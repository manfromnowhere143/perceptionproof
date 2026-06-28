"""PerceptionProof P2c — PARALLEL closed-loop scorer (collision-aligned signal vs PDMS gates).

Trains the ego-status ensemble in the main process, then fans the expensive per-scene PDM
simulation + collision-geometry signal across a worker pool (one PDM sim per core). Same science
as pp_p2c.py; this is the throughput version for the powered, drive-stratified decisive run.

Env: K, EPOCHS, H, TRAIN_CAP, NPROC, OUT.
"""

import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch import nn

from navsim.agents.ego_status_mlp_agent import EgoStatusFeatureBuilder, EgoStatusMLPAgent
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneFilter, SceneLoader
from navsim.evaluate.pdm_score import pdm_score

DATA = Path(os.environ["OPENSCENE_DATA_ROOT"])
LOGS = DATA / "trainval_navsim_logs" / "trainval"
CACHE = Path(os.environ["NAVSIM_EXP_ROOT"]) / "metric_cache"
CONFIG_DIR = str(Path.cwd() / "navsim/planning/script/config/pdm_scoring")

NUM_POSES = 8
H = int(os.environ.get("H", "256"))
K = int(os.environ.get("K", "4"))
EPOCHS = int(os.environ.get("EPOCHS", "300"))
TRAIN_CAP = int(os.environ.get("TRAIN_CAP", "2500"))
NPROC = int(os.environ.get("NPROC", str(max(1, os.cpu_count() - 1))))
OUT = os.environ.get("OUT", os.path.expanduser("~/pp_p2c_scaled.json"))
EGO_HALF_L, EGO_HALF_W = 2.3, 1.0


def native_mlp():
    return nn.Sequential(nn.Linear(8, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(),
                         nn.Linear(H, H), nn.ReLU(), nn.Linear(H, NUM_POSES * 3))


def ego_abs_xyh(traj, ego_state):
    poses = [StateSE2(float(p[0]), float(p[1]), float(p[2])) for p in traj.poses]
    ap = relative_to_absolute_poses(ego_state.rear_axle, poses)
    return np.array([[p.x, p.y, p.heading] for p in ap], dtype=float)


def footprint_points(ego_xyh):
    pts = []
    for x, y, h in ego_xyh:
        c, s = np.cos(h), np.sin(h)
        for dx, dy in [(EGO_HALF_L, EGO_HALF_W), (EGO_HALF_L, -EGO_HALF_W),
                       (-EGO_HALF_L, EGO_HALF_W), (-EGO_HALF_L, -EGO_HALF_W)]:
            pts.append([x + dx * c - dy * s, y + dx * s + dy * c])
    return np.array(pts, dtype=float)


def geom_signals(traj, mc):
    ego = ego_abs_xyh(traj, mc.ego_state)
    try:
        corners = footprint_points(ego)
        arr = np.asarray(mc.drivable_area_map.points_in_polygons(corners))
        in_da = arr.any(axis=0) if arr.ndim == 2 else arr
        off_road_frac = float(1.0 - in_da.mean())
    except Exception:
        off_road_frac = float("nan")
    fut = mc.future_tracked_objects
    min_dist = np.inf
    for i in range(ego.shape[0]):
        fidx = min((i + 1) * 5, len(fut) - 1)
        for o in (fut[fidx].tracked_objects if fidx >= 0 else []):
            d = float(np.hypot(ego[i, 0] - o.center.x, ego[i, 1] - o.center.y))
            if d < min_dist:
                min_dist = d
    min_dist = float(min_dist) if np.isfinite(min_dist) else 100.0
    return float(1.0 / (min_dist + 1.0)), off_road_frac, min_dist


# ---- worker globals + functions ----
SIM = SCORER = TRAFFIC = MCL = None


def _init_worker(config_dir, cache_dir):
    global SIM, SCORER, TRAFFIC, MCL
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="default_run_pdm_score")
    SIM = instantiate(cfg.simulator)
    SCORER = instantiate(cfg.scorer)
    TRAFFIC = instantiate(cfg.traffic_agents_policy.reactive, SIM.proposal_sampling)
    MCL = MetricCacheLoader(Path(cache_dir))


def _score_one(task):
    token, trajs = task
    try:
        mc = MCL.get_from_token(token)
        sr, _ = pdm_score(metric_cache=mc, model_trajectory=trajs[0],
                          future_sampling=SIM.proposal_sampling, simulator=SIM,
                          scorer=SCORER, traffic_agents_policy=TRAFFIC)

        def g(c):
            v = sr[c]
            return float(v.iloc[0]) if hasattr(v, "iloc") else float(v)

        crisk, offroad, mindist = geom_signals(trajs[0], mc)
        return {"token": token, "log": mc.log_name,
                "trajs": [t.poses[:, :2].tolist() for t in trajs],
                "collision_risk": crisk, "off_road": offroad, "min_dist": mindist,
                "nc": g("no_at_fault_collisions"), "dac": g("drivable_area_compliance"),
                "ttc": g("time_to_collision_within_bound"), "ep": g("ego_progress"),
                "pdms": g("pdm_score")}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{token}: {type(e).__name__}: {e}"}


def main():
    print(f"[par] K={K} EPOCHS={EPOCHS} NPROC={NPROC}", flush=True)
    mcl = MetricCacheLoader(CACHE)
    cached = list(mcl.tokens)
    cached_set = set(cached)
    print(f"[par] cached tokens: {len(cached)}", flush=True)

    sf = SceneFilter(num_history_frames=4, num_future_frames=10, has_route=True, tokens=None)
    loader = SceneLoader(LOGS, None, sf, SensorConfig.build_no_sensors())
    feat = EgoStatusFeatureBuilder()

    Xtr, Ytr = [], []
    for t in loader.tokens:
        if len(Xtr) >= TRAIN_CAP:
            break
        if t in cached_set:
            continue
        try:
            scene = loader.get_scene_from_token(t)
            x = feat.compute_features(scene.get_agent_input())["ego_status"].numpy().astype(np.float32)
            fut = np.asarray(scene.get_future_trajectory(NUM_POSES).poses, dtype=np.float32)
            if fut.shape[0] < NUM_POSES:
                continue
            Xtr.append(x)
            Ytr.append(fut[:NUM_POSES].reshape(-1))
        except Exception:
            continue
    Xtr = torch.tensor(np.stack(Xtr))
    Ytr = torch.tensor(np.stack(Ytr))
    print(f"[par] train scenes: {len(Xtr)}", flush=True)

    agents = []
    for seed in range(K):
        torch.manual_seed(seed)
        mlp = native_mlp()
        opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        for _ in range(EPOCHS):
            opt.zero_grad()
            loss = ((mlp(Xtr) - Ytr) ** 2).mean()
            loss.backward()
            opt.step()
        a = EgoStatusMLPAgent(hidden_layer_dim=H, lr=1e-3,
                              trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5))
        a._mlp.load_state_dict(mlp.state_dict())
        a._mlp.eval()
        agents.append(a)
    print(f"[par] trained {K} members; building trajectories...", flush=True)

    tasks = []
    for tok in cached:
        try:
            ai = loader.get_agent_input_from_token(tok)
            tasks.append((tok, [a.compute_trajectory(ai) for a in agents]))
        except Exception as e:  # noqa: BLE001
            print(f"[par] traj skip {tok}: {e}", flush=True)
    import gc
    del loader, Xtr, Ytr  # free the giant scene dict BEFORE forking workers (avoids OOM)
    gc.collect()
    print(f"[par] scoring {len(tasks)} scenes on {NPROC} cores...", flush=True)

    rows, errs = [], 0
    with Pool(NPROC, initializer=_init_worker, initargs=(CONFIG_DIR, str(CACHE)),
              maxtasksperchild=25) as pool:
        for i, res in enumerate(pool.imap_unordered(_score_one, tasks, chunksize=4)):
            if "error" in res:
                errs += 1
            else:
                rows.append(res)
            if (i + 1) % 100 == 0:
                json.dump({"K": K, "n": len(rows), "rows": rows}, open(OUT, "w"))
                print(f"[par] {i + 1}/{len(tasks)} done ({len(rows)} ok, {errs} err)", flush=True)
    json.dump({"K": K, "n": len(rows), "rows": rows}, open(OUT, "w"))
    print(f"[par] wrote {len(rows)} scenes to {OUT} ({errs} errors)", flush=True)


if __name__ == "__main__":
    main()
