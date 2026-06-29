"""PerceptionProof Phase 2B — closed-loop validity-bridge probe.

Scores PER-TRAJECTORY closed-loop PDMS (real NAVSIM PDM simulator) for a diverse set of
trajectories per scene (K ego-MLP ensemble members + constant-velocity + stopped heuristics),
and records each scene's human future. Downstream (local) we test whether a CHEAP learned
predictor of closed-loop PDMS recovers the closed-loop ranking better than the open-loop ADE
metric the field uses — the validity bridge. Abundant supervision (the simulator), no data wall.

Env: K, EPOCHS, H, TRAIN_CAP, NPROC, LIMIT, OUT.
"""
# ruff: noqa: E702  (compact research script)
import copy
import gc
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
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
K = int(os.environ.get("K", "6"))
EPOCHS = int(os.environ.get("EPOCHS", "400"))
TRAIN_CAP = int(os.environ.get("TRAIN_CAP", "3000"))
LIMIT = int(os.environ.get("LIMIT", "400"))
NPROC = int(os.environ.get("NPROC", str(max(1, os.cpu_count() - 2))))
OUT = os.environ.get("OUT", os.path.expanduser("~/pp_bridge.json"))


def native_mlp():
    return nn.Sequential(nn.Linear(8, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(),
                         nn.Linear(H, H), nn.ReLU(), nn.Linear(H, NUM_POSES * 3))


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
    token, trajs, human = task
    try:
        mc = MCL.get_from_token(token)

        def g(sr, c):
            v = sr[c]
            return float(v.iloc[0]) if hasattr(v, "iloc") else float(v)

        cands = []
        for t in trajs:
            try:
                sr, _ = pdm_score(metric_cache=mc, model_trajectory=t,
                                  future_sampling=SIM.proposal_sampling, simulator=SIM,
                                  scorer=SCORER, traffic_agents_policy=TRAFFIC)
                cands.append({"poses": t.poses[:, :2].tolist(), "pdms": g(sr, "pdm_score"),
                              "nc": g(sr, "no_at_fault_collisions"),
                              "dac": g(sr, "drivable_area_compliance")})
            except Exception:  # noqa: BLE001
                continue
        if len(cands) < 2:
            return {"error": f"{token}: too few scored"}
        return {"token": token, "log": mc.log_name, "human": human, "cands": cands}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{token}: {type(e).__name__}: {e}"}


def make_heuristics(member0):
    p0 = member0.poses.copy()
    out = []
    step = (p0[1] - p0[0]) if len(p0) > 1 else np.zeros(3)
    cv = np.stack([p0[0] + (i + 1) * step for i in range(len(p0))])
    cv[:, 2] = p0[0, 2]
    h1 = copy.deepcopy(member0)
    h1.poses = cv.astype(p0.dtype)
    out.append(h1)
    st = np.stack([p0[0].copy() for _ in range(len(p0))])
    h2 = copy.deepcopy(member0)
    h2.poses = st.astype(p0.dtype)
    out.append(h2)
    return out


def main():
    print(f"[bridge] K={K} EPOCHS={EPOCHS} NPROC={NPROC} LIMIT={LIMIT}", flush=True)
    mcl = MetricCacheLoader(CACHE)
    cached = list(mcl.tokens)
    cached_set = set(cached)
    print(f"[bridge] cached tokens: {len(cached)}", flush=True)

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
        except Exception:  # noqa: BLE001
            continue
    Xtr = torch.tensor(np.stack(Xtr))
    Ytr = torch.tensor(np.stack(Ytr))
    print(f"[bridge] train scenes: {len(Xtr)}", flush=True)

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
    print(f"[bridge] trained {K} members; building diverse trajectory sets...", flush=True)

    tasks = []
    for tok in cached[:LIMIT]:
        try:
            ai = loader.get_agent_input_from_token(tok)
            members = [a.compute_trajectory(ai) for a in agents]
            scene = loader.get_scene_from_token(tok)
            human = np.asarray(scene.get_future_trajectory(NUM_POSES).poses,
                               dtype=np.float32)[:NUM_POSES, :2].tolist()
            trajs = members + make_heuristics(members[0])
            tasks.append((tok, trajs, human))
        except Exception as e:  # noqa: BLE001
            print(f"[bridge] task skip {tok}: {e}", flush=True)
    del loader, Xtr, Ytr
    gc.collect()
    print(f"[bridge] scoring {len(tasks)} scenes x ~{K + 2} trajs on {NPROC} cores...", flush=True)

    rows, errs = [], 0
    with Pool(NPROC, initializer=_init_worker, initargs=(CONFIG_DIR, str(CACHE)),
              maxtasksperchild=20) as pool:
        for i, res in enumerate(pool.imap_unordered(_score_one, tasks, chunksize=2)):
            if "error" in res:
                errs += 1
            else:
                rows.append(res)
            if (i + 1) % 40 == 0:
                json.dump({"K": K, "n": len(rows), "rows": rows}, open(OUT, "w"))
                print(f"[bridge] {i + 1}/{len(tasks)} ({len(rows)} ok, {errs} err)", flush=True)
    json.dump({"K": K, "n": len(rows), "rows": rows}, open(OUT, "w"))
    print(f"[bridge] wrote {len(rows)} scenes to {OUT} ({errs} errors)", flush=True)


if __name__ == "__main__":
    main()
