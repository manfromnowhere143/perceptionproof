"""PerceptionProof P2b — closed-loop PDMS on NAVSIM.

Trains a native-architecture ego-status MLP ensemble on scenes disjoint from the metric-cached
tokens, then for each cached token computes each member's trajectory + its PDMS (NAVSIM's PDM
simulator), and dumps {token, trajs(xy), pdms} for downstream stats. PDMS is an independent
closed-loop-aligned outcome (not the structurally-coupled ADE), so disagreement-vs-PDMS is the
decisive test.

Env: LIMIT (cached tokens to score), TRAIN_CAP, K, EPOCHS, H, OUT.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
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

NUM_POSES = 8  # 4s @ 0.5s = PDM proposal sampling
H = int(os.environ.get("H", "256"))
K = int(os.environ.get("K", "4"))
EPOCHS = int(os.environ.get("EPOCHS", "400"))
TRAIN_CAP = int(os.environ.get("TRAIN_CAP", "3000"))
LIMIT = int(os.environ.get("LIMIT", "100000"))
OUT = os.environ.get("OUT", os.path.expanduser("~/pp_pdms.json"))


def native_mlp():
    return nn.Sequential(
        nn.Linear(8, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(),
        nn.Linear(H, H), nn.ReLU(), nn.Linear(H, NUM_POSES * 3),
    )


def main():
    print(f"[pdms] K={K} EPOCHS={EPOCHS} H={H} TRAIN_CAP={TRAIN_CAP} LIMIT={LIMIT}", flush=True)
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="default_run_pdm_score")
    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    traffic = instantiate(cfg.traffic_agents_policy.reactive, simulator.proposal_sampling)
    print("[pdms] simulator/scorer/traffic instantiated", flush=True)

    mcl = MetricCacheLoader(CACHE)
    cached = list(mcl.tokens)[:LIMIT]
    cached_set = set(cached)
    print(f"[pdms] cached tokens to score: {len(cached)}", flush=True)

    sf = SceneFilter(num_history_frames=4, num_future_frames=10, has_route=True, tokens=None)
    loader = SceneLoader(LOGS, None, sf, SensorConfig.build_no_sensors())
    feat = EgoStatusFeatureBuilder()

    # training data from scenes NOT in the cached (test) set
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
    print(f"[pdms] train scenes: {len(Xtr)}", flush=True)

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
        print(f"[pdms] member {seed} train MSE={loss.item():.3f}", flush=True)

    rows = []
    for i, tok in enumerate(cached):
        try:
            ai = loader.get_agent_input_from_token(tok)
            trajs = [a.compute_trajectory(ai) for a in agents]
            mc = mcl.get_from_token(tok)
            pdms = []
            for t in trajs:
                sr, _ = pdm_score(metric_cache=mc, model_trajectory=t,
                                  future_sampling=simulator.proposal_sampling,
                                  simulator=simulator, scorer=scorer, traffic_agents_policy=traffic)
                pdms.append(float(sr["pdm_score"]))
            rows.append({
                "token": tok,
                "log": mc.log_name,
                "trajs": [t.poses[:, :2].tolist() for t in trajs],
                "pdms": pdms,
            })
        except Exception as e:  # noqa: BLE001
            print(f"[pdms] skip {tok}: {e}", flush=True)
        if (i + 1) % 25 == 0:
            print(f"[pdms] scored {i + 1}/{len(cached)}", flush=True)

    json.dump({"K": K, "num_poses": NUM_POSES, "n": len(rows), "rows": rows}, open(OUT, "w"))
    print(f"[pdms] wrote {len(rows)} scored scenes to {OUT}", flush=True)


if __name__ == "__main__":
    main()
