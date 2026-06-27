"""PerceptionProof P2a — first real number on NAVSIM (CPU, no sensors, no metric cache).

Trains a K-member ego-status MLP ensemble and, on a held-out (by-log) test split, dumps each
member's predicted trajectory + the human future trajectory to JSON. Disagreement, ADE, and
all statistics are computed downstream by the already-tested PerceptionProof signals.py /
scoring.py — this script only produces real model trajectories on real frames.

Uses the real OpenScene/nuPlan trainval scenes we have locally, split into train/test by LOG
(disjoint, no leakage). This is NOT the official navtest benchmark split (that needs the
separate OpenScene test download); it is a legitimate first real measurement, documented as such.

Env knobs: CAP (total scenes), TEST_FRAC, K, EPOCHS, OUT.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml  # noqa: F401  (kept for parity; not required in this variant)
from torch import nn

from navsim.agents.ego_status_mlp_agent import EgoStatusFeatureBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneFilter, SceneLoader

DATA = Path(os.environ["OPENSCENE_DATA_ROOT"])
_LOGBASE = DATA / "trainval_navsim_logs"
LOGS = _LOGBASE / "trainval" if (_LOGBASE / "trainval").is_dir() else _LOGBASE
NUM_FUT = 10

CAP = int(os.environ.get("CAP", "4000"))
TEST_FRAC = float(os.environ.get("TEST_FRAC", "0.2"))
K = int(os.environ.get("K", "4"))
EPOCHS = int(os.environ.get("EPOCHS", "400"))
OUT = os.environ.get("OUT", os.path.expanduser("~/pp_result.json"))


class MLP(nn.Module):
    def __init__(self, din=8, dout=NUM_FUT * 3, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, dout))

    def forward(self, x):
        return self.net(x)


def main():
    print(f"[exp] CAP={CAP} TEST_FRAC={TEST_FRAC} K={K} EPOCHS={EPOCHS}", flush=True)
    sf = SceneFilter(num_history_frames=4, num_future_frames=NUM_FUT, has_route=True, tokens=None)
    print("[exp] building loader over ALL available trainval scenes (scans log pickles)...", flush=True)
    loader = SceneLoader(data_path=LOGS, original_sensor_path=None, scene_filter=sf,
                         sensor_config=SensorConfig.build_no_sensors())
    all_tokens = loader.tokens
    print(f"[exp] available scenes: {len(all_tokens)}", flush=True)

    feat = EgoStatusFeatureBuilder()
    recs = []
    for t in all_tokens:
        if len(recs) >= CAP:
            break
        try:
            scene = loader.get_scene_from_token(t)
            x = feat.compute_features(scene.get_agent_input())["ego_status"].numpy().astype(np.float32)
            fut = np.asarray(scene.get_future_trajectory(NUM_FUT).poses, dtype=np.float32)
            if fut.shape[0] < NUM_FUT:
                continue
            recs.append((t, getattr(scene.scene_metadata, "log_name", "unknown"), x, fut[:NUM_FUT]))
        except Exception:  # noqa: BLE001
            continue
    print(f"[exp] extracted {len(recs)} scenes", flush=True)
    assert len(recs) > 200, "not enough scenes extracted"

    # disjoint split by log
    logs = sorted({r[1] for r in recs})
    test_logs = set(logs[:: max(1, int(1 / TEST_FRAC))])
    train = [r for r in recs if r[1] not in test_logs]
    test = [r for r in recs if r[1] in test_logs]
    print(f"[exp] logs={len(logs)} train={len(train)} test={len(test)}", flush=True)

    Xtr = np.stack([r[2] for r in train])
    Ytr = np.stack([r[3] for r in train]).reshape(len(train), -1)
    Xte = np.stack([r[2] for r in test])

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_n = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    Ytr_t = torch.tensor(Ytr, dtype=torch.float32)
    Xte_n = torch.tensor((Xte - mu) / sd, dtype=torch.float32)

    models = []
    for seed in range(K):
        torch.manual_seed(seed)
        m = MLP()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(EPOCHS):
            opt.zero_grad()
            loss = ((m(Xtr_n) - Ytr_t) ** 2).mean()
            loss.backward()
            opt.step()
        print(f"[exp] member {seed} final train MSE={loss.item():.4f}", flush=True)
        models.append(m)

    preds = [m(Xte_n).detach().numpy().reshape(-1, NUM_FUT, 3) for m in models]
    rows = []
    for i, r in enumerate(test):
        rows.append({
            "token": r[0],
            "log": r[1],
            "trajs": [preds[k][i][:, :2].tolist() for k in range(K)],  # K x (10,2) xy
            "human": r[3][:, :2].tolist(),  # (10,2) xy
        })
    json.dump({"K": K, "num_fut": NUM_FUT, "n": len(rows),
               "split": "by-log disjoint over OpenScene trainval (NOT official navtest)",
               "rows": rows}, open(OUT, "w"))
    print(f"[exp] wrote {len(rows)} test scenes to {OUT}", flush=True)


if __name__ == "__main__":
    main()
