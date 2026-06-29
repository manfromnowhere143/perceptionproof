"""WOD-E2E RFS, perception-grounded ensemble (P2f).

The ego-only ensemble (P2e) predicted human RFS only weakly (rho=0.15, below the
0.30 bar) — its members disagree on ego kinematics, not on scene difficulty. This
adds a frozen DINOv2 front-camera embedding so the ensemble's disagreement reflects
actual perceptual scene content, and re-tests H1.

For a clean A/B it trains BOTH ensembles on the SAME frames:
  - ego  : features = ego_status(12)              (recomputes the P2e null here)
  - vis  : features = ego_status(12) + DINOv2(384)
and scores each frame's RFS (Waymo official) under each model. analyze_vision.py
then compares rho(disagreement, neg-RFS) for ego vs vis with cluster-bootstrap CIs.

Redistributes no Waymo frames — only derived scalars per (shard, frame).
"""
import glob
import io
import json
import os
from multiprocessing import Pool

import numpy as np

VAL_DIR = os.environ.get("VAL_DIR", "/home/dev_alfred_ai_app/wod/val")
OUT = os.environ.get("OUT", "/home/dev_alfred_ai_app/wod/wod_rfs_vision_out.json")
T = 20
N_INTENT = 8
K = 4
TRAIN_PER_SHARD = 150
SEED0 = 17
NPROC = int(os.environ.get("NPROC", "10"))
FRONT_CAM = 1  # Waymo CameraName.FRONT


def _build_encoder():
    import timm
    import torch
    torch.set_num_threads(1)
    m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                          num_classes=0, img_size=224, dynamic_img_size=True)
    m.eval()
    cfg = timm.data.resolve_model_data_config(m)
    mean = np.array(cfg["mean"], np.float32).reshape(3, 1, 1)
    std = np.array(cfg["std"], np.float32).reshape(3, 1, 1)
    return m, mean, std


def _embed(img_bytes, enc):
    import torch
    from PIL import Image
    m, mean, std = enc
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    x = (np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0 - mean) / std
    with torch.no_grad():
        e = m(torch.from_numpy(x[None]))  # [1,384]
    return e[0].numpy().astype(np.float32)


def parse_shard(args):
    si, sh = args
    os.environ["OMP_NUM_THREADS"] = "1"
    import tensorflow as tf
    from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as e2e
    enc = _build_encoder()
    X_ego, X_emb, Y, rater = [], [], [], []
    f = e2e.E2EDFrame()
    try:
        for rec in tf.data.TFRecordDataset(sh):
            f.ParseFromString(rec.numpy())
            ps, fs = f.past_states, f.future_states
            if len(ps.vel_x) == 0 or len(fs.pos_x) < T:
                continue
            front = next((im.image for im in f.frame.images if im.name == FRONT_CAM), None)
            if front is None:
                continue
            vx, vy = float(ps.vel_x[-1]), float(ps.vel_y[-1])
            ax = float(ps.accel_x[-1]) if len(ps.accel_x) else 0.0
            ay = float(ps.accel_y[-1]) if len(ps.accel_y) else 0.0
            intent = int(f.intent)
            oh = [0.0] * N_INTENT
            if 0 <= intent < N_INTENT:
                oh[intent] = 1.0
            ego = [vx, vy, ax, ay] + oh
            spd = (vx * vx + vy * vy) ** 0.5
            tgt = np.stack([np.array(fs.pos_x[:T]), np.array(fs.pos_y[:T])], axis=1).astype(np.float64)
            valid = [t for t in f.preference_trajectories
                     if len(t.pos_x) >= T and t.preference_score >= 0]
            is_rater = bool(valid)
            if not is_rater and len(X_ego) >= TRAIN_PER_SHARD:
                continue
            emb = _embed(front, enc)  # only embed frames we keep
            if is_rater:
                rtrajs = [np.stack([np.array(t.pos_x[:T]), np.array(t.pos_y[:T])], axis=1).astype(np.float64)
                          for t in valid]
                rater.append(dict(ego=ego, emb=emb.tolist(), init_speed=spd, future=tgt.tolist(),
                                  rtrajs=[r.tolist() for r in rtrajs],
                                  rscores=[float(t.preference_score) for t in valid], shard=si))
            else:
                X_ego.append(ego)
                X_emb.append(emb.tolist())
                Y.append(tgt.reshape(-1).tolist())
    except Exception as ex:  # noqa: BLE001
        return si, [], [], [], [], f"err:{ex}"
    return si, X_ego, X_emb, Y, rater, "ok"


def _ensemble_predict(models, scaler, feat):
    x = scaler.transform(np.array(feat, np.float64).reshape(1, -1))
    return np.stack([m.predict(x).reshape(T, 2) for m in models], axis=0)  # [K,T,2]


def _disagree(preds):
    return float(np.mean([np.mean(np.linalg.norm(preds[i] - preds[j], axis=1))
                          for i in range(K) for j in range(i + 1, K)]))


def main():
    shards = sorted(glob.glob(os.path.join(VAL_DIR, "val_*.tfrecord-*-of-00093")))
    shards = [s for s in shards if not s.endswith(".gstmp")]
    print(f"shards={len(shards)} nproc={NPROC}", flush=True)
    # warm the DINOv2 weight cache once before forking workers
    print("prefetch encoder weights...", flush=True)
    _build_encoder()
    print("  encoder ready", flush=True)

    Xe, Xv, Y, rater = [], [], [], []
    with Pool(NPROC, maxtasksperchild=1) as pool:
        for si, xe, xv, yy, rf, status in pool.imap_unordered(parse_shard, list(enumerate(shards))):
            Xe.extend(xe)
            Xv.extend(xv)
            Y.extend(yy)
            rater.extend(rf)
            print(f"  shard {si}: +train={len(xe)} +rater={len(rf)} [{status}] "
                  f"tot_train={len(Xe)} tot_rater={len(rater)}", flush=True)

    Xe = np.array(Xe, np.float64)
    Xv = np.concatenate([Xe, np.array(Xv, np.float64)], axis=1)  # ego + emb
    Y = np.array(Y, np.float64)
    print(f"TRAIN ego={Xe.shape} vis={Xv.shape} Y={Y.shape} RATER={len(rater)}", flush=True)

    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from waymo_open_dataset.metrics.python import rater_feedback_utils as rfu

    def train(X):
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        ms = []
        for k in range(K):
            m = MLPRegressor(hidden_layer_sizes=(256, 256), activation="relu", solver="adam",
                             alpha=1e-4, max_iter=300, random_state=SEED0 + 101 * k)
            m.fit(Xs, Y)
            ms.append(m)
        return sc, ms

    print("training ego ensemble...", flush=True)
    sc_e, ms_e = train(Xe)
    print("training vision ensemble...", flush=True)
    sc_v, ms_v = train(Xv)

    rows = []
    for r in rater:
        ego = r["ego"]
        vis = ego + r["emb"]
        rtrajs = [np.array(t, np.float64) for t in r["rtrajs"]]
        rscores = np.array(r["rscores"], np.float64)
        fut = np.array(r["future"], np.float64)
        out = {}
        for tag, sc, ms, feat in [("ego", sc_e, ms_e, ego), ("vis", sc_v, ms_v, vis)]:
            preds = _ensemble_predict(ms, sc, feat)
            mean = preds.mean(axis=0)
            rfs = float(np.array(rfu.get_rater_feedback_score(
                mean.reshape(1, 1, T, 2), np.array([[1.0]]), [rtrajs], [rscores],
                np.array([r["init_speed"]]), frequency=4, length_seconds=5)["rater_feedback_score"]).ravel()[0])
            out[f"{tag}_dis"] = _disagree(preds)
            out[f"{tag}_rfs"] = rfs
            out[f"{tag}_ade"] = float(np.mean(np.linalg.norm(mean - fut, axis=1)))
        out["shard"] = int(r["shard"])
        out["init_speed"] = float(r["init_speed"])
        rows.append(out)

    with open(OUT, "w") as fh:
        json.dump(dict(rows=rows, n_train=int(len(Xe)), K=K, T=T, emb_dim=int(Xv.shape[1] - Xe.shape[1])), fh)
    print(f"WROTE {OUT} rows={len(rows)}", flush=True)
    from scipy.stats import spearmanr
    for tag in ("ego", "vis"):
        d = np.array([x[f"{tag}_dis"] for x in rows])
        rr = np.array([x[f"{tag}_rfs"] for x in rows])
        print(f"SANITY {tag} disagreement-vs-RFS rho={spearmanr(d, rr).correlation:+.3f}", flush=True)


if __name__ == "__main__":
    main()
