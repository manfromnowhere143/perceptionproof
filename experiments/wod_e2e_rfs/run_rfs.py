"""WOD-E2E RFS experiment (parallel parse).

Same science as wod_rfs.py, but the 243 GB shard parse (each E2EDFrame embeds 8
camera images, so ParseFromString is CPU-heavy) is fanned out across cores. Each
worker returns only small derived arrays (features, 40-d future, rater scalars) —
no image bytes cross the process boundary. Main process trains the K-seed
ego-status MLP ensemble and scores RFS.
"""
import glob
import json
import os
from multiprocessing import Pool

import numpy as np

VAL_DIR = os.environ.get("VAL_DIR", "/home/dev_alfred_ai_app/wod/val")
OUT = os.environ.get("OUT", "/home/dev_alfred_ai_app/wod/wod_rfs_out.json")
T = 20
N_INTENT = 8
K = 4
TRAIN_PER_SHARD = 200
SEED0 = 17
NPROC = int(os.environ.get("NPROC", "12"))


def parse_shard(args):
    si, sh = args
    # import heavy libs inside the worker so the fork stays lean
    import tensorflow as tf
    from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as e2e
    X, Y, rater = [], [], []
    f = e2e.E2EDFrame()
    try:
        for rec in tf.data.TFRecordDataset(sh):
            f.ParseFromString(rec.numpy())
            ps, fs = f.past_states, f.future_states
            if len(ps.vel_x) == 0 or len(fs.pos_x) < T:
                continue
            vx, vy = float(ps.vel_x[-1]), float(ps.vel_y[-1])
            ax = float(ps.accel_x[-1]) if len(ps.accel_x) else 0.0
            ay = float(ps.accel_y[-1]) if len(ps.accel_y) else 0.0
            intent = int(f.intent)
            oh = [0.0] * N_INTENT
            if 0 <= intent < N_INTENT:
                oh[intent] = 1.0
            feat = [vx, vy, ax, ay] + oh
            spd = (vx * vx + vy * vy) ** 0.5
            tgt = np.stack([np.array(fs.pos_x[:T]), np.array(fs.pos_y[:T])], axis=1).astype(np.float64)
            valid = [t for t in f.preference_trajectories
                     if len(t.pos_x) >= T and t.preference_score >= 0]
            if valid:
                rtrajs = [np.stack([np.array(t.pos_x[:T]), np.array(t.pos_y[:T])], axis=1).astype(np.float64)
                          for t in valid]
                rscores = [float(t.preference_score) for t in valid]
                rater.append(dict(feat=feat, init_speed=spd, future=tgt.tolist(),
                                  rtrajs=[r.tolist() for r in rtrajs], rscores=rscores, shard=si))
            elif len(X) < TRAIN_PER_SHARD:
                X.append(feat)
                Y.append(tgt.reshape(-1).tolist())
    except Exception as ex:  # noqa: BLE001
        return si, [], [], [], f"err:{ex}"
    return si, X, Y, rater, "ok"


def main():
    shards = sorted(glob.glob(os.path.join(VAL_DIR, "val_*.tfrecord-*-of-00093")))
    shards = [s for s in shards if not s.endswith(".gstmp")]
    print(f"shards={len(shards)} nproc={NPROC}", flush=True)

    Xtr, Ytr, rater = [], [], []
    with Pool(NPROC, maxtasksperchild=2) as pool:
        for si, X, Y, rf, status in pool.imap_unordered(parse_shard, list(enumerate(shards))):
            Xtr.extend(X)
            Ytr.extend(Y)
            rater.extend(rf)
            print(f"  shard {si}: +train={len(X)} +rater={len(rf)} [{status}] "
                  f"tot_train={len(Xtr)} tot_rater={len(rater)}", flush=True)

    Xtr = np.array(Xtr, dtype=np.float64)
    Ytr = np.array(Ytr, dtype=np.float64)
    print(f"TRAIN X={Xtr.shape} Y={Ytr.shape} RATER={len(rater)}", flush=True)

    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from waymo_open_dataset.metrics.python import rater_feedback_utils as rfu

    scaler = StandardScaler().fit(Xtr)
    Xs = scaler.transform(Xtr)
    models = []
    for k in range(K):
        m = MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu", solver="adam",
                         alpha=1e-4, max_iter=300, random_state=SEED0 + 101 * k)
        m.fit(Xs, Ytr)
        models.append(m)
        print(f"  trained member {k} loss={m.loss_:.4f}", flush=True)

    rows = []
    for r in rater:
        x = scaler.transform(np.array(r["feat"], dtype=np.float64).reshape(1, -1))
        preds = np.stack([m.predict(x).reshape(T, 2) for m in models], axis=0)
        d = [float(np.mean(np.linalg.norm(preds[i] - preds[j], axis=1)))
             for i in range(K) for j in range(i + 1, K)]
        disagreement = float(np.mean(d))
        mean_traj = preds.mean(axis=0)
        fut = np.array(r["future"], dtype=np.float64)
        ade = float(np.mean(np.linalg.norm(mean_traj - fut, axis=1)))
        rtrajs = [np.array(t, dtype=np.float64) for t in r["rtrajs"]]
        out = rfu.get_rater_feedback_score(
            mean_traj.reshape(1, 1, T, 2), np.array([[1.0]]),
            [rtrajs], [np.array(r["rscores"], dtype=np.float64)],
            np.array([r["init_speed"]]), frequency=4, length_seconds=5)
        rfs = float(np.array(out["rater_feedback_score"]).ravel()[0])
        rows.append(dict(shard=int(r["shard"]), disagreement=disagreement, rfs=rfs,
                         ade=ade, init_speed=float(r["init_speed"]),
                         n_rater=int(len(r["rscores"]))))

    with open(OUT, "w") as fh:
        json.dump(dict(rows=rows, n_train=int(len(Xtr)), K=K, T=T), fh)
    print(f"WROTE {OUT} rows={len(rows)}", flush=True)
    if rows:
        from scipy.stats import spearmanr
        dd = np.array([x["disagreement"] for x in rows])
        rr = np.array([x["rfs"] for x in rows])
        aa = np.array([x["ade"] for x in rows])
        print(f"SANITY disagreement-vs-RFS rho={spearmanr(dd, rr).correlation:.3f}", flush=True)
        print(f"SANITY ade-vs-RFS rho={spearmanr(aa, rr).correlation:.3f}", flush=True)
        print(f"SANITY disagreement-vs-ade rho={spearmanr(dd, aa).correlation:.3f}", flush=True)
        print(f"RFS mean={rr.mean():.3f} sd={rr.std():.3f} min={rr.min():.3f} max={rr.max():.3f}", flush=True)


if __name__ == "__main__":
    main()
