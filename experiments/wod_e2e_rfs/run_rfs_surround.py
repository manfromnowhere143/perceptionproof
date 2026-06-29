"""WOD-E2E RFS, surround-view perception grounding (P2g).

P2f gave the ensemble one eye (front camera): disagreement-vs-RFS doubled (0.10 -> 0.22,
paired-significant). P2g tests the next rung on that slope: give it the FULL field of
view (all 8 cameras) and ask whether more perceptual coverage pushes the human-rating
correlation toward the pre-registered 0.30 bar.

A perception-coverage ablation ladder on the SAME 479 rater frames:
  ego      : ego_status(12)                      (blind baseline = P2e/P2f null)
  front    : ego + DINOv2(front, 384)            (reproduces P2f)
  surround : ego + PCA_256( DINOv2(all 8 cams) )  (full field of view, dim-controlled)

All 8 cameras of a frame are embedded in ONE batched forward. Embeddings are cached to
wod_surround_emb.npz so future ablations need no re-parse. Correlations target neg-RFS
with drive-cluster bootstrap CIs; the decisive comparisons are paired (surround vs front,
surround vs ego). No Waymo frames redistributed — only derived scalars + our embeddings.
"""
import glob
import io
import json
import os
from multiprocessing import Pool

import numpy as np

VAL_DIR = os.environ.get("VAL_DIR", "/home/dev_alfred_ai_app/wod/val")
OUT = os.environ.get("OUT", "/home/dev_alfred_ai_app/wod/wod_rfs_surround_out.json")
CACHE = os.environ.get("CACHE", "/home/dev_alfred_ai_app/wod/wod_surround_emb.npz")
T = 20
N_INTENT = 8
N_CAM = 8
EMB_D = 384
K = 4
PCA_D = 256
TRAIN_PER_SHARD = 120
SEED0 = 17
NPROC = int(os.environ.get("NPROC", "10"))


def _build_encoder():
    import timm
    import torch
    torch.set_num_threads(1)
    m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                          num_classes=0, img_size=224, dynamic_img_size=True)
    m.eval()
    cfg = timm.data.resolve_model_data_config(m)
    mean = np.array(cfg["mean"], np.float32).reshape(1, 3, 1, 1)
    std = np.array(cfg["std"], np.float32).reshape(1, 3, 1, 1)
    return m, mean, std


def _embed_batch(img_bytes_list, enc):
    """One batched forward over a frame's cameras -> [n_cam, 384]."""
    import torch
    from PIL import Image
    m, mean, std = enc
    arrs = []
    for b in img_bytes_list:
        img = Image.open(io.BytesIO(b)).convert("RGB").resize((224, 224))
        arrs.append(np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0)
    x = (np.stack(arrs, 0) - mean) / std
    with torch.no_grad():
        e = m(torch.from_numpy(x))  # [n,384]
    return e.numpy().astype(np.float32)


def parse_shard(args):
    si, sh = args
    os.environ["OMP_NUM_THREADS"] = "1"
    import tensorflow as tf
    from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as e2e
    enc = _build_encoder()
    tr, rater = [], []  # tr: (ego, emb8[8,384], Y40)
    f = e2e.E2EDFrame()
    try:
        for rec in tf.data.TFRecordDataset(sh):
            f.ParseFromString(rec.numpy())
            ps, fs = f.past_states, f.future_states
            if len(ps.vel_x) == 0 or len(fs.pos_x) < T:
                continue
            cams = {im.name: im.image for im in f.frame.images}
            if any(c not in cams for c in range(1, N_CAM + 1)):
                continue  # require full surround
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
            if not is_rater and len(tr) >= TRAIN_PER_SHARD:
                continue
            emb8 = _embed_batch([cams[c] for c in range(1, N_CAM + 1)], enc)  # [8,384] slot0=front
            if is_rater:
                rtrajs = [np.stack([np.array(t.pos_x[:T]), np.array(t.pos_y[:T])], axis=1).astype(np.float64)
                          for t in valid]
                rater.append(dict(ego=ego, emb8=emb8.tolist(), init_speed=spd, future=tgt.tolist(),
                                  rtrajs=[r.tolist() for r in rtrajs],
                                  rscores=[float(t.preference_score) for t in valid], shard=si))
            else:
                tr.append((ego, emb8.tolist(), tgt.reshape(-1).tolist()))
    except Exception as ex:  # noqa: BLE001
        return si, [], [], f"err:{ex}"
    return si, tr, rater, "ok"


def _disagree(preds):
    return float(np.mean([np.mean(np.linalg.norm(preds[i] - preds[j], axis=1))
                          for i in range(K) for j in range(i + 1, K)]))


def main():
    shards = sorted(glob.glob(os.path.join(VAL_DIR, "val_*.tfrecord-*-of-00093")))
    shards = [s for s in shards if not s.endswith(".gstmp")]
    print(f"shards={len(shards)} nproc={NPROC}", flush=True)
    print("prefetch encoder weights...", flush=True)
    _build_encoder()
    print("  encoder ready", flush=True)

    TR, rater = [], []
    with Pool(NPROC, maxtasksperchild=1) as pool:
        for si, tr, rf, status in pool.imap_unordered(parse_shard, list(enumerate(shards))):
            TR.extend(tr)
            rater.extend(rf)
            print(f"  shard {si}: +train={len(tr)} +rater={len(rf)} [{status}] "
                  f"tot_train={len(TR)} tot_rater={len(rater)}", flush=True)

    ego_tr = np.array([t[0] for t in TR], np.float64)
    emb_tr = np.array([t[1] for t in TR], np.float32)   # [Ntr,8,384]
    Y = np.array([t[2] for t in TR], np.float64)
    ego_r = np.array([r["ego"] for r in rater], np.float64)
    emb_r = np.array([r["emb8"] for r in rater], np.float32)  # [Nr,8,384]
    print(f"TRAIN ego={ego_tr.shape} emb={emb_tr.shape} Y={Y.shape} RATER={len(rater)}", flush=True)

    np.savez_compressed(CACHE, ego_tr=ego_tr, emb_tr=emb_tr, Y=Y,
                        ego_r=ego_r, emb_r=emb_r,
                        init_speed_r=np.array([r["init_speed"] for r in rater], np.float64),
                        shard_r=np.array([r["shard"] for r in rater], np.int32))
    with open(CACHE + ".rater.json", "w") as fh:
        json.dump([{"future": r["future"], "rtrajs": r["rtrajs"], "rscores": r["rscores"]} for r in rater], fh)
    print(f"cached embeddings -> {CACHE}", flush=True)

    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from waymo_open_dataset.metrics.python import rater_feedback_utils as rfu

    # feature sets for the ladder
    pca = PCA(n_components=PCA_D, random_state=SEED0).fit(emb_tr.reshape(len(emb_tr), -1))
    feats = {
        "ego": (ego_tr, ego_r),
        "front": (np.concatenate([ego_tr, emb_tr[:, 0]], 1),
                  np.concatenate([ego_r, emb_r[:, 0]], 1)),
        "surround": (np.concatenate([ego_tr, pca.transform(emb_tr.reshape(len(emb_tr), -1))], 1),
                     np.concatenate([ego_r, pca.transform(emb_r.reshape(len(emb_r), -1))], 1)),
    }
    print(f"PCA explained-var(256)={pca.explained_variance_ratio_.sum():.3f}", flush=True)

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

    trained = {}
    for tag, (Xtr, _) in feats.items():
        print(f"training {tag} ensemble ({Xtr.shape[1]}d)...", flush=True)
        trained[tag] = train(Xtr)

    rater_meta = json.load(open(CACHE + ".rater.json"))
    rows = []
    for i in range(len(rater)):
        rtrajs = [np.array(t, np.float64) for t in rater_meta[i]["rtrajs"]]
        rscores = np.array(rater_meta[i]["rscores"], np.float64)
        fut = np.array(rater_meta[i]["future"], np.float64)
        spd = float(ego_r[i][0] ** 2 + ego_r[i][1] ** 2) ** 0.5
        out = {"shard": int(rater[i]["shard"]), "init_speed": spd}
        for tag in feats:
            sc, ms = trained[tag]
            x = sc.transform(feats[tag][1][i].reshape(1, -1))
            preds = np.stack([m.predict(x).reshape(T, 2) for m in ms], 0)
            mean = preds.mean(0)
            rfs = float(np.array(rfu.get_rater_feedback_score(
                mean.reshape(1, 1, T, 2), np.array([[1.0]]), [rtrajs], [rscores],
                np.array([spd]), frequency=4, length_seconds=5)["rater_feedback_score"]).ravel()[0])
            out[f"{tag}_dis"] = _disagree(preds)
            out[f"{tag}_rfs"] = rfs
            out[f"{tag}_ade"] = float(np.mean(np.linalg.norm(mean - fut, axis=1)))
        rows.append(out)

    with open(OUT, "w") as fh:
        json.dump(dict(rows=rows, n_train=int(len(TR)), K=K, T=T, pca_d=PCA_D,
                       pca_evr=float(pca.explained_variance_ratio_.sum())), fh)
    print(f"WROTE {OUT} rows={len(rows)}", flush=True)
    from scipy.stats import spearmanr
    for tag in feats:
        d = np.array([x[f"{tag}_dis"] for x in rows])
        rr = np.array([x[f"{tag}_rfs"] for x in rows])
        print(f"SANITY {tag:8s} disagreement-vs-RFS rho={spearmanr(d, rr).correlation:+.3f}", flush=True)


if __name__ == "__main__":
    main()
