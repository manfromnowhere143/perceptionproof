"""Stability study (parallel). Same logic as stability_vision.py, but fans the
(instantiation b, rung) tasks across cores. Each task trains a K-member ensemble with
the seeds for instantiation b and returns rho(disagreement, neg-RFS). Same seeds across
rungs at a given b => paired comparison isolating the feature set.
"""
import json
import os
from multiprocessing import Pool

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from waymo_open_dataset.metrics.python import rater_feedback_utils as rfu

CACHE = os.environ.get("CACHE", "/home/dev_alfred_ai_app/wod/wod_surround_emb.npz")
OUT = os.environ.get("OUT", "/home/dev_alfred_ai_app/wod/stability_out.json")
T, K, B, PCA_D, NPROC = 20, 4, 20, 256, 12

d = np.load(CACHE)
ego_tr, emb_tr, Y = d["ego_tr"], d["emb_tr"], d["Y"]
ego_r, emb_r, ispd = d["ego_r"], d["emb_r"], d["init_speed_r"]
meta = json.load(open(CACHE + ".rater.json"))
RTRAJS = [[np.array(t, float) for t in m["rtrajs"]] for m in meta]
RSC = [np.array(m["rscores"], float) for m in meta]
N = len(ego_r)
pca = PCA(PCA_D, random_state=17).fit(emb_tr.reshape(len(emb_tr), -1))
FS = {
    "ego": (ego_tr, ego_r),
    "front": (np.c_[ego_tr, emb_tr[:, 0]], np.c_[ego_r, emb_r[:, 0]]),
    "surround": (np.c_[ego_tr, pca.transform(emb_tr.reshape(len(emb_tr), -1))],
                 np.c_[ego_r, pca.transform(emb_r.reshape(len(emb_r), -1))]),
}


def task(args):
    b, tag = args
    os.environ["OMP_NUM_THREADS"] = "1"
    seeds = [17 + 1000 * b + 7 * k for k in range(K)]
    Xtr, Xr = FS[tag]
    sc = StandardScaler().fit(Xtr)
    Xs, Xrs = sc.transform(Xtr), sc.transform(Xr)
    ms = [MLPRegressor(hidden_layer_sizes=(256, 256), alpha=1e-4, max_iter=250,
                       random_state=s).fit(Xs, Y) for s in seeds]
    P = np.stack([m.predict(Xrs).reshape(N, T, 2) for m in ms], 0)
    mean = P.mean(0)
    dis = np.array([np.mean([np.mean(np.linalg.norm(P[a, i] - P[c, i], axis=1))
                             for a in range(K) for c in range(a + 1, K)]) for i in range(N)])
    rfs = np.array([float(np.array(rfu.get_rater_feedback_score(
        mean[i].reshape(1, 1, T, 2), np.array([[1.0]]), [RTRAJS[i]], [RSC[i]],
        np.array([ispd[i]]), frequency=4, length_seconds=5)["rater_feedback_score"]).ravel()[0])
        for i in range(N)])
    return b, tag, float(spearmanr(dis, -rfs).correlation)


def main():
    print(f"train={len(ego_tr)} rater={N}  tasks={B * len(FS)}", flush=True)
    res = {t: [None] * B for t in FS}
    tasks = [(b, t) for b in range(B) for t in FS]
    with Pool(NPROC, maxtasksperchild=4) as pool:
        for b, tag, rho in pool.imap_unordered(task, tasks):
            res[tag][b] = rho
            print(f"  b={b} {tag}={rho:+.3f}", flush=True)
    json.dump(res, open(OUT, "w"))
    print("\n=== distribution of rho over", B, "seed-sets ===", flush=True)
    for t in FS:
        a = np.array(res[t])
        print(f"  {t:8s} mean={a.mean():+.3f} sd={a.std():.3f} range[{a.min():+.3f},{a.max():+.3f}]", flush=True)
    ego, fr, su = (np.array(res[t]) for t in ("ego", "front", "surround"))
    print("\n=== paired (same seeds per instantiation) ===", flush=True)
    print(f"  front-ego    : meanD={float((fr - ego).mean()):+.3f} P(front>ego)={float((fr > ego).mean()):.2f}", flush=True)
    print(f"  surround-ego : meanD={float((su - ego).mean()):+.3f} P(surr>ego)={float((su > ego).mean()):.2f}", flush=True)
    print(f"  surround-front: meanD={float((su - fr).mean()):+.3f} P(surr>front)={float((su > fr).mean()):.2f}", flush=True)


if __name__ == "__main__":
    main()
