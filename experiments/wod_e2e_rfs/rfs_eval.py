"""RFS evaluation of the jointly-trained vision ensemble (P2h). Runs in the `wod` env
(tf + waymo + scipy). Computes disagreement-vs-RFS for the full K-member ensemble and,
applying the P2g lesson, a member-bootstrap stability distribution over 4-of-K subsets.
"""
import itertools
import json
import os

import numpy as np
from scipy.stats import spearmanr

from waymo_open_dataset.metrics.python import rater_feedback_utils as rfu

PRED = os.environ["PRED"]
PIX = os.environ["PIX"]
OUT = os.environ["OUT"]
T = 20

p = np.load(PRED)
preds = p["preds"]  # [K, N, T, 2]
K, N = preds.shape[0], preds.shape[1]
meta = json.load(open(PIX + ".rater.json"))
d = np.load(PIX)
shard, spd = d["rater_shard"], d["rater_spd"]
rtrajs = [[np.array(t, float) for t in m["rtrajs"]] for m in meta]
rsc = [np.array(m["rscores"], float) for m in meta]
fut = [np.array(m["future"], float) for m in meta]


def disagree(P):
    k = len(P)
    return float(np.mean([np.mean(np.linalg.norm(P[a] - P[b], axis=1))
                          for a in range(k) for b in range(a + 1, k)]))


def rfs_of(mean, i):
    return float(np.array(rfu.get_rater_feedback_score(
        mean.reshape(1, 1, T, 2), np.array([[1.0]]), [rtrajs[i]], [rsc[i]],
        np.array([spd[i]]), frequency=4, length_seconds=5)["rater_feedback_score"]).ravel()[0])


rows = []
for i in range(N):
    P = preds[:, i]
    mean = P.mean(0)
    rows.append(dict(shard=int(shard[i]), dis=disagree(P), rfs=rfs_of(mean, i),
                     ade=float(np.mean(np.linalg.norm(mean - fut[i], axis=1)))))
dd = np.array([r["dis"] for r in rows])
rr = np.array([r["rfs"] for r in rows])
aa = np.array([r["ade"] for r in rows])
print(f"FULL-K(={K}) rho(dis,negRFS)={spearmanr(dd, -rr).correlation:+.3f}  "
      f"rho(ade,negRFS)={spearmanr(aa, -rr).correlation:+.3f}", flush=True)

subs = list(itertools.combinations(range(K), 4))
rhos = []
for sub in subs:
    Ps = preds[list(sub)]
    dis = np.array([disagree(Ps[:, i]) for i in range(N)])
    rfm = np.array([rfs_of(Ps[:, i].mean(0), i) for i in range(N)])
    rhos.append(float(spearmanr(dis, -rfm).correlation))
rhos = np.array(rhos)
print(f"SUBSET-4 stability: n={len(rhos)} mean rho={rhos.mean():+.3f} sd={rhos.std():.3f} "
      f"range[{rhos.min():+.3f},{rhos.max():+.3f}]", flush=True)

json.dump({"full": rows, "subset_rhos": rhos.tolist(), "K": int(K)}, open(OUT, "w"))
print(f"WROTE {OUT}", flush=True)
