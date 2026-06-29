"""Extract front-camera pixels (for jointly-trained vision planner, P2h).

Reads the WOD-E2E val shards (mounted read-only) and saves, for the SAME train/rater
split family as P2e-g: front-camera images resized to 224x224 (uint8), ego_status(12),
future trajectory(40), and rater metadata. Output goes to the GPU VM's own disk so
training (torch) needs no tf/waymo. Parse uses the mounted `wod` env (tf+waymo).

No Waymo frames are redistributed; this writes only to the local VM disk.
"""
import glob
import io
import json
import os
from multiprocessing import Pool

import numpy as np

VAL_DIR = os.environ.get("VAL_DIR", "/mnt/wod/home/dev_alfred_ai_app/wod/val")
OUT = os.environ.get("OUT", "/home/jupyter/work/pixels.npz")
T, N_INTENT, N_CAM, FRONT = 20, 8, 8, 1
TRAIN_PER_SHARD = 120
NPROC = int(os.environ.get("NPROC", "6"))


def parse_shard(args):
    si, sh = args
    os.environ["OMP_NUM_THREADS"] = "1"
    import tensorflow as tf
    from PIL import Image
    from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as e2e
    tr_img, tr_ego, tr_Y, rater = [], [], [], []
    f = e2e.E2EDFrame()
    try:
        for rec in tf.data.TFRecordDataset(sh):
            f.ParseFromString(rec.numpy())
            ps, fs = f.past_states, f.future_states
            if len(ps.vel_x) == 0 or len(fs.pos_x) < T:
                continue
            front = next((im.image for im in f.frame.images if im.name == FRONT), None)
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
            tgt = np.stack([np.array(fs.pos_x[:T]), np.array(fs.pos_y[:T])], axis=1).astype(np.float32)
            valid = [t for t in f.preference_trajectories
                     if len(t.pos_x) >= T and t.preference_score >= 0]
            is_rater = bool(valid)
            if not is_rater and len(tr_img) >= TRAIN_PER_SHARD:
                continue
            img = np.asarray(Image.open(io.BytesIO(front)).convert("RGB").resize((224, 224)), np.uint8)
            if is_rater:
                rtrajs = [np.stack([np.array(t.pos_x[:T]), np.array(t.pos_y[:T])], axis=1).astype(np.float32)
                          for t in valid]
                rater.append(dict(img=img, ego=ego, init_speed=spd, future=tgt.tolist(),
                                  rtrajs=[r.tolist() for r in rtrajs],
                                  rscores=[float(t.preference_score) for t in valid], shard=si))
            else:
                tr_img.append(img)
                tr_ego.append(ego)
                tr_Y.append(tgt.reshape(-1))
    except Exception as ex:  # noqa: BLE001
        return si, None, None, None, None, f"err:{ex}"
    timg = np.stack(tr_img, 0) if tr_img else np.zeros((0, 224, 224, 3), np.uint8)
    return si, timg, np.array(tr_ego, np.float32), np.array(tr_Y, np.float32), rater, "ok"


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    shards = sorted(glob.glob(os.path.join(VAL_DIR, "val_*.tfrecord-*-of-00093")))
    shards = [s for s in shards if not s.endswith(".gstmp")]
    print(f"shards={len(shards)} nproc={NPROC}", flush=True)
    Timg, Tego, TY, rater = [], [], [], []
    with Pool(NPROC, maxtasksperchild=1) as pool:
        for si, timg, tego, ty, rf, status in pool.imap_unordered(parse_shard, list(enumerate(shards))):
            if status != "ok":
                print(f"  shard {si} {status}", flush=True)
                continue
            Timg.append(timg)
            Tego.append(tego)
            TY.append(ty)
            rater.extend(rf)
            print(f"  shard {si}: +train={len(timg)} +rater={len(rf)} tot_train={sum(len(x) for x in Timg)} tot_rater={len(rater)}", flush=True)
    train_img = np.concatenate(Timg, 0)
    train_ego = np.concatenate(Tego, 0)
    train_Y = np.concatenate(TY, 0)
    rater_img = np.stack([r["img"] for r in rater], 0)
    rater_ego = np.array([r["ego"] for r in rater], np.float32)
    rater_spd = np.array([r["init_speed"] for r in rater], np.float32)
    rater_shard = np.array([r["shard"] for r in rater], np.int32)
    print(f"TRAIN img={train_img.shape} ego={train_ego.shape} Y={train_Y.shape} RATER={len(rater)}", flush=True)
    np.savez(OUT, train_img=train_img, train_ego=train_ego, train_Y=train_Y,
             rater_img=rater_img, rater_ego=rater_ego, rater_spd=rater_spd, rater_shard=rater_shard)
    json.dump([{"future": r["future"], "rtrajs": r["rtrajs"], "rscores": r["rscores"]} for r in rater],
              open(OUT + ".rater.json", "w"))
    print(f"WROTE {OUT}", flush=True)


if __name__ == "__main__":
    main()
