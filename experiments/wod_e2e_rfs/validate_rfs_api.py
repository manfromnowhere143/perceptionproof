import glob

import numpy as np
import tensorflow as tf

from waymo_open_dataset.metrics.python import rater_feedback_utils as rfu
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as e2e

T = 20
shard = sorted(glob.glob("/home/dev_alfred_ai_app/wod/val/val_*.tfrecord-00000-of-00093"))[0]
ds = tf.data.TFRecordDataset(shard)
f = e2e.E2EDFrame()
found = False
for rec in ds:
    f.ParseFromString(rec.numpy())
    valid = [t for t in f.preference_trajectories if len(t.pos_x) > 0 and t.preference_score >= 0]
    if valid:
        found = True
        break
print("found_rater_frame", found, flush=True)
rtrajs = [np.stack([np.array(t.pos_x[:T]), np.array(t.pos_y[:T])], axis=1).astype(np.float64) for t in valid]
rscores = np.array([t.preference_score for t in valid], dtype=np.float64)
fut = np.stack([np.array(f.future_states.pos_x[:T]), np.array(f.future_states.pos_y[:T])], axis=1).astype(np.float64)
spd = float((f.past_states.vel_x[-1] ** 2 + f.past_states.vel_y[-1] ** 2) ** 0.5)
out = rfu.get_rater_feedback_score(fut.reshape(1, 1, T, 2), np.array([[1.0]]), [rtrajs], [rscores], np.array([spd]))
print("RFS_KEYS", list(out.keys()), flush=True)
for k, v in out.items():
    print("VAL", k, np.array(v).ravel()[:3], flush=True)
