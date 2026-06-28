"""Generate a drive-stratified token list for the decisive P2c run: PER_LOG tokens from each of
LOGS_WANTED distinct driving logs, so the cluster-bootstrap CI (which resamples drives) is tight."""

import json
import os
from collections import defaultdict
from pathlib import Path

from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import SceneFilter, SceneLoader

DATA = Path(os.environ["OPENSCENE_DATA_ROOT"])
LOGS = DATA / "trainval_navsim_logs" / "trainval"
PER_LOG = int(os.environ.get("PER_LOG", "24"))
LOGS_WANTED = int(os.environ.get("LOGS_WANTED", "55"))

sf = SceneFilter(num_history_frames=4, num_future_frames=10, has_route=True, tokens=None)
loader = SceneLoader(LOGS, None, sf, SensorConfig.build_no_sensors())

bylog = defaultdict(list)
for t in loader.tokens:
    try:
        log = loader.get_scene_from_token(t).scene_metadata.log_name
    except Exception:
        continue
    if len(bylog[log]) < PER_LOG:
        bylog[log].append(t)
    full = [k for k, v in bylog.items() if len(v) >= PER_LOG]
    if len(full) >= LOGS_WANTED:
        break

chosen_logs = [k for k, v in bylog.items() if len(v) >= PER_LOG][:LOGS_WANTED]
toks = [t for k in chosen_logs for t in bylog[k][:PER_LOG]]
json.dump(toks, open(os.path.expanduser("~/p2c_tokens_list.json"), "w"))
with open(os.path.expanduser("~/p2c_tokens.txt"), "w") as f:
    f.write("[" + ",".join(toks) + "]")
print(f"[gen] tokens={len(toks)} logs={len(chosen_logs)}")
