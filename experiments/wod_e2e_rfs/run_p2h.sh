#!/bin/bash
# P2h unattended pipeline: wait for pixel parse -> train K vision planners (GPU) -> RFS eval (wod env).
set -e
cd "$HOME/work"
OLDPY=/mnt/wod/home/dev_alfred_ai_app/miniconda3/envs/wod/bin/python

echo "[wrap] $(date) waiting for pixels.npz + parse to finish..."
until [ -f "$HOME/work/pixels.npz" ] && ! pgrep -f parse_pixels.py >/dev/null; do sleep 15; done
echo "[wrap] $(date) pixels ready -> training K=6 jointly fine-tuned planners on the L4"

PIX="$HOME/work/pixels.npz" OUTPRED="$HOME/work/vision_preds.npz" K=6 EPOCHS=4 BS=96 \
  python3 "$HOME/train_vision.py"

echo "[wrap] $(date) training done -> RFS eval in the wod env"
PRED="$HOME/work/vision_preds.npz" PIX="$HOME/work/pixels.npz" OUT="$HOME/work/p2h_out.json" \
  "$OLDPY" "$HOME/rfs_eval.py"

echo "[wrap] $(date) DONE"
