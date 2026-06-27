#!/usr/bin/env bash
# Minimal NAVSIM data: maps + trainval metadata logs only (NO sensor blobs).
# Sufficient for ego-status agents. Respects dataset licenses (data not redistributed by this repo).
set -uo pipefail
DATA="$HOME/navsim_workspace/dataset"
mkdir -p "$DATA"; cd "$DATA"

# nuPlan maps (~1.4 GB)
if [ ! -d "$DATA/maps" ]; then
  wget -q https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip
  unzip -q nuplan-maps-v1.1.zip && rm -f nuplan-maps-v1.1.zip && mv nuplan-maps-v1.0 maps
fi

# trainval metadata logs (no sensors) — ~14 GB
if [ ! -d "$DATA/trainval_navsim_logs" ]; then
  wget -q https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz
  tar -xzf openscene_metadata_trainval.tgz && rm -f openscene_metadata_trainval.tgz
  mv openscene-v1.1/meta_datas trainval_navsim_logs && rm -rf openscene-v1.1
fi
du -sh "$DATA"/* 2>/dev/null
echo "[data] DONE"
