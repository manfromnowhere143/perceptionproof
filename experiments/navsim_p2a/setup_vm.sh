#!/usr/bin/env bash
# Stand up the NAVSIM environment on a fresh Linux CPU VM. Idempotent.
set -uo pipefail
echo "==== [setup] $(date -u) ===="

# system lib for opencv (headless)
sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libgl1 libglib2.0-0 || true

# miniconda
if [ ! -d "$HOME/miniconda3" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$HOME/miniconda3"
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"
# accept conda channel Terms of Service (required by recent conda before env create)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# clone + env
mkdir -p "$HOME/navsim_workspace" && cd "$HOME/navsim_workspace"
[ -d navsim ] || git clone --depth 1 https://github.com/autonomousvision/navsim.git
cd navsim
conda env list | grep -q '^navsim' || conda env create --name navsim -f environment.yml
conda activate navsim && pip install -e . -q
python -c "import navsim; print('[setup] import navsim OK')"

# env vars for the data + experiment
cat > "$HOME/navsim_env.sh" <<EOF
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="\$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="\$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="\$HOME/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="\$HOME/navsim_workspace/dataset"
EOF
echo "==== [setup] DONE $(date -u) ===="
