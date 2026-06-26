#!/bin/bash
# Convert the 16k training vision_tower.pth into ViQ weights and verify (reconstruction).
# Run:  bash run_convert.sh
#   IN_CKPT=/path/to/vision_tower.pth bash run_convert.sh    # override the source checkpoint

# Optional: set http(s)_proxy if your machine needs a proxy to fetch the
# verification images. Leave unset otherwise.
# export http_proxy=...
# export https_proxy=...

# make llava_viq (for the VAE decoder) importable -- auto-located, not hard-coded.
# This script lives at <repo>/viq_inference/converter/run_convert.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export VIQ_ROOT="$REPO_ROOT/viq_train"
export PYTHONPATH="$VIQ_ROOT:$PYTHONPATH"

cd "$SCRIPT_DIR"

# Source training checkpoint (override via the IN_CKPT env var).
IN_CKPT="${IN_CKPT:-/path/to/logs/viq_fsq16k/vision_tower.pth}"

# converted weights are written here (vision_tower / embedder.pth / index_drawer.pth)
OUT_DIR="$SCRIPT_DIR/converted"
mkdir -p "$OUT_DIR"

python3 convert_weight.py \
    --in_ckpt "$IN_CKPT" \
    --out_dir "$OUT_DIR" \
    --out_name model_viq_fsq.pth \
    --levels 8 8 8 6 5
