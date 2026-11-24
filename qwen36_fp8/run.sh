#!/bin/bash
# Usage: ./run.sh <work_dir>   needs about 90 GB free in work_dir and 60 GB RAM.
set -euo pipefail
W=${1:?work dir}; HERE=$(cd "$(dirname "$0")"; pwd)
PY=${PY:-/root/miniconda3/envs/furiosa02/bin/python}
REV=42c206e3eee5f771e1ad53ca201957c30f41bf4e
RAW=$W/raw; FP8=$W/fp8; mkdir -p "$RAW" "$FP8"; cd "$W"
# Xet stalled on the last shard here, so plain HTTP is forced.
HF_HOME=$W/hfhome HF_HUB_DISABLE_XET=1 "$PY" -I "$HERE/download.py" $REV "$RAW"
"$PY" -I "$HERE/verify_shards.py" $REV "$RAW" "$W/sha256.json"
"$PY" -I "$HERE/quant_fp8.py" "$RAW" "$FP8"
"$PY" -I "$HERE/make_config.py" "$RAW/config.json" "$FP8/config.json"
"$PY" -I "$HERE/check_fp8.py" "$FP8"
OMP_NUM_THREADS=14 "$PY" -I "$HERE/cpu_smoke.py" "$RAW" "$FP8" 48
# Official build config check only. Compiling needs the target NPU count in the README.
"$(dirname "$PY")/fxb" build --dry-run "$FP8" "$W/check.fxb"
