#!/bin/bash
# Build the one-card FXB. Usage: ./build_fxb.sh [out.fxb] [compile_cache_dir] [--force]
# Prefill runs in GDN_MAX_SEQ-token chunks: 256 by default, 128 reproduces the 2026-11-07 FXB.
# 256 is about 33% faster on long prompts; s512 does not fit in the 80 GiB pod limit.
# Wall time from an empty cache: about 27 min for 128, 42 min for 256 (peak 25 GiB). With the cache about 8 min.
# Defaults to ../_fxb & compile/logics-swe-1card-sdk5-p256.fxb and ../_fxb & compile/compile-cache-sdk5 when present.
# Skips building when <out.fxb> already exists and is non-empty unless --force or FORCE_COMPILE=1 is set.
# The FXB is reproducible: a rebuild differs only in manifest.json created_at and uuid.
set -euo pipefail
HERE=$(cd "$(dirname "$0")"; pwd)
BUNDLE="$HERE/../_fxb & compile"
force=${FORCE_COMPILE:-0}
pos=()
for arg in "$@"; do
  if [ "$arg" = "--force" ]; then
    force=1
  else
    pos+=("$arg")
  fi
done
OUT=${pos[0]:-$BUNDLE/logics-swe-1card-sdk5-p256.fxb}
if [ -n "${pos[1]:-}" ]; then
  CACHE=${pos[1]}
elif [ -d "$BUNDLE/compile-cache-sdk5" ]; then
  CACHE="$BUNDLE/compile-cache-sdk5"
else
  CACHE="$(dirname "$OUT")/compile-cache-sdk5"
fi
if [ "$force" != "1" ] && [ -s "$OUT" ]; then
  size=$(wc -c < "$OUT" | tr -d ' ')
  echo "skip $OUT (already exists, ${size} bytes; pass --force or set FORCE_COMPILE=1 to rebuild)"
  exit 0
fi
mkdir -p "$(dirname "$OUT")" "$CACHE"
ENV=${FURIOSA_ENV:-/root/miniconda3/envs/furiosa}
export GDN_MAX_SEQ=${GDN_MAX_SEQ:-256}
# Vision buckets up to VIS_MAX patches per frame get real kernels (256 = 320x192 for 16:9, 1024 = about 640x360).
export VIS_MAX=${VIS_MAX:-256}
# Context length (default 2048). Attention buckets use the same length.
ML=${MAX_LEN:-2048}
# Decode batch sizes with real kernels (default 1). For example DEC_BATCHES=1,2,4 adds tokenwise buckets 2 and 4,
# decode attention buckets (2,ML,1) and (4,ML,1), and real gdn_decode b2/b4 (fxb_single_card.py DEC_MAX).
DEC_BATCHES=${DEC_BATCHES:-1}
export DEC_MAX=$(echo "$DEC_BATCHES" | tr , '\n' | sort -n | tail -1)
case $GDN_MAX_SEQ in
  128) TW=1,128;     ATT="(1,128,128),(1,$ML,1),(1,$ML,128)" ;;
  256) TW=1,128,256; ATT="(1,128,128),(1,256,256),(1,$ML,1),(1,$ML,128),(1,$ML,256)" ;;
  *) echo "GDN_MAX_SEQ must be 128 or 256" >&2; exit 1 ;;
esac
for b in $(echo "$DEC_BATCHES" | tr , ' '); do
  [ "$b" = 1 ] && continue
  TW=$(echo "$TW,$b" | tr , "\n" | sort -nu | paste -sd,); ATT="$ATT,($b,$ML,1)"
done
export PATH=$ENV/bin:$PATH   # the 2026.5 furiosa-tcc must come first; /usr/local/bin has 2026.4
cd "$HERE"
exec "$ENV/bin/python" fxb_single_card.py build -tp 8 -O O3 --max-model-len "$ML" \
  --tokenwise-buckets "$TW" --attention-buckets "$ATT" \
  --concurrency 3 --no-fast-fail --compiler-cache-dir "$CACHE" --build-report fxb_config "$OUT"
