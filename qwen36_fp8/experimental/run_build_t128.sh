#!/bin/bash
# 128-token build. NOT RUN YET. Usage: run_build_t128.sh <fxb config dir> <compile cache dir> <output.fxb>
# The output still contains stub gdn_prefill s256/s512/s1024; remove them from manifest.json before loading.
cd "$(dirname "$0")"
export VIS_ALIAS=1 GDN_MAX_SEQ=128
exec /root/miniconda3/envs/furiosa02/bin/python fxb_single_card.py build -tp 8 -O O3 --max-model-len 2048 \
  --tokenwise-buckets 1,128 --attention-buckets '(1,128,128),(1,2048,1),(1,2048,128)' \
  --concurrency 2 --no-fast-fail --compiler-cache-dir "$2" --build-report --slow-kernel-threshold 60m "$1" "$3"
