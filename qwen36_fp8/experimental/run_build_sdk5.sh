#!/bin/bash
cd "$1"
exec /root/miniconda3/envs/furiosa/bin/python fxb_single_card.py build -tp 8 -O O3 --max-model-len 2048 --tokenwise-buckets 1,128 --attention-buckets '(1,128,128),(1,2048,1),(1,2048,128)' --concurrency 3 --no-fast-fail --compiler-cache-dir "$1/cc5" --build-report --slow-kernel-threshold 60m "$1/models/fxbcfg" "$1/build/logics-swe-1card-sdk5.fxb"
