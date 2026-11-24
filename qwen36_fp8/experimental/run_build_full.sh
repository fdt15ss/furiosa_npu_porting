#!/bin/bash
cd "$1"
exec /root/miniconda3/envs/furiosa02/bin/python fxb_single_card.py build -tp 8 -O O0 --max-model-len 2048 --concurrency 3 --no-fast-fail --compiler-cache-dir "$1/cc2" --build-report --slow-kernel-threshold 60m "$1/models/fxbcfg" "$1/build/logics-swe-1card-full.fxb"
