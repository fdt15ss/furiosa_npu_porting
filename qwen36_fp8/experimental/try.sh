#!/bin/bash
# usage: try.sh tag tokenwise attention  -> builds from cache then attempts load
SP=/tmp/claude-0/-root-furiosa-study/73b6e61b-7d28-43ac-bbcb-9c4c54e13f63/scratchpad; cd $SP
tag=$1; tw=$2; at=$3
GDN_MAX=128 VIS_ALIAS=1 timeout 1500 /root/miniconda3/envs/furiosa02/bin/python fxb_single_card.py build -tp 8 -O O3 --max-model-len 2048 --tokenwise-buckets $tw --attention-buckets "$at" --concurrency 3 --no-fast-fail --compiler-cache-dir cc2 --build-report models/fxbcfg build/t_$tag.fxb > logs/t_$tag.build 2>&1
sed 's/\x1b\[[0-9;]*m//g' logs/t_$tag.build | grep -E 'total \(' | tail -1 | cut -c1-80
[ -s build/t_$tag.fxb ] || { echo NOFXB; exit; }
OMP_NUM_THREADS=8 timeout 600 /root/miniconda3/envs/furiosa02/bin/python npu_smoke.py /root/models/Logics-SWE-Qwen3.6-27B-FP8 build/t_$tag.fxb > logs/t_$tag.npu 2>&1
sed 's/\x1b\[[0-9;]*m//g' logs/t_$tag.npu | grep -E -A1 'panicked|loaded in|TEXT:|MATCHES_CPU' | cut -c1-200 | tail -6
