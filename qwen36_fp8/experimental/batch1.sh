#!/bin/bash
cd "$1"; PY=/root/miniconda3/envs/furiosa02/bin/python; export TO=2400
run(){ timeout 2500 $PY kx/one.py "$@" 2>&1 | grep -aE ' rc |Traceback|Error' | tail -2 | cut -c1-260; }
run full_attention noctx_noar 1 1024 1 8192 &
run full_attention noctx_noar 1 2048 2047 8192 &
run gdn_decode noctx_noar 1 1024 &
run gated_delta_decode noctx_noar 1 1024 &
wait
run gdn_prefill noctx_noar 128 1024 &
run gated_delta_chunk noctx_noar 128 1024 1024 &
run full_in noctx_noar 1 8192 &
run random_sample noctx_noar 1 &
wait
echo BATCHDONE
