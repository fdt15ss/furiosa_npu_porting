#!/bin/bash
# Run long compiles one at a time: the 80 GiB memory limit allows only one.
cd "$(dirname "$0")"
PY=/root/miniconda3/envs/furiosa_yolo/bin/python
L=/tmp/claude-0/-root-furiosa-study/333c6b6e-a121-448c-87b2-adae57540ce1/scratchpad/dfine/lib
run() { tag=$1; shift; echo "$(date +%T) start $tag"; timeout 14400 "$@" > $tag.compile.log 2>&1 & c=$!; while kill -0 $c 2>/dev/null; do echo "$(date +%T) $(( $(cat /sys/fs/cgroup/memory.current)/1073741824 ))GiB" >> $tag.mem.log; sleep 60; done; wait $c; echo "$(date +%T) $tag exit $?"; }
run dfine-nano-backbone $PY -u compile_dfine.py $L backbone
run dfine-nano-encoder $PY -u compile_dfine.py $L encoder
run yolo26n-seg-proto $PY -u compile_model.py yolo26n-seg proto
echo "$(date +%T) chain done"
