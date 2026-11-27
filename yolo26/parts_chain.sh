#!/bin/bash
# Waits for the running chain, then compiles the D-FINE parts one at a time.
cd "$(dirname "$0")"
PY=/root/miniconda3/envs/furiosa_yolo/bin/python
L=/tmp/claude-0/-root-furiosa-study/333c6b6e-a121-448c-87b2-adae57540ce1/scratchpad/dfine/lib
until grep -q 'chain done' dfine_chain.log; do sleep 20; done
for p in a b c; do tag=dfine-nano-part$p; echo "$(date +%T) start $tag"; PYTHONPATH=. timeout 14400 $PY -u compile_dfine_parts.py $L $p > $tag.compile.log 2>&1 & c=$!; while kill -0 $c 2>/dev/null; do echo "$(date +%T) $(( $(cat /sys/fs/cgroup/memory.current)/1073741824 ))GiB" >> $tag.mem.log; sleep 60; done; wait $c; echo "$(date +%T) $tag exit $?"; done
echo "$(date +%T) parts done"
