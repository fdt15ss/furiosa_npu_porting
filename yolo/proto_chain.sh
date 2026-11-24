#!/bin/bash
# Waits for the D-FINE parts, then compiles the Proto26 graph.
cd /root/furiosa_study/yolo
PY=/root/miniconda3/envs/furiosa_yolo/bin/python
until grep -q 'parts done' parts_chain.log; do sleep 20; done
tag=yolo26n-seg-protohead; echo "$(date +%T) start $tag"; timeout 14400 $PY -u compile_proto.py > $tag.compile.log 2>&1 & c=$!; while kill -0 $c 2>/dev/null; do echo "$(date +%T) $(( $(cat /sys/fs/cgroup/memory.current)/1073741824 ))GiB" >> $tag.mem.log; sleep 60; done; wait $c; echo "$(date +%T) $tag exit $?"
echo "$(date +%T) proto done"
