#!/bin/bash
cd /root/furiosa_study/yolo
PY=/root/miniconda3/envs/furiosa_yolo/bin/python
L=/tmp/claude-0/-root-furiosa-study/333c6b6e-a121-448c-87b2-adae57540ce1/scratchpad/dfine/lib
one() { tag=dfine-nano-part$1; echo "$(date +%T) start $tag"; timeout 7200 $PY -u compile_dfine_parts.py $L $1 > $tag.compile.log 2>&1; r=$?; echo "$(date +%T) $tag exit $r"; return $r; }
one e
if ! one a; then one s0; one s1; one s2; fi
echo "$(date +%T) parts2 done"
