#!/bin/bash
# Compile one DSL file with the SDK 2026.5 tcc. usage: tcc_one.sh <in.tcl>  -> <in>.edf, <in>.log
T=/root/miniconda3/envs/furiosa/bin/furiosa-tcc
s=$(date +%s); $T --stdin --target-npu renegade-8pe --output "${1%.tcl}.edf" --no-compiler-cache < "$1" > "${1%.tcl}.log" 2>&1
echo "$1 rc $? $(( $(date +%s) - s ))s $(grep -m1 -oE 'failed to lower[^:]*|Compilation failed' "${1%.tcl}.log")"
