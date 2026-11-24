"""Compare NPU and CPU token ids. Usage: compare_eval.py <npu.json> <cpu.json>"""
import json
import sys

npu, cpu = (json.load(open(path)) for path in sys.argv[1:3])
failed = 0
for name, ref in cpu.items():
    a, b = npu[name]["tokens"], ref["tokens"]
    n = min(len(a), len(b))
    same = next((i for i in range(n) if a[i] != b[i]), n)
    match = same == n and len(a) == len(b)
    failed += not match
    print(f"{name}: prompt {ref['prompt_tokens']} tokens, NPU {len(a)}, CPU {len(b)}, same prefix {same}, {'MATCH' if match else 'DIFF'}")
    if not match:
        print("  NPU:", repr(npu[name]["text"][:160]))
        print("  CPU:", repr(ref["text"][:160]))
sys.exit(1 if failed else 0)
