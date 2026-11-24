"""Write the patched full_in DSL for T tokens. usage: full_in_dsl.py <harness.py> <T> <out.tcl>"""
import os, sys
harness, T, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
src = open(harness).read().replace('from furiosa_llm.cli.fxb_main import main\nsys.exit(main())', '')
g = {'__name__': 'harness', '__file__': harness}
exec(compile(src, harness, 'exec'), g)
k, _ = g['pk'].qwen3_6_27b_w8fa8fdkv16_txt_full_in(T, 8192)
open(out, 'w').write(k.dsl)
print(out, len(k.dsl))
