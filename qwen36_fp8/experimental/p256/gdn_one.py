"""Compile one txt_gdn_prefill kernel with the npu/fxb_single_card.py patches and SDK 2026.5 tcc.
usage: gdn_one.py <harness.py> <seq_len> <outdir> [chunk_size]
Loads the harness without running fxb main, lifts the 128 cap, and writes <outdir>/s<seq>.tcl, .edf, .log."""
import os, re, subprocess, sys, time
harness, seq, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
chunk = int(sys.argv[4]) if len(sys.argv) > 4 else None
src = open(harness).read()
src = src.replace('from furiosa_llm.cli.fxb_main import main\nsys.exit(main())', '')
src = src.replace('GM=128', 'GM=1<<30')  # older harness: no cap, compile the real seq_len
os.environ['GDN_MAX_SEQ'] = str(1 << 30)  # current harness reads the cap from the environment
assert 'fxb_main' not in src and ('GM=1<<30' in src or 'GDN_MAX_SEQ' in src)
g = {'__name__': 'harness', '__file__': harness}
exec(compile(src, harness, 'exec'), g)
pk = g['pk']
kw = {'chunk_size': chunk} if chunk else {}
k, _ = pk.qwen3_6_27b_w8fa8fdkv16_txt_gdn_prefill(seq, 1024, **kw)
dsl = k.dsl
import json
patch = json.loads(os.environ.get('CFG_PATCH', '{}'))  # merged into every @compiler_config of the kernel
if patch:
    def _m(mm):
        c = json.loads(mm[1]); c.update(patch)
        return '@compiler_config(' + json.dumps(c, indent=2) + ')'
    dsl, n = re.subn(r'@compiler_config\((\{.*?\n\})\)', _m, dsl, flags=re.S)
    print('patched compiler_config blocks', n, patch, flush=True)
tag = f's{seq}' + (f'_c{chunk}' if chunk else '') + os.environ.get('TAG', '')
os.makedirs(out, exist_ok=True)
open(f'{out}/{tag}.tcl', 'w').write(dsl)
print(tag, 'dsl bytes', len(dsl), 'f32 g/beta', bool(re.search(r'\bg: [^\n]*/f32', dsl)), flush=True)
tcc = os.path.join(os.path.dirname(sys.executable), 'furiosa-tcc')
t = time.time()
with open(f'{out}/{tag}.log', 'w') as lf:
    r = subprocess.run([tcc, '--stdin', '--target-npu', 'renegade-8pe', '--output', f'{out}/{tag}.edf',
                        '--no-compiler-cache'], input=dsl, text=True, stdout=lf, stderr=subprocess.STDOUT)
print(tag, 'rc', r.returncode, 'sec', round(time.time() - t), flush=True)
