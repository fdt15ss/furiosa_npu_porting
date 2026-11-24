"""Run SDK 2026.5.0b3 `fxb build` for Qwen3.6-27B on one RNGD card (8 PE) without changing SDK files.

The patches apply only inside this process:
  - one chip: num_chip=1, no Chip hints, all_reduce becomes reshape, T=1 rotary tiled to T=2
  - SDK 2026.5 on the 2026.4 kernel package: furiosa.megakernels alias, kernel shim,
    compiler option filter, f32 g/beta/SSM to match the runtime
  - text only: vision reuses 256 patches, gdn_decode reuses batch 1, gdn_prefill reuses 128
"""
import importlib,inspect,os,pkgutil,re,sys
import furiosa.kernels.qwen3_5 as pk
from furiosa.tcl import Kernel
# SDK 2026.5 looks kernels up in furiosa.megakernels, which is not published; point it at furiosa.kernels.
import types
_mk=types.ModuleType('furiosa.megakernels'); _mk.__path__=[]; _mk.qwen3_5=pk
pk.COMPONENT_MODULES=('furiosa.kernels.qwen3_5',)
sys.modules['furiosa.megakernels']=_mk; sys.modules['furiosa.megakernels.qwen3_5']=pk
THR=100000
mods=[importlib.import_module('furiosa.kernels.qwen3_5.'+m.name) for m in pkgutil.iter_modules(pk.__path__) if not m.ispkg]
def set_default(fn,name,val):
    if not inspect.isfunction(fn) or not fn.__defaults__: return 0
    sg=inspect.signature(fn,follow_wrapped=False)
    pos=[q for q in sg.parameters.values() if q.default is not inspect._empty and q.kind in(q.POSITIONAL_ONLY,q.POSITIONAL_OR_KEYWORD)]
    d=list(fn.__defaults__);n=0
    for i,q in enumerate(pos):
        if q.name in name: d[i]=val;n+=1
    fn.__defaults__=tuple(d);return n
c=0
for mod in mods:
    if hasattr(mod,'NUM_CHIP') and not mod.__name__.endswith('qwen3_6_27b'): mod.NUM_CHIP=1
    if hasattr(mod,'DEFAULT_TOKEN_SHARD_THRESHOLD'): mod.DEFAULT_TOKEN_SHARD_THRESHOLD=THR
    for f in list(vars(mod).values()):
        if not inspect.isfunction(f): continue
        ch=[f]
        while hasattr(ch[-1],'__wrapped__'): ch.append(ch[-1].__wrapped__)
        for fn in ch:
            c+=set_default(fn,('num_chip','num_chips'),1); c+=set_default(fn,('token_shard_threshold',),THR)
print('patched defaults',c,file=sys.stderr)

def tile_rotary(s):
    """T=1 full_in: compute the rotary section at T=2, keep the first token."""
    s=s.replace('    rope_val: [Three, T, Dhr, Ry, Rx]/bf16 = gather(rope_table_logical, position_ids_logical, Re)\n',
     '    rope_val0: [Three, T, Dhr, Ry, Rx]/bf16 = gather(rope_table_logical, position_ids_logical, Re)\n    T2: Axis = 2\n    rope_val: [Three, T2, Dhr, Ry, Rx]/bf16 = concat([rope_val0, rope_val0])\n')
    s=s.replace('rope_val_il: [T, Dhr, Ry, Rx]','rope_val_il: [T2, Dhr, Ry, Rx]')
    s=s.replace('dpe.exec(%0,%1 -> [T, Dhr, Ry, Rx])\n        write.to([T, Dhr, Ry, Rx]/bf16)','dpe.exec(%0,%1 -> [T2, Dhr, Ry, Rx])\n        write.to([T2, Dhr, Ry, Rx]/bf16)')
    for suf,H in (('','Hq'),('_1','Hkv')):
        o,i=f'{H}_OUTER',f'{H}_INNER'
        b=re.search(rf'    xr_rot{suf}: .*?\n    \)\n',s,re.S).group(0)
        s=s.replace(b,f'''    xr_c{suf}t: [T2, {o}, {i}, Rx, Dhr]/bf16 = concat([xr_c{suf}, xr_c{suf}])
    xr_rot{suf}t: [T2, {o}, {i}, Ry, Dhr]/bf16 = tk.EinsumByDpe(
        %0 = read(xr_c{suf}t)
        %1 = read(rope_val_il)
        %2 = dpe.exec(%0,%1 -> [T2, {o}, {i}, Ry, Dhr])
        write.to([T2, {o}, {i}, Ry, Dhr]/bf16)
    )
    xr_rot{suf}: [T, {o}, {i}, Ry, Dhr]/bf16 = slice(xr_rot{suf}t, T2, 0)
''')
    return s
def strip_cfg(s):
    """Drop compiler options that the 2026.5 compiler no longer accepts."""
    import json
    def f(m):
        c=json.loads(m[1])
        for k in ('allow_hp_lowering','lowering_mode','apply_adaptive_einsum_by_pattern','enable_vrf_half_mode'): c.pop(k,None)
        if isinstance(c.get('partial_sum_policy'),dict): c['partial_sum_policy']=next(iter(c['partial_sum_policy']))
        return '@compiler_config('+json.dumps(c,indent=2)+')'
    return re.sub(r'@compiler_config\((\{.*?\n\})\)',f,s,flags=re.S)
def to_f32(s,names):
    """Retype the declaration of each named tensor from bf16 to f32 (first type on its line)."""
    for n in names:
        s=re.sub(rf'^(\s*{n}: [^=\n]*?)/bf16',r'\1/f32',s,flags=re.M)
    return s
def write_f32(s,n):
    """Retype a block tensor and its write.to line."""
    return re.sub(rf'(^\s*{n}: [^=\n]*?)/bf16( = tk\.\w+\(.*?write\.to\([^\n]*?)/bf16',r'\1/f32\2/f32',s,count=1,flags=re.S|re.M)
def gdn_f32(s):
    """SDK 2026.5 runtime passes g, beta and the SSM cache as f32; 2026.4 kernels declare bf16."""
    if 'def lin_in_broadcast<' in s:
        s=write_f32(write_f32(s,'beta_b'),'g_b')
        s=to_f32(s,('_t2','_t3','beta_b_1','g_b_1'))
        m=re.search(r'^\) -> .*$',s[s.index('def lin_in_broadcast<'):],re.M); hdr=m[0]; p=hdr.split('dram(')
        for i in (3,4): p[i]=p[i].replace('/bf16','/f32',1)
        s=s.replace(hdr,'dram('.join(p))
    if 'def gdn_decode<' in s:
        i=s.index('def gdn_decode<'); h,t=s[:i],s[i:]
        t=to_f32(t,('g','beta','ssm_cache','ssm_state','g_bh','beta_bh','ssm_bh','ssm_new','ssm_cache_new'))
        t=write_f32(t,'ssm_new_bh')
        t=re.sub(r'^(\s*_: \[Mc, Hv, Dhk, Dhv\])/bf16( = copy\(src=ssm_cache_new)',r'\1/f32\2',t,flags=re.M)
        s=h+t
    if 'def gated_delta_prefill<' in s:
        i=s.index('def gated_delta_prefill<'); h,t=s[:i],s[i:]
        t=to_f32(t,('g','beta','ssm_cache','state0_b','state_h','g_r','beta_r','snaps_hv','ssm_cache_new'))
        t=write_f32(t,'snap_state')
        t=t.replace('[Nc, Ho, Hi, Dhk, Dhv]/bf16','[Nc, Ho, Hi, Dhk, Dhv]/f32')
        t=re.sub(r'^(\s*_: \[Mc, Hv, Dhk, Dhv\])/bf16( = copy\(src=ssm_cache_new)',r'\1/f32\2',t,flags=re.M)
        s=h+t
    return s
def fix(s):
    s=re.sub(r'\s*@ context\(\s*operator = \{ Chip: \w+ \}\s*\)','',s)
    s=re.sub(r'\n\s*operator = \{ Chip: \w+ \},','',s)
    s=re.sub(r'= all_reduce\((\w+), \w+\)',r'= reshape(\1)',s)
    s=gdn_f32(strip_cfg(s))
    if 'def full_in_broadcast<' in s and re.search(r'def full_in_broadcast<[^>]*\bT=1\b',s): s=tile_rotary(s)
    return s
orig=Kernel.from_edsl.__func__
def from_edsl(cls,*a,**k):
    r=orig(cls,*a,**k)
    new=fix(r.dsl)
    if new!=r.dsl:
        object.__setattr__(r,'dsl',new)
    return r
Kernel.from_edsl=classmethod(from_edsl)

# Decorated entries hide their defaults, so force num_chip at the public names.
import functools
import furiosa.kernels.qwen3_5.qwen3_6_27b as _m
for _mod in (_m,pk):
    for _n in dir(_mod):
        _f=getattr(_mod,_n)
        if callable(_n) or not callable(_f) or not _n.startswith(('qwen3_6_27b_','qwen3_5_','gdn_','mlp','lin_','full_','lm_head')): continue
        try: _p=inspect.signature(_f).parameters
        except (TypeError,ValueError): continue
        if 'num_chip' in _p:
            def _wrap(f):
                @functools.wraps(f)
                def g(*a,**k):
                    k['num_chip']=1
                    return f(*a,**k)
                return g
            setattr(_mod,_n,_wrap(_f))

# gdn_prefill above 128 does not compile on one chip, so those buckets reuse 128. The attention
# buckets are capped at 128, so the engine splits longer prompts and never selects them.
if True:
    import furiosa.kernels.qwen3_5.qwen3_6_27b as m
    GM=int(os.environ.get("GDN_MAX_SEQ","128"))
    def _gdn(f):
        @functools.wraps(f)
        def g(seq_len,*a,**k):
            return f(min(seq_len,GM),*a,**k)
        return g
    for _mod in (m,pk):
        for n in dir(_mod):
            if n.endswith('txt_gdn_prefill'): setattr(_mod,n,_gdn(getattr(_mod,n)))
# Text only: all vision buckets reuse 256 patches and gdn_decode reuses batch 1.
# Images above 256 patches and decode batches above 1 will not work.
if True:
    import furiosa.kernels.qwen3_5.qwen3_6_27b as m
    def _v(f):
        @functools.wraps(f)
        def g(batch_size,num_patches,*a,**k): return f(batch_size,256,*a,**k)
        return g
    def _d(f):
        @functools.wraps(f)
        def g(batch_size,*a,**k): return f(1,*a,**k)
        return g
    for _mod in (m,pk):
        for n in dir(_mod):
            if '_vis_' in n and n.endswith('_block'): setattr(_mod,n,_v(getattr(_mod,n)))
            if n.endswith('txt_gdn_decode'): setattr(_mod,n,_d(getattr(_mod,n)))
# SDK 2026.5 factories must return (Kernel, metadata).
if True:
    import json
    class _Src:
        def __init__(self,n): self.n=n
        def to_json(self,*a,**k): return json.dumps({'statement_provenance':[]})
    class _Meta:
        dependencies=()
    class _Shim:
        def __init__(self,k): self._k=k; self.source=_Src(k.name)
        def __getattr__(self,n):
            return getattr(self._k,n)
    def _tup(f):
        @functools.wraps(f)
        def g(*a,**k):
            return _Shim(f(*a,**k)),_Meta()
        return g
    for _n in dir(pk):
        _f=getattr(pk,_n)
        if _n.startswith('qwen3_6_27b_') and callable(_f): setattr(pk,_n,_tup(_f))
from furiosa_llm.cli.fxb_main import main
sys.exit(main())
