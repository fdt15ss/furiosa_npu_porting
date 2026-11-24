"""Compile one Qwen3.6 kernel with a config variant. args: kernel variant arg..."""
import subprocess,sys,time,tempfile,os,re,importlib,json
import furiosa.kernels.qwen3_5.entry_function as ef, furiosa.kernels.qwen3_5.tp_config as tc, furiosa.kernels.qwen3_5.qwen3_6_27b as m
import inspect,pkgutil,furiosa.kernels.qwen3_5 as pk
for mi in pkgutil.iter_modules(pk.__path__):
    if mi.ispkg: continue
    mod=importlib.import_module('furiosa.kernels.qwen3_5.'+mi.name)
    if hasattr(mod,'NUM_CHIP') and not mod.__name__.endswith('qwen3_6_27b'): mod.NUM_CHIP=1
    for nm,fn0 in list(vars(mod).items()):
        if not inspect.isfunction(fn0): continue
        c=[fn0]
        while hasattr(c[-1],'__wrapped__'): c.append(c[-1].__wrapped__)
        for fn in c:
            if not inspect.isfunction(fn) or not fn.__defaults__: continue
            sg=inspect.signature(fn,follow_wrapped=False)
            pos=[q for q in sg.parameters.values() if q.default is not inspect._empty and q.kind in(q.POSITIONAL_ONLY,q.POSITIONAL_OR_KEYWORD)]
            d=list(fn.__defaults__)
            for i,q in enumerate(pos):
                if q.name in('num_chip','num_chips'): d[i]=1
            fn.__defaults__=tuple(d)
THR=int(os.environ.get('THR','0'))
if THR:
    ef.DEFAULT_TOKEN_SHARD_THRESHOLD=THR; m.DEFAULT_TOKEN_SHARD_THRESHOLD=THR
    for fn in vars(ef).values():
        if inspect.isfunction(fn) and fn.__defaults__ and 'token_shard_threshold' in inspect.signature(fn,follow_wrapped=False).parameters:
            sg=inspect.signature(fn,follow_wrapped=False); pos=[q for q in sg.parameters.values() if q.default is not inspect._empty and q.kind in(q.POSITIONAL_ONLY,q.POSITIONAL_OR_KEYWORD)]
            d=list(fn.__defaults__)
            for i,q in enumerate(pos):
                if q.name=='token_shard_threshold': d[i]=THR
            fn.__defaults__=tuple(d)
name,variant,*args=sys.argv[1:]; args=[int(a) for a in args]
base=tc._base_config
def vb():
    c=base()
    if 'nops' in variant: c.pop('partial_sum_policy',None)
    if 'nofus' in variant: c['enable_einsum_fusion']=False
    if 'prune' in variant: c['enable_tactic_pruning']=True
    return c
tc._base_config=vb
f=getattr(m,'qwen3_6_27b_w8fa8fdkv16_txt_'+name)
import inspect
kw={'num_chip':1} if 'num_chip' in inspect.signature(f).parameters else {}
if not kw:
    d=ef.lin_out.__defaults__
k=f(*args,**kw)
src=k.dsl
if 'noctx' in variant:
    src,n=re.subn(r'\s*@ context\(\s*operator = \{ Chip: \w+ \}\s*\)','',src); print('removed ctx',n,flush=True)
    if 'noar' in variant:
        src,n2=re.subn(r'= all_reduce\((\w+), \w+\)',r'= reshape(\1)',src); print('all_reduce->reshape',n2,flush=True)
open(f'kx/{name}_{variant}.tcl','w').write(src)
t=time.time()
r=subprocess.run(['/usr/local/bin/furiosa-tcc','--stdin','--target-npu','renegade-8pe','--output',f'kx/{name}_{variant}.edf','--no-compiler-cache'],input=src,text=True,capture_output=True,timeout=int(os.environ.get('TO','900')))
e=r.stderr+r.stdout
stages=re.findall(r'\[(\d+)/13\]',e)
msg=[l for l in e.splitlines() if re.search(r'^\s*\d: |Option::unwrap|Result::unwrap|error',l)][-2:]
print(name,variant,'rc',r.returncode,'stage',stages[-1] if stages else '?',round(time.time()-t,1),'s',' | '.join(x.strip()[:120] for x in msg),flush=True)
