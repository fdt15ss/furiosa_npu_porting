import sys,inspect,pkgutil,importlib
CH=int(__import__('os').environ.get('CHIPS','1'))
import furiosa.kernels.qwen3_5 as pk
mods=[importlib.import_module('furiosa.kernels.qwen3_5.'+m.name) for m in pkgutil.iter_modules(pk.__path__) if not m.ispkg]
n=0
for mod in mods:
    if hasattr(mod,'NUM_CHIP') and not mod.__name__.endswith('qwen3_6_27b'): mod.NUM_CHIP=CH
    for name,f in vars(mod).items():
        if not inspect.isfunction(f): continue
        cand=[f]
        while hasattr(cand[-1],'__wrapped__'): cand.append(cand[-1].__wrapped__)
        for fn in cand:
            if not inspect.isfunction(fn): continue
            sig=inspect.signature(fn,follow_wrapped=False); p=sig.parameters.get('num_chip') or sig.parameters.get('num_chips')
            if p is None or p.default is inspect._empty: continue
            if fn.__defaults__:
                pos=[q for q in sig.parameters.values() if q.default is not inspect._empty and q.kind in(q.POSITIONAL_ONLY,q.POSITIONAL_OR_KEYWORD)]
                d=list(fn.__defaults__)
                for i,q in enumerate(pos):
                    if q.name in('num_chip','num_chips'): d[i]=CH;n+=1
                fn.__defaults__=tuple(d)
            if fn.__kwdefaults__:
                for k in('num_chip','num_chips'):
                    if k in fn.__kwdefaults__: fn.__kwdefaults__[k]=CH;n+=1
print('patched defaults',n,'chips',CH,file=sys.stderr)
from furiosa_llm.cli import fxb_main
for a in ('main','fxb_main','cli'):
    if hasattr(fxb_main,a): sys.exit(getattr(fxb_main,a)())
print(dir(fxb_main))
