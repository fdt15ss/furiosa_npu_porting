"""Convert a BF16 Qwen3.6-27B-structure safetensors checkpoint to block FP8 (E4M3, 128x128)."""
import json,os,shutil,sys,time,urllib.request,torch
from safetensors import safe_open
from safetensors.torch import save_file
src,dst=sys.argv[1:3]; B=128
off=json.loads(urllib.request.urlopen(urllib.request.Request('https://huggingface.co/Qwen/Qwen3.6-27B-FP8/raw/main/model.safetensors.index.json',headers={'User-Agent':'hf-check'}),timeout=30).read())['weight_map']
scaled={k[:-len('.weight_scale_inv')]+'.weight' for k in off if k.endswith('.weight_scale_inv')}
idx=json.load(open(os.path.join(src,'model.safetensors.index.json')))['weight_map']
miss=[k for k in idx if k not in off]; print('keys not in official',len(miss),miss[:5],flush=True)
if miss: sys.exit('structure mismatch')
torch.set_num_threads(12)
def q(w):
    r,c=w.shape
    if r%B or c%B: raise ValueError(f'shape not multiple of {B}: {w.shape}')
    f=w.float()
    if not torch.isfinite(f).all(): raise ValueError('nonfinite')
    a=f.abs().unfold(0,B,B).unfold(1,B,B).amax((-1,-2))
    s=(a/448).to(torch.bfloat16); s=torch.where(s>0,s,torch.ones_like(s))
    e=s.float().repeat_interleave(B,0).repeat_interleave(B,1)
    return (f/e).to(torch.float8_e4m3fn),s
wm={};n=0;t0=time.time();size=0
for sh in sorted(set(idx.values())):
    out={}
    with safe_open(os.path.join(src,sh),'pt') as f:
        for k in f.keys():
            t=f.get_tensor(k)
            if k in scaled:
                if t.ndim!=2: raise ValueError(k)
                qq,s=q(t); out[k]=qq; out[k[:-len('weight')]+'weight_scale_inv']=s; n+=1
                if not torch.isfinite(qq.float()).all(): raise ValueError('overflow '+k)
            else: out[k]=t
    p=os.path.join(dst,sh); save_file(out,p,metadata={'format':'pt'}); size+=os.path.getsize(p)
    for k in out: wm[k]=sh
    print(sh,'done',round(time.time()-t0),'s total',round(size/1e9,1),'GB',flush=True)
json.dump({'metadata':{'total_size':size},'weight_map':wm},open(os.path.join(dst,'model.safetensors.index.json'),'w'),indent=1)
print('quantized',n,'expected in checkpoint',len({k for k in idx if k in scaled}),flush=True)
for x in ('tokenizer.json','tokenizer_config.json','chat_template.jinja','generation_config.json','processor_config.json'):
    if os.path.exists(os.path.join(src,x)): shutil.copy(os.path.join(src,x),dst)
