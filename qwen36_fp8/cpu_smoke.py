"""CPU smoke test: raw BF16 vs FP8-dequantized logits, then greedy generation from FP8 weights."""
import json,os,sys,time,torch
from safetensors import safe_open
from transformers import AutoConfig,AutoTokenizer,Qwen3_5ForCausalLM
raw,fp8=sys.argv[1:3]; torch.set_num_threads(14)
cfg=AutoConfig.from_pretrained(raw,local_files_only=True).text_config
tok=AutoTokenizer.from_pretrained(raw,local_files_only=True)
with torch.device('meta'): m=Qwen3_5ForCausalLM(cfg)
print('params',sum(p.numel() for p in m.parameters())/1e9,flush=True)
m.model.rotary_emb=type(m.model.rotary_emb)(cfg)
meta=[n for n,b in m.named_buffers() if b.is_meta]; print('meta buffers',len(meta),meta[:4],flush=True)
if meta: raise SystemExit('meta buffers remain')
def key(n): return 'model.language_model.'+n[6:] if n.startswith('model.') else n
def load(d):
    idx=json.load(open(d+'/model.safetensors.index.json'))['weight_map']
    by={}
    for n,_ in m.named_parameters(): by.setdefault(idx[key(n)],[]).append(n)
    for sh,ns in by.items():
        with safe_open(os.path.join(d,sh),'pt') as f:
            for n in ns:
                k=key(n); t=f.get_tensor(k)
                if t.dtype==torch.float8_e4m3fn:
                    s=f.get_tensor(k[:-6]+'weight_scale_inv').float()
                    t=(t.float()*s.repeat_interleave(128,0).repeat_interleave(128,1)).to(torch.bfloat16)
                p,_,l=n.rpartition('.'); setattr(m.get_submodule(p),l,torch.nn.Parameter(t.to(torch.bfloat16),requires_grad=False))
msgs=[{'role':'user','content':'Write a Python function that reverses a string. Answer with only the code.'}]
ids=tok.apply_chat_template(msgs,add_generation_prompt=True,return_tensors='pt',return_dict=True,enable_thinking=False)['input_ids']
print('prompt tokens',ids.shape[1],flush=True)
def run(d):
    t=time.time(); load(d); print('loaded',d[-14:],round(time.time()-t),'s',flush=True)
    t=time.time()
    with torch.no_grad(): l=m(input_ids=ids,use_cache=False).logits[0,-1].float()
    print('forward',round(time.time()-t),'s top5',l.topk(5).indices.tolist(),flush=True); return l
m.tie_weights=lambda *a,**k:None
a=run(raw); b=run(fp8)
print('cosine',torch.nn.functional.cosine_similarity(a,b,dim=0).item(),'rmse',(a-b).pow(2).mean().sqrt().item(),'top1 match',bool(a.argmax()==b.argmax()),flush=True)
t=time.time()
with torch.no_grad(): o=m.generate(input_ids=ids,max_new_tokens=int(sys.argv[3]),do_sample=False,use_cache=True)
print('gen',round(time.time()-t),'s',flush=True); print('TEXT:',tok.decode(o[0,ids.shape[1]:],skip_special_tokens=False),flush=True); print('SMOKEDONE',flush=True)
