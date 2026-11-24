import zipfile,json,sys,glob
sys.path.insert(0,'pylib311')
from huggingface_hub import HfFileSystem
repo='furiosa-ai/Qwen3.6-27B-FP8'; name='qwen3-6-27b-fp8-5c8d4d6df9cdfc5328cb1d007eaa44233ec9e63d-e7189cf4f3-2611030849.fxb'
ours=zipfile.ZipFile('../build/logics-swe-1card-sdk5.fxb'); om=json.loads(ours.read('manifest.json'))
print('our format',om['format_version'],om.get('versions'))
print(json.dumps(om['kernels']['txt_mlp'])[:300])
def fname(m,k,pred):
    for e in m['kernels'][k]['entries']:
        if pred(e['buckets']): return e['file']
K=[('txt_lm_head',lambda b:b.get('tokenwise')==[1]),('txt_mlp',lambda b:b.get('tokenwise')==[1]),('txt_lin_in',lambda b:b.get('tokenwise')==[1]),('txt_full_out',lambda b:b.get('tokenwise')==[1]),('txt_full_in',lambda b:b.get('tokenwise')==[1])]
fs=HfFileSystem()
f=fs.open(f'{repo}/{name}','rb',block_size=2**20); off=zipfile.ZipFile(f); fm=json.loads(off.read('manifest.json'))



import io,struct,cbor2
def dec(b):
    n=struct.unpack('<Q',b[:8])[0]
    return cbor2.CBORDecoder(io.BytesIO(b[8:8+n])).decode()
def walk(x,dep=0,maxd=4):
    if isinstance(x,dict):
        for a,v in list(x.items())[:16]:
            print(' '*dep,repr(a)[:40],type(v).__name__,(len(v) if hasattr(v,'__len__') else v))
            if dep<maxd and not isinstance(v,(bytes,str)): walk(v,dep+1,maxd)
    elif isinstance(x,list) and x and dep<maxd: print(' '*dep,'[0]:'); walk(x[0],dep+1,maxd)

def sig(e):
    # find graph dict holding 'signature'
    st=[e]
    while st:
        x=st.pop()
        if isinstance(x,dict):
            if 'signature' in x and isinstance(x['signature'],dict) and 'inputs' in x['signature']: return x
            st+=[v for v in x.values() if isinstance(v,(dict,list))]
        elif isinstance(x,list): st+=x

def labels(x,out):
    if isinstance(x,dict):
        if 'tag' in x and 'size' in x:
            t=x['tag']; l=None
            if isinstance(t,dict):
                v=next(iter(t.values())); 
                l=(v.get('label',{}).get('inner') if isinstance(v,dict) and isinstance(v.get('label'),dict) else str(v))
            out.append(f"{l}:{x['size']}")
        else:
            for v in x.values(): labels(v,out)
    elif isinstance(x,list):
        for v in x: labels(v,out)
def rows(e):
    g=sig(e); out=[]
    for i,t in enumerate(g['signature']['inputs']):
        t0=t[0] if isinstance(t,list) else t
        L=[];labels(t0.get('original_tensor'),L)
        ref=g['inputs'][i] if i<len(g['inputs']) else None
        nm=None
        ph=g['nodes'] if False else None
        out.append((str(t0.get('element_type')),' '.join(L)[:70],len(t) if isinstance(t,list) else 1,str(ref)[:30]))
    return out

def outs(e):
    g=sig(e); r=[]
    for t in g['signature']['outputs']:
        t0=t[0] if isinstance(t,list) else t
        L=[];labels(t0.get('original_tensor'),L); r.append((str(t0.get('element_type')),' '.join(L)[:60]))
    return r
L1=lambda b:b.get('tokenwise')==[1]
K=[('txt_gdn_decode',lambda b:b.get('linear_attention')==[{'batch_size':1,'seq_len':1}]),('txt_gdn_prefill',lambda b:b.get('linear_attention')==[{'batch_size':1,'seq_len':128}]),('txt_lin_out',L1),('txt_lm_head',L1),('txt_mlp',L1),('txt_lin_in',L1),('txt_full_in',L1),('txt_full_out',L1),('txt_full_attention',lambda b:any(x.get('attention_size')==2048 and x.get('input_size')==1 and x.get('batch_size')==1 for x in b.get('attention',[])))]
K2=K
for k,p in K2:
    fa=fname(om,k,p); fb=fname(fm,k,p)
    if not fa or not fb:
        print('==',k,'missing',fa,fb,[e['buckets'] for e in (om['kernels'][k]['entries'])][:3]); continue
    ea,eb=dec(ours.read(fa)),dec(off.read(fb))
    a,b=rows(ea),rows(eb); oa,ob=outs(ea),outs(eb)
    d=[(i,a[i][:2],b[i][:2]) for i in range(min(len(a),len(b))) if a[i][:2]!=b[i][:2]]
    print('==',k,'in',len(a),len(b),'out',len(oa),len(ob),'in diffs',d)
    print('  out diffs',[(i,oa[i],ob[i]) for i in range(min(len(oa),len(ob))) if oa[i]!=ob[i]])
