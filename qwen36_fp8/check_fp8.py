import glob,json,os,struct,sys,urllib.request
d=sys.argv[1]
off=json.load(urllib.request.urlopen(urllib.request.Request('https://huggingface.co/Qwen/Qwen3.6-27B-FP8/raw/main/model.safetensors.index.json',headers={'User-Agent':'hf-check'})))['weight_map']
H={}
for f in glob.glob(d+'/*.safetensors'):
    with open(f,'rb') as x:
        n=struct.unpack('<Q',x.read(8))[0];h=json.loads(x.read(n));h.pop('__metadata__',None);H.update(h)
o={k for k in off if not k.startswith('mtp.')}
print('tensors',len(H),'official(no mtp)',len(o),'missing',len(o-set(H)),'extra',len(set(H)-o))
import collections
print(collections.Counter(v['dtype'] for v in H.values()))
print('fp8 without scale',sum(1 for k,v in H.items() if v['dtype']=='F8_E4M3' and k[:-6]+'weight_scale_inv' not in H))
print('size GB',round(sum(os.path.getsize(f) for f in glob.glob(d+'/*.safetensors'))/1e9,1))
