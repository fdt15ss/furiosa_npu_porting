"""prune.py in.fxb out.fxb: keep only linear buckets (b1,s1) and (b1,s128); drop their EDFs."""
import zipfile,json,sys
src,dst=sys.argv[1:3]
zi=zipfile.ZipFile(src); m=json.loads(zi.read('manifest.json')); drop=set()
for k in ('txt_gdn_decode','txt_gdn_prefill'):
    keep=[]
    for e in m['kernels'][k]:
        la=e['bucket']['linear_attention']
        if (la['batch_size'],la['seq_len']) in ((1,1),(1,128)): keep.append(e)
        else: drop.add(e['file'])
    m['kernels'][k]=keep
with zipfile.ZipFile(dst,'w',zipfile.ZIP_STORED) as zo:
    for i in zi.infolist():
        if i.filename in drop: continue
        zo.writestr(i.filename, json.dumps(m,indent=2) if i.filename=='manifest.json' else zi.read(i.filename))
print('dropped',len(drop))
