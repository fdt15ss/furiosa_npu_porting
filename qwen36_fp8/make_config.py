"""Write the FP8 config.json: finetune config + official quantization_config, minus vision_token_id."""
import json,sys,urllib.request
raw,out=sys.argv[1:3]
q=json.loads(urllib.request.urlopen(urllib.request.Request('https://huggingface.co/Qwen/Qwen3.6-27B-FP8/raw/main/config.json',headers={'User-Agent':'hf-check'}),timeout=30).read())
c=json.load(open(raw)); c['quantization_config']=q['quantization_config']; c.pop('vision_token_id',None)
json.dump(c,open(out,'w'),indent=1); print('wrote',out)
