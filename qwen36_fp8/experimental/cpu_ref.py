"""Greedy CPU reference from the FP8 weights. Usage: cpu_ref.py <model> <out.json> [compare.json]"""
import json
import os
import sys
import time

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

from eval_prompts import MAX_TOKENS, PROMPTS

model_dir, out_path = sys.argv[1:3]
torch.set_num_threads(14)
cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True).text_config
tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
with torch.device("meta"):
    model = Qwen3_5ForCausalLM(cfg)
model.model.rotary_emb = type(model.model.rotary_emb)(cfg)
index = json.load(open(model_dir + "/model.safetensors.index.json"))["weight_map"]


def key(name):
    return "model.language_model." + name[6:] if name.startswith("model.") else name


by_shard = {}
for name, _ in model.named_parameters():
    by_shard.setdefault(index[key(name)], []).append(name)
for shard, names in by_shard.items():
    with safe_open(os.path.join(model_dir, shard), "pt") as f:
        for name in names:
            k = key(name)
            t = f.get_tensor(k)
            if t.dtype == torch.float8_e4m3fn:
                s = f.get_tensor(k[:-6] + "weight_scale_inv").float()
                t = t.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent), leaf, torch.nn.Parameter(t.to(torch.bfloat16), requires_grad=False))
print("loaded", flush=True)
model.eval()
results = {}
for name, prompt in PROMPTS.items():
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )["input_ids"]
    start = time.time()
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=MAX_TOKENS, do_sample=False, use_cache=True)
    tokens = out[0, ids.shape[1]:].tolist()
    results[name] = {"prompt_tokens": ids.shape[1], "tokens": tokens, "text": tok.decode(tokens), "seconds": time.time() - start}
    print(name, ids.shape[1], "prompt tokens,", len(tokens), "tokens", repr(results[name]["text"][:120]), flush=True)
    json.dump(results, open(out_path, "w"), ensure_ascii=False, indent=1)
