"""Teacher-forced CPU check of NPU text answers. Usage: [FA8=1] cpu_text_check.py <model> <npu.json>
FA8=1 also quantizes the input of every FP8-weight linear to FP8 per token and 128-wide block (scale = amax / 448),
like the NPU w8fa8fd kernels, so the CPU sees the same activation rounding.
For each prompt, one CPU pass over prompt + NPU tokens; prints positions where the CPU top token differs and the logit gap."""
import json
import os
import sys
import time

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

from eval_prompts import MAX_TOKENS, PROMPTS

model_dir, npu_path = sys.argv[1:3]
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
fp8_modules = set()
for name, _ in model.named_parameters():
    by_shard.setdefault(index[key(name)], []).append(name)
for shard, names in by_shard.items():
    with safe_open(os.path.join(model_dir, shard), "pt") as f:
        for name in names:
            k = key(name)
            t = f.get_tensor(k)
            if t.dtype == torch.float8_e4m3fn:
                fp8_modules.add(name.rpartition(".")[0])
                s = f.get_tensor(k[:-6] + "weight_scale_inv").float()
                t = t.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent), leaf, torch.nn.Parameter(t.to(torch.bfloat16), requires_grad=False))
print("loaded", flush=True)


def fa8(module, args):
    x = args[0]
    shape = x.shape
    xb = x.float().reshape(*shape[:-1], shape[-1] // 128, 128)
    scale = xb.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
    q = (xb / scale).to(torch.float8_e4m3fn).float() * scale
    return (q.reshape(shape).to(x.dtype),) + tuple(args[1:])


if os.environ.get("FA8") == "1":
    for name in fp8_modules:
        model.get_submodule(name).register_forward_pre_hook(fa8)
    print("FA8: quantizing inputs of", len(fp8_modules), "linears", flush=True)
model.eval()
npu_all = json.load(open(npu_path))
for name, prompt in PROMPTS.items():
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True, enable_thinking=False)["input_ids"][0].tolist()
    npu = npu_all[name]["tokens"]
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids + npu])).logits[0, len(ids) - 1:-1].float()
    worst, agree = 0.0, 0
    print(f"{name}: {len(ids)} prompt tokens, {len(npu)} NPU tokens", flush=True)
    for i, t in enumerate(npu):
        row = logits[i]; top = int(row.argmax())
        if top == t:
            agree += 1; continue
        gap = float(row[top] - row[t]); worst = max(worst, gap); rank = int((row > row[t]).sum())
        print(f"  pos {i}: NPU {tok.decode([t])!r} (rank {rank}), CPU top {tok.decode([top])!r}, gap {gap:.3f} (CPU top logit {float(row[top]):.3f})", flush=True)
    print(f"  agree {agree}/{len(npu)}, largest gap {worst:.3f}", flush=True)
