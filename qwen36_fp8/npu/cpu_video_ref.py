"""Greedy CPU reference for a video request saved by serve_vision.py (MM_DUMP).

Usage: cpu_video_ref.py <model_dir> <out.json> <mm_*.pkl> [<mm_*.pkl> ...]
Uses the exact prompt ids and pixel values the NPU server used, so differences come from the
NPU kernels, not from video decoding or resizing. Run with the furiosa02 env and stop the
NPU server first: the BF16 model needs about 55 GB of the 80 GiB pod limit.
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))


def load_model(model_dir):
    """BF16 model with FP8 blocks dequantized, as in cpu_ref.py, including the vision tower."""
    torch.set_num_threads(14)
    cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    with torch.device("meta"):
        model = mq.Qwen3_5ForConditionalGeneration(cfg)
    # Buffers created on meta must be rebuilt on CPU.
    model.model.language_model.rotary_emb = mq.Qwen3_5TextRotaryEmbedding(config=cfg.text_config)
    model.model.visual.rotary_pos_emb = mq.Qwen3_5VisionRotaryEmbedding(cfg.vision_config)
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    by_shard = {}
    for name, _ in model.named_parameters():
        by_shard.setdefault(index[name], []).append(name)
    for shard, names in by_shard.items():
        with safe_open(os.path.join(model_dir, shard), "pt") as f:
            for name in names:
                t = f.get_tensor(name)
                if t.dtype == torch.float8_e4m3fn:
                    s = f.get_tensor(name[:-6] + "weight_scale_inv").float()
                    s = s.repeat_interleave(128, 0).repeat_interleave(128, 1)
                    t = t.float() * s[: t.shape[0], : t.shape[1]]
                parent, _, leaf = name.rpartition(".")
                param = torch.nn.Parameter(t.to(torch.bfloat16), requires_grad=False)
                setattr(model.get_submodule(parent), leaf, param)
    left = [n for n, b in model.named_buffers() if b.is_meta]
    assert not left, left
    model.eval()
    print("loaded", flush=True)
    return model, tok


def build_inputs(path):
    """Prompt ids, M-RoPE token types and pixel values exactly as the NPU server used them."""
    d = pickle.load(open(path, "rb"))
    ids, types = list(d["input_ids"]), [0] * len(d["input_ids"])
    for u in sorted(d["updates"], key=lambda u: u["start"], reverse=True):
        ids[u["start"]:u["end"]] = list(u["tokens"])
        types[u["start"]:u["end"]] = [2 if m else 0 for m in u["embedding_mask"]]
    pv, grids = [], []
    for p in d["payloads"]:
        x = torch.from_numpy(np.ascontiguousarray(p["pixel_values"]).view(np.int16)).view(torch.bfloat16)
        pv.append(x.reshape(-1, x.shape[-1]))
        grids.append(list(p["grid_thw"]))
    inputs = dict(
        input_ids=torch.tensor([ids]),
        attention_mask=torch.ones(1, len(ids), dtype=torch.long),
        mm_token_type_ids=torch.tensor([types]),
        pixel_values_videos=torch.cat(pv),
        video_grid_thw=torch.tensor(grids),
    )
    return inputs, ids, grids


def main():
    model_dir, out_path, *dumps = sys.argv[1:]
    model, tok = load_model(model_dir)
    results = {}
    for path in dumps:
        inputs, ids, grids = build_inputs(path)
        start = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False, use_cache=True)
        tokens = out[0, len(ids):].tolist()
        name = os.path.basename(path)
        results[name] = {"prompt_tokens": len(ids), "grid_thw": grids, "tokens": tokens,
                         "text": tok.decode(tokens, skip_special_tokens=True),
                         "seconds": time.time() - start}
        print(name, len(ids), "prompt tokens", grids, repr(results[name]["text"]), flush=True)
        json.dump(results, open(out_path, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
