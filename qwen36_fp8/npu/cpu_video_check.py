"""Score the NPU's answer with the CPU model to see how close each NPU token choice was.

Usage: cpu_video_check.py <model_dir> <mm_*.pkl> <npu_text_file> [<mm_*.pkl> <npu_text_file> ...]
The text file is video_client.py output; its last 3 lines (usage, blank, time) are dropped.
For every generated position this runs one teacher-forced CPU pass and prints the positions where
the CPU's top token differs from the NPU's, with the CPU logit gap between them. A gap of one or two
BF16 steps (0.125 near logit 20) points to rounding; a large gap points to a kernel error.
"""
import sys

import torch

from cpu_video_ref import build_inputs, load_model


def main():
    model_dir, *pairs = sys.argv[1:]
    model, tok = load_model(model_dir)
    for dump, text_file in zip(pairs[::2], pairs[1::2]):
        text = "".join(open(text_file).readlines()[:-3]).rstrip("\n")
        npu = tok.encode(text, add_special_tokens=False)
        inputs, ids, _ = build_inputs(dump)
        full = torch.tensor([ids + npu])
        inputs["input_ids"] = full
        inputs["attention_mask"] = torch.ones_like(full)
        inputs["mm_token_type_ids"] = torch.cat(
            [inputs["mm_token_type_ids"], torch.zeros(1, len(npu), dtype=torch.long)], 1)
        with torch.no_grad():
            logits = model(**inputs).logits[0, len(ids) - 1:-1].float()
        print(f"\n{dump}: {len(npu)} NPU tokens", flush=True)
        worst = 0.0
        for i, t in enumerate(npu):
            row = logits[i]
            top = int(row.argmax())
            if top != t:
                gap = float(row[top] - row[t])
                worst = max(worst, gap)
                rank = int((row > row[t]).sum())
                print(f"  pos {i}: NPU {tok.decode([t])!r} (rank {rank}), CPU top {tok.decode([top])!r}, "
                      f"gap {gap:.3f} (CPU top logit {float(row[top]):.3f})", flush=True)
        print(f"  agree {sum(int(logits[i].argmax()) == t for i, t in enumerate(npu))}/{len(npu)}, "
              f"largest gap {worst:.3f}", flush=True)


if __name__ == "__main__":
    main()
