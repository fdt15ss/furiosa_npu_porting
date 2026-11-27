"""usage: compile_fusion.py <fusion> [pad4]
Compile yoloe-26n-det for <fusion> fused PEs -> yoloe-26n-det-<fusion>pe[-pad4].edf

pad4: the input gets a 4th all-zero channel and the first conv gets a zero weight
for it to eliminate 3-channel input DMA bottlenecks on 2-PE fused devices.
"""
import sys
import time

from edf_skip import skip_if_edf_exists

if __name__ == "__main__" and len(sys.argv) >= 2:
    skip_if_edf_exists(f"yoloe-26n-det-{int(sys.argv[1])}pe{'-pad4' if 'pad4' in sys.argv[2:] else ''}.edf")

import torch
import furiosa.torch as ft
from furiosa.torch.export import passes

from compile_model import build, example_input


def pad_input(x, channels=4):
    return torch.cat([x, x.new_zeros(x.shape[0], channels - x.shape[1], *x.shape[2:])], 1)


def build_padded(name="yoloe-26n-det", channels=4):
    model = build(name)
    conv = model.model[0].conv
    w = conv.weight.data
    conv.weight = torch.nn.Parameter(torch.cat([w, w.new_zeros(w.shape[0], channels - w.shape[1], *w.shape[2:])], 1), requires_grad=False)
    conv.in_channels = channels
    return model


if __name__ == "__main__":
    fusion = int(sys.argv[1])
    pad = "pad4" in sys.argv
    print(ft.set_fusion(fusion), "pad4" if pad else "", flush=True)
    model = build_padded() if pad else build("yoloe-26n-det")
    x = pad_input(example_input()) if pad else example_input()
    t = time.time(); dec = passes.decompose(torch.export.export(model, (x,))); print(f"export+decompose {time.time() - t:.1f}s", flush=True)
    t = time.time(); rt = ft.compiler.compile(dec); print(f"compile ok {time.time() - t:.1f}s", flush=True)
    out = f"yoloe-26n-det-{fusion}pe{'-pad4' if pad else ''}.edf"
    with open(out, "wb") as f:
        f.write(rt.serialize())
    print("saved", out, flush=True)
