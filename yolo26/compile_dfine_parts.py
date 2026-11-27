"""Compile D-FINE as three NPU graphs without branches, then the CPU runs the decoder.

The graphs are small enough to compile in minutes:
  e: image -> embedder output
  a: embedder output -> stage2 output (stages 0 to 2)
  s0, s1, s2: single stages, used if `a` does not compile
  b: stage2 output -> stage3 output
  c: stage2 and stage3 outputs -> encoder outputs

usage: compile_dfine_parts.py <e|a|s0|s1|s2|b|c>
"""
import sys
import time

from edf_skip import skip_if_edf_exists

if __name__ == "__main__" and len(sys.argv) >= 2:
    skip_if_edf_exists(f"dfine-nano-part{sys.argv[1]}.edf")

import torch
from torch import nn

from compile_dfine import build, example_input
import furiosa.torch as ft


class Stages(nn.Module):
    def __init__(self, detector, first, last):
        super().__init__()
        self.stages = detector.model.backbone.model.encoder.stages[first:last]

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


class PartB(nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.stage = detector.model.backbone.model.encoder.stages[3]

    def forward(self, x):
        return self.stage(x)


class PartC(nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.proj = detector.model.encoder_input_proj
        self.encoder = detector.model.encoder

    def forward(self, s2, s3):
        return list(self.encoder([self.proj[0](s2), self.proj[1](s3)]).last_hidden_state)


def parts(detector):
    return {
        "e": detector.model.backbone.model.embedder.eval(),
        "a": Stages(detector, 0, 3).eval(),
        "s0": Stages(detector, 0, 1).eval(),
        "s1": Stages(detector, 1, 2).eval(),
        "s2": Stages(detector, 2, 3).eval(),
        "b": PartB(detector).eval(),
        "c": PartC(detector).eval(),
    }


if __name__ == "__main__":
    which = sys.argv[1]
    detector, x = build(), example_input()
    p = parts(detector)
    with torch.no_grad():
        e = p["e"](x)
        h0 = p["s0"](e)
        h1 = p["s1"](h0)
        s2 = p["s2"](h1)
        s3 = p["b"](s2)
    args = {"e": (x,), "a": (e,), "s0": (e,), "s1": (h0,), "s2": (h1,), "b": (s2,), "c": (s2, s3)}[which]
    t = time.time(); ep = torch.export.export(p[which], args); print(f"export ok {time.time() - t:.1f}s", flush=True)
    t = time.time(); compiled = ft.CompileModule.from_exported(ep); print(f"compile ok {time.time() - t:.1f}s", flush=True)
    print("order", compiled.output_reconstruction_indices, "passthrough", compiled.passthrough_pairs, flush=True)
    with open(f"dfine-nano-part{which}.edf", "wb") as f:
        f.write(compiled.runnable._inner.serialize())
    print("saved", flush=True)
