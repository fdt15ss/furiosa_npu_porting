"""Compile YOLOE-26 Proto26 alone as a second NPU graph and save yoloe-26n-seg-protohead.edf.

Compiling the whole segmentation model with Proto26 exceeds the 80 GiB memory
limit. As its own graph, it takes the feats output of yoloe-26n-seg.edf.
"""
import time

from edf_skip import skip_if_edf_exists

if __name__ == "__main__":
    skip_if_edf_exists("yoloe-26n-seg-protohead.edf")

import torch
import furiosa.torch as ft
from torch import nn

from compile_model import build, example_input, pixel_shuffle_upsample


class ProtoHead(nn.Module):
    def __init__(self, proto):
        super().__init__()
        self.proto = proto

    def forward(self, p3, p4, p5):
        return self.proto([p3, p4, p5], return_semantic=False)


def proto_head(model):
    proto = model.proto_tail
    proto.upsample = pixel_shuffle_upsample(proto.upsample)
    return ProtoHead(proto).eval()


if __name__ == "__main__":
    model, x = build("yoloe-26n-seg"), example_input()
    with torch.no_grad():
        feats = model(x)["feats"]
    head = proto_head(model)
    t = time.time(); ep = torch.export.export(head, tuple(feats)); print(f"export ok {time.time() - t:.1f}s", flush=True)
    t = time.time(); compiled = ft.CompileModule.from_exported(ep); print(f"compile ok {time.time() - t:.1f}s", flush=True)
    print("order", compiled.output_reconstruction_indices, "passthrough", compiled.passthrough_pairs, flush=True)
    with open("yoloe-26n-seg-protohead.edf", "wb") as f:
        f.write(compiled.runnable._inner.serialize())
    print("saved yoloe-26n-seg-protohead.edf", flush=True)
