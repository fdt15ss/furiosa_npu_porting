"""usage: compile_model.py <yolo26n|yolo26n-seg>; compile raw predictions and save <name>.edf.

Uses head.training = True from the Furiosa compile notes and drops the `feats`
output, which post-processing does not use.

Export with a real preprocessed image. A build exported with torch.zeros matched
the CPU only on all-zero input and gave wrong results on real images.

The compiler does not implement ConvTranspose2d, and moving only that layer to
the CPU still ran out of the 80 GiB container memory limit. So by default the
whole Proto26 runs on the CPU as model.proto_tail, using the `feats` output.
The `proto` option instead keeps it on the NPU using Conv + PixelShuffle.
"""
import sys
import time

from edf_skip import skip_if_edf_exists

if __name__ == "__main__" and len(sys.argv) >= 2:
    skip_if_edf_exists(sys.argv[1] + ("-proto" if "proto" in sys.argv[2:] else "") + ".edf")

import cv2
import torch
from torch import nn
import furiosa.torch as ft
from furiosa.torch.export import passes
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

EXAMPLE_IMAGE = "/root/programming/npu/media/bus.jpg"


class PassThrough(nn.Module):
    def forward(self, x):
        return x


def example_input():
    im = LetterBox(640, auto=False)(image=cv2.imread(EXAMPLE_IMAGE))
    return torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 255


def pixel_shuffle_upsample(up):
    """Equivalent of ConvTranspose2d(c, c, 2, 2) using only operators the compiler supports."""
    conv = nn.Conv2d(up.in_channels, up.out_channels * 4, 1)
    with torch.no_grad():
        for out in range(up.out_channels):
            for y in range(2):
                for x in range(2):
                    conv.weight[out * 4 + y * 2 + x, :, 0, 0] = up.weight[:, out, y, x]
                    conv.bias[out * 4 + y * 2 + x] = up.bias[out]
    return nn.Sequential(conv, nn.PixelShuffle(2)).eval()


def build(name, proto_on_npu=False):
    model = YOLO(name + ".pt").model
    model.fuse()
    model.eval()
    head = model.model[-1]
    head.training = True
    is_seg = hasattr(head, "proto")
    if is_seg and proto_on_npu:
        head.proto.upsample = pixel_shuffle_upsample(head.proto.upsample)
    elif is_seg:
        model.__dict__["proto_tail"] = head.proto.eval()
        head.proto = PassThrough()
    forward = head.forward

    def raw_forward(x):
        preds = dict(forward(x))
        preds.pop("proto" if is_seg and not proto_on_npu else "feats")
        return preds

    head.forward = raw_forward
    return model


if __name__ == "__main__":
    # usage: compile_model.py <name> [proto]; proto keeps Proto26 on the NPU using PixelShuffle
    name = sys.argv[1]
    proto_on_npu = "proto" in sys.argv
    model = build(name, proto_on_npu)
    x = example_input()
    t = time.time(); ep = torch.export.export(model, (x,)); print(f"export ok {time.time() - t:.1f}s", flush=True)
    t = time.time(); dec = passes.decompose(ep); print(f"decompose ok {time.time() - t:.1f}s", flush=True)
    t = time.time(); rt = ft.compiler.compile(dec); print(f"compile ok {time.time() - t:.1f}s", flush=True)
    tag = name + ("-proto" if proto_on_npu else "")
    with open(tag + ".edf", "wb") as f:
        f.write(rt.serialize())
    print("saved", tag + ".edf", flush=True)
