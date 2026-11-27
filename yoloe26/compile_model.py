"""usage: compile_model.py <yoloe-26n-det|yoloe-26n-seg> [proto]; compile raw predictions and save <name>.edf.

Fuses CLIP/MobileCLIP2 text prompt embeddings (COCO 80 classes by default, cached in
coco80_pe.pt) into the classification 1x1 convolutions via head.fuse(pe), then exports
raw predictions using a real preprocessed image (bus.jpg).

For segmentation, Proto26 uses ConvTranspose2d (unsupported by furiosa-torch) and
exceeds the 80 GiB container memory limit if compiled inside the main graph, so
Proto26 is split out via PassThrough and compiled separately by compile_proto.py.
"""
import pathlib
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
from ultralytics import YOLOE
from ultralytics.data.augment import LetterBox

ROOT = pathlib.Path(__file__).resolve().parent
WEIGHTS = str(ROOT / "yoloe-26n-seg.pt")
PE_CACHE = ROOT / "coco80_pe.pt"
EXAMPLE_IMAGE = "/root/programming/npu/media/bus.jpg"

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class PassThrough(nn.Module):
    def forward(self, x, return_semantic=False):
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


def load_coco80_pe(y):
    if PE_CACHE.exists():
        return torch.load(PE_CACHE, map_location="cpu", weights_only=True)
    pe = y.get_text_pe(COCO80).detach().cpu()
    tmp = PE_CACHE.with_suffix(".tmp.pt")
    torch.save(pe, tmp)
    tmp.replace(PE_CACHE)
    return pe


def build(name="yoloe-26n-seg", proto_on_npu=False):
    y = YOLOE(WEIGHTS)
    pe = load_coco80_pe(y)
    y.set_classes(COCO80, pe)
    model = y.model
    model.eval()
    model.fuse()
    head = model.model[-1]
    head.fuse(pe)
    for p in model.parameters():
        p.requires_grad_(False)
    head.training = True

    is_det_only = name == "yoloe-26n-det"
    if is_det_only:
        head.cv5 = None
        head.proto = PassThrough()
    elif proto_on_npu:
        head.proto.upsample = pixel_shuffle_upsample(head.proto.upsample)
    else:
        model.__dict__["proto_tail"] = head.proto.eval()
        head.proto = PassThrough()

    forward = head.forward

    def raw_forward(x):
        preds = dict(forward(x))
        preds.pop("proto" if is_det_only or not proto_on_npu else "feats")
        return preds

    head.forward = raw_forward
    return model


if __name__ == "__main__":
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
