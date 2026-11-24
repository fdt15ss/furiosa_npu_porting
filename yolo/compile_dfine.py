"""Compile the D-FINE nano backbone and encoder and save dfine-nano-encoder.edf.

grid_sampler is not supported by furiosa-torch 2026.3.0 or 2026.4.0, so the
decoder runs on the CPU.

Uses CompileModule.from_exported, which runs all SDK graph passes. decompose
alone leaves _assert_tensor_metadata nodes that the compiler rejects.

The stem uses max_pool_2x2 because max_pool2d with kernel 2 and stride 1 panics
the compiler ("empty transition cost table").

Three patches are required for decompose:
1. Disable torch_compilable_check, which adds .item() on tensors.
2. Use eager attention, which returns contiguous tensors.
3. Fuse BatchNorm2d into Conv2d, which removes mutable batch norm ops.
"""
import json
import sys
import time

import cv2
import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

import transformers.models.d_fine.modeling_d_fine as modeling
from transformers import DFineForObjectDetection
import furiosa.torch as ft

MODEL = "ustc-community/dfine-nano-coco"
IMAGE = "/root/furiosa_study/media/bus.jpg"


class BackbonePart(nn.Module):
    """Backbone and input projection only, for when the encoder does not compile."""

    def __init__(self, detector):
        super().__init__()
        self.backbone = detector.model.backbone
        self.encoder_input_proj = detector.model.encoder_input_proj

    def forward(self, x):
        return [proj(source) for proj, source in zip(self.encoder_input_proj, feature_maps(self.backbone, x))]


class EncoderPart(nn.Module):
    """Backbone and hybrid encoder, which run on the NPU."""

    def __init__(self, detector):
        super().__init__()
        self.backbone = detector.model.backbone
        self.encoder_input_proj = detector.model.encoder_input_proj
        self.encoder = detector.model.encoder

    def forward(self, x):
        proj = [layer(source) for layer, source in zip(self.encoder_input_proj, feature_maps(self.backbone, x))]
        return list(self.encoder(proj).last_hidden_state)


def run_decoder(detector, feats):
    """Decoder and box heads, which run on the CPU because grid_sampler is not supported."""
    with torch.no_grad():
        out = detector(pixel_values=None, inputs_embeds=feats[0], encoder_outputs=(list(feats),))
    return out.logits, out.pred_boxes


def max_pool_2x2(x):
    """F.max_pool2d(F.pad(x, (0, 1, 0, 1)), 2, stride=1) using only supported operators.

    max_pool2d with kernel 2 and stride 1 panics the compiler, and F.pad compiles
    only directly before a convolution.
    """
    b, c, h, w = x.shape
    col, row = torch.zeros(b, c, h, 1, device=x.device), torch.zeros(b, c, 1, w, device=x.device)
    right = torch.cat([x[:, :, :, 1:], col], 3)
    down = torch.cat([x[:, :, 1:, :], row], 2)
    both = torch.cat([down[:, :, :, 1:], col], 3)
    top = right + torch.relu(x - right)
    bottom = both + torch.relu(down - both)
    return bottom + torch.relu(top - bottom)


class Stem(nn.Module):
    """HGNetV2Embeddings.forward with max_pool_2x2."""

    def __init__(self, embedder):
        super().__init__()
        self.stem1, self.stem2a, self.stem2b = embedder.stem1, embedder.stem2a, embedder.stem2b
        self.stem3, self.stem4 = embedder.stem3, embedder.stem4

    def forward(self, x):
        e = self.stem1(x)
        h = self.stem2b(nn.functional.pad(self.stem2a(nn.functional.pad(e, (0, 1, 0, 1))), (0, 1, 0, 1)))
        return self.stem4(self.stem3(torch.cat([max_pool_2x2(e), h], 1)))


def fuse_conv_frozen_bn(conv, norm):
    """Same math as DFineFrozenBatchNorm2d.forward applied after conv."""
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups, bias=True, padding_mode=conv.padding_mode)
    with torch.no_grad():
        scale = norm.weight * (norm.running_var + 1e-5).rsqrt()
        shift = norm.bias - norm.running_mean * scale
        fused.weight.copy_(conv.weight * scale.reshape(-1, *[1] * (conv.weight.dim() - 1)))
        old_bias = conv.bias if conv.bias is not None else torch.zeros_like(scale)
        fused.bias.copy_(old_bias * scale + shift)
    return fused.eval()


def feature_maps(backbone, x):
    """Same as DFineConvEncoder.forward without the unused pixel mask resize."""
    return backbone.model(x).feature_maps


def build():
    modeling.torch_compilable_check = lambda *args, **kwargs: None
    model = DFineForObjectDetection.from_pretrained(MODEL, attn_implementation="eager").eval()
    for module in model.modules():
        names = list(module._modules)
        for a, b in zip(names, names[1:]):
            conv, norm = module._modules[a], module._modules[b]
            if isinstance(conv, nn.Conv2d) and isinstance(norm, nn.BatchNorm2d):
                module._modules[a] = fuse_conv_bn_eval(conv, norm)
                module._modules[b] = nn.Identity()
            elif isinstance(conv, nn.Conv2d) and isinstance(norm, modeling.DFineFrozenBatchNorm2d):
                module._modules[a] = fuse_conv_frozen_bn(conv, norm)
                module._modules[b] = nn.Identity()
    net = model.model.backbone.model
    net.embedder = Stem(net.embedder).eval()
    return model


def example_input():
    im = cv2.resize(cv2.imread(IMAGE), (640, 640))
    return torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 255


if __name__ == "__main__":
    # usage: compile_dfine.py <backbone|encoder> [nopruning]
    part = sys.argv[1]
    tag = f"dfine-nano-{part}" + ("-nopruning" if "nopruning" in sys.argv else "")
    config = ft.config.CompilerConfig()
    if "nopruning" in sys.argv:
        config.enable_tactic_pruning = False
    print("enable_tactic_pruning", config.enable_tactic_pruning, flush=True)
    detector, x = build(), example_input()
    model = {"backbone": BackbonePart, "encoder": EncoderPart}[part](detector).eval()
    t = time.time(); ep = torch.export.export(model, (x,)); print(f"export ok {time.time() - t:.1f}s", flush=True)
    t = time.time(); compiled = ft.CompileModule.from_exported(ep, config); print(f"compile ok {time.time() - t:.1f}s", flush=True)
    meta = {"order": list(compiled.output_reconstruction_indices), "passthrough": len(compiled.passthrough_pairs), "zero_sized": compiled.zero_sized_tensor_info is not None}
    print("meta", meta, flush=True)
    with open(tag + ".json", "w") as f:
        json.dump(meta, f)
    with open(tag + ".edf", "wb") as f:
        f.write(compiled.runnable._inner.serialize())
    print("saved", tag + ".edf", flush=True)
