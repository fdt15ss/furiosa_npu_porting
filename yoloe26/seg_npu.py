"""YOLOE-26 segmentation on two NPU graphs: the network and Proto26. CPU runs NMS and mask assembly.

usage: seg_npu.py check|files|fps  (run from this folder)
"""
import glob
import multiprocessing as mp
import pathlib
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils._pytree import tree_leaves
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.plotting import Annotator, colors

import furiosa.torch as ft
from furiosa.torch._C import ir
from furiosa.torch.export import ExportedProgramWeight, PASSES, make_pass_context

from compile_model import build
from compile_proto import proto_head
from detect_npu import MEDIA, load_npu, tensor

NAME = "yoloe-26n-seg"
RESULT = f"{MEDIA}/result"
DURATION = 10
letterbox = LetterBox(640, auto=False)

FEATS = [torch.zeros(1, 1, 640 // s, 640 // s) for s in (8, 16, 32)]


def post(model, head, boxes, scores, coef, proto):
    pred = head._inference({"boxes": boxes, "scores": scores, "feats": FEATS, "mask_coefficient": coef})
    det = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.7, nc=head.nc)[0]
    if not len(det):
        return det, None
    return det, ops.process_mask(proto[0], det[:, 6:], det[:, :4], (640, 640), upsample=True)


def load_proto(head_module, feats, dev):
    ep = torch.export.export(head_module, tuple(feats))
    ctx = make_pass_context({})
    for fx_pass in PASSES:
        ep = fx_pass(ep, ctx)
    inner = ir.Runnable.deserialize(open(f"{NAME}-protohead.edf", "rb").read())
    return ft.CompileModule(ft.Runnable(inner), ExportedProgramWeight(ep), output_reconstruction_indices=(0,)).to(dev)


def setup(core=6, threads=1):
    torch.set_num_threads(threads)
    model = build(NAME)
    _, x = tensor(cv2.imread(f"{MEDIA}/bus.jpg"), letterbox)
    cm, dev = load_npu(model, x, core, NAME)
    ph = proto_head(model)
    with torch.no_grad():
        feats = model(x)["feats"]
    pm = load_proto(ph, feats, dev)
    return model, model.model[-1], (cm, pm, ph), dev, x


def npu(nets, dev, x):
    """Returns boxes, scores, coef, proto on the CPU. The feats stay on the NPU device."""
    cm, pm, _ = nets
    boxes, scores, f0, f1, f2, coef = tree_leaves(cm(x.to(dev)))
    proto = pm(f0, f1, f2)
    proto = proto[0] if isinstance(proto, (list, tuple)) else proto
    return [t.to("cpu") for t in (boxes, scores, coef, proto)]


def cpu(model, nets, x):
    with torch.no_grad():
        o = model(x)
        return [o["boxes"], o["scores"], o["mask_coefficient"], nets[2](*o["feats"])]


def check():
    model, head, nets, dev, _ = setup()
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
    for img in ("bus", "zidane"):
        _, x = tensor(cv2.imread(f"{MEDIA}/{img}.jpg"), letterbox)
        n, c = npu(nets, dev, x), cpu(model, nets, x)
        print(img, "cos", [round(cos(a, b), 5) for a, b in zip(c, n)], flush=True)
        dc, mc = post(model, head, *c)
        dn, mn = post(model, head, *n)
        print(img, "cpu", [model.names[int(i)] for i in dc[:, 5]], "npu", [model.names[int(i)] for i in dn[:, 5]], flush=True)
        if len(dc) == len(dn) and len(dc):
            idx = torch.cdist(dc[:, :4], dn[:, :4]).argmin(1)
            m = dn[idx]
            a, b = mc.bool(), mn[idx].bool()
            iou = (a & b).sum((1, 2)).float() / (a | b).sum((1, 2)).clamp(min=1)
            print(f"{img} conf diff={(dc[:, 4] - m[:, 4]).abs().max():.4f} box diff={(dc[:, :4] - m[:, :4]).abs().max():.2f}px mask iou min={iou.min():.4f} mean={iou.mean():.4f}", flush=True)


def draw(model, frame, det, masks):
    h, w = frame.shape[:2]
    if len(det):
        full = ops.scale_masks(masks[None].float(), (h, w))[0] > 0.5
        for mask, c in zip(full, det[:, 5].int().tolist()):
            color = np.array(colors(c, True), dtype=np.float32)
            area = mask.numpy()
            frame[area] = (frame[area] * 0.55 + color * 0.45).astype(np.uint8)
    boxes = ops.scale_boxes((640, 640), det[:, :4].clone(), (h, w))
    a = Annotator(frame, line_width=2)
    for b, d in zip(boxes, det):
        a.box_label(b.tolist(), f"{model.names[int(d[5])]} {d[4]:.2f}", color=colors(int(d[5]), True))
    return a.result()


def files():
    model, head, nets, dev, _ = setup()

    def infer(frame):
        _, x = tensor(frame, letterbox)
        leaves = npu(nets, dev, x)
        return x, leaves, *post(model, head, *leaves)

    for path in sorted(glob.glob(f"{MEDIA}/*.jpg")):
        frame = cv2.imread(path)
        _, _, det, masks = infer(frame)
        cv2.imwrite(f"{RESULT}/{pathlib.Path(path).stem}_{NAME}_npu.jpg", draw(model, frame, det, masks))
        print(pathlib.Path(path).name, [model.names[int(c)] for c in det[:, 5]], [round(float(c), 2) for c in det[:, 4]], flush=True)
    for path in sorted(glob.glob(f"{MEDIA}/driving/*.mp4")):
        cap = cv2.VideoCapture(path)
        src, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
        stem = pathlib.Path(path).stem
        tmp, out = f"{RESULT}/.{stem}.{NAME}.tmp.mp4", f"{RESULT}/{stem}_{NAME}_npu.mp4"
        writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), src, (w, h))
        counts, tally, checks, n, start = [], {}, [], 0, time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            x, leaves, det, masks = infer(frame)
            if n % 30 == 0:
                ref = cpu(model, nets, x)
                c = min(float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)) for a, b in zip(ref, leaves))
                checks.append((c, len(post(model, head, *ref)[0]), len(det)))
            writer.write(draw(model, frame, det, masks))
            counts.append(len(det))
            for k in det[:, 5].int().tolist():
                tally[model.names[k]] = tally.get(model.names[k], 0) + 1
            n += 1
        total = time.time() - start
        cap.release()
        writer.release()
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
        pathlib.Path(tmp).unlink()
        print(f"{stem} {w}x{h} frames={n} fps={n / total:.1f} objects mean={np.mean(counts):.1f} max={max(counts)} zero_frames={counts.count(0)}", flush=True)
        print(f"  classes={dict(sorted(tally.items(), key=lambda kv: -kv[1]))}", flush=True)
        print(f"  cpu check n={len(checks)} min_cos={min(c[0] for c in checks):.5f} same_count={sum(1 for _, a, b in checks if a == b)}/{len(checks)}", flush=True)


def worker(core, mode, barrier, results):
    model, head, nets, dev, x = setup(core)
    for _ in range(5):
        npu(nets, dev, x)
    barrier.wait()
    count, start = 0, time.time()
    while time.time() - start < DURATION:
        leaves = npu(nets, dev, x)
        if mode == "full":
            post(model, head, *leaves)
        count += 1
    results.put(count / (time.time() - start))


def fps():
    ctx = mp.get_context("spawn")
    for cores, mode in [(1, "raw"), (2, "raw"), (4, "raw"), (8, "raw"), (1, "full"), (8, "full")]:
        barrier, results = ctx.Barrier(cores), ctx.Queue()
        procs = [ctx.Process(target=worker, args=(c, mode, barrier, results)) for c in range(cores)]
        for p in procs:
            p.start()
        per_core = [results.get(timeout=120) for _ in procs]
        for p in procs:
            p.join()
        print(f"npu {mode:4} cores={cores} total={sum(per_core):7.1f} fps  per_core={min(per_core):.1f}~{max(per_core):.1f}", flush=True)


if __name__ == "__main__":
    {"check": check, "files": files, "fps": fps}[sys.argv[1]]()
