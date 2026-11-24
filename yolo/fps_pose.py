"""Measure YOLO pose FPS on 1/2/4/8 NPU cores, with one process per core.

raw:  input transfer + NPU inference + output transfer
full: raw + CPU box/keypoint decode + NMS
Preprocessing (image decode, resize) is excluded. Run from this folder.
"""
import multiprocessing as mp
import time

import cv2
import numpy as np
import torch
from torch.utils._pytree import tree_leaves

DURATION = 10
IMAGE = "/root/furiosa_study/media/bus.jpg"
RUNS = [(1, "raw"), (2, "raw"), (4, "raw"), (8, "raw"), (1, "full"), (8, "full")]


def build(name):
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox

    model = YOLO(name + ".pt").model
    model.fuse()
    model.eval()
    head = model.model[-1]
    is26 = type(head).__name__ == "Pose26"
    if is26:
        branch = head.one2one if head.end2end else head.one2many

        def raw_forward(x):
            p = head.forward_head(x, **branch)
            return {k: p[k] for k in ("boxes", "scores", "kpts")}

        head.forward = raw_forward
    else:
        head.training = True
    im = LetterBox(640, auto=False)(image=cv2.imread(IMAGE))
    x = torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 255
    return model, head, is26, x


def decode(head, is26, leaves):
    from ultralytics.utils.nms import non_max_suppression

    feats = [torch.zeros(1, 1, 640 // s, 640 // s) for s in (8, 16, 32)]
    head.training = False
    pred = head._inference({"boxes": leaves[0], "scores": leaves[1], "kpts": leaves[-1], "feats": feats})
    if not is26:
        head.training = True
    return non_max_suppression(pred, conf_thres=0.25, iou_thres=0.7, nc=1)[0]


def worker(name, core, mode, barrier, results):
    import furiosa.torch as ft
    from furiosa.torch._C import ir
    from furiosa.torch.export import ExportedProgramWeight, passes

    torch.set_num_threads(1)
    model, head, is26, x = build(name)
    dec = passes.decompose(torch.export.export(model, (x,)))
    inner = ir.Runnable.deserialize(open(name + ".edf", "rb").read())
    cm = ft.CompileModule(
        ft.Runnable(inner),
        ExportedProgramWeight(dec),
        output_reconstruction_indices=tuple(range(len(inner.output_dram_shapes))),
    )
    dev = torch.device(f"furiosa:{core}")
    cm = cm.to(dev)
    xd = x.to(dev)
    for _ in range(5):
        cm(xd)
    barrier.wait()
    count, start = 0, time.time()
    while time.time() - start < DURATION:
        leaves = [o.to("cpu") for o in tree_leaves(cm(xd))]
        if mode == "full":
            decode(head, is26, leaves)
        count += 1
    results.put(count / (time.time() - start))


def run_npu(name, cores, mode):
    ctx = mp.get_context("spawn")
    barrier, results = ctx.Barrier(cores), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(name, c, mode, barrier, results)) for c in range(cores)]
    for p in procs:
        p.start()
    per_core = [results.get(timeout=120) for _ in procs]
    for p in procs:
        p.join()
    print(f"{name} npu {mode:4} cores={cores} total={sum(per_core):7.1f} fps  per_core={min(per_core):.1f}~{max(per_core):.1f}", flush=True)


def run_cpu(name):
    model, head, is26, x = build(name)
    for mode in ("raw", "full"):
        count, start = 0, time.time()
        with torch.no_grad():
            model(x)
            while time.time() - start < DURATION:
                leaves = tree_leaves(model(x))
                if mode == "full":
                    decode(head, is26, leaves)
                count += 1
        print(f"{name} cpu {mode:4} threads={torch.get_num_threads()} total={count / (time.time() - start):7.1f} fps", flush=True)


if __name__ == "__main__":
    for name in ("yolo11n-pose", "yolo26n-pose"):
        for cores, mode in RUNS:
            run_npu(name, cores, mode)
        run_cpu(name)
