"""YOLOE-26 detection helpers and FPS on 1/2/4/8 NPU cores. Run from this folder."""
import multiprocessing as mp
import time

import cv2
import numpy as np
import torch
from torch.utils._pytree import tree_leaves
from ultralytics.nn.modules.head import YOLOEDetect

from compile_model import build

NAME = "yoloe-26n-det"
MEDIA = "/root/programming/npu/media"
DURATION = 10


def tensor(img, letterbox):
    im = letterbox(image=img)
    return im, torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 255


def decode(head, leaves):
    from ultralytics.utils.nms import non_max_suppression

    feats = [torch.zeros(1, 1, 640 // s, 640 // s) for s in (8, 16, 32)]
    pred = YOLOEDetect._inference(head, {"boxes": leaves[0], "scores": leaves[1], "feats": feats})
    return non_max_suppression(pred, conf_thres=0.25, iou_thres=0.7, nc=head.nc)[0]


def load_npu(model, x, core, name=NAME):
    import furiosa.torch as ft
    from furiosa.torch._C import ir
    from furiosa.torch.export import ExportedProgramWeight, passes

    dec = passes.decompose(torch.export.export(model, (x,)))
    inner = ir.Runnable.deserialize(open(name + ".edf", "rb").read())
    cm = ft.CompileModule(
        ft.Runnable(inner),
        ExportedProgramWeight(dec),
        output_reconstruction_indices=tuple(range(len(inner.output_dram_shapes))),
    )
    dev = torch.device(f"furiosa:{core}")
    return cm.to(dev), dev


def worker(core, mode, barrier, results):
    from ultralytics.data.augment import LetterBox

    torch.set_num_threads(1)
    model = build(NAME)
    head = model.model[-1]
    _, x = tensor(cv2.imread(f"{MEDIA}/bus.jpg"), LetterBox(640, auto=False))
    cm, dev = load_npu(model, x, core)
    xd = x.to(dev)
    for _ in range(5):
        cm(xd)
    barrier.wait()
    count, start = 0, time.time()
    while time.time() - start < DURATION:
        leaves = [o.to("cpu") for o in tree_leaves(cm(xd))]
        if mode == "full":
            decode(head, leaves)
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


def cpu_fps():
    from ultralytics.data.augment import LetterBox

    model = build(NAME)
    head = model.model[-1]
    _, x = tensor(cv2.imread(f"{MEDIA}/bus.jpg"), LetterBox(640, auto=False))
    with torch.no_grad():
        model(x)
        for _ in range(3):
            count, start = 0, time.time()
            while time.time() - start < DURATION:
                decode(head, tree_leaves(model(x)))
                count += 1
            print(f"cpu full threads={torch.get_num_threads()} {count / (time.time() - start):.1f} fps", flush=True)


if __name__ == "__main__":
    fps()
    cpu_fps()
