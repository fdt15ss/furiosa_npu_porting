"""YOLO26n on fused NPU PEs: latency, throughput and CPU match.

usage: bench_fusion.py <fusion> <edf> [devices]
  fusion   PEs per logical device (1, 2, 4)
  edf      compiled model (yolo26n.edf is 1-PE, yolo26n-2pe.edf is 2-PE;
           a name with pad4 gets the 4-channel input from compile_fusion.py)
  devices  how many logical devices to run in parallel (default 1)

Prints single-stream latency (p50/p99) and FPS. With devices > 1, runs one
process per logical device and prints total FPS.
"""
import multiprocessing as mp
import statistics
import sys
import time

import cv2
import torch
from torch.utils._pytree import tree_leaves

from compile_model import build
from detect_npu import decode, tensor

MEDIA = "/root/programming/npu/media"
DURATION = 10


def load(fusion, edf, core):
    import furiosa.torch as ft
    from furiosa.torch._C import ir
    from furiosa.torch.export import ExportedProgramWeight, passes

    from compile_fusion import build_padded, pad_input
    from ultralytics.data.augment import LetterBox

    ft.set_fusion(fusion)
    pad = "pad4" in edf
    model = build_padded() if pad else build("yolo26n")
    _, x = tensor(cv2.imread(f"{MEDIA}/bus.jpg"), LetterBox(640, auto=False))
    if pad:
        x = pad_input(x)
    dec = passes.decompose(torch.export.export(model, (x,)))
    inner = ir.Runnable.deserialize(open(edf, "rb").read())
    cm = ft.CompileModule(
        ft.Runnable(inner),
        ExportedProgramWeight(dec),
        output_reconstruction_indices=tuple(range(len(inner.output_dram_shapes))),
    )
    dev = torch.device(f"furiosa:{core}")
    return model, x, cm.to(dev), dev


def check(model, x, cm, dev):
    head = model.model[-1]
    with torch.no_grad():
        ref = [t for t in tree_leaves(model(x))]
    out = [o.to("cpu") for o in tree_leaves(cm(x.to(dev)))]
    diff = max(float((a - b).abs().max()) for a, b in zip(ref, out))
    d_ref, d_npu = decode(head, ref), decode(head, out)
    same_cls = d_ref[:, 5].tolist() == d_npu[:, 5].tolist()
    box = float((d_ref[:, :4] - d_npu[:, :4]).abs().max()) if same_cls and len(d_ref) else float("nan")
    print(f"cpu match: raw maxdiff={diff:.4f} det cpu={len(d_ref)} npu={len(d_npu)} "
          f"same_classes={same_cls} box maxdiff={box:.2f}px", flush=True)


def worker(fusion, edf, core, barrier, results):
    torch.set_num_threads(1)
    _, x, cm, dev = load(fusion, edf, core)
    xd = x.to(dev)
    for _ in range(10):
        cm(xd)
    barrier.wait()
    count, start = 0, time.time()
    while time.time() - start < DURATION:
        [o.to("cpu") for o in tree_leaves(cm(xd))]
        count += 1
    results.put(count / (time.time() - start))


def single(fusion, edf):
    torch.set_num_threads(4)
    model, x, cm, dev = load(fusion, edf, 0)
    check(model, x, cm, dev)
    xd = x.to(dev)
    for _ in range(20):
        cm(xd)
    for label, sync in [("npu only", False), ("npu+copy", True)]:
        lat = []
        start = time.time()
        while time.time() - start < DURATION:
            t = time.perf_counter()
            out = tree_leaves(cm(xd))
            if sync:
                [o.to("cpu") for o in out]
            else:
                out[0].to("cpu")  # must sync somewhere; smallest output is enough
            lat.append((time.perf_counter() - t) * 1000)
        lat.sort()
        p99 = lat[int(len(lat) * 0.99)]
        print(f"fusion={fusion} {label:8} n={len(lat)} p50={statistics.median(lat):.2f}ms "
              f"p99={p99:.2f}ms fps={len(lat) / sum(lat) * 1000:.1f}", flush=True)


def parallel(fusion, edf, devices):
    ctx = mp.get_context("spawn")
    barrier, results = ctx.Barrier(devices), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(fusion, edf, c, barrier, results)) for c in range(devices)]
    for p in procs:
        p.start()
    per = [results.get(timeout=300) for _ in procs]
    for p in procs:
        p.join()
    print(f"fusion={fusion} devices={devices} ({fusion * devices} PE) total={sum(per):.1f} fps "
          f"per_device={min(per):.1f}~{max(per):.1f}", flush=True)


if __name__ == "__main__":
    fusion, edf = int(sys.argv[1]), sys.argv[2]
    devices = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    if devices == 1:
        single(fusion, edf)
    else:
        parallel(fusion, edf, devices)
