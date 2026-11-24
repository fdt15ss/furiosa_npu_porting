"""D-FINE nano inference: NPU backbone and encoder (one graph), CPU decoder.

usage: dfine_npu.py check|files|fps  (run from this folder)
"""
import glob
import json
import multiprocessing as mp
import pathlib
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics.utils.plotting import Annotator, colors
import furiosa.torch as ft
from furiosa.torch._C import ir
from furiosa.torch.export import (
    ExportedProgramWeight,
    PASSES,
    make_pass_context,
    remove_aliased_outputs,
    remove_zero_sized_outputs,
)

from compile_dfine import EncoderPart, build, example_input, run_decoder
from compile_dfine_parts import parts

MEDIA = "/root/furiosa_study/media"
RESULT = f"{MEDIA}/result"
NAME = "dfine-nano"
DURATION = 10
CONF = 0.4


def prepare(image):
    im = cv2.resize(image, (640, 640))
    return torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 255


def load_part(module, args, name, dev, outputs):
    ep = torch.export.export(module, args)
    ctx = make_pass_context({})
    for fx_pass in PASSES:
        ep = fx_pass(ep, ctx)
    ep, _ = remove_zero_sized_outputs(ep)
    ep, _ = remove_aliased_outputs(ep)
    inner = ir.Runnable.deserialize(open(f"{NAME}-{name}.edf", "rb").read())
    cm = ft.CompileModule(ft.Runnable(inner), ExportedProgramWeight(ep), output_reconstruction_indices=tuple(range(outputs)))
    return cm.to(dev)


class Pipeline:
    def __init__(self, core=6, threads=2, whole=True):
        torch.set_num_threads(threads)
        self.detector = build()
        self.p = parts(self.detector)
        x = example_input()
        with torch.no_grad():
            e = self.p["e"](x)
            s2 = self.p["a"](e)
            s3 = self.p["b"](s2)
        self.dev = torch.device(f"furiosa:{core}")
        self.whole = load_part(EncoderPart(self.detector).eval(), (x,), "encoder", self.dev, 2) if whole else None
        if not whole:
            self.e = load_part(self.p["e"], (x,), "parte", self.dev, 1)
            self.a = load_part(self.p["a"], (e,), "parta", self.dev, 1)
            self.b = load_part(self.p["b"], (s2,), "partb", self.dev, 1)
            self.c = load_part(self.p["c"], (s2, s3), "partc", self.dev, 2)
        self.times = {"npu": [], "decoder": []}

    def npu_part(self, x):
        t = time.time()
        if self.whole is not None:
            outputs = self.whole(x.to(self.dev))
        else:
            s2 = self.a(self.e(x.to(self.dev)))
            outputs = self.c(s2, self.b(s2))
        feats = [f.to("cpu") for f in outputs]
        self.times["npu"].append(time.time() - t)
        return feats

    def __call__(self, x):
        feats = self.npu_part(x)
        t = time.time()
        logits, boxes = run_decoder(self.detector, feats)
        self.times["decoder"].append(time.time() - t)
        return logits, boxes


def detections(logits, boxes, w, h, conf=CONF):
    scores, labels = logits.sigmoid()[0].max(-1)
    keep = scores >= conf
    b = boxes[0][keep]
    xyxy = torch.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1) * torch.tensor([w, h, w, h])
    return xyxy, scores[keep], labels[keep]


def check():
    pipe = Pipeline()
    names = pipe.detector.config.id2label
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
    for img in ("bus", "zidane"):
        frame = cv2.imread(f"{MEDIA}/{img}.jpg")
        h, w = frame.shape[:2]
        x = prepare(frame)
        with torch.no_grad():
            ref_feats = [t.clone() for t in parts_forward(pipe, x)]
        feats = pipe.npu_part(x)
        print(img, "feat cos", [round(cos(a, b), 5) for a, b in zip(ref_feats, feats)], flush=True)
        lc, bc = run_decoder(pipe.detector, ref_feats)
        ln, bn = run_decoder(pipe.detector, feats)
        dc, dn = detections(lc, bc, w, h), detections(ln, bn, w, h)
        print(img, "cpu", sorted(names[int(i)] for i in dc[2]), [round(float(s), 3) for s in sorted(dc[1], reverse=True)], flush=True)
        print(img, "npu", sorted(names[int(i)] for i in dn[2]), [round(float(s), 3) for s in sorted(dn[1], reverse=True)], flush=True)
        if len(dc[0]) == len(dn[0]) and len(dc[0]):
            idx = torch.cdist(dc[0], dn[0]).argmin(1)
            print(f"{img} conf diff={(dc[1] - dn[1][idx]).abs().max():.4f} box diff={(dc[0] - dn[0][idx]).abs().max():.2f}px", flush=True)


def parts_forward(pipe, x):
    e = pipe.p["e"](x)
    s2 = pipe.p["a"](e)
    s3 = pipe.p["b"](s2)
    return pipe.p["c"](s2, s3)


def draw(frame, det, names):
    a = Annotator(frame, line_width=2)
    for b, s, c in zip(*det):
        a.box_label(b.tolist(), f"{names[int(c)]} {s:.2f}", color=colors(int(c), True))
    return a.result()


def files():
    pipe = Pipeline()
    names = pipe.detector.config.id2label
    for path in sorted(glob.glob(f"{MEDIA}/*.jpg")):
        frame = cv2.imread(path)
        h, w = frame.shape[:2]
        det = detections(*pipe(prepare(frame)), w, h)
        cv2.imwrite(f"{RESULT}/{pathlib.Path(path).stem}_{NAME}_npu.jpg", draw(frame, det, names))
        print(pathlib.Path(path).name, [names[int(c)] for c in det[2]], [round(float(s), 2) for s in det[1]], flush=True)
    for path in sorted(glob.glob(f"{MEDIA}/driving/*.mp4")):
        cap = cv2.VideoCapture(path)
        src, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
        stem = pathlib.Path(path).stem
        tmp, out = f"{RESULT}/.{stem}.dfine.tmp.mp4", f"{RESULT}/{stem}_{NAME}_npu.mp4"
        writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), src, (w, h))
        counts, tally, n, start = [], {}, 0, time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            det = detections(*pipe(prepare(frame)), w, h)
            writer.write(draw(frame, det, names))
            counts.append(len(det[0]))
            for c in det[2].tolist():
                tally[names[c]] = tally.get(names[c], 0) + 1
            n += 1
        total = time.time() - start
        cap.release()
        writer.release()
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
        pathlib.Path(tmp).unlink()
        print(f"{stem} {w}x{h} frames={n} fps={n / total:.1f} objects mean={np.mean(counts):.1f} max={max(counts)} zero_frames={counts.count(0)}", flush=True)
        print(f"  classes={dict(sorted(tally.items(), key=lambda kv: -kv[1]))}", flush=True)
    print("ms per frame", {k: round(float(np.median(v)) * 1000, 2) for k, v in pipe.times.items()}, flush=True)


def worker(core, threads, whole, barrier, results):
    pipe = Pipeline(core, threads, whole)
    x = prepare(cv2.imread(f"{MEDIA}/bus.jpg"))
    for _ in range(3):
        pipe(x)
    for v in pipe.times.values():
        v.clear()
    barrier.wait()
    count, start = 0, time.time()
    while time.time() - start < DURATION:
        pipe(x)
        count += 1
    results.put((count / (time.time() - start), {k: float(np.median(v)) * 1000 for k, v in pipe.times.items()}))


def fps():
    ctx = mp.get_context("spawn")
    for whole, cores, threads in [(True, 1, 1), (True, 1, 2), (True, 2, 2), (True, 4, 2), (True, 8, 1), (True, 8, 2), (False, 8, 1)]:
        barrier, results = ctx.Barrier(cores), ctx.Queue()
        procs = [ctx.Process(target=worker, args=(c, threads, whole, barrier, results)) for c in range(cores)]
        for p in procs:
            p.start()
        got = [results.get(timeout=300) for _ in procs]
        for p in procs:
            p.join()
        print(f"npu whole={whole} cores={cores} threads={threads} total={sum(g[0] for g in got):6.1f} fps  ms={ {k: round(v, 1) for k, v in got[0][1].items()} }", flush=True)


if __name__ == "__main__":
    {"check": check, "files": files, "fps": fps}[sys.argv[1]]()
