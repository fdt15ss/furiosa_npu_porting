"""Run YOLOE-26 detection on media/*.jpg and media/driving/*.mp4; writes annotated files to media/result."""
import glob
import pathlib
import subprocess
import time

import cv2
import numpy as np
import torch
from torch.utils._pytree import tree_leaves
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.plotting import Annotator, colors

from compile_model import build
from detect_npu import MEDIA, NAME, decode, load_npu, tensor

RESULT = f"{MEDIA}/result"
torch.set_num_threads(1)
model = build(NAME)
head = model.model[-1]
letterbox = LetterBox(640, auto=False)
_, x0 = tensor(cv2.imread(f"{MEDIA}/bus.jpg"), letterbox)
cm, dev = load_npu(model, x0, 6)


def npu(x):
    return [o.to("cpu") for o in tree_leaves(cm(x.to(dev)))]


def cpu(x):
    with torch.no_grad():
        return tree_leaves(model(x))


def detect(frame):
    h, w = frame.shape[:2]
    _, x = tensor(frame, letterbox)
    leaves = npu(x)
    det = decode(head, leaves)
    det[:, :4] = ops.scale_boxes((640, 640), det[:, :4], (h, w))
    return x, leaves, det


def draw(frame, det):
    a = Annotator(frame, line_width=2)
    for d in det:
        c = int(d[5])
        a.box_label(d[:4].tolist(), f"{model.names[c]} {d[4]:.2f}", color=colors(c, True))
    return a.result()


def classes(det):
    return [model.names[int(c)] for c in det[:, 5]]


for path in sorted(glob.glob(f"{MEDIA}/*.jpg")):
    frame = cv2.imread(path)
    x, leaves, det = detect(frame)
    out = f"{RESULT}/{pathlib.Path(path).stem}_{NAME}_npu.jpg"
    cv2.imwrite(out, draw(frame, det))
    print(pathlib.Path(path).name, "objects", classes(det), "confs", [round(float(c), 2) for c in det[:, 4]], flush=True)

for path in sorted(glob.glob(f"{MEDIA}/driving/*.mp4")):
    cap = cv2.VideoCapture(path)
    src_fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
    stem = pathlib.Path(path).stem
    tmp, out = f"{RESULT}/.{stem}.{NAME}.tmp.mp4", f"{RESULT}/{stem}_{NAME}_npu.mp4"
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
    counts, tally, checks, n, start = [], {}, [], 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        x, leaves, det = detect(frame)
        if n % 30 == 0:
            ref = cpu(x)
            cos = min(float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)) for a, b in zip(ref, leaves))
            checks.append((cos, len(decode(head, ref)), len(det)))
        writer.write(draw(frame, det))
        counts.append(len(det))
        for name in classes(det):
            tally[name] = tally.get(name, 0) + 1
        n += 1
    total = time.time() - start
    cap.release()
    writer.release()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
    pathlib.Path(tmp).unlink()
    same = sum(1 for _, a, b in checks if a == b)
    print(f"{stem} {w}x{h} frames={n} fps={n / total:.1f} objects mean={np.mean(counts):.1f} max={max(counts)} zero_frames={counts.count(0)}", flush=True)
    print(f"  classes(box-frames)={dict(sorted(tally.items(), key=lambda kv: -kv[1]))}", flush=True)
    print(f"  cpu check n={len(checks)} min_cos={min(c[0] for c in checks):.5f} same_count={same}/{len(checks)}", flush=True)
