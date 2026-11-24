"""usage: video_pose.py <yolo11n-pose|yolo26n-pose> <video> [core]; writes an annotated mp4 to media/."""
import pathlib
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils._pytree import tree_leaves
import furiosa.torch as ft
from furiosa.torch._C import ir
from furiosa.torch.export import ExportedProgramWeight, passes
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.plotting import Annotator

from fps_pose import build, decode

torch.set_num_threads(1)  # with default threads, CPU decode + NMS took 77 ms per frame instead of 0.8 ms
name, video = sys.argv[1], sys.argv[2]
core = int(sys.argv[3]) if len(sys.argv) > 3 else 6
model, head, is26, x0 = build(name)
dec = passes.decompose(torch.export.export(model, (x0,)))
inner = ir.Runnable.deserialize(open(name + ".edf", "rb").read())
cm = ft.CompileModule(
    ft.Runnable(inner),
    ExportedProgramWeight(dec),
    output_reconstruction_indices=tuple(range(len(inner.output_dram_shapes))),
)
dev = torch.device(f"furiosa:{core}")
cm = cm.to(dev)

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
stem = pathlib.Path(video).stem
out_path = f"/root/furiosa_study/media/{stem}_{name}_npu.mp4"
tmp_path = f"/root/furiosa_study/yolo/.{stem}_{name}.tmp.mp4"
writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
letterbox = LetterBox(640, auto=False)
times = {"read": [], "pre": [], "npu": [], "post": [], "draw": []}
people, checks, frame_no = [], [], 0
start = time.time()
while True:
    t0 = time.time()
    ok, frame = cap.read()
    if not ok:
        break
    t1 = time.time()
    x = torch.from_numpy(letterbox(image=frame)[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 255
    t2 = time.time()
    leaves = [o.to("cpu") for o in tree_leaves(cm(x.to(dev)))]
    t3 = time.time()
    det = decode(head, is26, leaves)
    t4 = time.time()
    if frame_no % 30 == 0:
        with torch.no_grad():
            c = decode(head, is26, tree_leaves(model(x)))
        if len(c) == len(det) and len(c):
            d = torch.cdist(c[:, :4], det[:, :4]).min(1).values.max().item()
        else:
            d = None
        checks.append((frame_no, len(c), len(det), d))
    t5 = time.time()
    boxes = ops.scale_boxes((640, 640), det[:, :4].clone(), (h, w))
    kpts = ops.scale_coords((640, 640), det[:, 6:].view(-1, 17, 3).clone(), (h, w))
    a = Annotator(frame, line_width=2)
    for b, k, conf in zip(boxes, kpts, det[:, 4]):
        a.box_label(b.tolist(), f"{conf:.2f}")
        a.kpts(k)
    writer.write(a.result())
    t6 = time.time()
    for key, value in zip(times, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t6 - t5)):
        times[key].append(value)
    people.append(len(det))
    frame_no += 1
total = time.time() - start
cap.release()
writer.release()
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp_path, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path], check=True)
pathlib.Path(tmp_path).unlink()
print(f"{name} frames={frame_no} total={total:.2f}s fps={frame_no / total:.1f} (source {fps:.2f}) people mean={np.mean(people):.1f} max={max(people)} zero_frames={people.count(0)}")
print("ms per frame:", {k: round(float(np.mean(v)) * 1000, 2) for k, v in times.items()})
print("cpu check (frame, cpu, npu, max box dist px@640):", checks)
print("saved", out_path)
