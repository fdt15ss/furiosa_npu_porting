"""usage: run_model.py <yolo11n-pose|yolo26n-pose> ; compile, save, run on npu."""
import subprocess, sys, time, traceback
import cv2, numpy as np, torch
import furiosa.torch as ft
from furiosa.torch.export import passes
from torch.utils._pytree import tree_leaves
from ultralytics import YOLO, ASSETS

name = sys.argv[1]
yolo = YOLO(name + ".pt")
model = yolo.model
model.fuse()
model.eval()
head = model.model[-1]
print("head:", type(head).__name__, "end2end", head.end2end, flush=True)

if type(head).__name__ == "Pose26":
    # training=True takes the sigma branch that fuse() removed, so call the head directly in eval mode.
    branch = head.one2one if head.end2end else head.one2many
    def raw_forward(x):
        p = head.forward_head(x, **branch)
        return {k: p[k] for k in ("boxes", "scores", "kpts")}
    head.forward = raw_forward
else:
    head.training = True  # PDF: Raw Prediction

img = cv2.cvtColor(cv2.imread(str(ASSETS / "bus.jpg")), cv2.COLOR_BGR2RGB)
x = torch.from_numpy(cv2.resize(img, (640, 640))).permute(2, 0, 1)[None].float() / 255

with torch.no_grad():
    cpu_out = model(x)
cpu_leaves = tree_leaves(cpu_out)
print("outputs:", [tuple(t.shape) for t in cpu_leaves], flush=True)

t = time.time(); ep = torch.export.export(model, (x,)); print(f"export ok {time.time()-t:.1f}s", flush=True)
t = time.time(); dec = passes.decompose(ep); print(f"decompose ok {time.time()-t:.1f}s", flush=True)
t = time.time(); rt = ft.compiler.compile(dec); print(f"compile ok {time.time()-t:.1f}s", flush=True)

try:
    blob = rt.serialize()
    with open(name + ".edf", "wb") as f:
        f.write(blob if isinstance(blob, (bytes, bytearray)) else bytes(blob))
    print("saved", name + ".edf", len(blob), flush=True)
except Exception:
    traceback.print_exc()

busy = subprocess.run("ps -eo cmd | grep 'furiosa-llm serve' | grep -v -E 'grep|apt-get'", shell=True, capture_output=True, text=True).stdout.strip()
if busy:
    print("NPU in use by server, skipping inference:", busy); sys.exit(0)

dev = torch.device("furiosa:6")
rt = rt.to(dev)
xd = x.to(dev)
t = time.time(); out = rt(xd); print(f"first run {time.time()-t:.3f}s", flush=True)
leaves = [o.to("cpu") for o in tree_leaves(out)]
print("npu outputs:", [tuple(t.shape) for t in leaves], flush=True)
for i, (a, b) in enumerate(zip(cpu_leaves, leaves)):
    a, b = a.float(), b.float()
    if a.shape == b.shape:
        print(f"[{i}] maxdiff={(a-b).abs().max():.5f} meandiff={(a-b).abs().mean():.5f} cosine={torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0):.6f}", flush=True)
ts = []
for _ in range(100):
    t = time.time(); r = rt(xd); [o.to("cpu") for o in tree_leaves(r)]; ts.append(time.time()-t)
print(f"npu latency mean={np.mean(ts[10:])*1000:.2f}ms p50={np.median(ts[10:])*1000:.2f}ms", flush=True)
ts = []
with torch.no_grad():
    for _ in range(30):
        t = time.time(); model(x); ts.append(time.time()-t)
print(f"cpu latency mean={np.mean(ts[5:])*1000:.2f}ms", flush=True)
