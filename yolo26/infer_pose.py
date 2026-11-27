"""usage: infer.py <yolo11n-pose|yolo26n-pose>; load saved edf, run on npu, compare with cpu."""
import sys, time
import cv2, numpy as np, torch
import furiosa.torch as ft
from furiosa.torch._C import ir
from furiosa.torch.export import passes, ExportedProgramWeight
from torch.utils._pytree import tree_leaves
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.plotting import Annotator
from ultralytics.data.augment import LetterBox

name = sys.argv[1]
y = YOLO(name + ".pt"); model = y.model; model.fuse(); model.eval()
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

def load(path):
    im0 = cv2.imread(path)
    im = LetterBox(640, auto=False)(image=im0)
    t = torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 255
    return im, t

_, x0 = load("/root/programming/npu/media/bus.jpg")
ep = torch.export.export(model, (x0,))
dec = passes.decompose(ep)
w = ExportedProgramWeight(dec)
inner = ir.Runnable.deserialize(open(name + ".edf", "rb").read())
print("inputs", len(inner.input_dram_shapes), "outputs", len(inner.output_dram_shapes), flush=True)
cm = ft.CompileModule(ft.Runnable(inner), w, output_reconstruction_indices=tuple(range(len(inner.output_dram_shapes))))
dev = torch.device("furiosa:6")
t = time.time(); cm = cm.to(dev); print(f"load {time.time()-t:.2f}s", flush=True)

def decode(leaves):
    """leaves: boxes, scores, [feats...], kpts on cpu"""
    boxes, scores, kpts = leaves[0], leaves[1], leaves[-1]
    feats = [torch.zeros(1, 1, 640 // s, 640 // s) for s in (8, 16, 32)]
    head.training = False
    pred = head._inference({"boxes": boxes, "scores": scores, "kpts": kpts, "feats": feats})
    if not is26: head.training = True
    return non_max_suppression(pred, conf_thres=0.25, iou_thres=0.7, nc=1)[0]

for img in ("bus", "zidane"):
    im, x = load(f"/root/programming/npu/media/{img}.jpg")
    with torch.no_grad():
        cpu = [t for t in tree_leaves(model(x))]
    t = time.time(); out = cm(x.to(dev)); npu = [o.to("cpu") for o in tree_leaves(out)]; first = time.time()-t
    print(f"== {img} first={first:.3f}s shapes={[tuple(o.shape) for o in npu]}", flush=True)
    for i, (a, b) in enumerate(zip(cpu, npu)):
        a, b = a.float(), b.float()
        print(f" [{i}] maxdiff={(a-b).abs().max():.4f} rel={(a-b).abs().mean()/a.abs().mean():.5f} cos={torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0):.6f}", flush=True)
    dc, dn = decode(cpu), decode(npu)
    print(f" detections cpu={len(dc)} npu={len(dn)}", flush=True)
    if len(dc) == len(dn) and len(dc):
        idx = torch.cdist(dc[:, :4], dn[:, :4]).argmin(1)  # NMS order can swap near-equal scores
        m = dn[idx]
        k = (dc[:, 6:] - m[:, 6:]).view(-1, 17, 3)
        print(f" matched={idx.unique().numel()}/{len(dc)} conf diff max={(dc[:,4]-m[:,4]).abs().max():.4f} box diff max={(dc[:,:4]-m[:,:4]).abs().max():.2f}px kpt xy diff max={k[...,:2].abs().max():.2f}px", flush=True)
    a = Annotator(im.copy())
    for d in dn:
        a.box_label(d[:4].tolist(), f"person {d[4]:.2f}")
        a.kpts(d[6:].view(17, 3))
    cv2.imwrite(f"/root/programming/npu/media/{img}_{name}_npu.jpg", a.result())

xd = x0.to(dev); ts = []
for _ in range(110):
    t = time.time(); r = cm(xd); [o.to("cpu") for o in tree_leaves(r)]; ts.append(time.time()-t)
print(f"npu latency mean={np.mean(ts[10:])*1000:.2f}ms p50={np.median(ts[10:])*1000:.2f}ms p99={np.percentile(ts[10:],99)*1000:.2f}ms", flush=True)
ts = []
with torch.no_grad():
    for _ in range(35):
        t = time.time(); model(x0); ts.append(time.time()-t)
print(f"cpu ({torch.get_num_threads()} threads) latency mean={np.mean(ts[5:])*1000:.2f}ms", flush=True)
