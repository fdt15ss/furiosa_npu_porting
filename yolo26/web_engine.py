"""YOLO26n-seg Furiosa RNGD NPU inference engine for Detection & Segmentation.

Uses only the YOLO26 segmentation model (`yolo26n-seg.edf` + `yolo26n-seg-protohead.edf`)
to provide object detection, instance segmentation, or both simultaneously.
"""
from __future__ import annotations

import os
import pathlib
import threading
import time
from typing import Any

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
from furiosa.torch.export import ExportedProgramWeight, PASSES, make_pass_context, passes

YOLO26_DIR = pathlib.Path(__file__).resolve().parent
MODEL_NAME = "yolo26n-seg"
EXAMPLE_IMAGE = "/root/programming/npu/media/bus.jpg"
INPUT_SIZE = 640
FEATS_SHAPES = [torch.zeros(1, 1, INPUT_SIZE // s, INPUT_SIZE // s) for s in (8, 16, 32)]


def _ensure_yolo26_cwd() -> None:
    """Import helpers from yolo26/ and keep relative asset paths consistent."""
    import sys

    if str(YOLO26_DIR) not in sys.path:
        sys.path.insert(0, str(YOLO26_DIR))


_ensure_yolo26_cwd()
from compile_model import build  # noqa: E402
from compile_proto import proto_head  # noqa: E402


class Yolo26SegNPUEngine:
    """Thread-safe wrapper around YOLO26n-seg running on a Furiosa RNGD NPU core."""

    def __init__(self, core: int = 0, threads: int = 1) -> None:
        self.core = core
        self.threads = threads
        self._lock = threading.Lock()
        self.letterbox = LetterBox(INPUT_SIZE, auto=False)

        torch.set_num_threads(threads)
        prev_cwd = os.getcwd()
        try:
            os.chdir(YOLO26_DIR)
            self.model = build(MODEL_NAME)
            self.head = self.model.model[-1]
            self.names: dict[int, str] = dict(self.model.names)

            sample_img = self._load_sample_image()
            _, x = self._preprocess(sample_img)

            # Load main backbone+neck+head graph
            dec = passes.decompose(torch.export.export(self.model, (x,)))
            main_inner = ir.Runnable.deserialize((YOLO26_DIR / f"{MODEL_NAME}.edf").read_bytes())
            self.cm = ft.CompileModule(
                ft.Runnable(main_inner),
                ExportedProgramWeight(dec),
                output_reconstruction_indices=tuple(range(len(main_inner.output_dram_shapes))),
            )
            self.dev = torch.device(f"furiosa:{core}")
            self.cm = self.cm.to(self.dev)

            # Load Proto26 mask prototype graph
            ph = proto_head(self.model)
            with torch.no_grad():
                feats = self.model(x)["feats"]
            ep_proto = torch.export.export(ph, tuple(feats))
            ctx = make_pass_context({})
            for fx_pass in PASSES:
                ep_proto = fx_pass(ep_proto, ctx)
            proto_inner = ir.Runnable.deserialize((YOLO26_DIR / f"{MODEL_NAME}-protohead.edf").read_bytes())
            self.pm = ft.CompileModule(
                ft.Runnable(proto_inner),
                ExportedProgramWeight(ep_proto),
                output_reconstruction_indices=(0,),
            ).to(self.dev)

            # Warm up NPU graphs
            self._warmup(x)
        finally:
            os.chdir(prev_cwd)

    def _load_sample_image(self) -> np.ndarray:
        img = cv2.imread(EXAMPLE_IMAGE)
        if img is not None:
            return img
        return np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)

    def _warmup(self, x: torch.Tensor) -> None:
        with self._lock:
            for _ in range(3):
                boxes, scores, f0, f1, f2, coef = tree_leaves(self.cm(x.to(self.dev)))
                proto = self.pm(f0, f1, f2)
                _ = boxes.to("cpu"), scores.to("cpu"), coef.to("cpu")
                if isinstance(proto, (list, tuple)):
                    proto[0].to("cpu")
                else:
                    proto.to("cpu")

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        im = self.letterbox(image=frame_bgr)
        tensor = torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 255.0
        return im, tensor

    def _run_npu(self, x: torch.Tensor, need_proto: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        with self._lock:
            boxes, scores, f0, f1, f2, coef = tree_leaves(self.cm(x.to(self.dev)))
            if need_proto:
                proto = self.pm(f0, f1, f2)
                proto = proto[0] if isinstance(proto, (list, tuple)) else proto
                proto_cpu = proto.to("cpu")
            else:
                proto_cpu = None
            return boxes.to("cpu"), scores.to("cpu"), coef.to("cpu"), proto_cpu

    def _postprocess(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        coef: torch.Tensor,
        proto: torch.Tensor | None,
        conf_thres: float,
        iou_thres: float,
        need_masks: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pred = self.head._inference(
            {
                "boxes": boxes,
                "scores": scores,
                "feats": FEATS_SHAPES,
                "mask_coefficient": coef,
            }
        )
        det = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, nc=self.head.nc)[0]
        if not len(det) or not need_masks or proto is None:
            return det, None
        masks = ops.process_mask(proto[0], det[:, 6:], det[:, :4], (INPUT_SIZE, INPUT_SIZE), upsample=True)
        return det, masks

    @staticmethod
    def _letterbox_norm_params(h: int, w: int) -> tuple[float, float, float]:
        gain = min(INPUT_SIZE / h, INPUT_SIZE / w)
        pad_x = (INPUT_SIZE - w * gain) / 2.0
        pad_y = (INPUT_SIZE - h * gain) / 2.0
        return gain, pad_x, pad_y

    def _extract_polygons(self, mask_640: np.ndarray, h: int, w: int) -> list[list[list[float]]]:
        gain, pad_x, pad_y = self._letterbox_norm_params(h, w)
        inv_w_gain = 1.0 / (w * gain)
        inv_h_gain = 1.0 / (h * gain)
        contours, _ = cv2.findContours(mask_640, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys: list[list[list[float]]] = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 16.0:
                continue
            epsilon = max(1.0, 0.004 * cv2.arcLength(cnt, True))
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3:
                continue
            pts = approx.reshape(-1, 2).astype(np.float32)
            pts[:, 0] = np.clip((pts[:, 0] - pad_x) * inv_w_gain, 0.0, 1.0)
            pts[:, 1] = np.clip((pts[:, 1] - pad_y) * inv_h_gain, 0.0, 1.0)
            polys.append(np.round(pts, 4).tolist())
        return polys

    def infer(
        self,
        frame_bgr: np.ndarray,
        mode: str = "both",
        conf_thres: float = 0.25,
        iou_thres: float = 0.70,
    ) -> dict[str, Any]:
        """Run YOLO26n-seg on a BGR frame and return normalized detections & polygons."""
        if mode not in ("both", "detect", "segment"):
            mode = "both"
        need_masks = mode in ("both", "segment")

        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        _, x = self._preprocess(frame_bgr)
        t_pre = time.perf_counter()

        boxes, scores, coef, proto = self._run_npu(x, need_proto=need_masks)
        t_npu = time.perf_counter()

        det, masks = self._postprocess(
            boxes,
            scores,
            coef,
            proto,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            need_masks=need_masks,
        )

        detections: list[dict[str, Any]] = []
        if len(det):
            scaled_boxes = ops.scale_boxes((INPUT_SIZE, INPUT_SIZE), det[:, :4].clone(), (h, w))
            masks_np = (masks > 0.5).numpy().astype(np.uint8) if masks is not None else None
            for i in range(len(det)):
                x1, y1, x2, y2 = scaled_boxes[i].tolist()
                cls_id = int(det[i, 5])
                conf = float(det[i, 4])
                color_bgr = colors(cls_id, True)
                item: dict[str, Any] = {
                    "cls_id": cls_id,
                    "label": self.names.get(cls_id, str(cls_id)),
                    "conf": round(conf, 4),
                    "color": [int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])],  # RGB
                    "bbox": [
                        round(max(0.0, min(1.0, x1 / w)), 4),
                        round(max(0.0, min(1.0, y1 / h)), 4),
                        round(max(0.0, min(1.0, x2 / w)), 4),
                        round(max(0.0, min(1.0, y2 / h)), 4),
                    ],
                }
                if masks_np is not None:
                    item["polygons"] = self._extract_polygons(masks_np[i], h, w)
                detections.append(item)

        t_end = time.perf_counter()
        return {
            "model": MODEL_NAME,
            "mode": mode,
            "width": w,
            "height": h,
            "count": len(detections),
            "detections": detections,
            "timing_ms": {
                "pre": round((t_pre - t0) * 1000.0, 2),
                "npu": round((t_npu - t_pre) * 1000.0, 2),
                "post": round((t_end - t_npu) * 1000.0, 2),
                "total": round((t_end - t0) * 1000.0, 2),
            },
        }

    def infer_and_draw(
        self,
        frame_bgr: np.ndarray,
        mode: str = "both",
        conf_thres: float = 0.25,
        iou_thres: float = 0.70,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run YOLO26n-seg and render boxes and/or masks onto a BGR copy of the frame."""
        if mode not in ("both", "detect", "segment"):
            mode = "both"
        need_masks = mode in ("both", "segment")
        draw_boxes = mode in ("both", "detect")

        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        _, x = self._preprocess(frame_bgr)
        t_pre = time.perf_counter()

        boxes, scores, coef, proto = self._run_npu(x, need_proto=need_masks)
        t_npu = time.perf_counter()

        det, masks = self._postprocess(
            boxes,
            scores,
            coef,
            proto,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            need_masks=need_masks,
        )

        canvas = frame_bgr.copy()
        if len(det):
            if need_masks and masks is not None:
                full = ops.scale_masks(masks[None].float(), (h, w))[0] > 0.5
                for mask, c in zip(full, det[:, 5].int().tolist()):
                    color = np.array(colors(c, True), dtype=np.float32)
                    area = mask.numpy()
                    canvas[area] = (canvas[area] * 0.55 + color * 0.45).astype(np.uint8)
            if draw_boxes:
                scaled = ops.scale_boxes((INPUT_SIZE, INPUT_SIZE), det[:, :4].clone(), (h, w))
                annotator = Annotator(canvas, line_width=2)
                for b, d in zip(scaled, det):
                    c = int(d[5])
                    annotator.box_label(
                        b.tolist(),
                        f"{self.names.get(c, str(c))} {float(d[4]):.2f}",
                        color=colors(c, True),
                    )
                canvas = annotator.result()

        t_end = time.perf_counter()
        meta = {
            "count": int(len(det)),
            "classes": [self.names.get(int(c), str(int(c))) for c in det[:, 5]] if len(det) else [],
            "timing_ms": {
                "pre": round((t_pre - t0) * 1000.0, 2),
                "npu": round((t_npu - t_pre) * 1000.0, 2),
                "post": round((t_end - t_npu) * 1000.0, 2),
                "total": round((t_end - t0) * 1000.0, 2),
            },
        }
        return canvas, meta
