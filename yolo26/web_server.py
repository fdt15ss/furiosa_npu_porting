"""FastAPI + WebSocket server for real-time YOLO26n-seg NPU detection & segmentation."""
from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = pathlib.Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from web_engine import MODEL_NAME, Yolo26SegNPUEngine  # noqa: E402
STATIC_DIR = BASE_DIR / "static"
MEDIA_DIR = pathlib.Path("/root/programming/npu/media")
OUTPUT_DIR = BASE_DIR / "web_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NPU_CORE = int(os.environ.get("YOLO26_NPU_CORE", "0"))
_engine: Yolo26SegNPUEngine | None = None
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="yolo26-npu")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _engine
    _engine = Yolo26SegNPUEngine(core=NPU_CORE, threads=1)
    try:
        yield
    finally:
        _executor.shutdown(wait=False)


app = FastAPI(title="YOLO26 Furiosa RNGD NPU Vision Studio", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def get_engine() -> Yolo26SegNPUEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="NPU engine is still initializing")
    return _engine


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict:
    engine = get_engine()
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": f"furiosa:{engine.core}",
        "num_classes": len(engine.names),
        "classes": engine.names,
    }


@app.get("/api/samples")
async def list_samples() -> dict:
    samples: list[dict] = []
    if MEDIA_DIR.exists():
        for p in sorted(MEDIA_DIR.glob("*.mp4")) + sorted((MEDIA_DIR / "driving").glob("*.mp4")):
            if "_npu" in p.name or p.name.startswith("."):
                continue
            rel = p.relative_to(MEDIA_DIR).as_posix()
            samples.append({"name": p.name, "path": rel, "type": "video"})
        for p in sorted(MEDIA_DIR.glob("*.jpg")):
            if "_npu" in p.name:
                continue
            rel = p.relative_to(MEDIA_DIR).as_posix()
            samples.append({"name": p.name, "path": rel, "type": "image"})
    return {"samples": samples}


@app.get("/api/samples/{sample_path:path}")
async def get_sample(sample_path: str) -> FileResponse:
    target = (MEDIA_DIR / sample_path).resolve()
    if not str(target).startswith(str(MEDIA_DIR.resolve())) or not target.is_file():
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(target)


@app.post("/api/infer/image")
async def infer_image(
    file: UploadFile = File(...),
    mode: str = Form("both"),
    conf: float = Form(0.25),
    iou: float = Form(0.70),
) -> JSONResponse:
    engine = get_engine()
    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    loop = asyncio.get_running_loop()
    annotated, meta = await loop.run_in_executor(
        _executor,
        lambda: engine.infer_and_draw(frame, mode=mode, conf_thres=conf, iou_thres=iou),
    )
    ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode output image")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return JSONResponse({"image_base64": f"data:image/jpeg;base64,{b64}", **meta})


@app.post("/api/infer/video")
async def infer_video(
    file: UploadFile = File(...),
    mode: str = Form("both"),
    conf: float = Form(0.25),
    iou: float = Form(0.70),
) -> JSONResponse:
    engine = get_engine()
    job_id = uuid.uuid4().hex[:10]
    suffix = pathlib.Path(file.filename or "input.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(await file.read())
        in_path = pathlib.Path(tmp_in.name)

    tmp_mp4 = OUTPUT_DIR / f".{job_id}.tmp.mp4"
    out_name = f"yolo26n_seg_{job_id}.mp4"
    out_path = OUTPUT_DIR / out_name

    def _process_video() -> dict:
        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise ValueError("Cannot open uploaded video")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(tmp_mp4), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))

        frames = 0
        tally: dict[str, int] = {}
        npu_times: list[float] = []
        total_times: list[float] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                annotated, meta = engine.infer_and_draw(frame, mode=mode, conf_thres=conf, iou_thres=iou)
                writer.write(annotated)
                frames += 1
                npu_times.append(meta["timing_ms"]["npu"])
                total_times.append(meta["timing_ms"]["total"])
                for cls_name in meta["classes"]:
                    tally[cls_name] = tally.get(cls_name, 0) + 1
        finally:
            cap.release()
            writer.release()

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(tmp_mp4),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out_path),
            ],
            check=True,
        )
        tmp_mp4.unlink(missing_ok=True)
        in_path.unlink(missing_ok=True)
        avg_total = sum(total_times) / max(1, len(total_times))
        return {
            "video_url": f"/api/outputs/{out_name}",
            "frames": frames,
            "width": w,
            "height": h,
            "source_fps": round(src_fps, 2),
            "processing_fps": round(1000.0 / avg_total, 1) if avg_total > 0 else 0.0,
            "avg_npu_ms": round(sum(npu_times) / max(1, len(npu_times)), 2),
            "avg_total_ms": round(avg_total, 2),
            "classes": dict(sorted(tally.items(), key=lambda kv: -kv[1])),
        }

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(_executor, _process_video)
    except Exception as exc:
        in_path.unlink(missing_ok=True)
        tmp_mp4.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(result)


@app.get("/api/outputs/{filename}")
async def get_output(filename: str) -> FileResponse:
    target = (OUTPUT_DIR / filename).resolve()
    if not str(target).startswith(str(OUTPUT_DIR.resolve())) or not target.is_file():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(target, media_type="video/mp4")


@app.websocket("/ws/infer")
async def ws_infer(websocket: WebSocket) -> None:
    await websocket.accept()
    engine = get_engine()
    loop = asyncio.get_running_loop()

    config = {
        "mode": "both",
        "conf": 0.25,
        "iou": 0.70,
    }

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                    if payload.get("type") == "config":
                        if payload.get("mode") in ("both", "detect", "segment"):
                            config["mode"] = payload["mode"]
                        if "conf" in payload:
                            config["conf"] = max(0.05, min(0.95, float(payload["conf"])))
                        if "iou" in payload:
                            config["iou"] = max(0.05, min(0.95, float(payload["iou"])))
                        await websocket.send_json({"type": "config_ack", **config})
                except Exception:
                    pass
                continue

            raw_bytes = message.get("bytes")
            if not raw_bytes or len(raw_bytes) <= 4:
                continue

            # Wire format: 4-byte big-endian uint32 sequence id + JPEG bytes
            seq = struct.unpack(">I", raw_bytes[:4])[0]
            jpeg_bytes = raw_bytes[4:]
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json({"type": "error", "seq": seq, "detail": "decode failed"})
                continue

            mode = str(config["mode"])
            conf = float(config["conf"])
            iou = float(config["iou"])
            result = await loop.run_in_executor(
                _executor,
                lambda f=frame, m=mode, c=conf, i=iou: engine.infer(f, mode=m, conf_thres=c, iou_thres=i),
            )
            result["type"] = "result"
            result["seq"] = seq
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass
