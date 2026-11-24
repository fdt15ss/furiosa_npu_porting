"""Run `furiosa-llm serve` with only the vision buckets that hold real kernels.

The one-card FXB lists many vision buckets, but most reuse a smaller kernel (see fxb_single_card.py).
If the processor picked one of those, the encoder would run with wrong shapes and give wrong
output without an error. This wrapper limits the buckets the processor sees, so frames larger than
the biggest real bucket are rejected instead.

Usage: [VIDEO_MAX_TOKENS=1024] VIS_BUCKETS=256 python serve_vision.py serve <model_dir> --fxb <fxb> [serve options]
"""
import os
import sys

import furiosa_llm.llm_engine as engine
from furiosa_llm.multimodal.models.base import MultimodalConstraints

REAL = {int(b) for b in os.environ.get("VIS_BUCKETS", "256").split(",")}
_orig = engine.resolve_constraints


def resolve_constraints(native_engine):
    found = _orig(native_engine).vision_encoder_buckets
    kept = sorted(b for b in found if b in REAL)
    print(f"vision buckets in FXB {sorted(found)}; using {kept}", file=sys.stderr, flush=True)
    return MultimodalConstraints(kept)


engine.resolve_constraints = resolve_constraints

# VIDEO_MAX_TOKENS=<n>: per-frame vision-token cap of the Qwen3-VL video processor (default 768 tokens,
# i.e. 3072 patches). furiosa-llm turns on cap_pixels_per_frame and rejects max_video_tokens as a request
# option, so frames above 3072 patches (for example 1280x704 = 3520) need this set at server start.
if os.environ.get("VIDEO_MAX_TOKENS"):
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    Qwen3VLVideoProcessor.max_video_tokens = int(os.environ["VIDEO_MAX_TOKENS"])

# MM_DUMP=<dir>: save each request's prompt ids, prompt updates and encoder payload for cpu_video_ref.py.
if os.environ.get("MM_DUMP"):
    import dataclasses
    import pickle
    import time

    _fin = engine.LLMEngineBase._finalize_multimodal_inputs

    def _plain(x):
        if type(x).__name__ == "VisionPatches":  # native class; pixel_values is bf16 viewed as uint16
            return {"pixel_values": x.pixel_values.copy(), "grid_thw": tuple(x.grid_thw)}
        if dataclasses.is_dataclass(x):
            return {f.name: _plain(getattr(x, f.name)) for f in dataclasses.fields(x)}
        if isinstance(x, dict):
            return {k: _plain(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(_plain(v) for v in x)
        return x

    def _finalize(self, mm_inputs, batch_encoding, *a, **k):
        out = _fin(self, mm_inputs, batch_encoding, *a, **k)
        if out.inputs:
            path = os.path.join(os.environ["MM_DUMP"], f"mm_{time.time():.0f}.pkl")
            with open(path, "wb") as f:
                pickle.dump({
                    "input_ids": list(batch_encoding.input_ids),
                    "updates": [_plain(u) for u in out.prompt_updates],
                    "payloads": [_plain(i.payload) for i in out.inputs],
                    "metadata": [dict(i.metadata) for i in out.inputs],
                }, f)
            print("saved", path, file=sys.stderr, flush=True)
        return out

    engine.LLMEngineBase._finalize_multimodal_inputs = _finalize

from furiosa_llm.cli.main import main  # noqa: E402

sys.exit(main())