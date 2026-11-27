#!/bin/bash
# Start the OpenAI-compatible server with video input on one RNGD card.
# Usage: ./serve_video.sh [port] [media_dir]    then: python video_client.py <video.mp4> [num_frames]
# VIS_BUCKETS lists the real vision kernels of the FXB (default 256); larger frames are rejected.
# VIDEO_MAX_TOKENS=1024 lets a video frame go above 3072 patches (see serve_vision.py).
# MAX_CONC=<n> caps concurrent decode requests; keep it at or below the largest DEC_BATCHES of the FXB.
set -euo pipefail
PORT=${1:-8000}; MEDIA=${2:-/root/furiosa_study/media}
HERE=$(cd "$(dirname "$0")"; pwd)
ENV=${FURIOSA_ENV:-/root/miniconda3/envs/furiosa}
MODEL=${MODEL:-/root/models/Logics-SWE-Qwen3.6-27B-FP8}
BUNDLE_FXB="$HERE/../_fxb & compile/logics-swe-1card-sdk5-p256.fxb"
if [ -z "${FXB:-}" ]; then
  if [ -f "$BUNDLE_FXB" ]; then
    FXB="$BUNDLE_FXB"
  else
    FXB=/root/models/fxb/logics-swe-1card-sdk5-p256.fxb
  fi
fi
export PATH=$ENV/bin:$PATH VIS_BUCKETS=${VIS_BUCKETS:-256}   # MAX_LEN must match the FXB (default 2048)
cd "$HERE"
exec "$ENV/bin/python" serve_vision.py serve "$MODEL" --fxb "$FXB" --devices npu:6 --port "$PORT" \
  --max-model-len "${MAX_LEN:-2048}" --max-io-memory-mb 4096 \
  --no-profile-guided-decode-batching --no-enable-npu-sampling \
  --allowed-local-media-path "$MEDIA" ${MAX_CONC:+--max-concurrency "$MAX_CONC"}
