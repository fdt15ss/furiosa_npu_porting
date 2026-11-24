"""Send one video to the server started by serve_vision.py.

Usage: python video_client.py <video.mp4> [num_frames] [prompt]
Each frame is resized to at most MAX_PATCHES vision patches (16x16 px), so it fits the largest real
vision bucket. With 256 patches a 16:9 frame becomes 320x192.
"""
import os
import sys
import time

from openai import OpenAI

MAX_PATCHES = int(os.environ.get("MAX_PATCHES", "256"))
PATCH = 16


def main():
    video = os.path.abspath(sys.argv[1])
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    prompt = sys.argv[3] if len(sys.argv) > 3 else "이 영상에서 무슨 일이 일어나는지 한국어로 설명해줘."
    client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"), api_key="e")
    start = time.time()
    r = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=[{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": f"file://{video}"}},
            {"type": "text", "text": prompt},
        ]}],
        extra_body={
            # longest_edge is the pixel budget for the whole clip; the processor splits it per frame.
            "mm_processor_kwargs": {"video": {
                "num_frames": num_frames,
                "longest_edge": MAX_PATCHES * PATCH * PATCH * num_frames,
                "shortest_edge": 4096,
            }},
            "chat_template_kwargs": {"enable_thinking": False},
        },
        max_tokens=int(os.environ.get("MAX_TOKENS", "256")),
        temperature=0.0,
    )
    print(r.choices[0].message.content)
    print(r.usage)
    print(f"time {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
