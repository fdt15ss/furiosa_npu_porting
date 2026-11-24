"""Load the FP8 model with a single-card FXB and compare NPU output with the CPU reference.

Usage: python npu_smoke.py <model_dir> <fxb_path>
CPU reference: greedy output for the same prompt, from cpu_smoke.py.
"""
import os
import sys
import time

from furiosa_llm import LLM, SamplingParams
from furiosa_llm.metadata.config_types import SchedulerConfig

PROMPT = "Write a Python function that reverses a string. Answer with only the code."
CPU_TEXT = "```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```"


def main():
    model_dir, fxb = sys.argv[1:3]
    start = time.time()
    # Without max_io_memory_mb the engine estimates I/O memory with no linear buckets and panics
    # on this hybrid model. Decode-batching profiling is off for the same reason.
    llm = LLM(
        model_dir,
        fxb=fxb,
        max_model_len=2048,
        max_io_memory_mb=int(os.environ.get("IO_MB", "4096")),
        scheduler_config=SchedulerConfig(profile_guided_decode_batching=False, enable_npu_sampling=False),
    )
    print(f"loaded in {time.time() - start:.0f}s", flush=True)
    try:
        start = time.time()
        out = llm.chat(
            [{"role": "user", "content": PROMPT}],
            SamplingParams(temperature=0, max_tokens=48),
            chat_template_kwargs={"enable_thinking": False},
        )
        text = out[0].outputs[0].text
        print(f"generated in {time.time() - start:.0f}s", flush=True)
        print("TEXT:", repr(text), flush=True)
        print("TOKENS:", list(out[0].outputs[0].token_ids), flush=True)
        print("MATCHES_CPU:", text.strip() == CPU_TEXT, flush=True)
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
