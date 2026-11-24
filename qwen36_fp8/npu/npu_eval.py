"""Run the shared prompts on the NPU and save token ids. Usage: npu_eval.py <model> <fxb> <out.json>"""
import json
import os
import sys
import time

from furiosa_llm import LLM, SamplingParams
from furiosa_llm.metadata.config_types import SchedulerConfig

from eval_prompts import MAX_TOKENS, PROMPTS


def main():
    model, fxb, out_path = sys.argv[1:4]
    llm = LLM(
        model,
        fxb=fxb,
        max_model_len=2048,
        max_io_memory_mb=int(os.environ.get("IO_MB", "4096")),
        scheduler_config=SchedulerConfig(profile_guided_decode_batching=False, enable_npu_sampling=False),
    )
    results = {}
    try:
        for name, prompt in PROMPTS.items():
            start = time.time()
            out = llm.chat(
                [{"role": "user", "content": prompt}],
                SamplingParams(temperature=0, max_tokens=MAX_TOKENS),
                chat_template_kwargs={"enable_thinking": False},
            )[0].outputs[0]
            took = time.time() - start
            results[name] = {"tokens": list(out.token_ids), "text": out.text, "seconds": took}
            print(name, f"{len(out.token_ids)} tokens in {took:.2f}s", repr(out.text[:120]), flush=True)
            json.dump(results, open(out_path, "w"), ensure_ascii=False, indent=1)
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
