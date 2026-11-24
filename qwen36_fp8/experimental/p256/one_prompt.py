"""Prefill timing for one eval prompt. usage: one_prompt.py <npu_dir> <model> <fxb> <prompt_name> [reps]"""
import os, sys, time
sys.path.insert(0, sys.argv[1])
from eval_prompts import PROMPTS
from furiosa_llm import LLM, SamplingParams
from furiosa_llm.metadata.config_types import SchedulerConfig

model, fxb, name = sys.argv[2:5]
reps = int(sys.argv[5]) if len(sys.argv) > 5 else 5
llm = LLM(model, fxb=fxb, max_model_len=2048, max_io_memory_mb=4096,
          scheduler_config=SchedulerConfig(profile_guided_decode_batching=False, enable_npu_sampling=False))
try:
    msgs = [{"role": "user", "content": PROMPTS[name]}]
    sp = SamplingParams(temperature=0, max_tokens=1)
    times = []
    for i in range(reps + 1):
        t = time.time()
        llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
        if i: times.append(time.time() - t)  # first run is warm-up
    times.sort()
    print(f"RESULT {name} first-token median {times[len(times)//2]*1000:.1f} ms min {times[0]*1000:.1f} ms over {reps}", flush=True)
finally:
    llm.shutdown()
