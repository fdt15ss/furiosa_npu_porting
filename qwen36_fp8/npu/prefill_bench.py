"""Prefill latency without prefix-cache hits.
usage: prefill_bench.py <model> <fxb> <out.json> [lengths] [reps]
Each run starts with a fresh random tag, so no two prompts share a first token block.
Measures wall time of generate(max_tokens=1) and prints the median per prompt length."""
import json, random, sys, time
from transformers import AutoTokenizer
from furiosa_llm import LLM, SamplingParams
from furiosa_llm.metadata.config_types import SchedulerConfig

model, fxb, out = sys.argv[1:4]
lengths = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else '120,250,380,500,1000,1500,1900').split(',')]
reps = int(sys.argv[5]) if len(sys.argv) > 5 else 5
tok = AutoTokenizer.from_pretrained(model)
rng = random.Random(0)
words = 'river stone light paper green window quiet number market train garden yellow north metal cloud bridge'.split()
filler = ' '.join(rng.choice(words) for _ in range(4000))
filler_ids = tok(filler, add_special_tokens=False)['input_ids']

def prompt(n, tag):
    head = tok(f'Run {tag}. ', add_special_tokens=False)['input_ids']
    return tok.decode(head + filler_ids[:n - len(head)])

llm = LLM(model, fxb=fxb, max_model_len=2048, max_io_memory_mb=4096,
          scheduler_config=SchedulerConfig(profile_guided_decode_batching=False, enable_npu_sampling=False))
sp = SamplingParams(temperature=0, max_tokens=1)
res = {}
try:
    llm.generate(prompt(64, 'warmup'), sp)
    for n in lengths:
        ts = []
        for r in range(reps + 1):
            p = prompt(n, f'{n}-{r}-{rng.randrange(10**9)}')
            t = time.time()
            o = llm.generate(p, sp)
            dt = time.time() - t
            if r: ts.append(dt)  # first run of each length is warm-up
            plen = len(getattr(o if not isinstance(o, list) else o[0], 'prompt_token_ids', None) or [])
        ts.sort()
        res[n] = {'prompt_tokens': plen, 'median_ms': ts[len(ts) // 2] * 1000, 'min_ms': ts[0] * 1000, 'all_ms': [x * 1000 for x in ts]}
        print(f'RESULT len {n} actual {plen} median {res[n]["median_ms"]:.1f} ms min {res[n]["min_ms"]:.1f} ms', flush=True)
        json.dump(res, open(out, 'w'), indent=1)
finally:
    llm.shutdown()
