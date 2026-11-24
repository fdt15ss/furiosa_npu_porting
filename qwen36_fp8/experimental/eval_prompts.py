"""Prompts shared by the NPU and CPU comparison scripts."""
BODY = "The quick brown fox jumps over the lazy dog near the quiet river bank. "

PROMPTS = {
    "code": "Write a Python function that reverses a string. Answer with only the code.",
    "korean": "대한민국의 수도는 어디인가요? 한 문장으로 답하세요.",
    "reason": "A train travels 120 km in 1.5 hours. What is its average speed? Show the calculation.",
    "long_200": BODY * 12 + "\nHow many sentences are repeated above? Answer briefly.",
    "long_500": BODY * 32 + "\nSummarize the text above in one sentence.",
}
MAX_TOKENS = 64
