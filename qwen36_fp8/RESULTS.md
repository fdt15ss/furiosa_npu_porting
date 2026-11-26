# Qwen3.6-27B FP8 양자화 및 Furiosa RNGD NPU 실측 결과

측정일: 2026-11-26. 모델: `Logics-MLLM/Logics-SWE-Qwen3.6-27B` (`42c206e3eee5f771e1ad53ca201957c30f41bf4e`).
환경: CPU 검증 `furiosa02`, NPU 빌드·실행 `qwen36_npu_venv` (`furiosa-llm`/`furiosa-tcc 2026.5.0b3`, `furiosa-kernels 2026.4.0`), 작업 디렉터리 `/root/.cache/qwen36_work`.

## 단계별 실행 요약

| 단계 | 상태 | 소요 시간 |
|---|---|---|
| `01_download` | 성공 | 8.9분 (535.0s) |
| `02_verify_shards` | 성공 | 10.0초 |
| `03_quant_fp8` | 성공 | 2.8분 (165.0s) |
| `04_make_config` | 성공 | 5.0초 |
| `05_check_fp8` | 성공 | 5.0초 |
| `06_cpu_smoke` | 성공 | 2.3분 (135.1s) |
| `07_fxb_dry_run` | 실패 (exit=1) | 5.0초 |
| `08_build_fxb` | 성공 | 53.8분 (3225.3s) |
| `09_npu_smoke` | 성공 | 30.0초 |
| `10_npu_eval` | 성공 | 20.0초 |
| `11_cpu_ref` | 성공 | 4.7분 (280.0s) |
| `12_compare_eval` | 실패 (exit=1) | 5.0초 |
| `13_prefill_bench` | 성공 | 30.0초 |

## 양자화 및 CPU 검증

- 텐서 구성 (`check_fp8.py`): tensors 1584 official(no mtp) 1584 missing 0 extra 0
- FP8 출력 크기: 30.4 GB
- BF16 대비 FP8 logits (`cpu_smoke.py`): cosine 0.9819061756134033 rmse 0.35306262969970703 top1 match True

## NPU 빌드 및 스모크 테스트

- FXB 빌드 (`build_fxb.sh`, 256토큰 프리필): 53.8분 (3225.3s), 크기 107.3 MB
- NPU 로드 시간 (`npu_smoke.py`): 15s
- CPU 스모크 문자열 일치 (`MATCHES_CPU`): True

## NPU vs CPU 토큰 비교 (`npu_eval.py` vs `cpu_ref.py`)

```text
code: prompt 28 tokens, NPU 25, CPU 25, same prefix 25, MATCH
korean: prompt 26 tokens, NPU 11, CPU 11, same prefix 11, MATCH
reason: prompt 37 tokens, NPU 64, CPU 64, same prefix 19, DIFF
  NPU: 'To find the average speed of the train, we use the standard formula for speed.\n\n### Step 1: Identify the Formula\nThe formula for average speed is:\n$$ \\text{Spee'
  CPU: 'To find the average speed of the train, we use the standard formula for speed.\n\n### 1. Identify the Formula\nThe formula for average speed is:\n$$ \\text{Speed} = '
long_200: prompt 203 tokens, NPU 5, CPU 5, same prefix 5, MATCH
long_500: prompt 503 tokens, NPU 30, CPU 30, same prefix 30, MATCH
```

| 프롬프트 | 입력 토큰 | NPU 생성 토큰 (시간, tok/s) | CPU 생성 토큰 (시간) |
|---|---|---|---|
| `code` | 28 | 25 (0.77s, 32.5 tok/s) | 25 (31.5s) |
| `korean` | 26 | 11 (0.34s, 32.0 tok/s) | 11 (15.1s) |
| `reason` | 37 | 64 (1.84s, 34.9 tok/s) | 64 (75.4s) |
| `long_200` | 203 | 5 (0.21s, 24.1 tok/s) | 5 (20.4s) |
| `long_500` | 503 | 30 (0.98s, 30.7 tok/s) | 30 (64.2s) |

## 프리필 지연 측정 (`prefill_bench.py`, 프리픽스 캐시 미사용)

| 목표 길이 | 실제 토큰 | 중앙값 지연 (ms) | 최소 지연 (ms) |
|---|---|---|---|
| 120 | 119 | 62.5 | 62.5 |
| 250 | 249 | 81.9 | 81.8 |
| 380 | 379 | 140.1 | 140.0 |
| 500 | 499 | 159.7 | 159.6 |
| 1000 | 999 | 314.9 | 314.9 |
| 1500 | 1499 | 470.5 | 470.4 |
| 1900 | 1899 | 606.4 | 606.0 |

## 로그 경로

- 전체 로그: `qwen36_fp8/bench_logs/` (`master.log`, `summary.json`, `eval_npu.json`, `eval_cpu.json`, `prefill_bench.json`)
- 모델/FXB 작업 폴더: `/root/.cache/qwen36_work`
