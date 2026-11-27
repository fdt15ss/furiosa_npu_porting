# Furiosa RNGD YOLOE-26 컴파일 및 NPU 측정 결과

측정일: 2026-11-27. 환경: `yoloe26/.venv` (Python 3.11.18), `furiosa-torch` 2026.3.0, 펌웨어 `2026.4.0, 93d67f3`, RNGD 1장(코어 8개, `furiosa:0`~`furiosa:7`), 입력 640×640, COCO 80 클래스 텍스트 임베딩(`mobileclip2_b.ts`) 융합.

## 결과 요약

| 모델 | 태스크 | NPU 구간 | CPU 구간 | 컴파일 시간 | CPU 일치 (실측) |
|---|---|---|---|---|---|
| YOLOE-26n-det | Detection (fused vocab) | 모델 전체 | 박스 복원, NMS | 62.6분 (3758.6s) | same_count 13/13 (min_cos 0.99879) |
| YOLOE-26n-seg | Segmentation (fused vocab) | 본체 + Proto (그래프 2개) | NMS, 마스크 조립 | 68.9분 (4134.8s) + 0.5초 | bus IoU min=0.9935 mean=0.9970, zidane IoU min=0.8299 mean=0.9396 |

## FPS

입력 전처리 제외, 같은 이미지(`bus.jpg`) 반복 측정. `raw`는 NPU 추론과 출력 복사, `full`은 CPU 후처리(디코드·NMS)까지 포함합니다.

| 모델 | raw 1코어 | raw 8코어 | full 1코어 | full 8코어 |
|---|---|---|---|---|
| YOLOE-26n Detection | 169.0 | 946.3 | 138.0 | 601.0 |
| YOLOE-26n-seg | 129.0 | 677.6 | 56.1 | 251.8 |

- YOLOE-26n Detection CPU 기준값: 20.7 FPS (12 threads), 20.8 FPS (12 threads), 20.8 FPS (12 threads)
- 영상 파일 처리 (코어 1개, 읽기·전처리·저장 포함): YOLOE-26n Detection 62.9 FPS, YOLOE-26n-seg 51.5 FPS

## 1코어와 2코어 실험 (YOLOE-26n Detection)

| 모델 | 입력 | 지연 p50 (`npu only`) | 단일 스트림 FPS | 출력 복사 포함 FPS (`npu+copy`) | 컴파일 |
|---|---|---|---|---|---|
| 1-PE `yoloe-26n-det.edf` | 3채널 | 4.06 ms | 245.8 | 166.8 | 62.6분 (3758.6s), 최대 Δ53.9 GiB |
| 1-PE `yoloe-26n-det-1pe-pad4.edf` | 4채널 | 2.89 ms | 344.3 | 212.2 | 63.4분 (3804.6s), 최대 Δ53.8 GiB |
| 2-PE `yoloe-26n-det-2pe.edf` | 3채널 | 97.36 ms | 8.7 | 10.1 | 49.5분 (2967.2s), 최대 Δ39.7 GiB |
| 2-PE `yoloe-26n-det-2pe-pad4.edf` | 4채널 | 2.36 ms | 421.9 | 236.0 | 49.4분 (2964.8s), 최대 Δ41.1 GiB |

코어 수를 맞춘 병렬 처리량(`npu+copy`, 장치당 프로세스 1개):

| 사용 PE | 1-PE 복제 (4채널) | 2-PE 묶음 (4채널) |
|---|---|---|
| 2 | 421.4 | 236.0 |
| 4 | - | 445.5 |
| 8 | 979.0 | 851.1 |

## 실행 및 로그

```bash
cd yoloe26
.venv/bin/python -u run_full_bench.py
```

- 전체 단계 로그: `yoloe26/bench_logs/` (`master.log`, `summary.json`)
- 생성된 이미지·영상 결과: `/root/programming/npu/media/result`
