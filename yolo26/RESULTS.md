# Furiosa RNGD 비전 모델 컴파일 및 NPU 측정 결과

측정일: 2026-11-26. 환경: `yolo26/.venv` (Python 3.11.18), `furiosa-torch` 2026.3.0, 펌웨어 `2026.4.0, 93d67f3`, RNGD 1장(코어 8개, `furiosa:0`~`furiosa:7`), 입력 640×640.

## 결과 요약

| 모델 | 태스크 | NPU 구간 | CPU 구간 | 컴파일 시간 | CPU 일치 (실측) |
|---|---|---|---|---|---|
| YOLO11n-Pose | Pose | 모델 전체 | 박스 복원, NMS | 실패/미측정 | matched 4/4 (box 0.30px), matched 2/2 (box 0.13px) |
| YOLO26n-Pose | Pose | 모델 전체 | 박스 복원, NMS | 실패/미측정 | matched 4/4 (box 0.68px), matched 2/2 (box 0.30px) |
| YOLO26n | Detection | 모델 전체 | 박스 복원, NMS | 65.1분 (3907.2s) | same_count 13/13 (min_cos 0.99970) |
| YOLO26n-seg | Segmentation | 본체 + Proto (그래프 2개) | NMS, 마스크 조립 | 69.2분 (4150.7s) + 2.6분 (157.3s) | bus IoU min=0.9970 mean=0.9983, zidane IoU min=0.9912 mean=0.9958 |
| D-FINE nano | Detection | Backbone + Encoder | Decoder | 34.2분 (2054.2s) | bus conf 0.0078 box 1.17px, zidane conf 0.0131 box 0.70px |

## FPS

입력 전처리 제외, 같은 이미지(`bus.jpg`) 반복 측정. `raw`는 NPU 추론과 출력 복사, `full`은 CPU 후처리(디코드·NMS)까지 포함합니다.

| 모델 | raw 1코어 | raw 8코어 | full 1코어 | full 8코어 |
|---|---|---|---|---|
| YOLO11n-Pose | 157.6 | 862.2 | 128.8 | 542.8 |
| YOLO26n-Pose | 198.5 | 1160.1 | 170.2 | 822.9 |
| YOLO26n Detection | 213.2 | 1239.9 | 177.3 | 810.2 |
| YOLO26n-seg | 131.7 | 687.9 | 68.6 | 261.2 |
| D-FINE nano | - | - | 37.9 | 110.1 |

- YOLO26n Detection CPU 기준값: 21.2 FPS (12 threads), 21.3 FPS (12 threads), 21.3 FPS (12 threads)
- D-FINE nano 1코어(2 threads) 중앙값 지연: {'npu': 5.5, 'decoder': 20.6}
- D-FINE nano 8코어(1 thread) 중앙값 지연: {'npu': 15.7, 'decoder': 55.8}
- 영상 파일 처리 (코어 1개, 읽기·전처리·저장 포함): YOLO26n Detection 70.4 FPS, YOLO26n-seg 51.2 FPS, D-FINE nano 26.7 FPS, YOLO26n-Pose 62.5 FPS, YOLO11n-Pose 56.4 FPS

## 1코어와 2코어 실험 (YOLO26n Detection)

| 모델 | 입력 | 지연 p50 (`npu only`) | 단일 스트림 FPS | 출력 복사 포함 FPS (`npu+copy`) | 컴파일 |
|---|---|---|---|---|---|
| 1-PE `yolo26n.edf` | 3채널 | 4.00 ms | 248.4 | 210.9 | 65.1분 (3907.2s), 최대 Δ52.2 GiB |
| 1-PE `yolo26n-1pe-pad4.edf` | 4채널 | 2.82 ms | 352.8 | 284.6 | 65.1분 (3907.7s), 최대 Δ53.7 GiB |
| 2-PE `yolo26n-2pe.edf` | 3채널 | 86.73 ms | 9.9 | 11.4 | 50.2분 (3009.3s), 최대 Δ39.6 GiB |
| 2-PE `yolo26n-2pe-pad4.edf` | 4채널 | 2.29 ms | 435.2 | 335.4 | 49.8분 (2989.3s), 최대 Δ39.6 GiB |

코어 수를 맞춘 병렬 처리량(`npu+copy`, 장치당 프로세스 1개):

| 사용 PE | 1-PE 복제 (4채널) | 2-PE 묶음 (4채널) |
|---|---|---|
| 2 | 535.7 | 335.4 |
| 4 | - | 643.6 |
| 8 | 1458.6 | 1202.0 |

## 실패 또는 미완료 단계

- `28_compile_yolo11n_pose`: exit=1, timeout=False (`bench_logs/28_compile_yolo11n_pose.log`)
- `29_compile_yolo26n_pose`: exit=1, timeout=False (`bench_logs/29_compile_yolo26n_pose.log`)

## 실행 및 로그

```bash
cd yolo26
.venv/bin/python -u run_full_bench.py
```

- 전체 단계 로그: `yolo26/bench_logs/` (`master.log`, `summary.json`)
- 생성된 이미지·영상 결과: `/root/programming/npu/media/result` 및 `/root/programming/npu/media`
