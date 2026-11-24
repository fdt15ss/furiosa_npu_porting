# Furiosa RNGD 비전 모델 컴파일 결과

측정일: 2026-11-04. 환경: `furiosa_yolo` conda 환경, `furiosa-torch` 2026.3.0, 펌웨어 2026.4.0, RNGD 1장(코어 8개), 입력 640×640.

## 결과 요약

| 모델 | 태스크 | NPU 구간 | CPU 구간 | 컴파일 시간 | CPU 일치 |
|---|---|---|---|---|---|
| YOLO11n-Pose | Pose | 모델 전체 | 박스 복원, NMS | 41분 | 일치 |
| YOLO26n-Pose | Pose | 모델 전체 | 박스 복원, NMS | 52분 | 일치 |
| YOLO26n | Detection | 모델 전체 | 박스 복원, NMS | 51분 | 일치 |
| YOLO26n-seg | Segmentation | 본체 + Proto (그래프 2개) | NMS, 마스크 조립 | 54분 + 3분 | 마스크 IoU 0.99 이상 |
| D-FINE nano | Detection | Backbone + Encoder | Decoder | 27.5분 | 일치 |

"CPU 일치"는 `bus.jpg`, `zidane.jpg`에서 검출 수와 클래스가 같고 신뢰도와 박스 차이가 작다는 뜻입니다. COCO mAP 같은 공식 정확도는 측정하지 않았습니다.

## FPS

입력 전처리 제외, 같은 이미지 반복. `raw`는 NPU 추론과 전송, `full`은 후처리까지 포함합니다.

| 모델 | raw 1코어 | raw 8코어 | full 1코어 | full 8코어 |
|---|---|---|---|---|
| YOLO11n-Pose | 159 | 1040 | 124 | 728 |
| YOLO26n-Pose | 198 | 1339 | 166 | 1040 |
| YOLO26n Detection | 210 | 1408 | 162 | 1026 |
| YOLO26n-seg | 130 | 879 | 65 | 386 |
| D-FINE nano | - | - | 36 | 168 |

- D-FINE 값은 Decoder까지 포함합니다. 1코어는 CPU 2스레드, 8코어는 코어당 CPU 1스레드입니다.
- YOLO26n Detection의 CPU 기준값은 16스레드에서 약 18~21 FPS입니다. CPU 측정은 실행마다 많이 흔들렸으니 참고만 하세요.

영상 파일 처리(코어 1개, 읽기, 전처리, 저장 포함)는 YOLO26n Detection 57~69 FPS, Segmentation 30~44 FPS, D-FINE 26~29 FPS입니다. Detection과 Segmentation 값에는 30프레임마다 CPU 비교가 포함됩니다.

## 1코어와 2코어 실험 (YOLO26n Detection)

측정일: 2026-11-18. 같은 조건(`bus.jpg` 반복, 전처리 제외)에서 측정했고, 모든 경우에 CPU와 검출 5개와 클래스가 같았습니다. 박스 차이는 1px 미만입니다.

"2코어"는 두 가지 방식으로 시험했습니다.

- **복제:** 1-PE 모델을 코어 2개에 하나씩 올리고 프로세스 2개로 실행
- **묶음(fusion):** `ft.set_fusion(2)`로 PE 2개를 한 장치로 묶고, 그 설정으로 다시 컴파일

| 모델 | 입력 | 지연 p50 | 단일 스트림 FPS | 출력 복사 포함 FPS | 컴파일 |
|---|---|---|---|---|---|
| 1-PE `yolo26n.edf` | 3채널 | 3.97 ms | 250 | 217 | 51분 |
| 1-PE `yolo26n-1pe-pad4.edf` | 4채널 | 2.79 ms | 357 | 289 | 51분, 최대 54 GiB |
| 2-PE `yolo26n-2pe.edf` | 3채널 | 86.6 ms | 10 | 11 | 39분 |
| 2-PE `yolo26n-2pe-pad4.edf` | 4채널 | **2.24 ms** | **446** | **343** | 39분, 최대 40 GiB |

코어 수를 맞춘 처리량(출력 복사 포함, 장치당 프로세스 1개)은 다음과 같습니다.

| 사용 PE | 1-PE 복제 (4채널) | 2-PE 묶음 (4채널) |
|---|---|---|
| 2 | **570** (285 × 2) | 343 |
| 4 | – | 672 (336 × 2) |
| 8 | **1972** (8개) | 1220 (4개) |

### 해석

1. **3채널 입력은 2-PE에서 매우 느립니다.** 프로파일(`ProfileScope.Full`)에서 입력 전송 DMA 하나가 프레임당 약 84ms였습니다. 연산(TuExec) 사이클은 오히려 1-PE보다 25% 적었습니다. 첫 Conv 하나만 있는 그래프로도 재현됩니다(DMA 92ms).
2. **입력에 0 채널을 하나 붙여 4채널로 만들면 해결됩니다.** 첫 Conv 가중치에 0 채널을 붙이므로 결과는 수학적으로 같습니다. 그 DMA는 수십 µs로 줄었고, 1-PE도 지연이 3.97ms에서 2.79ms로 빨라졌습니다. 8채널도 효과가 있습니다. NHWC 입력은 2-PE 컴파일이 실패했고(`TransposedLower`), uint8 입력은 1·2-PE 모두 컴파일이 실패했습니다. `use_dma_bridges_only`나 `enable_tactic_pruning=False`는 효과가 없었습니다.
3. **2-PE 묶음은 지연시간만 줄여줍니다.** 2.79ms에서 2.24ms로 20% 줄었습니다. 같은 PE 2개라면 처리량은 1-PE 복제가 1.7배 높습니다(570 대 343). 작은 모델이라 PE 2개로 나눠도 연산이 반으로 줄지 않기 때문입니다.
4. 1-PE 모델을 fusion=2 장치에 그대로 올리면 실행은 되지만 PE 하나만 쓰고 속도도 1-PE와 같습니다.

결론: 처리량이 목표면 **4채널 1-PE 모델을 코어마다 복제**하고, 스트림 하나의 지연시간이 중요하면 4채널 2-PE 묶음을 쓰세요.

### 재현

```bash
python compile_fusion.py 2 pad4          # yolo26n-2pe-pad4.edf, 1이면 1-PE
python bench_fusion.py 2 yolo26n-2pe-pad4.edf      # 지연, FPS, CPU 비교
python bench_fusion.py 1 yolo26n-1pe-pad4.edf 2    # 장치 2개 병렬 처리량
```

## 컴파일 주의사항

1. **`torch.zeros`로 export하지 마세요.** 에러 없이 컴파일되지만 실제 이미지에서 결과가 틀립니다. 실제 전처리 이미지를 쓰고, 컴파일 후 반드시 CPU와 비교하세요.
2. **메모리 한도는 80GiB입니다.** `free`의 1TB는 서버 전체 값입니다. 긴 컴파일은 한 번에 하나만 돌리세요.
3. **PDF의 `head.training = True`는 YOLO26 Pose에 쓸 수 없습니다.** `fuse()`가 지운 층을 호출합니다. `compile_pose.py`처럼 헤드를 직접 호출합니다.
4. **`decompose`만으로 부족할 수 있습니다.** D-FINE은 `CompileModule.from_exported`로 SDK 패스를 모두 거쳐야 합니다.

## 컴파일러 제약과 대체 방법

| 연산 | 증상 | 대체 |
|---|---|---|
| `ConvTranspose2d` | `transposed convolution is not implemented` | 1×1 Conv + PixelShuffle |
| `max_pool2d(2, stride=1)` | `empty transition cost table` 패닉 | `relu`로 2×2 최댓값 계산 |
| `F.pad` | Conv 바로 앞이 아니면 미지원 | 0 행렬 `cat` |
| `torch.maximum` | 컴파일 실패 | `b + relu(a - b)` |
| `grid_sampler`, `gather`, `topk` | 미지원 (2026.4.0도 `grid_sampler` 동일) | CPU에서 실행 |

Segmentation의 Proto를 본체와 한 그래프로 컴파일하면 80GiB 메모리를 초과합니다. 그래서 별도 그래프로 분리했습니다.

## 남은 한계

- **D-FINE:** Decoder가 CPU에서 프레임당 약 22~41ms를 쓰며, 이것이 FPS를 제한합니다. 필요한 연산을 컴파일러가 지원해야 해결됩니다.
- **검출 품질:** 소형 모델이라 안개 영상에서 오탐이 있습니다. `yolo26n`은 신뢰도 0.35, 도로 클래스만 유지, 화면 30% 이상 박스 제거 조합으로 airplane 오탐을 없애면서 car 240건을 유지했습니다. 기준 0.5만 쓰면 car가 91건으로 줄고 airplane 13건이 남습니다.
- **검증 범위:** n 크기 모델과 640×640 입력만 시험했습니다.

## 실행 방법

```bash
cd /root/furiosa_study/yolo
conda activate furiosa_yolo
python run_detect.py          # Detection 결과 파일
python seg_npu.py files       # Segmentation 결과 파일
python dfine_npu.py files     # D-FINE 결과 파일
python seg_npu.py fps         # fps 측정, detect_npu.py, dfine_npu.py fps도 동일
```

결과 파일은 `media/result`에 저장됩니다.
