# FuriosaAI RNGD NPU 포팅 및 벤치마크 실험

FuriosaAI RNGD NPU(1장, 8 PE, 펌웨어 `2026.4.0`) 환경에서 비전 모델(YOLO / D-FINE)과 LLM/VLM(Qwen3.6-27B FP8)을 구동하기 위한 포팅 코드와 실측 결과 모음입니다.

## 디렉터리 구성

| 디렉터리 | 대상 모델 | 핵심 도구 / SDK | 요약 문서 |
|---|---|---|---|
| [yolo26/](yolo26/README.md) | YOLO11n-Pose, YOLO26n-Pose, YOLO26n, YOLO26n-seg, D-FINE nano | `furiosa-torch` 2026.3.0, `ultralytics` 8.4.202, `transformers` 5.22.1 | [yolo26/README.md](yolo26/README.md), [yolo26/RESULTS.md](yolo26/RESULTS.md) |
| [yoloe26/](yoloe26/README.md) | YOLOE-26n-det / YOLOE-26n-seg (Open-Vocabulary, fused text prompt) | `furiosa-torch` 2026.3.0, `ultralytics` 8.4.202, `clip` 1.0 | [yoloe26/README.md](yoloe26/README.md), [yoloe26/RESULTS.md](yoloe26/RESULTS.md) |
| [qwen36_fp8/](qwen36_fp8/README.md) | `Logics-MLLM/Logics-SWE-Qwen3.6-27B` (BF16 → Block FP8) | `furiosa-llm` / `furiosa-tcc` 2026.5.0b3, `furiosa-kernels` 2026.4.0 | [qwen36_fp8/README.md](qwen36_fp8/README.md), [qwen36_fp8/RESULTS.md](qwen36_fp8/RESULTS.md) |

## 최신 실측 요약 (2026-11-26 ~ 2026-11-27)

### 1. Vision 모델 (`yolo26/`)
- **YOLO26n Detection (1-PE 복제 8코어):** `raw` **1239.9 FPS**, `full`(후처리 포함) **810.2 FPS** (CPU 12스레드 21.2~21.3 FPS 대비 약 38배). 4채널 패딩(`pad4`) 적용 시 1-PE 단일 스트림 **284.6 FPS**(`p50 2.82ms`), 8 PE 복제 **1458.6 FPS**.
- **YOLO26n 2-PE Fusion (`pad4`):** 단일 스트림 지연 최소 **`p50 2.29ms` (`435.2 FPS` / copy 포함 `335.4 FPS`)**. 3채널 그대로 2-PE 컴파일 시 입력 DMA 병목으로 `86.73ms`까지 느려지던 문제를 4채널 0-패딩으로 해소.
- **YOLO26n-Pose / YOLO11n-Pose:** `full` 8코어 **822.9 FPS** / **542.8 FPS**, CPU 대비 박스 오차 `0.13~0.68px`.
- **YOLO26n-seg (2-그래프 분리):** `full` 8코어 **261.2 FPS**, 마스크 IoU `0.991~0.998`.
- **D-FINE nano (하이브리드):** Backbone+Encoder NPU + Decoder CPU로 8코어 **110.1 FPS** (`bus`/`zidane` 박스 오차 `0.70~1.17px`).

### 2. Open-Vocabulary Vision 모델 (`yoloe26/`, 2026-11-27)
- **YOLOE-26n Detection (COCO 80 클래스 텍스트 임베딩 융합):** `raw` 8코어 **946.3 FPS** / `full` 8코어 **601.0 FPS** (CPU 12스레드 20.7~20.8 FPS 대비 약 29배), 영상 1코어 **62.9 FPS**, CPU 대비 `same_count 13/13` (`min_cos 0.99879`).
- **YOLOE-26n 1-PE / 2-PE Fusion (`pad4`):** `2-PE pad4` 단일 스트림 최소 지연 **`p50 2.36ms` (`421.9 FPS` / copy 포함 `236.0 FPS`)**, 3채널 2-PE(`97.36ms`) DMA 병목 해소; `1-PE pad4` 8코어 복제 시 **979.0 FPS**.
- **YOLOE-26n-seg (2-그래프 분리):** `raw` 8코어 **677.6 FPS** / `full` 8코어 **251.8 FPS**, 영상 1코어 **51.5 FPS**, `bus` 마스크 IoU `0.9970` (`zidane` `0.9396`), 영상 `same_count 13/13` (`min_cos 0.99760`).

### 3. Qwen3.6-27B FP8 (`qwen36_fp8/`)
- **양자화 정확도:** 공식 FP8 스키마(E4M3, 128×128 블록)와 1,584개 텐서 구성 일치 (`30.4 GB`), `cpu_smoke.py` 기준 BF16 대비 logits 코사인 유사도 **0.9819**, Top-1 일치.
- **1칩(8 PE) FXB 빌드 & 추론:** 빌드 `53.8분` (`107.3 MB`, 256토큰 프리필), NPU 로드 `15s`, 생성 속도 **30.7~34.9 tok/s** (CPU 대비 5개 프롬프트 중 4개 토큰까지 완전 일치, `reason`은 19토큰까지 일치 후 동일 의미의 서식 차이로 분기).
- **프리필 지연 (프리픽스 캐시 미사용):** 119토큰 `62.5 ms`, 249토큰 `81.9 ms`, 499토큰 `159.7 ms`, 999토큰 `314.9 ms`, 1,899토큰 `606.4 ms`.

## 공통 NPU 포팅 주안점

1. **80 GiB 컨테이너 메모리 한도:** 호스트 전체 RAM과 무관하게 컨테이너 cgroup 한도가 80 GiB이므로 대용량 그래프·FXB 컴파일은 반드시 순차 실행하거나 메모리 초과 연산(`Proto26`, `s512` 프리필)을 분리/제한해야 합니다.
2. **컴파일러 미지원 연산 우회 및 그래프 분할:** NPU 컴파일러가 지원하지 않는 연산(`NMS`, `grid_sampler`, `ConvTranspose2d`, `max_pool2d(stride=1)`, `F.pad`)은 동등한 지원 연산(`1×1 Conv + PixelShuffle`, `cat + relu`)으로 치환하거나 CPU 구간으로 분리합니다.
3. **데이터 레이아웃 및 DMA/타입 정렬:** `yolo26`·`yoloe26` 2-PE에서의 3ch→4ch 패딩(`pad4`), `qwen36_fp8`에서의 `g`/`beta`/SSM `bf16→f32` 변환 및 `T=1`/`T=8` EDSL 패치처럼, 컴파일러와 런타임이 요구하는 텐서 형상·타입을 경계에서 맞춰주는 것이 성능과 정합성의 핵심입니다.
4. **멀티칩 커널의 단일 카드 포팅:** 공식 4칩 전제 커널(`qwen36_fp8`)은 `num_chip=1`, `Chip` 힌트 제거, `all_reduce → reshape` 치환으로 단일 RNGD 카드(8 PE)에서 동작하도록 패치했습니다.
