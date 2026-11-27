# Vision 모델 (YOLO / D-FINE) Furiosa RNGD NPU 포팅

`furiosa-torch` `2026.3.0` 환경에서 Ultralytics YOLO 계열(Detection, Pose, Segmentation)과 Hugging Face `D-FINE nano`를 RNGD 1장(8 PE, 펌웨어 `2026.4.0`)에 포팅하고 성능을 측정한 코드 및 결과입니다.

전체 실측 원본 표와 실행 로그는 [RESULTS.md](RESULTS.md)와 `bench_logs/`에 있습니다.

---

## 실측 결과 요약 (2026-11-26)

입력 해상도 `640×640`, `raw`는 NPU 추론 + 출력 CPU 복사, `full`은 CPU 후처리(디코드·NMS) 포함 기준입니다.

| 모델 | 태스크 | NPU / CPU 분할 | 컴파일 시간 | CPU 일치 (실측) | `raw` 1코어 / 8코어 (FPS) | `full` 1코어 / 8코어 (FPS) | 영상 1코어 (FPS) |
|---|---|---|---|---|---|---|---|
| YOLO11n-Pose | Pose | 전체 NPU / NMS CPU | `52.1분` (`3127.7s`)* | matched 4/4 (`0.30px`), 2/2 (`0.13px`) | 157.6 / 862.2 | 128.8 / 542.8 | 56.4 |
| YOLO26n-Pose | Pose | 전체 NPU / NMS CPU | `65.3분` (`3916.1s`)* | matched 4/4 (`0.68px`), 2/2 (`0.30px`) | 198.5 / 1160.1 | 170.2 / 822.9 | 62.5 |
| YOLO26n | Detection | 전체 NPU / NMS CPU | `65.1분` (`3907.2s`) | `same_count 13/13`, `min_cos 0.99970` | 213.2 / 1239.9 | 177.3 / 810.2 | 70.4 |
| YOLO26n-seg | Seg | 본체 + Proto (NPU 2그래프) / NMS CPU | `69.2분` + `2.6분` | `bus` IoU `0.9983`, `zidane` IoU `0.9958` | 131.7 / 687.9 | 68.6 / 261.2 | 51.2 |
| D-FINE nano | Detection | Backbone+Encoder NPU / Decoder CPU | `34.2분` (`2054.2s`) | `bus` `1.17px`, `zidane` `0.70px` | - | 37.9 / 110.1 | 26.7 |

> \* `compile_pose.py`는 `.edf` 저장(`compile ok 3127.7s` / `3916.1s`)까지 완료한 뒤 스크립트 하단 인라인 검증(`rt.to(dev)`)에서 `ir.Runnable`에 `.to()` 메서드가 없어 `exit=1`로 종료되었습니다. 실제 NPU 검증과 FPS 측정은 `CompileModule`을 사용하는 [infer_pose.py](infer_pose.py), [fps_pose.py](fps_pose.py), [video_pose.py](video_pose.py)에서 모두 정상 통과했습니다.

### YOLO26n 1-PE vs 2-PE (`pad4`) 실험

| 구성 | 입력 채널 | 지연 p50 (`npu only`) | 단일 스트림 FPS (`npu only` / `npu+copy`) | 8 PE 병렬 처리량 (`npu+copy`) |
|---|---|---|---|---|
| 1-PE `yolo26n.edf` | 3채널 | `4.00 ms` | `248.4` / `210.9` | - |
| 1-PE `yolo26n-1pe-pad4.edf` | 4채널 | `2.82 ms` | `352.8` / `284.6` | **`1458.6`** (8×1-PE) |
| 2-PE `yolo26n-2pe.edf` | 3채널 | `86.73 ms` | `9.9` / `11.4` | - |
| 2-PE `yolo26n-2pe-pad4.edf` | 4채널 | **`2.29 ms`** | **`435.2` / `335.4`** | `1202.0` (4×2-PE) |

---

## NPU 구동을 위한 코드 포팅 주안점

### 1. 추론 그래프와 후처리(NMS·디코딩) 분리
- **문제:** NMS, 동적 박스 필터링, 좌표 역변환은 제어 흐름과 동적 텐서 연산 때문에 NPU 컴파일이 불가능하거나 비효율적입니다.
- **해결:**
  - Detection / Segmentation ([compile_model.py](compile_model.py)): `head.training = True`로 설정하고 `forward`를 래핑해 `feats`를 제외한 raw 예측 텐서(`boxes`, `scores` 등)만 NPU 그래프로 export하고, `head._inference(...)` + `non_max_suppression`은 CPU에서 실행합니다([detect_npu.py](detect_npu.py), [run_detect.py](run_detect.py)).
  - Pose ([compile_pose.py](compile_pose.py), [infer_pose.py](infer_pose.py)): `YOLO26n-Pose`(`Pose26`)는 `model.fuse()` 이후 `head.training = True`를 쓰면 fuse로 제거된 sigma 분기를 타려다 실패하므로, `head.forward_head(x, **branch)`를 직접 호출하는 `raw_forward`로 교체해 `boxes`, `scores`, `kpts`만 export했습니다.

### 2. Dummy `torch.zeros` Export 금지 (실제 전처리 이미지 사용)
- **문제:** `torch.zeros`로 `torch.export` + `decompose`하면 컴파일은 에러 없이 통과하지만 실제 이미지 추론 시 출력이 틀어집니다.
- **해결:** [compile_model.py](compile_model.py), [compile_pose.py](compile_pose.py), [compile_dfine.py](compile_dfine.py) 모두 실제 이미지(`bus.jpg`, LetterBox/Resize `640×640`, RGB 정규화 `[0, 1]`)를 입력으로 사용해 export/decompose를 수행하고, 추론 시 CPU 레퍼런스와 박스·IoU·코사인 유사도를 교차 검증합니다.

### 3. 2-PE Fusion 시 3채널 입력 DMA 병목 해결 (`pad4`)
- **문제:** `ft.set_fusion(2)`로 PE 2개를 묶고 3채널 입력(`1×3×640×640`)을 그대로 컴파일하면 단일 입력 DMA 전송이 프레임당 ~84ms를 점유해 p50 지연이 `86.73 ms` (`9.9 FPS`)로 급락합니다.
- **해결:** [compile_fusion.py](compile_fusion.py)의 `build_padded` / `pad_input`에서 입력 텐서에 0 채널을 1개 붙여 4채널(`1×4×640×640`)로 만들고, 첫 번째 `Conv2d` 가중치에도 해당 입력 채널에 대해 0 가중치를 이어 붙였습니다. 수학적으로 출력은 100% 동일하지만 DMA 병목이 해소되어 2-PE 지연이 **`2.29 ms` (`435.2 FPS`)**, 1-PE 지연도 `4.00 ms → 2.82 ms`로 개선됩니다.
- **결론:** 단일 스트림 최저 지연이 목표면 `2-PE pad4`, 카드 전체 최대 처리량(Throughput)이 목표면 `1-PE pad4`를 코어마다 복제(8 프로세스, `1458.6 FPS`)하는 구성이 유리합니다.

### 4. `ConvTranspose2d` 대체 및 2-그래프 분리 (`YOLO26n-seg`)
- **문제:**
  1. `Proto26`의 `ConvTranspose2d(c, c, 2, 2)`는 컴파일러가 미지원(`transposed convolution is not implemented`).
  2. `ConvTranspose2d`를 대체해 본체와 `Proto26`을 한 그래프로 묶으면 80 GiB 컨테이너 메모리 한도를 초과해 컴파일 중 OOM 종료됩니다.
- **해결:**
  - [compile_model.py](compile_model.py)의 `pixel_shuffle_upsample`로 `ConvTranspose2d` 가중치를 재배치해 `1×1 Conv2d(in, out*4, 1) + nn.PixelShuffle(2)`로 수학적으로 동일하게 치환했습니다.
  - 본체 컴파일 시에는 `head.proto = PassThrough()`로 끊어 `yolo26n-seg.edf`를 만들고, [compile_proto.py](compile_proto.py)로 `ProtoHead`만 별도 그래프(`yolo26n-seg-protohead.edf`, 컴파일 2.6분)로 컴파일했습니다.
  - [seg_npu.py](seg_npu.py)는 본체의 `feats` 출력 텐서를 CPU로 내리지 않고 NPU 디바이스에 둔 채 `pm(f0, f1, f2)`를 바로 이어서 실행합니다.

### 5. `D-FINE nano` 연산 우회 및 하이브리드 분할 ([compile_dfine.py](compile_dfine.py))
- **문제:**
  1. Decoder의 `grid_sampler`가 NPU 컴파일러에서 미지원.
  2. Stem의 `max_pool2d(kernel=2, stride=1)`이 컴파일러 패닉(`empty transition cost table`)을 유발하고, `F.pad`는 Conv 직전 외에는 컴파일 실패.
  3. `torch.export` 시 `torch_compilable_check`가 `.item()`을 삽입하며, SDPA/BatchNorm이 그래프 분해에서 문제를 일으킴.
- **해결:**
  - **구간 분할:** Backbone + Hybrid Encoder만 NPU(`dfine-nano-encoder.edf`)로 내리고, Decoder는 CPU(`run_decoder`)에서 실행합니다. (메모리 부담을 줄이기 위해 4-part 분할 빌드인 [compile_dfine_parts.py](compile_dfine_parts.py)도 함께 제공)
  - **연산 치환 (`Stem`, `max_pool_2x2`):** `F.pad` 대신 0 행렬 `torch.cat`으로 우측·하단을 패딩하고, `max_pool2d(2, 1)` 대신 `b + torch.relu(a - b)`로 2×2 최댓값을 계산해 동일한 결과를 얻습니다.
  - **Export 패치:** `modeling.torch_compilable_check` 무력화, `attn_implementation="eager"`, `Conv+BN` 및 `DFineFrozenBatchNorm2d` 가중치 융합(`fuse_conv_frozen_bn`), `ft.CompileModule.from_exported`로 전체 SDK 그래프 패스 적용.

### 6. 호스트 CPU 스레드 경합 방지 및 순차 컴파일
- **스레드 제한:** 멀티코어/영상 파이프라인([detect_npu.py](detect_npu.py), [fps_pose.py](fps_pose.py), [video_pose.py](video_pose.py))에서 워커당 `torch.set_num_threads(1)`을 지정했습니다. 기본 스레드 수를 그대로 두면 CPU 디코드·NMS에서 스레드 경합이 발생해 프레임당 후처리 시간이 수십 ms(77 ms → 0.8 ms)까지 튈 수 있습니다.
- **메모리 관리:** 컴파일당 최대 Δ39.6~53.7 GiB(익명 메모리)를 사용하므로 80 GiB 컨테이너 한도에서 반드시 한 번에 하나씩 순차 컴파일해야 합니다([run_full_bench.py](run_full_bench.py)).

---

## 실행 방법

```bash
cd yolo26

# 전체 모델 컴파일 + 정확도/FPS/퓨전/영상 벤치마크 일괄 실행 및 RESULTS.md 생성
.venv/bin/python -u run_full_bench.py

# 이미 수집된 bench_logs/ 기반 RESULTS.md 재렌더링
.venv/bin/python -u run_full_bench.py --render-only
```

개별 테스트:

```bash
.venv/bin/python -u run_detect.py       # YOLO26n 이미지/영상 추론 -> /root/programming/npu/media/result
.venv/bin/python -u detect_npu.py       # YOLO26n 1/2/4/8코어 FPS
.venv/bin/python -u seg_npu.py check    # YOLO26n-seg CPU 대비 마스크 IoU 검증
.venv/bin/python -u seg_npu.py fps      # YOLO26n-seg FPS
.venv/bin/python -u dfine_npu.py check  # D-FINE CPU 대비 검증
.venv/bin/python -u dfine_npu.py fps    # D-FINE FPS
.venv/bin/python -u fps_pose.py         # YOLO11n-Pose / YOLO26n-Pose FPS
.venv/bin/python -u bench_fusion.py 2 yolo26n-2pe-pad4.edf
```
