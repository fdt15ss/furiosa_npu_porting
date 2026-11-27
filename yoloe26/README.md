# YOLOE-26 Open-Vocabulary 모델 Furiosa RNGD NPU 포팅

`furiosa-torch` `2026.3.0` 환경에서 Ultralytics Open-Vocabulary 모델인 **YOLOE-26 (`yoloe-26n-seg.pt`)** 을 RNGD 1장(8 PE, 펌웨어 `2026.4.0`)에 포팅하고 Detection / Segmentation 성능과 1-PE · 2-PE 퓨전을 측정한 코드 및 결과입니다.

전체 실측 원본 표와 실행 로그는 [RESULTS.md](RESULTS.md)와 `bench_logs/`에 있습니다.

---

## 실측 결과 요약 (2026-11-27)

입력 해상도 `640×640`, COCO 80 클래스 텍스트 임베딩(`mobileclip2_b.ts`)을 모델 가중치에 사전 융합(`head.fuse(pe)`)한 기준입니다. `raw`는 NPU 추론 + 출력 CPU 복사, `full`은 CPU 후처리(디코드·NMS) 포함 기준입니다.

| 모델 | 태스크 | NPU / CPU 분할 | 컴파일 시간 | CPU 일치 (실측) | `raw` 1코어 / 8코어 (FPS) | `full` 1코어 / 8코어 (FPS) | 영상 1코어 (FPS) |
|---|---|---|---|---|---|---|---|
| YOLOE-26n-det | Detection (fused vocab) | 전체 NPU / 박스 복원·NMS CPU | `62.6분` (`3758.6s`) | `same_count 13/13`, `min_cos 0.99879` | 169.0 / 946.3 | 138.0 / 601.0 | 62.9 |
| YOLOE-26n-seg | Segmentation (fused vocab) | 본체 + Proto (NPU 2그래프) / NMS·마스크 CPU | `68.9분` (`4134.8s`) + `0.5초`\* | `bus` IoU `0.9970`, `zidane` IoU `0.9396` | 129.0 / 677.6 | 56.1 / 251.8 | 51.5 |

- **CPU 기준값 (`YOLOE-26n-det`, 12 threads):** `20.7 ~ 20.8 FPS` (`full` 8코어 `601.0 FPS` 기준 약 29배, `raw` 8코어 `946.3 FPS` 기준 약 45배).
- **단일 이미지 정합성:** `bus.jpg` 기준 `YOLOE-26n-det`은 CPU·NPU 모두 6개 객체 클래스 100% 일치(박스 오차 `0.37 ~ 0.59px`), `YOLOE-26n-seg`는 `bus.jpg` 마스크 IoU `min=0.9935 / mean=0.9970`, `zidane.jpg` 마스크 IoU `min=0.8299 / mean=0.9396` 및 영상(`car-detection`) `same_count 13/13`, `min_cos 0.99760`을 기록했습니다.

> \* `Proto26` 헤드(`yoloe-26n-seg-protohead.edf`)는 그래프 구조가 `yolo26`과 동일해 컴파일러 캐시가 적중하며 `0.5초`(메모리 증분 `0.0 GiB`) 만에 완료되었습니다. 이후 [seg_npu.py](seg_npu.py) `check`/`files`/`fps` 단계에서 실제 NPU 로드 및 추론이 정상 검증되었습니다.

### YOLOE-26n Detection 1-PE vs 2-PE (`pad4`) 실험

| 구성 | 입력 채널 | 지연 p50 (`npu only`) | 단일 스트림 FPS (`npu only` / `npu+copy`) | 8 PE 병렬 처리량 (`npu+copy`) |
|---|---|---|---|---|
| 1-PE `yoloe-26n-det.edf` | 3채널 | `4.06 ms` | `245.8` / `166.8` | - |
| 1-PE `yoloe-26n-det-1pe-pad4.edf` | 4채널 | `2.89 ms` | `344.3` / `212.2` | **`979.0`** (8×1-PE) |
| 2-PE `yoloe-26n-det-2pe.edf` | 3채널 | `97.36 ms` | `8.7` / `10.1` | - |
| 2-PE `yoloe-26n-det-2pe-pad4.edf` | 4채널 | **`2.36 ms`** | **`421.9` / `236.0`** | `851.1` (4×2-PE) |

- 2-PE / 4-PE 확장 처리량(`npu+copy`): 2 PE 기준 `1-PE pad4 × 2` **`421.4 FPS`** vs `2-PE pad4 × 1` `236.0 FPS`, 4 PE 기준 `2-PE pad4 × 2` **`445.5 FPS`**.

---

## NPU 구동을 위한 코드 포팅 주안점

### 1. Text Prompt Embedding 사전 계산 및 가중치 융합 (`head.fuse(pe)`)
- **문제:** `YOLOE`는 원래 `mobileclip2_b.ts` 텍스트 인코더 → `reprta` → `BNContrastiveHead(cv4)`로 이미지 특징과 텍스트 임베딩의 유사도를 계산하므로, 그대로 export하면 CLIP 텍스트 인코더와 동적 어휘 경로까지 그래프에 포함됩니다.
- **해결:** [compile_model.py](compile_model.py)의 `load_coco80_pe`에서 대상 클래스(기본 COCO 80 클래스)의 텍스트 임베딩(`coco80_pe.pt`)을 한 번만 계산·캐시한 뒤 `y.set_classes(COCO80, pe)`와 `head.fuse(pe)`를 호출합니다. 이로써 텍스트 임베딩이 `cv3[-1]` `1×1 Conv2d` 가중치로 흡수되고 `BNContrastiveHead`와 `reprta`가 제거되어, NPU 추론 시 CLIP 연산 없이 단일 이미지 입력(`1×3×640×640`)만으로 동작합니다.

### 2. Detection-only 경로 분리 및 `YOLOEDetect._inference` 직접 호출
- **문제:** `yoloe-26n-seg.pt` 기반 모델의 `YOLOESegment26` 헤드에서 마스크 계수 분기(`head.cv5 = None`)를 생략해 Detection 전용 모델을 만들면, CPU 후처리 시 `YOLOESegment26._inference`가 `mask_coefficient` 키를 찾다가 `KeyError`가 발생합니다.
- **해결:** [compile_model.py](compile_model.py)에서 `yoloe-26n-det` 모드일 때 `head.cv5 = None`, `head.proto = PassThrough()`로 불필요한 헤드 연산을 제거하고, [detect_npu.py](detect_npu.py)의 `decode`에서는 `YOLOEDetect._inference(head, {"boxes": ..., "scores": ..., "feats": ...})`를 직접 호출해 박스와 클래스 확률만 복원한 뒤 `non_max_suppression`을 수행합니다.

### 3. Dummy `torch.zeros` Export 금지 (실제 전처리 이미지 사용)
- **문제:** `torch.zeros`로 `torch.export` + `decompose`하면 컴파일은 통과해도 실제 이미지 추론 시 출력이 어긋날 수 있습니다.
- **해결:** [compile_model.py](compile_model.py)의 `example_input()`에서 실제 이미지(`bus.jpg`, LetterBox `640×640`, RGB 정규화 `[0, 1]`)를 입력으로 사용해 export/decompose를 수행하고, 추론 시 CPU 레퍼런스와 박스·마스크 IoU·코사인 유사도를 교차 검증합니다([run_detect.py](run_detect.py), [bench_fusion.py](bench_fusion.py), [seg_npu.py](seg_npu.py)).

### 4. 2-PE Fusion 시 3채널 입력 DMA 병목 해결 (`pad4`)
- **문제:** `ft.set_fusion(2)`로 PE 2개를 묶고 3채널 입력(`1×3×640×640`)을 그대로 컴파일하면 입력 DMA 병목으로 p50 지연이 `97.36 ms` (`8.7 FPS`)까지 급락합니다.
- **해결:** [compile_fusion.py](compile_fusion.py)의 `build_padded` / `pad_input`에서 입력 텐서에 0 채널을 1개 붙여 4채널(`1×4×640×640`)로 만들고, 첫 번째 `Conv2d` 가중치에도 해당 채널에 대해 0 가중치를 이어 붙였습니다. 출력은 수학적으로 동일하며 2-PE 지연이 **`2.36 ms` (`421.9 FPS`)**, 1-PE 지연도 `4.06 ms → 2.89 ms`로 단축됩니다.
- **결론:** 단일 스트림 최저 지연이 목표면 `2-PE pad4`(`2.36 ms`), 카드 전체 최대 처리량(Throughput)이 목표면 `1-PE pad4`를 코어마다 복제(8 프로세스, `979.0 FPS`)하는 구성이 유리합니다.

### 5. `ConvTranspose2d` 대체 및 2-그래프 분리 (`YOLOE-26n-seg`)
- **문제:** `Proto26`의 `ConvTranspose2d(c, c, 2, 2)`는 컴파일러에서 미지원이며, 이를 우회해 본체와 한 그래프로 합쳐 컴파일하면 80 GiB 컨테이너 메모리 한도를 초과합니다.
- **해결:**
  - [compile_model.py](compile_model.py)의 `pixel_shuffle_upsample`로 `ConvTranspose2d`를 수학적으로 동등한 `1×1 Conv2d(in, out*4, 1) + nn.PixelShuffle(2)`로 치환했습니다.
  - 본체는 `head.proto = PassThrough()`로 끊어 `yoloe-26n-seg.edf`로 컴파일하고, `ProtoHead`는 [compile_proto.py](compile_proto.py)에서 별도 그래프(`yoloe-26n-seg-protohead.edf`)로 컴파일했습니다.
  - [seg_npu.py](seg_npu.py)는 본체의 `feats` 출력을 CPU로 내리지 않고 NPU 디바이스에 둔 채 `pm(f0, f1, f2)`를 연속 실행합니다.

### 6. 호스트 CPU 스레드 경합 방지 및 순차 컴파일
- **스레드 제한:** 멀티코어/영상 파이프라인([detect_npu.py](detect_npu.py), [run_detect.py](run_detect.py), [bench_fusion.py](bench_fusion.py), [seg_npu.py](seg_npu.py))에서 워커당 `torch.set_num_threads(1)`을 지정해 CPU 후처리 구간에서의 스레드 경합을 차단했습니다.
- **메모리 관리:** 컴파일당 최대 Δ39.7~55.5 GiB(익명 메모리)를 사용하므로 80 GiB 컨테이너 한도 내에서 [run_full_bench.py](run_full_bench.py)로 반드시 한 번에 하나씩 순차 컴파일합니다.

---

## 실행 방법

```bash
cd yoloe26

# 전체 컴파일 + 정확도/FPS/퓨전/영상 벤치마크 순차 실행 및 RESULTS.md 생성
.venv/bin/python -u run_full_bench.py

# 이미 수집된 bench_logs/ 기반 RESULTS.md 재렌더링
.venv/bin/python -u run_full_bench.py --render-only
```

개별 테스트:

```bash
.venv/bin/python -u run_detect.py                              # YOLOE-26n-det 이미지/영상 추론 -> /root/programming/npu/media/result
.venv/bin/python -u detect_npu.py                              # YOLOE-26n-det 1/2/4/8코어 FPS 및 CPU 기준값
.venv/bin/python -u seg_npu.py check                           # YOLOE-26n-seg CPU 대비 박스·마스크 IoU 검증
.venv/bin/python -u seg_npu.py files                           # YOLOE-26n-seg 이미지/영상 추론
.venv/bin/python -u seg_npu.py fps                             # YOLOE-26n-seg FPS
.venv/bin/python -u bench_fusion.py 2 yoloe-26n-det-2pe-pad4.edf  # 2-PE pad4 지연·처리량 측정
```
