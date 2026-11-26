# Qwen3.6-27B 파인튜닝 모델을 RNGD 1장에서 실행

Hugging Face의 파인튜닝 모델을 Furiosa FP8 형식으로 변환하고, RNGD 카드 1장(8 PE)에서 텍스트 추론을 실행하는 절차입니다.

**현재 상태:** 텍스트 추론과 저해상도 동영상 입력(프레임당 256패치 이하)이 NPU에서 동작합니다. 2026-11-09부터 프리필을 256토큰 단위로 처리해, 긴 프롬프트의 프리필이 128토큰 단위보다 약 33% 빠릅니다. 짧은 프롬프트 5개 중 4개는 CPU와 토큰까지 일치했고, 1개는 19번째 토큰부터 달랐습니다. 1,000토큰 이상의 프롬프트는 CPU와 1~20번째 토큰에서 갈라집니다. 비공식 패치와 베타 SDK를 쓰는 방식이며, 큰 프레임과 동시 처리(배치 2 이상)는 동작하지 않습니다.

NPU 실행 절차는 [NPU 실행](#npu-실행)을 보세요.

## 대상 모델

- 모델: `Logics-MLLM/Logics-SWE-Qwen3.6-27B`
- 커밋: `42c206e3eee5f771e1ad53ca201957c30f41bf4e`
- 라이선스: Apache-2.0
- 기반 모델: `Qwen/Qwen3.6-27B`

구조가 공식 모델과 같은 다른 BF16 파인튜닝 모델에도 쓸 수 있지만, 모델 이름과 커밋은 `run.sh`와 `download.py`, `verify_shards.py`에서 바꿔야 합니다.

## 실행

작업 폴더에 약 90GB, 메모리 약 60GB가 필요합니다. SDK 2026.4.0이 설치된 `furiosa02` 환경을 기준으로 합니다.

```bash
./run.sh /path/to/work
```

결과물은 `/path/to/work/fp8`에 생성됩니다. 다른 Python을 쓰려면 `PY=/path/to/python ./run.sh ...`로 지정합니다.

`run.sh`는 양자화와 CPU 검증까지만 다룹니다. 2026-11-06에 단계별로 수행한 명령을 묶은 것으로, 한 번에 처음부터 끝까지 실행해 본 적은 없습니다.

| 순서 | 파일 | 하는 일 |
|---|---|---|
| 1 | `download.py` | 커밋을 고정해 모델 다운로드 |
| 2 | `verify_shards.py` | 12개 파일의 SHA256 검증 |
| 3 | `quant_fp8.py` | FP8 변환 |
| 4 | `make_config.py` | 공식 양자화 설정을 넣은 `config.json` 생성 |
| 5 | `check_fp8.py` | 텐서 이름과 형식을 공식 FP8과 비교 |
| 6 | `cpu_smoke.py` | 원본과 FP8의 출력 비교, 문장 생성 |
| 7 | `fxb build --dry-run` | 공식 빌드 설정 확인 (컴파일 안 함) |

## 양자화 방식

공식 `Qwen/Qwen3.6-27B-FP8`과 같은 방식입니다.

- **형식:** `quant_method: fp8`, E4M3, 128×128 블록
- **블록 스케일:** `블록 절댓값 최대 / 448`, BF16으로 저장 (`weight_scale_inv`)
- **대상:** 공식 파일에서 `weight_scale_inv`가 있는 텐서. 이 모델에서는 400개
- **나머지:** BF16 유지 (비전 인코더, LM head 등)
- **제외:** `mtp.*` 가중치는 파인튜닝 모델에 없음

공식 원본의 텐서 1개로 검증한 결과, 블록 스케일은 100% 일치하고 FP8 값은 98.3%가 바이트까지 같습니다. 복원 오차는 공식 0.02652, 이 구현 0.02647로 사실상 같습니다.

## 검증 결과 (2026-11-06)

| 항목 | 결과 |
|---|---|
| 다운로드 무결성 | 12개 파일 SHA256 일치 |
| 실제 파인튜닝 여부 | 표본 3개 중 2개가 공식 원본과 다름. `norm.weight`는 동일 |
| 텐서 구성 | 1,584개, 공식 FP8(`mtp` 제외)과 일치 |
| 출력 크기 | 30.4GB |
| 공식 빌드 드라이런 | 기본 구성에서 통과, 커널 272개 |
| 원본 대비 logits (CPU, 프롬프트 1개) | 코사인 0.982, 상위 5개 토큰 순서까지 동일 |
| FP8 가중치로 문장 생성 (CPU) | 올바른 파이썬 함수 생성 |

프롬프트 1개로 한 동작 확인이며, 모델 품질 평가가 아닙니다.

## NPU 실행

### 준비물

- RNGD 카드 1장, 펌웨어 2026.4.0
- `furiosa02` 환경 (SDK 2026.4.0, 양자화와 CPU 비교용)
- `furiosa` 환경 (`furiosa-llm`, `furiosa-tcc` 2026.5.0b3, `furiosa-kernels` 2026.4.0, 빌드와 NPU 실행용)
- 메모리 80 GiB. CPU 비교는 혼자서도 약 55 GB를 씁니다. 빌드나 큰 커널 컴파일과 동시에 돌리지 마세요.

### 실행

```bash
cd npu
./build_fxb.sh /root/models/fxb/logics-swe-1card-sdk5-p256.fxb   # 프리필 256토큰 단위. 캐시가 있으면 약 8분
# GDN_MAX_SEQ=128 ./build_fxb.sh ...   로 2026-11-07의 128토큰 단위 FXB를 만듭니다

export PY=/root/miniconda3/envs/furiosa/bin/python FXB=/root/models/fxb/logics-swe-1card-sdk5-p256.fxb
$PY npu_smoke.py /root/models/Logics-SWE-Qwen3.6-27B-FP8 $FXB
$PY npu_eval.py  /root/models/Logics-SWE-Qwen3.6-27B-FP8 $FXB npu.json

# CPU 기준값(약 10분)과 토큰 비교
/root/miniconda3/envs/furiosa02/bin/python cpu_ref.py /root/models/Logics-SWE-Qwen3.6-27B-FP8 cpu.json
python3 compare_eval.py npu.json cpu.json
```

코드에서 쓸 때는 아래 두 설정이 꼭 필요합니다. 없으면 시작 단계에서 패닉이 나거나 NPU 샘플링이 실패합니다.

```python
LLM(model_dir, fxb=fxb, max_model_len=2048, max_io_memory_mb=4096,
    scheduler_config=SchedulerConfig(profile_guided_decode_batching=False, enable_npu_sampling=False))
```

### 결과 (2026-11-07)

모델 로드 10초. 생성 속도 약 35 토큰/초. 프롬프트마다 최대 64토큰을 생성해 FP8 가중치의 CPU 결과와 비교했습니다.

| 프롬프트 | 입력 토큰 | 생성 토큰 | CPU와 비교 |
|---|---|---|---|
| 코드 작성 | 28 | 25 | 일치 |
| 한국어 질문 | 26 | 11 | 일치 |
| 속력 계산 | 37 | 64 | 19번째 토큰부터 다름 |
| 긴 입력 | 203 | 5 | 일치 |
| 긴 입력 | 503 | 30 | 일치 |

속력 계산은 NPU가 `### Step 1:`, CPU가 `### 1.`로 시작했고 내용은 둘 다 같은 흐름입니다. 갈라진 위치에서 CPU의 1, 2위 로짓은 24.125와 24.0으로 BF16 한 칸 차이(확률 35%, 31%)라서 수치 반올림 차이로 보입니다. NPU 로짓을 직접 확인한 것은 아니라 확정은 아닙니다.

프롬프트 5개로 한 시험이며, 모델 품질이나 장시간 안정성을 평가한 것은 아닙니다.

### 프리필 256토큰 단위 (2026-11-09)

2026.5 컴파일러에서는 `gdn_prefill` s256이 컴파일됩니다(2026.4에서는 실패). `build_fxb.sh`의 기본값이 이제 256토큰 단위입니다.

**속도.** 매번 다른 프롬프트로 첫 토큰까지의 시간을 쟀습니다(`prefill_bench.py`, 길이마다 5회의 중앙값). 같은 프롬프트를 반복하면 접두사 캐시가 걸려 프리필을 재지 못합니다.

| 프롬프트 길이 | 128 단위 | 256 단위 |
|---|---|---|
| 119 | 62 ms | 63 ms |
| 249 | 122 ms | 82 ms |
| 499 | 241 ms | 159 ms |
| 999 | 480 ms | 314 ms |
| 1,899 | 898 ms | 605 ms |

청크 하나에 128토큰은 약 60 ms, 256토큰은 약 80 ms가 걸립니다. 생성 속도는 같습니다.

**정확도.** 위 짧은 프롬프트 5개와 긴 프롬프트 3개(`EVAL_SET=long`, 1,000·1,000·1,850토큰)에서 128 단위와 256 단위 FXB의 출력 토큰이 완전히 같습니다. 동영상은 256 단위가 CPU에 더 가깝습니다(아래 동영상 표 참고).

긴 프롬프트는 두 FXB 모두 CPU와 갈라집니다. `cpu_text_check.py`로 NPU 토큰을 CPU에 넣어 보면, NPU가 고른 토큰은 모두 CPU의 1위나 2위입니다. 다만 로짓 차가 짧은 프롬프트(0.125)보다 큽니다.

| 프롬프트 | CPU 1위와 일치 | 최대 로짓 차 |
|---|---|---|
| 코드 1,000토큰 | 62/64 | 0.750 |
| 한국어 1,000토큰 | 23/24 | 1.875 |
| 문서 1,850토큰 | 62/64 | 0.625 |

입력이 길수록 NPU와 CPU의 계산 차이가 쌓이는 것으로 보입니다. 원인은 확인하지 않았습니다. 로그는 `logs/*_2026-11-09.*`에 있습니다.

```bash
EVAL_SET=long $PY npu_eval.py /root/models/Logics-SWE-Qwen3.6-27B-FP8 $FXB npu_long.json
EVAL_SET=long /root/miniconda3/envs/furiosa02/bin/python cpu_text_check.py /root/models/Logics-SWE-Qwen3.6-27B-FP8 npu_long.json
$PY prefill_bench.py /root/models/Logics-SWE-Qwen3.6-27B-FP8 $FXB bench.json
```

### 동영상 입력 (2026-11-08)

FXB를 다시 빌드하지 않고 동작합니다. FXB에서 진짜 비전 커널은 256패치짜리뿐이고, 다른 비전 버킷은 이 커널을 그대로 재사용합니다. 그래서 프레임 하나를 256패치(16×16px 단위) 이하로 줄이면 됩니다. 16:9 영상은 프레임이 320×192가 됩니다.

```bash
cd npu
./serve_video.sh 8000 /root/furiosa_study/media          # 로드 약 10초
/root/miniconda3/envs/furiosa/bin/python video_client.py /root/furiosa_study/media/test_ball.mp4 8
```

- `serve_vision.py`: `furiosa-llm serve`를 감싸서 프로세서가 256 버킷만 보게 합니다. 이게 없으면 큰 프레임이 재사용 커널 버킷으로 가서 오류 없이 틀린 결과가 나옵니다. 이제는 256패치를 넘는 프레임은 오류로 거부됩니다.
- `video_client.py`: 요청마다 `mm_processor_kwargs.video.longest_edge = 256 × 16 × 16 × 프레임 수`를 넣습니다. 이 값은 클립 전체의 픽셀 예산이고, 프로세서가 프레임 수로 나눠 씁니다.
- 프레임 2장이 묶여 64토큰 이하가 됩니다. 8프레임이면 프롬프트가 약 300토큰입니다. 컨텍스트가 2048이라 프레임은 40장 정도가 한계입니다.

**검증.** 영상 2개를 8프레임으로 보내 NPU 답을 CPU와 비교했습니다. `MM_DUMP=<dir>`로 서버를 띄우면 NPU가 쓴 프롬프트 ID와 픽셀값이 저장되고, `cpu_video_ref.py`와 `cpu_video_check.py`가 같은 입력으로 CPU를 돌립니다(서버를 내리고 `furiosa02` 환경에서 실행).

| 영상 | NPU 답 | CPU 1위 토큰과 일치 | 다른 위치 |
|---|---|---|---|
| `test_ball.mp4` (640×360, 4초), 128 단위 | 회색 배경, 빨간 원이 왼쪽→오른쪽, 파란 사각형이 상단에 나타남 | 47/50 | 3곳, 모두 CPU 2위, 로짓 차 0.375·0.625·0.25 |
| `test_ball.mp4`, 256 단위 (11-09) | 흰색 배경, 빨간 원이 왼쪽→오른쪽, 파란 사각형이 위→아래 | 65/66 | 1곳, CPU 2위, 로짓 차 0.125 |
| `KakaoTalk_..._first10s.mp4` (1920×1080, 10초), 128 단위 | 소녀가 알람을 끄고, 신발을 신고, 가방을 메고 집을 나섬 | 40/41 | 1곳, CPU 2위, 로짓 차 0.125 |
| `KakaoTalk_..._first10s.mp4`, 256 단위 (11-09) | 같은 내용 ("아침 루틴" 대신 "아침 일상") | 40/40 | 없음 |

모든 답이 실제 프레임과 맞습니다. NPU가 고른 토큰은 모두 CPU의 1위나 2위였습니다. 128 단위에서는 차이가 최대 0.625였지만, 256 단위에서는 0.125로 줄었습니다. 입력(프롬프트 ID와 픽셀값)은 두 날짜가 바이트까지 같으므로 차이는 프리필 커널에서 옵니다. 256 단위가 왜 더 가까운지는 확인하지 않았습니다. 로그는 `logs/video_*_2026-11-08.*`, `logs/video_*_2026-11-09.*`에 있습니다.

### 제한 사항

| 항목 | 내용 |
|---|---|
| 입력 | 텍스트와 동영상. 비전 커널은 256패치만 진짜라서 프레임당 256패치 이하만 됩니다. 정지 이미지는 시험하지 않았습니다. 반드시 `serve_vision.py`로 띄워야 합니다. |
| 동시 처리 | 배치 1만 시험. 여러 요청을 동시에 처리하는 커널은 대체 상태입니다. |
| 컨텍스트 | 프롬프트와 생성을 합쳐 2048토큰. 프리필은 256토큰 단위로 나뉩니다. 512는 컴파일 메모리가 80 GiB를 넘어 만들지 못했습니다. |
| 샘플링 | 호스트에서 수행. NPU 샘플링은 끕니다. |
| SSM 정밀도 | `g`, `beta`는 프리필 안에서 BF16으로 내려서 계산합니다. |
| 지원 여부 | Furiosa 공식 지원 구성이 아닙니다. 베타 SDK와 2026.4 커널 패키지를 맞춰 쓴 조합입니다. |

### 어떻게 동작하게 만들었나

`npu/fxb_single_card.py`는 SDK 파일을 바꾸지 않고 빌드 프로세스 안에서만 커널을 고칩니다.

| 문제 | 해결 |
|---|---|
| 커널이 4칩 전제 | `num_chip=1`, Chip 힌트 제거, `all_reduce`를 `reshape`로 교체 |
| `full_in` T=1 컴파일 실패 | 회전 구간을 T=2로 복제해 계산 후 첫 값만 사용 |
| SDK 2026.4 런타임에 SSM 캐시가 없음 | 2026.5.0b3 런타임과 컴파일러 사용 |
| `furiosa.megakernels` 미공개 | 프로세스 안에서 `furiosa.kernels.qwen3_5`로 연결 |
| 컴파일러 설정 변경 | 없어진 옵션 4개 제거, `partial_sum_policy` 형식 변경 |
| 2026.5 런타임은 `g`, `beta`, SSM이 F32 | 공식 EDF의 입력과 비교해 커널 경계의 타입 수정 |
| `gdn_prefill` s256 이상 컴파일 실패 (2026.4) | 2026.5 컴파일러에서 s256은 통과. s512는 메모리 부족이라 프리필을 256토큰 단위로 제한 |

`-O O0`부터 `O2`까지는 `txt_gdn_*` 커널이 빠지므로 `-O O3`에 버킷을 직접 지정합니다.

`experimental/`은 시행착오 기록이고, 재현에는 `npu/`만 필요합니다.

## 다음 시도 후보

- 정식 지원 경로: `furiosa.megakernels`가 포함된 SDK가 나오면 패치 없이 빌드할 수 있는지 시험합니다.
- 프리필 s512와 배치 2 이상을 실제 커널로 컴파일합니다.
- 긴 프롬프트에서 CPU와의 로짓 차가 커지는 원인을 찾습니다.
- 더 많은 프롬프트로 CPU와 비교합니다.

## 포팅 주안점 한눈에 보기 (코드 변경 포인트)

이 모델을 RNGD 1장(8 PE)에서 돌리기 위해 코드상에서 바뀐 핵심 지점은 크게 세 영역입니다.

1. **BF16 → 공식 호환 Block FP8 변환 ([quant_fp8.py](quant_fp8.py), [make_config.py](make_config.py))**
   - 공식 `Qwen/Qwen3.6-27B-FP8` 인덱스를 참조해 400개 Linear 가중치만 `128×128` 블록 단위 `float8_e4m3fn` + `BF16 weight_scale_inv`(`max_abs / 448`)로 변환하고, 비전 인코더·LM head·norm 등은 BF16을 유지합니다.
   - `config.json`에 `quantization_config`(block `[128, 128]`, `quant_method: fp8`)와 `ignored_layers`를 공식 스키마에 맞게 주입합니다.
2. **SDK 파일 수정 없이 런타임 내 1칩/SDK 패치 ([npu/fxb_single_card.py](npu/fxb_single_card.py))**
   - **4칩 → 1칩 축소:** 커널 기본 인자 `num_chip=1` 강제, EDSL의 `operator = { Chip: X }` 힌트 제거, `all_reduce(...)` → `reshape(...)` 치환, `token_shard_threshold` 상향으로 토큰 샤딩 비활성화.
   - **EDSL 컴파일 실패 우회:** `full_in_broadcast` `T=1` 구간을 `T=2`로 복제 계산 후 슬라이스(`tile_rotary`), `T=8` RoPE interleave를 `2T`로 패딩 후 슬라이스(`pad_rope`).
   - **SDK `2026.5.0b3` ↔ `2026.4` 커널 패키지 연결:** 미공개 `furiosa.megakernels.qwen3_5`를 `furiosa.kernels.qwen3_5`로 모듈 에일리어싱하고, 커널 팩토리 반환 형식을 `(Kernel, metadata)` 튜플로 래핑하며, 2026.5 컴파일러에서 제거된 `@compiler_config` 옵션 4개와 `partial_sum_policy` 형식을 정리합니다.
   - **텐서 정밀도 정렬:** 2026.5 런타임과 일치하도록 `lin_in` / `gdn_prefill` / `gdn_decode` 경계의 `g`, `beta`, SSM 캐시 선언을 `bf16 → f32`로 변경합니다(`gdn_f32`).
   - **메모리 한도(80 GiB) 대응 버킷 제한:** `gdn_prefill`은 `GDN_MAX_SEQ`(256), 비전 인코더는 `VIS_MAX`(256패치), `gdn_decode`는 `DEC_MAX`(1) 초과 버킷을 해당 커널 재사용으로 우회합니다.
3. **NPU 런타임 및 비전 안전장치 ([npu/npu_eval.py](npu/npu_eval.py), [npu/serve_vision.py](npu/serve_vision.py))**
   - 하이브리드 SSM 모델 로더 패닉과 NPU 샘플링 실패를 막기 위해 `LLM(..., max_io_memory_mb=4096, scheduler_config=SchedulerConfig(profile_guided_decode_batching=False, enable_npu_sampling=False))`를 반드시 지정합니다.
   - `serve_vision.py`는 프로세서가 `256`패치 비전 버킷만 선택하도록 제한해, 큰 프레임이 재사용 커널로 들어가 오답이 나오는 현상을 차단합니다.

## 2026-11-26 재검증 실측 결과

[run_full_qwen36.py](run_full_qwen36.py)로 NVMe 작업 디렉터리(`/root/.cache/qwen36_work`) 및 `SDK 2026.5.0b3` 환경에서 다운로드 → FP8 양자화 → CPU 스모크 → FXB 빌드 → NPU/CPU 비교 → 프리필 벤치마크 전체를 재측정한 결과입니다. 상세 수치는 [RESULTS.md](RESULTS.md)에 기록되어 있습니다.

- **양자화 & CPU 검증:** 1,584개 텐서 일치(`30.4 GB`), `cpu_smoke.py` 기준 BF16 대비 logits 코사인 **0.9819**, Top-1 일치.
- **FXB 빌드 & 스모크:** `build_fxb.sh` 256토큰 프리필 빌드 `53.8분` (`3225.3s`, `107.3 MB`), NPU 로드 `15s`, `npu_smoke.py` `MATCHES_CPU: True`. (참고: 패치 없는 순정 `fxb build --dry-run`은 예상대로 `furiosa.megakernels.qwen3_5` 미존재로 종료됨)
- **NPU vs CPU 토큰 비교 (`npu_eval.py` vs `cpu_ref.py`):**
  - `code`(28→25토큰, `32.5 tok/s`), `korean`(26→11토큰, `32.0 tok/s`), `long_200`(203→5토큰, `24.1 tok/s`), `long_500`(503→30토큰, `30.7 tok/s`) **4개 완전 일치 (`MATCH`)**
  - `reason`(37→64토큰, `34.9 tok/s`): 19번째 토큰까지 일치 후 `### Step 1:` vs `### 1.` 서식 차이로 분기
- **프리필 지연 (`prefill_bench.py` 중앙값):** 119토큰 `62.5 ms` · 249토큰 `81.9 ms` · 499토큰 `159.7 ms` · 999토큰 `314.9 ms` · 1,499토큰 `470.5 ms` · 1,899토큰 `606.4 ms`.

## 주의

- 이 서버에서 Hugging Face Xet 전송이 마지막 파일에서 멈춘 적이 있어 `HF_HUB_DISABLE_XET=1`을 씁니다.
- `~/.cache`가 있는 NVMe에 공간이 부족해 작업 폴더를 다른 곳에 두는 편이 안전합니다.
