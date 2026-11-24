# Qwen3.6-27B NPU 빌드 작업 기록

RNGD 카드 1장(8 PE)에서 Qwen3.6-27B 파인튜닝 모델을 돌리기 위한 작업 기록입니다. 시도한 방법과 실패 원인, 남은 작업을 정리했습니다. 재현 절차만 필요하면 [README.md](README.md)의 "NPU 실행"을 보세요.

기준 시각: 2026-11-09 09:00 UTC

## 현재 상태

**RNGD 1장에서 텍스트 추론과 저해상도 동영상 입력이 동작합니다.** 2026-11-07 00:18 UTC에 NPU 출력이 처음으로 CPU 결과와 일치했고, 2026-11-08에 FXB를 다시 빌드하지 않고 동영상 입력을 확인했습니다. 2026-11-09부터 기본 FXB는 프리필을 256토큰 단위로 처리합니다(128 단위보다 약 33% 빠름).

| 단계 | 상태 |
|---|---|
| FP8 양자화, CPU 검증 | 완료 |
| 텍스트 커널 1칩 컴파일 | 완료 (SDK 2026.5.0b3 컴파일러, 100개) |
| NPU 로드 | 10~19초 |
| CPU 값 비교 | 프롬프트 5개 중 4개 토큰까지 일치, 1개는 19번째 토큰부터 다름 |
| 생성 속도 | 약 35 토큰/초 (배치 1) |
| 동영상 입력 | 프레임당 256패치 이하만 동작 (16:9는 320×192). 영상 2개, 8프레임으로 CPU와 비교 |
| 정지 이미지 입력 | 시험 안 함. 같은 256패치 커널을 쓰므로 같은 크기 제한이면 동작할 것으로 예상 |
| 256패치 초과 프레임 | 미지원 (큰 비전 버킷은 256패치 커널을 재사용). `serve_vision.py`가 오류로 거부 |
| 배치 2 이상 | 미지원 (`gdn_decode`를 배치 1로 대체) |
| 프리필 | 256토큰 단위 (`logics-swe-1card-sdk5-p256.fxb`). 128 단위보다 긴 프롬프트에서 약 33% 빠름. s512는 컴파일 메모리 부족 |
| 긴 프롬프트 CPU 비교 | 1,000~1,850토큰 3개에서 1~20번째 토큰부터 다름. NPU 토큰은 모두 CPU 1·2위, 로짓 차 최대 1.875. 128 단위 FXB도 같은 토큰 |

다른 프롬프트(속력 계산)는 NPU가 `### Step 1:`, CPU가 `### 1.`로 시작했습니다. 갈라진 위치에서 CPU의 1, 2위 로짓은 24.125와 24.0으로 BF16 한 칸 차이(확률 35%, 31%)라서 반올림 차이로 보입니다. NPU 로짓은 직접 확인하지 않았습니다.

비교 결과는 [logs/eval_npu_2026-11-07.json](logs/eval_npu_2026-11-07.json)과 [logs/eval_cpu_2026-11-07.json](logs/eval_cpu_2026-11-07.json)에 있습니다.

## 목표와 성공 기준

- **모델:** `Logics-MLLM/Logics-SWE-Qwen3.6-27B`, 커밋 `42c206e3eee5f771e1ad53ca201957c30f41bf4e`
- **성공 기준:** 핵심 커널이 컴파일되고, NPU 출력이 CPU 결과와 일치할 것
- **검증 범위:** 프롬프트 5개, 최대 64토큰입니다. 모델 품질이나 장시간 안정성은 평가하지 않았습니다.

## 환경

| 항목 | 값 |
|---|---|
| 하드웨어 | RNGD 1장, 8 PE, 메모리 47.5 GiB (`npu6`), 펌웨어 2026.4.0 |
| 빌드·실행 환경 | conda `furiosa`: furiosa-llm·tcc 2026.5.0b3, furiosa-kernels 2026.4.0 |
| 양자화·CPU 비교 환경 | conda `furiosa02`: SDK 2026.4.0 |
| 컴파일 메모리 | 컨테이너 한도 80 GiB (`free`에 보이는 1 TB가 아님) |
| FP8 모델 | `/root/models/Logics-SWE-Qwen3.6-27B-FP8` (텍스트 가중치 27.4 GiB) |
| FXB | `/root/models/fxb/logics-swe-1card-sdk5-p256.fxb` (112.5 MB, 프리필 256 단위, 기본)<br>`/root/models/fxb/logics-swe-1card-sdk5.fxb` (86 MB, 프리필 128 단위, 11-07) |

**SDK 2026.4.0으로는 실행할 수 없습니다.** 2026.4.0 런타임에는 Linear 레이어의 상태(SSM) 캐시가 없습니다. 런타임 라이브러리 문자열을 비교해 보면 `ssm_num_blocks` 같은 SSM 관련 코드는 2026.5.0b3에만 있습니다. 공식 FXB도 2026.5.0-beta.4로 빌드돼 있습니다.

공식 구성은 쓸 수 없습니다. 공식 FXB는 카드 2장(16 PE)이 필요하고, 공식 커널은 4칩 전제입니다.

## 빌드 방법

SDK 파일은 고치지 않습니다. [npu/fxb_single_card.py](npu/fxb_single_card.py)가 빌드 프로세스 안에서만 다음을 바꿉니다. 실행은 [npu/build_fxb.sh](npu/build_fxb.sh)로 합니다.

**1칩 구성**

1. 모든 `num_chip` 기본값을 1로 바꿉니다. `qwen3_6_27b.py`의 `NUM_CHIP`만 4로 둬서 4칩 샘플링 테이블을 건너뜁니다.
2. 데코레이터가 기본값을 가리므로 공개 진입점에서도 `num_chip=1`을 강제합니다.
3. DSL에서 `@ context(operator = { Chip: X })`를 지우고, `all_reduce(x, AXIS)`를 `reshape(x)`로 바꿉니다.
4. `full_in`이 T=1이면 rotary 구간을 T=2로 늘려 계산한 뒤 잘라 냅니다.
5. `token_shard_threshold`를 올려 토큰 샤딩을 끕니다.

**SDK 2026.5 연결**

6. 공개되지 않은 `furiosa.megakernels.qwen3_5`를 `furiosa.kernels.qwen3_5`로 연결하고 `COMPONENT_MODULES`를 지정합니다.
7. 커널 함수가 `(Kernel, metadata)`를 돌려주도록 감쌉니다. `Kernel.source.to_json()`은 `{"statement_provenance": []}`를 돌려줍니다.
8. 2026.5 컴파일러가 받지 않는 옵션 4개(`allow_hp_lowering`, `lowering_mode`, `apply_adaptive_einsum_by_pattern`, `enable_vrf_half_mode`)를 지우고, `partial_sum_policy`를 값 하나짜리 형식으로 바꿉니다.
9. `lin_in`의 출력 `g`, `beta`와 GDN 커널의 입력 `g`, `beta`, SSM 캐시를 BF16에서 F32로 바꿉니다. 공식 EDF의 입력 형식에 맞춘 것입니다.

**텍스트 전용 대체**

10. 비전 버킷은 모두 256패치 커널을 씁니다.
11. `gdn_decode`는 모두 배치 1 커널을 씁니다.
12. `gdn_prefill`이 `GDN_MAX_SEQ`(기본 256)를 넘는 버킷은 `GDN_MAX_SEQ` 커널을 씁니다. attention 버킷을 같은 크기까지만 두므로 엔진이 이 버킷을 고르지 않습니다. `GDN_MAX_SEQ=128`이면 11-07 FXB를 만듭니다.

빌드 옵션은 `-O O3`가 필수입니다.

```bash
cd npu && ./build_fxb.sh /root/models/fxb/logics-swe-1card-sdk5-p256.fxb
# 내부적으로 실행하는 명령 (GDN_MAX_SEQ=256)
python fxb_single_card.py build -tp 8 -O O3 --max-model-len 2048 \
  --tokenwise-buckets 1,128,256 --attention-buckets '(1,128,128),(1,256,256),(1,2048,1),(1,2048,128),(1,2048,256)' \
  --concurrency 3 --no-fast-fail --compiler-cache-dir <캐시> --build-report fxb_config <출력>.fxb
```

실행할 때는 아래 설정이 필요합니다.

```python
LLM(model_dir, fxb=fxb, max_model_len=2048, max_io_memory_mb=4096,
    scheduler_config=SchedulerConfig(profile_guided_decode_batching=False, enable_npu_sampling=False))
```

## 시도와 실패 기록

시각은 UTC입니다. 11-06 21:24 이전 로그는 [logs/](logs/)에 있고, 그 이후 로그는 스크래치 폴더에만 있습니다.

| 시각 | 시도 | 결과 | 원인과 조치 | 로그 |
|---|---|---|---|---|
| 11-06 08:12 | `num_chip`만 1로 강제한 빌드 | 45분간 14/89 진행, 10개 실패 | 4칩 구조가 남음. 오류 1 | `01_census_1chip.log` |
| 09:47 | 칩 힌트 제거 + `all_reduce`→`reshape`로 커널별 컴파일 | 대부분 통과, `full_in` T=1만 실패 | 오류 2 | `kernel_batches.log` |
| 10:50 | `gated_delta_chunk` 컴파일 | 40분 뒤 rc -9 | OOM. 빌드에 안 쓰이는 커널이라 제외 | 스크래치에만 있음 |
| 10:55 | 스텁 비전으로 전체 빌드 (`VIS_STUB`, `-O O0`) | 89/89 완료, 44.9 MB | 42분 소요 | `02_build_vis_stub.log` |
| 11:38 | 위 FXB를 NPU에 로드 | 패닉 | 오류 3 | `03_npu_load_vis_stub.log` |
| 11:40 | 입출력 구조만 같은 빈 비전 커널 | 실패, 포기 | `inputs must be non-empty`, 태그 shape 오류 | 없음 |
| 11:38~20:13 | 진짜 비전 커널로 전체 빌드 | 78개 성공, 11개 실패, 691분 | 오류 4 | `04_build_real_vision.log` |
| 20:14 | `VIS_ALIAS`로 재빌드 | 89/89, 17초(캐시), 496 MB | 실패한 버킷을 작은 커널로 대체 | `05_build_vis_alias.log` |
| 20:15 | 위 FXB를 NPU에 로드 | 패닉 | 오류 5. 비전 입출력 검사는 통과 | `06_npu_load_no_gdn.log` |
| 20:18~23:01 | `-O O3`, 버킷 지정으로 재빌드 | 100개 성공, `gdn_prefill` s256·s512·s1024 실패 | 오류 6 | `07_build_gdn_O3_running.log` (21:24까지) |
| 23:02~23:14 | `gdn_prefill` s256 이상을 s128로 대체, 프리필 128 | 103/103 | 대체된 TC가 s128과 같은지 해시로 확인 | 스크래치 `build6.log` |
| 23:15 | 위 FXB를 2026.4 런타임에 로드 | 패닉 | 오류 7 | 스크래치 `npu3.log` |
| 23:20~23:26 | `max_io_memory_mb` 지정 | 로드 성공(26초), 첫 요청에서 패닉 | 오류 7, 8 | 스크래치 `npu9.log` |
| 23:27 | 2026.5 런타임에 2026.4 EDF 로드 | 거부 | `loading EDF edfs/txt_full_attention_b1_a128_kv0.edf` | 스크래치 `npu11.log` |
| 23:29~23:36 | 2026.5 빌드 파이프라인 연결 | 드라이런 통과 | 오류 9 | 스크래치 `dry5.log` |
| 23:36~00:02 | 2026.5 컴파일러로 빌드 | 125/125 (의존 커널 25개 포함) | | 스크래치 `build7.log` |
| 11-07 00:03 | 2026.5 런타임에 로드 | 로드 10초, NPU 샘플러 패닉 | `NPU sampler drew Some(248320)`. 호스트 샘플링으로 전환 | 스크래치 `npu12.log` |
| 00:03 | 호스트 샘플링 | 48토큰 생성, 전부 토큰 0 | 오류 10 | 스크래치 `npu13.log` |
| 00:07~00:18 | GDN F32 수정 후 재빌드 | **CPU와 일치** | 처음 성공 | 스크래치 `npu14.log` |
| 00:19~00:30 | 프롬프트 5개로 CPU 비교 | 4개 일치, 1개 19번째 토큰부터 다름 | 현재 상태 참고 | `logs/eval_*_2026-11-07.json` |
| 00:20~00:28 | 정리한 `npu/` 스크립트로 재빌드 | EDF 100개가 성공 빌드와 바이트까지 동일 | 재현 확인 | 스크래치 `build9.log` |

### 오류 상세

**오류 1. 4칩 구조가 남은 커널**

```text
inter_chip_axes.volume() (4) != global_config().num_chips() (1)
called `Option::unwrap()` on a `None` value (tactic-populator/.../generic.rs:1924)
```

`lin_out`, `full_in`, `full_out`, `mlp`, `full_attention`에서 발생했습니다. `Chip` 힌트 제거와 `all_reduce`→`reshape`로 해결했습니다. 이 변경이 값을 바꾸지 않는다는 점은 최종 CPU 비교로 확인했습니다.

**오류 2. `full_in` T=1**

```text
Compilation failed: while lowering full_in_broadcast_xr_rot_1
```

크기 문제라는 가설은 틀렸습니다. 헤드 24개는 통과했고, T=2와 T=4도 통과했습니다. rotary 구간을 T=2로 계산하고 잘라 내는 방식으로 해결했습니다. 디코드는 T=1을 쓰므로 이 우회도 CPU 비교에 포함됐습니다.

**오류 3. 스텁 비전 커널 거부**

```text
operand layout length (24) != EDF I/O count (3) for kernel 'vis_first_block',
bucket VisionEncoder { batch_size: 1, num_patches: 256 }
(npu_module.rs:88)
```

텍스트만 쓰더라도 로더가 비전 커널의 입출력을 검사합니다. 진짜 비전 커널이 필요합니다.

**오류 4. 큰 비전 버킷 11개**

| 버킷 | 종료 | 원인 |
|---|---|---|
| b4 6400, b4 9216 (6개) | exit 1 | **컴파일 오류.** `dram_sizes_in_bytes <= dma_bridge_threshold_in_bytes` 조건 실패로 `BridgeTacticFailure` |
| b4 16384, b1 65536 (5개) | 시그널 종료 | OOM 추정. 같은 기간 `oom_kill`이 8회 기록됐으나 커널별 대응은 확인하지 못함 |

원문은 `logs/tcc_errors.txt`에 있습니다.

**오류 5. GDN 커널 누락**

```text
kernel 'txt_gdn_prefill' not found in EdfStore
```

`-O O0`, `O1`, `O2`는 `txt_gdn_*` 커널을 **오류 없이 빠뜨립니다.** `-O O3`를 써야 합니다.

**오류 6. gdn_prefill 시퀀서 한도 초과**

```text
[7/13] Compiling from precommandgen to commandgraph
insufficient sequencer entries: required 9
verification of operator failed: DmaCommand ... dram -> sram
```

- **통과와 실패의 경계:** s128(Nc=4)은 통과하고, s256(Nc=8)과 s512(Nc=16)는 실패합니다. s1024(Nc=32)는 약 2시간 뒤 시그널로 종료됐고, 그 사이 `oom_kill`이 8에서 9로 늘었습니다. `Nc`는 `seq_len / 32`입니다.
- **추정 원인(미검증):** DMA 하나의 stride를 풀어 보면 `[Nc, Ho, Hi, Ci, D]` 텐서로 읽힙니다. Nc=8에서 Nc 축이 `2 × 4` 두 루프로 쪼개져 9단이 되고, 명령 하나의 한도를 넘는 것으로 보입니다.
- **조치:** 프리필을 128토큰 단위로 제한했습니다.
- **2026-11-09:** 2026.5 컴파일러에서는 s256이 이 오류 없이 통과합니다. 원인 추정과 달리 청크 32(Nc=8)에서도 통과했습니다. 자세한 내용은 "남은 작업 > 프리필 s256 이상"에 있습니다.

**오류 7. 2026.4 런타임의 Linear 버킷 패닉**

```text
linear attention stage requires a linear bucket (executor.rs:700)
```

백트레이스 주소와 디스어셈블로 추적한 결과, 로드 단계에서는 I/O 메모리 추정 함수가 Linear 버킷 목록을 빈 채로 넘깁니다. `max_io_memory_mb`를 지정하면 이 함수를 건너뜁니다. 하지만 첫 요청에서 스케줄러가 같은 패닉을 냅니다(오류 8).

**오류 8. 2026.4 런타임에 SSM 캐시 없음**

로그의 `KVCachePlan`에 Linear 상태 항목이 없고, 요청마다 Linear 버킷 목록이 비어 있습니다. 2026.4.0 런타임은 이 모델의 하이브리드 구조를 실행할 수 없다고 판단했습니다. 2026.5.0b3에서는 `ssm_num_blocks: Some(60)`으로 SSM 캐시가 잡힙니다.

**오류 9. 2026.5 빌드 파이프라인 호환성**

| 오류 | 조치 |
|---|---|
| `megakernel module 'furiosa.megakernels.qwen3_5' ... is not importable` | 모듈 연결 |
| `must declare ... COMPONENT_MODULES` | `('furiosa.kernels.qwen3_5',)` 지정 |
| `must return a (Kernel, MegakernelMetadata) tuple` | 결과를 튜플로 감쌈 |
| `'Kernel' object has no attribute 'source'` | `to_json()`이 있는 객체 추가 |
| `missing field statement_provenance` | 빈 목록으로 채움 |
| `unknown field apply_adaptive_einsum_by_pattern` 등 4개 | 해당 옵션 제거 |
| `invalid type: map, expected unit` | `partial_sum_policy`를 이름만 남김 |

**오류 10. 출력이 전부 토큰 0**

실행은 되지만 값이 틀렸습니다. 공식 FXB를 HTTP 범위 요청으로 일부만 읽어 EDF 입출력을 비교했더니, 토큰 단위 커널과 attention은 순서·형식·크기가 모두 같았습니다. 다른 것은 세 곳뿐이었습니다.

| 커널 | 위치 | 공식 | 2026.4 커널 |
|---|---|---|---|
| `lin_in` | 출력 `g`, `beta` | F32 | BF16 |
| `gdn_decode` | 입력 `g`, `beta`, SSM 캐시 | F32 | BF16 |
| `gdn_prefill` | 입력 `g`, `beta`, SSM 캐시 | F32 | BF16 |

런타임이 F32로 넘긴 값을 커널이 BF16으로 읽어 값이 깨졌습니다. 경계의 타입만 F32로 바꾸고 내부 계산은 그대로 둬서 해결했습니다. 비교 도구는 [experimental/edfcmp.py](experimental/edfcmp.py)입니다.

## 남은 작업

### 배치 2 이상

**2026-11-12: 배치 2/4/8을 Qwen3.8 FXB로 빌드하고 시험했습니다(`DEC_BATCHES`). 결과는 `../qwen38_fp8/README.md` "동시 처리".** T=8 `full_in`은 rope einsum 우회(`pad_rope`)가 필요했습니다.

필요한 커널은 `gdn_decode` bN, 토큰 단위 커널 twN, `full_attention` 디코드 bN입니다.

- `gdn_decode` b2~b512는 2026.4 컴파일러로 모두 통과했지만, F32 수정 후에는 b1만 컴파일했습니다.
- 배치 attention은 1칩에서 한 번도 컴파일하지 않았습니다. 가장 불확실한 부분입니다.
- 제안: 배치 1, 2, 4, 8만 빌드하고, 동시 요청 결과를 단독 실행 결과와 비교합니다.

### 프리필 s256 이상

**2026-11-09: s256은 2026.5 컴파일러로 동작하고 기본값이 됐습니다. s512는 메모리가 부족합니다.**

`gdn_prefill` 하나만 2026.5 `furiosa-tcc`로 컴파일했습니다. 스크립트는 `experimental/p256/gdn_one.py`이고, s128 소스는 기존 빌드 캐시의 소스와 같습니다(끝의 빈 줄만 다름). 메모리 감시는 `experimental/p256/memguard.py`입니다. 컨테이너 cgroup이 읽기 전용이라 프로세스별 한도를 걸 수 없어서, 사용량을 2초마다 보고 한도를 넘으면 그 컴파일만 종료합니다.

| 커널 | 결과 | 시간 |
|---|---|---|
| s128, 청크 32 (대조군) | 통과 | 552초 |
| s256, 청크 32 | 통과. 2026.4의 오류 6이 2026.5에서는 나지 않음 | 714초 |
| s256, 청크 64 (Nc=4) | 통과 | 2,328초. 쓰지 않음 |
| s512, 청크 32 | 실패. 단계 5에서 3시간 17분 뒤 메모리 부족으로 강제 종료(`oom_kill` 9→10). 빌드와 동시에 돌림 | |
| s512, 청크 32, `enable_tactic_pruning` | 실패. 단계 5에서 1분에 약 4~5 GiB씩 늘어 10.8분 만에 감시 한도 55 GiB 도달 | |
| s512, 청크 64 (Nc=8) | 실패. 단계 5에서 21.4분 만에 감시 한도(anon 70 GiB) 도달, 최고 66.6 GiB | |

**256토큰 FXB.** `/root/models/fxb/logics-swe-1card-sdk5-p256.fxb`(112.5 MB, 커널 109개). 처음 빌드는 s512 컴파일과 동시에 돌린 탓에 s256 의존 커널이 2시간 53분 걸렸습니다(`logs/08_build_p256_2026-11-09.log`). 이후 `npu/build_fxb.sh` 기본값을 256으로 바꾸고 빈 캐시에서 다시 빌드했더니 41.7분 만에 끝났고(최고 25.2 GiB), FXB 안의 파일 110개 중 109개가 바이트까지 같았습니다. 다른 것은 `manifest.json`의 `created_at`과 `uuid`뿐입니다.

**버킷 선택 확인.** `RUST_LOG=debug`로 보면 203토큰 프롬프트에서 `Tokenwise bucket selected: input_size: 256`이 찍힙니다.

**속도.** 매번 다른 프롬프트로 첫 토큰 시간을 쟀습니다(`npu/prefill_bench.py`). 같은 프롬프트를 반복하면 접두사 캐시가 걸려 프리필을 재지 못합니다. 처음에 반복 측정으로 "256이 더 느리다(63→82 ms)"고 잘못 읽었던 것이 이 때문입니다.

| 프롬프트 길이 | 128 단위 | 256 단위 |
|---|---|---|
| 119 | 62 ms | 63 ms |
| 249 | 122 ms | 82 ms |
| 499 | 241 ms | 159 ms |
| 999 | 480 ms | 314 ms |
| 1,899 | 898 ms | 605 ms |

청크 하나에 128토큰은 약 60 ms, 256토큰은 약 80 ms입니다. 청크마다 드는 고정 비용이 커서, s512가 되면 더 빨라질 여지가 있습니다.

**정확도.**
- 짧은 프롬프트 5개와 긴 프롬프트 3개(`EVAL_SET=long`, 1,000·1,000·1,850토큰)에서 128 단위와 256 단위의 출력 토큰이 완전히 같습니다.
- 긴 프롬프트는 두 FXB 모두 CPU와 갈라집니다(1·19·20번째 토큰). `npu/cpu_text_check.py`로 보면 NPU 토큰은 모두 CPU 1·2위이고, 로짓 차는 0.750, 1.875, 0.625입니다. 짧은 프롬프트(0.125)보다 크며, 원인은 확인하지 않았습니다.
- 동영상은 256 단위가 CPU에 더 가깝습니다. test_ball 47/50(최대 0.625) → 65/66(0.125), KakaoTalk 40/41 → 40/40. 입력 덤프는 11-08과 바이트까지 같습니다.

로그는 `logs/*_2026-11-09.*`에 있습니다.

남은 일:
1. s512를 컴파일할 방법을 찾습니다. 후보는 컴파일 설정 `max_num_partitioning_axes`, `dma_segment_mode`, `enable_auto_loop_plan`, `scheduler_beam_search`입니다.
2. 긴 프롬프트에서 로짓 차가 커지는 원인을 찾습니다. CPU에서 FP8 가중치 선형층 400개의 입력을 NPU처럼 토큰·128 블록 단위로 FP8 양자화해도(`FA8=1 cpu_text_check.py`) 로짓 차는 0.875, 1.500, 0.500으로 거의 그대로였습니다(`logs/eval_long_cpu_check_fa8_2026-11-09.log`). 활성값 FP8만으로는 설명되지 않으므로, 다음은 층별 출력을 CPU와 비교해야 합니다.

### 그 외

- **비전:** 2026-11-08에 동영상 입력을 확인했습니다(프레임당 256패치 이하). 영상 2개에서 NPU가 고른 토큰은 모두 CPU의 1위나 2위였고, 로짓 차는 최대 0.625입니다. 2026-11-09의 256 단위 FXB에서는 최대 0.125입니다. 자세한 결과는 README의 "동영상 입력"에 있습니다. 남은 일은 다음과 같습니다.
  - 256패치를 넘는 비전 버킷을 2026.5 컴파일러로 진짜 커널로 빌드합니다. 2026.4에서는 b1 기준 4096패치까지 컴파일됐습니다.
  - 정지 이미지 입력을 시험합니다.
  - 비전 인코더 출력을 CPU와 직접 비교합니다. 256 단위에서 로짓 차가 줄었으므로, 0.625의 원인은 비전보다 128 단위 프리필 쪽에 있었을 수 있습니다. 미확인입니다.
- **NPU 샘플링:** `random_sample` 커널과 2026.5 런타임이 맞지 않습니다. 호스트 샘플링으로 우회 중입니다.
- **`--enable-mtp`:** 시험하지 않았습니다.
- **정식 경로:** `furiosa.megakernels`가 포함된 SDK가 나오면 패치 없이 빌드되는지 확인합니다.

## 작업 시 주의

- **`--no-fast-fail`이어도 실패 커널이 있으면 FXB는 안 나옵니다.** 빌드가 끝날 때까지 기다리지 말고 로그의 `Kernel compilation failed`를 확인하세요.
- **진행률로 성공을 판단하면 안 됩니다.** 실패한 커널도 진행 수에 포함될 수 있습니다.
- **실행이 된다고 값이 맞는 것은 아닙니다.** 출력이 이상하면 공식 FXB와 EDF 입출력을 먼저 비교하세요.
- **2026.5 빌드는 `PATH` 앞에 `furiosa` 환경의 `bin`이 있어야 합니다.** `/usr/local/bin/furiosa-tcc`는 2026.4입니다.
- **빌드와 CPU 비교를 동시에 돌리면 80 GiB 한도에 가까워집니다.** NPU 실행과 CPU 비교도 함께 돌리지 마세요. CPU 비교는 혼자서 anon 약 50 GiB, 파일 매핑까지 약 74 GiB입니다.
- **큰 커널 컴파일은 하나씩 돌리세요.** s512를 빌드와 함께 돌렸더니 빌드의 s256 의존 커널이 12분에서 2시간 53분으로 늘었습니다. 메모리 한도에 가까운 컴파일은 `experimental/p256/memguard.py`로 감싸세요. 한도를 넘으면 컨테이너 OOM이 다른 프로세스(VS Code, 서버)를 죽일 수 있습니다.
- **`pgrep -f`로 기다리는 루프는 자기 자신에 걸립니다.** 패턴 문자열이 대기 명령줄에도 들어가 영원히 끝나지 않습니다. `tail --pid=<PID> -f /dev/null`로 기다리세요.
- **프리필 속도는 매번 다른 프롬프트로 재세요.** 같은 프롬프트를 반복하면 접두사 캐시가 걸립니다.
- **dry-run은 몇 초면 끝납니다.** 버킷을 바꾸기 전에 먼저 돌리세요. 커널 목록만 알려 줄 뿐 컴파일 가능 여부는 알려 주지 않습니다.
- **`--dump-tc <폴더>`** 로 커널 DSL이 의도대로 생성됐는지 확인할 수 있습니다.
- **긴 컴파일은 `setsid nohup`으로** 실행하세요. VS Code를 닫아도 유지됩니다.
- **컴파일 캐시**는 커널 내용으로 키를 만듭니다. 같은 캐시를 쓰면 통과한 커널은 재컴파일되지 않습니다. 실패 원문은 캐시 폴더의 기록에서 읽을 수 있습니다.

## 파일 위치

| 경로 | 내용 | 비고 |
|---|---|---|
| `npu/` | 재현용 빌드·실행·비교 스크립트 | 현재 기준 |
| `npu/fxb_config/` | 빌드용 설정 | |
| `experimental/fxb_single_card.py` | 실험 중 하네스 | 11-07 버전으로 덮어써져 `GDN_MAX_SEQ`가 없습니다 |
| `experimental/run_build_t128.sh` | 128 토큰 빌드 (2026.4) | 미실행. `GDN_MAX_SEQ`에 의존해 지금 하네스로는 동작하지 않습니다 |
| `experimental/edfcmp.py` | 공식 FXB와 EDF 입출력 비교 | |
| `experimental/compile_one_kernel.py` | 커널 하나 컴파일 (2026.4) | |
| `logs/` | 11-06 21:24까지의 로그, `tcc_errors.txt`, 11-07 비교 결과 | |
| `/root/models/fxb/logics-swe-1card-sdk5-p256.fxb` | 동작하는 FXB, 프리필 256 단위 (기본) | 11-09 |
| `/root/models/fxb/logics-swe-1card-sdk5.fxb` | 동작하는 FXB, 프리필 128 단위 | 11-07 |
| `experimental/p256/` | 단일 커널 컴파일(`gdn_one.py`), 메모리 감시(`memguard.py`), 11-09 실험 하네스 | |
| `npu/prefill_bench.py`, `npu/cpu_text_check.py`, `npu/long_prompts.json` | 프리필 속도 측정, 긴 프롬프트 CPU 비교 | 11-09 |
| `logics-swe-1card-sdk5.fxb` | 위 FXB의 사본 | SHA256 `4f61a3c5…71493748` |
| `/root/models/fxb/compile-cache-sdk5` | 2026.5 컴파일 캐시 | 11-09에 s256 등 10개 항목 추가 |
| `/tmp/claude-0/.../73b6e61b-.../scratchpad/` | 11-06 21:24 이후 로그, 2026.4 캐시 `cc2` | 2026-11-08 삭제됨 |
