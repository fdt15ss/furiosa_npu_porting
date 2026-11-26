"""Wait for optional PID, then run full Qwen3.6-27B FP8 + NPU pipeline and write RESULTS.md.

Uses /root/.cache/qwen36_work (NVMe) for large checkpoints/FXB and
/root/.cache/qwen36_npu_venv (SDK 2026.5.0b3) for NPU build & inference.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
NPU_DIR = ROOT / "npu"
LOGS = ROOT / "bench_logs"
WORK = pathlib.Path("/root/.cache/qwen36_work")
RAW = WORK / "raw"
FP8 = WORK / "fp8"
FXB = WORK / "logics-swe-1card-sdk5-p256.fxb"
CACHE = WORK / "compile-cache-sdk5"

CPU_PY = "/root/miniconda3/envs/furiosa02/bin/python"
NPU_ENV = "/root/.cache/qwen36_npu_venv"
NPU_PY = f"{NPU_ENV}/bin/python"
FXB_BIN = f"{NPU_ENV}/bin/fxb"
REV = "42c206e3eee5f771e1ad53ca201957c30f41bf4e"


def log_master(msg: str) -> None:
    line = f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with (LOGS / "master.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def wait_for_pid(pid: int) -> None:
    log_master(f"WAIT start: waiting for PID {pid} (yolo) to finish before starting qwen36_fp8")
    while pathlib.Path(f"/proc/{pid}").exists():
        time.sleep(30)
    log_master(f"WAIT done: PID {pid} finished")


def run_step(name: str, cmd: list[str], cwd: pathlib.Path, extra_env: dict[str, str], timeout_s: int) -> dict:
    log_path = LOGS / f"{name}.log"
    log_master(f"START {name}: {' '.join(cmd)}")
    env = os.environ.copy()
    env.update(extra_env)
    start = time.time()
    rc = -1
    timed_out = False
    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            if time.time() - start > timeout_s:
                timed_out = True
                proc.kill()
                proc.wait()
                break
            time.sleep(5)
        if not timed_out:
            rc = proc.returncode
    elapsed = time.time() - start
    status = "TIMEOUT" if timed_out else f"EXIT={rc}"
    log_master(f"DONE  {name}: {status} elapsed={elapsed:.1f}s")
    return {
        "name": name,
        "cmd": cmd,
        "rc": rc,
        "timed_out": timed_out,
        "elapsed_s": round(elapsed, 2),
        "log": str(log_path.relative_to(ROOT)),
    }


def read_log(name: str) -> str:
    p = LOGS / f"{name}.log"
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def render_results(summary: list[dict]) -> None:
    by_name = {s["name"]: s for s in summary}
    today = dt.datetime.now().strftime("%Y-%m-%d")

    def fmt_time(sec: float | None) -> str:
        if sec is None:
            return "실패/미측정"
        return f"{sec / 60:.1f}분 ({sec:.1f}s)" if sec >= 60 else f"{sec:.1f}초"

    def step_time(name: str) -> float | None:
        s = by_name.get(name)
        return s["elapsed_s"] if s and s["rc"] == 0 else None

    chk_txt = read_log("05_check_fp8")
    smoke_txt = read_log("06_cpu_smoke")
    npu_smk_txt = read_log("09_npu_smoke")
    cmp_txt = read_log("12_compare_eval")

    cos_m = re.search(r"cosine\s+([\d.]+)\s+rmse\s+([\d.]+)\s+top1 match\s+(True|False)", smoke_txt)
    fp8_size_m = re.search(r"size GB\s+([\d.]+)", chk_txt)
    tensors_m = re.search(r"tensors\s+(\d+)\s+official\(no mtp\)\s+(\d+)\s+missing\s+(\d+)\s+extra\s+(\d+)", chk_txt)
    npu_load_m = re.search(r"loaded in\s+([\d.]+)s", npu_smk_txt)
    npu_match_m = re.search(r"MATCHES_CPU:\s+(True|False)", npu_smk_txt)

    eval_npu_path = LOGS / "eval_npu.json"
    eval_cpu_path = LOGS / "eval_cpu.json"
    prefill_path = LOGS / "prefill_bench.json"

    eval_npu = json.loads(eval_npu_path.read_text()) if eval_npu_path.exists() else {}
    eval_cpu = json.loads(eval_cpu_path.read_text()) if eval_cpu_path.exists() else {}
    prefill = json.loads(prefill_path.read_text()) if prefill_path.exists() else {}

    fxb_size_mb = f"{FXB.stat().st_size / (1024**2):.1f} MB" if FXB.exists() else "미생성"

    lines = [
        "# Qwen3.6-27B FP8 양자화 및 Furiosa RNGD NPU 실측 결과",
        "",
        f"측정일: {today}. 모델: `Logics-MLLM/Logics-SWE-Qwen3.6-27B` (`{REV}`).",
        f"환경: CPU 검증 `furiosa02`, NPU 빌드·실행 `qwen36_npu_venv` (`furiosa-llm`/`furiosa-tcc 2026.5.0b3`, `furiosa-kernels 2026.4.0`), 작업 디렉터리 `{WORK}`.",
        "",
        "## 단계별 실행 요약",
        "",
        "| 단계 | 상태 | 소요 시간 |",
        "|---|---|---|",
    ]
    for s in summary:
        st = "성공" if s["rc"] == 0 else ("시간 초과" if s["timed_out"] else f"실패 (exit={s['rc']})")
        lines.append(f"| `{s['name']}` | {st} | {fmt_time(s['elapsed_s'])} |")

    lines.extend([
        "",
        "## 양자화 및 CPU 검증",
        "",
        f"- 텐서 구성 (`check_fp8.py`): {tensors_m.group(0) if tensors_m else '실패/미측정'}",
        f"- FP8 출력 크기: {fp8_size_m.group(1) + ' GB' if fp8_size_m else '미측정'}",
        f"- BF16 대비 FP8 logits (`cpu_smoke.py`): {cos_m.group(0) if cos_m else '실패/미측정'}",
        "",
        "## NPU 빌드 및 스모크 테스트",
        "",
        f"- FXB 빌드 (`build_fxb.sh`, 256토큰 프리필): {fmt_time(step_time('08_build_fxb'))}, 크기 {fxb_size_mb}",
        f"- NPU 로드 시간 (`npu_smoke.py`): {npu_load_m.group(1) + 's' if npu_load_m else '미측정'}",
        f"- CPU 스모크 문자열 일치 (`MATCHES_CPU`): {npu_match_m.group(1) if npu_match_m else '미측정'}",
        "",
        "## NPU vs CPU 토큰 비교 (`npu_eval.py` vs `cpu_ref.py`)",
        "",
    ])

    if cmp_txt.strip():
        lines.append("```text")
        lines.extend(cmp_txt.strip().splitlines())
        lines.append("```")
        lines.append("")

    if eval_npu and eval_cpu:
        lines.extend([
            "| 프롬프트 | 입력 토큰 | NPU 생성 토큰 (시간, tok/s) | CPU 생성 토큰 (시간) |",
            "|---|---|---|---|",
        ])
        for k, cpu_v in eval_cpu.items():
            npu_v = eval_npu.get(k, {})
            ntok = len(npu_v.get("tokens", []))
            nsec = npu_v.get("seconds")
            tps = f"{ntok / nsec:.1f}" if nsec else "-"
            ctok = len(cpu_v.get("tokens", []))
            csec = cpu_v.get("seconds")
            lines.append(
                f"| `{k}` | {cpu_v.get('prompt_tokens', '-')} | "
                f"{ntok} ({nsec:.2f}s, {tps} tok/s) | {ctok} ({csec:.1f}s) |"
                if nsec and csec else f"| `{k}` | {cpu_v.get('prompt_tokens', '-')} | {ntok} | {ctok} |"
            )
        lines.append("")

    if prefill:
        lines.extend([
            "## 프리필 지연 측정 (`prefill_bench.py`, 프리픽스 캐시 미사용)",
            "",
            "| 목표 길이 | 실제 토큰 | 중앙값 지연 (ms) | 최소 지연 (ms) |",
            "|---|---|---|---|",
        ])
        for k, v in prefill.items():
            lines.append(f"| {k} | {v.get('prompt_tokens', '-')} | {v.get('median_ms', 0):.1f} | {v.get('min_ms', 0):.1f} |")
        lines.append("")

    lines.extend([
        "## 로그 경로",
        "",
        "- 전체 로그: `qwen36_fp8/bench_logs/` (`master.log`, `summary.json`, `eval_npu.json`, `eval_cpu.json`, `prefill_bench.json`)",
        f"- 모델/FXB 작업 폴더: `{WORK}`",
        "",
    ])

    (ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    LOGS.mkdir(exist_ok=True)
    if "--render-only" in sys.argv:
        summary = json.loads((LOGS / "summary.json").read_text(encoding="utf-8"))
        render_results(summary)
        return

    (LOGS / "master.log").write_text("", encoding="utf-8")
    if "--wait-pid" in sys.argv:
        idx = sys.argv.index("--wait-pid")
        wait_for_pid(int(sys.argv[idx + 1]))

    RAW.mkdir(parents=True, exist_ok=True)
    FP8.mkdir(parents=True, exist_ok=True)

    cpu_env = {
        "HF_HOME": str(WORK / "hfhome"),
        "HF_HUB_DISABLE_XET": "1",
        "OMP_NUM_THREADS": "14",
    }
    npu_env = {
        "PATH": f"{NPU_ENV}/bin:{os.environ.get('PATH', '')}",
        "FURIOSA_ENV": NPU_ENV,
        "HF_HOME": str(WORK / "hfhome"),
    }

    steps: list[tuple[str, list[str], pathlib.Path, dict[str, str], int]] = [
        ("01_download", [CPU_PY, "-I", str(ROOT / "download.py"), REV, str(RAW)], WORK, cpu_env, 7200),
        ("02_verify_shards", [CPU_PY, "-I", str(ROOT / "verify_shards.py"), REV, str(RAW), str(WORK / "sha256.json")], WORK, cpu_env, 1800),
        ("03_quant_fp8", [CPU_PY, "-I", str(ROOT / "quant_fp8.py"), str(RAW), str(FP8)], WORK, cpu_env, 3600),
        ("04_make_config", [CPU_PY, "-I", str(ROOT / "make_config.py"), str(RAW / "config.json"), str(FP8 / "config.json")], WORK, cpu_env, 300),
        ("05_check_fp8", [CPU_PY, "-I", str(ROOT / "check_fp8.py"), str(FP8)], WORK, cpu_env, 300),
        ("06_cpu_smoke", [CPU_PY, "-I", str(ROOT / "cpu_smoke.py"), str(RAW), str(FP8), "48"], WORK, cpu_env, 1800),
        ("07_fxb_dry_run", [FXB_BIN, "build", "--dry-run", str(FP8), str(WORK / "check.fxb")], WORK, npu_env, 600),
        ("08_build_fxb", [str(NPU_DIR / "build_fxb.sh"), str(FXB), str(CACHE)], NPU_DIR, npu_env, 7200),
        ("09_npu_smoke", [NPU_PY, "-u", str(NPU_DIR / "npu_smoke.py"), str(FP8), str(FXB)], NPU_DIR, npu_env, 900),
        ("10_npu_eval", [NPU_PY, "-u", str(NPU_DIR / "npu_eval.py"), str(FP8), str(FXB), str(LOGS / "eval_npu.json")], NPU_DIR, npu_env, 1800),
        ("11_cpu_ref", [CPU_PY, "-u", str(NPU_DIR / "cpu_ref.py"), str(FP8), str(LOGS / "eval_cpu.json")], NPU_DIR, cpu_env, 3600),
        ("12_compare_eval", [sys.executable, "-u", str(NPU_DIR / "compare_eval.py"), str(LOGS / "eval_npu.json"), str(LOGS / "eval_cpu.json")], NPU_DIR, {}, 300),
        ("13_prefill_bench", [NPU_PY, "-u", str(NPU_DIR / "prefill_bench.py"), str(FP8), str(FXB), str(LOGS / "prefill_bench.json")], NPU_DIR, npu_env, 1800),
    ]

    summary: list[dict] = []
    for name, cmd, cwd, env, timeout_s in steps:
        rec = run_step(name, cmd, cwd, env, timeout_s)
        summary.append(rec)
        (LOGS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    render_results(summary)
    log_master("ALL DONE -> qwen36_fp8/RESULTS.md written")


if __name__ == "__main__":
    main()
