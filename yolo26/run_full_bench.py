"""Sequential NPU YOLO/D-FINE compile + benchmark runner and RESULTS.md generator.

Runs all compile, accuracy check, file/video inference, FPS, and PE-fusion tests
one at a time (respecting the 80 GiB container memory limit), logs every step
under bench_logs/, and writes RESULTS.md solely from measured outputs.
"""
from __future__ import annotations

import datetime as dt
import importlib.metadata as meta
import json
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
PYTHON = str(ROOT / ".venv" / "bin" / "python")
LOGS = ROOT / "bench_logs"
MEDIA = "/root/programming/npu/media"

STEPS: list[tuple[str, list[str], int]] = [
    # Group A: YOLO26n Detection + Fusion
    ("01_compile_yolo26n", [PYTHON, "-u", "compile_model.py", "yolo26n"], 14400),
    ("02_run_detect", [PYTHON, "-u", "run_detect.py"], 1800),
    ("03_detect_fps", [PYTHON, "-u", "detect_npu.py"], 1800),
    ("04_compile_yolo26n_1pe_pad4", [PYTHON, "-u", "compile_fusion.py", "1", "pad4"], 14400),
    ("05_compile_yolo26n_2pe", [PYTHON, "-u", "compile_fusion.py", "2"], 14400),
    ("06_compile_yolo26n_2pe_pad4", [PYTHON, "-u", "compile_fusion.py", "2", "pad4"], 14400),
    ("07_bench_1pe_3ch", [PYTHON, "-u", "bench_fusion.py", "1", "yolo26n.edf"], 600),
    ("08_bench_1pe_pad4", [PYTHON, "-u", "bench_fusion.py", "1", "yolo26n-1pe-pad4.edf"], 600),
    ("09_bench_2pe_3ch", [PYTHON, "-u", "bench_fusion.py", "2", "yolo26n-2pe.edf"], 600),
    ("10_bench_2pe_pad4", [PYTHON, "-u", "bench_fusion.py", "2", "yolo26n-2pe-pad4.edf"], 600),
    ("11_bench_1pe_pad4_x2", [PYTHON, "-u", "bench_fusion.py", "1", "yolo26n-1pe-pad4.edf", "2"], 600),
    ("12_bench_1pe_pad4_x8", [PYTHON, "-u", "bench_fusion.py", "1", "yolo26n-1pe-pad4.edf", "8"], 600),
    ("13_bench_2pe_pad4_x2", [PYTHON, "-u", "bench_fusion.py", "2", "yolo26n-2pe-pad4.edf", "2"], 600),
    ("14_bench_2pe_pad4_x4", [PYTHON, "-u", "bench_fusion.py", "2", "yolo26n-2pe-pad4.edf", "4"], 600),
    # Group B: YOLO26n-seg
    ("15_compile_yolo26n_seg", [PYTHON, "-u", "compile_model.py", "yolo26n-seg"], 14400),
    ("16_compile_yolo26n_seg_protohead", [PYTHON, "-u", "compile_proto.py"], 7200),
    ("17_seg_check", [PYTHON, "-u", "seg_npu.py", "check"], 600),
    ("18_seg_files", [PYTHON, "-u", "seg_npu.py", "files"], 1800),
    ("19_seg_fps", [PYTHON, "-u", "seg_npu.py", "fps"], 1800),
    # Group C: D-FINE nano
    ("20_compile_dfine_encoder", [PYTHON, "-u", "compile_dfine.py", "encoder"], 14400),
    ("21_compile_dfine_parte", [PYTHON, "-u", "compile_dfine_parts.py", "e"], 7200),
    ("22_compile_dfine_parta", [PYTHON, "-u", "compile_dfine_parts.py", "a"], 7200),
    ("23_compile_dfine_partb", [PYTHON, "-u", "compile_dfine_parts.py", "b"], 7200),
    ("24_compile_dfine_partc", [PYTHON, "-u", "compile_dfine_parts.py", "c"], 7200),
    ("25_dfine_check", [PYTHON, "-u", "dfine_npu.py", "check"], 600),
    ("26_dfine_files", [PYTHON, "-u", "dfine_npu.py", "files"], 1800),
    ("27_dfine_fps", [PYTHON, "-u", "dfine_npu.py", "fps"], 1800),
    # Group D: Pose (runs last because infer_pose.py writes *_npu.jpg to MEDIA)
    ("28_compile_yolo11n_pose", [PYTHON, "-u", "compile_pose.py", "yolo11n-pose"], 14400),
    ("29_compile_yolo26n_pose", [PYTHON, "-u", "compile_pose.py", "yolo26n-pose"], 14400),
    ("30_infer_yolo11n_pose", [PYTHON, "-u", "infer_pose.py", "yolo11n-pose"], 600),
    ("31_infer_yolo26n_pose", [PYTHON, "-u", "infer_pose.py", "yolo26n-pose"], 600),
    ("32_fps_pose", [PYTHON, "-u", "fps_pose.py"], 1800),
    ("33_video_yolo26n_pose", [PYTHON, "-u", "video_pose.py", "yolo26n-pose", f"{MEDIA}/people-detection.mp4"], 1800),
    ("34_video_yolo11n_pose", [PYTHON, "-u", "video_pose.py", "yolo11n-pose", f"{MEDIA}/people-detection.mp4"], 1800),
]


def cgroup_anon_gib() -> float:
    """Return current cgroup anonymous memory in GiB (excludes reclaimable page cache)."""
    try:
        for line in pathlib.Path("/sys/fs/cgroup/memory.stat").read_text().splitlines():
            if line.startswith("anon "):
                return int(line.split()[1]) / (1024**3)
    except OSError:
        pass
    return 0.0


def log_master(msg: str) -> None:
    line = f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with (LOGS / "master.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run_step(name: str, cmd: list[str], timeout_s: int, extra_env: dict[str, str] | None = None) -> dict:
    log_path = LOGS / f"{name}.log"
    log_master(f"START {name}: {' '.join(cmd)}")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    base_anon = cgroup_anon_gib()
    peak_anon = base_anon
    start = time.time()
    rc = -1
    timed_out = False
    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            peak_anon = max(peak_anon, cgroup_anon_gib())
            if time.time() - start > timeout_s:
                timed_out = True
                proc.kill()
                proc.wait()
                break
            time.sleep(5)
        if not timed_out:
            rc = proc.returncode
    elapsed = time.time() - start
    peak_delta = max(0.0, peak_anon - base_anon)
    status = "TIMEOUT" if timed_out else f"EXIT={rc}"
    log_master(f"DONE  {name}: {status} elapsed={elapsed:.1f}s peak_anon_delta={peak_delta:.1f}GiB")
    return {
        "name": name,
        "cmd": cmd,
        "rc": rc,
        "timed_out": timed_out,
        "elapsed_s": round(elapsed, 2),
        "peak_anon_delta_gib": round(peak_delta, 2),
        "log": str(log_path.relative_to(ROOT)),
    }


def read_log(name: str) -> str:
    p = LOGS / f"{name}.log"
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def fmt_duration(sec: float | str | None) -> str:
    if sec == "skipped":
        return "건너뜀(기존 .edf)"
    if sec is None:
        return "실패/미측정"
    if sec >= 60:
        return f"{sec / 60:.1f}분 ({sec:.1f}s)"
    return f"{sec:.1f}초"


def parse_compile_time(text: str, fallback_s: float | None) -> float | None:
    m = re.search(r"compile ok\s+([\d.]+)s", text)
    if m:
        return float(m.group(1))
    return fallback_s


def parse_fps_lines(text: str, prefix: str = "") -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    pat = re.compile(rf"^{re.escape(prefix)}npu\s+(\w+)\s+cores=(\d+)\s+total=\s*([\d.]+)\s+fps")
    for line in text.splitlines():
        m = pat.search(line.strip())
        if m:
            out[(m.group(1), int(m.group(2)))] = float(m.group(3))
    return out


def parse_bench_single(text: str) -> dict:
    res: dict = {}
    m = re.search(r"cpu match: raw maxdiff=([\d.]+) det cpu=(\d+) npu=(\d+) same_classes=(True|False) box maxdiff=([\d.nan]+)px", text)
    if m:
        res["match"] = f"det {m.group(3)}/{m.group(2)}, same_classes={m.group(4)}, box maxdiff={m.group(5)}px"
    for label, key in [("npu only", "only"), ("npu+copy", "copy")]:
        lm = re.search(rf"{re.escape(label)}\s+n=\d+\s+p50=([\d.]+)ms\s+p99=([\d.]+)ms\s+fps=([\d.]+)", text)
        if lm:
            res[f"{key}_p50"] = float(lm.group(1))
            res[f"{key}_p99"] = float(lm.group(2))
            res[f"{key}_fps"] = float(lm.group(3))
    return res


def parse_bench_parallel(text: str) -> float | None:
    m = re.search(r"total=([\d.]+)\s+fps", text)
    return float(m.group(1)) if m else None


def get_firmware() -> str:
    try:
        out = subprocess.check_output(["furiosa-smi", "info"], text=True, stderr=subprocess.STDOUT)
        for line in out.splitlines():
            if "rngd" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 3:
                    return parts[2]
    except Exception:
        pass
    return "확인 불가"


def render_results(summary: list[dict]) -> None:
    by_name = {s["name"]: s for s in summary}
    today = dt.datetime.now().strftime("%Y-%m-%d")
    ft_ver = meta.version("furiosa-torch")
    fw = get_firmware()

    def ok_time(step: str) -> float | str | None:
        s = by_name.get(step)
        if not s or s["rc"] != 0:
            return None
        txt = read_log(step)
        if "compile ok" not in txt and txt.lstrip().startswith("skip "):
            return "skipped"
        return parse_compile_time(txt, s["elapsed_s"])

    # Compile times
    t_11p = ok_time("28_compile_yolo11n_pose")
    t_26p = ok_time("29_compile_yolo26n_pose")
    t_26 = ok_time("01_compile_yolo26n")
    t_seg = ok_time("15_compile_yolo26n_seg")
    t_proto = ok_time("16_compile_yolo26n_seg_protohead")
    t_dfine = ok_time("20_compile_dfine_encoder")
    t_1pe_pad4 = ok_time("04_compile_yolo26n_1pe_pad4")
    t_2pe = ok_time("05_compile_yolo26n_2pe")
    t_2pe_pad4 = ok_time("06_compile_yolo26n_2pe_pad4")

    # Pose check & FPS
    def pose_match(step: str) -> str:
        txt = read_log(step)
        matches = re.findall(r"matched=(\d+/\d+).*box diff max=([\d.]+)px", txt)
        if matches:
            return ", ".join(f"matched {m[0]} (box {m[1]}px)" for m in matches)
        return "실패/미측정"

    pose_fps_txt = read_log("32_fps_pose")
    fps_11p = parse_fps_lines(pose_fps_txt, "yolo11n-pose ")
    fps_26p = parse_fps_lines(pose_fps_txt, "yolo26n-pose ")

    # Detection check & FPS
    det_run_txt = read_log("02_run_detect")
    det_chk = re.search(r"cpu check n=(\d+) min_cos=([\d.]+) same_count=(\d+/\d+)", det_run_txt)
    det_match = f"same_count {det_chk.group(3)} (min_cos {det_chk.group(2)})" if det_chk else "실패/미측정"
    fps_det = parse_fps_lines(read_log("03_detect_fps"))
    det_cpu_matches = re.findall(r"cpu full threads=(\d+) ([\d.]+) fps", read_log("03_detect_fps"))

    # Seg check & FPS
    seg_chk_txt = read_log("17_seg_check")
    seg_ious = re.findall(r"(\w+) conf diff=([\d.]+) box diff=([\d.]+)px mask iou min=([\d.]+) mean=([\d.]+)", seg_chk_txt)
    seg_match = ", ".join(f"{x[0]} IoU min={x[3]} mean={x[4]}" for x in seg_ious) if seg_ious else "실패/미측정"
    fps_seg = parse_fps_lines(read_log("19_seg_fps"))

    # D-FINE check & FPS
    df_chk_txt = read_log("25_dfine_check")
    df_diffs = re.findall(r"(\w+) conf diff=([\d.]+) box diff=([\d.]+)px", df_chk_txt)
    df_match = ", ".join(f"{x[0]} conf {x[1]} box {x[2]}px" for x in df_diffs) if df_diffs else "실패/미측정"
    df_fps_txt = read_log("27_dfine_fps")
    df_1c = re.search(r"npu whole=True cores=1 threads=2 total=\s*([\d.]+)\s+fps\s+ms=(\{.*\})", df_fps_txt)
    df_8c = re.search(r"npu whole=True cores=8 threads=1 total=\s*([\d.]+)\s+fps\s+ms=(\{.*\})", df_fps_txt)

    # Video FPS
    def vid_fps(txt: str) -> str:
        m = re.search(r"fps=([\d.]+)", txt)
        return f"{m.group(1)} FPS" if m else "미측정"

    det_vid = vid_fps(det_run_txt)
    seg_vid = vid_fps(read_log("18_seg_files"))
    df_vid = vid_fps(read_log("26_dfine_files"))
    p26_vid = vid_fps(read_log("33_video_yolo26n_pose"))
    p11_vid = vid_fps(read_log("34_video_yolo11n_pose"))

    # Fusion
    b_1pe_3 = parse_bench_single(read_log("07_bench_1pe_3ch"))
    b_1pe_4 = parse_bench_single(read_log("08_bench_1pe_pad4"))
    b_2pe_3 = parse_bench_single(read_log("09_bench_2pe_3ch"))
    b_2pe_4 = parse_bench_single(read_log("10_bench_2pe_pad4"))
    p_1pe_2 = parse_bench_parallel(read_log("11_bench_1pe_pad4_x2"))
    p_1pe_8 = parse_bench_parallel(read_log("12_bench_1pe_pad4_x8"))
    p_2pe_2 = parse_bench_parallel(read_log("13_bench_2pe_pad4_x2"))
    p_2pe_4 = parse_bench_parallel(read_log("14_bench_2pe_pad4_x4"))

    def fnum(v: float | None, fmt: str = ".1f") -> str:
        return f"{v:{fmt}}" if v is not None else "-"

    def row_fps(label: str, d: dict[tuple[str, int], float]) -> str:
        return f"| {label} | {fnum(d.get(('raw', 1)))} | {fnum(d.get(('raw', 8)))} | {fnum(d.get(('full', 1)))} | {fnum(d.get(('full', 8)))} |"

    def row_fusion(label: str, ch: str, b: dict, ctime: float | str | None, step_name: str) -> str:
        s = by_name.get(step_name, {})
        mem = f", 최대 Δ{s.get('peak_anon_delta_gib', 0):.1f} GiB" if s.get("rc") == 0 and isinstance(ctime, (int, float)) else ""
        p50 = f"{b['only_p50']:.2f} ms" if "only_p50" in b else "-"
        fps_o = fnum(b.get("only_fps"))
        fps_c = fnum(b.get("copy_fps"))
        return f"| {label} | {ch} | {p50} | {fps_o} | {fps_c} | {fmt_duration(ctime)}{mem} |"

    failed = [s for s in summary if s["rc"] != 0]

    lines = [
        "# Furiosa RNGD 비전 모델 컴파일 및 NPU 측정 결과",
        "",
        f"측정일: {today}. 환경: `yolo26/.venv` (Python {sys.version.split()[0]}), `furiosa-torch` {ft_ver}, 펌웨어 `{fw}`, RNGD 1장(코어 8개, `furiosa:0`~`furiosa:7`), 입력 640×640.",
        "",
        "## 결과 요약",
        "",
        "| 모델 | 태스크 | NPU 구간 | CPU 구간 | 컴파일 시간 | CPU 일치 (실측) |",
        "|---|---|---|---|---|---|",
        f"| YOLO11n-Pose | Pose | 모델 전체 | 박스 복원, NMS | {fmt_duration(t_11p)} | {pose_match('30_infer_yolo11n_pose')} |",
        f"| YOLO26n-Pose | Pose | 모델 전체 | 박스 복원, NMS | {fmt_duration(t_26p)} | {pose_match('31_infer_yolo26n_pose')} |",
        f"| YOLO26n | Detection | 모델 전체 | 박스 복원, NMS | {fmt_duration(t_26)} | {det_match} |",
        f"| YOLO26n-seg | Segmentation | 본체 + Proto (그래프 2개) | NMS, 마스크 조립 | {fmt_duration(t_seg)} + {fmt_duration(t_proto)} | {seg_match} |",
        f"| D-FINE nano | Detection | Backbone + Encoder | Decoder | {fmt_duration(t_dfine)} | {df_match} |",
        "",
        "## FPS",
        "",
        "입력 전처리 제외, 같은 이미지(`bus.jpg`) 반복 측정. `raw`는 NPU 추론과 출력 복사, `full`은 CPU 후처리(디코드·NMS)까지 포함합니다.",
        "",
        "| 모델 | raw 1코어 | raw 8코어 | full 1코어 | full 8코어 |",
        "|---|---|---|---|---|",
        row_fps("YOLO11n-Pose", fps_11p),
        row_fps("YOLO26n-Pose", fps_26p),
        row_fps("YOLO26n Detection", fps_det),
        row_fps("YOLO26n-seg", fps_seg),
        f"| D-FINE nano | - | - | {df_1c.group(1) if df_1c else '-'} | {df_8c.group(1) if df_8c else '-'} |",
        "",
    ]

    if det_cpu_matches:
        vals = ", ".join(f"{fps} FPS ({thr} threads)" for thr, fps in det_cpu_matches)
        lines.append(f"- YOLO26n Detection CPU 기준값: {vals}")
    if df_1c:
        lines.append(f"- D-FINE nano 1코어(2 threads) 중앙값 지연: {df_1c.group(2)}")
    if df_8c:
        lines.append(f"- D-FINE nano 8코어(1 thread) 중앙값 지연: {df_8c.group(2)}")
    lines.extend([
        f"- 영상 파일 처리 (코어 1개, 읽기·전처리·저장 포함): YOLO26n Detection {det_vid}, YOLO26n-seg {seg_vid}, D-FINE nano {df_vid}, YOLO26n-Pose {p26_vid}, YOLO11n-Pose {p11_vid}",
        "",
        "## 1코어와 2코어 실험 (YOLO26n Detection)",
        "",
        "| 모델 | 입력 | 지연 p50 (`npu only`) | 단일 스트림 FPS | 출력 복사 포함 FPS (`npu+copy`) | 컴파일 |",
        "|---|---|---|---|---|---|",
        row_fusion("1-PE `yolo26n.edf`", "3채널", b_1pe_3, t_26, "01_compile_yolo26n"),
        row_fusion("1-PE `yolo26n-1pe-pad4.edf`", "4채널", b_1pe_4, t_1pe_pad4, "04_compile_yolo26n_1pe_pad4"),
        row_fusion("2-PE `yolo26n-2pe.edf`", "3채널", b_2pe_3, t_2pe, "05_compile_yolo26n_2pe"),
        row_fusion("2-PE `yolo26n-2pe-pad4.edf`", "4채널", b_2pe_4, t_2pe_pad4, "06_compile_yolo26n_2pe_pad4"),
        "",
        "코어 수를 맞춘 병렬 처리량(`npu+copy`, 장치당 프로세스 1개):",
        "",
        "| 사용 PE | 1-PE 복제 (4채널) | 2-PE 묶음 (4채널) |",
        "|---|---|---|",
        f"| 2 | {fnum(p_1pe_2)} | {fnum(b_2pe_4.get('copy_fps'))} |",
        f"| 4 | - | {fnum(p_2pe_2)} |",
        f"| 8 | {fnum(p_1pe_8)} | {fnum(p_2pe_4)} |",
        "",
    ])

    if failed:
        lines.extend(["## 실패 또는 미완료 단계", ""])
        for s in failed:
            lines.append(f"- `{s['name']}`: exit={s['rc']}, timeout={s['timed_out']} (`{s['log']}`)")
        lines.append("")

    lines.extend([
        "## 실행 및 로그",
        "",
        "```bash",
        "cd yolo26",
        ".venv/bin/python -u run_full_bench.py",
        "```",
        "",
        f"- 전체 단계 로그: `yolo26/bench_logs/` (`master.log`, `summary.json`)",
        f"- 생성된 이미지·영상 결과: `{MEDIA}/result` 및 `{MEDIA}`",
        "",
    ])

    (ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    LOGS.mkdir(exist_ok=True)
    if "--render-only" in sys.argv:
        summary = json.loads((LOGS / "summary.json").read_text(encoding="utf-8"))
        render_results(summary)
        return

    extra_env = {"FORCE_COMPILE": "1"} if "--force" in sys.argv else None
    (LOGS / "master.log").write_text("", encoding="utf-8")
    summary: list[dict] = []
    for name, cmd, timeout_s in STEPS:
        rec = run_step(name, cmd, timeout_s, extra_env)
        summary.append(rec)
        (LOGS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    render_results(summary)
    log_master("ALL DONE -> RESULTS.md written")


if __name__ == "__main__":
    main()
