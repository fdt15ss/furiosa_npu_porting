"""Skip NPU compilation when the target .edf file already exists."""
import os
import pathlib
import sys


def should_force(argv=None) -> bool:
    """Return True when --force or FORCE_COMPILE=1 is set."""
    args = sys.argv[1:] if argv is None else argv
    if "--force" in args:
        return True
    return os.environ.get("FORCE_COMPILE", "").strip().lower() in {"1", "true", "yes"}


def skip_if_edf_exists(path: str | pathlib.Path, argv=None) -> None:
    """Print a skip notice and exit(0) if `path` exists and is non-empty unless forced."""
    if should_force(argv):
        return
    edf = pathlib.Path(path)
    if edf.is_file() and edf.stat().st_size > 0:
        print(
            f"skip {edf} (already exists, {edf.stat().st_size} bytes; pass --force or FORCE_COMPILE=1 to recompile)",
            flush=True,
        )
        raise SystemExit(0)
