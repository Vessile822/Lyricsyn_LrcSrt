"""Run from any working directory and keep complete output for troubleshooting."""

import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    args = sys.argv[1:] or [str(project_root / "songs")]
    log = project_root / "last_run.log"
    print(
        "LyricSync: processing. The first run downloads AI models; please keep this window open.",
        flush=True,
    )
    print("Log:", log, flush=True)
    with log.open("w", encoding="utf-8") as out:
        out.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        out.flush()
        proc = subprocess.Popen(
            [sys.executable, "-u", str(src_dir / "lyricsync.py"), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            out.write(line)
            out.flush()
        code = proc.wait()
    if code:
        print(
            f"Processing failed (exit code {code}). Please send last_run.log back for help.",
            flush=True,
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
