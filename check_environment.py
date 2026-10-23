"""Import and computation checks; no model downloads or song processing."""

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    for name in (
        "numpy",
        "soundfile",
        "scipy",
        "torch",
        "torchaudio",
        "demucs.pretrained",
        "demucs.apply",
        "transformers",
        "uroman",
        "stable_whisper",
        "whisper",
        "pypinyin",
        "mms_refine",
        "lyricsync",
    ):
        importlib.import_module(name)
        print("OK:", name, flush=True)

    import torch
    import torchaudio

    scripts = Path(sys.executable).parent
    ffmpeg = scripts / "ffmpeg.exe"
    ffprobe = scripts / "ffprobe.exe"

    # 優先使用系統原生硬體加速 FFmpeg 與 ffprobe
    sys_ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg.exists():
        if sys_ffmpeg and Path(sys_ffmpeg).resolve() != ffmpeg.resolve():
            try:
                shutil.copy2(sys_ffmpeg, ffmpeg)
            except Exception:
                pass
        if not ffmpeg.exists():
            try:
                import imageio_ffmpeg

                shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), ffmpeg)
            except Exception:
                pass

    sys_ffprobe = shutil.which("ffprobe")
    if not ffprobe.exists() and sys_ffprobe:
        try:
            shutil.copy2(sys_ffprobe, ffprobe)
        except Exception:
            pass

    ffmpeg_target = ffmpeg if ffmpeg.exists() else (sys_ffmpeg or "ffmpeg")
    subprocess.run(
        [str(ffmpeg_target), "-version"], check=True, stdout=subprocess.DEVNULL
    )

    device = "cpu"
    reason = ""
    if torch.cuda.is_available():
        try:
            x = torch.ones((16, 16), device="cuda")
            y = x @ x
            torch.cuda.synchronize()
            assert y[0, 0].item() == 16
            device = "cuda"
            print("GPU:", torch.cuda.get_device_name(0))
        except Exception as exc:
            reason = str(exc)
            print("GPU computation unavailable. CPU mode will be used:", exc)
    else:
        reason = "CUDA is unavailable."

    cfg = {
        "device": device,
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "gpu_note": reason,
    }
    (root / "runtime_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("FFmpeg: OK")
    print("Runtime device:", device)
    print("Package checks passed. AI models will download on first song.")


if __name__ == "__main__":
    main()
