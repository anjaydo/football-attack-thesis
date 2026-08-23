from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from importlib import metadata


def _package_version(distribution: str, module: str | None = None) -> str | None:
    module_name = module or distribution.replace("-", "_")
    if importlib.util.find_spec(module_name) is None:
        return None
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "installed-unknown-version"


def environment_report() -> dict[str, object]:
    packages = {
        "numpy": _package_version("numpy"),
        "opencv": _package_version("opencv-python", "cv2"),
        "torch": _package_version("torch"),
        "ultralytics": _package_version("ultralytics"),
        "soccernet": _package_version("SoccerNet"),
    }
    cuda_available = False
    if packages["torch"] is not None:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False

    core_ready = packages["numpy"] is not None
    video_ready = packages["opencv"] is not None
    vision_ready = all(
        packages[name] is not None for name in ("numpy", "opencv", "torch", "ultralytics")
    )
    recommendations: list[str] = []
    if not core_ready:
        recommendations.append('install core: python -m pip install -e .')
    if not vision_ready:
        recommendations.append('install vision: python -m pip install -e ".[vision]"')
    if vision_ready and not cuda_available:
        recommendations.append("use --device cpu and --max-frames 300 for the first run")
    if shutil.which("ffmpeg") is None:
        recommendations.append("FFmpeg is optional but required to remux original audio")
    if shutil.which("curl") is None:
        recommendations.append("curl is required for partial SoccerNet sequence downloads")

    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "packages": packages,
        "tools": {
            "ffmpeg": shutil.which("ffmpeg"),
            "nvidia_smi": shutil.which("nvidia-smi"),
            "curl": shutil.which("curl"),
            "cuda_available": cuda_available,
        },
        "core_ready": core_ready,
        "video_ready": video_ready,
        "vision_ready": vision_ready,
        "recommendations": recommendations,
    }
