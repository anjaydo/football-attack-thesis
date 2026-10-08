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


def _bundled_ffmpeg_executable() -> str | None:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError:
        return None
    try:
        return get_ffmpeg_exe()
    except Exception:
        return None


def default_torch_device() -> str:
    """Use GPU 0 when this interpreter's PyTorch was built with CUDA."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def default_inference_imgsz() -> int:
    """1080p broadcast balls are ~12px. 1280 on the RTX 4050 upscales them; CPU stays at 960."""
    if default_torch_device() == "0":
        return 1280
    return 960


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
    if vision_ready and cuda_available:
        recommendations.append(
            "CUDA is available; annotate-video, run-pipeline and train-ball-model "
            "use --device 0 unless you pass --device cpu. Ball tiles are off by default; "
            "--ball-tile-size runs them at batch 2 so a 6GB laptop GPU can keep the "
            "player, pose and ball models loaded."
        )
    elif vision_ready and not cuda_available:
        recommendations.append("use --device cpu and --max-frames 300 for the first run")
        recommendations.append(
            "ball F1 0.35 vs gate 0.80: train-ball-model --imgsz 1280 needs NVIDIA GPU "
            "(--device 0). FOOTPASS+HDF5 xG does not wait on this gate"
        )
    if shutil.which("ffmpeg") is None and _package_version("imageio-ffmpeg") is None:
        recommendations.append(
            'install browser H.264 support: python -m pip install "imageio-ffmpeg>=0.5"'
        )
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
            "ffmpeg": shutil.which("ffmpeg") or _bundled_ffmpeg_executable(),
            "nvidia_smi": shutil.which("nvidia-smi"),
            "curl": shutil.which("curl"),
            "cuda_available": cuda_available,
        },
        "core_ready": core_ready,
        "video_ready": video_ready,
        "vision_ready": vision_ready,
        "recommendations": recommendations,
    }
