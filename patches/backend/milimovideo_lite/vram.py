import os
import subprocess
from typing import Optional

import torch


def _read_nvidia_smi() -> Optional[float]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode("utf-8").strip()
        if not out:
            return None
        first = float(out.splitlines()[0].strip())
        return first / 1024.0
    except Exception:
        return None


def get_total_vram_gb() -> Optional[float]:
    if torch.cuda.is_available():
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            return total / (1024.0 ** 3)
        except Exception:
            pass
    return _read_nvidia_smi()


def get_vram_mode() -> str:
    """Return configured mode: 'high', 'low', or 'auto'."""
    mode = os.environ.get("MILIMO_VRAM_MODE", "auto").strip().lower()
    if mode in {"high", "low", "auto"}:
        return mode
    return "auto"


def resolve_runtime_mode() -> str:
    """Return effective runtime profile: high, low, or cpu."""
    configured = get_vram_mode()
    if configured in {"high", "low"}:
        return configured

    total = get_total_vram_gb()
    if total is None:
        return "low"
    if total >= 16.0:
        return "high"
    if total >= 6.0:
        return "low"
    return "cpu"
