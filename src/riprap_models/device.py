"""Device + dtype selection.

``get_device()`` returns ``(kind, dtype, label)``. Order: CUDA, MPS, CPU.
Dtype: float16 on cuda/mps, float32 on cpu.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Literal

DeviceKind = Literal["cuda", "mps", "cpu"]


@dataclass(frozen=True)
class DeviceInfo:
    kind: DeviceKind
    dtype: str  # torch dtype name; the loader resolves it via getattr(torch, dtype)
    label: str  # human-readable, e.g. "Apple M3 (MPS, fp16)"


def get_device(prefer: DeviceKind | None = None) -> DeviceInfo:
    """Pick a device + dtype. ``prefer`` forces a kind for testing."""

    try:
        import torch  # noqa: F401
    except ImportError:
        # The base install does not pull torch. Callers who need a device for
        # actual inference must install one of the model extras. We still
        # return CPU here so smoke tests don't crash.
        return DeviceInfo(kind="cpu", dtype="float32", label=f"{_cpu_label()} (CPU, fp32, no torch)")

    import torch

    if prefer == "cuda" or (prefer is None and torch.cuda.is_available()):
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA"
        return DeviceInfo(kind="cuda", dtype="float16", label=f"{name} (CUDA, fp16)")

    if prefer == "mps" or (prefer is None and _mps_available(torch)):
        return DeviceInfo(kind="mps", dtype="float16", label=f"{_cpu_label()} (MPS, fp16)")

    return DeviceInfo(kind="cpu", dtype="float32", label=f"{_cpu_label()} (CPU, fp32)")


def _mps_available(torch_mod) -> bool:
    backends = getattr(torch_mod, "backends", None)
    mps = getattr(backends, "mps", None) if backends else None
    if mps is None:
        return False
    try:
        return bool(mps.is_available()) and bool(mps.is_built())
    except Exception:
        return False


def _cpu_label() -> str:
    sysname = platform.system()
    machine = platform.machine()
    if sysname == "Darwin" and machine == "arm64":
        # Apple Silicon. ``platform.processor()`` returns "arm" on macOS, not
        # the chip name, so we read sysctl.
        try:
            import subprocess

            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=1
            ).strip()
            return out or "Apple Silicon"
        except Exception:
            return "Apple Silicon"
    return f"{sysname} {machine}"


def mps_fallback_enabled() -> bool:
    return os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
