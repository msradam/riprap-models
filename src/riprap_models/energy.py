"""Per-call energy measurement.

    with measure_energy() as m:
        run_inference(...)
    print(m.joules, m.duration_s, m.peak_memory_mb, m.method)

Methods, auto-selected by platform: ``nvml`` (NVIDIA), ``rapl`` (Linux x86 CPU),
``powermetrics`` (macOS, sudo without prompt required), ``estimated`` (fallback,
duration × platform envelope; see ``docs/ENERGY.md``).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

import psutil

Method = Literal["nvml", "rapl", "powermetrics", "estimated"]


# Conservative power envelopes for the "estimated" fallback. These are not
# claims about peak silicon TDP; they are realistic average package + DRAM
# draw under sustained inference workloads. Sourced in docs/ENERGY.md.
_ESTIMATE_W: dict[str, float] = {
    "darwin-arm64": 12.0,   # Apple M-series Air, no fan, sustained inference
    "linux-x86_64": 35.0,   # generic desktop CPU package + DRAM
    "linux-aarch64": 8.0,   # generic ARM SBC
    "default": 25.0,
}


@dataclass
class EnergyResult:
    joules: float
    duration_s: float
    peak_memory_mb: float
    method: Method
    notes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "joules": round(self.joules, 4),
            "duration_s": round(self.duration_s, 4),
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "method": self.method,
            "notes": dict(self.notes),
        }


def _platform_key() -> str:
    sysname = platform.system().lower()
    arch = platform.machine().lower()
    return f"{sysname}-{arch}"


def _estimate_watts() -> float:
    return _ESTIMATE_W.get(_platform_key(), _ESTIMATE_W["default"])


# ---------- NVML (NVIDIA) -----------------------------------------------------


def _try_nvml_sampler():
    try:
        import pynvml  # type: ignore
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None
    return pynvml, handle


# ---------- Linux RAPL --------------------------------------------------------


_RAPL_PATH = "/sys/class/powercap/intel-rapl:0/energy_uj"


def _try_rapl_read() -> int | None:
    try:
        with open(_RAPL_PATH) as f:
            return int(f.read().strip())
    except Exception:
        return None


# ---------- macOS powermetrics ------------------------------------------------


def _powermetrics_available() -> bool:
    """True iff we can run ``sudo -n powermetrics`` without a TTY prompt."""
    if platform.system() != "Darwin":
        return False
    if not shutil.which("powermetrics"):
        return False
    if os.environ.get("RIPRAP_MODELS_NO_POWERMETRICS"):
        return False
    try:
        # -n: non-interactive. Returns non-zero immediately if a prompt
        # would be required. We only care about exit status, not output.
        r = subprocess.run(
            ["sudo", "-n", "true"],
            capture_output=True,
            timeout=1,
        )
        return r.returncode == 0
    except Exception:
        return False


# ---------- Memory peak sampler (cross-platform) ------------------------------


class _PeakMemorySampler(threading.Thread):
    def __init__(self, interval_s: float = 0.05):
        super().__init__(daemon=True)
        self._interval = interval_s
        self._stop_evt = threading.Event()
        self._proc = psutil.Process(os.getpid())
        self.peak_mb: float = self._sample()

    def _sample(self) -> float:
        try:
            return self._proc.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def run(self) -> None:
        while not self._stop_evt.is_set():
            cur = self._sample()
            if cur > self.peak_mb:
                self.peak_mb = cur
            self._stop_evt.wait(self._interval)

    def stop(self) -> float:
        self._stop_evt.set()
        self.join(timeout=1.0)
        return self.peak_mb


# ---------- NVML continuous sampler ------------------------------------------


class _NvmlSampler(threading.Thread):
    def __init__(self, pynvml, handle, interval_s: float = 0.02):
        super().__init__(daemon=True)
        self._pynvml = pynvml
        self._handle = handle
        self._interval = interval_s
        self._stop_evt = threading.Event()
        self._samples: list[tuple[float, float]] = []  # (timestamp, watts)

    def _read_watts(self) -> float:
        try:
            mw = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
            return mw / 1000.0
        except Exception:
            return 0.0

    def run(self) -> None:
        while not self._stop_evt.is_set():
            self._samples.append((time.perf_counter(), self._read_watts()))
            self._stop_evt.wait(self._interval)

    def stop(self) -> float:
        """Return integrated joules (trapezoid rule on the sample series)."""
        self._stop_evt.set()
        self.join(timeout=1.0)
        s = self._samples
        if len(s) < 2:
            return 0.0
        joules = 0.0
        for i in range(1, len(s)):
            dt = s[i][0] - s[i - 1][0]
            avg_w = (s[i][1] + s[i - 1][1]) / 2.0
            joules += dt * avg_w
        return joules


# ---------- Public context manager -------------------------------------------


@contextmanager
def measure_energy(prefer: Method | None = None) -> Iterator[EnergyResult]:
    """Measure energy + duration + peak RSS for the enclosed block.

    ``prefer`` lets a caller force a method (mostly for tests). When None,
    we pick the best available for the current platform.
    """

    result = EnergyResult(joules=0.0, duration_s=0.0, peak_memory_mb=0.0, method="estimated")
    mem_sampler = _PeakMemorySampler()
    mem_sampler.start()
    t0 = time.perf_counter()

    nvml_state = None
    nvml_sampler: _NvmlSampler | None = None
    rapl_start: int | None = None
    method: Method = "estimated"

    if prefer in (None, "nvml"):
        nvml_state = _try_nvml_sampler()
        if nvml_state is not None:
            method = "nvml"
            nvml_sampler = _NvmlSampler(*nvml_state)
            nvml_sampler.start()

    if method == "estimated" and prefer in (None, "rapl"):
        rapl_start = _try_rapl_read()
        if rapl_start is not None:
            method = "rapl"

    powermetrics_proc: subprocess.Popen | None = None
    powermetrics_log: str | None = None
    if method == "estimated" and prefer in (None, "powermetrics") and _powermetrics_available():
        method = "powermetrics"
        powermetrics_log = f"/tmp/riprap-models-pm-{os.getpid()}-{int(t0)}.txt"
        # 100ms sample interval, plain text output. We collect the whole
        # block then average package power across CPU + GPU planes.
        try:
            powermetrics_proc = subprocess.Popen(
                ["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power",
                 "-i", "100", "-f", "text", "-o", powermetrics_log],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            method = "estimated"
            powermetrics_proc = None

    try:
        yield result
    finally:
        duration = time.perf_counter() - t0
        peak_mb = mem_sampler.stop()

        joules = 0.0
        notes: dict[str, str] = {}

        if method == "nvml" and nvml_sampler is not None:
            joules = nvml_sampler.stop()
            try:
                nvml_state[0].nvmlShutdown()  # type: ignore[index]
            except Exception:
                pass
        elif method == "rapl" and rapl_start is not None:
            rapl_end = _try_rapl_read()
            if rapl_end is not None and rapl_end >= rapl_start:
                joules = (rapl_end - rapl_start) / 1_000_000.0  # uJ → J
            else:
                # Counter wrapped or read failed. Fall back to estimate.
                method = "estimated"
                joules = duration * _estimate_watts()
                notes["rapl_fallback"] = "wrapped_or_read_failed"
        elif method == "powermetrics" and powermetrics_proc is not None:
            try:
                powermetrics_proc.terminate()
                powermetrics_proc.wait(timeout=2)
            except Exception:
                try:
                    powermetrics_proc.kill()
                except Exception:
                    pass
            joules, pm_notes = _parse_powermetrics(powermetrics_log, duration)
            notes.update(pm_notes)
            if joules <= 0:
                method = "estimated"
                joules = duration * _estimate_watts()
                notes["powermetrics_fallback"] = "no_samples_parsed"
        else:
            method = "estimated"
            watts = _estimate_watts()
            joules = duration * watts
            notes["estimate_watts"] = str(watts)
            notes["estimate_basis"] = _platform_key()

        result.joules = joules
        result.duration_s = duration
        result.peak_memory_mb = peak_mb
        result.method = method
        result.notes = notes


def _parse_powermetrics(path: str | None, duration_s: float) -> tuple[float, dict[str, str]]:
    """Parse a powermetrics text dump and return integrated joules + notes."""
    if not path:
        return 0.0, {}
    try:
        with open(path) as f:
            text = f.read()
    except Exception:
        return 0.0, {"powermetrics_read": "failed"}

    # powermetrics text output reports per-sample lines like:
    #   CPU Power: 1234 mW
    #   GPU Power: 567 mW
    # Sum them across samples, then average × duration.
    cpu_mw: list[int] = []
    gpu_mw: list[int] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("CPU Power:"):
            cpu_mw.append(_extract_mw(s))
        elif s.startswith("GPU Power:"):
            gpu_mw.append(_extract_mw(s))

    if not cpu_mw and not gpu_mw:
        return 0.0, {"powermetrics_samples": "0"}

    avg_w = (sum(cpu_mw) / max(1, len(cpu_mw)) + sum(gpu_mw) / max(1, len(gpu_mw))) / 1000.0
    notes = {
        "powermetrics_samples_cpu": str(len(cpu_mw)),
        "powermetrics_samples_gpu": str(len(gpu_mw)),
        "powermetrics_avg_w": f"{avg_w:.2f}",
    }
    return avg_w * duration_s, notes


def _extract_mw(line: str) -> int:
    # "CPU Power: 1234 mW" → 1234. Defensive: powermetrics has occasionally
    # emitted "N/A" on cold boot.
    parts = line.split(":", 1)
    if len(parts) != 2:
        return 0
    val = parts[1].strip().split()[0]
    try:
        return int(val)
    except ValueError:
        return 0
