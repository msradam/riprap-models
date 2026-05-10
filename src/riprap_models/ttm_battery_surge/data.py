"""NOAA CO-OPS data loader for The Battery (station 8518750).

The CO-OPS API is public and unauthenticated. We pull the
``water_level`` product (observed) and the ``predictions`` product
(astronomical tide). The surge residual = observed − predicted.

Reference: https://api.tidesandcurrents.noaa.gov/api/prod/
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import requests

COOPS_BASE = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
DEFAULT_STATION = "8518750"  # The Battery, NY


def _coops_get(product: str, station: str, begin: str, end: str) -> list[dict]:
    """Fetch a CO-OPS product. ``begin``/``end`` are YYYYMMDD strings."""
    params = {
        "product": product,
        "application": "riprap-models",
        "station": station,
        "begin_date": begin,
        "end_date": end,
        "datum": "MLLW",
        "units": "metric",
        "time_zone": "gmt",
        "format": "json",
    }
    r = requests.get(COOPS_BASE, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if "data" in payload:
        return payload["data"]
    if "predictions" in payload:
        return payload["predictions"]
    return []


def fetch_residual_series(station: str, begin: str, end: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(timestamps_iso, residual_m)`` for the requested window.

    Residual is in metres. Timestamps follow CO-OPS' GMT format and are
    aligned across observed and predicted by intersecting on timestamp.
    """
    obs = _coops_get("water_level", station, begin, end)
    pred = _coops_get("predictions", station, begin, end)
    by_t_obs = {row["t"]: float(row["v"]) for row in obs if row.get("v") not in (None, "")}
    by_t_pred = {row["t"]: float(row["v"]) for row in pred if row.get("v") not in (None, "")}
    common = sorted(set(by_t_obs) & set(by_t_pred))
    ts = np.array(common)
    res = np.array([by_t_obs[t] - by_t_pred[t] for t in common], dtype=np.float64)
    return ts, res


def fetch_window(
    station: str,
    history_start: str,
    history_end: str,
    target_start: str,
    target_end: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(history_residual, target_residual)`` for one eval window."""
    _, hist = fetch_residual_series(station, history_start, history_end)
    _, tgt = fetch_residual_series(station, target_start, target_end)
    return hist, tgt


@dataclass
class _ZeroForecaster:
    """Minimal forecaster used when the real fine-tune isn't installed.

    Predicts zero residual everywhere. Lets the bench path return real
    timing numbers without weights, but the eval path will be obviously
    wrong (large MAE) so a reviewer can't mistake it for a measurement.
    """

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        return np.zeros((horizon,), dtype=np.float32)


def load_finetune(cfg: dict):
    """Load the Battery surge fine-tune via tsfm_public.

    Returns an object with ``predict(history: np.ndarray, horizon: int) -> np.ndarray``.
    """
    try:
        from tsfm_public.toolkit.get_model import get_model  # type: ignore  # noqa: F401
    except ImportError:
        return _ZeroForecaster()

    revision = cfg.get("model_revision")
    raise NotImplementedError(
        "Wire up tsfm_public.get_model('msradam/Granite-TTM-r2-Battery-Surge') "
        f"with revision={revision}. The riprap-nyc reference is "
        "app/live/ttm_forecast.py — that file uses a 512-step context and "
        "96-step horizon at 6-minute cadence."
    )
