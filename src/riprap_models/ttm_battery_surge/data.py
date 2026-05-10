"""NOAA CO-OPS loader for The Battery (8518750). Surge residual =
``water_level`` minus ``predictions`` (astronomical tide). Hourly-mean
resampler matches the fine-tune's 1024-step training cadence.

API: https://api.tidesandcurrents.noaa.gov/api/prod/ (public, no auth).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import requests

COOPS_BASE = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
DEFAULT_STATION = "8518750"  # The Battery, NY


def _coops_get_one(product: str, station: str, begin: str, end: str) -> list[dict]:
    """Fetch a CO-OPS product for a single window. Both inputs YYYYMMDD."""
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
    r = requests.get(COOPS_BASE, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()
    if "data" in payload:
        return payload["data"]
    if "predictions" in payload:
        return payload["predictions"]
    return []


def _coops_get(product: str, station: str, begin: str, end: str) -> list[dict]:
    """Fetch a CO-OPS product, chunking 31 days at a time to satisfy the
    water_level endpoint's window cap. ``predictions`` accepts longer
    windows but the same chunking works for both."""
    from datetime import datetime, timedelta

    fmt = "%Y%m%d"
    b = datetime.strptime(begin, fmt)
    e = datetime.strptime(end, fmt)
    out: list[dict] = []
    cursor = b
    chunk = timedelta(days=30)
    while cursor <= e:
        next_end = min(cursor + chunk, e)
        out.extend(
            _coops_get_one(product, station, cursor.strftime(fmt), next_end.strftime(fmt))
        )
        cursor = next_end + timedelta(days=1)
    return out


def fetch_residual_series(
    station: str, begin: str, end: str, hourly: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(timestamps_iso, residual_m)`` for the requested window.

    Residual is in metres. When ``hourly=True`` (the default, matching the
    fine-tune training cadence), the 6-min CO-OPS samples are averaged into
    hourly buckets keyed by ``YYYY-MM-DD HH:00``. ``hourly=False`` returns
    the raw 6-min series for callers that want it.
    """
    obs = _coops_get("water_level", station, begin, end)
    pred = _coops_get("predictions", station, begin, end)
    by_t_obs = {row["t"]: float(row["v"]) for row in obs if row.get("v") not in (None, "")}
    by_t_pred = {row["t"]: float(row["v"]) for row in pred if row.get("v") not in (None, "")}
    common = sorted(set(by_t_obs) & set(by_t_pred))
    if not hourly:
        ts = np.array(common)
        res = np.array([by_t_obs[t] - by_t_pred[t] for t in common], dtype=np.float64)
        return ts, res

    # Hourly mean. NOAA timestamps are "YYYY-MM-DD HH:MM" GMT; the bucket key
    # is the same string truncated to "YYYY-MM-DD HH:00".
    buckets: dict[str, list[float]] = {}
    for t in common:
        bucket = t[:13] + ":00"
        buckets.setdefault(bucket, []).append(by_t_obs[t] - by_t_pred[t])
    keys = sorted(buckets)
    ts = np.array(keys)
    res = np.array([sum(buckets[k]) / len(buckets[k]) for k in keys], dtype=np.float64)
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
    """Fallback when granite-tsfm isn't installed. Predicts zero residual."""

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        return np.zeros((horizon,), dtype=np.float32)


class _TTMForecaster:
    """``predict(history, horizon)``: standardize, ``past_values=(1,T,1)``,
    de-standardize. Same call shape as riprap-nyc/app/live/ttm_forecast.py.
    """

    def __init__(self, model, context_length: int, prediction_length: int):
        self.model = model
        self.context_length = context_length
        self.prediction_length = prediction_length

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        import torch

        if history.size < self.context_length:
            raise ValueError(
                f"history has {history.size} steps; model needs {self.context_length}"
            )
        h = history[-self.context_length :].astype(np.float64)
        mu = float(h.mean())
        sigma = float(h.std() + 1e-6)
        normed = (h - mu) / sigma
        x = torch.from_numpy(normed.astype(np.float32))[None, :, None]
        with torch.no_grad():
            out = self.model(past_values=x)
        pred = out.prediction_outputs[0, :, 0].cpu().numpy()
        pred = pred * sigma + mu
        return pred[:horizon].astype(np.float32)


def load_finetune(cfg: dict):
    """Load the Battery surge fine-tune via tsfm_public.

    Returns an object with ``predict(history, horizon) -> np.ndarray``.
    Falls back to ``_ZeroForecaster`` when granite-tsfm isn't installed.
    """
    try:
        from tsfm_public.toolkit.get_model import get_model
    except ImportError:
        return _ZeroForecaster()

    model_id = cfg.get("model_id", "msradam/Granite-TTM-r2-Battery-Surge")
    context_length = int(cfg.get("context_steps", 1024))
    prediction_length = int(cfg.get("horizon_steps", 96))
    kwargs = dict(context_length=context_length, prediction_length=prediction_length)
    if cfg.get("model_revision"):
        kwargs["revision"] = cfg["model_revision"]
    model = get_model(model_id, **kwargs)
    model.eval()
    return _TTMForecaster(model, context_length, prediction_length)
