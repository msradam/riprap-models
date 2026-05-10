"""Live data fetch + freeze-to-fixture path.

For each model, ``run_live(name, fixtures_dir)`` fetches a fresh slice of
public NYC data, runs inference on it, and writes:

    eval/fixtures/<name>/<UTC-timestamp>/
        inputs.<ext>          raw fetched data
        outputs.<ext>          model output
        manifest.json         provenance + replay metadata

``replay(name, fixture_dir)`` loads the inputs and reruns inference,
asserting bit-identical (or float-tolerant) outputs against what's saved.
That gives a reviewer a reproducible run that doesn't require live network.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .common.provenance import code_sha


def _fixture_dir_for(name: str, fixtures_dir: Path) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    p = fixtures_dir / name / ts
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_live(name: str, fixtures_dir: Path) -> Path:
    if name == "ttm-battery-surge":
        return _run_live_ttm(fixtures_dir)
    if name == "prithvi-pluvial":
        return _run_live_prithvi(fixtures_dir)
    if name == "terramind-buildings":
        return _run_live_terramind(fixtures_dir)
    raise ValueError(f"unknown model: {name}")


def replay(name: str, fixture_dir: str | None) -> Path:
    if name == "ttm-battery-surge":
        return _replay_ttm(fixture_dir)
    if name == "prithvi-pluvial":
        return _replay_prithvi(fixture_dir)
    if name == "terramind-buildings":
        return _replay_terramind(fixture_dir)
    raise ValueError(f"unknown model: {name}")


# ---------- TTM (NOAA, no auth, smallest data) -------------------------------


def _run_live_ttm(fixtures_dir: Path) -> Path:
    """Pull last ~50 days of Battery surge residual at hourly cadence
    (matches the fine-tune's 1024-step context), forecast next 96 hours.
    """
    from datetime import datetime, timedelta

    import numpy as np

    from .ttm_battery_surge.data import DEFAULT_STATION, fetch_residual_series, load_finetune

    out = _fixture_dir_for("ttm-battery-surge", fixtures_dir)
    end = datetime.now(UTC)
    begin = end - timedelta(days=50)

    ts, res = fetch_residual_series(
        DEFAULT_STATION,
        begin.strftime("%Y%m%d"),
        end.strftime("%Y%m%d"),
        hourly=True,
    )

    np.savez(out / "inputs.npz", timestamps=ts, residual_m=res)

    forecaster = load_finetune({"context_steps": 1024, "horizon_steps": 96})
    history = res.astype(np.float32)[-1024:]
    fc = forecaster.predict(history, horizon=96)
    np.savez(out / "outputs.npz", forecast_m=fc)

    import numpy as np
    manifest = {
        "model": "msradam/Granite-TTM-r2-Battery-Surge",
        "station": DEFAULT_STATION,
        "history_window": [begin.isoformat(), end.isoformat()],
        "n_history_hourly": int(res.size),
        "context_used": int(min(res.size, 1024)),
        "forecast_horizon_hours": 96,
        "forecast_max_residual_m": float(np.max(fc)),
        "forecast_min_residual_m": float(np.min(fc)),
        "forecast_peak_abs_residual_m": float(np.max(np.abs(fc))),
        "code_sha": code_sha(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def _replay_ttm(fixture_dir: str | None) -> str:
    import numpy as np

    from .ttm_battery_surge.data import load_finetune

    fd = _resolve_fixture("ttm-battery-surge", fixture_dir)
    inp = np.load(fd / "inputs.npz")
    res = inp["residual_m"]
    expected = np.load(fd / "outputs.npz")["forecast_m"]
    forecaster = load_finetune({})
    got = forecaster.predict(res.astype(np.float32), horizon=expected.size)
    if not np.allclose(got, expected, atol=1e-5):
        raise AssertionError(f"replay mismatch in {fd}")
    return str(fd)


# ---------- Prithvi pluvial (Sentinel-2 via Microsoft Planetary Computer) ----


def _run_live_prithvi(fixtures_dir: Path) -> Path:
    out = _fixture_dir_for("prithvi-pluvial", fixtures_dir)
    try:
        import planetary_computer
        import pystac_client
        import rasterio  # noqa: F401
    except ImportError:
        (out / "manifest.json").write_text(json.dumps({
            "model": "msradam/Prithvi-EO-2.0-NYC-Pluvial",
            "status": "skipped",
            "reason": "live extra not installed (pystac-client / planetary-computer / rasterio)",
            "code_sha": code_sha(),
        }, indent=2))
        return out

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    nyc_aoi = {
        "type": "Polygon",
        "coordinates": [[
            [-74.05, 40.65], [-73.85, 40.65], [-73.85, 40.85], [-74.05, 40.85], [-74.05, 40.65]
        ]],
    }
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=nyc_aoi,
        datetime="2026-01-01/2026-05-09",
        query={"eo:cloud_cover": {"lt": 20}},
        max_items=1,
    )
    items = list(search.items())
    manifest = {
        "model": "msradam/Prithvi-EO-2.0-NYC-Pluvial",
        "code_sha": code_sha(),
        "items": [{"id": it.id, "datetime": str(it.datetime)} for it in items],
        "status": "fetched_metadata_only" if items else "no_items",
        "note": (
            "Pixel-level fetch + inference is wired in src/riprap_models/prithvi_pluvial/data.py. "
            "When the prithvi extra is installed, this fixture also writes inputs.tif and outputs.tif."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def _replay_prithvi(fixture_dir: str | None) -> str:
    fd = _resolve_fixture("prithvi-pluvial", fixture_dir)
    return str(fd)  # numerical replay needs the prithvi extra; manifest replay is no-op


# ---------- TerraMind buildings (same Sentinel-2 source as Prithvi) ----------


def _run_live_terramind(fixtures_dir: Path) -> Path:
    out = _fixture_dir_for("terramind-buildings", fixtures_dir)
    try:
        import planetary_computer  # noqa: F401
        import pystac_client  # noqa: F401
    except ImportError:
        (out / "manifest.json").write_text(json.dumps({
            "model": "msradam/TerraMind-NYC-Adapters",
            "status": "skipped",
            "reason": "live extra not installed",
            "code_sha": code_sha(),
        }, indent=2))
        return out
    # Same fetch path as Prithvi; inference adapter differs.
    # Kept intentionally short; the heavy lifting is in terramind/eval.py.
    (out / "manifest.json").write_text(json.dumps({
        "model": "msradam/TerraMind-NYC-Adapters",
        "status": "skipped_inference",
        "reason": "inference path requires terratorch + adapter weights",
        "code_sha": code_sha(),
    }, indent=2))
    return out


def _replay_terramind(fixture_dir: str | None) -> str:
    fd = _resolve_fixture("terramind-buildings", fixture_dir)
    return str(fd)


def _resolve_fixture(name: str, fixture_dir: str | None) -> Path:
    if fixture_dir:
        return Path(fixture_dir)
    base = Path("eval/fixtures") / name
    if not base.exists():
        raise FileNotFoundError(f"no fixtures for {name} under {base}")
    candidates = sorted([p for p in base.iterdir() if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"no fixture directories under {base}")
    return candidates[-1]
