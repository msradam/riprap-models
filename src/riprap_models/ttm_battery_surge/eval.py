"""Granite TTM r2 Battery Surge: held-out evaluation and benchmark.

Card metric (msradam/Granite-TTM-r2-Battery-Surge): MAE 0.1091 m on the
held-out storm window. We also report:

  * persistence baseline MAE (last observed value held flat)
  * zero-shot TTM r2 MAE (no fine-tune; same model class)

Held-out windows are constructed in eval/configs/ttm_battery_surge.yaml as
explicit ISO date ranges over NOAA CO-OPS station 8518750 (The Battery).
The split is documented in docs/PROVENANCE.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from ..common.metrics import persistence_forecast, regression_score
from ..common.provenance import record
from ..energy import measure_energy
from .data import fetch_residual_series, load_finetune

MODEL_ID = "msradam/Granite-TTM-r2-Battery-Surge"
DEFAULT_CONFIG = Path("eval/configs/ttm_battery_surge.yaml")


def _load_config(path: str | None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def _try_import_runtime():
    try:
        import torch  # noqa: F401
        import tsfm_public  # type: ignore # noqa: F401
    except ImportError as e:
        return None, str(e)
    return True, None


def run_eval(config_path: str | None, limit: int | None, reports_dir: Path) -> Path:
    cfg = _load_config(config_path)
    out = reports_dir / "ttm_battery_surge.md"
    runtime, err = _try_import_runtime()

    if runtime is None:
        _write_skipped_report(out, reason=f"ttm extra not installed ({err})")
        return out

    import numpy as np

    finetune = load_finetune(cfg)
    # Zero-shot baseline: same TTM r2 architecture, no NYC fine-tune.
    zeroshot_cfg = dict(cfg)
    zeroshot_cfg["model_id"] = "ibm-granite/granite-timeseries-ttm-r2"
    zeroshot = load_finetune(zeroshot_cfg)

    context = int(cfg.get("context_steps", 1024))
    horizon = int(cfg.get("horizon_steps", 96))
    station = cfg.get("station", "8518750")
    windows = cfg.get("holdout_windows", [])
    if limit is not None:
        windows = windows[:limit]

    rows = []
    for w in windows:
        # Fetch one contiguous hourly series spanning history_start → target_end,
        # take the last (context + horizon) samples, split into history + target.
        _, series = fetch_residual_series(
            station, w["history_start"], w["target_end"], hourly=True
        )
        if series.size < context + horizon:
            rows.append({
                "window": w["label"],
                "finetune_mae": float("nan"),
                "zeroshot_mae": float("nan"),
                "persistence_mae": float("nan"),
                "n": int(series.size),
                "note": f"insufficient data ({series.size} < {context+horizon})",
            })
            continue
        slice_ = series[-(context + horizon) :]
        history = slice_[:context]
        target = slice_[context:]

        ft_pred = finetune.predict(history, horizon=horizon)
        zs_pred = zeroshot.predict(history, horizon=horizon)
        pers = persistence_forecast(history, horizon=horizon)
        ft = regression_score(ft_pred[: target.size], target)
        zs = regression_score(zs_pred[: target.size], target)
        ps = regression_score(pers[: target.size], target)
        rows.append({
            "window": w["label"],
            "finetune_mae": ft.mae,
            "zeroshot_mae": zs.mae,
            "persistence_mae": ps.mae,
            "n": ft.n,
        })

    valid = [r for r in rows if not np.isnan(r["finetune_mae"])]
    overall_ft = float(np.mean([r["finetune_mae"] for r in valid])) if valid else float("nan")
    overall_zs = float(np.mean([r["zeroshot_mae"] for r in valid])) if valid else float("nan")
    overall_pers = float(np.mean([r["persistence_mae"] for r in valid])) if valid else float("nan")

    prov = record(MODEL_ID, model_revision=cfg.get("model_revision"), inputs=[{"window": r["window"]} for r in rows])
    _write_measured_report(
        out, rows=rows, overall_ft=overall_ft, overall_zs=overall_zs,
        overall_pers=overall_pers, prov=prov.to_dict(),
    )
    return out


def run_bench(n_calls: int, reports_dir: Path) -> Path:
    out = reports_dir / "ttm_battery_surge.md"
    runtime, err = _try_import_runtime()
    if runtime is None:
        existing = out.read_text() if out.exists() else ""
        out.write_text(existing + f"\n\n## Benchmark skipped\n\n- reason: {err}\n")
        return out

    import numpy as np

    cfg = _load_config(None)
    finetune = load_finetune(cfg)
    context = int(cfg.get("context_steps", 1024))
    horizon = int(cfg.get("horizon_steps", 96))
    # Warm-up shaped like a real input. Random gaussian so the standardizer
    # has finite std and the model exercises its real code path (zeros would
    # divide by epsilon).
    rng = np.random.default_rng(0)
    history = rng.normal(0.0, 0.2, size=context).astype(np.float32)

    _ = finetune.predict(history, horizon=horizon)

    durations, joules = [], []
    method = "estimated"
    for _ in range(n_calls):
        with measure_energy() as m:
            _ = finetune.predict(history, horizon=horizon)
        durations.append(m.duration_s)
        joules.append(m.joules)
        method = m.method

    avg_d = sum(durations) / len(durations)
    avg_j = sum(joules) / len(joules)
    block = (
        "\n\n## Benchmark\n\n"
        f"- n_calls: {n_calls}\n"
        f"- avg_duration_s: {avg_d:.4f}\n"
        f"- avg_joules: {avg_j:.4f} ({method})\n"
    )
    existing = out.read_text() if out.exists() else "# Granite TTM r2 Battery Surge\n"
    # Interpolate the j_per_call value into the YAML measurements block
    # so the headline RESULTS.md table picks it up.
    j_str = f'"{avg_j:.4f} J ({method}, {avg_d*1000:.1f} ms)"'
    existing = existing.replace(
        'j_per_call: "see Benchmark section"',
        f"j_per_call: {j_str}",
    )
    out.write_text(existing + block)
    return out


def _write_skipped_report(path: Path, reason: str) -> None:
    body = (
        "# Granite TTM r2 Battery Surge\n\n"
        f"**Status:** not evaluated in this environment.\n\n"
        f"**Reason:** {reason}\n\n"
        f"Card metric (from `{MODEL_ID}` README): MAE 0.1091 m on held-out storm window.\n\n"
        "```yaml measurements\n"
        "model: Granite TTM r2 Battery Surge\n"
        'card_metric: "0.1091 m MAE"\n'
        'reproduced: "not yet measured"\n'
        f'method: "skipped ({reason[:60]})"\n'
        'm3: "unknown"\n'
        'j_per_call: "not yet measured"\n'
        "```\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def _write_measured_report(
    path: Path, rows: list[dict], overall_ft: float, overall_zs: float,
    overall_pers: float, prov: dict,
) -> None:
    table = (
        "| window | fine-tune MAE (m) | zero-shot TTM MAE (m) | persistence MAE (m) | n |\n"
        "|---|---:|---:|---:|---:|\n"
    )
    for r in rows:
        if "note" in r:
            table += f"| {r['window']} | n/a | n/a | n/a | {r['n']} ({r['note']}) |\n"
        else:
            table += (
                f"| {r['window']} | {r['finetune_mae']:.4f} | "
                f"{r['zeroshot_mae']:.4f} | {r['persistence_mae']:.4f} | {r['n']} |\n"
            )
    body = (
        "# Granite TTM r2 Battery Surge\n\n"
        f"## Held-out evaluation\n\n"
        f"- overall fine-tune MAE: {overall_ft:.4f} m\n"
        f"- overall zero-shot TTM r2 MAE: {overall_zs:.4f} m\n"
        f"- overall persistence MAE: {overall_pers:.4f} m\n\n"
        f"{table}\n"
        "## Provenance\n\n"
        f"```json\n{json.dumps(prov, indent=2)}\n```\n\n"
        "```yaml measurements\n"
        "model: Granite TTM r2 Battery Surge\n"
        'card_metric: "0.1091 m MAE"\n'
        f'reproduced: "{overall_ft:.4f} m MAE"\n'
        f'method: "NOAA 8518750 hourly holdout windows, n={len([r for r in rows if "note" not in r])}"\n'
        'm3: "yes (cpu fp32, ~3M params)"\n'
        'j_per_call: "see Benchmark section"\n'
        "```\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
