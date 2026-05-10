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

    rows: list[dict] = []
    sliding_rows: list[dict] = []

    sliding_cfg = cfg.get("sliding_windows")
    if sliding_cfg:
        _, big_series = fetch_residual_series(
            station, sliding_cfg["range_start"], sliding_cfg["range_end"], hourly=True
        )
        stride = int(sliding_cfg.get("stride_hours", 240))
        max_w = int(sliding_cfg.get("max_windows", 40))
        starts: list[int] = []
        s = 0
        while s + context + horizon <= big_series.size and len(starts) < max_w:
            starts.append(s)
            s += stride
        for i, s in enumerate(starts):
            history = big_series[s : s + context]
            target = big_series[s + context : s + context + horizon]
            ft_pred = finetune.predict(history, horizon=horizon)
            zs_pred = zeroshot.predict(history, horizon=horizon)
            pers = persistence_forecast(history, horizon=horizon)
            sliding_rows.append({
                "idx": i,
                "finetune_mae": regression_score(ft_pred[: target.size], target).mae,
                "zeroshot_mae": regression_score(zs_pred[: target.size], target).mae,
                "persistence_mae": regression_score(pers[: target.size], target).mae,
            })

    windows = cfg.get("holdout_windows", [])
    if limit is not None:
        windows = windows[:limit]

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
    named_ft = float(np.mean([r["finetune_mae"] for r in valid])) if valid else float("nan")
    named_zs = float(np.mean([r["zeroshot_mae"] for r in valid])) if valid else float("nan")
    named_pers = float(np.mean([r["persistence_mae"] for r in valid])) if valid else float("nan")

    if sliding_rows:
        sliding_ft = float(np.mean([r["finetune_mae"] for r in sliding_rows]))
        sliding_zs = float(np.mean([r["zeroshot_mae"] for r in sliding_rows]))
        sliding_pers = float(np.mean([r["persistence_mae"] for r in sliding_rows]))
        # Headline number = sliding-window mean (the larger N).
        overall_ft = sliding_ft
        overall_zs = sliding_zs
        overall_pers = sliding_pers
    else:
        sliding_ft = sliding_zs = sliding_pers = float("nan")
        overall_ft, overall_zs, overall_pers = named_ft, named_zs, named_pers

    prov = record(
        MODEL_ID, model_revision=cfg.get("model_revision"),
        inputs=[{"sliding_n": len(sliding_rows)}] + [{"window": r["window"]} for r in rows],
    )
    _write_measured_report(
        out, rows=rows, sliding_rows=sliding_rows,
        overall_ft=overall_ft, overall_zs=overall_zs, overall_pers=overall_pers,
        named_ft=named_ft, named_zs=named_zs, named_pers=named_pers,
        sliding_ft=sliding_ft, sliding_zs=sliding_zs, sliding_pers=sliding_pers,
        prov=prov.to_dict(),
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
    path: Path, rows: list[dict], sliding_rows: list[dict],
    overall_ft: float, overall_zs: float, overall_pers: float,
    named_ft: float, named_zs: float, named_pers: float,
    sliding_ft: float, sliding_zs: float, sliding_pers: float,
    prov: dict,
) -> None:
    parts: list[str] = ["# Granite TTM r2 Battery Surge\n\n"]

    if sliding_rows:
        parts.append(
            "## Sliding-window evaluation (post-training-cutoff range)\n\n"
            f"- windows: {len(sliding_rows)}\n"
            f"- fine-tune MAE: {sliding_ft:.4f} m\n"
            f"- zero-shot TTM r2 MAE: {sliding_zs:.4f} m\n"
            f"- persistence MAE: {sliding_pers:.4f} m\n"
            f"- fine-tune vs persistence: {(1 - sliding_ft/sliding_pers)*100:+.1f}% MAE reduction\n"
            f"- fine-tune vs zero-shot:   {(1 - sliding_ft/sliding_zs)*100:+.1f}% MAE reduction\n\n"
        )

    if rows:
        table = (
            "## Named-window breakdown\n\n"
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
        parts.append(table + "\n")

    parts.append(
        "## Provenance\n\n"
        f"```json\n{json.dumps(prov, indent=2)}\n```\n\n"
    )
    parts.append(
        "```yaml measurements\n"
        "model: Granite TTM r2 Battery Surge\n"
        'card_metric: "0.1091 m MAE"\n'
        f'reproduced: "{overall_ft:.4f} m MAE"\n'
        f'method: "NOAA 8518750 hourly, sliding n={len(sliding_rows)} + named n={len([r for r in rows if "note" not in r])}"\n'
        'm3: "yes (cpu fp32, ~3M params)"\n'
        'j_per_call: "see Benchmark section"\n'
        "```\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts))
