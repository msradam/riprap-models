"""TerraMind buildings adapter: eval + bench.

Card metric: 0.5511 mIoU / 0.2928 building IoU. Loads via terratorch.
LULC adapter eval lives in ``eval_lulc.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from ..common.provenance import record
from ..energy import measure_energy
from .data import iter_holdout_tiles

MODEL_ID = "msradam/TerraMind-NYC-Adapters"
DEFAULT_CONFIG = Path("eval/configs/terramind_buildings.yaml")


def _load_config(path: str | None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def _try_import_runtime():
    try:
        import terratorch  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        return None, str(e)
    return True, None


def run_eval(config_path: str | None, limit: int | None, reports_dir: Path) -> Path:
    cfg = _load_config(config_path)
    out = reports_dir / "terramind_buildings.md"
    runtime, err = _try_import_runtime()

    if runtime is None:
        _write_skipped_report(
            out,
            reason=f"terramind extra not installed ({err}). install with: uv pip install -e \".[terramind]\"",
            cfg=cfg,
        )
        return out

    import numpy as np

    from ..common.metrics import confusion_matrix, iou_from_confusion
    from .data import load_buildings_adapter

    model, preprocess, num_classes = load_buildings_adapter(cfg)

    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    n_tiles = 0
    tile_ids: list[str] = []
    per_tile: list[dict] = []

    for tile_id, inputs, label, _kind in iter_holdout_tiles(cfg, limit=limit):
        x = preprocess(inputs)
        logits = model(x)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        cm_total += confusion_matrix(pred, label, num_classes=num_classes, ignore_index=255)
        tp = int(((pred == 1) & (label == 1)).sum())
        fp = int(((pred == 1) & (label == 0)).sum())
        fn = int(((pred == 0) & (label == 1)).sum())
        bld_iou = tp / max(1, tp + fp + fn)
        per_tile.append({
            "tile_id": tile_id,
            "gt_pix": int((label == 1).sum()),
            "pred_pix": int((pred == 1).sum()),
            "bld_iou": bld_iou,
        })
        n_tiles += 1
        tile_ids.append(tile_id)

    iou = iou_from_confusion(cm_total)
    miou_macro = float(np.nanmean(list(iou.values())))
    bld_iou_micro = iou.get(1, float("nan"))
    nonbld_iou = iou.get(0, float("nan"))

    prov = record(MODEL_ID, model_revision=cfg.get("model_revision"),
                  inputs=[{"tile_id": t} for t in tile_ids])
    _write_measured_report(
        out, miou=miou_macro, bld_iou=bld_iou_micro, nonbld_iou=nonbld_iou,
        per_tile=per_tile, n_tiles=n_tiles, prov=prov.to_dict(),
    )
    return out


def run_bench(n_calls: int, reports_dir: Path) -> Path:
    out = reports_dir / "terramind_buildings.md"
    runtime, err = _try_import_runtime()
    if runtime is None:
        # Append-only: keep whatever the last eval wrote and add a benchmark
        # note so the missing extras are obvious.
        existing = out.read_text() if out.exists() else ""
        out.write_text(existing + f"\n\n## Benchmark skipped\n\n- reason: {err}\n")
        return out

    from .data import dummy_input, load_buildings_adapter

    model, preprocess, _ = load_buildings_adapter({})
    x = preprocess(dummy_input())
    _ = model(x)  # warm-up

    durations = []
    joules = []
    method = "estimated"
    for _ in range(n_calls):
        with measure_energy() as m:
            _ = model(x)
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
    existing = out.read_text() if out.exists() else "# TerraMind Buildings\n"
    j_str = f'"{avg_j:.2f} J ({method}, {avg_d*1000:.0f} ms)"'
    existing = existing.replace(
        'j_per_call: "see Benchmark section"', f"j_per_call: {j_str}"
    )
    out.write_text(existing + block)
    return out


def _write_skipped_report(path: Path, reason: str, cfg: dict) -> None:
    body = (
        "# TerraMind Buildings\n\n"
        f"**Status:** not evaluated in this environment.\n\n"
        f"**Reason:** {reason}\n\n"
        f"Card metric (from `{MODEL_ID}` README): 0.5511 mIoU on held-out NYC tiles.\n\n"
        "```yaml measurements\n"
        "model: TerraMind Buildings\n"
        'card_metric: "0.5511 mIoU"\n'
        'reproduced: "not yet measured"\n'
        f'method: "skipped ({reason[:60]})"\n'
        'm3: "unknown"\n'
        'j_per_call: "not yet measured"\n'
        "```\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def _write_measured_report(
    path: Path, miou: float, bld_iou: float, nonbld_iou: float,
    per_tile: list[dict], n_tiles: int, prov: dict,
) -> None:
    table = "| tile | gt_pix | pred_pix | building IoU |\n|---|---:|---:|---:|\n"
    for r in per_tile:
        table += f"| `{r['tile_id']}` | {r['gt_pix']} | {r['pred_pix']} | {r['bld_iou']:.4f} |\n"
    body = (
        "# TerraMind Buildings\n\n"
        "## Independent reconstruction\n\n"
        "Construction: 6 NYC AOIs (Manhattan midtown, Brooklyn downtown, Queens Jamaica,\n"
        "Bronx Morrisania, Staten Island St. George, Manhattan Lower waterfront), each\n"
        "a 224×224 chip at 10 m / pixel. Multi-modal input: Sentinel-2 L2A 12 bands ×\n"
        "4 timesteps + Sentinel-1 RTC 2 bands × 4 timesteps + Copernicus DEM GLO-30,\n"
        "all from Microsoft Planetary Computer. Labels: NYC DOITT building footprints\n"
        "(`5zhs-2jue` on NYC OpenData) fetched per chip via Socrata REST and\n"
        "rasterized to the chip grid.\n\n"
        f"## Held-out evaluation (n={n_tiles})\n\n"
        f"- mIoU (macro): {miou:.4f}\n"
        f"- building IoU (micro): {bld_iou:.4f}\n"
        f"- non-building IoU (micro): {nonbld_iou:.4f}\n\n"
        "## Per-tile detail\n\n"
        + table + "\n"
        "## Provenance\n\n"
        f"```json\n{json.dumps(prov, indent=2)}\n```\n\n"
        "```yaml measurements\n"
        "model: TerraMind Buildings\n"
        'card_metric: "0.5511 mIoU"\n'
        f'reproduced: "{miou:.4f} mIoU"\n'
        f'method: "6 NYC AOIs, S2L2A+S1RTC+DEM 4 timesteps, DOITT labels"\n'
        'm3: "yes (cpu fp32, ~168M params multi-modal)"\n'
        'j_per_call: "see Benchmark section"\n'
        "```\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
