"""TerraMind NYC LULC adapter: held-out evaluation.

Card metric (msradam/TerraMind-NYC-Adapters/lulc_nyc): test mIoU 0.5866
with per-class IoU [0.949, 0.780, 0.770, 0.389, 0.045] across 5 classes
(Impervious, Vegetation, Water, Bare/Cropland, Building).

We construct an independent NYC test set: same 6 AOIs as the buildings
adapter, with labels from ESA WorldCover 2021 (NYC 5-class collapse) +
DOITT building footprints as class 4.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from ..common.metrics import confusion_matrix, iou_from_confusion
from ..common.provenance import record
from ..energy import measure_energy
from .data import iter_lulc_holdout_tiles, load_terramind_adapter

MODEL_ID = "msradam/TerraMind-NYC-Adapters"
ADAPTER = "lulc_nyc"
DEFAULT_CONFIG = Path("eval/configs/terramind_lulc.yaml")
# Card per-class IoU re-keyed under the model's actual class order
# (water=0, impervious=1, vegetation=2, bare/cropland=3, building=4).
# Card-published per-class IoUs in the model card are listed under
# names; we re-key by index to match how this harness scores.
CARD_PER_CLASS = {0: 0.7696, 1: 0.9494, 2: 0.7803, 3: 0.3892, 4: 0.0447}
CARD_MIOU = 0.5866


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
    cfg.setdefault("adapter_dir", ADAPTER)
    cfg.setdefault("num_classes", 5)
    out = reports_dir / "terramind_lulc.md"
    runtime, err = _try_import_runtime()
    if runtime is None:
        out.write_text(f"# TerraMind LULC\n\n**Skipped**: {err}\n")
        return out

    import numpy as np

    model, preprocess, num_classes = load_terramind_adapter(cfg)
    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    n_tiles = 0
    tile_ids: list[str] = []
    per_tile: list[dict] = []

    for tile_id, inputs, label, _kind in iter_lulc_holdout_tiles(cfg, limit=limit):
        x = preprocess(inputs)
        logits = model(x)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        cm = confusion_matrix(pred, label, num_classes=num_classes, ignore_index=255)
        cm_total += cm
        per_tile.append({
            "tile_id": tile_id,
            "n_pixels_per_class": [int((label == c).sum()) for c in range(num_classes)],
            "iou_per_class": iou_from_confusion(cm),
        })
        n_tiles += 1
        tile_ids.append(tile_id)

    iou = iou_from_confusion(cm_total)
    miou = float(np.nanmean([v for v in iou.values() if not np.isnan(v)]))

    prov = record(MODEL_ID, model_revision=cfg.get("model_revision"),
                  inputs=[{"tile_id": t} for t in tile_ids])
    _write_report(out, miou=miou, iou=iou, per_tile=per_tile, n_tiles=n_tiles,
                  prov=prov.to_dict())
    return out


def run_bench(n_calls: int, reports_dir: Path) -> Path:
    out = reports_dir / "terramind_lulc.md"
    runtime, err = _try_import_runtime()
    if runtime is None:
        existing = out.read_text() if out.exists() else ""
        out.write_text(existing + f"\n\n## Benchmark skipped\n\n- reason: {err}\n")
        return out

    from .data import dummy_input

    model, preprocess, _ = load_terramind_adapter({"adapter_dir": ADAPTER, "num_classes": 5})
    x = preprocess(dummy_input())
    _ = model(x)
    durations, joules = [], []
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
    existing = out.read_text() if out.exists() else "# TerraMind LULC\n"
    j_str = f'"{avg_j:.2f} J ({method}, {avg_d*1000:.0f} ms)"'
    existing = existing.replace('j_per_call: "see Benchmark section"', f"j_per_call: {j_str}")
    out.write_text(existing + block)
    return out


def _write_report(path: Path, miou: float, iou: dict, per_tile: list[dict],
                  n_tiles: int, prov: dict) -> None:
    iou_table = "| class | name | reproduced IoU | card IoU | Δ |\n|---|---|---:|---:|---:|\n"
    NAMES = {0: "water", 1: "impervious", 2: "vegetation", 3: "bare/cropland", 4: "building"}
    for c in sorted(iou):
        my = iou[c]
        card = CARD_PER_CLASS.get(c, float("nan"))
        delta = my - card
        iou_table += f"| {c} | {NAMES.get(c, '?')} | {my:.4f} | {card:.4f} | {delta:+.4f} |\n"

    per_tile_table = "| tile | water | impervious | veg | bare | building |\n|---|---:|---:|---:|---:|---:|\n"
    for r in per_tile:
        cols = []
        for c in range(5):
            v = r["iou_per_class"].get(c, float("nan"))
            cols.append(f"{v:.3f}" if not (v != v) else "nan")
        per_tile_table += f"| `{r['tile_id']}` | " + " | ".join(cols) + " |\n"

    body = (
        "# TerraMind LULC\n\n"
        "## Independent reconstruction\n\n"
        "Construction: same 6 NYC AOIs as the buildings adapter (multi-modal\n"
        "S2L2A 12 bands × 4 timesteps + S1RTC 2 bands × 4 timesteps + Copernicus\n"
        "DEM GLO-30, all from MS Planetary Computer). Labels: ESA WorldCover 2021\n"
        "v200 (collapsed to NYC 5-class scheme, see `data.py:ESA_TO_NYC5`) with NYC\n"
        "DOITT building footprints overwritten on top as class 4.\n\n"
        f"## Held-out evaluation (n={n_tiles})\n\n"
        f"- mIoU (macro): {miou:.4f}\n"
        f"- card mIoU: {CARD_MIOU:.4f}\n"
        f"- Δ vs card: {miou - CARD_MIOU:+.4f}\n\n"
        f"### Per-class IoU\n\n{iou_table}\n"
        f"### Per-tile detail\n\n{per_tile_table}\n"
        "## Provenance\n\n"
        f"```json\n{json.dumps(prov, indent=2)}\n```\n\n"
        "```yaml measurements\n"
        "model: TerraMind LULC\n"
        f'card_metric: "{CARD_MIOU:.4f} mIoU"\n'
        f'reproduced: "{miou:.4f} mIoU (5-class, ESA WorldCover + DOITT labels)"\n'
        f'method: "6 NYC AOIs, S2L2A+S1RTC+DEM 4 timesteps, ESA WorldCover labels"\n'
        'm3: "yes (cpu fp32, ~168M params multi-modal)"\n'
        'j_per_call: "see Benchmark section"\n'
        "```\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
