"""TerraMind NYC adapters: held-out evaluation and benchmark.

Three adapters live under ``msradam/TerraMind-NYC-Adapters``:

  * Buildings (binary segmentation: building / not building)
  * LULC      (Sentinel-2 land-cover, 9 classes)
  * TiM       (Tile-in-Mosaic context probe, regression)

The `Buildings` adapter is the smallest test set, so we evaluate it first
per the build order in WORKLOG.md. The other two follow the same pattern.

This module loads the adapter via terratorch when the ``terramind`` extra
is installed. When it isn't (e.g. base CI smoke run), the eval functions
write a status report flagging that the model wasn't evaluated and why.
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

    # Real eval path. Numbers are populated from the held-out tile loop.
    import numpy as np
    import torch  # noqa: F401

    from .data import load_buildings_adapter

    model, preprocess, num_classes = load_buildings_adapter()
    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    n_tiles = 0
    tile_ids: list[str] = []

    for tile_id, image, label in iter_holdout_tiles(cfg, limit=limit):
        with torch.no_grad():
            logits = model(preprocess(image).unsqueeze(0))
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        from ..common.metrics import confusion_matrix

        cm_total += confusion_matrix(pred, label, num_classes=num_classes, ignore_index=255)
        n_tiles += 1
        tile_ids.append(tile_id)

    from ..common.metrics import iou_from_confusion

    iou = iou_from_confusion(cm_total)
    miou = float(np.nanmean(list(iou.values())))

    prov = record(MODEL_ID, model_revision=cfg.get("model_revision"), inputs=[{"tile_id": t} for t in tile_ids])
    _write_measured_report(out, miou=miou, iou=iou, n_tiles=n_tiles, prov=prov.to_dict())
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

    import torch

    from .data import dummy_input, load_buildings_adapter

    model, preprocess, _ = load_buildings_adapter()
    x = preprocess(dummy_input()).unsqueeze(0)

    # Warm-up to exclude lazy compilation / first-call costs.
    with torch.no_grad():
        _ = model(x)

    durations = []
    joules = []
    method = "estimated"
    for _ in range(n_calls):
        with measure_energy() as m:
            with torch.no_grad():
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


def _write_measured_report(path: Path, miou: float, iou: dict, n_tiles: int, prov: dict) -> None:
    iou_lines = "\n".join(f"- class {c}: {v:.4f}" for c, v in sorted(iou.items()))
    body = (
        "# TerraMind Buildings\n\n"
        f"## Held-out evaluation\n\n"
        f"- tiles: {n_tiles}\n"
        f"- mIoU (macro): {miou:.4f}\n"
        f"- per-class IoU:\n{iou_lines}\n\n"
        "## Provenance\n\n"
        f"```json\n{json.dumps(prov, indent=2)}\n```\n\n"
        "```yaml measurements\n"
        "model: TerraMind Buildings\n"
        'card_metric: "0.5511 mIoU"\n'
        f'reproduced: "{miou:.4f} mIoU"\n'
        f'method: "held-out NYC tiles, n={n_tiles}"\n'
        'm3: "see docs/M3_NOTES.md"\n'
        'j_per_call: "see Benchmark section"\n'
        "```\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
