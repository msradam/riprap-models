"""Prithvi-EO 2.0 NYC Pluvial: held-out evaluation and benchmark.

Card metric (msradam/Prithvi-EO-2.0-NYC-Pluvial): test flood IoU 0.5979,
zero-shot baseline IoU around 0.10. We report both, and we deliberately
include three adversarial tile classes:

  * Ida 2021 footprint (the training-distribution case)
  * Sandy 2012 (older event, different optics)
  * non-event control (no flood, must report background)

The split is defined in eval/configs/prithvi_pluvial.yaml.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from ..common.metrics import segmentation_score
from ..common.provenance import record
from ..energy import measure_energy
from .data import iter_holdout_tiles

MODEL_ID = "msradam/Prithvi-EO-2.0-NYC-Pluvial"
DEFAULT_CONFIG = Path("eval/configs/prithvi_pluvial.yaml")


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
    out = reports_dir / "prithvi_pluvial.md"
    runtime, err = _try_import_runtime()

    if runtime is None:
        _write_skipped_report(out, reason=f"prithvi extra not installed ({err})")
        return out

    import numpy as np
    import torch

    from .data import load_pluvial_finetune

    model, preprocess, num_classes = load_pluvial_finetune(cfg)

    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    n_tiles = 0
    by_class: dict[str, list[float]] = {"ida": [], "sandy": [], "control": []}
    tile_ids: list[str] = []

    for tile_id, image, label, kind in iter_holdout_tiles(cfg, limit=limit):
        with torch.no_grad():
            logits = model(preprocess(image).unsqueeze(0))
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        s = segmentation_score(pred, label, num_classes=num_classes, ignore_index=255)
        # Flood class IoU is class 1 by Sen1Floods11 convention.
        flood_iou = s.iou_per_class.get(1, float("nan"))
        if kind in by_class and not (flood_iou != flood_iou):  # skip NaN
            by_class[kind].append(flood_iou)
        from ..common.metrics import confusion_matrix

        cm_total += confusion_matrix(pred, label, num_classes=num_classes, ignore_index=255)
        n_tiles += 1
        tile_ids.append(tile_id)

    from ..common.metrics import iou_from_confusion

    iou = iou_from_confusion(cm_total)
    flood_iou_overall = iou.get(1, float("nan"))

    prov = record(MODEL_ID, model_revision=cfg.get("model_revision"), inputs=[{"tile_id": t} for t in tile_ids])
    _write_measured_report(
        out,
        flood_iou=flood_iou_overall,
        per_class_iou=iou,
        by_kind=by_class,
        n_tiles=n_tiles,
        prov=prov.to_dict(),
    )
    return out


def run_bench(n_calls: int, reports_dir: Path) -> Path:
    out = reports_dir / "prithvi_pluvial.md"
    runtime, err = _try_import_runtime()
    if runtime is None:
        existing = out.read_text() if out.exists() else ""
        out.write_text(existing + f"\n\n## Benchmark skipped\n\n- reason: {err}\n")
        return out

    import torch

    from .data import dummy_input, load_pluvial_finetune

    model, preprocess, _ = load_pluvial_finetune({})
    x = preprocess(dummy_input()).unsqueeze(0)
    with torch.no_grad():
        _ = model(x)

    durations, joules = [], []
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
    existing = out.read_text() if out.exists() else "# Prithvi-EO 2.0 NYC Pluvial\n"
    out.write_text(existing + block)
    return out


def _write_skipped_report(path: Path, reason: str) -> None:
    body = (
        "# Prithvi-EO 2.0 NYC Pluvial\n\n"
        f"**Status:** not evaluated in this environment.\n\n"
        f"**Reason:** {reason}\n\n"
        f"Card metric (from `{MODEL_ID}` README): 0.5979 flood IoU on held-out NYC tiles, baseline IoU ~0.10.\n\n"
        "```yaml measurements\n"
        "model: Prithvi-EO 2.0 NYC Pluvial\n"
        'card_metric: "0.5979 flood IoU"\n'
        'reproduced: "not yet measured"\n'
        f'method: "skipped ({reason[:60]})"\n'
        'm3: "unknown"\n'
        'j_per_call: "not yet measured"\n'
        "```\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def _write_measured_report(
    path: Path, flood_iou: float, per_class_iou: dict, by_kind: dict, n_tiles: int, prov: dict
) -> None:
    by_kind_lines = "\n".join(
        f"- {k}: mean flood IoU = {sum(v)/len(v):.4f} (n={len(v)})" if v else f"- {k}: no tiles"
        for k, v in by_kind.items()
    )
    pc_lines = "\n".join(f"- class {c}: {v:.4f}" for c, v in sorted(per_class_iou.items()))
    body = (
        "# Prithvi-EO 2.0 NYC Pluvial\n\n"
        f"## Held-out evaluation\n\n"
        f"- tiles: {n_tiles}\n"
        f"- flood IoU (overall): {flood_iou:.4f}\n"
        f"- per-class IoU:\n{pc_lines}\n\n"
        "## By tile kind\n\n"
        f"{by_kind_lines}\n\n"
        "## Provenance\n\n"
        f"```json\n{json.dumps(prov, indent=2)}\n```\n\n"
        "```yaml measurements\n"
        "model: Prithvi-EO 2.0 NYC Pluvial\n"
        'card_metric: "0.5979 flood IoU"\n'
        f'reproduced: "{flood_iou:.4f} flood IoU"\n'
        f'method: "Ida + Sandy + control tiles, n={n_tiles}"\n'
        'm3: "see docs/M3_NOTES.md"\n'
        'j_per_call: "see Benchmark section"\n'
        "```\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
