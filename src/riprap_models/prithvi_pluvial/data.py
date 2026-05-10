"""Held-out tile loader for the Prithvi-EO 2.0 NYC pluvial fine-tune."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np


def iter_holdout_tiles(cfg: dict, limit: int | None = None) -> Iterator[tuple[str, np.ndarray, np.ndarray, str]]:
    """Yield ``(tile_id, image, label, kind)`` for each held-out tile.

    ``kind`` is one of "ida" / "sandy" / "control" so the eval can break
    down flood IoU by event type. The manifest format is one tile per line:

        tile_id,image_path,label_path,kind
    """
    try:
        import rasterio
    except ImportError:
        return

    manifest_path = cfg.get("manifest")
    if not manifest_path:
        return
    manifest = Path(manifest_path)
    if not manifest.exists():
        return

    n = 0
    with manifest.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            tile_id, image_path, label_path, kind = parts[0], parts[1], parts[2], parts[3]
            with rasterio.open(image_path) as src:
                image = src.read().transpose(1, 2, 0).astype(np.float32)
            with rasterio.open(label_path) as src:
                label = src.read(1).astype(np.int64)
            yield tile_id, image, label, kind
            n += 1
            if limit is not None and n >= limit:
                return


def load_pluvial_finetune():
    """Load the NYC pluvial fine-tune via terratorch.

    Returns ``(model, preprocess_fn, num_classes)``.
    """
    import terratorch  # noqa: F401
    import torch  # noqa: F401

    raise NotImplementedError(
        "Wire up the terratorch loader for msradam/Prithvi-EO-2.0-NYC-Pluvial. "
        "The Sen1Floods11 fine-tune recipe in riprap-nyc/scripts/run_prithvi_ida.py "
        "is the canonical reference for input shape and band order."
    )


def dummy_input(h: int = 512, w: int = 512, c: int = 6) -> np.ndarray:
    return np.zeros((h, w, c), dtype=np.float32)
