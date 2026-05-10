"""Held-out NYC tile loader for the TerraMind Buildings adapter.

The loader reads tile IDs + label paths from the YAML config under
``eval/configs/terramind_buildings.yaml``. The config points at a public
manifest (S3 URL or local path); we do not commit Sentinel-2 GeoTIFFs to
the repo. See ``data/README.md`` for the source-data pointer.

When the optional ``rasterio`` import is missing we degrade gracefully and
yield nothing, so the eval module's "skipped" path stays clean.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np


def iter_holdout_tiles(cfg: dict, limit: int | None = None) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(tile_id, image, label)`` for each held-out tile.

    Image is HxWxC float32, label is HxW int with class indices and 255 for ignore.
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
            if len(parts) < 3:
                continue
            tile_id, image_path, label_path = parts[0], parts[1], parts[2]
            with rasterio.open(image_path) as src:
                image = src.read().transpose(1, 2, 0).astype(np.float32)
            with rasterio.open(label_path) as src:
                label = src.read(1).astype(np.int64)
            yield tile_id, image, label
            n += 1
            if limit is not None and n >= limit:
                return


def load_buildings_adapter():
    """Load the TerraMind Buildings adapter via terratorch.

    Returns ``(model, preprocess_fn, num_classes)``. The preprocess function
    accepts an HxWxC numpy array and returns a torch.Tensor in CHW form.
    """
    import terratorch  # noqa: F401

    # The exact terratorch loader signature for the published adapter lives
    # in the model card. We resolve it by name so a future revision (e.g.
    # the LULC or TiM adapters) can reuse this with one argument flipped.
    raise NotImplementedError(
        "Wire up the terratorch loader for msradam/TerraMind-NYC-Adapters "
        "(buildings head). See the model card for the canonical kwargs."
    )


def dummy_input(h: int = 224, w: int = 224, c: int = 12) -> np.ndarray:
    """A zero tensor shaped like a Sentinel-2 patch, for benchmark warm-ups."""
    return np.zeros((h, w, c), dtype=np.float32)
