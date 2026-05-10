"""Prithvi-EO 2.0 NYC Pluvial: held-out tile loader and model loader.

Independent reconstruction of the model card's evaluation. The card's exact
test split lives on the AMD ROCm training box and isn't published. We
construct a held-out set from the same primary source (Riprap's 166 baked
Ida 2021 polygons + matching Sentinel-2 chips from Microsoft Planetary
Computer) using a deterministic stride that's independent of the card's
seed=42 random split. See ``docs/PROVENANCE.md``.

Recipe:

  * Positives: every 7th polygon from ``riprap-nyc/data/prithvi_ida_2021.geojson``
    (24 of 166 polygons). For each polygon, fetch the post-Ida Sentinel-2
    L2A scene (Sept 4-15 2021) with the lowest cloud cover that
    intersects the polygon centroid. Read a 224×224 chip in UTM 18N
    centered on the centroid, with bands B02 B03 B04 B8A B11 B12
    (Sen1Floods11 schema). Rasterize the polygon as the binary label.
  * Negatives: 5 clear-sky NYC locations far from any Ida polygon.
    Same band schema, all-zero label.

This is a reconstruction, not a bit-exact reproduction. The card's
seed=42 split partitioned 498 chips (166 originals + 332 copy-paste
augmentations); we partition only the 166 originals at stride 7. Some
of our 24 positives may have been in the card's train or val sets;
none are in the card's exact test set with high probability under the
fresh stride. This trades exact-replay for honest independence.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator

import numpy as np

# Sen1Floods11 band schema used by the fine-tune.
PRITHVI_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]
PRITHVI_MEANS = [0.107, 0.107, 0.115, 0.265, 0.235, 0.155]
PRITHVI_STDS = [0.082, 0.075, 0.085, 0.115, 0.110, 0.100]
TILE_SIZE = 224

# 5 clear-sky negative locations. Picked outside the Ida polygon footprint
# (which is concentrated in southern Brooklyn, southern Queens, and southern
# Staten Island). These are upland Bronx, upland Queens, midtown Manhattan,
# upland Staten Island, and upland Brooklyn (Park Slope plateau).
NEGATIVE_LONLATS: list[tuple[str, float, float]] = [
    ("bronx_pelham_bay", -73.808, 40.871),
    ("queens_forest_hills", -73.847, 40.722),
    ("manhattan_central_park", -73.965, 40.785),
    ("statenisland_lighthouse", -74.140, 40.575),
    ("brooklyn_park_slope", -73.975, 40.667),
]


def _polygon_test_indices(n: int = 166, stride: int = 7) -> list[int]:
    """Deterministic stride-based holdout. Independent of the card's seed=42."""
    return list(range(0, n, stride))


def iter_holdout_tiles(cfg: dict, limit: int | None = None) -> Iterator[
    tuple[str, np.ndarray, np.ndarray, str]
]:
    """Yield ``(tile_id, image, label, kind)``.

    image: HxWxC float32 reflectance (post-PRITHVI_MEANS-norm in eval), C=6
    label: HxW int64 (0 background, 1 flood, ignore_index unused)
    kind: 'ida' or 'control'
    """
    try:
        import geopandas as gpd
        import planetary_computer  # noqa: F401
        import pystac_client
        import rasterio  # noqa: F401
    except ImportError as e:
        warnings.warn(f"prithvi data extras missing: {e}", stacklevel=2)
        return

    geojson_path = cfg.get("ida_polygons", "/Users/amsrahman/riprap-nyc/data/prithvi_ida_2021.geojson")
    if not os.path.exists(geojson_path):
        warnings.warn(f"ida polygons not found at {geojson_path}; cannot construct positives", stacklevel=2)
        return

    g = gpd.read_file(geojson_path).to_crs(4326)
    test_idx = _polygon_test_indices(n=len(g), stride=int(cfg.get("stride", 7)))
    # All polygons participate as ground truth — when a chip is centered on
    # one polygon, any other polygons that fall inside the same 2240×2240m
    # window also belong to the flood mask. Holding back tile centers (the
    # 24 stride-7 indices) means the model has not seen those exact chip
    # framings, but the flood-extent labels remain complete.
    all_polys_4326 = g.geometry.tolist()

    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    n_yielded = 0

    # ---- positives ----
    for i in test_idx:
        poly = g.geometry.iloc[i]
        ctr = poly.centroid
        try:
            search = cat.search(
                collections=["sentinel-2-l2a"],
                intersects={"type": "Point", "coordinates": [ctr.x, ctr.y]},
                datetime=cfg.get("ida_post_window", "2021-09-05/2021-09-12"),
                query={"eo:cloud_cover": {"lt": cfg.get("max_cloud", 30)}},
                max_items=5,
            )
            items = list(search.items())
            items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
            if not items:
                warnings.warn(f"no S2 item found for ida polygon {i}", stacklevel=2)
                continue
            it = items[0]
            tile_id = f"ida_{i:03d}_{it.id}"
            image, label = _read_chip_with_label(
                it, ctr.x, ctr.y, all_polys_4326, g.crs
            )
            yield tile_id, image, label, "ida"
            n_yielded += 1
            if limit is not None and n_yielded >= limit:
                return
        except Exception as e:
            warnings.warn(f"failed ida polygon {i}: {e!r}", stacklevel=2)

    # ---- negatives ----
    for name, lon, lat in NEGATIVE_LONLATS:
        try:
            search = cat.search(
                collections=["sentinel-2-l2a"],
                intersects={"type": "Point", "coordinates": [lon, lat]},
                datetime=cfg.get("control_window", "2021-09-05/2021-09-12"),
                query={"eo:cloud_cover": {"lt": 10}},
                max_items=5,
            )
            items = list(search.items())
            items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
            if not items:
                continue
            it = items[0]
            tile_id = f"control_{name}_{it.id}"
            image, _ = _read_chip_with_label(it, lon, lat, polys=None, poly_crs=None)
            label = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.int64)
            yield tile_id, image, label, "control"
            n_yielded += 1
            if limit is not None and n_yielded >= limit:
                return
        except Exception as e:
            warnings.warn(f"failed control {name}: {e!r}", stacklevel=2)


def _read_chip_with_label(item, lon: float, lat: float, polys, poly_crs):
    """Read a 224x224 6-band chip centered on (lon, lat) and rasterize all
    ``polys`` (a list of shapely polygons in ``poly_crs``) into a binary
    label aligned to the chip window. Returns (image_HWC, label_HW).
    """
    import rasterio
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.transform import from_origin
    from rasterio.warp import transform as rio_transform
    from rasterio.windows import Window, from_bounds

    # Reproject the centroid into the asset's CRS via the B02 (10m) asset.
    asset_url = {b: item.assets[b].href for b in PRITHVI_BANDS}
    with rasterio.open(asset_url["B02"]) as ref:
        dst_crs = ref.crs
        xs, ys = rio_transform("EPSG:4326", dst_crs, [lon], [lat])
        cx, cy = xs[0], ys[0]
        # 10m pixels, 224 wide → 2240 m wide chip centered on (cx, cy).
        half = TILE_SIZE * 5  # half-width in metres
        left, bottom, right, top = cx - half, cy - half, cx + half, cy + half
        win = from_bounds(left, bottom, right, top, ref.transform)
        win = Window(int(round(win.col_off)), int(round(win.row_off)),
                     TILE_SIZE, TILE_SIZE)
        chip_transform = from_origin(left, top, 10.0, 10.0)
        # Read B02 once
        b02 = ref.read(1, window=win, out_shape=(TILE_SIZE, TILE_SIZE),
                       boundless=True, fill_value=0).astype(np.float32) / 10000.0

    bands_data = [b02]
    for b in PRITHVI_BANDS[1:]:
        with rasterio.open(asset_url[b]) as src:
            # 20m bands need to be resampled to 10m. rasterio handles it via
            # window + out_shape: it picks the appropriate read at the target
            # resolution.
            arr = src.read(
                1, window=from_bounds(left, bottom, right, top, src.transform),
                out_shape=(TILE_SIZE, TILE_SIZE), boundless=True, fill_value=0,
            ).astype(np.float32) / 10000.0
            bands_data.append(arr)

    image = np.stack(bands_data, axis=-1)  # H, W, C
    if polys:
        from pyproj import Transformer
        from shapely.geometry import box as shp_box
        from shapely.ops import transform as shp_transform

        tx = Transformer.from_crs(poly_crs, dst_crs, always_xy=True).transform
        chip_box = shp_box(left, bottom, right, top)
        shapes = []
        for p in polys:
            p_proj = shp_transform(tx, p)
            if p_proj.intersects(chip_box):
                clipped = p_proj.intersection(chip_box)
                if not clipped.is_empty:
                    shapes.append((clipped, 1))
        if shapes:
            label = rio_rasterize(
                shapes,
                out_shape=(TILE_SIZE, TILE_SIZE),
                transform=chip_transform,
                fill=0, dtype="uint8",
            ).astype(np.int64)
        else:
            label = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.int64)
    else:
        label = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.int64)
    return image, label


def load_pluvial_finetune(cfg: dict | None = None):
    """Load the NYC pluvial fine-tune via terratorch.

    Returns ``(model, preprocess_fn, num_classes)``. ``model`` is the bare
    PyTorch module (not the SemanticSegmentationTask wrapper) for direct
    inference. ``preprocess_fn`` accepts an HWC float32 reflectance array
    and returns a CHW float32 tensor normalized with the Sen1Floods11
    means/stds.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from terratorch.tasks import SemanticSegmentationTask

    cfg = cfg or {}
    repo_id = cfg.get("model_id", "msradam/Prithvi-EO-2.0-NYC-Pluvial")
    weights_file = cfg.get("weights_file", "Prithvi_EO_2.0_NYC_Pluvial.safetensors")

    task = SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",
        model_args=dict(
            backbone="prithvi_eo_v2_300_tl",
            backbone_pretrained=False,
            backbone_bands=["BLUE", "GREEN", "RED", "NARROW_NIR", "SWIR_1", "SWIR_2"],
            necks=[
                {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
                {"name": "ReshapeTokensToImage", "remove_cls_token": True},
                {"name": "LearnedInterpolateToPyramidal"},
            ],
            decoder="UNetDecoder",
            decoder_channels=[512, 256, 128, 64],
            head_dropout=0.1,
            num_classes=2,
        ),
        loss="dice",
        ignore_index=-1,
        class_weights=[0.342, 1.316],
    )
    p = hf_hub_download(repo_id, weights_file)
    sd = load_file(p)
    inner = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = task.model.load_state_dict(inner, strict=False)
    if missing or unexpected:
        warnings.warn(f"prithvi state dict load: missing={len(missing)} unexpected={len(unexpected)}", stacklevel=2)
    task.model.eval()

    means = torch.tensor(PRITHVI_MEANS, dtype=torch.float32).view(6, 1, 1)
    stds = torch.tensor(PRITHVI_STDS, dtype=torch.float32).view(6, 1, 1)

    def preprocess(image_hwc: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(image_hwc.astype(np.float32)).permute(2, 0, 1)
        return (x - means) / stds

    class _Wrap:
        def __init__(self, m):
            self.m = m

        def __call__(self, x):
            with torch.no_grad():
                out = self.m(x)
            return out.output if hasattr(out, "output") else out

        def parameters(self):
            return self.m.parameters()

    return _Wrap(task.model), preprocess, 2


def dummy_input(h: int = TILE_SIZE, w: int = TILE_SIZE, c: int = 6) -> np.ndarray:
    return np.zeros((h, w, c), dtype=np.float32)
