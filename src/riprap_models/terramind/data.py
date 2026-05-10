"""TerraMind NYC adapters: tile loader + model loader.

6 NYC AOIs, 224×224 chips at 10 m in UTM 18N. Inputs: Sentinel-2 L2A
(12 bands × 4 timesteps), Sentinel-1 RTC (2 bands × 4 timesteps),
Copernicus DEM GLO-30 (1 band × 4). Labels: NYC DOITT footprints
(buildings adapter) or ESA WorldCover + DOITT overlay (LULC adapter).
All sources public, no auth.
"""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Iterator

import numpy as np
import requests

S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
S1_BANDS = ["vv", "vh"]
TILE_SIZE = 224
N_TIMESTEPS = 4

# TerraMind 1.0 base pretraining input statistics. From the IBM reference
# config at ibm-esa-geospatial/TerraMind-base-Flood/terramind_v1_base_impactmesh_flood.yaml.
# S2L2A is on the raw 0-10000 reflectance scale (do NOT divide). S1RTC is
# in dB (10 * log10(linear)). DEM is in metres above geoid.
S2_MEANS = [1390.458, 1503.317, 1718.197, 1853.91, 2199.1, 2779.975,
            2987.011, 3083.234, 3132.22, 3162.988, 2424.884, 1857.648]
S2_STDS = [2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926,
           1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311]
S1_MEANS = [-10.93, -17.329]
S1_STDS = [4.391, 4.459]
DEM_MEAN = 670.665
DEM_STD = 951.272

# Chip centres in lon/lat. Each chip is 2240 × 2240 m at 10 m.
NYC_AOIS: list[tuple[str, float, float]] = [
    ("manhattan_midtown", -73.984, 40.755),
    ("brooklyn_downtown", -73.989, 40.692),
    ("queens_jamaica", -73.794, 40.702),
    ("bronx_morrisania", -73.911, 40.829),
    ("statenisland_stgeorge", -74.075, 40.643),
    ("manhattan_lower_waterfront", -74.014, 40.706),
]

NYC_BUILDINGS_API = "https://data.cityofnewyork.us/resource/5zhs-2jue.json"


def iter_holdout_tiles(cfg: dict, limit: int | None = None) -> Iterator[
    tuple[str, dict, np.ndarray, str]
]:
    """Yield ``(tile_id, inputs, label, kind)`` for each held-out NYC AOI.

    inputs: dict with keys ``S2L2A``, ``S1RTC``, ``DEM`` (CHWxT-shaped
    numpy arrays in raw reflectance / backscatter / metres units; the
    eval module normalizes and converts to torch).
    label: HxW int64 (0 not-building, 1 building).
    kind: 'nyc'.
    """
    try:
        import planetary_computer  # noqa: F401
        import pystac_client
        import rasterio  # noqa: F401
    except ImportError as e:
        warnings.warn(f"terramind data extras missing: {e}", stacklevel=2)
        return

    aois = cfg.get("aois", NYC_AOIS)
    if limit is not None:
        aois = aois[:limit]

    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=__import__("planetary_computer").sign_inplace,
    )

    s2_window = cfg.get("s2_window", "2024-04-01/2024-09-30")
    s1_window = cfg.get("s1_window", "2024-04-01/2024-09-30")
    max_cloud = cfg.get("max_cloud", 20)

    for name, lon, lat in aois:
        try:
            inputs, label = _build_chip(cat, lon, lat, s2_window, s1_window, max_cloud)
        except Exception as e:
            warnings.warn(f"failed AOI {name}: {e!r}", stacklevel=2)
            continue
        if inputs is None:
            continue
        yield f"nyc_{name}", inputs, label, "nyc"


def _build_chip(cat, lon: float, lat: float, s2_window: str, s1_window: str, max_cloud: int):
    import planetary_computer
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform as rio_transform

    # ---- pick 4 cloud-free S2 scenes spread across the window
    s2_search = cat.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=s2_window,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=80,
    )
    s2_items = sorted(s2_search.items(), key=lambda it: it.datetime)
    if len(s2_items) < N_TIMESTEPS:
        return None, None
    # Evenly spaced sample of N_TIMESTEPS dates from the cloud-free list.
    step = max(1, len(s2_items) // N_TIMESTEPS)
    s2_picked = [s2_items[i * step] for i in range(N_TIMESTEPS)]

    # ---- chip framing: force UTM 18N (EPSG:32618), which is correct for
    # all of NYC. Reads from rasters in any CRS get reprojected via the
    # WarpedVRT-style ``vrt_options`` argument to rasterio.open with
    # ``CRS`` and ``transform`` overrides.
    from rasterio.vrt import WarpedVRT

    dst_crs = "EPSG:32618"
    xs, ys = rio_transform("EPSG:4326", dst_crs, [lon], [lat])
    cx, cy = xs[0], ys[0]
    half = TILE_SIZE * 5
    left, bottom, right, top = cx - half, cy - half, cx + half, cy + half
    chip_transform = from_origin(left, top, 10.0, 10.0)

    def _read_to_chip(href: str) -> np.ndarray:
        with rasterio.open(href) as src:
            with WarpedVRT(
                src,
                crs=dst_crs,
                transform=chip_transform,
                width=TILE_SIZE,
                height=TILE_SIZE,
                resampling=rasterio.enums.Resampling.bilinear,
            ) as vrt:
                return vrt.read(1)

    # ---- read S2L2A bands across timesteps. Returns [12, T, 224, 224].
    s2_stack = np.zeros((12, N_TIMESTEPS, TILE_SIZE, TILE_SIZE), dtype=np.float32)
    for t, item in enumerate(s2_picked):
        signed = planetary_computer.sign(item)
        for bi, b in enumerate(S2_BANDS):
            asset = signed.assets.get(b)
            if asset is None:
                continue
            try:
                # Keep raw 0-10000 scale: TerraMind pretraining stats are on
                # this scale, not on the [0, 1] reflectance scale.
                s2_stack[bi, t] = _read_to_chip(asset.href).astype(np.float32)
            except Exception as e:
                warnings.warn(f"S2 band {b} read failed: {e!r}", stacklevel=2)

    # ---- pick 4 S1 RTC scenes near the S2 dates
    s1_picked = []
    for s2_item in s2_picked:
        s2_date = s2_item.datetime
        s1_search = cat.search(
            collections=["sentinel-1-rtc"],
            intersects={"type": "Point", "coordinates": [lon, lat]},
            datetime=(
                f"{(s2_date.replace(day=1)).date().isoformat()}/"
                f"{s2_window.split('/')[1]}"
            ),
            max_items=10,
        )
        opts = sorted(s1_search.items(), key=lambda it: abs((it.datetime - s2_date).total_seconds()))
        s1_picked.append(opts[0] if opts else None)
    s1_stack = np.zeros((2, N_TIMESTEPS, TILE_SIZE, TILE_SIZE), dtype=np.float32)
    for t, item in enumerate(s1_picked):
        if item is None:
            continue
        signed = planetary_computer.sign(item)
        for bi, b in enumerate(S1_BANDS):
            asset = signed.assets.get(b)
            if asset is None:
                continue
            # PC's sentinel-1-rtc is published in linear backscatter; convert
            # to dB to match TerraMind pretraining stats (S1_MEANS in dB).
            lin = _read_to_chip(asset.href).astype(np.float32)
            lin = np.clip(lin, 1e-6, None)
            s1_stack[bi, t] = 10.0 * np.log10(lin)

    # ---- DEM (single timestep, replicated to T)
    dem_search = cat.search(
        collections=["cop-dem-glo-30"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        max_items=1,
    )
    dem_items = list(dem_search.items())
    dem_stack = np.zeros((1, N_TIMESTEPS, TILE_SIZE, TILE_SIZE), dtype=np.float32)
    if dem_items:
        signed = planetary_computer.sign(dem_items[0])
        asset = signed.assets.get("data")
        if asset is not None:
            arr = _read_to_chip(asset.href).astype(np.float32)
            for t in range(N_TIMESTEPS):
                dem_stack[0, t] = arr

    # ---- DOITT building label
    label = _fetch_doitt_label(left, bottom, right, top, dst_crs, chip_transform)

    inputs = {"S2L2A": s2_stack, "S1RTC": s1_stack, "DEM": dem_stack}
    return inputs, label


def _fetch_doitt_label(left, bottom, right, top, dst_crs, chip_transform):
    """Pull NYC DOITT building footprints inside the chip bbox via Socrata,
    transform from EPSG:4326 to ``dst_crs``, rasterize to 224×224.
    """
    import rasterio  # noqa: F401
    from pyproj import Transformer
    from rasterio.features import rasterize as rio_rasterize
    from shapely.geometry import shape as shp_shape
    from shapely.ops import transform as shp_transform

    # Reproject the chip bbox back to lon/lat for the Socrata where-clause.
    tx_back = Transformer.from_crs(dst_crs, "EPSG:4326", always_xy=True).transform
    ll_x_min, ll_y_min = tx_back(left, bottom)
    ll_x_max, ll_y_max = tx_back(right, top)

    where = (
        f"within_box(the_geom, {ll_y_max}, {ll_x_min}, {ll_y_min}, {ll_x_max})"
    )
    params = {"$where": where, "$limit": 50000, "$select": "the_geom"}
    try:
        r = requests.get(NYC_BUILDINGS_API, params=params, timeout=60)
        r.raise_for_status()
        rows = r.json()
    except Exception as e:
        warnings.warn(f"DOITT fetch failed: {e!r}", stacklevel=2)
        return np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.int64)

    tx_fwd = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True).transform
    shapes = []
    for row in rows:
        g = row.get("the_geom")
        if not g:
            continue
        try:
            geom = shp_shape(g)
            geom_proj = shp_transform(tx_fwd, geom)
            shapes.append((geom_proj, 1))
        except Exception:
            continue
    if not shapes:
        return np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.int64)
    label = rio_rasterize(
        shapes,
        out_shape=(TILE_SIZE, TILE_SIZE),
        transform=chip_transform,
        fill=0, dtype="uint8",
    ).astype(np.int64)
    return label


def load_buildings_adapter(cfg: dict | None = None):
    """Backwards-compatible alias: load the buildings adapter (2 classes)."""
    cfg = dict(cfg) if cfg else {}
    cfg.setdefault("adapter_dir", "buildings_nyc")
    cfg.setdefault("num_classes", 2)
    return load_terramind_adapter(cfg)


def load_terramind_adapter(cfg: dict | None = None):
    """Load TerraMind 1.0 base + the requested NYC LoRA + decoder head.

    Returns ``(model_callable, preprocess_fn, num_classes)``. The model
    accepts a dict with ``S2L2A``, ``S1RTC``, ``DEM`` torch tensors of
    shape ``[1, C, T, H, W]``. The preprocess function converts numpy
    inputs to that dict.

    cfg keys:
      adapter_dir: one of buildings_nyc / lulc_nyc / tim_nyc
      num_classes: must match the adapter (2 buildings, 5 lulc, 5 tim)
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from terratorch.tasks import SemanticSegmentationTask

    cfg = cfg or {}
    base_id = cfg.get("base_id", "ibm-esa-geospatial/TerraMind-1.0-base")
    adapters_id = cfg.get("model_id", "msradam/TerraMind-NYC-Adapters")
    adapter_dir = cfg.get("adapter_dir", "buildings_nyc")
    num_classes = int(cfg.get("num_classes", 2))

    task = SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",
        model_args=dict(
            backbone="terramind_v1_base", backbone_pretrained=False,
            backbone_modalities=["S2L2A", "S1RTC", "DEM"],
            backbone_use_temporal=True, backbone_temporal_pooling="concat",
            backbone_temporal_n_timestamps=4,
            necks=[
                {"name": "SelectIndices", "indices": [2, 5, 8, 11]},
                {"name": "ReshapeTokensToImage", "remove_cls_token": False},
                {"name": "LearnedInterpolateToPyramidal"},
            ],
            decoder="UNetDecoder", decoder_channels=[512, 256, 128, 64],
            head_dropout=0.1, num_classes=num_classes,
        ),
        loss="ce", ignore_index=-1,
        class_weights=cfg.get("class_weights", [1.0] * num_classes),
    )
    base_p = hf_hub_download(base_id, "TerraMind_v1_base.pt")
    base_sd = torch.load(base_p, map_location="cpu", weights_only=False)
    task.model.encoder.load_state_dict({f"encoder.{k}": v for k, v in base_sd.items()}, strict=False)

    dec_p = hf_hub_download(adapters_id, f"{adapter_dir}/decoder_head.safetensors")
    task.model.load_state_dict(load_file(dec_p), strict=False)

    # Merge LoRA Δ into the base qkv/proj weights.
    lora_p = hf_hub_download(adapters_id, f"{adapter_dir}/adapter_model.safetensors")
    cfg_p = hf_hub_download(adapters_id, f"{adapter_dir}/adapter_config.json")
    with open(cfg_p) as f:
        ad_cfg = json.load(f)
    scale = ad_cfg["lora_alpha"] / ad_cfg["r"]

    lora = load_file(lora_p)
    sd = task.model.state_dict()
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in lora.items():
        m = re.match(r"(.+)\.lora_([AB])\.default\.weight$", k)
        if m:
            pairs.setdefault(m.group(1), {})[m.group(2)] = v
    for layer, parts in pairs.items():
        if "A" not in parts or "B" not in parts:
            continue
        full_key = layer + ".weight"
        if full_key not in sd:
            continue
        delta = scale * (parts["B"].float() @ parts["A"].float())
        sd[full_key] = sd[full_key] + delta.to(sd[full_key].dtype)
    task.model.load_state_dict(sd, strict=False)
    task.model.eval()

    s2_means_t = torch.tensor(S2_MEANS, dtype=torch.float32).view(12, 1, 1, 1)
    s2_stds_t = torch.tensor(S2_STDS, dtype=torch.float32).view(12, 1, 1, 1)
    s1_means_t = torch.tensor(S1_MEANS, dtype=torch.float32).view(2, 1, 1, 1)
    s1_stds_t = torch.tensor(S1_STDS, dtype=torch.float32).view(2, 1, 1, 1)

    def preprocess(inputs: dict) -> dict:
        # numpy [C, T, H, W] -> torch [1, C, T, H, W], with TerraMind
        # pretraining-stat normalization. The S2L2A input may carry 12
        # bands (the model expects 12) but we only normalize the first
        # 12; if the caller passed AOT/SCL as auxiliary they sit beyond
        # index 11 and this still indexes correctly.
        s2 = torch.from_numpy(inputs["S2L2A"]).float()[:12]
        s2 = (s2 - s2_means_t) / s2_stds_t
        s1 = torch.from_numpy(inputs["S1RTC"]).float()
        s1 = (s1 - s1_means_t) / s1_stds_t
        dem = torch.from_numpy(inputs["DEM"]).float()
        dem = (dem - DEM_MEAN) / DEM_STD
        return {
            "S2L2A": s2.unsqueeze(0),
            "S1RTC": s1.unsqueeze(0),
            "DEM": dem.unsqueeze(0),
        }

    class _Wrap:
        def __init__(self, m):
            self.m = m

        def __call__(self, inputs):
            with torch.no_grad():
                out = self.m(inputs)
            return out.output if hasattr(out, "output") else out

        def parameters(self):
            return self.m.parameters()

    return _Wrap(task.model), preprocess, num_classes


# ---------- LULC labels (ESA WorldCover 2021 + NYC DOITT building overlay) ----

# ESA WorldCover 2021 → NYC 5-class collapse. Class order matches what the
# LULC adapter actually predicts (recovered by permutation search against
# the loaded weights — the published model card doesn't number-list the
# classes explicitly, just names them):
#   0 Water                  ← ESA 80 (Permanent water), 90 (Wetland)
#   1 Impervious / urban     ← ESA 50 (Built-up) MINUS DOITT polygons
#   2 Vegetation             ← ESA 10 (Tree), 20 (Shrub), 30 (Grass)
#   3 Bare / cropland        ← ESA 40 (Crop), 60 (Bare), 70 (Snow), 95 (Mangrove), 100 (Moss)
#   4 Building (LULC scope)  ← NYC DOITT footprints OVERWRITE ESA built-up
ESA_TO_NYC5: dict[int, int] = {
    80: 0, 90: 0,
    50: 1,
    10: 2, 20: 2, 30: 2,
    40: 3, 60: 3, 70: 3, 95: 3, 100: 3,
    0: 3,  # nodata → bare-ish (rare in NYC)
}
LULC_CLASS_NAMES = {0: "water", 1: "impervious", 2: "vegetation",
                    3: "bare/cropland", 4: "building"}


def iter_lulc_holdout_tiles(cfg: dict, limit: int | None = None) -> Iterator[
    tuple[str, dict, np.ndarray, str]
]:
    """Same chip-fetching path as buildings, but with ESA WorldCover +
    DOITT building overlay as the 5-class label.
    """
    try:
        import planetary_computer  # noqa: F401
        import pystac_client
        import rasterio  # noqa: F401
    except ImportError as e:
        warnings.warn(f"terramind data extras missing: {e}", stacklevel=2)
        return

    aois = cfg.get("aois", NYC_AOIS)
    if limit is not None:
        aois = aois[:limit]
    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=__import__("planetary_computer").sign_inplace,
    )
    s2_window = cfg.get("s2_window", "2024-04-01/2024-09-30")
    s1_window = cfg.get("s1_window", "2024-04-01/2024-09-30")
    max_cloud = cfg.get("max_cloud", 20)

    for name, lon, lat in aois:
        try:
            inputs, _ = _build_chip(cat, lon, lat, s2_window, s1_window, max_cloud)
            if inputs is None:
                continue
            label = _build_lulc_label(cat, lon, lat)
            yield f"nyc_{name}", inputs, label, "nyc"
        except Exception as e:
            warnings.warn(f"failed AOI {name}: {e!r}", stacklevel=2)
            continue


def _build_lulc_label(cat, lon: float, lat: float) -> np.ndarray:
    """Fetch ESA WorldCover 2021 over the chip and rasterize NYC DOITT
    building polygons on top. Returns 224×224 int64.
    """
    import planetary_computer
    import rasterio
    from pyproj import Transformer
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform as rio_transform
    from shapely.geometry import shape as shp_shape
    from shapely.ops import transform as shp_transform

    dst_crs = "EPSG:32618"
    xs, ys = rio_transform("EPSG:4326", dst_crs, [lon], [lat])
    cx, cy = xs[0], ys[0]
    half = TILE_SIZE * 5
    left, top = cx - half, cy + half
    chip_transform = from_origin(left, top, 10.0, 10.0)

    # ESA WorldCover, 2021 v200 preferred
    search = cat.search(
        collections=["esa-worldcover"],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        max_items=2,
    )
    items = sorted(search.items(), key=lambda it: it.id, reverse=True)  # 2021 > 2020
    if not items:
        return np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.int64)
    item = planetary_computer.sign(items[0])
    asset = item.assets["map"]

    with rasterio.open(asset.href) as src:
        with WarpedVRT(
            src, crs=dst_crs, transform=chip_transform,
            width=TILE_SIZE, height=TILE_SIZE,
            resampling=rasterio.enums.Resampling.nearest,
        ) as vrt:
            esa = vrt.read(1)

    label = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.int64)
    for esa_val, nyc_cls in ESA_TO_NYC5.items():
        label[esa == esa_val] = nyc_cls

    # Overlay DOITT building footprints as class 4
    where = (
        f"within_box(the_geom, {lat + 0.025}, {lon - 0.025}, {lat - 0.025}, {lon + 0.025})"
    )
    params = {"$where": where, "$limit": 50000, "$select": "the_geom"}
    try:
        rows = requests.get(NYC_BUILDINGS_API, params=params, timeout=60).json()
    except Exception:
        rows = []
    if rows:
        tx_fwd = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True).transform
        shapes = []
        for row in rows:
            g = row.get("the_geom")
            if not g:
                continue
            try:
                shapes.append((shp_transform(tx_fwd, shp_shape(g)), 4))
            except Exception:
                continue
        if shapes:
            bld_mask = rio_rasterize(
                shapes, out_shape=(TILE_SIZE, TILE_SIZE),
                transform=chip_transform, fill=0, dtype="uint8",
            )
            label[bld_mask > 0] = 4
    return label


def dummy_input(h: int = TILE_SIZE, w: int = TILE_SIZE) -> dict:
    return {
        "S2L2A": np.zeros((12, N_TIMESTEPS, h, w), dtype=np.float32),
        "S1RTC": np.zeros((2, N_TIMESTEPS, h, w), dtype=np.float32),
        "DEM": np.zeros((1, N_TIMESTEPS, h, w), dtype=np.float32),
    }
