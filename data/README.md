# `data/`

This directory holds **manifests** that point at public open data, plus the
small fixtures that the replay path depends on. We deliberately do not commit
Sentinel-2 GeoTIFFs or other rasters to git: they are large, they go stale,
and re-fetching them from a versioned source is the right way to reproduce.

## Sources

| Manifest                                  | Backing source                                                   |
|---|---|
| `manifests/terramind_buildings_holdout.csv` | Sentinel-2 L2A via Microsoft Planetary Computer (`sentinel-2-l2a` collection) |
| `manifests/prithvi_pluvial_holdout.csv`     | Sentinel-2 L2A + Sen1Floods11 label rasters                      |
| (no manifest)                                | NOAA CO-OPS station 8518750 surge residual, fetched live in `src/riprap_models/ttm_battery_surge/data.py` |

## Building the manifests

The build scripts live next to the model evals (TODO: ship as
`scripts/build_*_holdout.py`). Each script writes a CSV of tile IDs and
public URLs; the eval modules then stream rasters through `rasterio` /
`planetary_computer.sign_inplace`.

## Why no rasters in git

A reproduction harness should be small. Anyone with internet can re-pull
the rasters from Planetary Computer in minutes; nobody can pull them out
of a 5 GB git repo any faster. The provenance block in each report
records the exact tile IDs and timestamps used.
