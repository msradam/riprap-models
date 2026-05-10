# Provenance

Every report under `eval/reports/` carries a JSON provenance block with
the exact tile IDs, station IDs, timestamps, and code SHA used to
produce it. This document explains where those identifiers come from
and how they were chosen.

## TerraMind Buildings (`terramind_buildings.md`)

Held-out tiles are Sentinel-2 L2A scenes from the
`microsoft/sentinel-2-l2a` STAC collection that:

- intersect the NYC five-borough polygon
- have `eo:cloud_cover < 20`
- are dated after the published fine-tune cutoff (2024-12-31, per the
  model card)

Tile IDs follow Sentinel-2's standard MGRS form (e.g.
`S2A_MSIL2A_20260204T155321_R011_T18TWL_20260204T201200`). The label
rasters are NYC Department of Information Technology and
Telecommunications building footprints, rasterized to 10 m at the same
MGRS grid.

## Prithvi-EO 2.0 NYC Pluvial (`prithvi_pluvial.md`)

Held-out tiles are split into three kinds and labelled in the manifest:

| kind     | source                                                                  |
|---|---|
| `ida`    | Hurricane Ida 2021 pre/post Sentinel-2 pair, Sen1Floods11-style label   |
| `sandy`  | Hurricane Sandy 2012 Landsat archive, ground-truth from FEMA HSDC       |
| `control`| Fair-weather scene, label = all-zero                                    |

The control class is what catches a model that has memorized "NYC means
flood." Prithvi-EO 2.0's NYC fine-tune should report flood IoU close to
zero on a non-event control tile.

## Granite TTM r2 Battery Surge (`ttm_battery_surge.md`)

NOAA CO-OPS station 8518750, "The Battery, NY". Three held-out windows:

| label                          | dates (GMT)              | character                       |
|---|---|---|
| `noreaster_2024_12_18`         | 2024-12-15 → 2024-12-19  | nor'easter, large positive surge |
| `calm_2024_09_15`              | 2024-09-12 → 2024-09-16  | fair weather, small residual     |
| `ida_remnants_2021_09_02`      | 2021-08-30 → 2021-09-03  | Hurricane Ida remnants           |

These windows are deliberately constructed to **not overlap** the
fine-tune training windows. The training windows are listed in the
model card under "Training data" (see the model card on Hugging Face);
the published cutoff is 2021-08-29 (Ida pre-impact) and after the
2024-12-19 nor'easter dissipated. The Ida remnants window therefore
falls in the train-time blackout, which is exactly the point: it is the
held-out story the model must get right.

## What "code_sha" in each report means

The SHA is `git rev-parse HEAD` at the time the report was generated.
A reviewer can `git checkout <sha>` and re-run the same eval to bisect
any drift between report runs. If you regenerate a report on a dirty
working tree, the SHA still reflects the last commit, so prefer to run
evals on a clean tree (the harness does not error on dirty trees,
because that would be over-strict for a research repo).

## Construction-bias caveat

If the published training split for a model is thin or undocumented in
its riprap-nyc training experiment, this repo constructs a fresh
held-out split as described above and reports under both labels. Any
such case is named explicitly in the relevant report's body, not just
in the YAML measurements block.
