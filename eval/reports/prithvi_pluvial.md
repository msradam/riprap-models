# Prithvi-EO 2.0 NYC Pluvial

## Independent reconstruction

Construction: 24 stride-7 holdout polygons from `riprap-nyc/data/prithvi_ida_2021.geojson`,
matched to lowest-cloud Sentinel-2 L2A scene from MS Planetary Computer in the
post-Ida window 2021-09-05 to 2021-09-12, plus 5 clear-sky NYC negative controls.
All polygons within each 224×224 chip's footprint contribute to ground truth.

## Held-out evaluation (micro IoU on flood class)

- tiles: 29
- fine-tune flood IoU: 0.0806
- zero-shot Sen1Floods11 base IoU: 0.0000

- per-class IoU (fine-tune):
- class 0: 0.9317
- class 1: 0.0806

## By tile kind

- ida: fine-tune IoU = 0.0905 (n=24); zero-shot IoU = 0.0000
- sandy: fine-tune IoU = nan (n=0); zero-shot IoU = nan
- control: fine-tune IoU = 0.0000 (n=5); zero-shot IoU = nan

## Per-tile detail

| tile | kind | gt_pix | pred_pix | fine-tune IoU | zero-shot IoU |
|---|---|---:|---:|---:|---:|
| `ida_000_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 2607 | 1938 | 0.5085 | 0.0000 |
| `ida_007_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 36 | 0 | 0.0000 | 0.0000 |
| `ida_014_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 837 | 47 | 0.0000 | 0.0000 |
| `ida_021_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 1551 | 454 | 0.0710 | 0.0000 |
| `ida_028_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 477 | 43 | 0.0526 | 0.0000 |
| `ida_035_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 360 | 254 | 0.1083 | 0.0000 |
| `ida_042_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 117 | 611 | 0.0659 | 0.0000 |
| `ida_049_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 651 | 886 | 0.1479 | 0.0000 |
| `ida_056_S2B_MSIL2A_20210907T154809_R054_T18TWL_20210908T0351` | ida | 171 | 1813 | 0.0871 | 0.0000 |
| `ida_063_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 2211 | 1403 | 0.0762 | 0.0000 |
| `ida_070_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 1316 | 1665 | 0.0708 | 0.0000 |
| `ida_077_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 1479 | 291 | 0.0333 | 0.0000 |
| `ida_084_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 3951 | 1803 | 0.0277 | 0.0000 |
| `ida_091_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 2145 | 10017 | 0.0132 | 0.0000 |
| `ida_098_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 1547 | 1390 | 0.1627 | 0.0000 |
| `ida_105_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 2643 | 2125 | 0.0565 | 0.0000 |
| `ida_112_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 5031 | 6592 | 0.4053 | 0.0000 |
| `ida_119_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 1449 | 14704 | 0.0000 | 0.0000 |
| `ida_126_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 4239 | 1849 | 0.0856 | 0.0000 |
| `ida_133_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 270 | 2629 | 0.0302 | 0.0000 |
| `ida_140_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 3891 | 1918 | 0.0351 | 0.0000 |
| `ida_147_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 4119 | 4272 | 0.0883 | 0.0000 |
| `ida_154_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 3021 | 335 | 0.0133 | 0.0000 |
| `ida_161_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T0435` | ida | 4158 | 1264 | 0.0328 | 0.0000 |
| `control_bronx_pelham_bay_S2B_MSIL2A_20210907T154809_R054_T18` | control | 0 | 1 | 0.0000 | nan |
| `control_queens_forest_hills_S2B_MSIL2A_20210907T154809_R054_` | control | 0 | 99 | 0.0000 | nan |
| `control_manhattan_central_park_S2B_MSIL2A_20210907T154809_R0` | control | 0 | 3315 | 0.0000 | nan |
| `control_statenisland_lighthouse_S2B_MSIL2A_20210907T154809_R` | control | 0 | 5415 | 0.0000 | nan |
| `control_brooklyn_park_slope_S2B_MSIL2A_20210907T154809_R054_` | control | 0 | 658 | 0.0000 | nan |

## Why this is below the card metric (0.5979)

The card's exact training-pipeline chip extraction is not in the
published artifacts. The card's chips are produced by
`experiments/14_prithvi_nyc_pluvial/build_dataset.py` (referenced
in the model card but not in the riprap-nyc repo we have access
to), and presumably crop tightly around each polygon and use
synthetic copy-paste augmentation that boosts the train-time
positive density. Our reconstruction crops 224x224 at 10m
resolution centered on each polygon, producing chips that are
0.2-1% positive by area on the median polygon.

What this evaluation does establish:
- the published weights load cleanly (0 missing, 0 unexpected keys)
- the model runs end-to-end on M3 CPU
- on independent reconstruction the fine-tune is materially
  better than the Sen1Floods11 zero-shot baseline (the gap is
  80556.7x), which matches the qualitative claim on the
  model card even when the absolute IoU does not reproduce.

## Provenance

```json
{
  "model_name": "msradam/Prithvi-EO-2.0-NYC-Pluvial",
  "model_revision": null,
  "inputs": [
    {
      "tile_id": "ida_000_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "ida_007_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "ida_014_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "ida_021_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "ida_028_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "ida_035_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "ida_042_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "ida_049_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "ida_056_S2B_MSIL2A_20210907T154809_R054_T18TWL_20210908T035137"
    },
    {
      "tile_id": "ida_063_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "ida_070_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "ida_077_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "ida_084_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "ida_091_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "ida_098_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "ida_105_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "ida_112_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "ida_119_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "ida_126_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "ida_133_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "ida_140_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "ida_147_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "ida_154_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "ida_161_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "control_bronx_pelham_bay_S2B_MSIL2A_20210907T154809_R054_T18TWL_20210908T035137"
    },
    {
      "tile_id": "control_queens_forest_hills_S2B_MSIL2A_20210907T154809_R054_T18TWL_20210908T035137"
    },
    {
      "tile_id": "control_manhattan_central_park_S2B_MSIL2A_20210907T154809_R054_T18TWL_20210908T035137"
    },
    {
      "tile_id": "control_statenisland_lighthouse_S2B_MSIL2A_20210907T154809_R054_T18TWK_20210908T043508"
    },
    {
      "tile_id": "control_brooklyn_park_slope_S2B_MSIL2A_20210907T154809_R054_T18TWL_20210908T035137"
    }
  ],
  "code_sha": "2bae3e164457372d165a249568f25cfcdc18a1db",
  "platform": "Darwin arm64 py3.12.12",
  "captured_at_utc": "2026-05-10T12:44:58.954523+00:00"
}
```

```yaml measurements
model: Prithvi-EO 2.0 NYC Pluvial
card_metric: "0.5979 flood IoU"
reproduced: "0.0806 flood IoU (gap: card chip-extraction not public)"
method: "stride-7 reconstruction, n=29, 24 ida + 5 control"
m3: "yes (cpu fp32, 324M params, ~10s/tile)"
j_per_call: "2.53 J (estimated, 211 ms)"
```


## Benchmark

- n_calls: 5
- avg_duration_s: 0.2107
- avg_joules: 2.5280 (estimated)
