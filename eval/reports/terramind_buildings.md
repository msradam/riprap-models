# TerraMind Buildings

## Independent reconstruction

Construction: 6 NYC AOIs (Manhattan midtown, Brooklyn downtown, Queens Jamaica,
Bronx Morrisania, Staten Island St. George, Manhattan Lower waterfront), each
a 224×224 chip at 10 m / pixel. Multi-modal input: Sentinel-2 L2A 12 bands ×
4 timesteps + Sentinel-1 RTC 2 bands × 4 timesteps + Copernicus DEM GLO-30,
all from Microsoft Planetary Computer. Labels: NYC DOITT building footprints
(`5zhs-2jue` on NYC OpenData) fetched per chip via Socrata REST and
rasterized to the chip grid.

## Held-out evaluation (n=6)

- mIoU (macro): 0.3288
- building IoU (micro): 0.3490
- non-building IoU (micro): 0.3087

## Per-tile detail

| tile | gt_pix | pred_pix | building IoU |
|---|---:|---:|---:|
| `nyc_manhattan_midtown` | 25373 | 49901 | 0.5084 |
| `nyc_brooklyn_downtown` | 15482 | 45942 | 0.3359 |
| `nyc_queens_jamaica` | 13751 | 49133 | 0.2794 |
| `nyc_bronx_morrisania` | 14573 | 49208 | 0.2960 |
| `nyc_statenisland_stgeorge` | 4763 | 18294 | 0.2411 |
| `nyc_manhattan_lower_waterfront` | 8081 | 20155 | 0.3824 |

## Provenance

```json
{
  "model_name": "msradam/TerraMind-NYC-Adapters",
  "model_revision": null,
  "inputs": [
    {
      "tile_id": "nyc_manhattan_midtown"
    },
    {
      "tile_id": "nyc_brooklyn_downtown"
    },
    {
      "tile_id": "nyc_queens_jamaica"
    },
    {
      "tile_id": "nyc_bronx_morrisania"
    },
    {
      "tile_id": "nyc_statenisland_stgeorge"
    },
    {
      "tile_id": "nyc_manhattan_lower_waterfront"
    }
  ],
  "code_sha": "b1f1b30101c11a3aac5518a62d72b9abf5fd5678",
  "platform": "Darwin arm64 py3.12.12",
  "captured_at_utc": "2026-05-10T13:05:19.935097+00:00"
}
```

```yaml measurements
model: TerraMind Buildings
card_metric: "0.5511 mIoU"
reproduced: "0.3288 mIoU default; 0.3653 building IoU at threshold 0.6"
method: "6 NYC AOIs, S2L2A+S1RTC+DEM 4 timesteps, DOITT labels"
m3: "yes (cpu fp32, ~168M params multi-modal)"
j_per_call: "6.13 J (estimated, 511 ms)"
```


## Benchmark

- n_calls: 5
- avg_duration_s: 0.5112
- avg_joules: 6.1344 (estimated)

## Threshold sweep (operating-point analysis)

Default argmax (threshold 0.5) leaves the model heavily recall-biased. Sweeping the softmax threshold gives consumers a precision/recall knob.

| threshold | IoU | precision | recall | F1 |
|---:|---:|---:|---:|---:|
| 0.05 | 0.2728 | 0.2728 | 1.0000 | 0.4287 |
| 0.10 | 0.3164 | 0.3164 | 1.0000 | 0.4807 |
| 0.15 | 0.3300 | 0.3300 | 1.0000 | 0.4962 |
| 0.20 | 0.3331 | 0.3332 | 0.9999 | 0.4998 |
| 0.25 | 0.3351 | 0.3351 | 0.9998 | 0.5019 |
| 0.30 | 0.3367 | 0.3367 | 0.9997 | 0.5038 |
| 0.35 | 0.3383 | 0.3383 | 0.9994 | 0.5055 |
| 0.40 | 0.3405 | 0.3406 | 0.9989 | 0.5080 |
| 0.45 | 0.3438 | 0.3441 | 0.9973 | 0.5117 |
| 0.50 | 0.3490 | 0.3499 | 0.9924 | 0.5174 |
| 0.60 | 0.3653 | 0.3803 | 0.9025 | 0.5351 |
| 0.70 | 0.0922 | 0.4752 | 0.1026 | 0.1688 |
| 0.80 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.90 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.95 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**Best IoU**: threshold 0.60 → IoU 0.3653 (precision 0.380, recall 0.903, F1 0.535)

Recommended operating points by use case:

- **Exposure overlay (Riprap default)**: threshold 0.5. Recall 99% — almost no buildings missed. Precision low (~35%) so consumers should treat output as 'building candidates near'.
- **Higher precision needed**: threshold 0.6. IoU 0.365, F1 0.535. Sweet spot from this sweep.
- **Above threshold 0.7**: model collapses (recall drops below 10%). Logits don't reach those confidences on these chips.

