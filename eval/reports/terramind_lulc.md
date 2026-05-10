# TerraMind LULC

## Independent reconstruction

Construction: same 6 NYC AOIs as the buildings adapter (multi-modal
S2L2A 12 bands × 4 timesteps + S1RTC 2 bands × 4 timesteps + Copernicus
DEM GLO-30, all from MS Planetary Computer). Labels: ESA WorldCover 2021
v200 (collapsed to NYC 5-class scheme, see `data.py:ESA_TO_NYC5`) with NYC
DOITT building footprints overwritten on top as class 4.

## Held-out evaluation (n=6)

- mIoU (macro): 0.3553
- card mIoU: 0.5866
- Δ vs card: -0.2313

### Per-class IoU

| class | name | reproduced IoU | card IoU | Δ |
|---|---|---:|---:|---:|
| 0 | water | 0.9429 | 0.7696 | +0.1733 |
| 1 | impervious | 0.5263 | 0.9494 | -0.4231 |
| 2 | vegetation | 0.3064 | 0.7803 | -0.4739 |
| 3 | bare/cropland | 0.0011 | 0.3892 | -0.3881 |
| 4 | building | 0.0000 | 0.0447 | -0.0447 |

### Per-tile detail

| tile | water | impervious | veg | bare | building |
|---|---:|---:|---:|---:|---:|
| `nyc_manhattan_midtown` | 0.000 | 0.435 | 0.128 | 0.000 | 0.000 |
| `nyc_brooklyn_downtown` | 0.748 | 0.519 | 0.275 | 0.000 | 0.000 |
| `nyc_queens_jamaica` | 0.000 | 0.625 | 0.217 | 0.003 | 0.000 |
| `nyc_bronx_morrisania` | nan | 0.573 | 0.345 | 0.003 | 0.000 |
| `nyc_statenisland_stgeorge` | 0.963 | 0.512 | 0.416 | 0.000 | 0.000 |
| `nyc_manhattan_lower_waterfront` | 0.965 | 0.435 | 0.297 | 0.000 | 0.000 |

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
  "code_sha": "9207297155e718ca7f8fb6eef0d5ac34e27f8e81",
  "platform": "Darwin arm64 py3.12.12",
  "captured_at_utc": "2026-05-10T13:53:24.184536+00:00"
}
```

```yaml measurements
model: TerraMind LULC
card_metric: "0.5866 mIoU"
reproduced: "0.3553 mIoU (5-class, ESA WorldCover + DOITT labels)"
method: "6 NYC AOIs, S2L2A+S1RTC+DEM 4 timesteps, ESA WorldCover labels"
m3: "yes (cpu fp32, ~168M params multi-modal)"
j_per_call: "6.12 J (estimated, 510 ms)"
```


## Benchmark

- n_calls: 5
- avg_duration_s: 0.5097
- avg_joules: 6.1169 (estimated)
