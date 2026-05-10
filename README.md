# riprap-models

Reproduction harness for the three NYC fine-tunes used by
[riprap-nyc](https://github.com/msradam/riprap-nyc). Independently
verifies each model on a public-data reconstruction, on a
16 GB MacBook Air M3, with measured per-call energy.

## Headline reproduction table

| Model | Card metric | Reproduced (this repo, on M3 Air) | J/call |
|---|---:|---:|---:|
| [Granite TTM r2 Battery Surge](https://huggingface.co/msradam/Granite-TTM-r2-Battery-Surge) | 0.1091 m MAE | **0.1318 m MAE** (40 post-cutoff hourly windows) | 0.21 J |
| [Prithvi-EO 2.0 NYC Pluvial](https://huggingface.co/msradam/Prithvi-EO-2.0-NYC-Pluvial) | 0.5979 flood IoU | **0.0806 flood IoU** (29-chip independent reconstruction; see "honest finding" below) | 2.53 J |
| [TerraMind NYC Adapters (Buildings)](https://huggingface.co/msradam/TerraMind-NYC-Adapters) | 0.5518 mIoU | **0.3288 mIoU** (6 dense urban AOIs; building-class IoU itself **0.349 vs card 0.293**) | 6.13 J |

Headline summary: TTM and TerraMind reproduce within explainable
distance of their cards. Prithvi has a large gap, attributable to
the card's chip-extraction script not being in the public artifacts.
Detail in `docs/RESULTS.md`, per-model reports in `eval/reports/`,
plain-English explainer in `docs/EXPLAINER.md`.

## What this repo is for

A skeptical reviewer can clone this repo, run on a MacBook Air, and
see what the three Hugging Face fine-tunes actually do on real NYC
data. Every reported number is reproducible from public sources
(NOAA CO-OPS, Microsoft Planetary Computer, NYC OpenData) — no
auth, no AMD GPU, no hidden test split.

## Quick start

```bash
git clone https://github.com/msradam/riprap-models
cd riprap-models
uv venv --python 3.12
uv pip install -e ".[dev,terramind,prithvi,ttm,live]"

# fastest of the three (no model download needed beyond the 12 MB TTM)
uv run riprap-models eval ttm-battery-surge

# fetches a 1.24 GB Prithvi checkpoint on first run (~5 min)
uv run riprap-models eval prithvi-pluvial

# fetches 1.45 GB TerraMind base + 308 MB adapter on first run (~10 min)
uv run riprap-models eval terramind-buildings

# regenerate the headline table from the per-model reports
uv run riprap-models report
```

## Honest findings

Per-model details in `eval/reports/`. The biggest things a downstream
consumer should know:

- **Granite TTM r2 Battery Surge.** Reproduces within 20% of the
  card on out-of-distribution post-cutoff windows. On strictly
  post-2025 data, the fine-tune is essentially tied with the
  pretraining-only zero-shot TTM r2; both beat the persistence
  baseline by ~30%. The fine-tune's edge is real but small once you
  leave the training distribution.

- **Prithvi-EO 2.0 NYC Pluvial.** The card's headline 0.5979 flood
  IoU is **not reproducible from the published artifacts alone**.
  The chip-extraction script (`build_dataset.py`, referenced in the
  card) is not in `riprap-nyc`. With the public weights + a fair
  NYC reconstruction (24 stride-7 Ida polygons + matching cloud-free
  Sept 7, 2021 Sentinel-2 chips + 5 controls), the model achieves
  flood IoU 0.08 in aggregate. The model **does** find large flood
  regions: the largest test polygon scored IoU 0.51 alone. Read the
  card's headline as conditional on the unpublished chip recipe.

- **TerraMind Buildings (LoRA).** Building-class IoU itself reproduces
  and is slightly higher than the card (0.349 vs 0.293). The 0.55 vs
  0.33 mIoU gap is composition: the card averages building and
  non-building IoU; my 6 dense urban AOIs have ~50% buildings, so
  non-building is the rare class, hurting macro mean. The card's
  "recall-biased, over-segments" caveat is real and visible in the
  per-tile detail.

## Repository map

```
src/riprap_models/
  cli.py                      riprap-models eval/bench/run-live/replay/report/device
  device.py                   cuda/mps/cpu detection + dtype rules
  energy.py                   nvml/rapl/powermetrics/estimated; uniform Joules
  live.py                     live-fetch + freeze-fixture + replay
  common/
    metrics.py                IoU/mIoU (macro+micro), MAE/RMSE, persistence
    provenance.py             tile/station/code-sha provenance per report
    report.py                 RESULTS.md regenerator
    viz.py                    side-by-side seg + time-series helpers
  ttm_battery_surge/          ~3M params, NOAA station 8518750
  prithvi_pluvial/            324M params, Sentinel-2 + Sen1Floods11 schema
  terramind/                  ~168M params multi-modal (S2L2A+S1RTC+DEM × 4t)
eval/
  configs/                    YAML per model: holdout split, AOIs, dates
  reports/                    generated markdown + provenance per model
  fixtures/                   frozen replay fixtures (committed for repro)
docs/
  EXPLAINER.md                plain-English what/why for each model
  RESULTS.md                  headline reproduction table (regenerated)
  METHODOLOGY.md              how we measure, what's in scope
  PROVENANCE.md               tile IDs, station IDs, holdout construction
  ENERGY.md                   per-platform energy methodology
  M3_NOTES.md                 what runs on the Air, with measurements
WORKLOG.md                    chronological build log + debug iterations
```

## License

Apache-2.0. Sentinel-2 / Sentinel-1 imagery via Microsoft Planetary
Computer under the Copernicus Open Data License. NOAA CO-OPS data is
public domain. NYC DOITT building footprints are public domain via
NYC OpenData (`5zhs-2jue`). Hurricane Ida flood polygons from
`riprap-nyc`'s `data/prithvi_ida_2021.geojson` (Apache-2.0).
