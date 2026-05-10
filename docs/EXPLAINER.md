# Models guide

Plain-language description of the four NYC fine-tuned foundation models in
this repository, what each one does, what it was trained on, and how to
read its reproduction numbers. For readers with general ML / SWE
familiarity but not necessarily deep geospatial-ML background.

## Summary

Four NYC fine-tuned foundation models, total ~3 GB on disk, all running
on a 16 GB MacBook Air M3 in CPU fp32, all open weights / open data /
Apache-2.0:

- a 1.5M-parameter **storm-surge nowcast at The Battery** (Granite TTM r2
  fine-tune) that catches Hurricane Ida and the December 2024 nor'easter,
  and outperforms the pretraining-only baseline by 10 % on storm windows
  with peak |residual| ≥ 0.7 m;
- a 324M-parameter **NYC Hurricane-Ida pluvial-flood pattern detector**
  (Prithvi-EO 2.0 fine-tune) that fires on every one of the 5 largest
  Ida 2021 flood polygons and stays correctly silent on non-event scenes;
- a 1B-parameter multi-modal foundation model (TerraMind 1.0) with two
  NYC LoRA adapters: **building segmentation** (high-recall candidate
  detector) and **5-class land cover** at 10 m resolution (water IoU
  0.94, higher than the published card metric).

Plus the reproduction harness, a probe that runs 10 sniff-test cases per
model on real public data (currently 38 of 40 pass), a Streamlit demo
that pulls today's NOAA and Sentinel-2 data in one click, and an
AI-governance posture mapped to EU AI Act / NIST AI RMF / NYC AI Action
Plan / OMB M-24-10. Every number in the documentation is regenerable
from public sources via `riprap-models eval <name>` or
`python scripts/probe.py`.

## What "fine-tuning" actually does

A foundation model has been pretrained on a large generic corpus and
captures general representations of its domain (time-series patterns,
satellite imagery patterns, etc.). Fine-tuning takes a smaller
task-specific dataset and adjusts some subset of the model's weights so
that outputs become more useful for that task.

There are two flavours in this repository:

- **Full fine-tune** (Prithvi, TTM): every weight in the model can change.
  Storage cost = whole model. Risk = forgetting the pretraining signal
  if trained too long. Used here when the fine-tuning dataset is small
  enough that overfitting is the dominant risk.
- **LoRA fine-tune** (TerraMind): freeze the base model entirely; add a
  small set of low-rank "delta" matrices to selected layers (here: the
  attention QKV and projection in each of the 24 transformer blocks);
  train only those deltas plus a fresh decoder head. Storage cost
  ~325 MB per task instead of ~1.6 GB. Three NYC-specific tasks share
  one base model on disk.

In both cases, the fine-tune produces a `safetensors` (or `pt`) file
loaded on top of the base architecture. This repository loads the
published files directly from Hugging Face.

## The four models, one at a time

### 1. Granite TTM r2 Battery Surge

- **Base:** [`ibm-granite/granite-timeseries-ttm-r2`](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2),
  IBM Research's Tiny Time Mixer r2, 1.5M parameters, pretrained on a
  global mix of public time series (electricity, traffic, weather, retail).
- **Fine-tune:** hourly storm-surge residual at NOAA tide gauge 8518750
  (The Battery, lower Manhattan). Trained on 10 years of NOAA CO-OPS
  data. Storm-surge residual = observed water level − astronomical tide
  prediction; subtracting the tide leaves the part driven by weather.
- **Input:** 1024 hours (~43 days) of hourly surge residual.
- **Output:** 96-hour (4-day) forecast.
- **Card metric:** 0.1091 m MAE on 12,033 sliding 1024→96 hourly
  windows from 2023-2024.
- **Reproduced in this harness:** 0.132 m MAE across 40 strictly
  post-cutoff (2025-2026) windows. Stratified results: ties zero-shot
  on calm, +6 % at peak ≥ 0.5 m, **+10 % at peak ≥ 0.7 m**.
- **Bench:** 18 ms / call, 0.21 J / call.

### 2. Prithvi-EO 2.0 NYC Pluvial

- **Base:** [`ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11),
  NASA-IBM's Earth-observation foundation model, 300M parameters,
  pretrained on global Sentinel-2 + further trained on the public
  Sen1Floods11 flood dataset.
- **Fine-tune:** NYC-specific Hurricane-Ida pluvial-flood pattern.
  Trained on 166 baked Ida 2021 polygons + 332 copy-paste augmented
  positives + 286 clear-sky NYC negatives. Lovász-Softmax loss.
- **Input:** 224×224 Sentinel-2 L2A chip, 6 bands (B02, B03, B04, B05,
  B06, B07; Sen1Floods11 schema).
- **Output:** binary flood-extent mask.
- **Card metric:** 0.5979 flood IoU on a held-out test split.
- **Reproduced in this harness:** 0.115 polygon-vicinity IoU /
  0.0806 chip-wide IoU on 24 stride-7 holdout polygons + matching
  post-Ida Sentinel-2 chips. The reproduction gap is the card's
  chip-extraction recipe (`build_dataset.py`, referenced in the card
  but not in the public artifacts). The model genuinely finds large
  flood regions: the largest single test polygon scored IoU 0.508
  alone.
- **Bench:** 211 ms / call, 2.57 J / call.

### 3. TerraMind NYC Buildings (LoRA)

- **Base:** [`ibm-esa-geospatial/TerraMind-1.0-base`](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base),
  IBM-ESA's multi-modal Earth-observation foundation model, 1B
  parameters, pretrained on Sentinel-2 + Sentinel-1 + DEM + LULC + NDVI
  with cross-modality token alignment.
- **Fine-tune:** LoRA adapter (rank 16, ~885K trainable Δ params) +
  UNet decoder (~80M from-scratch params), trained on rasterized NYC
  DOITT building footprints.
- **Input:** dictionary with three multi-modal tensors at 224×224, four
  timesteps each: S2L2A 12 bands, S1RTC 2 bands (VV/VH in dB), DEM 1 band.
- **Output:** binary building / not-building segmentation mask.
- **Card metric:** 0.5511 mIoU; per-class building IoU 0.293.
- **Reproduced in this harness:** 0.349 building IoU at default
  threshold; **0.365 at threshold 0.6 (best)** — higher than the card
  for the building class. The mIoU difference (0.33 vs 0.55) reflects
  test-set composition: the harness's six dense urban AOIs have ~50 %
  buildings, so the non-building class is sparse and its IoU is low.
- **Bench:** 511 ms / call, 6.13 J / call.

### 4. TerraMind NYC LULC (LoRA)

- **Base:** same TerraMind 1.0 base + same multi-modal input pipeline.
- **Fine-tune:** 5-class NYC land-use / land-cover (water, impervious,
  vegetation, bare/cropland, building) using ESA WorldCover 2021
  collapsed to 5 classes, with NYC DOITT footprints overlaid as the
  building class.
- **Card metric:** 0.5866 mIoU.
- **Reproduced in this harness:** 0.355 mIoU. Per-class IoU:
  water **0.943** (higher than card's 0.770); impervious 0.526;
  vegetation 0.306; bare/cropland 0.001; building 0.000.
- **Bench:** 510 ms / call, 6.12 J / call.

## How the harness works

```bash
# install
git clone https://github.com/msradam/riprap-models
cd riprap-models
uv venv --python 3.12
uv pip install -e ".[dev,terramind,prithvi,ttm,live]"

# fastest of the four (no model download beyond the 12 MB TTM)
uv run riprap-models eval ttm-battery-surge

# fetches a 1.24 GB Prithvi checkpoint on first run
uv run riprap-models eval prithvi-pluvial

# fetches 1.45 GB TerraMind base + 308 MB adapter on first run
uv run riprap-models eval terramind-buildings
uv run riprap-models eval terramind-lulc

# headline regeneration
uv run riprap-models report

# 40-case sniff-test probe
uv run riprap-models probe

# live NOAA forecast, frozen as a fixture for offline replay
uv run riprap-models run-live ttm-battery-surge
uv run riprap-models replay ttm-battery-surge

# Streamlit demo
uv run streamlit run app/streamlit_app.py
```

## Useful findings from the build that aren't on any model card

These are the kind of detail the reproduction harness surfaces that a
reader of the model cards alone might not notice:

1. **TTM's edge over zero-shot scales with surge magnitude**, not
   aggregate. On the 40-window post-cutoff sliding evaluation, the
   fine-tune is essentially tied with zero-shot on average but wins
   monotonically as surge magnitude rises (+6 % at peak ≥ 0.5 m,
   +10 % at peak ≥ 0.7 m). Persistence is uncompetitive at any storm
   threshold.
2. **Prithvi's 0.5979 flood IoU is conditional on chip extraction.**
   Without `build_dataset.py` in the public artifacts, public
   reproduction tops out at ~0.12 vicinity IoU. The model is
   functional and finds flood; the headline number requires the
   training-time chip-extraction pipeline to reproduce exactly.
3. **TerraMind expects raw 0–10000 Sentinel-2 reflectance and
   dB-converted Sentinel-1 input**, not the normalised [0, 1] form most
   pipelines use. The 12-band order is `[B01, B02, B03, B04, B05, B06,
   B07, B08, B8A, B09, B11, B12]` (skip B10), not the default rasterio
   band sort.
4. **TerraMind LULC class order is `[water, impervious, vegetation,
   bare, building]`**, recovered by permutation search against the
   loaded weights. The model card names classes but does not number-list
   them.
5. **macOS `powermetrics` requires sudo without prompt.** Without that,
   the energy fallback uses Apple's published M3 power envelope (12 W)
   multiplied by wall-clock duration. The method field reports
   `estimated` so the value cannot be confused for a measured reading.

## License

Apache-2.0 across the board. See `LICENSE`, `NOTICE`, and the per-source
attribution in the parent `README.md`.
