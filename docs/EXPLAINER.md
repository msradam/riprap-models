# What I shipped

Internal-facing brief for walking up to Noel Hidalgo at BetaNYC, or
someone in the Mayor's office, or anyone in NYC civic-tech, and being
able to say with a straight face "here's what I shipped with AI-assisted
coding."

## The one-paragraph version

Three NYC fine-tuned foundation models — a 1.5M-param time-series model
that nowcasts storm surge at the Battery, a 324M-param vision transformer
that finds flood water in Sentinel-2 imagery, and a 1B-param multi-modal
model with three NYC LoRA adapters (buildings, land-use, building-context)
— published on Hugging Face, all open weights, all open data, all
Apache-2.0. Plus a reproduction harness that loads the published weights,
constructs held-out NYC test sets from public sources (NOAA, MS Planetary
Computer, NYC OpenData, ESA WorldCover), runs each model on a 16 GB
MacBook Air M3, reports per-call energy, and surfaces the gaps to the
card claims honestly. Plus a Streamlit demo that pulls real-time NYC data
into all four models in one click. Total: ~3 GB of model weights, four
measured rows in `RESULTS.md`, every number reproducible, runs without
GPU or vendor LLM.

## What's in the bag

| Artifact | Where | Use |
|---|---|---|
| 4 fine-tuned models | `huggingface.co/msradam/{Granite-TTM-r2-Battery-Surge, Prithvi-EO-2.0-NYC-Pluvial, TerraMind-NYC-Adapters}` | Drop into any Python pipeline |
| Reproduction harness | `github.com/msradam/riprap-models` | `riprap-models eval <name>` regenerates every number |
| Live Streamlit demo | `uv run streamlit run app/streamlit_app.py` | One-button live forecast / segmentation on real NYC data |
| Gap-analysis reports | `eval/reports/*.md` | Per-tile / per-window / per-threshold detail |
| AI-regulation mapping | `docs/COMPLIANCE.md` | EU AI Act / NIST AI RMF / NYC AI Action Plan / OMB M-24-10 |
| Plain-English brief | `docs/EXPLAINER.md` (this file) | Walk-up explanation |
| Differentiation pitch | `docs/PITCH.md` | One-pager vs prior work |
| Build log | `WORKLOG.md` | Chronological with debug iterations |

## The three (well, four) models

### 1. Granite TTM r2 Battery Surge

- **Base**: `ibm-granite/granite-timeseries-ttm-r2`, IBM Research's Tiny
  Time Mixer r2, 1.5 M params, pretrained on a global mix of public
  time series (electricity, traffic, weather, retail).
- **Fine-tune**: hourly storm-surge residual at NOAA tide gauge 8518750
  (The Battery, lower Manhattan). Trained on 10 years of CO-OPS data.
- **Input**: 1024 hours (~43 days) of hourly surge residual.
- **Output**: 96-hour (4-day) forecast of surge residual.
- **Card metric**: 0.1091 m MAE on 12k 2023-2024 sliding windows.
- **Reproduced**: 0.1318 m MAE on 40 strictly post-cutoff (2025-2026)
  windows. **Stratified**: tied with zero-shot on calm, +6% better at
  peak ≥ 0.5 m, **+10% at peak ≥ 0.7 m** (the storm regime).
- **Bench**: 18 ms / call, 0.21 J / call.
- **Use case**: nor'easter / hurricane surge nowcasts. Drop-in
  alternative to NOAA ETSS API where you want it embedded locally.

### 2. Prithvi-EO 2.0 NYC Pluvial

- **Base**: `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11`,
  NASA / IBM's Earth-observation foundation model, 300M params,
  pretrained on global Sentinel-2 imagery + Sen1Floods11 fine-tune.
- **Fine-tune**: NYC-specific pluvial flood segmentation (Hurricane Ida
  pattern: rapid stormwater accumulation, basement flooding). Trained
  on Riprap's 166 baked Ida 2021 polygons + 332 copy-paste
  augmentations + Major-TOM clear-sky negatives, with Lovász-Softmax
  loss tuned for rare-class IoU.
- **Input**: 224×224 Sentinel-2 L2A chip (6 bands: B02/B03/B04/B05/B06/B07).
- **Output**: binary flood segmentation mask.
- **Card metric**: 0.5979 flood IoU on a held-out test split.
- **Reproduced**: 0.115 vicinity IoU / 0.080 chip-wide IoU on 24
  stride-7 holdout polygons + matching post-Ida Sentinel-2 chips.
- **Honest gap**: card's headline is conditional on an unpublished
  chip-extraction script (`build_dataset.py` referenced in the card,
  not in the public artifacts). The model **does** find large flood
  regions (largest test polygon scored IoU 0.508 alone), but absolute
  reproducibility from public weights + public data tops out around
  0.115. Documented in `eval/reports/prithvi_pluvial.md`.
- **Bench**: 211 ms / call, 2.57 J / call.
- **Use case**: flag candidate flood regions in S2 imagery for human
  review. Not a substitute for hydraulic models or FEMA floodplains.

### 3. TerraMind NYC Buildings (LoRA)

- **Base**: `ibm-esa-geospatial/TerraMind-1.0-base`, IBM-ESA's
  multi-modal Earth-observation foundation model, 1 B params,
  pretrained on global Sentinel-2 + Sentinel-1 + DEM + LULC + NDVI
  with cross-modality token alignment.
- **Fine-tune**: LoRA adapter (rank 16, ~885 K trainable params on
  attention QKV/proj across 24 transformer blocks) + UNet decoder
  (~80M from-scratch params), trained on rasterized NYC DOITT
  building footprints.
- **Input**: dictionary of three multi-modal tensors at 224×224, four
  timesteps each: S2L2A 12 bands, S1RTC 2 bands (VV/VH in dB), DEM 1 band.
- **Output**: binary building / not-building segmentation mask.
- **Card metric**: 0.5511 mIoU; per-class building IoU 0.293.
- **Reproduced**: 0.349 building IoU at default threshold;
  **0.365 at threshold 0.6 (best)** — *higher* than the card's 0.293.
  The mIoU gap (0.33 vs 0.55) is composition: my 6 dense urban AOIs
  have ~50% buildings, so non-building is the rare class and its IoU
  drops the macro mean.
- **Bench**: 511 ms / call, 6.13 J / call.
- **Use case**: building exposure overlay (flag any building near
  floodwater). Recall-biased per the card's own caveat. Not a
  substitute for NYC DOITT (the training labels).

### 4. TerraMind NYC LULC (LoRA)

- **Base**: same TerraMind 1.0 base + same multi-modal input pipeline.
- **Fine-tune**: 5-class NYC land-use / land-cover (water, impervious,
  vegetation, bare/cropland, building) using ESA WorldCover 2021
  collapsed to 5 classes, with NYC DOITT footprints overlaid as the
  building class.
- **Card metric**: 0.5866 mIoU.
- **Reproduced**: 0.355 mIoU. Per-class breakdown:
  - water: **0.943** (*higher* than card's 0.770)
  - impervious: 0.526 (vs card 0.949 — composition difference)
  - vegetation: 0.306 (vs card 0.780)
  - bare/cropland: 0.001 (no bare in dense urban AOIs)
  - building: 0.000 (model never predicts class 4 — matches card's 0.045)
- **Bench**: 510 ms / call, 6.12 J / call.
- **Use case**: rough land-use overlays at 10 m, especially in
  non-NYC cities where ESA WorldCover is the only available data.

## How the harness works

```bash
# install
git clone https://github.com/msradam/riprap-models
cd riprap-models
uv venv --python 3.12
uv pip install -e ".[dev,terramind,prithvi,ttm,live]"

# fastest of the four (no model download beyond the 12 MB TTM)
uv run riprap-models eval ttm-battery-surge

# fetches a 1.24 GB Prithvi checkpoint on first run (~5 min)
uv run riprap-models eval prithvi-pluvial

# fetches 1.45 GB TerraMind base + 308 MB adapter on first run (~10 min)
uv run riprap-models eval terramind-buildings
uv run riprap-models eval terramind-lulc

# regenerate the headline table from per-model reports
uv run riprap-models report

# live forecast on today's NOAA data, frozen as a fixture
uv run riprap-models run-live ttm-battery-surge

# bit-identical replay against frozen fixture
uv run riprap-models replay ttm-battery-surge

# Streamlit demo
uv run streamlit run app/streamlit_app.py
```

## How to demo this

For a 5-minute show-and-tell:

1. **Open Streamlit app** (`uv run streamlit run app/streamlit_app.py`).
   Browser opens to localhost:8501. Header shows "Local · Apple M3"
   with a green dot — proof of locality.
2. **Battery Surge tab** → "Run live forecast". 5–10 seconds: NOAA
   pull, model load, inference. 18 ms inference time prominently
   displayed. Chart shows last 14 days of surge + 96-hour forecast.
3. **NYC Satellite tab** → pick "Manhattan midtown", click both
   buttons. Maps + side-by-side input/label/prediction PNGs render.
   Per-call timing and joules in metrics row.
4. **About tab** → reproducibility + honest limitations sections.
   "Here's where the card overstates; here's the reproducible number."

For a longer conversation, walk through `WORKLOG.md` (the debug
iterations are the real story) and `COMPLIANCE.md` (the EU AI Act
table is the procurement-friendly version).

## What the audience cares about, mapped

| Person / org | What they care about | What to say |
|---|---|---|
| **Noel Hidalgo (BetaNYC)** | Civic tech that runs on a laptop, open source, NYC-specific, gets non-ML people in the door | "Three NYC foundation models, one Streamlit app, runs on a MacBook Air, every number is reproducible from public data" |
| **NYC OpenData / DOITT** | Maps to NYC's existing open-data ecosystem (DOITT footprints, NOAA Battery, etc.) with provenance | "TerraMind buildings is a Sentinel-2-derived approximation of `5zhs-2jue`. TTM uses station 8518750. Prithvi uses Sentinel-2 + Ida event polygons." |
| **NYCEM / MOCEJ** | Tools their analysts can use without procurement, no vendor lock, survives a journalist's reproduction attempt | "Compliance posture in `COMPLIANCE.md`: EU AI Act, NIST AI RMF, NYC AI Action Plan. Honest gap analysis in WORKLOG.md." |
| **NYC tech firms (CARTO / Mapbox / Foursquare)** | Open-source EO foundation models they can build commercial layers on | "LoRA adapter pattern: one base on disk, three NYC adapters. Add a fourth (impervious surface, heat islands) without retraining the base." |
| **Climate NGOs (NRDC / Riverkeeper)** | Open access to climate data without commercial cloud accounts | "0.21 J / 2.5 J / 6.1 J per call. No vendor lock. Apache 2.0 throughout. Energy methodology in `ENERGY.md`." |
| **Mayor Mamdani's tech team** | "Built with AI-assisted coding, but here's the receipts" | "Model cards on HF, reproduction harness on GitHub, Streamlit demo runs in 5 minutes on a borrowed laptop. Every claim is testable in the harness." |

## What's not in scope

- **Hydraulic flood modelling.** Use HEC-RAS or InfoWorks ICM.
- **Engineering-grade fragility.** Building IoU is recall-biased.
- **A SaaS dashboard with SLAs.** This is open-source civic infrastructure.
- **Replacing NOAA ETSS / NYC FloodHelp / DOITT.** These models add
  signal to those authoritative sources; they don't replace them.

## What I learned that's worth knowing

Findings from the build that aren't in any model card:

1. **TTM's edge is storm-magnitude-conditional**, not aggregate. The
   "marginal lift" you'd see from a flat eval is calm-weather
   selection bias. The fine-tune wins where it matters.
2. **Prithvi's 0.60 IoU is conditional on chip extraction.** Without
   `build_dataset.py` published, public reproduction tops out at
   ~0.12. The model is real and finds flood; the headline number
   needs the script.
3. **TerraMind expects raw 0-10000 S2 reflectance and dB-converted
   S1**, not the normalized [0,1] form most pipelines use. Plus the
   12-band order is `[B01, B02, B03, B04, B05, B06, B07, B08, B8A,
   B09, B11, B12]` (skip B10), not the default rasterio sort.
4. **TerraMind LULC class order is `[water, impervious, vegetation,
   bare, building]`**, recovered by permutation search against the
   loaded weights. The card names classes but doesn't number-list
   them.
5. **macOS `powermetrics` requires sudo without prompt.** Without
   that, the energy fallback uses Apple's published M3 power envelope
   (12 W) × wall-clock. Method field reports `estimated` so a reader
   can't mistake it for measured.

These five findings are the kind of thing AI-assisted coding caught
that a human reading the model cards alone might not have. The
harness exists to surface them.

## License

Apache-2.0 across the board.
