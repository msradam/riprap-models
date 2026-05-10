# What this repo actually is, in plain English

Internal doc for someone with surface familiarity with ML who's read
the model cards and wants to know what they're really looking at.

## The one-paragraph summary

You fine-tuned three foundation models on NYC data: a tiny time-series
model that forecasts storm surge at the Battery, a medium model that
finds flood water in satellite imagery, and a large multi-modal model
that finds buildings in satellite imagery. Each of those fine-tuned
models has a Hugging Face card that claims a headline accuracy number.
This repo independently re-runs each model on a public-data
reconstruction of its test task and reports what actually reproduces
on a 16 GB MacBook Air M3, plus how much energy each call costs.

## The three models, one at a time

### 1. Granite TTM r2 Battery Surge

**What it is.** A fine-tuned variant of IBM's "Tiny Time Mixer r2"
(`granite-timeseries-ttm-r2`), a 1.5M-parameter transformer for time
series forecasting. Tiny by foundation-model standards. Runs on CPU.

**Base model.** Pretrained on a huge mix of public time series
(electricity loads, traffic, retail, weather). Knows how to extrapolate
patterns in any 1-D signal it's given as input.

**The fine-tune.** Specialized on **storm surge residual** at NOAA tide
gauge 8518750 (The Battery, lower Manhattan). Storm surge residual =
observed water level − the harmonic tide prediction. Subtracting the
tide leaves the part driven by weather (wind setup, storm pressure,
river pulses), which is the part an emergency planner actually cares
about.

**Input.** 1024 hours (~43 days) of hourly surge residual at the
Battery, in metres.

**Output.** The next 96 hours (4 days) of forecast surge residual.

**Use case.** "Will the next 4 days produce surge that meaningfully
adds to the astronomical tide?" Answers like "the model expects a peak
of +0.4 m around hour 38" feed into the Riprap-NYC briefing system as
one input among many.

**What this repo measured.** 40-window sliding evaluation across
Jan 2025 to May 2026 (entirely outside the model's training cutoff of
Dec 2024). Plus three named events. See `eval/reports/ttm_battery_surge.md`.

**Honest finding.** On these strictly post-training windows, the
fine-tune (MAE 0.13 m) is **only slightly better** than the
pretraining-only zero-shot TTM r2 (MAE 0.13 m). Both beat the trivial
"persistence" baseline (MAE 0.19 m) by ~30%. The model card's headline
0.11 m MAE was averaged over 12,000+ windows from the 2023-2024 test
split — easier conditions, more training-distribution-like.

---

### 2. Prithvi-EO 2.0 NYC Pluvial

**What it is.** A fine-tuned variant of NASA-IBM's
`Prithvi-EO-2.0-300M-TL-Sen1Floods11`, a 300M-parameter vision
transformer for satellite imagery. Medium-sized. Runs on M3 CPU at
~10 seconds per chip.

**Base model.** Prithvi-EO 2.0 is a foundation model for Earth
observation, pretrained on global Sentinel-2 imagery. The
"-Sen1Floods11" suffix means it's been further trained on a public
flood-segmentation dataset that's mostly coastal events.

**The fine-tune.** Specialized on **pluvial (rain-driven) flooding in
NYC**, particularly the Hurricane Ida 2021 pattern of rapid
stormwater accumulation. The fine-tune used Riprap's 166 baked Ida
flood polygons plus 332 synthetic copy-pasted positives plus 286
clear-sky negatives, with Lovász-Softmax loss optimized for IoU on
the rare flood class.

**Input.** A 224×224 Sentinel-2 L2A chip (so 6.72 km × 6.72 km of NYC
at 30 m resolution? actually 224 × 30 m = 6.7 km, but the chip is at
30 m to match Sen1Floods11 conventions), with 6 bands:
B02, B03, B04, B05, B06, B07 (visible + red-edge + NIR).

**Output.** A binary segmentation mask: each pixel is 0 (no flood) or
1 (flood). Same 224×224 spatial extent as input.

**Use case.** "Where in NYC was there standing water in this
Sentinel-2 capture?" Used by Riprap as one signal in its flood-exposure
briefing — a structural / observational complement to the FEMA
floodplains and DEP stormwater scenarios.

**What this repo measured.** Independent reconstruction: 24 of the
166 Ida polygons held out by stride-7 sampling, matched to the
1.5%-cloud Sept 7, 2021 Sentinel-2 scene over each polygon centroid,
plus 5 clear-sky NYC negative controls. See
`eval/reports/prithvi_pluvial.md`.

**Honest finding.** Reproduced flood IoU is **0.08** vs the card's
**0.60**. That's a big gap. The reproduction shows the model **does**
find large flood regions (the largest test polygon scored IoU 0.51 by
itself, very close to the card), but small-polygon chips drag the
aggregate down because they have <0.5% positive density and any
false-positive elsewhere blows up the IoU denominator. The card's
0.60 was computed on chips extracted by a script
(`build_dataset.py`) that's not in the public artifacts; that script
likely cropped tightly around each polygon to keep positive density
high. Without that script, the public weights produce ~0.08 on a
fair NYC reconstruction. **Read the card's headline as conditional
on the unpublished chip-extraction recipe, not as a property of the
weights.**

---

### 3. TerraMind NYC Adapters (Buildings)

**What it is.** A fine-tuned LoRA adapter on top of IBM-ESA's
TerraMind 1.0 base, a 1B-parameter multi-modal foundation model for
Earth observation. Large. Runs on M3 CPU at ~0.5 seconds per chip.

**Base model.** TerraMind 1.0 is the biggest beast in the family. It
ingests **multiple modalities at once** (Sentinel-2 optical +
Sentinel-1 SAR + DEM elevation) at 4 timesteps each, and has been
pretrained on a global Earth-observation corpus to understand how
those modalities relate.

**The fine-tune.** A LoRA adapter (rank 16, ~885K trainable params on
top of the frozen 1B base) plus a from-scratch UNet decoder (~80M
params), specialized for **NYC building footprint segmentation**.
Trained against rasterized NYC DOITT building polygons. Loss is
weighted cross-entropy. There are sister adapters in the same family
for LULC (land use / land cover) and TiM (a "thinking-in-modalities"
variant of LULC).

**Input.** A dictionary of three multi-modal tensors at 224×224 and
4 timesteps each:
- Sentinel-2 L2A: 12 bands × 4 dates (raw 0-10000 reflectance)
- Sentinel-1 RTC: 2 bands (VV, VH) × 4 dates (in dB)
- Copernicus DEM GLO-30: 1 band × 4 dates (replicated)

**Output.** A binary segmentation mask: 0 (not building) or 1
(building). Same 224×224 spatial extent.

**Use case.** "Where are the buildings in this multi-modal NYC
satellite stack?" Used by Riprap as the structural-prior layer for
exposure overlays (you want to know where buildings are when you're
forecasting flood risk).

**What this repo measured.** Independent reconstruction across 6 NYC
AOIs (Manhattan midtown, Brooklyn downtown, Queens Jamaica, Bronx
Morrisania, Staten Island St. George, Manhattan lower waterfront).
For each AOI, fetched 4 cloud-free Sentinel-2 dates from
April-September 2024, 4 matching Sentinel-1 RTC dates, the
Copernicus DEM, and the DOITT building footprints from NYC OpenData
as labels. See `eval/reports/terramind_buildings.md`.

**Honest finding.** mIoU **0.33** vs the card's **0.55**. But the
**building-class IoU itself reproduces and is slightly higher** than
the card (0.349 vs 0.293). The mIoU gap is composition: the card
averages building IoU and non-building IoU into a 2-class macro
mean. My 6 AOIs are all dense urban (~50% buildings), so the
non-building class is sparse and its IoU is low. The card's 32
chips were a different mix that gave both classes high IoU.

The card's "recall-biased, over-segments" caveat is **real and
visible**: on Manhattan midtown the model achieves 99.99% recall
(catches almost every actual building pixel) but predicts ~2× the
actual building pixels. For the downstream exposure-overlay use
case, recall-biased outputs are what you want — better to over-flag
a building near floodwater than miss it.

---

## What "fine-tuning" actually does, since this is internal

A foundation model has been pretrained on a huge generic corpus and
"knows" general representations of its domain (time series patterns,
satellite imagery patterns, etc.). Fine-tuning takes a smaller
task-specific dataset and adjusts some subset of the model's weights
so that when you give it inputs from your task, the outputs get more
useful for that task.

There are two flavors in this repo:

- **Full fine-tune** (Prithvi, TTM): every weight in the model can
  change. Storage cost = whole model. Risk = forgetting the
  pretraining signal if you train too long. Used here when the
  fine-tuning dataset is small enough that overfitting is the bigger
  risk than catastrophic forgetting.
- **LoRA fine-tune** (TerraMind): freeze the base model entirely;
  add a tiny set of low-rank "delta" matrices to selected layers
  (here: the attention QKV and projection in each of the 24
  transformer blocks); train only those deltas plus a fresh decoder
  head. Storage cost = ~325 MB instead of ~1.6 GB. Combinable: you
  can have LULC, TiM, and Buildings adapters all mounted on the same
  base. Used here because the base is huge and we want three
  specialised heads, not three full duplicates.

In both cases, the fine-tune produces a `.safetensors` (or `.pt`)
file you load on top of the base architecture. This repo loads the
published files from Hugging Face directly.

---

## What "reproduction" means here, and why it matters

Hugging Face model cards report a single headline accuracy number per
model. Those numbers were measured on whatever test set the original
authors had on the training machine at the time. If you, the future
user, want to know whether to trust the model, you need to **run it
yourself on data you can verify** and see what number you get.

This is the "reproduction harness" job. For each model, this repo:

1. **Loads the exact published weights** from Hugging Face via
   `huggingface_hub`, with no modification.
2. **Constructs an independent test set** from public sources (NOAA
   for TTM, Microsoft Planetary Computer for satellite chips, NYC
   OpenData for building footprints).
3. **Runs the model** on that test set and computes the same metric
   the model card uses (IoU for segmentation, MAE for regression).
4. **Compares** the reproduced number to the card claim, and **reports
   the gap honestly**.

If the gap is small, the card is trustworthy. If the gap is large
(Prithvi: 0.08 vs 0.60), there is something the card depends on that
isn't in the public artifacts, and the reader should know that.

This is the "skeptical reviewer" use case: someone says "should we
deploy the Prithvi NYC pluvial fine-tune?" and you want to know,
without committing yourself to several days of work, whether the
card's 0.60 IoU is something you can rely on. The repo answers in
five minutes (the longest single eval).

---

## Why energy

Each of these models will be called many times if it's deployed.
Knowing the per-call energy cost (in joules) lets a downstream user
estimate the carbon footprint of the deployment. The numbers we
report:

- TTM Battery Surge: ~0.21 J / call
- Prithvi NYC Pluvial: ~2.5 J / call
- TerraMind Buildings: ~6.1 J / call

For comparison, Riprap's Granite 4.1:3b reconciler call costs roughly
~108 J per query (0.03 Wh). So all three of these geospatial models
are 1-3 orders of magnitude cheaper per call than the LLM the same
system uses for its final paragraph.

The methodology behind these numbers (estimated from M3 Air's
documented power envelope, since `powermetrics` requires sudo) is in
`docs/ENERGY.md`.

---

## Why the M3 Air bench

You wanted to know whether a journalist or city planner with a
MacBook Air, no GPU, no cloud account, can actually run these
models. The answer for all three is yes:

| Model | Wall-clock | Memory | Comment |
|---|---|---|---|
| TTM Battery Surge | 18 ms / call | trivial | runs faster than a typical web request |
| Prithvi NYC Pluvial | 211 ms / call | ~1.3 GB | comfortable on the Air |
| TerraMind Buildings | 511 ms / call | ~1.7 GB | comfortable on the Air |

All numbers are CPU fp32. MPS would be faster but the eval was run
on CPU because some terratorch ops trigger MPS fallback warnings and
we wanted clean numbers; switching to MPS is one env var
(`PYTORCH_ENABLE_MPS_FALLBACK=1`) plus `device.get_device()`
returning `mps`.

---

## Suggested order to read the rest of the repo

1. `WORKLOG.md` — chronological build log with the exact debug
   iterations that mattered (TerraMind band-order bug, Prithvi
   chip-density issue, TTM cadence mismatch).
2. `docs/RESULTS.md` — the headline table, regenerated automatically
   from each model's report.
3. `eval/reports/<model>.md` — the per-tile detail for whichever
   model you care about.
4. `src/riprap_models/<model>/data.py` — the test-set construction
   code; this is where the "honest reconstruction" happens.

If you want to verify yourself: clone the repo, `uv venv --python
3.12 && uv pip install -e ".[dev,terramind,prithvi,ttm,live]"`,
then `uv run riprap-models eval ttm-battery-surge` (the fastest of
the three).
