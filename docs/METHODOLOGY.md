# Methodology

This document records the rules the harness follows when it produces a
number. The intent: a reader should be able to reproduce any cell of
`RESULTS.md` from this file and the configs under `eval/configs/`.

## What we measure

Three quantities per model:

1. **Headline accuracy on held-out NYC data.** For the segmentation
   models (TerraMind Buildings, Prithvi-EO 2.0 NYC Pluvial), this is
   IoU computed exactly as defined in `src/riprap_models/common/metrics.py`.
   For the regression model (Granite TTM r2 Battery Surge), this is MAE
   in metres against persistence as a baseline.
2. **Wall-clock and peak resident memory** during one forward pass on
   the smallest valid input shape for each model.
3. **Joules per inference call**, measured by the best method available
   on the runtime platform. See `ENERGY.md`.

## What "held-out" means here

For the segmentation models, the held-out manifest is constructed from
public Sentinel-2 tiles that fall in NYC and were not used during the
fine-tune. The exact tile IDs are recorded in each report's provenance
block. Where the upstream training split was thin or undocumented (a
finding noted in the riprap-nyc experiments), we construct a fresh split
from public sources and report under both labels.

For TTM Battery Surge, the held-out windows are explicit ISO date ranges
at NOAA station 8518750 chosen to not overlap the training window. The
windows are listed in `eval/configs/ttm_battery_surge.yaml` and explained
in `PROVENANCE.md`.

## What we report on a failed reproduction

If a measured number disagrees with the model card by more than the
documented tolerance, the report keeps the measured value, names the
gap, and offers a hypothesis. The repo does not retry-with-tweaks until
the number "matches"; that pattern produces unreproducible results.

## What's deliberately out of scope

- Hyperparameter search over the fine-tunes. We evaluate the published
  weights, not retraining variants.
- Per-token energy attribution. The granularity here is per inference
  call, which is the unit a downstream caller actually sees.
- Engineering-grade hydraulic simulation. Same caveat as `riprap-nyc`:
  this is exposure triage, not flood vulnerability modelling.
