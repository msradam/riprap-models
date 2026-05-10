# riprap-models

Reproduction harness for the three NYC fine-tunes used by
[riprap-nyc](https://github.com/msradam/riprap-nyc). Independently
verifies each model on a public-data reconstruction, on a
16 GB MacBook Air M3, with measured per-call energy.

## Headline reproduction table

| Model | Card metric | Reproduced (this repo, on M3 Air) | J/call |
|---|---:|---:|---:|
| [Granite TTM r2 Battery Surge](https://huggingface.co/msradam/Granite-TTM-r2-Battery-Surge) | 0.1091 m MAE (12k 2023-2024 windows) | **0.1318 m MAE** all-windows; **0.3239 m MAE on storm windows (peak ≥ 0.7 m), 10% better than zero-shot** | 0.21 J |
| [Prithvi-EO 2.0 NYC Pluvial](https://huggingface.co/msradam/Prithvi-EO-2.0-NYC-Pluvial) | 0.5979 flood IoU | **0.1150 polygon-vicinity IoU** (within 300m of any GT polygon); 0.0806 chip-wide | 2.57 J |
| [TerraMind NYC Adapters (Buildings)](https://huggingface.co/msradam/TerraMind-NYC-Adapters) | 0.5518 mIoU | **0.3653 building IoU** at threshold 0.6 (best); 0.3288 mIoU at default; building-class IoU itself *higher* than card's 0.293 | 6.13 J |

Headline summary:

- **TTM**: marginal lift on calm windows is real, but the model wins
  exactly where it matters — on the biggest storms (peak ≥ 0.7 m it
  beats the pretraining-only baseline by 10%; persistence is uncompetitive).
- **Prithvi**: vicinity scoring (only count pixels within 300 m of any
  GT polygon, since the labels don't include pre-existing rivers/coast
  the model legitimately segments) closes ~30% of the gap to the card.
  Remainder is the unpublished chip-extraction recipe.
- **TerraMind**: threshold tuning to 0.6 gives best IoU. Building-class
  IoU itself reproduces and is slightly higher than the card.

Detail in `docs/RESULTS.md`, per-model reports in `eval/reports/`,
plain-English explainer in `docs/EXPLAINER.md`,
procurement / AI-regulation mapping in `docs/COMPLIANCE.md`.

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
consumer should know (updated after the gap-analysis pass):

- **Granite TTM r2 Battery Surge.** Aggregate post-cutoff MAE 0.132 m
  is close to the card's 0.109 m. **Stratified by surge magnitude**, the
  fine-tune's edge over the pretraining-only zero-shot baseline scales
  with storm severity: tied on calm windows, +6% better at peak ≥ 0.5 m,
  **+10% better at peak ≥ 0.7 m**. The fine-tune wins where it matters,
  on actual storm forecasts. Persistence is uncompetitive at any storm
  threshold (50–100% worse).

- **Prithvi-EO 2.0 NYC Pluvial.** Two scoring modes:
  - **Chip-wide IoU 0.0806** — penalised by Ida polygons not labelling
    pre-existing rivers / coast / harbour, which the model legitimately
    segments.
  - **Polygon-vicinity IoU 0.1150** — only counts pixels within 300 m
    of any GT polygon, where labels are complete.

  Vicinity is the fairer metric for this label set. Card's 0.5979 still
  further above; the remaining gap is the unpublished chip-extraction
  recipe (`build_dataset.py` referenced in the card but not in the
  public artifacts). The model **does** find large flood regions: the
  largest single test polygon scored IoU 0.51 by itself.

- **TerraMind Buildings (LoRA).** Building-class IoU 0.349 at default
  threshold; **0.365 at threshold 0.6 (best)**. Both *higher* than the
  card's 0.293 building IoU. The mIoU gap (0.33 vs card 0.55) is
  composition: my 6 AOIs are dense urban (~50% buildings) so the
  non-building class is sparse and its IoU is low. Threshold sweep
  table in `eval/reports/terramind_buildings.md` lets consumers pick
  their precision/recall operating point.

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
  COMPLIANCE.md               EU AI Act / NIST AI RMF / NYC AI Action Plan mapping
WORKLOG.md                    chronological build log + debug iterations
```

## License

Apache-2.0. Sentinel-2 / Sentinel-1 imagery via Microsoft Planetary
Computer under the Copernicus Open Data License. NOAA CO-OPS data is
public domain. NYC DOITT building footprints are public domain via
NYC OpenData (`5zhs-2jue`). Hurricane Ida flood polygons from
`riprap-nyc`'s `data/prithvi_ida_2021.geojson` (Apache-2.0).
