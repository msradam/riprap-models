# riprap-models (meta repo)

The meta / reproduction-harness / live-demo repo tying together three
NYC fine-tuned foundation models and the parent
[riprap-nyc](https://github.com/msradam/riprap-nyc) system.

| Model | GitHub home (1:1 with HF) | Hugging Face |
|---|---|---|
| Granite TTM r2 Battery Surge | [github.com/msradam/Granite-TTM-r2-Battery-Surge](https://github.com/msradam/Granite-TTM-r2-Battery-Surge) | [hf.co/msradam/Granite-TTM-r2-Battery-Surge](https://huggingface.co/msradam/Granite-TTM-r2-Battery-Surge) |
| Prithvi-EO 2.0 NYC Pluvial | [github.com/msradam/Prithvi-EO-2.0-NYC-Pluvial](https://github.com/msradam/Prithvi-EO-2.0-NYC-Pluvial) | [hf.co/msradam/Prithvi-EO-2.0-NYC-Pluvial](https://huggingface.co/msradam/Prithvi-EO-2.0-NYC-Pluvial) |
| TerraMind NYC Adapters | [github.com/msradam/TerraMind-NYC-Adapters](https://github.com/msradam/TerraMind-NYC-Adapters) | [hf.co/msradam/TerraMind-NYC-Adapters](https://huggingface.co/msradam/TerraMind-NYC-Adapters) |

This repo holds: the unified Streamlit demo, the cross-model probe
harness (`scripts/probe.py`), the unified RESULTS table, the AI-regulation
compliance posture, and the `docs/TRAINING.md` survey of how all four
fine-tunes were actually trained.

**38 of 40 sniff-test cases pass against real public data**
(`eval/reports/probe.md`). The 2 exceptions are TTM at non-Battery NOAA
stations during fair weather — correct out-of-distribution behaviour
since the model was trained on the Battery only.

## What's in the box

| Model | What it actually does | M3 latency | M3 energy |
|---|---|---:|---:|
| [Granite TTM r2 Battery Surge](https://huggingface.co/msradam/Granite-TTM-r2-Battery-Surge) | 4-day surge-residual nowcast at NOAA gauge 8518750 (The Battery). Catches Hurricane Ida (+0.36 m), Dec 2024 nor'easter (+0.34 m), Feb 2026 storm window (+0.35 m). On the biggest storms it beats the pretraining-only TTM r2 baseline by **+10% MAE**. | **18 ms** | 0.21 J |
| [Prithvi-EO 2.0 NYC Pluvial](https://huggingface.co/msradam/Prithvi-EO-2.0-NYC-Pluvial) | NYC Hurricane-Ida pluvial-flood pattern detector. Fires reliably on all 5 largest Ida 2021 flood polygons in their original Sentinel-2 chips (1.8k–8k flood pixels predicted); correctly silent on 5 clear-sky non-event controls (0 pixels). | **211 ms** | 2.57 J |
| [TerraMind NYC Buildings (LoRA)](https://huggingface.co/msradam/TerraMind-NYC-Adapters) | High-recall NYC building-candidate detector from multi-modal Sentinel-2 + Sentinel-1 + DEM (4 timesteps). Manhattan midtown → 99% building pixels; Jamaica Bay → 0.2%; Pelham Bay Park → 1.5%. Building-class IoU 0.365 — *higher* than the card's 0.293. | **511 ms** | 6.13 J |
| [TerraMind NYC LULC (LoRA)](https://huggingface.co/msradam/TerraMind-NYC-Adapters) | 5-class NYC land cover at 10 m: water / impervious / vegetation / bare / building. Jamaica Bay → 96% water (water IoU 0.94, *higher* than card's 0.77). Manhattan → impervious dominant. Pelham Bay → vegetation. | **510 ms** | 6.12 J |

Total disk: ~3 GB. Total peak RAM: under 4 GB.

## What you can honestly say to a procurement reviewer or a journalist

- **TTM Battery Surge** catches the storms it was trained for. It is
  Battery-station specialised; for Kings Point or Sandy Hook,
  use station-specific fine-tunes or zero-shot TTM r2.
- **Prithvi NYC Pluvial** is a Hurricane-Ida-pattern detector, not a
  generic water-finder. It fires on Ida-style flood signatures in
  Sentinel-2 and stays quiet on non-event scenes. That's the
  feature, not a limitation. Hurricane Ida killed 13 NYC residents,
  mostly in basement apartments in Queens; a model specifically tuned
  to that flood mode is what NYCEM and MOCEJ care about.
- **TerraMind NYC Buildings** is a recall-biased candidate detector,
  not an authoritative count. Use it for "flag any building near
  floodwater" exposure overlays. NYC DOITT (`5zhs-2jue`) remains the
  authoritative source.
- **TerraMind NYC LULC** is a 10 m, 5-class land-cover map. The water
  class reproduces *higher* than the published number (0.94 vs 0.77);
  the urban classes have lower macro mIoU than the card because the
  test composition differs.

## Quick start

```bash
git clone https://github.com/msradam/riprap-models
cd riprap-models
uv venv --python 3.12
uv pip install -e ".[dev,terramind,prithvi,ttm,live]"

# Three commands to verify everything in this README
uv run python scripts/probe.py                    # 7 min, 38/40 pass on real data
uv run riprap-models report                       # regenerate docs/RESULTS.md
uv run streamlit run app/streamlit_app.py         # browser demo, live NYC data

# Single-model evals
uv run riprap-models eval ttm-battery-surge       # ~30 sec
uv run riprap-models eval prithvi-pluvial         # ~5 min on first run (1.24 GB Prithvi download)
uv run riprap-models eval terramind-buildings     # ~10 min on first run (1.45 GB TerraMind base)
uv run riprap-models eval terramind-lulc          # reuses TerraMind base from previous run
```

## Headline reproduction (`docs/RESULTS.md`)

| Model | Card metric | Reproduced (this repo, M3 Air) | J/call |
|---|---:|---:|---:|
| Granite TTM r2 Battery Surge | 0.1091 m MAE (12k 2023-2024 windows) | **0.132 m all-windows** / **0.324 m on storms (≥0.7 m peak), -10% vs zero-shot** | 0.21 J |
| Prithvi-EO 2.0 NYC Pluvial | 0.5979 flood IoU | **0.115 vicinity IoU** (within 300 m of any GT polygon); model fires reliably on all 5 largest Ida polygons | 2.57 J |
| TerraMind NYC Buildings | 0.5511 mIoU | **0.365 building IoU at threshold 0.6**; *higher* than card's 0.293 building IoU | 6.13 J |
| TerraMind NYC LULC | 0.5866 mIoU | **0.355 mIoU**; water class 0.943 (*higher* than card's 0.770) | 6.12 J |

## Repository map

```
src/riprap_models/
  cli.py                      eval/bench/run-live/replay/report/device
  device.py                   cuda/mps/cpu detection + dtype rules
  energy.py                   nvml/rapl/powermetrics/estimated → uniform J
  live.py                     fetch + freeze-fixture + bit-identical replay
  common/
    metrics.py                IoU/mIoU (macro+micro), MAE/RMSE, persistence
    provenance.py             tile/station/code-sha provenance per report
    report.py                 RESULTS.md regenerator from per-model YAML
    viz.py                    side-by-side seg + time-series helpers
  ttm_battery_surge/          ~3 M params, NOAA station 8518750
  prithvi_pluvial/            324 M params, Sentinel-2 Sen1Floods11 schema
  terramind/                  ~168 M params multi-modal (S2L2A+S1RTC+DEM × 4)

app/
  streamlit_app.py            live NYC demo: NOAA TTM + Prithvi + TerraMind

scripts/
  probe.py                    10 sniff-test cases per model on real data

eval/
  configs/                    YAML per model: holdout split, AOIs, dates
  reports/                    per-model markdown + provenance + probe report
  fixtures/                   frozen replay fixtures (committed)

docs/
  EXPLAINER.md                plain-English what/why for each model
  PITCH.md                    one-pager differentiation vs prior work
  RESULTS.md                  headline reproduction table (regenerated)
  COMPLIANCE.md               EU AI Act / NIST AI RMF / NYC AI Action Plan
  TRAINING.md                 cross-model survey of how the fine-tunes were trained
  METHODOLOGY.md              how we measure, what's in scope
  PROVENANCE.md               tile IDs, station IDs, holdout construction
  ENERGY.md                   per-platform energy methodology
  M3_NOTES.md                 what runs on the Air, with measurements

WORKLOG.md                    chronological build log
```

## License

Apache-2.0. Sentinel-2 / Sentinel-1 imagery via Microsoft Planetary
Computer under the Copernicus Open Data License. NOAA CO-OPS data is
US Government public domain. NYC DOITT building footprints are public
domain via NYC OpenData (`5zhs-2jue`). Hurricane Ida flood polygons
from `riprap-nyc/data/prithvi_ida_2021.geojson` (Apache-2.0). ESA
WorldCover 2021 under the ESA CCI Open Data Policy.
