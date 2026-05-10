# WORKLOG

Chronological build log. Newest entry on top. Each entry: what was built,
what was measured, what failed, what was decided.

## 2026-05-10 — TTM Battery Surge wired and measured on M3

After the initial skeleton landed, I noticed the build environment was
the user's actual M3 Air. So instead of stopping at "skeleton plus
skipped reports", I wired the smallest of the three loaders end to end
and produced real numbers.

### What changed

- `ttm_battery_surge/data.py`: replaced the `NotImplementedError` with
  a real `_TTMForecaster` that wraps `tsfm_public.get_model(...)`. The
  call surface mirrors `riprap-nyc/app/live/ttm_forecast.py` (channel-
  wise standardize, `past_values=(1, T, 1)`, de-standardize).
- Added hourly-bucket resampling in `fetch_residual_series`. The
  fine-tune was trained on hourly data with a 1024-step context, not
  6-min/512 like the live nowcast in riprap-nyc.
- Added 31-day chunking in `_coops_get` so a 43-day history fetch
  stops blowing the NOAA water_level endpoint's window cap.
- `eval/configs/ttm_battery_surge.yaml`: cadence + context updated to
  match the fine-tune (1024 hourly in, 96 hourly out). Three holdout
  windows constructed from outside the documented training range.
- `eval.py`: added zero-shot baseline (`ibm-granite/granite-timeseries-ttm-r2`)
  alongside the fine-tune and persistence baselines, mirroring the
  three-row table on the model card. Also fixed the bench to update
  the YAML measurements block in place so `RESULTS.md` picks up the
  joules figure automatically.

### Measured numbers

Apple M3 Air, 16 GB unified, macOS 25.4, Python 3.12.12, torch 2.11.0:

| Model                       | Card metric | Reproduced (3 windows) | Notes |
|---|---:|---:|---|
| Granite TTM r2 Battery Surge | 0.1091 m MAE | 0.1364 m MAE | Card was 12,033 sliding windows; this is 3 windows. Honest gap. |

Per-window breakdown lives in `eval/reports/ttm_battery_surge.md`.
Bench (30 calls, post-warm-up): 17.7 ms / call, 0.21 J / call
(estimated against the 12 W M3 Air envelope; see `ENERGY.md` for the
sudoers one-liner that swaps to real `powermetrics`).

### Honest finding

On the calm fair-weather window, **persistence beats both TTM
variants** (MAE 0.067 m vs fine-tune 0.098 m, zero-shot 0.077 m).
On the December 2024 nor'easter, **zero-shot beats the fine-tune**
(0.154 m vs 0.174 m). Only on the Feb 2026 post-training window does
the fine-tune lead. Three windows is a small N; the headline 0.1091 m
in the card was averaged over twelve thousand windows and reflects
that scale. This repo's first job is to make the asymmetry visible,
not to hide it.

### What still isn't wired

The two satellite loaders (`load_buildings_adapter`,
`load_pluvial_finetune`) still raise `NotImplementedError`. The
canonical reference is `riprap-nyc/scripts/run_prithvi_ida.py` for
Prithvi; for TerraMind, the published model card kwargs need pinning
against `terratorch>=0.10`. Each is a fifteen to thirty minute job
on a GPU box.

## 2026-05-10 — initial autonomous build (Adam offline studying for finals)

### Phase 0: orient

Read `riprap-nyc/README.md`, `ARCHITECTURE.md` (relevant sections on
TerraMind, Prithvi-EO 2.0, TTM r2), and `CLAUDE.md` for voice + critical
constraints. Read `app/live/ttm_forecast.py` for the canonical TTM call
signature (512-step context, 96-step horizon at 6-minute cadence,
station 8518750).

### Findings vs the brief

The brief assumed paths that don't exist in `riprap-nyc`:

- `experiments/18..21/` → actual repo has `experiments/00..10`. Mapped
  TerraMind → `04_terramind_synthetic_sar`, `05_terramind_nyc_finetune`,
  `05a_terramind_finetune_micro`; Prithvi → `01_prithvi_live_water`;
  TTM Battery Surge → `app/live/ttm_forecast.py` + `ARCHITECTURE.md §7.2`
  (no dedicated experiment dir).
- `docs/BENCHMARKS.md` and `docs/EMISSIONS.md` referenced in the brief
  do not exist in `riprap-nyc`. Voice constraints taken from the brief
  itself (no em-dashes, no banned words, etc.) and from
  `riprap-nyc/CLAUDE.md`.
- `services/riprap-models/`, `inference-vllm/`, `Dockerfile.l4` referenced
  in the brief: not present. Used the root `Dockerfile` and `app/llm.py`
  pinning notes as the dependency reference.

### Phase 1: scope and structure

Created `~/riprap-models`, `gh repo create msradam/riprap-models`,
seeded `main` with LICENSE / NOTICE / .gitignore / README stub.
Branched `eod-build` for all subsequent slices.

### Slice 1 — skeleton + CI

- `pyproject.toml` (uv-managed, Python 3.12, hatchling backend, model
  extras for `terramind` / `prithvi` / `ttm` / `live` / `dev`)
- `src/riprap_models/` package layout matching the brief
- `.github/workflows/smoke.yml` matrix on `ubuntu-latest` + `macos-14`
- `.github/workflows/benchmark.yml` for tag-triggered heavy bench
- `tests/test_smoke.py` confirms the package imports under base install

What works on this slice: `uv pip install -e ".[dev]"`, `pytest -q`,
`riprap-models --help`.

### Slice 2 — device + energy plumbing

- `device.py`: cuda → mps → cpu detection; dtype rules per device;
  Apple Silicon detection via `sysctl machdep.cpu.brand_string`;
  graceful no-torch fallback so smoke tests pass without torch.
- `energy.py`: four methods auto-selected by platform.
  - `nvml`: pynvml polled at 50 Hz, trapezoid-integrated.
  - `rapl`: `/sys/class/powercap/intel-rapl:0/energy_uj` before/after.
  - `powermetrics`: macOS, opt-in via passwordless sudo; parses CPU+GPU
    mW from text output; falls back to estimated if no samples parse.
  - `estimated`: documented platform envelopes (M3 Air = 12 W package
    midpoint; sources in `docs/ENERGY.md`).
- `tests/test_device.py` + `tests/test_energy.py` cover the cross-platform
  behaviour (method ∈ {nvml, rapl, powermetrics, estimated}; duration
  positive; peak RSS positive; estimated-fallback path exercised).

Decision: macOS defaults to `estimated`, not `powermetrics`. Reason:
prompting for sudo from inside a measurement context biases the
wall-clock figure. Documented the sudoers entry that flips macOS to
real `powermetrics` measurement in `docs/ENERGY.md`.

### Slice 3 — metrics

- `common/metrics.py`: confusion matrix, per-class IoU, macro/micro
  mIoU, pixel accuracy, ignore_index handling, regression MAE+RMSE,
  persistence forecast.
- `tests/test_metrics.py`: hand-computed expected values for each
  metric, including the absent-class NaN case and the ignore_index
  exclusion case.

### Slice 4 — TerraMind Buildings

- `terramind/eval.py`: full eval + bench loop. Uses `iter_holdout_tiles`
  from `data.py`, accumulates a global confusion matrix, writes a
  measured report with provenance + per-class IoU + a YAML
  `measurements` block that `report.py` reads.
- `terramind/data.py`: manifest reader (CSV: tile_id,image_path,
  label_path), `dummy_input` for warm-ups, `load_buildings_adapter()`
  raises `NotImplementedError` until the terratorch loader kwargs are
  pinned against the published adapter weights.
- `eval/configs/terramind_buildings.yaml`: 12-band Sentinel-2 input,
  224 tiles, 2 classes, ignore_index 255.

Status: code path complete. `riprap-models eval terramind-buildings`
on a fresh checkout writes a "skipped" report explaining the missing
extra. Real numbers pending GPU access + adapter-loader wiring.

### Slice 5 — Prithvi Pluvial

- Same shape as Slice 4. Adversarial split has three kinds: `ida`,
  `sandy`, `control`. Report breaks down flood IoU by kind so a
  reviewer can see the model handles non-event tiles correctly.
- `eval/configs/prithvi_pluvial.yaml`: 6-band 512-tile Sen1Floods11-style.

### Slice 6 — TTM Battery Surge

- Three held-out windows in the config: 2024 nor'easter, 2024 fair-weather,
  2021 Hurricane Ida remnants. Documented why the windows do not overlap
  the training data in `docs/PROVENANCE.md`.
- `ttm_battery_surge/data.py`: NOAA CO-OPS fetcher (no auth required;
  `water_level` minus `predictions` = surge residual). `_ZeroForecaster`
  fallback so the bench path returns real timing numbers without
  weights, but the eval is obviously wrong (huge MAE) so it cannot be
  mistaken for a measurement.

### Slice 7 — live data path

- `live.py`: per-model `run_live(name, fixtures_dir)` and `replay(name,
  fixture_dir)`. TTM live works without any extras: pulls last 96 hours
  of NOAA data, forecasts (zero-stub if extras absent), saves to
  `eval/fixtures/ttm-battery-surge/<UTC-ts>/{inputs.npz, outputs.npz,
  manifest.json}`. Prithvi + TerraMind live use Microsoft Planetary
  Computer (`pystac-client`, `planetary-computer`); both write a
  manifest even without `[live]` extra so a reviewer can see the path.

### Slice 8 — M3 benchmark pass

- `docs/M3_NOTES.md` written from the build environment's perspective:
  what the smoke matrix (macos-14 arm64 runner) can confirm structurally,
  what needs a real M3 + extras to measure. Documents the expected
  per-model behaviour on M3 from `riprap-nyc` experience (Conv3d MPS
  fallback, fp16 vs bfloat16, first-call compile).

### Slice 9 — CLI

- `cli.py`: `device`, `eval`, `bench`, `run-live`, `replay`, `report`.
  All six wired and in the smoke matrix via `riprap-models --help`.

### Slice 10 — Dockerfiles

- `Dockerfile`: Linux/CUDA, ubuntu 22.04, CUDA 12.4, Python 3.12, all
  model extras. Pins match riprap-nyc's transformers + huggingface_hub
  range so adapter loading does not silently regress.
- `Dockerfile.mps`: arm64 parity image. MPS is not exposed inside Docker;
  the image runs on CPU. Documented in the file header that real M3
  measurement needs `uv` directly on the host, not Docker.
- `docker-compose.yml`: both services with persistent HF cache volume.

### Slice 11 — docs

- `docs/METHODOLOGY.md`: what we measure, what "held-out" means,
  what we do on a failed reproduction, what's out of scope.
- `docs/RESULTS.md`: headline table. Currently every cell is "not yet
  measured" because the loader wire-ups raise `NotImplementedError`
  pending GPU access; the YAML `measurements` blocks will populate
  the table automatically the first time `riprap-models eval` returns
  a real number.
- `docs/ENERGY.md`: the four methods, why macOS defaults to estimated,
  the envelope numbers and their sources, what the joules figure does
  not include.
- `docs/PROVENANCE.md`: tile ID / station ID / window provenance per
  model, plus the construction-bias caveat for any model whose
  upstream training split was thin.
- `docs/M3_NOTES.md`: above.

### Process notes for the reviewer

- One PR open against `main` from `eod-build`. Slicing into 11 stacked
  PRs was de-prioritized in favour of getting more substantive content
  shipped in the available time. Per-slice attribution is preserved in
  this WORKLOG and in the commit history (one commit per slice; `git
  log --oneline` reads as a slice list).
- The biggest remaining wire-up: `load_buildings_adapter()`,
  `load_pluvial_finetune()`, and `load_finetune()` in the three
  `data.py` modules. Each `NotImplementedError` names the canonical
  reference in `riprap-nyc` to consult. Fifteen to thirty minutes of
  GPU-attached work per model should land real numbers in `RESULTS.md`.

## Honest "done definition" checklist (per the brief)

1. **WORKLOG.md exists and explains what happened** — yes (this file).
2. **RESULTS.md has the headline table** — yes, populated with "not yet
   measured" for the three rows that need GPU wire-up. The table format
   and the regenerator (`riprap-models report`) are in place.
3. **M3_NOTES.md says what runs on the Air** — yes, with the caveat
   that the structural verification is from `macos-14` CI; full
   numerical bench needs an Air with the model extras installed.
4. **ENERGY.md explains how joules were measured per platform** — yes.
5. **`riprap-models replay <name>` works** — yes for `ttm-battery-surge`
   end-to-end (no extras required, since the NOAA path needs no
   auth and the `_ZeroForecaster` fallback is deterministic). For the
   two satellite models, replay needs the `live` + `prithvi` /
   `terramind` extras.
6. **PRs in order with the diff that produced each measurement** —
   one PR contains the eleven slices as eleven commits; per the brief's
   decision rule, this trade-off (one PR vs eleven stacked PRs) was
   made to ship more content in the available time.
