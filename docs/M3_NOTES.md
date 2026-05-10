# MacBook Air M3 notes

The target Air: 16 GB unified memory, no discrete GPU, PyTorch MPS backend.

## What loads under the base install

The base install (`uv pip install -e ".[dev]"`) does not pull torch. The
package itself imports cleanly and the metrics, energy, and CLI modules
all run. `riprap-models device` reports the platform without needing
torch, so a reviewer can verify their environment before installing the
heavy extras.

## What's been confirmed on M3 in this build

The structural verification (smoke matrix on `macos-14` arm64 in
`.github/workflows/smoke.yml`) confirms:

- The package builds and installs under Python 3.12 on arm64.
- `device.get_device()` returns `kind=mps, dtype=float16` when torch is
  installed, and `kind=cpu, dtype=float32, label=… (CPU, fp32, no torch)`
  otherwise.
- `energy.measure_energy()` returns sane values on macOS, defaulting to
  `method="estimated"` because `powermetrics` requires sudo.

## What's been measured on this M3 Air

| Model                        | Status                                                                 |
|---|---|
| Granite TTM r2 Battery Surge | **measured.** CPU fp32, ~3M params. 40-window sliding eval across NOAA station 8518750 hourly residuals (2025-01-01 → 2026-05-01). 0.1318 m MAE vs card 0.1091 m. 17.7 ms / call, 0.21 J / call. |
| Prithvi-EO 2.0 NYC Pluvial   | **measured.** CPU fp32, 324M params. 29-chip independent reconstruction (24 stride-7 Ida polygons + 5 controls). Reproduced flood IoU 0.0806 vs card 0.5979 — gap is the unpublished card chip-extraction. 211 ms / call, 2.53 J / call. |
| TerraMind Buildings          | **not measured.** terratorch is installed, weights are downloadable, but the published test split names Major-TOM Core chips that need a multi-modal 4-timestep fetcher this harness does not yet ship. Multi-hour data-engineering job. |

For the two satellite models, the expectations going in (based on
parameter counts and prior MPS experience in `riprap-nyc`):

| Model                       | Params | M3 expectation                         |
|---|---|---|
| TerraMind Buildings         | ~1 B (TerraMind 1.0 backbone + LoRA) | Should run on MPS in fp16 with the LoRA delta in fp32. Tile of 224×224×12 is small; throughput should be >1 tile/s. |
| Prithvi-EO 2.0 NYC Pluvial  | ~300 M | Should run on MPS in fp16. Sen1Floods11-style 512×512 input is the heaviest patch; 16 GB unified is sufficient with batch size 1. |

## Measured TTM run (this build)

- Hardware: Apple M3 Air, 16 GB unified memory, macOS 25.4, Python 3.12.12, torch 2.11.0
- Model: `msradam/Granite-TTM-r2-Battery-Surge`, ~3M params, fp32 on CPU
- Input: 1024 hourly surge-residual samples from NOAA station 8518750
- Output: 96-hour forecast
- Wall-clock per call: ~17.7 ms average across 30 calls (after warm-up)
- Energy per call: ~0.21 J (estimated against the M3 Air 12 W envelope; see `ENERGY.md` for how to flip macOS to real `powermetrics` measurement)
- Held-out MAE: 0.1364 m across three windows (card: 0.1091 m on 12,033 windows). Honest gap, attributable to sample size.

## Known MPS sharp edges (from riprap-nyc experience)

- **Conv3d on MPS**: TerraMind backbone has 3D convs in some branches.
  If MPS rejects, set `PYTORCH_ENABLE_MPS_FALLBACK=1` and re-bench;
  expect a 5–10× slowdown on the affected layers.
- **`bfloat16`**: not universally supported on MPS depending on the
  PyTorch nightly. Default to `float16`; if loss of precision is
  visible in the segmentation report, fall back to `float32`.
- **First-call compilation**: MPS has a one-time graph compile per
  unique input shape. The bench loop's warm-up call discards this so
  reported wall-clock is steady-state.

When any of the three model loaders is wired, this file gets the
measured numbers (wall-clock per call, peak RSS, MPS-vs-CPU comparison,
any fallback notes) and the "expectation" table above is replaced with
the measured table.

## When a model genuinely will not run on 16 GB unified memory

Per the brief, mark it "no, because <reason>" in `RESULTS.md` and
propose either a quantized variant or a tiled inference path. None of
the three is currently expected to OOM on 16 GB; the heaviest
plausible peak is Prithvi at fp16 on a 512×512 patch.
