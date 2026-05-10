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

## What I have not yet verified end-to-end on M3

Each of the three model loaders raises `NotImplementedError` until the
matching extra is wired against the published adapter weights. The eval
modules detect this and write a "skipped" report, so a reviewer who
runs `riprap-models eval terramind-buildings` on a fresh checkout will
see an honest report explaining what's missing rather than a fabricated
metric. See `WORKLOG.md` for the slice that wires each adapter.

The expectations going in, based on parameter counts and prior MPS
experience in `riprap-nyc`:

| Model                       | Params | M3 expectation                         |
|---|---|---|
| TerraMind Buildings         | ~1 B (TerraMind 1.0 backbone + LoRA) | Should run on MPS in fp16 with the LoRA delta only kept in fp32. Tile of 224×224×12 is small; throughput should be >1 tile/s. |
| Prithvi-EO 2.0 NYC Pluvial  | ~300 M | Should run on MPS in fp16. Sen1Floods11-style 512×512 input is the heaviest patch; 16 GB unified should be sufficient with batch size 1. |
| Granite TTM r2 Battery Surge | ~1.5 M | Trivially fits. The CPU path is fast enough that MPS is not needed; the bench will likely show MPS within noise of CPU for this model. |

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
