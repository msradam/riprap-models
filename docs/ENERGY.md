# Energy methodology

`src/riprap_models/energy.py` exposes a single context manager,
`measure_energy()`, that picks the best available power-accounting method
for the runtime platform. The chosen method is reported in every report
under the `method:` field of the `measurements` YAML block, so a reader
can never confuse an estimate for a measurement.

## The four methods

| Method        | Where it works                                       | What it reads                                  |
|---|---|---|
| `nvml`         | Linux + NVIDIA GPU                                  | `nvmlDeviceGetPowerUsage` polled at ~50 Hz, integrated trapezoidally |
| `rapl`         | Linux + Intel x86 CPU                                | `/sys/class/powercap/intel-rapl:0/energy_uj` before/after |
| `powermetrics` | macOS, opt-in only (sudo without prompt)             | `sudo -n powermetrics --samplers cpu_power,gpu_power`, parsed for CPU + GPU mW |
| `estimated`    | Anything else                                        | `duration_s × platform_envelope_W` (envelopes below) |

The selector tries them in that order and falls back to `estimated` only
when nothing better is reachable. A reader can force a method via the
`prefer=` argument for testing.

## Why macOS defaults to `estimated`

`powermetrics` requires root privileges. We refuse to prompt the user
for sudo from inside a measurement context (an interactive prompt would
itself bias the wall-clock figure), so the default macOS path is the
estimated fallback. To get a real measurement on macOS, configure
sudoers to permit `powermetrics` without a password:

```
%admin ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
```

then re-run; the harness auto-detects the change and switches to
`powermetrics` on its own.

## Estimation envelopes

The estimated fallback multiplies wall-clock seconds by a conservative
platform-typical sustained power draw:

| Platform key    | Envelope (W) | Source                                                |
|---|---|---|
| `darwin-arm64`  | 12.0         | Apple M3 / M3 Pro Air sustained inference; observed 8–14 W package + DRAM in published reviews of MLX/PyTorch workloads on the M3 Air. Conservative midpoint. |
| `linux-x86_64`  | 35.0         | Generic desktop CPU package + DRAM under sustained inference. |
| `linux-aarch64` | 8.0          | Generic ARM SBC.                                      |
| _default_       | 25.0         | Catch-all when none of the above match.               |

These are envelopes for the **estimated** fallback only. When the
harness can read RAPL or NVML or powermetrics directly, the envelope is
not consulted.

## What the joules figure does not include

- LLM token streaming costs upstream of the model (e.g. the Granite 4.1
  reconciler in riprap-nyc; that is measured separately in
  riprap-nyc/app/energy.py).
- Network egress for fetching model weights or tiles.
- Cooling. The Air's chassis fan does not consume power; if you run
  this on a desktop, add fan + chassis power separately.

## Cross-checking against riprap-nyc

`riprap-nyc/README.md` reports approximately 0.03 Wh per query for
Granite 4.1:3b on commodity CPU (108 J). That figure was produced by a
prior version of this same `energy.py` module running in
`riprap-nyc/app/energy.py`. The methodology here is identical; the
joule-per-call values reported in `RESULTS.md` should be directly
comparable.
