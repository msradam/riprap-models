# TerraMind Buildings

**Status:** not evaluated in this environment.

**Reason:** loader runtime is installed (terratorch), but the 32-chip test split listed at the model repo (buildings_nyc/splits/test.txt) names Major-TOM Core chips like 'nyc_452U_625L_r0c0' that this harness does not yet reconstruct. The adapter expects S2L2A+S1RTC+DEM at 4 timesteps; the local riprap-nyc cache ships single-timestep chips. The wire-up is one Major-TOM fetcher away. See docs/M3_NOTES.md for the gap.

Card metric (from `msradam/TerraMind-NYC-Adapters` README): 0.5511 mIoU on held-out NYC tiles.

```yaml measurements
model: TerraMind Buildings
card_metric: "0.5511 mIoU"
reproduced: "not yet measured"
method: "skipped (loader runtime is installed (terratorch), but the 32-chip te)"
m3: "unknown"
j_per_call: "not yet measured"
```
