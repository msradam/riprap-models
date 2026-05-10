# Headline reproduction table

This file is regenerated from `eval/reports/*.md` by `riprap-models report`.
Do not edit by hand. Each row reflects the most recent measurement on disk.

| Model | Card metric | Reproduced | Method | M3? | J/call |
|---|---|---|---|---|---|
| Prithvi-EO 2.0 NYC Pluvial | 0.5979 flood IoU | not yet measured | skipped (prithvi extra not installed (No module named 'terratorch')) | unknown | not yet measured |
| TerraMind Buildings | 0.5511 mIoU | not yet measured | skipped (terramind extra not installed (No module named 'terratorch')) | unknown | not yet measured |
| Granite TTM r2 Battery Surge | 0.1091 m MAE | 0.1318 m MAE | NOAA 8518750 hourly, sliding n=40 + named n=3 | yes (cpu fp32, ~3M params) | 0.2125 J (estimated, 17.7 ms) |
