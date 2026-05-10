# Headline reproduction table

This file is regenerated from `eval/reports/*.md` by `riprap-models report`.
Do not edit by hand. Each row reflects the most recent measurement on disk.

| Model | Card metric | Reproduced | Method | M3? | J/call |
|---|---|---|---|---|---|
| Prithvi-EO 2.0 NYC Pluvial | 0.5979 flood IoU | 0.0806 chip-wide / 0.1150 vicinity flood IoU (within 300m of any GT polygon) | stride-7 reconstruction, n=29, 24 ida + 5 control | yes (cpu fp32, 324M params, ~10s/tile) | 2.57 J (estimated, 214 ms) |
| TerraMind Buildings | 0.5511 mIoU | 0.3288 mIoU default; 0.3653 building IoU at threshold 0.6 | 6 NYC AOIs, S2L2A+S1RTC+DEM 4 timesteps, DOITT labels | yes (cpu fp32, ~168M params multi-modal) | 6.13 J (estimated, 511 ms) |
| Granite TTM r2 Battery Surge | 0.1091 m MAE | 0.1318 m MAE all-windows; 0.3239 m MAE on storm windows (peak >=0.7m), -10% vs zero-shot | NOAA 8518750 hourly, sliding n=40 + named n=3 | yes (cpu fp32, ~3M params) | 0.2125 J (estimated, 17.7 ms) |
