# riprap-models

Reproduction harness for the three NYC fine-tunes used by
[riprap-nyc](https://github.com/msradam/riprap-nyc).

This repository exists to answer three questions about three model artifacts:

1. Do the headline numbers on each Hugging Face model card reproduce on
   held-out NYC data?
2. Do the models run on a 16 GB MacBook Air M3 (PyTorch MPS, no discrete GPU)?
3. What does one inference call cost in joules, and how was that measured?

The three models under test:

- `msradam/TerraMind-NYC-Adapters` — LULC, Tile-in-Mosaic, and Buildings
  LoRA adapters on TerraMind 1.0
- `msradam/Prithvi-EO-2.0-NYC-Pluvial` — Prithvi-EO 2.0 fine-tune for
  NYC pluvial flood segmentation
- `msradam/Granite-TTM-r2-Battery-Surge` — Granite TimeSeries TTM r2
  fine-tune for surge residual nowcasting at NOAA station 8518750

See `docs/RESULTS.md` for the headline reproduction table, `docs/M3_NOTES.md`
for what runs on the Air, and `docs/ENERGY.md` for the per-platform energy
methodology. `WORKLOG.md` is the chronological build log.

This is a reproduction harness, not a production inference path. For the
production FSM that consumes these models, see riprap-nyc.

## License

Apache-2.0.
