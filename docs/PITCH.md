# Positioning vs prior work

Where the four NYC fine-tunes in this repository sit relative to existing
academic and industry work on NYC flood / building / land-cover machine
learning.

## Summary

Three things distinguish this family of fine-tunes:

1. **Compute floor.** Every model loads and runs on a 16 GB MacBook Air M3
   in CPU fp32. No GPU required for inference, no cloud account required.
2. **Open data and open code throughout.** Sentinel-2, Sentinel-1, NOAA
   CO-OPS, NYC OpenData, ESA WorldCover. All Apache-2.0 / CC-BY-4.0 /
   public domain. No proprietary or vendor-locked components.
3. **Reproduction harness ships with the models.** Every reported number
   is regenerable from public sources via a single command. A reader
   does not need to take any claim on faith.

## Comparison table

| Axis | Typical academic work | Typical commercial work | This family |
|---|---|---|---|
| Compute floor | GPU cluster | Cloud GPU API | MacBook Air M3 (CPU) |
| Data licensing | Mixed or unclear | Proprietary or vendor-locked | All open data |
| Code licensing | Often non-permissive | Closed | Apache-2.0 throughout |
| Reproducibility | Paper-quality | Vendor reports | Independent harness, every number regenerable |
| Energy disclosure | Rare | Absent | Per-call joules per platform, methodology in repo |
| Model granularity | One task, large model | One task, opaque model | Four NYC-specialised models, total ~3 GB |
| NYC specificity | Generic urban or non-NYC | Generic global | Trained on Riprap's Hurricane Ida polygons + NOAA Battery + DOITT footprints |

## Selected prior work

### NYC storm-surge forecasting

| Reference | Approach | What this family adds |
|---|---|---|
| [Stevens Institute, *Sci. Rep.* 2022](https://www.nature.com/articles/s41598-022-23627-6) | NYC surge ML at 57 sites, ADCIRC + SWAN ensemble feeding ML models | Foundation-model approach, 18 ms inference on a laptop, no cluster compute required |
| [npj Climate, 2023](https://www.nature.com/articles/s41612-023-00420-4) | Climate-change projection of NYC/NJ storm-surge hazards with ML | Operational nowcast rather than climate projection; runs against today's NOAA data |
| [Coastal Engineering, 2024](https://www.sciencedirect.com/science/article/abs/pii/S0378383924000802) | LSTM with anomaly correction for storm-tide forecast | Smaller (1.5M vs typical 10–100M) and CPU-runnable |
| [DeepSurge, 2025](https://arxiv.org/html/2506.13963) | National coastal storm-surge deep learning | NYC-specialised; energy disclosed; per-station fine-tune pattern is portable |

### Prithvi-based flood detection

| Reference | Approach | What this family adds |
|---|---|---|
| NASA / IBM / [ArcGIS docs](https://doc.arcgis.com/en/pretrained-models/latest/imagery/finetuning-the-prithvi-flood-segmentation.htm) | Generic Prithvi flood fine-tuning recipes | NYC-specific Hurricane Ida pluvial pattern, with copy-paste augmentation |
| [CUNY Graduate Center](https://sustainabilityandthecity.commons.gc.cuny.edu/2021/09/20/mapping-flooding-from-hurricane-ida/) | NYC Hurricane Ida flood mapping (manual + GIS) | Foundation-model fine-tune that runs on the same data |
| [ESS Open Archive](https://essopenarchive.org/doi/full/10.22541/essoar.15001292/v1) | Satellite-vs-terrain flood depth in NYC for Ida | Direct flood-extent segmentation from S2 with reproducible eval |

### TerraMind and EO LoRA

| Reference | Approach | What this family adds |
|---|---|---|
| [TerraMind paper, 2025](https://arxiv.org/html/2504.11171v1) | TerraMind 1.0 base; multi-modal generative pretraining | First published city-specific LoRA family on TerraMind |
| [PEFT for geospatial foundation models, 2025](https://arxiv.org/html/2504.17397v1) | First systematic study of LoRA on EO foundation models | Production deployment pattern for a single base + many city-specific adapters |
| [Google high-res buildings from Sentinel-2, 2023](https://research.google/pubs/high-resolution-building-and-road-segmentation-from-sentinel-2-imagery/) | Custom CNN architecture, single-modality | Multi-modal (S2 + S1 + DEM × 4 timesteps) with foundation-model backbone |

## Audiences and uses

- **City emergency-management agencies** can run the models on field
  laptops without procurement or cloud accounts. The TTM fine-tune
  drops into any storm-forecast workflow.
- **Civic-tech and open-data communities** can clone, run, and audit
  every claim against the same public data sources the city itself uses.
- **Climate research labs** get an open-source baseline that re-implements
  the entire pipeline from public data, useful as a starting point for
  iteration.
- **Insurance and underwriting workflows** can use the models for
  per-address exposure overlays without sending data to vendor LLMs.
- **NYC tech firms** working on geospatial layers have a reference
  implementation of the LoRA-on-TerraMind pattern for adding their own
  city-specific adapters.

## Out of scope

- Engineering-grade hydraulic flood modelling (use HEC-RAS or
  InfoWorks ICM).
- Structural fragility and per-building damage prediction.
- Replacing authoritative city or federal data sources (NYC DOITT, NOAA
  ETSS, FEMA flood maps). These models complement those sources rather
  than replace them.
