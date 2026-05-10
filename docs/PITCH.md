# What's differentiated about this work

A one-pager you can hand to anyone who asks "why these three models, why
now, why riprap."

## The bet

Three foundation models, each fine-tuned on NYC, each running on a
laptop. That combination is rare. The published academic and industry
work on NYC flood ML usually picks one axis (model size, training
data, deployment target) and the other two suffer.

| Axis | Typical academic work | Typical commercial work | Riprap |
|---|---|---|---|
| **Compute floor** | GPU cluster | Cloud GPU API | **MacBook Air M3 (CPU)** |
| **Data licensing** | Mixed / unclear | Proprietary / vendor lock | **All open data** |
| **Code licensing** | Often non-permissive | Closed | **Apache-2.0 throughout** |
| **Reproducibility** | Paper-quality | Vendor reports | **Independent harness, every number regeneratable** |
| **Energy disclosure** | Rare | None | **Per-call joules per platform, methodology in repo** |
| **Model granularity** | One task, big model | One task, opaque model | **Three NYC-specialised models, total ~500M params** |
| **NYC specificity** | Generic urban or non-NYC | Generic global | **Trained on Riprap's Hurricane Ida polygons + NOAA Battery + DOITT footprints** |

## The three differentiators that compound

### 1. Efficiency

- TTM Battery Surge: **1.5M params**, 18 ms / call, 0.21 J / call.
  Compare: prior NYC surge ML work uses LSTMs and ADCIRC + SWAN
  ensemble runs that need GPU clusters.
- Prithvi NYC Pluvial: **324M params**, 211 ms / call. Runs on M3
  CPU. Compare: typical Prithvi flood work assumes A100 / H100.
- TerraMind Buildings: **~168M effective params** (1B base + LoRA),
  511 ms / call. The LoRA approach itself is the efficiency story:
  one base model on disk, three small adapters.

Why this matters: a journalist, a city planner, an emergency manager,
a researcher — none of them need cloud GPU credits to verify or use
these. The whole stack runs on what they already own.

### 2. Open data + open code + open weights

Every model is Apache-2.0. Every input dataset is open data:

- Sentinel-2 / Sentinel-1 (Copernicus Open Data via Microsoft
  Planetary Computer)
- NOAA CO-OPS station 8518750 (US Government public domain)
- NYC DOITT building footprints (NYC OpenData public domain)
- Hurricane Ida polygons (Apache-2.0, derived from FEMA HSDC)
- Copernicus DEM GLO-30 (open)

Why this matters: a public agency or non-profit can deploy without
license review or vendor procurement. A researcher can publish results
without IRB / data-use-agreement friction.

### 3. Compliance posture is built in

`docs/COMPLIANCE.md` maps the repo to:

- EU AI Act (in force, high-risk obligations through Aug 2027)
- NIST AI Risk Management Framework
- NYC AI Action Plan (Oct 2023)
- OMB M-24-10 (federal AI use, FEMA / NOAA grant context)
- DHS / CISA AI Roadmap (open-source preference for emergency
  management)

What this gives a procurement team for free:

- **Data lineage**: every report has a JSON provenance block with
  tile IDs, station IDs, code SHA, platform.
- **Energy disclosure**: per-call joules with methodology.
- **Honest accuracy**: each report names the gap to the card claim
  and explains it. No "best-case" cherry-picking.
- **Local inference**: no data leaves the deployer's environment.
- **Algorithmic impact assessment material**: the harness doubles as
  the documentation a city or agency needs to file under most AI
  governance frameworks.

## Where this slots in

### Public sector / non-profit

- **NYC Mayor's Office of Emergency Management**: tabletop exercises,
  field briefings, situational awareness during storms. Locally-
  runnable means a planner can take this to the field with a laptop.
- **NYC DEP stormwater planning**: building footprint extraction at
  scale for impervious-surface mapping; pluvial flood priors for the
  8000+ sub-catchments.
- **FEMA / NOAA grant-funded research**: the open-source compliance
  posture is what M-24-10 wants.
- **CDBG-DR resilience programmes**: the per-address exposure
  briefings Riprap produces feed directly into hazard mitigation
  planning.

### Newsrooms

- **NYT, ProPublica, The City**: flood reporting where the
  newsroom can show its work. Newsroom legal can sign off on
  Apache-2.0 + open data; cannot sign off on vendor LLMs that
  ingest FOIA'd documents.
- **Local TV / digital outlets**: TTM Battery Surge run as a
  morning forecast tool. 18 ms per call means embedded in any
  publishing workflow.

### Insurance / underwriting

- **Marsh, AIG, Lloyd's NYC**: per-address flood-exposure briefings
  without sending data to vendor LLMs. The "runs locally" angle is
  genuinely differentiated for insurance with broker-confidential
  data.

### Climate research

- **Columbia Climate School, NYU CUSP, Stevens DAVI**: an
  open-source baseline they can iterate on without re-implementing
  the full chain.

### Civic tech

- **BetaNYC, NYC Planning Labs, Open New York**: open-source NYC
  flood-data infrastructure. The HF model cards pointing back at
  this reproduction harness make it easy for civic tech volunteers
  to get involved.

## What this is not

- **Not** a real-time hydraulic flood model. Use HEC-RAS or
  InfoWorks ICM for engineering-grade flood vulnerability.
- **Not** a structural fragility tool. Building IoU is recall-
  biased; consumers should treat outputs as candidates for human
  review, not authoritative.
- **Not** a vendor product. There is no SLA, no support team, no
  SaaS dashboard. There is a repo, and a maintainer.

## Where the prior art doesn't reach

| Prior work | What they did | What they didn't do |
|---|---|---|
| [Stevens 2022 (Nature)](https://www.nature.com/articles/s41598-022-23627-6) | NYC surge ML at 57 sites, ADCIRC + SWAN ensemble | Need cluster compute; not deployable on a laptop |
| [DeepSurge 2025](https://arxiv.org/html/2506.13963) | National coastal storm-surge deep learning | Generic, not NYC-specialised; no energy disclosure |
| [LSTM storm surge 2024](https://www.sciencedirect.com/science/article/abs/pii/S0378383924000802) | LSTM + tide-gauge anomaly correction | Single-task, no foundation-model framework |
| Prithvi flood docs (NASA / IBM / ArcGIS) | Generic Prithvi flood fine-tuning | Not NYC-specific; no held-out reconstruction harness |
| [Google high-res buildings 2023](https://research.google/pubs/high-resolution-building-and-road-segmentation-from-sentinel-2-imagery/) | Buildings from S2 with custom CNN | Not foundation-model based; not multi-modal |
| [TerraMind paper (Apr 2025)](https://arxiv.org/html/2504.11171v1) | TerraMind 1.0 base | No published city-specific LoRA fine-tunes |
| [PEFT for geospatial (2025)](https://arxiv.org/html/2504.17397v1) | First systematic LoRA-on-EO study | No NYC-specific applications |

The combination — three NYC-specialised foundation-model fine-tunes,
all open, all M3-runnable, with an independent reproduction harness
and a procurement-ready compliance doc — is genuinely first-mover.

## One sentence to remember

> Three NYC foundation-model fine-tunes that fit in a laptop, ship
> with their own reproducibility harness, document their own energy
> cost, and clear EU AI Act / NIST AI RMF / NYC AI Action Plan
> documentation requirements out of the box.
