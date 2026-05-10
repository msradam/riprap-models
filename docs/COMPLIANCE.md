# AI compliance posture

What this repo gives a procurement / legal / risk team for free, mapped to
the regulations and frameworks that actually apply to municipal and
public-sector AI deployment in 2026.

This is internal-facing. Not legal advice. The point is: when an
agency reviewer asks "can we deploy this model?", the answer is in
this repo, not in slideware.

## Frameworks this aligns to

| Framework | Status (mid-2026) | What it requires from a deployed model |
|---|---|---|
| **EU AI Act** | In force, high-risk obligations phased in through Aug 2027 | Risk assessment, technical documentation, data governance, transparency, human oversight, accuracy/robustness/cybersecurity, post-market monitoring |
| **NIST AI Risk Management Framework (AI RMF 1.0)** | Voluntary, broadly adopted | Map / Measure / Manage / Govern functions; trustworthiness characteristics |
| **NYC Local Law 144 (auto employment decisions)** | In force since 2023 | Bias audit; not directly applicable to flood models, but sets the procurement-team mental model |
| **NYC AI Action Plan (Oct 2023)** | Citywide guidance | Documentation, transparency, equity review for any city-deployed AI |
| **OMB M-24-10 (federal AI use)** | In force; impacts FEMA / NOAA federal-grant context | Algorithmic impact assessment; minimum risk practices for "rights-impacting" and "safety-impacting" AI |
| **EO 14110 (revoked Jan 2025) → EO 14179 (current)** | Federal | Reduced federal-side disclosure requirements vs EO 14110, but agencies retain their own |
| **DHS/CISA AI Roadmap** | Guidance | Open-source preference for emergency-management AI; supply-chain transparency |

## What this repo provides, item by item

### Provenance and reproducibility

- Every reported number is regenerable from public sources
  (NOAA CO-OPS, Microsoft Planetary Computer, NYC OpenData). No
  paywalled APIs, no proprietary test sets.
- Every report under `eval/reports/` carries a JSON provenance block
  with: tile/station IDs used, timestamp of capture, code SHA at
  generation time, platform.
- The exact SHA pinning means a reviewer can `git checkout <sha>`
  and re-run the same eval; bisection of any drift is supported.
- Maps to: **EU AI Act Art. 10 (data governance)**, **NIST AI RMF
  Measure 2.7 (data and input integrity)**, **OMB M-24-10
  algorithmic impact assessment data lineage**.

### Open weights, open data, open code

- All three models are Apache-2.0. So is this harness. So is each
  base model (TerraMind 1.0, Prithvi-EO 2.0, Granite TTM r2).
- All evaluation data is under a permissive open data license:
  Sentinel-2 / Sentinel-1 (Copernicus Open Data), NOAA CO-OPS
  (US Government public domain), NYC DOITT building footprints
  (NYC OpenData public domain), Hurricane Ida polygons (Apache-2.0
  via riprap-nyc).
- No vendor LLM is contacted at inference time.
- Maps to: **DHS/CISA open-source preference**, **EU AI Act Annex IV
  technical documentation transparency**, **NIST AI RMF Govern 6.1
  (third-party software supply chain)**.

### Energy and environmental disclosure

- `docs/ENERGY.md` documents the per-platform energy methodology.
- Each model report carries a measured (or honestly-labeled
  estimated) joules-per-call number.
- No "energy is too hard to measure" hand-waving. The fallback to
  estimated is explicit and the estimation envelope is sourced.
- Maps to: **EU AI Act Annex IV §2(g) (energy use disclosure for
  general-purpose AI models)**, **EU CSRD (corporate sustainability
  reporting)** alignment for downstream city procurement.

### Honest accuracy reporting

- `docs/RESULTS.md` reports the reproduced accuracy, not the model
  card claim, and names the gap when there is one.
- Per-model reports include per-tile / per-window detail so a
  skeptical reviewer can see which slices of the evaluation drove
  the headline number.
- For Prithvi specifically, the report explains that the card's
  0.5979 IoU is conditional on a chip-extraction recipe not in the
  public artifacts, and reports both chip-wide and polygon-vicinity
  IoU so the user can pick the metric that fits their use case.
- Maps to: **EU AI Act Art. 13 (transparency)**, **NIST AI RMF
  Measure 2.5 (accountable measurement)**, **NYC AI Action Plan
  honesty-about-limits principle**.

### M3 / commodity-hardware runnable

- All three models load and run on a 16 GB MacBook Air M3 in CPU
  fp32. No GPU dependency for inference.
- Means: a city emergency manager doing a tabletop exercise on a
  laptop, a journalist working from home, a researcher at a desk —
  none need cloud GPU credits to verify or use these models.
- Maps to: **EU AI Act Art. 10 (proportionality of computing
  resources)**, **DHS/CISA equity-of-access guidance**.

### No training-data PII

- Sentinel-2 / Sentinel-1 imagery is below the resolution where
  individual-person identification is possible (10 m / pixel).
- NOAA CO-OPS station data is aggregate water level, no person-level
  data.
- NYC DOITT building footprints are public records of physical
  structures, not occupancy data.
- Maps to: **GDPR Art. 9 (special category data)**, **EU AI Act
  Art. 10(5) (sensitive data minimization)**.

### Post-market / deployed-model monitoring

- The harness is designed to be re-run on a schedule. The CLI's
  `riprap-models eval <model>` regenerates the whole accuracy
  report; `riprap-models bench <model>` regenerates the energy
  number; `riprap-models report` regenerates the headline table.
- A deployer can wire any of these into a daily / weekly cron and
  alert on accuracy or energy drift.
- Maps to: **EU AI Act Art. 72 (post-market monitoring system)**,
  **NIST AI RMF Manage 4.1 (monitoring deployed systems)**.

## What this repo does not provide

Stating these explicitly so a procurement reviewer is not surprised.

- **Bias audit in the LL144 sense.** These are flood and building
  segmentation models, not employment-decision models, so a
  disparate-impact analysis on protected classes is not
  immediately applicable. A targeted analysis (e.g. does the model
  systematically under-predict flood in low-income neighborhoods?)
  would be a useful add-on but is not currently in scope.
- **Adversarial robustness testing.** No FGSM / PGD / corruption
  benchmarks. The models' inputs are public satellite imagery and
  NOAA tide data, where adversarial inputs are not the realistic
  threat model.
- **Formal model card per the original Mitchell et al. spec.**
  The HF model cards are close to that spec; this repo's reports
  add the reproduction layer on top.
- **Watermarking of model outputs.** The output is a binary
  segmentation mask or a numeric forecast, not generative content
  that requires AI-output disclosure under California SB 942 or
  similar.
- **Continuous accuracy monitoring in production.** The harness
  supports this (the CLI is designed for it) but the cron / alert
  wiring is the deployer's job, not the model's.

## Procurement summary, one sentence per row

For a procurement intake form:

| Question | Answer |
|---|---|
| License | Apache-2.0 across all three models, the harness, and every evaluation dataset. |
| Vendor lock-in | None. Models run locally; harness has no vendor SDK dependencies. |
| Data residency | All inference is local; no data leaves the deployer's environment. |
| Reproducibility | Every reported number is regenerable from public sources via `riprap-models eval <name>`. |
| Energy disclosure | Per-call joules reported per model, methodology in `docs/ENERGY.md`. |
| Hardware floor | 16 GB MacBook Air M3 (CPU, no GPU). |
| Accuracy honesty | Each report names the gap to the card claim and explains the cause. |
| Maintenance | Open source, bug reports + PRs at https://github.com/msradam/riprap-models. |
