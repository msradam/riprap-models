# Training origins

How the four NYC fine-tunes consumed by this repo were actually trained.
Source code for all of it lives in
[`riprap-nyc/experiments/`](https://github.com/msradam/riprap-nyc/tree/main/experiments).
Per-model deep-dives are in each model's own GitHub repo.

## Cross-cutting

| | |
|---|---|
| Hardware | 1× AMD Instinct MI300X (192 GB HBM3) on AMD Developer Cloud |
| ROCm | 4.0.0+1a5c7ec |
| Python | 3.12 |
| Random seed | 42 (every fine-tune) |
| License | Apache-2.0 across all checkpoints |

Three persistent containers on the droplet (sibling of one another from
the same `rocm:latest` image):

- `terramind` — for TerraMind LoRA adapters (Phases 13, 18) and Prithvi
  fine-tunes (Phase 14, 19), uses TerraTorch 1.2.7 + Lightning 2.6.1.
- `ttm` — for Granite TTM r2 fine-tune (Phase 16, 20), uses
  granite-tsfm + transformers 4.55.x.
- `rocm` — clean sibling for future work.

Each container runs end-to-end reproducibly: clone riprap-nyc, run the
phase-specific command, get the same checkpoint back.

## Per-model

### Granite TTM r2 Battery Surge (Phase 20)

Source: [`riprap-nyc/experiments/20_ttm_battery_surge/`](https://github.com/msradam/riprap-nyc/tree/main/experiments/20_ttm_battery_surge).
Per-model TRAINING.md: [Granite-TTM-r2-Battery-Surge/docs/TRAINING.md](https://github.com/msradam/Granite-TTM-r2-Battery-Surge/blob/main/docs/TRAINING.md).

| | |
|---|---|
| Base | `ibm-granite/granite-timeseries-ttm-r2` (1.5M params) |
| Data | NOAA CO-OPS station 8518750, 2015-2024 (10 years), hourly resampled |
| Training samples | 60,251 sliding 1024→96 hourly windows |
| Loss | MSE on standardized residuals |
| Optimizer | AdamW, lr 1e-4, batch 64, 10 epochs |
| Wall-clock | **~5 minutes** on MI300X |
| Test result | MAE 0.1091 m on 12,033 test windows (vs persistence 0.1861, zero-shot 0.1467) |

### Prithvi-EO 2.0 NYC Pluvial v2 (Phase 19)

Source: [`riprap-nyc/experiments/19_prithvi_nyc_v2/`](https://github.com/msradam/riprap-nyc/tree/main/experiments/19_prithvi_nyc_v2).
Per-model TRAINING.md: [Prithvi-EO-2.0-NYC-Pluvial/docs/TRAINING.md](https://github.com/msradam/Prithvi-EO-2.0-NYC-Pluvial/blob/main/docs/TRAINING.md).

| | |
|---|---|
| Base | `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11` (300M params) |
| Data | Riprap's 166 baked Hurricane Ida 2021 polygons + 332 copy-paste augmented + 286 clear-sky NYC negatives = **784 chips** |
| Augmentation | Simple Copy-Paste (Ghiasi et al. CVPR 2021) with Gaussian-feathered alpha blending on polygon edges |
| Loss | Lovász-Softmax (direct IoU surrogate; replaced Dice in v1) |
| Optimizer | AdamW, lr 3e-5, batch 8, max 60 epochs (early-stop on val/loss) |
| Wall-clock | **~2 hours** on MI300X |
| Test result | mIoU 0.7974, **flood IoU 0.5979** (vs v1's flood IoU 0.10) |

The v1 → v2 lift (5.4× on flood IoU) came from copy-paste augmentation
+ Lovász-Softmax loss + 3× learning rate + 60 epochs. The augmentation
is the single biggest contributor.

### TerraMind NYC Adapters: Buildings + LULC + TiM (Phase 18)

Source: [`riprap-nyc/experiments/18_terramind_nyc_lora/`](https://github.com/msradam/riprap-nyc/tree/main/experiments/18_terramind_nyc_lora).
Per-adapter TRAINING.md: [TerraMind-NYC-Adapters/docs/TRAINING.md](https://github.com/msradam/TerraMind-NYC-Adapters/blob/main/docs/TRAINING.md).

| | |
|---|---|
| Base | `ibm-esa-geospatial/TerraMind-1.0-base` (1B params, **frozen**) |
| Adapter shape | LoRA rank 16, alpha 32, target `attn.qkv` + `attn.proj` across 24 transformer blocks (~885K trainable Δ params) |
| Decoder | UNet (channels [512, 256, 128, 64]), trained from scratch (~80M params) |
| Data | Major-TOM Core (S2L2A + S1RTC + DEM × 4 timesteps) over NYC bbox, 22 parents → 352 sub-chips |
| Labels (LULC) | ESA WorldCover 2021 v200, collapsed 11→5 NYC macro classes |
| Labels (Buildings) | NYC DOITT footprints (`5zhs-2jue`) rasterized |
| Loss (LULC, TiM) | Class-weighted CE (inverse-frequency on train) |
| Loss (Buildings v2) | CE with class weights [0.6, 1.6] (replaced Focal-Tversky from v1 which didn't converge under LoRA) |
| Optimizer | AdamW, two-LR (LoRA params 5e-4, decoder/head 1e-4) |
| Batch | 8 |
| Epochs | 30 (LULC, TiM) / 40 (Buildings) |
| Wall-clock | **25–40 min** per adapter |
| Test mIoU | LULC 0.5866 / Buildings 0.5511 / TiM 0.6023 |

## Why these three (and only these three)

The Riprap-NYC parent system has nine specialists in its FSM. Three of
them are foundation-model fine-tunes; the other six are rule-based
spatial joins (NYC 311, FloodNet sensors, Sandy Inundation Zone, DEP
stormwater scenarios, Ida high-water marks, microtopography). This
repo only documents the fine-tunes — the rule-based specialists don't
need foundation-model training and aren't the right shape for this kind
of repository.

The three fine-tunes, in size order:

- **TTM Battery Surge (1.5M)** — time-series regression on NOAA gauge
  data, the smallest model and the fastest to train (5 min).
- **Prithvi NYC Pluvial (300M)** — vision transformer for Hurricane-Ida
  flood-pattern segmentation in Sentinel-2.
- **TerraMind NYC Adapters (1B base + LoRA Δ)** — multi-modal foundation
  model with three NYC LoRA adapters (buildings, LULC, TiM).

Total params on disk: ~1.5 B + ~300 M + ~1 B = ~2.8 B. On an M3 Air
running CPU fp32, peak RAM stays under 4 GB because only one model
loads at a time.

## Reproducibility receipt

Every step is documented in code:

- **TTM**: `riprap-nyc/experiments/20_ttm_battery_surge/finetune_ttm_battery.py`
  + `fetch_noaa_battery.py` for the data pull.
- **Prithvi v2**: `phase19_prithvi_v2.yaml` + `copy_paste_aug.py` for
  the augmentation step.
- **TerraMind LoRA**: `shared/train_lora.py` + per-adapter
  `adapters/<name>/config.yaml`.

Plus the Hugging Face model cards for each include their own MODEL_CARD.md
that mirrors what's published. Plus this repo's `eval/reports/*.md`
records what actually reproduces from the published artifacts on a
laptop.

The deepest version of the story is in
[`riprap-nyc/experiments/`](https://github.com/msradam/riprap-nyc/tree/main/experiments).
