"""riprap-models live demo: pulls real NYC data, runs three NYC fine-tuned
foundation models on local hardware, reports per-call energy.

    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Riprap Models — Live NYC",
    page_icon="🪨",
    layout="wide",
)

# IBM Plex + civic-hydrology accents on top of the base theme in
# .streamlit/config.toml. Tokens copied from riprap-nyc/web/sveltekit/src/lib/tokens.css.
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@400;500;600&display=swap');

      :root {
        --paper: #F4F6F9;
        --paper-deep: #E8ECF2;
        --ink: #0F172A;
        --ink-secondary: #334155;
        --ink-tertiary: #64748B;
        --rule: #0F172A;
        --rule-soft: #CBD5E1;
        --accent: #005EA2;
        --accent-deep: #1A4480;
        --accent-warn: #92400E;
        --accent-alert: #B91C1C;
        --stone-touchstone: #0E7490;
      }

      html, body, [class*="css"] {
        font-family: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif !important;
        color: var(--ink);
      }
      code, pre, .stCodeBlock, [data-testid="stMetricValue"] {
        font-family: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace !important;
      }
      h1, h2, h3, h4 {
        font-family: "IBM Plex Sans", system-ui, sans-serif !important;
        letter-spacing: -0.01em;
        color: var(--ink);
      }
      h1 { font-weight: 600; border-bottom: 2px solid var(--ink); padding-bottom: 8px; }
      h2 { font-weight: 600; }

      /* Riprap-style top header rule */
      [data-testid="stAppViewContainer"] > .main {
        background: var(--paper);
      }
      [data-testid="stHeader"] {
        background: var(--paper);
        border-bottom: 1px solid var(--rule-soft);
      }
      [data-testid="stSidebar"] {
        background: var(--paper-deep);
        border-right: 1px solid var(--rule-soft);
      }
      [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--ink-tertiary);
        font-weight: 600;
        border-bottom: none;
        margin-bottom: 4px;
      }

      /* Metrics: civic-mono register */
      [data-testid="stMetric"] {
        background: var(--paper);
        border: 1px solid var(--rule-soft);
        border-radius: 0;
        padding: 10px 14px;
      }
      [data-testid="stMetricLabel"] {
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--ink-tertiary) !important;
      }
      [data-testid="stMetricValue"] {
        color: var(--accent-deep) !important;
        font-weight: 600 !important;
      }

      /* Buttons: federal-blue primary */
      .stButton > button[kind="primary"] {
        background: var(--accent);
        border: 1px solid var(--accent-deep);
        border-radius: 0;
        font-family: "IBM Plex Sans", sans-serif !important;
        font-weight: 500;
        letter-spacing: 0.02em;
      }
      .stButton > button[kind="primary"]:hover {
        background: var(--accent-deep);
      }
      .stButton > button {
        border-radius: 0;
        border: 1px solid var(--rule-soft);
        font-family: "IBM Plex Sans", sans-serif !important;
      }

      /* Alerts: tiered colour register from tokens.css */
      [data-testid="stAlert"] {
        border-radius: 0;
        border-left-width: 4px;
      }
      [data-testid="stAlert"][data-baseweb="notification"] [aria-label="Error icon"] ~ div {
        color: var(--accent-alert);
      }

      /* Tabs */
      [data-baseweb="tab-list"] {
        border-bottom: 1px solid var(--rule-soft);
        gap: 24px;
      }
      [data-baseweb="tab"] {
        font-family: "IBM Plex Sans", sans-serif !important;
        font-weight: 500;
        letter-spacing: 0.02em;
        color: var(--ink-secondary);
      }
      [data-baseweb="tab"][aria-selected="true"] {
        color: var(--accent-deep);
      }

      /* Caption + small text */
      .stCaption, [data-testid="stCaptionContainer"] {
        font-family: "IBM Plex Mono", monospace !important;
        font-size: 12px;
        color: var(--ink-tertiary);
      }

      /* Selectbox / dropdown */
      .stSelectbox > div > div {
        border-radius: 0;
        border: 1px solid var(--rule-soft);
      }

      /* Code blocks */
      .stCodeBlock {
        background: var(--paper-deep) !important;
        border-left: 3px solid var(--accent);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---- Header --------------------------------------------------------------

col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("# 🪨 riprap-models")
    st.caption(
        "Three NYC fine-tuned foundation models, fetched live from public sources, "
        "running on this laptop. No vendor LLM contacted. No GPU required. "
        "Apache-2.0."
    )
with col_status:
    from riprap_models.device import get_device
    info = get_device()
    st.markdown(
        f"<div style='text-align:right; font-family: IBM Plex Mono; font-size: 11px; "
        f"color: var(--ink-tertiary); text-transform: uppercase; letter-spacing: 0.08em; "
        f"padding-top: 18px;'>"
        f"<span style='display:inline-block; width:6px; height:6px; border-radius:50%; "
        f"background:#0E7490; margin-right:6px;'></span>"
        f"Local · {info.label}"
        f"</div>",
        unsafe_allow_html=True,
    )

# ---- Sidebar -------------------------------------------------------------

with st.sidebar:
    st.markdown("## Reproduction")
    results_path = Path("docs/RESULTS.md")
    if results_path.exists():
        st.markdown(results_path.read_text())
    st.markdown("---")
    st.markdown("## Source")
    st.markdown(
        "[github.com/msradam/riprap-models](https://github.com/msradam/riprap-models)\n\n"
        "All models, harness, and docs are Apache-2.0. Every number on this page "
        "is regenerable from public sources via `riprap-models eval <name>`."
    )

# ---- Tabs ----------------------------------------------------------------

tab_surge, tab_eo, tab_about = st.tabs(
    ["⛈ Battery Surge (TTM)", "🛰 NYC Satellite (Prithvi + TerraMind)", "ℹ About"]
)

# ---- TAB 1: TTM Battery Surge -------------------------------------------

with tab_surge:
    st.markdown("## Granite TTM r2 — Battery surge nowcast")
    st.caption(
        "NOAA station 8518750 (The Battery, lower Manhattan). 1024-hour hourly "
        "context → 96-hour forecast. ~3M-param fine-tune of IBM Granite TimeSeries TTM r2."
    )

    if st.button("Run live forecast", type="primary", key="run_ttm"):
        with st.spinner("Fetching NOAA · loading model · forecasting"):
            from riprap_models.energy import measure_energy
            from riprap_models.ttm_battery_surge.data import (
                DEFAULT_STATION,
                fetch_residual_series,
                load_finetune,
            )

            t0 = time.time()
            end = datetime.now(timezone.utc)
            begin = end - timedelta(days=50)
            ts, res = fetch_residual_series(
                DEFAULT_STATION,
                begin.strftime("%Y%m%d"),
                end.strftime("%Y%m%d"),
                hourly=True,
            )
            fetch_s = time.time() - t0

            t1 = time.time()
            forecaster = load_finetune({"context_steps": 1024, "horizon_steps": 96})
            load_s = time.time() - t1

            history = res.astype(np.float32)[-1024:]
            with measure_energy() as m:
                fc = forecaster.predict(history, horizon=96)

            st.session_state["ttm_history"] = res
            st.session_state["ttm_forecast"] = fc
            st.session_state["ttm_fetch_s"] = fetch_s
            st.session_state["ttm_load_s"] = load_s
            st.session_state["ttm_inference_ms"] = m.duration_s * 1000
            st.session_state["ttm_joules"] = m.joules
            st.session_state["ttm_method"] = m.method
            st.session_state["ttm_n_history"] = len(res)
            st.session_state["ttm_when"] = end.isoformat()

    if "ttm_history" in st.session_state:
        res = st.session_state["ttm_history"]
        fc = st.session_state["ttm_forecast"]
        peak = float(np.max(np.abs(fc)))
        peak_signed = float(np.max(fc)) if abs(np.max(fc)) > abs(np.min(fc)) else float(np.min(fc))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Peak forecast", f"{peak_signed:+.2f} m")
        c2.metric("Inference", f"{st.session_state['ttm_inference_ms']:.0f} ms")
        c3.metric("Energy", f"{st.session_state['ttm_joules']:.3f} J",
                  help=f"Method: {st.session_state['ttm_method']}")
        c4.metric("History", f"{st.session_state['ttm_n_history']} h")

        if peak >= 0.50:
            st.error(f"⚠ Storm-class forecast: peak |residual| {peak:.2f} m "
                     "(regime where the fine-tune materially beats persistence).")
        elif peak >= 0.30:
            st.warning(f"Moderate surge: peak |residual| {peak:.2f} m.")
        else:
            st.success(f"Calm forecast: peak |residual| {peak:.2f} m.")

        import altair as alt
        import pandas as pd

        h_recent = res[-336:]
        h_idx = np.arange(-len(h_recent), 0)
        f_idx = np.arange(0, 96)
        df = pd.concat([
            pd.DataFrame({"hour": h_idx, "residual_m": h_recent, "kind": "history"}),
            pd.DataFrame({"hour": f_idx, "residual_m": fc, "kind": "forecast"}),
        ])
        chart = (
            alt.Chart(df)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("hour:Q", title="hours from now"),
                y=alt.Y("residual_m:Q", title="surge residual (m)"),
                color=alt.Color(
                    "kind:N",
                    scale=alt.Scale(
                        domain=["history", "forecast"],
                        range=["#1A4480", "#B91C1C"],
                    ),
                    legend=alt.Legend(title=None, orient="top-right"),
                ),
            )
            .properties(height=320, padding={"left": 8, "top": 16})
        )
        st.altair_chart(chart, use_container_width=True)

        with st.expander("Reproduction notes"):
            st.markdown(
                f"- Fetched **{st.session_state['ttm_n_history']} hourly samples** "
                f"in {st.session_state['ttm_fetch_s']:.1f} s from NOAA CO-OPS\n"
                f"- Model load: {st.session_state['ttm_load_s']:.1f} s (cached after first run)\n"
                f"- Inference: **{st.session_state['ttm_inference_ms']:.1f} ms**\n"
                f"- Energy: **{st.session_state['ttm_joules']:.4f} J** "
                f"({st.session_state['ttm_method']})\n"
                f"- Capture: {st.session_state['ttm_when']}\n\n"
                "On strictly post-cutoff data the fine-tune wins over zero-shot "
                "TTM r2 by **+10% MAE on storm windows (peak ≥ 0.7 m)** and "
                "ties on calm. Persistence is uncompetitive at any storm threshold."
            )

# ---- TAB 2: Satellite (Prithvi + TerraMind) -----------------------------

with tab_eo:
    st.markdown("## NYC satellite analysis")
    st.caption(
        "Prithvi-EO 2.0 NYC Pluvial (324M) on Sentinel-2 + TerraMind (~168M, multi-modal "
        "S2L2A + S1RTC + DEM × 4 timesteps) on the same chip. Fetched from MS Planetary Computer."
    )

    aoi_options = {
        "Manhattan midtown": (-73.984, 40.755),
        "Brooklyn downtown": (-73.989, 40.692),
        "Queens Jamaica": (-73.794, 40.702),
        "Bronx Morrisania": (-73.911, 40.829),
        "Staten Island St. George": (-74.075, 40.643),
        "Manhattan lower waterfront": (-74.014, 40.706),
    }
    aoi_name = st.selectbox("NYC AOI", list(aoi_options))
    lon, lat = aoi_options[aoi_name]

    c1, c2 = st.columns(2)
    run_prithvi = c1.button("Run Prithvi (pluvial flood)", key="run_prithvi")
    run_terramind = c2.button("Run TerraMind (buildings + LULC)", key="run_tm")

    if run_prithvi:
        with st.spinner("Fetching Sept 2021 Sentinel-2 chip · Prithvi inference"):
            from riprap_models.energy import measure_energy
            from riprap_models.prithvi_pluvial.data import (
                _read_chip_with_label,
                load_pluvial_finetune,
            )

            cfg = {
                "ida_polygons": "/Users/amsrahman/riprap-nyc/data/prithvi_ida_2021.geojson",
                "test_mode": "polygon",
            }
            model, preprocess, _ = load_pluvial_finetune(cfg)

            import geopandas as gpd
            from shapely.geometry import Point

            g = gpd.read_file(cfg["ida_polygons"]).to_crs(4326)
            distances = g.geometry.apply(lambda p: p.centroid.distance(Point(lon, lat)))
            nearest_idx = int(distances.idxmin())
            nlon = float(g.geometry.iloc[nearest_idx].centroid.x)
            nlat = float(g.geometry.iloc[nearest_idx].centroid.y)
            st.caption(f"Nearest Ida polygon: idx {nearest_idx} at ({nlon:.4f}, {nlat:.4f})")

            import planetary_computer
            import pystac_client

            cat = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=planetary_computer.sign_inplace,
            )
            search = cat.search(
                collections=["sentinel-2-l2a"],
                intersects={"type": "Point", "coordinates": [nlon, nlat]},
                datetime="2021-09-05/2021-09-12",
                query={"eo:cloud_cover": {"lt": 30}},
                max_items=5,
            )
            items = sorted(search.items(), key=lambda it: it.properties.get("eo:cloud_cover", 100))
            if not items:
                st.error("No cloud-free post-Ida S2 scene found")
            else:
                image, label = _read_chip_with_label(
                    items[0], nlon, nlat, g.geometry.tolist(), g.crs,
                )
                x = preprocess(image).unsqueeze(0)
                with measure_energy() as m:
                    pred = model(x).argmax(dim=1).squeeze(0).cpu().numpy()

                col1, col2, col3 = st.columns(3)
                col1.metric("Inference", f"{m.duration_s*1000:.0f} ms")
                col2.metric("Energy", f"{m.joules:.2f} J")
                col3.metric("Pred flood", f"{int(pred.sum())} px")

                import matplotlib.pyplot as plt

                plt.rcParams["font.family"] = "IBM Plex Sans"
                fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor="#F4F6F9")
                rgb = image[..., [2, 1, 0]]
                rgb = np.clip(rgb / np.percentile(rgb, 99), 0, 1)
                axes[0].imshow(rgb)
                axes[0].set_title("Sentinel-2 (Sept 7 2021)", color="#0F172A", fontsize=11)
                axes[1].imshow(label, cmap="Blues", vmin=0, vmax=1)
                axes[1].set_title("Ida polygons (label)", color="#0F172A", fontsize=11)
                axes[2].imshow(pred, cmap="Blues", vmin=0, vmax=1)
                axes[2].set_title("Prithvi prediction", color="#0F172A", fontsize=11)
                for a in axes:
                    a.set_xticks([]); a.set_yticks([])
                    for s in a.spines.values():
                        s.set_color("#CBD5E1")
                st.pyplot(fig)
                plt.close(fig)

                tp = int(((pred == 1) & (label == 1)).sum())
                fp = int(((pred == 1) & (label == 0)).sum())
                fn = int(((pred == 0) & (label == 1)).sum())
                iou = tp / max(1, tp + fp + fn)
                st.info(f"IoU on this polygon: **{iou:.3f}** (TP={tp}, FP={fp}, FN={fn})")

    if run_terramind:
        with st.spinner("Fetching multi-modal stack · TerraMind buildings + LULC inference"):
            from riprap_models.energy import measure_energy
            from riprap_models.terramind.data import (
                iter_holdout_tiles as iter_tm,
                load_terramind_adapter,
            )

            cfg_aoi = {"aois": [(aoi_name.replace(" ", "_").lower(), lon, lat)]}

            bld_model, bld_pre, _ = load_terramind_adapter(
                {"adapter_dir": "buildings_nyc", "num_classes": 2}
            )
            lulc_model, lulc_pre, _ = load_terramind_adapter(
                {"adapter_dir": "lulc_nyc", "num_classes": 5}
            )

            tile_id, inputs, bld_label, _ = next(iter_tm(cfg_aoi))
            x = bld_pre(inputs)
            with measure_energy() as m_bld:
                bld_pred = bld_model(x).argmax(dim=1).squeeze(0).cpu().numpy()

            x_lulc = lulc_pre(inputs)
            with measure_energy() as m_lulc:
                lulc_pred = lulc_model(x_lulc).argmax(dim=1).squeeze(0).cpu().numpy()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Buildings", f"{m_bld.duration_s*1000:.0f} ms")
            c2.metric("LULC", f"{m_lulc.duration_s*1000:.0f} ms")
            c3.metric("Bld energy", f"{m_bld.joules:.2f} J")
            c4.metric("LULC energy", f"{m_lulc.joules:.2f} J")

            import matplotlib.pyplot as plt

            plt.rcParams["font.family"] = "IBM Plex Sans"
            fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), facecolor="#F4F6F9")
            rgb = np.stack(
                [inputs["S2L2A"][3, 0], inputs["S2L2A"][2, 0], inputs["S2L2A"][1, 0]],
                axis=-1,
            )
            rgb = np.clip(rgb / np.percentile(rgb, 99), 0, 1)
            axes[0].imshow(rgb)
            axes[0].set_title(f"Sentinel-2 RGB · {aoi_name}", color="#0F172A", fontsize=11)
            axes[1].imshow(bld_label, cmap="Greys", vmin=0, vmax=1)
            axes[1].set_title("DOITT footprints (label)", color="#0F172A", fontsize=11)
            axes[2].imshow(bld_pred, cmap="Greys", vmin=0, vmax=1)
            axes[2].set_title("TerraMind buildings", color="#0F172A", fontsize=11)
            axes[3].imshow(lulc_pred, cmap="tab10", vmin=0, vmax=5)
            axes[3].set_title("TerraMind LULC (5-class)", color="#0F172A", fontsize=11)
            for a in axes:
                a.set_xticks([]); a.set_yticks([])
                for s in a.spines.values():
                    s.set_color("#CBD5E1")
            st.pyplot(fig)
            plt.close(fig)

            class_names = ["water", "impervious", "vegetation", "bare/cropland", "building"]
            counts = [int((lulc_pred == c).sum()) for c in range(5)]
            total = sum(counts)
            st.markdown("**LULC composition of this chip:**")
            for name, count in zip(class_names, counts):
                pct = 100 * count / max(1, total)
                st.markdown(f"- {name}: {count} px ({pct:.1f}%)")

# ---- TAB 3: About --------------------------------------------------------

with tab_about:
    st.markdown("## What this is")
    st.markdown(
        """
Three NYC fine-tuned foundation models, all running locally on your laptop.
No data leaves your machine. No vendor LLM is contacted. Every model loads
from Hugging Face's public hub under Apache-2.0.

- **Granite TTM r2 Battery Surge** — 1.5 M-param time-series fine-tune
- **Prithvi-EO 2.0 NYC Pluvial** — 324 M-param vision transformer fine-tune
- **TerraMind NYC Adapters** — 1 B-param multi-modal foundation model with
  three NYC LoRA adapters (Buildings, LULC, TiM)

Total disk: ~3 GB. Total peak RAM: under 4 GB.
        """
    )

    st.markdown("## Reproducibility")
    st.markdown(
        """
Every number is regenerable from public sources:

- **NOAA CO-OPS** for Battery surge data (no auth, public domain)
- **Microsoft Planetary Computer** for Sentinel-2 / Sentinel-1 / DEM (no auth)
- **NYC OpenData** for DOITT building footprints (`5zhs-2jue`, public domain)
- **ESA WorldCover 2021** for LULC labels
- **riprap-nyc** for the 166 Hurricane Ida flood polygons (Apache-2.0)

Clone https://github.com/msradam/riprap-models, run
`riprap-models eval <name>`, verify every sidebar number.
        """
    )

    st.markdown("## Honest limitations")
    st.markdown(
        """
- **TTM**: marginal advantage on calm weather; +10% vs zero-shot on storms (peak ≥ 0.7 m).
- **Prithvi**: card claims 0.60 flood IoU; reproduction from public artifacts yields
  0.115 vicinity IoU. The model finds large flood regions (largest test polygon
  scored IoU 0.51 alone); the headline number depends on an unpublished
  chip-extraction script. Treat outputs as candidate flood regions for human review.
- **TerraMind buildings**: high recall (99%), low precision (35%). Use as candidate
  overlay, not as authoritative footprint count. NYC DOITT is the authoritative source.
- **TerraMind LULC**: 0.36 mIoU vs card 0.59. Water class IoU 0.94 (higher than card).

Full per-model gap analysis in `WORKLOG.md` and `eval/reports/*.md`.
        """
    )
