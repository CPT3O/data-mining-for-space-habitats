# ==== Section 11: Streamlit Dashboard (safe image handling + purpose) ====
import os, json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Space Habitat • Anomalies • Forecasts • Policy • Omics • QA",
                   layout="wide")
st.title("Space Habitat • Anomalies • Forecasts • Policy • Omics • QA")

# ---------- Path helpers ----------
def first_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]

BASE = Path.cwd()
OUT  = first_existing([BASE / "outputs", BASE])  # prefer ./outputs
ANOM = first_existing([OUT / "anomaly_forecast", OUT])
PAT  = first_existing([OUT / "pattern_analysis", OUT])
REL  = first_existing([OUT / "relationships", OUT])
INS  = first_existing([OUT / "insights", OUT])
POL  = first_existing([OUT / "policy", OUT])
OMX  = first_existing([OUT / "omics", OUT])
SEC9 = first_existing([OUT / "section9_global_qa", OUT])

telemetry_cols = ["Temp_degC_ISS", "RH_percent_ISS", "CO2_ppm_ISS"]
radiation_cols = ["GCR_Dose_mGy_d", "SAA_Dose_mGy_d", "Total_Dose_mGy_d", "Accumulated_Dose_mGy_d"]
all_vars = telemetry_cols + radiation_cols
GLDS_LIST = ["GLDS-98","GLDS-99","GLDS-104"]

# ---------- Utilities ----------
@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def list_missions() -> list[str]:
    if ANOM.exists():
        return sorted([p.name for p in ANOM.iterdir() if p.is_dir() and p.name.startswith("RR-")])
    return []

@st.cache_data(show_spinner=False)
def variables_for(rr: str) -> dict:
    out = {"telemetry": [], "radiation": []}
    mission_dir = ANOM / rr
    if not mission_dir.exists():
        return out
    for p in mission_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if name.startswith("telemetry_"):
            out["telemetry"].append(name.replace("telemetry_", ""))
        if name.startswith("radiation_"):
            out["radiation"].append(name.replace("radiation_", ""))
    out["telemetry"].sort()
    out["radiation"].sort()
    return out

@st.cache_data(show_spinner=False)
def load_policy() -> dict | None:
    for fname in ["prescriptive_policy_calibrated.json", "prescriptive_policy.json"]:
        p = POL / fname
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
    return None

@st.cache_data(show_spinner=False)
def load_relationships_table() -> pd.DataFrame | None:
    p = REL / "granger_var_results.csv"
    return load_csv(p) if p.exists() else None

@st.cache_data(show_spinner=False)
def load_seasonality_summary() -> pd.DataFrame | None:
    p = PAT / "seasonality_summary.csv"
    return load_csv(p) if p.exists() else None

@st.cache_data(show_spinner=False)
def load_scoreboards() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if ANOM.exists():
        for p in ANOM.glob("*_forecast_scoreboard.csv"):
            df = load_csv(p)
            if df is not None and not df.empty:
                out[p.stem.replace("_forecast_scoreboard", "")] = df
    return out

def safe_image(path: Path | None, caption: str, *, width='stretch'):
    """Render only if it's a real file; 'width' supports 'stretch'/'content' or int px."""
    if isinstance(path, Path) and path.is_file():
        st.image(str(path), caption=caption, width=width)
        return True
    return False

def safe_df(df: pd.DataFrame | None, *, max_rows: int = 200, note: str | None = None):
    if df is None:
        if note: st.caption(note); 
        return False
    if df.empty:
        st.caption("Empty table.")
        return False
    n = min(len(df), max_rows)
    st.dataframe(df.head(n))
    if len(df) > n:
        st.caption(f"Showing first {n} of {len(df)} rows.")
    return True

def list_all_artifacts(root: Path) -> pd.DataFrame:
    rows = []
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root)
            try:
                s = p.stat()
                rows.append({"path": str(rel), "size_bytes": s.st_size,
                             "modified": datetime.fromtimestamp(s.st_mtime)})
            except Exception:
                rows.append({"path": str(rel), "size_bytes": None, "modified": None})
    return pd.DataFrame(rows).sort_values("path")

# --- Robust image discovery in a series folder ---
def find_first_with_keys(folder: Path, *keys: str) -> Path | None:
    if not folder.exists(): return None
    keys = tuple(k.lower() for k in keys)
    for f in sorted(folder.glob("*.png")):
        name = f.name.lower()
        if all(k in name for k in keys):
            return f
    return None

def pick_series_images(vdir: Path) -> dict[str, Path | None]:
    return {
        "anoms":  (vdir/"anoms.png" if (vdir/"anoms.png").is_file() else
                   find_first_with_keys(vdir, "anom") or
                   find_first_with_keys(vdir, "z", "anom") or
                   find_first_with_keys(vdir, "ewma")),
        "ae_error": (vdir/"ae_error.png" if (vdir/"ae_error.png").is_file() else
                     find_first_with_keys(vdir, "ae", "error") or
                     find_first_with_keys(vdir, "autoencoder") or
                     find_first_with_keys(vdir, "recon", "error")),
        "lstm":   (vdir/"lstm_forecast.png" if (vdir/"lstm_forecast.png").is_file() else
                   find_first_with_keys(vdir, "lstm", "forecast") or
                   find_first_with_keys(vdir, "lstm")),
        "sarimax":(vdir/"sarimax_forecast.png" if (vdir/"sarimax_forecast.png").is_file() else
                   find_first_with_keys(vdir, "sarimax", "forecast") or
                   find_first_with_keys(vdir, "sarimax")),
    }

# ---------- Sidebar ----------
missions = list_missions()
with st.sidebar:
    st.header("Controls")
    if not missions:
        st.error("No missions found under outputs/anomaly_forecast/* .")
    rr = st.selectbox("Mission", missions, index=0 if missions else None)
    kinds = variables_for(rr) if missions else {"telemetry": [], "radiation": []}
    vtype = st.radio("Type", ["telemetry", "radiation"], horizontal=True) if missions else "telemetry"
    vlist = kinds.get(vtype, [])
    var = st.selectbox("Variable", vlist) if vlist else None
    st.caption(f"Outputs root: {OUT.resolve()}")

# ---------- Tabs ----------
tabs = st.tabs([
    "About & Purpose",
    "Overview",
    "Anomalies & Forecasts",
    "Pattern Extraction",
    "Relationships",
    "Policy & Compliance",
    "Omics (RR-1: GLDS-98/99/104)",
    "Global QA (Sec. 9)",
    "Publication Figures (Sec. 10)",
    "All Artifacts Browser"
])

# ---------- About & Purpose ----------
with tabs[0]:
    st.subheader("How to use this dashboard")
    st.markdown("""
- **Sidebar (left)** drives context:
  - **Mission** selects which RR mission’s artifacts to view.
  - **Type** + **Variable** pick a specific telemetry/radiation series.
  - These selections affect: **Overview**, **Anomalies & Forecasts**, **Pattern Extraction**, **Relationships**, **Policy & Compliance**.
- **Tab purposes**
  - **Overview**: Mission summary & scoreboard for the selected mission; quick preview of the selected series.
  - **Anomalies & Forecasts**: Per-series images (anomalies, AE error, LSTM/SARIMAX) + mission composite & calendar.
  - **Pattern Extraction**: STL/FFT/Welch plots for the selected series; seasonality table + download.
  - **Relationships**: Correlation & directed Granger figures; significant edges table for the selected mission.
  - **Policy & Compliance**: Prescriptive policy and compliance for the selected mission, with downloads.
  - **Omics (RR-1 only)**: GLDS-98/99/104 tables & figures; these datasets are associated with **RR-1**.
  - **Global QA (Sec. 9)**: Cross-mission statistics and QA figures (sidebar selections do **not** change these).
  - **Publication Figures (Sec. 10)**: Clean paper/deck figures (scoreboard, seasonal overlays, mission composites).
  - **All Artifacts Browser**: Explore everything under `outputs/`, preview & download files.
    """)

# ---------- Overview ----------
with tabs[1]:
    st.subheader(f"Summary — {rr}")
    summary_path = ANOM / f"{rr}_forecast_summary.csv"
    summary_df = load_csv(summary_path) if summary_path.exists() else None
    if (summary_df is None or summary_df.empty) and (INS / "all_missions_metrics_raw.csv").exists():
        tmp = load_csv(INS / "all_missions_metrics_raw.csv")
        if tmp is not None and not tmp.empty and "RR" in tmp.columns:
            tmp_rr = tmp[tmp["RR"].astype(str) == str(rr)].copy()
            if not tmp_rr.empty:
                summary_df = tmp_rr
    if summary_df is not None and not summary_df.empty:
        cols_show = [c for c in ["Type","Variable","Model","RMSE","MAE","AE_MeanMSE","AE_Threshold"] if c in summary_df.columns]
        st.dataframe(summary_df[cols_show].sort_values(["RMSE","MAE"]).reset_index(drop=True))
        st.download_button(f"Download {rr}_forecast_summary.csv",
                           summary_path.read_bytes() if summary_path.exists() else summary_df.to_csv(index=False).encode(),
                           file_name=f"{rr}_forecast_summary.csv", mime="text/csv", key=f"dl_summary_{rr}")
    else:
        st.info("No per-mission summary found.")
    st.markdown("### Scoreboard")
    boards = load_scoreboards()
    if boards and rr in boards:
        st.dataframe(boards[rr])
    elif boards:
        st.caption(f"No scoreboard CSV for {rr}. Available: {', '.join(sorted(boards.keys()))}")
    else:
        st.caption("No scoreboard CSVs found.")
    st.markdown("### Quick preview — selected series")
    if rr and var:
        vdir = ANOM / rr / f"{vtype}_{var}"
        if vdir.exists():
            imgs = pick_series_images(vdir)
            c1, c2 = st.columns(2)
            with c1:
                if not safe_image(imgs["anoms"], "Z & EWMA Anomalies"): st.caption("No anomalies image found.")
                if not safe_image(imgs["ae_error"], "LSTM AE Reconstruction Error"): st.caption("No AE error image found.")
            with c2:
                safe_image(imgs["lstm"], "LSTM Forecast (with PI)")
                safe_image(imgs["sarimax"], "SARIMAX Forecast")
        else:
            st.caption(f"No artifacts for {rr}/{vtype}_{var} at {vdir}")

# ---------- Anomalies & Forecasts ----------
with tabs[2]:
    st.subheader(f"{rr} • {vtype}:{var}")
    st.caption("This tab follows your **Mission/Type/Variable** selections in the sidebar.")
    if rr and var:
        vdir = ANOM / rr / f"{vtype}_{var}"
        if vdir.exists():
            imgs = pick_series_images(vdir)
            c1, c2 = st.columns(2)
            with c1:
                if not safe_image(imgs["anoms"], "Z & EWMA Anomalies"): st.caption("No anomalies image found in this folder.")
                if not safe_image(imgs["ae_error"], "LSTM AE Reconstruction Error"): st.caption("No AE error image found in this folder.")
            with c2:
                safe_image(imgs["lstm"], "LSTM Forecast (with PI)")
                safe_image(imgs["sarimax"], "SARIMAX Forecast")
            with st.expander("Files in this series folder"):
                files = [f.name for f in sorted(vdir.glob("*.png"))]
                st.code("\n".join(files) if files else "(none)")
        else:
            st.info("No anomaly/forecast artifacts for the selected series.")
    st.markdown("### Mission-wide")
    safe_image(ANOM / rr / "fig_composite_timeline.png", f"{rr} • Composite anomaly timeline")
    safe_image(ANOM / rr / "fig_anomaly_calendar.png", f"{rr} • Anomaly calendar (Z>3, per hour)")

# ---------- Pattern Extraction ----------
with tabs[3]:
    st.subheader(f"STL • FFT • Welch — {rr} • {vtype}:{var}")
    st.caption("This tab follows your **Mission/Type/Variable** selections in the sidebar.")
    if rr and var:
        if vtype == "telemetry":
            safe_image(PAT / f"{var}_{rr}_Raw.png", "Raw")
            safe_image(PAT / f"{var}_{rr}_STL.png", "STL")
            c1, c2 = st.columns(2)
            with c1: safe_image(PAT / f"{var}_{rr}_FFT.png", "FFT")
            with c2: safe_image(PAT / f"{var}_{rr}_WELCH.png", "Welch PSD")
        else:
            sub = PAT / "radiation"
            safe_image(sub / f"{var}_{rr}_Raw.png", "Raw")
            safe_image(sub / f"{var}_{rr}_STL.png", "STL")
            c1, c2 = st.columns(2)
            with c1: safe_image(sub / f"{var}_{rr}_FFT.png", "FFT")
            with c2: safe_image(sub / f"{var}_{rr}_WELCH.png", "Welch PSD")
    ss = load_seasonality_summary()
    if ss is not None and not ss.empty:
        st.markdown("Seasonality summary (top 25 by strength/amp)")
        top = (ss.sort_values(["Seasonality_Strength","Seasonal_Amplitude_P95_P05"], ascending=[False, False]).head(25))
        safe_df(top)
        p = PAT / "seasonality_summary.csv"
        if p.exists():
            st.download_button("Download seasonality_summary.csv", p.read_bytes(),
                               file_name="seasonality_summary.csv", mime="text/csv", key="dl_seasonality")
    else:
        st.caption("No seasonality_summary.csv found.")

# ---------- Relationships ----------
with tabs[4]:
    st.subheader(f"Correlation • Granger — {rr}")
    st.caption("Figures respect your **Mission** choice; tables filter to that mission.")
    safe_image(REL / "plots" / f"{rr}_pearson.png", "Pearson Correlation")
    safe_image(REL / "plots" / f"{rr}_spearman.png", "Spearman Correlation")
    safe_image(REL / "plots" / f"{rr}_granger_directed.png", "Directed Granger (−log10 q)")
    safe_image(REL / "plots" / f"{rr}_network.png", "Correlation Network")
    gr = load_relationships_table()
    if gr is not None and not gr.empty:
        st.markdown("Significant directed edges (q ≤ 0.05)")
        sig_col = "Significant" if "Significant" in gr.columns else None
        if sig_col:
            sub = gr[(gr["Mission"].astype(str) == str(rr)) & (gr[sig_col] == True)].copy()
            show_cols = [c for c in ["Source","Target","Lag","pvalue","qvalue","MinusLog10_q"] if c in sub.columns]
            if not safe_df(sub.sort_values(show_cols[-1] if "MinusLog10_q" in show_cols else "qvalue"), max_rows=200):
                st.caption("No significant directed relations at q ≤ 0.05.")
        else:
            st.caption("‘Significant’ indicator not present in table.")
        p = REL / "granger_var_results.csv"
        if p.exists():
            st.download_button("Download granger_var_results.csv", p.read_bytes(),
                               file_name="granger_var_results.csv", mime="text/csv", key="dl_granger")
    else:
        st.caption("No granger_var_results.csv found.")

# ---------- Policy & Compliance ----------
with tabs[5]:
    st.subheader("Policy (preliminary)")
    st.caption("This tab uses your **Mission** selection for thresholds and compliance.")
    pol = load_policy()
    if not pol:
        st.info("No policy JSON found under outputs/policy.")
    else:
        st.json(pol.get("meta", {}))
        if rr in pol.get("missions", {}):
            st.markdown(f"**Thresholds** — {rr}")
            st.json(pol["missions"][rr].get("thresholds", {}))
            st.markdown("**Radiation guidance**")
            st.json(pol["missions"][rr].get("radiation", {}))
            st.markdown("**Scheduling preferences**")
            st.json({"prefer_hours_utc_low_CO2": pol["missions"][rr].get("scheduling",{}).get("prefer_hours_utc_low_CO2",[])})
        comp_path = POL / "compliance" / f"{rr}_compliance.csv"
        if comp_path.exists():
            comp = load_csv(comp_path)
            st.markdown("Compliance rates")
            if comp is not None and not comp.empty:
                rate = comp.groupby("rule")["ok"].mean().reset_index()
                rate["ok"] = rate["ok"].round(3)
                safe_df(rate.sort_values("ok", ascending=False))
                st.download_button(f"Download {comp_path.name}", comp_path.read_bytes(),
                                   file_name=comp_path.name, mime="text/csv", key=f"dl_comp_{rr}")
        for fname in ["prescriptive_policy_calibrated.json","prescriptive_policy.json","policy_preview.csv","compliance_summary.csv"]:
            p = POL / fname
            if p.exists():
                st.download_button(f"Download {fname}", p.read_bytes(), file_name=fname,
                                   mime="application/json" if fname.endswith(".json") else "text/csv",
                                   key=f"dl_policy_{fname}")

# ---------- Omics (RR-1: GLDS-98/99/104) ----------
with tabs[6]:
    st.subheader("RR-1 Omics snapshot (GLDS-98/99/104)")
    st.caption("These datasets are associated with **RR-1**.")
    for glds in GLDS_LIST:
        st.markdown(f"### {glds}")
        up = INS / f"{glds}_top_upregulated.csv"
        dn = INS / f"{glds}_top_downregulated.csv"
        pca_tbl = INS / f"{glds}_pca_table.csv"
        if up.exists():
            st.caption("Top upregulated");  safe_df(load_csv(up), max_rows=50)
            st.download_button(f"Download {up.name}", up.read_bytes(), file_name=up.name, mime="text/csv", key=f"dl_up_{glds}")
        if dn.exists():
            st.caption("Top downregulated"); safe_df(load_csv(dn), max_rows=50)
            st.download_button(f"Download {dn.name}", dn.read_bytes(), file_name=dn.name, mime="text/csv", key=f"dl_dn_{glds}")
        if pca_tbl.exists():
            st.caption("PCA table (samples × PC1/PC2)"); safe_df(load_csv(pca_tbl), max_rows=100)
            st.download_button(f"Download {pca_tbl.name}", pca_tbl.read_bytes(), file_name=pca_tbl.name, mime="text/csv", key=f"dl_pca_{glds}")
        safe_image(OMX / f"fig_volcano_{glds}.png", f"{glds} — Volcano")
        safe_image(OMX / f"fig_pca_{glds}.png", f"{glds} — PCA")

# ---------- Global QA (Sec. 9) ----------
with tabs[7]:
    st.subheader("Section 9 — Global QA & Summary Statistics")
    st.caption("Cross-mission statistics and figures; sidebar selections do not change these.")
    kf_json = SEC9 / "key_findings.json"
    kf_md   = SEC9 / "key_findings.md"
    if kf_json.exists():
        st.markdown("**Key findings (JSON)**")
        st.json(json.loads(kf_json.read_text()))
        st.download_button("Download key_findings.json", kf_json.read_bytes(), file_name="key_findings.json",
                           mime="application/json", key="dl_kf_json")
    if kf_md.exists():
        st.markdown("**Key findings (Markdown)**")
        st.markdown(kf_md.read_text())
        st.download_button("Download key_findings.md", kf_md.read_bytes(), file_name="key_findings.md",
                           mime="text/markdown", key="dl_kf_md")
    for fname in ["coverage_summary.csv","quality_counts.csv","anomaly_rate_per_day.csv",
                  "granger_edge_counts.csv","forecast_best_by_variable.csv","omics_deg_counts.csv"]:
        p = SEC9 / fname
        if p.exists():
            st.markdown(f"**{fname}**")
            safe_df(load_csv(p), max_rows=200)
            st.download_button(f"Download {fname}", p.read_bytes(), file_name=fname,
                               mime="text/csv", key=f"dl_sec9_{fname}")
    tables_dir = SEC9 / "tables"
    if tables_dir.exists():
        st.markdown("**Top lists (tables/)**")
        for i, p in enumerate(sorted(tables_dir.glob("*.csv"))):
            with st.expander(p.name):
                safe_df(load_csv(p), max_rows=200)
                st.download_button(f"Download {p.name}", p.read_bytes(), file_name=p.name,
                                   mime="text/csv", key=f"dl_top_{i}")
    fig_names = [
        "fig_coverage_heatmap.png", "fig_granger_edges.png", "fig_rmse_by_model.png",
        "fig_seasonality_strength_hist.png", "fig_anomaly_rate_by_mission.png",
        "fig_deg_counts.png",
        "fig_quality_RR-1.png","fig_quality_RR-3.png","fig_quality_RR-6.png",
        "fig_quality_RR-9.png","fig_quality_RR-12.png","fig_quality_RR-19.png",
    ]
    st.markdown("**QA Figures**")
    cols = st.columns(3)
    i = 0
    for fname in fig_names:
        p = SEC9 / fname
        if p.is_file():
            with cols[i % 3]:
                safe_image(p, fname)
            i += 1
    if i == 0:
        st.caption("No QA figures found in section9_global_qa/")

# ---------- Publication Figures (Sec. 10) ----------
with tabs[8]:
    st.subheader("Section 10 — Publication Figures & Repro Plots")
    st.caption("Cross-mission figures; sidebar selections only affect which mission’s composites you expand.")
    st.markdown("**Scoreboard**")
    safe_image(ANOM / "fig_model_scoreboard.png", "Model comparison (RMSE)")
    st.markdown("**Seasonal overlays**")
    colsa = st.columns(3)
    j = 0
    for v in all_vars:
        p = PAT / f"fig_seasonal_overlay_{v}.png"
        if p.is_file():
            with colsa[j % 3]:
                safe_image(p, f"Seasonal overlay — {v}")
            j += 1
    st.markdown("**Mission composites**")
    for m in missions:
        with st.expander(m, expanded=False):
            safe_image(ANOM / m / "fig_composite_timeline.png", f"{m} • Composite anomaly timeline")
            safe_image(ANOM / m / "fig_anomaly_calendar.png", f"{m} • Anomaly calendar")

# ---------- All Artifacts Browser ----------
with tabs[9]:
    st.subheader("All Artifacts Browser")
    st.caption(str(OUT.resolve()))
    files_df = list_all_artifacts(OUT)
    if not files_df.empty:
        st.dataframe(files_df)
        rel_sel = st.selectbox("Preview/download a file", files_df["path"].tolist(), index=0, key="file_picker")
        sel_path = OUT / rel_sel
        if sel_path.is_file():
            st.download_button("Download selected", sel_path.read_bytes(), file_name=sel_path.name, key="dl_selected")
            if sel_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                safe_image(sel_path, sel_path.name, width=720)
            elif sel_path.suffix.lower() == ".csv":
                safe_df(load_csv(sel_path), max_rows=200)
            elif sel_path.suffix.lower() in [".json", ".md", ".txt"]:
                with st.expander("Preview text", expanded=True):
                    st.code(sel_path.read_text()[:20000],
                            language="json" if sel_path.suffix.lower()==".json" else "markdown")
    else:
        st.caption("No files found under outputs/.")
