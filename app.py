print("--- APP BOOTING ---")

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_gsheets import GSheetsConnection

import data_processing as dp

st.set_page_config(page_title="IPMS Viewer", layout="wide")

conn = st.connection("gsheets", type=GSheetsConnection)

DATA_ROOT = Path("Data")
OVERRIDES_PATH = Path("metadata_overrides.json")


def ensure_metadata_overrides_file() -> None:
    """Create an empty overrides file if missing so the app and Git can track it."""
    if not OVERRIDES_PATH.exists():
        OVERRIDES_PATH.write_text("{}\n", encoding="utf-8")


def get_file_binary(
    file_path: Optional[str] = None,
    investigator_folder: Optional[str] = None,
    filename: Optional[str] = None,
) -> bytes:
    """
    Read a raw experiment file from disk as bytes.

    Prefer ``file_path`` from the manifest (already joins ``Data/<Investigator>/<filename>`` when indexed).

    Alternatively pass ``investigator_folder`` + ``filename`` to build
    ``Data/<investigator_folder>/<filename>`` under the app working directory.
    """
    if file_path and str(file_path).strip():
        p = Path(str(file_path).strip())
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
    elif investigator_folder and filename:
        inv = str(investigator_folder).strip().strip("/\\")
        fn = str(filename).strip().strip("/\\")
        p = (DATA_ROOT / inv / fn).resolve()
    else:
        raise ValueError("get_file_binary requires file_path, or investigator_folder and filename.")
    if not p.is_file():
        raise FileNotFoundError(f"Not found or not a file: {p}")
    return p.read_bytes()


def load_metadata_overrides() -> Dict[str, Dict[str, str]]:
    if not OVERRIDES_PATH.exists():
        return {}
    try:
        raw = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for k, v in raw.items():
        if not isinstance(v, dict):
            continue
        out[str(k)] = {
            "display_name": str(v.get("display_name", "")).strip(),
            "details": str(v.get("details", "")).strip(),
        }
    return out


def save_metadata_overrides(overrides: Dict[str, Dict[str, str]]) -> None:
    OVERRIDES_PATH.write_text(json.dumps(overrides, indent=2, sort_keys=True), encoding="utf-8")


ensure_metadata_overrides_file()
# Eager read at startup: same JSON drives enrich_manifest (overrides beat filename parsing).
load_metadata_overrides()


BAF_RED = "#FF4B4B"
RANK_PLOT_MUTED_BLUE = "#A0C4FF"
OCEAN_BLUE = "#007BFF"
EMERALD = "#28A745"
CARD_BORDER = "#F0F2F6"
BG_WHITE = "#FFFFFF"
TEXT_DARK = "#1F2937"
SOFT_BLUE = "#E0F2FE"
CEBPE_TEAL = "#008080"
CONTROL_GREY_BG = "#F0F2F6"
CONTROL_GREY_FG = "#31333F"


def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {BG_WHITE};
                font-family: "Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
                color: {TEXT_DARK};
            }}
            h1, h2, h3, h4, h5, h6, p, label, span, div {{
                color: {TEXT_DARK};
            }}
            [data-testid="stMetric"] {{
                background-color: white;
                border: 1px solid {CARD_BORDER};
                border-radius: 10px;
                padding: 8px;
            }}
            [data-testid="stDataFrame"] {{
                width: 100%;
            }}
            div[data-testid="stDataFrame"] > div {{
                width: 100%;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, accent: str = OCEAN_BLUE) -> None:
    st.markdown(
        f"<hr style='border: none; border-top: 1px solid {CARD_BORDER}; margin: 0.2rem 0 0.9rem 0;'>",
        unsafe_allow_html=True,
    )
    st.subheader(title)
    st.markdown(
        f"<div style='height:3px; width:70px; background:{accent}; border-radius:4px; margin-top:-8px; margin-bottom:12px;'></div>",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def index_library() -> pd.DataFrame:
    """Index CSV paths only; expand each .xlsx into one row per TMT channel (filename parse only for CSV)."""
    rows = []
    for p in DATA_ROOT.rglob("*.csv"):
        if "_PROCESSED" in p.name.upper():
            continue
        rows.append({"path": str(p), "file_name": p.name, "investigator": p.parent.name})
    for p in DATA_ROOT.rglob("*.xlsx"):
        if "_PROCESSED" in p.name.upper():
            continue
        try:
            tmt_rows = dp.build_tmt_manifest_rows(p)
            rows.extend(tmt_rows)
        except Exception as exc:
            print(f"TMT manifest skipped for {p}: {exc}")
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_path_metadata(path: str) -> dict:
    """Filename-based metadata for one path (cached per path)."""
    return dp.extract_metadata(Path(path))


def enrich_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach filename-derived metadata, then apply ``metadata_overrides.json`` on top.

    Priority: manual overrides (gold) win over automatic filename parsing (fallback).
    """
    if df.empty:
        return df
    overrides = load_metadata_overrides()
    out_rows = []
    for _, row in df.iterrows():
        rd = row.to_dict()
        pth = Path(str(rd["path"]))
        inv_from_folder = pth.parent.name or str(rd.get("investigator", "") or "")
        sn = rd.get("tmt_sn_sum_column")
        if sn is not None and str(sn).strip() != "" and (not isinstance(sn, float) or not pd.isna(sn)):
            key = str(rd.get("full_filename") or pth.name)
            ov = overrides.get(key, {})
            rd = {**rd, "investigator": str(rd.get("investigator") or inv_from_folder)}
            rd.setdefault("experiment_type", "TMT Multiplex")
            if not rd.get("biological_condition"):
                rd["biological_condition"] = dp.tmt_biological_condition_from_channel_label(str(rd.get("tmt_channel", "")))
            rd.setdefault("details", "N/A")
            rd.setdefault("display_name", str(rd.get("file_name", "")))
            disp_o = str(ov.get("display_name", "")).strip()
            det_o = str(ov.get("details", "")).strip()
            if disp_o:
                if str(rd.get("tmt_channel", "")).strip():
                    rd["display_name"] = f"{disp_o} | Channel: {rd.get('tmt_channel')}"
                else:
                    rd["display_name"] = disp_o
            if det_o:
                rd["details"] = det_o
            if not str(rd.get("details", "")).strip():
                rd["details"] = "N/A"
            if not str(rd.get("display_name", "")).strip():
                rd["display_name"] = str(rd.get("file_name", ""))
            out_rows.append(rd)
            continue
        m = get_path_metadata(str(row["path"]))
        m["investigator"] = inv_from_folder
        m["biological_condition"] = "Single Run"
        key = str(m.get("full_filename") or pth.name)
        ov = overrides.get(key, {})
        base: Dict[str, Any] = {
            **rd,
            **m,
            "experiment_type": "Label-Free",
            "details": "Single Run",
            "display_name": str(rd.get("file_name", "")),
        }
        disp_o = str(ov.get("display_name", "")).strip()
        det_o = str(ov.get("details", "")).strip()
        if disp_o:
            base["display_name"] = disp_o
        if det_o:
            base["details"] = det_o
        if not str(base.get("details", "")).strip():
            base["details"] = "N/A"
        if not str(base.get("display_name", "")).strip():
            base["display_name"] = str(rd.get("file_name", ""))
        out_rows.append(base)
    out_df = pd.DataFrame(out_rows)
    if out_df.empty:
        return out_df
    return dp.apply_manifest_discovery_defaults(out_df)


@st.cache_data(show_spinner=False)
def load_experiment_summary(path: str, tmt_sn_sum_column: Optional[str] = None) -> pd.DataFrame:
    return dp.summarize_experiment_any(Path(path), tmt_sn_sum_column)


def tmt_sn_col_from_row(row: Union[pd.Series, Dict[str, Any]]) -> Optional[str]:
    if isinstance(row, dict):
        v = row.get("tmt_sn_sum_column")
    else:
        if "tmt_sn_sum_column" not in row.index:
            return None
        v = row.get("tmt_sn_sum_column")
    if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
        return None
    return str(v).strip()


def load_and_process_file(path: str, tmt_sn_sum_column: Optional[str] = None) -> pd.DataFrame:
    df = load_experiment_summary(path, tmt_sn_sum_column).copy()
    lda = pd.to_numeric(df["LDA Probability"], errors="coerce")
    valid_lda = (lda > 0) & (lda <= 1)
    y_axis = np.full(len(df), np.nan, dtype=float)
    if valid_lda.any():
        y_axis[valid_lda.to_numpy()] = -np.log10(lda[valid_lda].to_numpy())
    df["y_axis_val"] = y_axis

    y_std = float(df["y_axis_val"].std(skipna=True)) if not df["y_axis_val"].dropna().empty else np.nan
    stats_valid = not (np.isnan(y_std) or y_std == 0.0)
    st.session_state["stats_valid"] = stats_valid
    return df


def is_baf_gene(gene: str) -> bool:
    return dp.identify_baf_target(gene) is not None


def highlight_target_cell(val: str) -> str:
    s = str(val)
    if dp.identify_baf_target(s):
        return f"background-color: {BAF_RED}; color: white;"
    if s == "CEBPE":
        return f"background-color: {CEBPE_TEAL}; color: white;"
    if s == "Control (IgG/Mock)":
        return f"background-color: {CONTROL_GREY_BG}; color: {CONTROL_GREY_FG};"
    return f"background-color: {SOFT_BLUE}; color: {TEXT_DARK};"


def rank_abundance_hockey_stick_figure(
    df: pd.DataFrame,
    title: str,
    *,
    abundance_axis_is_tmt: bool = False,
    use_log10: bool = True,
) -> go.Figure:
    """
    Rank–abundance hockey stick: X = rank (1..N), Y = abundance (log10 or linear).
    Light gray rank curve + markers; blue interactors and larger red BAF diamonds.
    Labels annotate top winners for quick experiment readout.
    """
    _sans = "Inter, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
    fig = go.Figure()
    if df.empty or "Gene Symbol" not in df.columns or "Spectral Count" not in df.columns:
        fig.update_layout(title=title, plot_bgcolor=BG_WHITE, paper_bgcolor=BG_WHITE, font={"family": _sans})
        return fig

    work = df[["Gene Symbol", "Spectral Count"]].copy()
    work["Spectral Count"] = pd.to_numeric(work["Spectral Count"], errors="coerce").fillna(0.0)
    work = work.sort_values("Spectral Count", ascending=False).reset_index(drop=True)
    work["Rank"] = np.arange(1, len(work) + 1, dtype=int)
    work["is_baf"] = work["Gene Symbol"].apply(is_baf_gene)
    work["x_plot"] = work["Rank"].astype(float)
    # Safety guard for log plotting: clip to >=1 before log10 to avoid -inf/NaN.
    work["plot_y"] = np.log10(work["Spectral Count"].clip(lower=1.0))
    if use_log10:
        work["y_plot"] = work["plot_y"].astype(float)
        yaxis_title = "Log10(summed S:N)" if abundance_axis_is_tmt else "Log10(spectral count)"
    else:
        work["y_plot"] = work["Spectral Count"].astype(float)
        yaxis_title = "Summed signal-to-noise (TMT)" if abundance_axis_is_tmt else "Spectral count"

    x_all = work["x_plot"].to_numpy()
    y_all = work["y_plot"].to_numpy()
    # Light gray curve linking all ranked proteins (hockey-stick backbone).
    fig.add_trace(
        go.Scatter(
            x=x_all,
            y=y_all,
            mode="lines",
            line={"color": "lightgray", "width": 2},
            name="Rank curve",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    nb_all = work[~work["is_baf"]]
    nb_vis = nb_all[nb_all["Rank"] <= 500]
    baf = work[work["is_baf"]]
    _ht = "%{customdata[0]}<br>Rank %{customdata[1]}<br>Abundance %{customdata[2]:,.0f}<extra></extra>"
    if not nb_vis.empty:
        fig.add_trace(
            go.Scatter(
                x=nb_vis["x_plot"],
                y=nb_vis["y_plot"],
                mode="markers",
                name="Interactors (rank <= 500)",
                marker={
                    "size": 6,
                    "color": RANK_PLOT_MUTED_BLUE,
                    "opacity": 0.88,
                    "line": {"width": 0},
                },
                customdata=np.column_stack(
                    [
                        nb_vis["Gene Symbol"].astype(str),
                        nb_vis["Rank"].astype(int),
                        nb_vis["Spectral Count"].astype(float),
                    ]
                ),
                hovertemplate=_ht,
            )
        )
    if not baf.empty:
        fig.add_trace(
            go.Scatter(
                x=baf["x_plot"],
                y=baf["y_plot"],
                mode="markers",
                name="BAF subunits",
                marker={
                    "size": 12,
                    "color": BAF_RED,
                    "symbol": "diamond",
                    "opacity": 0.96,
                    "line": {"width": 0.5, "color": "#cc3333"},
                },
                customdata=np.column_stack(
                    [baf["Gene Symbol"].astype(str), baf["Rank"].astype(int), baf["Spectral Count"].astype(float)]
                ),
                hovertemplate=_ht,
            )
        )

    # Leader lines: top winners (top 10) with staggered offsets.
    label_ix = set(work.head(min(10, len(work))).index.tolist())
    label_rows = (
        work.loc[sorted(label_ix)]
        .drop_duplicates(subset=["Rank"], keep="first")
        .sort_values("Rank")
        .copy()
    )
    label_rows["Gene Symbol"] = label_rows["Gene Symbol"].fillna("")

    _ay_offsets = [-40.0, -70.0, -100.0]
    _ax_offsets = [44.0, -44.0, 58.0, -58.0]
    for i, (_, r) in enumerate(label_rows.iterrows()):
        _is_b = bool(r["is_baf"])
        rk = int(r["Rank"])
        ay_p = _ay_offsets[i % len(_ay_offsets)]
        ax_p = _ax_offsets[i % len(_ax_offsets)]
        if _is_b:
            # Give BAF labels a little extra radial push away from the cluster.
            ax_p *= 1.35
            ay_p *= 1.20
        raw_rank = float(rk)
        raw_abundance = float(pd.to_numeric(r["Spectral Count"], errors="coerce"))
        if not np.isfinite(raw_rank) or not np.isfinite(raw_abundance):
            continue
        if raw_rank <= 0 or raw_abundance <= 0:
            continue
        x_val = float(raw_rank)
        if use_log10:
            y_val = float(np.log10(max(raw_abundance, 1.0)))
        else:
            y_val = float(raw_abundance)
        if pd.isna(x_val) or pd.isna(y_val):
            continue
        if not np.isfinite(x_val) or not np.isfinite(y_val):
            continue
        gene_txt = str(r["Gene Symbol"]).strip()
        if not gene_txt:
            continue
        try:
            fig.add_annotation(
                xref="x",
                yref="y",
                x=float(x_val),
                y=float(y_val),
                ax=float(ax_p),
                ay=float(ay_p),
                text=gene_txt,
                showarrow=True,
                arrowhead=0,
                arrowwidth=0.5,
                arrowcolor="darkgray",
                arrowsize=0.6,
                font={"family": _sans, "size": 11, "color": TEXT_DARK},
                bgcolor="rgba(255,255,255,0.85)",
                borderpad=2,
                cliponaxis=False,
            )
        except Exception:
            continue

    _max_r = int(work["Rank"].max())
    _x_lo = float(_max_r) * 1.02 if _max_r > 0 else 1.0
    fig.update_layout(
        title=title,
        font={"family": _sans},
        xaxis_title="Rank",
        yaxis_title=yaxis_title,
        plot_bgcolor=BG_WHITE,
        paper_bgcolor=BG_WHITE,
        margin={"l": 72, "r": 72, "t": 96, "b": 72},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        xaxis={
            "range": [0.0, _x_lo],
            "showgrid": False,
            "zeroline": False,
            "automargin": True,
        },
        yaxis={"showgrid": False, "zeroline": False, "automargin": True},
    )
    return fig


@st.cache_data(show_spinner=False)
def load_tmt_wide_cached(path: str) -> pd.DataFrame:
    return dp.read_tmt_excel_wide(Path(path))


def tmt_ma_scatter_figure(ma_df: pd.DataFrame, title: str) -> go.Figure:
    """MA-style comparison: BAF = red, CEBPE = teal, others = ocean blue."""
    fig = go.Figure()
    if ma_df.empty:
        fig.update_layout(title=title, plot_bgcolor=BG_WHITE, paper_bgcolor=BG_WHITE)
        return fig

    def cls_for(g: str) -> str:
        u = str(g).upper()
        if u == "CEBPE":
            return "CEBPE"
        if dp.identify_baf_target(str(g)):
            return "BAF subunit"
        return "Other"

    work = ma_df.copy()
    work["_cls"] = work["Gene Symbol"].map(cls_for)

    for label, color in (("Other", OCEAN_BLUE), ("BAF subunit", BAF_RED), ("CEBPE", CEBPE_TEAL)):
        sub = work[work["_cls"] == label]
        if sub.empty:
            continue
        fc_sub = pd.to_numeric(sub["fold_change"], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=sub["x_avg_log10"],
                y=sub["y_log2_ratio"],
                mode="markers",
                name=label,
                marker={"size": 9, "color": color, "opacity": 0.75},
                text=sub["Gene Symbol"],
                customdata=np.column_stack(
                    [sub["Gene Symbol"].astype(str), sub["hover_conditions"].astype(str), fc_sub]
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Biological condition: %{customdata[1]}<br>"
                    "Fold change (mean T / mean R): %{customdata[2]:.3f}<br>"
                    "Avg log10(intensity): %{x:.4f}<br>"
                    "log2(T/R): %{y:.4f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Average log10 (all selected channels)",
        yaxis_title="log2(mean target / mean reference)",
        plot_bgcolor=BG_WHITE,
        paper_bgcolor=BG_WHITE,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


inject_theme()
st.title("IPMS Viewer")
meta_df = index_library()
if meta_df.empty:
    st.error("No CSV or Excel files found in `Data/`.")
    st.stop()

MODE_CSV = "🔬 Single-Bait Discovery (CSV)"
MODE_TMT = "📊 Multiplex Comparison (TMT)"

if "quick_open_file" not in st.session_state:
    st.session_state["quick_open_file"] = None
if "selected_file" not in st.session_state:
    st.session_state["selected_file"] = None
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Dataset Browser"
if "qc_path" not in st.session_state:
    st.session_state["qc_path"] = None
if "stats_valid" not in st.session_state:
    st.session_state["stats_valid"] = True
if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = MODE_CSV

tab_options = ["Dataset Browser", "Discovery Hub", "Comparative Analysis", "Data Management", "Feedback"]
if st.session_state["active_tab"] not in tab_options:
    st.session_state["active_tab"] = "Dataset Browser"

with st.sidebar:
    admin_mode = st.checkbox("Admin Mode", value=False)
    nav_options = tab_options + (["Admin Control"] if admin_mode else [])
    if st.session_state["active_tab"] not in nav_options:
        st.session_state["active_tab"] = "Dataset Browser"
    st.session_state["app_mode"] = st.selectbox(
        "Analysis Pipeline",
        options=[MODE_CSV, MODE_TMT],
        index=0 if st.session_state["app_mode"] == MODE_CSV else 1,
    )
    st.session_state["active_tab"] = st.radio(
        "Navigation",
        options=nav_options,
        index=nav_options.index(st.session_state["active_tab"]),
    )

mode_is_tmt = st.session_state["app_mode"] == MODE_TMT
meta_df_mode = meta_df[
    meta_df["path"].apply(lambda p: Path(str(p)).suffix.lower() == (".xlsx" if mode_is_tmt else ".csv"))
].copy()

if st.session_state["active_tab"] == "Dataset Browser":
    section_header("Dataset Browser (Control Center)", BAF_RED)
    if meta_df_mode.empty:
        st.warning("No experiments found for the selected analysis pipeline.")
        st.stop()
    meta_search = st.text_input(
        "Filter experiments",
        value="",
        key="dataset_browser_meta_search",
        placeholder="Search Target, Biological Condition, file name, cell line… (e.g. GPF)",
    ).strip()
    investigators = sorted(meta_df_mode["investigator"].unique().tolist())
    default_inv = investigators[0]
    preferred_file = st.session_state.get("selected_file") or st.session_state.get("quick_open_file")
    if preferred_file in meta_df_mode["file_name"].tolist():
        default_inv = meta_df_mode.loc[meta_df_mode["file_name"] == preferred_file, "investigator"].iloc[0]
    selected_inv = st.selectbox("Investigator", options=investigators, index=investigators.index(default_inv))
    inv_df_raw = meta_df_mode[meta_df_mode["investigator"] == selected_inv].copy()
    inv_df = enrich_manifest(inv_df_raw)
    if "experiment_type" not in inv_df.columns:
        inv_df = inv_df.copy()
        inv_df["experiment_type"] = "Label-Free"
    if "biological_condition" not in inv_df.columns:
        inv_df = inv_df.copy()
        inv_df["biological_condition"] = "Single Run"
    if meta_search:
        q = meta_search.upper()

        def _contains(series: pd.Series) -> pd.Series:
            return series.astype(str).str.upper().str.contains(q, na=False, regex=False)

        mask = _contains(inv_df["target"])
        for col in (
            "biological_condition",
            "details",
            "display_name",
            "cell_line",
            "file_name",
            "full_filename",
            "sample_label",
            "session_id",
            "initials",
            "tmt_channel",
        ):
            if col in inv_df.columns:
                mask = mask | _contains(inv_df[col])
        inv_df = inv_df[mask].copy()
    show_tmt_columns = mode_is_tmt
    browse_cols = ["session_id", "initials", "target", "details"]
    if show_tmt_columns:
        browse_cols.extend(["biological_condition", "experiment_type"])
    browse_cols.extend(["cell_line", "sample_label", "full_filename"])
    browse_cols = [c for c in browse_cols if c in inv_df.columns]
    table_df = inv_df[browse_cols].rename(
        columns={
            "session_id": "Session ID",
            "initials": "Initials",
            "target": "Target",
            "details": "Details",
            "biological_condition": "Biological Condition",
            "cell_line": "Cell Line",
            "sample_label": "Sample Label",
            "experiment_type": "Type",
            "full_filename": "Full Filename",
        }
    )
    if not table_df.empty:
        col_cfg = {
            "Biological Condition": st.column_config.TextColumn(
                "Biological Condition",
                width="large",
                help="Row-1 multiplex label (Kevin / Jessica) or Single Run for CSV.",
            ),
            "Details": st.column_config.TextColumn(
                "Details",
                width="large",
                help="Specific experimental details",
            ),
        }
        try:
            st.dataframe(
                table_df.style.map(highlight_target_cell, subset=["Target"]),
                use_container_width=True,
                column_config=col_cfg,
            )
        except Exception as e:
            print(f"Styled dataframe / column_config failed: {e}")
            try:
                st.dataframe(table_df, use_container_width=True, column_config=col_cfg)
            except Exception:
                st.dataframe(table_df, use_container_width=True)
    else:
        st.info("No experiments available for this investigator.")
    options = inv_df["file_name"].tolist()
    if not options:
        st.warning("No matching experiments found. Please adjust your filters.")
        st.stop()
    default_idx = 0
    if preferred_file in options:
        default_idx = options.index(preferred_file)
    idx_options = inv_df.index.tolist()
    default_row_idx = idx_options[default_idx]
    selected_idx = st.selectbox(
        "Select experiment",
        options=idx_options,
        index=idx_options.index(default_row_idx),
        format_func=lambda i: str(inv_df.loc[i, "display_name"]) if "display_name" in inv_df.columns else str(inv_df.loc[i, "file_name"]),
    )
    if selected_idx is None:
        st.stop()
    selected_row = inv_df.loc[selected_idx]
    st.session_state["selected_file"] = str(selected_row["file_name"])
    st.session_state["qc_path"] = selected_row["path"]
    st.markdown("")
    raw_fname = str(selected_row.get("full_filename") or Path(str(selected_row["path"])).name)
    try:
        raw_bytes = get_file_binary(str(selected_row["path"]))
    except Exception as exc:
        st.caption(f"Raw download unavailable: {exc}")
    else:
        raw_mime = (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            if raw_fname.lower().endswith(".xlsx")
            else "text/csv"
        )
        st.download_button(
            label=f"📥 Download {raw_fname}",
            data=raw_bytes,
            file_name=raw_fname,
            mime=raw_mime,
            key=f"dl_raw_{hashlib.md5(str(selected_row['path']).encode()).hexdigest()[:16]}",
        )
    tmt_sn = tmt_sn_col_from_row(selected_row)
    is_tmt = tmt_sn is not None
    selected_suffix = Path(str(selected_row["path"])).suffix.lower()
    if st.session_state["app_mode"] == MODE_CSV and selected_suffix == ".xlsx":
        st.warning("Please switch to TMT mode to view Excel data.")
        exp = pd.DataFrame(columns=["Gene Symbol", "Spectral Count", "Unique Peptides", "LDA Probability"])
    elif st.session_state["app_mode"] == MODE_TMT and selected_suffix == ".csv":
        st.warning("Please switch to CSV mode to view label-free data.")
        exp = pd.DataFrame(columns=["Gene Symbol", "Spectral Count", "Unique Peptides", "LDA Probability"])
    else:
        try:
            exp = load_and_process_file(selected_row["path"], tmt_sn)
        except Exception as exc:
            st.error(f"TMT / file load failed: {exc}")
            exp = pd.DataFrame(columns=["Gene Symbol", "Spectral Count", "Unique Peptides", "LDA Probability"])
    if is_tmt:
        st.caption("📍 Mode: TMT Multiplexed Data")
    if exp.empty:
        st.warning("No quantitative rows loaded for this selection.")
    exp["is_baf"] = exp["Gene Symbol"].apply(is_baf_gene)

    total_proteins = int(exp["Gene Symbol"].nunique()) if not exp.empty else 0
    baf_ranked = exp[exp["is_baf"]].sort_values("Spectral Count", ascending=False) if not exp.empty else exp
    baf_count = int(baf_ranked["Gene Symbol"].nunique()) if not baf_ranked.empty else 0
    bait_label = str(selected_row["target"])
    if exp.empty:
        top_inter = pd.Series(dtype=object)
    elif bait_label in ("Unknown", "Control (IgG/Mock)"):
        top_inter = exp.sort_values("Spectral Count", ascending=False)["Gene Symbol"].head(1)
    else:
        top_inter = exp[exp["Gene Symbol"] != bait_label.upper()].sort_values("Spectral Count", ascending=False)["Gene Symbol"].head(1)
    top_interactor = top_inter.iloc[0] if not top_inter.empty else "N/A"
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Proteins", f"{total_proteins:,}")
    m2.metric("BAF Subunits (Ranked)", f"{baf_count}/26")
    m3.metric("Top Interactor", top_interactor)

    if not exp.empty:
        st.markdown("#### Rank–abundance (hockey stick)")
        _scale = st.radio(
            "Scale Type",
            ["Log10", "Linear"],
            horizontal=True,
            key="dataset_browser_rank_scale",
        )
        st.plotly_chart(
            rank_abundance_hockey_stick_figure(
                exp,
                "Experiment rank plot",
                abundance_axis_is_tmt=is_tmt,
                use_log10=(_scale == "Log10"),
            ),
            use_container_width=True,
        )
        st.caption(
            "Proteins ranked by abundance (spectral count for CSV; summed S:N for TMT). "
            "Log10: **Log10** uses rank on X and log10(abundance, clipped >=1) on Y; **Linear** uses rank on X with linear abundance. "
            "Light gray rank curve; blue interactors (top 500 only); red BAF diamonds. Hover: symbol, rank, raw abundance."
        )

    coverage = exp.copy()
    if not exp.empty:
        coverage["canonical"] = coverage["Gene Symbol"].apply(lambda g: dp.identify_baf_target(g) or "")
        coverage = coverage[coverage["canonical"] != ""].groupby("canonical", as_index=False)["Spectral Count"].sum()
        missing = [x for x in dp.BAF_CORE_CANONICAL if x not in coverage["canonical"].tolist()]
        if missing:
            coverage = pd.concat([coverage, pd.DataFrame({"canonical": missing, "Spectral Count": [0] * len(missing)})], ignore_index=True)
        coverage = coverage.sort_values("Spectral Count", ascending=False)
        cov_fig = px.bar(coverage, x="Spectral Count", y="canonical", orientation="h", title="Complex Coverage (BAF Core)", color_discrete_sequence=[BAF_RED])
        cov_fig.update_yaxes(autorange="reversed")
        cov_x = "Summed Signal-to-Noise (TMT)" if is_tmt else "Spectral Count"
        cov_fig.update_layout(plot_bgcolor=BG_WHITE, paper_bgcolor=BG_WHITE, xaxis_title=cov_x)
        st.plotly_chart(cov_fig, use_container_width=True)
    with st.expander("📂 View Raw Experiment Data", expanded=False):
        if exp.empty:
            st.caption("No rows to display.")
        else:
            raw_df = exp.sort_values("Spectral Count", ascending=False).reset_index(drop=True)
            if is_tmt:
                # UI-only projection for TMT protein list (do not mutate raw_df / exp)
                tmt_view = pd.DataFrame({"Gene Symbol": raw_df["Gene Symbol"]})
                tmt_view["Description"] = raw_df["Description"] if "Description" in raw_df.columns else ""
                tmt_view["Biological Condition"] = str(selected_row.get("biological_condition") or "Unlabeled")
                tmt_view["Log2 Fold Change"] = (
                    pd.to_numeric(raw_df["Log2 Fold Change"], errors="coerce")
                    if "Log2 Fold Change" in raw_df.columns
                    else np.nan
                )
                tmt_view["Average Intensity"] = pd.to_numeric(raw_df["Spectral Count"], errors="coerce")
                st.dataframe(tmt_view, use_container_width=True)
            else:
                st.dataframe(raw_df, use_container_width=True)

    xlsx_path = Path(selected_row["path"])
    wide_ma: Optional[pd.DataFrame] = None
    channel_opts: list = []
    ref_sel: list = []
    tgt_sel: list = []
    if st.session_state["app_mode"] == MODE_TMT and xlsx_path.suffix.lower() == ".xlsx":
        with st.sidebar:
            st.markdown("---")
            st.markdown("**TMT comparison (MA)**")
            try:
                wide_ma = load_tmt_wide_cached(str(xlsx_path))
                channel_opts = dp.list_tmt_sn_sum_columns(wide_ma)
            except Exception as exc:
                st.caption(f"Could not load wide sheet for comparison: {exc}")
            if channel_opts:
                sid = hashlib.md5(str(xlsx_path).encode()).hexdigest()[:12]
                ref_sel = st.multiselect(
                    "Reference channels (e.g. IgG)",
                    options=channel_opts,
                    key=f"tmt_ma_ref_{sid}",
                )
                tgt_sel = st.multiselect(
                    "Target channels (e.g. BRG1)",
                    options=channel_opts,
                    key=f"tmt_ma_tgt_{sid}",
                )
            else:
                st.caption("No _sn_sum channel columns found.")

        if wide_ma is not None and ref_sel and tgt_sel:
            st.markdown("#### TMT comparison (MA plot)")
            try:
                ma_df = dp.build_tmt_ma_comparison_df(wide_ma, ref_sel, tgt_sel)
                st.plotly_chart(
                    tmt_ma_scatter_figure(
                        ma_df,
                        f"MA: {xlsx_path.name} — reference vs target channels",
                    ),
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"TMT comparison plot failed: {exc}")

if st.session_state["active_tab"] == "Discovery Hub":
    section_header("Discovery Hub", OCEAN_BLUE)
    _, center, _ = st.columns([1, 2, 1])
    with center:
        gene_query = st.text_input("Gene of Interest", value="SS18").strip().upper()
    compare_core = st.toggle("Compare with Core BAF", value=False)

    if not gene_query:
        st.info("Enter a gene symbol to search.")
        st.stop()

    def _normalize_for_details_search(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()

    def _details_match(details_value: str, raw_query: str) -> bool:
        q = _normalize_for_details_search(raw_query)
        if not q:
            return True
        d = _normalize_for_details_search(details_value)
        q_tokens = [t for t in q.split() if t]
        return bool(q_tokens) and all(t in d for t in q_tokens)

    def _discovery_select_options(series: pd.Series) -> List[str]:
        vals = sorted(
            {
                str(x).strip()
                for x in series.dropna().unique()
                if str(x).strip() and str(x).strip().lower() not in ("nan", "none")
            }
        )
        return ["All"] + vals

    hub_meta = enrich_manifest(meta_df_mode)
    if hub_meta.empty:
        st.warning("No experiments available for the selected analysis pipeline.")
        st.stop()

    _tag_keys = list(dp.DISCOVERY_TAG_KEYS)
    try:
        _tag_df = hub_meta.apply(
            lambda r: pd.Series(dp.discovery_filter_tag_row_from_record(r)),
            axis=1,
        )
    except Exception:
        _tag_df = pd.DataFrame(
            [dp.discovery_filter_tag_defaults_unknown() for _ in range(len(hub_meta))],
            index=hub_meta.index,
        )
    if _tag_df.empty or set(_tag_df.columns) != set(_tag_keys):
        _tag_df = pd.DataFrame(
            [dp.discovery_filter_tag_defaults_unknown() for _ in range(len(hub_meta))],
            index=hub_meta.index,
        )
        _tag_df = _tag_df[_tag_keys]
    hub_ctx = pd.concat([hub_meta.reset_index(drop=True), _tag_df.reset_index(drop=True)], axis=1)
    if "experiment_type" in hub_ctx.columns:
        hub_ctx["experiment_type"] = hub_ctx["experiment_type"].fillna("Unknown").astype(str)
    else:
        hub_ctx["experiment_type"] = "Unknown"

    st.markdown("#### Filter experiments")
    bio1, bio2, bio3, bio4, bio5 = st.columns(5)
    with bio1:
        sel_investigator = st.selectbox(
            "Investigator",
            _discovery_select_options(hub_ctx["investigator"]),
            key="dh_bio_investigator",
        )
    with bio2:
        sel_target = st.selectbox("Bait / Target", _discovery_select_options(hub_ctx["target"]), key="dh_bio_target")
    with bio3:
        sel_cell = st.selectbox("Cell Line", _discovery_select_options(hub_ctx["cell_line"]), key="dh_bio_cell")
    with bio4:
        sel_genetic = st.selectbox(
            "Genetic background",
            _discovery_select_options(hub_ctx["genetic_background"]),
            key="dh_bio_genetic",
        )
    with bio5:
        sel_treatment = st.selectbox(
            "Treatment",
            _discovery_select_options(hub_ctx["treatment"]),
            key="dh_bio_treatment",
        )
    tech1, tech2, tech3, tech4, tech5 = st.columns(5)
    with tech1:
        sel_exp_type = st.selectbox(
            "Experiment type",
            _discovery_select_options(hub_ctx["experiment_type"]),
            key="dh_tech_exptype",
        )
    with tech2:
        sel_tag = st.selectbox(
            "Purification tag",
            _discovery_select_options(hub_ctx["purification_tag"]),
            key="dh_tech_tag",
        )
    with tech3:
        sel_conc = st.selectbox(
            "Concentration method",
            _discovery_select_options(hub_ctx["concentration_method"]),
            key="dh_tech_conc",
        )
    with tech4:
        sel_fix = st.selectbox(
            "Fixation / crosslinking",
            _discovery_select_options(hub_ctx["fixation_crosslinking"]),
            key="dh_tech_fix",
        )
    with tech5:
        sel_nuc = st.selectbox(
            "Nuclease treatment",
            _discovery_select_options(hub_ctx["nuclease_treatment"]),
            key="dh_tech_nuc",
        )
    details_keyword = st.text_input(
        "Keyword Search in Details",
        value="",
        key="dh_details_keyword",
        placeholder="e.g. sonication settings, buffer notes",
    ).strip()

    _mask = pd.Series(True, index=hub_ctx.index)
    if sel_investigator != "All":
        _mask &= hub_ctx["investigator"].astype(str).str.strip() == sel_investigator
    if sel_target != "All":
        _mask &= hub_ctx["target"].astype(str).str.strip() == sel_target
    if sel_cell != "All":
        _mask &= hub_ctx["cell_line"].astype(str).str.strip() == sel_cell
    if sel_genetic != "All":
        _mask &= hub_ctx["genetic_background"].astype(str).str.strip() == sel_genetic
    if sel_treatment != "All":
        _mask &= hub_ctx["treatment"].astype(str).str.strip() == sel_treatment
    if sel_exp_type != "All":
        _mask &= hub_ctx["experiment_type"].astype(str).str.strip() == sel_exp_type
    if sel_tag != "All":
        _mask &= hub_ctx["purification_tag"].astype(str).str.strip() == sel_tag
    if sel_conc != "All":
        _mask &= hub_ctx["concentration_method"].astype(str).str.strip() == sel_conc
    if sel_fix != "All":
        _mask &= hub_ctx["fixation_crosslinking"].astype(str).str.strip() == sel_fix
    if sel_nuc != "All":
        _mask &= hub_ctx["nuclease_treatment"].astype(str).str.strip() == sel_nuc
    hub_scan = hub_ctx.loc[_mask].copy()

    bait_for_consensus = dp.resolve_search_as_bait(gene_query)

    def _manifest_row_is_primary_bait(manifest_row: Dict[str, Any], gq: str) -> bool:
        """True when this run's indexed bait matches the gene query (symbol or BAF alias, e.g. BRG1 vs SMARCA4)."""
        q = (gq or "").strip().upper()
        if not q:
            return False
        tgt = str(manifest_row.get("target", "")).strip()
        if not tgt:
            return False
        if tgt.upper() == q:
            return True
        if q == "CEBPE" and tgt.upper() == "CEBPE":
            return True
        q_can = dp.identify_baf_target(q)
        t_can = dp.identify_baf_target(tgt)
        return bool(q_can and t_can and q_can == t_can)

    st.divider()
    st.markdown("### Global overview")
    st.caption("Runs matching your gene search after filters. Rows highlighted in amber: this run’s indexed bait matches the gene you typed.")
    rows = []
    if hub_scan.empty:
        st.warning("No experiments match the selected filters.")
    else:
        prog = st.progress(0)
        files = hub_scan.to_dict(orient="records")
        for i, row in enumerate(files, start=1):
            e = load_experiment_summary(row["path"], tmt_sn_col_from_row(row))
            hit = dp.filter_rows_for_gene(e, gene_query)
            meta_hit = dp.hub_manifest_row_matches_global_query(row, gene_query)
            if not hit.empty:
                best = hit.sort_values("Spectral Count", ascending=False).iloc[0]
                rows.append(
                    {
                        "Investigator": row["investigator"],
                        "Target": row["target"],
                        "Cell Line": row["cell_line"],
                        "Exp ID": row["session_id"],
                        "Spectral Count": float(best["Spectral Count"]),
                        "Unique Peptides": int(best["Unique Peptides"]),
                        "Core BAF IP": (
                            dp.is_core_baf_canonical(dp.identify_baf_target(str(row["target"])))
                            or (str(row["target"]).upper() == "CEBPE")
                        ),
                        "Details": str(row.get("details", "N/A")),
                        "File Name": row["file_name"],
                        "_is_primary_bait": _manifest_row_is_primary_bait(row, gene_query),
                    }
                )
            elif meta_hit:
                rows.append(
                    {
                        "Investigator": row["investigator"],
                        "Target": row["target"],
                        "Cell Line": row["cell_line"],
                        "Exp ID": row["session_id"],
                        "Spectral Count": 0.0,
                        "Unique Peptides": 0,
                        "Core BAF IP": (
                            dp.is_core_baf_canonical(dp.identify_baf_target(str(row["target"])))
                            or (str(row["target"]).upper() == "CEBPE")
                        ),
                        "Details": str(row.get("details", "N/A")),
                        "File Name": row["file_name"],
                        "_is_primary_bait": _manifest_row_is_primary_bait(row, gene_query),
                    }
                )
            prog.progress(i / len(files))
        prog.empty()

    if rows:
        res_all = pd.DataFrame(rows).sort_values("Spectral Count", ascending=False)
        if details_keyword:
            res = res_all[res_all["Details"].apply(lambda d: _details_match(d, details_keyword))].copy()
        else:
            res = res_all.copy()
        if res.empty and details_keyword:
            st.info("No experiments found matching those technical criteria.")
            st.stop()
        st.markdown("#### Global results")
        res_view = res.drop(columns=["_is_primary_bait"], errors="ignore")
        if mode_is_tmt:
            res_view = res_view.drop(columns=["Exp ID"], errors="ignore")
        if not mode_is_tmt:
            res_view = res_view.drop(columns=["Type", "Biological Condition"], errors="ignore")
        _disp_cols = ["Investigator", "Target", "Cell Line", "Spectral Count", "Unique Peptides", "Details"]
        if not mode_is_tmt:
            _disp_cols.insert(3, "Exp ID")
        if compare_core and "Core BAF IP" in res_view.columns:
            _disp_cols.append("Core BAF IP")
        res_disp = res_view[[c for c in _disp_cols if c in res_view.columns]].copy()
        res_disp["Spectral Count"] = pd.to_numeric(res_disp["Spectral Count"], errors="coerce").fillna(0.0).round().astype(np.int64)
        res_disp["Unique Peptides"] = pd.to_numeric(res_disp["Unique Peptides"], errors="coerce").fillna(0).round().astype(np.int64)
        _bait_mask = res["_is_primary_bait"].reindex(res_disp.index).fillna(False)

        def _row_highlight_primary_bait(row: pd.Series) -> List[str]:
            if bool(_bait_mask.loc[row.name]):
                return ["background-color: #fff3cd; color: #1f2937;"] * len(row)
            return [""] * len(row)

        _global_height = min(560, 52 + 38 * max(1, len(res_disp)))
        _global_col_cfg: Dict[str, Any] = {
            "Spectral Count": st.column_config.NumberColumn(
                "Spectral Count",
                help="Summed intensity (TMT) or spectral-derived count (CSV).",
                format="%d",
                width="medium",
            ),
            "Unique Peptides": st.column_config.NumberColumn(
                "Unique Peptides",
                format="%d",
                width="small",
            ),
            "Details": st.column_config.TextColumn("Details", width="large", help="Experiment notes (from manifest / overrides)."),
        }
        try:
            _styled = res_disp.style.apply(_row_highlight_primary_bait, axis=1)
            if compare_core and "Core BAF IP" in res_disp.columns:
                _styled = _styled.map(
                    lambda v: f"background-color: {SOFT_BLUE}; color: {TEXT_DARK};" if v else "",
                    subset=["Core BAF IP"],
                )
            st.dataframe(_styled, use_container_width=True, height=_global_height, column_config=_global_col_cfg)
        except Exception:
            st.dataframe(res_disp, use_container_width=True, height=_global_height, column_config=_global_col_cfg)

        qf = st.selectbox("Quick Open experiment", options=res["File Name"].tolist())
        if st.button("Open in Dataset Browser"):
            st.session_state["selected_file"] = qf
            st.session_state["quick_open_file"] = qf
            st.session_state["active_tab"] = "Dataset Browser"
            st.rerun()

        inv_dist = res.groupby("Investigator", as_index=False)["Spectral Count"].count().rename(columns={"Spectral Count": "Hit Count"}).sort_values("Hit Count", ascending=False)
        _cell_plot_src = res.copy()
        _cell_plot_src["_upep_cell"] = dp.unique_peptides_numeric_series(_cell_plot_src)
        cell_enrich = (
            _cell_plot_src.groupby("Cell Line", as_index=False)
            .agg(
                Unique_Peptides_total=("_upep_cell", "sum"),
                Investigator=(
                    "Investigator",
                    lambda s: ", ".join(
                        sorted(
                            {
                                str(x).strip()
                                for x in s.dropna().unique()
                                if str(x).strip() and str(x).strip().lower() not in ("nan", "none")
                            }
                        )
                    )
                    or "N/A",
                ),
            )
            .rename(columns={"Unique_Peptides_total": "Unique Peptides"})
            .sort_values("Unique Peptides", ascending=False)
        )
        cell_enrich["Unique Peptides"] = pd.to_numeric(cell_enrich["Unique Peptides"], errors="coerce").fillna(0.0).round().astype(int)
        st.markdown("#### Enrichment across filtered runs")
        g1, g2 = st.columns(2)
        with g1:
            fig1 = px.bar(inv_dist, x="Investigator", y="Hit Count", title="Enrichment by Investigator", color_discrete_sequence=[OCEAN_BLUE])
            fig1.update_layout(plot_bgcolor=BG_WHITE, paper_bgcolor=BG_WHITE)
            st.plotly_chart(fig1, use_container_width=True)
        with g2:
            fig2 = px.bar(
                cell_enrich,
                x="Cell Line",
                y="Unique Peptides",
                title="Enrichment by Cell Line",
                color_discrete_sequence=[EMERALD],
            )
            fig2.update_layout(
                plot_bgcolor=BG_WHITE,
                paper_bgcolor=BG_WHITE,
                yaxis_title="Unique Peptides",
            )
            fig2.update_traces(
                hovertemplate=(
                    "<b>Cell Line:</b> %{x}<br>"
                    "<b>Unique Peptides:</b> %{y}<br>"
                    "<b>Investigator:</b> %{customdata[0]}<extra></extra>"
                ),
                customdata=cell_enrich[["Investigator"]].to_numpy(),
            )
            st.plotly_chart(fig2, use_container_width=True)
    elif not hub_scan.empty:
        st.info("No hits found for this gene in the filtered experiments.")

    st.divider()
    st.header("\U0001F3AF Consensus analysis")
    if bait_for_consensus is not None:
        bait_runs = hub_meta[
            hub_meta["target"].apply(lambda t: dp.manifest_row_matches_bait(str(t), bait_for_consensus))
        ].copy()
        st.markdown("#### Indexed runs for this bait")
        if bait_for_consensus == "CEBPE":
            st.caption(f"Experiments indexed with **CEBPE** as the IP bait (outside the 26-gene BAF core list).")
        else:
            st.caption(f"Experiments indexed with **{bait_for_consensus}** (or an accepted alias) as the IP bait.")
        if not bait_runs.empty:
            enrich_cols = bait_runs[
                ["investigator", "session_id", "cell_line", "sample_label", "details", "display_name", "full_filename"]
            ].rename(
                columns={
                    "investigator": "Investigator",
                    "session_id": "Exp ID",
                    "cell_line": "Cell Line",
                    "sample_label": "Sample Label",
                    "details": "Details",
                    "display_name": "Display Name",
                    "full_filename": "Full Filename",
                }
            )
            enrich_view = enrich_cols.sort_values("Exp ID")
            if mode_is_tmt and "Exp ID" in enrich_view.columns:
                enrich_view = enrich_view.drop(columns=["Exp ID"])
            if not mode_is_tmt:
                enrich_view = enrich_view.drop(columns=["Type", "Biological Condition"], errors="ignore")
            st.dataframe(
                enrich_view,
                use_container_width=True,
                height=min(420, 48 + 34 * max(1, len(enrich_view))),
                column_config={
                    "Details": st.column_config.TextColumn("Details", width="large", help="Specific experimental details")
                },
            )
            with st.expander("\U0001F50D View consensus interactor table", expanded=False):
                st.caption(
                    f"Proteins meeting prevalence across **{len(bait_runs)}** indexed run(s) with bait **{bait_for_consensus}** "
                    "(≥50% of runs). Opens here so global catalog stays separate from bait-specific discovery."
                )
                all_rows_c = []
                for _, row in bait_runs.iterrows():
                    e = load_experiment_summary(row["path"], tmt_sn_col_from_row(row)).copy()
                    e["exp_id"] = row["session_id"]
                    all_rows_c.append(e)
                merged_c = pd.concat(all_rows_c, ignore_index=True)
                prevalence = merged_c.groupby("Gene Symbol")["exp_id"].nunique()
                thr = max(1, math.ceil(len(bait_runs) * 0.5))
                keep = prevalence[prevalence >= thr].index
                consensus = (
                    merged_c[merged_c["Gene Symbol"].isin(keep)]
                    .groupby("Gene Symbol", as_index=False)
                    .agg(
                        Mean_Spectral_Count=("Spectral Count", "mean"),
                        Mean_Unique_Peptides=("Unique Peptides", "mean"),
                        Runs_Present=("exp_id", "nunique"),
                    )
                    .sort_values("Mean_Spectral_Count", ascending=False)
                )
                consensus["Prevalence Score (%)"] = (consensus["Runs_Present"] / len(bait_runs) * 100).round(1)
                st.dataframe(consensus, use_container_width=True)
        else:
            st.info(f"No runs indexed with bait target **{bait_for_consensus}**.")
    else:
        st.caption("Search a **core BAF** bait symbol (or **CEBPE**) to show bait-indexed runs and a consensus interactor table.")

if st.session_state["active_tab"] == "Comparative Analysis":
    section_header("Comparative Analysis", EMERALD)
    if meta_df_mode.empty:
        st.warning("No experiments available for the selected analysis pipeline.")
        st.stop()
    picks = st.multiselect("Choose 2 to 4 experiments", options=meta_df_mode["file_name"].tolist(), max_selections=4)
    if len(picks) == 2:
        a_file, b_file = picks[0], picks[1]
        a_row = enrich_manifest(meta_df_mode[meta_df_mode["file_name"] == a_file]).iloc[0]
        b_row = enrich_manifest(meta_df_mode[meta_df_mode["file_name"] == b_file]).iloc[0]
        a_sn, b_sn = tmt_sn_col_from_row(a_row), tmt_sn_col_from_row(b_row)
        a_exp = load_and_process_file(a_row["path"], a_sn)
        b_exp = load_and_process_file(b_row["path"], b_sn)
        st.markdown("#### Rank–abundance (pair)")
        _cmp_scale = st.radio(
            "Scale Type",
            ["Log10", "Linear"],
            horizontal=True,
            key="comparative_rank_scale",
        )
        _cmp_log = _cmp_scale == "Log10"
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                rank_abundance_hockey_stick_figure(
                    a_exp,
                    f"Rank plot: {a_file}",
                    abundance_axis_is_tmt=bool(a_sn),
                    use_log10=_cmp_log,
                ),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                rank_abundance_hockey_stick_figure(
                    b_exp,
                    f"Rank plot: {b_file}",
                    abundance_axis_is_tmt=bool(b_sn),
                    use_log10=_cmp_log,
                ),
                use_container_width=True,
            )
        st.caption(
            "Same scale options as Dataset Browser: **Log10** = rank on X with log10(abundance, clipped >=1) on Y; "
            "**Linear** = rank on X with linear abundance. Blue markers show top 500 interactors; BAF are always shown."
        )

        df1 = a_exp[["Gene Symbol", "Spectral Count"]]
        df2 = b_exp[["Gene Symbol", "Spectral Count"]]
        overlap_df = pd.merge(df1, df2, on="Gene Symbol", how="outer", suffixes=("_A", "_B")).fillna(0)
        overlap_df = overlap_df.sort_values(by=["Spectral Count_A", "Spectral Count_B"], ascending=False)
        st.markdown("#### Spectral overlap (outer join)")
        st.dataframe(overlap_df, use_container_width=True)
        only_a = overlap_df[(overlap_df["Spectral Count_A"] > 0) & (overlap_df["Spectral Count_B"] == 0)]["Gene Symbol"].head(200)
        only_b = overlap_df[(overlap_df["Spectral Count_B"] > 0) & (overlap_df["Spectral Count_A"] == 0)]["Gene Symbol"].head(200)
        both = overlap_df[(overlap_df["Spectral Count_A"] > 0) & (overlap_df["Spectral Count_B"] > 0)]["Gene Symbol"].head(200)
        summary = pd.concat(
            [
                only_a.reset_index(drop=True).rename("Unique to Exp A"),
                only_b.reset_index(drop=True).rename("Unique to Exp B"),
                both.reset_index(drop=True).rename("Common Interactors"),
            ],
            axis=1,
        )
        st.markdown("#### Category preview")
        st.dataframe(summary, use_container_width=True)
    elif len(picks) >= 3:
        subset = meta_df_mode[meta_df_mode["file_name"].isin(picks)].copy()
        merged = None
        for _, row in subset.iterrows():
            e = load_experiment_summary(row["path"], tmt_sn_col_from_row(row))[
                ["Gene Symbol", "Spectral Count"]
            ].rename(columns={"Spectral Count": row["file_name"]})
            merged = e if merged is None else merged.merge(e, on="Gene Symbol", how="outer")
        if merged is not None:
            merged = merged.fillna(0)
            corr = merged[picks].corr(method="pearson")
            heat = px.imshow(corr, text_auto=True, title="Pearson Correlation Heatmap", color_continuous_scale="Blues")
            heat.update_layout(plot_bgcolor=BG_WHITE, paper_bgcolor=BG_WHITE)
            st.plotly_chart(heat, use_container_width=True)
            baf_matrix = merged[merged["Gene Symbol"].apply(is_baf_gene)].copy()
            baf_matrix["Canonical"] = baf_matrix["Gene Symbol"].apply(lambda g: dp.identify_baf_target(g) or g)
            baf_matrix = baf_matrix.groupby("Canonical", as_index=False)[picks].sum().sort_values("Canonical")
            st.markdown("#### BAF Subunit Spectral Matrix")
            st.dataframe(baf_matrix, use_container_width=True)
    else:
        st.info("Select at least 2 experiments.")

if st.session_state["active_tab"] == "Data Management":
    section_header("Data Management", OCEAN_BLUE)
    c1, c2 = st.columns([2, 1])
    with c1:
        upload_target = st.selectbox("Upload target investigator folder", options=sorted(meta_df["investigator"].unique().tolist()))
        uploads = st.file_uploader("Upload CSV or Excel files", type=["csv", "xlsx"], accept_multiple_files=True)
        if st.button("Save Uploads") and uploads:
            folder = DATA_ROOT / upload_target
            folder.mkdir(parents=True, exist_ok=True)
            for f in uploads:
                (folder / f.name).write_bytes(f.getvalue())
            index_library.clear()
            get_path_metadata.clear()
            load_experiment_summary.clear()
            st.success(f"Saved {len(uploads)} file(s).")
    with c2:
        if st.button("Refresh Metadata"):
            index_library.clear()
            get_path_metadata.clear()
            load_experiment_summary.clear()
            st.success("Metadata recrawl enabled.")

    tree = []
    for inv_dir in sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()]):
        count = len([x for x in inv_dir.glob("*.csv") if "_PROCESSED" not in x.name.upper()])
        tree.append({"Investigator Folder": inv_dir.name, "CSV Files": count})
    st.markdown("#### Folder Mapping")
    st.dataframe(pd.DataFrame(tree), use_container_width=True)

    delete_file = st.selectbox("Delete file", options=meta_df["file_name"].tolist())
    if st.button("Delete Selected File"):
        hit = meta_df[meta_df["file_name"] == delete_file].head(1)
        if not hit.empty:
            Path(hit.iloc[0]["path"]).unlink(missing_ok=True)
            index_library.clear()
            get_path_metadata.clear()
            load_experiment_summary.clear()
            st.warning(f"Deleted: {delete_file}")

if st.session_state["active_tab"] == "Feedback":
    st.header("🚀 Help us improve the BAF IP-MS Viewer")
    st.subheader("Found a bug? Want a new chart? Let us know below.")
    with st.form("feedback_form"):
        name = st.text_input("Name (Optional)")
        category = st.selectbox(
            "Category",
            ["Bug Report", "Feature Request", "Data Issue", "General Feedback"],
        )
        suggestion = st.text_area("Your Suggestion", help="Tell us what's on your mind...")
        submit = st.form_submit_button("Submit 🚀")
    if submit:
        if not suggestion or not str(suggestion).strip():
            st.warning("Please enter a suggestion before submitting.")
        else:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            name_part = str(name).strip() if str(name).strip() else "Anonymous"
            sug = str(suggestion).strip()
            _fb_cols = ["Timestamp", "Name", "Category", "Suggestion"]
            try:
                existing_data = conn.read(worksheet="Sheet1", usecols=[0, 1, 2, 3], ttl=0)
                if existing_data is None or existing_data.empty or existing_data.shape[1] < 4:
                    existing_data = pd.DataFrame(columns=_fb_cols)
                else:
                    existing_data = existing_data.iloc[:, :4].copy()
                    existing_data.columns = _fb_cols
                new_row_df = pd.DataFrame([[ts, name_part, category, sug]], columns=_fb_cols)
                updated_df = pd.concat([existing_data, new_row_df], ignore_index=True)
                conn.update(worksheet="Sheet1", data=updated_df)
                st.success("Feedback sent directly to Google Sheets!")
            except Exception:
                st.warning("Could not connect to Google Sheets. Please check back later!")

if st.session_state["active_tab"] == "Admin Control":
    section_header("Admin Control", OCEAN_BLUE)
    st.caption("Manual metadata overrides for display name and experiment details.")
    all_files = sorted([p for p in DATA_ROOT.rglob("*") if p.is_file() and p.suffix.lower() in (".csv", ".xlsx")])
    if not all_files:
        st.info("No files found in Data/.")
    else:
        labels = [str(p.relative_to(DATA_ROOT)) for p in all_files]
        pick_rel = st.selectbox("Pick file", options=labels, key="admin_pick_file")
        pick_path = DATA_ROOT / pick_rel
        key = pick_path.name
        overrides = load_metadata_overrides()
        current = overrides.get(key, {})
        disp_default = current.get("display_name", "")
        details_default = current.get("details", "")
        new_display = st.text_input("Display Name Override", value=disp_default, key="admin_display_name")
        new_details = st.text_area(
            "Experiment Details",
            value=details_default,
            key="admin_details",
            placeholder="e.g., 50nM Purified cBAF + EN119",
            height=120,
        )
        if st.button("💾 Save Changes", key="admin_save_overrides"):
            payload = {
                "display_name": new_display.strip(),
                "details": new_details.strip(),
            }
            if payload["display_name"] or payload["details"]:
                overrides[key] = payload
            elif key in overrides:
                overrides.pop(key, None)
            save_metadata_overrides(overrides)
            index_library.clear()
            get_path_metadata.clear()
            load_experiment_summary.clear()
            st.success(f"Saved overrides for {key} (session and local {OVERRIDES_PATH.name}).")
            st.info(
                "Copy the code below and paste it into your metadata_overrides.json file on GitHub "
                "to make these changes permanent for everyone who pulls the repo."
            )
            st.code(json.dumps(overrides, indent=2, sort_keys=True), language="json")

with st.sidebar:
    with st.expander("QC Summary", expanded=True):
        qc_path = st.session_state.get("qc_path")
        if qc_path and Path(qc_path).exists():
            qc = dp.compute_qc_metrics(Path(qc_path))
            st.caption(Path(qc_path).name)
            if qc.get("lda_column"):
                std = qc.get("lda_std")
                std_txt = f"{std:.4f}" if std is not None else "—"
                st.caption(f"LDA: `{qc['lda_column']}` · n={qc['n_lda_values']} · σ={std_txt}")
            if qc.get("flatline_significance"):
                st.warning(dp.FLATLINE_SIGNIFICANCE_MSG)
            for msg in qc.get("warnings", []):
                if msg == dp.FLATLINE_SIGNIFICANCE_MSG:
                    continue
                st.caption(msg)
        else:
            st.caption("Select an experiment in **Dataset Browser**.")
