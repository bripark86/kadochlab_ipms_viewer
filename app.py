import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data_processing as dp

st.set_page_config(page_title="IPMS Viewer", layout="wide")

DATA_ROOT = Path("Data")
BAF_RED = "#FF4B4B"
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
def crawl_library() -> pd.DataFrame:
    rows = []
    for p in DATA_ROOT.rglob("*.csv"):
        if "_PROCESSED" in p.name.upper():
            continue
        meta = dp.extract_metadata(p)
        rows.append(
            {
                "path": str(p),
                "file_name": p.name,
                "investigator": meta["investigator"],
                "session_id": meta["session_id"],
                "initials": meta["initials"],
                "target": meta["target"],
                "cell_line": meta["cell_line"],
                "sample_label": meta["sample_label"],
                "full_filename": meta["full_filename"],
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_experiment_summary(path: str) -> pd.DataFrame:
    return dp.summarize_experiment(Path(path))


def load_and_process_file(path: str) -> pd.DataFrame:
    df = load_experiment_summary(path).copy()
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


def volcano_plot(df: pd.DataFrame, title: str, stats_valid: bool = True) -> go.Figure:
    plot_df = df.copy()
    plot_df["is_baf"] = plot_df["Gene Symbol"].apply(is_baf_gene)
    plot_df["x"] = plot_df["Spectral Count"].apply(lambda v: math.log10(float(v) + 1))
    if not stats_valid:
        plot_df["y"] = 0.0
    elif "y_axis_val" in plot_df.columns:
        plot_df["y"] = plot_df["y_axis_val"]
    else:
        lda = pd.to_numeric(plot_df["LDA Probability"], errors="coerce")
        valid_lda = (lda > 0) & (lda <= 1)
        y_axis = np.full(len(plot_df), np.nan, dtype=float)
        if valid_lda.any():
            y_axis[valid_lda.to_numpy()] = -np.log10(lda[valid_lda].to_numpy())
        plot_df["y"] = y_axis
    plot_df["high_conf"] = plot_df["LDA Probability"] >= 0.8
    if stats_valid:
        plot_df = plot_df.dropna(subset=["y"])

    fig = go.Figure()
    other = plot_df[(~plot_df["is_baf"]) & (~plot_df["high_conf"])]
    high = plot_df[(~plot_df["is_baf"]) & (plot_df["high_conf"])]
    baf = plot_df[plot_df["is_baf"]]
    fig.add_trace(
        go.Scatter(
            x=other["x"], y=other["y"], mode="markers", name="Interactors", marker={"size": 7, "color": OCEAN_BLUE, "opacity": 0.55}, text=other["Gene Symbol"]
        )
    )
    fig.add_trace(
        go.Scatter(
            x=high["x"], y=high["y"], mode="markers", name="High-confidence", marker={"size": 8, "color": EMERALD, "opacity": 0.9}, text=high["Gene Symbol"]
        )
    )
    fig.add_trace(
        go.Scatter(
            x=baf["x"], y=baf["y"], mode="markers", name="BAF subunits", marker={"size": 12, "color": BAF_RED, "symbol": "diamond"}, text=baf["Gene Symbol"]
        )
    )
    if not stats_valid:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"opacity": 0},
                name="Y-axis values are placeholders due to missing stats.",
                showlegend=True,
            )
        )
    fig.update_layout(title=title, xaxis_title="Log10(Spectral Count + 1)", yaxis_title="-Log10(LDA Probability)", plot_bgcolor=BG_WHITE, paper_bgcolor=BG_WHITE)
    if not stats_valid:
        fig.update_yaxes(range=[0, 1])
    return fig


inject_theme()
st.title("IPMS Viewer")
meta_df = crawl_library()
if meta_df.empty:
    st.error("No CSV files found in `Data/`.")
    st.stop()

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

tab_options = ["Dataset Browser", "Discovery Hub", "Comparative Analysis", "Data Management"]
if st.session_state["active_tab"] not in tab_options:
    st.session_state["active_tab"] = "Dataset Browser"

with st.sidebar:
    st.session_state["active_tab"] = st.radio(
        "Navigation",
        options=tab_options,
        index=tab_options.index(st.session_state["active_tab"]),
    )

if st.session_state["active_tab"] == "Dataset Browser":
    section_header("Dataset Browser (Control Center)", BAF_RED)
    investigators = sorted(meta_df["investigator"].unique().tolist())
    default_inv = investigators[0]
    preferred_file = st.session_state.get("selected_file") or st.session_state.get("quick_open_file")
    if preferred_file in meta_df["file_name"].tolist():
        default_inv = meta_df.loc[meta_df["file_name"] == preferred_file, "investigator"].iloc[0]
    selected_inv = st.selectbox("Investigator", options=investigators, index=investigators.index(default_inv))
    inv_df = meta_df[meta_df["investigator"] == selected_inv].copy()
    table_df = inv_df[["session_id", "initials", "target", "cell_line", "sample_label", "full_filename"]].rename(
        columns={
            "session_id": "Session ID",
            "initials": "Initials",
            "target": "Target",
            "cell_line": "Cell Line",
            "sample_label": "Sample Label",
            "full_filename": "Full Filename",
        }
    )
    if not table_df.empty:
        try:
            st.dataframe(table_df.style.map(highlight_target_cell, subset=["Target"]), use_container_width=True)
        except Exception:
            st.dataframe(table_df, use_container_width=True)
    else:
        st.info("No experiments available for this investigator.")
    options = inv_df["file_name"].tolist()
    default_idx = 0
    if preferred_file in options:
        default_idx = options.index(preferred_file)
    selected_file = st.selectbox("Select experiment", options=options, index=default_idx)
    st.session_state["selected_file"] = selected_file
    selected_row = inv_df[inv_df["file_name"] == selected_file].iloc[0]
    st.session_state["qc_path"] = selected_row["path"]
    exp = load_and_process_file(selected_row["path"])
    exp["is_baf"] = exp["Gene Symbol"].apply(is_baf_gene)

    total_proteins = int(exp["Gene Symbol"].nunique())
    baf_ranked = exp[exp["is_baf"]].sort_values("Spectral Count", ascending=False)
    baf_count = int(baf_ranked["Gene Symbol"].nunique())
    bait_label = str(selected_row["target"])
    if bait_label in ("Unknown", "Control (IgG/Mock)"):
        top_inter = exp.sort_values("Spectral Count", ascending=False)["Gene Symbol"].head(1)
    else:
        top_inter = exp[exp["Gene Symbol"] != bait_label.upper()].sort_values("Spectral Count", ascending=False)["Gene Symbol"].head(1)
    top_interactor = top_inter.iloc[0] if not top_inter.empty else "N/A"
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Proteins", f"{total_proteins:,}")
    m2.metric("BAF Subunits (Ranked)", f"{baf_count}/26")
    m3.metric("Top Interactor", top_interactor)

    st.plotly_chart(volcano_plot(exp, "Experiment Scatter", stats_valid=st.session_state["stats_valid"]), use_container_width=True)

    coverage = exp.copy()
    coverage["canonical"] = coverage["Gene Symbol"].apply(lambda g: dp.identify_baf_target(g) or "")
    coverage = coverage[coverage["canonical"] != ""].groupby("canonical", as_index=False)["Spectral Count"].sum()
    missing = [x for x in dp.BAF_CORE_CANONICAL if x not in coverage["canonical"].tolist()]
    if missing:
        coverage = pd.concat([coverage, pd.DataFrame({"canonical": missing, "Spectral Count": [0] * len(missing)})], ignore_index=True)
    coverage = coverage.sort_values("Spectral Count", ascending=False)
    cov_fig = px.bar(coverage, x="Spectral Count", y="canonical", orientation="h", title="Complex Coverage (BAF Core)", color_discrete_sequence=[BAF_RED])
    cov_fig.update_yaxes(autorange="reversed")
    cov_fig.update_layout(plot_bgcolor=BG_WHITE, paper_bgcolor=BG_WHITE)
    st.plotly_chart(cov_fig, use_container_width=True)
    with st.expander("📂 View Raw Experiment Data", expanded=False):
        raw_df = exp.sort_values("Spectral Count", ascending=False).reset_index(drop=True)
        st.dataframe(raw_df, use_container_width=True)

if st.session_state["active_tab"] == "Discovery Hub":
    section_header("Discovery Hub", OCEAN_BLUE)
    _, center, _ = st.columns([1, 2, 1])
    with center:
        gene_query = st.text_input("Gene of Interest", value="SS18").strip().upper()
    compare_core = st.toggle("Compare with Core BAF", value=False)

    if gene_query:
        bait_for_consensus = dp.resolve_search_as_bait(gene_query)

        if bait_for_consensus is not None:
            bait_runs = meta_df[meta_df["target"] == bait_for_consensus].copy()
            st.markdown("### Primary target — enrichment profile")
            if bait_for_consensus == "CEBPE":
                st.caption(f"Indexed experiments where **CEBPE** is the IP bait (primary target outside the 26 BAF core).")
            else:
                st.caption(f"Indexed experiments where **{bait_for_consensus}** is the IP bait.")
            if not bait_runs.empty:
                enrich_cols = bait_runs[
                    ["investigator", "session_id", "cell_line", "sample_label", "file_name", "full_filename"]
                ].rename(
                    columns={
                        "investigator": "Investigator",
                        "session_id": "Exp ID",
                        "cell_line": "Cell Line",
                        "sample_label": "Sample Label",
                        "file_name": "File Name",
                        "full_filename": "Full Filename",
                    }
                )
                st.dataframe(enrich_cols.sort_values("Exp ID"), use_container_width=True)
            else:
                st.info(f"No runs indexed with bait target **{bait_for_consensus}**.")

            st.markdown("### Primary target — consensus interactors")
            st.caption(f"Mean spectral counts across {len(bait_runs)} run(s) with bait **{bait_for_consensus}**.")
            if not bait_runs.empty:
                all_rows = []
                for _, row in bait_runs.iterrows():
                    e = load_experiment_summary(row["path"]).copy()
                    e["exp_id"] = row["session_id"]
                    all_rows.append(e)
                merged_c = pd.concat(all_rows, ignore_index=True)
                prevalence = merged_c.groupby("Gene Symbol")["exp_id"].nunique()
                thr = max(1, math.ceil(len(bait_runs) * 0.5))
                keep = prevalence[prevalence >= thr].index
                consensus = (
                    merged_c[merged_c["Gene Symbol"].isin(keep)]
                    .groupby("Gene Symbol", as_index=False)
                    .agg(Mean_Spectral_Count=("Spectral Count", "mean"), Mean_Unique_Peptides=("Unique Peptides", "mean"), Runs_Present=("exp_id", "nunique"))
                    .sort_values("Mean_Spectral_Count", ascending=False)
                )
                consensus["Prevalence Score (%)"] = (consensus["Runs_Present"] / len(bait_runs) * 100).round(1)
                st.dataframe(consensus, use_container_width=True)
            else:
                st.info("No consensus table (no bait-matched runs).")

        st.markdown("### Global Results")
        rows = []
        prog = st.progress(0)
        files = meta_df.to_dict(orient="records")
        for i, row in enumerate(files, start=1):
            e = load_experiment_summary(row["path"])
            hit = e[e["Gene Symbol"] == gene_query]
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
                        "File Name": row["file_name"],
                    }
                )
            prog.progress(i / len(files))
        prog.empty()

        if rows:
            res = pd.DataFrame(rows).sort_values("Spectral Count", ascending=False)
            if compare_core:
                try:
                    st.dataframe(
                        res.style.map(lambda v: f"background-color: {SOFT_BLUE}; color: {TEXT_DARK};" if v else "", subset=["Core BAF IP"]),
                        use_container_width=True,
                    )
                except Exception:
                    st.dataframe(res, use_container_width=True)
            else:
                st.dataframe(res[["Investigator", "Target", "Cell Line", "Exp ID", "Spectral Count", "Unique Peptides"]], use_container_width=True)

            qf = st.selectbox("Quick Open experiment", options=res["File Name"].tolist())
            if st.button("Open in Dataset Browser"):
                st.session_state["selected_file"] = qf
                st.session_state["quick_open_file"] = qf
                st.session_state["active_tab"] = "Dataset Browser"
                st.rerun()

            inv_dist = res.groupby("Investigator", as_index=False)["Spectral Count"].count().rename(columns={"Spectral Count": "Hit Count"}).sort_values("Hit Count", ascending=False)
            cell_enrich = res.groupby("Cell Line", as_index=False)["Spectral Count"].sum().sort_values("Spectral Count", ascending=False)
            g1, g2 = st.columns(2)
            with g1:
                fig1 = px.bar(inv_dist, x="Investigator", y="Hit Count", title="Enrichment by Investigator", color_discrete_sequence=[OCEAN_BLUE])
                fig1.update_layout(plot_bgcolor=BG_WHITE, paper_bgcolor=BG_WHITE)
                st.plotly_chart(fig1, use_container_width=True)
            with g2:
                fig2 = px.bar(cell_enrich, x="Cell Line", y="Spectral Count", title="Enrichment by Cell Line", color_discrete_sequence=[EMERALD])
                fig2.update_layout(plot_bgcolor=BG_WHITE, paper_bgcolor=BG_WHITE)
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No hits found.")

if st.session_state["active_tab"] == "Comparative Analysis":
    section_header("Comparative Analysis", EMERALD)
    picks = st.multiselect("Choose 2 to 4 experiments", options=meta_df["file_name"].tolist(), max_selections=4)
    if len(picks) == 2:
        a_file, b_file = picks[0], picks[1]
        a_row = meta_df[meta_df["file_name"] == a_file].iloc[0]
        b_row = meta_df[meta_df["file_name"] == b_file].iloc[0]
        a_exp = load_and_process_file(a_row["path"])
        a_stats_valid = st.session_state.get("stats_valid", True)
        b_exp = load_and_process_file(b_row["path"])
        b_stats_valid = st.session_state.get("stats_valid", True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(volcano_plot(a_exp, f"Scatter: {a_file}", stats_valid=a_stats_valid), use_container_width=True)
        with c2:
            st.plotly_chart(volcano_plot(b_exp, f"Scatter: {b_file}", stats_valid=b_stats_valid), use_container_width=True)

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
        subset = meta_df[meta_df["file_name"].isin(picks)].copy()
        merged = None
        for _, row in subset.iterrows():
            e = load_experiment_summary(row["path"])[["Gene Symbol", "Spectral Count"]].rename(columns={"Spectral Count": row["file_name"]})
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
        uploads = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
        if st.button("Save Uploads") and uploads:
            folder = DATA_ROOT / upload_target
            folder.mkdir(parents=True, exist_ok=True)
            for f in uploads:
                (folder / f.name).write_bytes(f.getvalue())
            crawl_library.clear()
            load_experiment_summary.clear()
            st.success(f"Saved {len(uploads)} file(s).")
    with c2:
        if st.button("Refresh Metadata"):
            crawl_library.clear()
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
            crawl_library.clear()
            load_experiment_summary.clear()
            st.warning(f"Deleted: {delete_file}")

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
