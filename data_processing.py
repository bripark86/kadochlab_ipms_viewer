import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BAF_ALIASES: Dict[str, List[str]] = {
    "SMARCA4": ["SMARCA4", "BRG1", "BRG-1"],
    "SMARCA2": ["SMARCA2", "BRM"],
    "SMARCC1": ["SMARCC1", "BAF155", "BAF-155"],
    "SMARCC2": ["SMARCC2", "BAF170", "BAF-170"],
    "SMARCB1": ["SMARCB1", "BAF47", "INI1", "SNF5"],
    "ARID1A": ["ARID1A", "BAF250A"],
    "ARID1B": ["ARID1B", "BAF250B"],
    "PBRM1": ["PBRM1", "BAF180", "PB1"],
    "ARID2": ["ARID2", "BAF200"],
    "BRD7": ["BRD7"],
    "BRD9": ["BRD9"],
    "BICRA": ["BICRA", "GLTSCR1", "C7ORF26"],
    "BICRAL": ["BICRAL", "GLTSCR1L", "C19ORF26"],
    "ACTL6A": ["ACTL6A", "BAF53A"],
    "ACTL6B": ["ACTL6B", "BAF53B"],
    "SMARCE1": ["SMARCE1", "BAF57"],
    "SMARCD1": ["SMARCD1", "BAF60A"],
    "SMARCD2": ["SMARCD2", "BAF60B"],
    "SMARCD3": ["SMARCD3", "BAF60C"],
    "DPF2": ["DPF2", "BAF45D"],
    "DPF1": ["DPF1", "BAF45B"],
    "DPF3": ["DPF3", "BAF45C"],
    "PHF10": ["PHF10", "BAF45A"],
    "SS18": ["SS18"],
    "SS18L1": ["SS18L1", "CREST"],
    "BCL7A": ["BCL7A"],
}

KNOWN_CELL_LINES = [
    "K562",
    "HEK293T",
    "293T",
    "SW13",
    "OCILY1",
    "OCI-LY1",
    "SCCOHT-1",
    "BIN67",
    "HAP1",
    "U2OS",
    "HELA",
    "MCF7",
    "RPE1",
    "RPE-1",
    "SHI1",
    "MOLM13",
    "AN3CA",
    "SUDHL1",
    "EOLI",
    "H23",
    "A549",
    "H1299",
]

# 26 core BAF genes (canonical keys)
BAF_CORE_CANONICAL = tuple(BAF_ALIASES.keys())

# Recognized non-BAF primary baits (filename / metadata target string)
PRIMARY_TARGET_EXTRA = frozenset({"CEBPE"})

_CEBPE_IN_STEM = re.compile(r"(?:^|[^A-Za-z0-9])CEBPE(?:$|[^A-Za-z0-9])", re.IGNORECASE)

# \b fails for _IgG because '_' is a "word" char in Python. Use delimiter-aware match on full stem.
_CONTROL_IN_FILENAME = re.compile(
    r"(?:^|[^A-Za-z0-9])(?P<ctrl>igg|mock|control|ev)(?:$|[^A-Za-z0-9])",
    re.IGNORECASE,
)


def norm_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


ALIAS_LOOKUP: Dict[str, str] = {}
for _canonical, _aliases in BAF_ALIASES.items():
    for _alias in _aliases:
        ALIAS_LOOKUP[norm_token(_alias)] = _canonical


def identify_baf_target(text: str) -> Optional[str]:
    tokens = re.split(r"[_\-\s]+", text.upper())
    for token in tokens + [norm_token(text)]:
        token_norm = norm_token(token)
        if token_norm in ALIAS_LOOKUP:
            return ALIAS_LOOKUP[token_norm]
    return None


def get_biological_target(rem_text: str, stem: str) -> Optional[str]:
    """
    Resolve biological bait from filename remainder and full stem.
    BAF subunits take precedence; then CEBPE; caller applies control / Unknown after.
    """
    detected_baf = identify_baf_target(rem_text)
    if detected_baf:
        return detected_baf
    if _CEBPE_IN_STEM.search(stem) or _CEBPE_IN_STEM.search(rem_text):
        return "CEBPE"
    parts = [p.upper() for p in re.split(r"[^A-Za-z0-9]+", stem) if p]
    if "CEBPE" in parts:
        return "CEBPE"
    return None


def filename_indicates_control(stem: str) -> bool:
    if _CONTROL_IN_FILENAME.search(stem):
        return True
    parts = [p.upper() for p in re.split(r"[^A-Za-z0-9]+", stem) if p]
    return any(p in ("IGG", "MOCK", "CONTROL", "EV") for p in parts)


def extract_cell_line(rem_text: str) -> str:
    if re.search(r"\bKT\d+\b", rem_text.upper()):
        return "KT"
    if re.search(r"\bKOB\d+\b", rem_text.upper()):
        return "KOB"
    for cell in KNOWN_CELL_LINES:
        if norm_token(cell) in norm_token(rem_text):
            return cell
    return "Unknown"


def extract_metadata(csv_path: Path) -> Dict[str, str]:
    stem = csv_path.stem
    parts = stem.split("_")
    session_id = "Unknown"
    initials = "Unknown"
    target = "Unknown"
    sample_label = "Unknown"
    investigator = csv_path.parent.name

    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        session_id = f"{parts[0]}_{parts[1]}"
    if len(parts) >= 3:
        initials = parts[2]
    remainder = parts[3:] if len(parts) > 3 else []
    rem_text = "_".join(remainder)

    bio = get_biological_target(rem_text, stem)
    if bio:
        target = bio
    elif filename_indicates_control(stem):
        target = "Control (IgG/Mock)"

    cell_line = extract_cell_line(rem_text)
    leftovers = []
    for token in remainder:
        if identify_baf_target(token):
            continue
        if token.upper() == "CEBPE":
            continue
        if token.upper() in {"IGG", "MOCK", "CONTROL", "EV"}:
            continue
        if re.fullmatch(r"(?i)(KT|KOB)\d+", token):
            continue
        if any(norm_token(token) == norm_token(c) for c in KNOWN_CELL_LINES):
            continue
        leftovers.append(token)
    if leftovers:
        sample_label = "_".join(leftovers)

    return {
        "investigator": investigator,
        "session_id": session_id,
        "initials": initials,
        "target": target,
        "cell_line": cell_line,
        "sample_label": sample_label,
        "full_filename": csv_path.name,
    }


def infer_tmt_investigator(path: Path) -> str:
    """
    Deprecated for manifest rows: investigator must be ``path.parent.name`` (physical Data subfolder).
    Kept for backward compatibility; returns parent folder name when possible.
    """
    inv = path.parent.name
    return inv if inv else "JSL"


def select_tmt_sheet(xl: pd.ExcelFile) -> str:
    for name in xl.sheet_names:
        if "protein_quant_" in name.lower():
            return name
    return xl.sheet_names[0]


def _find_tmt_header_row(preview: pd.DataFrame) -> int:
    """
    Prefer a row containing both Protein Id and Gene Symbol; else first row with Protein Id
    (Jessica_StL two-row header: short codes on row above).
    """
    max_r = min(5, len(preview))
    best_pid_only: Optional[int] = None
    for i in range(max_r):
        cells = [str(preview.iat[i, j]).strip().lower() for j in range(preview.shape[1]) if pd.notna(preview.iat[i, j])]
        has_pid = any("protein id" in c for c in cells)
        has_gene = any("gene symbol" in c for c in cells)
        if has_pid and has_gene:
            return i
        if has_pid and best_pid_only is None:
            best_pid_only = i
    return best_pid_only if best_pid_only is not None else 0


_SN_SUM_TAIL = re.compile(r"[_\s]sn[_\s]sum\s*$", re.IGNORECASE)


def is_tmt_sn_sum_column(name: str) -> bool:
    """Strict TMT intensity: ends with sn sum / sn_sum; excludes scaled columns."""
    s = str(name).strip()
    low = s.lower().replace(" ", "_")
    if "scaled" in low:
        return False
    return bool(_SN_SUM_TAIL.search(s.replace(" ", "_")))


def assemble_tmt_column_names(full: pd.DataFrame, hrow: int) -> List[str]:
    """Merge metadata rows above channel header for sn-sum columns (supports 2-row and 3-row styles)."""
    n = int(full.shape[1])
    header_vals = [full.iat[hrow, j] if j < full.shape[1] else "" for j in range(n)]
    row_minus_1 = [full.iat[hrow - 1, j] if hrow > 0 and j < full.shape[1] else "" for j in range(n)]
    row_minus_2 = [full.iat[hrow - 2, j] if hrow > 1 and j < full.shape[1] else "" for j in range(n)]

    def _clean_header_cell(v: Any) -> str:
        if pd.isna(v):
            return ""
        s = str(v).strip()
        if s.lower() in ("nan", "none"):
            return ""
        return s

    def _is_numericish(v: str) -> bool:
        t = (v or "").strip()
        if not t:
            return False
        try:
            float(t)
            return True
        except ValueError:
            return False

    new_cols: List[str] = []
    for j in range(n):
        base = str(header_vals[j]).strip() if pd.notna(header_vals[j]) else ""
        if base and is_tmt_sn_sum_column(base):
            # Kevin-style: bio labels are often two rows above Protein Id row, while row above is numeric.
            cand_top = _clean_header_cell(row_minus_2[j])
            cand_near = _clean_header_cell(row_minus_1[j])
            if cand_top and not _is_numericish(cand_top):
                code = cand_top
            elif cand_near and not _is_numericish(cand_near):
                code = cand_near
            elif cand_top:
                code = cand_top
            else:
                code = cand_near
            if code:
                new_cols.append(f"{code} | {base}")
            else:
                new_cols.append(base)
        else:
            new_cols.append(base if base else f"Unnamed_{j}")
    return new_cols


def tmt_short_label_from_column(col: str) -> str:
    """Biological / short-code label from Row 1 (before ' | ')."""
    s = str(col).strip()
    if " | " in s:
        return s.split(" | ", 1)[0].strip()
    return ""


def tmt_biological_condition_from_channel_label(channel_label: str) -> str:
    """
    Row-1 biological label for manifest ``Biological_Condition`` / table column.
    Kevin: e.g. ``GPF_1``; Jessica: e.g. ``VOA_ARID1A+_BRG1_1`` (prefix before isobar / channel id).
    """
    bio = tmt_short_label_from_column(channel_label)
    return bio if bio else "N/A"


def read_tmt_excel_wide(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path, engine="openpyxl")
    sheet = select_tmt_sheet(xl)
    full = pd.read_excel(path, sheet_name=sheet, header=None, engine="openpyxl")
    if full.empty:
        return pd.DataFrame()
    preview = full.iloc[: min(5, len(full)), :]
    hrow = _find_tmt_header_row(preview)
    new_cols = assemble_tmt_column_names(full, hrow)
    df = full.iloc[hrow + 1 :].copy()
    n = min(len(new_cols), df.shape[1])
    df = df.iloc[:, :n].copy()
    df.columns = new_cols[:n]

    pid_col = resolve_column(df, ["Protein Id", "Protein ID", "Protein"])
    if not pid_col:
        raise ValueError("TMT sheet: Protein Id column not found after header detection")
    df = df.dropna(subset=[pid_col])

    gene_col = resolve_column(df, ["Gene Symbol", "Gene", "Symbol"])
    desc_col = resolve_column(df, ["Description", "Protein Description"])
    pep_col = resolve_column(df, ["No. of peptides", "No Of Peptides", "Number of peptides", "Peptides"])
    id_keep = [c for c in [pid_col, gene_col, desc_col, pep_col] if c and c in df.columns]
    sig_cols = [c for c in df.columns if is_tmt_sn_sum_column(c)]
    if not sig_cols:
        raise ValueError("TMT sheet: no _sn_sum intensity columns found (scaled columns are ignored)")
    if not gene_col or gene_col not in df.columns:
        raise ValueError("TMT sheet: Gene Symbol column not found")
    df = df[[c for c in id_keep if c in df.columns] + [c for c in sig_cols if c not in id_keep]].copy()

    num_sig = df[sig_cols].apply(pd.to_numeric, errors="coerce")
    arr = num_sig.replace(0, np.nan).to_numpy(dtype=float).ravel()
    pos = arr[(arr > 0) & np.isfinite(arr)]
    repl = float(np.nanmin(pos)) if pos.size else 1.0
    for c in sig_cols:
        colv = pd.to_numeric(df[c], errors="coerce")
        colv = colv.where(colv != 0, repl)
        colv = colv.fillna(repl)
        df[c] = colv
    return df


def list_tmt_sn_sum_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if is_tmt_sn_sum_column(c)]


def _normalize_tmt_iso_fragment(p: str) -> str:
    if not p:
        return p
    if len(p) >= 2 and p[-1].lower() in "nc" and p[-2].isdigit():
        return p[:-1] + p[-1].upper()
    return p.upper() if p.isascii() else p


# Wells like 1A / 12B (not TMT isobar tags such as 127N)
_PLATE_CHANNEL_RE = re.compile(r"^[0-9]{1,2}[A-Za-z]$")


def _tmt_core_between_first_last_underscore(s: str) -> Optional[str]:
    """Substring strictly between the first and last ``_`` in *s* (None if not applicable)."""
    s = (s or "").strip()
    first = s.find("_")
    last = s.rfind("_")
    if first == -1 or last == -1 or first >= last:
        return None
    mid = s[first + 1 : last].strip()
    return mid if mid else None


def _tmt_format_plus_bait_label(core: str) -> str:
    """e.g. ``ARID1A+_BRG1`` → ``ARID1A + BRG1``."""
    t = (core or "").strip()
    t = t.replace("+_", " + ").replace("_+", " + ")
    if "+" in t and " + " not in t:
        t = re.sub(r"\+", " + ", t, count=1)
    t = re.sub(r"\s+", " ", t.replace("_", " ")).strip()
    return t


def _tmt_target_from_bio(bio: str) -> str:
    """
    Fine-grained target from Row-1 label (before ``|``).

    - IgG → IgG Control.
    - ARID1A + BRG1 style (``ARID1A+_BRG1`` in label) → ``ARID1A + BRG1``.
    - BRG1-only multiplex (e.g. ``VOA_BRG1_2``) → ``BRG1`` (distinct from dual-bait rows).
    - Else: core between first/last underscores, or humanized full label.
    """
    bio = (bio or "").strip()
    if not bio:
        return "Unknown"
    bio_u = bio.upper()
    mid = _tmt_core_between_first_last_underscore(bio)

    if "IGG" in bio_u:
        return "IgG Control"
    if "ARID1A" in bio_u and "BRG1" in bio_u:
        core = mid if mid else bio
        return _tmt_format_plus_bait_label(core)
    if "BRG1" in bio_u:
        return "BRG1"
    if _PLATE_CHANNEL_RE.fullmatch(bio):
        return f"TMT Reference {bio}"
    if bio_u.startswith("VOA_"):
        rest = bio[4:].strip() if len(bio) >= 4 else ""
        if not rest:
            return "Unknown"
        inner = _tmt_core_between_first_last_underscore(rest)
        if inner:
            return _tmt_format_plus_bait_label(inner) if ("+" in inner or "_" in inner) else inner.replace("_", " ").strip()
        return rest.replace("_", " ").strip()
    if mid:
        return _tmt_format_plus_bait_label(mid) if "+" in mid else mid.replace("_", " ").strip()
    return bio.replace("_", " ").strip()


def parse_tmt_virtual_channel_metadata(channel_label: str) -> Dict[str, str]:
    """
    Derive Target and Cell Line from the virtual channel label only (not the .xlsx filename).

    Uses the substring before ' | ' (Row-1 short code + TMT column), e.g. ``VOA_BRG1_2 | 128C`` → bio ``VOA_BRG1_2``.

    Rules:
    - Cell line: ``VOA_`` prefix → VOA; plate-style ``1A`` / ``1B`` → JSL_Ref; else Unknown.
    - Target: see ``_tmt_target_from_bio`` (IgG Control, ``ARID1A + BRG1``, BRG1-only, plate wells, VOA remainder).
    """
    ch = (channel_label or "").strip()
    if not ch:
        return {"cell_line": "Unknown", "target": "Unknown", "sample_label": "Unknown"}
    bio = ch.split(" | ", 1)[0].strip() if " | " in ch else ch
    iso = ch.split(" | ", 1)[1].strip() if " | " in ch else ""
    iso_core = _normalize_tmt_iso_fragment(iso) if iso else ""
    bio_u = bio.upper()

    target = _tmt_target_from_bio(bio)
    if (
        target not in {"Unknown", "IgG Control", "BRG1"}
        and bio
        and iso_core
        and (not bio_u.startswith("VOA_"))
        and (not _PLATE_CHANNEL_RE.fullmatch(bio))
    ):
        # Kevin-style labels (e.g., GPF_1 row + 126 channel row) keep both parts searchable.
        target = f"{bio} | {iso_core}"

    if bio_u.startswith("VOA_"):
        cell_line = "VOA"
    elif _PLATE_CHANNEL_RE.fullmatch(bio):
        cell_line = "JSL_Ref"
    else:
        cell_line = "Unknown"

    return {"cell_line": cell_line, "target": target, "sample_label": iso if iso else "Unknown"}


_TMT_STEM_DATE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}_", re.IGNORECASE)


def tmt_fallback_target_cellline_from_filename(xlsx_path: Path) -> Tuple[str, str]:
    """
    When Row-1 channel labels do not encode biology (no VOA_/plate style), derive Target from the
    workbook stem (e.g. ``2025-09-25_KSO_IP9`` → Target ``IP9``). Cell line is ``Unknown/TMT``.
    """
    stem = _TMT_STEM_DATE_PREFIX.sub("", xlsx_path.stem, count=1)
    parts = [p for p in stem.split("_") if str(p).strip()]
    if not parts:
        return "Unknown", "Unknown/TMT"
    return str(parts[-1]).strip(), "Unknown/TMT"


def merge_tmt_parsed_with_filename_fallback(parsed: Dict[str, str], xlsx_path: Path) -> Dict[str, str]:
    """
    When Row-1 labels do not encode a cell line (no VOA_/plate), use the workbook stem for Target
    (e.g. ``..._KSO_IP9`` → ``IP9``) and ``Unknown/TMT`` for Cell line. If only Target was unknown, fill it from stem.
    """
    out = {**parsed}
    ft, fc = tmt_fallback_target_cellline_from_filename(xlsx_path)
    if out.get("cell_line") == "Unknown":
        out["cell_line"] = fc
        out["target"] = ft
    elif out.get("target") == "Unknown":
        out["target"] = ft
    return out


def manifest_row_matches_bait(target_str: str, bait_canonical: str) -> bool:
    """True if manifest row target string denotes the same bait as bait_canonical (BAF aliases, CEBPE)."""
    if not bait_canonical:
        return False
    t = str(target_str).strip()
    if bait_canonical == "CEBPE":
        return "CEBPE" in t.upper()
    if t == bait_canonical:
        return True
    can = identify_baf_target(t)
    return can == bait_canonical


def sn_sum_col_to_channel_label(col: str) -> str:
    """Human-readable channel label; strips trailing sn_sum / 'sn sum' including combined '1A | 127n_sn_sum' forms."""
    s = str(col).strip()
    if " | " in s:
        left, right = s.split(" | ", 1)
        rnorm = _SN_SUM_TAIL.sub("", right.strip().replace(" ", "_")).strip("_")
        rnorm = _normalize_tmt_iso_fragment(rnorm) if rnorm else ""
        return f"{left.strip()} | {rnorm}".strip(" |") if rnorm else left.strip()
    rnorm = _SN_SUM_TAIL.sub("", s.replace(" ", "_")).strip("_")
    return _normalize_tmt_iso_fragment(rnorm) if rnorm else s


def build_tmt_ma_comparison_df(
    wide_df: pd.DataFrame,
    reference_columns: List[str],
    target_columns: List[str],
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    TMT comparison (MA-style): X = mean(log10 intensity) over all selected channels (ref ∪ target);
    Y = log2(mean(target) / mean(reference)).
    """
    empty_cols = ["Gene Symbol", "x_avg_log10", "y_log2_ratio", "fold_change", "hover_conditions", "ref_labels", "tgt_labels"]
    gene_col = resolve_column(wide_df, ["Gene Symbol", "Gene", "Symbol"])
    if not gene_col:
        return pd.DataFrame(columns=empty_cols)
    ref = [c for c in reference_columns if c in wide_df.columns]
    tgt = [c for c in target_columns if c in wide_df.columns]
    if not ref or not tgt:
        return pd.DataFrame(columns=empty_cols)

    all_c = list(dict.fromkeys(ref + tgt))
    genes = wide_df[gene_col].astype(str).str.strip().str.upper()
    mask = (genes != "") & (genes != "NAN")
    sub = wide_df.loc[mask].copy()
    gser = genes[mask]

    num = pd.DataFrame({c: pd.to_numeric(sub[c], errors="coerce") for c in all_c}, index=sub.index)
    log_block = np.log10(num.clip(lower=eps))
    x_avg = log_block.mean(axis=1)
    mean_r = num[ref].mean(axis=1)
    mean_t = num[tgt].mean(axis=1)
    ratio = mean_t / (mean_r + eps)
    y_m = np.log2(np.maximum(ratio, eps))

    ref_labs = ", ".join(sorted({tmt_short_label_from_column(c) for c in ref if tmt_short_label_from_column(c)}))
    tgt_labs = ", ".join(sorted({tmt_short_label_from_column(c) for c in tgt if tmt_short_label_from_column(c)}))
    hover_bio = f"Ref: {ref_labs or '—'} | Tgt: {tgt_labs or '—'}"

    out = pd.DataFrame(
        {
            "Gene Symbol": gser.values,
            "x_avg_log10": x_avg.values,
            "y_log2_ratio": y_m.values,
            "fold_change": ratio.values,
            "hover_conditions": hover_bio,
            "ref_labels": ref_labs,
            "tgt_labels": tgt_labs,
        }
    )
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Gene Symbol", "x_avg_log10", "y_log2_ratio"])


def summarize_tmt_channel(path: Path, sn_sum_col: str, wide_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    One TMT channel -> same schema as summarize_experiment (gene-level).
    Spectral Count = sn_sum intensity; synthetic LDA from peptide counts.
    """
    df = wide_df if wide_df is not None else read_tmt_excel_wide(path)
    if sn_sum_col not in df.columns:
        raise ValueError(f"TMT column not found: {sn_sum_col}")

    gene_col = resolve_column(df, ["Gene Symbol", "Gene", "Symbol"])
    if not gene_col:
        raise ValueError("TMT sheet: Gene Symbol column not found")

    pep_col = resolve_column(df, ["No. of peptides", "No Of Peptides", "Number of peptides", "Peptides"])
    desc_col = resolve_column(df, ["Description", "Protein Description"])

    use = df[[gene_col, sn_sum_col] + [c for c in [pep_col, desc_col] if c]].copy()
    use[gene_col] = use[gene_col].astype(str).str.strip().str.upper()
    use = use[(use[gene_col] != "") & (use[gene_col] != "NAN")]

    spec = pd.to_numeric(use[sn_sum_col], errors="coerce").fillna(0.0)
    use["_spec"] = spec

    if pep_col:
        npep = pd.to_numeric(use[pep_col], errors="coerce").fillna(0)
        use["_npep"] = npep
        use["_lda"] = np.where(use["_npep"] > 1, 0.99, 0.50)
        use["_upep"] = npep.astype(int)
    else:
        use["_lda"] = 0.50
        use["_upep"] = 0

    agg = (
        use.groupby(gene_col, dropna=True)
        .agg(Spectral_Count=("_spec", "sum"), Unique_Peptides=("_upep", "max"), LDA_Probability=("_lda", "mean"))
        .reset_index()
        .rename(columns={gene_col: "Gene Symbol"})
    )
    agg = agg.rename(columns={"Spectral_Count": "Spectral Count", "LDA_Probability": "LDA Probability", "Unique_Peptides": "Unique Peptides"})
    ch_label = sn_sum_col_to_channel_label(sn_sum_col)
    agg["TMT_Channel"] = ch_label
    return agg.sort_values("Spectral Count", ascending=False)


def build_tmt_manifest_rows(xlsx_path: Path) -> List[Dict[str, Any]]:
    """
    One .xlsx -> one manifest row per TMT channel (virtual experiments).
    """
    wide = read_tmt_excel_wide(xlsx_path)
    sn_cols = list_tmt_sn_sum_columns(wide)
    if not sn_cols:
        return []

    base_meta = extract_metadata(xlsx_path)
    inv_folder = xlsx_path.parent.name or "Unknown"
    inv = inv_folder.replace("_", " ")
    initials_display = inv
    rows_out: List[Dict[str, Any]] = []
    for sn_col in sn_cols:
        ch = sn_sum_col_to_channel_label(sn_col)
        biological_condition = tmt_biological_condition_from_channel_label(ch)
        parsed = merge_tmt_parsed_with_filename_fallback(parse_tmt_virtual_channel_metadata(ch), xlsx_path)
        cell_line = parsed["cell_line"]
        target = parsed["target"]
        sample_label = parsed.get("sample_label") or "Unknown"
        # Select experiment: Filename | Target | Cell | TMT channel (isobar / column id)
        display = f"{xlsx_path.name} | Target: {target} | Cell: {cell_line} | Channel: {ch}"
        rows_out.append(
            {
                "path": str(xlsx_path),
                "file_name": display,
                "investigator": inv,
                "session_id": base_meta["session_id"],
                "initials": initials_display,
                "target": target,
                "biological_condition": biological_condition,
                "cell_line": cell_line,
                "sample_label": sample_label,
                "full_filename": xlsx_path.name,
                "tmt_channel": ch,
                "tmt_sn_sum_column": sn_col,
                "experiment_type": "TMT Multiplex",
            }
        )
    return rows_out


def summarize_experiment_any(path: Path, tmt_sn_sum_column: Optional[str] = None) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        if not tmt_sn_sum_column:
            return pd.DataFrame(columns=["Gene Symbol", "Spectral Count", "Unique Peptides", "LDA Probability"])
        try:
            return summarize_tmt_channel(path, tmt_sn_sum_column)
        except Exception as exc:
            print(f"TMT summarize skipped for {path}: {exc}")
            return pd.DataFrame(columns=["Gene Symbol", "Spectral Count", "Unique Peptides", "LDA Probability"])
    return summarize_experiment(path)


def read_csv_flexible(csv_path: Path) -> pd.DataFrame:
    raw = csv_path.read_text(errors="ignore")
    lines = raw.splitlines()
    header_idx = 0
    for idx, line in enumerate(lines[:35]):
        if "Gene Symbol" in line and "," in line:
            header_idx = idx
            break
    cleaned = "\n".join(lines[header_idx:])
    return pd.read_csv(io.StringIO(cleaned), low_memory=False)


def resolve_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for low, original in lowered.items():
            if cand.lower() == low or cand.lower() in low:
                return original
    return None


FLATLINE_SIGNIFICANCE_MSG = "Flatline Significance Detected."
LDA_STD_OK_THRESHOLD = 0.1


def compute_qc_metrics(csv_path: Path) -> Dict[str, Any]:
    """
    Assess LDA (significance) spread in raw peptide-level rows.
    If standard deviation is not greater than 0.1 (or LDA is missing / insufficient data),
    flags flatline significance for QC display.
    """
    out: Dict[str, Any] = {
        "lda_column": None,
        "n_lda_values": 0,
        "lda_std": None,
        "flatline_significance": False,
        "warnings": [],
    }

    if csv_path.suffix.lower() == ".xlsx":
        out["warnings"].append("TMT multiplex: open a channel in Dataset Browser for per-channel QC-style stats.")
        return out

    try:
        df = read_csv_flexible(csv_path)
    except Exception:
        out["warnings"].append("Could not read file for QC.")
        out["flatline_significance"] = True
        return out

    lda_col = resolve_column(df, ["LDA Probability", "LDA", "DeltaCorr", "Corr"])
    if not lda_col:
        out["warnings"].append("No LDA column found.")
        return out

    out["lda_column"] = lda_col
    series = pd.to_numeric(df[lda_col], errors="coerce").dropna()
    out["n_lda_values"] = int(series.shape[0])

    if series.shape[0] < 2:
        out["lda_std"] = None
        out["flatline_significance"] = True
        out["warnings"].append(FLATLINE_SIGNIFICANCE_MSG)
        return out

    std = float(series.std(ddof=1))
    out["lda_std"] = std
    if std <= LDA_STD_OK_THRESHOLD:
        out["flatline_significance"] = True
        out["warnings"].append(FLATLINE_SIGNIFICANCE_MSG)
    return out


def summarize_experiment(csv_path: Path) -> pd.DataFrame:
    df = read_csv_flexible(csv_path)
    gene_col = resolve_column(df, ["Gene Symbol", "Gene", "Symbol"])
    peptide_col = resolve_column(df, ["Peptide", "Peptide Sequence"])
    lda_col = resolve_column(df, ["LDA Probability", "DeltaCorr", "Corr"])
    spectral_col = resolve_column(df, ["Spectral Count", "Count", "Max"])

    columns = ["Gene Symbol", "Spectral Count", "Unique Peptides", "LDA Probability"]
    if gene_col is None:
        return pd.DataFrame(columns=columns)

    use_cols = [gene_col] + [c for c in [peptide_col, lda_col, spectral_col] if c]
    work = df[use_cols].copy()
    work[gene_col] = work[gene_col].astype(str).str.strip().str.upper()
    work = work[(work[gene_col] != "") & (work[gene_col] != "NAN")]

    out = work.groupby(gene_col).size().rename("Spectral Count").to_frame()
    if spectral_col:
        spec = pd.to_numeric(work[spectral_col], errors="coerce").fillna(0)
        out["Spectral Count"] = spec.groupby(work[gene_col]).sum()
    if peptide_col:
        out["Unique Peptides"] = work.groupby(gene_col)[peptide_col].nunique().reindex(out.index).fillna(0).astype(int)
    else:
        out["Unique Peptides"] = 0
    if lda_col:
        lda = pd.to_numeric(work[lda_col], errors="coerce").fillna(0.05)
        out["LDA Probability"] = lda.groupby(work[gene_col]).mean().reindex(out.index).fillna(0.05)
    else:
        out["LDA Probability"] = 0.05

    out = out.reset_index().rename(columns={gene_col: "Gene Symbol"})
    return out.sort_values("Spectral Count", ascending=False)


def is_core_baf_canonical(name: Optional[str]) -> bool:
    return name is not None and name in BAF_CORE_CANONICAL


def is_primary_target_bait(name: Optional[str]) -> bool:
    """True if this metadata target is a BAF core bait or an extra primary (e.g. CEBPE)."""
    if name is None:
        return False
    if name in PRIMARY_TARGET_EXTRA:
        return True
    return name in BAF_CORE_CANONICAL


def hub_manifest_row_matches_global_query(row: Dict[str, Any], query: str) -> bool:
    """
    True if query appears in manifest metadata (cell line, target, TMT channel, filenames, etc.).
    Used so Discovery Global Results can find runs by e.g. BRG1 or VOA without a protein hit.
    Includes canonical BAF bait resolved from the parsed target (e.g. SMARCA4 for BRG1).
    """
    q = (query or "").strip().upper()
    if not q:
        return False
    tgt = str(row.get("target", ""))
    can = identify_baf_target(tgt) or ""
    parts = [
        tgt,
        can,
        str(row.get("biological_condition", "")),
        str(row.get("cell_line", "")),
        str(row.get("tmt_channel", "")),
        str(row.get("file_name", "")),
        str(row.get("full_filename", "")),
        str(row.get("sample_label", "")),
        str(row.get("investigator", "")),
        str(row.get("session_id", "")),
    ]
    blob = " ".join(parts).upper()
    return q in blob


def resolve_search_as_bait(gene_query: str) -> Optional[str]:
    """
    Map a Discovery Hub search term to the indexed bait label in meta_df['target'], if any.
    """
    g = (gene_query or "").strip().upper()
    if not g:
        return None
    if g == "CEBPE":
        return "CEBPE"
    can = identify_baf_target(g)
    if is_core_baf_canonical(can):
        return can
    return None
