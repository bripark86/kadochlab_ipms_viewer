import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    detected_baf = identify_baf_target(rem_text)
    if detected_baf:
        target = detected_baf
    elif filename_indicates_control(stem):
        target = "Control (IgG/Mock)"

    cell_line = extract_cell_line(rem_text)
    leftovers = []
    for token in remainder:
        if identify_baf_target(token):
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
