from __future__ import annotations

import os
import sys
import builtins as _builtins
from datetime import datetime as _dt
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

_ORIG_PRINT = _builtins.print

def _timestamped_print(*args, **kwargs):
    if not args:  # bare print() used only for blank spacer lines -> leave as-is
        _ORIG_PRINT(*args, **kwargs)
        return
    stamp = _dt.now().strftime("%H:%M:%S.%f")[:-3] + " | "
    _ORIG_PRINT(stamp + str(args[0]), *args[1:], **kwargs)

def enable_timestamped_output() -> None:
    """Install the timestamped print globally (idempotent)."""
    if _builtins.print is not _timestamped_print:
        _builtins.print = _timestamped_print

enable_timestamped_output()

REPO_ROOT = Path("/home/mht/17MHT/BSPC")
INPUT_ROOT = REPO_ROOT / "Input"
OUTPUT_ROOT = REPO_ROOT / "Output"

TRAIN_MAIN_ROOT = INPUT_ROOT / "Train_Main (ProstateX)"
TRAIN_HELPER_ROOT = INPUT_ROOT / "Train_Helper (PICAI)"
TEST_ROOT = INPUT_ROOT / "Test (P158)"

MAIN_ROOT = TRAIN_MAIN_ROOT / "preprocessed"
MAIN_AI1_MASK = "mask_organ"
MAIN_AI2_MASK = "mask_organ (b)"
PRETRAIN_ROOT = TRAIN_HELPER_ROOT / "preprocessed"
P158_ROOT = TEST_ROOT / "preprocessed"

DATASET_XLSX = TRAIN_MAIN_ROOT / "ProstateX-Dataset.xlsx"
P158_DATASET_XLSX = TEST_ROOT / "P158-Dataset.xlsx"

A_PATCH_DIR = OUTPUT_ROOT / "A_Patch"
B_ORGAN_DIR = OUTPUT_ROOT / "B_Organ"
B_ORGAN_SAME_DIR = OUTPUT_ROOT / "B_Organ_same"
C_EARLY_DIR = OUTPUT_ROOT / "C_Early"
D_LATE_DIR = OUTPUT_ROOT / "D_Late"

PATCH_HELPER_INPUT_FEATURE_DIR = A_PATCH_DIR / "01_raw_features_helper"
PATCH_HELPER_MODEL_DIR = A_PATCH_DIR / "02_models_helper"
PATCH_RAW_MAIN_DIR = A_PATCH_DIR / "03_raw_features_main"
PATCH_DETECTION_DIR = A_PATCH_DIR / "04_detection_by_helper"
PATCH_ONLY_WORKBOOK_DIR = A_PATCH_DIR / "05_summary_features"

ORGAN_RAW_DIR = B_ORGAN_DIR / "01_raw_features"
ORGAN_ICC_DIR = ORGAN_RAW_DIR
ORGAN_ONLY_WORKBOOK_DIR = B_ORGAN_DIR / "02_selected_features"

ORGAN_SAME_RAW_DIR = B_ORGAN_SAME_DIR / "01_raw_features"
ORGAN_SAME_WORKBOOK_DIR = B_ORGAN_SAME_DIR / "02_selected_features"

EARLY_FUSION_WORKBOOK_DIR = C_EARLY_DIR / "01_fused_features"

EXPERIMENT_LAYOUT = {
    "A":  {"root": A_PATCH_DIR,      "results": "06_results", "final": "07_final_model", "external": "08_external_testing", "prefix": "patch"},
    "B":  {"root": B_ORGAN_DIR,      "results": "03_results", "final": "04_final_model", "external": "05_external_testing", "prefix": "organ"},
    "B0": {"root": B_ORGAN_SAME_DIR, "results": "03_results", "final": None,             "external": None,                 "prefix": "organ_same"},
    "C":  {"root": C_EARLY_DIR,      "results": "02_results", "final": "03_final_model", "external": "04_external_testing", "prefix": "early"},
    "D":  {"root": D_LATE_DIR,       "results": "01_results", "final": "02_final_model", "external": "03_external_testing", "prefix": "late"},
}

def results_dir(exp: str, seed: int) -> Path:
    layout = EXPERIMENT_LAYOUT[exp]
    return layout["root"] / layout["results"] / f"random_seed{seed}"

def final_model_dir(exp: str, seed: int) -> Path:
    layout = EXPERIMENT_LAYOUT[exp]
    if layout["final"] is None:
        raise ValueError(f"experiment {exp!r} has no final-model stage")
    return layout["root"] / layout["final"] / f"random_seed{seed}"

def external_summary_dir(exp: str) -> Path:
    layout = EXPERIMENT_LAYOUT[exp]
    if layout["external"] is None:
        raise ValueError(f"experiment {exp!r} has no external-testing stage")
    return layout["root"] / layout["external"] / "01_summary_features"

def external_results_dir(exp: str) -> Path:
    layout = EXPERIMENT_LAYOUT[exp]
    if layout["external"] is None:
        raise ValueError(f"experiment {exp!r} has no external-testing stage")
    return layout["root"] / layout["external"] / "02_results"

def final_model_path(exp: str, task: str, seed: int) -> Path:
    return final_model_dir(exp, seed) / f"{EXPERIMENT_LAYOUT[exp]['prefix']}_{task}.joblib"

def split_json_name(seed: int) -> str:
    return "data_split.json"

def split_json_path(exp: str, seed: int) -> Path:
    return results_dir(exp, seed) / split_json_name(seed)

def file_tag(feature_file: str) -> str:
    return Path(feature_file).stem

def read_table(path, *, id_col: str | None = None):
    import pandas as pd
    path = Path(path)
    dtype = {id_col: str} if id_col else None
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=dtype)
    return pd.read_csv(path, dtype=dtype)

def write_table(df, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def read_features(path):
    return read_table(path, id_col=ID_COL)

def read_predictions(path):
    return read_table(path, id_col=ID_COL)

COMPLETION_MARKER = "[ML COMPLETE]"

METRIC_ROUND_DECIMALS = 4

NO_ROUND_COLS = {
    "outer_fold", "inner_fold", "n", "n_train", "n_test", "n_oof",
    "n_input_features", "n_after_quasi", "n_after_corr", "n_selected",
    "n_base_features", "n_meta_features", "n_meta_train",
    "n_cache_candidates_selected", "n_candidates_recomputed", "n_candidates_used",
    "n_base", "pca_components", "inner_scores_used", "cache_inner_scores_used",
    "rank", "selected_k", "freq", "complexity", "candidate_rank",
    "y_true", "y_pred", "is_selected", "is_chosen",
}

def round_metrics(df, extra_exclude=()):
    """Round float metric columns to METRIC_ROUND_DECIMALS dp for result output."""
    import pandas as pd
    exclude = NO_ROUND_COLS | set(extra_exclude)
    df = df.copy()
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].round(METRIC_ROUND_DECIMALS)
    return df

def write_predictions(df, path) -> None:
    write_table(round_metrics(df), path)

def format_float_cells_4dp(writer) -> None:
    """Set the 0.0000 number format on every float cell (data rows only) of an
    openpyxl-backed ExcelWriter. Shared by every stage that writes result workbooks."""
    for ws in writer.book.worksheets:
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = "0.0000"

_MEAN_STD_SEPARATORS = ("_", " ")

def collapse_mean_std_columns(df):
    """Merge every {base}<sep>mean/{base}<sep>std column pair (sep is '_', ' ', or
    nothing, matched case-insensitively) into a single {base} column formatted as
    'mean ± std', dropping the separate mean/std columns."""
    import numpy as np
    import pandas as pd
    order = list(df.columns)
    df = df.copy()
    replaced = {}
    consumed_std_cols = set()
    for col in order:
        lower = col.lower()
        sep = next((s for s in _MEAN_STD_SEPARATORS if lower.endswith(f"{s}mean")), None)
        if sep is None:
            continue
        base = col[: len(col) - len(f"{sep}mean")]
        std_col = next(
            (c for c in df.columns if c not in replaced and c.lower() == f"{base.lower()}{sep}std"),
            None,
        )
        if std_col is None:
            continue
        merged = [
            np.nan if (pd.isna(m) and pd.isna(s)) else f"{m} ± {s}"
            for m, s in zip(df[col], df[std_col])
        ]
        df[base] = merged
        df = df.drop(columns=[col, std_col])
        replaced[col] = base
        consumed_std_cols.add(std_col)

    if not replaced:
        return df

    new_order = []
    seen = set()
    for col in order:
        if col in replaced:
            base = replaced[col]
        elif col in consumed_std_cols:
            continue
        else:
            base = col
        if base not in seen and base in df.columns:
            new_order.append(base)
            seen.add(base)
    return df[new_order]

def patch_workbook_done_ids(path) -> set:
    import pandas as pd
    p = Path(path)
    if not p.exists():
        return set()
    df = pd.read_csv(p, dtype={ID_COL: str})
    label_cols = {ID_COL, CLIN_COL, ZONE_COL}
    feat_cols = [c for c in df.columns if c not in label_cols]
    if not feat_cols:
        return set()
    mask = df[feat_cols].notna().any(axis=1)
    return {str(c).strip() for c in df.loc[mask, ID_COL]}

def log_progress(stage: str, completed: int, total: int, metric: str = "items") -> None:
    if total <= 0:
        return
    completed = min(completed, total)
    milestones = (25, 50, 75, 100)
    prev_bucket = min(4, (completed - 1) * 4 // total)
    curr_bucket = min(4, completed * 4 // total)
    if curr_bucket > prev_bucket and curr_bucket > 0:
        pct = milestones[curr_bucket - 1]
        print(f"[{stage}] {completed}/{total} {metric} ({pct}%)")

RANDOM_STATE = 7
RANDOM_SEEDS = [7, 21, 42, 73, 101]

DEFAULT_INTERNAL_EXPERIMENTS = "A,B,C"
DEFAULT_EXTERNAL_EXPERIMENTS = "A,B,C,D"
DEFAULT_STACK_EXPERIMENTS = "A,B,C,D"
DEFAULT_PIPELINE_EXPERIMENTS = "A,B,B0,C,D"
DEFAULT_LATE_FUSION_ALPHA_GRID = "0.25,0.5,0.75"
DEFAULT_STACKING_TOP_K = 8

PATCH_ONLY_RESULTS_DIR = results_dir("A", RANDOM_STATE)
ORGAN_ONLY_RESULTS_DIR = results_dir("B", RANDOM_STATE)
ORGAN_SAME_RESULTS_DIR = results_dir("B0", RANDOM_STATE)
EARLY_FUSION_RESULTS_DIR = results_dir("C", RANDOM_STATE)
LATE_FUSION_RESULTS_DIR = results_dir("D", RANDOM_STATE)

def results_dirs_for_seed(seed: int) -> dict:
    return {
        "organ": results_dir("B", seed),
        "organ_same": results_dir("B0", seed),
        "patch": results_dir("A", seed),
        "late_fusion": results_dir("D", seed),
        "early_fusion": results_dir("C", seed),
        "split_json": split_json_path("A", seed),
    }

_CPU = os.cpu_count() or 4
FEATURE_WORKERS = max(1, min(40, _CPU))
PATCH_TRAIN_WORKERS = max(1, min(40, _CPU))
PATCH_APPLY_WORKERS = max(1, min(10, _CPU))
PATCH_MODEL_TRAIN_N_JOBS = 1
EXTERNAL_PATCH_APPLY_WORKERS = max(1, min(10, _CPU))
EXCEL_FLUSH_EVERY = 5

ID_COL = "case_id"
CLIN_COL = "clin_sig"
ZONE_COL = "binary"

T2W_FOLDER = "t2w"
ADC_FOLDER = "adc"
MASK_FOLDER = "mask_organ"
LESION_MASK_FOLDER = "mask_lesion"
ZONE_MASK_FOLDER = "mask_zone"

ZONE_VALUE_TO_LABEL = {
    1: "PZ",
    2: "TZ",
}

_PATCH_FEATURE_SET_PREFIX = "feature--"

FEATURE_FILES = [
    "feature--t2w.csv",
    "feature--adc.csv",
    "feature--concat.csv",
    "feature--hada.csv",
    "feature--diff.csv",
    "feature--fusion(cd).csv",
    "feature--fusion(dh).csv",
    "feature--fusion(ch).csv",
    "feature--fusion(cdh).csv",
]

PATCH_FEATURE_SETS = [Path(name).stem[len(_PATCH_FEATURE_SET_PREFIX):] for name in FEATURE_FILES]

PATCH_FEATURE_FILE_MAP = dict(zip(PATCH_FEATURE_SETS, FEATURE_FILES))
PATCH_FEATURE_SET_BY_FILE = dict(zip(FEATURE_FILES, PATCH_FEATURE_SETS))
PATCH_MODALITIES = ["t2w", "adc"]

ML_FEATURE_SELECTORS = [
    "anova_f",
    "l1_embedded",
]
ML_CLASSIFIERS = [
    "logreg",
    "linear_svm",
    "gaussian_nb",
    "lda",
]

PATCH_SCALES = [
    {"name": "s1", "size_mm_zyx": (3.0, 3.5, 3.5), "stride_mm_zyx": (3.0, 1.0, 1.0)},
    {"name": "s2", "size_mm_zyx": (3.0, 6.5, 6.5), "stride_mm_zyx": (3.0, 2.0, 2.0)},
    {"name": "s3", "size_mm_zyx": (3.0, 9.5, 9.5), "stride_mm_zyx": (3.0, 3.5, 3.5)},
]

PATCH_ORGAN_MIN_OVERLAP = 0.90
PATCH_FIXED_POSITIVE_PATCHES_PER_CASE = 20
PATCH_MIN_POSITIVE_LESION_OVERLAP = 0.5
PATCH_NONCSPCA_MAX_LESION_OVERLAP = 0.0
PATCH_ZONE_MIN_OVERLAP = 0.80
PATCH_HEATMAP_THRESHOLD = 0.60
PATCH_CLUSTER_FEATURE_THRESHOLDS = [0.50, 0.60, 0.70]
PATCH_CSPCA_MIN_CLUSTER_Z_SLICES = 2
PATCH_CSPCA_MIN_CLUSTER_VOXELS = 1
PATCH_CLUSTER_ZONE_CONFIDENCE = 0.65
PATCH_CSPCA_Z_SUPPORT_XY_RADIUS_VOXELS = 1
PATCH_CSPCA_XY_CLOSING_ITERATIONS = 1
PATCH_SAVE_CLUSTER_OVERLAYS = True
PATCH_MAX_PATCHES_PER_CLASS_PER_SCALE = 20000
PATCH_MAX_MAJORITY_TO_MINORITY_RATIO = 2
PATCH_HELPER_RANDOM_STATE = RANDOM_STATE
