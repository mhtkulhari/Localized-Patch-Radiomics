import sys
sys.stdout.reconfigure(line_buffering=True)

import os
from pathlib import Path
import json
import logging
import traceback
import argparse
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from itertools import product
from typing import Dict, List, Tuple, Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent

import all_config as _acfg
from all_config import (
    FEATURE_FILES,
    ML_CLASSIFIERS,
    ML_FEATURE_SELECTORS,
    RANDOM_STATE,
    RANDOM_SEEDS,
)

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)


CONFIG = {

    "base_dir": "",
    "json_path": "",
    "results_dir_name": "ml_results",

    "file_order": [],

    "id_col": "case_id",
    "clin_col": "clin_sig",
    "zone_col": "binary",

    "outer_folds": 4,
    "inner_folds": 4,
    "random_state": RANDOM_STATE,

    "quasi_constant_threshold": 0.95,
    "corr_threshold": 0.95,
    "scaler": "zscore",

    "enabled_feature_selectors": list(ML_FEATURE_SELECTORS),
    "enabled_models": list(ML_CLASSIFIERS),

    "fs_k_grid": [10,15,20,30],
    "fs_l1_C_grid": [0.05, 0.1, 0.5, 1.0],

    "pca_var_grid": ["off", 0.90, 0.95, 0.99],

    "model_grids": {
        "logreg": {"C": [0.1, 0.3, 1.0, 3.0]},
        "linear_svm": {"C": [0.1, 0.3, 1.0, 3.0]},
        "gaussian_nb": {"var_smoothing": [1e-9, 1e-8, 1e-7]},
        "lda": {"shrinkage": ["auto", 0.2, 0.5]},
    },

    "max_iter": 12000,
    "tol": 1e-3,
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def setup_logger(log_path: str):
    logger = logging.getLogger("multi_ml_feature_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d | %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def append_text(path: str, text: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def is_valid_excel_workbook(path: str) -> bool:
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    if not zipfile.is_zipfile(path):
        return False
    try:
        with pd.ExcelFile(path, engine="openpyxl") as xls:
            _ = xls.sheet_names
        return True
    except Exception:
        return False


def write_workbook(path: str, sheets: Dict[str, pd.DataFrame], logger: Optional[logging.Logger] = None):
    ensure_dir(os.path.dirname(path))

    tmp_fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp.xlsx", dir=os.path.dirname(path))
    os.close(tmp_fd)

    try:
        with pd.ExcelWriter(tmp_path, engine="openpyxl", mode="w") as writer:
            for s, d in sheets.items():
                d.to_excel(writer, sheet_name=str(s)[:31], index=False)
            _acfg.format_float_cells_4dp(writer)

        if not is_valid_excel_workbook(tmp_path):
            raise RuntimeError(f"Temp workbook validation failed: {tmp_path}")

        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise


def resolve_readable_workbook(path: str, logger: logging.Logger) -> Optional[str]:
    # Workbooks are written atomically (tempfile + os.replace), so a partial file never
    # lands at `path`; either it is a valid workbook or it does not exist / is corrupt.
    if is_valid_excel_workbook(path):
        return path
    if os.path.exists(path):
        logger.warning(f"Workbook corrupt/unreadable: {path}")
    return None


def sort_output_df(df: pd.DataFrame, kind: str, final_combo_sort: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    if kind == "combo_summary" and final_combo_sort:
        metric_col = "auc_mean" if "auc_mean" in df.columns else "auc"
        sort_cols = [metric_col]
        ascending = [False]
        tie_cols = [c for c in ["file", "fs_method", "clf_model"] if c in df.columns]
        sort_cols.extend(tie_cols)
        ascending.extend([True] * len(tie_cols))
        return df.sort_values(sort_cols, ascending=ascending, kind="stable").reset_index(drop=True)

    order_map = {
        "fold_metrics": ["file", "fs_method", "clf_model", "outer_fold"],
        "inner_perf": ["file", "fs_method", "clf_model", "outer_fold", "inner_fold"],
        "selected_features": ["file", "fs_method", "clf_model", "outer_fold", "rank"],
        "combo_summary": ["file", "fs_method", "clf_model"],
        "agg_feat_rank": ["file", "fs_method", "clf_model", "freq", "mean_score", "mean_rank"],
        "pred_rows": ["file", "fs_method", "clf_model", "outer_fold", "case_id"],
    }

    cols = [c for c in order_map.get(kind, []) if c in df.columns]
    if not cols:
        return df

    ascending = [True] * len(cols)
    if kind == "agg_feat_rank":
        ascending = []
        for c in cols:
            if c in {"freq", "mean_score"}:
                ascending.append(False)
            else:
                ascending.append(True)

    return df.sort_values(cols, ascending=ascending, kind="stable").reset_index(drop=True)


def read_feature_workbook(path: str) -> pd.DataFrame:

    df = _acfg.read_table(path)
    if "case_id" not in df.columns:
        raise RuntimeError(f"{os.path.basename(path)} missing 'case_id' column")
    df["case_id"] = df["case_id"].astype(str).str.strip()
    dup = df["case_id"].duplicated(keep=False)
    if dup.any():
        examples = sorted(df.loc[dup, "case_id"].astype(str).unique().tolist())[:10]
        raise RuntimeError(
            f"{os.path.basename(path)} has duplicate case_id rows, "
            f"which can leak the same patient/case across folds. Examples: {examples}"
        )
    return df


def parse_clin(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return 1
    if s in {"false", "0", "no", "n", "f"}:
        return 0
    return np.nan


def parse_zone(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().upper()
    if s in {"PZ", "0"}:
        return "PZ"
    if s in {"TZ", "1"}:
        return "TZ"
    return np.nan


def parse_3class(clin_raw, zone_raw):
    c = parse_clin(clin_raw)
    z = parse_zone(zone_raw)
    if pd.isna(c):
        return np.nan
    if c == 0:
        return "FALSE"
    if c == 1 and z == "PZ":
        return "TRUE_PZ"
    if c == 1 and z == "TZ":
        return "TRUE_TZ"
    return np.nan


def get_feature_columns(df: pd.DataFrame, config: Dict) -> List[str]:
    non_feat = {config["id_col"], config["clin_col"], config["zone_col"]}
    return [c for c in df.columns if c not in non_feat]


def get_class_weight(task_name: str, config: Dict):
    return "balanced"


def make_hashable_params(d: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted(d.items(), key=lambda x: x[0]))


def params_to_json(d: Dict[str, Any]) -> str:
    return json.dumps(d, sort_keys=True)


_CLASS_ORDER = ["FALSE", "TRUE_TZ", "TRUE_PZ"]

def ordered_dist(counter_or_dict) -> dict:
    d = dict(counter_or_dict)
    return {k: d.get(k, 0) for k in _CLASS_ORDER if k in d or d.get(k, 0) > 0}


def parse_csv_list(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    values = [x.strip() for x in str(raw).split(",") if x.strip()]
    return values or None


def param_grid_list(grid_dict: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not grid_dict:
        return [{}]
    keys = list(grid_dict.keys())
    values = [grid_dict[k] for k in keys]
    out = []
    for combo in product(*values):
        out.append(dict(zip(keys, combo)))
    return out


def build_or_load_split_json(config: Dict, reference_file: str,
                             default_json_path: str, logger):
    user_json = config["json_path"].strip() if config["json_path"] else ""
    split_json_path = user_json or default_json_path

    if not split_json_path:
        raise RuntimeError("No split JSON path configured.")


    if os.path.exists(split_json_path):
        with open(split_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        logger.info(f"Loaded split JSON: {split_json_path}")
        return payload, split_json_path

    df = read_feature_workbook(reference_file)
    req = [config["id_col"], config["clin_col"], config["zone_col"]]
    for c in req:
        if c not in df.columns:
            raise RuntimeError(f"Reference file missing required column: {c}")

    df = df[req].copy()
    df["y3"] = df.apply(lambda r: parse_3class(r[config["clin_col"]], r[config["zone_col"]]), axis=1)
    df = df.dropna(subset=["y3"]).drop_duplicates(subset=[config["id_col"]]).copy()

    case_ids = df[config["id_col"]].astype(str).tolist()
    y3 = df["y3"].astype(str).tolist()

    case_arr = np.array(case_ids)
    y3_arr = np.array(y3)

    skf_outer = StratifiedKFold(
        n_splits=config["outer_folds"],
        shuffle=True,
        random_state=config["random_state"],
    )

    outer_splits_summary = []
    outer_splits = []
    for o_idx, (tr_idx, te_idx) in enumerate(skf_outer.split(case_arr, y3_arr), start=1):
        outer_train_ids = case_arr[tr_idx].tolist()
        outer_test_ids = case_arr[te_idx].tolist()

        case_outer = case_arr[tr_idx]
        y3_outer = y3_arr[tr_idx]

        skf_inner = StratifiedKFold(
            n_splits=config["inner_folds"],
            shuffle=True,
            random_state=config["random_state"] + o_idx,
        )

        inner_splits_summary = []
        inner_splits = []
        for i_idx, (itr, iva) in enumerate(skf_inner.split(case_outer, y3_outer), start=1):
            tr_ids = case_outer[itr].tolist()
            va_ids = case_outer[iva].tolist()
            inner_splits_summary.append({
                "inner_fold": i_idx,
                "train": {"n": len(tr_ids), "dist": ordered_dist(Counter(y3_outer[itr].tolist()))},
                "val":   {"n": len(va_ids), "dist": ordered_dist(Counter(y3_outer[iva].tolist()))},
            })
            inner_splits.append({
                "inner_fold": i_idx,
                "train_case_ids": tr_ids,
                "val_case_ids": va_ids,
            })

        outer_train_dist = ordered_dist(Counter(y3_outer.tolist()))
        outer_test_dist  = ordered_dist(Counter(y3_arr[te_idx].tolist()))
        outer_splits_summary.append({
            "outer_fold": o_idx,
            "train": {"n": len(outer_train_ids), "dist": outer_train_dist},
            "test":  {"n": len(outer_test_ids),  "dist": outer_test_dist},
            "inner_folds": inner_splits_summary,
        })
        outer_splits.append({
            "outer_fold": o_idx,
            "train_case_ids": outer_train_ids,
            "test_case_ids": outer_test_ids,
            "inner_splits": inner_splits,
        })

    overall_dist = ordered_dist(Counter(y3))
    payload = {
        "meta": {
            "description": "Shared 3-class stratified nested CV splits for all files",
            "classes": ["FALSE", "TRUE_PZ", "TRUE_TZ"],
            "outer_folds": config["outer_folds"],
            "inner_folds": config["inner_folds"],
            "random_state": config["random_state"],
            "reference_file": os.path.basename(reference_file),
        },
        "summary": {
            "n_total": len(case_ids),
            "overall": overall_dist,
            "outer_folds": outer_splits_summary,
        },
        "outer_splits": outer_splits,
    }

    ensure_dir(os.path.dirname(split_json_path))
    with open(split_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Created split JSON: {split_json_path}")
    return payload, split_json_path


class TrainPreprocessor:
    def __init__(self, quasi_thresh=0.95, corr_thresh=0.95, scaler_type="zscore"):
        self.quasi_thresh = quasi_thresh
        self.corr_thresh = corr_thresh
        self.scaler_type = scaler_type

        self.orig_cols = None
        self.imputer = None
        self.keep_quasi = None
        self.keep_corr = None
        self.scaler = None
        self.stats = {}

    def fit(self, X_df: pd.DataFrame):
        self.orig_cols = list(X_df.columns)

        self.imputer = SimpleImputer(strategy="median")
        X_imp = self.imputer.fit_transform(X_df.values.astype(float))
        X_imp_df = pd.DataFrame(X_imp, columns=self.orig_cols)

        keep_q = []
        for c in X_imp_df.columns:
            vc = X_imp_df[c].value_counts(normalize=True, dropna=False)
            top_ratio = float(vc.iloc[0]) if len(vc) > 0 else 1.0
            if top_ratio < self.quasi_thresh:
                keep_q.append(c)

        if len(keep_q) == 0 and len(X_imp_df.columns) > 0:
            vari = X_imp_df.var(axis=0).sort_values(ascending=False)
            keep_q = vari.head(1).index.tolist()

        X_q = X_imp_df[keep_q].copy()

        if X_q.shape[1] <= 1:
            keep_c = list(X_q.columns)
        else:
            arr = X_q.values.astype(float)
            corr = np.corrcoef(arr, rowvar=False)
            corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
            upper = np.triu(corr, k=1)
            drop_idx = np.where((upper > self.corr_thresh).any(axis=0))[0]
            drop_set = set(drop_idx.tolist())
            keep_c = [c for i, c in enumerate(X_q.columns) if i not in drop_set]
            if len(keep_c) == 0:
                keep_c = [X_q.columns[0]]

        X_c = X_q[keep_c].copy()

        self.scaler = StandardScaler() if self.scaler_type.lower() == "zscore" else RobustScaler()
        self.scaler.fit(X_c.values.astype(float))

        self.keep_quasi = keep_q
        self.keep_corr = keep_c
        self.stats = {
            "n_input_features": len(self.orig_cols),
            "n_after_quasi": len(self.keep_quasi),
            "n_after_corr": len(self.keep_corr),
        }

    def transform(self, X_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        X = X_df.reindex(columns=self.orig_cols).copy()
        X_imp = self.imputer.transform(X.values.astype(float))
        X_imp_df = pd.DataFrame(X_imp, columns=self.orig_cols)

        X_q = X_imp_df[self.keep_quasi].copy()
        X_c = X_q[self.keep_corr].copy()
        X_sc = self.scaler.transform(X_c.values.astype(float))
        return X_sc, list(self.keep_corr)


def fit_l1_selector(X: np.ndarray, y: np.ndarray, C: float, class_weight, config: Dict):
    # scikit-learn >= 1.8: `l1_ratio=1.0` (with the saga solver) selects a pure L1
    # penalty; `penalty=` is deprecated. This gives sparse, embedded selection.
    # NOTE: requires sklearn >= 1.8 — on older versions l1_ratio needs penalty="elasticnet".
    return LogisticRegression(
        solver="saga",
        l1_ratio=1.0,
        C=C,
        class_weight=class_weight,
        max_iter=config["max_iter"],
        tol=config["tol"],
        random_state=config["random_state"],
    ).fit(X, y)


def select_features(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_apply: np.ndarray,
    feature_names: List[str],
    params: Dict[str, Any],
    class_weight,
    config: Dict,
    logger: logging.Logger,
    context: str,
    selector_cache: Optional[Dict[Tuple[Any, ...], Tuple[List[int], np.ndarray]]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[float]]:
    n_feat = X_train.shape[1]
    if n_feat == 0:
        raise RuntimeError(f"{context} no features after preprocessing")

    k = int(min(max(params.get("k", n_feat), 1), n_feat))

    def cached_ranking(cache_key: Tuple[Any, ...], build_scores) -> Tuple[List[int], np.ndarray]:
        if selector_cache is not None and cache_key in selector_cache:
            return selector_cache[cache_key]
        scores_all = np.nan_to_num(build_scores(), nan=0.0, posinf=0.0, neginf=0.0)
        order = np.argsort(scores_all)[::-1].tolist()
        if selector_cache is not None:
            selector_cache[cache_key] = (order, scores_all)
        return order, scores_all

    if method == "none":
        idx = list(range(n_feat))
        names = list(feature_names)
        scores = [1.0] * n_feat
    elif method == "anova_f":
        order, f_vals = cached_ranking(("anova_f",), lambda: f_classif(X_train, y_train)[0])
        idx = order[:k]
        names = [feature_names[i] for i in idx]
        scores = [float(f_vals[i]) for i in idx]
    elif method == "l1_embedded":
        C = float(params.get("l1_C", 0.1))
        order, coef = cached_ranking(
            ("l1_embedded", C),
            lambda: np.abs(np.ravel(fit_l1_selector(X_train, y_train, C, class_weight, config).coef_)),
        )
        idx = order[:k]
        names = [feature_names[i] for i in idx]
        scores = [float(coef[i]) for i in idx]
    else:
        raise ValueError(f"Unknown feature selector: {method}")

    if len(idx) == 0:
        raise RuntimeError(f"{context} selector {method} returned zero features")

    Xtr_sel = X_train[:, idx]
    Xap_sel = X_apply[:, idx]
    return Xtr_sel, Xap_sel, names, scores


def build_selector_param_grid(fs_method: str, config: Dict, n_base_features: int) -> List[Dict[str, Any]]:
    if fs_method == "none":
        return [{}]

    k_valid = sorted({int(k) for k in config["fs_k_grid"] if 1 <= int(k) <= n_base_features})

    if fs_method == "anova_f":
        return [{"k": k} for k in k_valid]

    if fs_method == "l1_embedded":
        out = []
        for k in k_valid:
            for C in config["fs_l1_C_grid"]:
                out.append({"k": k, "l1_C": float(C)})
        return out

    raise ValueError(f"Unknown feature selector: {fs_method}")


def make_estimator(model_key: str, params: Dict[str, Any], task_name: str, config: Dict):
    class_weight = get_class_weight(task_name, config)

    if model_key == "logreg":
        return LogisticRegression(
            solver="lbfgs",
            l1_ratio=0.0,
            C=float(params["C"]),
            class_weight=class_weight,
            max_iter=config["max_iter"],
            tol=config["tol"],
            random_state=config["random_state"],
        )
    if model_key == "linear_svm":
        return LinearSVC(
            C=float(params["C"]),
            class_weight=class_weight,
            max_iter=config["max_iter"],
            random_state=config["random_state"],
        )
    if model_key == "gaussian_nb":
        return GaussianNB(var_smoothing=float(params["var_smoothing"]))
    if model_key == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=params["shrinkage"])

    raise ValueError(f"Unknown model_key: {model_key}")


def get_model_grid(model_key: str, config: Dict) -> List[Dict[str, Any]]:
    return param_grid_list(config["model_grids"].get(model_key, {}))


def get_scores_and_preds(clf, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X)[:, 1]
        pred = clf.predict(X)
        return prob.astype(float), pred.astype(int), "predict_proba"

    if hasattr(clf, "decision_function"):
        score = clf.decision_function(X)
        score = np.ravel(score).astype(float)
        pred = clf.predict(X)
        return score, pred.astype(int), "decision_function"

    pred = clf.predict(X)
    return pred.astype(float), pred.astype(int), "predict"


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(np.unique(y_true)) >= 2:
        auc = float(roc_auc_score(y_true, y_score))
        ap = float(average_precision_score(y_true, y_score))
    else:
        auc = np.nan
        ap = np.nan

    bal = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan

    return {
        "auc": round(auc, 4),
        "ap": round(ap, 4),
        "balanced_acc": round(bal, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
    }


def choose_best_params(
    param_scores: Dict[Tuple, Dict[str, List[float]]],
    expected_folds: int,
    logger: logging.Logger,
    context: str,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    best_key = None
    best_rank = None
    best_stats = None

    if expected_folds <= 0:
        return {}, {}

    issue_txt_path = None
    for h in getattr(logger, "handlers", []):
        if hasattr(h, "baseFilename"):
            issue_txt_path = os.path.join(os.path.dirname(h.baseFilename), "inner_selection_excluded_cases.txt")
            break

    for key, vals in param_scores.items():
        aucs = np.array(vals.get("auc", []), dtype=float)
        aps = np.array(vals.get("ap", []), dtype=float)

        reject_reason = None
        if len(aucs) != expected_folds or len(aps) != expected_folds:
            reject_reason = (
                f"incomplete_inner_coverage expected_folds={expected_folds} "
                f"got_auc={len(aucs)} got_ap={len(aps)}"
            )
        elif np.any(np.isnan(aucs)) or np.any(np.isnan(aps)):
            reject_reason = (
                f"nan_scores_present expected_folds={expected_folds} "
                f"valid_auc={int(np.sum(~np.isnan(aucs)))} valid_ap={int(np.sum(~np.isnan(aps)))}"
            )

        if reject_reason is not None:
            if issue_txt_path is not None:
                append_text(
                    issue_txt_path,
                    (
                        f"[{context}] rejected_param_set reason={reject_reason} | "
                        f"key={json.dumps(dict(key), sort_keys=True)} | "
                        f"auc={aucs.tolist()} | ap={aps.tolist()}"
                    ),
                )
            continue

        mean_auc = float(np.mean(aucs))
        mean_ap = float(np.mean(aps))
        std_auc = float(np.std(aucs))

        rank = (mean_auc, mean_ap, -std_auc)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_key = key
            best_stats = {
                "inner_mean_auc": round(mean_auc, 4),
                "inner_mean_ap": round(mean_ap, 4),
                "inner_std_auc": round(std_auc, 4),
                "inner_scores_used": int(len(aucs)),
            }

    if best_key is None:
        return {}, {}

    out = dict(best_key)
    out.update(best_stats)
    return out, best_stats


def apply_optional_pca(
    X_train: np.ndarray,
    X_apply: np.ndarray,
    pca_var: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, Optional[PCA], int, float]:
    if pca_var in {None, "off", "none", False} or X_train.shape[1] <= 1:
        return X_train, X_apply, None, int(X_train.shape[1]), 1.0

    pca = PCA(n_components=pca_var, svd_solver="full")
    Xtr_p = pca.fit_transform(X_train)
    Xap_p = pca.transform(X_apply)
    return (
        Xtr_p,
        Xap_p,
        pca,
        int(getattr(pca, "n_components_", Xtr_p.shape[1])),
        round(float(np.sum(pca.explained_variance_ratio_)), 4),
    )


def prepare_model_data(df: pd.DataFrame, task_name: str, config: Dict) -> pd.DataFrame:
    d = df.copy()
    d[config["clin_col"]] = d[config["clin_col"]].apply(parse_clin)
    d[config["zone_col"]] = d[config["zone_col"]].apply(parse_zone)

    if task_name == "cs":
        d = d.dropna(subset=[config["clin_col"]]).copy()
        d["y"] = d[config["clin_col"]].astype(int)
        d["y_label"] = d["y"].map({0: "FALSE", 1: "TRUE"})
    elif task_name == "zone":
        d = d[d[config["clin_col"]] == 1].copy()
        d = d[d[config["zone_col"]].isin(["PZ", "TZ"])].copy()
        d["y"] = d[config["zone_col"]].map({"PZ": 0, "TZ": 1}).astype(int)
        d["y_label"] = d["y"].map({0: "PZ", 1: "TZ"})
    else:
        raise ValueError("task_name must be 'cs' or 'zone'")

    d = d.drop_duplicates(subset=[config["id_col"]]).copy()
    return d


def prepare_base_fold_data(
    train_df: pd.DataFrame,
    apply_df: pd.DataFrame,
    feature_cols: List[str],
    config: Dict,
) -> Optional[Dict[str, Any]]:
    Xtr_df = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    Xap_df = apply_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    ytr = train_df["y"].values.astype(int)
    yap = apply_df["y"].values.astype(int)

    prep = TrainPreprocessor(
        quasi_thresh=config["quasi_constant_threshold"],
        corr_thresh=config["corr_threshold"],
        scaler_type=config["scaler"],
    )
    prep.fit(Xtr_df)
    Xtr_base, base_names = prep.transform(Xtr_df)
    Xap_base, _ = prep.transform(Xap_df)

    if Xtr_base.shape[1] < 1:
        return None

    return {
        "Xtr_base": Xtr_base,
        "Xap_base": Xap_base,
        "base_names": base_names,
        "ytr": ytr,
        "yap": yap,
        "prep": prep,
        "n_base_features": int(Xtr_base.shape[1]),
        "selector_cache": {},
        "selected_cache": {},
        "pca_cache": {},
    }


def run_inner_search_combo(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    inner_splits: List[Dict],
    task_name: str,
    fs_method: str,
    clf_model: str,
    config: Dict,
    logger: logging.Logger,
    context: str,
    prepared_folds_cache: Optional[Dict[Any, List[Dict[str, Any]]]] = None,
    prepared_folds_cache_key: Optional[Any] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    param_scores = defaultdict(lambda: {"auc": [], "ap": []})
    rows = []
    class_weight = get_class_weight(task_name, config)

    prepared_folds = None
    if prepared_folds_cache is not None and prepared_folds_cache_key in prepared_folds_cache:
        prepared_folds = prepared_folds_cache[prepared_folds_cache_key]

    if prepared_folds is None:
        prepared_folds = []

        for inner in inner_splits:
            i_fold = inner["inner_fold"]
            tr_ids = set(inner["train_case_ids"])
            va_ids = set(inner["val_case_ids"])

            dtr = train_df[train_df["case_id"].isin(tr_ids)].copy()
            dva = train_df[train_df["case_id"].isin(va_ids)].copy()

            if dtr.empty or dva.empty:
                continue
            if dtr["y"].nunique() < 2 or dva["y"].nunique() < 2:
                continue

            fold_data = prepare_base_fold_data(dtr, dva, feature_cols, config)
            if fold_data is None:
                continue

            fold_data["inner_fold"] = i_fold
            prepared_folds.append(fold_data)

        if prepared_folds_cache is not None:
            prepared_folds_cache[prepared_folds_cache_key] = prepared_folds

    valid_inner_folds = len(prepared_folds)
    if valid_inner_folds == 0:
        return {}, pd.DataFrame(rows)

    common_max_k = min(f["n_base_features"] for f in prepared_folds)
    fs_param_grid = build_selector_param_grid(fs_method, config, common_max_k)
    model_param_grid = get_model_grid(clf_model, config)

    if fs_method != "none" and not fs_param_grid:
        logger.warning(f"[INNER] No valid fs params for {context}. common_max_k={common_max_k}")
        return {}, pd.DataFrame(rows)

    for fold_data in prepared_folds:
        i_fold = fold_data["inner_fold"]
        Xtr_base = fold_data["Xtr_base"]
        Xva_base = fold_data["Xap_base"]
        base_names = fold_data["base_names"]
        ytr = fold_data["ytr"]
        yva = fold_data["yap"]
        prep = fold_data["prep"]
        selector_cache = fold_data.setdefault("selector_cache", {})
        selected_cache = fold_data.setdefault("selected_cache", {})
        pca_cache = fold_data.setdefault("pca_cache", {})

        for fs_params in fs_param_grid:
            fs_params_json = params_to_json(fs_params)
            selected_key = (fs_method, fs_params_json)
            try:
                if selected_key in selected_cache:
                    Xtr_sel, Xva_sel, sel_names, _ = selected_cache[selected_key]
                else:
                    Xtr_sel, Xva_sel, sel_names, sel_scores = select_features(
                        method=fs_method,
                        X_train=Xtr_base,
                        y_train=ytr,
                        X_apply=Xva_base,
                        feature_names=base_names,
                        params=fs_params,
                        class_weight=class_weight,
                        config=config,
                        logger=logger,
                        context=f"{context}:inner{i_fold}:{fs_method}",
                        selector_cache=selector_cache,
                    )
                    selected_cache[selected_key] = (Xtr_sel, Xva_sel, sel_names, sel_scores)
            except Exception as e:
                logger.warning(f"[INNER][FS FAIL] {context}:inner{i_fold} fs={fs_method} params={fs_params} err={repr(e)}")
                continue

            for pca_var in config["pca_var_grid"]:
                pca_key = (fs_method, fs_params_json, json.dumps(pca_var, sort_keys=True))
                try:
                    if pca_key in pca_cache:
                        Xtr_final, Xva_final, pca_components, pca_explained = pca_cache[pca_key]
                    else:
                        Xtr_final, Xva_final, _, pca_components, pca_explained = apply_optional_pca(
                            Xtr_sel, Xva_sel, pca_var
                        )
                        pca_cache[pca_key] = (Xtr_final, Xva_final, pca_components, pca_explained)
                except Exception as e:
                    logger.warning(
                        f"[INNER][PCA FAIL] {context}:inner{i_fold} fs={fs_method} params={fs_params} pca={pca_var} err={repr(e)}"
                    )
                    continue

                for model_params in model_param_grid:
                    auc = np.nan
                    ap = np.nan
                    score_source = "unknown"
                    try:
                        clf = make_estimator(clf_model, model_params, task_name, config)
                        clf.fit(Xtr_final, ytr)
                        y_score, y_pred, score_source = get_scores_and_preds(clf, Xva_final)
                        if len(np.unique(yva)) >= 2:
                            auc = float(roc_auc_score(yva, y_score))
                            ap = float(average_precision_score(yva, y_score))
                    except Exception as e:
                        logger.warning(
                            f"[INNER][CLF FAIL] {context}:inner{i_fold} fs={fs_method} params={fs_params} "
                            f"pca={pca_var} clf={clf_model} clf_params={model_params} err={repr(e)}"
                        )

                    key = {
                        "fs_method": fs_method,
                        "fs_params": params_to_json(fs_params),
                        "pca_var": pca_var,
                        "clf_model": clf_model,
                        "clf_params": params_to_json(model_params),
                    }
                    param_scores[make_hashable_params(key)]["auc"].append(auc)
                    param_scores[make_hashable_params(key)]["ap"].append(ap)

                    rows.append({
                        "inner_fold": i_fold,
                        "fs_method": fs_method,
                        "fs_params": fs_params_json,
                        "n_selected": int(len(sel_names)),
                        "pca_var": pca_var,
                        "pca_components": pca_components,
                        "pca_explained_var_sum": pca_explained,
                        "clf_model": clf_model,
                        "clf_params": params_to_json(model_params),
                        "score_source": score_source,
                        "auc": round(auc, 4),
                        "ap": round(ap, 4),
                        "n_base_features": int(Xtr_base.shape[1]),
                        "n_after_quasi": prep.stats.get("n_after_quasi", np.nan),
                        "n_after_corr": prep.stats.get("n_after_corr", np.nan),
                    })

    best_params, _ = choose_best_params(
        param_scores=param_scores,
        expected_folds=valid_inner_folds,
        logger=logger,
        context=context,
    )
    return best_params, pd.DataFrame(rows)
    
def evaluate_outer_fold_combo(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    best_params: Dict[str, Any],
    task_name: str,
    config: Dict,
    outer_fold: int,
    logger: logging.Logger,
    context: str,
    outer_base_cache: Optional[Dict[Any, Dict[str, Any]]] = None,
    outer_base_cache_key: Optional[Any] = None,
):
    base_data = None
    if outer_base_cache is not None and outer_base_cache_key in outer_base_cache:
        base_data = outer_base_cache[outer_base_cache_key]

    if base_data is None:
        base_data = prepare_base_fold_data(train_df, test_df, feature_cols, config)
        if base_data is None:
            raise RuntimeError(f"{context}:outer{outer_fold} no features after preprocessing")
        if outer_base_cache is not None:
            outer_base_cache[outer_base_cache_key] = base_data

    Xtr_base = base_data["Xtr_base"]
    Xte_base = base_data["Xap_base"]
    base_names = base_data["base_names"]
    ytr = base_data["ytr"]
    yte = base_data["yap"]
    prep = base_data["prep"]

    class_weight = get_class_weight(task_name, config)
    fs_method = best_params["fs_method"]
    fs_params = json.loads(best_params["fs_params"])
    clf_model = best_params["clf_model"]
    clf_params = json.loads(best_params["clf_params"])
    pca_var = best_params["pca_var"]
    selector_cache = base_data.setdefault("selector_cache", {})
    selected_cache = base_data.setdefault("selected_cache", {})
    pca_cache = base_data.setdefault("pca_cache", {})

    selected_key = (fs_method, best_params["fs_params"])
    if selected_key in selected_cache:
        Xtr_sel, Xte_sel, sel_names, sel_scores = selected_cache[selected_key]
    else:
        Xtr_sel, Xte_sel, sel_names, sel_scores = select_features(
            method=fs_method,
            X_train=Xtr_base,
            y_train=ytr,
            X_apply=Xte_base,
            feature_names=base_names,
            params=fs_params,
            class_weight=class_weight,
            config=config,
            logger=logger,
            context=f"{context}:outer{outer_fold}:{fs_method}",
            selector_cache=selector_cache,
        )
        selected_cache[selected_key] = (Xtr_sel, Xte_sel, sel_names, sel_scores)

    pca_key = (fs_method, best_params["fs_params"], json.dumps(pca_var, sort_keys=True))
    if pca_key in pca_cache:
        Xtr_final, Xte_final, pca_components, pca_explained = pca_cache[pca_key]
    else:
        Xtr_final, Xte_final, _, pca_components, pca_explained = apply_optional_pca(Xtr_sel, Xte_sel, pca_var)
        pca_cache[pca_key] = (Xtr_final, Xte_final, pca_components, pca_explained)

    clf = make_estimator(clf_model, clf_params, task_name, config)
    clf.fit(Xtr_final, ytr)
    y_score, y_pred, score_source = get_scores_and_preds(clf, Xte_final)
    mets = compute_metrics(yte, y_score, y_pred)

    fold_row = {
        "outer_fold": outer_fold,
        "fs_method": fs_method,
        "fs_params": best_params["fs_params"],
        "clf_model": clf_model,
        "clf_params": best_params["clf_params"],
        "score_source": score_source,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "train_class_dist": json.dumps(dict(Counter(ytr.tolist())), sort_keys=True),
        "test_class_dist": json.dumps(dict(Counter(yte.tolist())), sort_keys=True),
        "n_input_features": prep.stats.get("n_input_features", np.nan),
        "n_after_quasi": prep.stats.get("n_after_quasi", np.nan),
        "n_after_corr": prep.stats.get("n_after_corr", np.nan),
        "n_selected": int(len(sel_names)),
        "pca_var": pca_var,
        "pca_components": pca_components,
        "pca_explained_var_sum": pca_explained,
        "auc": mets["auc"],
        "ap": mets["ap"],
        "balanced_acc": mets["balanced_acc"],
        "f1": mets["f1"],
        "accuracy": mets["accuracy"],
        "sensitivity": mets["sensitivity"],
        "specificity": mets["specificity"],
        "inner_mean_auc": best_params.get("inner_mean_auc", np.nan),
        "inner_mean_ap": best_params.get("inner_mean_ap", np.nan),
        "inner_std_auc": best_params.get("inner_std_auc", np.nan),
        "inner_scores_used": best_params.get("inner_scores_used", np.nan),
    }

    feat_df = pd.DataFrame({
        "rank": np.arange(1, len(sel_names) + 1),
        "feature": sel_names,
        "score": sel_scores,
        "selected_k": len(sel_names),
    }).sort_values(["score", "rank"], ascending=[False, True]).reset_index(drop=True)
    feat_df["score"] = feat_df["score"].round(4)

    pred_df = pd.DataFrame({
        "case_id": test_df["case_id"].astype(str).values,
        "y_true": yte,
        "y_score": np.round(y_score, 4),
        "y_pred": y_pred,
    })

    return fold_row, feat_df, pred_df


def summarize_combo_metrics(
    df_fold: pd.DataFrame,
    file_tag: str,
    fs_method: str,
    clf_model: str,
) -> pd.DataFrame:
    if df_fold.empty:
        return pd.DataFrame([{
            "file": file_tag,
            "fs_method": fs_method,
            "clf_model": clf_model,
            "n_outer_folds": 0,
        }])

    out = {
        "file": file_tag,
        "fs_method": fs_method,
        "clf_model": clf_model,
        "n_outer_folds": int(df_fold["outer_fold"].nunique()),
    }

    metrics = ["auc", "ap", "balanced_acc", "f1", "accuracy", "sensitivity", "specificity"]
    for m in metrics:
        s = pd.to_numeric(df_fold[m], errors="coerce")
        out[f"{m}_mean"] = float(s.mean())
        out[f"{m}_std"] = float(s.std(ddof=0))

    out["stability_index_auc"] = out["auc_mean"] - out["auc_std"]

    def mode_or_nan(series):
        vals = series.dropna().tolist()
        if not vals:
            return np.nan
        return Counter(vals).most_common(1)[0][0]

    out["mode_fs_params"] = mode_or_nan(df_fold["fs_params"])
    out["mode_clf_params"] = mode_or_nan(df_fold["clf_params"])
    out["mode_pca_var"] = mode_or_nan(df_fold["pca_var"])
    out["mean_n_after_corr"] = float(pd.to_numeric(df_fold["n_after_corr"], errors="coerce").mean())
    out["mean_n_selected"] = float(pd.to_numeric(df_fold["n_selected"], errors="coerce").mean())
    out["mean_pca_components"] = float(pd.to_numeric(df_fold["pca_components"], errors="coerce").mean())

    for key in (
        [f"{m}_mean" for m in metrics] + [f"{m}_std" for m in metrics]
        + ["stability_index_auc", "mean_n_after_corr", "mean_n_selected", "mean_pca_components"]
    ):
        out[key] = round(out[key], 4)

    return pd.DataFrame([out])


def aggregate_selected_features(
    sel_df: pd.DataFrame,
    file_tag: str,
    fs_method: str,
    clf_model: str,
) -> pd.DataFrame:
    if sel_df.empty:
        return pd.DataFrame(columns=[
            "file", "fs_method", "clf_model",
            "feature", "freq", "mean_score", "mean_rank",
        ])

    g = sel_df.groupby("feature", as_index=False).agg(
        freq=("feature", "count"),
        mean_score=("score", "mean"),
        mean_rank=("rank", "mean"),
    )
    g["mean_score"] = g["mean_score"].round(4)
    g["mean_rank"] = g["mean_rank"].round(4)
    g["file"] = file_tag
    g["fs_method"] = fs_method
    g["clf_model"] = clf_model
    g = g.sort_values(["freq", "mean_score", "mean_rank"], ascending=[False, False, True]).reset_index(drop=True)
    return g[[
        "file", "fs_method", "clf_model",
        "feature", "freq", "mean_score", "mean_rank",
    ]]


def build_task_workbooks(task_store: Dict[str, List[Dict]], file_out_dir: str, file_tag: str, task_name: str, logger: Optional[logging.Logger] = None):
    analysis_book = os.path.join(file_out_dir, f"results_{task_name}_{file_tag}.xlsx")
    pred_csv = os.path.join(file_out_dir, f"predictions_{task_name}_{file_tag}.csv")

    fold_df = sort_output_df(pd.DataFrame(task_store["fold_rows"]), "fold_metrics")
    inner_df = sort_output_df(pd.DataFrame(task_store["inner_rows"]), "inner_perf")
    sel_df = sort_output_df(pd.DataFrame(task_store["sel_rows"]), "selected_features")
    summary_df = sort_output_df(pd.DataFrame(task_store["summary_rows"]), "combo_summary", final_combo_sort=True)
    agg_df = sort_output_df(pd.DataFrame(task_store["agg_feature_rows"]), "agg_feat_rank")
    pred_df = sort_output_df(pd.DataFrame(task_store["pred_rows"]), "pred_rows")

    write_workbook(analysis_book, {
        "combo_summary": _acfg.collapse_mean_std_columns(_acfg.round_metrics(summary_df)),
        "fold_metrics": _acfg.round_metrics(fold_df),
        "inner_perf": _acfg.round_metrics(inner_df),
        "selected_features": _acfg.round_metrics(sel_df),
        "agg_feat_rank": _acfg.round_metrics(agg_df),
    }, logger=logger)
    _acfg.write_predictions(pred_df, pred_csv)


def build_minimal_task_summary(summary_df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    cols_needed = [
        "file", "fs_method", "clf_model",
        "auc", "f1", "sensitivity", "specificity", "accuracy",
    ]
    if summary_df is None or summary_df.empty:
        return pd.DataFrame(columns=[
            "file", "fs_method", "clf_model",
            f"auc_{suffix}", f"f1_{suffix}", f"sensitivity_{suffix}", f"specificity_{suffix}", f"accuracy_{suffix}",
        ])

    out = summary_df[cols_needed].copy()
    out = out.rename(columns={
        "auc": f"auc_{suffix}",
        "f1": f"f1_{suffix}",
        "sensitivity": f"sensitivity_{suffix}",
        "specificity": f"specificity_{suffix}",
        "accuracy": f"accuracy_{suffix}",
    })
    return out


def reorder_df_like_reference(df: pd.DataFrame, ref_df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if ref_df is None or ref_df.empty:
        return df.reset_index(drop=True)

    ref_keys = ref_df[key_cols].drop_duplicates().copy()
    ref_keys["__sort_order"] = np.arange(len(ref_keys))
    merged = df.merge(ref_keys, on=key_cols, how="left")
    merged = merged.sort_values(["__sort_order"] + key_cols, kind="stable").drop(columns=["__sort_order"])
    return merged.reset_index(drop=True)


def build_combined_average_workbook(file_out_dir: str, file_tag: str, cs_summary_full: pd.DataFrame, zone_summary_full: pd.DataFrame, logger: Optional[logging.Logger] = None):
    def _normalize_summary_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        work = _acfg.round_metrics(df).copy()
        metric_pairs = {
            "auc": ("auc_mean", "auc"),
            "f1": ("f1_mean", "f1"),
            "sensitivity": ("sensitivity_mean", "sensitivity"),
            "specificity": ("specificity_mean", "specificity"),
            "accuracy": ("accuracy_mean", "accuracy"),
        }
        rename_map = {}
        for raw_name, (mean_name, raw_alt) in metric_pairs.items():
            if raw_name in work.columns:
                continue
            if mean_name in work.columns:
                rename_map[mean_name] = raw_name
            elif raw_alt in work.columns:
                continue
            else:
                raise KeyError(f"Missing metric column for {raw_name}: expected {mean_name} or {raw_alt}")
        if rename_map:
            work = work.rename(columns=rename_map)
        keep_cols = ["file", "fs_method", "clf_model"] + list(metric_pairs.keys())
        return work[keep_cols]

    cs_summary_full = _normalize_summary_df(cs_summary_full)
    zone_summary_full = _normalize_summary_df(zone_summary_full)
    cs_min = build_minimal_task_summary(cs_summary_full, "cs")
    zone_min = build_minimal_task_summary(zone_summary_full, "zone")

    key_cols = ["file", "fs_method", "clf_model"]
    merged = cs_min.merge(zone_min, on=key_cols, how="inner")

    combo = merged[key_cols].copy()
    combo["auc"] = (merged["auc_cs"] + merged["auc_zone"]) / 2.0
    combo["f1"] = (merged["f1_cs"] + merged["f1_zone"]) / 2.0
    combo["sensitivity"] = (merged["sensitivity_cs"] + merged["sensitivity_zone"]) / 2.0
    combo["specificity"] = (merged["specificity_cs"] + merged["specificity_zone"]) / 2.0
    combo["accuracy"] = (merged["accuracy_cs"] + merged["accuracy_zone"]) / 2.0

    combo = sort_output_df(combo, "combo_summary", final_combo_sort=True)
    cs_min = reorder_df_like_reference(cs_min, combo, key_cols)
    zone_min = reorder_df_like_reference(zone_min, combo, key_cols)

    combo = _acfg.round_metrics(combo)
    return combo[["file", "fs_method", "clf_model", "auc", "f1", "sensitivity", "specificity", "accuracy"]]


def filter_summary_to_active_grid(summary_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    if summary_df is None or summary_df.empty:
        return summary_df

    out = summary_df.copy()
    filters = {
        "fs_method": set(map(str, config["enabled_feature_selectors"])),
        "clf_model": set(map(str, config["enabled_models"])),
    }
    for col, allowed in filters.items():
        if col in out.columns:
            out = out[out[col].astype(str).isin(allowed)].copy()
    return out.reset_index(drop=True)


def _top_report_df(df: pd.DataFrame, metric_cols: List[str], top_n: int = 10) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    key_cols = [col for col in ["file", "fs_method", "clf_model", "alpha"] if col in work.columns]
    cols = key_cols + [col for col in metric_cols if col in work.columns]
    if "n_outer_folds" in work.columns:
        cols.append("n_outer_folds")
    cols = list(dict.fromkeys(cols))
    sort_cols = [col for col in metric_cols if col in work.columns]
    ascending = [False] * len(sort_cols)
    return work.sort_values(sort_cols + key_cols, ascending=ascending + [True] * len(key_cols), kind="stable")[cols].head(top_n).reset_index(drop=True)


def _write_report_section(lines: List[str], title: str, df: pd.DataFrame) -> None:
    lines.append(title)
    if df is None or df.empty:
        lines.append("No rows.")
    else:
        lines.append(df.to_string(index=False))
    lines.append("")


def write_top10_final_report(
    report_txt: str,
    split_used: str,
    cs_summaries: List[pd.DataFrame],
    zone_summaries: List[pd.DataFrame],
    combo_summaries: List[pd.DataFrame],
    notes: Optional[List[str]] = None,
) -> None:
    cs_all = pd.concat([df for df in cs_summaries if df is not None and not df.empty], ignore_index=True) if cs_summaries else pd.DataFrame()
    zone_all = pd.concat([df for df in zone_summaries if df is not None and not df.empty], ignore_index=True) if zone_summaries else pd.DataFrame()
    combo_all = pd.concat([df for df in combo_summaries if df is not None and not df.empty], ignore_index=True) if combo_summaries else pd.DataFrame()

    task_metrics = ["auc_mean", "f1_mean", "sensitivity_mean", "specificity_mean", "accuracy_mean"]
    combo_metrics = ["auc", "f1", "sensitivity", "specificity", "accuracy"]

    lines = [
        "================ FINAL TOP-10 REPORT ================",
        f"Split JSON used: {split_used}",
        "",
    ]
    if notes:
        lines.append("[Notes]")
        lines.extend(notes)
        lines.append("")

    _write_report_section(lines, "[CS summary]", _top_report_df(cs_all, task_metrics))
    _write_report_section(lines, "[ZONE summary]", _top_report_df(zone_all, task_metrics))
    _write_report_section(lines, "[CS/ZONE average summary]", _top_report_df(combo_all, combo_metrics))

    ensure_dir(os.path.dirname(report_txt))
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def safe_read_workbook_sheet(path: str, sheet_name: str, logger: logging.Logger) -> pd.DataFrame:
    resolved_path = resolve_readable_workbook(path, logger)
    if resolved_path is None:
        return pd.DataFrame()

    try:
        return pd.read_excel(resolved_path, sheet_name=sheet_name)
    except Exception as e:
        logger.warning(f"Failed to read workbook sheet {resolved_path}:{sheet_name}. err={repr(e)}")
        return pd.DataFrame()


def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    return df.where(pd.notnull(df), None).to_dict("records")


def make_empty_task_store() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "fold_rows": [],
        "inner_rows": [],
        "sel_rows": [],
        "pred_rows": [],
        "summary_rows": [],
        "agg_feature_rows": [],
    }


def load_existing_task_results(file_out_dir: str, file_tag: str, task_name: str, logger: logging.Logger) -> Dict[str, List[Dict[str, Any]]]:
    task_store = make_empty_task_store()

    analysis_book = os.path.join(file_out_dir, f"results_{task_name}_{file_tag}.xlsx")
    pred_csv = os.path.join(file_out_dir, f"predictions_{task_name}_{file_tag}.csv")

    analysis_map = {
        "fold_rows": "fold_metrics",
        "inner_rows": "inner_perf",
        "sel_rows": "selected_features",
        "summary_rows": "combo_summary",
        "agg_feature_rows": "agg_feat_rank",
    }

    resolved_analysis = resolve_readable_workbook(analysis_book, logger)

    for key, sheet_name in analysis_map.items():
        if resolved_analysis is None:
            task_store[key] = []
        else:
            task_store[key] = df_to_records(safe_read_workbook_sheet(resolved_analysis, sheet_name, logger))

    if os.path.exists(pred_csv):
        task_store["pred_rows"] = df_to_records(_acfg.read_predictions(pred_csv))
    else:
        task_store["pred_rows"] = []
    return task_store


def remove_combo_from_task_store(
    task_store: Dict[str, List[Dict[str, Any]]],
    file_tag: str,
    fs_method: str,
    clf_model: str,
):
    def _keep(row: Dict[str, Any]) -> bool:
        return not (
            str(row.get("file")) == str(file_tag)
            and str(row.get("fs_method")) == str(fs_method)
            and str(row.get("clf_model")) == str(clf_model)
        )

    for key in ["fold_rows", "inner_rows", "sel_rows", "pred_rows", "summary_rows", "agg_feature_rows"]:
        task_store[key] = [row for row in task_store.get(key, []) if _keep(row)]


def _row_matches_combo(row: Dict[str, Any], file_tag: str, fs_method: str, clf_model: str) -> bool:
    return (
        str(row.get("file")) == str(file_tag)
        and str(row.get("fs_method")) == str(fs_method)
        and str(row.get("clf_model")) == str(clf_model)
    )


def _fold_set(rows: List[Dict[str, Any]], file_tag: str, fs_method: str, clf_model: str) -> set:
    folds = set()
    for row in rows:
        if not _row_matches_combo(row, file_tag, fs_method, clf_model):
            continue
        try:
            folds.add(int(row.get("outer_fold")))
        except Exception:
            pass
    return folds


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def task_store_has_completed_combo(
    task_store: Dict[str, List[Dict[str, Any]]],
    file_tag: str,
    fs_method: str,
    clf_model: str,
    expected_outer_folds: Optional[set] = None,
) -> bool:
    expected_outer_folds = set(expected_outer_folds or [])
    has_summary = any(
        _row_matches_combo(row, file_tag, fs_method, clf_model)
        and _safe_int(row.get("n_outer_folds", 0), default=0) >= len(expected_outer_folds)
        for row in task_store.get("summary_rows", [])
    )
    if not has_summary:
        return False

    fold_folds = _fold_set(task_store.get("fold_rows", []), file_tag, fs_method, clf_model)
    pred_folds = _fold_set(task_store.get("pred_rows", []), file_tag, fs_method, clf_model)
    if expected_outer_folds:
        return expected_outer_folds.issubset(fold_folds) and expected_outer_folds.issubset(pred_folds)
    return bool(fold_folds) and fold_folds.issubset(pred_folds)


def validate_task_store_complete(
    task_store: Dict[str, List[Dict[str, Any]]],
    file_tag: str,
    task_name: str,
    config: Dict[str, Any],
    expected_outer_folds: set,
    max_examples: int = 12,
) -> List[str]:
    problems = []
    for fs_method in config["enabled_feature_selectors"]:
        for clf_model in config["enabled_models"]:
            key = (str(fs_method), str(clf_model))
            if task_store_has_completed_combo(
                task_store, file_tag, fs_method, clf_model, expected_outer_folds=expected_outer_folds
            ):
                continue
            fold_folds = _fold_set(task_store.get("fold_rows", []), file_tag, fs_method, clf_model)
            pred_folds = _fold_set(task_store.get("pred_rows", []), file_tag, fs_method, clf_model)
            summary_folds = [
                _safe_int(row.get("n_outer_folds", 0), default=0)
                for row in task_store.get("summary_rows", [])
                if _row_matches_combo(row, file_tag, fs_method, clf_model)
            ]
            problems.append(
                f"{file_tag}:{task_name}:{key} "
                f"summary_n_outer={summary_folds[:3]} "
                f"missing_fold_metrics={sorted(expected_outer_folds - fold_folds)} "
                f"missing_predictions={sorted(expected_outer_folds - pred_folds)}"
            )
            if len(problems) >= max_examples:
                return problems
    return problems


def parse_last_running_combo_from_log(log_path: str) -> Optional[Dict[str, str]]:
    if not os.path.exists(log_path):
        return None

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return None

    marker = "| INFO | Running "
    for line in reversed(lines):
        if marker not in line:
            continue
        combo = line.split(marker, 1)[1].strip()
        parts = combo.split(":")
        if len(parts) != 4:
            continue
        return {
            "file_tag": parts[0],
            "task_name": parts[1],
            "fs_method": parts[2],
            "clf_model": parts[3],
            "raw": combo,
        }
    return None


def is_resume_target(
    target: Optional[Dict[str, str]],
    file_tag: str,
    task_name: str,
    fs_method: str,
    clf_model: str,
) -> bool:
    if target is None:
        return False
    return (
        target["file_tag"] == file_tag
        and target["task_name"] == task_name
        and target["fs_method"] == fs_method
        and target["clf_model"] == clf_model
    )


def _clean_case_id_for_verify(v) -> Optional[str]:
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null", "<na>", "na"}:
        return None
    return s


def _parse_clin_for_verify(v) -> Optional[int]:
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return 1
    if s in {"false", "0", "no", "n", "f"}:
        return 0
    return None


def _parse_zone_for_verify(v) -> Optional[str]:
    if pd.isna(v):
        return None
    s = str(v).strip().upper()
    if s in {"PZ", "0"}:
        return "PZ"
    if s in {"TZ", "1"}:
        return "TZ"
    return None


def _load_reference_label_maps_for_verify(logger: logging.Logger) -> Dict[str, Dict[str, Any]]:
    try:
        df = pd.read_excel(DATASET_XLSX, dtype={ID_COL: str})
    except Exception as exc:
        raise RuntimeError(f"could_not_read_reference_labels_for_coverage: {DATASET_XLSX} ({repr(exc)})") from exc

    df.columns = [str(c).strip() for c in df.columns]
    if ID_COL not in df.columns:
        lowered = {str(c).strip().lower(): c for c in df.columns}
        if ID_COL.lower() not in lowered:
            raise RuntimeError(f"reference_label_file_missing_case_id_for_coverage: {DATASET_XLSX}")
        df = df.rename(columns={lowered[ID_COL.lower()]: ID_COL})

    needed = {ID_COL, CLIN_COL, ZONE_COL}
    if not needed.issubset(df.columns):
        raise RuntimeError(f"reference_label_file_missing_columns_for_coverage: {sorted(needed - set(df.columns))}")


    df[ID_COL] = df[ID_COL].map(_clean_case_id_for_verify)
    df = df.dropna(subset=[ID_COL])

    cs_map: Dict[str, int] = {}
    zone_map: Dict[str, int] = {}
    y3_map: Dict[str, str] = {}
    dropped_blank = 0
    for _, row in df.drop_duplicates(subset=[ID_COL]).iterrows():
        cid = _clean_case_id_for_verify(row[ID_COL])
        if cid is None:
            dropped_blank += 1
            continue
        clin = _parse_clin_for_verify(row[CLIN_COL])
        zone = _parse_zone_for_verify(row[ZONE_COL])
        if clin is None:
            continue
        cs_map[cid] = int(clin)
        if clin == 0:
            y3_map[cid] = "FALSE"
        elif clin == 1 and zone in {"PZ", "TZ"}:
            zone_map[cid] = 1 if zone == "TZ" else 0
            y3_map[cid] = "TRUE_TZ" if zone == "TZ" else "TRUE_PZ"
    if dropped_blank:
        logger.info(f"[COVERAGE] Ignored {dropped_blank} blank/NaN reference case_id rows during strict coverage check")
    return {"cs": cs_map, "zone": zone_map, "y3": y3_map}


def _expected_ids_for_outer_for_verify(
    outer: Dict[str, Any],
    task: str,
    label_maps: Dict[str, Dict[str, Any]],
) -> Tuple[set, set, set]:
    train_ids = [cid for cid in (_clean_case_id_for_verify(x) for x in outer.get("train_case_ids", [])) if cid is not None]
    test_ids = [cid for cid in (_clean_case_id_for_verify(x) for x in outer.get("test_case_ids", [])) if cid is not None]
    task_map = label_maps.get(task, {})
    y3_map = label_maps.get("y3", {})
    expected_train = {cid for cid in train_ids if cid in task_map}
    expected_eval = {cid for cid in test_ids if cid in task_map}
    expected_all_test = {cid for cid in test_ids if cid in y3_map}
    return expected_train, expected_eval, expected_all_test


def validate_feature_workbook_coverage_for_stacking(
    file_path: str,
    file_tag: str,
    split_payload: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    outer_splits = split_payload.get("outer_splits", []) if isinstance(split_payload, dict) else []
    if not outer_splits:
        raise RuntimeError("coverage_check_failed_missing_outer_splits")
    label_maps = _load_reference_label_maps_for_verify(logger)

    try:
        df = _acfg.read_table(file_path, id_col=config["id_col"])
    except Exception as exc:
        raise RuntimeError(f"coverage_check_cannot_read_feature_workbook {file_tag}: {repr(exc)}") from exc
    if config["id_col"] not in df.columns:
        raise RuntimeError(f"coverage_check_missing_id_col {file_tag}:{config['id_col']}")
    got_ids = {cid for cid in (_clean_case_id_for_verify(x) for x in df[config["id_col"]].tolist()) if cid is not None}
    for outer in outer_splits:
        outer_fold = int(outer.get("outer_fold", -1))
        for task in ("cs", "zone"):
            expected_train, expected_eval, expected_all_test = _expected_ids_for_outer_for_verify(outer, task, label_maps)
            for part_name, expected_ids in (
                ("train", expected_train),
                ("eval", expected_eval),
                ("all_test", expected_all_test),
            ):
                missing = sorted(expected_ids - got_ids)
                if missing:
                    raise RuntimeError(
                        f"feature_workbook_incomplete_for_stacking {file_tag} "
                        f"task={task} outer={outer_fold} missing_{part_name}={len(missing)}/{len(expected_ids)} "
                        f"examples={missing[:8]}"
                    )


def run_pipeline(config: Dict):
    base_dir = config["base_dir"]
    results_dir = os.path.join(base_dir, config["results_dir_name"])
    ensure_dir(results_dir)

    log_path = os.path.join(results_dir, "run_log.txt")
    logger = setup_logger(log_path)
    report_txt = os.path.join(results_dir, "ml_top10.txt")
    failed_marker = os.path.join(results_dir, "_ML_FAILED.txt")

    for stale_path in [report_txt, failed_marker]:
        try:
            if os.path.exists(stale_path):
                os.remove(stale_path)
        except Exception:
            pass
    report_notes = []
    report_cs_summaries = []
    report_zone_summaries = []
    report_combo_summaries = []

    logger.info("Pipeline started")

    ref_file = os.path.join(base_dir, config["file_order"][0])
    configured_json = config["json_path"].strip() if config["json_path"] else ""
    default_json = configured_json or os.path.join(
        str(results_dir),
        _acfg.split_json_name(config["random_state"]),
    )
    split_payload, split_used = build_or_load_split_json(
        config=config,
        reference_file=ref_file,
        default_json_path=default_json,
        logger=logger,
    )

    for file_name in config["file_order"]:
        file_path = os.path.join(base_dir, file_name)
        file_tag = os.path.splitext(file_name)[0]

        if not os.path.exists(file_path):
            logger.warning(f"Missing file skipped: {file_path}")
            report_notes.append(f"[SKIP] Missing file: {file_name}")
            continue

        file_out_dir = os.path.join(results_dir, file_tag)
        try:
            validate_feature_workbook_coverage_for_stacking(
                file_path=file_path,
                file_tag=file_tag,
                split_payload=split_payload,
                config=config,
                logger=logger,
            )
            ensure_dir(file_out_dir)

            task_results = {
                "cs": load_existing_task_results(file_out_dir, file_tag, "cs", logger),
                "zone": load_existing_task_results(file_out_dir, file_tag, "zone", logger),
            }

            df = read_feature_workbook(file_path)

            req_cols = [config["id_col"], config["clin_col"], config["zone_col"]]
            for c in req_cols:
                if c not in df.columns:
                    raise RuntimeError(f"{file_name} missing required column: {c}")

            feature_cols = get_feature_columns(df, config)
            if len(feature_cols) == 0:
                logger.warning(f"No features in {file_name}")
                continue

            for task_name in ["cs", "zone"]:
                d_task = prepare_model_data(df, task_name, config)
                if d_task.empty or d_task["y"].nunique() < 2:
                    logger.warning(f"Insufficient data/classes for {file_tag}:{task_name}")
                    continue

                inner_preprocess_cache = {}
                outer_preprocess_cache = {}
                outer_split_cache = {}

                for fs_method in config["enabled_feature_selectors"]:
                    for clf_model in config["enabled_models"]:
                        if task_store_has_completed_combo(
                            task_results[task_name],
                            file_tag=file_tag,
                            fs_method=fs_method,
                            clf_model=clf_model,
                            expected_outer_folds={int(x["outer_fold"]) for x in split_payload["outer_splits"]},
                        ):
                            continue

                        remove_combo_from_task_store(
                            task_results[task_name],
                            file_tag=file_tag,
                            fs_method=fs_method,
                            clf_model=clf_model,
                        )

                        combo_fold_rows = []
                        combo_sel_rows = []
                        combo_pred_rows = []

                        logger.info(
                            f"Running {file_tag}:{task_name}:{fs_method}:{clf_model}"
                        )

                        for outer in split_payload["outer_splits"]:
                            outer_fold = outer["outer_fold"]

                            if outer_fold in outer_split_cache:
                                cached_split = outer_split_cache[outer_fold]
                                if cached_split is None:
                                    continue
                                dtr, dte = cached_split
                            else:
                                tr_ids = set(outer["train_case_ids"])
                                te_ids = set(outer["test_case_ids"])

                                dtr = d_task[d_task["case_id"].isin(tr_ids)].copy()
                                dte = d_task[d_task["case_id"].isin(te_ids)].copy()

                                if dtr.empty or dte.empty or dtr["y"].nunique() < 2 or dte["y"].nunique() < 2:
                                    outer_split_cache[outer_fold] = None
                                    continue

                                outer_split_cache[outer_fold] = (dtr, dte)

                            ctx = f"{file_tag}:{task_name}:{fs_method}:{clf_model}:outer{outer_fold}"

                            best_params, inner_df = run_inner_search_combo(
                                train_df=dtr,
                                feature_cols=feature_cols,
                                inner_splits=outer["inner_splits"],
                                task_name=task_name,
                                fs_method=fs_method,
                                clf_model=clf_model,
                                config=config,
                                logger=logger,
                                context=ctx,
                                prepared_folds_cache=inner_preprocess_cache,
                                prepared_folds_cache_key=outer_fold,
                            )

                            if not inner_df.empty:
                                inner_df.insert(0, "file", file_tag)
                                inner_df.insert(1, "outer_fold", outer_fold)
                                task_results[task_name]["inner_rows"].extend(inner_df.to_dict("records"))

                            if not best_params:
                                logger.warning(f"No valid inner params for {ctx}")
                                continue

                            try:
                                fold_row, feat_df, pred_df = evaluate_outer_fold_combo(
                                    train_df=dtr,
                                    test_df=dte,
                                    feature_cols=feature_cols,
                                    best_params=best_params,
                                    task_name=task_name,
                                    config=config,
                                    outer_fold=outer_fold,
                                    logger=logger,
                                    context=ctx,
                                    outer_base_cache=outer_preprocess_cache,
                                    outer_base_cache_key=outer_fold,
                                )
                            except Exception as e:
                                logger.error(f"Outer fold failed {ctx}: {repr(e)}")
                                logger.error(traceback.format_exc())
                                continue

                            fold_row["file"] = file_tag

                            feat_df.insert(0, "file", file_tag)
                            feat_df.insert(1, "fs_method", fs_method)
                            feat_df.insert(2, "clf_model", clf_model)
                            feat_df.insert(3, "outer_fold", outer_fold)

                            pred_df.insert(0, "file", file_tag)
                            pred_df.insert(1, "fs_method", fs_method)
                            pred_df.insert(2, "clf_model", clf_model)
                            pred_df.insert(3, "outer_fold", outer_fold)

                            combo_fold_rows.append(fold_row)
                            combo_sel_rows.extend(feat_df.to_dict("records"))
                            combo_pred_rows.extend(pred_df.to_dict("records"))

                        combo_fold_df = pd.DataFrame(combo_fold_rows)
                        combo_sel_df = pd.DataFrame(combo_sel_rows)

                        sum_df = summarize_combo_metrics(
                            combo_fold_df,
                            file_tag=file_tag,
                            fs_method=fs_method,
                            clf_model=clf_model,
                        )
                        agg_df = aggregate_selected_features(
                            combo_sel_df,
                            file_tag=file_tag,
                            fs_method=fs_method,
                            clf_model=clf_model,
                        )

                        task_results[task_name]["fold_rows"].extend(combo_fold_rows)
                        task_results[task_name]["sel_rows"].extend(combo_sel_rows)
                        task_results[task_name]["pred_rows"].extend(combo_pred_rows)
                        task_results[task_name]["summary_rows"].extend(sum_df.to_dict("records"))
                        task_results[task_name]["agg_feature_rows"].extend(agg_df.to_dict("records"))

                        build_task_workbooks(task_results[task_name], file_out_dir, file_tag, task_name, logger=logger)

            build_task_workbooks(task_results["cs"], file_out_dir, file_tag, "cs", logger=logger)
            build_task_workbooks(task_results["zone"], file_out_dir, file_tag, "zone", logger=logger)

            expected_outer_folds = {int(x["outer_fold"]) for x in split_payload["outer_splits"]}
            incomplete = []
            incomplete.extend(validate_task_store_complete(task_results["cs"], file_tag, "cs", config, expected_outer_folds))
            incomplete.extend(validate_task_store_complete(task_results["zone"], file_tag, "zone", config, expected_outer_folds))
            if incomplete:
                msg = "Incomplete ML output after resume/build; refusing to write final report as complete. Examples: " + " | ".join(incomplete[:12])
                logger.error(msg)
                report_notes.append(f"[ERROR] {file_name}: {msg}")
                shutil.rmtree(file_out_dir, ignore_errors=True)
                raise RuntimeError(msg)

            cs_summary_full = sort_output_df(pd.DataFrame(task_results["cs"]["summary_rows"]), "combo_summary", final_combo_sort=True)
            zone_summary_full = sort_output_df(pd.DataFrame(task_results["zone"]["summary_rows"]), "combo_summary", final_combo_sort=True)
            combo_summary_full = build_combined_average_workbook(file_out_dir, file_tag, cs_summary_full, zone_summary_full, logger=logger)
            report_cs_summaries.append(filter_summary_to_active_grid(cs_summary_full, config))
            report_zone_summaries.append(filter_summary_to_active_grid(zone_summary_full, config))
            report_combo_summaries.append(filter_summary_to_active_grid(combo_summary_full, config))
            logger.info(f"Completed file: {file_tag}")

        except Exception as e:
            logger.error(f"Error in file {file_name}: {e}")
            logger.error(traceback.format_exc())
            report_notes.append(f"[ERROR] {file_name}: {str(e)}")
            shutil.rmtree(file_out_dir, ignore_errors=True)
            try:
                with open(failed_marker, "w", encoding="utf-8") as f:
                    f.write("ML pipeline failed; partial result directory was removed.\n")
                    f.write(f"file={file_name}\n")
                    f.write(f"error={str(e)}\n")
            except Exception:
                pass
            raise

    write_top10_final_report(
        report_txt=report_txt,
        split_used=split_used,
        cs_summaries=report_cs_summaries,
        zone_summaries=report_zone_summaries,
        combo_summaries=report_combo_summaries,
        notes=report_notes,
    )
    if report_notes:
        logger.error("Pipeline finished with incomplete/error notes; exiting non-zero so z_main will not proceed to stacking/external testing")
        raise RuntimeError("ML pipeline incomplete; see ml_top10.txt and run_log.txt for [ERROR] notes")
    name = config.get("experiment_name")
    logger.info(f"{_acfg.COMPLETION_MARKER} {name} pipeline finished" if name else "Pipeline finished")


from all_config import (
    EARLY_FUSION_WORKBOOK_DIR,
    ORGAN_ONLY_WORKBOOK_DIR,
    ORGAN_SAME_WORKBOOK_DIR,
    PATCH_ONLY_WORKBOOK_DIR,
    DATASET_XLSX,
    ID_COL,
    CLIN_COL,
    ZONE_COL,
)


def _ml_profile_for_base_dir(base_dir: Path) -> dict:
    # All experiments (A_Patch, B_Organ, B_Organ_same, C_Early) train on the same
    # feature files / feature selectors / classifiers grid; base_dir is unused but
    # kept so g_train.py can call this the same way regardless of workbook type.
    return {
        "feature_files": list(FEATURE_FILES),
        "feature_selectors": list(ML_FEATURE_SELECTORS),
        "classifiers": list(ML_CLASSIFIERS),
    }


def _run_one_ml(
    base_dir: Path,
    results_dir: Path,
    seed: int,
    json_path: str | None,
    experiment_name: str | None = None,
) -> None:
    profile = _ml_profile_for_base_dir(base_dir)
    feature_files = list(profile["feature_files"])
    missing = [str(base_dir / name) for name in feature_files if not (base_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Prepare the requested experiment workbooks first, or pass a prepared --base-dir. "
            "Missing workbooks:\n"
            + "\n".join(missing)
        )

    cfg = dict(CONFIG)
    cfg["base_dir"] = str(base_dir)
    cfg["results_dir_name"] = str(results_dir)
    cfg["random_state"] = int(seed)
    cfg["enabled_feature_selectors"] = list(profile["feature_selectors"])
    cfg["enabled_models"] = list(profile["classifiers"])
    cfg["experiment_name"] = experiment_name

    if json_path:
        cfg["json_path"] = str(Path(json_path))
    else:
        cfg["json_path"] = str(Path(results_dir) / _acfg.split_json_name(cfg["random_state"]))

    cfg["file_order"] = feature_files
    run_pipeline(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run nested-CV ML search on patch-helper-augmented main workbooks."
    )
    parser.add_argument("--only", default=_acfg.DEFAULT_INTERNAL_EXPERIMENTS, help="Comma-separated workbook-based experiments to run: A (A_Patch), B (B_Organ), C (C_Early), B0 (B_Organ_same). D (D_Late) is handled by d_fusion, not here.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEEDS[0])
    parser.add_argument("--all-seeds", action="store_true", help="Run for all seeds in RANDOM_SEEDS sequentially.")
    parser.add_argument(
        "--json-path",
        default=None,
        help="Optional exact split JSON path. If not given, uses the experiment's results-dir split file.",
    )
    parser.add_argument("--base-dir", default=None, help="Override workbook directory (used by z_main).")
    parser.add_argument("--results-dir-name", default=None, help="Override results directory (used by z_main).")
    parser.add_argument("--name", default=None, help="Experiment name for the completion log line (used by z_main).")
    args = parser.parse_args()

    if args.base_dir is not None and args.results_dir_name is not None:
        _run_one_ml(
            base_dir=Path(args.base_dir),
            results_dir=Path(args.results_dir_name),
            seed=args.seed,
            json_path=args.json_path,
            experiment_name=args.name,
        )
        return

    requested = [x.strip().upper() for x in args.only.split(",") if x.strip()]
    invalid = sorted(set(requested) - {"A", "B", "C", "B0"})
    if invalid:
        raise ValueError(f"Unknown experiments in --only: {invalid}. Choices: A, B, C, B0")

    seeds = RANDOM_SEEDS if args.all_seeds else [args.seed]
    experiment_names = {"A": "A_Patch", "B": "B_Organ", "C": "C_Early", "B0": "B_Organ_same"}

    for seed in seeds:
        rdirs = _acfg.results_dirs_for_seed(seed)
        configured_parts = {
            "A":  (Path(PATCH_ONLY_WORKBOOK_DIR), rdirs["patch"]),
            "B":  (Path(ORGAN_ONLY_WORKBOOK_DIR), rdirs["organ"]),
            "C":  (Path(EARLY_FUSION_WORKBOOK_DIR), rdirs["early_fusion"]),
            "B0": (Path(ORGAN_SAME_WORKBOOK_DIR), rdirs["organ_same"]),
        }
        for part in requested:
            base_dir, results_dir = configured_parts[part]
            _run_one_ml(
                base_dir=base_dir,
                results_dir=results_dir,
                seed=seed,
                json_path=args.json_path,
                experiment_name=experiment_names[part],
            )


if __name__ == "__main__":
    main()
