import sys
sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import logging
import os
import tempfile
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import all_config as cfg_module
from all_config import (
    D_LATE_DIR,
    EARLY_FUSION_RESULTS_DIR,
    EARLY_FUSION_WORKBOOK_DIR,
    FEATURE_FILES,
    LATE_FUSION_RESULTS_DIR,
    ML_CLASSIFIERS,
    ML_FEATURE_SELECTORS,
    ORGAN_ONLY_RESULTS_DIR,
    ORGAN_ONLY_WORKBOOK_DIR,
    ORGAN_SAME_RESULTS_DIR,
    ORGAN_SAME_WORKBOOK_DIR,
    PATCH_ONLY_RESULTS_DIR,
    PATCH_ONLY_WORKBOOK_DIR,
)
import e_ml as base


TASKS = ("cs", "zone")


CONFIG = dict(base.CONFIG)
CONFIG.update({
    "results_dir": "",
    "previous_results_dir": "",


    "meta_top_k": cfg_module.DEFAULT_STACKING_TOP_K,
    "meta_diversity_max_per_file": 0,
    "meta_diversity_max_per_fs": 0,
    "meta_diversity_max_per_model": 0,
    "meta_std_auc_penalty": 0.0,


    "meta_lr_C_grid": [0.05, 0.1, 0.3, 1.0, 3.0],
    "meta_min_candidates": 1,
    "meta_min_oof_coverage": 1.0,
    "meta_score_transform": "rank",
})


EXPERIMENTS = {
    "A": {
        "name": "A_Patch",
        "workbook_dir": PATCH_ONLY_WORKBOOK_DIR,
        "results_dir": PATCH_ONLY_RESULTS_DIR,
        "feature_files": list(FEATURE_FILES),
        "feature_selectors": list(ML_FEATURE_SELECTORS),
        "classifiers": list(ML_CLASSIFIERS),
        "applicable": True,
        "stack_kind": "single",
    },
    "B": {
        "name": "B_Organ",
        "workbook_dir": ORGAN_ONLY_WORKBOOK_DIR,
        "results_dir": ORGAN_ONLY_RESULTS_DIR,
        "feature_files": list(FEATURE_FILES),
        "feature_selectors": list(ML_FEATURE_SELECTORS),
        "classifiers": list(ML_CLASSIFIERS),
        "applicable": True,
        "stack_kind": "single",
    },
    "B0": {
        "name": "B_Organ_same",
        "workbook_dir": ORGAN_SAME_WORKBOOK_DIR,
        "results_dir": ORGAN_SAME_RESULTS_DIR,
        "feature_files": list(FEATURE_FILES),
        "feature_selectors": list(ML_FEATURE_SELECTORS),
        "classifiers": list(ML_CLASSIFIERS),
        "applicable": True,
        "stack_kind": "single",
    },
    "C": {
        "name": "C_Early",
        "workbook_dir": EARLY_FUSION_WORKBOOK_DIR,
        "results_dir": EARLY_FUSION_RESULTS_DIR,
        "feature_files": list(FEATURE_FILES),
        "feature_selectors": list(ML_FEATURE_SELECTORS),
        "classifiers": list(ML_CLASSIFIERS),
        "applicable": True,
        "stack_kind": "single",
    },
    "D": {
        "name": "D_Late",
        "workbook_dir": D_LATE_DIR,
        "results_dir": LATE_FUSION_RESULTS_DIR,
        "feature_files": list(FEATURE_FILES),
        "feature_selectors": list(ML_FEATURE_SELECTORS),
        "classifiers": list(ML_CLASSIFIERS),
        "applicable": True,
        "stack_kind": "late_fusion",
    },
}


CANDIDATE_COLS = [
    "task",
    "outer_fold",
    "file",
    "fs_method",
    "fs_params",
    "pca_var",
    "clf_model",
    "clf_params",
]


def setup_meta_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("meta_ml_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(sh)

    return logger


def setup_stdout_logger() -> logging.Logger:
    logger = logging.getLogger("meta_ml_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(sh)
    return logger


def normalize_param_json(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return "{}"
    try:
        if pd.isna(value):
            return "{}"
    except TypeError:
        pass
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return "{}"
    try:
        obj = json.loads(text)
    except Exception:
        obj = {}
    return json.dumps(obj, sort_keys=True)


def normalize_pca_var(value: Any) -> Any:
    if value is None:
        return "off"
    try:
        if pd.isna(value):
            return "off"
    except TypeError:
        pass
    if isinstance(value, str):
        s = value.strip()
        if s.lower() in {"", "off", "none", "nan", "false", "null"}:
            return "off"
        try:
            return float(s)
        except ValueError:
            return s
    if isinstance(value, (np.floating, float, np.integer, int)):
        return float(value)
    return value


def pca_key(value: Any) -> str:
    v = normalize_pca_var(value)
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def make_candidate_id(row: Dict[str, Any]) -> str:
    parts = [
        str(row.get("task")),
        str(row.get("outer_fold")),
        str(row.get("file")),
        str(row.get("fs_method")),
        str(row.get("fs_params")),
        pca_key(row.get("pca_var")),
        str(row.get("clf_model")),
        str(row.get("clf_params")),
    ]
    return "||".join(parts)


def resolve_workbook_readonly(path: str) -> Optional[str]:
    if base.is_valid_excel_workbook(path):
        return path
    return None


def canonicalize_fold_metrics(df: pd.DataFrame, task: str, file_tag: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "file" not in out.columns:
        out["file"] = file_tag
    out["task"] = task
    out["file"] = out["file"].fillna(file_tag).astype(str)
    out["outer_fold"] = pd.to_numeric(out["outer_fold"], errors="coerce").astype("Int64")
    out["fs_params"] = out["fs_params"].apply(normalize_param_json)
    out["clf_params"] = out["clf_params"].apply(normalize_param_json)
    out["pca_var"] = out["pca_var"].apply(normalize_pca_var)
    out["pca_key"] = out["pca_var"].apply(pca_key)

    for col in ["inner_mean_auc", "inner_mean_ap", "inner_std_auc", "inner_scores_used"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["outer_fold", "inner_mean_auc"]).copy()
    out["outer_fold"] = out["outer_fold"].astype(int)
    out["candidate_id"] = out.apply(lambda r: make_candidate_id(r.to_dict()), axis=1)
    return out


def load_candidate_cache(config: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    previous_root = config["previous_results_dir"]
    rows = []
    active_selectors = {str(x) for x in config.get("enabled_feature_selectors", [])}
    active_models = {str(x) for x in config.get("enabled_models", [])}

    for file_name in config["file_order"]:
        file_tag = os.path.splitext(file_name)[0]
        file_dir = os.path.join(previous_root, file_tag)

        for task in TASKS:
            book = os.path.join(file_dir, f"results_{task}_{file_tag}.xlsx")
            resolved = resolve_workbook_readonly(book)
            if resolved is None:
                logger.debug(f"[CACHE] Missing/unreadable workbook skipped: {book}")
                continue
            try:
                df = pd.read_excel(resolved, sheet_name="fold_metrics")
                df = canonicalize_fold_metrics(df, task=task, file_tag=file_tag)
                if not df.empty:
                    rows.append(df)
                    logger.debug(f"[CACHE] Loaded {len(df)} fold rows from {resolved}")
            except Exception as e:
                logger.debug(f"[CACHE] Failed reading {resolved}: {repr(e)}")

    if not rows:
        return pd.DataFrame()

    cache = pd.concat(rows, ignore_index=True)
    if active_selectors and "fs_method" in cache.columns:
        cache = cache[cache["fs_method"].astype(str).isin(active_selectors)].copy()
    if active_models and "clf_model" in cache.columns:
        cache = cache[cache["clf_model"].astype(str).isin(active_models)].copy()
    if cache.empty:
        return cache

    sort_cols = ["task", "outer_fold", "inner_mean_auc", "inner_mean_ap", "inner_std_auc"]
    cache = cache.sort_values(
        sort_cols,
        ascending=[True, True, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    return cache


def select_cached_candidates(
    cache_df: pd.DataFrame,
    task: str,
    outer_fold: int,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    subset = cache_df[
        (cache_df["task"] == task)
        & (cache_df["outer_fold"] == int(outer_fold))
    ].copy()

    if subset.empty:
        return []

    subset = subset.drop_duplicates(subset=["candidate_id"], keep="first")
    std_for_penalty = pd.to_numeric(subset["inner_std_auc"], errors="coerce")
    if std_for_penalty.notna().any():
        std_for_penalty = std_for_penalty.fillna(float(std_for_penalty.max()))
    else:
        std_for_penalty = std_for_penalty.fillna(0.0)
    subset["inner_selection_score"] = (
        pd.to_numeric(subset["inner_mean_auc"], errors="coerce").fillna(-np.inf)
        - float(config.get("meta_std_auc_penalty", 0.0)) * std_for_penalty
    )
    subset = subset.sort_values(
        [
            "inner_selection_score",
            "inner_mean_auc",
            "inner_mean_ap",
            "inner_std_auc",
            "file",
            "clf_model",
        ],
        ascending=[False, False, False, True, True, True],
        kind="stable",
    )

    top_k = int(config["meta_top_k"])
    max_per_file = int(config.get("meta_diversity_max_per_file", 2))
    max_per_fs   = int(config.get("meta_diversity_max_per_fs", 3))
    max_per_clf  = int(config.get("meta_diversity_max_per_model", 3))

    selected = []
    used_ids = set()
    counts_file = defaultdict(int)
    counts_fs   = defaultdict(int)
    counts_clf  = defaultdict(int)

    for _, row in subset.iterrows():
        file_key = str(row.get("file", ""))
        fs_key   = str(row.get("fs_method", ""))
        clf_key  = str(row.get("clf_model", ""))
        if max_per_file > 0 and counts_file[file_key] >= max_per_file:
            continue
        if max_per_fs > 0 and counts_fs[fs_key] >= max_per_fs:
            continue
        if max_per_clf > 0 and counts_clf[clf_key] >= max_per_clf:
            continue
        rec = row.to_dict()
        selected.append(rec)
        used_ids.add(rec["candidate_id"])
        counts_file[file_key] += 1
        counts_fs[fs_key]     += 1
        counts_clf[clf_key]   += 1
        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        for _, row in subset.iterrows():
            rec = row.to_dict()
            if rec["candidate_id"] in used_ids:
                continue
            selected.append(rec)
            used_ids.add(rec["candidate_id"])
            if len(selected) >= top_k:
                break

    for rank, rec in enumerate(selected, start=1):
        rec["candidate_rank"] = rank
    return selected


def tag_candidate_source(candidate: Dict[str, Any], source: str, base_dir: Path) -> Dict[str, Any]:
    out = dict(candidate)
    out["source"] = source
    out["_base_dir"] = str(base_dir)
    out["candidate_id"] = f"{source}::{out['candidate_id']}"
    return out


def select_late_fusion_candidates(
    cache_a: pd.DataFrame,
    cache_b: pd.DataFrame,
    task: str,
    outer_fold: int,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:

    def _tagged(cache: pd.DataFrame, source: str, base_dir: Path) -> pd.DataFrame:
        sub = cache[(cache["task"] == task) & (cache["outer_fold"] == int(outer_fold))].copy()
        if sub.empty:
            return sub
        sub = sub.drop_duplicates(subset=["candidate_id"], keep="first")
        sub["source"] = source
        sub["_base_dir"] = str(base_dir)
        sub["candidate_id"] = source + "::" + sub["candidate_id"].astype(str)

        sub["_file_key"] = source + "::" + sub["file"].astype(str)
        return sub

    pool = pd.concat([
        _tagged(cache_a, "A_organ", ORGAN_ONLY_WORKBOOK_DIR),
        _tagged(cache_b, "B_patch", PATCH_ONLY_WORKBOOK_DIR),
    ], ignore_index=True)

    if pool.empty:
        return []

    std_for_penalty = pd.to_numeric(pool["inner_std_auc"], errors="coerce")
    if std_for_penalty.notna().any():
        std_for_penalty = std_for_penalty.fillna(float(std_for_penalty.max()))
    else:
        std_for_penalty = std_for_penalty.fillna(0.0)
    pool["inner_selection_score"] = (
        pd.to_numeric(pool["inner_mean_auc"], errors="coerce").fillna(-np.inf)
        - float(config.get("meta_std_auc_penalty", 0.0)) * std_for_penalty
    )
    pool = pool.sort_values(
        ["inner_selection_score", "inner_mean_auc", "inner_mean_ap", "inner_std_auc", "file", "clf_model"],
        ascending=[False, False, False, True, True, True],
        kind="stable",
    )

    top_k       = int(config["meta_top_k"])
    max_per_file = int(config.get("meta_diversity_max_per_file", 2))
    max_per_fs   = int(config.get("meta_diversity_max_per_fs", 3))
    max_per_clf  = int(config.get("meta_diversity_max_per_model", 3))

    selected = []
    used_ids = set()
    counts_file = defaultdict(int)
    counts_fs   = defaultdict(int)
    counts_clf  = defaultdict(int)

    for _, row in pool.iterrows():
        file_key = str(row.get("_file_key", row.get("file", "")))
        fs_key   = str(row.get("fs_method", ""))
        clf_key  = str(row.get("clf_model", ""))
        cid      = str(row["candidate_id"])
        if cid in used_ids:
            continue
        if max_per_file > 0 and counts_file[file_key] >= max_per_file:
            continue
        if max_per_fs > 0 and counts_fs[fs_key] >= max_per_fs:
            continue
        if max_per_clf > 0 and counts_clf[clf_key] >= max_per_clf:
            continue
        selected.append(row.to_dict())
        used_ids.add(cid)
        counts_file[file_key] += 1
        counts_fs[fs_key]     += 1
        counts_clf[clf_key]   += 1
        if len(selected) >= top_k:
            break


    if len(selected) < top_k:
        for _, row in pool.iterrows():
            cid = str(row["candidate_id"])
            if cid in used_ids:
                continue
            selected.append(row.to_dict())
            used_ids.add(cid)
            if len(selected) >= top_k:
                break

    for rank, cand in enumerate(selected, start=1):
        cand["candidate_rank"] = rank
    return selected


def build_label_maps(reference_file: str, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    df = base.read_feature_workbook(reference_file)
    df = df.drop_duplicates(subset=[config["id_col"]]).copy()

    cs_map = {}
    zone_map = {}

    for _, row in df.iterrows():
        cid = _clean_case_id(row[config["id_col"]])
        if cid is None:
            continue
        clin = base.parse_clin(row[config["clin_col"]])
        zone = base.parse_zone(row[config["zone_col"]])

        if not pd.isna(clin):
            cs_map[cid] = int(clin)
        if clin == 1 and zone in {"PZ", "TZ"}:
            zone_map[cid] = 1 if zone == "TZ" else 0

    return {"cs": cs_map, "zone": zone_map}


def get_inner_fold_by_case(outer: Dict[str, Any]) -> Dict[str, int]:
    out = {}
    for inner in outer["inner_splits"]:
        fold = int(inner["inner_fold"])
        for raw_cid in inner.get("val_case_ids", []):
            cid = _clean_case_id(raw_cid)
            if cid is not None:
                out[cid] = fold
    return out


def ordered_ids(ids: List[str], allowed_map: Dict[str, Any]) -> List[str]:
    out = []
    for raw_cid in ids:
        cid = _clean_case_id(raw_cid)
        if cid is not None and cid in allowed_map:
            out.append(cid)
    return out


def _clean_case_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>", "na"}:
        return None
    return text


def validate_inner_oof_split_coverage(
    split_payload: Dict[str, Any],
    label_maps: Dict[str, Dict[str, Any]],
) -> List[str]:
    errors: List[str] = []
    for outer in split_payload.get("outer_splits", []):
        outer_fold = int(outer.get("outer_fold", -1))
        inner_val_union = {
            cid
            for inner in outer.get("inner_splits", [])
            for cid in (_clean_case_id(x) for x in inner.get("val_case_ids", []))
            if cid is not None
        }
        outer_train_all = {
            cid for cid in (_clean_case_id(x) for x in outer.get("train_case_ids", [])) if cid is not None
        }
        extra_inner = sorted(inner_val_union - outer_train_all)
        if extra_inner:
            errors.append(
                f"[SPLIT OOF COVERAGE FAIL] outer={outer_fold} inner validation contains "
                f"{len(extra_inner)} IDs not present in outer train; examples={extra_inner[:8]}"
            )
        for task in TASKS:
            task_map = label_maps.get(task, {})
            expected_train = {cid for cid in outer_train_all if cid in task_map}
            covered_train = {cid for cid in inner_val_union if cid in task_map}
            missing = sorted(expected_train - covered_train)
            if missing:
                errors.append(
                    f"[SPLIT OOF COVERAGE FAIL] task={task} outer={outer_fold} "
                    f"inner_val_union missing {len(missing)}/{len(expected_train)} outer-train IDs; "
                    f"examples={missing[:8]}"
                )
    return errors


def fit_fixed_candidate(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    candidate: Dict[str, Any],
    task_name: str,
    config: Dict[str, Any],
    logger: logging.Logger,
    context: str,
) -> Dict[str, Any]:
    if train_df.empty:
        raise RuntimeError(f"{context} empty training data")
    if train_df["y"].nunique() < 2:
        raise RuntimeError(f"{context} training data has one class")

    Xtr_df = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    ytr = train_df["y"].values.astype(int)

    prep = base.TrainPreprocessor(
        quasi_thresh=config["quasi_constant_threshold"],
        corr_thresh=config["corr_threshold"],
        scaler_type=config["scaler"],
    )
    prep.fit(Xtr_df)
    Xtr_base, base_names = prep.transform(Xtr_df)

    fs_method = str(candidate["fs_method"])
    fs_params = json.loads(candidate["fs_params"])
    clf_model = str(candidate["clf_model"])
    clf_params = json.loads(candidate["clf_params"])
    pca_var = normalize_pca_var(candidate["pca_var"])
    class_weight = base.get_class_weight(task_name, config)

    Xtr_sel, _, sel_names, sel_scores = base.select_features(
        method=fs_method,
        X_train=Xtr_base,
        y_train=ytr,
        X_apply=Xtr_base,
        feature_names=base_names,
        params=fs_params,
        class_weight=class_weight,
        config=config,
        logger=logger,
        context=context,
    )

    name_to_idx = {name: idx for idx, name in enumerate(base_names)}
    sel_idx = [name_to_idx[name] for name in sel_names]

    Xtr_final, _, pca, pca_components, pca_explained = base.apply_optional_pca(
        Xtr_sel,
        Xtr_sel,
        pca_var,
    )

    clf = base.make_estimator(clf_model, clf_params, task_name, config)
    clf.fit(Xtr_final, ytr)

    return {
        "prep": prep,
        "feature_cols": list(feature_cols),
        "base_names": base_names,
        "sel_idx": sel_idx,
        "sel_names": sel_names,
        "sel_scores": sel_scores,
        "pca": pca,
        "pca_components": pca_components,
        "pca_explained_var_sum": pca_explained,
        "clf": clf,
        "score_source": None,
        "n_input_features": prep.stats.get("n_input_features", np.nan),
        "n_after_quasi": prep.stats.get("n_after_quasi", np.nan),
        "n_after_corr": prep.stats.get("n_after_corr", np.nan),
        "n_selected": int(len(sel_names)),
    }


def predict_fixed_candidate(
    state: Dict[str, Any],
    predict_df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    id_col = config["id_col"]
    if predict_df.empty:
        return pd.DataFrame(columns=["case_id", "raw_score", "raw_pred", "score_source"])

    Xdf = predict_df[state["feature_cols"]].apply(pd.to_numeric, errors="coerce")
    Xbase, _ = state["prep"].transform(Xdf)
    Xsel = Xbase[:, state["sel_idx"]]

    if state["pca"] is not None:
        Xfinal = state["pca"].transform(Xsel)
    else:
        Xfinal = Xsel

    y_score, y_pred, score_source = base.get_scores_and_preds(state["clf"], Xfinal)
    pred_ids = [_clean_case_id(x) for x in predict_df[id_col].tolist()]
    return pd.DataFrame({
        "case_id": pred_ids,
        "raw_score": y_score.astype(float),
        "raw_pred": y_pred.astype(int),
        "score_source": score_source,
    }).dropna(subset=["case_id"])


def load_candidate_frames(
    candidate: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    candidate_base_dir = candidate.get("_base_dir", config["base_dir"])
    file_path = os.path.join(candidate_base_dir, f"{candidate['file']}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    df = base.read_feature_workbook(file_path)
    id_col = config["id_col"]
    if id_col not in df.columns:
        raise RuntimeError(f"{file_path} missing {id_col}")
    df[id_col] = df[id_col].apply(_clean_case_id)
    df = df.dropna(subset=[id_col]).copy()
    dup = df[id_col].duplicated(keep=False)
    if dup.any():
        examples = sorted(df.loc[dup, id_col].astype(str).unique().tolist())[:10]
        raise RuntimeError(
            f"{file_path} has duplicate case_id rows after normalization; "
            f"examples={examples}"
        )
    feature_cols = base.get_feature_columns(df, config)
    if not feature_cols:
        raise RuntimeError(f"{file_path} has no features")

    task_df = base.prepare_model_data(df, str(candidate["task"]), config)
    return task_df, feature_cols


def recompute_candidate_predictions(
    candidate: Dict[str, Any],
    outer: Dict[str, Any],
    task_name: str,
    train_ids: List[str],
    eval_ids: List[str],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    task_df, feature_cols = load_candidate_frames(candidate, config)
    id_col = config["id_col"]
    candidate_id = candidate["candidate_id"]

    oof_parts = []
    for inner in outer["inner_splits"]:
        inner_fold = int(inner["inner_fold"])
        inner_train_ids = {cid for cid in (_clean_case_id(x) for x in inner.get("train_case_ids", [])) if cid is not None}
        inner_val_ids = {cid for cid in (_clean_case_id(x) for x in inner.get("val_case_ids", [])) if cid is not None}

        dtr = task_df[task_df[id_col].isin(inner_train_ids)].copy()
        dva = task_df[task_df[id_col].isin(inner_val_ids)].copy()
        if dtr.empty or dva.empty or dtr["y"].nunique() < 2:
            logger.warning(f"[OOF SKIP] {candidate_id} inner={inner_fold} insufficient data")
            continue

        state = fit_fixed_candidate(
            dtr,
            feature_cols,
            candidate,
            task_name,
            config,
            logger,
            context=f"{candidate_id}:inner{inner_fold}",
        )
        pred = predict_fixed_candidate(state, dva, config)
        pred["y_true"] = pred["case_id"].map(dict(zip(dva[id_col].apply(_clean_case_id), dva["y"].astype(int))))
        pred["inner_fold"] = inner_fold
        oof_parts.append(pred)

    oof_df = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()

    train_set = set(train_ids)
    eval_set = set(eval_ids)

    dtr_outer = task_df[task_df[id_col].isin(train_set)].copy()
    dte_eval = task_df[task_df[id_col].isin(eval_set)].copy()

    state = fit_fixed_candidate(
        dtr_outer,
        feature_cols,
        candidate,
        task_name,
        config,
        logger,
        context=f"{candidate_id}:outer{outer['outer_fold']}",
    )
    eval_df = predict_fixed_candidate(state, dte_eval, config)
    if not eval_df.empty:
        eval_df["y_true"] = eval_df["case_id"].map(dict(zip(dte_eval[id_col].apply(_clean_case_id), dte_eval["y"].astype(int))))

    return {
        "candidate": candidate,
        "candidate_id": candidate_id,
        "oof_df": oof_df,
        "eval_df": eval_df,
        "outer_state": state,
        "state_summary": {
            "score_source": None,
            "n_input_features": state["n_input_features"],
            "n_after_quasi": state["n_after_quasi"],
            "n_after_corr": state["n_after_corr"],
            "n_selected": state["n_selected"],
            "pca_components": state["pca_components"],
            "pca_explained_var_sum": state["pca_explained_var_sum"],
        },
    }


def safe_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            "auc": np.nan,
            "ap": np.nan,
            "balanced_acc": np.nan,
            "f1": np.nan,
            "accuracy": np.nan,
            "sensitivity": np.nan,
            "specificity": np.nan,
        }
    return base.compute_metrics(y_true.astype(int), y_score.astype(float), y_pred.astype(int))


def score_to_pred(score: np.ndarray) -> np.ndarray:
    return (np.asarray(score, dtype=float) >= 0.5).astype(int)


def series_from_pred_df(df: pd.DataFrame, ids: List[str], col: str) -> pd.Series:
    norm_ids = [cid for cid in (_clean_case_id(x) for x in ids) if cid is not None]
    if df is None or df.empty:
        return pd.Series(index=norm_ids, dtype=float)
    work = df.copy()
    work["case_id"] = work["case_id"].apply(_clean_case_id)
    work = work.dropna(subset=["case_id"]).drop_duplicates(subset=["case_id"])
    s = work.set_index("case_id")[col]
    return s.reindex(norm_ids)


def rank_normalize_matrices(
    z_oof: np.ndarray,
    z_eval: np.ndarray,
    z_all: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_oof_t = np.zeros_like(z_oof, dtype=float)
    z_eval_t = np.zeros_like(z_eval, dtype=float)
    z_all_t = np.zeros_like(z_all, dtype=float)

    for j in range(z_oof.shape[1]):
        ref = np.asarray(z_oof[:, j], dtype=float)
        if len(np.unique(ref)) <= 1:
            z_oof_t[:, j] = 0.5
            if z_eval_t.shape[0]:
                z_eval_t[:, j] = 0.5
            if z_all_t.shape[0]:
                z_all_t[:, j] = 0.5
            continue

        ref_sorted = np.sort(ref)
        denom = float(len(ref_sorted))

        def transform(values: np.ndarray) -> np.ndarray:
            values = np.asarray(values, dtype=float)
            ranks = np.searchsorted(ref_sorted, values, side="right") / denom
            eps = 0.5 / denom
            return np.clip(ranks, eps, 1.0 - eps)

        z_oof_t[:, j] = transform(z_oof[:, j])
        if z_eval_t.shape[0]:
            z_eval_t[:, j] = transform(z_eval[:, j])
        if z_all_t.shape[0]:
            z_all_t[:, j] = transform(z_all[:, j])

    return z_oof_t, z_eval_t, z_all_t


def assemble_meta_matrices(
    candidate_results: List[Dict[str, Any]],
    train_ids: List[str],
    eval_ids: List[str],
    y_train: np.ndarray,
    y_eval: np.ndarray,
    task_name: str,
    outer_fold: int,
    config: Dict[str, Any],
    all_test_ids: List[str] = [],
) -> Dict[str, Any]:
    kept = []
    selected_status = {}
    base_oof_rows = []
    base_outer_rows = []

    min_cov = float(config["meta_min_oof_coverage"])
    for res in candidate_results:
        cand = res["candidate"]
        cid = res["candidate_id"]

        oof_score = series_from_pred_df(res["oof_df"], train_ids, "raw_score")
        oof_pred = series_from_pred_df(res["oof_df"], train_ids, "raw_pred")
        eval_score = series_from_pred_df(res["eval_df"], eval_ids, "raw_score")
        eval_pred = series_from_pred_df(res["eval_df"], eval_ids, "raw_pred")
        all_score = series_from_pred_df(res.get("all_df", pd.DataFrame()), all_test_ids, "raw_score") if all_test_ids else pd.Series([], dtype=float)

        oof_coverage = float(oof_score.notna().mean()) if len(train_ids) else 0.0
        missing_eval = int(eval_score.isna().sum())
        missing_all = int(all_score.isna().sum()) if all_test_ids else 0

        if oof_coverage < min_cov or missing_eval > 0 or missing_all > 0:
            missing_oof_ids = oof_score[oof_score.isna()].index.astype(str).tolist()[:8]
            missing_eval_ids = eval_score[eval_score.isna()].index.astype(str).tolist()[:8]
            missing_all_ids = all_score[all_score.isna()].index.astype(str).tolist()[:8] if all_test_ids else []
            selected_status[cid] = (
                f"dropped_missing_predictions oof_coverage={oof_coverage:.3f} "
                f"missing_oof={int(oof_score.isna().sum())} missing_oof_examples={missing_oof_ids} "
                f"missing_eval={missing_eval} missing_eval_examples={missing_eval_ids} "
                f"missing_all={missing_all} missing_all_examples={missing_all_ids}"
            )
            continue

        oof_score_arr = oof_score.astype(float).values
        oof_pred_values = oof_pred.astype(float).values
        oof_pred_arr = np.where(
            np.isnan(oof_pred_values),
            score_to_pred(oof_score_arr),
            oof_pred_values,
        ).astype(int)

        eval_score_arr = eval_score.astype(float).values
        eval_pred_values = eval_pred.astype(float).values
        eval_pred_arr = np.where(
            np.isnan(eval_pred_values),
            score_to_pred(eval_score_arr),
            eval_pred_values,
        ).astype(int)

        oof_metrics = safe_binary_metrics(y_train, oof_score_arr, oof_pred_arr)
        eval_metrics = safe_binary_metrics(y_eval, eval_score_arr, eval_pred_arr)

        common = {
            "source": cand.get("source", ""),
            "task": task_name,
            "outer_fold": outer_fold,
            "candidate_rank": cand["candidate_rank"],
            "candidate_id": cid,
            "file": cand["file"],
            "fs_method": cand["fs_method"],
            "fs_params": cand["fs_params"],
            "pca_var": cand["pca_var"],
            "clf_model": cand["clf_model"],
            "clf_params": cand["clf_params"],
            "cache_inner_mean_auc": cand.get("inner_mean_auc", np.nan),
            "cache_inner_mean_ap": cand.get("inner_mean_ap", np.nan),
            "cache_inner_std_auc": cand.get("inner_std_auc", np.nan),
            "cache_inner_selection_score": cand.get("inner_selection_score", np.nan),
        }

        base_oof_rows.append({
            **common,
            "n_oof": int(len(y_train)),
            **{f"oof_{k}": v for k, v in oof_metrics.items()},
        })
        base_outer_rows.append({
            **common,
            "n_test": int(len(y_eval)),
            **{f"outer_{k}": v for k, v in eval_metrics.items()},
        })

        all_score_arr = all_score.astype(float).values if all_test_ids else np.array([], dtype=float)

        selected_status[cid] = "used"
        kept.append({
            **res,
            "oof_score": oof_score_arr,
            "eval_score": eval_score_arr,
            "all_score": all_score_arr,
            "oof_auc": oof_metrics["auc"],
            "oof_ap": oof_metrics["ap"],
        })

    if len(kept) < int(config["meta_min_candidates"]):
        return {
            "ok": False,
            "reason": f"not_enough_usable_candidates kept={len(kept)}",
            "selected_status": selected_status,
            "base_oof_rows": base_oof_rows,
            "base_outer_rows": base_outer_rows,
        }

    z_oof = np.column_stack([res["oof_score"] for res in kept])
    z_eval = np.column_stack([res["eval_score"] for res in kept]) if len(eval_ids) else np.empty((0, len(kept)))
    z_all = np.column_stack([res["all_score"] for res in kept]) if all_test_ids else np.empty((0, len(kept)))

    if config.get("meta_score_transform", "rank") == "rank":
        z_oof, z_eval, z_all = rank_normalize_matrices(z_oof, z_eval, z_all)

    return {
        "ok": True,
        "kept": kept,
        "z_oof": z_oof,
        "z_eval": z_eval,
        "z_all": z_all,
        "selected_status": selected_status,
        "base_oof_rows": base_oof_rows,
        "base_outer_rows": base_outer_rows,
    }


def metric_rank(metrics: Dict[str, float], complexity: int) -> Tuple[float, float, int]:
    auc = metrics.get("auc", np.nan)
    ap = metrics.get("ap", np.nan)
    auc_rank = float(auc) if not pd.isna(auc) else -np.inf
    ap_rank = float(ap) if not pd.isna(ap) else -np.inf
    return auc_rank, ap_rank, -int(complexity)


def fit_final_meta_lr(
    z_oof: np.ndarray,
    y_train: np.ndarray,
    z_eval: np.ndarray,
    z_all: np.ndarray,
    C: float,
    task_name: str,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    ztr = scaler.fit_transform(z_oof)
    clf = LogisticRegression(
        C=float(C),
        solver="lbfgs",
        class_weight=base.get_class_weight(task_name, config),
        max_iter=config["max_iter"],
        tol=config["tol"],
        random_state=config["random_state"],
    )
    clf.fit(ztr, y_train.astype(int))

    eval_score = (
        clf.predict_proba(scaler.transform(z_eval))[:, 1]
        if z_eval.shape[0]
        else np.array([], dtype=float)
    )
    all_score = (
        clf.predict_proba(scaler.transform(z_all))[:, 1]
        if z_all.shape[0]
        else np.array([], dtype=float)
    )
    return eval_score, all_score


def meta_lr_cv_oof(
    z_oof: np.ndarray,
    y_train: np.ndarray,
    fold_ids: np.ndarray,
    C: float,
    task_name: str,
    config: Dict[str, Any],
) -> Optional[np.ndarray]:
    pred = np.full(len(y_train), np.nan, dtype=float)
    for fold in sorted(set(fold_ids.tolist())):
        va = fold_ids == fold
        tr = ~va
        if np.sum(va) == 0 or len(np.unique(y_train[tr])) < 2:
            return None
        scaler = StandardScaler()
        ztr = scaler.fit_transform(z_oof[tr])
        clf = LogisticRegression(
            C=float(C),
            solver="lbfgs",
            class_weight=base.get_class_weight(task_name, config),
            max_iter=config["max_iter"],
            tol=config["tol"],
            random_state=config["random_state"],
        )
        clf.fit(ztr, y_train[tr].astype(int))
        pred[va] = clf.predict_proba(scaler.transform(z_oof[va]))[:, 1]

    if np.any(np.isnan(pred)):
        return None
    return pred


def choose_and_fit_meta(
    z_oof: np.ndarray,
    y_train: np.ndarray,
    fold_ids: np.ndarray,
    z_eval: np.ndarray,
    y_eval: np.ndarray,
    z_all: np.ndarray,
    kept: List[Dict[str, Any]],
    task_name: str,
    outer_fold: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    method_rows = []
    choices = []

    def add_choice(method: str, score_oof: np.ndarray, score_eval: np.ndarray, score_all: np.ndarray, complexity: int, extra: Dict[str, Any]):
        pred_oof = score_to_pred(score_oof)
        pred_eval = score_to_pred(score_eval) if len(score_eval) else np.array([], dtype=int)
        oof_metrics = safe_binary_metrics(y_train, score_oof, pred_oof)
        eval_metrics = safe_binary_metrics(y_eval, score_eval, pred_eval)
        row = {
            "task": task_name,
            "outer_fold": outer_fold,
            "meta_method": method,
            "complexity": complexity,
            "n_meta_features": int(z_oof.shape[1]),
            **extra,
            **{f"meta_inner_{k}": v for k, v in oof_metrics.items()},
            **{f"outer_{k}": v for k, v in eval_metrics.items()},
        }
        method_rows.append(row)
        choices.append({
            "method": method,
            "rank": metric_rank(oof_metrics, complexity),
            "score_eval": score_eval,
            "score_all": score_all,
            "oof_metrics": oof_metrics,
            "eval_metrics": eval_metrics,
            "extra": extra,
        })


    best_j = None
    best_rank = None
    for j, res in enumerate(kept):
        score = z_oof[:, j]
        mets = safe_binary_metrics(y_train, score, score_to_pred(score))
        r = metric_rank(mets, complexity=0)
        if best_rank is None or r > best_rank:
            best_rank = r
            best_j = j
    if best_j is not None:
        add_choice(
            "single_best",
            z_oof[:, best_j],
            z_eval[:, best_j] if z_eval.shape[0] else np.array([], dtype=float),
            z_all[:, best_j] if z_all.shape[0] else np.array([], dtype=float),
            complexity=0,
            extra={
                "selected_base_candidate_id": kept[best_j]["candidate_id"],
                "meta_C": np.nan,
            },
        )

    add_choice(
        "simple_average",
        np.mean(z_oof, axis=1),
        np.mean(z_eval, axis=1) if z_eval.shape[0] else np.array([], dtype=float),
        np.mean(z_all, axis=1) if z_all.shape[0] else np.array([], dtype=float),
        complexity=1,
        extra={"selected_base_candidate_id": "", "meta_C": np.nan},
    )

    weights = np.array([
        max(float(res.get("oof_auc", np.nan)) - 0.5, 0.001)
        if not pd.isna(res.get("oof_auc", np.nan))
        else 0.001
        for res in kept
    ], dtype=float)
    weights = weights / weights.sum()
    add_choice(
        "weighted_average",
        np.average(z_oof, axis=1, weights=weights),
        np.average(z_eval, axis=1, weights=weights) if z_eval.shape[0] else np.array([], dtype=float),
        np.average(z_all, axis=1, weights=weights) if z_all.shape[0] else np.array([], dtype=float),
        complexity=1,
        extra={
            "selected_base_candidate_id": "",
            "meta_C": np.nan,
            "weights_json": json.dumps({kept[i]["candidate_id"]: round(float(weights[i]), 4) for i in range(len(kept))}, sort_keys=True),
        },
    )

    for C in config["meta_lr_C_grid"]:
        score_oof_cv = meta_lr_cv_oof(z_oof, y_train, fold_ids, C, task_name, config)
        if score_oof_cv is None:
            continue
        score_eval, score_all = fit_final_meta_lr(z_oof, y_train, z_eval, z_all, C, task_name, config)
        add_choice(
            f"logreg_stack_C={C}",
            score_oof_cv,
            score_eval,
            score_all,
            complexity=2,
            extra={"selected_base_candidate_id": "", "meta_C": float(C)},
        )

    if not choices:
        raise RuntimeError("No valid meta choices")

    chosen = max(choices, key=lambda x: x["rank"])
    return {
        "chosen": chosen,
        "method_rows": method_rows,
        "all_choices": choices,
    }


def validate_candidate_inputs_for_stacking(
    task_name: str,
    outer: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    train_ids: List[str],
    eval_ids: List[str],
    config: Dict[str, Any],
) -> List[str]:
    errors: List[str] = []
    train_set = set(train_ids)
    eval_set = set(eval_ids)
    id_col = config["id_col"]

    for cand in candidates:
        cid = cand.get("candidate_id", "<unknown_candidate>")
        try:
            task_df, _feature_cols = load_candidate_frames(cand, config)
        except Exception as exc:
            errors.append(f"{cid} input_load_failed={repr(exc)}")
            continue

        task_ids = {x for x in ( _clean_case_id(v) for v in task_df[id_col].tolist() ) if x is not None}

        missing_train = sorted(train_set - task_ids)
        missing_eval = sorted(eval_set - task_ids)
        if missing_train or missing_eval:
            errors.append(
                f"{cid} candidate_input_coverage_failed "
                f"missing_train={len(missing_train)}/{len(train_set)} examples={missing_train[:8]} "
                f"missing_eval={len(missing_eval)}/{len(eval_set)} examples={missing_eval[:8]}"
            )
            continue

        for inner in outer.get("inner_splits", []):
            inner_fold = int(inner.get("inner_fold", -1))
            inner_train_ids = {x for x in (_clean_case_id(v) for v in inner.get("train_case_ids", [])) if x is not None}
            inner_val_ids = {x for x in (_clean_case_id(v) for v in inner.get("val_case_ids", [])) if x is not None}
            expected_val = sorted((inner_val_ids & train_set) - task_ids)
            if expected_val:
                errors.append(
                    f"{cid} inner={inner_fold} inner_val_missing_from_task_df "
                    f"count={len(expected_val)} examples={expected_val[:8]}"
                )
                continue
            dtr = task_df[task_df[id_col].isin(inner_train_ids)].copy()
            dva = task_df[task_df[id_col].isin(inner_val_ids)].copy()
            if dtr.empty or dva.empty or dtr["y"].nunique() < 2:
                errors.append(
                    f"{cid} inner={inner_fold} cannot_make_oof "
                    f"n_train={len(dtr)} n_val={len(dva)} n_train_classes={int(dtr['y'].nunique()) if 'y' in dtr.columns else 0}"
                )
                continue
    return errors


def run_task_outer_meta(
    task_name: str,
    outer: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    label_maps: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    outer_fold = int(outer["outer_fold"])
    train_ids = ordered_ids(outer["train_case_ids"], label_maps[task_name])
    eval_ids = ordered_ids(outer["test_case_ids"], label_maps[task_name])
    inner_fold_map = get_inner_fold_by_case(outer)

    y_train = np.array([label_maps[task_name][cid] for cid in train_ids], dtype=int)
    y_eval = np.array([label_maps[task_name][cid] for cid in eval_ids], dtype=int)
    fold_ids = np.array([inner_fold_map[cid] for cid in train_ids], dtype=int)

    if len(train_ids) == 0 or len(np.unique(y_train)) < 2:
        raise RuntimeError(f"{task_name} outer {outer_fold} has insufficient meta training data")

    input_errors = validate_candidate_inputs_for_stacking(
        task_name,
        outer,
        candidates,
        train_ids,
        eval_ids,
        config,
    )
    if input_errors:
        detail = (
            f"selected_candidate_input_preflight_failed task={task_name} outer={outer_fold}; "
            + " | ".join(input_errors[:12])
        )
        logger.error("[CANDIDATE PREFLIGHT FAIL] " + detail)
        raise RuntimeError(detail)

    selected_rows = []
    candidate_results = []

    for cand in candidates:
        selected_row = {
            "source": cand.get("source", ""),
            "task": task_name,
            "outer_fold": outer_fold,
            "candidate_rank": cand["candidate_rank"],
            "candidate_id": cand["candidate_id"],
            "file": cand["file"],
            "fs_method": cand["fs_method"],
            "fs_params": cand["fs_params"],
            "pca_var": cand["pca_var"],
            "clf_model": cand["clf_model"],
            "clf_params": cand["clf_params"],
            "cache_inner_mean_auc": cand.get("inner_mean_auc", np.nan),
            "cache_inner_mean_ap": cand.get("inner_mean_ap", np.nan),
            "cache_inner_std_auc": cand.get("inner_std_auc", np.nan),
            "cache_inner_selection_score": cand.get("inner_selection_score", np.nan),
            "cache_inner_scores_used": cand.get("inner_scores_used", np.nan),
            "status": "selected",
        }
        try:
            logger.debug(f"[RECOMPUTE] task={task_name} outer={outer_fold} rank={cand['candidate_rank']} {cand['candidate_id']}")
            res = recompute_candidate_predictions(
                cand,
                outer,
                task_name,
                train_ids,
                eval_ids,
                config,
                logger,
            )
            candidate_results.append(res)
            selected_row.update(res["state_summary"])
            selected_row["status"] = "recomputed"
        except Exception as e:
            selected_row["status"] = f"failed: {repr(e)}"
            logger.error(f"[RECOMPUTE FAIL] {cand['candidate_id']} err={repr(e)}")
            logger.error(traceback.format_exc())
        selected_rows.append(selected_row)

    assembled = assemble_meta_matrices(
        candidate_results,
        train_ids,
        eval_ids,
        y_train,
        y_eval,
        task_name,
        outer_fold,
        config,
    )

    status_by_id = assembled.get("selected_status", {})
    for row in selected_rows:
        if row["candidate_id"] in status_by_id:
            row["status"] = status_by_id[row["candidate_id"]]

    if not assembled["ok"]:
        status_items = [(row["candidate_id"], row.get("status", "")) for row in selected_rows]
        status_counts = Counter(status for _, status in status_items)
        examples = [f"{cid} => {status}" for cid, status in status_items[:8]]
        detail = (
            f"{assembled['reason']}; "
            f"status_counts={dict(status_counts)}; "
            f"examples={examples}"
        )
        logger.error(f"[NO CANDIDATES] task={task_name} outer={outer_fold} {detail}")
        raise RuntimeError(detail)

    meta = choose_and_fit_meta(
        assembled["z_oof"],
        y_train,
        fold_ids,
        assembled["z_eval"],
        y_eval,
        assembled["z_all"],
        assembled["kept"],
        task_name,
        outer_fold,
        config,
    )

    chosen = meta["chosen"]

    eval_score = chosen["score_eval"]
    eval_pred = score_to_pred(eval_score)
    eval_metrics = safe_binary_metrics(y_eval, eval_score, eval_pred)

    fold_row = {
        "task": task_name,
        "outer_fold": outer_fold,
        "meta_method": chosen["method"],
        "n_cache_candidates_selected": int(len(candidates)),
        "n_candidates_recomputed": int(len(candidate_results)),
        "n_candidates_used": int(len(assembled["kept"])),
        "n_meta_train": int(len(y_train)),
        "n_test": int(len(y_eval)),
        "train_class_dist": json.dumps(dict(Counter(y_train.tolist())), sort_keys=True),
        "test_class_dist": json.dumps(dict(Counter(y_eval.tolist())), sort_keys=True),
        **chosen["extra"],
        **{f"meta_inner_{k}": v for k, v in chosen["oof_metrics"].items()},
        **eval_metrics,
    }

    pred_rows = []
    for cid, yt, score, pred in zip(eval_ids, y_eval, eval_score, eval_pred):
        pred_rows.append({
            "task": task_name,
            "outer_fold": outer_fold,
            "case_id": cid,
            "y_true": int(yt),
            "y_score": float(score),
            "y_pred": int(pred),
            "meta_method": chosen["method"],
        })

    return {
        "fold_row": fold_row,
        "pred_rows": pred_rows,
        "selected_rows": selected_rows,
        "base_oof_rows": assembled["base_oof_rows"],
        "base_outer_rows": assembled["base_outer_rows"],
        "meta_method_rows": meta["method_rows"],
    }


_OVERALL_METRIC_COLS = ["auc", "ap", "balanced_acc", "f1", "sensitivity", "specificity", "accuracy"]
_OVERALL_METRIC_OUT_NAMES = {"balanced_acc": "bal_acc"}


def build_overall_summary(
    meta_fold_rows: List[Dict[str, Any]],
    meta_pred_rows: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred_df = pd.DataFrame(meta_pred_rows)
    fold_df = pd.DataFrame(meta_fold_rows)
    out_cols = ["task", "n"] + [_OVERALL_METRIC_OUT_NAMES.get(m, m) for m in _OVERALL_METRIC_COLS]

    pooled_by_task: Dict[str, Dict[str, Any]] = {}
    mean_by_task: Dict[str, Dict[str, Any]] = {}
    for task in TASKS:
        sub = pred_df[pred_df["task"] == task] if not pred_df.empty else pd.DataFrame()
        if not sub.empty:
            y = sub["y_true"].astype(int).values
            score = sub["y_score"].astype(float).values
            pred = sub["y_pred"].astype(int).values
            mets = safe_binary_metrics(y, score, pred)
            row = {"task": task, "n": int(len(sub))}
            for metric in _OVERALL_METRIC_COLS:
                row[_OVERALL_METRIC_OUT_NAMES.get(metric, metric)] = mets[metric]
            pooled_by_task[task] = row

        fsub = fold_df[fold_df["task"] == task] if not fold_df.empty else pd.DataFrame()
        if not fsub.empty:
            n = int(len(fsub))
            stats = {}
            row = {"task": task, "n": n}
            for metric in _OVERALL_METRIC_COLS:
                vals = pd.to_numeric(fsub[metric], errors="coerce")
                mean_v = round(float(vals.mean()), 4)
                std_v = round(float(vals.std(ddof=0)), 4)
                stats[metric] = (mean_v, std_v)
                row[_OVERALL_METRIC_OUT_NAMES.get(metric, metric)] = f"{mean_v} ± {std_v}"
            mean_by_task[task] = {"n": n, "row": row, "stats": stats}

    def _avg_n(n0: int, n1: int):
        total = n0 + n1
        return total // 2 if total % 2 == 0 else total / 2.0

    pooled_rows = list(pooled_by_task.values())
    if "cs" in pooled_by_task and "zone" in pooled_by_task:
        cs_row, zone_row = pooled_by_task["cs"], pooled_by_task["zone"]
        avg_row = {"task": "average", "n": _avg_n(cs_row["n"], zone_row["n"])}
        for col in out_cols[2:]:
            v0, v1 = cs_row[col], zone_row[col]
            avg_row[col] = round((v0 + v1) / 2.0, 4) if pd.notna(v0) and pd.notna(v1) else np.nan
        pooled_rows.append(avg_row)

    mean_rows = [entry["row"] for entry in mean_by_task.values()]
    if "cs" in mean_by_task and "zone" in mean_by_task:
        cs_entry, zone_entry = mean_by_task["cs"], mean_by_task["zone"]
        avg_row = {"task": "average", "n": _avg_n(cs_entry["n"], zone_entry["n"])}
        for metric in _OVERALL_METRIC_COLS:
            m0, s0 = cs_entry["stats"][metric]
            m1, s1 = zone_entry["stats"][metric]
            avg_m = round((m0 + m1) / 2.0, 4)
            avg_s = round((s0 + s1) / 2.0, 4)
            avg_row[_OVERALL_METRIC_OUT_NAMES.get(metric, metric)] = f"{avg_m} ± {avg_s}"
        mean_rows.append(avg_row)

    pooled_df = pd.DataFrame(pooled_rows, columns=out_cols)
    mean_df = pd.DataFrame(mean_rows, columns=out_cols)
    return pooled_df, mean_df


def cleanup_incomplete_stacking_outputs(result_root: str, logger: logging.Logger) -> None:
    root = Path(result_root)
    removed = []
    for path in [root / "final_stacking.xlsx"]:
        if path.exists():
            try:
                path.unlink()
                removed.append(str(path))
            except Exception as e:
                logger.warning(f"[CLEANUP] Could not remove partial workbook {path}: {repr(e)}")
    if removed:
        logger.error(f"[CLEANUP] Removed incomplete stacking outputs/artifacts: {len(removed)}")


def write_stacking_workbook(
    path: str,
    sheets: Dict[str, pd.DataFrame],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".",
        suffix=".tmp.xlsx",
        dir=os.path.dirname(path),
    )
    os.close(fd)
    try:
        with pd.ExcelWriter(tmp_path, engine="openpyxl", mode="w") as writer:
            for sheet, df in sheets.items():
                df.to_excel(writer, sheet_name=str(sheet)[:31], index=False)
            cfg_module.format_float_cells_4dp(writer)
        if not base.is_valid_excel_workbook(tmp_path):
            raise RuntimeError(f"Temp workbook validation failed: {tmp_path}")
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise


def write_meta_outputs(
    result_root: str,
    stores: Dict[str, List[Dict[str, Any]]],
    overall_summary: Tuple[pd.DataFrame, pd.DataFrame],
    logger: logging.Logger,
) -> None:
    book = os.path.join(result_root, "final_stacking.xlsx")
    pooled_df, mean_outer_df = overall_summary

    sheets = {
        "mean_outer": mean_outer_df,
        "pooled": pooled_df,
        "meta_fold_metrics": pd.DataFrame(stores["meta_fold_rows"]),
        "selected_candidates": pd.DataFrame(stores["selected_rows"]),
        "meta_predictions": pd.DataFrame(stores["meta_pred_rows"]),
        "base_oof_metrics": pd.DataFrame(stores["base_oof_rows"]),
        "base_outer_metrics": pd.DataFrame(stores["base_outer_rows"]),
        "meta_method_perf": pd.DataFrame(stores["meta_method_rows"]),
    }

    sort_specs = {
        "meta_fold_metrics": ["task", "outer_fold"],
        "selected_candidates": ["task", "outer_fold", "candidate_rank"],
        "meta_predictions": ["task", "outer_fold", "case_id"],
        "base_oof_metrics": ["task", "outer_fold", "candidate_rank"],
        "base_outer_metrics": ["task", "outer_fold", "candidate_rank"],
        "meta_method_perf": ["task", "outer_fold", "meta_method"],
    }

    for sheet, cols in sort_specs.items():
        df = sheets[sheet]
        present = [c for c in cols if c in df.columns]
        if present:
            sheets[sheet] = df.sort_values(present, kind="stable").reset_index(drop=True)

    sheets = {name: cfg_module.round_metrics(df) for name, df in sheets.items()}
    write_stacking_workbook(book, sheets)
    logger.debug(f"[SAVE] {book}")


def load_existing_split_payload(config: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    path = Path(config["json_path"]) if config.get("json_path") else Path(config["results_dir"]) / cfg_module.split_json_name(config["random_state"])
    path = path.expanduser()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), str(path)

    raise FileNotFoundError(
        "No existing split JSON found for stacking. Run the requested ML step first or pass --json-path.\n"
        f"Expected:\n  - {path}"
    )


def dry_run_candidate_cache(cache_df: pd.DataFrame, split_payload: Dict[str, Any], config: Dict[str, Any]) -> None:
    print(f"candidate_cache_rows={len(cache_df)}")
    for task in TASKS:
        print(f"\n[{task}]")
        for outer in split_payload["outer_splits"]:
            outer_fold = int(outer["outer_fold"])
            selected = select_cached_candidates(cache_df, task, outer_fold, config)
            print(f"outer_fold={outer_fold} selected={len(selected)}")
            for cand in selected[: min(5, len(selected))]:
                print(
                    "  "
                    f"rank={cand['candidate_rank']} "
                    f"auc={float(cand['inner_mean_auc']):.4f} "
                    f"std={float(cand.get('inner_std_auc', np.nan)):.4f} "
                    f"sel={float(cand.get('inner_selection_score', np.nan)):.4f} "
                    f"file={cand['file']} "
                    f"fs={cand['fs_method']} clf={cand['clf_model']} pca={cand['pca_var']}"
                )


def run_pipeline(config: Dict[str, Any], dry_run: bool = False) -> None:
    if dry_run:
        logger = setup_stdout_logger()
        split_payload, split_used = load_existing_split_payload(config)
        logger.debug(f"Split JSON used: {split_used}")
        cache_df = load_candidate_cache(config, logger)
        if cache_df.empty:
            raise RuntimeError(
                f"No cached fold_metrics found under {config['previous_results_dir']}"
            )
        dry_run_candidate_cache(cache_df, split_payload, config)
        return

    result_root = config["results_dir"]
    base.ensure_dir(result_root)
    logger = setup_stdout_logger()

    logger.debug(f"Stacking pipeline started: {config.get('experiment_name', result_root)}")

    reference_file = os.path.join(config["base_dir"], config["file_order"][0])
    split_payload, split_used = load_existing_split_payload(config)
    logger.debug(f"Split JSON used: {split_used}")

    cache_df = load_candidate_cache(config, logger)
    if cache_df.empty:
        raise RuntimeError(
            f"No cached fold_metrics found under {config['previous_results_dir']}"
        )
    logger.debug(f"Candidate cache rows loaded: {len(cache_df)}")

    if dry_run:
        dry_run_candidate_cache(cache_df, split_payload, config)
        logger.debug("Dry run finished")
        return

    label_maps = build_label_maps(reference_file, config)

    split_coverage_errors = validate_inner_oof_split_coverage(split_payload, label_maps)
    if split_coverage_errors:
        for msg in split_coverage_errors[:20]:
            logger.error(msg)
        cleanup_incomplete_stacking_outputs(result_root, logger)
        raise RuntimeError("stacking_preflight_split_oof_coverage_failed: " + " | ".join(split_coverage_errors[:5]))

    stores = {
        "meta_fold_rows": [],
        "meta_pred_rows": [],
        "selected_rows": [],
        "base_oof_rows": [],
        "base_outer_rows": [],
        "meta_method_rows": [],
    }

    failed_or_missing = []
    for outer in split_payload["outer_splits"]:
        outer_fold = int(outer["outer_fold"])
        for task in TASKS:
            logger.debug(f"[OUTER START] task={task} outer={outer_fold}")
            candidates = select_cached_candidates(cache_df, task, outer_fold, config)
            if not candidates:
                msg = f"[NO CANDIDATES] task={task} outer={outer_fold}"
                logger.error(msg)
                cleanup_incomplete_stacking_outputs(result_root, logger)
                raise RuntimeError("stacking_stopped_no_candidates: " + msg)

            try:
                result = run_task_outer_meta(task, outer, candidates, label_maps, config, logger)
            except Exception as e:
                msg = f"[OUTER FAIL] task={task} outer={outer_fold} err={repr(e)}"
                logger.error(msg)
                logger.error(traceback.format_exc())
                cleanup_incomplete_stacking_outputs(result_root, logger)
                raise RuntimeError("stacking_stopped_outer_fail: " + msg) from e

            stores["meta_fold_rows"].append(result["fold_row"])
            stores["meta_pred_rows"].extend(result["pred_rows"])
            stores["selected_rows"].extend(result["selected_rows"])
            stores["base_oof_rows"].extend(result["base_oof_rows"])
            stores["base_outer_rows"].extend(result["base_outer_rows"])
            stores["meta_method_rows"].extend(result["meta_method_rows"])

            logger.info(f"[OUTER DONE] task={task} outer={outer_fold}")

    overall_df = build_overall_summary(stores["meta_fold_rows"], stores["meta_pred_rows"])
    write_meta_outputs(result_root, stores, overall_df, logger)

    expected_pairs = {(task, int(outer["outer_fold"])) for outer in split_payload["outer_splits"] for task in TASKS}
    completed_pairs = {
        (str(row.get("task")), int(row.get("outer_fold")))
        for row in stores["meta_fold_rows"]
        if row.get("task") is not None and row.get("outer_fold") is not None
    }
    missing_pairs = sorted(expected_pairs - completed_pairs)
    if missing_pairs:
        failed_or_missing.extend([f"[MISSING STACK RESULT] task={task} outer={outer}" for task, outer in missing_pairs])
    if failed_or_missing:
        logger.error("Stacking incomplete; no final workbook/model artifacts will be kept")
        for msg in failed_or_missing[:20]:
            logger.error(msg)
        cleanup_incomplete_stacking_outputs(result_root, logger)
        raise RuntimeError("stacking_incomplete: " + " | ".join(failed_or_missing[:10]))

    logger.debug("Stacking pipeline finished")


def run_late_fusion_stacking_pipeline(config: Dict[str, Any], dry_run: bool = False) -> None:
    logger = setup_stdout_logger()
    split_payload, split_used = load_existing_split_payload(config)
    logger.debug(f"Split JSON used: {split_used}")

    cache_a_cfg = dict(config)
    cache_a_cfg["previous_results_dir"] = str(ORGAN_ONLY_RESULTS_DIR)
    cache_a_cfg["enabled_feature_selectors"] = list(EXPERIMENTS["B"]["feature_selectors"])
    cache_a_cfg["enabled_models"] = list(EXPERIMENTS["B"]["classifiers"])
    cache_a = load_candidate_cache(cache_a_cfg, logger)

    cache_b_cfg = dict(config)
    cache_b_cfg["previous_results_dir"] = str(PATCH_ONLY_RESULTS_DIR)
    cache_b_cfg["enabled_feature_selectors"] = list(EXPERIMENTS["A"]["feature_selectors"])
    cache_b_cfg["enabled_models"] = list(EXPERIMENTS["A"]["classifiers"])
    cache_b = load_candidate_cache(cache_b_cfg, logger)

    if cache_a.empty or cache_b.empty:
        raise RuntimeError(
            "C stacking needs both B_Organ and A_Patch fold_metrics workbooks."
        )

    if dry_run:
        print(f"A_candidate_cache_rows={len(cache_a)}")
        print(f"B_candidate_cache_rows={len(cache_b)}")
        for task in TASKS:
            print(f"\n[C stacked late fusion: {task}]")
            for outer in split_payload["outer_splits"]:
                outer_fold = int(outer["outer_fold"])
                selected = select_late_fusion_candidates(cache_a, cache_b, task, outer_fold, config)
                print(f"outer_fold={outer_fold} selected={len(selected)} A+B candidates")
                for cand in selected[: min(5, len(selected))]:
                    print(
                        "  "
                        f"rank={cand['candidate_rank']} source={cand['source']} "
                        f"auc={float(cand['inner_mean_auc']):.4f} "
                        f"file={cand['file']} "
                        f"fs={cand['fs_method']} clf={cand['clf_model']} pca={cand['pca_var']}"
                    )
        return

    result_root = config["results_dir"]
    base.ensure_dir(result_root)
    logger.debug("Stacking pipeline started: D_Late")

    reference_file = os.path.join(str(ORGAN_ONLY_WORKBOOK_DIR), config["file_order"][0])
    label_maps = build_label_maps(reference_file, config)

    split_coverage_errors = validate_inner_oof_split_coverage(split_payload, label_maps)
    if split_coverage_errors:
        for msg in split_coverage_errors[:20]:
            logger.error(msg)
        cleanup_incomplete_stacking_outputs(result_root, logger)
        raise RuntimeError("stacking_preflight_split_oof_coverage_failed: " + " | ".join(split_coverage_errors[:5]))

    stores = {
        "meta_fold_rows": [],
        "meta_pred_rows": [],
        "selected_rows": [],
        "base_oof_rows": [],
        "base_outer_rows": [],
        "meta_method_rows": [],
    }

    for outer in split_payload["outer_splits"]:
        outer_fold = int(outer["outer_fold"])
        for task in TASKS:
            logger.debug(f"[OUTER START] task={task} outer={outer_fold}")
            candidates = select_late_fusion_candidates(cache_a, cache_b, task, outer_fold, config)
            if not candidates:
                msg = f"[NO CANDIDATES] task={task} outer={outer_fold}"
                logger.error(msg)
                cleanup_incomplete_stacking_outputs(result_root, logger)
                raise RuntimeError("stacking_stopped_no_candidates: " + msg)

            try:
                result = run_task_outer_meta(task, outer, candidates, label_maps, config, logger)
            except Exception as e:
                msg = f"[OUTER FAIL] task={task} outer={outer_fold} err={repr(e)}"
                logger.error(msg)
                logger.error(traceback.format_exc())
                cleanup_incomplete_stacking_outputs(result_root, logger)
                raise RuntimeError("stacking_stopped_outer_fail: " + msg) from e

            stores["meta_fold_rows"].append(result["fold_row"])
            stores["meta_pred_rows"].extend(result["pred_rows"])
            stores["selected_rows"].extend(result["selected_rows"])
            stores["base_oof_rows"].extend(result["base_oof_rows"])
            stores["base_outer_rows"].extend(result["base_outer_rows"])
            stores["meta_method_rows"].extend(result["meta_method_rows"])

            logger.info(f"[OUTER DONE] task={task} outer={outer_fold}")

    overall_df = build_overall_summary(stores["meta_fold_rows"], stores["meta_pred_rows"])
    write_meta_outputs(result_root, stores, overall_df, logger)
    logger.debug("Stacking pipeline finished: D_Late")


def build_config_for_experiment(args: argparse.Namespace, experiment_key: str) -> Optional[Dict[str, Any]]:
    meta = EXPERIMENTS[experiment_key]
    if not meta["applicable"]:
        print(f"[SKIP] {meta['name']}: {meta['skip_reason']}")
        return None

    cfg = dict(CONFIG)
    cfg["experiment_key"] = experiment_key
    cfg["experiment_name"] = meta["name"]
    cfg["stack_kind"] = meta["stack_kind"]
    cfg["base_dir"] = str(Path(args.base_dir) if args.base_dir else meta["workbook_dir"])
    cfg["previous_results_dir"] = str(Path(args.previous_results_dir) if args.previous_results_dir else meta["results_dir"])
    cfg["results_dir"] = str(Path(args.results_dir) if args.results_dir else meta["results_dir"])
    cfg["json_path"] = str(Path(args.json_path)) if args.json_path else str(
        Path(cfg["results_dir"]) / cfg_module.split_json_name(cfg["random_state"])
    )
    cfg["file_order"] = list(meta["feature_files"])
    cfg["enabled_feature_selectors"] = list(meta["feature_selectors"])
    cfg["enabled_models"] = list(meta["classifiers"])
    cfg["meta_top_k"] = int(args.top_k)
    cfg["meta_diversity_max_per_file"]  = int(args.max_per_file)
    cfg["meta_diversity_max_per_fs"]    = int(args.max_per_fs)
    cfg["meta_diversity_max_per_model"] = int(args.max_per_model)
    cfg["meta_std_auc_penalty"] = float(args.std_auc_penalty)
    cfg["meta_score_transform"] = args.score_transform
    return cfg


def parse_experiment_list(raw: str) -> List[str]:
    vals = [x.strip().upper() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("--only cannot be empty")
    invalid = sorted(set(vals) - set(EXPERIMENTS))
    if invalid:
        raise ValueError(f"Unknown experiments in --only: {invalid}. Choices: A,B,B0,C,D")
    return list(dict.fromkeys(vals))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run final_meta stacking logic on the updated A/B/C/D experiment outputs. "
            "Writes only final_stacking.xlsx inside each selected result folder."
        )
    )
    parser.add_argument(
        "--only",
        default=cfg_module.DEFAULT_STACK_EXPERIMENTS,
        help="Comma-separated experiments to stack. Choices: A,B,B0,C,D. D uses stacked late fusion from B+A candidates.",
    )
    parser.add_argument("--base-dir", default=None, help="Override workbook folder. Use only when running one experiment.")
    parser.add_argument("--previous-results-dir", default=None, help="Override prior ML result folder. Use only when running one experiment.")
    parser.add_argument("--results-dir", default=None, help="Override output result folder. Use only when running one experiment.")
    parser.add_argument("--json-path", default=None, help="Experiment-local split JSON path. Defaults to the selected results folder.")
    parser.add_argument("--top-k", type=int, default=CONFIG["meta_top_k"])
    parser.add_argument("--max-per-file", type=int, default=CONFIG["meta_diversity_max_per_file"])
    parser.add_argument("--max-per-fs", type=int, default=CONFIG["meta_diversity_max_per_fs"])
    parser.add_argument("--max-per-model", type=int, default=CONFIG["meta_diversity_max_per_model"])
    parser.add_argument(
        "--std-auc-penalty",
        type=float,
        default=CONFIG["meta_std_auc_penalty"],
        help="Candidate selection score is inner_mean_auc - penalty * inner_std_auc.",
    )
    parser.add_argument("--score-transform", choices=["rank", "none"], default=CONFIG["meta_score_transform"])
    parser.add_argument("--dry-run", action="store_true", help="Only print cached candidate selections; do not recompute models.")
    parser.add_argument("--seed", type=int, default=cfg_module.RANDOM_SEEDS[0])
    parser.add_argument("--all-seeds", action="store_true", help="Run for all seeds in RANDOM_SEEDS sequentially.")

    args = parser.parse_args()
    requested = parse_experiment_list(args.only)
    if len(requested) != 1 and (args.base_dir or args.previous_results_dir or args.results_dir):
        raise ValueError("--base-dir, --previous-results-dir, and --results-dir overrides require exactly one experiment in --only")

    seeds = cfg_module.RANDOM_SEEDS if args.all_seeds else [args.seed]

    for seed in seeds:
        rdirs = cfg_module.results_dirs_for_seed(seed)

        EXPERIMENTS["A"]["results_dir"]  = str(rdirs["patch"])
        EXPERIMENTS["B"]["results_dir"]  = str(rdirs["organ"])
        EXPERIMENTS["B0"]["results_dir"] = str(rdirs["organ_same"])
        EXPERIMENTS["C"]["results_dir"]  = str(rdirs["early_fusion"])
        EXPERIMENTS["D"]["results_dir"]  = str(rdirs["late_fusion"])
        CONFIG["random_state"] = seed

        for experiment_key in requested:
            experiment_cfg = build_config_for_experiment(args, experiment_key)
            if experiment_cfg is None:
                continue
            if experiment_cfg.get("stack_kind") == "late_fusion":
                run_late_fusion_stacking_pipeline(experiment_cfg, dry_run=args.dry_run)
            else:
                run_pipeline(experiment_cfg, dry_run=args.dry_run)
