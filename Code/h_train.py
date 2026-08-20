from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

import all_config as cfg
import e_ml as base
import f_stack
import g_maxPool
from e_ml import (
    get_feature_columns,
    prepare_model_data,
    run_inner_search_combo,
)


def _setup_stdout_logger() -> logging.Logger:
    logger = logging.getLogger("h_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(sh)
    return logger

TASKS = tuple(cfg.CLASSICAL_TASKS)

EXPERIMENTS = {
    "A": ("A_Patch", cfg.PATCH_ONLY_WORKBOOK_DIR),
    "B": ("B_Organ", cfg.ORGAN_ONLY_WORKBOOK_DIR),
    "B0": ("B_Organ_same", cfg.ORGAN_SAME_WORKBOOK_DIR),
    "C": ("C_Early", cfg.EARLY_FUSION_WORKBOOK_DIR),
}


def _config_for(train_dir: Path, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = base._ml_profile_for_base_dir(train_dir)
    c = dict(f_stack.CONFIG)
    c["base_dir"] = str(train_dir)
    c["random_state"] = int(seed)
    c["file_order"] = list(profile["feature_files"])
    c["enabled_feature_selectors"] = list(profile["feature_selectors"])
    c["enabled_models"] = list(profile["classifiers"])
    c["id_col"] = cfg.ID_COL
    c["clin_col"] = cfg.CLIN_COL
    c["zone_col"] = cfg.ZONE_COL
    return c, profile


def _fit_candidate_oof(candidate, full_train_task_df, feature_cols, task_name, inner_splits, config, logger):
    id_col = config["id_col"]
    candidate_id = candidate["candidate_id"]
    oof_parts = []
    for inner in inner_splits:
        tr_ids = set(inner["train_case_ids"])
        va_ids = set(inner["val_case_ids"])
        dtr = full_train_task_df[full_train_task_df[id_col].isin(tr_ids)].copy()
        dva = full_train_task_df[full_train_task_df[id_col].isin(va_ids)].copy()
        if dtr.empty or dva.empty or dtr["y"].nunique() < 2:
            continue
        try:
            state = f_stack.fit_fixed_candidate(dtr, feature_cols, candidate, task_name, config, logger, context=f"{candidate_id}:inner{inner['inner_fold']}")
            pred = f_stack.predict_fixed_candidate(state, dva, config)
            pred["y_true"] = pred["case_id"].map({f_stack._clean_case_id(x): int(y) for x, y in zip(dva[id_col], dva["y"])})
            pred["inner_fold"] = inner["inner_fold"]
            oof_parts.append(pred)
        except Exception as e:
            logger.warning(f"[OOF FAIL] {candidate_id} inner={inner['inner_fold']} err={repr(e)}")
    oof_df = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    final_state = f_stack.fit_fixed_candidate(full_train_task_df, feature_cols, candidate, task_name, config, logger, context=f"{candidate_id}:full_refit")
    return {
        "candidate": candidate,
        "candidate_id": candidate_id,
        "oof_df": oof_df,
        "eval_df": pd.DataFrame(),
        "all_df": pd.DataFrame(),
        "outer_state": final_state,
    }


def _oof_auc(oof_df: pd.DataFrame) -> float:
    if oof_df is None or oof_df.empty or "y_true" not in oof_df.columns:
        return -1.0
    d = oof_df.dropna(subset=["y_true", "raw_score"])
    if d.empty or d["y_true"].nunique() < 2:
        return -1.0
    try:
        return float(roc_auc_score(d["y_true"].astype(int).to_numpy(), d["raw_score"].astype(float).to_numpy()))
    except Exception:
        return -1.0


def _select_base_models(candidate_results, train_ids, y_train, task_name, config, logger):
    for res in candidate_results:
        res["oof_auc"] = _oof_auc(res["oof_df"])
    top_k = int(config.get("meta_top_k", 8))
    selected = sorted(candidate_results, key=lambda r: r["oof_auc"], reverse=True)[:top_k]

    assembled = f_stack.assemble_meta_matrices(
        candidate_results=selected, train_ids=train_ids, eval_ids=[],
        y_train=y_train, y_eval=np.array([], dtype=int), task_name=task_name, outer_fold=0, config=config,
    )
    if not assembled["ok"]:
        raise RuntimeError(f"meta assembly failed: {assembled.get('reason')}")
    kept = assembled["kept"]
    kept_ids = [k["candidate_id"] for k in kept]
    id_to_res = {r["candidate_id"]: r for r in selected}
    ordered = [id_to_res[cid] for cid in kept_ids if cid in id_to_res]
    logger.info(f"[MODEL TRAIN] selected {len(ordered)} base models for weighted averaging")
    return ordered, kept_ids


def train_experiment(exp_key: str, exp_name: str, train_dir: Path, seed: int) -> None:
    train_dir = Path(train_dir)
    config, profile = _config_for(train_dir, seed)
    models_dir = cfg.final_model_dir(exp_key, seed)
    models_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_stdout_logger()

    ref_file = profile["feature_files"][0]
    ref_df = cfg.read_features(train_dir / ref_file)
    results_split_path = cfg.split_json_path(exp_key, seed)
    if not results_split_path.exists():
        raise FileNotFoundError(f"Missing outer split {results_split_path} for {exp_name}. Run nested-CV ML first.")
    outer_splits = json.loads(results_split_path.read_text())["outer_splits"]
    shared = [
        {"inner_fold": s["outer_fold"], "train_case_ids": s["train_case_ids"], "val_case_ids": s["test_case_ids"]}
        for s in outer_splits
    ]

    for task_name in cfg.classical_tasks_for_experiment(exp_key):
        task_ref = prepare_model_data(ref_df, task_name, config)
        task_ids = set(task_ref[cfg.ID_COL].astype(str).tolist())
        inner_splits = [
            {
                "inner_fold": s["inner_fold"],
                "train_case_ids": [c for c in s["train_case_ids"] if c in task_ids],
                "val_case_ids": [c for c in s["val_case_ids"] if c in task_ids],
            }
            for s in shared
        ]
        id_to_y = {f_stack._clean_case_id(x): int(y) for x, y in zip(task_ref[cfg.ID_COL], task_ref["y"])}

        total_combos = len(profile["feature_files"]) * len(config["enabled_feature_selectors"]) * len(config["enabled_models"])
        milestones = {int(total_combos * p / 100) for p in (25, 50, 75, 100)} - {0}
        done = 0

        candidate_results = []
        rank = 0
        for file_name in profile["feature_files"]:
            file_tag = Path(file_name).stem
            train_path = train_dir / file_name
            if not train_path.exists():
                logger.warning(f"[MODEL TRAIN] missing {train_path}")
                done += len(config["enabled_feature_selectors"]) * len(config["enabled_models"])
                continue
            train_df = cfg.read_features(train_path)
            feature_cols = get_feature_columns(train_df, config)
            dtrain = prepare_model_data(train_df, task_name, config)
            skip_fit = dtrain.empty or dtrain["y"].nunique() < 2
            for fs_method in config["enabled_feature_selectors"]:
                for clf_model in config["enabled_models"]:
                    done += 1
                    if skip_fit:
                        if done in milestones:
                            pct = round(done * 100 / total_combos)
                            print(f"[MODEL TRAIN] {exp_name} task={task_name} {pct}% ({done}/{total_combos})")
                        continue
                    ctx = f"{exp_name}:{file_tag}:{task_name}:{fs_method}:{clf_model}"
                    best_params, _ = run_inner_search_combo(
                        train_df=dtrain, feature_cols=feature_cols, inner_splits=inner_splits,
                        task_name=task_name, fs_method=fs_method, clf_model=clf_model,
                        config=config, logger=logger, context=ctx,
                    )
                    if best_params:
                        rank += 1
                        candidate = {
                            "file": file_tag,
                            "task": task_name,
                            "fs_method": best_params["fs_method"],
                            "fs_params": best_params["fs_params"],
                            "pca_var": best_params["pca_var"],
                            "clf_model": best_params["clf_model"],
                            "clf_params": best_params["clf_params"],
                            "inner_mean_auc": best_params.get("inner_mean_auc", np.nan),
                            "inner_mean_ap": best_params.get("inner_mean_ap", np.nan),
                            "inner_std_auc": best_params.get("inner_std_auc", np.nan),
                            "candidate_rank": rank,
                            "candidate_id": f"refit::{task_name}::{file_tag}::{fs_method}::{clf_model}",
                            "outer_fold": 0,
                        }
                        try:
                            res = _fit_candidate_oof(candidate, dtrain, feature_cols, task_name, inner_splits, config, logger)
                            candidate_results.append(res)
                        except Exception as e:
                            logger.error(f"[MODEL TRAIN FAIL] {candidate['candidate_id']} err={repr(e)}")
                    if done in milestones:
                        pct = round(done * 100 / total_combos)
                        print(f"[MODEL TRAIN] {exp_name} task={task_name} {pct}% ({done}/{total_combos})")

        if not candidate_results:
            raise RuntimeError(f"No deployable candidates for {exp_name}:{task_name}")

        expected_combos = {
            (Path(file_name).stem, fs_method, clf_model)
            for file_name in profile["feature_files"]
            for fs_method in config["enabled_feature_selectors"]
            for clf_model in config["enabled_models"]
        }
        fitted_combos = {
            (
                str(res["candidate"]["file"]),
                str(res["candidate"]["fs_method"]),
                str(res["candidate"]["clf_model"]),
            )
            for res in candidate_results
        }
        missing_combos = sorted(expected_combos - fitted_combos)
        if missing_combos:
            raise RuntimeError(
                f"Deployable candidate grid incomplete for {exp_name}:{task_name}; "
                f"missing examples: {missing_combos[:8]}"
            )

        train_ids_set = set()
        for res in candidate_results:
            if not res["oof_df"].empty and "case_id" in res["oof_df"].columns:
                train_ids_set.update(res["oof_df"]["case_id"].dropna().tolist())
        train_ids = sorted(train_ids_set)
        y_train_full = np.array([id_to_y.get(c, np.nan) for c in train_ids])
        valid = ~np.isnan(y_train_full)
        train_ids = [c for c, v in zip(train_ids, valid) if v]
        y_train = y_train_full[valid].astype(int)
        selected, kept_ids = _select_base_models(
            candidate_results, train_ids, y_train, task_name, config, logger
        )

        bundle = {
            "experiment": exp_name,
            "task": task_name,
            "seed": seed,
            "candidate_results": selected,
            "kept_ids": kept_ids,
            "train_ids": train_ids,
            "y_train": y_train,
            "feature_files": list(profile["feature_files"]),
            "config": config,
        }
        out_path = cfg.final_model_path(exp_key, task_name, seed)
        joblib.dump(bundle, out_path)
        logger.info(f"[MODEL TRAIN] saved {out_path} base={len(selected)} stacker=weighted_average")
        print(f"[MODEL TRAIN] saved {out_path} (final model: {len(selected)} base + weighted_average)")

    missing_models = [
        cfg.final_model_path(exp_key, task, seed)
        for task in cfg.classical_tasks_for_experiment(exp_key)
        if not cfg.final_model_path(exp_key, task, seed).exists()
    ]
    if missing_models:
        raise RuntimeError(
            f"Deployable training incomplete for {exp_name}; missing: "
            + ", ".join(str(path) for path in missing_models)
        )

def train_seed(seed: int, requested: set[str]) -> None:
    for key, (exp_name, train_dir) in EXPERIMENTS.items():
        if requested and key not in requested:
            continue
        if not Path(train_dir).exists():
            print(f"[MODEL TRAIN] skip {exp_name}: feature dir not built yet ({train_dir})")
            continue
        train_experiment(key, exp_name, train_dir, seed)
    if not requested or "A" in requested:
        g_maxPool.train_deployable(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train deployable models on full ProstateX (all file/fs/classifier combos, "
                    "tuned, with OOF for stacking) and save per seed to models/seed{N}/. i_test "
                    "loads these and applies them to P158 with no refit."
    )
    parser.add_argument("--only", default=cfg.DEFAULT_INTERNAL_EXPERIMENTS,
                        help="Comma-separated experiment keys to train (A, B, B0, C).")
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_STATE)
    parser.add_argument("--all-seeds", action="store_true")
    args = parser.parse_args()

    requested = {x.strip().upper() for x in args.only.split(",") if x.strip()}
    invalid = requested - set(EXPERIMENTS)
    if invalid:
        raise ValueError(f"Unknown experiments in --only: {sorted(invalid)}. Choices: {sorted(EXPERIMENTS)}")
    seeds = cfg.RANDOM_SEEDS if args.all_seeds else [args.seed]
    for seed in seeds:
        print(f"\n========== h_train seed={seed} ==========")
        train_seed(seed, requested)


if __name__ == "__main__":
    main()
