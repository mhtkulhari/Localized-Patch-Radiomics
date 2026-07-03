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
from e_ml import (
    get_feature_columns,
    prepare_model_data,
    run_inner_search_combo,
)


def _setup_stdout_logger() -> logging.Logger:
    logger = logging.getLogger("g_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(sh)
    return logger

TASKS = ("cs", "zone")

EXPERIMENTS = {
    "A": ("A_Patch", cfg.PATCH_ONLY_WORKBOOK_DIR),
    "B": ("B_Organ", cfg.ORGAN_ONLY_WORKBOOK_DIR),
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


def _select_and_choose_stacker(candidate_results, train_ids, y_train, fold_ids, task_name, config, logger):
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
    meta = f_stack.choose_and_fit_meta(
        assembled["z_oof"], y_train, fold_ids, assembled["z_eval"], np.array([], dtype=int),
        assembled["z_all"], kept, task_name, 0, config,
    )
    chosen_method = meta["chosen"]["method"]
    id_to_res = {r["candidate_id"]: r for r in selected}
    ordered = [id_to_res[cid] for cid in kept_ids if cid in id_to_res]
    logger.info(f"[MODEL TRAIN] selected {len(ordered)} base models, stacker={chosen_method}")
    return ordered, kept_ids, chosen_method, meta


def _da_oof_metrics(oof_df: pd.DataFrame) -> dict:
    if oof_df is None or oof_df.empty:
        return {}
    d = oof_df.dropna(subset=["y_true", "raw_score"])
    if d.empty or d["y_true"].nunique() < 2:
        return {}
    yt = d["y_true"].astype(int).to_numpy()
    ys = d["raw_score"].astype(float).to_numpy()
    return f_stack.safe_binary_metrics(yt, ys, f_stack.score_to_pred(ys))


def _da_individual_row(exp_name: str, task_name: str, res: dict, is_selected: bool) -> dict:
    cand = res["candidate"]
    oof_mets = _da_oof_metrics(res["oof_df"])
    return {
        "experiment": exp_name, "task": task_name,
        "candidate_id": res["candidate_id"],
        "file": cand["file"], "fs_method": cand["fs_method"], "clf_model": cand["clf_model"],
        "is_selected": is_selected,
        **{f"oof_{k}": v for k, v in oof_mets.items()},
    }


def _da_stacking_row(exp_name: str, task_name: str, choice: dict, is_chosen: bool) -> dict:
    return {
        "experiment": exp_name, "task": task_name,
        "meta_method": choice["method"], "is_chosen": is_chosen,
        "oof_auc": choice.get("oof_metrics", {}).get("auc", np.nan),
    }


def _da_base_row(exp_name: str, task_name: str, meta: dict, kept_ids: list) -> dict:
    chosen = meta["chosen"]
    om = chosen.get("oof_metrics", {})
    return {
        "experiment": exp_name, "task": task_name,
        "n_base": len(kept_ids), "chosen_stacker": chosen["method"],
        "selected_candidates": ", ".join(kept_ids),
        **{f"oof_{k}": v for k, v in om.items()},
    }


def _write_analysis_xlsx(individual_rows: list, stacking_rows: list, base_rows: list, path: Path) -> None:
    def _sort(df: pd.DataFrame, auc_col: str) -> pd.DataFrame:
        if auc_col in df.columns:
            return df.sort_values(["task", auc_col], ascending=[True, False], na_position="last").reset_index(drop=True)
        return df

    indiv_df = cfg.round_metrics(_sort(pd.DataFrame(individual_rows), "oof_auc"))
    stack_df = cfg.round_metrics(_sort(pd.DataFrame(stacking_rows), "oof_auc"))
    base_df = cfg.round_metrics(pd.DataFrame(base_rows))
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        indiv_df.to_excel(writer, sheet_name="individual_metrics", index=False)
        stack_df.to_excel(writer, sheet_name="stacking_all_methods", index=False)
        base_df.to_excel(writer, sheet_name="base_metrics", index=False)
    print(f"[MODEL TRAIN] saved analysis {path}")


def train_experiment(exp_key: str, exp_name: str, train_dir: Path, seed: int) -> dict:
    train_dir = Path(train_dir)
    config, profile = _config_for(train_dir, seed)
    models_dir = cfg.final_model_dir(exp_key, seed)
    models_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_stdout_logger()
    da_individual, da_stacking, da_base = [], [], []

    ref_file = profile["feature_files"][0]
    ref_df = cfg.read_features(train_dir / ref_file)
    results_split_path = cfg.split_json_path(exp_key, seed)
    if not results_split_path.exists():
        logger.warning(f"[MODEL TRAIN] missing outer split {results_split_path} for {exp_name}; skipping")
        return {"individual": da_individual, "stacking": da_stacking, "base": da_base}
    outer_splits = json.loads(results_split_path.read_text())["outer_splits"]
    shared = [
        {"inner_fold": s["outer_fold"], "train_case_ids": s["train_case_ids"], "val_case_ids": s["test_case_ids"]}
        for s in outer_splits
    ]

    for task_name in TASKS:
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
        fold_id_map = {c: s["inner_fold"] for s in inner_splits for c in s["val_case_ids"]}
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
            logger.warning(f"[MODEL TRAIN] no candidates for {exp_name}:{task_name}")
            continue

        train_ids_set = set()
        for res in candidate_results:
            if not res["oof_df"].empty and "case_id" in res["oof_df"].columns:
                train_ids_set.update(res["oof_df"]["case_id"].dropna().tolist())
        train_ids = sorted(train_ids_set)
        y_train_full = np.array([id_to_y.get(c, np.nan) for c in train_ids])
        valid = ~np.isnan(y_train_full)
        train_ids = [c for c, v in zip(train_ids, valid) if v]
        y_train = y_train_full[valid].astype(int)
        fold_ids = np.array([fold_id_map.get(c, 1) for c in train_ids])

        selected, kept_ids, chosen_method, meta = _select_and_choose_stacker(
            candidate_results, train_ids, y_train, fold_ids, task_name, config, logger
        )

        kept_set = set(kept_ids)
        for res in candidate_results:
            da_individual.append(_da_individual_row(exp_name, task_name, res, res["candidate_id"] in kept_set))
        chosen_name = meta["chosen"]["method"]
        for ch in meta["all_choices"]:
            da_stacking.append(_da_stacking_row(exp_name, task_name, ch, ch["method"] == chosen_name))
        da_base.append(_da_base_row(exp_name, task_name, meta, kept_ids))

        bundle = {
            "experiment": exp_name,
            "task": task_name,
            "seed": seed,
            "candidate_results": selected,
            "kept_ids": kept_ids,
            "chosen_stacker": chosen_method,
            "train_ids": train_ids,
            "y_train": y_train,
            "fold_ids": fold_ids,
            "feature_files": list(profile["feature_files"]),
            "config": config,
        }
        out_path = cfg.final_model_path(exp_key, task_name, seed)
        joblib.dump(bundle, out_path)
        logger.info(f"[MODEL TRAIN] saved {out_path} base={len(selected)} stacker={chosen_method}")
        print(f"[MODEL TRAIN] saved {out_path} (final model: {len(selected)} base + {chosen_method})")

    if da_individual or da_base:
        _write_analysis_xlsx(da_individual, da_stacking, da_base, models_dir / "analysis.xlsx")

    return {"individual": da_individual, "stacking": da_stacking, "base": da_base}


def train_seed(seed: int, requested: set[str]) -> None:
    for key, (exp_name, train_dir) in EXPERIMENTS.items():
        if requested and key not in requested:
            continue
        if not Path(train_dir).exists():
            print(f"[MODEL TRAIN] skip {exp_name}: feature dir not built yet ({train_dir})")
            continue
        train_experiment(key, exp_name, train_dir, seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train deployable models on full ProstateX (all file/fs/classifier combos, "
                    "tuned, with OOF for stacking) and save per seed to models/seed{N}/. h_test "
                    "loads these and applies them to P158 with no refit."
    )
    parser.add_argument("--only", default=cfg.DEFAULT_INTERNAL_EXPERIMENTS,
                        help="Comma-separated experiment keys to train (A, B, C).")
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEEDS[0])
    parser.add_argument("--all-seeds", action="store_true")
    args = parser.parse_args()

    requested = {x.strip().upper() for x in args.only.split(",") if x.strip()}
    seeds = cfg.RANDOM_SEEDS if args.all_seeds else [args.seed]
    for seed in seeds:
        print(f"\n========== g_train seed={seed} ==========")
        train_seed(seed, requested)


if __name__ == "__main__":
    main()
