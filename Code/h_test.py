from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import all_config as cfg
import a_preprocess
import c_organ
import b_patch
import d_fusion
import f_stack
from e_ml import compute_metrics, prepare_model_data

TASKS = ("cs", "zone")

EXPERIMENT_KEY = {"A_Patch": "A", "B_Organ": "B", "C_Early": "C", "D_Late": "D"}

EXPERIMENT_P158_FEATURE_DIR = {
    "B_Organ": cfg.external_summary_dir("B"),
    "A_Patch": cfg.external_summary_dir("A"),
    "C_Early": cfg.external_summary_dir("C"),
}
EXPERIMENT_P158_RESULTS_DIR = {
    "B_Organ": cfg.external_results_dir("B"),
    "A_Patch": cfg.external_results_dir("A"),
    "C_Early": cfg.external_results_dir("C"),
}


def load_p158_labels() -> pd.DataFrame:
    return c_organ.load_main_labels(cfg.P158_DATASET_XLSX)




def prepare_p158_organ_features() -> None:
    out_dir = cfg.external_summary_dir("B")
    expected_ids = set(c_organ.p158_case_ids(cfg.P158_ROOT))

    def _file_complete(path: Path) -> bool:
        if not path.exists():
            return False
        got = set(pd.read_csv(path, dtype={cfg.ID_COL: str})[cfg.ID_COL].astype(str))
        return expected_ids.issubset(got)

    missing = [f for f in cfg.FEATURE_FILES if not _file_complete(out_dir / f)]
    if not missing:
        print(f"[P158 ORGAN] external B_Organ features already exist: {out_dir}")
        return

    keep = c_organ.load_keep_features(cfg.ORGAN_ICC_DIR)
    labels = load_p158_labels()

    existing_by_file: dict[str, pd.DataFrame] = {}
    done_ids: set[str] | None = None
    for f in missing:
        p = out_dir / f
        if p.exists():
            df = pd.read_csv(p, dtype={cfg.ID_COL: str})
            existing_by_file[f] = df
            ids = set(df[cfg.ID_COL].astype(str))
            done_ids = ids if done_ids is None else (done_ids & ids)
        else:
            done_ids = set()
    done_ids = done_ids or set()
    if not existing_by_file:
        print(f"[P158 ORGAN] Summary Feature Files are built and will be updated per {cfg.EXCEL_FLUSH_EVERY} cases")

    def _publish(new_tables: dict) -> None:
        if not any(len(df) for df in new_tables.values()):
            return
        new_books = c_organ.build_requested_books(labels, new_tables, keep, missing)
        for file_name, new_df in new_books.items():
            old_df = existing_by_file.get(file_name)
            combined = pd.concat([old_df, new_df], ignore_index=True) if old_df is not None and not old_df.empty else new_df
            combined = combined.drop_duplicates(cfg.ID_COL, keep="last")
            c_organ.write_book(out_dir / file_name, combined, quiet=True)

    c_organ.extract_p158_organ_features_combined(cfg.P158_ROOT, keep, already_done_ids=done_ids, on_progress=_publish)
    print(f"[P158 ORGAN] saved external B_Organ features: {out_dir}")


def _patch_module_to_p158() -> None:
    b_patch.MAIN_ROOT = cfg.P158_ROOT
    b_patch.DATASET_XLSX = cfg.P158_DATASET_XLSX
    b_patch.PATCH_ONLY_WORKBOOK_DIR = cfg.external_summary_dir("A")
    b_patch.EARLY_FUSION_WORKBOOK_DIR = cfg.external_summary_dir("C")
    b_patch.PATCH_APPLY_WORKERS = cfg.EXTERNAL_PATCH_APPLY_WORKERS


def prepare_p158_patch_and_fusion(build_early_fusion: bool = True) -> None:
    patch_out = cfg.external_summary_dir("A")
    early_out = cfg.external_summary_dir("C")
    organ_out = cfg.external_summary_dir("B")

    patch_files = cfg.FEATURE_FILES
    early_files = cfg.FEATURE_FILES if build_early_fusion else []
    expected_patch_ids = {str(c).strip() for c in b_patch.list_case_ids(cfg.P158_ROOT)}

    patch_missing = [
        f for f in patch_files
        if not expected_patch_ids or not expected_patch_ids.issubset(cfg.patch_workbook_done_ids(patch_out / f))
    ]
    early_missing = [f for f in early_files if not (early_out / f).exists()] if build_early_fusion else []
    
    if not patch_missing and not early_missing:
        print(f"[P158 A/C] external patch/fusion features already exist: {patch_out}")
        return
    
    _patch_module_to_p158()
    if patch_missing:
        b_patch.apply_patch_helpers(write_views=False)
        print(f"[P158 A] saved patch features: {patch_out}")
    else:
        print(f"[P158 A] patch features already exist: {patch_out}")
    
    if build_early_fusion and early_missing:
        d_fusion.build_early_fusion_workbooks(organ_dir=organ_out, patch_dir=patch_out, out_dir=early_out)
        print(f"[P158 C] saved early-fusion features: {early_out}")
    elif build_early_fusion:
        print(f"[P158 C] early-fusion features already exist: {early_out}")


def prepare_p158_features(requested: set[str]) -> None:
    if requested & {"B_Organ", "C_Early"}:
        prepare_p158_organ_features()
    if requested & {"A_Patch", "C_Early"}:
        prepare_p158_patch_and_fusion(build_early_fusion="C_Early" in requested)


def _apply_experiment_task(bundle: dict, p158_feature_dir: Path) -> dict:
    config = bundle["config"]
    task = bundle["task"]
    exp = bundle["experiment"]
    _METHOD_ALIASES = {
        "weighted_oof_auc": "weighted_average",
        "mean_rank": "simple_average",
        "best_single_oof": "single_best",
    }
    chosen_method = _METHOD_ALIASES.get(bundle["chosen_stacker"], bundle["chosen_stacker"])
    y_true_map = _p158_truth_map(task, config)

    file_task_cache: dict[str, pd.DataFrame] = {}
    candidate_results = []
    indiv_metric_rows = []
    indiv_pred_rows = []
    for res in bundle["candidate_results"]:
        file_tag = res["candidate"]["file"]
        if file_tag not in file_task_cache:
            fpath = p158_feature_dir / f"{file_tag}.csv"
            if not fpath.exists():
                raise FileNotFoundError(f"Missing P158 feature table for apply: {fpath}")
            df = cfg.read_features(fpath)
            file_task_cache[file_tag] = prepare_model_data(df, task, config)
        all_df = f_stack.predict_fixed_candidate(res["outer_state"], file_task_cache[file_tag], config)
        candidate_results.append({**res, "all_df": all_df})
        if not all_df.empty:
            cids = all_df["case_id"].tolist()
            ys = all_df["raw_score"].astype(float).to_numpy()
            yp = f_stack.score_to_pred(ys)
            yt = np.array([y_true_map.get(c, np.nan) for c in cids])
            v = ~np.isnan(yt)
            mets = compute_metrics(yt[v].astype(int), ys[v], yp[v]) if v.sum() >= 2 and len(np.unique(yt[v])) >= 2 else {}
            cand = res["candidate"]
            indiv_metric_rows.append({"experiment": exp, "task": task, "file": cand["file"], "fs_method": cand["fs_method"], "clf_model": cand["clf_model"], "candidate_id": res["candidate_id"], "n": int(v.sum()), **mets})
            for c, t, s, p in zip(cids, yt, ys, yp):
                indiv_pred_rows.append({"experiment": exp, "task": task, "candidate_id": res["candidate_id"], cfg.ID_COL: c, "y_true": t, "y_score": float(s), "y_pred": int(p)})

    p158_ids = sorted({cid for r in candidate_results if not r["all_df"].empty for cid in r["all_df"]["case_id"].dropna().tolist()})
    train_ids = list(bundle["train_ids"])
    y_train = np.asarray(bundle["y_train"], dtype=int)
    fold_ids = np.asarray(bundle["fold_ids"], dtype=int)

    assembled = f_stack.assemble_meta_matrices(
        candidate_results=candidate_results, train_ids=train_ids, eval_ids=[], all_test_ids=p158_ids,
        y_train=y_train, y_eval=np.array([], dtype=int), task_name=task, outer_fold=0, config=config,
    )
    if not assembled["ok"]:
        raise RuntimeError(f"meta assembly failed for {exp}:{task}: {assembled.get('reason')}")
    meta = f_stack.choose_and_fit_meta(
        assembled["z_oof"], y_train, fold_ids, assembled["z_eval"], np.array([], dtype=int),
        assembled["z_all"], assembled["kept"], task, 0, config,
    )
    choice = next((c for c in meta["all_choices"] if c["method"] == chosen_method), meta["chosen"])
    y_score = np.asarray(choice["score_all"], dtype=float)
    y_pred = f_stack.score_to_pred(y_score)

    used_method = choice["method"]
    final_rows = [
        {"experiment": exp, "task": task, "meta_method": used_method, cfg.ID_COL: cid, "y_true": y_true_map.get(cid, np.nan), "y_score": float(s), "y_pred": int(p)}
        for cid, s, p in zip(p158_ids, y_score, y_pred)
    ]
    yt_all = np.array([y_true_map.get(c, np.nan) for c in p158_ids])
    v_all = ~np.isnan(yt_all)
    all_methods = []
    for ch in meta["all_choices"]:
        sa = np.asarray(ch["score_all"], dtype=float)
        if sa.shape[0] != len(p158_ids):
            continue
        yp = f_stack.score_to_pred(sa)
        mm = compute_metrics(yt_all[v_all].astype(int), sa[v_all], yp[v_all]) if v_all.sum() >= 2 and len(np.unique(yt_all[v_all])) >= 2 else {}
        all_methods.append({"experiment": exp, "task": task, "meta_method": ch["method"], "is_chosen": ch["method"] == chosen_method, **mm})

    return {
        "final": pd.DataFrame(final_rows),
        "individual_metrics": pd.DataFrame(indiv_metric_rows),
        "individual_predictions": pd.DataFrame(indiv_pred_rows),
        "stacking_all_methods": pd.DataFrame(all_methods),
    }


def _p158_truth_map(task: str, config: dict) -> dict:
    ref = None
    for d in EXPERIMENT_P158_FEATURE_DIR.values():
        cand = Path(d) / cfg.FEATURE_FILES[0]
        if cand.exists():
            ref = cand
            break
    if ref is None:
        return {}
    df = cfg.read_features(ref)
    dtask = prepare_model_data(df, task, config)
    return {f_stack._clean_case_id(x): int(y) for x, y in zip(dtask[cfg.ID_COL], dtask["y"])}


def _metrics_row(experiment: str, task: str, pred_df: pd.DataFrame) -> dict:
    valid = pred_df.dropna(subset=["y_true"])
    if valid.empty or valid["y_true"].nunique() < 2:
        return {"experiment": experiment, "task": task, "n": int(len(valid))}
    m = compute_metrics(valid["y_true"].astype(int).to_numpy(), valid["y_score"].astype(float).to_numpy(), valid["y_pred"].astype(int).to_numpy())
    return {"experiment": experiment, "task": task, "n": int(len(valid)), **m}


_DA_SKIP_INDIV = {"experiment", "task", "file", "fs_method", "clf_model", "candidate_id", "n"}
_DA_SKIP_STACK = {"experiment", "task", "meta_method", "is_chosen"}


def _da_collect_task_ext(exp_name: str, task: str, out: dict) -> tuple[list, list, dict]:
    indiv = out["individual_metrics"]
    indiv_rows = []
    if not indiv.empty and "candidate_id" in indiv.columns:
        for _, row in indiv.iterrows():
            indiv_rows.append({
                "candidate_id": row["candidate_id"],
                **{f"ext_{c}": row[c] for c in indiv.columns if c not in _DA_SKIP_INDIV},
            })

    stack = out["stacking_all_methods"]
    stack_rows = []
    if not stack.empty and "meta_method" in stack.columns:
        for _, row in stack.iterrows():
            stack_rows.append({
                "experiment": exp_name, "task": task, "meta_method": row["meta_method"],
                **{f"ext_{c}": row[c] for c in stack.columns if c not in _DA_SKIP_STACK},
            })

    final_valid = out["final"].dropna(subset=["y_true"]) if not out["final"].empty else pd.DataFrame()
    ext_base = {}
    if not final_valid.empty and final_valid["y_true"].nunique() >= 2:
        yt = final_valid["y_true"].astype(int).to_numpy()
        ys = final_valid["y_score"].astype(float).to_numpy()
        yp = final_valid["y_pred"].astype(int).to_numpy()
        ext_base = {f"ext_{k}": v for k, v in compute_metrics(yt, ys, yp).items()}
    base_row = {"experiment": exp_name, "task": task, **ext_base}

    return indiv_rows, stack_rows, base_row


def _da_update_analysis(seed: int, models_dir: Path, da_indiv: list, da_stack: list, da_base: list) -> None:
    analysis_path = Path(models_dir) / "analysis.xlsx"
    if not analysis_path.exists():
        print(f"[H TEST] deep-analysis skip: {analysis_path} not found (run g_train first)")
        return

    sheets = pd.read_excel(analysis_path, sheet_name=None)

    if "individual_metrics" in sheets and da_indiv:
        ext_df = pd.DataFrame(da_indiv).drop_duplicates("candidate_id")
        base = sheets["individual_metrics"]
        base = base[[c for c in base.columns if not c.startswith("ext_")]]
        ext_cols = ["candidate_id"] + [c for c in ext_df.columns if c.startswith("ext_")]
        merged = base.merge(ext_df[ext_cols], on="candidate_id", how="left")
        if "ext_auc" in merged.columns:
            merged = merged.sort_values(["task", "ext_auc"], ascending=[True, False], na_position="last").reset_index(drop=True)
        sheets["individual_metrics"] = merged

    if "stacking_all_methods" in sheets and da_stack:
        ext_df = pd.DataFrame(da_stack)
        base = sheets["stacking_all_methods"]
        base = base[[c for c in base.columns if not c.startswith("ext_")]]
        ext_cols = ["experiment", "task", "meta_method"] + [c for c in ext_df.columns if c.startswith("ext_")]
        merged = base.merge(ext_df[ext_cols], on=["experiment", "task", "meta_method"], how="left")
        if "ext_auc" in merged.columns:
            merged = merged.sort_values(["task", "ext_auc"], ascending=[True, False], na_position="last").reset_index(drop=True)
        sheets["stacking_all_methods"] = merged

    if "base_metrics" in sheets and da_base:
        ext_df = pd.DataFrame(da_base)
        base = sheets["base_metrics"]
        base = base[[c for c in base.columns if not c.startswith("ext_")]]
        ext_cols = ["experiment", "task"] + [c for c in ext_df.columns if c.startswith("ext_")]
        merged = base.merge(ext_df[ext_cols], on=["experiment", "task"], how="left")
        sheets["base_metrics"] = merged

    with pd.ExcelWriter(analysis_path, engine="openpyxl") as writer:
        for sname, df in sheets.items():
            cfg.round_metrics(df).to_excel(writer, sheet_name=sname, index=False)
    print(f"[H TEST] updated analysis {analysis_path}")



def apply_experiment(seed: int, exp_name: str, deep_analysis: bool = False) -> dict | None:
    exp_key = EXPERIMENT_KEY[exp_name]
    feature_dir = Path(EXPERIMENT_P158_FEATURE_DIR[exp_name])
    results_dir = Path(EXPERIMENT_P158_RESULTS_DIR[exp_name])
    results_dir.mkdir(parents=True, exist_ok=True)

    final_frames, metric_rows, indiv_metrics, indiv_preds, all_methods = [], [], [], [], []
    task_outs: dict = {}
    for task in TASKS:
        bundle_path = cfg.final_model_path(exp_key, task, seed)
        if not bundle_path.exists():
            raise FileNotFoundError(f"Missing trained model: {bundle_path}. Run g_train.py for seed {seed} first.")
        bundle = joblib.load(bundle_path)
        out = _apply_experiment_task(bundle, feature_dir)
        task_outs[task] = out
        final_frames.append(out["final"])
        metric_rows.append(_metrics_row(exp_name, task, out["final"]))
        indiv_metrics.append(out["individual_metrics"])
        indiv_preds.append(out["individual_predictions"])
        all_methods.append(out["stacking_all_methods"])

    indiv_df = pd.concat(indiv_metrics, ignore_index=True)
    if "auc" in indiv_df.columns:
        indiv_df = indiv_df.sort_values(["task", "auc"], ascending=[True, False], na_position="last").reset_index(drop=True)
    stack_df = pd.concat(all_methods, ignore_index=True)
    if "auc" in stack_df.columns:
        stack_df = stack_df.sort_values(["task", "auc"], ascending=[True, False], na_position="last").reset_index(drop=True)

    out_path = results_dir / f"ext_results_seed{seed}.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        cfg.round_metrics(pd.DataFrame(metric_rows)).to_excel(writer, sheet_name="stacking_metrics", index=False)
        cfg.round_metrics(pd.concat(final_frames, ignore_index=True)).to_excel(writer, sheet_name="stacking_predictions", index=False)
        cfg.round_metrics(indiv_df).to_excel(writer, sheet_name="individual_metrics", index=False)
        cfg.round_metrics(pd.concat(indiv_preds, ignore_index=True)).to_excel(writer, sheet_name="individual_predictions", index=False)
        cfg.round_metrics(stack_df).to_excel(writer, sheet_name="stacking_all_methods", index=False)
    print(f"[H TEST] saved {out_path}")

    if deep_analysis:
        da_i, da_s, da_b = [], [], []
        for task, out in task_outs.items():
            ir, sr, br = _da_collect_task_ext(exp_name, task, out)
            da_i.extend(ir); da_s.extend(sr); da_b.append(br)
        return {"individual": da_i, "stacking": da_s, "base": da_b}
    return None


def apply_late_fusion(seed: int, alpha_grid: list[float]) -> None:
    a_path = cfg.external_results_dir("B") / f"ext_results_seed{seed}.xlsx"
    b_path = cfg.external_results_dir("A") / f"ext_results_seed{seed}.xlsx"
    if not a_path.exists() or not b_path.exists():
        print(f"[H TEST D] skip late fusion: need B_Organ and A_Patch external results first ({a_path}, {b_path})")
        return
    a = pd.read_excel(a_path, sheet_name="stacking_predictions", dtype={cfg.ID_COL: str})
    b = pd.read_excel(b_path, sheet_name="stacking_predictions", dtype={cfg.ID_COL: str})
    metric_rows = []
    pred_frames = []
    for task in TASKS:
        at = a[a["task"].astype(str) == task]
        bt = b[b["task"].astype(str) == task]
        merged = at[[cfg.ID_COL, "y_true", "y_score"]].merge(
            bt[[cfg.ID_COL, "y_true", "y_score"]], on=[cfg.ID_COL, "y_true"], suffixes=("_a", "_b"), how="inner"
        )
        if merged.empty:
            continue
        merged["a_norm"] = d_fusion._rank01(merged["y_score_a"])
        merged["b_norm"] = d_fusion._rank01(merged["y_score_b"])
        for alpha in alpha_grid:
            ys = alpha * merged["a_norm"].to_numpy() + (1.0 - alpha) * merged["b_norm"].to_numpy()
            yp = (ys >= 0.5).astype(int)
            yt = merged["y_true"]
            valid = yt.notna()
            row = {"experiment": "D_Late", "task": task, "alpha": float(alpha), "n": int(valid.sum())}
            if valid.sum() >= 2 and yt[valid].nunique() >= 2:
                row.update(d_fusion._metrics(yt[valid].astype(int).to_numpy(), ys[valid.to_numpy()], yp[valid.to_numpy()]))
            metric_rows.append(row)
            block = merged[[cfg.ID_COL, "y_true"]].copy()
            block["experiment"], block["task"], block["alpha"] = "D_Late", task, float(alpha)
            block["y_score"], block["y_pred"] = ys, yp
            pred_frames.append(block)

    out_dir = cfg.external_results_dir("D")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ext_results_seed{seed}.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        cfg.round_metrics(pd.DataFrame(metric_rows)).to_excel(writer, sheet_name="late_fusion_metrics", index=False)
        cfg.round_metrics(pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()).to_excel(writer, sheet_name="late_fusion_predictions", index=False)
    print(f"[H TEST] saved {out_path}")


def run_external_test(seed: int, requested: set[str], do_prepare: bool, alpha_grid: list[float], deep_analysis: bool = False) -> None:
    if do_prepare:
        prepare_p158_features(requested)
    for exp_name in ["B_Organ", "A_Patch", "C_Early"]:
        if exp_name in requested:
            result = apply_experiment(seed, exp_name, deep_analysis=deep_analysis)
            if deep_analysis and result and (result["individual"] or result["base"]):
                _da_update_analysis(
                    seed,
                    cfg.final_model_dir(EXPERIMENT_KEY[exp_name], seed),
                    result["individual"], result["stacking"], result["base"],
                )
    if "D_Late" in requested:
        apply_late_fusion(seed, alpha_grid)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="External P158 test: prepare ICC-stable P158 features and apply the "
                    "g_train-saved models per seed (no refit). Writes "
                    "<Experiment>/0N_external_testing/02_results/ext_results_seed{N}.xlsx."
    )
    parser.add_argument("--only", default=cfg.DEFAULT_EXTERNAL_EXPERIMENTS,
                        help="Comma-separated experiment keys: A, B, C, D.")
    parser.add_argument("--prepare", action="store_true", help="Build P158 feature tables before applying models (skips any already present).")
    parser.add_argument("--alpha-grid", default=cfg.DEFAULT_LATE_FUSION_ALPHA_GRID)
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEEDS[0])
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--no-deep-analysis", dest="deep_analysis", action="store_false",
                        help="Skip merging external metrics into each experiment's analysis.xlsx. "
                             "Deep analysis runs by default wherever valid.")
    args = parser.parse_args()

    print("[PREPROCESS] ensuring external P158 test data is preprocessed (resample + crop)")
    a_preprocess.preprocess_test_dataset()

    _key_to_name = {"A": "A_Patch", "B": "B_Organ", "C": "C_Early", "D": "D_Late"}
    requested = {_key_to_name[k.strip().upper()] for k in args.only.split(",") if k.strip().upper() in _key_to_name}
    alpha_grid = [float(x) for x in str(args.alpha_grid).split(",") if str(x).strip()]
    seeds = cfg.RANDOM_SEEDS if args.all_seeds else [args.seed]
    for seed in seeds:
        print(f"\n========== h_test seed={seed} ==========")
        run_external_test(
            seed=seed, requested=requested,
            do_prepare=bool(args.prepare),
            alpha_grid=alpha_grid, deep_analysis=bool(args.deep_analysis),
        )


if __name__ == "__main__":
    main()
