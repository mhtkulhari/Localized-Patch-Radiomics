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
import g_maxPool
from e_ml import compute_metrics, prepare_model_data

TASKS = tuple(cfg.CLASSICAL_TASKS)

EXPERIMENT_KEY = {"A_Patch": "A", "B_Organ": "B", "B_Organ_same": "B0", "C_Early": "C", "D_Late": "D"}

EXPERIMENT_P158_FEATURE_DIR = {
    "B_Organ": cfg.external_summary_dir("B"),
    "B_Organ_same": cfg.external_summary_dir("B0"),
    "A_Patch": cfg.external_summary_dir("A"),
    "C_Early": cfg.external_summary_dir("C"),
}
EXPERIMENT_P158_RESULTS_DIR = {
    "B_Organ": cfg.external_results_dir("B"),
    "B_Organ_same": cfg.external_results_dir("B0"),
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


def prepare_p158_organ_same_features() -> None:
    out_dir = cfg.external_summary_dir("B0")
    expected_ids = set(c_organ.p158_case_ids(cfg.P158_ROOT))

    def _file_complete(path: Path) -> bool:
        if not path.exists():
            return False
        got = set(pd.read_csv(path, dtype={cfg.ID_COL: str})[cfg.ID_COL].astype(str))
        return expected_ids.issubset(got)

    missing = [f for f in cfg.FEATURE_FILES if not _file_complete(out_dir / f)]
    if not missing:
        print(f"[P158 ORGAN SAME] external B_Organ_same features already exist: {out_dir}")
        return

    keep = c_organ.load_keep_features(cfg.ORGAN_SAME_RAW_DIR)
    labels = load_p158_labels()
    # P158 has one provided organ segmentation. Internal B0 uses AI2 after the
    # AI1/AI2 ICC filter, while external B0 must use the available test mask.
    tables = c_organ.extract_patchstyle_tables(cfg.P158_ROOT, cfg.MASK_FOLDER)
    books = c_organ.build_requested_books(labels, tables, keep, missing)
    for file_name, work in books.items():
        c_organ.write_book(out_dir / file_name, work, quiet=True)
    print(f"[P158 ORGAN SAME] saved external B_Organ_same features: {out_dir}")


def _patch_module_to_p158() -> None:
    b_patch.MAIN_ROOT = cfg.P158_ROOT
    b_patch.DATASET_XLSX = cfg.P158_DATASET_XLSX
    b_patch.PATCH_ONLY_WORKBOOK_DIR = cfg.external_summary_dir("A")
    b_patch.EARLY_FUSION_WORKBOOK_DIR = cfg.external_summary_dir("C")
    b_patch.PATCH_INFERENCE_RAW_PRED_DIR = cfg.external_detection_dir()
    b_patch.PATCH_APPLY_WORKERS = cfg.EXTERNAL_PATCH_APPLY_WORKERS


def prepare_p158_patch_and_fusion(build_early_fusion: bool = True) -> None:
    patch_out = cfg.external_summary_dir("A")
    early_out = cfg.external_summary_dir("C")
    organ_out = cfg.external_summary_dir("B")

    _patch_module_to_p158()
    patch_files = cfg.FEATURE_FILES
    early_files = cfg.FEATURE_FILES if build_early_fusion else []
    expected_patch_ids = {str(c).strip() for c in b_patch.list_case_ids(cfg.P158_ROOT)}

    patch_missing = [
        f for f in patch_files
        if not expected_patch_ids or not expected_patch_ids.issubset(cfg.patch_workbook_done_ids(patch_out / f))
    ]
    if not b_patch._all_maxpool_detection_caches_exist(sorted(expected_patch_ids)):
        patch_missing = list(patch_files)
    early_missing = [f for f in early_files if not (early_out / f).exists()] if build_early_fusion else []
    
    if not patch_missing and not early_missing:
        print(f"[P158 A/C] external patch/fusion features already exist: {patch_out}")
        return
    
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
    if "B_Organ_same" in requested:
        prepare_p158_organ_same_features()
    if requested & {"A_Patch", "C_Early"}:
        prepare_p158_patch_and_fusion(build_early_fusion="C_Early" in requested)


def _apply_experiment_task(bundle: dict, p158_feature_dir: Path) -> dict:
    config = bundle["config"]
    task = bundle["task"]
    exp = bundle["experiment"]
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
            yp = f_stack.score_to_pred(ys, f_stack.fixed_threshold(task))
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
    assembled = f_stack.assemble_meta_matrices(
        candidate_results=candidate_results, train_ids=train_ids, eval_ids=[], all_test_ids=p158_ids,
        y_train=y_train, y_eval=np.array([], dtype=int), task_name=task, outer_fold=0, config=config,
    )
    if not assembled["ok"]:
        raise RuntimeError(f"meta assembly failed for {exp}:{task}: {assembled.get('reason')}")
    meta = f_stack.fit_weighted_average(
        assembled["z_oof"], y_train, assembled["z_eval"], np.array([], dtype=int),
        assembled["z_all"], assembled["kept"], task, 0,
    )
    choice = meta["chosen"]
    y_score = np.asarray(choice["score_all"], dtype=float)
    y_pred = f_stack.score_to_pred(y_score, f_stack.fixed_threshold(task))

    used_method = choice["method"]
    final_rows = [
        {"experiment": exp, "task": task, "meta_method": used_method, cfg.ID_COL: cid, "y_true": y_true_map.get(cid, np.nan), "y_score": float(s), "y_pred": int(p)}
        for cid, s, p in zip(p158_ids, y_score, y_pred)
    ]
    yt_all = np.array([y_true_map.get(c, np.nan) for c in p158_ids])
    v_all = ~np.isnan(yt_all)
    weighted_metrics = (
        compute_metrics(yt_all[v_all].astype(int), y_score[v_all], y_pred[v_all])
        if v_all.sum() >= 2 and len(np.unique(yt_all[v_all])) >= 2
        else {}
    )

    return {
        "final": pd.DataFrame(final_rows),
        "individual_metrics": pd.DataFrame(indiv_metric_rows),
        "individual_predictions": pd.DataFrame(indiv_pred_rows),
        "weighted_metrics": pd.DataFrame([{
            "experiment": exp,
            "task": task,
            "meta_method": "weighted_average",
            **weighted_metrics,
        }]),
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


def apply_experiment(seed: int, exp_name: str) -> None:
    exp_key = EXPERIMENT_KEY[exp_name]
    feature_dir = Path(EXPERIMENT_P158_FEATURE_DIR[exp_name])
    results_dir = Path(EXPERIMENT_P158_RESULTS_DIR[exp_name])
    results_dir.mkdir(parents=True, exist_ok=True)

    final_frames, metric_rows, indiv_metrics, indiv_preds, weighted_rows = [], [], [], [], []
    zone_predictions = pd.DataFrame()
    for task in cfg.classical_tasks_for_experiment(exp_key):
        bundle_path = cfg.final_model_path(exp_key, task, seed)
        if not bundle_path.exists():
            raise FileNotFoundError(f"Missing trained model: {bundle_path}. Run h_train.py for seed {seed} first.")
        bundle = joblib.load(bundle_path)
        out = _apply_experiment_task(bundle, feature_dir)
        final_frames.append(out["final"])
        metric_rows.append(_metrics_row(exp_name, task, out["final"]))
        indiv_metrics.append(out["individual_metrics"])
        indiv_preds.append(out["individual_predictions"])
        weighted_rows.append(out["weighted_metrics"])

    if exp_name == "A_Patch":
        zone_metrics, zone_predictions = g_maxPool.predict_external(seed)
        metric_rows.extend(zone_metrics.to_dict("records"))

    indiv_df = pd.concat(indiv_metrics, ignore_index=True)
    if "auc" in indiv_df.columns:
        indiv_df = indiv_df.sort_values(["task", "auc"], ascending=[True, False], na_position="last").reset_index(drop=True)
    stack_df = pd.concat(weighted_rows, ignore_index=True)
    if "auc" in stack_df.columns:
        stack_df = stack_df.sort_values(["task", "auc"], ascending=[True, False], na_position="last").reset_index(drop=True)

    out_path = results_dir / f"ext_results_seed{seed}.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        cfg.round_metrics(pd.DataFrame(metric_rows)).to_excel(writer, sheet_name="metrics", index=False)
        cfg.round_metrics(pd.concat(final_frames, ignore_index=True)).to_excel(writer, sheet_name="stacking_predictions", index=False)
        if exp_name == "A_Patch":
            cfg.round_metrics(zone_predictions).to_excel(writer, sheet_name="zone_predictions", index=False)
        cfg.round_metrics(indiv_df).to_excel(writer, sheet_name="individual_metrics", index=False)
        cfg.round_metrics(pd.concat(indiv_preds, ignore_index=True)).to_excel(writer, sheet_name="individual_predictions", index=False)
        cfg.round_metrics(stack_df).to_excel(writer, sheet_name="weighted_metrics", index=False)
    print(f"[H TEST] saved {out_path}")


def apply_late_fusion(seed: int) -> None:
    alpha = float(cfg.LATE_FUSION_ALPHA)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    a_path = cfg.external_results_dir("A") / f"ext_results_seed{seed}.xlsx"
    b_path = cfg.external_results_dir("B") / f"ext_results_seed{seed}.xlsx"
    if not a_path.exists() or not b_path.exists():
        print(f"[H TEST D] skip late fusion: need A_Patch and B_Organ external results first ({a_path}, {b_path})")
        return
    a = pd.read_excel(a_path, sheet_name="stacking_predictions", dtype={cfg.ID_COL: str})
    b = pd.read_excel(b_path, sheet_name="stacking_predictions", dtype={cfg.ID_COL: str})
    try:
        a_zone = pd.read_excel(a_path, sheet_name="zone_predictions", dtype={cfg.ID_COL: str})
    except Exception:
        a_zone = pd.DataFrame()
    metric_rows = []
    pred_frames = []
    for task in TASKS:
        at = a_zone if task == "zone" else a[a["task"].astype(str) == task]
        bt = b[b["task"].astype(str) == task]
        if at.empty or bt.empty:
            raise RuntimeError(f"Missing external A/B predictions required for D_Late task={task}")
        merged = at[[cfg.ID_COL, "y_true", "y_score"]].merge(
            bt[[cfg.ID_COL, "y_true", "y_score"]], on=[cfg.ID_COL, "y_true"],
            suffixes=("_a", "_b"), how="inner", validate="one_to_one",
        )
        if merged.empty:
            raise RuntimeError(f"No matching external A/B prediction rows for D_Late task={task}")
        if len(merged) != len(at) or len(merged) != len(bt):
            raise RuntimeError(
                f"External A/B prediction coverage differs for D_Late task={task}: "
                f"A={len(at)} B={len(bt)} matched={len(merged)}"
            )
        merged["a_norm"] = d_fusion._rank01(merged["y_score_a"])
        merged["b_norm"] = d_fusion._rank01(merged["y_score_b"])
        ys = alpha * merged["a_norm"].to_numpy() + (1.0 - alpha) * merged["b_norm"].to_numpy()
        yp = (ys >= d_fusion._fixed_threshold(task)).astype(int)
        yt = merged["y_true"]
        valid = yt.notna()
        row = {
            "experiment": "D_Late", "task": task, "alpha": alpha,
            "alpha_source": "A_Patch", "n": int(valid.sum()),
        }
        if valid.sum() >= 2 and yt[valid].nunique() >= 2:
            row.update(d_fusion._metrics(yt[valid].astype(int).to_numpy(), ys[valid.to_numpy()], yp[valid.to_numpy()]))
        metric_rows.append(row)
        block = merged[[cfg.ID_COL, "y_true"]].copy()
        block["experiment"], block["task"], block["alpha"] = "D_Late", task, alpha
        block["alpha_source"] = "A_Patch"
        block["a_score"] = merged["y_score_a"]
        block["b_score"] = merged["y_score_b"]
        block["a_score_norm"] = merged["a_norm"]
        block["b_score_norm"] = merged["b_norm"]
        block["y_score"], block["y_pred"] = ys, yp
        pred_frames.append(block)

    out_dir = cfg.external_results_dir("D")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ext_results_seed{seed}.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        cfg.round_metrics(pd.DataFrame(metric_rows)).to_excel(writer, sheet_name="late_fusion_metrics", index=False)
        cfg.round_metrics(pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()).to_excel(writer, sheet_name="late_fusion_predictions", index=False)
    print(f"[H TEST] saved {out_path}")


def _external_result_ready(seed: int, exp_name: str) -> bool:
    exp_key = EXPERIMENT_KEY[exp_name]
    path = cfg.external_results_dir(exp_key) / f"ext_results_seed{seed}.xlsx"
    if not path.exists():
        return False
    try:
        xl = pd.ExcelFile(path)
        if exp_name == "D_Late":
            required = {"late_fusion_metrics", "late_fusion_predictions"}
            if not required.issubset(xl.sheet_names):
                return False
            metrics = pd.read_excel(path, sheet_name="late_fusion_metrics")
            predictions = pd.read_excel(path, sheet_name="late_fusion_predictions")
            alphas = set(pd.to_numeric(metrics.get("alpha", pd.Series(dtype=float)), errors="coerce").dropna().round(8))
            alpha_sources = set(metrics.get("alpha_source", pd.Series(dtype=str)).dropna().astype(str))
            if alphas != {round(float(cfg.LATE_FUSION_ALPHA), 8)} or alpha_sources != {"A_Patch"}:
                return False
        else:
            required = {
                "metrics", "stacking_predictions", "individual_metrics",
                "individual_predictions", "weighted_metrics",
            }
            if exp_name == "A_Patch":
                required.add("zone_predictions")
            if not required.issubset(xl.sheet_names):
                return False
            metrics = pd.read_excel(path, sheet_name="metrics")
            predictions = pd.read_excel(path, sheet_name="stacking_predictions")
            if exp_name == "A_Patch":
                zone_predictions = pd.read_excel(path, sheet_name="zone_predictions")
                if zone_predictions.empty:
                    return False
        tasks = set(metrics.get("task", pd.Series(dtype=str)).astype(str))
        return not predictions.empty and set(TASKS).issubset(tasks)
    except Exception:
        return False


def run_external_test(seed: int, requested: set[str]) -> None:
    prepare_p158_features(requested)
    for exp_name in ["B_Organ", "B_Organ_same", "A_Patch", "C_Early"]:
        if exp_name in requested:
            if _external_result_ready(seed, exp_name):
                print(f"[CACHE] {exp_name} external result exists; skipping apply")
            else:
                apply_experiment(seed, exp_name)
    if "D_Late" in requested:
        if _external_result_ready(seed, "D_Late"):
            print("[CACHE] D_Late external result exists; skipping fusion")
        else:
            apply_late_fusion(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="External P158 test: prepare ICC-stable P158 features and apply the "
                    "h_train-saved models per seed (no refit). Writes "
                    "<Experiment>/0N_external_testing/02_results/ext_results_seed{N}.xlsx."
    )
    parser.add_argument("--only", default=cfg.DEFAULT_EXTERNAL_EXPERIMENTS,
                        help="Comma-separated experiment keys: A, B, B0, C, D.")
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_STATE)
    parser.add_argument("--all-seeds", action="store_true")
    args = parser.parse_args()

    _key_to_name = {"A": "A_Patch", "B": "B_Organ", "B0": "B_Organ_same", "C": "C_Early", "D": "D_Late"}
    requested_keys = {k.strip().upper() for k in args.only.split(",") if k.strip()}
    invalid = requested_keys - set(_key_to_name)
    if invalid:
        raise ValueError(f"Unknown experiments in --only: {sorted(invalid)}. Choices: {sorted(_key_to_name)}")
    requested = {_key_to_name[key] for key in requested_keys}
    print("[PREPROCESS] ensuring external P158 test data is preprocessed (resample + crop)")
    a_preprocess.preprocess_test_dataset()
    seeds = cfg.RANDOM_SEEDS if args.all_seeds else [args.seed]
    for seed in seeds:
        print(f"\n========== i_test seed={seed} ==========")
        run_external_test(seed=seed, requested=requested)


if __name__ == "__main__":
    main()
