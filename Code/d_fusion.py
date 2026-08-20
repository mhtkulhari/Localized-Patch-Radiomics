from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
)

from all_config import (
    ID_COL,
    CLIN_COL,
    CLASSICAL_TASKS,
    ZONE_COL,
    FEATURE_FILES,
    ORGAN_ONLY_WORKBOOK_DIR,
    PATCH_ONLY_WORKBOOK_DIR,
    EARLY_FUSION_WORKBOOK_DIR,
    RANDOM_SEEDS,
    RANDOM_STATE,
    results_dirs_for_seed,
    read_features,
    write_table,
)
import all_config as cfg


def _early_fusion_sheet(main_df: pd.DataFrame, helper: pd.DataFrame) -> pd.DataFrame:
    df = main_df.copy()
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    duplicate_helper_cols = [c for c in helper.columns if c != ID_COL and c in df.columns]
    return df.merge(helper.drop(columns=duplicate_helper_cols), on=ID_COL, how="left")


def build_early_fusion_workbooks(
    organ_dir: Path = ORGAN_ONLY_WORKBOOK_DIR,
    patch_dir: Path = PATCH_ONLY_WORKBOOK_DIR,
    out_dir: Path = EARLY_FUSION_WORKBOOK_DIR,
    file_names: list[str] | None = None,
) -> list[dict]:
    organ_dir = Path(organ_dir)
    patch_dir = Path(patch_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(FEATURE_FILES if file_names is None else file_names)
    report = []
    completed = 0
    total = len(names)
    for file_name in names:
        completed += 1
        cfg.log_progress("EARLY_FUSION", completed, total)
        src = organ_dir / file_name
        patch_dst = patch_dir / file_name
        if not src.exists():
            print(f"[C EARLY] skip missing organ workbook used for labels: {src}")
            continue
        if not patch_dst.exists():
            raise FileNotFoundError(f"Missing patch workbook needed for early fusion: {patch_dst}. Run the patch apply stage first.")
        patch_df = read_features(patch_dst)
        patch_df[ID_COL] = patch_df[ID_COL].astype(str).str.strip()
        patch_features = patch_df.drop(columns=[CLIN_COL, ZONE_COL], errors="ignore")
        main_df = read_features(src)
        main_df[ID_COL] = main_df[ID_COL].astype(str).str.strip()
        early_df = _early_fusion_sheet(main_df, patch_features)
        early_dst = out_dir / file_name
        write_table(early_df, early_dst)
        report.append({
            "experiment": "C_Early",
            "file": file_name,
            "rows": int(len(early_df)),
            "features": int(max(0, early_df.shape[1] - 3)),
            "path": str(early_dst),
        })
        print(f"[C EARLY] saved {early_dst} rows={len(early_df)} features={early_df.shape[1] - 3}")
    return report


def _read_maxpool_zone_predictions(patch_results_dir: Path) -> pd.DataFrame:
    path = Path(patch_results_dir) / "maxpool_zone.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Missing A_Patch max-pool zone predictions for late fusion: {path}")
    df = pd.read_excel(path, sheet_name="predictions", dtype={ID_COL: str})
    if df.empty:
        raise RuntimeError(f"A_Patch max-pool zone prediction sheet has no rows: {path}")
    return df


def _rank01(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    if x.notna().sum() <= 1:
        return pd.Series(np.full(len(x), 0.5), index=x.index)
    ranks = x.rank(method="average", na_option="keep")
    out = (ranks - 1.0) / max(1.0, float(x.notna().sum() - 1))
    return out.fillna(0.5)


def _metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) >= 2:
        auc = float(roc_auc_score(y_true, y_score))
        ap = float(average_precision_score(y_true, y_score))
    else:
        auc = np.nan
        ap = np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) else np.nan
    spec = float(tn / (tn + fp)) if (tn + fp) else np.nan
    return {
        "auc": round(auc, 4),
        "ap": round(ap, 4),
        "balanced_acc": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
    }


def _fixed_threshold(task: str) -> float:
    return 0.45 if str(task).lower() == "zone" else 0.5


def _read_final_stacking_predictions(results_dir: Path, task: str) -> pd.DataFrame:
    path = Path(results_dir) / "final_stacking.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Missing final stacking predictions required for late fusion: {path}")
    df = pd.read_excel(path, sheet_name="meta_predictions", dtype={ID_COL: str})
    required = {"task", "outer_fold", ID_COL, "y_true", "y_score"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Final stacking prediction sheet has missing columns at {path}: {sorted(required - set(df.columns))}")
    out = df[df["task"].astype(str) == task].copy()
    if out.empty:
        raise RuntimeError(f"No final stacking predictions for task={task} at {path}")
    return out


def compute_late_fusion(
    patch_results_dir: Path,
    organ_results_dir: Path,
    late_results_dir: Path,
) -> list[dict]:
    """Fuse final A_Patch and B_Organ out-of-fold prediction scores."""
    import f_stack

    alpha = float(cfg.LATE_FUSION_ALPHA)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    patch_results_dir = Path(patch_results_dir)
    organ_results_dir = Path(organ_results_dir)
    late_results_dir = Path(late_results_dir)

    fold_rows: list[dict] = []
    pred_rows: list[dict] = []
    selected_rows: list[dict] = []
    base_outer_rows: list[dict] = []
    method_rows: list[dict] = []

    for task in CLASSICAL_TASKS:
        if task == "zone":
            patch = _read_maxpool_zone_predictions(patch_results_dir)
        else:
            patch = _read_final_stacking_predictions(patch_results_dir, task)
        organ = _read_final_stacking_predictions(organ_results_dir, task)

        keep = ["outer_fold", ID_COL, "y_true", "y_score"]
        merged = patch[keep].merge(
            organ[keep],
            on=["outer_fold", ID_COL, "y_true"],
            suffixes=("_a", "_b"),
            how="inner",
            validate="one_to_one",
        )
        if merged.empty:
            raise RuntimeError(f"No matching final A/B prediction rows for late fusion task={task}")
        if len(merged) != len(patch) or len(merged) != len(organ):
            raise RuntimeError(
                f"A/B final prediction coverage differs for task={task}: "
                f"A={len(patch)} B={len(organ)} matched={len(merged)}"
            )
        merged["a_score_norm"] = merged.groupby("outer_fold")["y_score_a"].transform(_rank01)
        merged["b_score_norm"] = merged.groupby("outer_fold")["y_score_b"].transform(_rank01)
        merged["y_score"] = alpha * merged["a_score_norm"] + (1.0 - alpha) * merged["b_score_norm"]
        merged["y_pred"] = (merged["y_score"] >= _fixed_threshold(task)).astype(int)

        for outer_fold, fold in merged.groupby("outer_fold", sort=True):
            outer_fold = int(outer_fold)
            fusion_metrics = _metrics(fold["y_true"], fold["y_score"], fold["y_pred"])
            fold_rows.append({
                "task": task,
                "outer_fold": outer_fold,
                "meta_method": "equal_score_fusion",
                "alpha": alpha,
                "alpha_source": "A_Patch",
                "n_candidates_used": 2,
                "n_test": int(len(fold)),
                **fusion_metrics,
            })
            method_rows.append({
                "task": task,
                "outer_fold": outer_fold,
                "meta_method": "equal_score_fusion",
                "complexity": 1,
                "n_meta_features": 2,
                "alpha": alpha,
                **{f"outer_{key}": value for key, value in fusion_metrics.items()},
            })
            for rank, (source, score_col) in enumerate(
                (("A_Patch", "a_score_norm"), ("B_Organ", "b_score_norm")), start=1
            ):
                source_metrics = _metrics(
                    fold["y_true"], fold[score_col], (fold[score_col] >= _fixed_threshold(task)).astype(int)
                )
                selected_rows.append({
                    "source": source,
                    "task": task,
                    "outer_fold": outer_fold,
                    "candidate_rank": rank,
                    "candidate_id": f"{source}::{task}::{outer_fold}::final_score",
                    "status": "used",
                    "score_source": "final_prediction_score",
                    "fusion_weight": alpha if source == "A_Patch" else 1.0 - alpha,
                })
                base_outer_rows.append({
                    "source": source,
                    "task": task,
                    "outer_fold": outer_fold,
                    "candidate_rank": rank,
                    "candidate_id": f"{source}::{task}::{outer_fold}::final_score",
                    "n_test": int(len(fold)),
                    **{f"outer_{key}": value for key, value in source_metrics.items()},
                })

            pred_rows.extend({
                "task": task,
                "outer_fold": outer_fold,
                ID_COL: str(row[ID_COL]),
                "y_true": int(row["y_true"]),
                "y_score": float(row["y_score"]),
                "y_pred": int(row["y_pred"]),
                "meta_method": "equal_score_fusion",
                "alpha": alpha,
                "alpha_source": "A_Patch",
                "a_score": float(row["y_score_a"]),
                "b_score": float(row["y_score_b"]),
                "a_score_norm": float(row["a_score_norm"]),
                "b_score_norm": float(row["b_score_norm"]),
            } for _, row in fold.iterrows())

    pooled_df, mean_outer_df = f_stack.build_overall_summary(fold_rows, pred_rows)
    sheets = {
        "mean_outer": mean_outer_df,
        "pooled": pooled_df,
        "meta_fold_metrics": pd.DataFrame(fold_rows),
        "selected_candidates": pd.DataFrame(selected_rows),
        "meta_predictions": pd.DataFrame(pred_rows),
        "base_oof_metrics": pd.DataFrame(),
        "base_outer_metrics": pd.DataFrame(base_outer_rows),
        "meta_method_perf": pd.DataFrame(method_rows),
    }
    out_path = late_results_dir / "final_stacking.xlsx"
    f_stack.write_stacking_workbook(
        str(out_path), {name: cfg.round_metrics(frame) for name, frame in sheets.items()}
    )
    print(f"[D LATE] saved final-score fusion with alpha(A_Patch)={alpha}: {out_path}")
    return [{"task": task, "alpha": alpha, "path": str(out_path)} for task in CLASSICAL_TASKS]


def run_late_fusion_for_seed(seed: int) -> None:
    rd = results_dirs_for_seed(seed)
    compute_late_fusion(rd["patch"], rd["organ"], rd["late_fusion"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fusion stage. --stage early builds C_Early feature workbooks "
                    "(organ + patch features merged). --stage late blends organ and patch "
                    "predictions (per seed) into D_Late results."
    )
    parser.add_argument("--stage", choices=["early", "late"], required=True)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    if args.stage == "early":
        build_early_fusion_workbooks()
    else:
        run_late_fusion_for_seed(args.seed)


if __name__ == "__main__":
    main()
