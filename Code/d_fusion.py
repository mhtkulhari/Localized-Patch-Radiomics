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
    ZONE_COL,
    FEATURE_FILES,
    ORGAN_ONLY_WORKBOOK_DIR,
    PATCH_ONLY_WORKBOOK_DIR,
    EARLY_FUSION_WORKBOOK_DIR,
    RANDOM_SEEDS,
    results_dirs_for_seed,
    read_features,
    read_predictions,
    write_table,
    write_predictions,
)
import all_config as cfg

LATE_KEY_COLS = ["file", "fs_method", "clf_model", "alpha"]
LATE_METRICS = ["auc", "f1", "sensitivity", "specificity", "accuracy"]
FINAL_REPORT_NAME = "ml_top10.txt"


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


def _read_predictions(results_dir: Path, file_tag: str, task: str) -> pd.DataFrame:
    path = Path(results_dir) / file_tag / f"predictions_{task}_{file_tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required prediction table for late fusion: {path}")
    df = read_predictions(path)
    if df.empty:
        raise RuntimeError(f"Prediction table has no rows for late fusion: {path}")
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


def _summarize_late_fusion(fold_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame()
    group_cols = ["file", "fs_method", "clf_model", "alpha"]
    rows = []
    for keys, group in fold_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_outer_folds"] = int(group["outer_fold"].nunique())
        for metric in ["auc", "ap", "balanced_acc", "f1", "accuracy", "sensitivity", "specificity"]:
            vals = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = round(float(vals.mean()), 4)
            row[f"{metric}_std"] = round(float(vals.std(ddof=0)), 4)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["auc_mean", "f1_mean", "accuracy_mean"], ascending=[False, False, False]).reset_index(drop=True)


def _write_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
        cfg.format_float_cells_4dp(writer)


def _minimal_late_summary(summary_df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    out_cols = LATE_KEY_COLS + [f"{metric}_{suffix}" for metric in LATE_METRICS]
    if summary_df is None or summary_df.empty:
        return pd.DataFrame(columns=out_cols)
    needed = LATE_KEY_COLS + [f"{metric}_mean" for metric in LATE_METRICS]
    work = summary_df.copy()
    for col in needed:
        if col not in work.columns:
            work[col] = np.nan
    out = work[needed].copy()
    out = out.rename(columns={f"{metric}_mean": f"{metric}_{suffix}" for metric in LATE_METRICS})
    return out[out_cols]


def _sort_combo(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    sort_cols = [col for col in metric_cols if col in df.columns] + [col for col in LATE_KEY_COLS if col in df.columns]
    ascending = [False] * len([col for col in metric_cols if col in df.columns])
    ascending.extend([True] * (len(sort_cols) - len(ascending)))
    return df.sort_values(sort_cols, ascending=ascending, kind="stable").reset_index(drop=True)


def _write_late_fusion_combined(cs_summary: pd.DataFrame, zone_summary: pd.DataFrame) -> pd.DataFrame:
    cs_min = _minimal_late_summary(cs_summary, "cs")
    zone_min = _minimal_late_summary(zone_summary, "zone")
    merged = cs_min.merge(zone_min, on=LATE_KEY_COLS, how="inner")
    if merged.empty:
        combo = pd.DataFrame(columns=LATE_KEY_COLS + LATE_METRICS)
    else:
        combo = merged[LATE_KEY_COLS].copy()
        for metric in LATE_METRICS:
            combo[metric] = ((merged[f"{metric}_cs"] + merged[f"{metric}_zone"]) / 2.0).round(4)
        combo = _sort_combo(combo, ["auc", "f1", "accuracy"])
    return combo


def _top_report_df(df: pd.DataFrame, metric_cols: list[str], top_n: int = 10) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    key_cols = [col for col in LATE_KEY_COLS if col in work.columns]
    cols = key_cols + [col for col in metric_cols if col in work.columns]
    if "n_outer_folds" in work.columns:
        cols.append("n_outer_folds")
    cols = list(dict.fromkeys(cols))
    sort_cols = [col for col in metric_cols if col in work.columns]
    return work.sort_values(
        sort_cols + key_cols,
        ascending=[False] * len(sort_cols) + [True] * len(key_cols),
        kind="stable",
    )[cols].head(top_n).reset_index(drop=True)


def _write_top10_report_section(lines: list[str], title: str, df: pd.DataFrame) -> None:
    lines.append(title)
    if df is None or df.empty:
        lines.append("No rows.")
    else:
        lines.append(df.to_string(index=False))
    lines.append("")


def _write_late_fusion_top10_report(report_txt: Path, alpha_grid: list[float], cs_summaries, zone_summaries, combo_summaries) -> None:
    cs_all = pd.concat([df for df in cs_summaries if df is not None and not df.empty], ignore_index=True) if cs_summaries else pd.DataFrame()
    zone_all = pd.concat([df for df in zone_summaries if df is not None and not df.empty], ignore_index=True) if zone_summaries else pd.DataFrame()
    combo_all = pd.concat([df for df in combo_summaries if df is not None and not df.empty], ignore_index=True) if combo_summaries else pd.DataFrame()
    task_metrics = ["auc_mean", "f1_mean", "sensitivity_mean", "specificity_mean", "accuracy_mean"]
    combo_metrics = ["auc", "f1", "sensitivity", "specificity", "accuracy"]
    lines = [
        "================ FINAL TOP-10 REPORT ================",
        "Experiment: D_Late",
        "Alpha grid: " + ", ".join(str(x) for x in alpha_grid),
        "",
    ]
    _write_top10_report_section(lines, "[CS summary]", _top_report_df(cs_all, task_metrics))
    _write_top10_report_section(lines, "[ZONE summary]", _top_report_df(zone_all, task_metrics))
    _write_top10_report_section(lines, "[CS/ZONE average summary]", _top_report_df(combo_all, combo_metrics))
    report_txt.parent.mkdir(parents=True, exist_ok=True)
    report_txt.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def compute_late_fusion(
    alpha_grid: list[float],
    organ_results_dir: Path,
    patch_results_dir: Path,
    late_results_dir: Path,
    file_names: list[str] | None = None,
) -> list[dict]:
    organ_results_dir = Path(organ_results_dir)
    patch_results_dir = Path(patch_results_dir)
    late_results_dir = Path(late_results_dir)
    late_results_dir.mkdir(parents=True, exist_ok=True)
    names = list(FEATURE_FILES if file_names is None else file_names)
    report_txt = late_results_dir / FINAL_REPORT_NAME
    all_reports = []
    report_cs_summaries = []
    report_zone_summaries = []
    report_combo_summaries = []

    for file_name in names:
        file_tag = Path(file_name).stem
        file_dir = late_results_dir / file_tag
        file_dir.mkdir(parents=True, exist_ok=True)
        task_summaries = {}
        for task in ["cs", "zone"]:
            organ = _read_predictions(organ_results_dir, file_tag, task)
            patch = _read_predictions(patch_results_dir, file_tag, task)
            key_cols = ["file", "fs_method", "clf_model", "outer_fold", "case_id"]
            keep = key_cols + ["y_true", "y_score"]
            merged = organ[keep].merge(
                patch[keep],
                on=key_cols + ["y_true"],
                suffixes=("_organ", "_patch"),
                how="inner",
            )
            if merged.empty:
                raise RuntimeError(f"No matching A/B prediction rows for late fusion: {file_tag}:{task}")
            pred_parts = []
            norm_group = ["file", "fs_method", "clf_model", "outer_fold"]
            merged["organ_score_norm"] = merged.groupby(norm_group)["y_score_organ"].transform(_rank01)
            merged["patch_score_norm"] = merged.groupby(norm_group)["y_score_patch"].transform(_rank01)
            for alpha in alpha_grid:
                d = merged.copy()
                d["alpha"] = float(alpha)
                d["y_score"] = alpha * d["organ_score_norm"] + (1.0 - alpha) * d["patch_score_norm"]
                d["y_pred"] = (d["y_score"] >= 0.5).astype(int)
                pred_parts.append(d)
            pred_df = pd.concat(pred_parts, ignore_index=True)
            fold_rows = []
            fold_group_cols = ["file", "fs_method", "clf_model", "alpha", "outer_fold"]
            for keys, group in pred_df.groupby(fold_group_cols, dropna=False):
                row = dict(zip(fold_group_cols, keys))
                row.update(_metrics(group["y_true"].to_numpy(), group["y_score"].to_numpy(), group["y_pred"].to_numpy()))
                row["n_test"] = int(len(group))
                fold_rows.append(row)
            fold_df = pd.DataFrame(fold_rows)
            summary_df = _summarize_late_fusion(fold_df)
            task_summaries[task] = summary_df
            out_book = file_dir / f"results_{task}_{file_tag}.xlsx"
            _write_excel(out_book, {"combo_summary": cfg.collapse_mean_std_columns(cfg.round_metrics(summary_df)), "fold_metrics": cfg.round_metrics(fold_df)})
            write_predictions(pred_df, file_dir / f"predictions_{task}_{file_tag}.csv")
            all_reports.append({"file": file_tag, "task": task, "prediction_rows": int(len(pred_df)), "fold_rows": int(len(fold_df)), "summary_rows": int(len(summary_df)), "path": str(out_book)})
            print(f"[D LATE] saved {out_book}")
        combined_df = _write_late_fusion_combined(task_summaries.get("cs", pd.DataFrame()), task_summaries.get("zone", pd.DataFrame()))
        report_cs_summaries.append(task_summaries.get("cs", pd.DataFrame()))
        report_zone_summaries.append(task_summaries.get("zone", pd.DataFrame()))
        report_combo_summaries.append(combined_df)

    _write_late_fusion_top10_report(report_txt, alpha_grid, report_cs_summaries, report_zone_summaries, report_combo_summaries)
    return all_reports


def run_late_fusion_for_seed(seed: int, alpha_grid: list[float]) -> None:
    rd = results_dirs_for_seed(seed)
    compute_late_fusion(alpha_grid, rd["organ"], rd["patch"], rd["late_fusion"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fusion stage. --stage early builds C_Early feature workbooks "
                    "(organ + patch features merged). --stage late blends organ and patch "
                    "predictions (per seed) into D_Late results."
    )
    parser.add_argument("--stage", choices=["early", "late"], required=True)
    parser.add_argument("--seed", type=int, default=RANDOM_SEEDS[0])
    parser.add_argument("--alpha-grid", default=cfg.DEFAULT_LATE_FUSION_ALPHA_GRID)
    args = parser.parse_args()

    if args.stage == "early":
        build_early_fusion_workbooks()
    else:
        alpha_grid = [float(x) for x in str(args.alpha_grid).split(",") if str(x).strip()]
        run_late_fusion_for_seed(args.seed, alpha_grid)


if __name__ == "__main__":
    main()
