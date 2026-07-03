from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

import a_preprocess
import d_fusion

from all_config import (
    COMPLETION_MARKER,
    DEFAULT_LATE_FUSION_ALPHA_GRID,
    DEFAULT_PIPELINE_EXPERIMENTS,
    DEFAULT_STACKING_TOP_K,
    EARLY_FUSION_RESULTS_DIR,
    EARLY_FUSION_WORKBOOK_DIR,
    FEATURE_FILES,
    LATE_FUSION_RESULTS_DIR,
    ML_CLASSIFIERS,
    ML_FEATURE_SELECTORS,
    ORGAN_SAME_WORKBOOK_DIR,
    ORGAN_SAME_RESULTS_DIR,
    ORGAN_ONLY_RESULTS_DIR,
    ORGAN_ONLY_WORKBOOK_DIR,
    P158_DATASET_XLSX,
    TEST_ROOT,
    PATCH_FEATURE_SETS,
    PATCH_HELPER_MODEL_DIR,
    PATCH_SCALES,
    PATCH_ONLY_RESULTS_DIR,
    PATCH_ONLY_WORKBOOK_DIR,
    external_results_dir as _external_results_dir,
    final_model_path as _final_model_path,
    RANDOM_SEEDS,
    DATASET_XLSX,
    ID_COL,
    CLIN_COL,
    ZONE_COL,
    results_dir as _results_dir,
    split_json_name as _split_json_name,
)


PYTHON = sys.executable
RUN_LOG_NAME = "run_log.txt"
FINAL_REPORT_NAME = "ml_top10.txt"
LEGACY_DONE_MARKER = "_DONE.txt"


def run_cmd(cmd: list[str]) -> None:
    print("\n" + "#" * 72 + "\n")
    print("[RUN] " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(SCRIPT_DIR.parent))


def all_exist(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def _split_json_for_experiment(exp_key: str, seed: int) -> Path:
    return _results_dir(exp_key, seed) / _split_json_name(seed)


def _final_model_paths_for_seed(seed: int, exp_keys: list[str]) -> list[Path]:
    paths: list[Path] = []
    for exp_key in exp_keys:
        if exp_key not in {"A", "B", "C"}:
            continue
        for task in ("cs", "zone"):
            paths.append(_final_model_path(exp_key, task, seed))
    return paths


def _external_result_paths_for_seed(seed: int, exp_keys: list[str]) -> list[Path]:
    return [_external_results_dir(exp_key) / f"ext_results_seed{seed}.xlsx" for exp_key in exp_keys]


def workbook_files(folder: Path, file_names: list[str] | None = None) -> list[Path]:
    names = FEATURE_FILES if file_names is None else file_names
    return [folder / name for name in names]


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def log_line(results_dir: Path, message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    append_text(results_dir / RUN_LOG_NAME, f"{stamp} | {message}")


def completion_logged(results_dir: Path, name: str) -> bool:
    log_path = results_dir / RUN_LOG_NAME
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return f"{COMPLETION_MARKER} {name} pipeline finished" in text


def mark_complete(results_dir: Path, name: str) -> None:
    log_line(results_dir, f"{COMPLETION_MARKER} {name} pipeline finished")


def _file_tags(file_names: list[str]) -> list[str]:
    return [Path(name).stem for name in file_names]


def _expected_outer_folds(split_json_path: Path, default: int = 4) -> set[int]:
    path = split_json_path
    try:
        if path and Path(path).exists():
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            outer_splits = payload.get("outer_splits", [])
            folds = {int(item["outer_fold"]) for item in outer_splits if "outer_fold" in item}
            if folds:
                return folds
    except Exception:
        pass
    return set(range(1, int(default) + 1))


def _load_outer_splits(split_json_path: Path) -> list[dict]:
    path = split_json_path
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        outer_splits = payload.get("outer_splits", [])
        if isinstance(outer_splits, list) and outer_splits:
            return outer_splits
    except Exception as exc:
        print(f"[CACHE] could not read split JSON for coverage check: {path} ({repr(exc)})")
    return []


def _clean_case_id_for_verify(v) -> str | None:
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null", "<na>", "na"}:
        return None
    return s


def _parse_clin_for_verify(v):
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return 1
    if s in {"false", "0", "no", "n", "f"}:
        return 0
    return None


def _parse_zone_for_verify(v):
    if pd.isna(v):
        return None
    s = str(v).strip().upper()
    if s in {"PZ", "0"}:
        return "PZ"
    if s in {"TZ", "1"}:
        return "TZ"
    return None


def _load_reference_label_maps() -> dict[str, dict[str, object]]:
    try:
        df = pd.read_excel(DATASET_XLSX, dtype={ID_COL: str})
    except Exception as exc:
        print(f"[CACHE] could not read reference labels for coverage check: {DATASET_XLSX} ({repr(exc)})")
        return {"cs": {}, "zone": {}, "y3": {}}

    df.columns = [str(c).strip() for c in df.columns]
    if ID_COL not in df.columns:
        lowered = {str(c).strip().lower(): c for c in df.columns}
        if ID_COL.lower() not in lowered:
            print(f"[CACHE] reference label file missing case_id for coverage check: {DATASET_XLSX}")
            return {"cs": {}, "zone": {}, "y3": {}}
        df = df.rename(columns={lowered[ID_COL.lower()]: ID_COL})

    needed = {ID_COL, CLIN_COL, ZONE_COL}
    if not needed.issubset(df.columns):
        print(f"[CACHE] reference label file missing columns for coverage check: {sorted(needed - set(df.columns))}")
        return {"cs": {}, "zone": {}, "y3": {}}


    df[ID_COL] = df[ID_COL].map(_clean_case_id_for_verify)
    df = df.dropna(subset=[ID_COL])

    cs_map: dict[str, int] = {}
    zone_map: dict[str, int] = {}
    y3_map: dict[str, str] = {}
    for _, row in df.drop_duplicates(subset=[ID_COL]).iterrows():
        cid = _clean_case_id_for_verify(row[ID_COL])
        clin = _parse_clin_for_verify(row[CLIN_COL])
        zone = _parse_zone_for_verify(row[ZONE_COL])
        if cid is None or clin is None:
            continue
        cs_map[cid] = int(clin)
        if clin == 0:
            y3_map[cid] = "FALSE"
        elif clin == 1 and zone in {"PZ", "TZ"}:
            zone_map[cid] = 1 if zone == "TZ" else 0
            y3_map[cid] = "TRUE_TZ" if zone == "TZ" else "TRUE_PZ"
    return {"cs": cs_map, "zone": zone_map, "y3": y3_map}


def _expected_ids_for_outer(outer: dict, task: str, label_maps: dict[str, dict[str, object]]) -> tuple[set[str], set[str], set[str]]:
    train_ids = [cid for cid in (_clean_case_id_for_verify(x) for x in outer.get("train_case_ids", [])) if cid is not None]
    test_ids = [cid for cid in (_clean_case_id_for_verify(x) for x in outer.get("test_case_ids", [])) if cid is not None]
    task_map = label_maps.get(task, {})
    y3_map = label_maps.get("y3", {})
    expected_train = {cid for cid in train_ids if cid in task_map}
    expected_eval = {cid for cid in test_ids if cid in task_map}
    expected_all_test = {cid for cid in test_ids if cid in y3_map}
    return expected_train, expected_eval, expected_all_test


def _assert_inner_oof_split_coverage_complete(
    outer_splits: list[dict],
    label_maps: dict[str, dict[str, object]],
    expected_folds: set[int],
) -> None:
    errors: list[str] = []
    for outer in outer_splits:
        outer_fold = int(outer.get("outer_fold", -1))
        if outer_fold not in expected_folds:
            continue
        outer_train_all = {
            cid for cid in (_clean_case_id_for_verify(x) for x in outer.get("train_case_ids", [])) if cid is not None
        }
        inner_val_union = {
            cid
            for inner in outer.get("inner_splits", [])
            for cid in (_clean_case_id_for_verify(x) for x in inner.get("val_case_ids", []))
            if cid is not None
        }
        extra_inner = sorted(inner_val_union - outer_train_all)
        if extra_inner:
            errors.append(
                f"outer={outer_fold} inner validation has {len(extra_inner)} IDs outside outer train; "
                f"examples={extra_inner[:8]}"
            )
        for task in ("cs", "zone"):
            task_map = label_maps.get(task, {})
            expected_train = {cid for cid in outer_train_all if cid in task_map}
            covered_train = {cid for cid in inner_val_union if cid in task_map}
            missing = sorted(expected_train - covered_train)
            if missing:
                errors.append(
                    f"task={task} outer={outer_fold} inner_val_union missing "
                    f"{len(missing)}/{len(expected_train)} outer-train IDs; examples={missing[:8]}"
                )
    if errors:
        msg = "[CACHE] split JSON cannot support complete stacking OOF coverage; " + " | ".join(errors[:8])
        print(msg)
        raise RuntimeError("split_oof_coverage_incomplete: " + " | ".join(errors[:8]))


def _read_workbook_case_ids(workbook_dir: Path, file_tag: str) -> set[str] | None:
    path = workbook_dir / f"{file_tag}.csv"
    if not path.exists():
        print(f"[CACHE] stacking coverage check missing feature workbook: {path}")
        return None
    try:
        df = pd.read_csv(path, dtype={ID_COL: str})
    except Exception as exc:
        print(f"[CACHE] stacking coverage check cannot read {path.name}: {repr(exc)}")
        return None
    if ID_COL not in df.columns:
        print(f"[CACHE] stacking coverage check {path.name} missing {ID_COL}")
        return None
    ids = {_clean_case_id_for_verify(x) for x in df[ID_COL].tolist()}
    return {cid for cid in ids if cid is not None}


def _stacking_feature_coverage_complete(
    workbook_dir: Path,
    file_names: list[str],
    expected_outer_folds: set[int] | None = None,
    split_json_path: Path | None = None,
) -> bool:
    outer_splits = _load_outer_splits(split_json_path)
    label_maps = _load_reference_label_maps()
    if not outer_splits or not label_maps.get("cs") or not label_maps.get("y3"):
        print("[CACHE] cannot verify stacking coverage because split/label maps are unavailable")
        return False

    expected_folds = set(expected_outer_folds or _expected_outer_folds(split_json_path))
    split_by_fold = {
        int(outer.get("outer_fold")): outer
        for outer in outer_splits
        if int(outer.get("outer_fold", -1)) in expected_folds
    }
    missing_split_folds = expected_folds - set(split_by_fold)
    if missing_split_folds:
        print(f"[CACHE] split JSON missing expected folds for stacking coverage: {sorted(missing_split_folds)}")
        return False

    for file_tag in _file_tags(file_names):
        case_ids = _read_workbook_case_ids(workbook_dir, file_tag)
        if case_ids is None:
            return False
        for outer_fold in sorted(expected_folds):
            outer = split_by_fold[outer_fold]
            for task in ("cs", "zone"):
                expected_train, expected_eval, expected_all_test = _expected_ids_for_outer(outer, task, label_maps)
                checks = [
                    ("train", expected_train),
                    ("eval", expected_eval),
                    ("all_test", expected_all_test),
                ]
                for part_name, expected_ids in checks:
                    missing = sorted(expected_ids - case_ids)
                    if missing:
                        print(
                            f"[CACHE] {workbook_dir.name}/{file_tag} incomplete for stacking; "
                            f"task={task} outer={outer_fold} missing_{part_name}={len(missing)}/"
                            f"{len(expected_ids)} examples={missing[:8]}"
                        )
                        return False
    return True


def _prediction_rows_cover_expected_cases(
    pred_rows: pd.DataFrame,
    pmask: pd.Series,
    task: str,
    outer_fold: int,
    expected_ids_by_task_outer: dict[tuple[str, int], set[str]],
) -> bool:
    expected_ids = expected_ids_by_task_outer.get((task, int(outer_fold)), set())
    if not expected_ids:
        return True
    fold_mask = pmask & (pd.to_numeric(pred_rows["outer_fold"], errors="coerce") == int(outer_fold))
    got_ids = {cid for cid in (_clean_case_id_for_verify(x) for x in pred_rows.loc[fold_mask, "case_id"].tolist()) if cid is not None}
    missing = sorted(expected_ids - got_ids)
    if missing:
        return False
    dup = pred_rows.loc[fold_mask, "case_id"].astype(str).str.strip().duplicated().sum()
    return int(dup) == 0


def _report_is_complete(report_path: Path) -> bool:
    if not report_path.exists():
        print(f"[CACHE] missing final report: {report_path}")
        return False
    try:
        text = report_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    if "[ERROR]" in text or "Traceback" in text:
        print(f"[CACHE] final report contains errors: {report_path}")
        return False
    required_sections = ["[CS summary]", "[ZONE summary]", "[CS/ZONE average summary]"]
    if not all(section in text for section in required_sections):
        print(f"[CACHE] final report missing required sections: {report_path}")
        return False
    if text.count("No rows.") >= 3:
        print(f"[CACHE] final report has no usable rows: {report_path}")
        return False
    return True


def _active_ml_grid_complete(
    results_dir: Path,
    file_names: list[str],
    feature_selectors: list[str],
    classifiers: list[str],
    expected_outer_folds: set[int] | None = None,
    workbook_dir: Path | None = None,
    split_json_path: Path | None = None,
) -> bool:
    if not results_dir.exists():
        return False
    if not _report_is_complete(results_dir / FINAL_REPORT_NAME):
        return False

    expected_folds = set(expected_outer_folds or _expected_outer_folds(split_json_path))

    outer_splits = _load_outer_splits(split_json_path)
    label_maps = _load_reference_label_maps()
    expected_ids_by_task_outer: dict[tuple[str, int], set[str]] = {}
    if not outer_splits or not label_maps.get("cs") or not label_maps.get("y3"):
        print(f"[CACHE] cannot verify case-level ML prediction coverage for {results_dir}")
        return False
    _assert_inner_oof_split_coverage_complete(outer_splits, label_maps, expected_folds)

    for outer in outer_splits:
        outer_fold = int(outer.get("outer_fold", -1))
        if outer_fold not in expected_folds:
            continue
        for task in ("cs", "zone"):
            _, expected_eval, _ = _expected_ids_for_outer(outer, task, label_maps)
            expected_ids_by_task_outer[(task, outer_fold)] = expected_eval

    if workbook_dir is not None and not _stacking_feature_coverage_complete(workbook_dir, file_names, expected_folds, split_json_path):
        return False

    expected = {
        (str(fs_method), str(clf_model))
        for fs_method in feature_selectors
        for clf_model in classifiers
    }
    if not expected or not expected_folds:
        return False

    def _file_tag_complete(file_tag: str) -> bool:
        file_dir = results_dir / file_tag
        if not file_dir.exists():
            return False

        required_files = [
            file_dir / f"results_cs_{file_tag}.xlsx",
            file_dir / f"predictions_cs_{file_tag}.csv",
            file_dir / f"results_zone_{file_tag}.xlsx",
            file_dir / f"predictions_zone_{file_tag}.csv",
        ]
        if any(not path.exists() for path in required_files):
            return False

        for task in ("cs", "zone"):
            result_path = file_dir / f"results_{task}_{file_tag}.xlsx"
            pred_path = file_dir / f"predictions_{task}_{file_tag}.csv"
            try:
                summary = pd.read_excel(result_path, sheet_name="combo_summary")
                fold_metrics = pd.read_excel(result_path, sheet_name="fold_metrics")
                pred_rows = pd.read_csv(pred_path)
            except Exception:
                return False

            needed_cols = {"fs_method", "clf_model"}
            fold_cols = needed_cols | {"outer_fold"}
            pred_cols = needed_cols | {"outer_fold", "case_id", "y_true", "y_score"}
            if (
                summary.empty or fold_metrics.empty or pred_rows.empty
                or not needed_cols.issubset(summary.columns)
                or not fold_cols.issubset(fold_metrics.columns)
                or not pred_cols.issubset(pred_rows.columns)
            ):
                return False

            summary_got = {
                (str(row["fs_method"]), str(row["clf_model"]))
                for _, row in summary.iterrows()
            }
            if expected - summary_got:
                return False

            for key in sorted(expected):
                fs_method, clf_model = key
                fmask = (
                    (fold_metrics["fs_method"].astype(str) == fs_method)
                    & (fold_metrics["clf_model"].astype(str) == clf_model)
                )
                pmask = (
                    (pred_rows["fs_method"].astype(str) == fs_method)
                    & (pred_rows["clf_model"].astype(str) == clf_model)
                )
                got_folds = set(pd.to_numeric(fold_metrics.loc[fmask, "outer_fold"], errors="coerce").dropna().astype(int).tolist())
                pred_folds = set(pd.to_numeric(pred_rows.loc[pmask, "outer_fold"], errors="coerce").dropna().astype(int).tolist())
                if (expected_folds - got_folds) or (expected_folds - pred_folds):
                    return False

                for outer_fold in sorted(expected_folds):
                    if not _prediction_rows_cover_expected_cases(
                        pred_rows, pmask, task, outer_fold, expected_ids_by_task_outer,
                    ):
                        return False

                bad_scores = pd.to_numeric(pred_rows.loc[pmask, "y_score"], errors="coerce").isna().sum()
                if int(bad_scores) > 0:
                    return False
        return True

    incomplete_tags = [file_tag for file_tag in _file_tags(file_names) if not _file_tag_complete(file_tag)]
    if incomplete_tags:
        print(f"[CACHE] Files not available for {', '.join(incomplete_tags)}")
        return False
    return True


def ml_result_ready(
    results_dir: Path,
    name: str,
    file_names: list[str],
    feature_selectors: list[str],
    classifiers: list[str],
    expected_outer_folds: set[int] | None = None,
    workbook_dir: Path | None = None,
    split_json_path: Path | None = None,
) -> bool:
    if _active_ml_grid_complete(
        results_dir,
        file_names,
        feature_selectors,
        classifiers,
        expected_outer_folds,
        workbook_dir=workbook_dir,
        split_json_path=split_json_path,
    ):
        print(f"[CACHE] {name} active ML grid complete and verified for stacking: {results_dir}")
        if not completion_logged(results_dir, name):
            mark_complete(results_dir, name)
        return True
    if completion_logged(results_dir, name):
        print(f"[RESUME] {name} completion marker exists, but verified ML Excel/report check failed; rerunning ML resume")
    return False


def late_fusion_files_complete(
    results_dir: Path,
    file_names: list[str],
    feature_selectors: list[str],
    classifiers: list[str],
    alpha_grid: list[float],
) -> bool:
    if not results_dir.exists():
        return False
    expected = {
        (str(fs_method), str(clf_model), float(alpha))
        for fs_method in feature_selectors
        for clf_model in classifiers
        for alpha in alpha_grid
    }
    if not expected:
        return False

    for file_tag in _file_tags(file_names):
        file_dir = results_dir / file_tag
        required_files = [
            file_dir / f"results_cs_{file_tag}.xlsx",
            file_dir / f"results_zone_{file_tag}.xlsx",
        ]
        if not all(path.exists() for path in required_files):
            return False
        for task in ("cs", "zone"):
            try:
                summary = pd.read_excel(file_dir / f"results_{task}_{file_tag}.xlsx", sheet_name="combo_summary")
            except Exception:
                return False
            needed_cols = {"fs_method", "clf_model", "alpha"}
            if summary.empty or not needed_cols.issubset(summary.columns):
                return False
            got = {
                (str(row["fs_method"]), str(row["clf_model"]), float(row["alpha"]))
                for _, row in summary.iterrows()
            }
            missing = expected - got
            if missing:
                sample = sorted(missing)[:5]
                print(
                    f"[CACHE] D_Late/{file_tag}/{task} missing active fusion combos; "
                    f"examples: {sample}"
                )
                return False
    return (results_dir / FINAL_REPORT_NAME).exists()


def late_fusion_ready(alpha_grid: list[float]) -> bool:
    name = "D_Late"
    if late_fusion_files_complete(
        LATE_FUSION_RESULTS_DIR,
        list(FEATURE_FILES),
        list(ML_FEATURE_SELECTORS),
        list(ML_CLASSIFIERS),
        alpha_grid,
    ):
        print(f"[CACHE] D_Late active fusion grid complete: {LATE_FUSION_RESULTS_DIR}")
        if not completion_logged(LATE_FUSION_RESULTS_DIR, name):
            mark_complete(LATE_FUSION_RESULTS_DIR, name)
        return True
    if completion_logged(LATE_FUSION_RESULTS_DIR, name):
        print("[RESUME] D_Late completion marker exists, but active config has missing combos; rerunning late fusion")
    return False


def organ_workbooks_ready() -> bool:
    return all_exist(workbook_files(ORGAN_ONLY_WORKBOOK_DIR, list(FEATURE_FILES)))


def patch_workbooks_ready() -> bool:
    paths = workbook_files(PATCH_ONLY_WORKBOOK_DIR, list(FEATURE_FILES))
    if not paths or not all(p.exists() for p in paths):
        return False
    import b_patch
    try:
        return b_patch._all_final_patch_workbooks_exist()
    except FileNotFoundError:
        return True


def early_fusion_workbooks_ready() -> bool:
    return all_exist(workbook_files(EARLY_FUSION_WORKBOOK_DIR, list(FEATURE_FILES)))


def migrate_legacy_done_marker(results_dir: Path, name: str) -> None:
    marker = results_dir / LEGACY_DONE_MARKER
    if not marker.exists():
        return
    if not completion_logged(results_dir, name):
        mark_complete(results_dir, name)
    try:
        marker.unlink()
        log_line(results_dir, f"Removed legacy {LEGACY_DONE_MARKER}; completion is tracked in {RUN_LOG_NAME}")
    except Exception as e:
        log_line(results_dir, f"Could not remove legacy {LEGACY_DONE_MARKER}: {repr(e)}")


def _safe_feature_set_tag(feature_set: str) -> str:
    return str(feature_set).strip()


def _task_model_path(scale_name: str, task: str, feature_set: str) -> Path:
    # Must mirror b_patch._task_model_path exactly; otherwise patch_helper_training_complete()
    # checks for files that b_patch never writes and always reports "not trained".
    return PATCH_HELPER_MODEL_DIR / f"{scale_name}_{task}_{_safe_feature_set_tag(feature_set)}.joblib"


def patch_helper_training_complete() -> bool:
    for scale in PATCH_SCALES:
        scale_name = str(scale["name"])
        for feature_set in PATCH_FEATURE_SETS:
            for task in ("cspca", "zone"):
                if not _task_model_path(scale_name, task, str(feature_set)).exists():
                    return False
    return True


def ensure_main_organ_workbooks() -> None:
    if all_exist(workbook_files(ORGAN_ONLY_WORKBOOK_DIR, list(FEATURE_FILES))):
        print("[CACHE] organ experiment workbooks already exist")
        return
    run_cmd([PYTHON, str(SCRIPT_DIR / "c_organ.py")])


def organ_same_workbooks_ready() -> bool:
    return all_exist(workbook_files(ORGAN_SAME_WORKBOOK_DIR, list(FEATURE_FILES)))


def ensure_organ_same_workbooks() -> None:
    if organ_same_workbooks_ready():
        print("[CACHE] B_Organ_same workbooks already exist")
        return
    run_cmd([PYTHON, str(SCRIPT_DIR / "c_organ.py"), "--same"])


def ensure_patch_helper_outputs(build_early_fusion: bool = True) -> None:
    if patch_workbooks_ready() and (not build_early_fusion or early_fusion_workbooks_ready()):
        print("[CACHE] A_Patch (+C_Early) workbooks already exist; skipping patch stages")
        return

    if not patch_helper_training_complete():
        run_cmd([PYTHON, str(SCRIPT_DIR / "b_patch.py"), "--stage", "train"])
    else:
        print("[CACHE] external patch helper models already exist; skipping patch train")

    if not patch_workbooks_ready():
        run_cmd([PYTHON, str(SCRIPT_DIR / "b_patch.py"), "--stage", "apply"])
    else:
        print("[CACHE] A_Patch workbooks already exist; skipping patch apply")

    if build_early_fusion:
        if not early_fusion_workbooks_ready():
            if not organ_workbooks_ready():
                raise FileNotFoundError(
                    "C_Early workbooks are missing and B_Organ workbooks are not ready. "
                    "Run B preparation first or request C through z_main.py so dependencies are built."
                )
            if not patch_workbooks_ready():
                raise FileNotFoundError(
                    "C_Early workbooks are missing and A_Patch workbooks are not ready. "
                    "Patch apply should have created them before merge."
                )
            run_cmd([PYTHON, str(SCRIPT_DIR / "d_fusion.py"), "--stage", "early"])
        else:
            print("[CACHE] C_Early workbooks already exist; skipping early-fusion merge")


def run_ml_experiment(
    name: str,
    workbook_dir: Path,
    results_dir: Path,
    seed: int,
    split_json_path: Path | None = None,
    feature_files: list[str] | None = None,
) -> bool:
    if feature_files is not None:
        expected_folds = _expected_outer_folds(split_json_path, default=4)
        if ml_result_ready(
            results_dir,
            name,
            feature_files,
            list(ML_FEATURE_SELECTORS),
            list(ML_CLASSIFIERS),
            expected_outer_folds=expected_folds,
            workbook_dir=workbook_dir,
            split_json_path=split_json_path,
        ):
            return False
    results_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON,
        str(SCRIPT_DIR / "e_ml.py"),
        "--base-dir",
        str(workbook_dir),
        "--results-dir-name",
        str(results_dir),
        "--name",
        name,
        "--seed",
        str(int(seed)),
    ]
    if split_json_path is not None:
        cmd.extend(["--json-path", str(split_json_path)])
    run_cmd(cmd)
    return True


def run_late_fusion(alpha_grid: list[float], force: bool = False) -> bool:
    name = "D_Late"
    migrate_legacy_done_marker(LATE_FUSION_RESULTS_DIR, name)
    if not force and late_fusion_ready(alpha_grid):
        print(f"[CACHE] D late-fusion results already complete: {LATE_FUSION_RESULTS_DIR}")
        return False
    LATE_FUSION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_line(LATE_FUSION_RESULTS_DIR, "D late-fusion pipeline started")
    reports = d_fusion.compute_late_fusion(
        alpha_grid, ORGAN_ONLY_RESULTS_DIR, PATCH_ONLY_RESULTS_DIR, LATE_FUSION_RESULTS_DIR
    )
    log_line(LATE_FUSION_RESULTS_DIR, f"D late-fusion wrote {len(reports)} task result workbooks")
    mark_complete(LATE_FUSION_RESULTS_DIR, name)
    return True


def parse_alpha_grid(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("alpha grid cannot be empty")
    for val in vals:
        if val < 0.0 or val > 1.0:
            raise ValueError("alpha values must be in [0, 1]")
    return vals


def _result_dir_for_experiment(experiment_key: str) -> Path:
    mapping = {
        "A":  PATCH_ONLY_RESULTS_DIR,
        "B":  ORGAN_ONLY_RESULTS_DIR,
        "B0": ORGAN_SAME_RESULTS_DIR,
        "C":  EARLY_FUSION_RESULTS_DIR,
        "D":  LATE_FUSION_RESULTS_DIR,
    }
    return mapping[experiment_key]


def stacking_ready(experiment_key: str, split_json_path: Path | None = None) -> bool:
    result_dir = _result_dir_for_experiment(experiment_key)
    book = result_dir / "final_stacking.xlsx"
    if not book.exists():
        return False
    try:
        xl = pd.ExcelFile(book)
        required_sheets = {
            "meta_fold_metrics",
            "meta_predictions",
            "selected_candidates",
            "mean_outer",
            "pooled",
        }
        if not required_sheets.issubset(set(xl.sheet_names)):
            print(f"[CACHE] {experiment_key} stacking workbook missing sheets; rerunning stacking")
            return False
        fold_df = pd.read_excel(book, sheet_name="meta_fold_metrics")
        pred_df = pd.read_excel(book, sheet_name="meta_predictions")
        sel_df = pd.read_excel(book, sheet_name="selected_candidates")
    except Exception as exc:
        print(f"[CACHE] {experiment_key} stacking workbook unreadable: {repr(exc)}; rerunning stacking")
        return False

    needed_fold_cols = {"task", "outer_fold"}
    needed_pred_cols = {"task", "outer_fold", "case_id", "y_true", "y_score"}
    if (
        fold_df.empty or pred_df.empty or sel_df.empty
        or not needed_fold_cols.issubset(fold_df.columns)
        or not needed_pred_cols.issubset(pred_df.columns)
    ):
        print(f"[CACHE] {experiment_key} stacking workbook partial/empty; rerunning stacking")
        return False

    expected_folds = _expected_outer_folds(split_json_path, default=4)
    expected_pairs = {(task, fold) for task in ("cs", "zone") for fold in expected_folds}
    got_pairs = {
        (str(row["task"]), int(row["outer_fold"]))
        for _, row in fold_df.iterrows()
        if pd.notna(row.get("task")) and pd.notna(row.get("outer_fold"))
    }
    missing_pairs = expected_pairs - got_pairs
    if missing_pairs:
        print(f"[CACHE] {experiment_key} stacking workbook missing task/fold rows; examples: {sorted(missing_pairs)[:5]}")
        return False

    return True


def run_stacking(
    requested: set[str],
    top_k: int,
    seed: int,
    split_json_path: Path | None = None,
) -> bool:
    ordered = [key for key in ["A", "B", "B0", "C", "D"] if key in requested]
    if not ordered:
        print("[STACKING] no applicable A/B/B0/C/D experiments requested")
        return False
    cmd = [
        PYTHON,
        str(SCRIPT_DIR / "f_stack.py"),
        "--only",
        ",".join(ordered),
        "--top-k",
        str(int(top_k)),
        "--seed",
        str(int(seed)),
    ]
    if len(ordered) == 1 and split_json_path is not None:
        cmd.extend(["--json-path", str(split_json_path)])
    run_cmd(cmd)
    return True


def maybe_run_stacking(
    experiment_key: str,
    requested: set[str],
    skip_stacking: bool,
    top_k: int,
    seed: int,
    split_json_path: Path | None = None,
    upstream_changed: bool = False,
) -> bool:
    if skip_stacking or experiment_key not in requested:
        return False
    _exp_name = {"A": "A_Patch", "B": "B_Organ", "B0": "B_Organ_same", "C": "C_Early", "D": "D_Late"}.get(experiment_key, experiment_key)
    if upstream_changed or not stacking_ready(experiment_key, split_json_path):
        if upstream_changed:
            print(f"[STACKING] {_exp_name} upstream ML/fusion changed; rerunning stacking")
        else:
            print(f"[STACKING] {_exp_name} stacking output missing; running stacking")
        return run_stacking({experiment_key}, top_k, seed, split_json_path=split_json_path)
    print(f"[CACHE] {_exp_name} stacking output exists and upstream did not change; skipping stacking")
    return False


def _ordered_experiments(raw: str | None, fallback: set[str]) -> list[str]:
    if raw is None:
        requested = set(fallback)
    else:
        requested = {x.strip().upper() for x in raw.split(",") if x.strip()}
    invalid = requested - {"A", "B", "B0", "C", "D"}
    if invalid:
        raise ValueError(f"Unknown experiments: {sorted(invalid)}")
    return [key for key in ["A", "B", "B0", "C", "D"] if key in requested]


def external_data_available() -> bool:
    # Checks raw input, not the preprocessed output folder - preprocessing itself
    # happens lazily inside h_test.py's own startup, not here.
    return Path(TEST_ROOT / "original").exists() and Path(P158_DATASET_XLSX).exists()


def _final_models_ready_for_external(seed: int, exp_keys: list[str]) -> bool:
    return all_exist(_final_model_paths_for_seed(seed, exp_keys))


def _external_results_ready(seed: int, exp_keys: list[str]) -> bool:
    return all_exist(_external_result_paths_for_seed(seed, exp_keys))


def run_external_test(
    args: argparse.Namespace,
    keys: list[str],
) -> bool:
    test_keys = [e for e in keys if e in ("A", "B", "C", "D")]
    if not test_keys:
        return False
    if not external_data_available():
        print(f"[TEST] external P158 raw data not available ({TEST_ROOT / 'original'}); skipping external test for {test_keys}")
        return False

    if _external_results_ready(args.seed, test_keys):
        print(f"[CACHE] external results already exist for {test_keys}; skipping g_train and h_test")
        return False

    model_keys = [e for e in test_keys if e != "D"]
    if model_keys:
        if _final_models_ready_for_external(args.seed, model_keys):
            print(f"[CACHE] deployable models already exist for {model_keys}; skipping g_train")
        else:
            print("[TRAIN] g_train: refitting deployable models on full training data")
            run_cmd([PYTHON, str(SCRIPT_DIR / "g_train.py"), "--seed", str(args.seed), "--only", ",".join(model_keys)])
    print("[TEST] h_test: preparing test features and applying saved models")
    test_cmd = [PYTHON, str(SCRIPT_DIR / "h_test.py"), "--prepare", "--seed", str(args.seed),
                "--only", ",".join(test_keys), "--alpha-grid", str(args.alpha_grid)]
    if not args.deep_analysis:
        test_cmd.append("--no-deep-analysis")
    run_cmd(test_cmd)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run A/B/B0/C/D ML experiments, stacking, and P158 external testing unless skipped."
    )
    parser.add_argument("--alpha-grid", default=DEFAULT_LATE_FUSION_ALPHA_GRID, help="Late-fusion organ-score weight grid.")
    parser.add_argument("--skip-stacking", action="store_true", help="Do not run f_stack.py after each requested experiment completes.")
    parser.add_argument("--stacking-top-k", type=int, default=DEFAULT_STACKING_TOP_K, help="Number of cached base candidates per task/fold for f_stack.py.")
    parser.add_argument("--skip-external", action="store_true", help="Do not run the P158 external test phase.")
    parser.add_argument("--no-deep-analysis", dest="deep_analysis", action="store_false",
                        help="During external testing, skip merging external metrics into each experiment's "
                             "analysis.xlsx. Deep analysis runs by default wherever valid.")
    parser.add_argument("--external-experiments", default=None, help="Comma-separated P158 experiments. Default: same as --only.")
    parser.add_argument(
        "--only",
        default=DEFAULT_PIPELINE_EXPERIMENTS,
        help="Comma-separated experiments to run after preparation. Choices: A,B,B0,C,D. "
             "B0 (B_Organ_same) runs through nested-CV + stacking only; it is never refit/tested externally.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEEDS[0],
                        help="Run a single seed. Per-experiment results go to <Exp>/.../random_seed{N}/.")
    parser.add_argument("--all-seeds", action="store_true",
                        help="Run all seeds from RANDOM_SEEDS sequentially. Equivalent to running --seed N for each N.")
    args = parser.parse_args()

    print("[PREPROCESS] ensuring original training data is preprocessed (resample + crop)")
    a_preprocess.preprocess_train_datasets()

    requested = {x.strip().upper() for x in args.only.split(",") if x.strip()}
    invalid = requested - {"A", "B", "B0", "C", "D"}
    if invalid:
        raise ValueError(f"Unknown experiments in --only: {sorted(invalid)}")

    if args.all_seeds:
        external_requested = [key for key in _ordered_experiments(args.external_experiments, requested) if key in {"A", "B", "C", "D"}]
        requested_model_keys = [key for key in requested if key in {"A", "B", "C"}]
        for s in RANDOM_SEEDS:
            if not args.skip_external and requested_model_keys and external_requested:
                if _final_models_ready_for_external(s, requested_model_keys) and _external_results_ready(s, external_requested):
                    print(f"\n{'='*60}\n[z_main] seed={s} already complete; skipping\n{'='*60}")
                    continue
            print(f"\n{'='*60}\n[z_main] Running seed={s}\n{'='*60}")
            import subprocess as _sp
            cmd = [sys.executable, __file__] + [a for a in sys.argv[1:] if a != "--all-seeds"] + [f"--seed={s}"]
            ret = _sp.run(cmd)
            if ret.returncode != 0:
                print(f"[z_main] seed={s} FAILED with returncode={ret.returncode}")
        return

    import all_config as _acfg
    _seed = args.seed
    _rdirs = _acfg.results_dirs_for_seed(_seed)
    global ORGAN_ONLY_RESULTS_DIR, PATCH_ONLY_RESULTS_DIR
    global LATE_FUSION_RESULTS_DIR, EARLY_FUSION_RESULTS_DIR
    global ORGAN_SAME_RESULTS_DIR
    ORGAN_ONLY_RESULTS_DIR   = _rdirs["organ"]
    ORGAN_SAME_RESULTS_DIR   = _rdirs["organ_same"]
    PATCH_ONLY_RESULTS_DIR   = _rdirs["patch"]
    LATE_FUSION_RESULTS_DIR  = _rdirs["late_fusion"]
    EARLY_FUSION_RESULTS_DIR = _rdirs["early_fusion"]

    print(f"[z_main] seed={_seed}  results → {_acfg.OUTPUT_ROOT}")
    split_json_paths = {key: _split_json_for_experiment(key, _seed) for key in ["A", "B", "B0", "C", "D"]}
    ordered_requested = [key for key in ["A", "B", "B0", "C", "D"] if key in requested]
    primary_split_json = split_json_paths[ordered_requested[0]] if ordered_requested else split_json_paths["A"]
    expected_outer_folds = _expected_outer_folds(primary_split_json, default=4)

    alpha_grid = parse_alpha_grid(args.alpha_grid)


    organ_ready = patch_ready = organ_same_ready = early_ready = False

    if "B" in requested or "D" in requested:
        organ_ready = ml_result_ready(
            ORGAN_ONLY_RESULTS_DIR,
            "B_Organ",
            list(FEATURE_FILES),
            list(ML_FEATURE_SELECTORS),
            list(ML_CLASSIFIERS),
            expected_outer_folds=expected_outer_folds,
            workbook_dir=ORGAN_ONLY_WORKBOOK_DIR,
            split_json_path=split_json_paths["B"],
        )
    if "A" in requested or "D" in requested:
        patch_ready = ml_result_ready(
            PATCH_ONLY_RESULTS_DIR,
            "A_Patch",
            list(FEATURE_FILES),
            list(ML_FEATURE_SELECTORS),
            list(ML_CLASSIFIERS),
            expected_outer_folds=expected_outer_folds,
            workbook_dir=PATCH_ONLY_WORKBOOK_DIR,
            split_json_path=split_json_paths["A"],
        )
    if "B0" in requested:
        organ_same_ready = ml_result_ready(
            ORGAN_SAME_RESULTS_DIR,
            "B_Organ_same",
            list(FEATURE_FILES),
            list(ML_FEATURE_SELECTORS),
            list(ML_CLASSIFIERS),
            expected_outer_folds=expected_outer_folds,
            workbook_dir=ORGAN_SAME_WORKBOOK_DIR,
            split_json_path=split_json_paths["B0"],
        )
    if "C" in requested:
        early_ready = ml_result_ready(
            EARLY_FUSION_RESULTS_DIR,
            "C_Early",
            list(FEATURE_FILES),
            list(ML_FEATURE_SELECTORS),
            list(ML_CLASSIFIERS),
            expected_outer_folds=expected_outer_folds,
            workbook_dir=EARLY_FUSION_WORKBOOK_DIR,
            split_json_path=split_json_paths["C"],
        )

    def ensure_organ_inputs_and_results() -> bool:
        nonlocal organ_ready
        if organ_ready:
            print("[CACHE] B_Organ final ML results available; skipping B_Organ preparation/training")
            return False
        if not organ_workbooks_ready():
            ensure_main_organ_workbooks()
        else:
            print("[CACHE] B_Organ workbooks already exist; skipping raw organ extraction/ICC/workbook build")
        changed = run_ml_experiment(
            "B_Organ",
            ORGAN_ONLY_WORKBOOK_DIR,
            ORGAN_ONLY_RESULTS_DIR,
            _seed,
            split_json_path=split_json_paths["B"],
            feature_files=list(FEATURE_FILES),
        )
        organ_ready = ml_result_ready(
            ORGAN_ONLY_RESULTS_DIR,
            "B_Organ",
            list(FEATURE_FILES),
            list(ML_FEATURE_SELECTORS),
            list(ML_CLASSIFIERS),
            expected_outer_folds=expected_outer_folds,
            workbook_dir=ORGAN_ONLY_WORKBOOK_DIR,
            split_json_path=split_json_paths["B"],
        )
        return changed

    def ensure_organ_same_inputs_and_results() -> bool:
        nonlocal organ_same_ready
        if organ_same_ready:
            print("[CACHE] B_Organ_same final ML results available; skipping B_Organ_same preparation/training")
            return False
        if not organ_same_workbooks_ready():
            ensure_organ_same_workbooks()
        else:
            print("[CACHE] B_Organ_same workbooks already exist; skipping raw organ_same extraction/ICC/workbook build")
        changed = run_ml_experiment(
            "B_Organ_same",
            ORGAN_SAME_WORKBOOK_DIR,
            ORGAN_SAME_RESULTS_DIR,
            _seed,
            split_json_path=split_json_paths["B0"],
            feature_files=list(FEATURE_FILES),
        )
        organ_same_ready = ml_result_ready(
            ORGAN_SAME_RESULTS_DIR,
            "B_Organ_same",
            list(FEATURE_FILES),
            list(ML_FEATURE_SELECTORS),
            list(ML_CLASSIFIERS),
            expected_outer_folds=expected_outer_folds,
            workbook_dir=ORGAN_SAME_WORKBOOK_DIR,
            split_json_path=split_json_paths["B0"],
        )
        return changed

    def ensure_patch_inputs_and_results() -> bool:
        nonlocal patch_ready
        if patch_ready:
            print("[CACHE] A_Patch final ML results available; skipping A_Patch preparation/training")
            return False
        if not patch_workbooks_ready():
            ensure_patch_helper_outputs(build_early_fusion=False)
        else:
            print("[CACHE] A_Patch workbooks already exist; skipping patch train/apply")
        changed = run_ml_experiment(
            "A_Patch",
            PATCH_ONLY_WORKBOOK_DIR,
            PATCH_ONLY_RESULTS_DIR,
            _seed,
            split_json_path=split_json_paths["A"],
            feature_files=list(FEATURE_FILES),
        )
        patch_ready = ml_result_ready(
            PATCH_ONLY_RESULTS_DIR,
            "A_Patch",
            list(FEATURE_FILES),
            list(ML_FEATURE_SELECTORS),
            list(ML_CLASSIFIERS),
            expected_outer_folds=expected_outer_folds,
            workbook_dir=PATCH_ONLY_WORKBOOK_DIR,
            split_json_path=split_json_paths["A"],
        )
        return changed

    def ensure_early_fusion_inputs() -> None:
        if early_fusion_workbooks_ready():
            print("[CACHE] C_Early workbooks already exist; skipping A/B feature preparation and merge")
            return
        if not organ_workbooks_ready():
            ensure_main_organ_workbooks()
        else:
            print("[CACHE] B_Organ workbooks already exist; skipping raw organ extraction/ICC/workbook build")
        ensure_patch_helper_outputs(build_early_fusion=True)

    external_requested = set(_ordered_experiments(args.external_experiments, requested))
    if not external_data_available() and not args.skip_external:
        print(f"[TEST] external P158 raw data not available ({TEST_ROOT / 'original'}); external testing will be skipped for every experiment")

    # Each experiment runs all the way through its own external test (when requested
    # and external data is available) before the next experiment starts.
    if "A" in requested:
        patch_changed = ensure_patch_inputs_and_results()
        maybe_run_stacking(
            "A", requested, args.skip_stacking, args.stacking_top_k, _seed,
            split_json_path=split_json_paths["A"], upstream_changed=patch_changed
        )
        if not args.skip_external and "A" in external_requested:
            run_external_test(args, ["A"])

    if "B" in requested:
        organ_changed = ensure_organ_inputs_and_results()
        maybe_run_stacking(
            "B", requested, args.skip_stacking, args.stacking_top_k, _seed,
            split_json_path=split_json_paths["B"], upstream_changed=organ_changed
        )
        if not args.skip_external and "B" in external_requested:
            run_external_test(args, ["B"])

    if "B0" in requested:
        organ_same_changed = ensure_organ_same_inputs_and_results()
        maybe_run_stacking(
            "B0", requested, args.skip_stacking, args.stacking_top_k, _seed,
            split_json_path=split_json_paths["B0"], upstream_changed=organ_same_changed
        )
        # B0 (B_Organ_same) is never refit/tested externally.

    if "C" in requested:
        early_changed = False
        if early_ready:
            print("[CACHE] C_Early final ML results available; skipping C preparation/training")
        else:
            ensure_early_fusion_inputs()
            early_changed = run_ml_experiment(
                "C_Early",
                EARLY_FUSION_WORKBOOK_DIR,
                EARLY_FUSION_RESULTS_DIR,
                _seed,
                split_json_path=split_json_paths["C"],
                feature_files=list(FEATURE_FILES),
            )
        maybe_run_stacking(
            "C", requested, args.skip_stacking, args.stacking_top_k, _seed,
            split_json_path=split_json_paths["C"], upstream_changed=early_changed
        )
        if not args.skip_external and "C" in external_requested:
            run_external_test(args, ["C"])

    if "D" in requested:
        organ_changed = ensure_organ_inputs_and_results()
        patch_changed = ensure_patch_inputs_and_results()
        late_now_ready = late_fusion_ready(alpha_grid)
        if late_now_ready and not organ_changed and not patch_changed:
            print("[CACHE] D_Late final results available and A/B dependencies unchanged; skipping D_Late")
            late_changed = False
        else:
            late_changed = run_late_fusion(alpha_grid, force=organ_changed or patch_changed)
        maybe_run_stacking(
            "D", requested, args.skip_stacking, args.stacking_top_k, _seed,
            split_json_path=split_json_paths["B"], upstream_changed=late_changed
        )
        if not args.skip_external and "D" in external_requested:
            run_external_test(args, ["D"])

    print("[DONE] full A/B/B0/C/D + external runner finished")


if __name__ == "__main__":
    main()
