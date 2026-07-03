from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from openpyxl import Workbook

import all_config as cfg

FOLDER_T2W = cfg.T2W_FOLDER
FOLDER_MASK_ORGAN = cfg.MASK_FOLDER
FOLDER_MASK_LESION = cfg.LESION_MASK_FOLDER
FOLDER_MASK_ZONE = cfg.ZONE_MASK_FOLDER
FOLDER_ADC = cfg.ADC_FOLDER
REQUIRED_FOLDERS = (FOLDER_T2W, FOLDER_MASK_ORGAN)

IMAGE_SUFFIXES = (".nii.gz", ".mha", ".nii")
SPACE_SIZE_SUFFIXES = (".nii.gz", ".mha")

RESAMPLE_SPACING = (0.5, 0.5, 3.0)

MIN_LESION_INSIDE_ORGAN = 0.95
MIN_ZONE_INSIDE_ORGAN = 0.80
MIN_ADC_ORGAN_COVERAGE = 0.95
MIN_ADC_LESION_COVERAGE = 0.95

DATASETS = {
    "train_main": {
        "original": cfg.TRAIN_MAIN_ROOT / "original",
        "preprocessed": cfg.MAIN_ROOT,
        "crop_size": (320, 320, 19),
        "filter_cropped_final": False,
    },
    "train_helper": {
        "original": cfg.TRAIN_HELPER_ROOT / "original",
        "preprocessed": cfg.PRETRAIN_ROOT,
        "crop_size": (320, 320, 19),
        "filter_cropped_final": True,
    },
    "test_p158": {
        "original": cfg.TEST_ROOT / "original",
        "preprocessed": cfg.P158_ROOT,
        "crop_size": (192, 192, 19),
        "filter_cropped_final": False,
    },
}

TRAIN_DATASETS = ["train_main", "train_helper"]
TEST_DATASETS = ["test_p158"]


def strip_image_suffix(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for suffix in IMAGE_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def is_image_file(path: Path, suffixes: tuple[str, ...] = IMAGE_SUFFIXES) -> bool:
    lower = path.name.lower()
    return path.is_file() and any(lower.endswith(suffix) for suffix in suffixes)


def is_mask_folder(name: str) -> bool:
    return "mask" in name.lower()


def find_case_file(folder: Path, case_id: str, prefer_mask: bool = False) -> Path | None:
    suffixes = IMAGE_SUFFIXES if prefer_mask else (".mha", ".nii.gz", ".nii")
    for suffix in suffixes:
        path = folder / f"{case_id}{suffix}"
        if path.exists():
            return path
    return None


def input_dirs(root: Path) -> dict[str, Path]:
    dirs = {p.name: p for p in sorted(root.iterdir()) if p.is_dir()} if root.exists() else {}
    for name in REQUIRED_FOLDERS:
        dirs.setdefault(name, root / name)
    return dirs


def existing_input_dirs(root: Path) -> dict[str, Path]:
    dirs = input_dirs(root)
    missing = [str(dirs[name]) for name in REQUIRED_FOLDERS if not dirs[name].exists()]
    if missing:
        raise FileNotFoundError("Missing required input folders:\n" + "\n".join(missing))
    return {name: path for name, path in dirs.items() if path.exists()}


def case_ids_from_t2w(t2w_dir: Path) -> list[str]:
    return sorted(strip_image_suffix(p) for p in t2w_dir.iterdir() if is_image_file(p))


def dataset_ready(root: Path) -> bool:
    t2w_dir = root / FOLDER_T2W
    organ_dir = root / FOLDER_MASK_ORGAN
    return t2w_dir.exists() and organ_dir.exists() and any(is_image_file(p) for p in t2w_dir.iterdir())


def fit_excel_columns(sheet, max_width: int = 28) -> None:
    for column_cells in sheet.columns:
        width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
        sheet.column_dimensions[column_cells[0].column_letter].width = min(max(width + 2, 10), max_width)


def write_space_size_report(dataset_root: Path) -> Path:
    image_folders = []
    for folder in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        files = sorted(p for p in folder.iterdir() if is_image_file(p, SPACE_SIZE_SUFFIXES))
        if files:
            image_folders.append((folder.name, files))
    if not image_folders:
        raise RuntimeError(f"No .mha or .nii.gz files found inside child folders of: {dataset_root}")

    all_case_ids, folder_case_map = set(), {}
    for folder_name, files in image_folders:
        case_map = {strip_image_suffix(p): p for p in files}
        folder_case_map[folder_name] = case_map
        all_case_ids.update(case_map)

    case_ids = sorted(all_case_ids)
    columns = ["case_id"]
    for folder_name in folder_case_map:
        columns += [f"{folder_name}_{suffix}" for suffix in ("spacing_x", "spacing_y", "spacing_z", "size_x", "size_y", "size_z")]

    rows = []
    for case_id in case_ids:
        row = {"case_id": case_id}
        for folder_name, case_map in folder_case_map.items():
            img_path = case_map.get(case_id)
            vals = (None,) * 6
            if img_path is not None:
                img = sitk.ReadImage(str(img_path))
                spacing, size = img.GetSpacing(), img.GetSize()
                vals = (float(spacing[0]), float(spacing[1]), float(spacing[2]), int(size[0]), int(size[1]), int(size[2]))
            for suffix, val in zip(("spacing_x", "spacing_y", "spacing_z", "size_x", "size_y", "size_z"), vals):
                row[f"{folder_name}_{suffix}"] = val
        rows.append(row)

    average_row = {"case_id": "AVERAGE"}
    for col in columns[1:]:
        values = [row[col] for row in rows if row.get(col) is not None]
        average_row[col] = float(np.mean(values)) if values else None

    xlsx_path = dataset_root / "space_size.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "space_size"
    ws.append(columns)
    ws.append([average_row.get(col) for col in columns])
    for row in rows:
        ws.append([row.get(col) for col in columns])
    ws.freeze_panes = "A3"
    fit_excel_columns(ws)
    wb.save(xlsx_path)
    return xlsx_path


def ensure_space_size_report(dataset_root: Path) -> None:
    if (dataset_root / "space_size.xlsx").exists():
        return
    path = write_space_size_report(dataset_root)
    print(f"[PREPROCESS] saved {path}")


def resample_to_ref(moving: sitk.Image, ref: sitk.Image, is_label: bool) -> sitk.Image:
    filt = sitk.ResampleImageFilter()
    filt.SetReferenceImage(ref)
    filt.SetTransform(sitk.Transform())
    filt.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    filt.SetDefaultPixelValue(0)
    return filt.Execute(moving)


def make_ref_like(t2w_img: sitk.Image, target_spacing: tuple[float, float, float]) -> sitk.Image:
    old_size = np.array(t2w_img.GetSize(), dtype=float)
    old_spacing = np.array(t2w_img.GetSpacing(), dtype=float)
    new_spacing = np.array(target_spacing, dtype=float)
    new_size = [max(1, int(v)) for v in np.round(old_size * old_spacing / new_spacing)]
    ref = sitk.Image(new_size, t2w_img.GetPixelID())
    ref.SetSpacing(tuple(float(v) for v in target_spacing))
    ref.SetOrigin(t2w_img.GetOrigin())
    ref.SetDirection(t2w_img.GetDirection())
    return ref


def crop_pad_2d_center(slice2d: np.ndarray, cx: int, cy: int, out_w: int, out_h: int) -> np.ndarray:
    h, w = slice2d.shape
    x0, y0 = cx - out_w // 2, cy - out_h // 2
    x1, y1 = x0 + out_w, y0 + out_h
    pad_left, pad_right = max(0, -x0), max(0, x1 - w)
    pad_top, pad_bottom = max(0, -y0), max(0, y1 - h)
    if pad_left or pad_right or pad_top or pad_bottom:
        slice2d = np.pad(slice2d, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)
        x0, x1, y0, y1 = x0 + pad_left, x1 + pad_left, y0 + pad_top, y1 + pad_top
    out = slice2d[y0:y1, x0:x1]
    if out.shape != (out_h, out_w):
        out = np.pad(out[:out_h, :out_w], ((0, max(0, out_h - out.shape[0])), (0, max(0, out_w - out.shape[1]))), mode="constant", constant_values=0)
    return out[:out_h, :out_w]


def make_z_list(areas: np.ndarray, z0: int, need: int) -> list[int]:
    z_count = int(len(areas))
    if z_count < need:
        missing = need - z_count
        left_pad = missing // 2 + int(missing % 2 == 1 and z0 < z_count // 2)
        right_pad = missing - left_pad
        return ([0] * left_pad) + list(range(z_count)) + ([z_count - 1] * right_pad)
    csum = np.concatenate([[0], np.cumsum(np.asarray(areas, dtype=np.int64))])
    scores = csum[need:] - csum[:-need]
    starts = np.flatnonzero(scores == scores.max())
    centers = starts + (need - 1) / 2.0
    start = int(starts[np.argmin(np.abs(centers - z0))])
    return list(range(start, start + need))


def physical_point_from_continuous_index(img: sitk.Image, index_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    origin = np.array(img.GetOrigin(), dtype=float)
    spacing = np.array(img.GetSpacing(), dtype=float)
    direction = np.array(img.GetDirection(), dtype=float).reshape(3, 3)
    return tuple(float(v) for v in origin + direction.dot(np.array(index_xyz, dtype=float) * spacing))


def image_from_array(arr: np.ndarray, ref: sitk.Image, x0: int, y0: int, z0: int, is_mask: bool) -> sitk.Image:
    img = sitk.GetImageFromArray(arr.astype(np.uint8 if is_mask else np.float32))
    img.SetSpacing(ref.GetSpacing())
    img.SetDirection(ref.GetDirection())
    img.SetOrigin(physical_point_from_continuous_index(ref, (x0, y0, z0)))
    return img


def crop_volume(arr: np.ndarray, z_list: list[int], cx: int, cy: int, crop_size: tuple[int, int, int]) -> np.ndarray:
    out_w, out_h, _ = crop_size
    return np.stack([crop_pad_2d_center(arr[z], cx, cy, out_w, out_h) for z in z_list], axis=0)


def write_image(img: sitk.Image, out_dir: Path, case_id: str, is_mask: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_dir / f"{case_id}{'.nii.gz' if is_mask else '.mha'}"))


def mask_inside_ratio(mask_arr: np.ndarray, organ_arr: np.ndarray) -> float:
    mask = mask_arr > 0
    n_mask = int(mask.sum())
    if n_mask == 0:
        return float("nan")
    return float(np.logical_and(mask, organ_arr > 0).sum() / n_mask)


def finite_nonzero_fraction(values: np.ndarray, mask: np.ndarray) -> float:
    region = values[mask > 0]
    if region.size == 0:
        return float("nan")
    return float((np.isfinite(region) & (np.abs(region) > 1e-6)).mean())


def passes_final_filter(cropped_np: dict[str, np.ndarray]) -> tuple[bool, list[str]]:
    organ_bin = cropped_np[FOLDER_MASK_ORGAN] > 0
    reasons = []

    lesion = cropped_np.get(FOLDER_MASK_LESION)
    if lesion is not None:
        ratio = mask_inside_ratio(lesion, organ_bin)
        if not (np.isfinite(ratio) and ratio >= MIN_LESION_INSIDE_ORGAN):
            reasons.append("lesion_inside_organ")

    zone = cropped_np.get(FOLDER_MASK_ZONE)
    if zone is not None:
        ratio = mask_inside_ratio(zone, organ_bin)
        if not (np.isfinite(ratio) and ratio >= MIN_ZONE_INSIDE_ORGAN):
            reasons.append("zone_inside_organ")

    adc = cropped_np.get(FOLDER_ADC)
    if adc is not None:
        cov = finite_nonzero_fraction(adc, organ_bin)
        if not (np.isfinite(cov) and cov >= MIN_ADC_ORGAN_COVERAGE):
            reasons.append("adc_organ_coverage")
        if lesion is not None:
            lcov = finite_nonzero_fraction(adc, lesion > 0)
            if not (np.isfinite(lcov) and lcov >= MIN_ADC_LESION_COVERAGE):
                reasons.append("adc_lesion_coverage")

    return not reasons, reasons


def process_case(
    case_id: str, in_dirs: dict[str, Path], target_spacing: tuple[float, float, float], crop_size: tuple[int, int, int]
) -> tuple[dict[str, sitk.Image], dict[str, np.ndarray]] | str:
    paths = {name: find_case_file(folder, case_id, prefer_mask=is_mask_folder(name)) for name, folder in in_dirs.items()}
    missing_required = [name for name in REQUIRED_FOLDERS if paths.get(name) is None]
    if missing_required:
        return f"missing_required:{missing_required}"

    imgs = {name: sitk.ReadImage(str(path)) for name, path in paths.items() if path is not None}
    ref = make_ref_like(imgs[FOLDER_T2W], target_spacing)

    resampled: dict[str, sitk.Image] = {}
    for name, img in imgs.items():
        is_mask = is_mask_folder(name)
        out = resample_to_ref(img, ref, is_label=is_mask)
        resampled[name] = sitk.Cast(out, sitk.sitkUInt8) if is_mask else out

    arrays = {
        name: (sitk.GetArrayFromImage(img).astype(np.uint8) if is_mask_folder(name) else sitk.GetArrayFromImage(img).astype(np.float32))
        for name, img in resampled.items()
    }

    organ = arrays[FOLDER_MASK_ORGAN] > 0
    if int(organ.sum()) == 0:
        return "empty_organ"

    z_count, height, width = organ.shape
    areas = organ.reshape(z_count, -1).sum(axis=1)
    z0 = int(np.argmax(areas))
    coords = np.argwhere(organ[z0])
    cy = int(round(coords[:, 0].mean())) if coords.size else height // 2
    cx = int(round(coords[:, 1].mean())) if coords.size else width // 2
    target_w, target_h, need_slices = crop_size
    z_list = make_z_list(areas, z0, need_slices)
    x0, y0, first_z = cx - target_w // 2, cy - target_h // 2, int(z_list[0])

    t2w_ref = resampled[FOLDER_T2W]
    cropped_imgs: dict[str, sitk.Image] = {}
    cropped_np: dict[str, np.ndarray] = {}
    for name, arr in arrays.items():
        is_mask = is_mask_folder(name)
        crop_arr = crop_volume(arr, z_list, cx, cy, crop_size)
        out_img = image_from_array(crop_arr, t2w_ref, x0, y0, first_z, is_mask)
        if is_mask:
            out_img = sitk.Cast(out_img, sitk.sitkUInt8)
        cropped_imgs[name] = out_img
        cropped_np[name] = crop_arr

    return cropped_imgs, cropped_np


def target_spacing_str(spacing: tuple[float, float, float]) -> str:
    return "x".join(f"{v:g}" for v in spacing)


def preprocess_dataset(name: str) -> None:
    spec = DATASETS[name]
    original_root = Path(spec["original"])
    preprocessed_root = Path(spec["preprocessed"])
    crop_size = tuple(spec["crop_size"])
    filter_final = bool(spec.get("filter_cropped_final", False))

    if not original_root.exists():
        print(f"[PREPROCESS {name}] original data not found, skipping ({original_root})")
        return

    ensure_space_size_report(original_root)

    if dataset_ready(preprocessed_root):
        print(f"[PREPROCESS {name}] already preprocessed, skipping ({preprocessed_root})")
        return

    in_dirs = existing_input_dirs(original_root)
    out_dirs = {folder_name: preprocessed_root / folder_name for folder_name in in_dirs}
    case_ids = case_ids_from_t2w(in_dirs[FOLDER_T2W])
    print(f"[PREPROCESS {name}] {original_root} -> {preprocessed_root} | cases={len(case_ids)} | spacing={target_spacing_str(RESAMPLE_SPACING)} | crop={crop_size}")

    kept = skipped = filtered = 0
    filter_reason_counts: dict[str, int] = {}
    for i, case_id in enumerate(case_ids, 1):
        try:
            result = process_case(case_id, in_dirs, RESAMPLE_SPACING, crop_size)
        except Exception as exc:
            skipped += 1
            print(f"[PREPROCESS {name}] ERROR {case_id}: {exc!r}")
            continue

        if isinstance(result, str):
            skipped += 1
            print(f"[PREPROCESS {name}] SKIP {case_id}: {result}")
            continue

        cropped_imgs, cropped_np = result
        if filter_final:
            ok, reasons = passes_final_filter(cropped_np)
            if not ok:
                filtered += 1
                for reason in reasons:
                    filter_reason_counts[reason] = filter_reason_counts.get(reason, 0) + 1
                continue

        for folder_name, img in cropped_imgs.items():
            write_image(img, out_dirs[folder_name], case_id, is_mask_folder(folder_name))
        kept += 1
        cfg.log_progress(f"PREPROCESS {name}", i, len(case_ids))

    write_space_size_report(preprocessed_root)
    summary = f"[PREPROCESS {name}] done cases={len(case_ids)} kept={kept} skipped={skipped}"
    if filter_final:
        summary += f" filtered_by_qc={filtered} {filter_reason_counts}"
    print(summary)


def preprocess(names: list[str]) -> None:
    for name in names:
        preprocess_dataset(name)


def preprocess_train_datasets() -> None:
    preprocess(TRAIN_DATASETS)


def preprocess_test_dataset() -> None:
    preprocess(TEST_DATASETS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample + organ-centered crop of original datasets straight into their preprocessed folders.")
    parser.add_argument("--only", default=",".join(DATASETS), help=f"Comma-separated dataset keys: {', '.join(DATASETS)}.")
    args = parser.parse_args()
    names = [x.strip() for x in args.only.split(",") if x.strip()]
    invalid = [n for n in names if n not in DATASETS]
    if invalid:
        raise ValueError(f"Unknown dataset keys in --only: {invalid}")
    preprocess(names)


if __name__ == "__main__":
    main()
