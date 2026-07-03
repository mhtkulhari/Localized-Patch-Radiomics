from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage


SCRIPT_DIR = Path(__file__).resolve().parent

from all_config import (
    ADC_FOLDER,
    LESION_MASK_FOLDER,
    MASK_FOLDER,
    PATCH_FIXED_POSITIVE_PATCHES_PER_CASE,
    PATCH_HELPER_RANDOM_STATE,
    PATCH_MAX_MAJORITY_TO_MINORITY_RATIO,
    PATCH_MIN_POSITIVE_LESION_OVERLAP,
    PATCH_NONCSPCA_MAX_LESION_OVERLAP,
    PATCH_ORGAN_MIN_OVERLAP,
    PATCH_ZONE_MIN_OVERLAP,
    T2W_FOLDER,
    ZONE_MASK_FOLDER,
    ZONE_VALUE_TO_LABEL,
)


EPS = 1e-12
IMAGE_SUFFIXES = (".nii.gz", ".mha", ".nii", ".nrrd", ".mhd")

PATCH_META_COLS = {
    "case_id",
    "scale",
    "center_z",
    "center_y",
    "center_x",
    "patch_z",
    "patch_y",
    "patch_x",
    "organ_overlap",
    "patch_source",
    "lesion_overlap",
    "lesion_coverage",
    "center_in_lesion",
    "pz_overlap",
    "tz_overlap",
    "cs_patch_label",
    "zone_patch_label",
    "sel_lesion_voxels",
    "sel_lesion_candidates_sampled",
    "sel_lesion_inside_image",
    "sel_lesion_overlap_ge_threshold",
    "sel_lesion_after_top_cap",
    "sel_negative_grid_candidates",
    "sel_negative_no_lesion_pz_pool",
    "sel_negative_no_lesion_tz_pool",
    "sel_negative_target_total",
    "sel_negative_selected_pz",
    "sel_negative_selected_tz",
    "sel_final_positive",
    "sel_final_negative",
}


@dataclass(frozen=True)
class PatchScale:
    name: str
    size_mm_zyx: tuple[float, float, float]
    stride_mm_zyx: tuple[float, float, float]


@dataclass
class CaseArrays:
    case_id: str
    t2w: np.ndarray
    adc: np.ndarray
    organ: np.ndarray
    lesion: np.ndarray | None
    zone: np.ndarray | None
    reference_image: sitk.Image
    spacing_zyx: tuple[float, float, float]


def normalize_scale(scale: dict | PatchScale) -> PatchScale:
    if isinstance(scale, PatchScale):
        return scale
    return PatchScale(
        name=str(scale["name"]),
        size_mm_zyx=tuple(float(v) for v in scale["size_mm_zyx"]),
        stride_mm_zyx=tuple(float(v) for v in scale["stride_mm_zyx"]),
    )


def strip_image_suffix(path: Path) -> str:
    name = path.name
    for suffix in IMAGE_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def find_case_file(folder: Path, case_id: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        path = folder / f"{case_id}{suffix}"
        if path.exists():
            return path
    return None


def list_case_ids(root: Path, image_folder: str = T2W_FOLDER) -> list[str]:
    folder = root / image_folder
    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")
    ids = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.name.lower().endswith(IMAGE_SUFFIXES):
            ids.append(strip_image_suffix(path))
    return ids


def read_image_array(path: Path) -> tuple[np.ndarray, sitk.Image, tuple[float, float, float]]:
    image = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing_xyz = image.GetSpacing()
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    return arr, image, spacing_zyx


def _read_mask(path: Path, dtype=np.uint8) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(dtype)


def load_case_arrays(root: Path, case_id: str, require_lesion: bool, require_zone: bool) -> CaseArrays | None:
    t2w_path = find_case_file(root / T2W_FOLDER, case_id)
    adc_path = find_case_file(root / ADC_FOLDER, case_id)
    organ_path = find_case_file(root / MASK_FOLDER, case_id)
    lesion_path = find_case_file(root / LESION_MASK_FOLDER, case_id)
    zone_path = find_case_file(root / ZONE_MASK_FOLDER, case_id)

    if t2w_path is None or adc_path is None or organ_path is None:
        return None
    if require_lesion and lesion_path is None:
        return None
    if require_zone and zone_path is None:
        return None

    t2w, reference_image, spacing_zyx = read_image_array(t2w_path)
    adc, _, _ = read_image_array(adc_path)
    organ = _read_mask(organ_path) > 0
    lesion = (_read_mask(lesion_path) > 0) if require_lesion and lesion_path is not None else None
    zone = _read_mask(zone_path, dtype=np.int16) if require_zone and zone_path is not None else None

    shapes = [t2w.shape, adc.shape, organ.shape]
    if lesion is not None:
        shapes.append(lesion.shape)
    if zone is not None:
        shapes.append(zone.shape)
    if len(set(shapes)) != 1:
        return None

    return CaseArrays(
        case_id=case_id,
        t2w=t2w,
        adc=adc,
        organ=organ,
        lesion=lesion,
        zone=zone,
        reference_image=reference_image,
        spacing_zyx=spacing_zyx,
    )


def bbox_from_mask(mask: np.ndarray) -> tuple[slice, slice, slice]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        z, y, x = mask.shape
        return slice(0, z), slice(0, y), slice(0, x)
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    return slice(zmin, zmax + 1), slice(ymin, ymax + 1), slice(xmin, xmax + 1)


def _odd_voxels_from_mm(size_mm_zyx: Iterable[float], spacing_zyx: Iterable[float]) -> tuple[int, int, int]:
    voxels = []
    for size_mm, spacing in zip(size_mm_zyx, spacing_zyx):
        n = max(1, int(round(float(size_mm) / max(float(spacing), EPS))))
        if n % 2 == 0:
            n += 1
        voxels.append(n)
    return tuple(voxels)


def _stride_voxels_from_mm(stride_mm_zyx: Iterable[float], spacing_zyx: Iterable[float]) -> tuple[int, int, int]:
    return tuple(
        max(1, int(round(float(stride_mm) / max(float(spacing), EPS))))
        for stride_mm, spacing in zip(stride_mm_zyx, spacing_zyx)
    )


def patch_shape_vox(scale: PatchScale, spacing_zyx: tuple[float, float, float]) -> tuple[int, int, int]:
    return _odd_voxels_from_mm(scale.size_mm_zyx, spacing_zyx)


def patch_radius_vox(scale: PatchScale, spacing_zyx: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(n // 2 for n in patch_shape_vox(scale, spacing_zyx))


def patch_slices_from_center(
    center_zyx: tuple[int, int, int],
    radius_zyx: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    z, y, x = center_zyx
    rz, ry, rx = radius_zyx
    return (
        slice(z - rz, z + rz + 1),
        slice(y - ry, y + ry + 1),
        slice(x - rx, x + rx + 1),
    )


def _patch_if_inside_image(
    image_shape: tuple[int, int, int],
    center_zyx: tuple[int, int, int],
    radius_zyx: tuple[int, int, int],
) -> tuple[slice, slice, slice] | None:
    z, y, x = center_zyx
    rz, ry, rx = radius_zyx
    if (
        z - rz < 0
        or y - ry < 0
        or x - rx < 0
        or z + rz + 1 > image_shape[0]
        or y + ry + 1 > image_shape[1]
        or x + rx + 1 > image_shape[2]
    ):
        return None
    return patch_slices_from_center(center_zyx, radius_zyx)


def _patch_if_inside_organ(
    organ_mask: np.ndarray,
    center_zyx: tuple[int, int, int],
    radius_zyx: tuple[int, int, int],
    min_organ_overlap: float = PATCH_ORGAN_MIN_OVERLAP,
) -> tuple[tuple[slice, slice, slice], float] | None:
    z, y, x = center_zyx
    rz, ry, rx = radius_zyx
    if (
        z - rz < 0
        or y - ry < 0
        or x - rx < 0
        or z + rz + 1 > organ_mask.shape[0]
        or y + ry + 1 > organ_mask.shape[1]
        or x + rx + 1 > organ_mask.shape[2]
    ):
        return None
    sl = patch_slices_from_center(center_zyx, radius_zyx)
    organ_patch = organ_mask[sl]
    organ_overlap = float(np.mean(organ_patch))
    if min_organ_overlap >= 1.0:
        ok = bool(np.all(organ_patch))
    else:
        ok = organ_overlap >= min_organ_overlap
    if not ok:
        return None
    return sl, organ_overlap


def iter_patch_centers(
    organ_mask: np.ndarray,
    scale: dict | PatchScale,
    spacing_zyx: tuple[float, float, float],
    min_organ_overlap: float = PATCH_ORGAN_MIN_OVERLAP,
) -> Iterable[tuple[tuple[int, int, int], tuple[slice, slice, slice], float]]:
    sc = normalize_scale(scale)
    shape = patch_shape_vox(sc, spacing_zyx)
    radius = tuple(n // 2 for n in shape)
    stride = _stride_voxels_from_mm(sc.stride_mm_zyx, spacing_zyx)
    zsl, ysl, xsl = bbox_from_mask(organ_mask)

    z_start = max(zsl.start + radius[0], radius[0])
    y_start = max(ysl.start + radius[1], radius[1])
    x_start = max(xsl.start + radius[2], radius[2])
    z_stop = min(zsl.stop - radius[0], organ_mask.shape[0] - radius[0])
    y_stop = min(ysl.stop - radius[1], organ_mask.shape[1] - radius[1])
    x_stop = min(xsl.stop - radius[2], organ_mask.shape[2] - radius[2])

    if z_start >= z_stop or y_start >= y_stop or x_start >= x_stop:
        return

    for z in range(z_start, z_stop, stride[0]):
        for y in range(y_start, y_stop, stride[1]):
            for x in range(x_start, x_stop, stride[2]):
                center = (z, y, x)
                patch = _patch_if_inside_organ(organ_mask, center, radius, min_organ_overlap)
                if patch is not None:
                    sl, organ_overlap = patch
                    yield center, sl, organ_overlap


def iter_lesion_guided_patch_centers(
    organ_mask: np.ndarray,
    lesion_mask: np.ndarray,
    scale: dict | PatchScale,
    spacing_zyx: tuple[float, float, float],
    min_organ_overlap: float = PATCH_ORGAN_MIN_OVERLAP,
) -> Iterable[tuple[tuple[int, int, int], tuple[slice, slice, slice], float]]:
    sc = normalize_scale(scale)
    radius = patch_radius_vox(sc, spacing_zyx)
    coords = np.argwhere(lesion_mask > 0)
    if coords.size == 0:
        return

    candidates = [tuple(np.round(coords.mean(axis=0)).astype(int).tolist())]
    max_centers = max(1, int(PATCH_FIXED_POSITIVE_PATCHES_PER_CASE))
    step = max(1, int(math.ceil(coords.shape[0] / max_centers)))
    candidates.extend(tuple(int(v) for v in c) for c in coords[::step])

    seen = set()
    for center in candidates:
        if center in seen:
            continue
        seen.add(center)
        patch = _patch_if_inside_organ(organ_mask, center, radius, min_organ_overlap)
        if patch is not None:
            sl, organ_overlap = patch
            yield center, sl, organ_overlap


def _safe_stats(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "var": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "iqr": np.nan,
            "skew": np.nan,
            "kurt": np.nan,
            "rms": np.nan,
            "energy": np.nan,
        }
    mean = float(np.mean(x))
    std = float(np.std(x))
    centered = x - mean
    p10, p25, p75, p90 = np.percentile(x, [10, 25, 75, 90])
    skew = float(np.mean((centered / (std + EPS)) ** 3)) if std > EPS else 0.0
    kurt = float(np.mean((centered / (std + EPS)) ** 4)) if std > EPS else 0.0
    return {
        "mean": mean,
        "std": std,
        "var": float(np.var(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "p10": float(p10),
        "p25": float(p25),
        "p75": float(p75),
        "p90": float(p90),
        "iqr": float(p75 - p25),
        "skew": skew,
        "kurt": kurt,
        "rms": float(np.sqrt(np.mean(x * x))),
        "energy": float(np.mean(x * x)),
    }


def _quantize(img: np.ndarray, levels: int = 16) -> np.ndarray:
    x = img.astype(np.float32, copy=False)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.uint8)
    mn = float(np.min(x[finite]))
    mx = float(np.max(x[finite]))
    if mx - mn < EPS:
        return np.zeros_like(x, dtype=np.uint8)
    q = np.floor((x - mn) / (mx - mn + EPS) * levels)
    return np.clip(q, 0, levels - 1).astype(np.uint8)


def _glcm_features_2d(img2d: np.ndarray, levels: int = 16) -> dict[str, float]:
    q = _quantize(img2d, levels=levels)
    offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]
    glcm = np.zeros((levels, levels), dtype=np.float64)
    h, w = q.shape
    for dy, dx in offsets:
        y0 = max(0, dy)
        y1 = h + min(0, dy)
        x0 = max(0, dx)
        x1 = w + min(0, dx)
        a = q[y0:y1, x0:x1].ravel()
        b = q[y0 - dy:y1 - dy, x0 - dx:x1 - dx].ravel()
        np.add.at(glcm, (a, b), 1)
        np.add.at(glcm, (b, a), 1)

    total = float(glcm.sum())
    if total <= 0:
        return {
            "glcm_contrast": np.nan,
            "glcm_dissimilarity": np.nan,
            "glcm_homogeneity": np.nan,
            "glcm_asm": np.nan,
            "glcm_energy": np.nan,
            "glcm_entropy": np.nan,
            "glcm_correlation": np.nan,
        }

    p = glcm / total
    i = np.arange(levels, dtype=np.float64).reshape(-1, 1)
    j = np.arange(levels, dtype=np.float64).reshape(1, -1)
    diff = i - j
    asm = float(np.sum(p * p))
    pi = p.sum(axis=1)
    pj = p.sum(axis=0)
    mui = float(np.sum(np.arange(levels) * pi))
    muj = float(np.sum(np.arange(levels) * pj))
    sdi = math.sqrt(float(np.sum(((np.arange(levels) - mui) ** 2) * pi)) + EPS)
    sdj = math.sqrt(float(np.sum(((np.arange(levels) - muj) ** 2) * pj)) + EPS)
    corr = float(np.sum((i - mui) * (j - muj) * p) / (sdi * sdj + EPS))
    nz = p[p > 0]
    return {
        "glcm_contrast": float(np.sum((diff * diff) * p)),
        "glcm_dissimilarity": float(np.sum(np.abs(diff) * p)),
        "glcm_homogeneity": float(np.sum(p / (1.0 + np.abs(diff)))),
        "glcm_asm": asm,
        "glcm_energy": math.sqrt(asm),
        "glcm_entropy": float(-np.sum(nz * np.log(nz))),
        "glcm_correlation": corr,
    }


def _modality_features(prefix: str, patch: np.ndarray) -> dict[str, float]:
    out = {f"{prefix}_{k}": v for k, v in _safe_stats(patch).items()}
    if min(patch.shape) > 1:
        grads = np.gradient(patch.astype(np.float32), edge_order=1)
        grad_mag = np.sqrt(sum(g * g for g in grads))
        out.update({f"{prefix}_grad_{k}": v for k, v in _safe_stats(grad_mag).items()})
    elif patch.ndim == 3 and min(patch.shape[1:]) > 1:
        central_z = patch.shape[0] // 2
        grads = np.gradient(patch[central_z].astype(np.float32), edge_order=1)
        grad_mag = np.sqrt(sum(g * g for g in grads))
        out.update({f"{prefix}_grad_{k}": v for k, v in _safe_stats(grad_mag).items()})
    else:
        out.update({f"{prefix}_grad_{k}": np.nan for k in _safe_stats(np.array([])).keys()})

    central_z = patch.shape[0] // 2
    out.update({f"{prefix}_{k}": v for k, v in _glcm_features_2d(patch[central_z]).items()})
    return out


def _gradient_magnitude(arr: np.ndarray) -> np.ndarray:
    if min(arr.shape) > 1:
        grads = np.gradient(arr.astype(np.float32), edge_order=1)
        return np.sqrt(sum(g * g for g in grads))
    if arr.ndim == 3 and min(arr.shape[1:]) > 1:
        central_z = arr.shape[0] // 2
        grads = np.gradient(arr[central_z].astype(np.float32), edge_order=1)
        grad2d = np.sqrt(sum(g * g for g in grads))
        out = np.zeros_like(arr, dtype=np.float32)
        out[central_z] = grad2d
        return out
    return np.full(arr.shape, np.nan, dtype=np.float32)


def _masked_glcm_features_2d(img2d: np.ndarray, mask2d: np.ndarray) -> dict[str, float]:
    mask = np.asarray(mask2d, dtype=bool)
    if not mask.any():
        return _glcm_features_2d(np.array([[np.nan]], dtype=np.float32))
    coords = np.argwhere(mask)
    ymin, xmin = coords.min(axis=0)
    ymax, xmax = coords.max(axis=0)
    crop = np.asarray(img2d[ymin : ymax + 1, xmin : xmax + 1], dtype=np.float32).copy()
    crop_mask = mask[ymin : ymax + 1, xmin : xmax + 1]
    vals = crop[crop_mask]
    vals = vals[np.isfinite(vals)]
    fill = float(np.median(vals)) if vals.size else 0.0
    crop[~crop_mask] = fill
    return _glcm_features_2d(crop)


def _unprefixed_modality_features(patch: np.ndarray) -> dict[str, float]:
    out = dict(_safe_stats(patch))
    grad = _gradient_magnitude(patch)
    out.update({f"grad_{k}": v for k, v in _safe_stats(grad).items()})
    central_z = patch.shape[0] // 2
    out.update(_glcm_features_2d(patch[central_z]))
    return out


def _unprefixed_ring_features(context_patch: np.ndarray, ring_mask: np.ndarray) -> dict[str, float]:
    out = dict(_safe_stats(context_patch[ring_mask]))
    grad = _gradient_magnitude(context_patch)
    out.update({f"grad_{k}": v for k, v in _safe_stats(grad[ring_mask]).items()})
    central_z = context_patch.shape[0] // 2
    out.update(_masked_glcm_features_2d(context_patch[central_z], ring_mask[central_z]))
    return out


def _slice_for_radius(
    center_zyx: tuple[int, int, int],
    radius_zyx: tuple[int, int, int],
    shape: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    out = []
    for c, r, dim in zip(center_zyx, radius_zyx, shape):
        lo = max(0, int(c) - int(r))
        hi = min(int(dim), int(c) + int(r) + 1)
        out.append(slice(lo, hi))
    return tuple(out)


def _ring_mask_for_context(
    context_sl: tuple[slice, slice, slice],
    core_sl: tuple[slice, slice, slice],
) -> np.ndarray:
    shape = tuple(int(s.stop - s.start) for s in context_sl)
    ring = np.ones(shape, dtype=bool)
    rel = tuple(slice(int(c.start - ctx.start), int(c.stop - ctx.start)) for c, ctx in zip(core_sl, context_sl))
    ring[rel] = False
    return ring


def _context_modality_features(
    prefix: str,
    core_patch: np.ndarray,
    context_patch: np.ndarray,
    ring_mask: np.ndarray,
) -> dict[str, float]:
    out = _modality_features(f"{prefix}_core", core_patch)
    ring_features = _unprefixed_ring_features(context_patch, ring_mask)
    out.update({f"{prefix}_ring_{k}": v for k, v in ring_features.items()})

    core_features = _unprefixed_modality_features(core_patch)
    for key, core_value in core_features.items():
        ring_value = ring_features.get(key, np.nan)
        out[f"{prefix}_core_minus_ring_{key}"] = float(core_value - ring_value) if np.isfinite(core_value) and np.isfinite(ring_value) else np.nan
        out[f"{prefix}_core_ring_ratio_{key}"] = float(core_value / (ring_value + EPS)) if np.isfinite(core_value) and np.isfinite(ring_value) else np.nan
    return out


def extract_patch_features(
    t2w_patch: np.ndarray,
    adc_patch: np.ndarray,
    t2w_context_patch: np.ndarray | None = None,
    adc_context_patch: np.ndarray | None = None,
    ring_mask: np.ndarray | None = None,
) -> dict[str, float]:
    out = {}
    if t2w_context_patch is not None and adc_context_patch is not None and ring_mask is not None and bool(np.any(ring_mask)):
        out.update(_context_modality_features("t2w", t2w_patch, t2w_context_patch, ring_mask))
        out.update(_context_modality_features("adc", adc_patch, adc_context_patch, ring_mask))
    else:
        out.update(_modality_features("t2w_core", t2w_patch))
        out.update(_modality_features("adc_core", adc_patch))

    t = t2w_patch.astype(np.float64).ravel()
    a = adc_patch.astype(np.float64).ravel()
    finite = np.isfinite(t) & np.isfinite(a)
    if finite.sum() >= 3 and np.std(t[finite]) > EPS and np.std(a[finite]) > EPS:
        corr = float(np.corrcoef(t[finite], a[finite])[0, 1])
    else:
        corr = 0.0
    out["t2w_adc_core_corr"] = corr
    out["t2w_adc_core_mean_diff"] = float(np.nanmean(t) - np.nanmean(a))
    out["t2w_adc_core_median_diff"] = float(np.nanmedian(t) - np.nanmedian(a))
    out["t2w_adc_core_std_ratio"] = float(np.nanstd(t) / (np.nanstd(a) + EPS))
    out["t2w_adc_core_mean_ratio"] = float(np.nanmean(t) / (np.nanmean(a) + EPS))
    return out


def patch_label_values(
    center_zyx: tuple[int, int, int],
    sl: tuple[slice, slice, slice],
    lesion: np.ndarray | None,
    zone: np.ndarray | None,
    lesion_zone_label: int | None = None,
) -> dict[str, float | int]:
    out: dict[str, float | int] = {
        "lesion_overlap": np.nan,
        "lesion_coverage": np.nan,
        "center_in_lesion": np.nan,
        "pz_overlap": np.nan,
        "tz_overlap": np.nan,
        "cs_patch_label": np.nan,
        "zone_patch_label": np.nan,
    }
    if lesion is not None:
        lesion_mask = lesion > 0
        patch_lesion_voxels = int(np.sum(lesion_mask[sl]))
        total_lesion_voxels = int(np.sum(lesion_mask))
        lesion_overlap = float(patch_lesion_voxels / max(1, lesion_mask[sl].size))
        lesion_coverage = float(patch_lesion_voxels / max(1, total_lesion_voxels))
        z, y, x = center_zyx
        center_in_lesion = bool(lesion_mask[z, y, x])
        out["lesion_overlap"] = lesion_overlap
        out["lesion_coverage"] = lesion_coverage
        out["center_in_lesion"] = int(center_in_lesion)
        if lesion_overlap >= PATCH_MIN_POSITIVE_LESION_OVERLAP:
            out["cs_patch_label"] = 1
        elif lesion_overlap <= PATCH_NONCSPCA_MAX_LESION_OVERLAP:
            out["cs_patch_label"] = 0

    if zone is not None:
        pz_value = next(k for k, v in ZONE_VALUE_TO_LABEL.items() if v == "PZ")
        tz_value = next(k for k, v in ZONE_VALUE_TO_LABEL.items() if v == "TZ")
        patch_zone = zone[sl]
        pz_overlap = float(np.mean(patch_zone == pz_value))
        tz_overlap = float(np.mean(patch_zone == tz_value))
        out["pz_overlap"] = pz_overlap
        out["tz_overlap"] = tz_overlap
        if out["cs_patch_label"] == 1:
            if lesion_zone_label is not None:
                out["zone_patch_label"] = int(lesion_zone_label)
        elif pz_overlap >= PATCH_ZONE_MIN_OVERLAP and pz_overlap >= tz_overlap:
            out["zone_patch_label"] = 0
        elif tz_overlap >= PATCH_ZONE_MIN_OVERLAP and tz_overlap > pz_overlap:
            out["zone_patch_label"] = 1
    return out


def lesion_zone_label_for_case(lesion: np.ndarray | None, zone: np.ndarray | None) -> int | None:
    if lesion is None or zone is None:
        return None
    lesion_mask = lesion > 0
    if not np.any(lesion_mask):
        return None
    pz_value = next(k for k, v in ZONE_VALUE_TO_LABEL.items() if v == "PZ")
    tz_value = next(k for k, v in ZONE_VALUE_TO_LABEL.items() if v == "TZ")
    lesion_zone = zone[lesion_mask]
    pz_count = int(np.sum(lesion_zone == pz_value))
    tz_count = int(np.sum(lesion_zone == tz_value))
    if pz_count == 0 and tz_count == 0:
        return None
    return 1 if tz_count > pz_count else 0


def _stable_rng(case_id: str, scale_name: str) -> np.random.Generator:
    import hashlib

    text = f"{PATCH_HELPER_RANDOM_STATE}|{case_id}|{scale_name}"
    seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def _sample_items(items: list, max_n: int, rng: np.random.Generator) -> list:
    if max_n <= 0 or len(items) <= max_n:
        return list(items)
    idx = rng.choice(len(items), size=int(max_n), replace=False)
    return [items[int(i)] for i in idx]


def _training_patch_centers_for_case(
    case: CaseArrays,
    sc: PatchScale,
) -> tuple[list[tuple[tuple[int, int, int], tuple[slice, slice, slice], float, str]], dict[str, int]]:
    empty_diag = {
        "sel_lesion_voxels": 0,
        "sel_lesion_candidates_sampled": 0,
        "sel_lesion_inside_image": 0,
        "sel_lesion_overlap_ge_threshold": 0,
        "sel_lesion_after_top_cap": 0,
        "sel_negative_grid_candidates": 0,
        "sel_negative_no_lesion_pz_pool": 0,
        "sel_negative_no_lesion_tz_pool": 0,
        "sel_negative_target_total": 0,
        "sel_negative_selected_pz": 0,
        "sel_negative_selected_tz": 0,
        "sel_final_positive": 0,
        "sel_final_negative": 0,
    }
    if case.lesion is None or case.zone is None:
        return [], empty_diag

    rng = _stable_rng(case.case_id, sc.name)
    max_pos = max(1, int(PATCH_FIXED_POSITIVE_PATCHES_PER_CASE))
    radius = patch_radius_vox(sc, case.spacing_zyx)
    lesion_coords = np.argwhere(case.lesion > 0)
    lesion_zone_label = lesion_zone_label_for_case(case.lesion, case.zone)
    diag = dict(empty_diag)
    diag["sel_lesion_voxels"] = int(len(lesion_coords))

    high_pos = []
    if lesion_coords.size:
        diag["sel_lesion_candidates_sampled"] = int(len(lesion_coords))
        for coord in lesion_coords:
            center = tuple(int(v) for v in coord)
            sl = _patch_if_inside_image(case.organ.shape, center, radius)
            if sl is None:
                continue
            diag["sel_lesion_inside_image"] += 1
            labels = patch_label_values(center, sl, case.lesion, case.zone, lesion_zone_label=lesion_zone_label)
            lesion_overlap = float(labels["lesion_overlap"])
            if lesion_overlap >= PATCH_MIN_POSITIVE_LESION_OVERLAP:
                diag["sel_lesion_overlap_ge_threshold"] += 1
                high_pos.append((center, sl, np.nan, "lesion_core", lesion_overlap, float(rng.random())))
    high_pos = [
        item[:4]
        for item in sorted(high_pos, key=lambda item: (-item[4], item[5]))[:max_pos]
    ]
    diag["sel_lesion_after_top_cap"] = int(len(high_pos))

    neg_by_zone: dict[int, list] = {0: [], 1: []}
    for center, sl, organ_overlap in iter_patch_centers(case.organ, sc, case.spacing_zyx):
        diag["sel_negative_grid_candidates"] += 1
        labels = patch_label_values(center, sl, case.lesion, case.zone, lesion_zone_label=lesion_zone_label)
        lesion_overlap = float(labels["lesion_overlap"])
        zone_label = labels["zone_patch_label"]
        if lesion_overlap <= PATCH_NONCSPCA_MAX_LESION_OVERLAP and not pd.isna(zone_label):
            neg_by_zone[int(zone_label)].append((center, sl, organ_overlap, "organ_far_negative"))
    diag["sel_negative_no_lesion_pz_pool"] = int(len(neg_by_zone[0]))
    diag["sel_negative_no_lesion_tz_pool"] = int(len(neg_by_zone[1]))

    positives = list(high_pos)
    seen = {item[0] for item in positives}
    positives = [item for item in positives if item[0] in seen]

    target_neg_total = int(math.ceil(len(positives) * float(PATCH_MAX_MAJORITY_TO_MINORITY_RATIO)))
    diag["sel_negative_target_total"] = int(target_neg_total)
    target_pz = target_neg_total // 2
    target_tz = target_neg_total - target_pz
    neg_pz = _sample_items(neg_by_zone[0], target_pz, rng) if target_pz > 0 else []
    neg_tz = _sample_items(neg_by_zone[1], target_tz, rng) if target_tz > 0 else []
    negatives = neg_pz + neg_tz
    diag["sel_negative_selected_pz"] = int(len(neg_pz))
    diag["sel_negative_selected_tz"] = int(len(neg_tz))

    if len(negatives) < target_neg_total:
        remaining = [item for item in (neg_by_zone[0] + neg_by_zone[1]) if item not in negatives]
        extra_negatives = _sample_items(remaining, target_neg_total - len(negatives), rng)
        negatives.extend(extra_negatives)
        for _, _, _, source in extra_negatives:
            pass

    selected = []
    selected_seen = set()
    for item in positives + negatives:
        if item[0] in selected_seen:
            continue
        selected_seen.add(item[0])
        selected.append(item)
    diag["sel_final_positive"] = int(len(positives))
    diag["sel_final_negative"] = int(len(negatives))
    return selected, diag


def build_patch_dataframe_for_case(
    case: CaseArrays,
    scale: dict | PatchScale,
    include_labels: bool,
) -> pd.DataFrame:
    sc = normalize_scale(scale)
    rows = []
    seen_centers = set()
    patch_shape = patch_shape_vox(sc, case.spacing_zyx)
    core_radius = patch_radius_vox(sc, case.spacing_zyx)
    context_radius = tuple(max(1, int(r) * 2) for r in core_radius)
    lesion_zone_label = lesion_zone_label_for_case(case.lesion, case.zone) if include_labels else None

    if include_labels and case.lesion is not None and case.zone is not None:
        selected_centers, selection_diag = _training_patch_centers_for_case(case, sc)
    else:
        selection_diag = {}
        selected_centers = [
            (center, sl, organ_overlap, "organ_grid")
            for center, sl, organ_overlap in iter_patch_centers(case.organ, sc, case.spacing_zyx)
        ]

    for center, sl, organ_overlap, patch_source in selected_centers:
        if center in seen_centers:
            continue
        seen_centers.add(center)
        z, y, x = center
        row = {
            "case_id": str(case.case_id),
            "scale": sc.name,
            "center_z": int(z),
            "center_y": int(y),
            "center_x": int(x),
            "patch_z": int(patch_shape[0]),
            "patch_y": int(patch_shape[1]),
            "patch_x": int(patch_shape[2]),
            "organ_overlap": float(organ_overlap),
            "patch_source": str(patch_source),
        }
        if include_labels:
            row.update(patch_label_values(center, sl, case.lesion, case.zone, lesion_zone_label=lesion_zone_label))
            row.update(selection_diag)
        context_sl = _slice_for_radius(center, context_radius, case.organ.shape)
        ring_mask = _ring_mask_for_context(context_sl, sl)
        row.update(
            extract_patch_features(
                case.t2w[sl],
                case.adc[sl],
                case.t2w[context_sl],
                case.adc[context_sl],
                ring_mask,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in PATCH_META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def copy_array_to_image(arr: np.ndarray, reference_image: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(arr.astype(np.float32))
    image.CopyInformation(reference_image)
    return image


def write_heatmap(path: Path, heatmap: np.ndarray, reference_image: sitk.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(copy_array_to_image(heatmap, reference_image), str(path))


def make_average_heatmap(
    shape: tuple[int, int, int],
    patch_rows: pd.DataFrame,
    score_col: str,
    scale: dict | PatchScale,
    spacing_zyx: tuple[float, float, float],
) -> np.ndarray:
    sc = normalize_scale(scale)
    radius = patch_radius_vox(sc, spacing_zyx)
    acc = np.zeros(shape, dtype=np.float32)
    cnt = np.zeros(shape, dtype=np.float32)
    for row in patch_rows.itertuples(index=False):
        score = getattr(row, score_col)
        if not np.isfinite(score):
            continue
        center = (int(row.center_z), int(row.center_y), int(row.center_x))
        sl = patch_slices_from_center(center, radius)
        acc[sl] += float(score)
        cnt[sl] += 1.0
    heatmap = np.zeros(shape, dtype=np.float32)
    np.divide(acc, cnt, out=heatmap, where=cnt > 0)
    return heatmap


def largest_component_volume_mm3(
    heatmap: np.ndarray,
    threshold: float,
    spacing_zyx: tuple[float, float, float],
) -> tuple[int, float]:
    mask = heatmap >= threshold
    if not mask.any():
        return 0, 0.0
    labels, n_labels = ndimage.label(mask)
    if n_labels == 0:
        return 0, 0.0
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest_voxels = int(counts.max())
    voxel_volume = float(np.prod(spacing_zyx))
    return largest_voxels, largest_voxels * voxel_volume


def _empty_cluster_features(prefix: str) -> dict[str, float]:
    keys = [
        "raw_num_clusters",
        "raw_total_hot_voxels",
        "raw_largest_cluster_voxels",
        "raw_largest_cluster_mm3",
        "raw_largest_cluster_z_slices",
        "raw_largest_cluster_z_span_mm",
        "raw_largest_cluster_mean_score",
        "raw_largest_cluster_max_score",
        "raw_largest_cluster_p90_score",
        "valid_num_clusters",
        "valid_total_hot_voxels",
        "valid_total_hot_mm3",
        "valid_hot_density",
        "valid_largest_cluster_voxels",
        "valid_largest_cluster_mm3",
        "valid_largest_cluster_z_slices",
        "valid_largest_cluster_z_span_mm",
        "valid_largest_cluster_mean_score",
        "valid_largest_cluster_max_score",
        "valid_largest_cluster_p90_score",
        "valid_largest_cluster_weighted_score",
        "ignored_single_slice_clusters",
        "ignored_single_slice_voxels",
    ]
    return {f"{prefix}_{k}": np.nan for k in keys}


def _component_rows(
    heatmap: np.ndarray,
    threshold: float,
    spacing_zyx: tuple[float, float, float],
    organ_mask: np.ndarray | None = None,
    structure: np.ndarray | None = None,
) -> list[dict[str, float]]:
    h = np.asarray(heatmap, dtype=np.float32)
    finite = np.isfinite(h)
    mask = finite & (h >= float(threshold))
    if organ_mask is not None:
        mask &= organ_mask > 0
    if not mask.any():
        return []

    if structure is None:
        structure = np.ones((3, 3, 3), dtype=bool)
    labels, n_labels = ndimage.label(mask, structure=structure)
    voxel_volume = float(np.prod(spacing_zyx))
    rows: list[dict[str, float]] = []
    for label in range(1, n_labels + 1):
        comp = labels == label
        if not comp.any():
            continue
        vals = h[comp]
        z_idx = np.where(np.any(comp, axis=(1, 2)))[0]
        z_slices = int(len(z_idx))
        z_span_vox = int(z_idx[-1] - z_idx[0] + 1) if z_slices else 0
        voxels = int(np.sum(comp))
        rows.append(
            {
                "label": int(label),
                "voxels": voxels,
                "mm3": float(voxels * voxel_volume),
                "z_slices": z_slices,
                "z_span_vox": z_span_vox,
                "z_span_mm": float(z_span_vox * spacing_zyx[0]),
                "mean_score": float(np.nanmean(vals)),
                "max_score": float(np.nanmax(vals)),
                "p90_score": float(np.nanpercentile(vals, 90)),
            }
        )
    return rows


def z_continuous_hotspot_features(
    prefix: str,
    heatmap: np.ndarray,
    threshold: float,
    spacing_zyx: tuple[float, float, float],
    organ_mask: np.ndarray | None = None,
    min_z_slices: int = 2,
) -> dict[str, float]:
    rows = _component_rows(heatmap, threshold, spacing_zyx, organ_mask=organ_mask)
    if not rows:
        out = _empty_cluster_features(prefix)
        return {k: 0.0 for k in out}

    raw_largest = max(rows, key=lambda r: r["voxels"])
    valid = [r for r in rows if r["z_slices"] >= int(min_z_slices)]
    ignored = [r for r in rows if r["z_slices"] < int(min_z_slices)]
    organ_voxels = int(np.sum(organ_mask > 0)) if organ_mask is not None else int(np.prod(np.asarray(heatmap).shape))
    voxel_volume = float(np.prod(spacing_zyx))

    out = {
        f"{prefix}_raw_num_clusters": int(len(rows)),
        f"{prefix}_raw_total_hot_voxels": int(sum(r["voxels"] for r in rows)),
        f"{prefix}_raw_largest_cluster_voxels": int(raw_largest["voxels"]),
        f"{prefix}_raw_largest_cluster_mm3": float(raw_largest["mm3"]),
        f"{prefix}_raw_largest_cluster_z_slices": int(raw_largest["z_slices"]),
        f"{prefix}_raw_largest_cluster_z_span_mm": float(raw_largest["z_span_mm"]),
        f"{prefix}_raw_largest_cluster_mean_score": float(raw_largest["mean_score"]),
        f"{prefix}_raw_largest_cluster_max_score": float(raw_largest["max_score"]),
        f"{prefix}_raw_largest_cluster_p90_score": float(raw_largest["p90_score"]),
        f"{prefix}_ignored_single_slice_clusters": int(len(ignored)),
        f"{prefix}_ignored_single_slice_voxels": int(sum(r["voxels"] for r in ignored)),
    }

    if not valid:
        out.update(
            {
                f"{prefix}_valid_num_clusters": 0,
                f"{prefix}_valid_total_hot_voxels": 0,
                f"{prefix}_valid_total_hot_mm3": 0.0,
                f"{prefix}_valid_hot_density": 0.0,
                f"{prefix}_valid_largest_cluster_voxels": 0,
                f"{prefix}_valid_largest_cluster_mm3": 0.0,
                f"{prefix}_valid_largest_cluster_z_slices": 0,
                f"{prefix}_valid_largest_cluster_z_span_mm": 0.0,
                f"{prefix}_valid_largest_cluster_mean_score": 0.0,
                f"{prefix}_valid_largest_cluster_max_score": 0.0,
                f"{prefix}_valid_largest_cluster_p90_score": 0.0,
                f"{prefix}_valid_largest_cluster_weighted_score": 0.0,
            }
        )
        return out

    largest = max(valid, key=lambda r: r["voxels"])
    valid_total_voxels = int(sum(r["voxels"] for r in valid))
    z_weight = float(largest["z_slices"] / max(1, int(min_z_slices)))
    out.update(
        {
            f"{prefix}_valid_num_clusters": int(len(valid)),
            f"{prefix}_valid_total_hot_voxels": valid_total_voxels,
            f"{prefix}_valid_total_hot_mm3": float(valid_total_voxels * voxel_volume),
            f"{prefix}_valid_hot_density": float(valid_total_voxels / max(1, organ_voxels)),
            f"{prefix}_valid_largest_cluster_voxels": int(largest["voxels"]),
            f"{prefix}_valid_largest_cluster_mm3": float(largest["mm3"]),
            f"{prefix}_valid_largest_cluster_z_slices": int(largest["z_slices"]),
            f"{prefix}_valid_largest_cluster_z_span_mm": float(largest["z_span_mm"]),
            f"{prefix}_valid_largest_cluster_mean_score": float(largest["mean_score"]),
            f"{prefix}_valid_largest_cluster_max_score": float(largest["max_score"]),
            f"{prefix}_valid_largest_cluster_p90_score": float(largest["p90_score"]),
            f"{prefix}_valid_largest_cluster_weighted_score": float(largest["mean_score"] * z_weight * math.log1p(largest["voxels"])),
        }
    )
    return out


def _fill_2d_holes_per_slice(mask: np.ndarray, organ_mask: np.ndarray | None = None) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for z in range(mask.shape[0]):
        sl = mask[z]
        if organ_mask is not None:
            sl = sl & (organ_mask[z] > 0)
        filled = ndimage.binary_fill_holes(sl)
        if organ_mask is not None:
            filled &= organ_mask[z] > 0
        out[z] = filled
    return out


def zone_continuity_features(
    prefix: str,
    heatmap: np.ndarray,
    threshold: float,
    spacing_zyx: tuple[float, float, float],
    organ_mask: np.ndarray | None = None,
) -> dict[str, float]:
    h = np.asarray(heatmap, dtype=np.float32)
    finite = np.isfinite(h)
    raw = finite & (h >= float(threshold))
    if organ_mask is not None:
        raw &= organ_mask > 0

    if not raw.any():
        keys = [
            "raw_voxels",
            "smoothed_voxels",
            "filled_gap_voxels",
            "largest_component_voxels",
            "largest_component_mm3",
            "largest_component_z_slices",
            "largest_component_mean_score",
            "largest_component_max_score",
            "largest_component_p90_score",
            "smooth_density",
        ]
        return {f"{prefix}_{k}": 0.0 for k in keys}

    closed = ndimage.binary_closing(raw, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
    if organ_mask is not None:
        closed &= organ_mask > 0
    filled = _fill_2d_holes_per_slice(closed, organ_mask=organ_mask)
    if organ_mask is not None:
        filled &= organ_mask > 0

    labels, n_labels = ndimage.label(filled, structure=np.ones((3, 3, 3), dtype=bool))
    voxel_volume = float(np.prod(spacing_zyx))
    organ_voxels = int(np.sum(organ_mask > 0)) if organ_mask is not None else int(np.prod(h.shape))
    if n_labels == 0:
        largest = filled
    else:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        largest = labels == int(np.argmax(counts))

    vals = h[largest & finite]
    z_slices = int(np.sum(np.any(largest, axis=(1, 2))))
    largest_voxels = int(np.sum(largest))
    return {
        f"{prefix}_raw_voxels": int(np.sum(raw)),
        f"{prefix}_smoothed_voxels": int(np.sum(filled)),
        f"{prefix}_filled_gap_voxels": int(max(0, np.sum(filled) - np.sum(raw))),
        f"{prefix}_largest_component_voxels": largest_voxels,
        f"{prefix}_largest_component_mm3": float(largest_voxels * voxel_volume),
        f"{prefix}_largest_component_z_slices": z_slices,
        f"{prefix}_largest_component_mean_score": float(np.nanmean(vals)) if vals.size else 0.0,
        f"{prefix}_largest_component_max_score": float(np.nanmax(vals)) if vals.size else 0.0,
        f"{prefix}_largest_component_p90_score": float(np.nanpercentile(vals, 90)) if vals.size else 0.0,
        f"{prefix}_smooth_density": float(np.sum(filled) / max(1, organ_voxels)),
    }


def _empty_dominant_cluster_features(prefix: str) -> dict[str, float]:
    keys = [
        "raw_num_clusters",
        "valid_num_clusters",
        "ignored_single_slice_clusters",
        "ignored_small_clusters",
        "best_cluster_score",
        "best_cluster_voxels",
        "best_cluster_mm3",
        "best_cluster_z_slices",
        "best_cluster_z_span_mm",
        "best_cluster_mean_cspca",
        "best_cluster_max_cspca",
        "best_cluster_p90_cspca",
        "best_cluster_top10_mean_cspca",
        "second_cluster_score",
        "second_cluster_voxels",
        "best_minus_second_cluster_score",
        "dominance_ratio",
        "top2_cluster_score_mean",
        "top3_cluster_score_mean",
        "top2_cluster_voxels_total",
        "top3_cluster_voxels_total",
        "top2_cluster_mm3_total",
        "top3_cluster_mm3_total",
        "cluster_score_concentration",
        "dominant_pz_weighted_score",
        "dominant_tz_weighted_score",
        "dominant_pz_ratio",
        "dominant_tz_ratio",
        "dominant_zone_margin",
        "dominant_zone_entropy",
        "dominant_zone_label_code",
        "dominant_zone_is_uncertain",
        "dominant_pz_mask_fraction",
        "dominant_tz_mask_fraction",
    ]
    return {f"{prefix}_{k}": 0.0 for k in keys}


def _top_fraction_mean(values: np.ndarray, frac: float = 0.10) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    n = max(1, int(math.ceil(float(frac) * x.size)))
    return float(np.mean(np.sort(x)[-n:]))


def _zone_smooth_mask(
    heatmap: np.ndarray | None,
    organ_mask: np.ndarray | None,
    threshold: float = 0.50,
) -> np.ndarray | None:
    if heatmap is None:
        return None
    h = np.asarray(heatmap, dtype=np.float32)
    mask = np.isfinite(h) & (h >= float(threshold))
    if organ_mask is not None:
        mask &= organ_mask > 0
    if not mask.any():
        return mask
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
    if organ_mask is not None:
        mask &= organ_mask > 0
    mask = _fill_2d_holes_per_slice(mask, organ_mask=organ_mask)
    if organ_mask is not None:
        mask &= organ_mask > 0
    return mask


def dominant_cspca_cluster_zone_features(
    prefix: str,
    cspca_heatmap: np.ndarray,
    pz_heatmap: np.ndarray | None,
    tz_heatmap: np.ndarray | None,
    threshold: float,
    spacing_zyx: tuple[float, float, float],
    organ_mask: np.ndarray | None = None,
    min_z_slices: int = 2,
    min_cluster_voxels: int = 1,
    zone_confidence: float = 0.65,
) -> dict[str, float]:
    h = np.asarray(cspca_heatmap, dtype=np.float32)
    if organ_mask is not None:
        organ_bool = organ_mask > 0
    else:
        organ_bool = np.ones_like(h, dtype=bool)

    finite = np.isfinite(h)
    hot = finite & (h >= float(threshold)) & organ_bool
    empty = _empty_dominant_cluster_features(prefix)
    if not hot.any():
        empty[f"{prefix}_dominant_zone_label_code"] = -1.0
        return empty

    labels, n_labels = ndimage.label(hot, structure=np.ones((3, 3, 3), dtype=bool))
    if n_labels == 0:
        empty[f"{prefix}_dominant_zone_label_code"] = -1.0
        return empty

    voxel_volume = float(np.prod(spacing_zyx))
    rows: list[dict[str, float]] = []
    ignored_single = 0
    ignored_small = 0

    for lab in range(1, n_labels + 1):
        mask = labels == lab
        if not mask.any():
            continue
        vals = h[mask]
        z_idx = np.where(np.any(mask, axis=(1, 2)))[0]
        z_slices = int(len(z_idx))
        z_span_vox = int(z_idx[-1] - z_idx[0] + 1) if z_slices else 0
        voxels = int(np.sum(mask))
        if z_slices < int(min_z_slices):
            ignored_single += 1
            continue
        if voxels < int(min_cluster_voxels):
            ignored_small += 1
            continue

        mean_score = float(np.nanmean(vals)) if vals.size else 0.0
        max_score = float(np.nanmax(vals)) if vals.size else 0.0
        p90_score = float(np.nanpercentile(vals, 90)) if vals.size else 0.0
        top10_score = _top_fraction_mean(vals, 0.10)
        volume_norm = min(1.0, math.log1p(voxels) / math.log1p(1000.0))
        z_norm = min(1.0, z_slices / 5.0)
        cluster_score = float(
            0.35 * mean_score
            + 0.25 * p90_score
            + 0.15 * top10_score
            + 0.10 * max_score
            + 0.10 * volume_norm
            + 0.05 * z_norm
        )
        rows.append(
            {
                "label": int(lab),
                "mask": mask,
                "score": cluster_score,
                "voxels": voxels,
                "mm3": float(voxels * voxel_volume),
                "z_slices": z_slices,
                "z_span_mm": float(z_span_vox * spacing_zyx[0]),
                "mean": mean_score,
                "max": max_score,
                "p90": p90_score,
                "top10": top10_score,
            }
        )

    out = _empty_dominant_cluster_features(prefix)
    out[f"{prefix}_raw_num_clusters"] = int(n_labels)
    out[f"{prefix}_ignored_single_slice_clusters"] = int(ignored_single)
    out[f"{prefix}_ignored_small_clusters"] = int(ignored_small)
    out[f"{prefix}_valid_num_clusters"] = int(len(rows))

    if not rows:
        out[f"{prefix}_dominant_zone_label_code"] = -1.0
        return out

    rows = sorted(rows, key=lambda r: r["score"], reverse=True)
    best = rows[0]
    second = rows[1] if len(rows) > 1 else None
    best_score = float(best["score"])
    second_score = float(second["score"]) if second is not None else 0.0
    gap = float(best_score - second_score)
    dominance_ratio = float(best_score / max(second_score, EPS)) if second is not None else float(best_score)

    out.update(
        {
            f"{prefix}_best_cluster_score": best_score,
            f"{prefix}_best_cluster_voxels": int(best["voxels"]),
            f"{prefix}_best_cluster_mm3": float(best["mm3"]),
            f"{prefix}_best_cluster_z_slices": int(best["z_slices"]),
            f"{prefix}_best_cluster_z_span_mm": float(best["z_span_mm"]),
            f"{prefix}_best_cluster_mean_cspca": float(best["mean"]),
            f"{prefix}_best_cluster_max_cspca": float(best["max"]),
            f"{prefix}_best_cluster_p90_cspca": float(best["p90"]),
            f"{prefix}_best_cluster_top10_mean_cspca": float(best["top10"]),
            f"{prefix}_second_cluster_score": second_score,
            f"{prefix}_second_cluster_voxels": int(second["voxels"]) if second is not None else 0,
            f"{prefix}_best_minus_second_cluster_score": gap,
            f"{prefix}_dominance_ratio": dominance_ratio,
        }
    )

    best_mask = best["mask"]
    c_vals = np.where(np.isfinite(h), h, 0.0)
    pz_vals = np.where(np.isfinite(pz_heatmap), pz_heatmap, 0.0) if pz_heatmap is not None else np.zeros_like(h, dtype=np.float32)
    tz_vals = np.where(np.isfinite(tz_heatmap), tz_heatmap, 0.0) if tz_heatmap is not None else np.zeros_like(h, dtype=np.float32)

    pz_weighted = float(np.sum(c_vals[best_mask] * pz_vals[best_mask]))
    tz_weighted = float(np.sum(c_vals[best_mask] * tz_vals[best_mask]))
    denom = pz_weighted + tz_weighted
    if denom <= EPS:
        pz_ratio = 0.0
        tz_ratio = 0.0
        label_code = -1.0
        is_uncertain = 1.0
        margin = 0.0
        entropy = 0.0
    else:
        pz_ratio = float(pz_weighted / denom)
        tz_ratio = float(tz_weighted / denom)
        margin = float(abs(pz_ratio - tz_ratio))
        entropy = float(-(pz_ratio * math.log(max(pz_ratio, EPS)) + tz_ratio * math.log(max(tz_ratio, EPS))))
        if pz_ratio >= float(zone_confidence):
            label_code = 0.0
            is_uncertain = 0.0
        elif tz_ratio >= float(zone_confidence):
            label_code = 1.0
            is_uncertain = 0.0
        else:
            label_code = 0.5
            is_uncertain = 1.0

    pz_mask = _zone_smooth_mask(pz_heatmap, organ_mask=organ_bool, threshold=0.50)
    tz_mask = _zone_smooth_mask(tz_heatmap, organ_mask=organ_bool, threshold=0.50)
    best_voxels = max(1, int(best["voxels"]))
    pz_mask_frac = float(np.sum(best_mask & pz_mask) / best_voxels) if pz_mask is not None else 0.0
    tz_mask_frac = float(np.sum(best_mask & tz_mask) / best_voxels) if tz_mask is not None else 0.0

    out.update(
        {
            f"{prefix}_dominant_pz_weighted_score": pz_weighted,
            f"{prefix}_dominant_tz_weighted_score": tz_weighted,
            f"{prefix}_dominant_pz_ratio": pz_ratio,
            f"{prefix}_dominant_tz_ratio": tz_ratio,
            f"{prefix}_dominant_zone_margin": margin,
            f"{prefix}_dominant_zone_entropy": entropy,
            f"{prefix}_dominant_zone_label_code": label_code,
            f"{prefix}_dominant_zone_is_uncertain": is_uncertain,
            f"{prefix}_dominant_pz_mask_fraction": pz_mask_frac,
            f"{prefix}_dominant_tz_mask_fraction": tz_mask_frac,
        }
    )
    return out


def _dilate_2d_per_slice(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    radius = max(0, int(radius))
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    for z in range(mask.shape[0]):
        out[z] = ndimage.binary_dilation(mask[z], structure=structure, iterations=1)
    return out


def _z_supported_hot_mask(
    hot_mask: np.ndarray,
    organ_mask: np.ndarray | None = None,
    xy_radius: int = 1,
) -> np.ndarray:
    hot = np.asarray(hot_mask, dtype=bool)
    if organ_mask is not None:
        hot = hot & (organ_mask > 0)
    if not hot.any():
        return hot.copy()


    dil = _dilate_2d_per_slice(hot, radius=xy_radius)
    support = np.zeros_like(hot, dtype=bool)
    if hot.shape[0] > 1:
        support[1:] |= dil[:-1]
        support[:-1] |= dil[1:]
    supported = hot & support
    if organ_mask is not None:
        supported &= organ_mask > 0
    return supported


def _xy_close_and_fill_per_slice(
    mask: np.ndarray,
    organ_mask: np.ndarray | None = None,
    iterations: int = 1,
) -> np.ndarray:
    iterations = max(0, int(iterations))
    structure = np.ones((3, 3), dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    for z in range(mask.shape[0]):
        sl = np.asarray(mask[z], dtype=bool)
        if organ_mask is not None:
            sl &= organ_mask[z] > 0
        if iterations > 0 and sl.any():
            sl = ndimage.binary_closing(sl, structure=structure, iterations=iterations)
        if sl.any():
            sl = ndimage.binary_fill_holes(sl)
        if organ_mask is not None:
            sl &= organ_mask[z] > 0
        out[z] = sl
    return out


def dominant_cspca_cluster_zone_features_v2(
    prefix: str,
    cspca_heatmap: np.ndarray,
    pz_heatmap: np.ndarray | None,
    tz_heatmap: np.ndarray | None,
    threshold: float,
    spacing_zyx: tuple[float, float, float],
    organ_mask: np.ndarray | None = None,
    min_cluster_voxels: int = 1,
    zone_confidence: float = 0.65,
    z_support_xy_radius: int = 1,
    xy_closing_iterations: int = 1,
) -> dict[str, float]:
    h = np.asarray(cspca_heatmap, dtype=np.float32)
    organ_bool = (organ_mask > 0) if organ_mask is not None else np.ones_like(h, dtype=bool)
    finite = np.isfinite(h)
    raw_hot = finite & (h >= float(threshold)) & organ_bool

    out = _empty_dominant_cluster_features(prefix)
    out.update({
        f"{prefix}_raw_hot_voxels": 0.0,
        f"{prefix}_z_supported_hot_voxels": 0.0,
        f"{prefix}_z_unsupported_hot_voxels": 0.0,
        f"{prefix}_xy_filled_cluster_voxels": 0.0,
        f"{prefix}_xy_filled_gap_voxels": 0.0,
        f"{prefix}_best_cluster_core_hot_voxels": 0.0,
        f"{prefix}_best_cluster_filled_gap_voxels": 0.0,
        f"{prefix}_best_cluster_core_mean_cspca": 0.0,
        f"{prefix}_best_cluster_core_p90_cspca": 0.0,
    })
    out[f"{prefix}_dominant_zone_label_code"] = -1.0

    if not raw_hot.any():
        return out

    z_supported = _z_supported_hot_mask(raw_hot, organ_mask=organ_bool, xy_radius=z_support_xy_radius)
    unsupported = raw_hot & ~z_supported
    filled = _xy_close_and_fill_per_slice(z_supported, organ_mask=organ_bool, iterations=xy_closing_iterations)

    out[f"{prefix}_raw_hot_voxels"] = float(np.sum(raw_hot))
    out[f"{prefix}_z_supported_hot_voxels"] = float(np.sum(z_supported))
    out[f"{prefix}_z_unsupported_hot_voxels"] = float(np.sum(unsupported))
    out[f"{prefix}_xy_filled_cluster_voxels"] = float(np.sum(filled))
    out[f"{prefix}_xy_filled_gap_voxels"] = float(max(0, np.sum(filled) - np.sum(z_supported)))

    if not filled.any():


        out[f"{prefix}_ignored_single_slice_clusters"] = 1.0
        return out

    labels, n_labels = ndimage.label(filled, structure=np.ones((3, 3, 3), dtype=bool))
    out[f"{prefix}_raw_num_clusters"] = int(n_labels)
    voxel_volume = float(np.prod(spacing_zyx))
    rows: list[dict[str, float]] = []
    ignored_small = 0

    pz_arr = np.asarray(pz_heatmap, dtype=np.float32) if pz_heatmap is not None else None
    tz_arr = np.asarray(tz_heatmap, dtype=np.float32) if tz_heatmap is not None else None

    for lab in range(1, n_labels + 1):
        comp = labels == lab
        voxels = int(np.sum(comp))
        if voxels < int(min_cluster_voxels):
            ignored_small += 1
            continue

        core = comp & z_supported
        gaps = comp & ~z_supported
        comp_vals = h[comp & finite]
        core_vals = h[core & finite]
        if comp_vals.size == 0:
            continue

        z_idx = np.where(np.any(comp, axis=(1, 2)))[0]
        z_slices = int(len(z_idx))
        z_span_vox = int(z_idx[-1] - z_idx[0] + 1) if z_slices else 0
        mean_score = float(np.nanmean(comp_vals))
        max_score = float(np.nanmax(comp_vals))
        p90_score = float(np.nanpercentile(comp_vals, 90))
        top10_score = _top_fraction_mean(comp_vals, 0.10)
        core_mean = float(np.nanmean(core_vals)) if core_vals.size else mean_score
        core_p90 = float(np.nanpercentile(core_vals, 90)) if core_vals.size else p90_score


        volume_norm = min(1.0, math.log1p(voxels) / math.log1p(1000.0))
        z_norm = min(1.0, z_slices / 5.0)
        core_frac = float(np.sum(core) / max(1, voxels))
        cluster_score = float(
            0.35 * core_mean
            + 0.25 * core_p90
            + 0.15 * top10_score
            + 0.10 * max_score
            + 0.08 * volume_norm
            + 0.04 * z_norm
            + 0.03 * core_frac
        )
        rows.append({
            "label": int(lab),
            "mask": comp,
            "core": core,
            "gaps": gaps,
            "score": cluster_score,
            "voxels": voxels,
            "mm3": float(voxels * voxel_volume),
            "z_slices": z_slices,
            "z_span_mm": float(z_span_vox * spacing_zyx[0]),
            "mean": mean_score,
            "max": max_score,
            "p90": p90_score,
            "top10": top10_score,
            "core_voxels": int(np.sum(core)),
            "gap_voxels": int(np.sum(gaps)),
            "core_mean": core_mean,
            "core_p90": core_p90,
        })

    out[f"{prefix}_ignored_small_clusters"] = int(ignored_small)
    out[f"{prefix}_valid_num_clusters"] = int(len(rows))
    if not rows:
        return out

    rows = sorted(rows, key=lambda r: r["score"], reverse=True)
    best = rows[0]
    second = rows[1] if len(rows) > 1 else None
    best_score = float(best["score"])
    second_score = float(second["score"]) if second is not None else 0.0
    gap = float(best_score - second_score)
    dominance_ratio = float(best_score / max(second_score, EPS)) if second is not None else float(best_score)
    top2 = rows[:2]
    top3 = rows[:3]
    top3_score_sum = float(sum(r["score"] for r in top3))
    cluster_score_concentration = float(best_score / max(top3_score_sum, EPS))

    out.update({
        f"{prefix}_best_cluster_score": best_score,
        f"{prefix}_best_cluster_voxels": int(best["voxels"]),
        f"{prefix}_best_cluster_mm3": float(best["mm3"]),
        f"{prefix}_best_cluster_z_slices": int(best["z_slices"]),
        f"{prefix}_best_cluster_z_span_mm": float(best["z_span_mm"]),
        f"{prefix}_best_cluster_mean_cspca": float(best["mean"]),
        f"{prefix}_best_cluster_max_cspca": float(best["max"]),
        f"{prefix}_best_cluster_p90_cspca": float(best["p90"]),
        f"{prefix}_best_cluster_top10_mean_cspca": float(best["top10"]),
        f"{prefix}_best_cluster_core_hot_voxels": int(best["core_voxels"]),
        f"{prefix}_best_cluster_filled_gap_voxels": int(best["gap_voxels"]),
        f"{prefix}_best_cluster_core_mean_cspca": float(best["core_mean"]),
        f"{prefix}_best_cluster_core_p90_cspca": float(best["core_p90"]),
        f"{prefix}_second_cluster_score": second_score,
        f"{prefix}_second_cluster_voxels": int(second["voxels"]) if second is not None else 0,
        f"{prefix}_best_minus_second_cluster_score": gap,
        f"{prefix}_dominance_ratio": dominance_ratio,
        f"{prefix}_top2_cluster_score_mean": float(np.mean([r["score"] for r in top2])),
        f"{prefix}_top3_cluster_score_mean": float(np.mean([r["score"] for r in top3])),
        f"{prefix}_top2_cluster_voxels_total": int(sum(r["voxels"] for r in top2)),
        f"{prefix}_top3_cluster_voxels_total": int(sum(r["voxels"] for r in top3)),
        f"{prefix}_top2_cluster_mm3_total": float(sum(r["mm3"] for r in top2)),
        f"{prefix}_top3_cluster_mm3_total": float(sum(r["mm3"] for r in top3)),
        f"{prefix}_cluster_score_concentration": cluster_score_concentration,
    })

    best_mask = best["mask"]
    c_vals = np.where(np.isfinite(h), h, 0.0)
    pz_vals = np.where(np.isfinite(pz_arr), pz_arr, 0.0) if pz_arr is not None else np.zeros_like(h, dtype=np.float32)
    tz_vals = np.where(np.isfinite(tz_arr), tz_arr, 0.0) if tz_arr is not None else np.zeros_like(h, dtype=np.float32)


    pz_weighted = float(np.sum(c_vals[best_mask] * pz_vals[best_mask]))
    tz_weighted = float(np.sum(c_vals[best_mask] * tz_vals[best_mask]))
    denom = pz_weighted + tz_weighted
    if denom <= EPS:
        pz_ratio = tz_ratio = margin = entropy = 0.0
        label_code = -1.0
        is_uncertain = 1.0
    else:
        pz_ratio = float(pz_weighted / denom)
        tz_ratio = float(tz_weighted / denom)
        margin = float(abs(pz_ratio - tz_ratio))
        entropy = float(-(pz_ratio * math.log(max(pz_ratio, EPS)) + tz_ratio * math.log(max(tz_ratio, EPS))))
        if pz_ratio >= float(zone_confidence):
            label_code = 0.0
            is_uncertain = 0.0
        elif tz_ratio >= float(zone_confidence):
            label_code = 1.0
            is_uncertain = 0.0
        else:
            label_code = 0.5
            is_uncertain = 1.0

    pz_mask = _zone_smooth_mask(pz_arr, organ_mask=organ_bool, threshold=0.50) if pz_arr is not None else None
    tz_mask = _zone_smooth_mask(tz_arr, organ_mask=organ_bool, threshold=0.50) if tz_arr is not None else None
    best_voxels = max(1, int(best["voxels"]))
    pz_mask_frac = float(np.sum(best_mask & pz_mask) / best_voxels) if pz_mask is not None else 0.0
    tz_mask_frac = float(np.sum(best_mask & tz_mask) / best_voxels) if tz_mask is not None else 0.0

    out.update({
        f"{prefix}_dominant_pz_weighted_score": pz_weighted,
        f"{prefix}_dominant_tz_weighted_score": tz_weighted,
        f"{prefix}_dominant_pz_ratio": pz_ratio,
        f"{prefix}_dominant_tz_ratio": tz_ratio,
        f"{prefix}_dominant_zone_margin": margin,
        f"{prefix}_dominant_zone_entropy": entropy,
        f"{prefix}_dominant_zone_label_code": label_code,
        f"{prefix}_dominant_zone_is_uncertain": is_uncertain,
        f"{prefix}_dominant_pz_mask_fraction": pz_mask_frac,
        f"{prefix}_dominant_tz_mask_fraction": tz_mask_frac,
    })
    return out


def threshold_tag(threshold: float) -> str:
    return ("thr_" + f"{float(threshold):.2f}".replace(".", "_")).replace("-", "m")


def cspca_cluster_overlay_masks(
    cspca_heatmap: np.ndarray,
    threshold: float,
    organ_mask: np.ndarray | None = None,
    min_cluster_voxels: int = 1,
    z_support_xy_radius: int = 1,
    xy_closing_iterations: int = 1,
) -> dict[str, np.ndarray]:
    h = np.asarray(cspca_heatmap, dtype=np.float32)
    organ_bool = (organ_mask > 0) if organ_mask is not None else np.ones_like(h, dtype=bool)
    finite = np.isfinite(h)
    raw_hot = finite & (h >= float(threshold)) & organ_bool
    z_supported = _z_supported_hot_mask(raw_hot, organ_mask=organ_bool, xy_radius=z_support_xy_radius) if raw_hot.any() else np.zeros_like(raw_hot, dtype=bool)
    z_unsupported = raw_hot & ~z_supported
    xy_filled = _xy_close_and_fill_per_slice(z_supported, organ_mask=organ_bool, iterations=xy_closing_iterations) if z_supported.any() else np.zeros_like(raw_hot, dtype=bool)

    dominant = np.zeros_like(raw_hot, dtype=bool)
    dominant_core = np.zeros_like(raw_hot, dtype=bool)
    labels = np.zeros_like(raw_hot, dtype=np.int32)
    n_labels = 0
    if xy_filled.any():
        labels, n_labels = ndimage.label(xy_filled, structure=np.ones((3, 3, 3), dtype=bool))

    best_label = 0
    best_score = -np.inf
    for lab in range(1, int(n_labels) + 1):
        comp = labels == lab
        voxels = int(np.sum(comp))
        if voxels < int(min_cluster_voxels):
            continue
        core = comp & z_supported
        comp_vals = h[comp & finite]
        core_vals = h[core & finite]
        if comp_vals.size == 0:
            continue
        z_idx = np.where(np.any(comp, axis=(1, 2)))[0]
        z_slices = int(len(z_idx))
        mean_score = float(np.nanmean(comp_vals))
        max_score = float(np.nanmax(comp_vals))
        p90_score = float(np.nanpercentile(comp_vals, 90))
        top10_score = _top_fraction_mean(comp_vals, 0.10)
        core_mean = float(np.nanmean(core_vals)) if core_vals.size else mean_score
        core_p90 = float(np.nanpercentile(core_vals, 90)) if core_vals.size else p90_score
        volume_norm = min(1.0, math.log1p(voxels) / math.log1p(1000.0))
        z_norm = min(1.0, z_slices / 5.0)
        core_frac = float(np.sum(core) / max(1, voxels))
        score = float(
            0.35 * core_mean
            + 0.25 * core_p90
            + 0.15 * top10_score
            + 0.10 * max_score
            + 0.08 * volume_norm
            + 0.04 * z_norm
            + 0.03 * core_frac
        )
        if score > best_score:
            best_score = score
            best_label = lab

    if best_label > 0:
        dominant = labels == best_label
        dominant_core = dominant & z_supported

    return {
        "raw_hot": raw_hot.astype(np.float32),
        "z_supported_hot": z_supported.astype(np.float32),
        "z_unsupported_hot": z_unsupported.astype(np.float32),
        "xy_filled": xy_filled.astype(np.float32),
        "dominant_cluster": dominant.astype(np.float32),
        "dominant_core": dominant_core.astype(np.float32),
    }

def summarize_scores(prefix: str, values: np.ndarray, threshold: float) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        keys = [
            "count",
            "mean",
            "std",
            "max",
            "median",
            "p90",
            "p95",
            "p99",
            "top5_mean",
            "top10_mean",
            "top20_mean",
            "top1pct_mean",
            "top5pct_mean",
            "count_ge_0_50",
            "count_ge_0_60",
            "count_ge_0_70",
            "count_ge_0_80",
            "frac_ge_0_50",
            "frac_ge_0_60",
            "frac_ge_0_70",
            "frac_ge_0_80",
            "count_ge_threshold",
            "frac_ge_threshold",
        ]
        return {f"{prefix}_{k}": np.nan for k in keys}
    sorted_x = np.sort(x)

    def top_mean(frac: float) -> float:
        n = max(1, int(math.ceil(frac * x.size)))
        return float(np.mean(sorted_x[-n:]))

    def top_n_mean(n: int) -> float:
        n = max(1, min(int(n), x.size))
        return float(np.mean(sorted_x[-n:]))

    return {
        f"{prefix}_count": int(x.size),
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_p90": float(np.percentile(x, 90)),
        f"{prefix}_p95": float(np.percentile(x, 95)),
        f"{prefix}_p99": float(np.percentile(x, 99)),
        f"{prefix}_top5_mean": top_n_mean(5),
        f"{prefix}_top10_mean": top_n_mean(10),
        f"{prefix}_top20_mean": top_n_mean(20),
        f"{prefix}_top1pct_mean": top_mean(0.01),
        f"{prefix}_top5pct_mean": top_mean(0.05),
        f"{prefix}_count_ge_0_50": int(np.sum(x >= 0.50)),
        f"{prefix}_count_ge_0_60": int(np.sum(x >= 0.60)),
        f"{prefix}_count_ge_0_70": int(np.sum(x >= 0.70)),
        f"{prefix}_count_ge_0_80": int(np.sum(x >= 0.80)),
        f"{prefix}_frac_ge_0_50": float(np.mean(x >= 0.50)),
        f"{prefix}_frac_ge_0_60": float(np.mean(x >= 0.60)),
        f"{prefix}_frac_ge_0_70": float(np.mean(x >= 0.70)),
        f"{prefix}_frac_ge_0_80": float(np.mean(x >= 0.80)),
        f"{prefix}_count_ge_threshold": int(np.sum(x >= threshold)),
        f"{prefix}_frac_ge_threshold": float(np.mean(x >= threshold)),
    }


import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from joblib import dump, load
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MHT_VERBOSE = os.environ.get("MHT_VERBOSE", "0").strip().lower() in {"1", "true", "yes", "y"}

def _vprint(*args, **kwargs):
    if MHT_VERBOSE:
        print(*args, **kwargs)

def _compact_progress(prefix: str, total: int, reused: int = 0, pending: int = 0, extra: str = "") -> None:
    parts = [prefix, f"total={total}"]
    if reused:
        parts.append(f"reused={reused}")
    if pending:
        parts.append(f"to_process={pending}")
    if extra:
        parts.append(extra)
    print(" | ".join(parts))

from all_config import (
    CLIN_COL,
    DATASET_XLSX,
    EARLY_FUSION_WORKBOOK_DIR,
    FEATURE_FILES,
    ID_COL,
    MAIN_ROOT,
    PATCH_DETECTION_DIR,
    PATCH_FEATURE_FILE_MAP,
    PATCH_FEATURE_SET_BY_FILE,
    PATCH_FEATURE_SETS,
    PATCH_HEATMAP_THRESHOLD,
    PATCH_CLUSTER_FEATURE_THRESHOLDS,
    PATCH_SAVE_CLUSTER_OVERLAYS,
    PATCH_CSPCA_MIN_CLUSTER_VOXELS,
    PATCH_CLUSTER_ZONE_CONFIDENCE,
    PATCH_CSPCA_Z_SUPPORT_XY_RADIUS_VOXELS,
    PATCH_CSPCA_XY_CLOSING_ITERATIONS,
    PATCH_HELPER_INPUT_FEATURE_DIR,
    PATCH_HELPER_MODEL_DIR,
    PATCH_MAX_PATCHES_PER_CLASS_PER_SCALE,
    PATCH_MODALITIES as CONFIG_PATCH_MODALITIES,
    PATCH_RAW_MAIN_DIR,
    PATCH_SCALES,
    PATCH_ONLY_WORKBOOK_DIR,
    PRETRAIN_ROOT,
    ZONE_COL,
)


SCORE_COLS = [
    "csPCa_likelihood",
    "non_csPCa_likelihood",
    "PZ_likelihood",
    "TZ_likelihood",
]
SELECTION_STAGE_COLS = [
    "sel_lesion_voxels",
    "sel_lesion_candidates_sampled",
    "sel_lesion_inside_image",
    "sel_lesion_overlap_ge_threshold",
    "sel_lesion_after_top_cap",
    "sel_negative_grid_candidates",
    "sel_negative_no_lesion_pz_pool",
    "sel_negative_no_lesion_tz_pool",
    "sel_negative_target_total",
    "sel_negative_selected_pz",
    "sel_negative_selected_tz",
    "sel_final_positive",
    "sel_final_negative",
]
PATCH_MODALITIES = tuple(CONFIG_PATCH_MODALITIES)
PATCH_FEATURE_SETS = tuple(PATCH_FEATURE_SETS)
PATCH_FEATURE_FILE_MAP = dict(PATCH_FEATURE_FILE_MAP)
PATCH_FEATURE_SET_BY_FILE = dict(PATCH_FEATURE_SET_BY_FILE)
PATCH_INFERENCE_RAW_FEATURE_DIR = PATCH_RAW_MAIN_DIR
PATCH_INFERENCE_RAW_PRED_DIR = PATCH_DETECTION_DIR
PATCH_HELPER_HEATMAP_DIR = PATCH_HELPER_MODEL_DIR.parent / ".heatmaps"

from all_config import (
    PATCH_TRAIN_WORKERS as PATCH_WORKERS,
    PATCH_APPLY_WORKERS,
    PATCH_MODEL_TRAIN_N_JOBS,
)
import all_config as cfg
PATCH_MODEL_PREDICT_N_JOBS = 1
PATCH_INITIAL_EXPORT_CASE_LIMIT = 1
DEFAULT_PATCH_STAGE = "all"
_MODEL_CACHE: dict[tuple[str, str, str], dict] = {}


def _case_key(case_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(case_id).strip().lower())


def validate_external_main_case_separation() -> None:
    main_ids = [str(x).strip() for x in list_case_ids(MAIN_ROOT)]
    ext_ids = [str(x).strip() for x in list_case_ids(PRETRAIN_ROOT)]

    exact_overlap = sorted(set(main_ids) & set(ext_ids))
    main_norm = {_case_key(x): x for x in main_ids}
    norm_overlap = []
    for ext in ext_ids:
        key = _case_key(ext)
        if key and key in main_norm:
            norm_overlap.append((main_norm[key], ext))
    if exact_overlap or norm_overlap:
        examples = []
        examples.extend([f"exact:{x}" for x in exact_overlap[:10]])
        examples.extend([f"normalized:main={m}, external={e}" for m, e in norm_overlap[:10]])
        raise RuntimeError(
            "Patient/case leakage risk: PRETRAIN_ROOT external helper set overlaps with "
            "MAIN_ROOT main dataset. Remove PROSTATEx-overlapping cases before training/applying helpers. "
            "Examples: " + "; ".join(examples)
        )
    print(f"[LEAK CHECK] OK: {len(ext_ids)} external cases and {len(main_ids)} main cases have no ID overlap")




def _ensure_dirs() -> None:
    for path in [
        PATCH_HELPER_INPUT_FEATURE_DIR,
        PATCH_HELPER_MODEL_DIR,
        PATCH_INFERENCE_RAW_FEATURE_DIR,
        PATCH_INFERENCE_RAW_PRED_DIR,
        PATCH_ONLY_WORKBOOK_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def _safe_feature_set_tag(feature_set: str) -> str:
    return str(feature_set).strip()


def _task_model_path(scale_name: str, task: str, feature_set: str) -> Path:
    return PATCH_HELPER_MODEL_DIR / f"{scale_name}_{task}_{_safe_feature_set_tag(feature_set)}.joblib"


def _set_model_n_jobs(package: dict, n_jobs: int) -> None:
    try:
        package["model"][-1].set_params(n_jobs=n_jobs)
    except Exception:
        pass


def _load_task_model(scale_name: str, task: str, feature_set: str) -> dict | None:
    key = (str(scale_name), str(task), str(feature_set))
    if key not in _MODEL_CACHE:
        path = _task_model_path(scale_name, task, feature_set)
        if not path.exists():
            return None
        # mmap_mode="r": each ProcessPoolExecutor worker keeps its own _MODEL_CACHE
        # (54 CalibratedClassifierCV(cv=3) ExtraTrees models, ~9.2GB on disk uncompressed).
        # A normal load() deserializes a private copy per worker (measured ~2x the
        # on-disk size in RSS); mmap_mode maps the same file read-only, so worker
        # processes share the underlying pages via the OS page cache instead of each
        # duplicating the full model set. Inference-only use (predict/predict_proba
        # never writes to the arrays), and dump() saves uncompressed, so this is safe.
        package = load(path, mmap_mode="r")
        _set_model_n_jobs(package, PATCH_MODEL_PREDICT_N_JOBS)
        _MODEL_CACHE[key] = package
    return _MODEL_CACHE[key]


def _balanced_sample(
    df: pd.DataFrame,
    label_col: str,
    max_per_class: int,
    max_majority_to_minority_ratio: float,
    random_state: int,
) -> pd.DataFrame:
    d = df.dropna(subset=[label_col]).copy()
    if d.empty:
        return d
    d[label_col] = d[label_col].astype(int)
    counts = d[label_col].value_counts()
    minority_count = int(counts.min())
    ratio_cap = max(1, int(np.floor(float(max_majority_to_minority_ratio) * minority_count)))
    parts = []
    for label, group in d.groupby(label_col):
        n = min(len(group), max_per_class, ratio_cap)
        parts.append(group.sample(n=n, random_state=random_state + int(label)))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def _label_count_json(df: pd.DataFrame, label_col: str) -> str:
    if df.empty or label_col not in df.columns:
        return "{}"
    vals = df.dropna(subset=[label_col])[label_col].astype(int)
    counts = vals.value_counts().sort_index().to_dict()
    return json.dumps({str(k): int(v) for k, v in counts.items()}, sort_keys=True)


def _selection_stage_summary(df: pd.DataFrame) -> dict[str, int]:
    out = {col: 0 for col in SELECTION_STAGE_COLS}
    if df.empty or ID_COL not in df.columns:
        return out
    present = [col for col in SELECTION_STAGE_COLS if col in df.columns]
    if not present:
        return out
    group_cols = [ID_COL]
    if "scale" in df.columns:
        group_cols.append("scale")
    per_case = df[group_cols + present].drop_duplicates(group_cols, keep="first")
    for col in present:
        out[col] = int(pd.to_numeric(per_case[col], errors="coerce").fillna(0).sum())
    return out


def _prefixed_counts(prefix: str, df: pd.DataFrame, label_col: str) -> dict[str, int]:
    if df.empty or label_col not in df.columns:
        return {}
    vals = df.dropna(subset=[label_col])[label_col].astype(int)
    counts = vals.value_counts().sort_index().to_dict()
    return {f"{prefix}_class_{int(k)}": int(v) for k, v in counts.items()}


def _cap_zone_task_by_overlap(df: pd.DataFrame, max_per_zone: int) -> pd.DataFrame:
    if df.empty or "zone_patch_label" not in df.columns or "lesion_overlap" not in df.columns:
        return df
    work = df.copy()
    work["__lesion_overlap_sort"] = pd.to_numeric(work["lesion_overlap"], errors="coerce").fillna(-1.0)
    parts = []
    group_cols = [ID_COL, "zone_patch_label"] if ID_COL in work.columns else ["zone_patch_label"]
    for _, group in work.groupby(group_cols, dropna=True):
        parts.append(
            group.sort_values("__lesion_overlap_sort", ascending=False, kind="stable").head(int(max_per_zone))
        )
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True).drop(columns=["__lesion_overlap_sort"], errors="ignore")


def _sampling_meta() -> dict:
    return {
        "max_per_class": int(PATCH_MAX_PATCHES_PER_CLASS_PER_SCALE),
        "max_majority_to_minority_ratio": float(PATCH_MAX_MAJORITY_TO_MINORITY_RATIO),
        "fixed_positive_patches_per_case": int(PATCH_FIXED_POSITIVE_PATCHES_PER_CASE),
        "min_positive_lesion_overlap": float(PATCH_MIN_POSITIVE_LESION_OVERLAP),
        "patch_helper_classifier": "extra_trees_sigmoid_calibrated_cv3_when_possible",
        "concat_includes_mixed_t2w_adc_features": True,
    }


def _model_sampling_matches(path: Path) -> bool:
    try:
        package = load(path)
    except Exception:
        return False
    return package.get("sampling") == _sampling_meta()


def _modality_feature_cols(cols: list[str], modality: str) -> list[str]:
    prefixes = (f"{modality}_core_", f"{modality}_ring_")
    return [c for c in cols if str(c).startswith(prefixes)]


def _neutral_modality_table(df: pd.DataFrame, modality: str, include_meta: bool = True) -> pd.DataFrame:
    prefix = f"{modality}_"
    cols = _modality_feature_cols(list(df.columns), modality)
    rename = {c: c[len(prefix):] for c in cols}
    meta_cols = [c for c in df.columns if c in _patch_meta_cols()]
    out = df[meta_cols + list(rename.keys())].rename(columns=rename) if include_meta else df[list(rename.keys())].rename(columns=rename)
    return out.copy()


def _mixed_patch_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    excluded = set(_patch_meta_cols())
    modality_cols = set(_modality_feature_cols(list(df.columns), "t2w")) | set(_modality_feature_cols(list(df.columns), "adc"))
    cols = [
        c
        for c in df.columns
        if c not in excluded
        and c not in modality_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df[cols].copy() if cols else pd.DataFrame(index=df.index)


def _patch_meta_cols() -> list[str]:
    preferred = [
        ID_COL,
        "scale",
        "center_z",
        "center_y",
        "center_x",
        "patch_z",
        "patch_y",
        "patch_x",
        "organ_overlap",
        "patch_source",
        "lesion_overlap",
        "lesion_coverage",
        "center_in_lesion",
        "pz_overlap",
        "tz_overlap",
        "cs_patch_label",
        "zone_patch_label",
        *SELECTION_STAGE_COLS,
    ]
    return preferred


def _label_cols_present(df: pd.DataFrame) -> list[str]:
    return [c for c in ["lesion_overlap", "lesion_coverage", "center_in_lesion", "pz_overlap", "tz_overlap", "cs_patch_label", "zone_patch_label"] if c in df.columns]


def _selection_cols_present(df: pd.DataFrame) -> list[str]:
    return [c for c in SELECTION_STAGE_COLS if c in df.columns]


def _base_patch_meta(df: pd.DataFrame, include_labels: bool) -> pd.DataFrame:
    cols = [ID_COL, "scale", "center_z", "center_y", "center_x", "patch_z", "patch_y", "patch_x", "organ_overlap"]
    if include_labels:
        cols += _label_cols_present(df)
        cols += _selection_cols_present(df)
    return df[[c for c in cols if c in df.columns]].copy()


def _feature_set_table(raw: pd.DataFrame, feature_set: str, include_labels: bool = True) -> pd.DataFrame:
    meta = _base_patch_meta(raw, include_labels=include_labels)
    t2w = _neutral_modality_table(raw, "t2w", include_meta=False)
    adc = _neutral_modality_table(raw, "adc", include_meta=False)
    common = sorted(set(t2w.columns) & set(adc.columns))
    fs = str(feature_set)

    if fs == "t2w":
        feat = t2w.add_suffix("_t2w")
    elif fs == "adc":
        feat = adc.add_suffix("_adc")
    elif fs == "concat":
        mixed = _mixed_patch_feature_table(raw).add_suffix("_mixed")
        feat = pd.concat([t2w.add_suffix("_t2w"), adc.add_suffix("_adc"), mixed], axis=1)
    elif fs in {"hada", "diff"}:
        if not common:
            raise RuntimeError(f"No common T2W/ADC patch features for {fs}")
        a = t2w[common].to_numpy(dtype=float)
        b = adc[common].to_numpy(dtype=float)
        values = a * b if fs == "hada" else (a - b)
        feat = pd.DataFrame(values, columns=[f"{name}_{fs}" for name in common], index=raw.index)
    elif fs.startswith("fusion(") and fs.endswith(")"):
        inner = fs[len("fusion(") : -1]
        parts = []
        if "c" in inner:
            parts.append(_feature_set_table(raw, "concat", include_labels=False).drop(columns=[c for c in _patch_meta_cols() if c in raw.columns], errors="ignore"))
        if "d" in inner:
            parts.append(_feature_set_table(raw, "diff", include_labels=False).drop(columns=[c for c in _patch_meta_cols() if c in raw.columns], errors="ignore"))
        if "h" in inner:
            parts.append(_feature_set_table(raw, "hada", include_labels=False).drop(columns=[c for c in _patch_meta_cols() if c in raw.columns], errors="ignore"))
        feat = pd.concat(parts, axis=1)
        feat = feat.loc[:, ~feat.columns.duplicated()]
    else:
        raise RuntimeError(f"Unknown patch feature set: {feature_set}")

    return pd.concat([meta.reset_index(drop=True), feat.reset_index(drop=True)], axis=1)


def _feature_columns_for_set(df: pd.DataFrame) -> list[str]:
    excluded = set(_patch_meta_cols())
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def _feature_file_for_set(feature_set: str) -> str:
    return PATCH_FEATURE_FILE_MAP[str(feature_set)]


def _train_extra_trees(df: pd.DataFrame, label_col: str, cols: list[str], random_state: int):
    y = df[label_col].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise RuntimeError(f"Need both classes for {label_col}, got {sorted(np.unique(y).tolist())}")
    x = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    class_counts = pd.Series(y).value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    base_classifier = ExtraTreesClassifier(
        n_estimators=600,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=PATCH_MODEL_TRAIN_N_JOBS,
    )
    if min_class_count >= 3:
        try:
            classifier = CalibratedClassifierCV(
                estimator=base_classifier,
                method="sigmoid",
                cv=min(3, min_class_count),
            )
        except TypeError:
            classifier = CalibratedClassifierCV(
                base_estimator=base_classifier,
                method="sigmoid",
                cv=min(3, min_class_count),
            )
    else:
        classifier = base_classifier
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        classifier,
    )
    model.fit(x, y)
    return model


def _write_training_feature_set_table(feature_set: str, df: pd.DataFrame) -> None:
    out_path = PATCH_HELPER_INPUT_FEATURE_DIR / _feature_file_for_set(feature_set)
    preview = _first_n_case_rows(df, PATCH_INITIAL_EXPORT_CASE_LIMIT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    preview.to_csv(tmp, index=False)
    os.replace(tmp, out_path)
    print(f"[PATCH TRAIN][{feature_set}] saved first-{PATCH_INITIAL_EXPORT_CASE_LIMIT}-case view {out_path} rows={len(preview)}")


def _first_n_case_rows(df: pd.DataFrame, n_cases: int) -> pd.DataFrame:
    if df.empty or ID_COL not in df.columns:
        return df.copy()
    ids = df[ID_COL].astype(str).str.strip()
    keep_ids = list(dict.fromkeys(ids.tolist()))[: int(n_cases)]
    return df[ids.isin(keep_ids)].copy()


def _model_signature() -> list[dict]:
    rows = []
    for scale in PATCH_SCALES:
        scale_name = normalize_scale(scale).name
        for feature_set in PATCH_FEATURE_SETS:
            for task in ("cspca", "zone"):
                path = _task_model_path(scale_name, task, feature_set)
                if path.exists():
                    stat = path.stat()
                    rows.append(
                        {
                            "scale": scale_name,
                            "feature_set": feature_set,
                            "task": task,
                            "path": str(path),
                            "size": int(stat.st_size),
                            "mtime_ns": int(stat.st_mtime_ns),
                        }
                    )
                else:
                    rows.append({"scale": scale_name, "feature_set": feature_set, "task": task, "path": str(path), "missing": True})
    return rows


def _all_patch_models_exist() -> bool:
    for scale in PATCH_SCALES:
        scale_name = normalize_scale(scale).name
        for feature_set in PATCH_FEATURE_SETS:
            for task in ("cspca", "zone"):
                if not _task_model_path(scale_name, task, feature_set).exists():
                    return False
    return True


def _all_patch_models_match_sampling() -> bool:
    if not _all_patch_models_exist():
        return False
    for scale in PATCH_SCALES:
        scale_name = normalize_scale(scale).name
        for feature_set in PATCH_FEATURE_SETS:
            for task in ("cspca", "zone"):
                if not _model_sampling_matches(_task_model_path(scale_name, task, feature_set)):
                    return False
    return True


def _expected_patch_workbook_paths() -> list[Path]:
    return [PATCH_ONLY_WORKBOOK_DIR / _feature_file_for_set(feature_set) for feature_set in PATCH_FEATURE_SETS]


def _all_final_patch_workbooks_exist() -> bool:
    paths = _expected_patch_workbook_paths()
    if not paths:
        return False
    expected_ids = {str(c).strip() for c in list_case_ids(MAIN_ROOT)}
    if not expected_ids:
        return False
    return all(expected_ids.issubset(cfg.patch_workbook_done_ids(p)) for p in paths)


def _expected_early_fusion_workbook_paths() -> list[Path]:
    return [EARLY_FUSION_WORKBOOK_DIR / name for name in FEATURE_FILES]


def _all_early_fusion_workbooks_exist() -> bool:
    paths = _expected_early_fusion_workbook_paths()
    return bool(paths) and all(path.exists() for path in paths)


def _extract_external_case_task(task: tuple[dict, str]) -> tuple[str, "pd.DataFrame | None", str]:
    scale, case_id = task
    sc = normalize_scale(scale)
    case = load_case_arrays(PRETRAIN_ROOT, case_id, require_lesion=True, require_zone=True)
    if case is None:
        return case_id, None, "skip_missing_file_or_shape_mismatch"
    df = build_patch_dataframe_for_case(case, sc, include_labels=True)
    if df.empty:
        return case_id, None, "skip_no_intra_organ_patches"
    return case_id, df, "ok"


def _extract_external_scale_table(scale: dict) -> pd.DataFrame:
    sc = normalize_scale(scale)
    case_ids = list_case_ids(PRETRAIN_ROOT)
    tasks = [(scale, case_id) for case_id in case_ids]
    frames: list[pd.DataFrame] = []

    def _run_seq() -> None:
        for task in tasks:
            case_id, df, status = _extract_external_case_task(task)
            if df is not None:
                frames.append(df)
            _vprint(f"[PATCH TRAIN][{sc.name}] {case_id}: {status}")

    if PATCH_WORKERS <= 1:
        _run_seq()
    else:
        try:
            with ProcessPoolExecutor(max_workers=PATCH_WORKERS) as ex:
                futures = [ex.submit(_extract_external_case_task, task) for task in tasks]
                completed = 0
                total = len(futures)
                for f in as_completed(futures):
                    completed += 1
                    cfg.log_progress("PATCH_TRAIN", completed, total)
                    case_id, df, status = f.result()
                    if df is not None:
                        frames.append(df)
                    _vprint(f"[PATCH TRAIN][{sc.name}] {case_id}: {status}")
        except BrokenProcessPool:
            print(f"[PATCH TRAIN][{sc.name}] WARNING: a worker died; retrying this scale sequentially.")
            frames.clear()
            _run_seq()

    if not frames:
        raise RuntimeError(f"No external patches produced for scale {sc.name}")
    out = pd.concat(frames, ignore_index=True)
    print(f"[PATCH TRAIN][{sc.name}] built aggregate table rows={len(out)}")
    return out


def train_patch_helpers(
    allow_skipped_models: bool = False,
) -> pd.DataFrame:
    _ensure_dirs()
    validate_external_main_case_separation()
    if _all_patch_models_match_sampling():
        print("[PATCH TRAIN] all expected .joblib models already exist; skipping training")
        rows = []
        for item in _model_signature():
            rows.append({
                "scale": item.get("scale"),
                "feature_set": item.get("feature_set"),
                "task": item.get("task"),
                "model_path": item.get("path"),
                "status": "existing_model_reused",
            })
        return pd.DataFrame(rows)
    report_rows = []
    per_set_views: dict[str, list[pd.DataFrame]] = {fs: [] for fs in PATCH_FEATURE_SETS}
    for scale in PATCH_SCALES:
        sc = normalize_scale(scale)
        table = _extract_external_scale_table(scale)
        view_preview = _first_n_case_rows(table, PATCH_INITIAL_EXPORT_CASE_LIMIT)
        task_specs = [
            ("cspca", "cs_patch_label", {0: "non-csPCa/reference", 1: "csPCa"}),
            ("zone", "zone_patch_label", {0: "PZ", 1: "TZ"}),
        ]
        for feature_set in PATCH_FEATURE_SETS:
            feature_table = _feature_set_table(table, feature_set, include_labels=True)
            view_ft = _feature_set_table(view_preview, feature_set, include_labels=True)
            if "scale" in view_ft.columns:
                view_ft = view_ft.drop(columns=["scale"])
            view_ft.insert(0, "scale", sc.name)
            per_set_views[feature_set].append(view_ft)
            feature_cols = [c for c in _feature_columns_for_set(feature_table) if feature_table[c].notna().any()]
            if not feature_cols:
                raise RuntimeError(f"No {feature_set} patch feature columns for scale {sc.name}")
            for task, label_col, label_names in task_specs:
                task_table = feature_table
                zone_before_cap = pd.DataFrame()
                if task == "zone":
                    zone_before_cap = feature_table[
                        pd.to_numeric(feature_table.get("cs_patch_label"), errors="coerce") == 1
                    ].copy()
                    task_table = zone_before_cap
                    task_table = _cap_zone_task_by_overlap(
                        task_table,
                        max_per_zone=PATCH_FIXED_POSITIVE_PATCHES_PER_CASE,
                    )
                out_path = _task_model_path(sc.name, task, feature_set)
                label_values = task_table.dropna(subset=[label_col])[label_col].astype(int).to_numpy()
                unique_labels = sorted(np.unique(label_values).tolist()) if label_values.size else []
                eligible_class_counts = _label_count_json(task_table, label_col)
                report_stage_counts = _selection_stage_summary(feature_table)
                report_stage_counts.update(_prefixed_counts("eligible", task_table, label_col))
                if task == "zone":
                    report_stage_counts.update(_prefixed_counts("zone_before_cap", zone_before_cap, label_col))
                    report_stage_counts["zone_before_cap_total"] = int(len(zone_before_cap))
                    report_stage_counts["zone_after_cap_total"] = int(len(task_table))
                if len(unique_labels) < 2:
                    if out_path.exists():
                        out_path.unlink()
                        print(f"[PATCH TRAIN][{sc.name}][{feature_set}] removed stale {task} model because current labels have one class")
                    message = (
                        f"[PATCH TRAIN][{sc.name}][{feature_set}] cannot train {task}: need both classes, "
                        f"got {unique_labels}. Increase external cases or relax patch-label rules."
                    )
                    if not allow_skipped_models:
                        raise RuntimeError(message)
                    report_rows.append(
                    {
                        "scale": sc.name,
                            "feature_set": feature_set,
                            "task": task,
                            "n_total_patches": int(len(task_table)),
                            "n_train_patches": int(label_values.size),
                            "eligible_class_counts": eligible_class_counts,
                            "train_class_counts": eligible_class_counts,
                            "class_counts": eligible_class_counts,
                            **report_stage_counts,
                            "max_majority_to_minority_ratio": float(PATCH_MAX_MAJORITY_TO_MINORITY_RATIO),
                            "n_features": int(len(feature_cols)),
                            "model_path": str(out_path),
                            "status": f"skipped_need_two_classes_got_{unique_labels}",
                        }
                    )
                    _vprint(message)
                    continue

                if out_path.exists() and _model_sampling_matches(out_path):
                    train_preview = _balanced_sample(
                        task_table,
                        label_col=label_col,
                        max_per_class=PATCH_MAX_PATCHES_PER_CLASS_PER_SCALE,
                        max_majority_to_minority_ratio=PATCH_MAX_MAJORITY_TO_MINORITY_RATIO,
                        random_state=PATCH_HELPER_RANDOM_STATE,
                    )
                    train_class_counts = _label_count_json(train_preview, label_col)
                    train_count_cols = _prefixed_counts("train", train_preview, label_col)
                    report_rows.append(
                        {
                            "scale": sc.name,
                            "feature_set": feature_set,
                            "task": task,
                            "n_total_patches": int(len(task_table)),
                            "n_train_patches": int(len(train_preview)),
                            "eligible_class_counts": eligible_class_counts,
                            "train_class_counts": train_class_counts,
                            "class_counts": train_class_counts,
                            **report_stage_counts,
                            **train_count_cols,
                            "max_majority_to_minority_ratio": float(PATCH_MAX_MAJORITY_TO_MINORITY_RATIO),
                            "n_features": int(len(feature_cols)),
                            "model_path": str(out_path),
                            "status": "reused",
                        }
                    )
                    _vprint(f"[PATCH TRAIN][{sc.name}][{feature_set}] reuse model {task}: {out_path}")
                    continue

                train_df = _balanced_sample(
                    task_table,
                    label_col=label_col,
                    max_per_class=PATCH_MAX_PATCHES_PER_CLASS_PER_SCALE,
                    max_majority_to_minority_ratio=PATCH_MAX_MAJORITY_TO_MINORITY_RATIO,
                    random_state=PATCH_HELPER_RANDOM_STATE,
                )
                train_class_counts = _label_count_json(train_df, label_col)
                train_count_cols = _prefixed_counts("train", train_df, label_col)
                model = _train_extra_trees(train_df, label_col, feature_cols, PATCH_HELPER_RANDOM_STATE)
                package = {
                    "task": task,
                    "feature_set": feature_set,
                    "scale": sc.__dict__,
                    "feature_cols": feature_cols,
                    "label_col": label_col,
                    "label_names": label_names,
                    "sampling": _sampling_meta(),
                    "model": model,
                }
                dump(package, out_path)
                report_rows.append(
                    {
                        "scale": sc.name,
                        "feature_set": feature_set,
                        "task": task,
                        "n_total_patches": int(len(task_table)),
                        "n_train_patches": int(len(train_df)),
                        "eligible_class_counts": eligible_class_counts,
                        "train_class_counts": train_class_counts,
                        "class_counts": train_class_counts,
                        **report_stage_counts,
                        **train_count_cols,
                        "max_majority_to_minority_ratio": float(PATCH_MAX_MAJORITY_TO_MINORITY_RATIO),
                        "n_features": int(len(feature_cols)),
                        "model_path": str(out_path),
                        "status": "trained",
                    }
                )
                print(f"[PATCH TRAIN][{sc.name}][{feature_set}] trained {task} rows={len(train_df)} features={len(feature_cols)}")

    for feature_set, parts in per_set_views.items():
        combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        _write_training_feature_set_table(feature_set, combined)

    return pd.DataFrame(report_rows)


def _model_proba(package: dict, df: pd.DataFrame, class_label: int) -> np.ndarray:
    if df.empty:
        return np.array([], dtype=float)
    cols = package["feature_cols"]
    x = df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    model = package["model"]
    proba = model.predict_proba(x)
    classes = list(model[-1].classes_)
    if class_label not in classes:
        return np.full(len(df), np.nan, dtype=float)
    return proba[:, classes.index(class_label)].astype(float)


def _score_patch_table(df: pd.DataFrame, scale_name: str, feature_set: str) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        for col in SCORE_COLS:
            out[col] = []
        return out

    cspca_path = _task_model_path(scale_name, "cspca", feature_set)
    zone_path = _task_model_path(scale_name, "zone", feature_set)

    if cspca_path.exists():
        cspca_model = _load_task_model(scale_name, "cspca", feature_set)
        out["csPCa_likelihood"] = _model_proba(cspca_model, out, 1)
        out["non_csPCa_likelihood"] = _model_proba(cspca_model, out, 0)
    else:
        out["csPCa_likelihood"] = np.nan
        out["non_csPCa_likelihood"] = np.nan

    if zone_path.exists():
        zone_model = _load_task_model(scale_name, "zone", feature_set)
        out["PZ_likelihood"] = _model_proba(zone_model, out, 0)
        out["TZ_likelihood"] = _model_proba(zone_model, out, 1)
    else:
        out["PZ_likelihood"] = np.nan
        out["TZ_likelihood"] = np.nan
    return out


def _csv_feature_file_name(feature_set: str) -> str:
    return Path(_feature_file_for_set(feature_set)).with_suffix(".csv").name


def _patch_prediction_path(feature_set: str) -> Path:
    return PATCH_INFERENCE_RAW_PRED_DIR / _csv_feature_file_name(feature_set)


def _raw_feature_path(feature_set: str) -> Path:
    return PATCH_INFERENCE_RAW_FEATURE_DIR / _csv_feature_file_name(feature_set)


def _write_scale_csv_table(path: Path, df: pd.DataFrame, value_cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = [c for c in value_cols if c in df.columns]
    out = df.reindex(columns=keep).copy() if not df.empty else pd.DataFrame(columns=value_cols)
    sort_cols = [c for c in ["scale", ID_COL, "center_z", "center_y", "center_x"] if c in out.columns]
    if sort_cols and not out.empty:
        out = out.sort_values(sort_cols, kind="stable")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    out.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def _write_patch_prediction_table(feature_set: str, df: pd.DataFrame) -> None:
    path = _patch_prediction_path(feature_set)
    pred_cols = [
        ID_COL,
        "scale",
        "center_z",
        "center_y",
        "center_x",
        "patch_z",
        "patch_y",
        "patch_x",
        "csPCa_likelihood",
        "PZ_likelihood",
    ]
    _write_scale_csv_table(path, df, pred_cols)
    print(f"[PATCH APPLY][{feature_set}] saved first-{PATCH_INITIAL_EXPORT_CASE_LIMIT}-case detection view {path} rows={len(df)}")


def _write_inference_raw_feature_table(feature_set: str, df: pd.DataFrame) -> None:
    path = _raw_feature_path(feature_set)
    meta_cols = [ID_COL, "scale", "center_z", "center_y", "center_x", "patch_z", "patch_y", "patch_x"]
    feat_cols = [c for c in df.columns if c not in set(_patch_meta_cols()) and pd.api.types.is_numeric_dtype(df[c])]
    _write_scale_csv_table(path, df, [*meta_cols, *feat_cols])
    print(f"[PATCH APPLY][{feature_set}] saved first-{PATCH_INITIAL_EXPORT_CASE_LIMIT}-case raw-feature view {path} rows={len(df)}")


def _smooth_probability_heatmap(heatmap: np.ndarray, organ_mask: np.ndarray, sigma: tuple[float, float, float] = (0.5, 1.0, 1.0)) -> np.ndarray:
    h = np.asarray(heatmap, dtype=np.float32)
    organ = np.asarray(organ_mask, dtype=bool)
    valid = np.isfinite(h) & organ
    if not valid.any():
        return np.where(organ, 0.0, 0.0).astype(np.float32)
    values = np.where(valid, h, 0.0).astype(np.float32)
    weights = valid.astype(np.float32)
    num = ndimage.gaussian_filter(values, sigma=sigma, mode="nearest")
    den = ndimage.gaussian_filter(weights, sigma=sigma, mode="nearest")
    out = np.zeros_like(h, dtype=np.float32)
    np.divide(num, den, out=out, where=den > 1e-6)
    out[~organ] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _top_n_mean(values: np.ndarray, n: int) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    n = max(1, min(int(n), x.size))
    return float(np.mean(np.sort(x)[-n:]))


def _scale_signal_metrics(scale_name: str, scored: pd.DataFrame, score_col: str = "csPCa_likelihood") -> dict[str, float | str]:
    if scored.empty or score_col not in scored.columns:
        return {
            "scale": str(scale_name),
            "max": np.nan,
            "p95": np.nan,
            "top10_mean": np.nan,
            "frac_ge_0_60": np.nan,
            "frac_ge_0_70": np.nan,
            "frac_ge_0_80": np.nan,
        }
    values = pd.to_numeric(scored[score_col], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "scale": str(scale_name),
            "max": np.nan,
            "p95": np.nan,
            "top10_mean": np.nan,
            "frac_ge_0_60": np.nan,
            "frac_ge_0_70": np.nan,
            "frac_ge_0_80": np.nan,
        }
    return {
        "scale": str(scale_name),
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
        "top10_mean": _top_n_mean(values, 10),
        "frac_ge_0_60": float(np.mean(values >= 0.60)),
        "frac_ge_0_70": float(np.mean(values >= 0.70)),
        "frac_ge_0_80": float(np.mean(values >= 0.80)),
    }


def _multiscale_agreement_features(feature_set: str, scale_metrics: list[dict[str, float | str]]) -> dict[str, float]:
    tag = _safe_feature_set_tag(feature_set)
    prefix = f"patch_{tag}_multiscale_csPCa_likelihood"
    rows = [m for m in scale_metrics if np.isfinite(float(m.get("max", np.nan)))]
    out: dict[str, float] = {
        f"{prefix}_n_scales": float(len(rows)),
        f"{prefix}_all_scales_max_ge_0_60": 0.0,
        f"{prefix}_all_scales_max_ge_0_70": 0.0,
        f"{prefix}_all_scales_max_ge_0_80": 0.0,
    }
    if not rows:
        return out
    for metric in ["max", "p95", "top10_mean"]:
        vals = np.asarray([float(m[metric]) for m in rows], dtype=float)
        out[f"{prefix}_{metric}_mean_across_scales"] = float(np.mean(vals))
        out[f"{prefix}_{metric}_min_across_scales"] = float(np.min(vals))
        out[f"{prefix}_{metric}_max_across_scales"] = float(np.max(vals))
        out[f"{prefix}_{metric}_std_across_scales"] = float(np.std(vals))
    for thr_tag, thr in [("0_60", 0.60), ("0_70", 0.70), ("0_80", 0.80)]:
        max_vals = np.asarray([float(m["max"]) for m in rows], dtype=float)
        p95_vals = np.asarray([float(m["p95"]) for m in rows], dtype=float)
        out[f"{prefix}_n_scales_max_ge_{thr_tag}"] = float(np.sum(max_vals >= thr))
        out[f"{prefix}_n_scales_p95_ge_{thr_tag}"] = float(np.sum(p95_vals >= thr))
        out[f"{prefix}_all_scales_max_ge_{thr_tag}"] = float(np.all(max_vals >= thr))
        out[f"{prefix}_all_scales_p95_ge_{thr_tag}"] = float(np.all(p95_vals >= thr))
    best = max(rows, key=lambda m: float(m["p95"]))
    out[f"{prefix}_best_scale_p95"] = float(best["p95"])
    out[f"{prefix}_best_scale_max"] = float(best["max"])
    return out

def _summarize_case_scale(case, scale: dict, scored: pd.DataFrame, save_heatmaps: bool, feature_set: str) -> dict:
    sc = normalize_scale(scale)
    tag = _safe_feature_set_tag(feature_set)
    prefix_base = f"patch_{tag}_{sc.name}"
    row = {
        f"{prefix_base}_n_patches": int(len(scored)),
    }
    if scored.empty:
        for score_col in SCORE_COLS:
            row.update(summarize_scores(f"{prefix_base}_{score_col}", np.array([]), PATCH_HEATMAP_THRESHOLD))
        return row

    scored = scored.copy()
    scored["csPCa_x_PZ_likelihood"] = scored["csPCa_likelihood"] * scored["PZ_likelihood"]
    scored["csPCa_x_TZ_likelihood"] = scored["csPCa_likelihood"] * scored["TZ_likelihood"]

    summary_cols = SCORE_COLS + ["csPCa_x_PZ_likelihood", "csPCa_x_TZ_likelihood"]
    for score_col in summary_cols:
        row.update(
            summarize_scores(
                f"{prefix_base}_{score_col}",
                scored[score_col].to_numpy(dtype=float),
                PATCH_HEATMAP_THRESHOLD,
            )
        )

    heatmaps: dict[str, np.ndarray] = {}
    smoothed_heatmaps: dict[str, np.ndarray] = {}
    for score_col in SCORE_COLS:
        score_values = scored[score_col].to_numpy(dtype=float)
        if np.isfinite(score_values).any():
            heatmap = make_average_heatmap(case.organ.shape, scored, score_col, sc, case.spacing_zyx)
            heatmap = np.where(case.organ, heatmap, 0.0).astype(np.float32)
            smoothed_heatmap = _smooth_probability_heatmap(heatmap, case.organ)
            organ_vals = heatmap[case.organ > 0]
            smoothed_organ_vals = smoothed_heatmap[case.organ > 0]
        else:
            heatmap = np.full(case.organ.shape, np.nan, dtype=np.float32)
            smoothed_heatmap = np.full(case.organ.shape, np.nan, dtype=np.float32)
            organ_vals = np.array([], dtype=float)
            smoothed_organ_vals = np.array([], dtype=float)
        heatmaps[score_col] = heatmap
        smoothed_heatmaps[score_col] = smoothed_heatmap
        row.update(
            summarize_scores(
                f"{prefix_base}_{score_col}_heatmap",
                organ_vals,
                PATCH_HEATMAP_THRESHOLD,
            )
        )
        row.update(
            summarize_scores(
                f"{prefix_base}_{score_col}_smoothed_heatmap",
                smoothed_organ_vals,
                PATCH_HEATMAP_THRESHOLD,
            )
        )
        largest_vox, largest_mm3 = largest_component_volume_mm3(
            smoothed_heatmap,
            threshold=PATCH_HEATMAP_THRESHOLD,
            spacing_zyx=case.spacing_zyx,
        )
        row[f"{prefix_base}_{score_col}_smoothed_largest_hotspot_voxels"] = largest_vox
        row[f"{prefix_base}_{score_col}_smoothed_largest_hotspot_mm3"] = largest_mm3
        row[f"{prefix_base}_{score_col}_largest_hotspot_voxels"] = largest_vox
        row[f"{prefix_base}_{score_col}_largest_hotspot_mm3"] = largest_mm3


        if score_col in {"PZ_likelihood", "TZ_likelihood"}:
            row.update(
                zone_continuity_features(
                    f"{prefix_base}_{score_col}_continuous",
                    smoothed_heatmap,
                    threshold=PATCH_HEATMAP_THRESHOLD,
                    spacing_zyx=case.spacing_zyx,
                    organ_mask=case.organ,
                )
            )

        if save_heatmaps:
            heatmap_path = PATCH_HELPER_HEATMAP_DIR / tag / score_col / sc.name / f"{case.case_id}.mha"
            write_heatmap(heatmap_path, heatmap, case.reference_image)


    if "csPCa_likelihood" in heatmaps:
        thresholds = sorted({float(PATCH_HEATMAP_THRESHOLD), *[float(x) for x in PATCH_CLUSTER_FEATURE_THRESHOLDS]})
        for thr in thresholds:
            tag = threshold_tag(thr)
            cluster_prefix = f"{prefix_base}_dominant_cspca_cluster_{tag}"
            feats = dominant_cspca_cluster_zone_features_v2(
                cluster_prefix,
                cspca_heatmap=smoothed_heatmaps.get("csPCa_likelihood"),
                pz_heatmap=smoothed_heatmaps.get("PZ_likelihood"),
                tz_heatmap=smoothed_heatmaps.get("TZ_likelihood"),
                threshold=thr,
                spacing_zyx=case.spacing_zyx,
                organ_mask=case.organ,
                min_cluster_voxels=PATCH_CSPCA_MIN_CLUSTER_VOXELS,
                zone_confidence=PATCH_CLUSTER_ZONE_CONFIDENCE,
                z_support_xy_radius=PATCH_CSPCA_Z_SUPPORT_XY_RADIUS_VOXELS,
                xy_closing_iterations=PATCH_CSPCA_XY_CLOSING_ITERATIONS,
            )
            row.update(feats)


            if abs(thr - float(PATCH_HEATMAP_THRESHOLD)) < 1e-12:
                row.update(
                    dominant_cspca_cluster_zone_features_v2(
                        f"{prefix_base}_dominant_cspca_cluster",
                        cspca_heatmap=smoothed_heatmaps.get("csPCa_likelihood"),
                        pz_heatmap=smoothed_heatmaps.get("PZ_likelihood"),
                        tz_heatmap=smoothed_heatmaps.get("TZ_likelihood"),
                        threshold=thr,
                        spacing_zyx=case.spacing_zyx,
                        organ_mask=case.organ,
                        min_cluster_voxels=PATCH_CSPCA_MIN_CLUSTER_VOXELS,
                        zone_confidence=PATCH_CLUSTER_ZONE_CONFIDENCE,
                        z_support_xy_radius=PATCH_CSPCA_Z_SUPPORT_XY_RADIUS_VOXELS,
                        xy_closing_iterations=PATCH_CSPCA_XY_CLOSING_ITERATIONS,
                    )
                )

            if save_heatmaps and PATCH_SAVE_CLUSTER_OVERLAYS:
                masks = cspca_cluster_overlay_masks(
                    heatmaps.get("csPCa_likelihood"),
                    threshold=thr,
                    organ_mask=case.organ,
                    min_cluster_voxels=PATCH_CSPCA_MIN_CLUSTER_VOXELS,
                    z_support_xy_radius=PATCH_CSPCA_Z_SUPPORT_XY_RADIUS_VOXELS,
                    xy_closing_iterations=PATCH_CSPCA_XY_CLOSING_ITERATIONS,
                )
                for mask_name, mask_arr in masks.items():
                    overlay_path = (
                    PATCH_HELPER_HEATMAP_DIR
                    / tag
                    / "cluster_overlays"
                    / mask_name
                        / tag
                        / sc.name
                        / f"{case.case_id}.mha"
                    )
                    write_heatmap(overlay_path, mask_arr.astype(np.float32), case.reference_image)

    return row


def _process_main_patch_case_task(task: tuple[str, bool]):
    case_id, is_view = task
    case = load_case_arrays(MAIN_ROOT, case_id, require_lesion=False, require_zone=False)
    if case is None:
        return case_id, None, None, None

    out_row = {ID_COL: str(case_id)}
    multiscale_parts: dict[str, list[dict[str, float | str]]] = {feature_set: [] for feature_set in PATCH_FEATURE_SETS}
    prediction_parts = {feature_set: [] for feature_set in PATCH_FEATURE_SETS} if is_view else None
    raw_feature_parts = {feature_set: [] for feature_set in PATCH_FEATURE_SETS} if is_view else None
    for scale in PATCH_SCALES:
        sc = normalize_scale(scale)
        patch_df = build_patch_dataframe_for_case(case, sc, include_labels=False)
        for feature_set in PATCH_FEATURE_SETS:
            feature_df = _feature_set_table(patch_df, feature_set, include_labels=False)
            scored = _score_patch_table(feature_df, sc.name, feature_set)
            multiscale_parts[feature_set].append(_scale_signal_metrics(sc.name, scored, "csPCa_likelihood"))
            out_row.update(_summarize_case_scale(case, sc, scored, save_heatmaps=False, feature_set=feature_set))
            if is_view:
                raw_feature_parts[feature_set].append(feature_df.copy())
                pred_cols = [
                    ID_COL, "scale", "center_z", "center_y", "center_x",
                    "patch_z", "patch_y", "patch_x", "organ_overlap", *SCORE_COLS,
                ]
                prediction_parts[feature_set].append(scored.reindex(columns=[c for c in pred_cols if c in scored.columns]))

    for feature_set, metrics in multiscale_parts.items():
        out_row.update(_multiscale_agreement_features(feature_set, metrics))

    raw_views = pred_views = None
    if is_view:
        raw_views = {fs: (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()) for fs, parts in raw_feature_parts.items()}
        pred_views = {fs: (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()) for fs, parts in prediction_parts.items()}
    return case_id, out_row, raw_views, pred_views


def apply_patch_helpers(allow_missing_models: bool = False, write_views: bool = True) -> pd.DataFrame:
    _ensure_dirs()
    validate_external_main_case_separation()
    missing_models = []
    for scale in PATCH_SCALES:
        sc = normalize_scale(scale)
        for feature_set in PATCH_FEATURE_SETS:
            for task in ("cspca", "zone"):
                path = _task_model_path(sc.name, task, feature_set)
                if not path.exists():
                    missing_models.append(str(path))
    if missing_models and not allow_missing_models:
        raise FileNotFoundError(
            "Missing required patch-helper models. Run b_patch.py --stage train "
            "with enough external cases; no NaN-score fallback is used by default.\n"
            + "\n".join(missing_models)
        )
    for path in missing_models:
        print(f"[PATCH APPLY] explicit debug fallback enabled; missing model scores will be NaN: {path}")

    case_ids = list_case_ids(MAIN_ROOT)
    expected_ids = {str(c).strip() for c in case_ids}

    paths = _expected_patch_workbook_paths()
    if _all_final_patch_workbooks_exist():
        print("[PATCH APPLY] final A_Patch summary features already exist; skipping apply")
        return pd.DataFrame({ID_COL: sorted(expected_ids)})

    existing_by_file: dict[str, pd.DataFrame] = {}
    done_case_ids: set[str] | None = None
    for p in paths:
        if p.exists():
            existing_by_file[p.name] = pd.read_csv(p, dtype={ID_COL: str})
            ids = cfg.patch_workbook_done_ids(p)
            done_case_ids = ids if done_case_ids is None else (done_case_ids & ids)
        else:
            done_case_ids = set()
    done_case_ids = done_case_ids or set()
    if done_case_ids:
        print(f"[PATCH APPLY] resume: {len(done_case_ids)} cases already published in {PATCH_ONLY_WORKBOOK_DIR}")
    elif not existing_by_file:
        print(f"[PATCH APPLY] Summary Feature Files are built and will be updated per {cfg.EXCEL_FLUSH_EVERY} cases")

    pending_ids = [c for c in case_ids if str(c).strip() not in done_case_ids]
    view_case = (str(pending_ids[0]).strip() if (pending_ids and write_views) else None)
    tasks = [(case_id, str(case_id).strip() == view_case) for case_id in pending_ids]
    rows: list[dict] = []
    view_raw: dict[str, pd.DataFrame] = {}
    view_pred: dict[str, pd.DataFrame] = {}
    completed = 0
    total = len(tasks)

    def _publish() -> None:
        if not rows:
            return
        new_helper = pd.DataFrame(rows)
        new_helper[ID_COL] = new_helper[ID_COL].astype(str).str.strip()
        this_run_ids = set(new_helper[ID_COL])
        new_books = _build_patch_books(new_helper)
        for file_name, new_df in new_books.items():
            old_df = existing_by_file.get(file_name)
            if old_df is not None and not old_df.empty:
                new_only = new_df[new_df[ID_COL].astype(str).str.strip().isin(this_run_ids)]
                combined = pd.concat([old_df, new_only], ignore_index=True).drop_duplicates(ID_COL, keep="last")
            else:
                combined = new_df
            _write_feature_workbook(PATCH_ONLY_WORKBOOK_DIR / file_name, combined, quiet=True)

    def _handle(result) -> None:
        nonlocal completed
        case_id, out_row, raw_views, pred_views = result
        completed += 1
        if out_row is not None:
            rows.append(out_row)
        if raw_views is not None:
            view_raw.update(raw_views)
            view_pred.update(pred_views)
        _vprint(f"[PATCH APPLY] {case_id}: {'ok' if out_row is not None else 'skip'}")
        if completed % cfg.EXCEL_FLUSH_EVERY == 0:
            _publish()

    if PATCH_APPLY_WORKERS <= 1:
        for task in tasks:
            cfg.log_progress("PATCH_APPLY", completed + 1, total)
            _handle(_process_main_patch_case_task(task))
    else:
        try:
            with ProcessPoolExecutor(max_workers=PATCH_APPLY_WORKERS) as ex:
                futures = [ex.submit(_process_main_patch_case_task, task) for task in tasks]
                for f in as_completed(futures):
                    cfg.log_progress("PATCH_APPLY", completed + 1, total)
                    _handle(f.result())
        except BrokenProcessPool:
            done_this_run = {str(r[ID_COL]).strip() for r in rows}
            remaining = [t for t in tasks if str(t[0]).strip() not in done_this_run]
            print(f"[PATCH APPLY] WARNING: a worker died; retrying {len(remaining)} remaining cases sequentially.")
            for task in remaining:
                _handle(_process_main_patch_case_task(task))

    _publish()
    print(f"[PATCH APPLY] applied {len(rows)} new cases (resumed={len(done_case_ids)})")

    features = pd.DataFrame(rows)
    if ID_COL in features.columns:
        features[ID_COL] = features[ID_COL].astype(str).str.strip()
    if write_views and view_case is not None:
        for feature_set in PATCH_FEATURE_SETS:
            _write_inference_raw_feature_table(feature_set, view_raw.get(feature_set, pd.DataFrame()))
            _write_patch_prediction_table(feature_set, view_pred.get(feature_set, pd.DataFrame()))
    return features


def _clean_case_id_value(value) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    return s


def _load_master_case_labels(dataset_xlsx: Path) -> pd.DataFrame:
    labels = pd.read_excel(dataset_xlsx, dtype={ID_COL: str})
    labels.columns = [str(c).strip() for c in labels.columns]

    if ID_COL not in labels.columns:
        lowered = {str(c).strip().lower(): c for c in labels.columns}
        if ID_COL.lower() not in lowered:
            raise KeyError(f"Could not find ID column {ID_COL!r} in {dataset_xlsx}")
        labels = labels.rename(columns={lowered[ID_COL.lower()]: ID_COL})

    required = [ID_COL, CLIN_COL, ZONE_COL]
    missing = [c for c in required if c not in labels.columns]
    if missing:
        raise KeyError(f"Missing required label columns in {dataset_xlsx}: {missing}")

    labels = labels[required].copy()
    labels[ID_COL] = labels[ID_COL].map(_clean_case_id_value)
    labels = labels.dropna(subset=[ID_COL])
    return labels.drop_duplicates(ID_COL, keep="first").reset_index(drop=True)


def _patch_labels_for_file(file_name: str) -> pd.DataFrame:
    return _load_master_case_labels(DATASET_XLSX)


def _write_feature_workbook(path: Path, df: pd.DataFrame, quiet: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.sort_values(ID_COL).to_csv(tmp, index=False)
    os.replace(tmp, path)
    if not quiet:
        print(f"[PATCH] saved {path}")


def _final_suffix_for_feature_set(feature_set: str) -> str:
    fs = str(feature_set).strip()
    if fs in {"t2w", "adc", "diff", "hada", "concat"}:
        return f"_{fs}"
    if fs.startswith("fusion(") and fs.endswith(")"):
        return f"_{fs}"
    safe = re.sub(r"[^A-Za-z0-9]+", "_", fs).strip("_")
    return f"_{safe}"


def _helper_feature_block(helper: pd.DataFrame, source_feature_set: str, final_suffix: str | None = None) -> pd.DataFrame:
    tag = _safe_feature_set_tag(source_feature_set)
    suffix = final_suffix or _final_suffix_for_feature_set(source_feature_set)
    prefix = f"patch_{tag}_"
    cols = [c for c in helper.columns if str(c).startswith(prefix)]
    block = helper[[ID_COL] + cols].copy()
    rename = {c: f"{str(c)[len(prefix):]}{suffix}" for c in cols}
    return block.rename(columns=rename)


def _component_feature_sets(feature_set: str) -> list[tuple[str, str]]:
    return [(str(feature_set), _final_suffix_for_feature_set(feature_set))]


def _case_features_for_set(labels: pd.DataFrame, helper: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    out = labels.copy()
    block = _helper_feature_block(helper, feature_set, _final_suffix_for_feature_set(feature_set))
    duplicate_cols = [c for c in block.columns if c != ID_COL and c in out.columns]
    out = out.merge(block.drop(columns=duplicate_cols), on=ID_COL, how="left")
    return out

def _build_patch_workbooks_for_file(file_name: str, helper: pd.DataFrame) -> dict[str, pd.DataFrame]:
    labels = _patch_labels_for_file(file_name)
    return {
        _feature_file_for_set(feature_set): _case_features_for_set(labels, helper, feature_set)
        for feature_set in PATCH_FEATURE_SETS
    }


def _patch_only_sheet(main_df: pd.DataFrame, helper: pd.DataFrame) -> pd.DataFrame:
    label_cols = [ID_COL, CLIN_COL, ZONE_COL]
    labels = main_df[label_cols].copy()
    labels[ID_COL] = labels[ID_COL].astype(str).str.strip()
    return labels.merge(helper, on=ID_COL, how="left")


def _write_patch_and_early_workbooks() -> list[dict]:
    if _all_final_patch_workbooks_exist() and _all_early_fusion_workbooks_exist():
        print("[PATCH] final patch and early-fusion workbooks already exist; skipping merge")
        return []
    import d_fusion
    return d_fusion.build_early_fusion_workbooks()


def _build_patch_books(helper: pd.DataFrame) -> dict[str, pd.DataFrame]:
    helper = helper.copy()
    helper[ID_COL] = helper[ID_COL].astype(str).str.strip()
    books: dict[str, pd.DataFrame] = {}
    for file_name in FEATURE_FILES:
        patch_books = _build_patch_workbooks_for_file(file_name, helper)
        books[file_name] = patch_books[file_name]
    return books


def write_patch_workbooks_from_helper(helper: pd.DataFrame) -> list[dict]:
    books = _build_patch_books(helper)
    report = []
    for file_name, patch_df in books.items():
        patch_dst = PATCH_ONLY_WORKBOOK_DIR / file_name
        _write_feature_workbook(patch_dst, patch_df)
        report.append(
            {
                "experiment": "A_Patch",
                "file": file_name,
                "rows": int(len(patch_df)),
                "features": int(max(0, patch_df.shape[1] - 3)),
                "path": str(patch_dst),
            }
        )
    return report


def build_experiment_workbooks() -> pd.DataFrame:
    _ensure_dirs()
    if _all_final_patch_workbooks_exist() and _all_early_fusion_workbooks_exist():
        print("[PATCH] all final patch and early-fusion workbooks already exist; nothing to merge")
        return pd.DataFrame()
    report = _write_patch_and_early_workbooks()
    return pd.DataFrame(report)


def build_final_workbooks() -> pd.DataFrame:
    return build_experiment_workbooks()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train external multi-scale patch helper models, apply them over main organ ROI, "
            "and build final patch-only and early-fusion feature workbooks."
        )
    )
    parser.add_argument("--stage", choices=["train", "apply", "merge", "all"], default=DEFAULT_PATCH_STAGE)
    parser.add_argument(
        "--allow-skipped-models",
        action="store_true",
        help="Debug only: continue when a helper task has one class. Default is to fail loudly.",
    )
    parser.add_argument(
        "--allow-missing-models",
        action="store_true",
        help="Debug only: fill missing helper-model scores with NaN. Default is to fail loudly.",
    )
    args = parser.parse_args()

    if args.stage in {"train", "all"}:
        train_patch_helpers(allow_skipped_models=args.allow_skipped_models)
    if args.stage in {"apply", "all"}:
        apply_patch_helpers(allow_missing_models=args.allow_missing_models)
    if args.stage in {"merge", "all"}:
        build_final_workbooks()


if __name__ == "__main__":
    main()
