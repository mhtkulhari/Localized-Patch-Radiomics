#!/usr/bin/env python3
from __future__ import annotations

import argparse
import string
import textwrap
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mht_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import ScaledTranslation
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from all_config import RANDOM_SEEDS, RANDOM_STATE


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "Output"
DEFAULT_OUTPUT = DEFAULT_INPUT / "graphs"
DEFAULT_SEED = RANDOM_STATE

EXPERIMENTS = ("A_Patch", "B_Organ", "C_Early", "D_Late")
EXPERIMENT_LABELS = {
    "A_Patch": "A: Patch",
    "B_Organ": "B: Organ",
    "C_Early": "C: Early Fusion",
    "D_Late": "D: Late Fusion",
    "A_Organ": "A: Organ",
    "B_Patch": "B: Patch",
    "C_LateFusion": "C: Late Fusion",
    "D_EarlyFusion": "D: Early Fusion",
}
TASKS = ("cs", "zone")
TASK_LABELS = {"cs": "csPCa", "zone": "Zone"}
BINARY_CLASS_LABELS = {
    "cs": ("non-csPCa", "csPCa"),
    "zone": ("PZ", "TZ"),
}
BINARY_CLASS_Y_LABELS = {
    "cs": ("non\ncsPCa", "csPCa"),
    "zone": ("PZ", "TZ"),
}
BINARY_CONFUSION_ORDER = [1, 0]


def fixed_threshold(task: str) -> float:
    return 0.45 if str(task).lower() == "zone" else 0.5


def fixed_pred(score: pd.Series | np.ndarray, task: str) -> np.ndarray:
    return (pd.to_numeric(pd.Series(score), errors="coerce").to_numpy(dtype=float) >= fixed_threshold(task)).astype(int)


CLASS_ORDER_3 = ("FALSE", "TRUE_PZ", "TRUE_TZ")
CLASS_LABELS_3 = ("non-csPCa", "csPCa PZ", "csPCa TZ")
CLASS_Y_LABELS_3 = ("non\ncsPCa", "csPCa\nPZ", "csPCa\nTZ")

NON_FEATURE_COLS = {
    "case_id",
    "patient_id",
    "clin_sig",
    "binary",
    "y",
    "y_label",
    "fold",
    "outer_fold",
}

PAPER_COLORS = [
    "#2b6cb0",
    "#2f855a",
    "#c05621",
    "#805ad5",
    "#b83280",
    "#718096",
    "#d69e2e",
    "#319795",
    "#e53e3e",
]

CAPTIONS_ENABLED = False
# Physical spacing controls in inches.  These are converted per figure so the
# same values look consistent for short, wide, and tall figures.
width_ver = 0.10
width_hor = 0.10
save_pad_inches = 0.10
box_body_width = 0.25
external_bar_width = 0.34
combined_axis_label_fontsize = 10
combined_tick_label_fontsize = 9
combined_legend_fontsize = 8


@dataclass
class RunContext:
    input_path: Path
    run_root: Path
    output_root: Path
    dpi: int
    top_n: int
    skip_heavy: bool
    no_external: bool
    captions: bool
    seed: int


@dataclass
class RocPrCurve:
    task: str
    best_fold: Optional[int]
    fpr: np.ndarray
    tpr: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    roc_auc: float
    pr_auc: float
    prevalence: float


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "STIXGeneral", "CMU Serif", "Computer Modern Roman", "cmr10"],
            "mathtext.fontset": "dejavuserif",
            "axes.formatter.use_mathtext": True,
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "text.color": "#050505",
            "axes.labelcolor": "#050505",
            "xtick.color": "#050505",
            "ytick.color": "#050505",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.2,
            "lines.markersize": 5,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.65,
            "axes.axisbelow": True,
        }
    )


def clean_name(raw: object) -> str:
    text = str(raw)
    text = text.replace(".xlsx", "")
    text = text.replace("feature--", "")
    return text


def experiment_label(raw: object) -> str:
    return EXPERIMENT_LABELS.get(str(raw), clean_name(raw))


def experiment_axis_label(raw: object) -> str:
    return experiment_label(raw)


def compact_feature_name(name: object, max_len: int = 54) -> str:
    text = str(name)
    text = re.sub(r"^s([123])_", r"S\1 ", text)
    text = text.replace("_count_ge_threshold", " count ge thr")
    text = text.replace("_frac_ge_threshold", " frac ge thr")
    text = text.replace("_likelihood", "")
    text = text.replace("_smoothed_heatmap", " smooth")
    text = text.replace("_heatmap", " heat")
    text = text.replace("_dominant_cspca_cluster", " cluster")
    text = text.replace("_largest_hotspot", " hotspot")
    text = text.replace("_count_ge_", " ge")
    text = text.replace("_frac_ge_", " frac ge")
    text = text.replace("_top", " top")
    text = text.replace("_mean", " mean")
    text = text.replace("_std", " sd")
    text = text.replace("_median", " med")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "."


def feature_family(name: object) -> str:
    text = str(name).lower()
    if "cluster" in text:
        return "cluster"
    if "hotspot" in text:
        return "hotspot"
    if "smoothed_heatmap" in text:
        return "smooth heat"
    if "heatmap" in text:
        return "heatmap"
    if "_x_" in text or "x_pz" in text or "x_tz" in text:
        return "cs-zone cross"
    if "pz_" in text or "tz_" in text:
        return "zone prob"
    if "count_ge" in text or "frac_ge" in text or "threshold" in text:
        return "threshold"
    if "likelihood" in text:
        return "patch stats"
    if "n_patches" in text:
        return "patch count"
    return "other"


def resolve_input_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def resolve_run_root(input_path: Path) -> Path:
    if (input_path / "04_results").exists() or (input_path / "05_external_P158").exists():
        return input_path
    if (input_path / "results_stacking.xlsx").exists():
        return input_path
    if (input_path / "final_stacking.xlsx").exists():
        return input_path
    if any((input_path / name).exists() for name in EXPERIMENTS):
        return input_path

    sibling = input_path.parent / f"#{input_path.name.lstrip('#')}"
    if sibling.exists() and ((sibling / "04_results").exists() or (sibling / "05_external_P158").exists()):
        return sibling

    if input_path.name.startswith("#"):
        plain = input_path.parent / input_path.name.lstrip("#")
        if (plain / "a_config.py").exists():
            return input_path

    raise FileNotFoundError(
        f"Could not resolve an experiment output root from {input_path}. "
        "Pass a run folder such as '#new_config_12' or a code folder such as 'new_config_12'."
    )


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_axes(fig: plt.Figure) -> List[plt.Axes]:
    axes: List[plt.Axes] = []
    for ax in fig.axes:
        if not ax.get_visible() or str(ax.get_label()).startswith("<colorbar"):
            continue
        if ax.has_data() or ax.get_xlabel() or ax.get_ylabel():
            axes.append(ax)
    return axes


def task_from_stem(stem: str) -> str:
    tokens = stem.split("_")
    if "cs" in tokens:
        return "csPCa"
    if "zone" in tokens:
        return "zone"
    return ""


def variant_from_stem(stem: str) -> str:
    if stem.endswith("_pooled"):
        return "pooled "
    if stem.endswith("_best"):
        return "best outer-fold "
    return ""


def caption_for(out_path: Path) -> str:
    stem = out_path.stem
    task = task_from_stem(stem)
    variant = variant_from_stem(stem)
    captions = {
        "stack_cv_stability": "Fig: Outer-fold stability for stacked classifiers: (a) csPCa vs non-csPCa and (b) PZ vs TZ.",
        "stack_confusion_binary": "Fig: Stacked binary confusion matrices: (a) csPCa and (b) zone.",
        "stack_confusion_binary_pooled": "Fig: Pooled outer-fold stacked binary confusion matrices: (a) csPCa and (b) zone.",
        "stack_confusion_binary_best": "Fig: Best outer-fold stacked binary confusion matrices: (a) csPCa and (b) zone.",
        "stack_confusion_3class": "Fig: Three-class stacked confusion matrix for non-csPCa, csPCa PZ, and csPCa TZ.",
        "stack_confusion_3class_pooled": "Fig: Pooled outer-fold three-class stacked confusion matrix for non-csPCa, csPCa PZ, and csPCa TZ.",
        "stack_confusion_3class_best": "Fig: Best outer-fold three-class stacked confusion matrix for non-csPCa, csPCa PZ, and csPCa TZ.",
        "stack_roc_pr_combined": "Fig: Stacked discrimination curves: (a) ROC-AUC for csPCa vs non-csPCa, (b) average precision for csPCa vs non-csPCa, (c) ROC-AUC for PZ vs TZ, and (d) average precision for PZ vs TZ.",
        "stack_roc_pr_combined_pooled": "Fig: Pooled outer-fold stacked discrimination curves: (a) ROC-AUC for csPCa vs non-csPCa, (b) average precision for csPCa vs non-csPCa, (c) ROC-AUC for PZ vs TZ, and (d) average precision for PZ vs TZ.",
        "stack_roc_pr_combined_best": "Fig: Best outer-fold stacked discrimination curves: (a) ROC-AUC for csPCa vs non-csPCa, (b) average precision for csPCa vs non-csPCa, (c) ROC-AUC for PZ vs TZ, and (d) average precision for PZ vs TZ.",
        "stack_score_distributions": "Fig: Probability prediction distributions across outer folds: (a) csPCa and (b) zone.",
        "stack_score_distributions_pooled": "Fig: Pooled outer-fold probability prediction distributions: (a) csPCa and (b) zone.",
        "stack_score_distributions_best_auc_fold": "Fig: Best-AUC outer-fold probability prediction distributions: (a) csPCa and (b) zone.",
        "stack_score_distributions_best": "Fig: Best outer-fold probability prediction distributions: (a) csPCa and (b) zone.",
        "stack_feature_set_use": "Fig: Feature-set use in stacked models: (a) selected base models and (b) aggregate stack weights.",
        "stack_feature_reduction": "Fig: Median feature reduction across stacked-model candidate pipelines.",
        "external_validation_metrics": "Fig: External P158 validation summary for stacked csPCa and zone models.",
        "external_cv_stability": "Fig: External P158 outer-fold stability for stacked classifiers: (a) csPCa vs non-csPCa and (b) PZ vs TZ.",
        "external_confusion_binary": "Fig: External binary confusion matrices: (a) csPCa and (b) zone.",
        "external_confusion_binary_pooled": "Fig: External P158 pooled binary confusion matrices averaged across saved outer-fold models: (a) csPCa and (b) zone.",
        "external_confusion_binary_best": "Fig: External P158 best outer-fold binary confusion matrices: (a) csPCa and (b) zone.",
        "external_roc_pr_combined": "Fig: External P158 discrimination curves: (a) ROC-AUC for csPCa vs non-csPCa, (b) average precision for csPCa vs non-csPCa, (c) ROC-AUC for PZ vs TZ, and (d) average precision for PZ vs TZ.",
        "external_roc_pr_combined_pooled": "Fig: External P158 pooled discrimination curves averaged across saved outer-fold models: (a) ROC-AUC for csPCa vs non-csPCa, (b) average precision for csPCa vs non-csPCa, (c) ROC-AUC for PZ vs TZ, and (d) average precision for PZ vs TZ.",
        "external_roc_pr_combined_best": "Fig: External P158 best outer-fold discrimination curves: (a) ROC-AUC for csPCa vs non-csPCa, (b) average precision for csPCa vs non-csPCa, (c) ROC-AUC for PZ vs TZ, and (d) average precision for PZ vs TZ.",
        "external_score_distributions": "Fig: External P158 probability prediction distributions averaged across saved outer-fold models: (a) csPCa and (b) zone.",
        "external_score_distributions_pooled": "Fig: External P158 pooled probability prediction distributions averaged across saved outer-fold models: (a) csPCa and (b) zone.",
        "external_score_distributions_best_auc_fold": "Fig: External P158 best-AUC outer-fold probability prediction distributions: (a) csPCa and (b) zone.",
        "external_score_distributions_best": "Fig: External P158 best outer-fold probability prediction distributions: (a) csPCa and (b) zone.",
    }
    if stem in captions:
        return captions[stem]
    if stem.startswith("stack_roc_pr_") and task:
        return f"Fig: Stacked {variant}{task} discrimination: (a) ROC-AUC score and (b) average precision score."
    if stem.startswith("external_roc_pr_") and task:
        return f"Fig: External {variant}{task} discrimination: (a) ROC-AUC score and (b) average precision score."
    if stem.startswith("selected_features_top_") and task:
        return f"Fig: Top selected {task} features across outer folds."
    if stem.startswith("selected_feature_families_") and task:
        return f"Fig: Selected {task} feature-family frequencies across feature sets."
    if stem.startswith("feature_effect_top_") and task:
        return f"Fig: Top univariate {task} feature effects measured by standardized mean difference."
    if stem.startswith("feature_effect_family_heatmap_") and task:
        return f"Fig: {task} feature-family separation summarized by top mean absolute SMD."
    return f"Fig: {stem.replace('_', ' ').capitalize()}."


def panel_label_for(out_path: Path, idx: int) -> str:
    if out_path.stem in {"stack_cv_stability", "external_cv_stability"} or out_path.stem.startswith(
        (
            "stack_confusion_binary",
            "external_confusion_binary",
            "stack_roc_combined",
            "external_roc_combined",
            "stack_ap_combined",
            "external_ap_combined",
            "stack_score_distributions",
            "external_score_distributions",
            "internal_auc",
            "external_auc",
            "internal_box",
            "external_box",
        )
    ):
        labels = ["(a) csPCa vs non-csPCa", "(b) TZ vs PZ"]
        if idx < len(labels):
            return labels[idx]
    if out_path.stem.startswith(("stack_roc_pr_combined", "external_roc_pr_combined")):
        labels = [
            "(a) ROC-AUC (csPCa vs non-csPCa)",
            "(b) Average Precision (csPCa vs non-csPCa)",
            "(c) ROC-AUC (PZ vs TZ)",
            "(d) Average Precision (PZ vs TZ)",
        ]
        if idx < len(labels):
            return labels[idx]
    if out_path.stem.startswith(("stack_roc_pr_", "external_roc_pr_")):
        labels = ["(a) ROC-AUC Score", "(b) Average Precision Score"]
        if idx < len(labels):
            return labels[idx]
    return f"({string.ascii_lowercase[idx]})"


def panel_label_fontsize(out_path: Path) -> float:
    if out_path.stem.startswith(("stack_roc_pr_combined", "external_roc_pr_combined")):
        return 8.5
    return 10.0


def wrap_caption(fig: plt.Figure, caption: str, width_frac: float = 1.0) -> str:
    width_inches = max(fig.get_figwidth() * width_frac, 1.0)
    width = int(width_inches * 11.5)
    width = max(36, min(95, width))
    return textwrap.fill(caption, width=width)


def text_height_frac(fig: plt.Figure, fontsize: float, lines: int = 1, linespacing: float = 1.15) -> float:
    return (fontsize / 72.0 * max(lines, 1) * linespacing) / max(fig.get_figheight(), 1e-6)


def width_ver_frac(fig: plt.Figure) -> float:
    return width_ver / max(fig.get_figheight(), 1e-6)


def width_hor_frac(fig: plt.Figure) -> float:
    return width_hor / max(fig.get_figwidth(), 1e-6)


def axis_legend(
    ax: plt.Axes,
    loc: str = "best",
    handles: Optional[Sequence[object]] = None,
    labels: Optional[Sequence[str]] = None,
    frame_alpha: float = 0.88,
    borderpad: float = 0.35,
    handlelength: float = 1.8,
    labelspacing: float = 0.35,
    handletextpad: float = 0.42,
    fontsize: Optional[float] = None,
) -> None:
    leg = ax.legend(
        handles=handles,
        labels=labels,
        frameon=True,
        loc=loc,
        borderpad=borderpad,
        handlelength=handlelength,
        labelspacing=labelspacing,
        handletextpad=handletextpad,
        fontsize=fontsize,
    )
    if leg is not None:
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_alpha(frame_alpha)
        frame.set_edgecolor("#e6e6e6")
        frame.set_linewidth(0.5)


def apply_combined_axis_typography(ax: plt.Axes) -> None:
    ax.xaxis.label.set_fontsize(combined_axis_label_fontsize)
    ax.yaxis.label.set_fontsize(combined_axis_label_fontsize)
    ax.tick_params(
        axis="both",
        labelsize=combined_tick_label_fontsize,
    )


def add_top_legend(
    fig: plt.Figure,
    handles: Sequence[object],
    labels: Sequence[str],
    ncol: Optional[int] = None,
) -> None:
    if not handles or not labels:
        return
    ncol = ncol or min(len(labels), 4)
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0 - width_ver_frac(fig)),
        ncol=ncol,
        columnspacing=1.25,
        handlelength=2.0,
        borderaxespad=0.0,
    )


def collect_legend_items(axes: Sequence[plt.Axes]) -> Tuple[List[object], List[str]]:
    handles: List[object] = []
    labels: List[str] = []
    seen = set()
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if not label or label.startswith("_") or label in seen:
                continue
            handles.append(handle)
            labels.append(label)
            seen.add(label)
    return handles, labels


def visible_axis_bboxes(fig: plt.Figure, axes: Sequence[plt.Axes]):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    bboxes = []
    for ax in axes:
        bbox = ax.get_tightbbox(renderer)
        if bbox is None:
            bbox = ax.bbox
        bboxes.append(inv.transform_bbox(bbox))
    return bboxes


def equalize_horizontal_visible_gaps(fig: plt.Figure, axes: Sequence[plt.Axes]) -> None:
    if len(axes) <= 1:
        return

    bboxes = visible_axis_bboxes(fig, axes)
    block_widths = [bbox.width for bbox in bboxes]
    total_width = float(sum(block_widths))
    available_gap = 1.0 - total_width
    if available_gap <= 0:
        return

    # Measure from each subplot's visible bbox, not just the axes spine.  If the
    # requested padding cannot fit because labels are wide, keep all horizontal
    # gaps equal and use the largest gap that fits.
    hgap = width_hor_frac(fig)
    gap = hgap if total_width + hgap * (len(axes) + 1) <= 1.0 else available_gap / (len(axes) + 1)
    gap = max(0.010, gap)

    next_left = gap
    for ax, bbox in zip(axes, bboxes):
        delta = next_left - bbox.x0
        pos = ax.get_position()
        ax.set_position([pos.x0 + delta, pos.y0, pos.width, pos.height])
        next_left += bbox.width + gap


def content_width_fraction(fig: plt.Figure, axes: Sequence[plt.Axes]) -> float:
    if not axes:
        return 1.0
    bboxes = visible_axis_bboxes(fig, axes)
    left = min(bbox.x0 for bbox in bboxes)
    right = max(bbox.x1 for bbox in bboxes)
    return max(0.20, min(1.0, right - left))


def decorate_figure(fig: plt.Figure, out_path: Path) -> None:
    axes = plot_axes(fig)
    for ax in axes:
        ax.set_title("")

    show_panels = len(axes) > 1
    raw_caption = caption_for(out_path) if CAPTIONS_ENABLED else ""
    caption = wrap_caption(fig, raw_caption) if raw_caption else ""
    caption_lines = caption.count("\n") + 1 if caption else 0
    panel_fontsize = panel_label_fontsize(out_path)

    vgap = width_ver_frac(fig)
    hgap = width_hor_frac(fig)

    # width_ver controls vertical rows; width_hor controls left/right padding
    # and the gap between panels in multi-plot figures.  They are stored in
    # inches and converted to figure fractions here.
    cursor = vgap
    caption_y = None
    if caption:
        caption_y = cursor
        cursor += text_height_frac(fig, fontsize=10, lines=caption_lines, linespacing=1.15) + vgap

    panel_y = None
    if show_panels:
        panel_y = cursor
        cursor += text_height_frac(fig, fontsize=panel_fontsize, lines=1, linespacing=1.0) + vgap

    bottom = min(cursor, 0.42)
    top = 1.0 - vgap
    if fig.legends:
        top -= text_height_frac(fig, fontsize=9, lines=1, linespacing=1.0) + vgap

    try:
        fig.tight_layout(rect=[hgap, bottom, 1.0 - hgap, top], pad=0.08, h_pad=0.20, w_pad=0.20)
    except Exception:
        pass
    if show_panels:
        equalize_horizontal_visible_gaps(fig, axes)

    if raw_caption:
        content_frac = content_width_fraction(fig, axes)
        caption2 = wrap_caption(fig, raw_caption, width_frac=content_frac)
        if caption2 != caption:
            caption = caption2
            caption_lines = caption.count("\n") + 1
            cursor = vgap
            caption_y = cursor
            cursor += text_height_frac(fig, fontsize=10, lines=caption_lines, linespacing=1.15) + vgap
            if show_panels:
                panel_y = cursor
                cursor += text_height_frac(fig, fontsize=panel_fontsize, lines=1, linespacing=1.0) + vgap
            bottom = min(cursor, 0.42)
            try:
                fig.tight_layout(rect=[hgap, bottom, 1.0 - hgap, top], pad=0.08, h_pad=0.20, w_pad=0.20)
            except Exception:
                pass
            if show_panels:
                equalize_horizontal_visible_gaps(fig, axes)

    if show_panels and panel_y is not None:
        for idx, ax in enumerate(axes):
            if idx >= len(string.ascii_lowercase):
                break
            box = ax.get_position()
            fig.text(
                (box.x0 + box.x1) / 2.0,
                panel_y,
                panel_label_for(out_path, idx),
                ha="center",
                va="bottom",
                fontsize=panel_fontsize,
            )

    if caption and caption_y is not None:
        fig.text(0.5, caption_y, caption, ha="center", va="bottom", fontsize=10, linespacing=1.15)


def save_fig(fig: plt.Figure, out_path: Path, dpi: int) -> Path:
    safe_mkdir(out_path.parent)
    decorate_figure(fig, out_path)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=save_pad_inches, facecolor="white")
    plt.close(fig)
    return out_path


def excel_sheet(path: Path, sheet: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except ValueError:
        return pd.DataFrame()


def available_sheets(path: Path) -> List[str]:
    try:
        return pd.ExcelFile(path).sheet_names
    except Exception:
        return []


def is_number_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_bool_label(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return int(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "cs", "cspca", "positive", "pos"}:
        return 1
    if text in {"0", "false", "no", "n", "non-cspca", "negative", "neg"}:
        return 0
    return None


def parse_zone_label(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        text = value.strip().upper()
        if text == "PZ":
            return 0
        if text == "TZ":
            return 1
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return int(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)
    return None


def normalize_3class(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return "TRUE_PZ" if bool(value) else "FALSE"
    text = str(value).strip().upper()
    text = text.replace(" ", "_").replace("-", "_")
    if text in {"FALSE", "0", "NON_CSPCA", "NON_CS", "NEGATIVE"}:
        return "FALSE"
    if text in {"TRUE_PZ", "PZ", "CSPCA_PZ"}:
        return "TRUE_PZ"
    if text in {"TRUE_TZ", "TZ", "CSPCA_TZ"}:
        return "TRUE_TZ"
    return None


def get_experiment_dirs(run_root: Path, wanted: Sequence[str], seed: int) -> Dict[str, Path]:
    if (run_root / "results_stacking.xlsx").exists():
        return {run_root.name: run_root}
    if (run_root / "final_stacking.xlsx").exists():
        return {run_root.parent.parent.name if run_root.parent.name.startswith("random_seed") else run_root.name: run_root}

    result_root = run_root / "04_results"
    dirs: Dict[str, Path] = {}
    for name in wanted:
        path = result_root / name
        if (path / "results_stacking.xlsx").exists():
            dirs[name] = path
        direct_book = run_root / name / "final_stacking.xlsx"
        if direct_book.exists():
            dirs[name] = run_root / name
            continue
        for rel in (
            "06_results",
            "03_results",
            "02_results",
            "01_results",
        ):
            book = run_root / name / rel / f"random_seed{seed}" / "final_stacking.xlsx"
            if book.exists():
                dirs[name] = book.parent
                break
    return dirs


def stacking_book_for(experiment_dir: Path) -> Path:
    for name in ("results_stacking.xlsx", "final_stacking.xlsx"):
        path = experiment_dir / name
        if path.exists():
            return path
    return experiment_dir / "results_stacking.xlsx"


def maxpool_zone_book_for(experiment_name: str, experiment_dir: Path) -> Optional[Path]:
    if experiment_name != "A_Patch":
        return None
    path = experiment_dir / "maxpool_zone.xlsx"
    return path if path.exists() else None


def stack_predictions_for_experiment(book: Path, experiment_name: str, experiment_dir: Path) -> pd.DataFrame:
    pred = excel_sheet(book, "meta_predictions")
    maxpool_book = maxpool_zone_book_for(experiment_name, experiment_dir)
    if maxpool_book is None:
        return pred
    zone = excel_sheet(maxpool_book, "predictions")
    if zone.empty:
        return pred
    zone = zone.copy()
    zone["task"] = "zone"
    zone["meta_method"] = "maxpool"
    columns = [c for c in ["task", "outer_fold", "case_id", "y_true", "y_score", "y_pred", "meta_method"] if c in zone.columns]
    zone = zone.reindex(columns=columns)
    if pred.empty:
        return zone
    pred = pred[pred["task"].astype(str) != "zone"].copy()
    return pd.concat([pred, zone], ignore_index=True, sort=False)


def stack_metrics_for_experiment(book: Path, experiment_name: str, experiment_dir: Path) -> pd.DataFrame:
    metrics = excel_sheet(book, "meta_fold_metrics")
    maxpool_book = maxpool_zone_book_for(experiment_name, experiment_dir)
    if maxpool_book is None:
        return metrics
    zone = excel_sheet(maxpool_book, "fold_metrics")
    if zone.empty:
        return metrics
    zone = zone.copy()
    zone["task"] = "zone"
    zone["meta_method"] = "maxpool"
    if metrics.empty:
        return zone
    metrics = metrics[metrics["task"].astype(str) != "zone"].copy()
    return pd.concat([metrics, zone], ignore_index=True, sort=False)


def add_cv_metric_box(
    ax: plt.Axes,
    work: pd.DataFrame,
    metrics: Sequence[Tuple[str, str]],
    x0: float = 2.50,
    y0: float = 36.5,
) -> None:
    short_labels = {
        "AUC": "AUC",
        "AP": "AP",
        "Bal. Acc.": "BAcc",
        "F1": "F1",
    }
    box_w = 1.25
    box_h = 22.0
    box = FancyBboxPatch(
        (x0, y0),
        box_w,
        box_h,
        boxstyle="round,pad=0.0,rounding_size=0.014",
        transform=ax.transData,
        facecolor="white",
        edgecolor="#bfbfbf",
        linewidth=0.65,
        alpha=0.98,
        zorder=6,
        clip_on=False,
    )
    ax.add_patch(box)

    ys = np.linspace(y0 + box_h - 3.6, y0 + 3.4, len(metrics))
    for idx, ((metric, label), y) in enumerate(zip(metrics, ys)):
        vals = pd.to_numeric(work[metric], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size:
            mean = float(np.mean(vals)) * 100.0
            std = float(np.std(vals, ddof=0)) * 100.0
            text = f"{short_labels.get(label, label)}: {mean:.2f} ± {std:.2f}"
        else:
            text = f"{short_labels.get(label, label)}: n/a"
        color = PAPER_COLORS[idx % len(PAPER_COLORS)]
        ax.plot(
            [x0 + 0.035, x0 + 0.125],
            [y, y],
            color=color,
            linewidth=1.25,
            zorder=7,
            clip_on=False,
        )
        ax.plot(
            [x0 + 0.080],
            [y],
            marker="o",
            markersize=3.2,
            markerfacecolor=color,
            markeredgecolor=color,
            linestyle="None",
            zorder=8,
            clip_on=False,
        )
        ax.text(
            x0 + 0.148,
            y,
            text,
            ha="left",
            va="center",
            color="#000000",
            fontsize=7.7,
            fontfamily="DejaVu Serif",
            fontweight="normal",
            zorder=8,
            clip_on=False,
        )


def make_metric_stability_figure(df: pd.DataFrame, out_dir: Path, dpi: int, filename: str) -> List[Path]:
    if df.empty:
        return []
    if "outer_fold" not in df.columns:
        return []

    metrics = [
        ("auc", "AUC"),
        ("ap", "AP"),
        ("balanced_acc", "Bal. Acc."),
        ("f1", "F1"),
    ]
    metrics = [(c, lab) for c, lab in metrics if c in df.columns]
    if not metrics:
        return []

    tasks = [t for t in TASKS if t in set(df["task"].astype(str))]
    if not tasks:
        tasks = sorted(df["task"].dropna().astype(str).unique().tolist())

    fig, axes = plt.subplots(1, len(tasks), figsize=(3.8 * len(tasks), 2.85), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        work = df[df["task"].astype(str) == task].copy()
        work = work.sort_values("outer_fold")
        for i, (metric, label) in enumerate(metrics):
            ax.plot(
                work["outer_fold"],
                pd.to_numeric(work[metric], errors="coerce") * 100.0,
                marker="o",
                linewidth=2.1,
                markersize=5,
                color=PAPER_COLORS[i % len(PAPER_COLORS)],
                label=label,
            )
        ax.set_title(f"{TASK_LABELS.get(task, task)} CV Stability")
        ax.set_xlabel("Outer fold")
        ax.set_ylabel("Score (%)")
        ax.set_xlim(0.9, 4.12)
        ax.set_ylim(35, 92)
        ax.set_yticks(np.arange(40, 91, 10))
        guides = (40, 50, 60, 70) if filename == "external_cv_stability.png" and task == "zone" else (60, 70, 80, 90)
        for guide in guides:
            ax.axhline(guide, color="#778899", linestyle="--", linewidth=0.9, alpha=0.62, zorder=1)
        ax.set_xticks(sorted(work["outer_fold"].dropna().astype(int).unique()))
        ax.tick_params(axis="y", labelleft=True)
        legend_y = 70.4 if filename == "external_cv_stability.png" and task == "zone" else 36.5
        add_cv_metric_box(ax, work, metrics, y0=legend_y)
    fig.tight_layout()
    return [save_fig(fig, out_dir / filename, dpi)]


def make_metric_stability_plot(
    book: Path,
    out_dir: Path,
    dpi: int,
    experiment_name: str = "",
    experiment_dir: Optional[Path] = None,
) -> List[Path]:
    metrics = (
        stack_metrics_for_experiment(book, experiment_name, experiment_dir)
        if experiment_dir is not None
        else excel_sheet(book, "meta_fold_metrics")
    )
    return make_metric_stability_figure(
        metrics,
        out_dir,
        dpi,
        "stack_cv_stability.png",
    )


def make_sota_stability_plot(out_dir: Path, dpi: int) -> List[Path]:
    folds = np.array([1, 2, 3, 4], dtype=int)
    x_positions = 1.0 + 0.5 * (folds - 1)
    rows = [
        ("UNet", [0.8104, 0.7030, 0.8350, 0.8916], "0.810 ± 0.079"),
        ("nnUNet", [0.8250, 0.7387, 0.8794, 0.7649], "0.802 ± 0.063"),
        ("nnDetection", [0.8669, 0.7338, 0.8866, 0.8206], "0.827 ± 0.068"),
        ("UMamba ProSSL", [0.8739, 0.7916, 0.9277, 0.8628], "0.864 ± 0.056"),
        ("Ours", [0.8704, 0.7897, 0.8738, 0.8580], "0.848 ± 0.039"),
    ]

    fig, ax = plt.subplots(figsize=(4.6, 2.90))
    for idx, (method, aucs, summary) in enumerate(rows):
        mean, spread = summary.split(" ± ")
        ax.plot(
            x_positions,
            np.asarray(aucs, dtype=float),
            marker="o",
            linewidth=2.1,
            markersize=5,
            color=PAPER_COLORS[idx % len(PAPER_COLORS)],
            label=f"{method}\n${mean}\\ \mathbf{{\\pm\\ {spread}}}$",
        )

    yticks = np.arange(0.70, 0.951, 0.05)
    for guide in yticks:
        ax.axhline(guide, color="#778899", linestyle="--", linewidth=0.9, alpha=0.62, zorder=1)

    ax.set_xlabel("Outer fold")
    ax.set_ylabel("AUC")
    ax.set_xlim(0.86, 3.50)
    ax.set_ylim(0.69, 0.96)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(fold) for fold in folds])
    ax.set_yticks(yticks)
    ax.tick_params(axis="y", labelleft=True)
    for label in ax.get_xticklabels():
        label.set_fontfamily("DejaVu Serif")
        label.set_fontstyle("normal")
        label.set_fontweight("normal")
    leg = ax.legend(
        loc="center left",
        bbox_to_anchor=(2.607, 0.825),
        bbox_transform=ax.transData,
        frameon=True,
        borderpad=0.5,
        handlelength=0.85,
        labelspacing=1.2,
        handletextpad=0.42,
        fontsize=7.2,
    )
    if leg is not None:
        for text in leg.get_texts():
            text.set_linespacing(1.5)
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_alpha(0.88)
        frame.set_edgecolor("#e6e6e6")
        frame.set_linewidth(0.5)
    fig.tight_layout()
    return [save_fig(fig, out_dir / "sota_stability.png", dpi)]


def annotate_confusion(
    ax: plt.Axes,
    cm: np.ndarray,
    labels: Sequence[str],
    title: str,
    y_labels: Optional[Sequence[str]] = None,
) -> None:
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
    cmap = LinearSegmentedColormap.from_list("paper_blues", ["#f7fafc", "#1f5f9f"])
    ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_yticklabels(y_labels if y_labels is not None else labels)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.grid(False, which="both")
    ax.minorticks_off()
    lo, hi = -0.5, len(labels) - 0.5
    for boundary in np.arange(0.5, len(labels) - 0.5, 1):
        ax.plot([lo, hi], [boundary, boundary], color="#000000", linewidth=0.55, linestyle=(0, (4.5, 2.6)), alpha=0.9, zorder=3)
        ax.plot([boundary, boundary], [lo, hi], color="#000000", linewidth=0.55, linestyle=(0, (4.5, 2.6)), alpha=0.9, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(hi, lo)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = 100.0 * cm_pct[i, j]
            color = "white" if cm_pct[i, j] > 0.55 else "#1a202c"
            ax.text(j, i - 0.080, f"{cm[i, j]}", ha="center", va="center", color=color, fontsize=9.0)
            ax.text(j, i + 0.125, f"({pct:.0f}%)", ha="center", va="center", color=color, fontsize=8.2)


def task_prediction_frame(
    pred: pd.DataFrame,
    task: str,
    best_folds: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    work = pred[pred["task"].astype(str) == task].copy()
    if best_folds is None:
        return work
    if task not in best_folds or "outer_fold" not in work.columns:
        return pd.DataFrame()
    work["outer_fold"] = pd.to_numeric(work["outer_fold"], errors="coerce")
    return work[work["outer_fold"] == best_folds[task]].copy()


def positive_first_binary_labels(task: str) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    labels = BINARY_CLASS_LABELS.get(task, ("0", "1"))
    y_labels = BINARY_CLASS_Y_LABELS.get(task, labels)
    return (labels[1], labels[0]), (y_labels[1], y_labels[0])


def save_binary_confusion_figure(
    pred: pd.DataFrame,
    tasks: Sequence[str],
    out_dir: Path,
    dpi: int,
    filename: str,
    best_folds: Optional[Dict[str, int]] = None,
) -> Optional[Path]:
    matrices: List[Tuple[str, np.ndarray]] = []
    for task in tasks:
        work = task_prediction_frame(pred, task, best_folds=best_folds)
        if work.empty or not {"y_true", "y_pred"}.issubset(work.columns):
            continue
        y_true = pd.to_numeric(work["y_true"], errors="coerce")
        y_pred = pd.to_numeric(work["y_pred"], errors="coerce")
        valid = y_true.notna() & y_pred.notna()
        if not valid.any():
            continue
        cm = confusion_matrix(
            y_true[valid].astype(int),
            y_pred[valid].astype(int),
            labels=BINARY_CONFUSION_ORDER,
        )
        matrices.append((task, cm))

    if not matrices:
        return None

    fig, axes = plt.subplots(1, len(matrices), figsize=(3.2 * len(matrices), 3.0))
    if len(matrices) == 1:
        axes = [axes]
    for ax, (task, cm) in zip(axes, matrices):
        x_labels, y_labels = positive_first_binary_labels(task)
        annotate_confusion(
            ax,
            cm,
            x_labels,
            f"{TASK_LABELS.get(task, task)} Confusion",
            y_labels=y_labels,
        )
    fig.tight_layout()
    return save_fig(fig, out_dir / filename, dpi)


def save_3class_confusion_figure(
    comb: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    filename: str,
    best_fold: Optional[int] = None,
) -> Optional[Path]:
    work = comb.copy()
    if best_fold is not None:
        if "outer_fold" not in work.columns:
            return None
        work["outer_fold"] = pd.to_numeric(work["outer_fold"], errors="coerce")
        work = work[work["outer_fold"] == best_fold].copy()
    if work.empty or not {"y3_true", "y3_pred"}.issubset(work.columns):
        return None

    y_true = work["y3_true"].map(normalize_3class)
    y_pred = work["y3_pred"].map(normalize_3class)
    valid = y_true.notna() & y_pred.notna()
    if not valid.any():
        return None

    cm = confusion_matrix(y_true[valid], y_pred[valid], labels=list(CLASS_ORDER_3))
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    annotate_confusion(ax, cm, CLASS_LABELS_3, "3-Class Confusion", y_labels=CLASS_Y_LABELS_3)
    fig.tight_layout()
    return save_fig(fig, out_dir / filename, dpi)


def make_confusion_plots(
    book: Path,
    out_dir: Path,
    dpi: int,
    experiment_name: str = "",
    experiment_dir: Optional[Path] = None,
) -> List[Path]:
    paths: List[Path] = []
    pred = (
        stack_predictions_for_experiment(book, experiment_name, experiment_dir)
        if experiment_dir is not None
        else excel_sheet(book, "meta_predictions")
    )
    metrics = (
        stack_metrics_for_experiment(book, experiment_name, experiment_dir)
        if experiment_dir is not None
        else excel_sheet(book, "meta_fold_metrics")
    )
    best_folds = best_auc_outer_folds(metrics)
    if not pred.empty and {"task", "y_true", "y_pred"}.issubset(pred.columns):
        tasks = [t for t in TASKS if t in set(pred["task"].astype(str))]
        if tasks:
            pooled = save_binary_confusion_figure(pred, tasks, out_dir, dpi, "stack_confusion_binary_pooled.png")
            if pooled is not None:
                paths.append(pooled)
            best = save_binary_confusion_figure(
                pred,
                tasks,
                out_dir,
                dpi,
                "stack_confusion_binary_best.png",
                best_folds=best_folds,
            )
            if best is not None:
                paths.append(best)

    comb = excel_sheet(book, "combined_3class")
    if not comb.empty and {"y3_true", "y3_pred"}.issubset(comb.columns):
        pooled_3 = save_3class_confusion_figure(comb, out_dir, dpi, "stack_confusion_3class_pooled.png")
        if pooled_3 is not None:
            paths.append(pooled_3)
        best_3 = save_3class_confusion_figure(
            comb,
            out_dir,
            dpi,
            "stack_confusion_3class_best.png",
            best_fold=best_mean_auc_outer_fold(metrics),
        )
        if best_3 is not None:
            paths.append(best_3)

    return paths


def best_auc_outer_folds(metrics: pd.DataFrame, auc_col: str = "auc") -> Dict[str, int]:
    required = {"task", "outer_fold", auc_col}
    if metrics.empty or not required.issubset(metrics.columns):
        return {}

    work = metrics.copy()
    work["task"] = work["task"].astype(str)
    work["outer_fold"] = pd.to_numeric(work["outer_fold"], errors="coerce")
    work[auc_col] = pd.to_numeric(work[auc_col], errors="coerce")
    work = work.dropna(subset=["task", "outer_fold", auc_col])
    if work.empty:
        return {}

    work = work.sort_values(["task", auc_col, "outer_fold"], ascending=[True, False, True])
    best = work.groupby("task", sort=False).head(1)
    return {str(row["task"]): int(row["outer_fold"]) for _, row in best.iterrows()}


def best_mean_auc_outer_fold(metrics: pd.DataFrame, auc_col: str = "auc") -> Optional[int]:
    required = {"outer_fold", auc_col}
    if metrics.empty or not required.issubset(metrics.columns):
        return None

    work = metrics.copy()
    work["outer_fold"] = pd.to_numeric(work["outer_fold"], errors="coerce")
    work[auc_col] = pd.to_numeric(work[auc_col], errors="coerce")
    work = work.dropna(subset=["outer_fold", auc_col])
    if work.empty:
        return None

    fold_scores = work.groupby("outer_fold", as_index=False)[auc_col].mean()
    fold_scores = fold_scores.sort_values([auc_col, "outer_fold"], ascending=[False, True])
    return int(fold_scores.iloc[0]["outer_fold"])


def roc_pr_curve_for_task(pred: pd.DataFrame, task: str, best_fold: Optional[int]) -> Optional[RocPrCurve]:
    work = pred[pred["task"].astype(str) == task].copy()
    if best_fold is not None and "outer_fold" in work.columns:
        work["outer_fold"] = pd.to_numeric(work["outer_fold"], errors="coerce")
        work = work[work["outer_fold"] == best_fold]

    y_true = pd.to_numeric(work["y_true"], errors="coerce")
    y_score = pd.to_numeric(work["y_score"], errors="coerce")
    valid = y_true.notna() & y_score.notna()
    y_true = y_true[valid].astype(int).to_numpy()
    y_score = y_score[valid].astype(float).to_numpy()
    if len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return RocPrCurve(
        task=task,
        best_fold=best_fold,
        fpr=fpr,
        tpr=tpr,
        precision=precision,
        recall=recall,
        roc_auc=roc_auc_score(y_true, y_score),
        pr_auc=average_precision_score(y_true, y_score),
        prevalence=float(np.mean(y_true)),
    )


def draw_roc_axis(ax: plt.Axes, curve: RocPrCurve) -> None:
    title_suffix = f" Outer Fold {curve.best_fold}" if curve.best_fold is not None else ""
    ax.plot(curve.fpr, curve.tpr, color=PAPER_COLORS[0], lw=2.4, label=f"AUC {curve.roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#8fa1b7", lw=1.2, linestyle="--")
    ax.set_title(f"{TASK_LABELS.get(curve.task, curve.task)} ROC{title_suffix}")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    axis_legend(ax, loc="lower right")


def draw_pr_axis(ax: plt.Axes, curve: RocPrCurve) -> None:
    title_suffix = f" Outer Fold {curve.best_fold}" if curve.best_fold is not None else ""
    ax.plot(curve.recall, curve.precision, color=PAPER_COLORS[1], lw=2.4, label=f"AP {curve.pr_auc:.3f}")
    ax.hlines(curve.prevalence, 0, 1, color="#8fa1b7", lw=1.2, linestyle="--", label=f"Prev. {curve.prevalence:.2f}")
    ax.set_title(f"{TASK_LABELS.get(curve.task, curve.task)} PR{title_suffix}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    axis_legend(ax, loc="lower left")


def save_roc_pr_plot_set(
    pred: pd.DataFrame,
    tasks: Sequence[str],
    out_dir: Path,
    dpi: int,
    prefix: str,
    variant: str,
    best_folds: Optional[Dict[str, int]] = None,
) -> List[Path]:
    paths: List[Path] = []
    curves: Dict[str, RocPrCurve] = {}
    for task in tasks:
        if best_folds is not None and task not in best_folds:
            continue
        best_fold = best_folds.get(task) if best_folds is not None else None
        curve = roc_pr_curve_for_task(pred, task, best_fold)
        if curve is None:
            continue
        curves[task] = curve

    plot_tasks = [task for task in TASKS if task in curves]
    if not plot_tasks:
        return paths

    fig, axes = plt.subplots(
        1,
        len(plot_tasks),
        figsize=(3.45 * len(plot_tasks), 2.85),
        sharex=True,
        sharey=True,
    )
    if len(plot_tasks) == 1:
        axes = [axes]
    for ax, task in zip(axes, plot_tasks):
        draw_roc_axis(ax, curves[task])
    fig.tight_layout()
    paths.append(save_fig(fig, out_dir / f"{prefix}_roc_combined_{variant}.png", dpi))

    fig, axes = plt.subplots(
        1,
        len(plot_tasks),
        figsize=(3.45 * len(plot_tasks), 2.85),
        sharex=True,
        sharey=True,
    )
    if len(plot_tasks) == 1:
        axes = [axes]
    for ax, task in zip(axes, plot_tasks):
        draw_pr_axis(ax, curves[task])
    fig.tight_layout()
    paths.append(save_fig(fig, out_dir / f"{prefix}_ap_combined_{variant}.png", dpi))

    if all(task in curves for task in ("cs", "zone")):
        fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.65))
        draw_roc_axis(axes[0, 0], curves["cs"])
        draw_roc_axis(axes[0, 1], curves["zone"])
        draw_pr_axis(axes[1, 0], curves["cs"])
        draw_pr_axis(axes[1, 1], curves["zone"])
        fig.tight_layout()
        paths.append(save_fig(fig, out_dir / f"{prefix}_roc_pr_combined_{variant}.png", dpi))
    return paths


def make_roc_pr_plots(
    book: Path,
    out_dir: Path,
    dpi: int,
    experiment_name: str = "",
    experiment_dir: Optional[Path] = None,
) -> List[Path]:
    pred = (
        stack_predictions_for_experiment(book, experiment_name, experiment_dir)
        if experiment_dir is not None
        else excel_sheet(book, "meta_predictions")
    )
    if pred.empty or not {"task", "y_true", "y_score"}.issubset(pred.columns):
        return []

    tasks = [t for t in TASKS if t in set(pred["task"].astype(str))]
    if not tasks:
        return []

    paths: List[Path] = []
    paths.extend(save_roc_pr_plot_set(pred, tasks, out_dir, dpi, "stack", "pooled"))
    metrics = (
        stack_metrics_for_experiment(book, experiment_name, experiment_dir)
        if experiment_dir is not None
        else excel_sheet(book, "meta_fold_metrics")
    )
    best_folds = best_auc_outer_folds(metrics)
    paths.extend(save_roc_pr_plot_set(pred, tasks, out_dir, dpi, "stack", "best", best_folds=best_folds))
    return paths


def draw_score_distribution_axis(ax: plt.Axes, work: pd.DataFrame, task: str) -> None:
    groups = [work.loc[work["y_true"] == lab, "y_score"].to_numpy(dtype=float) for lab in (0, 1)]
    positions = [0, 1]
    bp = ax.boxplot(
        groups,
        positions=positions,
        widths=box_body_width,
        patch_artist=True,
        showfliers=False,
        boxprops={"linewidth": 1.3, "edgecolor": "#1a202c"},
        medianprops={"linewidth": 1.7, "color": "#111111"},
        whiskerprops={"linewidth": 1.2, "color": "#1a202c"},
        capprops={"linewidth": 1.2, "color": "#1a202c"},
    )
    for patch, color in zip(bp["boxes"], [PAPER_COLORS[5], PAPER_COLORS[0]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.34)
        patch.set_edgecolor("#1a202c")
    for i, values in enumerate(groups):
        if len(values) == 0:
            continue
        rng = np.random.default_rng(1234 + i)
        jitter = rng.uniform(-0.13, 0.13, len(values))
        ax.scatter(
            np.full(len(values), positions[i]) + jitter,
            values,
            s=20,
            alpha=0.68,
            color=[PAPER_COLORS[5], PAPER_COLORS[0]][i],
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
        )
    ax.set_title(f"{TASK_LABELS.get(task, task)} Scores")
    ax.set_xticks(positions)
    ax.set_xticklabels(BINARY_CLASS_LABELS.get(task, ("0", "1")))
    ax.set_ylabel("Predicted score")
    ax.tick_params(axis="y", labelleft=True)
    ax.set_ylim(-0.02, 1.02)


def save_score_distribution_figure(
    pred: pd.DataFrame,
    tasks: Sequence[str],
    out_dir: Path,
    dpi: int,
    filename: str,
    best_folds: Optional[Dict[str, int]] = None,
) -> Optional[Path]:
    task_frames: Dict[str, pd.DataFrame] = {}
    for task in tasks:
        work = pred[pred["task"].astype(str) == task].copy()
        if best_folds is not None:
            if task not in best_folds or "outer_fold" not in work.columns:
                continue
            work["outer_fold"] = pd.to_numeric(work["outer_fold"], errors="coerce")
            work = work[work["outer_fold"] == best_folds[task]]
        work["y_true"] = pd.to_numeric(work["y_true"], errors="coerce")
        work["y_score"] = pd.to_numeric(work["y_score"], errors="coerce")
        work = work.dropna(subset=["y_true", "y_score"])
        if not work.empty:
            task_frames[task] = work

    plot_tasks = [task for task in tasks if task in task_frames]
    if not plot_tasks:
        return None

    fig, axes = plt.subplots(1, len(plot_tasks), figsize=(3.7 * len(plot_tasks), 2.9), sharey=True)
    if len(plot_tasks) == 1:
        axes = [axes]
    for ax, task in zip(axes, plot_tasks):
        draw_score_distribution_axis(ax, task_frames[task], task)
    fig.tight_layout()
    return save_fig(fig, out_dir / filename, dpi)


def make_score_distribution_plot(
    book: Path,
    out_dir: Path,
    dpi: int,
    experiment_name: str = "",
    experiment_dir: Optional[Path] = None,
) -> List[Path]:
    pred = (
        stack_predictions_for_experiment(book, experiment_name, experiment_dir)
        if experiment_dir is not None
        else excel_sheet(book, "meta_predictions")
    )
    if pred.empty or not {"task", "y_true", "y_score"}.issubset(pred.columns):
        return []
    tasks = [t for t in TASKS if t in set(pred["task"].astype(str))]
    if not tasks:
        return []

    paths: List[Path] = []
    pooled = save_score_distribution_figure(pred, tasks, out_dir, dpi, "stack_score_distributions_pooled.png")
    if pooled is not None:
        paths.append(pooled)

    metrics = (
        stack_metrics_for_experiment(book, experiment_name, experiment_dir)
        if experiment_dir is not None
        else excel_sheet(book, "meta_fold_metrics")
    )
    best_folds = best_auc_outer_folds(metrics)
    best_auc_fold = save_score_distribution_figure(
        pred,
        tasks,
        out_dir,
        dpi,
        "stack_score_distributions_best.png",
        best_folds=best_folds,
    )
    if best_auc_fold is not None:
        paths.append(best_auc_fold)
    return paths


def parse_weight_json(raw: object) -> Dict[str, float]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return {}
    if not str(raw).strip():
        return {}
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return {}
    out: Dict[str, float] = {}
    for key, value in parsed.items():
        try:
            out[str(key)] = float(value)
        except Exception:
            continue
    return out


def candidate_file_from_id(candidate_id: object) -> str:
    parts = str(candidate_id).split("||")
    if len(parts) >= 3:
        return clean_name(parts[2])
    return clean_name(candidate_id)


def make_feature_set_use_plot(
    book: Path,
    out_dir: Path,
    dpi: int,
    experiment_name: str = "",
    experiment_dir: Optional[Path] = None,
) -> List[Path]:
    selected = excel_sheet(book, "selected_candidates")
    fold = excel_sheet(book, "meta_fold_metrics")
    if selected.empty and fold.empty:
        return []

    use_rows = []
    if not selected.empty and {"task", "file"}.issubset(selected.columns):
        work = selected.copy()
        if "status" in work.columns:
            work = work[work["status"].fillna("used").astype(str).str.lower().eq("used")]
        for _, row in work.iterrows():
            use_rows.append(
                {
                    "task": str(row.get("task", "")),
                    "feature_set": clean_name(row.get("file", "")),
                    "selected_count": 1.0,
                    "weight": np.nan,
                }
            )

    if not fold.empty and {"task", "weights_json"}.issubset(fold.columns):
        for _, row in fold.iterrows():
            task = str(row.get("task", ""))
            weights = parse_weight_json(row.get("weights_json"))
            for candidate_id, weight in weights.items():
                use_rows.append(
                    {
                        "task": task,
                        "feature_set": candidate_file_from_id(candidate_id),
                        "selected_count": np.nan,
                        "weight": weight,
                    }
                )

    if not use_rows:
        return []
    use = pd.DataFrame(use_rows)
    tasks = [t for t in TASKS if t in set(use["task"].astype(str))]
    feature_order = sorted(use["feature_set"].dropna().astype(str).unique().tolist())
    if not tasks or not feature_order:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(7.6, max(2.8, 0.32 * len(feature_order) + 1.2)))

    count_pivot = (
        use.dropna(subset=["selected_count"])
        .pivot_table(index="feature_set", columns="task", values="selected_count", aggfunc="sum", fill_value=0)
        .reindex(feature_order)
    )
    if count_pivot.empty:
        axes[0].axis("off")
    else:
        left = np.zeros(len(count_pivot))
        y = np.arange(len(count_pivot))
        for i, task in enumerate(tasks):
            vals = count_pivot.get(task, pd.Series(0, index=count_pivot.index)).to_numpy(dtype=float)
            axes[0].barh(
                y,
                vals,
                left=left,
                height=0.62,
                color=PAPER_COLORS[i],
                alpha=0.94,
                edgecolor="white",
                linewidth=0.4,
                label=TASK_LABELS.get(task, task),
            )
            left += vals
        axes[0].set_yticks(y)
        axes[0].set_yticklabels([clean_name(v) for v in count_pivot.index])
        axes[0].invert_yaxis()
        axes[0].set_title("Selected Base Sets")
        axes[0].set_xlabel("Count")

    weight_pivot = (
        use.dropna(subset=["weight"])
        .pivot_table(index="feature_set", columns="task", values="weight", aggfunc="sum", fill_value=0)
        .reindex(feature_order)
    )
    if weight_pivot.empty:
        axes[1].axis("off")
        axes[1].set_title("Stack Weights")
    else:
        left = np.zeros(len(weight_pivot))
        y = np.arange(len(weight_pivot))
        for i, task in enumerate(tasks):
            vals = weight_pivot.get(task, pd.Series(0, index=weight_pivot.index)).to_numpy(dtype=float)
            axes[1].barh(
                y,
                vals,
                left=left,
                height=0.62,
                color=PAPER_COLORS[i],
                alpha=0.94,
                edgecolor="white",
                linewidth=0.4,
                label=TASK_LABELS.get(task, task),
            )
            left += vals
        axes[1].set_yticks(y)
        axes[1].set_yticklabels([clean_name(v) for v in weight_pivot.index])
        axes[1].invert_yaxis()
        axes[1].set_title("Weighted Stack Share")
        axes[1].set_xlabel("Summed weight")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        add_top_legend(fig, handles, labels, ncol=max(1, len(labels)))
    fig.tight_layout()
    return [save_fig(fig, out_dir / "stack_feature_set_use.png", dpi)]


def make_feature_reduction_plot(
    book: Path,
    out_dir: Path,
    dpi: int,
    experiment_name: str = "",
    experiment_dir: Optional[Path] = None,
) -> List[Path]:
    df = excel_sheet(book, "selected_candidates")
    if df.empty or "task" not in df.columns:
        return []

    stage_cols = [
        ("n_input_features", "Input"),
        ("n_after_quasi", "Quasi"),
        ("n_after_corr", "Corr."),
        ("n_selected", "Selected"),
        ("pca_components", "PCA"),
    ]
    stage_cols = [(c, lab) for c, lab in stage_cols if c in df.columns]
    if len(stage_cols) < 3:
        return []

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for i, task in enumerate([t for t in TASKS if t in set(df["task"].astype(str))]):
        work = df[df["task"].astype(str) == task].copy()
        vals = []
        for col, _ in stage_cols:
            vals.append(pd.to_numeric(work[col], errors="coerce").median())
        ax.plot(
            np.arange(len(stage_cols)),
            vals,
            marker="o",
            lw=2.3,
            markersize=5.5,
            color=PAPER_COLORS[i],
            label=TASK_LABELS.get(task, task),
        )
    ax.set_xticks(np.arange(len(stage_cols)))
    ax.set_xticklabels([lab for _, lab in stage_cols])
    ax.set_yscale("log")
    ax.set_ylabel("Median features")
    ax.set_title("Feature Reduction")
    handles, labels = collect_legend_items([ax])
    add_top_legend(fig, handles, labels, ncol=len(labels))
    fig.tight_layout()
    return [save_fig(fig, out_dir / "stack_feature_reduction.png", dpi)]


def result_feature_workbooks(experiment_dir: Path, task: str) -> List[Path]:
    return sorted(experiment_dir.glob(f"feature--*/results_{task}_feature--*.xlsx"))


def load_selected_feature_ranks(experiment_dir: Path, task: str) -> pd.DataFrame:
    rows = []
    for path in result_feature_workbooks(experiment_dir, task):
        sheets = available_sheets(path)
        file_label = clean_name(path.parent.name)
        if "agg_feat_rank" in sheets:
            df = pd.read_excel(path, sheet_name="agg_feat_rank")
            if df.empty or "feature" not in df.columns:
                continue
            for _, row in df.iterrows():
                freq = pd.to_numeric(pd.Series([row.get("freq", 1)]), errors="coerce").iloc[0]
                rank = pd.to_numeric(pd.Series([row.get("mean_rank", np.nan)]), errors="coerce").iloc[0]
                score = pd.to_numeric(pd.Series([row.get("mean_score", np.nan)]), errors="coerce").iloc[0]
                rows.append(
                    {
                        "feature_set": file_label,
                        "feature": str(row["feature"]),
                        "freq": float(freq) if not pd.isna(freq) else 1.0,
                        "mean_rank": float(rank) if not pd.isna(rank) else np.nan,
                        "mean_score": float(score) if not pd.isna(score) else np.nan,
                    }
                )
        elif "selected_features" in sheets:
            df = pd.read_excel(path, sheet_name="selected_features")
            if df.empty or "feature" not in df.columns:
                continue
            for _, row in df.iterrows():
                rows.append(
                    {
                        "feature_set": file_label,
                        "feature": str(row["feature"]),
                        "freq": 1.0,
                        "mean_rank": pd.to_numeric(pd.Series([row.get("rank", np.nan)]), errors="coerce").iloc[0],
                        "mean_score": pd.to_numeric(pd.Series([row.get("score", np.nan)]), errors="coerce").iloc[0],
                    }
                )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["family"] = df["feature"].map(feature_family)
    return df


def make_selected_feature_rank_plots(experiment_dir: Path, out_dir: Path, dpi: int, top_n: int) -> List[Path]:
    paths: List[Path] = []
    for task in TASKS:
        df = load_selected_feature_ranks(experiment_dir, task)
        if df.empty:
            continue

        grouped = (
            df.groupby(["feature", "family"], as_index=False)
            .agg(freq=("freq", "sum"), mean_rank=("mean_rank", "mean"), mean_score=("mean_score", "mean"))
            .sort_values(["freq", "mean_rank"], ascending=[False, True])
            .head(top_n)
        )
        if not grouped.empty:
            fig, ax = plt.subplots(figsize=(7.2, max(3.4, 0.33 * len(grouped) + 0.9)))
            y = np.arange(len(grouped))
            colors = [PAPER_COLORS[i % len(PAPER_COLORS)] for i in range(len(grouped))]
            ax.barh(y, grouped["freq"], color=colors, alpha=0.94, edgecolor="white", linewidth=0.35)
            ax.set_yticks(y)
            ax.set_yticklabels([compact_feature_name(v) for v in grouped["feature"]])
            ax.invert_yaxis()
            ax.set_xlabel("Selection frequency")
            ax.set_title(f"{TASK_LABELS.get(task, task)} Selected Features")
            fig.tight_layout()
            paths.append(save_fig(fig, out_dir / f"selected_features_top_{task}.png", dpi))

        fam = (
            df.pivot_table(index="family", columns="feature_set", values="freq", aggfunc="sum", fill_value=0)
            .sort_index()
        )
        if not fam.empty:
            fig, ax = plt.subplots(figsize=(max(5.8, 0.55 * fam.shape[1] + 2.4), max(3.1, 0.38 * fam.shape[0] + 1.2)))
            im = ax.imshow(fam.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")
            ax.set_xticks(np.arange(fam.shape[1]))
            ax.set_xticklabels([clean_name(c) for c in fam.columns], rotation=35, ha="right")
            ax.set_yticks(np.arange(fam.shape[0]))
            ax.set_yticklabels(fam.index)
            ax.set_title(f"{TASK_LABELS.get(task, task)} Feature Families")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Selection frequency")
            ax.grid(False)
            fig.tight_layout()
            paths.append(save_fig(fig, out_dir / f"selected_feature_families_{task}.png", dpi))
    return paths


def final_feature_workbooks(run_root: Path, experiment_name: str) -> List[Path]:
    return sorted((run_root / "03_final_features" / experiment_name).glob("feature--*.xlsx"))


def choose_feature_sheet(path: Path) -> Optional[str]:
    sheets = available_sheets(path)
    if "all" in sheets:
        return "all"
    return sheets[0] if sheets else None


def task_labels_from_feature_df(df: pd.DataFrame, task: str) -> pd.Series:
    if task == "cs":
        if "clin_sig" not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        return df["clin_sig"].map(parse_bool_label)
    if task == "zone":
        if "clin_sig" not in df.columns or "binary" not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        clin = df["clin_sig"].map(parse_bool_label)
        zone = df["binary"].map(parse_zone_label)
        labels = zone.where(clin == 1)
        return labels
    return pd.Series(index=df.index, dtype=float)


def standardized_mean_difference(values: pd.Series, labels: pd.Series) -> float:
    work = pd.DataFrame({"x": pd.to_numeric(values, errors="coerce"), "y": labels}).dropna()
    if work["y"].nunique() < 2:
        return np.nan
    x0 = work.loc[work["y"] == 0, "x"].astype(float)
    x1 = work.loc[work["y"] == 1, "x"].astype(float)
    if len(x0) < 3 or len(x1) < 3:
        return np.nan
    var0 = float(x0.var(ddof=1))
    var1 = float(x1.var(ddof=1))
    pooled = math.sqrt(max((var0 + var1) / 2.0, 0.0))
    if pooled <= 0:
        return np.nan
    return float((x1.mean() - x0.mean()) / pooled)


def feature_effect_rows(run_root: Path, experiment_name: str, task: str, max_files: Optional[int] = None) -> pd.DataFrame:
    rows = []
    paths = final_feature_workbooks(run_root, experiment_name)
    if max_files is not None:
        paths = paths[:max_files]
    for path in paths:
        sheet = choose_feature_sheet(path)
        if sheet is None:
            continue
        try:
            df = pd.read_excel(path, sheet_name=sheet)
        except Exception:
            continue
        labels = task_labels_from_feature_df(df, task)
        if labels.dropna().nunique() < 2:
            continue

        numeric_cols = []
        for col in df.columns:
            if str(col) in NON_FEATURE_COLS:
                continue
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() >= 8 and converted.nunique(dropna=True) > 1:
                numeric_cols.append(col)

        for col in numeric_cols:
            smd = standardized_mean_difference(df[col], labels)
            if pd.isna(smd):
                continue
            rows.append(
                {
                    "feature_set": clean_name(path.name),
                    "feature": str(col),
                    "family": feature_family(col),
                    "smd": smd,
                    "abs_smd": abs(smd),
                }
            )
    return pd.DataFrame(rows)


def make_univariate_feature_effect_plots(
    run_root: Path,
    experiment_name: str,
    out_dir: Path,
    dpi: int,
    top_n: int,
) -> List[Path]:
    paths: List[Path] = []
    for task in TASKS:
        effects = feature_effect_rows(run_root, experiment_name, task)
        if effects.empty:
            continue

        top = effects.sort_values("abs_smd", ascending=False).head(top_n).copy()
        if not top.empty:
            fig, ax = plt.subplots(figsize=(7.2, max(3.4, 0.33 * len(top) + 0.9)))
            y = np.arange(len(top))
            colors = [PAPER_COLORS[0] if v >= 0 else PAPER_COLORS[2] for v in top["smd"]]
            ax.barh(y, top["smd"], color=colors, alpha=0.94, edgecolor="white", linewidth=0.35)
            ax.axvline(0, color="#111111", lw=1.0)
            ax.set_yticks(y)
            labels = [f"{compact_feature_name(f)} ({clean_name(s)})" for f, s in zip(top["feature"], top["feature_set"])]
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlabel("Standardized mean difference")
            ax.set_title(f"{TASK_LABELS.get(task, task)} Feature Effect")
            fig.tight_layout()
            paths.append(save_fig(fig, out_dir / f"feature_effect_top_{task}.png", dpi))

        fam_rows = []
        for (feature_set, family), grp in effects.groupby(["feature_set", "family"]):
            vals = grp["abs_smd"].sort_values(ascending=False).head(min(10, len(grp)))
            fam_rows.append({"feature_set": feature_set, "family": family, "top_abs_smd": vals.mean()})
        fam = pd.DataFrame(fam_rows)
        if not fam.empty:
            heat = fam.pivot_table(index="family", columns="feature_set", values="top_abs_smd", aggfunc="mean")
            heat = heat.loc[heat.max(axis=1).sort_values(ascending=False).index]
            fig, ax = plt.subplots(figsize=(max(5.8, 0.55 * heat.shape[1] + 2.4), max(3.1, 0.38 * heat.shape[0] + 1.2)))
            im = ax.imshow(heat.to_numpy(dtype=float), cmap="PuBuGn", aspect="auto")
            ax.set_xticks(np.arange(heat.shape[1]))
            ax.set_xticklabels([clean_name(c) for c in heat.columns], rotation=35, ha="right")
            ax.set_yticks(np.arange(heat.shape[0]))
            ax.set_yticklabels(heat.index)
            ax.set_title(f"{TASK_LABELS.get(task, task)} Effect by Family")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Top mean |SMD|")
            ax.grid(False)
            fig.tight_layout()
            paths.append(save_fig(fig, out_dir / f"feature_effect_family_heatmap_{task}.png", dpi))
    return paths


def external_book_for(run_root: Path, experiment_name: str, seed: int) -> Path:
    old_path = (
        run_root
        / "05_external_P158"
        / "04_external_results"
        / experiment_name
        / "external_results_from_saved_stacking_models.xlsx"
    )
    if old_path.exists():
        return old_path
    matches = sorted(
        (run_root / experiment_name).glob(
            f"*external_testing/02_results/ext_results_seed{seed}.xlsx"
        )
    )
    if matches:
        return matches[0]
    return old_path


def external_predictions_for_book(book: Path) -> pd.DataFrame:
    pred = excel_sheet(book, "all_8_model_predictions")
    if not pred.empty:
        return pred
    pred = excel_sheet(book, "stacking_predictions")
    zone = excel_sheet(book, "zone_predictions")
    if not zone.empty:
        pred = pd.concat([pred[pred["task"].astype(str) != "zone"], zone], ignore_index=True, sort=False)
    late = excel_sheet(book, "late_fusion_predictions")
    if not late.empty:
        metrics = excel_sheet(book, "late_fusion_metrics")
        if not metrics.empty and {"task", "alpha", "auc"}.issubset(metrics.columns):
            best = (
                metrics.assign(auc=pd.to_numeric(metrics["auc"], errors="coerce"))
                .dropna(subset=["auc"])
                .sort_values(["task", "auc"], ascending=[True, False])
                .drop_duplicates("task")
                .set_index("task")["alpha"]
                .to_dict()
            )
            late = late[
                late.apply(
                    lambda row: row.get("alpha") == best.get(str(row.get("task"))),
                    axis=1,
                )
            ].copy()
        pred = late
    return pred


def external_metrics_for_book(book: Path) -> pd.DataFrame:
    metrics = excel_sheet(book, "all_8_model_metrics")
    if not metrics.empty:
        return metrics
    metrics = excel_sheet(book, "metrics")
    late = excel_sheet(book, "late_fusion_metrics")
    if not late.empty:
        metrics = late
    return metrics


def make_external_plots(book: Path, out_dir: Path, dpi: int) -> List[Path]:
    if not book.exists():
        return []

    paths: List[Path] = []
    fold_metrics = external_metrics_for_book(book)
    paths.extend(make_metric_stability_figure(fold_metrics, out_dir, dpi, "external_cv_stability.png"))

    metrics = excel_sheet(book, "task_fold_average")
    if metrics.empty:
        metrics = external_metrics_for_book(book)
    if not metrics.empty and "task" in metrics.columns:
        metric_cols = [
            ("auc_mean", "AUC"),
            ("ap_mean", "AP"),
            ("balanced_acc_mean", "Bal. Acc."),
            ("f1_mean", "F1"),
            ("auc", "AUC"),
            ("ap", "AP"),
            ("balanced_acc", "Bal. Acc."),
            ("f1", "F1"),
        ]
        metric_cols = [(c, lab) for c, lab in metric_cols if c in metrics.columns]
        tasks = [t for t in TASKS if t in set(metrics["task"].astype(str))]
        if metric_cols and tasks:
            x = np.arange(len(metric_cols))
            width = external_bar_width
            fig, ax = plt.subplots(figsize=(5.2, 2.9))
            for i, task in enumerate(tasks):
                row = metrics[metrics["task"].astype(str) == task].iloc[0]
                vals = [float(row[col]) for col, _ in metric_cols]
                err = []
                for col, _ in metric_cols:
                    std_col = col.replace("_mean", "_std")
                    err.append(float(row[std_col]) if std_col in row and not pd.isna(row[std_col]) else 0.0)
                ax.bar(
                    x + (i - (len(tasks) - 1) / 2) * width,
                    vals,
                    yerr=err,
                    width=width,
                    color=PAPER_COLORS[i],
                    alpha=0.94,
                    edgecolor="white",
                    linewidth=0.4,
                    capsize=2.5,
                    error_kw={"elinewidth": 1.2, "capthick": 1.2},
                    label=TASK_LABELS.get(task, task),
                )
            ax.set_xticks(x)
            ax.set_xticklabels([lab for _, lab in metric_cols])
            ax.set_ylim(0, 1.02)
            ax.set_ylabel("External score")
            ax.set_title("External Validation")
            handles, labels = collect_legend_items([ax])
            add_top_legend(fig, handles, labels, ncol=len(labels))
            fig.tight_layout()
            paths.append(save_fig(fig, out_dir / "external_validation_metrics.png", dpi))

    pred = external_predictions_for_book(book)
    if not pred.empty and {"case_id", "task", "y_true", "y_score"}.issubset(pred.columns):
        avg = (
            pred.assign(y_score=pd.to_numeric(pred["y_score"], errors="coerce"))
            .dropna(subset=["y_score"])
            .groupby(["case_id", "task"], as_index=False)
            .agg(y_true=("y_true", "first"), y_score=("y_score", "mean"))
        )
        avg["y_pred"] = [
            int(score >= fixed_threshold(task))
            for score, task in zip(avg["y_score"], avg["task"].astype(str))
        ]
        tasks = [t for t in TASKS if t in set(avg["task"].astype(str))]
        if tasks:
            averaged_scores = save_score_distribution_figure(
                avg,
                tasks,
                out_dir,
                dpi,
                "external_score_distributions_pooled.png",
            )
            if averaged_scores is not None:
                paths.append(averaged_scores)

            pooled_cm = save_binary_confusion_figure(
                avg,
                tasks,
                out_dir,
                dpi,
                "external_confusion_binary_pooled.png",
            )
            if pooled_cm is not None:
                paths.append(pooled_cm)

            paths.extend(save_roc_pr_plot_set(avg, tasks, out_dir, dpi, "external", "pooled"))

            best_folds = best_auc_outer_folds(excel_sheet(book, "all_8_model_metrics"))
            best_fold_scores = save_score_distribution_figure(
                pred,
                tasks,
                out_dir,
                dpi,
                "external_score_distributions_best.png",
                best_folds=best_folds,
            )
            if best_fold_scores is not None:
                paths.append(best_fold_scores)

            best_cm = save_binary_confusion_figure(
                pred,
                tasks,
                out_dir,
                dpi,
                "external_confusion_binary_best.png",
                best_folds=best_folds,
            )
            if best_cm is not None:
                paths.append(best_cm)

            paths.extend(save_roc_pr_plot_set(pred, tasks, out_dir, dpi, "external", "best", best_folds=best_folds))
    return paths


def process_experiment(ctx: RunContext, experiment_name: str, experiment_dir: Path) -> Tuple[List[Path], List[str]]:
    out_dir = ctx.output_root / experiment_name
    safe_mkdir(out_dir)
    book = stacking_book_for(experiment_dir)
    made: List[Path] = []
    notes: List[str] = []

    makers = [
        make_metric_stability_plot,
        make_confusion_plots,
        make_roc_pr_plots,
        make_score_distribution_plot,
        make_feature_set_use_plot,
        make_feature_reduction_plot,
    ]
    for maker in makers:
        try:
            made.extend(maker(book, out_dir, ctx.dpi, experiment_name, experiment_dir))
        except Exception as exc:
            notes.append(f"{experiment_name}: skipped {maker.__name__}: {exc}")

    try:
        made.extend(make_selected_feature_rank_plots(experiment_dir, out_dir, ctx.dpi, ctx.top_n))
    except Exception as exc:
        notes.append(f"{experiment_name}: skipped selected feature rank plots: {exc}")

    if not ctx.skip_heavy:
        try:
            made.extend(make_univariate_feature_effect_plots(ctx.run_root, experiment_name, out_dir, ctx.dpi, ctx.top_n))
        except Exception as exc:
            notes.append(f"{experiment_name}: skipped feature effect plots: {exc}")

    return made, notes


def combined_prediction_frames(experiment_dirs: Dict[str, Path], variant: str) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for experiment_name, experiment_dir in experiment_dirs.items():
        book = stacking_book_for(experiment_dir)
        pred = stack_predictions_for_experiment(book, experiment_name, experiment_dir)
        if pred.empty or not {"task", "y_true", "y_score"}.issubset(pred.columns):
            continue
        if variant == "best":
            metrics = stack_metrics_for_experiment(book, experiment_name, experiment_dir)
            best_folds = best_auc_outer_folds(metrics)
            parts = []
            for task in TASKS:
                work = task_prediction_frame(pred, task, best_folds=best_folds)
                if not work.empty:
                    parts.append(work)
            pred = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
        if not pred.empty:
            frames[experiment_name] = pred
    return frames


def save_combined_roc_auc_plot(
    frames: Dict[str, pd.DataFrame],
    out_dir: Path,
    dpi: int,
    variant: str,
    source: str = "internal",
) -> Optional[Path]:
    if not frames:
        return None
    fig, axes = plt.subplots(1, len(TASKS), figsize=(3.8 * len(TASKS), 3.0), sharey=True)
    if len(TASKS) == 1:
        axes = [axes]
    made_any = False
    for ax, task in zip(axes, TASKS):
        for i, (experiment_name, pred) in enumerate(frames.items()):
            curve = roc_pr_curve_for_task(pred, task, None)
            if curve is None:
                continue
            made_any = True
            color = "#e53e3e" if variant == "pooled" and i == 0 else PAPER_COLORS[i % len(PAPER_COLORS)]
            ax.plot(
                curve.fpr,
                curve.tpr,
                color=color,
                lw=2.2,
                label=f"{experiment_label(experiment_name)} ({curve.roc_auc:.3f})",
            )
        ax.plot([0, 1], [0, 1], color="#8fa1b7", lw=1.1, linestyle="--")
        ax.set_title(f"{TASK_LABELS.get(task, task)} ROC")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax.tick_params(axis="y", labelleft=True)
        apply_combined_axis_typography(ax)
        axis_legend(ax, loc="lower right", handlelength=0.6, fontsize=combined_legend_fontsize)
    if not made_any:
        plt.close(fig)
        return None
    fig.tight_layout()
    return save_fig(fig, out_dir / f"{source}_auc_{variant}.png", dpi)


def save_combined_score_distribution_plot(
    frames: Dict[str, pd.DataFrame],
    out_dir: Path,
    dpi: int,
    variant: str,
    source: str = "internal",
) -> Optional[Path]:
    if not frames:
        return None

    fig, axes = plt.subplots(1, len(TASKS), figsize=(4.35 * len(TASKS), 3.0), sharey=True)
    if len(TASKS) == 1:
        axes = [axes]
    made_any = False

    for ax, task in zip(axes, TASKS):
        valid_frames: List[Tuple[str, pd.DataFrame]] = []
        for experiment_name, frame in frames.items():
            work = frame.copy()
            work = work[work["task"].astype(str) == task].copy()
            work["y_true"] = pd.to_numeric(work["y_true"], errors="coerce")
            work["y_score"] = pd.to_numeric(work["y_score"], errors="coerce")
            work = work.dropna(subset=["y_true", "y_score"])
            if work.empty:
                continue
            work["y_true"] = work["y_true"].astype(int)
            valid_frames.append((experiment_name, work))

        if not valid_frames:
            ax.set_visible(False)
            continue

        made_any = True
        class_labels = BINARY_CLASS_LABELS.get(task, ("0", "1"))
        class_colors = [PAPER_COLORS[5], PAPER_COLORS[0]]
        centers = np.arange(len(valid_frames), dtype=float) * 0.85
        offsets = (-0.18, 0.18)
        rng = np.random.default_rng(2026)

        for experiment_index, (_, work) in enumerate(valid_frames):
            for class_index, class_value in enumerate((0, 1)):
                values = work.loc[work["y_true"] == class_value, "y_score"].to_numpy(dtype=float)
                if values.size == 0:
                    continue

                position = centers[experiment_index] + offsets[class_index]
                boxplot = ax.boxplot(
                    [values],
                    positions=[position],
                    widths=0.26,
                    patch_artist=True,
                    showfliers=False,
                    boxprops={"linewidth": 1.15, "edgecolor": "#1a202c"},
                    medianprops={"linewidth": 1.55, "color": "#111111"},
                    whiskerprops={"linewidth": 1.05, "color": "#1a202c"},
                    capprops={"linewidth": 1.05, "color": "#1a202c"},
                )
                for patch in boxplot["boxes"]:
                    patch.set_facecolor(class_colors[class_index])
                    patch.set_alpha(0.34)
                    patch.set_edgecolor("#1a202c")

                jitter = rng.uniform(-0.075, 0.075, values.size)
                label = class_labels[class_value] if experiment_index == 0 else "_nolegend_"
                ax.scatter(
                    np.full(values.size, position) + jitter,
                    values,
                    s=17,
                    alpha=0.62,
                    color=class_colors[class_index],
                    edgecolors="white",
                    linewidths=0.25,
                    zorder=3,
                    label=label,
                )

        ax.set_xticks(centers)
        ax.set_xticklabels([experiment_axis_label(name) for name, _ in valid_frames], rotation=0, ha="center")
        ax.set_xlim(-0.5, centers[-1] + 0.6)
        for tick_label, (experiment_name, _) in zip(ax.get_xticklabels(), valid_frames):
            if experiment_name in {"B_Organ", "C_Early"}:
                tick_label.set_transform(
                    tick_label.get_transform() + ScaledTranslation(-7.0 / 72.0, 0.0, fig.dpi_scale_trans)
                )
            elif experiment_name == "D_Late":
                tick_label.set_transform(
                    tick_label.get_transform() + ScaledTranslation(4.0 / 72.0, 0.0, fig.dpi_scale_trans)
                )
        ax.set_ylabel("Predicted score")
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax.tick_params(axis="x", labelrotation=0)
        ax.tick_params(axis="y", labelleft=True)
        apply_combined_axis_typography(ax)

        handles, labels = collect_legend_items([ax])
        label_order = [class_labels[1], class_labels[0]]
        handle_map = {label: handle for handle, label in zip(handles, labels)}
        ordered_labels = [label for label in label_order if label in handle_map]
        ordered_handles = [handle_map[label] for label in ordered_labels]
        axis_legend(
            ax,
            loc="lower right",
            handles=ordered_handles,
            labels=ordered_labels,
            frame_alpha=1.0,
            borderpad=0.28,
            labelspacing=0.28,
            handletextpad=0.28,
            fontsize=combined_legend_fontsize,
        )

    if not made_any:
        plt.close(fig)
        return None
    fig.tight_layout()
    return save_fig(
        fig,
        out_dir / f"{source}_box_{variant}.png",
        dpi,
    )


def make_combined_internal_plots(ctx: RunContext, experiment_dirs: Dict[str, Path]) -> Tuple[List[Path], List[str]]:
    out_dir = ctx.output_root / "Combined ABCD"
    safe_mkdir(out_dir)
    made: List[Path] = []
    notes: List[str] = []
    for variant in ("pooled", "best"):
        try:
            frames = combined_prediction_frames(experiment_dirs, variant)
            for path in (
                save_combined_roc_auc_plot(frames, out_dir, ctx.dpi, variant),
                save_combined_score_distribution_plot(frames, out_dir, ctx.dpi, variant),
            ):
                if path is not None:
                    made.append(path)
        except Exception as exc:
            notes.append(f"combined_all_experiments: skipped {variant} combined internal plots: {exc}")
    return made, notes


def combined_external_prediction_frames(
    ctx: RunContext,
    experiment_dirs: Dict[str, Path],
) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for experiment_name in experiment_dirs:
        book = external_book_for(ctx.run_root, experiment_name, ctx.seed)
        if not book.exists():
            continue
        pred = external_predictions_for_book(book)
        required = {"case_id", "task", "y_true", "y_score"}
        if pred.empty or not required.issubset(pred.columns):
            continue
        pred = pred.copy()
        pred["y_score"] = pd.to_numeric(pred["y_score"], errors="coerce")
        pred = pred.dropna(subset=["y_score"])
        pred = (
            pred.groupby(["case_id", "task"], as_index=False)
            .agg(y_true=("y_true", "first"), y_score=("y_score", "mean"))
        )
        pred["y_pred"] = [
            int(score >= fixed_threshold(task))
            for score, task in zip(pred["y_score"], pred["task"].astype(str))
        ]
        if not pred.empty:
            frames[experiment_name] = pred
    return frames


def make_combined_external_plots(
    ctx: RunContext,
    experiment_dirs: Dict[str, Path],
) -> Tuple[List[Path], List[str]]:
    if ctx.no_external:
        return [], []
    out_dir = ctx.output_root / "Combined ABCD"
    safe_mkdir(out_dir)
    made: List[Path] = []
    notes: List[str] = []
    try:
        frames = combined_external_prediction_frames(ctx, experiment_dirs)
        for path in (
            save_combined_roc_auc_plot(frames, out_dir, ctx.dpi, "pooled", source="external"),
            save_combined_score_distribution_plot(
                frames,
                out_dir,
                ctx.dpi,
                "pooled",
                source="external",
            ),
        ):
            if path is not None:
                made.append(path)
    except Exception as exc:
        notes.append(f"combined_all_experiments: skipped combined external plots: {exc}")
    return made, notes


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate BMVC-style A_Patch and combined BSPC graphs for one random seed."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Run output folder or code folder.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Base graphs output folder.")
    parser.add_argument(
        "--experiments",
        default=",".join(EXPERIMENTS),
        help="Comma-separated experiment names to scan. Missing folders are skipped.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Figure DPI.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed to plot.")
    parser.add_argument("--all-seeds", action="store_true", help="Generate graphs for all seeds in RANDOM_SEEDS.")
    parser.add_argument("--top-n", type=int, default=15, help="Top features to show.")
    parser.add_argument("--add-heavy", action="store_true", help="Generate univariate feature-effect scans.")
    parser.add_argument("--no-external", action="store_true", help="Skip external P158 plots.")
    parser.add_argument("--add-captions", action="store_true", help="Include panel labels and figure captions.")
    parser.add_argument("--output-name", default=None, help="Override the final subfolder name under --output.")
    return parser.parse_args(argv)


def _run(args: argparse.Namespace, seed: int) -> int:
    global CAPTIONS_ENABLED
    CAPTIONS_ENABLED = bool(args.add_captions)
    input_path = resolve_input_path(args.input)
    run_root = resolve_run_root(input_path)
    config_name = str(args.output_name) if args.output_name else f"seed{seed}"
    output_base = resolve_input_path(args.output)
    output_root = output_base / config_name
    safe_mkdir(output_root)

    wanted = [item.strip() for item in str(args.experiments).split(",") if item.strip()]
    experiment_dirs = get_experiment_dirs(run_root, wanted, seed)
    if not experiment_dirs:
        print(f"No available stacking workbooks found under {run_root}", file=sys.stderr)
        return 2

    ctx = RunContext(
        input_path=input_path,
        run_root=run_root,
        output_root=output_root,
        dpi=int(args.dpi),
        top_n=int(args.top_n),
        skip_heavy=not bool(args.add_heavy),
        no_external=bool(args.no_external),
        captions=CAPTIONS_ENABLED,
        seed=seed,
    )

    all_notes: List[str] = []
    print(f"Input: {run_root}")
    print(f"Output: {output_root}")
    print(f"Experiments: {', '.join(experiment_dirs)}")

    try:
        made = make_sota_stability_plot(output_root, ctx.dpi)
        if made:
            print(f"sota_stability: wrote {len(made)} graph(s)")
    except Exception as exc:
        all_notes.append(f"sota_stability: skipped: {exc}")

    patch_dir = experiment_dirs.get("A_Patch")
    if patch_dir is not None:
        experiment_name = "A_Patch"
        experiment_dir = patch_dir
        made, notes = process_experiment(ctx, experiment_name, experiment_dir)
        all_notes.extend(notes)
        print(f"{experiment_name}: wrote {len(made)} graph(s)")

    made, notes = make_combined_internal_plots(ctx, experiment_dirs)
    all_notes.extend(notes)
    if made:
        print(f"combined_all_experiments: wrote {len(made)} graph(s)")

    made, notes = make_combined_external_plots(ctx, experiment_dirs)
    all_notes.extend(notes)
    if made:
        print(f"combined_all_experiments external: wrote {len(made)} graph(s)")

    if all_notes:
        notes_path = output_root / "notes.txt"
        notes_path.write_text("\n".join(all_notes) + "\n", encoding="utf-8")
        print(f"Notes: {notes_path}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_matplotlib()
    args = parse_args(argv)
    seeds = RANDOM_SEEDS if args.all_seeds else [args.seed]
    for seed in seeds:
        status = _run(args, int(seed))
        if status != 0:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
