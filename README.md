# Localized Patch Radiomics for Clinically Significant Prostate Cancer and Zone Classification on Biparametric MRI

A reproducible machine-learning pipeline for detecting **clinically significant prostate cancer (csPCa)** and classifying its **anatomical zone** (peripheral vs. transition) from **biparametric prostate MRI** (T2-weighted and apparent-diffusion-coefficient maps). Its central method is **localized patch radiomics** — dense, multi-scale patch descriptors extracted across the gland — which we benchmark against whole-organ handcrafted radiomics and their early- and late-fusion combinations under a single, leakage-controlled nested cross-validation protocol, with final validation on an independent external cohort.

> **Research use only.** This software is provided for methodological research and reproducibility. It is **not** a medical device and must **not** be used for clinical decision-making.

---

## Table of Contents

1. [Overview](#overview)
2. [Clinical Motivation](#clinical-motivation)
3. [Prediction Tasks](#prediction-tasks)
4. [Data](#data)
5. [Experimental Design](#experimental-design)
6. [Pipeline Architecture](#pipeline-architecture)
7. [Repository Layout](#repository-layout)
8. [Installation](#installation)
9. [Commands](#commands)
10. [Outputs](#outputs)

---

## Overview

Prostate MRI is highly informative but operator-dependent, and radiomic signatures can vary with segmentation, feature representation, and validation design. In this work I set out to answer a focused methodological question:

> *Given the same patients and the same rigorously controlled validation, which image representation — localized patch descriptors, whole-organ handcrafted radiomics, or a fusion of the two — best supports the detection and zonal characterisation of clinically significant prostate cancer, and does any advantage survive external validation?*

To answer it, I build a family of feature representations from the same cohort, evaluate every representation with an identical **nested cross-validation** procedure, combine the strongest base models through a **stacked ensemble**, and finally re-fit deployable models on the full training set for **external validation** on a held-out cohort. I seed-control every stochastic choice and repeat it across multiple seeds, so that the differences I report reflect signal rather than a fortunate data split.

---

## Clinical Motivation

Not every prostate lesion warrants intervention. Distinguishing **clinically significant** disease from indolent findings is central to reducing over-treatment, and the **zone** in which a significant lesion arises (peripheral vs. transition) carries different diagnostic priors and imaging appearances. I therefore treat detection and zonal characterisation as two linked but distinct problems, and evaluate both.

---

## Prediction Tasks

I train and evaluate two binary tasks derived from the reference labels:

| Task key | Question | Positive / Negative | Cohort used |
|----------|----------|---------------------|-------------|
| `cs`   | Is clinically significant cancer present? | `TRUE` vs `FALSE` (`clin_sig`) | All labelled cases |
| `zone` | For significant cases, which zone? | Transition (`TZ`) vs Peripheral (`PZ`) | csPCa-positive cases only |

For cross-validation, cases are stratified on a combined **three-class** label — `FALSE`, `TRUE_PZ`, `TRUE_TZ` — so that both disease status and zone are balanced across folds.

---

## Data

I use three cohorts, each organised into modality/mask sub-folders with per-case image files and an accompanying label spreadsheet:

| Role | Cohort | Purpose |
|------|--------|---------|
| **Main (internal)** | ProstateX | Model development and nested cross-validation |
| **Helper (external)** | PICAI | Training the patch-level helper models only |
| **Test (external)** | Independent cohort (`P158`) | Final held-out validation, no refitting |

Each case provides co-registered **T2W** and **ADC** volumes with an organ (prostate) mask; the main cohort additionally provides **two independent organ segmentations** (`mask_organ`, `mask_organ (b)`), which are used to quantify feature reproducibility (see [Methodological Safeguards](#methodological-safeguards)). Lesion and zone masks, where available, support patch labelling and quality control.

```
Input/
├── Train_Main (ProstateX)/
│   ├── original/{t2w, adc, mask_organ, mask_organ (b), mask_lesion, mask_zone}/
│   └── ProstateX-Dataset.xlsx
├── Train_Helper (PICAI)/
│   └── original/{t2w, adc, mask_organ, mask_lesion, mask_zone}/
└── Test (P158)/
    ├── original/{t2w, adc, mask_organ}/
    └── P158-Dataset.xlsx
```

The preprocessing stage resamples each volume to a common spacing and produces an organ-centred crop under a parallel `preprocessed/` folder; all downstream stages read from there.

### Obtaining the data

The complete `Input/` folder — organised exactly as shown above — is available for download:

📁 **[Download the `Input/` folder (Google Drive)](https://drive.google.com/drive/folders/1oW_m37vEN7BitG1HHcwMgKGtvctIAdc4?usp=sharing)**

Download it and place it under your project root, next to `Code/` (see [Installation → Step 1](#installation)).

Every cohort I use comes **exclusively from publicly released prostate-MRI datasets** (ProstateX for development and PICAI for the patch-helper training, plus the public external test cohort). Crucially, **I removed no case**: I keep every publicly available case rather than curating the data down to an easier subset. I made this choice deliberately to keep the evaluation unfiltered, so that the performance I report reflects a realistic and more generalizable cohort.

---

## Experimental Design

I evaluate five experiment arms under the same folds and seeds. Each produces the same nine feature files (single-modality, concatenation, element-wise interactions, and their multi-way fusions), so I compare the representations on completely equal footing.

| Arm | Name | Representation |
|-----|------|----------------|
| **A** | `A_Patch` | Multi-scale **patch descriptors**: helper models trained on the external cohort, applied across the prostate ROI, and aggregated into case-level features |
| **B** | `B_Organ` | **Handcrafted organ radiomics** (Laws, Gabor, Haralick/GLCM, gradient, gray-level, and topological descriptors), filtered for inter-segmentation stability |
| **B0** | `B_Organ_same` | The patch model's raw-image features computed over the whole organ ROI — an internal control isolating *representation* from *localization* (nested CV only; never externally tested) |
| **C** | `C_Early` | **Early fusion**: organ and patch features merged at the feature level |
| **D** | `D_Late` | **Late fusion**: organ and patch prediction scores rank-normalised and blended over a weight grid |

**Feature sets (per arm):** `t2w`, `adc`, `concat`, `hada` (element-wise product), `diff` (difference), and the fusions `fusion(cd)`, `fusion(dh)`, `fusion(ch)`, `fusion(cdh)`.

**Modelling grid.** Within each outer fold, an inner search tunes:

- **Feature selection** — ANOVA *F*-test filter, or L1-embedded logistic-regression selection.
- **Dimensionality reduction** — optional PCA (retained variance grid).
- **Classifier** — logistic regression, linear SVM, Gaussian naïve Bayes, or linear discriminant analysis.

All per-fold preprocessing (median imputation, quasi-constant removal, correlation pruning, z-score scaling, PCA) is fitted on the **training partition only** and applied to the validation/test partition, so no target information leaks across the fold boundary.

**Stacking.** For each task and outer fold, the top base candidates (ranked by inner out-of-fold AUC) are combined by a meta-learner; the ensemble rule — single best, simple average, performance-weighted average, or a logistic-regression stacker — is selected by out-of-fold performance rather than test performance.

**External validation.** After cross-validation, deployable models are re-fitted on the entire main cohort (with out-of-fold predictions preserved for the stacker) and applied **without refitting** to the independent test cohort.

---

## Pipeline Architecture

The stages run in sequence and are chained by the orchestrator, but each is also runnable in isolation:

```
a_preprocess  →  b_patch      →  c_organ     →  d_fusion    →  e_ml        →  f_stack     →  g_train      →  h_test
 resample/crop   patch models    organ         early & late    nested-CV      stacked        deployable      external
                 + application    radiomics     fusion          feature+model  ensemble       refit           validation
                                  + ICC filter                  search
```

| Stage | Script | Responsibility |
|-------|--------|----------------|
| Config | `all_config.py` | Single source of truth: paths, seeds, feature sets, hyper-parameter grids, I/O helpers |
| Preprocess | `a_preprocess.py` | Resampling to common spacing and organ-centred cropping; quality-control filtering |
| Patch | `b_patch.py` | Trains multi-scale patch classifiers on the helper cohort and applies them across the organ ROI to build patch features |
| Organ | `c_organ.py` | Extracts handcrafted radiomic and topological features and retains only inter-segmentation–stable features (ICC) |
| Fusion | `d_fusion.py` | Builds early-fusion feature workbooks and computes late-fusion score blends |
| ML | `e_ml.py` | Nested cross-validated feature selection and classification for every feature set |
| Stacking | `f_stack.py` | Selects and combines base models into a stacked ensemble per task and fold |
| Train | `g_train.py` | Re-fits deployable models on the full main cohort |
| Test | `h_test.py` | Applies deployable models to the external cohort; optionally merges external metrics into the analysis workbooks |
| Orchestrator | `z_main.py` | Runs the entire study end-to-end with caching and per-seed bookkeeping |

---

## Repository Layout

```
Code/
├── all_config.py        # central configuration and shared helpers
├── a_preprocess.py      # resample + organ-centred crop
├── b_patch.py           # patch helper models + application (Experiment A)
├── c_organ.py           # handcrafted organ radiomics + ICC filtering (Experiment B / B0)
├── d_fusion.py          # early- and late-fusion feature/score construction (C / D)
├── e_ml.py              # nested cross-validation engine
├── f_stack.py           # stacked-ensemble meta-learner
├── g_train.py           # deployable model refit
├── h_test.py            # external validation
├── z_main.py            # end-to-end orchestrator
└── README.md
```

---

## Installation

The pipeline targets **Python 3.11** and pins its dependencies for reproducibility.

### Step 1 — Place the Code and data under one project root

Put the `Code/` folder (this repository) and the `Input/` data folder side by side inside a single directory of your choice. That directory is your **project root**; the pipeline creates `Output/` and `logs/` next to them automatically.

```
<PROJECT_ROOT>/            # any location you choose, e.g. /home/user/folder
├── Code/             # this repository (contains z_main.py, all_config.py, …)
├── Input/                 # datasets — see the Data section for the exact structure
├── Output/                # created automatically when you run the pipeline
└── logs/                  # created for background-run logs (Step 4)
```

### Step 2 — Point the pipeline at your project root ⚠️ **required**

All paths are resolved from a single constant. Open **`Code/all_config.py`** and change the first line to the absolute path of *your* project root:

```python
# all_config.py (line 6)
REPO_ROOT = Path("/home/mht/17MHT/BSPC")     # ← replace with YOUR project root
# e.g.  REPO_ROOT = Path("/home/user/folder")
```

`Input/`, `Output/`, and every dataset path are derived from `REPO_ROOT`, so this one edit is all that is needed to relocate the project. If you skip it, the pipeline will keep looking under the default path from my own setup and fail.

### Step 3 — Create the environment and install dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate

pip install numpy==2.4.3 pandas==3.0.0 scipy==1.17.0 scikit-learn==1.8.0 joblib==1.5.3 SimpleITK==2.5.3 scikit-image==0.26.0 openpyxl==3.1.5

pip install cupy-cuda12x     # or cupy-cuda11x, depending on your CUDA version (Optional, falls back to NumPy)
```

Find your interpreter path for later (needed for background runs):

```bash
which python        # e.g. /home/user/folder/.venv/bin/python
```

---

## Commands

```bash
python z_main.py
```

This preprocesses the data, builds every feature representation, runs nested cross-validation and stacking for all arms, and — if the external cohort is present — refits deployable models and validates them externally. Intermediate results are cached, so re-running resumes rather than recomputing.

| Flag | Effect |
|------|--------|
| `--only A,B,C` | Restrict to specific experiment arms (`A,B,B0,C,D`) |
| `--seed N` | Run a single random seed (default `7`; valid seeds `7,21,42,73,101`) |
| `--all-seeds` | Repeat the whole study across all seeds for robustness |
| `--skip-stacking` | Skip the stacked-ensemble stage |
| `--skip-external` | Skip external validation |
| `--no-deep-analysis` | Run external validation without merging external metrics into the analysis workbooks |
| `--stacking-top-k K` | Number of base candidates retained per task/fold (default `8`) |
| `--alpha-grid a,b,c` | Late-fusion weight grid (default `0.25,0.5,0.75`) |

### Running in the background

A full run can take hours, so I launch it with `nohup` from the project root (the folder holding `Code/` and `Input/`), with all output streamed to a timestamped file under `logs/`:

```bash
cd <PROJECT_ROOT>
mkdir -p logs

# Full study — all experiments (A, B, B0, C, D)
nohup <PYTHON> -u "Code/z_main.py" --all-seeds > "logs/Code_$(date +%Y%m%d_%H%M%S)_z_main.log" 2>&1 &

# A single experiment arm, e.g. A_Patch only
nohup <PYTHON> -u "Code/z_main.py" --all-seeds --only A > "logs/Code_$(date +%Y%m%d_%H%M%S)_z_main_A.log" 2>&1 &
```

| Placeholder | Replace with |
|-------------|--------------|
| `<PROJECT_ROOT>` | The folder holding `Code/` and `Input/` e.g. `/home/user/folder` |
| `<PYTHON>` | Your environment's Python interpreter, e.g. `/home/user/folder/.venv/bin/python` |

Every stage can also be run on its own, the same way:

```bash
nohup <PYTHON> -u "Code/b_patch.py" > "logs/Code_$(date +%Y%m%d_%H%M%S)_b_patch.log" 2>&1 &          # A_Patch
nohup <PYTHON> -u "Code/c_organ.py"  > "logs/Code_$(date +%Y%m%d_%H%M%S)_c_organ.log" 2>&1 &         # B_Organ
nohup <PYTHON> -u "Code/c_organ.py" --same > "logs/Code_$(date +%Y%m%d_%H%M%S)_c_organ_same.log" 2>&1 &    # B_Organ_same
nohup <PYTHON> -u "Code/d_fusion.py" --stage early > "logs/Code_$(date +%Y%m%d_%H%M%S)_d_fusion_early.log" 2>&1 &  # C_Early
nohup <PYTHON> -u "Code/d_fusion.py" --stage late > "logs/Code_$(date +%Y%m%d_%H%M%S)_d_fusion_late.log" 2>&1 &   # D_Late
nohup <PYTHON> -u "Code/e_ml.py" > "logs/Code_$(date +%Y%m%d_%H%M%S)_e_ml.log" 2>&1 &            # nested-CV ML search
nohup <PYTHON> -u "Code/f_stack.py" > "logs/Code_$(date +%Y%m%d_%H%M%S)_f_stack.log" 2>&1 &         # stacking
nohup <PYTHON> -u "Code/g_train.py" > "logs/Code_$(date +%Y%m%d_%H%M%S)_g_train.log" 2>&1 &         # deployable train
nohup <PYTHON> -u "Code/h_test.py" --prepare > "logs/Code_$(date +%Y%m%d_%H%M%S)_h_test.log" 2>&1 &         # external test
```

---

## Outputs

Results are written under `Output/`, organised by experiment arm and, within each, by random seed:

```
Output/
├── A_Patch/      # Arm A — localized multi-scale patch-radiomics experiment
├── B_Organ/      # Arm B — whole-organ improved handcrafted-radiomics experiment
├── B_Organ_same/ # Arm B0 — whole-organ handcrafted-radiomics experiment (same raw features as A)
├── C_Early/      # Arm C — early (extracted-feature level) fusion of A + B
└── D_Late/       # Arm D — late (prediction-score level) fusion of A + B
```

Each arm produces per-feature-set result workbooks (per-fold metrics, combination summaries, selected features), a top-model text report, a stacked-ensemble workbook (`final_stacking.xlsx`), and — for externally validated arms — an external-testing workbook. Reported metrics include AUC, average precision, balanced accuracy, F1, accuracy, sensitivity, and specificity, summarised as mean ± standard deviation across outer folds.

---

## Status

The associated manuscript — *"Localized Patch Radiomics for Clinically Significant Prostate Cancer and Zone Classification on Biparametric MRI"* — is currently submitted to *Biomedical Signal Processing and Control* (BSPC). A citation will be added here once the paper is published.

---

## Acknowledgements

This work makes use of the publicly available **ProstateX**, **PICAI** and **P158** prostate-MRI datasets. I thank their organisers and the participating institutions for enabling reproducible prostate-cancer imaging research.
