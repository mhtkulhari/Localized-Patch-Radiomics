# Localized Patch Radiomics for Clinically Significant Prostate Cancer and Zone Classification on Biparametric MRI

A reproducible machine-learning pipeline for detecting **clinically significant prostate cancer (csPCa)** and classifying its **anatomical zone** (peripheral vs. transition) from **biparametric prostate MRI** (T2-weighted and apparent-diffusion-coefficient maps). Its central method is **localized patch radiomics** — dense, multi-scale patch descriptors extracted across the gland — which we benchmark against whole-organ handcrafted radiomics and their early- and late-fusion combinations under a single, leakage-controlled nested cross-validation protocol, with final validation on an independent external cohort.

> **Research use only.** This software is provided for methodological research and reproducibility. It is **not** a medical device and must **not** be used for clinical decision-making.

---

## Table of Contents

1. [Data](#data)
2. [Experiments](#experiments)
3. [Task Routes](#task-routes)
4. [Pipeline](#pipeline)
5. [Setup](#setup)
6. [Running](#running)
7. [Full Run](#full-run)
8. [Individual Run](#individual-run)
9. [Outputs](#outputs)
10. [Status](#status)
11. [Acknowledgements](#acknowledgements)

---

## Data

The configured datasets are:

| Role | Folder | Use |
|------|--------|-----|
| Main | `Train_Main (ProstateX)` | Internal model development and nested cross-validation |
| Helper | `Train_Helper (PICAI)` | Patch helper-model training |
| External test | `Test (P158)` | Held-out external evaluation |

Expected input layout:

```text
Input/
├── Train_Main (ProstateX)/
│   ├── original/{t2w,adc,mask_organ,mask_organ (b),mask_lesion,mask_zone}/
│   └── ProstateX-Dataset.xlsx
├── Train_Helper (PICAI)/
│   └── original/{t2w,adc,mask_organ,mask_lesion,mask_zone}/
└── Test (P158)/
    ├── original/{t2w,adc,mask_organ}/
    └── P158-Dataset.xlsx
```

The input folder is available from [Google Drive](https://drive.google.com/drive/folders/1oW_m37vEN7BitG1HHcwMgKGtvctIAdc4?usp=sharing).

Preprocessing resamples each volume to the configured spacing and writes organ-centred crops under each cohort's `preprocessed/` directory.

Original dataset references:

- ProstateX: Litjens G, Debats O, Barentsz J, Karssemeijer N, Huisman H. *SPIE-AAPM PROSTATEx Challenge Data (Version 2)*. The Cancer Imaging Archive; 2017. [doi:10.7937/K9TCIA.2017.MURS5CL](https://doi.org/10.7937/K9TCIA.2017.MURS5CL)
- PI-CAI: Saha A, Bosma JS, Twilt JJ, et al. Artificial intelligence and radiologists in prostate cancer detection on MRI (PI-CAI): an international, paired, non-inferiority, confirmatory study. *Lancet Oncology*. 2024;25:879-887. [doi:10.1016/S1470-2045(24)00220-1](https://doi.org/10.1016/S1470-2045(24)00220-1)
- P158: Adams LC, Makowski MR, Engel G, et al. Prostate158 - An expert-annotated 3T MRI dataset and algorithm for prostate cancer detection. *Computers in Biology and Medicine*. 2022;148:105817. [doi:10.1016/j.compbiomed.2022.105817](https://doi.org/10.1016/j.compbiomed.2022.105817)

---

## Experiments

| Key | Name | Method |
|-----|------|--------|
| `A` | `A_Patch` | Multi-scale localized patch descriptors |
| `B` | `B_Organ` | Whole-organ handcrafted radiomic and topological features retained after inter-segmentation ICC filtering |
| `B0` | `B_Organ_same` | Patch-style raw-image features calculated over the whole-organ ROI |
| `C` | `C_Early` | A and B feature columns merged by `case_id` before machine learning |
| `D` | `D_Late` | Rank-normalized final A and B scores combined as `alpha * A + (1 - alpha) * B`; `alpha = 0.5` is configured in `all_config.py` |

The configured feature files are:

- `feature--concat.csv`
- `feature--fusion(dh).csv`

The classical ML configuration uses L1-embedded feature selection, optional PCA, and logistic regression or linear discriminant analysis. Preprocessing and feature selection are fitted only on each training partition.

### Task Routes

- A csPCa: summary features, nested CV, and final stacking
- A zone: max-pool multiple-instance model from patch detection bags
- B, B0, and C: classical ML and final stacking for both tasks
- D: final A and B scores for each task, using the same score-fusion rule internally and externally

---

## Pipeline

```text
a_preprocess
├── b_patch
│   ├── A summary features -> e_ml (cs) -> f_stack
│   └── A detection bags -> g_maxPool (zone)
├── c_organ
│   └── B/B0 features -> e_ml (cs, zone) -> f_stack
├── d_fusion --stage early
│   └── C features -> e_ml (cs, zone) -> f_stack
└── d_fusion --stage late
    └── final A/B scores -> D final_stacking.xlsx

h_train -> deployable A/B/B0/C models
i_test  -> external A/B/B0/C predictions -> external D score fusion
```

| Script | Responsibility |
|--------|----------------|
| `all_config.py` | Paths, seeds, feature sets, model configuration, and late-fusion alpha |
| `a_preprocess.py` | Resampling and organ-centred cropping |
| `b_patch.py` | Patch helper-model training, inference, and summary-feature construction |
| `c_organ.py` | Organ feature extraction and ICC filtering |
| `d_fusion.py` | Early-fusion feature construction and internal D score fusion |
| `e_ml.py` | Nested cross-validation for workbook-based experiments |
| `f_stack.py` | Final prediction aggregation for the configured experiments |
| `g_maxPool.py` | Internal and deployable A zone max-pool models |
| `h_train.py` | Full-training-set refit for deployable A, B, B0, and C models |
| `i_test.py` | External feature preparation, prediction, and D score fusion |
| `z_main.py` | End-to-end orchestration, dependency checks, and caching |
| `x_results.py` | Internal and external results workbook generation |
| `x_graphs.py` | Internal and external results graph generation |

---

## Setup

The project root must contain the code and input folders:

```text
<PROJECT_ROOT>/
├── Code/                   # All python .py code files
├── Input/                  # Download from provided g-drive link 
├── logs/                   # Create manually or mkdir -p logs
└── Output/                 # Created by the pipeline
```

Set `REPO_ROOT` in `all_config.py`:

```python
REPO_ROOT = Path("/absolute/path/to/project/root")
```

The code uses Python 3.11 and the following main packages:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy scikit-learn joblib SimpleITK scikit-image openpyxl matplotlib torch
```

CuPy is optional. `c_organ.py` falls back to NumPy when CuPy is unavailable.

---

## Running

Replace `<PYTHON>` with Python executable from configured environment. i.e `/home/mht/miniconda3/envs/mht/bin/python`

Run commands from `<PROJECT_ROOT>` i.e. cd "/home/mht/17MHT/BSPC"

Supported arguments:

| Option | Effect |
|--------|--------|
| `--only A,B,B0,C,D` | Run only the listed experiments |
| `--seed N` | Run one seed; default is `42` |
| `--all-seeds` | Run seeds `7,21,42,73,101` |
| `--skip-external` | Skip external evaluation |
| `--external-experiments A,B,B0,C,D` | Select external experiments; default matches `--only` |

### Full Run

```bash
nohup <PYTHON> -u "Code/z_main.py" > "logs/Code_$(date +%Y%m%d_%H%M%S)_z_main.log" 2>&1 &
```

### Individual Run

Run an individual command after its required upstream outputs are available.

```bash
nohup <PYTHON> -u "Code/z_main.py" --only A > "logs/Code_$(date +%Y%m%d_%H%M%S)_z_main_A.log" 2>&1 &

nohup <PYTHON> -u "Code/a_preprocess.py" > "logs/Code_$(date +%Y%m%d_%H%M%S)_a_preprocess.log" 2>&1 &
nohup <PYTHON> -u "Code/b_patch.py" --stage all > "logs/Code_$(date +%Y%m%d_%H%M%S)_b_patch.log" 2>&1 &
nohup <PYTHON> -u "Code/c_organ.py" > "logs/Code_$(date +%Y%m%d_%H%M%S)_c_organ_B.log" 2>&1 &
nohup <PYTHON> -u "Code/c_organ.py" --same > "logs/Code_$(date +%Y%m%d_%H%M%S)_c_organ_B0.log" 2>&1 &
nohup <PYTHON> -u "Code/d_fusion.py" --stage early > "logs/Code_$(date +%Y%m%d_%H%M%S)_d_fusion_early.log" 2>&1 &
nohup <PYTHON> -u "Code/d_fusion.py" --stage late > "logs/Code_$(date +%Y%m%d_%H%M%S)_d_fusion_late.log" 2>&1 &
nohup <PYTHON> -u "Code/e_ml.py" > "logs/Code_$(date +%Y%m%d_%H%M%S)_e_ml.log" 2>&1 &
nohup <PYTHON> -u "Code/f_stack.py" > "logs/Code_$(date +%Y%m%d_%H%M%S)_f_stack.log" 2>&1 &
nohup <PYTHON> -u "Code/g_maxPool.py" > "logs/Code_$(date +%Y%m%d_%H%M%S)_g_maxPool.log" 2>&1 &
nohup <PYTHON> -u "Code/h_train.py" > "logs/Code_$(date +%Y%m%d_%H%M%S)_h_train.log" 2>&1 &
nohup <PYTHON> -u "Code/i_test.py" > "logs/Code_$(date +%Y%m%d_%H%M%S)_i_test.log" 2>&1 &

nohup <PYTHON> -u "Code/x_results.py" > "logs/Code_$(date +%Y%m%d_%H%M%S)_x_results.log" 2>&1 &
nohup <PYTHON> -u "Code/x_graphs.py" > "logs/Code_$(date +%Y%m%d_%H%M%S)_x_graphs.log" 2>&1 &
```

---

## Outputs

Results are written under `Output/`:

```text
Output/
├── A_Patch/
│   ├── 01_raw_features_helper/
│   ├── 02_models_helper/
│   ├── 03_raw_features_main/
│   ├── 04_detection_by_helper/
│   │   ├── concat/
│   │   └── fusion(dh)/
│   ├── 05_summary_features/
│   ├── 06_results/random_seed{N}/
│   │   ├── final_stacking.xlsx
│   │   └── maxpool_zone.xlsx
│   ├── 07_final_model/random_seed{N}/
│   └── 08_external_testing/
│       ├── 01_summary_features/
│       └── 02_results/ext_results_seed{N}.xlsx
├── B_Organ/
│   ├── 01_raw_features/
│   ├── 02_selected_features/
│   ├── 03_results/random_seed{N}/final_stacking.xlsx
│   ├── 04_final_model/random_seed{N}/
│   └── 05_external_testing/
│       ├── 01_summary_features/
│       └── 02_results/ext_results_seed{N}.xlsx
├── B_Organ_same/
│   ├── 01_raw_features/
│   ├── 02_selected_features/
│   ├── 03_results/random_seed{N}/final_stacking.xlsx
│   ├── 04_final_model/random_seed{N}/
│   └── 05_external_testing/
│       ├── 01_summary_features/
│       └── 02_results/ext_results_seed{N}.xlsx
├── C_Early/
│   ├── 01_fused_features/
│   ├── 02_results/random_seed{N}/final_stacking.xlsx
│   ├── 03_final_model/random_seed{N}/
│   └── 04_external_testing/
│       ├── 01_summary_features/
│       └── 02_results/ext_results_seed{N}.xlsx
└── D_Late/
    ├── 01_results/random_seed{N}/final_stacking.xlsx
    └── 02_external_testing/02_results/ext_results_seed{N}.xlsx
```

---

## Status

The associated manuscript — *"Localized Patch Radiomics for Clinically Significant Prostate Cancer and Zone Classification on Biparametric MRI"* — is currently submitted to *Biomedical Signal Processing and Control* (BSPC). A citation will be added here once the paper is published.

---

## Acknowledgements

This work makes use of the publicly available **ProstateX**, **PICAI** and **P158** prostate-MRI datasets. I thank their organisers and the participating institutions for enabling reproducible prostate-cancer imaging research.
