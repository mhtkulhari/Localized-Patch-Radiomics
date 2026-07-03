from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

import argparse
import csv
import json
from pathlib import Path
import os


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import re
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from skimage.measure import marching_cubes, mesh_surface_area, euler_number
from concurrent.futures import ProcessPoolExecutor, as_completed

from all_config import FEATURE_WORKERS as N_WORKERS

MHT_VERBOSE = os.environ.get("MHT_VERBOSE", "0").strip().lower() in {"1", "true", "yes", "y"}

def _vprint(*args, **kwargs):
    if MHT_VERBOSE:
        print(*args, **kwargs)

ROI_BBOX_MARGIN = 8
EPS = 1e-12

SCRIPT_DIR = Path(__file__).resolve().parent

HAR_LEVELS = 8
HAR_OFFSETS_2D = [(0, 1), (1, 0), (1, 1), (1, -1)]


TEXTURE_FEATURE_TABLE = [
    ('kurtosis-Laws W5W5W5', 'median-Laws W5W5W5', 'skewness-Laws W5W5W5', 'var-Laws W5W5W5'),
    ('kurtosis-Laws W5W5S5', 'median-Laws W5W5S5', 'skewness-Laws W5W5S5', 'var-Laws W5W5S5'),
    ('kurtosis-Laws W5W5R5', 'median-Laws W5W5R5', 'skewness-Laws W5W5R5', 'var-Laws W5W5R5'),
    ('kurtosis-Laws W5W5L5', 'median-Laws W5W5L5', 'skewness-Laws W5W5L5', 'var-Laws W5W5L5'),
    ('kurtosis-Laws W5W5E5', 'median-Laws W5W5E5', 'skewness-Laws W5W5E5', 'var-Laws W5W5E5'),
    ('kurtosis-Laws W5S5W5', 'median-Laws W5S5W5', 'skewness-Laws W5S5W5', 'var-Laws W5S5W5'),
    ('kurtosis-Laws W5S5S5', 'median-Laws W5S5S5', 'skewness-Laws W5S5S5', 'var-Laws W5S5S5'),
    ('kurtosis-Laws W5S5R5', 'median-Laws W5S5R5', 'skewness-Laws W5S5R5', 'var-Laws W5S5R5'),
    ('kurtosis-Laws W5S5L5', 'median-Laws W5S5L5', 'skewness-Laws W5S5L5', 'var-Laws W5S5L5'),
    ('kurtosis-Laws W5S5E5', 'median-Laws W5S5E5', 'skewness-Laws W5S5E5', 'var-Laws W5S5E5'),
    ('kurtosis-Laws W5R5W5', 'median-Laws W5R5W5', 'skewness-Laws W5R5W5', 'var-Laws W5R5W5'),
    ('kurtosis-Laws W5R5S5', 'median-Laws W5R5S5', 'skewness-Laws W5R5S5', 'var-Laws W5R5S5'),
    ('kurtosis-Laws W5R5R5', 'median-Laws W5R5R5', 'skewness-Laws W5R5R5', 'var-Laws W5R5R5'),
    ('kurtosis-Laws W5R5L5', 'median-Laws W5R5L5', 'skewness-Laws W5R5L5', 'var-Laws W5R5L5'),
    ('kurtosis-Laws W5R5E5', 'median-Laws W5R5E5', 'skewness-Laws W5R5E5', 'var-Laws W5R5E5'),
    ('kurtosis-Laws W5L5W5', 'median-Laws W5L5W5', 'skewness-Laws W5L5W5', 'var-Laws W5L5W5'),
    ('kurtosis-Laws W5L5S5', 'median-Laws W5L5S5', 'skewness-Laws W5L5S5', 'var-Laws W5L5S5'),
    ('kurtosis-Laws W5L5R5', 'median-Laws W5L5R5', 'skewness-Laws W5L5R5', 'var-Laws W5L5R5'),
    ('kurtosis-Laws W5L5L5', 'median-Laws W5L5L5', 'skewness-Laws W5L5L5', 'var-Laws W5L5L5'),
    ('kurtosis-Laws W5L5E5', 'median-Laws W5L5E5', 'skewness-Laws W5L5E5', 'var-Laws W5L5E5'),
    ('kurtosis-Laws W5E5W5', 'median-Laws W5E5W5', 'skewness-Laws W5E5W5', 'var-Laws W5E5W5'),
    ('kurtosis-Laws W5E5S5', 'median-Laws W5E5S5', 'skewness-Laws W5E5S5', 'var-Laws W5E5S5'),
    ('kurtosis-Laws W5E5R5', 'median-Laws W5E5R5', 'skewness-Laws W5E5R5', 'var-Laws W5E5R5'),
    ('kurtosis-Laws W5E5L5', 'median-Laws W5E5L5', 'skewness-Laws W5E5L5', 'var-Laws W5E5L5'),
    ('kurtosis-Laws W5E5E5', 'median-Laws W5E5E5', 'skewness-Laws W5E5E5', 'var-Laws W5E5E5'),
    ('kurtosis-Laws S5W5W5', 'median-Laws S5W5W5', 'skewness-Laws S5W5W5', 'var-Laws S5W5W5'),
    ('kurtosis-Laws S5W5S5', 'median-Laws S5W5S5', 'skewness-Laws S5W5S5', 'var-Laws S5W5S5'),
    ('kurtosis-Laws S5W5R5', 'median-Laws S5W5R5', 'skewness-Laws S5W5R5', 'var-Laws S5W5R5'),
    ('kurtosis-Laws S5W5L5', 'median-Laws S5W5L5', 'skewness-Laws S5W5L5', 'var-Laws S5W5L5'),
    ('kurtosis-Laws S5W5E5', 'median-Laws S5W5E5', 'skewness-Laws S5W5E5', 'var-Laws S5W5E5'),
    ('kurtosis-Laws S5S5W5', 'median-Laws S5S5W5', 'skewness-Laws S5S5W5', 'var-Laws S5S5W5'),
    ('kurtosis-Laws S5S5S5', 'median-Laws S5S5S5', 'skewness-Laws S5S5S5', 'var-Laws S5S5S5'),
    ('kurtosis-Laws S5S5R5', 'median-Laws S5S5R5', 'skewness-Laws S5S5R5', 'var-Laws S5S5R5'),
    ('kurtosis-Laws S5S5L5', 'median-Laws S5S5L5', 'skewness-Laws S5S5L5', 'var-Laws S5S5L5'),
    ('kurtosis-Laws S5S5E5', 'median-Laws S5S5E5', 'skewness-Laws S5S5E5', 'var-Laws S5S5E5'),
    ('kurtosis-Laws S5R5W5', 'median-Laws S5R5W5', 'skewness-Laws S5R5W5', 'var-Laws S5R5W5'),
    ('kurtosis-Laws S5R5S5', 'median-Laws S5R5S5', 'skewness-Laws S5R5S5', 'var-Laws S5R5S5'),
    ('kurtosis-Laws S5R5R5', 'median-Laws S5R5R5', 'skewness-Laws S5R5R5', 'var-Laws S5R5R5'),
    ('kurtosis-Laws S5R5L5', 'median-Laws S5R5L5', 'skewness-Laws S5R5L5', 'var-Laws S5R5L5'),
    ('kurtosis-Laws S5R5E5', 'median-Laws S5R5E5', 'skewness-Laws S5R5E5', 'var-Laws S5R5E5'),
    ('kurtosis-Laws S5L5W5', 'median-Laws S5L5W5', 'skewness-Laws S5L5W5', 'var-Laws S5L5W5'),
    ('kurtosis-Laws S5L5S5', 'median-Laws S5L5S5', 'skewness-Laws S5L5S5', 'var-Laws S5L5S5'),
    ('kurtosis-Laws S5L5R5', 'median-Laws S5L5R5', 'skewness-Laws S5L5R5', 'var-Laws S5L5R5'),
    ('kurtosis-Laws S5L5L5', 'median-Laws S5L5L5', 'skewness-Laws S5L5L5', 'var-Laws S5L5L5'),
    ('kurtosis-Laws S5L5E5', 'median-Laws S5L5E5', 'skewness-Laws S5L5E5', 'var-Laws S5L5E5'),
    ('kurtosis-Laws S5E5W5', 'median-Laws S5E5W5', 'skewness-Laws S5E5W5', 'var-Laws S5E5W5'),
    ('kurtosis-Laws S5E5S5', 'median-Laws S5E5S5', 'skewness-Laws S5E5S5', 'var-Laws S5E5S5'),
    ('kurtosis-Laws S5E5R5', 'median-Laws S5E5R5', 'skewness-Laws S5E5R5', 'var-Laws S5E5R5'),
    ('kurtosis-Laws S5E5L5', 'median-Laws S5E5L5', 'skewness-Laws S5E5L5', 'var-Laws S5E5L5'),
    ('kurtosis-Laws S5E5E5', 'median-Laws S5E5E5', 'skewness-Laws S5E5E5', 'var-Laws S5E5E5'),
    ('kurtosis-Laws S3S3S3', 'median-Laws S3S3S3', 'skewness-Laws S3S3S3', 'var-Laws S3S3S3'),
    ('kurtosis-Laws S3S3L3', 'median-Laws S3S3L3', 'skewness-Laws S3S3L3', 'var-Laws S3S3L3'),
    ('kurtosis-Laws S3S3E3', 'median-Laws S3S3E3', 'skewness-Laws S3S3E3', 'var-Laws S3S3E3'),
    ('kurtosis-Laws S3L3S3', 'median-Laws S3L3S3', 'skewness-Laws S3L3S3', 'var-Laws S3L3S3'),
    ('kurtosis-Laws S3L3L3', 'median-Laws S3L3L3', 'skewness-Laws S3L3L3', 'var-Laws S3L3L3'),
    ('kurtosis-Laws S3L3E3', 'median-Laws S3L3E3', 'skewness-Laws S3L3E3', 'var-Laws S3L3E3'),
    ('kurtosis-Laws S3E3S3', 'median-Laws S3E3S3', 'skewness-Laws S3E3S3', 'var-Laws S3E3S3'),
    ('kurtosis-Laws S3E3L3', 'median-Laws S3E3L3', 'skewness-Laws S3E3L3', 'var-Laws S3E3L3'),
    ('kurtosis-Laws S3E3E3', 'median-Laws S3E3E3', 'skewness-Laws S3E3E3', 'var-Laws S3E3E3'),
    ('kurtosis-Laws R5W5W5', 'median-Laws R5W5W5', 'skewness-Laws R5W5W5', 'var-Laws R5W5W5'),
    ('kurtosis-Laws R5W5S5', 'median-Laws R5W5S5', 'skewness-Laws R5W5S5', 'var-Laws R5W5S5'),
    ('kurtosis-Laws R5W5R5', 'median-Laws R5W5R5', 'skewness-Laws R5W5R5', 'var-Laws R5W5R5'),
    ('kurtosis-Laws R5W5L5', 'median-Laws R5W5L5', 'skewness-Laws R5W5L5', 'var-Laws R5W5L5'),
    ('kurtosis-Laws R5W5E5', 'median-Laws R5W5E5', 'skewness-Laws R5W5E5', 'var-Laws R5W5E5'),
    ('kurtosis-Laws R5S5W5', 'median-Laws R5S5W5', 'skewness-Laws R5S5W5', 'var-Laws R5S5W5'),
    ('kurtosis-Laws R5S5S5', 'median-Laws R5S5S5', 'skewness-Laws R5S5S5', 'var-Laws R5S5S5'),
    ('kurtosis-Laws R5S5R5', 'median-Laws R5S5R5', 'skewness-Laws R5S5R5', 'var-Laws R5S5R5'),
    ('kurtosis-Laws R5S5L5', 'median-Laws R5S5L5', 'skewness-Laws R5S5L5', 'var-Laws R5S5L5'),
    ('kurtosis-Laws R5S5E5', 'median-Laws R5S5E5', 'skewness-Laws R5S5E5', 'var-Laws R5S5E5'),
    ('kurtosis-Laws R5R5W5', 'median-Laws R5R5W5', 'skewness-Laws R5R5W5', 'var-Laws R5R5W5'),
    ('kurtosis-Laws R5R5S5', 'median-Laws R5R5S5', 'skewness-Laws R5R5S5', 'var-Laws R5R5S5'),
    ('kurtosis-Laws R5R5R5', 'median-Laws R5R5R5', 'skewness-Laws R5R5R5', 'var-Laws R5R5R5'),
    ('kurtosis-Laws R5R5L5', 'median-Laws R5R5L5', 'skewness-Laws R5R5L5', 'var-Laws R5R5L5'),
    ('kurtosis-Laws R5R5E5', 'median-Laws R5R5E5', 'skewness-Laws R5R5E5', 'var-Laws R5R5E5'),
    ('kurtosis-Laws R5L5W5', 'median-Laws R5L5W5', 'skewness-Laws R5L5W5', 'var-Laws R5L5W5'),
    ('kurtosis-Laws R5L5S5', 'median-Laws R5L5S5', 'skewness-Laws R5L5S5', 'var-Laws R5L5S5'),
    ('kurtosis-Laws R5L5R5', 'median-Laws R5L5R5', 'skewness-Laws R5L5R5', 'var-Laws R5L5R5'),
    ('kurtosis-Laws R5L5L5', 'median-Laws R5L5L5', 'skewness-Laws R5L5L5', 'var-Laws R5L5L5'),
    ('kurtosis-Laws R5L5E5', 'median-Laws R5L5E5', 'skewness-Laws R5L5E5', 'var-Laws R5L5E5'),
    ('kurtosis-Laws R5E5W5', 'median-Laws R5E5W5', 'skewness-Laws R5E5W5', 'var-Laws R5E5W5'),
    ('kurtosis-Laws R5E5S5', 'median-Laws R5E5S5', 'skewness-Laws R5E5S5', 'var-Laws R5E5S5'),
    ('kurtosis-Laws R5E5R5', 'median-Laws R5E5R5', 'skewness-Laws R5E5R5', 'var-Laws R5E5R5'),
    ('kurtosis-Laws R5E5L5', 'median-Laws R5E5L5', 'skewness-Laws R5E5L5', 'var-Laws R5E5L5'),
    ('kurtosis-Laws R5E5E5', 'median-Laws R5E5E5', 'skewness-Laws R5E5E5', 'var-Laws R5E5E5'),
    ('kurtosis-Laws L5W5W5', 'median-Laws L5W5W5', 'skewness-Laws L5W5W5', 'var-Laws L5W5W5'),
    ('kurtosis-Laws L5W5S5', 'median-Laws L5W5S5', 'skewness-Laws L5W5S5', 'var-Laws L5W5S5'),
    ('kurtosis-Laws L5W5R5', 'median-Laws L5W5R5', 'skewness-Laws L5W5R5', 'var-Laws L5W5R5'),
    ('kurtosis-Laws L5W5L5', 'median-Laws L5W5L5', 'skewness-Laws L5W5L5', 'var-Laws L5W5L5'),
    ('kurtosis-Laws L5W5E5', 'median-Laws L5W5E5', 'skewness-Laws L5W5E5', 'var-Laws L5W5E5'),
    ('kurtosis-Laws L5S5W5', 'median-Laws L5S5W5', 'skewness-Laws L5S5W5', 'var-Laws L5S5W5'),
    ('kurtosis-Laws L5S5S5', 'median-Laws L5S5S5', 'skewness-Laws L5S5S5', 'var-Laws L5S5S5'),
    ('kurtosis-Laws L5S5R5', 'median-Laws L5S5R5', 'skewness-Laws L5S5R5', 'var-Laws L5S5R5'),
    ('kurtosis-Laws L5S5L5', 'median-Laws L5S5L5', 'skewness-Laws L5S5L5', 'var-Laws L5S5L5'),
    ('kurtosis-Laws L5S5E5', 'median-Laws L5S5E5', 'skewness-Laws L5S5E5', 'var-Laws L5S5E5'),
    ('kurtosis-Laws L5R5W5', 'median-Laws L5R5W5', 'skewness-Laws L5R5W5', 'var-Laws L5R5W5'),
    ('kurtosis-Laws L5R5S5', 'median-Laws L5R5S5', 'skewness-Laws L5R5S5', 'var-Laws L5R5S5'),
    ('kurtosis-Laws L5R5R5', 'median-Laws L5R5R5', 'skewness-Laws L5R5R5', 'var-Laws L5R5R5'),
    ('kurtosis-Laws L5R5L5', 'median-Laws L5R5L5', 'skewness-Laws L5R5L5', 'var-Laws L5R5L5'),
    ('kurtosis-Laws L5R5E5', 'median-Laws L5R5E5', 'skewness-Laws L5R5E5', 'var-Laws L5R5E5'),
    ('kurtosis-Laws L5L5W5', 'median-Laws L5L5W5', 'skewness-Laws L5L5W5', 'var-Laws L5L5W5'),
    ('kurtosis-Laws L5L5S5', 'median-Laws L5L5S5', 'skewness-Laws L5L5S5', 'var-Laws L5L5S5'),
    ('kurtosis-Laws L5L5R5', 'median-Laws L5L5R5', 'skewness-Laws L5L5R5', 'var-Laws L5L5R5'),
    ('kurtosis-Laws L5L5L5', 'median-Laws L5L5L5', 'skewness-Laws L5L5L5', 'var-Laws L5L5L5'),
    ('kurtosis-Laws L5L5E5', 'median-Laws L5L5E5', 'skewness-Laws L5L5E5', 'var-Laws L5L5E5'),
    ('kurtosis-Laws L5E5W5', 'median-Laws L5E5W5', 'skewness-Laws L5E5W5', 'var-Laws L5E5W5'),
    ('kurtosis-Laws L5E5S5', 'median-Laws L5E5S5', 'skewness-Laws L5E5S5', 'var-Laws L5E5S5'),
    ('kurtosis-Laws L5E5R5', 'median-Laws L5E5R5', 'skewness-Laws L5E5R5', 'var-Laws L5E5R5'),
    ('kurtosis-Laws L5E5L5', 'median-Laws L5E5L5', 'skewness-Laws L5E5L5', 'var-Laws L5E5L5'),
    ('kurtosis-Laws L5E5E5', 'median-Laws L5E5E5', 'skewness-Laws L5E5E5', 'var-Laws L5E5E5'),
    ('kurtosis-Laws L3S3S3', 'median-Laws L3S3S3', 'skewness-Laws L3S3S3', 'var-Laws L3S3S3'),
    ('kurtosis-Laws L3S3L3', 'median-Laws L3S3L3', 'skewness-Laws L3S3L3', 'var-Laws L3S3L3'),
    ('kurtosis-Laws L3S3E3', 'median-Laws L3S3E3', 'skewness-Laws L3S3E3', 'var-Laws L3S3E3'),
    ('kurtosis-Laws L3L3S3', 'median-Laws L3L3S3', 'skewness-Laws L3L3S3', 'var-Laws L3L3S3'),
    ('kurtosis-Laws L3L3L3', 'median-Laws L3L3L3', 'skewness-Laws L3L3L3', 'var-Laws L3L3L3'),
    ('kurtosis-Laws L3L3E3', 'median-Laws L3L3E3', 'skewness-Laws L3L3E3', 'var-Laws L3L3E3'),
    ('kurtosis-Laws L3E3S3', 'median-Laws L3E3S3', 'skewness-Laws L3E3S3', 'var-Laws L3E3S3'),
    ('kurtosis-Laws L3E3L3', 'median-Laws L3E3L3', 'skewness-Laws L3E3L3', 'var-Laws L3E3L3'),
    ('kurtosis-Laws L3E3E3', 'median-Laws L3E3E3', 'skewness-Laws L3E3E3', 'var-Laws L3E3E3'),
    ('kurtosis-Laws E5W5W5', 'median-Laws E5W5W5', 'skewness-Laws E5W5W5', 'var-Laws E5W5W5'),
    ('kurtosis-Laws E5W5S5', 'median-Laws E5W5S5', 'skewness-Laws E5W5S5', 'var-Laws E5W5S5'),
    ('kurtosis-Laws E5W5R5', 'median-Laws E5W5R5', 'skewness-Laws E5W5R5', 'var-Laws E5W5R5'),
    ('kurtosis-Laws E5W5L5', 'median-Laws E5W5L5', 'skewness-Laws E5W5L5', 'var-Laws E5W5L5'),
    ('kurtosis-Laws E5W5E5', 'median-Laws E5W5E5', 'skewness-Laws E5W5E5', 'var-Laws E5W5E5'),
    ('kurtosis-Laws E5S5W5', 'median-Laws E5S5W5', 'skewness-Laws E5S5W5', 'var-Laws E5S5W5'),
    ('kurtosis-Laws E5S5S5', 'median-Laws E5S5S5', 'skewness-Laws E5S5S5', 'var-Laws E5S5S5'),
    ('kurtosis-Laws E5S5R5', 'median-Laws E5S5R5', 'skewness-Laws E5S5R5', 'var-Laws E5S5R5'),
    ('kurtosis-Laws E5S5L5', 'median-Laws E5S5L5', 'skewness-Laws E5S5L5', 'var-Laws E5S5L5'),
    ('kurtosis-Laws E5S5E5', 'median-Laws E5S5E5', 'skewness-Laws E5S5E5', 'var-Laws E5S5E5'),
    ('kurtosis-Laws E5R5W5', 'median-Laws E5R5W5', 'skewness-Laws E5R5W5', 'var-Laws E5R5W5'),
    ('kurtosis-Laws E5R5S5', 'median-Laws E5R5S5', 'skewness-Laws E5R5S5', 'var-Laws E5R5S5'),
    ('kurtosis-Laws E5R5R5', 'median-Laws E5R5R5', 'skewness-Laws E5R5R5', 'var-Laws E5R5R5'),
    ('kurtosis-Laws E5R5L5', 'median-Laws E5R5L5', 'skewness-Laws E5R5L5', 'var-Laws E5R5L5'),
    ('kurtosis-Laws E5R5E5', 'median-Laws E5R5E5', 'skewness-Laws E5R5E5', 'var-Laws E5R5E5'),
    ('kurtosis-Laws E5L5W5', 'median-Laws E5L5W5', 'skewness-Laws E5L5W5', 'var-Laws E5L5W5'),
    ('kurtosis-Laws E5L5S5', 'median-Laws E5L5S5', 'skewness-Laws E5L5S5', 'var-Laws E5L5S5'),
    ('kurtosis-Laws E5L5R5', 'median-Laws E5L5R5', 'skewness-Laws E5L5R5', 'var-Laws E5L5R5'),
    ('kurtosis-Laws E5L5L5', 'median-Laws E5L5L5', 'skewness-Laws E5L5L5', 'var-Laws E5L5L5'),
    ('kurtosis-Laws E5L5E5', 'median-Laws E5L5E5', 'skewness-Laws E5L5E5', 'var-Laws E5L5E5'),
    ('kurtosis-Laws E5E5W5', 'median-Laws E5E5W5', 'skewness-Laws E5E5W5', 'var-Laws E5E5W5'),
    ('kurtosis-Laws E5E5S5', 'median-Laws E5E5S5', 'skewness-Laws E5E5S5', 'var-Laws E5E5S5'),
    ('kurtosis-Laws E5E5R5', 'median-Laws E5E5R5', 'skewness-Laws E5E5R5', 'var-Laws E5E5R5'),
    ('kurtosis-Laws E5E5L5', 'median-Laws E5E5L5', 'skewness-Laws E5E5L5', 'var-Laws E5E5L5'),
    ('kurtosis-Laws E5E5E5', 'median-Laws E5E5E5', 'skewness-Laws E5E5E5', 'var-Laws E5E5E5'),
    ('kurtosis-Laws E3S3S3', 'median-Laws E3S3S3', 'skewness-Laws E3S3S3', 'var-Laws E3S3S3'),
    ('kurtosis-Laws E3S3L3', 'median-Laws E3S3L3', 'skewness-Laws E3S3L3', 'var-Laws E3S3L3'),
    ('kurtosis-Laws E3S3E3', 'median-Laws E3S3E3', 'skewness-Laws E3S3E3', 'var-Laws E3S3E3'),
    ('kurtosis-Laws E3L3S3', 'median-Laws E3L3S3', 'skewness-Laws E3L3S3', 'var-Laws E3L3S3'),
    ('kurtosis-Laws E3L3L3', 'median-Laws E3L3L3', 'skewness-Laws E3L3L3', 'var-Laws E3L3L3'),
    ('kurtosis-Laws E3L3E3', 'median-Laws E3L3E3', 'skewness-Laws E3L3E3', 'var-Laws E3L3E3'),
    ('kurtosis-Laws E3E3S3', 'median-Laws E3E3S3', 'skewness-Laws E3E3S3', 'var-Laws E3E3S3'),
    ('kurtosis-Laws E3E3L3', 'median-Laws E3E3L3', 'skewness-Laws E3E3L3', 'var-Laws E3E3L3'),
    ('kurtosis-Laws E3E3E3', 'median-Laws E3E3E3', 'skewness-Laws E3E3E3', 'var-Laws E3E3E3'),
    ('kurtosis-Haralick sum-var ws=9', 'median-Haralick sum-var ws=9', 'skewness-Haralick sum-var ws=9', 'var-Haralick sum-var ws=9'),
    ('kurtosis-Haralick sum-var ws=7', 'median-Haralick sum-var ws=7', 'skewness-Haralick sum-var ws=7', 'var-Haralick sum-var ws=7'),
    ('kurtosis-Haralick sum-var ws=5', 'median-Haralick sum-var ws=5', 'skewness-Haralick sum-var ws=5', 'var-Haralick sum-var ws=5'),
    ('kurtosis-Haralick sum-var ws=3', 'median-Haralick sum-var ws=3', 'skewness-Haralick sum-var ws=3', 'var-Haralick sum-var ws=3'),
    ('kurtosis-Haralick sum-var ws=11', 'median-Haralick sum-var ws=11', 'skewness-Haralick sum-var ws=11', 'var-Haralick sum-var ws=11'),
    ('kurtosis-Haralick sum-ent ws=9', 'median-Haralick sum-ent ws=9', 'skewness-Haralick sum-ent ws=9', 'var-Haralick sum-ent ws=9'),
    ('kurtosis-Haralick sum-ent ws=7', 'median-Haralick sum-ent ws=7', 'skewness-Haralick sum-ent ws=7', 'var-Haralick sum-ent ws=7'),
    ('kurtosis-Haralick sum-ent ws=5', 'median-Haralick sum-ent ws=5', 'skewness-Haralick sum-ent ws=5', 'var-Haralick sum-ent ws=5'),
    ('kurtosis-Haralick sum-ent ws=3', 'median-Haralick sum-ent ws=3', 'skewness-Haralick sum-ent ws=3', 'var-Haralick sum-ent ws=3'),
    ('kurtosis-Haralick sum-ent ws=11', 'median-Haralick sum-ent ws=11', 'skewness-Haralick sum-ent ws=11', 'var-Haralick sum-ent ws=11'),
    ('kurtosis-Haralick sum-av ws=9', 'median-Haralick sum-av ws=9', 'skewness-Haralick sum-av ws=9', 'var-Haralick sum-av ws=9'),
    ('kurtosis-Haralick sum-av ws=7', 'median-Haralick sum-av ws=7', 'skewness-Haralick sum-av ws=7', 'var-Haralick sum-av ws=7'),
    ('kurtosis-Haralick sum-av ws=5', 'median-Haralick sum-av ws=5', 'skewness-Haralick sum-av ws=5', 'var-Haralick sum-av ws=5'),
    ('kurtosis-Haralick sum-av ws=3', 'median-Haralick sum-av ws=3', 'skewness-Haralick sum-av ws=3', 'var-Haralick sum-av ws=3'),
    ('kurtosis-Haralick sum-av ws=11', 'median-Haralick sum-av ws=11', 'skewness-Haralick sum-av ws=11', 'var-Haralick sum-av ws=11'),
    ('kurtosis-Haralick info2 ws=9', 'median-Haralick info2 ws=9', 'skewness-Haralick info2 ws=9', 'var-Haralick info2 ws=9'),
    ('kurtosis-Haralick info2 ws=7', 'median-Haralick info2 ws=7', 'skewness-Haralick info2 ws=7', 'var-Haralick info2 ws=7'),
    ('kurtosis-Haralick info2 ws=5', 'median-Haralick info2 ws=5', 'skewness-Haralick info2 ws=5', 'var-Haralick info2 ws=5'),
    ('kurtosis-Haralick info2 ws=3', 'median-Haralick info2 ws=3', 'skewness-Haralick info2 ws=3', 'var-Haralick info2 ws=3'),
    ('kurtosis-Haralick info2 ws=11', 'median-Haralick info2 ws=11', 'skewness-Haralick info2 ws=11', 'var-Haralick info2 ws=11'),
    ('kurtosis-Haralick info1 ws=9', 'median-Haralick info1 ws=9', 'skewness-Haralick info1 ws=9', 'var-Haralick info1 ws=9'),
    ('kurtosis-Haralick info1 ws=7', 'median-Haralick info1 ws=7', 'skewness-Haralick info1 ws=7', 'var-Haralick info1 ws=7'),
    ('kurtosis-Haralick info1 ws=5', 'median-Haralick info1 ws=5', 'skewness-Haralick info1 ws=5', 'var-Haralick info1 ws=5'),
    ('kurtosis-Haralick info1 ws=3', 'median-Haralick info1 ws=3', 'skewness-Haralick info1 ws=3', 'var-Haralick info1 ws=3'),
    ('kurtosis-Haralick info1 ws=11', 'median-Haralick info1 ws=11', 'skewness-Haralick info1 ws=11', 'var-Haralick info1 ws=11'),
    ('kurtosis-Haralick inertia ws=9', 'median-Haralick inertia ws=9', 'skewness-Haralick inertia ws=9', 'var-Haralick inertia ws=9'),
    ('kurtosis-Haralick inertia ws=7', 'median-Haralick inertia ws=7', 'skewness-Haralick inertia ws=7', 'var-Haralick inertia ws=7'),
    ('kurtosis-Haralick inertia ws=5', 'median-Haralick inertia ws=5', 'skewness-Haralick inertia ws=5', 'var-Haralick inertia ws=5'),
    ('kurtosis-Haralick inertia ws=3', 'median-Haralick inertia ws=3', 'skewness-Haralick inertia ws=3', 'var-Haralick inertia ws=3'),
    ('kurtosis-Haralick inertia ws=11', 'median-Haralick inertia ws=11', 'skewness-Haralick inertia ws=11', 'var-Haralick inertia ws=11'),
    ('kurtosis-Haralick idm ws=9', 'median-Haralick idm ws=9', 'skewness-Haralick idm ws=9', 'var-Haralick idm ws=9'),
    ('kurtosis-Haralick idm ws=7', 'median-Haralick idm ws=7', 'skewness-Haralick idm ws=7', 'var-Haralick idm ws=7'),
    ('kurtosis-Haralick idm ws=5', 'median-Haralick idm ws=5', 'skewness-Haralick idm ws=5', 'var-Haralick idm ws=5'),
    ('kurtosis-Haralick idm ws=3', 'median-Haralick idm ws=3', 'skewness-Haralick idm ws=3', 'var-Haralick idm ws=3'),
    ('kurtosis-Haralick idm ws=11', 'median-Haralick idm ws=11', 'skewness-Haralick idm ws=11', 'var-Haralick idm ws=11'),
    ('kurtosis-Haralick entropy ws=9', 'median-Haralick entropy ws=9', 'skewness-Haralick entropy ws=9', 'var-Haralick entropy ws=9'),
    ('kurtosis-Haralick entropy ws=7', 'median-Haralick entropy ws=7', 'skewness-Haralick entropy ws=7', 'var-Haralick entropy ws=7'),
    ('kurtosis-Haralick entropy ws=5', 'median-Haralick entropy ws=5', 'skewness-Haralick entropy ws=5', 'var-Haralick entropy ws=5'),
    ('kurtosis-Haralick entropy ws=3', 'median-Haralick entropy ws=3', 'skewness-Haralick entropy ws=3', 'var-Haralick entropy ws=3'),
    ('kurtosis-Haralick entropy ws=11', 'median-Haralick entropy ws=11', 'skewness-Haralick entropy ws=11', 'var-Haralick entropy ws=11'),
    ('kurtosis-Haralick energy ws=9', 'median-Haralick energy ws=9', 'skewness-Haralick energy ws=9', 'var-Haralick energy ws=9'),
    ('kurtosis-Haralick energy ws=7', 'median-Haralick energy ws=7', 'skewness-Haralick energy ws=7', 'var-Haralick energy ws=7'),
    ('kurtosis-Haralick energy ws=5', 'median-Haralick energy ws=5', 'skewness-Haralick energy ws=5', 'var-Haralick energy ws=5'),
    ('kurtosis-Haralick energy ws=3', 'median-Haralick energy ws=3', 'skewness-Haralick energy ws=3', 'var-Haralick energy ws=3'),
    ('kurtosis-Haralick energy ws=11', 'median-Haralick energy ws=11', 'skewness-Haralick energy ws=11', 'var-Haralick energy ws=11'),
    ('kurtosis-Haralick diff-var ws=9', 'median-Haralick diff-var ws=9', 'skewness-Haralick diff-var ws=9', 'var-Haralick diff-var ws=9'),
    ('kurtosis-Haralick diff-var ws=7', 'median-Haralick diff-var ws=7', 'skewness-Haralick diff-var ws=7', 'var-Haralick diff-var ws=7'),
    ('kurtosis-Haralick diff-var ws=5', 'median-Haralick diff-var ws=5', 'skewness-Haralick diff-var ws=5', 'var-Haralick diff-var ws=5'),
    ('kurtosis-Haralick diff-var ws=3', 'median-Haralick diff-var ws=3', 'skewness-Haralick diff-var ws=3', 'var-Haralick diff-var ws=3'),
    ('kurtosis-Haralick diff-var ws=11', 'median-Haralick diff-var ws=11', 'skewness-Haralick diff-var ws=11', 'var-Haralick diff-var ws=11'),
    ('kurtosis-Haralick diff-ent ws=9', 'median-Haralick diff-ent ws=9', 'skewness-Haralick diff-ent ws=9', 'var-Haralick diff-ent ws=9'),
    ('kurtosis-Haralick diff-ent ws=7', 'median-Haralick diff-ent ws=7', 'skewness-Haralick diff-ent ws=7', 'var-Haralick diff-ent ws=7'),
    ('kurtosis-Haralick diff-ent ws=5', 'median-Haralick diff-ent ws=5', 'skewness-Haralick diff-ent ws=5', 'var-Haralick diff-ent ws=5'),
    ('kurtosis-Haralick diff-ent ws=3', 'median-Haralick diff-ent ws=3', 'skewness-Haralick diff-ent ws=3', 'var-Haralick diff-ent ws=3'),
    ('kurtosis-Haralick diff-ent ws=11', 'median-Haralick diff-ent ws=11', 'skewness-Haralick diff-ent ws=11', 'var-Haralick diff-ent ws=11'),
    ('kurtosis-Haralick diff-av ws=9', 'median-Haralick diff-av ws=9', 'skewness-Haralick diff-av ws=9', 'var-Haralick diff-av ws=9'),
    ('kurtosis-Haralick diff-av ws=7', 'median-Haralick diff-av ws=7', 'skewness-Haralick diff-av ws=7', 'var-Haralick diff-av ws=7'),
    ('kurtosis-Haralick diff-av ws=5', 'median-Haralick diff-av ws=5', 'skewness-Haralick diff-av ws=5', 'var-Haralick diff-av ws=5'),
    ('kurtosis-Haralick diff-av ws=3', 'median-Haralick diff-av ws=3', 'skewness-Haralick diff-av ws=3', 'var-Haralick diff-av ws=3'),
    ('kurtosis-Haralick diff-av ws=11', 'median-Haralick diff-av ws=11', 'skewness-Haralick diff-av ws=11', 'var-Haralick diff-av ws=11'),
    ('kurtosis-Haralick correlation ws=9', 'median-Haralick correlation ws=9', 'skewness-Haralick correlation ws=9', 'var-Haralick correlation ws=9'),
    ('kurtosis-Haralick correlation ws=7', 'median-Haralick correlation ws=7', 'skewness-Haralick correlation ws=7', 'var-Haralick correlation ws=7'),
    ('kurtosis-Haralick correlation ws=5', 'median-Haralick correlation ws=5', 'skewness-Haralick correlation ws=5', 'var-Haralick correlation ws=5'),
    ('kurtosis-Haralick correlation ws=3', 'median-Haralick correlation ws=3', 'skewness-Haralick correlation ws=3', 'var-Haralick correlation ws=3'),
    ('kurtosis-Haralick correlation ws=11', 'median-Haralick correlation ws=11', 'skewness-Haralick correlation ws=11', 'var-Haralick correlation ws=11'),
    ('kurtosis-Gray std_dev ws=9', 'median-Gray std_dev ws=9', 'skewness-Gray std_dev ws=9', 'var-Gray std_dev ws=9'),
    ('kurtosis-Gray std_dev ws=7', 'median-Gray std_dev ws=7', 'skewness-Gray std_dev ws=7', 'var-Gray std_dev ws=7'),
    ('kurtosis-Gray std_dev ws=5', 'median-Gray std_dev ws=5', 'skewness-Gray std_dev ws=5', 'var-Gray std_dev ws=5'),
    ('kurtosis-Gray std_dev ws=3', 'median-Gray std_dev ws=3', 'skewness-Gray std_dev ws=3', 'var-Gray std_dev ws=3'),
    ('kurtosis-Gray std_dev ws=11', 'median-Gray std_dev ws=11', 'skewness-Gray std_dev ws=11', 'var-Gray std_dev ws=11'),
    ('kurtosis-Gray range ws=9', 'median-Gray range ws=9', 'skewness-Gray range ws=9', 'var-Gray range ws=9'),
    ('kurtosis-Gray range ws=7', 'median-Gray range ws=7', 'skewness-Gray range ws=7', 'var-Gray range ws=7'),
    ('kurtosis-Gray range ws=5', 'median-Gray range ws=5', 'skewness-Gray range ws=5', 'var-Gray range ws=5'),
    ('kurtosis-Gray range ws=3', 'median-Gray range ws=3', 'skewness-Gray range ws=3', 'var-Gray range ws=3'),
    ('kurtosis-Gray range ws=11', 'median-Gray range ws=11', 'skewness-Gray range ws=11', 'var-Gray range ws=11'),
    ('kurtosis-Gray median ws=9', 'median-Gray median ws=9', 'skewness-Gray median ws=9', 'var-Gray median ws=9'),
    ('kurtosis-Gray median ws=7', 'median-Gray median ws=7', 'skewness-Gray median ws=7', 'var-Gray median ws=7'),
    ('kurtosis-Gray median ws=5', 'median-Gray median ws=5', 'skewness-Gray median ws=5', 'var-Gray median ws=5'),
    ('kurtosis-Gray median ws=3', 'median-Gray median ws=3', 'skewness-Gray median ws=3', 'var-Gray median ws=3'),
    ('kurtosis-Gray median ws=11', 'median-Gray median ws=11', 'skewness-Gray median ws=11', 'var-Gray median ws=11'),
    ('kurtosis-Gray mean ws=9', 'median-Gray mean ws=9', 'skewness-Gray mean ws=9', 'var-Gray mean ws=9'),
    ('kurtosis-Gray mean ws=7', 'median-Gray mean ws=7', 'skewness-Gray mean ws=7', 'var-Gray mean ws=7'),
    ('kurtosis-Gray mean ws=5', 'median-Gray mean ws=5', 'skewness-Gray mean ws=5', 'var-Gray mean ws=5'),
    ('kurtosis-Gray mean ws=3', 'median-Gray mean ws=3', 'skewness-Gray mean ws=3', 'var-Gray mean ws=3'),
    ('kurtosis-Gray mean ws=11', 'median-Gray mean ws=11', 'skewness-Gray mean ws=11', 'var-Gray mean ws=11'),
    ('kurtosis-Gradient z', 'median-Gradient z', 'skewness-Gradient z', 'var-Gradient z'),
    ('kurtosis-Gradient y', 'median-Gradient y', 'skewness-Gradient y', 'var-Gradient y'),
    ('kurtosis-Gradient x', 'median-Gradient x', 'skewness-Gradient x', 'var-Gradient x'),
    ('kurtosis-Gradient sobelzy', 'median-Gradient sobelzy', 'skewness-Gradient sobelzy', 'var-Gradient sobelzy'),
    ('kurtosis-Gradient sobelzx', 'median-Gradient sobelzx', 'skewness-Gradient sobelzx', 'var-Gradient sobelzx'),
    ('kurtosis-Gradient sobelz', 'median-Gradient sobelz', 'skewness-Gradient sobelz', 'var-Gradient sobelz'),
    ('kurtosis-Gradient sobelyz', 'median-Gradient sobelyz', 'skewness-Gradient sobelyz', 'var-Gradient sobelyz'),
    ('kurtosis-Gradient sobelyx', 'median-Gradient sobelyx', 'skewness-Gradient sobelyx', 'var-Gradient sobelyx'),
    ('kurtosis-Gradient sobely', 'median-Gradient sobely', 'skewness-Gradient sobely', 'var-Gradient sobely'),
    ('kurtosis-Gradient sobelxz', 'median-Gradient sobelxz', 'skewness-Gradient sobelxz', 'var-Gradient sobelxz'),
    ('kurtosis-Gradient sobelxy', 'median-Gradient sobelxy', 'skewness-Gradient sobelxy', 'var-Gradient sobelxy'),
    ('kurtosis-Gradient sobelx', 'median-Gradient sobelx', 'skewness-Gradient sobelx', 'var-Gradient sobelx'),
    ('kurtosis-Gradient magnitude', 'median-Gradient magnitude', 'skewness-Gradient magnitude', 'var-Gradient magnitude'),
    ('kurtosis-Gabor XY-?=2.749, XZ-?=2.356, ?=3.261, BW=1', 'median-Gabor XY-?=2.749, XZ-?=2.356, ?=3.261, BW=1', 'skewness-Gabor XY-?=2.749, XZ-?=2.356, ?=3.261, BW=1', 'var-Gabor XY-?=2.749, XZ-?=2.356, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=2.749, XZ-?=2.356, ?=2.668, BW=1', 'median-Gabor XY-?=2.749, XZ-?=2.356, ?=2.668, BW=1', 'skewness-Gabor XY-?=2.749, XZ-?=2.356, ?=2.668, BW=1', 'var-Gabor XY-?=2.749, XZ-?=2.356, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=2.749, XZ-?=2.356, ?=2.075, BW=1', 'median-Gabor XY-?=2.749, XZ-?=2.356, ?=2.075, BW=1', 'skewness-Gabor XY-?=2.749, XZ-?=2.356, ?=2.075, BW=1', 'var-Gabor XY-?=2.749, XZ-?=2.356, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=2.749, XZ-?=2.356, ?=1.482, BW=1', 'median-Gabor XY-?=2.749, XZ-?=2.356, ?=1.482, BW=1', 'skewness-Gabor XY-?=2.749, XZ-?=2.356, ?=1.482, BW=1', 'var-Gabor XY-?=2.749, XZ-?=2.356, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=2.749, XZ-?=2.356, ?=0.880, BW=1', 'median-Gabor XY-?=2.749, XZ-?=2.356, ?=0.880, BW=1', 'skewness-Gabor XY-?=2.749, XZ-?=2.356, ?=0.880, BW=1', 'var-Gabor XY-?=2.749, XZ-?=2.356, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=2.749, XZ-?=1.571, ?=3.261, BW=1', 'median-Gabor XY-?=2.749, XZ-?=1.571, ?=3.261, BW=1', 'skewness-Gabor XY-?=2.749, XZ-?=1.571, ?=3.261, BW=1', 'var-Gabor XY-?=2.749, XZ-?=1.571, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=2.749, XZ-?=1.571, ?=2.668, BW=1', 'median-Gabor XY-?=2.749, XZ-?=1.571, ?=2.668, BW=1', 'skewness-Gabor XY-?=2.749, XZ-?=1.571, ?=2.668, BW=1', 'var-Gabor XY-?=2.749, XZ-?=1.571, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=2.749, XZ-?=1.571, ?=2.075, BW=1', 'median-Gabor XY-?=2.749, XZ-?=1.571, ?=2.075, BW=1', 'skewness-Gabor XY-?=2.749, XZ-?=1.571, ?=2.075, BW=1', 'var-Gabor XY-?=2.749, XZ-?=1.571, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=2.749, XZ-?=1.571, ?=1.482, BW=1', 'median-Gabor XY-?=2.749, XZ-?=1.571, ?=1.482, BW=1', 'skewness-Gabor XY-?=2.749, XZ-?=1.571, ?=1.482, BW=1', 'var-Gabor XY-?=2.749, XZ-?=1.571, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=2.749, XZ-?=1.571, ?=0.880, BW=1', 'median-Gabor XY-?=2.749, XZ-?=1.571, ?=0.880, BW=1', 'skewness-Gabor XY-?=2.749, XZ-?=1.571, ?=0.880, BW=1', 'var-Gabor XY-?=2.749, XZ-?=1.571, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=2.749, XZ-?=0.785, ?=3.261, BW=1', 'median-Gabor XY-?=2.749, XZ-?=0.785, ?=3.261, BW=1', 'skewness-Gabor XY-?=2.749, XZ-?=0.785, ?=3.261, BW=1', 'var-Gabor XY-?=2.749, XZ-?=0.785, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=2.749, XZ-?=0.785, ?=2.668, BW=1', 'median-Gabor XY-?=2.749, XZ-?=0.785, ?=2.668, BW=1', 'skewness-Gabor XY-?=2.749, XZ-?=0.785, ?=2.668, BW=1', 'var-Gabor XY-?=2.749, XZ-?=0.785, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=2.749, XZ-?=0.785, ?=2.075, BW=1', 'median-Gabor XY-?=2.749, XZ-?=0.785, ?=2.075, BW=1', 'skewness-Gabor XY-?=2.749, XZ-?=0.785, ?=2.075, BW=1', 'var-Gabor XY-?=2.749, XZ-?=0.785, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=2.749, XZ-?=0.785, ?=1.482, BW=1', 'median-Gabor XY-?=2.749, XZ-?=0.785, ?=1.482, BW=1', 'skewness-Gabor XY-?=2.749, XZ-?=0.785, ?=1.482, BW=1', 'var-Gabor XY-?=2.749, XZ-?=0.785, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=2.749, XZ-?=0.785, ?=0.880, BW=1', 'median-Gabor XY-?=2.749, XZ-?=0.785, ?=0.880, BW=1', 'skewness-Gabor XY-?=2.749, XZ-?=0.785, ?=0.880, BW=1', 'var-Gabor XY-?=2.749, XZ-?=0.785, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=2.749, XZ-?=0.000, ?=3.261, BW=1', 'median-Gabor XY-?=2.749, XZ-?=0.000, ?=3.261, BW=1', 'skewness-Gabor XY-?=2.749, XZ-?=0.000, ?=3.261, BW=1', 'var-Gabor XY-?=2.749, XZ-?=0.000, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=2.749, XZ-?=0.000, ?=2.668, BW=1', 'median-Gabor XY-?=2.749, XZ-?=0.000, ?=2.668, BW=1', 'skewness-Gabor XY-?=2.749, XZ-?=0.000, ?=2.668, BW=1', 'var-Gabor XY-?=2.749, XZ-?=0.000, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=2.749, XZ-?=0.000, ?=2.075, BW=1', 'median-Gabor XY-?=2.749, XZ-?=0.000, ?=2.075, BW=1', 'skewness-Gabor XY-?=2.749, XZ-?=0.000, ?=2.075, BW=1', 'var-Gabor XY-?=2.749, XZ-?=0.000, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=2.749, XZ-?=0.000, ?=1.482, BW=1', 'median-Gabor XY-?=2.749, XZ-?=0.000, ?=1.482, BW=1', 'skewness-Gabor XY-?=2.749, XZ-?=0.000, ?=1.482, BW=1', 'var-Gabor XY-?=2.749, XZ-?=0.000, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=2.749, XZ-?=0.000, ?=0.880, BW=1', 'median-Gabor XY-?=2.749, XZ-?=0.000, ?=0.880, BW=1', 'skewness-Gabor XY-?=2.749, XZ-?=0.000, ?=0.880, BW=1', 'var-Gabor XY-?=2.749, XZ-?=0.000, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=2.356, XZ-?=2.356, ?=3.261, BW=1', 'median-Gabor XY-?=2.356, XZ-?=2.356, ?=3.261, BW=1', 'skewness-Gabor XY-?=2.356, XZ-?=2.356, ?=3.261, BW=1', 'var-Gabor XY-?=2.356, XZ-?=2.356, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=2.356, XZ-?=2.356, ?=2.668, BW=1', 'median-Gabor XY-?=2.356, XZ-?=2.356, ?=2.668, BW=1', 'skewness-Gabor XY-?=2.356, XZ-?=2.356, ?=2.668, BW=1', 'var-Gabor XY-?=2.356, XZ-?=2.356, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=2.356, XZ-?=2.356, ?=2.075, BW=1', 'median-Gabor XY-?=2.356, XZ-?=2.356, ?=2.075, BW=1', 'skewness-Gabor XY-?=2.356, XZ-?=2.356, ?=2.075, BW=1', 'var-Gabor XY-?=2.356, XZ-?=2.356, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=2.356, XZ-?=2.356, ?=1.482, BW=1', 'median-Gabor XY-?=2.356, XZ-?=2.356, ?=1.482, BW=1', 'skewness-Gabor XY-?=2.356, XZ-?=2.356, ?=1.482, BW=1', 'var-Gabor XY-?=2.356, XZ-?=2.356, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=2.356, XZ-?=2.356, ?=0.880, BW=1', 'median-Gabor XY-?=2.356, XZ-?=2.356, ?=0.880, BW=1', 'skewness-Gabor XY-?=2.356, XZ-?=2.356, ?=0.880, BW=1', 'var-Gabor XY-?=2.356, XZ-?=2.356, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=2.356, XZ-?=1.571, ?=3.261, BW=1', 'median-Gabor XY-?=2.356, XZ-?=1.571, ?=3.261, BW=1', 'skewness-Gabor XY-?=2.356, XZ-?=1.571, ?=3.261, BW=1', 'var-Gabor XY-?=2.356, XZ-?=1.571, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=2.356, XZ-?=1.571, ?=2.668, BW=1', 'median-Gabor XY-?=2.356, XZ-?=1.571, ?=2.668, BW=1', 'skewness-Gabor XY-?=2.356, XZ-?=1.571, ?=2.668, BW=1', 'var-Gabor XY-?=2.356, XZ-?=1.571, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=2.356, XZ-?=1.571, ?=2.075, BW=1', 'median-Gabor XY-?=2.356, XZ-?=1.571, ?=2.075, BW=1', 'skewness-Gabor XY-?=2.356, XZ-?=1.571, ?=2.075, BW=1', 'var-Gabor XY-?=2.356, XZ-?=1.571, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=2.356, XZ-?=1.571, ?=1.482, BW=1', 'median-Gabor XY-?=2.356, XZ-?=1.571, ?=1.482, BW=1', 'skewness-Gabor XY-?=2.356, XZ-?=1.571, ?=1.482, BW=1', 'var-Gabor XY-?=2.356, XZ-?=1.571, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=2.356, XZ-?=1.571, ?=0.880, BW=1', 'median-Gabor XY-?=2.356, XZ-?=1.571, ?=0.880, BW=1', 'skewness-Gabor XY-?=2.356, XZ-?=1.571, ?=0.880, BW=1', 'var-Gabor XY-?=2.356, XZ-?=1.571, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=2.356, XZ-?=0.785, ?=3.261, BW=1', 'median-Gabor XY-?=2.356, XZ-?=0.785, ?=3.261, BW=1', 'skewness-Gabor XY-?=2.356, XZ-?=0.785, ?=3.261, BW=1', 'var-Gabor XY-?=2.356, XZ-?=0.785, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=2.356, XZ-?=0.785, ?=2.668, BW=1', 'median-Gabor XY-?=2.356, XZ-?=0.785, ?=2.668, BW=1', 'skewness-Gabor XY-?=2.356, XZ-?=0.785, ?=2.668, BW=1', 'var-Gabor XY-?=2.356, XZ-?=0.785, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=2.356, XZ-?=0.785, ?=2.075, BW=1', 'median-Gabor XY-?=2.356, XZ-?=0.785, ?=2.075, BW=1', 'skewness-Gabor XY-?=2.356, XZ-?=0.785, ?=2.075, BW=1', 'var-Gabor XY-?=2.356, XZ-?=0.785, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=2.356, XZ-?=0.785, ?=1.482, BW=1', 'median-Gabor XY-?=2.356, XZ-?=0.785, ?=1.482, BW=1', 'skewness-Gabor XY-?=2.356, XZ-?=0.785, ?=1.482, BW=1', 'var-Gabor XY-?=2.356, XZ-?=0.785, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=2.356, XZ-?=0.785, ?=0.880, BW=1', 'median-Gabor XY-?=2.356, XZ-?=0.785, ?=0.880, BW=1', 'skewness-Gabor XY-?=2.356, XZ-?=0.785, ?=0.880, BW=1', 'var-Gabor XY-?=2.356, XZ-?=0.785, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=2.356, XZ-?=0.000, ?=3.261, BW=1', 'median-Gabor XY-?=2.356, XZ-?=0.000, ?=3.261, BW=1', 'skewness-Gabor XY-?=2.356, XZ-?=0.000, ?=3.261, BW=1', 'var-Gabor XY-?=2.356, XZ-?=0.000, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=2.356, XZ-?=0.000, ?=2.668, BW=1', 'median-Gabor XY-?=2.356, XZ-?=0.000, ?=2.668, BW=1', 'skewness-Gabor XY-?=2.356, XZ-?=0.000, ?=2.668, BW=1', 'var-Gabor XY-?=2.356, XZ-?=0.000, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=2.356, XZ-?=0.000, ?=2.075, BW=1', 'median-Gabor XY-?=2.356, XZ-?=0.000, ?=2.075, BW=1', 'skewness-Gabor XY-?=2.356, XZ-?=0.000, ?=2.075, BW=1', 'var-Gabor XY-?=2.356, XZ-?=0.000, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=2.356, XZ-?=0.000, ?=1.482, BW=1', 'median-Gabor XY-?=2.356, XZ-?=0.000, ?=1.482, BW=1', 'skewness-Gabor XY-?=2.356, XZ-?=0.000, ?=1.482, BW=1', 'var-Gabor XY-?=2.356, XZ-?=0.000, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=2.356, XZ-?=0.000, ?=0.880, BW=1', 'median-Gabor XY-?=2.356, XZ-?=0.000, ?=0.880, BW=1', 'skewness-Gabor XY-?=2.356, XZ-?=0.000, ?=0.880, BW=1', 'var-Gabor XY-?=2.356, XZ-?=0.000, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=1.963, XZ-?=2.356, ?=3.261, BW=1', 'median-Gabor XY-?=1.963, XZ-?=2.356, ?=3.261, BW=1', 'skewness-Gabor XY-?=1.963, XZ-?=2.356, ?=3.261, BW=1', 'var-Gabor XY-?=1.963, XZ-?=2.356, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=1.963, XZ-?=2.356, ?=2.668, BW=1', 'median-Gabor XY-?=1.963, XZ-?=2.356, ?=2.668, BW=1', 'skewness-Gabor XY-?=1.963, XZ-?=2.356, ?=2.668, BW=1', 'var-Gabor XY-?=1.963, XZ-?=2.356, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=1.963, XZ-?=2.356, ?=2.075, BW=1', 'median-Gabor XY-?=1.963, XZ-?=2.356, ?=2.075, BW=1', 'skewness-Gabor XY-?=1.963, XZ-?=2.356, ?=2.075, BW=1', 'var-Gabor XY-?=1.963, XZ-?=2.356, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=1.963, XZ-?=2.356, ?=1.482, BW=1', 'median-Gabor XY-?=1.963, XZ-?=2.356, ?=1.482, BW=1', 'skewness-Gabor XY-?=1.963, XZ-?=2.356, ?=1.482, BW=1', 'var-Gabor XY-?=1.963, XZ-?=2.356, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=1.963, XZ-?=2.356, ?=0.880, BW=1', 'median-Gabor XY-?=1.963, XZ-?=2.356, ?=0.880, BW=1', 'skewness-Gabor XY-?=1.963, XZ-?=2.356, ?=0.880, BW=1', 'var-Gabor XY-?=1.963, XZ-?=2.356, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=1.963, XZ-?=1.571, ?=3.261, BW=1', 'median-Gabor XY-?=1.963, XZ-?=1.571, ?=3.261, BW=1', 'skewness-Gabor XY-?=1.963, XZ-?=1.571, ?=3.261, BW=1', 'var-Gabor XY-?=1.963, XZ-?=1.571, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=1.963, XZ-?=1.571, ?=2.668, BW=1', 'median-Gabor XY-?=1.963, XZ-?=1.571, ?=2.668, BW=1', 'skewness-Gabor XY-?=1.963, XZ-?=1.571, ?=2.668, BW=1', 'var-Gabor XY-?=1.963, XZ-?=1.571, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=1.963, XZ-?=1.571, ?=2.075, BW=1', 'median-Gabor XY-?=1.963, XZ-?=1.571, ?=2.075, BW=1', 'skewness-Gabor XY-?=1.963, XZ-?=1.571, ?=2.075, BW=1', 'var-Gabor XY-?=1.963, XZ-?=1.571, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=1.963, XZ-?=1.571, ?=1.482, BW=1', 'median-Gabor XY-?=1.963, XZ-?=1.571, ?=1.482, BW=1', 'skewness-Gabor XY-?=1.963, XZ-?=1.571, ?=1.482, BW=1', 'var-Gabor XY-?=1.963, XZ-?=1.571, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=1.963, XZ-?=1.571, ?=0.880, BW=1', 'median-Gabor XY-?=1.963, XZ-?=1.571, ?=0.880, BW=1', 'skewness-Gabor XY-?=1.963, XZ-?=1.571, ?=0.880, BW=1', 'var-Gabor XY-?=1.963, XZ-?=1.571, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=1.963, XZ-?=0.785, ?=3.261, BW=1', 'median-Gabor XY-?=1.963, XZ-?=0.785, ?=3.261, BW=1', 'skewness-Gabor XY-?=1.963, XZ-?=0.785, ?=3.261, BW=1', 'var-Gabor XY-?=1.963, XZ-?=0.785, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=1.963, XZ-?=0.785, ?=2.668, BW=1', 'median-Gabor XY-?=1.963, XZ-?=0.785, ?=2.668, BW=1', 'skewness-Gabor XY-?=1.963, XZ-?=0.785, ?=2.668, BW=1', 'var-Gabor XY-?=1.963, XZ-?=0.785, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=1.963, XZ-?=0.785, ?=2.075, BW=1', 'median-Gabor XY-?=1.963, XZ-?=0.785, ?=2.075, BW=1', 'skewness-Gabor XY-?=1.963, XZ-?=0.785, ?=2.075, BW=1', 'var-Gabor XY-?=1.963, XZ-?=0.785, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=1.963, XZ-?=0.785, ?=1.482, BW=1', 'median-Gabor XY-?=1.963, XZ-?=0.785, ?=1.482, BW=1', 'skewness-Gabor XY-?=1.963, XZ-?=0.785, ?=1.482, BW=1', 'var-Gabor XY-?=1.963, XZ-?=0.785, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=1.963, XZ-?=0.785, ?=0.880, BW=1', 'median-Gabor XY-?=1.963, XZ-?=0.785, ?=0.880, BW=1', 'skewness-Gabor XY-?=1.963, XZ-?=0.785, ?=0.880, BW=1', 'var-Gabor XY-?=1.963, XZ-?=0.785, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=1.963, XZ-?=0.000, ?=3.261, BW=1', 'median-Gabor XY-?=1.963, XZ-?=0.000, ?=3.261, BW=1', 'skewness-Gabor XY-?=1.963, XZ-?=0.000, ?=3.261, BW=1', 'var-Gabor XY-?=1.963, XZ-?=0.000, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=1.963, XZ-?=0.000, ?=2.668, BW=1', 'median-Gabor XY-?=1.963, XZ-?=0.000, ?=2.668, BW=1', 'skewness-Gabor XY-?=1.963, XZ-?=0.000, ?=2.668, BW=1', 'var-Gabor XY-?=1.963, XZ-?=0.000, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=1.963, XZ-?=0.000, ?=2.075, BW=1', 'median-Gabor XY-?=1.963, XZ-?=0.000, ?=2.075, BW=1', 'skewness-Gabor XY-?=1.963, XZ-?=0.000, ?=2.075, BW=1', 'var-Gabor XY-?=1.963, XZ-?=0.000, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=1.963, XZ-?=0.000, ?=1.482, BW=1', 'median-Gabor XY-?=1.963, XZ-?=0.000, ?=1.482, BW=1', 'skewness-Gabor XY-?=1.963, XZ-?=0.000, ?=1.482, BW=1', 'var-Gabor XY-?=1.963, XZ-?=0.000, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=1.963, XZ-?=0.000, ?=0.880, BW=1', 'median-Gabor XY-?=1.963, XZ-?=0.000, ?=0.880, BW=1', 'skewness-Gabor XY-?=1.963, XZ-?=0.000, ?=0.880, BW=1', 'var-Gabor XY-?=1.963, XZ-?=0.000, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=1.571, XZ-?=2.356, ?=3.261, BW=1', 'median-Gabor XY-?=1.571, XZ-?=2.356, ?=3.261, BW=1', 'skewness-Gabor XY-?=1.571, XZ-?=2.356, ?=3.261, BW=1', 'var-Gabor XY-?=1.571, XZ-?=2.356, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=1.571, XZ-?=2.356, ?=2.668, BW=1', 'median-Gabor XY-?=1.571, XZ-?=2.356, ?=2.668, BW=1', 'skewness-Gabor XY-?=1.571, XZ-?=2.356, ?=2.668, BW=1', 'var-Gabor XY-?=1.571, XZ-?=2.356, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=1.571, XZ-?=2.356, ?=2.075, BW=1', 'median-Gabor XY-?=1.571, XZ-?=2.356, ?=2.075, BW=1', 'skewness-Gabor XY-?=1.571, XZ-?=2.356, ?=2.075, BW=1', 'var-Gabor XY-?=1.571, XZ-?=2.356, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=1.571, XZ-?=2.356, ?=1.482, BW=1', 'median-Gabor XY-?=1.571, XZ-?=2.356, ?=1.482, BW=1', 'skewness-Gabor XY-?=1.571, XZ-?=2.356, ?=1.482, BW=1', 'var-Gabor XY-?=1.571, XZ-?=2.356, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=1.571, XZ-?=2.356, ?=0.880, BW=1', 'median-Gabor XY-?=1.571, XZ-?=2.356, ?=0.880, BW=1', 'skewness-Gabor XY-?=1.571, XZ-?=2.356, ?=0.880, BW=1', 'var-Gabor XY-?=1.571, XZ-?=2.356, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=1.571, XZ-?=1.571, ?=3.261, BW=1', 'median-Gabor XY-?=1.571, XZ-?=1.571, ?=3.261, BW=1', 'skewness-Gabor XY-?=1.571, XZ-?=1.571, ?=3.261, BW=1', 'var-Gabor XY-?=1.571, XZ-?=1.571, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=1.571, XZ-?=1.571, ?=2.668, BW=1', 'median-Gabor XY-?=1.571, XZ-?=1.571, ?=2.668, BW=1', 'skewness-Gabor XY-?=1.571, XZ-?=1.571, ?=2.668, BW=1', 'var-Gabor XY-?=1.571, XZ-?=1.571, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=1.571, XZ-?=1.571, ?=2.075, BW=1', 'median-Gabor XY-?=1.571, XZ-?=1.571, ?=2.075, BW=1', 'skewness-Gabor XY-?=1.571, XZ-?=1.571, ?=2.075, BW=1', 'var-Gabor XY-?=1.571, XZ-?=1.571, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=1.571, XZ-?=1.571, ?=1.482, BW=1', 'median-Gabor XY-?=1.571, XZ-?=1.571, ?=1.482, BW=1', 'skewness-Gabor XY-?=1.571, XZ-?=1.571, ?=1.482, BW=1', 'var-Gabor XY-?=1.571, XZ-?=1.571, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=1.571, XZ-?=1.571, ?=0.880, BW=1', 'median-Gabor XY-?=1.571, XZ-?=1.571, ?=0.880, BW=1', 'skewness-Gabor XY-?=1.571, XZ-?=1.571, ?=0.880, BW=1', 'var-Gabor XY-?=1.571, XZ-?=1.571, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=1.571, XZ-?=0.785, ?=3.261, BW=1', 'median-Gabor XY-?=1.571, XZ-?=0.785, ?=3.261, BW=1', 'skewness-Gabor XY-?=1.571, XZ-?=0.785, ?=3.261, BW=1', 'var-Gabor XY-?=1.571, XZ-?=0.785, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=1.571, XZ-?=0.785, ?=2.668, BW=1', 'median-Gabor XY-?=1.571, XZ-?=0.785, ?=2.668, BW=1', 'skewness-Gabor XY-?=1.571, XZ-?=0.785, ?=2.668, BW=1', 'var-Gabor XY-?=1.571, XZ-?=0.785, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=1.571, XZ-?=0.785, ?=2.075, BW=1', 'median-Gabor XY-?=1.571, XZ-?=0.785, ?=2.075, BW=1', 'skewness-Gabor XY-?=1.571, XZ-?=0.785, ?=2.075, BW=1', 'var-Gabor XY-?=1.571, XZ-?=0.785, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=1.571, XZ-?=0.785, ?=1.482, BW=1', 'median-Gabor XY-?=1.571, XZ-?=0.785, ?=1.482, BW=1', 'skewness-Gabor XY-?=1.571, XZ-?=0.785, ?=1.482, BW=1', 'var-Gabor XY-?=1.571, XZ-?=0.785, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=1.571, XZ-?=0.785, ?=0.880, BW=1', 'median-Gabor XY-?=1.571, XZ-?=0.785, ?=0.880, BW=1', 'skewness-Gabor XY-?=1.571, XZ-?=0.785, ?=0.880, BW=1', 'var-Gabor XY-?=1.571, XZ-?=0.785, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=1.571, XZ-?=0.000, ?=3.261, BW=1', 'median-Gabor XY-?=1.571, XZ-?=0.000, ?=3.261, BW=1', 'skewness-Gabor XY-?=1.571, XZ-?=0.000, ?=3.261, BW=1', 'var-Gabor XY-?=1.571, XZ-?=0.000, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=1.571, XZ-?=0.000, ?=2.668, BW=1', 'median-Gabor XY-?=1.571, XZ-?=0.000, ?=2.668, BW=1', 'skewness-Gabor XY-?=1.571, XZ-?=0.000, ?=2.668, BW=1', 'var-Gabor XY-?=1.571, XZ-?=0.000, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=1.571, XZ-?=0.000, ?=2.075, BW=1', 'median-Gabor XY-?=1.571, XZ-?=0.000, ?=2.075, BW=1', 'skewness-Gabor XY-?=1.571, XZ-?=0.000, ?=2.075, BW=1', 'var-Gabor XY-?=1.571, XZ-?=0.000, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=1.571, XZ-?=0.000, ?=1.482, BW=1', 'median-Gabor XY-?=1.571, XZ-?=0.000, ?=1.482, BW=1', 'skewness-Gabor XY-?=1.571, XZ-?=0.000, ?=1.482, BW=1', 'var-Gabor XY-?=1.571, XZ-?=0.000, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=1.571, XZ-?=0.000, ?=0.880, BW=1', 'median-Gabor XY-?=1.571, XZ-?=0.000, ?=0.880, BW=1', 'skewness-Gabor XY-?=1.571, XZ-?=0.000, ?=0.880, BW=1', 'var-Gabor XY-?=1.571, XZ-?=0.000, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=1.178, XZ-?=2.356, ?=3.261, BW=1', 'median-Gabor XY-?=1.178, XZ-?=2.356, ?=3.261, BW=1', 'skewness-Gabor XY-?=1.178, XZ-?=2.356, ?=3.261, BW=1', 'var-Gabor XY-?=1.178, XZ-?=2.356, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=1.178, XZ-?=2.356, ?=2.668, BW=1', 'median-Gabor XY-?=1.178, XZ-?=2.356, ?=2.668, BW=1', 'skewness-Gabor XY-?=1.178, XZ-?=2.356, ?=2.668, BW=1', 'var-Gabor XY-?=1.178, XZ-?=2.356, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=1.178, XZ-?=2.356, ?=2.075, BW=1', 'median-Gabor XY-?=1.178, XZ-?=2.356, ?=2.075, BW=1', 'skewness-Gabor XY-?=1.178, XZ-?=2.356, ?=2.075, BW=1', 'var-Gabor XY-?=1.178, XZ-?=2.356, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=1.178, XZ-?=2.356, ?=1.482, BW=1', 'median-Gabor XY-?=1.178, XZ-?=2.356, ?=1.482, BW=1', 'skewness-Gabor XY-?=1.178, XZ-?=2.356, ?=1.482, BW=1', 'var-Gabor XY-?=1.178, XZ-?=2.356, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=1.178, XZ-?=2.356, ?=0.880, BW=1', 'median-Gabor XY-?=1.178, XZ-?=2.356, ?=0.880, BW=1', 'skewness-Gabor XY-?=1.178, XZ-?=2.356, ?=0.880, BW=1', 'var-Gabor XY-?=1.178, XZ-?=2.356, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=1.178, XZ-?=1.571, ?=3.261, BW=1', 'median-Gabor XY-?=1.178, XZ-?=1.571, ?=3.261, BW=1', 'skewness-Gabor XY-?=1.178, XZ-?=1.571, ?=3.261, BW=1', 'var-Gabor XY-?=1.178, XZ-?=1.571, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=1.178, XZ-?=1.571, ?=2.668, BW=1', 'median-Gabor XY-?=1.178, XZ-?=1.571, ?=2.668, BW=1', 'skewness-Gabor XY-?=1.178, XZ-?=1.571, ?=2.668, BW=1', 'var-Gabor XY-?=1.178, XZ-?=1.571, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=1.178, XZ-?=1.571, ?=2.075, BW=1', 'median-Gabor XY-?=1.178, XZ-?=1.571, ?=2.075, BW=1', 'skewness-Gabor XY-?=1.178, XZ-?=1.571, ?=2.075, BW=1', 'var-Gabor XY-?=1.178, XZ-?=1.571, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=1.178, XZ-?=1.571, ?=1.482, BW=1', 'median-Gabor XY-?=1.178, XZ-?=1.571, ?=1.482, BW=1', 'skewness-Gabor XY-?=1.178, XZ-?=1.571, ?=1.482, BW=1', 'var-Gabor XY-?=1.178, XZ-?=1.571, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=1.178, XZ-?=1.571, ?=0.880, BW=1', 'median-Gabor XY-?=1.178, XZ-?=1.571, ?=0.880, BW=1', 'skewness-Gabor XY-?=1.178, XZ-?=1.571, ?=0.880, BW=1', 'var-Gabor XY-?=1.178, XZ-?=1.571, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=1.178, XZ-?=0.785, ?=3.261, BW=1', 'median-Gabor XY-?=1.178, XZ-?=0.785, ?=3.261, BW=1', 'skewness-Gabor XY-?=1.178, XZ-?=0.785, ?=3.261, BW=1', 'var-Gabor XY-?=1.178, XZ-?=0.785, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=1.178, XZ-?=0.785, ?=2.668, BW=1', 'median-Gabor XY-?=1.178, XZ-?=0.785, ?=2.668, BW=1', 'skewness-Gabor XY-?=1.178, XZ-?=0.785, ?=2.668, BW=1', 'var-Gabor XY-?=1.178, XZ-?=0.785, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=1.178, XZ-?=0.785, ?=2.075, BW=1', 'median-Gabor XY-?=1.178, XZ-?=0.785, ?=2.075, BW=1', 'skewness-Gabor XY-?=1.178, XZ-?=0.785, ?=2.075, BW=1', 'var-Gabor XY-?=1.178, XZ-?=0.785, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=1.178, XZ-?=0.785, ?=1.482, BW=1', 'median-Gabor XY-?=1.178, XZ-?=0.785, ?=1.482, BW=1', 'skewness-Gabor XY-?=1.178, XZ-?=0.785, ?=1.482, BW=1', 'var-Gabor XY-?=1.178, XZ-?=0.785, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=1.178, XZ-?=0.785, ?=0.880, BW=1', 'median-Gabor XY-?=1.178, XZ-?=0.785, ?=0.880, BW=1', 'skewness-Gabor XY-?=1.178, XZ-?=0.785, ?=0.880, BW=1', 'var-Gabor XY-?=1.178, XZ-?=0.785, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=1.178, XZ-?=0.000, ?=3.261, BW=1', 'median-Gabor XY-?=1.178, XZ-?=0.000, ?=3.261, BW=1', 'skewness-Gabor XY-?=1.178, XZ-?=0.000, ?=3.261, BW=1', 'var-Gabor XY-?=1.178, XZ-?=0.000, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=1.178, XZ-?=0.000, ?=2.668, BW=1', 'median-Gabor XY-?=1.178, XZ-?=0.000, ?=2.668, BW=1', 'skewness-Gabor XY-?=1.178, XZ-?=0.000, ?=2.668, BW=1', 'var-Gabor XY-?=1.178, XZ-?=0.000, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=1.178, XZ-?=0.000, ?=2.075, BW=1', 'median-Gabor XY-?=1.178, XZ-?=0.000, ?=2.075, BW=1', 'skewness-Gabor XY-?=1.178, XZ-?=0.000, ?=2.075, BW=1', 'var-Gabor XY-?=1.178, XZ-?=0.000, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=1.178, XZ-?=0.000, ?=1.482, BW=1', 'median-Gabor XY-?=1.178, XZ-?=0.000, ?=1.482, BW=1', 'skewness-Gabor XY-?=1.178, XZ-?=0.000, ?=1.482, BW=1', 'var-Gabor XY-?=1.178, XZ-?=0.000, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=1.178, XZ-?=0.000, ?=0.880, BW=1', 'median-Gabor XY-?=1.178, XZ-?=0.000, ?=0.880, BW=1', 'skewness-Gabor XY-?=1.178, XZ-?=0.000, ?=0.880, BW=1', 'var-Gabor XY-?=1.178, XZ-?=0.000, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=0.785, XZ-?=2.356, ?=3.261, BW=1', 'median-Gabor XY-?=0.785, XZ-?=2.356, ?=3.261, BW=1', 'skewness-Gabor XY-?=0.785, XZ-?=2.356, ?=3.261, BW=1', 'var-Gabor XY-?=0.785, XZ-?=2.356, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=0.785, XZ-?=2.356, ?=2.668, BW=1', 'median-Gabor XY-?=0.785, XZ-?=2.356, ?=2.668, BW=1', 'skewness-Gabor XY-?=0.785, XZ-?=2.356, ?=2.668, BW=1', 'var-Gabor XY-?=0.785, XZ-?=2.356, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=0.785, XZ-?=2.356, ?=2.075, BW=1', 'median-Gabor XY-?=0.785, XZ-?=2.356, ?=2.075, BW=1', 'skewness-Gabor XY-?=0.785, XZ-?=2.356, ?=2.075, BW=1', 'var-Gabor XY-?=0.785, XZ-?=2.356, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=0.785, XZ-?=2.356, ?=1.482, BW=1', 'median-Gabor XY-?=0.785, XZ-?=2.356, ?=1.482, BW=1', 'skewness-Gabor XY-?=0.785, XZ-?=2.356, ?=1.482, BW=1', 'var-Gabor XY-?=0.785, XZ-?=2.356, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=0.785, XZ-?=2.356, ?=0.880, BW=1', 'median-Gabor XY-?=0.785, XZ-?=2.356, ?=0.880, BW=1', 'skewness-Gabor XY-?=0.785, XZ-?=2.356, ?=0.880, BW=1', 'var-Gabor XY-?=0.785, XZ-?=2.356, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=0.785, XZ-?=1.571, ?=3.261, BW=1', 'median-Gabor XY-?=0.785, XZ-?=1.571, ?=3.261, BW=1', 'skewness-Gabor XY-?=0.785, XZ-?=1.571, ?=3.261, BW=1', 'var-Gabor XY-?=0.785, XZ-?=1.571, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=0.785, XZ-?=1.571, ?=2.668, BW=1', 'median-Gabor XY-?=0.785, XZ-?=1.571, ?=2.668, BW=1', 'skewness-Gabor XY-?=0.785, XZ-?=1.571, ?=2.668, BW=1', 'var-Gabor XY-?=0.785, XZ-?=1.571, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=0.785, XZ-?=1.571, ?=2.075, BW=1', 'median-Gabor XY-?=0.785, XZ-?=1.571, ?=2.075, BW=1', 'skewness-Gabor XY-?=0.785, XZ-?=1.571, ?=2.075, BW=1', 'var-Gabor XY-?=0.785, XZ-?=1.571, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=0.785, XZ-?=1.571, ?=1.482, BW=1', 'median-Gabor XY-?=0.785, XZ-?=1.571, ?=1.482, BW=1', 'skewness-Gabor XY-?=0.785, XZ-?=1.571, ?=1.482, BW=1', 'var-Gabor XY-?=0.785, XZ-?=1.571, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=0.785, XZ-?=1.571, ?=0.880, BW=1', 'median-Gabor XY-?=0.785, XZ-?=1.571, ?=0.880, BW=1', 'skewness-Gabor XY-?=0.785, XZ-?=1.571, ?=0.880, BW=1', 'var-Gabor XY-?=0.785, XZ-?=1.571, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=0.785, XZ-?=0.785, ?=3.261, BW=1', 'median-Gabor XY-?=0.785, XZ-?=0.785, ?=3.261, BW=1', 'skewness-Gabor XY-?=0.785, XZ-?=0.785, ?=3.261, BW=1', 'var-Gabor XY-?=0.785, XZ-?=0.785, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=0.785, XZ-?=0.785, ?=2.668, BW=1', 'median-Gabor XY-?=0.785, XZ-?=0.785, ?=2.668, BW=1', 'skewness-Gabor XY-?=0.785, XZ-?=0.785, ?=2.668, BW=1', 'var-Gabor XY-?=0.785, XZ-?=0.785, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=0.785, XZ-?=0.785, ?=2.075, BW=1', 'median-Gabor XY-?=0.785, XZ-?=0.785, ?=2.075, BW=1', 'skewness-Gabor XY-?=0.785, XZ-?=0.785, ?=2.075, BW=1', 'var-Gabor XY-?=0.785, XZ-?=0.785, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=0.785, XZ-?=0.785, ?=1.482, BW=1', 'median-Gabor XY-?=0.785, XZ-?=0.785, ?=1.482, BW=1', 'skewness-Gabor XY-?=0.785, XZ-?=0.785, ?=1.482, BW=1', 'var-Gabor XY-?=0.785, XZ-?=0.785, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=0.785, XZ-?=0.785, ?=0.880, BW=1', 'median-Gabor XY-?=0.785, XZ-?=0.785, ?=0.880, BW=1', 'skewness-Gabor XY-?=0.785, XZ-?=0.785, ?=0.880, BW=1', 'var-Gabor XY-?=0.785, XZ-?=0.785, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=0.785, XZ-?=0.000, ?=3.261, BW=1', 'median-Gabor XY-?=0.785, XZ-?=0.000, ?=3.261, BW=1', 'skewness-Gabor XY-?=0.785, XZ-?=0.000, ?=3.261, BW=1', 'var-Gabor XY-?=0.785, XZ-?=0.000, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=0.785, XZ-?=0.000, ?=2.668, BW=1', 'median-Gabor XY-?=0.785, XZ-?=0.000, ?=2.668, BW=1', 'skewness-Gabor XY-?=0.785, XZ-?=0.000, ?=2.668, BW=1', 'var-Gabor XY-?=0.785, XZ-?=0.000, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=0.785, XZ-?=0.000, ?=2.075, BW=1', 'median-Gabor XY-?=0.785, XZ-?=0.000, ?=2.075, BW=1', 'skewness-Gabor XY-?=0.785, XZ-?=0.000, ?=2.075, BW=1', 'var-Gabor XY-?=0.785, XZ-?=0.000, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=0.785, XZ-?=0.000, ?=1.482, BW=1', 'median-Gabor XY-?=0.785, XZ-?=0.000, ?=1.482, BW=1', 'skewness-Gabor XY-?=0.785, XZ-?=0.000, ?=1.482, BW=1', 'var-Gabor XY-?=0.785, XZ-?=0.000, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=0.785, XZ-?=0.000, ?=0.880, BW=1', 'median-Gabor XY-?=0.785, XZ-?=0.000, ?=0.880, BW=1', 'skewness-Gabor XY-?=0.785, XZ-?=0.000, ?=0.880, BW=1', 'var-Gabor XY-?=0.785, XZ-?=0.000, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=0.393, XZ-?=2.356, ?=3.261, BW=1', 'median-Gabor XY-?=0.393, XZ-?=2.356, ?=3.261, BW=1', 'skewness-Gabor XY-?=0.393, XZ-?=2.356, ?=3.261, BW=1', 'var-Gabor XY-?=0.393, XZ-?=2.356, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=0.393, XZ-?=2.356, ?=2.668, BW=1', 'median-Gabor XY-?=0.393, XZ-?=2.356, ?=2.668, BW=1', 'skewness-Gabor XY-?=0.393, XZ-?=2.356, ?=2.668, BW=1', 'var-Gabor XY-?=0.393, XZ-?=2.356, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=0.393, XZ-?=2.356, ?=2.075, BW=1', 'median-Gabor XY-?=0.393, XZ-?=2.356, ?=2.075, BW=1', 'skewness-Gabor XY-?=0.393, XZ-?=2.356, ?=2.075, BW=1', 'var-Gabor XY-?=0.393, XZ-?=2.356, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=0.393, XZ-?=2.356, ?=1.482, BW=1', 'median-Gabor XY-?=0.393, XZ-?=2.356, ?=1.482, BW=1', 'skewness-Gabor XY-?=0.393, XZ-?=2.356, ?=1.482, BW=1', 'var-Gabor XY-?=0.393, XZ-?=2.356, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=0.393, XZ-?=2.356, ?=0.880, BW=1', 'median-Gabor XY-?=0.393, XZ-?=2.356, ?=0.880, BW=1', 'skewness-Gabor XY-?=0.393, XZ-?=2.356, ?=0.880, BW=1', 'var-Gabor XY-?=0.393, XZ-?=2.356, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=0.393, XZ-?=1.571, ?=3.261, BW=1', 'median-Gabor XY-?=0.393, XZ-?=1.571, ?=3.261, BW=1', 'skewness-Gabor XY-?=0.393, XZ-?=1.571, ?=3.261, BW=1', 'var-Gabor XY-?=0.393, XZ-?=1.571, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=0.393, XZ-?=1.571, ?=2.668, BW=1', 'median-Gabor XY-?=0.393, XZ-?=1.571, ?=2.668, BW=1', 'skewness-Gabor XY-?=0.393, XZ-?=1.571, ?=2.668, BW=1', 'var-Gabor XY-?=0.393, XZ-?=1.571, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=0.393, XZ-?=1.571, ?=2.075, BW=1', 'median-Gabor XY-?=0.393, XZ-?=1.571, ?=2.075, BW=1', 'skewness-Gabor XY-?=0.393, XZ-?=1.571, ?=2.075, BW=1', 'var-Gabor XY-?=0.393, XZ-?=1.571, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=0.393, XZ-?=1.571, ?=1.482, BW=1', 'median-Gabor XY-?=0.393, XZ-?=1.571, ?=1.482, BW=1', 'skewness-Gabor XY-?=0.393, XZ-?=1.571, ?=1.482, BW=1', 'var-Gabor XY-?=0.393, XZ-?=1.571, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=0.393, XZ-?=1.571, ?=0.880, BW=1', 'median-Gabor XY-?=0.393, XZ-?=1.571, ?=0.880, BW=1', 'skewness-Gabor XY-?=0.393, XZ-?=1.571, ?=0.880, BW=1', 'var-Gabor XY-?=0.393, XZ-?=1.571, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=0.393, XZ-?=0.785, ?=3.261, BW=1', 'median-Gabor XY-?=0.393, XZ-?=0.785, ?=3.261, BW=1', 'skewness-Gabor XY-?=0.393, XZ-?=0.785, ?=3.261, BW=1', 'var-Gabor XY-?=0.393, XZ-?=0.785, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=0.393, XZ-?=0.785, ?=2.668, BW=1', 'median-Gabor XY-?=0.393, XZ-?=0.785, ?=2.668, BW=1', 'skewness-Gabor XY-?=0.393, XZ-?=0.785, ?=2.668, BW=1', 'var-Gabor XY-?=0.393, XZ-?=0.785, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=0.393, XZ-?=0.785, ?=2.075, BW=1', 'median-Gabor XY-?=0.393, XZ-?=0.785, ?=2.075, BW=1', 'skewness-Gabor XY-?=0.393, XZ-?=0.785, ?=2.075, BW=1', 'var-Gabor XY-?=0.393, XZ-?=0.785, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=0.393, XZ-?=0.785, ?=1.482, BW=1', 'median-Gabor XY-?=0.393, XZ-?=0.785, ?=1.482, BW=1', 'skewness-Gabor XY-?=0.393, XZ-?=0.785, ?=1.482, BW=1', 'var-Gabor XY-?=0.393, XZ-?=0.785, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=0.393, XZ-?=0.785, ?=0.880, BW=1', 'median-Gabor XY-?=0.393, XZ-?=0.785, ?=0.880, BW=1', 'skewness-Gabor XY-?=0.393, XZ-?=0.785, ?=0.880, BW=1', 'var-Gabor XY-?=0.393, XZ-?=0.785, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=0.393, XZ-?=0.000, ?=3.261, BW=1', 'median-Gabor XY-?=0.393, XZ-?=0.000, ?=3.261, BW=1', 'skewness-Gabor XY-?=0.393, XZ-?=0.000, ?=3.261, BW=1', 'var-Gabor XY-?=0.393, XZ-?=0.000, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=0.393, XZ-?=0.000, ?=2.668, BW=1', 'median-Gabor XY-?=0.393, XZ-?=0.000, ?=2.668, BW=1', 'skewness-Gabor XY-?=0.393, XZ-?=0.000, ?=2.668, BW=1', 'var-Gabor XY-?=0.393, XZ-?=0.000, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=0.393, XZ-?=0.000, ?=2.075, BW=1', 'median-Gabor XY-?=0.393, XZ-?=0.000, ?=2.075, BW=1', 'skewness-Gabor XY-?=0.393, XZ-?=0.000, ?=2.075, BW=1', 'var-Gabor XY-?=0.393, XZ-?=0.000, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=0.393, XZ-?=0.000, ?=1.482, BW=1', 'median-Gabor XY-?=0.393, XZ-?=0.000, ?=1.482, BW=1', 'skewness-Gabor XY-?=0.393, XZ-?=0.000, ?=1.482, BW=1', 'var-Gabor XY-?=0.393, XZ-?=0.000, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=0.393, XZ-?=0.000, ?=0.880, BW=1', 'median-Gabor XY-?=0.393, XZ-?=0.000, ?=0.880, BW=1', 'skewness-Gabor XY-?=0.393, XZ-?=0.000, ?=0.880, BW=1', 'var-Gabor XY-?=0.393, XZ-?=0.000, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=0.000, XZ-?=2.356, ?=3.261, BW=1', 'median-Gabor XY-?=0.000, XZ-?=2.356, ?=3.261, BW=1', 'skewness-Gabor XY-?=0.000, XZ-?=2.356, ?=3.261, BW=1', 'var-Gabor XY-?=0.000, XZ-?=2.356, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=0.000, XZ-?=2.356, ?=2.668, BW=1', 'median-Gabor XY-?=0.000, XZ-?=2.356, ?=2.668, BW=1', 'skewness-Gabor XY-?=0.000, XZ-?=2.356, ?=2.668, BW=1', 'var-Gabor XY-?=0.000, XZ-?=2.356, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=0.000, XZ-?=2.356, ?=2.075, BW=1', 'median-Gabor XY-?=0.000, XZ-?=2.356, ?=2.075, BW=1', 'skewness-Gabor XY-?=0.000, XZ-?=2.356, ?=2.075, BW=1', 'var-Gabor XY-?=0.000, XZ-?=2.356, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=0.000, XZ-?=2.356, ?=1.482, BW=1', 'median-Gabor XY-?=0.000, XZ-?=2.356, ?=1.482, BW=1', 'skewness-Gabor XY-?=0.000, XZ-?=2.356, ?=1.482, BW=1', 'var-Gabor XY-?=0.000, XZ-?=2.356, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=0.000, XZ-?=2.356, ?=0.880, BW=1', 'median-Gabor XY-?=0.000, XZ-?=2.356, ?=0.880, BW=1', 'skewness-Gabor XY-?=0.000, XZ-?=2.356, ?=0.880, BW=1', 'var-Gabor XY-?=0.000, XZ-?=2.356, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=0.000, XZ-?=1.571, ?=3.261, BW=1', 'median-Gabor XY-?=0.000, XZ-?=1.571, ?=3.261, BW=1', 'skewness-Gabor XY-?=0.000, XZ-?=1.571, ?=3.261, BW=1', 'var-Gabor XY-?=0.000, XZ-?=1.571, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=0.000, XZ-?=1.571, ?=2.668, BW=1', 'median-Gabor XY-?=0.000, XZ-?=1.571, ?=2.668, BW=1', 'skewness-Gabor XY-?=0.000, XZ-?=1.571, ?=2.668, BW=1', 'var-Gabor XY-?=0.000, XZ-?=1.571, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=0.000, XZ-?=1.571, ?=2.075, BW=1', 'median-Gabor XY-?=0.000, XZ-?=1.571, ?=2.075, BW=1', 'skewness-Gabor XY-?=0.000, XZ-?=1.571, ?=2.075, BW=1', 'var-Gabor XY-?=0.000, XZ-?=1.571, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=0.000, XZ-?=1.571, ?=1.482, BW=1', 'median-Gabor XY-?=0.000, XZ-?=1.571, ?=1.482, BW=1', 'skewness-Gabor XY-?=0.000, XZ-?=1.571, ?=1.482, BW=1', 'var-Gabor XY-?=0.000, XZ-?=1.571, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=0.000, XZ-?=1.571, ?=0.880, BW=1', 'median-Gabor XY-?=0.000, XZ-?=1.571, ?=0.880, BW=1', 'skewness-Gabor XY-?=0.000, XZ-?=1.571, ?=0.880, BW=1', 'var-Gabor XY-?=0.000, XZ-?=1.571, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=0.000, XZ-?=0.785, ?=3.261, BW=1', 'median-Gabor XY-?=0.000, XZ-?=0.785, ?=3.261, BW=1', 'skewness-Gabor XY-?=0.000, XZ-?=0.785, ?=3.261, BW=1', 'var-Gabor XY-?=0.000, XZ-?=0.785, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=0.000, XZ-?=0.785, ?=2.668, BW=1', 'median-Gabor XY-?=0.000, XZ-?=0.785, ?=2.668, BW=1', 'skewness-Gabor XY-?=0.000, XZ-?=0.785, ?=2.668, BW=1', 'var-Gabor XY-?=0.000, XZ-?=0.785, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=0.000, XZ-?=0.785, ?=2.075, BW=1', 'median-Gabor XY-?=0.000, XZ-?=0.785, ?=2.075, BW=1', 'skewness-Gabor XY-?=0.000, XZ-?=0.785, ?=2.075, BW=1', 'var-Gabor XY-?=0.000, XZ-?=0.785, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=0.000, XZ-?=0.785, ?=1.482, BW=1', 'median-Gabor XY-?=0.000, XZ-?=0.785, ?=1.482, BW=1', 'skewness-Gabor XY-?=0.000, XZ-?=0.785, ?=1.482, BW=1', 'var-Gabor XY-?=0.000, XZ-?=0.785, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=0.000, XZ-?=0.785, ?=0.880, BW=1', 'median-Gabor XY-?=0.000, XZ-?=0.785, ?=0.880, BW=1', 'skewness-Gabor XY-?=0.000, XZ-?=0.785, ?=0.880, BW=1', 'var-Gabor XY-?=0.000, XZ-?=0.785, ?=0.880, BW=1'),
    ('kurtosis-Gabor XY-?=0.000, XZ-?=0.000, ?=3.261, BW=1', 'median-Gabor XY-?=0.000, XZ-?=0.000, ?=3.261, BW=1', 'skewness-Gabor XY-?=0.000, XZ-?=0.000, ?=3.261, BW=1', 'var-Gabor XY-?=0.000, XZ-?=0.000, ?=3.261, BW=1'),
    ('kurtosis-Gabor XY-?=0.000, XZ-?=0.000, ?=2.668, BW=1', 'median-Gabor XY-?=0.000, XZ-?=0.000, ?=2.668, BW=1', 'skewness-Gabor XY-?=0.000, XZ-?=0.000, ?=2.668, BW=1', 'var-Gabor XY-?=0.000, XZ-?=0.000, ?=2.668, BW=1'),
    ('kurtosis-Gabor XY-?=0.000, XZ-?=0.000, ?=2.075, BW=1', 'median-Gabor XY-?=0.000, XZ-?=0.000, ?=2.075, BW=1', 'skewness-Gabor XY-?=0.000, XZ-?=0.000, ?=2.075, BW=1', 'var-Gabor XY-?=0.000, XZ-?=0.000, ?=2.075, BW=1'),
    ('kurtosis-Gabor XY-?=0.000, XZ-?=0.000, ?=1.482, BW=1', 'median-Gabor XY-?=0.000, XZ-?=0.000, ?=1.482, BW=1', 'skewness-Gabor XY-?=0.000, XZ-?=0.000, ?=1.482, BW=1', 'var-Gabor XY-?=0.000, XZ-?=0.000, ?=1.482, BW=1'),
    ('kurtosis-Gabor XY-?=0.000, XZ-?=0.000, ?=0.880, BW=1', 'median-Gabor XY-?=0.000, XZ-?=0.000, ?=0.880, BW=1', 'skewness-Gabor XY-?=0.000, XZ-?=0.000, ?=0.880, BW=1', 'var-Gabor XY-?=0.000, XZ-?=0.000, ?=0.880, BW=1'),
]


def load_feature_specs() -> Tuple[List[str], List[str], List[str], List[str], List[Tuple[str, str]]]:
    k_list = [row[0] for row in TEXTURE_FEATURE_TABLE]
    m_list = [row[1] for row in TEXTURE_FEATURE_TABLE]
    s_list = [row[2] for row in TEXTURE_FEATURE_TABLE]
    v_list = [row[3] for row in TEXTURE_FEATURE_TABLE]

    if not (len(k_list) == len(m_list) == len(s_list) == len(v_list) == 410):
        raise ValueError(f"Expected 410 entries per column, got {len(k_list)} / {len(m_list)} / {len(s_list)} / {len(v_list)}")

    specs: List[Tuple[str, str]] = []
    for name in k_list:
        m = re.match(r"kurtosis-([A-Za-z0-9]+)\s*(.*)$", name)
        if not m:
            raise ValueError(f"Cannot parse feature name: {name}")
        fam = m.group(1)
        params = m.group(2).strip()
        specs.append((fam, params))

    return k_list, m_list, s_list, v_list, specs


def stats_4(x: np.ndarray) -> Tuple[float, float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)

    med = float(np.median(x))
    var = float(np.var(x, ddof=0))

    mu = float(np.mean(x))
    s2 = float(np.mean((x - mu) ** 2))
    if s2 < EPS:
        return (0.0, med, 0.0, 0.0)

    s = math.sqrt(s2)
    skew = float(np.mean(((x - mu) / (s + EPS)) ** 3))
    kurt = float(np.mean(((x - mu) / (s + EPS)) ** 4))

    return (kurt, med, skew, var)


def list_case_ids(folder: str, ext: str = ".mha") -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    ids = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(ext):
            ids.append(fn[:-len(ext)])
    ids.sort()
    return ids

def find_mask_path(mask_dir: str, cid: str) -> Optional[str]:
    p1 = os.path.join(mask_dir, f"{cid}.nii.gz")
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(mask_dir, f"{cid}.mha")
    if os.path.exists(p2):
        return p2
    return None

def read_volume(path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    sp = img.GetSpacing()
    return arr, (float(sp[0]), float(sp[1]), float(sp[2]))

def bbox_from_mask(mask: np.ndarray, margin: int) -> Tuple[slice, slice, slice]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        z, y, x = mask.shape
        return slice(0, z), slice(0, y), slice(0, x)

    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)

    zmin = max(0, zmin - margin); ymin = max(0, ymin - margin); xmin = max(0, xmin - margin)
    zmax = min(mask.shape[0] - 1, zmax + margin)
    ymax = min(mask.shape[1] - 1, ymax + margin)
    xmax = min(mask.shape[2] - 1, xmax + margin)

    return slice(zmin, zmax + 1), slice(ymin, ymax + 1), slice(xmin, xmax + 1)


LAWS_5 = {
    "L5": np.array([1, 4, 6, 4, 1], dtype=np.float32),
    "E5": np.array([-1, -2, 0, 2, 1], dtype=np.float32),
    "S5": np.array([-1, 0, 2, 0, -1], dtype=np.float32),
    "W5": np.array([-1, 2, 0, -2, 1], dtype=np.float32),
    "R5": np.array([1, -4, 6, -4, 1], dtype=np.float32),
}
LAWS_3 = {
    "L3": np.array([1, 2, 1], dtype=np.float32),
    "E3": np.array([-1, 0, 1], dtype=np.float32),
    "S3": np.array([-1, 2, -1], dtype=np.float32),
}

def laws_filter_3d(vol: np.ndarray, code: str, mode: str = "reflect") -> np.ndarray:
    code = code.strip()
    blocks = re.findall(r"[A-Z]\d", code)
    if len(blocks) != 3:
        raise ValueError(f"Invalid Laws code: {code}")

    ks = []
    for b in blocks:
        if b.endswith("5"):
            ks.append(LAWS_5[b].astype(np.float32))
        elif b.endswith("3"):
            ks.append(LAWS_3[b].astype(np.float32))
        else:
            raise ValueError(f"Unknown Laws kernel: {b}")

    out = vol.astype(np.float32, copy=False)
    out = ndimage.convolve1d(out, ks[2], axis=2, mode=mode)
    out = ndimage.convolve1d(out, ks[1], axis=1, mode=mode)
    out = ndimage.convolve1d(out, ks[0], axis=0, mode=mode)
    return out

def local_mean(vol: np.ndarray, ws: int) -> np.ndarray:
    return ndimage.uniform_filter(vol, size=(ws, ws, ws), mode="reflect")

def local_std(vol: np.ndarray, ws: int) -> np.ndarray:
    mu = local_mean(vol, ws)
    mu2 = local_mean(vol * vol, ws)
    var = np.maximum(0.0, mu2 - mu * mu)
    return np.sqrt(var)

def local_median(vol: np.ndarray, ws: int) -> np.ndarray:
    return ndimage.median_filter(vol, size=(ws, ws, ws), mode="reflect")

def local_range(vol: np.ndarray, ws: int) -> np.ndarray:
    mx = ndimage.maximum_filter(vol, size=(ws, ws, ws), mode="reflect")
    mn = ndimage.minimum_filter(vol, size=(ws, ws, ws), mode="reflect")
    return mx - mn

def gradient_maps(vol: np.ndarray) -> Dict[str, np.ndarray]:
    dz, dy, dx = np.gradient(vol.astype(np.float32), edge_order=1)
    mag = np.sqrt(dx * dx + dy * dy + dz * dz)

    sobx = ndimage.sobel(vol, axis=2, mode="reflect")
    soby = ndimage.sobel(vol, axis=1, mode="reflect")
    sobz = ndimage.sobel(vol, axis=0, mode="reflect")

    sobxy = ndimage.sobel(sobx, axis=1, mode="reflect")
    sobyx = ndimage.sobel(soby, axis=2, mode="reflect")

    sobxz = ndimage.sobel(sobx, axis=0, mode="reflect")
    sobzx = ndimage.sobel(sobz, axis=2, mode="reflect")

    sobyz = ndimage.sobel(soby, axis=0, mode="reflect")
    sobzy = ndimage.sobel(sobz, axis=1, mode="reflect")

    return {
        "x": dx, "y": dy, "z": dz, "magnitude": mag,
        "sobelx": sobx, "sobely": soby, "sobelz": sobz,
        "sobelxy": sobxy, "sobelyx": sobyx,
        "sobelxz": sobxz, "sobelzx": sobzx,
        "sobelyz": sobyz, "sobelzy": sobzy,
    }

def _sigma_from_lambda_bw(lam: float, bw: int) -> float:
    if bw <= 0:
        bw = 1
    return (lam / math.pi) * math.sqrt(math.log(2) / 2) * ((2**bw + 1) / (2**bw - 1))

def gabor_kernel_3d(xy_theta: float, xz_theta: float, lam: float, bw: int,
                    spacing: Tuple[float, float, float]) -> np.ndarray:
    spx, spy, spz = spacing
    sigma = _sigma_from_lambda_bw(lam, bw)

    rad_mm = 3.0 * sigma
    rx = max(1, int(math.ceil(rad_mm / (spx + EPS))))
    ry = max(1, int(math.ceil(rad_mm / (spy + EPS))))
    rz = max(1, int(math.ceil(rad_mm / (spz + EPS))))

    z = (np.arange(-rz, rz + 1) * spz).astype(np.float32)
    y = (np.arange(-ry, ry + 1) * spy).astype(np.float32)
    x = (np.arange(-rx, rx + 1) * spx).astype(np.float32)
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    az = xy_theta
    el = xz_theta
    kx = math.cos(el) * math.cos(az)
    ky = math.cos(el) * math.sin(az)
    kz = math.sin(el)

    u = (kx * X + ky * Y + kz * Z)

    gauss = np.exp(-(X*X + Y*Y + Z*Z) / (2.0 * sigma * sigma + EPS))
    wave = np.cos(2.0 * math.pi * u / (lam + EPS))

    ker = (gauss * wave).astype(np.float32)
    ker -= float(ker.mean())
    norm = float(np.sqrt(np.sum(ker * ker)) + EPS)
    ker /= norm
    return ker

def gabor_filter_3d(vol: np.ndarray, params: str, spacing: Tuple[float, float, float]) -> np.ndarray:
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", params)]
    if len(nums) < 4:
        raise ValueError(f"Cannot parse Gabor params: {params}")
    xy_theta, xz_theta, lam, bw = nums[0], nums[1], nums[2], int(nums[3])

    ker = gabor_kernel_3d(xy_theta, xz_theta, lam, bw, spacing)
    out = ndimage.convolve(vol.astype(np.float32), ker, mode="reflect")
    return out


def _quantize(img: np.ndarray, levels: int) -> np.ndarray:
    img = img.astype(np.float32)
    mn = float(np.min(img))
    mx = float(np.max(img))
    if mx - mn < EPS:
        return np.zeros_like(img, dtype=np.uint8)
    q = np.floor((img - mn) / (mx - mn + EPS) * levels).astype(np.int32)
    q = np.clip(q, 0, levels - 1).astype(np.uint8)
    return q

def _haralick_maps_2d(q: np.ndarray, ws: int, levels: int) -> Dict[str, np.ndarray]:
    H, W = q.shape
    win_area = ws * ws

    C = np.zeros((levels, levels, H, W), dtype=np.float32)

    for dy, dx in HAR_OFFSETS_2D:
        y1s = max(0, dy); y1e = H + min(0, dy)
        x1s = max(0, dx); x1e = W + min(0, dx)

        a = q[y1s:y1e, x1s:x1e]
        b = q[y1s - dy:y1e - dy, x1s - dx:x1e - dx]

        for i in range(levels):
            ai = (a == i)
            if not ai.any():
                continue
            for j in range(levels):
                ind = (ai & (b == j)).astype(np.float32)
                s = ndimage.uniform_filter(ind, size=ws, mode="reflect") * win_area
                C[i, j, y1s:y1e, x1s:x1e] += s

    Psum = np.sum(C, axis=(0, 1), keepdims=True) + EPS
    P = C / Psum

    I = np.arange(levels, dtype=np.float32).reshape(levels, 1, 1, 1)
    J = np.arange(levels, dtype=np.float32).reshape(1, levels, 1, 1)
    IJ = I * J
    D2 = (I - J) ** 2

    px = np.sum(P, axis=1)
    py = np.sum(P, axis=0)

    mux = np.sum((np.arange(levels, dtype=np.float32).reshape(levels, 1, 1) * px), axis=0)
    muy = np.sum((np.arange(levels, dtype=np.float32).reshape(levels, 1, 1) * py), axis=0)

    sigx = np.sqrt(np.sum(((np.arange(levels, dtype=np.float32).reshape(levels, 1, 1) - mux) ** 2) * px, axis=0) + EPS)
    sigy = np.sqrt(np.sum(((np.arange(levels, dtype=np.float32).reshape(levels, 1, 1) - muy) ** 2) * py, axis=0) + EPS)

    energy = np.sum(P * P, axis=(0, 1))
    entropy = -np.sum(P * np.log(P + EPS), axis=(0, 1))
    inertia = np.sum(D2 * P, axis=(0, 1))
    idm = np.sum(P / (1.0 + D2), axis=(0, 1))

    corr_num = np.sum(IJ * P, axis=(0, 1)) - (mux * muy)
    correlation = corr_num / (sigx * sigy + EPS)

    kmax = 2 * (levels - 1)
    p_sum = np.zeros((kmax + 1, H, W), dtype=np.float32)
    p_diff = np.zeros((levels, H, W), dtype=np.float32)

    for i in range(levels):
        for j in range(levels):
            p_sum[i + j] += P[i, j]
            p_diff[abs(i - j)] += P[i, j]

    k = np.arange(kmax + 1, dtype=np.float32).reshape(-1, 1, 1)
    sum_av = np.sum(k * p_sum, axis=0)
    sum_ent = -np.sum(p_sum * np.log(p_sum + EPS), axis=0)
    sum_var = np.sum(((k - sum_av) ** 2) * p_sum, axis=0)

    k2 = np.arange(levels, dtype=np.float32).reshape(-1, 1, 1)
    diff_av = np.sum(k2 * p_diff, axis=0)
    diff_ent = -np.sum(p_diff * np.log(p_diff + EPS), axis=0)
    diff_var = np.sum(((k2 - diff_av) ** 2) * p_diff, axis=0)

    HX = -np.sum(px * np.log(px + EPS), axis=0)
    HY = -np.sum(py * np.log(py + EPS), axis=0)

    px_py = px[:, None, :, :] * py[None, :, :, :]
    HXY1 = -np.sum(P * np.log(px_py + EPS), axis=(0, 1))
    HXY2 = -np.sum(px_py * np.log(px_py + EPS), axis=(0, 1))
    HXY = entropy

    info1 = (HXY - HXY1) / (np.maximum(HX, HY) + EPS)
    info2 = np.sqrt(np.maximum(0.0, 1.0 - np.exp(-2.0 * (HXY2 - HXY))))

    return {
        "energy": energy,
        "entropy": entropy,
        "inertia": inertia,
        "idm": idm,
        "correlation": correlation,
        "sum-av": sum_av,
        "sum-var": sum_var,
        "sum-ent": sum_ent,
        "diff-av": diff_av,
        "diff-var": diff_var,
        "diff-ent": diff_ent,
        "info1": info1,
        "info2": info2,
    }

def haralick_feature_values(vol: np.ndarray, mask: np.ndarray, feature: str, ws: int) -> np.ndarray:
    values = []
    for z in range(vol.shape[0]):
        m2d = mask[z] > 0
        if not m2d.any():
            continue

        img2d = vol[z].astype(np.float32)
        coords = np.argwhere(m2d)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        pad = ws // 2

        ys = max(0, y0 - pad); ye = min(img2d.shape[0], y1 + pad + 1)
        xs = max(0, x0 - pad); xe = min(img2d.shape[1], x1 + pad + 1)

        img_crop = img2d[ys:ye, xs:xe]
        m_crop = m2d[ys:ye, xs:xe]

        q = _quantize(img_crop, HAR_LEVELS)
        maps = _haralick_maps_2d(q, ws=ws, levels=HAR_LEVELS)
        fmap = maps[feature]
        values.append(fmap[m_crop])

    if not values:
        return np.array([], dtype=np.float32)

    return np.concatenate(values).astype(np.float32)


def compute_feature_map(vol: np.ndarray, fam: str, params: str,
                        spacing: Tuple[float, float, float],
                        cache: Dict[str, Dict]) -> np.ndarray:
    fam = fam.strip()

    if fam == "Gray":
        m = re.match(r"(mean|median|range|std_dev)\s+ws=(\d+)", params)
        if not m:
            raise ValueError(f"Bad Gray params: {params}")
        metric = m.group(1)
        ws = int(m.group(2))
        key = f"Gray::{metric}::ws={ws}"
        if key in cache["maps"]:
            return cache["maps"][key]

        if metric == "mean":
            out = local_mean(vol, ws)
        elif metric == "std_dev":
            out = local_std(vol, ws)
        elif metric == "median":
            out = local_median(vol, ws)
        elif metric == "range":
            out = local_range(vol, ws)
        else:
            raise ValueError(metric)

        cache["maps"][key] = out
        return out

    if fam == "Gradient":
        if cache["Gradient"] is None:
            cache["Gradient"] = gradient_maps(vol)
        name = params.strip()
        if name not in cache["Gradient"]:
            raise ValueError(f"Unknown Gradient feature: {name}")
        return cache["Gradient"][name]

    if fam == "Laws":
        code = params.strip()
        key = f"Laws::{code}"
        if key in cache["maps"]:
            return cache["maps"][key]
        out = laws_filter_3d(vol, code)
        cache["maps"][key] = out
        return out

    if fam == "Gabor":
        key = f"Gabor::{params}"
        if key in cache["maps"]:
            return cache["maps"][key]
        out = gabor_filter_3d(vol, params, spacing=spacing)
        cache["maps"][key] = out
        return out

    raise ValueError(f"Unknown family: {fam}")

def extract_case_features(vol: np.ndarray, mask: np.ndarray, spacing: Tuple[float, float, float],
                          specs: List[Tuple[str, str]],
                          k_names: List[str], m_names: List[str], s_names: List[str], v_names: List[str]) -> Dict[str, float]:
    if vol.shape != mask.shape:
        raise ValueError(f"Shape mismatch: vol{vol.shape} mask{mask.shape}")

    zsl, ysl, xsl = bbox_from_mask(mask, margin=ROI_BBOX_MARGIN)
    v = vol[zsl, ysl, xsl]
    m = (mask[zsl, ysl, xsl] > 0)

    out: Dict[str, float] = {}
    cache = {"maps": {}, "Gradient": None}

    for i, (fam, params) in enumerate(specs):
        if fam == "Haralick":
            mm = re.match(r"(.*)\s+ws=(\d+)", params)
            if not mm:
                raise ValueError(f"Bad Haralick params: {params}")
            feat = mm.group(1).strip()
            ws = int(mm.group(2))

            vals = haralick_feature_values(v, m.astype(np.uint8), feature=feat, ws=ws)
            kurt, med, skew, var = stats_4(vals)
        else:
            fmap = compute_feature_map(v, fam, params, spacing=spacing, cache=cache)
            vals = fmap[m]
            kurt, med, skew, var = stats_4(vals)

        out[k_names[i]] = kurt
        out[m_names[i]] = med
        out[s_names[i]] = skew
        out[v_names[i]] = var

    return out


try:
    import cupy as cp
except Exception:
    cp = np

TOPO_PYRAMID_LEVELS = 4


def topo_find_mask_path(mask_dir: str, cid: str):
    for ext in (".nii.gz", ".mha"):
        p = os.path.join(mask_dir, cid + ext)
        if os.path.exists(p):
            return p
    return None


def topo_pyramid(img, mask, lvl):
    if lvl == 0:
        return img, mask
    for _ in range(lvl):
        img = sitk.DiscreteGaussian(img, 1.0)
        sz, sp = img.GetSize(), img.GetSpacing()
        ns = [max(1, s // 2) for s in sz]
        nsp = [sp[i] * 2 for i in range(3)]

        r = sitk.ResampleImageFilter()
        r.SetSize(ns)
        r.SetOutputSpacing(tuple(nsp))
        r.SetOutputOrigin(img.GetOrigin())
        r.SetOutputDirection(img.GetDirection())

        r.SetInterpolator(sitk.sitkLinear)
        img = r.Execute(img)

        r.SetInterpolator(sitk.sitkNearestNeighbor)
        mask = r.Execute(mask)

    return img, mask


def topo_gpu_intensity(arr, mask):
    roi = arr[mask > 0]
    if roi.size == 0:
        return [0.0] * 5
    x = cp.asarray(roi)
    mu = cp.mean(x)
    mean = float(mu)
    mx = float(cp.max(x))
    tot = float(cp.sum(x))
    skw = float(cp.mean((x - mu) ** 3) / (cp.std(x) ** 3 + EPS))
    krt = float(cp.mean((x - mu) ** 4) / (cp.var(x) ** 2 + EPS))
    return [mean, mx, tot, skw, krt]


def topo_topology(mask, spacing):
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    vol = float(np.sum(mask) * voxel_vol)

    try:
        v, f, _, _ = marching_cubes(mask.astype(np.float32), 0.5, spacing=spacing)
        area = float(mesh_surface_area(v, f))
    except Exception:
        area = 0.0

    try:
        eul = int(euler_number(mask.astype(np.uint8), connectivity=3))
    except Exception:
        eul = 0

    return vol, area, eul


def topo_extract_case(img_path, mask_path):
    img = sitk.ReadImage(img_path)
    mask = sitk.ReadImage(mask_path)

    feats = []
    for lvl in range(TOPO_PYRAMID_LEVELS):
        i, m = topo_pyramid(img, mask, lvl)

        arr = sitk.GetArrayFromImage(i).astype(np.float32)
        msk = (sitk.GetArrayFromImage(m) > 0)

        sp_xyz = i.GetSpacing()
        sp_zyx = sp_xyz[::-1]
        vol, area, eul = topo_topology(msk, sp_zyx)

        mean, mx, tot, skw, krt = topo_gpu_intensity(arr, msk)

        feats.extend([vol, area, eul, mean, mx, tot, skw, krt])

    return feats


from all_config import (
    ADC_FOLDER,
    CLIN_COL,
    DATASET_XLSX,
    EXCEL_FLUSH_EVERY,
    FEATURE_FILES,
    ID_COL,
    log_progress,
    MAIN_ROOT,
    MAIN_AI1_MASK,
    MAIN_AI2_MASK,
    ORGAN_ICC_DIR,
    ORGAN_RAW_DIR,
    MASK_FOLDER,
    ORGAN_SAME_WORKBOOK_DIR,
    ORGAN_ONLY_WORKBOOK_DIR,
    ORGAN_SAME_RAW_DIR,
    T2W_FOLDER,
    ZONE_COL,
)


import b_patch


def feature_paths(root: Path, mask_folder: str = MASK_FOLDER) -> tuple[str, str, str]:
    return str(root / T2W_FOLDER), str(root / ADC_FOLDER), str(root / mask_folder)


def _topo_feature_cols() -> list[str]:
    names = ["Volume", "Surface", "Euler", "Mean", "Max", "Sum", "Skew", "Kurt"]
    return [f"{n}_L{l}" for l in range(TOPO_PYRAMID_LEVELS) for n in names]


def _process_combined_case(args):
    cid, modality_dir, mask_dir, specs, k_names, m_names, s_names, v_names = args
    img_path = os.path.join(modality_dir, f"{cid}.mha")
    pm_path = find_mask_path(mask_dir, cid)
    if pm_path is None:
        return None
    vol, spacing = read_volume(img_path)
    pm, _ = read_volume(pm_path)
    pm = (pm > 0).astype(np.uint8)
    if vol.shape != pm.shape:
        return None
    feats = extract_case_features(vol, pm, spacing, specs, k_names, m_names, s_names, v_names)
    feats.update(dict(zip(_topo_feature_cols(), topo_extract_case(img_path, pm_path))))
    feats["case_id"] = str(cid)
    return feats


def _handcrafted_fieldnames(specs, k_names, m_names, s_names, v_names) -> list[str]:
    cols = ["case_id"]
    for i in range(len(specs)):
        cols.extend([k_names[i], m_names[i], s_names[i], v_names[i]])
    cols.extend(_topo_feature_cols())
    return cols


def stream_handcrafted_modality(modality_dir: str, mask_dir: str, out_path: Path,
                                specs, k_names, m_names, s_names, v_names) -> None:
    fieldnames = _handcrafted_fieldnames(specs, k_names, m_names, s_names, v_names)
    case_ids = list_case_ids(modality_dir, ext=".mha")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    file_exists = out_path.exists()
    if file_exists:
        existing = pd.read_csv(out_path, dtype={"case_id": str}, on_bad_lines="skip")
        done_ids = {str(c) for c in existing["case_id"]}
        print(f"[ORGAN] resume: {len(done_ids)} cases already in {out_path}")

    pending_ids = [cid for cid in case_ids if cid not in done_ids]
    tasks = [(cid, modality_dir, mask_dir, specs, k_names, m_names, s_names, v_names) for cid in pending_ids]

    written = skipped = 0
    with open(out_path, "a" if file_exists else "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
            fh.flush()

        def _emit(row):
            nonlocal written, skipped
            if row is None:
                skipped += 1
                return
            writer.writerow({c: row.get(c, "") for c in fieldnames})
            fh.flush()
            written += 1

        if N_WORKERS <= 1:
            for task in tasks:
                _emit(_process_combined_case(task))
        else:
            with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
                futures = [ex.submit(_process_combined_case, t) for t in tasks]
                for f in as_completed(futures):
                    _emit(f.result())
    print(f"[ORGAN] streamed {out_path} new={written} skipped={skipped} cols={len(fieldnames) - 1}")


def _read_mask_tables_handcrafted(raw_dir: Path, prefix: str) -> dict[str, pd.DataFrame]:
    k_names, m_names, s_names, v_names, _specs = load_feature_specs()
    texture_names = set(k_names) | set(m_names) | set(s_names) | set(v_names)
    topo_names = set(_topo_feature_cols())
    tables: dict[str, pd.DataFrame] = {}
    for mod in ("adc", "t2w"):
        df = pd.read_csv(Path(raw_dir) / f"mask{prefix}_feature--{mod}.csv", dtype={"case_id": str}, on_bad_lines="skip")
        tex_cols = [c for c in df.columns if c in texture_names]
        top_cols = [c for c in df.columns if c in topo_names]
        tables[f"texture_{mod}"] = df[["case_id"] + tex_cols].copy()
        tables[f"topology_{mod}"] = df[["case_id"] + top_cols].copy()
    return tables


def load_keep_features(icc_dir: Path = ORGAN_ICC_DIR) -> dict[str, set[str]]:
    keep_path = Path(icc_dir) / "selected_features_icc.json"
    if not keep_path.exists():
        raise FileNotFoundError(f"Run c_organ.py --stage icc first. Missing: {keep_path}")
    with keep_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {key: set(vals) for key, vals in raw.items()}


def selected_texture_specs(modality_key: str, keep_features: set[str]):
    k_names, m_names, s_names, v_names, specs = load_feature_specs()
    selected = []
    output_cols = ["case_id"]
    for i, spec in enumerate(specs):
        names = [k_names[i], m_names[i], s_names[i], v_names[i]]
        keep_names = [name for name in names if name in keep_features]
        if keep_names:
            selected.append((i, spec, names, keep_names))
            output_cols.extend(keep_names)
    if not selected:
        raise RuntimeError(f"No ICC-stable texture features found for {modality_key}")
    return selected, output_cols


def extract_reduced_texture_case(cid: str, modality_dir: str, mask_dir: str, selected) -> dict | None:
    img_path = os.path.join(modality_dir, f"{cid}.mha")
    pm_path = find_mask_path(mask_dir, cid)
    if pm_path is None:
        return None

    vol, spacing = read_volume(img_path)
    pm, _ = read_volume(pm_path)
    pm = (pm > 0).astype(np.uint8)
    if vol.shape != pm.shape:
        return None

    zsl, ysl, xsl = bbox_from_mask(pm, margin=ROI_BBOX_MARGIN)
    v = vol[zsl, ysl, xsl]
    m = (pm[zsl, ysl, xsl] > 0)
    cache = {"maps": {}, "Gradient": None}
    row = {"case_id": cid}

    for _, (fam, params), names, keep_names in selected:
        if fam == "Haralick":
            mm = re.match(r"(.*)\s+ws=(\d+)", params)
            if not mm:
                raise ValueError(f"Bad Haralick params: {params}")
            feat = mm.group(1).strip()
            ws = int(mm.group(2))
            vals = haralick_feature_values(v, m.astype(np.uint8), feature=feat, ws=ws)
        else:
            fmap = compute_feature_map(v, fam, params, spacing=spacing, cache=cache)
            vals = fmap[m]

        stats = stats_4(vals)
        for name, value in zip(names, stats):
            if name in keep_names:
                row[name] = value
    return row


def _process_p158_case_combined(task):
    cid, root, specs_t2w, specs_adc, topo_cols = task
    root = Path(root)
    t2w_dir = str(root / "t2w")
    adc_dir = str(root / "adc")
    mask_dir = str(root / MASK_FOLDER)
    
    result = {"case_id": str(cid)}
    
    selected_t2w, cols_t2w = specs_t2w
    selected_adc, cols_adc = specs_adc
    
    for mod_key, mod_dir, selected, output_cols in [("t2w", t2w_dir, selected_t2w, cols_t2w), 
                                                       ("adc", adc_dir, selected_adc, cols_adc)]:
        tex_row = extract_reduced_texture_case(cid, mod_dir, mask_dir, selected)
        if tex_row:
            for col in output_cols:
                if col != "case_id":
                    result[f"tex_{mod_key}_{col}"] = tex_row.get(col, np.nan)
    
    for mod_key, mod_dir in [("t2w", t2w_dir), ("adc", adc_dir)]:
        mpath = topo_find_mask_path(mask_dir, cid)
        if mpath:
            topo_feats = topo_extract_case(os.path.join(mod_dir, f"{cid}.mha"), mpath)
            if topo_feats:
                for i, feat in enumerate(topo_feats):
                    result[f"top_{mod_key}_{i}"] = feat
    
    return result


def _p158_tables_from_rows(rows: list[dict], specs_t2w, specs_adc, topo_cols: list[str]) -> dict[str, pd.DataFrame]:
    combined_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    tables = {}
    _, cols_t2w = specs_t2w
    _, cols_adc = specs_adc

    for mod_key, cols in [("t2w", cols_t2w), ("adc", cols_adc)]:
        tex_cols = [c for c in combined_df.columns if c.startswith(f"tex_{mod_key}_")]
        if tex_cols:
            df_tex = combined_df[["case_id"] + tex_cols].copy()
            df_tex.columns = ["case_id"] + [c[len(f"tex_{mod_key}_"):] for c in tex_cols]
            tables[f"texture_{mod_key}"] = df_tex

        top_cols = [c for c in combined_df.columns if c.startswith(f"top_{mod_key}_")]
        if top_cols:
            df_top = combined_df[["case_id"] + top_cols].copy()
            df_top.columns = ["case_id"] + topo_cols[:len(top_cols)]
            tables[f"topology_{mod_key}"] = df_top

    for key in ["texture_t2w", "texture_adc", "topology_t2w", "topology_adc"]:
        if key not in tables:
            tables[key] = pd.DataFrame(columns=["case_id"])
        else:
            tables[key]["case_id"] = tables[key]["case_id"].astype(str)
            tables[key] = tables[key].sort_values("case_id").reset_index(drop=True)
    return tables


def p158_case_ids(dataset_root: Path) -> list[str]:
    dataset_root = Path(dataset_root)
    case_ids_t2w = set(list_case_ids(str(dataset_root / "t2w"), ext=".mha"))
    case_ids_adc = set(list_case_ids(str(dataset_root / "adc"), ext=".mha"))
    return sorted(case_ids_t2w & case_ids_adc)


def extract_p158_organ_features_combined(
    dataset_root: Path,
    keep_features: dict[str, set[str]],
    already_done_ids: set[str] | None = None,
    on_progress=None,
) -> dict[str, pd.DataFrame]:
    dataset_root = Path(dataset_root)
    specs_t2w = selected_texture_specs("t2w", keep_features["texture_t2w"])
    specs_adc = selected_texture_specs("adc", keep_features["texture_adc"])
    topo_names = ["Volume", "Surface", "Euler", "Mean", "Max", "Sum", "Skew", "Kurt"]
    topo_cols = [f"{n}_L{l}" for l in range(TOPO_PYRAMID_LEVELS) for n in topo_names]

    case_ids = p158_case_ids(dataset_root)
    already_done_ids = already_done_ids or set()
    pending_ids = [cid for cid in case_ids if cid not in already_done_ids]
    if already_done_ids:
        print(f"[P158 ORGAN] resume: {len(already_done_ids)} cases already published; {len(pending_ids)} remaining")

    tasks = [(cid, dataset_root, specs_t2w, specs_adc, topo_cols) for cid in pending_ids]
    rows: list[dict] = []
    completed = 0
    total = len(tasks)

    def _emit(result) -> None:
        nonlocal completed
        completed += 1
        log_progress("FEATURES", completed, total)
        if result and len(result) > 1:
            rows.append(result)
        if on_progress is not None and completed % EXCEL_FLUSH_EVERY == 0:
            on_progress(_p158_tables_from_rows(rows, specs_t2w, specs_adc, topo_cols))

    if N_WORKERS <= 1:
        for task in tasks:
            _emit(_process_p158_case_combined(task))
    else:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = [ex.submit(_process_p158_case_combined, t) for t in tasks]
            for f in as_completed(futures):
                _emit(f.result())

    tables = _p158_tables_from_rows(rows, specs_t2w, specs_adc, topo_cols)
    for key, df in tables.items():
        print(f"[P158 {key}] new rows={len(df)} cols={len(df.columns) - 1}")
    if on_progress is not None:
        on_progress(tables)
    return tables


def _publish_merged_raw_features(tables: dict[str, pd.DataFrame], out_dir: Path, prefix: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for mod in ("adc", "t2w"):
        frames = [tables[f"{kind}_{mod}"] for kind in ("texture", "topology") if f"{kind}_{mod}" in tables]
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            continue
        merged = frames[0].copy()
        merged = merged.rename(columns={find_id_col(merged): ID_COL})
        for nxt in frames[1:]:
            nxt = nxt.rename(columns={find_id_col(nxt): ID_COL})
            dup = [c for c in nxt.columns if c != ID_COL and c in merged.columns]
            merged = merged.merge(nxt.drop(columns=dup), on=ID_COL, how="outer")
        out_path = out_dir / f"mask{prefix}_feature--{mod}.csv"
        merged.sort_values(ID_COL).to_csv(out_path, index=False)
        print(f"[ORGAN] published {out_path} cases={len(merged)} features={merged.shape[1] - 1}")


def _organ_roi_patch_features(vol: np.ndarray, mask: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
    organ = np.asarray(mask) > 0
    if not bool(np.any(organ)):
        stats = b_patch._safe_stats(np.array([], dtype=np.float64))
        glcm = b_patch._glcm_features_2d(np.array([[np.nan]], dtype=np.float32))
        grad_stats = b_patch._safe_stats(np.array([], dtype=np.float64))
        tex = {**stats, **glcm}
        top = {f"grad_{k}": v for k, v in grad_stats.items()}
        return tex, top

    sl = bbox_from_mask(organ, margin=0)
    vol_c = np.asarray(vol[sl], dtype=np.float32)
    organ_c = organ[sl]

    stats = b_patch._safe_stats(vol_c[organ_c])
    central_z = vol_c.shape[0] // 2
    glcm = b_patch._masked_glcm_features_2d(vol_c[central_z], organ_c[central_z])
    grad = b_patch._gradient_magnitude(vol_c)
    grad_stats = b_patch._safe_stats(grad[organ_c])

    tex = {**stats, **glcm}
    top = {f"grad_{k}": v for k, v in grad_stats.items()}
    return tex, top


def extract_patchstyle_tables(root: Path, mask_folder: str = MASK_FOLDER) -> dict[str, pd.DataFrame]:
    root = Path(root)
    t2w_dir, adc_dir, mask_dir = feature_paths(root, mask_folder)
    case_ids = list_case_ids(t2w_dir, ext=".mha")

    rows = {"texture_t2w": [], "texture_adc": [], "topology_t2w": [], "topology_adc": []}
    skipped = 0
    for cid in case_ids:
        mask_path = find_mask_path(mask_dir, cid)
        t2w_path = os.path.join(t2w_dir, f"{cid}.mha")
        adc_path = os.path.join(adc_dir, f"{cid}.mha")
        if mask_path is None or not (os.path.exists(t2w_path) and os.path.exists(adc_path)):
            _vprint(f"[patchstyle] missing volume/mask for {cid}, skipping")
            skipped += 1
            continue
        mask, _ = read_volume(mask_path)
        t2w, _ = read_volume(t2w_path)
        adc, _ = read_volume(adc_path)
        t2w_tex, t2w_top = _organ_roi_patch_features(t2w, mask)
        adc_tex, adc_top = _organ_roi_patch_features(adc, mask)
        rows["texture_t2w"].append({"case_id": cid, **t2w_tex})
        rows["texture_adc"].append({"case_id": cid, **adc_tex})
        rows["topology_t2w"].append({"case_id": cid, **t2w_top})
        rows["topology_adc"].append({"case_id": cid, **adc_top})

    out = {}
    for key, recs in rows.items():
        df = pd.DataFrame(recs)
        if df.empty:
            raise RuntimeError(
                f"No patch-style organ features extracted for {key}. Check that "
                f"{root} has {T2W_FOLDER}/{ADC_FOLDER}/{MASK_FOLDER} volumes."
            )
        out[key] = df.sort_values("case_id").reset_index(drop=True)
    if skipped:
        print(f"[PATCHSTYLE] skipped {skipped} case(s) with missing volume/mask")
    return out


ATOL = 1e-8
RTOL = 1e-6
EPS = 1e-12
CONST_FRAC = 0.90
TH = 0.75


def find_id_col(df: pd.DataFrame) -> str:
    for col in df.columns:
        if str(col).strip().lower() == "case_id":
            return col
    return df.columns[0]


def constant_fraction(x: np.ndarray, atol=ATOL, rtol=RTOL) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return 1.0
    ref = float(np.median(x))
    tol = atol + rtol * max(1.0, abs(ref), np.max(np.abs(x)))
    return float(np.mean(np.abs(x - ref) <= tol))


def icc2_1_absolute(x1: np.ndarray, x2: np.ndarray) -> float:
    m = np.vstack([x1, x2]).T
    m = m[~np.isnan(m).any(axis=1)]
    n, k = m.shape
    if n < 3:
        return np.nan

    mean_subject = m.mean(axis=1, keepdims=True)
    mean_rater = m.mean(axis=0, keepdims=True)
    grand = m.mean()

    MSR = (k * ((mean_subject.squeeze() - grand) ** 2).sum()) / (n - 1)
    MSC = (n * ((mean_rater.squeeze() - grand) ** 2).sum()) / (k - 1)

    resid = m - mean_subject - mean_rater + grand
    MSE = (resid ** 2).sum() / ((n - 1) * (k - 1))

    denom = MSR + (k - 1) * MSE + (k * (MSC - MSE) / n)
    if abs(denom) < EPS:
        return np.nan
    return float((MSR - MSE) / denom)


def _align_common(df1: pd.DataFrame, df2: pd.DataFrame):
    id1 = find_id_col(df1)
    id2 = find_id_col(df2)

    df1 = df1.rename(columns={id1: "case_id"}).copy()
    df2 = df2.rename(columns={id2: "case_id"}).copy()

    df1["case_id"] = df1["case_id"].astype(str).str.strip()
    df2["case_id"] = df2["case_id"].astype(str).str.strip()

    ids = df1[["case_id"]].merge(df2[["case_id"]], on="case_id", how="inner")
    df1 = ids.merge(df1, on="case_id", how="left")
    df2 = ids.merge(df2, on="case_id", how="left")

    feats1 = [c for c in df1.columns if c != "case_id"]
    feats2 = [c for c in df2.columns if c != "case_id"]
    common = sorted(list(set(feats1) & set(feats2)))

    df1[common] = df1[common].apply(pd.to_numeric, errors="coerce")
    df2[common] = df2[common].apply(pd.to_numeric, errors="coerce")

    df1 = df1.copy()
    df2 = df2.copy()
    return df1, df2, common


def _icc_bins_counts(icc_vals: pd.Series) -> dict:
    x = icc_vals.dropna().astype(float).to_numpy()
    return {
        "icc [0.0,0.5)": int(np.sum((x >= 0.0) & (x < 0.5))),
        "icc [0.5,0.75)": int(np.sum((x >= 0.5) & (x < 0.75))),
        "icc [0.75,0.9)": int(np.sum((x >= 0.75) & (x < 0.9))),
        "icc [0.9,1.0]": int(np.sum((x >= 0.9) & (x <= 1.0))),
    }


def _icc_bins_counts_th(icc_vals: pd.Series, th: float) -> dict:
    x = icc_vals.dropna().astype(float).to_numpy()
    return {f"icc [{TH},1]": int(np.sum((x >= th) & (x <= 1.0)))}


def _icc_bins_counts_both(icc_vals: pd.Series) -> dict:
    x = icc_vals.dropna().astype(float).to_numpy()
    return {
        "icc_both [0.0,0.5)": int(np.sum((x >= 0.0) & (x < 0.5))),
        "icc_both [0.5,0.75)": int(np.sum((x >= 0.5) & (x < 0.75))),
        "icc_both [0.75,0.9)": int(np.sum((x >= 0.75) & (x < 0.9))),
        "icc_both [0.9,1.0]": int(np.sum((x >= 0.9) & (x <= 1.0))),
    }


def _icc_bins_counts_both_th(icc_vals: pd.Series, th: float) -> dict:
    x = icc_vals.dropna().astype(float).to_numpy()
    return {f"icc_both [{TH},1]": int(np.sum((x >= th) & (x <= 1.0)))}


def compute_icc_table_with_constant_append(df_ai1: pd.DataFrame, df_ai2: pd.DataFrame):
    df1, df2, common = _align_common(df_ai1, df_ai2)

    const_info, keep_feats = [], []
    for f in common:
        a = df1[f].to_numpy(dtype=float)
        b = df2[f].to_numpy(dtype=float)

        fa = constant_fraction(a)
        fb = constant_fraction(b)

        if max(fa, fb) >= CONST_FRAC:
            if fa >= fb:
                ref = float(np.nanmedian(a))
                frac = fa
            else:
                ref = float(np.nanmedian(b))
                frac = fb

            pct = int(round(frac * 100))
            const_info.append((f, f"Constant {ref:.6g} in {pct}%"))
        else:
            keep_feats.append(f)

    rows = []
    for f in keep_feats:
        a = df1[f].to_numpy(dtype=float)
        b = df2[f].to_numpy(dtype=float)
        rows.append((f, icc2_1_absolute(a, b)))

    icc_df = pd.DataFrame(rows, columns=["feature", "icc2_1_abs"]).sort_values(
        "icc2_1_abs", ascending=False
    )

    stable = icc_df.loc[
        icc_df["icc2_1_abs"].notna() & (icc_df["icc2_1_abs"] >= TH),
        "feature",
    ].tolist()

    const_df = pd.DataFrame(const_info, columns=["feature", "icc2_1_abs"])
    out_df = pd.concat([icc_df, const_df], ignore_index=True) if len(const_df) > 0 else icc_df

    total_common = len(common)
    n_constant = len(const_info)
    n_remaining = len(keep_feats)

    bins = _icc_bins_counts(icc_df["icc2_1_abs"])
    bins_th = _icc_bins_counts_th(icc_df["icc2_1_abs"], TH)

    summary_row = {
        "Total": total_common,
        "Constant": n_constant,
        "Remaining": n_remaining,
        **bins,
        **bins_th,
    }

    return stable, df1, df2, out_df, summary_row, icc_df[["feature", "icc2_1_abs"]].copy()


def run_icc_stage(mask1: dict[str, pd.DataFrame], mask2: dict[str, pd.DataFrame], icc_dir: Path) -> dict[str, list[str]]:
    icc_dir.mkdir(parents=True, exist_ok=True)
    PAIRS = {key: (mask1[key], mask2[key]) for key in ("texture_adc", "texture_t2w", "topology_adc", "topology_t2w")}

    OUT_ICC_ALL = icc_dir / "rank_icc.xlsx"
    OUT_KEEP_JSON = icc_dir / "selected_features_icc.json"

    keep = {}
    summary_records = []

    with pd.ExcelWriter(OUT_ICC_ALL, engine="openpyxl") as writer:
        stable_texture_adc, _, _, icc_texture_adc_df, sum_texture_adc, icc_texture_adc_core = compute_icc_table_with_constant_append(
            PAIRS["texture_adc"][0], PAIRS["texture_adc"][1]
        )
        icc_texture_adc_df.to_excel(writer, sheet_name="texture_adc", index=False)
        keep["texture_adc"] = stable_texture_adc

        stable_texture_t2w, _, _, icc_texture_t2w_df, sum_texture_t2w, icc_texture_t2w_core = compute_icc_table_with_constant_append(
            PAIRS["texture_t2w"][0], PAIRS["texture_t2w"][1]
        )
        icc_texture_t2w_df.to_excel(writer, sheet_name="texture_t2w", index=False)
        keep["texture_t2w"] = stable_texture_t2w

        stable_topology_adc, _, _, icc_topology_adc_df, sum_topology_adc, icc_topology_adc_core = compute_icc_table_with_constant_append(
            PAIRS["topology_adc"][0], PAIRS["topology_adc"][1]
        )
        icc_topology_adc_df.to_excel(writer, sheet_name="topology_adc", index=False)
        keep["topology_adc"] = stable_topology_adc

        stable_topology_t2w, _, _, icc_topology_t2w_df, sum_topology_t2w, icc_topology_t2w_core = compute_icc_table_with_constant_append(
            PAIRS["topology_t2w"][0], PAIRS["topology_t2w"][1]
        )
        icc_topology_t2w_df.to_excel(writer, sheet_name="topology_t2w", index=False)
        keep["topology_t2w"] = stable_topology_t2w

        tex_both = icc_texture_adc_core.merge(
            icc_texture_t2w_core,
            on="feature",
            how="inner",
            suffixes=("_adc", "_t2w"),
        )
        tex_both["icc_both_min"] = tex_both[
            ["icc2_1_abs_adc", "icc2_1_abs_t2w"]
        ].min(axis=1)
        tex_both_bins = _icc_bins_counts_both(tex_both["icc_both_min"])
        tex_both_bins_th = _icc_bins_counts_both_th(tex_both["icc_both_min"], TH)

        top_both = icc_topology_adc_core.merge(
            icc_topology_t2w_core,
            on="feature",
            how="inner",
            suffixes=("_adc", "_t2w"),
        )
        top_both["icc_both_min"] = top_both[
            ["icc2_1_abs_adc", "icc2_1_abs_t2w"]
        ].min(axis=1)
        top_both_bins = _icc_bins_counts_both(top_both["icc_both_min"])
        top_both_bins_th = _icc_bins_counts_both_th(top_both["icc_both_min"], TH)

        summary_records.append({"type": "texture_adc", **sum_texture_adc, **tex_both_bins, **tex_both_bins_th})
        summary_records.append({"type": "texture_t2w", **sum_texture_t2w, **tex_both_bins, **tex_both_bins_th})
        summary_records.append({"type": "topology_adc", **sum_topology_adc, **top_both_bins, **top_both_bins_th})
        summary_records.append({"type": "topology_t2w", **sum_topology_t2w, **top_both_bins, **top_both_bins_th})

        summary_cols = [
            "type",
            "Total",
            "Constant",
            "Remaining",
            "icc [0.0,0.5)",
            "icc [0.5,0.75)",
            "icc [0.75,0.9)",
            "icc [0.9,1.0]",
            "icc_both [0.0,0.5)",
            "icc_both [0.5,0.75)",
            "icc_both [0.75,0.9)",
            "icc_both [0.9,1.0]",
            "",
            f"icc [{TH},1]",
            f"icc_both [{TH},1]",
        ]

        summary_df = pd.DataFrame(summary_records)
        summary_df[""] = ""

        for c in summary_cols:
            if c not in summary_df.columns:
                summary_df[c] = 0 if c not in ["type", ""] else ""

        summary_df = summary_df[summary_cols]

        sum_row = {"type": "SUM", "": ""}
        for c in summary_cols:
            if c in ["type", ""]:
                continue
            sum_row[c] = pd.to_numeric(summary_df[c], errors="coerce").fillna(0).sum()

        summary_df = pd.concat(
            [summary_df, pd.DataFrame([sum_row])[summary_cols]],
            ignore_index=True,
        )

        summary_df.to_excel(writer, sheet_name="summary", index=False)

        wb = writer.book
        if "summary" in wb.sheetnames:
            wb._sheets.insert(0, wb._sheets.pop(wb.sheetnames.index("summary")))

    with OUT_KEEP_JSON.open("w", encoding="utf-8") as f:
        json.dump(keep, f, indent=2)

    print(f"[SAVED] {OUT_KEEP_JSON}")
    print(f"[SAVED] {OUT_ICC_ALL}")
    return keep


def _clean_case_id_value(value) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    return s


def load_main_labels(dataset_xlsx: Path = DATASET_XLSX) -> pd.DataFrame:
    lab = pd.read_excel(dataset_xlsx, dtype={ID_COL: str})
    lab.columns = [str(c).strip() for c in lab.columns]
    lab = lab.rename(columns={find_id_col(lab): ID_COL})

    required = [ID_COL, CLIN_COL, ZONE_COL]
    missing = [c for c in required if c not in lab.columns]
    if missing:
        raise KeyError(f"Missing required label columns in {dataset_xlsx}: {missing}")

    lab = lab[required].copy()
    lab[ID_COL] = lab[ID_COL].map(_clean_case_id_value)
    lab = lab.dropna(subset=[ID_COL])
    return lab.drop_duplicates(ID_COL, keep="first").reset_index(drop=True)


def prep_feats(df: pd.DataFrame, suffix: str | None = None, keep: set[str] | None = None) -> pd.DataFrame:
    df = df.rename(columns={find_id_col(df): ID_COL}).copy()
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    feats = [c for c in df.columns if c != ID_COL]
    if keep is not None:
        feats = [f for f in feats if f in keep]
    df = df[[ID_COL] + feats]
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce")
    if suffix:
        df = df.rename(columns={c: f"{c}{suffix}" for c in feats})
    return df


def build_concat(labels: pd.DataFrame, tables: dict[str, pd.DataFrame], keep_feats: dict[str, set[str]]) -> pd.DataFrame:
    t2w_top = prep_feats(tables["topology_t2w"], "_t2w-top", keep=keep_feats["topology_t2w"])
    t2w_tex = prep_feats(tables["texture_t2w"], "_t2w-tex", keep=keep_feats["texture_t2w"])
    adc_top = prep_feats(tables["topology_adc"], "_adc-top", keep=keep_feats["topology_adc"])
    adc_tex = prep_feats(tables["texture_adc"], "_adc-tex", keep=keep_feats["texture_adc"])

    all_df = labels.merge(t2w_top, on=ID_COL).merge(t2w_tex, on=ID_COL).merge(adc_top, on=ID_COL).merge(adc_tex, on=ID_COL)
    return all_df


def build_modality(labels: pd.DataFrame, tables: dict[str, pd.DataFrame], modality: str, keep_feats: dict[str, set[str]]) -> pd.DataFrame:
    top = prep_feats(tables[f"topology_{modality}"], f"_{modality}-top", keep=keep_feats[f"topology_{modality}"])
    tex = prep_feats(tables[f"texture_{modality}"], f"_{modality}-tex", keep=keep_feats[f"texture_{modality}"])
    all_df = labels.merge(tex, on=ID_COL).merge(top, on=ID_COL)
    return all_df


def paired_block(t2w: pd.DataFrame, adc: pd.DataFrame, suffix: str, op: str) -> pd.DataFrame:
    common = sorted((set(t2w.columns) & set(adc.columns)) - {ID_COL})
    if not common:
        raise RuntimeError(f"No common features for {suffix}")
    merged = t2w[[ID_COL] + common].merge(adc[[ID_COL] + common], on=ID_COL, suffixes=("_t2w", "_adc"))
    a = merged[[f"{c}_t2w" for c in common]].to_numpy(dtype=float)
    b = merged[[f"{c}_adc" for c in common]].to_numpy(dtype=float)
    out = a * b if op == "hada" else (a - b)
    return pd.concat(
        [merged[[ID_COL]].reset_index(drop=True), pd.DataFrame(out, columns=[f"{c}{suffix}" for c in common])],
        axis=1,
    )


def build_pairwise(labels: pd.DataFrame, tables: dict[str, pd.DataFrame], op: str, keep_feats: dict[str, set[str]]) -> pd.DataFrame:
    t2w_top = prep_feats(tables["topology_t2w"], keep=keep_feats["topology_t2w"])
    t2w_tex = prep_feats(tables["texture_t2w"], keep=keep_feats["texture_t2w"])
    adc_top = prep_feats(tables["topology_adc"], keep=keep_feats["topology_adc"])
    adc_tex = prep_feats(tables["texture_adc"], keep=keep_feats["texture_adc"])
    tex = paired_block(t2w_tex, adc_tex, f"_{op}-tex", op)
    top = paired_block(t2w_top, adc_top, f"_{op}-top", op)
    all_df = labels.merge(tex, on=ID_COL).merge(top, on=ID_COL)
    return all_df


def write_book(path: Path, df: pd.DataFrame, quiet: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.sort_values(ID_COL).to_csv(tmp, index=False)
    os.replace(tmp, path)
    if not quiet:
        print(f"[SAVED] {path}")


def merge_sheetwise(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = frames[0].copy()
    for nxt in frames[1:]:
        dup = [c for c in nxt.columns if c != ID_COL and c in out.columns]
        out = out.merge(nxt.drop(columns=dup), on=ID_COL, how="inner")
    return out


# Multi-way combos mirroring b_patch.py's PATCH_FEATURE_SETS naming (c=concat, d=diff,
# h=hada): each is the column-wise union of its named component organ workbooks.
FUSION_COMPONENTS = {
    "fusion(cd)": ("concat", "diff"),
    "fusion(dh)": ("diff", "hada"),
    "fusion(ch)": ("concat", "hada"),
    "fusion(cdh)": ("concat", "diff", "hada"),
}


def build_requested_books(
    labels: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    keep_feats: dict[str, set[str]],
    file_names: list[str],
) -> dict[str, pd.DataFrame]:
    requested = set(file_names)
    unknown = sorted(requested - set(FEATURE_FILES))
    if unknown:
        raise ValueError(f"Unknown feature files requested: {unknown}")

    fusion_files = {f"feature--{name}.csv" for name in FUSION_COMPONENTS}

    books: dict[str, pd.DataFrame] = {}
    if requested & ({"feature--t2w.csv", "feature--adc.csv", "feature--concat.csv", "feature--hada.csv", "feature--diff.csv"} | fusion_files):
        books["t2w"] = build_modality(labels, tables, "t2w", keep_feats)
        books["adc"] = build_modality(labels, tables, "adc", keep_feats)
    if requested & ({"feature--concat.csv", "feature--hada.csv", "feature--diff.csv"} | fusion_files):
        books["concat"] = build_concat(labels, tables, keep_feats)
    if requested & ({"feature--hada.csv", "feature--diff.csv"} | fusion_files):
        books["hada"] = build_pairwise(labels, tables, "hada", keep_feats)
        books["diff"] = build_pairwise(labels, tables, "diff", keep_feats)

    for fusion_name, parts in FUSION_COMPONENTS.items():
        if f"feature--{fusion_name}.csv" in requested:
            books[fusion_name] = merge_sheetwise([books[part] for part in parts])

    workbook_map = {
        "feature--t2w.csv": books.get("t2w"),
        "feature--adc.csv": books.get("adc"),
        "feature--hada.csv": books.get("hada"),
        "feature--diff.csv": books.get("diff"),
        "feature--concat.csv": books.get("concat"),
        **{f"feature--{name}.csv": books.get(name) for name in FUSION_COMPONENTS},
    }
    out: dict[str, pd.DataFrame] = {}
    for file_name in requested:
        work = workbook_map[file_name]
        if work is None:
            raise RuntimeError(f"Unable to build requested workbook {file_name}")
        out[file_name] = work
    return out


def build_all_workbooks(
    labels: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    keep_feats: dict[str, set[str]],
    out_dir: Path,
    file_names: list[str] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    file_names = list(file_names) if file_names is not None else list(FEATURE_FILES)
    if not file_names:
        return
    books = build_requested_books(labels, tables, keep_feats, file_names)
    completed = 0
    total = len(books)
    for file_name, work in books.items():
        completed += 1
        log_progress("WORKBOOK", completed, total)
        write_book(out_dir / file_name, work)


def run_organ(same: bool = False) -> None:
    raw_dir = ORGAN_SAME_RAW_DIR if same else ORGAN_RAW_DIR
    out_dir = ORGAN_SAME_WORKBOOK_DIR if same else ORGAN_ONLY_WORKBOOK_DIR

    expected = [raw_dir / f"mask{p}_feature--{m}.csv" for p in ("1", "2") for m in ("adc", "t2w")]
    expected += [raw_dir / "rank_icc.xlsx", raw_dir / "selected_features_icc.json"]
    expected += [out_dir / f for f in FEATURE_FILES]
    if all(p.exists() for p in expected):
        print(f"[ORGAN] all final outputs already exist; skipping ({raw_dir.parent.name})")
        return

    if same:
        mask1 = extract_patchstyle_tables(MAIN_ROOT, MAIN_AI1_MASK)
        mask2 = extract_patchstyle_tables(MAIN_ROOT, MAIN_AI2_MASK)
        _publish_merged_raw_features(mask1, raw_dir, "1")
        _publish_merged_raw_features(mask2, raw_dir, "2")
    else:
        k_names, m_names, s_names, v_names, specs = load_feature_specs()
        for prefix, mask_folder in (
            ("1", MAIN_AI1_MASK),
            ("2", MAIN_AI2_MASK),
        ):
            t2w_dir, adc_dir, mask_dir = feature_paths(MAIN_ROOT, mask_folder)
            stream_handcrafted_modality(adc_dir, mask_dir, raw_dir / f"mask{prefix}_feature--adc.csv",
                                        specs, k_names, m_names, s_names, v_names)
            stream_handcrafted_modality(t2w_dir, mask_dir, raw_dir / f"mask{prefix}_feature--t2w.csv",
                                        specs, k_names, m_names, s_names, v_names)
        mask1 = _read_mask_tables_handcrafted(raw_dir, "1")
        mask2 = _read_mask_tables_handcrafted(raw_dir, "2")
    keep = run_icc_stage(mask1, mask2, raw_dir)
    build_all_workbooks(load_main_labels(), mask2, keep, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organ-feature stage. Default branch extracts handcrafted "
                    "Laws/Gabor/Haralick/Gradient/Gray radiomics + topology with "
                    "AI1/AI2 ICC stability (B_Organ, the canonical organ used in "
                    "fusion and external testing). --same extracts the patch model's "
                    "raw-image features over the organ ROI (B_Organ_same), kept for "
                    "the internal comparison with A_Patch."
    )
    parser.add_argument(
        "--same",
        action="store_true",
        help="Use the patch-style branch (B_Organ_same) instead of the default "
             "handcrafted-radiomics branch (B_Organ).",
    )
    args = parser.parse_args()
    run_organ(same=args.same)


if __name__ == "__main__":
    main()
