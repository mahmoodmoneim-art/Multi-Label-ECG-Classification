# ================================
# Key safeguards implemented here:
# - source-defined external cohorts (never the aggregated Challenge 2020 bundle)
# - patient-wise split audit before windowing
# - common physical sampling frequency and window duration
# - explicit label-mapping and provenance audits
# - identical training budget for proposed, baseline, and ablation models
# - true post-hoc fixed-threshold comparison using identical probabilities
# - paired patient-level bootstrap confidence intervals and p-values
# - reproducibility/configuration manifests suitable for Zenodo deposition
# ============================================================

import sys
import subprocess

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "wfdb", "scipy", "iterative-stratification"
])

import os
import re
import gc
import ast
import math
import glob
import copy
import random
import warnings
import hashlib
import json
import platform
import time
import tempfile
import shutil
import urllib.request
import urllib.error
from functools import lru_cache
from datetime import datetime, timezone
from math import gcd
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import wfdb
from scipy import signal as scipy_signal
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    roc_curve,
    precision_recall_curve,
    jaccard_score,
    multilabel_confusion_matrix,
    classification_report
)
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from IPython.display import display


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_all(RANDOM_STATE)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


class CFG:
    # -----------------------------
    # Internal PTB-XL reformatted dataset
    # -----------------------------
    PTBXL_PATH_CANDIDATES = [
        "/kaggle/input/ptb-xl-dataset-reformatted",
        "/kaggle/input/ptb-xl-dataset-reforma",
        "/kaggle/input/datasets/khyeh0719/ptb-xl-dataset-reformatted",
    ]

    # -----------------------------
    # Source-defined external cohorts. The complete Challenge 2020 root is
    # NEVER evaluated as one cohort because it contains PTB-XL, CPSC,
    # CPSC-Extra and INCART. Only its Georgia source is retained here.
    # -----------------------------
    EXTERNAL_DATASETS = {
        "Georgia 12-Lead (Challenge 2020 source)": {
            "paths": [
                "/kaggle/input/datasets/gamalasran/physionet-challenge-2020",
                "/kaggle/input/datasets/gamalasran/physionet-challenge-2020/classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2",
            ],
            "required_path_tokens": ["georgia"],
            "source_note": "Georgia source only; PTB/PTB-XL/CPSC/CPSC-Extra/INCART excluded",
        },
        "CPSC-Extra (China 12-Lead ECG Challenge)": {
            "paths": [
            "/kaggle/input/datasets/organizations/physionet/china-12lead-ecg-challenge-database",
            "/kaggle/input/datasets/organizations/physionet/china-12lead-ecg-challenge-database/Training_2",
            ],
            "required_path_tokens": [],
            "source_note": "Unused/extra CPSC 2018 source distributed as China 12-Lead",
        },
        "CPSC 2018": {
            "paths": [
            "/kaggle/input/datasets/organizations/physionet/china-physiological-signal-challenge-in-2018",
            "/kaggle/input/datasets/organizations/physionet/china-physiological-signal-challenge-in-2018/Training_WFDB",
            ],
            "required_path_tokens": [],
            "source_note": "CPSC 2018 public training cohort",
        },
        "St Petersburg INCART": {
            "paths": [
            "/kaggle/input/datasets/organizations/physionet/st-petersburg-incart-12lead-arrhythmia-database",
            "/kaggle/input/datasets/organizations/physionet/st-petersburg-incart-12lead-arrhythmia-database/Training_StPetersburg",
            ],
            "required_path_tokens": [],
            "source_note": "INCART source cohort",
        },
    }

    OUT_DIR = "/kaggle/working/reviewer_external_dataset_results"

    LABEL_COLS = ["NORM", "MI", "STTC", "CD", "HYP"]
    C = len(LABEL_COLS)

    # All sample counts are derived from physical time after resampling.
    TARGET_FS = 500
    WINDOW_SECONDS = 4.0
    STRIDE_TRAIN_SECONDS = 1.0
    STRIDE_EVAL_SECONDS = 4.0
    T_WINDOW = int(TARGET_FS * WINDOW_SECONDS)
    STRIDE_TRAIN = int(TARGET_FS * STRIDE_TRAIN_SECONDS)
    STRIDE_EVAL = int(TARGET_FS * STRIDE_EVAL_SECONDS)

    # Half the previous peak activation/pre-fetch memory. This does not
    # change dataset records, labels, model layers, or inference thresholds.
    BATCH_SIZE = 32
    EPOCHS = 10
    # Fair-comparison rule: all trained variants receive the same budget.
    BASELINE_EPOCHS = EPOCHS

    LR = 2e-4
    BASELINE_LR = 2e-4
    WD = 1e-4

    AMP = True
    # Avoid retaining two dataloader worker pools (train + validation), each
    # of which keeps a reference to the entire in-memory ECG dictionary.
    NUM_WORKERS = 2
    PERSISTENT_WORKERS = False
    PREFETCH_FACTOR = 1

    D_MODEL = 256
    TX_DEPTH = 3
    NHEAD = 8
    DROPOUT = 0.10

    EMA_DECAY = 0.999
    WARMUP_EPOCHS = 1
    THR_GRID = 101
    THR_MIN = 0.05
    THR_MAX = 0.95

    RUN_EXTERNAL_DATASETS = True
    RUN_BASELINES = True
    RUN_ABLATIONS = True
    RUN_PAIRED_BOOTSTRAP = True
    RUN_FIVE_FOLD_CV = True  # proposed-model robustness analysis; distinct from fixed held-out test
    RUN_TSNE = True
    # External feature extraction repeats a full pass over long WFDB records.
    # It is not needed for the reviewer's quantitative claims; internal t-SNE remains enabled.
    RUN_EXTERNAL_TSNE = False

    # Runtime controls. The 15-minute reserve is used for saving results and
    # notebook cleanup. Exact wall time still depends on Kaggle GPU/CPU load.
    # Fixed run budget. Optional studies are skipped if their estimated
    # minimum duration would exceed the reserve; incomplete folds are rejected.
    MAX_RUNTIME_HOURS = 11.0
    MIN_COMPLETED_CV_EPOCHS = 5
    FINALIZATION_RESERVE_MINUTES = 15
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_MIN_EPOCHS = 5

    STRICT_PATIENT_AUDIT = True
    AUTO_REBUILD_LEAKING_SPLITS = True
    ALLOW_OFFICIAL_METADATA_DOWNLOAD = True
    PTBXL_METADATA_URLS = [
        # v1.0.1/v1.0.2 contain the historical 21,837-record release used by
        # the reformatted Kaggle dataset. v1.0.3 contains 21,799 records.
        "https://physionet.org/files/ptb-xl/1.0.1/ptbxl_database.csv",
        "https://physionet.org/files/ptb-xl/1.0.2/ptbxl_database.csv",
        "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv",
    ]
    PATIENTWISE_VALID_FRACTION = 0.10
    PATIENTWISE_TEST_FRACTION = 0.10
    ALLOW_TEXT_LABEL_FALLBACK = False
    RUN_SIGNAL_FINGERPRINT_AUDIT = True
    FINGERPRINT_SECONDS = 10
    # 3,000 patient-level resamples provide stable CIs while avoiding the large
    # CPU cost of 10,000 repeats. Raise this only for the final archival run.
    BOOTSTRAP_REPEATS = 3000
    BOOTSTRAP_SEED = 42
    ZENODO_DOI = "TO_BE_ADDED_BEFORE_SUBMISSION"

    TSNE_MAX_SAMPLES = 2500
    FIG_DPI = 300

    # Use None for all available records.
    # Set to a number, e.g., 2000, if Kaggle RAM/time is limited.
    EXTERNAL_MAX_RECORDS_PER_DATASET = None
    EXTERNAL_RANDOM_SAMPLE = False

    # Each worker keeps only the most recently used long WFDB records. Without
    # this cache, a 30-minute INCART record is read/resampled once per window.
    EXTERNAL_RECORD_CACHE_SIZE = 1
    SIGNAL_CSV_CHUNK_ROWS = 25000


os.makedirs(CFG.OUT_DIR, exist_ok=True)

RUN_STARTED_MONOTONIC = time.monotonic()


def elapsed_runtime_hours():
    return (time.monotonic() - RUN_STARTED_MONOTONIC) / 3600.0


def runtime_seconds_remaining(include_finalization_reserve=True):
    reserve = 60.0 * CFG.FINALIZATION_RESERVE_MINUTES if include_finalization_reserve else 0.0
    return max(0.0, CFG.MAX_RUNTIME_HOURS * 3600.0 -
               (time.monotonic() - RUN_STARTED_MONOTONIC) - reserve)


def runtime_allows_optional_stage(stage_name, estimated_seconds):
    remaining = runtime_seconds_remaining()
    if remaining >= estimated_seconds:
        return True
    print(
        f"RUNTIME BUDGET: skipping {stage_name}; "
        f"{remaining / 60.0:.1f} minutes remain before the finalization reserve."
    )
    return False


def dataloader_runtime_kwargs():
    """Arguments shared by all loaders without invalid worker-zero options."""
    kwargs = {
        "num_workers": CFG.NUM_WORKERS,
        "pin_memory": bool(torch.cuda.is_available()),
    }
    if CFG.NUM_WORKERS > 0:
        kwargs["persistent_workers"] = bool(CFG.PERSISTENT_WORKERS)
        kwargs["prefetch_factor"] = int(CFG.PREFETCH_FACTOR)
    return kwargs

LABEL_COLS = CFG.LABEL_COLS
C = CFG.C


def first_existing_path(path_candidates):
    for p in path_candidates:
        if os.path.exists(p):
            return p
    return None


PTBXL_PATH = first_existing_path(CFG.PTBXL_PATH_CANDIDATES)
if PTBXL_PATH is None:
    raise FileNotFoundError("PTB-XL reformatted dataset folder not found in /kaggle/input.")

EXTERNAL_PATHS = {}
for ds_name, spec in CFG.EXTERNAL_DATASETS.items():
    EXTERNAL_PATHS[ds_name] = first_existing_path(spec["paths"])

print("PTB-XL path:", PTBXL_PATH)
print("\nExternal dataset paths:")
for k, v in EXTERNAL_PATHS.items():
    print(f"{k}: {v}")


# ============================================================
# 2. LABEL MAPPING TO PTB-XL FIVE SUPERCLASSES
# ============================================================

def normalize_dx_token(x):
    if x is None:
        return ""
    x = str(x).strip()
    x = x.replace('"', "").replace("'", "")
    x = x.replace("[", "").replace("]", "")
    x = x.replace("(", "").replace(")", "")
    if x == "" or x.lower() in ["nan", "none", "null"]:
        return ""
    if re.fullmatch(r"\d+\.0+", x):
        x = x.split(".")[0]
    return x.strip()


def split_dx_field(raw):
    if raw is None:
        return []
    raw = str(raw).strip()
    if raw == "" or raw.lower() in ["nan", "none", "null"]:
        return []

    try:
        val = ast.literal_eval(raw)
        if isinstance(val, (list, tuple, set)):
            toks = [normalize_dx_token(item) for item in val]
            return [t for t in toks if t]
    except Exception:
        pass

    raw = raw.replace(";", ",").replace("|", ",")
    raw = raw.replace("[", "").replace("]", "")
    raw = raw.replace("'", "").replace('"', "")

    toks = []
    for part in raw.split(","):
        part = normalize_dx_token(part)
        if part:
            toks.append(part)
    return toks


# SNOMED and short-code mapping to PTB-XL diagnostic superclasses.
SNOMED_TO_SUPERCLASS = {
    # Normal
    "426783006": "NORM", "426177001": "NORM", "251146004": "NORM",
    "NORM": "NORM", "NORMAL": "NORM", "NSR": "NORM", "SR": "NORM",
    "426783006.0": "NORM",

    # Myocardial infarction
    "164865005": "MI", "164866006": "MI", "164867002": "MI",
    "164861001": "MI", "57054005": "MI", "54329005": "MI",
    "22298006": "MI", "233843008": "MI",
    "MI": "MI", "AMI": "MI", "IMI": "MI", "ASMI": "MI",
    "ILMI": "MI", "ALMI": "MI", "INJAS": "MI", "INJAL": "MI",
    "INJIN": "MI", "INJLA": "MI", "IPLMI": "MI", "IPMI": "MI",

    # ST/T change
    "164934002": "STTC", "59931005": "STTC", "428750005": "STTC",
    "164930006": "STTC", "164931005": "STTC", "164947007": "STTC",
    "428417006": "STTC", "111975006": "STTC", "164917005": "STTC",
    "429622005": "STTC", "39732003": "STTC", "47665007": "STTC",
    "STTC": "STTC", "STD": "STTC", "STE": "STTC",
    "TAB": "STTC", "TAb": "STTC", "TINV": "STTC", "TInv": "STTC",
    "NDT": "STTC", "NST": "STTC", "LNGQT": "STTC", "LQT": "STTC",
    "QAB": "STTC",

    # Conduction disturbance
    "270492004": "CD", "164909002": "CD", "59118001": "CD",
    "713427006": "CD", "713426002": "CD", "445118002": "CD",
    "445211001": "CD", "698252002": "CD", "195042002": "CD",
    "195039008": "CD", "27885002": "CD", "233917008": "CD",
    "6374002": "CD", "733534002": "CD", "10370003": "CD",
    "251173003": "CD",
    "CD": "CD", "LBBB": "CD", "RBBB": "CD", "IRBBB": "CD",
    "CRBBB": "CD", "CLBBB": "CD", "1AVB": "CD", "IAVB": "CD",
    "2AVB": "CD", "3AVB": "CD", "AVB": "CD", "LAFB": "CD",
    "LPFB": "CD", "IVCD": "CD", "NSIVCB": "CD", "WPW": "CD",

    # Hypertrophy
    "164873001": "HYP", "89792004": "HYP", "266249003": "HYP",
    "251199005": "HYP",
    "HYP": "HYP", "LVH": "HYP", "RVH": "HYP",
    "LAO/LAE": "HYP", "RAO/RAE": "HYP", "LAE": "HYP", "RAE": "HYP",
}

# Correct clinical meanings used in every table and exported audit.
SUPERCLASS_DESCRIPTIONS = {
    "NORM": "Normal ECG",
    "MI": "Myocardial infarction",
    "STTC": "ST-segment/T-wave change",
    "CD": "Conduction disturbance",
    "HYP": "Cardiac hypertrophy",
}

MAPPING_RULES = {
    "NORM": "Explicit normal ECG/rhythm code; removed if any mapped abnormal superclass is present.",
    "MI": "Diagnostic codes explicitly denoting myocardial infarction.",
    "STTC": "Compatible ST-segment or T-wave abnormality codes.",
    "CD": "Compatible atrioventricular, bundle-branch, fascicular, or intraventricular conduction codes.",
    "HYP": "Compatible atrial or ventricular hypertrophy/enlargement codes.",
}


def map_dx_tokens_to_labels(dx_tokens):
    labels = {c: 0 for c in LABEL_COLS}
    mapped = []

    for token in dx_tokens:
        token = normalize_dx_token(token)
        if token in SNOMED_TO_SUPERCLASS:
            cls = SNOMED_TO_SUPERCLASS[token]
            labels[cls] = 1
            mapped.append(cls)

    # A normal label is not allowed to coexist with a mapped abnormal
    # superclass. This avoids an artificial NORM/abnormal contradiction.
    if any(labels[c] == 1 for c in ["MI", "STTC", "CD", "HYP"]):
        labels["NORM"] = 0
        mapped = [c for c in mapped if c != "NORM"]

    return labels, sorted(set(mapped))


# Optional fallback text parser for datasets with weak header comments.
# This is used only when no formal Dx field exists.
def fallback_text_to_superclasses(text):
    text_low = str(text).lower()
    labels = {c: 0 for c in LABEL_COLS}

    if any(w in text_low for w in ["normal", "healthy control", "healthy subject"]):
        labels["NORM"] = 1

    if any(w in text_low for w in ["myocardial infarction", " infarction", "old mi", "acute mi"]):
        labels["MI"] = 1

    if any(w in text_low for w in ["st depression", "st elevation", "st-t", "st/t", "t wave", "t-wave", "qt prolong"]):
        labels["STTC"] = 1

    if any(w in text_low for w in ["lbbb", "rbbb", "bundle branch", "av block", "atrioventricular block", "conduction"]):
        labels["CD"] = 1

    if any(w in text_low for w in ["hypertrophy", "lvh", "rvh", "left ventricular hypertrophy"]):
        labels["HYP"] = 1

    mapped = [c for c in LABEL_COLS if labels[c] == 1]
    return labels, mapped


# ============================================================
# 3. LOAD PTB-XL REFORMATTED DATASET
# ============================================================

def read_ptbxl_csv(name):
    path = os.path.join(PTBXL_PATH, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


train_meta = read_ptbxl_csv("train_meta.csv")
valid_meta = read_ptbxl_csv("valid_meta.csv")
test_meta  = read_ptbxl_csv("test_meta.csv")

CHANNEL_COLS = [c for c in pd.read_csv(
    os.path.join(PTBXL_PATH, "train_signal.csv"), nrows=0
).columns if c.startswith("channel-")]
if len(CHANNEL_COLS) != 12:
    raise RuntimeError(f"Expected 12 channel-* columns, found {len(CHANNEL_COLS)}")

print("Channel columns:", CHANNEL_COLS)


SIGNAL_CACHE_DIR = tempfile.mkdtemp(
    prefix="ptbxl_signal_cache_",
    dir="/kaggle/temp" if os.path.isdir("/kaggle/temp") else CFG.OUT_DIR,
)


def read_signal_map_chunked(name):
    """Two CSV passes; ECG samples live in disk-backed float32 arrays, not notebook RAM."""
    path = os.path.join(PTBXL_PATH, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    stem = os.path.splitext(name)[0]
    dtype = {"ecg_id": str, **{c: np.float32 for c in CHANNEL_COLS}}
    sizes = {}
    for chunk in pd.read_csv(path, usecols=["ecg_id"], dtype={"ecg_id": str},
                             chunksize=CFG.SIGNAL_CSV_CHUNK_ROWS):
        for key, count in chunk["ecg_id"].value_counts(sort=False).items():
            sizes[key] = sizes.get(key, 0) + int(count)
    if not sizes:
        raise RuntimeError(f"No ECG signals in {path}")
    source_fs = int(round(float(np.median(list(sizes.values()))) / 10.0))
    if source_fs not in (100, 250, 500, 1000):
        raise RuntimeError(f"Unexpected source frequency {source_fs} Hz in {name}; check signal lengths")
    offsets = {}
    cursor = 0
    for key, n in sizes.items():
        offsets[key] = cursor
        cursor += n
    target_rows = sum(math.ceil(n * CFG.TARGET_FS / source_fs) for n in sizes.values())
    required_bytes = 4 * len(CHANNEL_COLS) * (cursor + (target_rows if source_fs != CFG.TARGET_FS else 0))
    free_bytes = shutil.disk_usage(SIGNAL_CACHE_DIR).free
    if free_bytes < required_bytes + 512 * 1024**2:
        raise RuntimeError(
            f"Insufficient temporary disk for {name}: requires approximately "
            f"{required_bytes / 1024**3:.1f} GiB plus 0.5 GiB free; "
            f"{free_bytes / 1024**3:.1f} GiB currently available."
        )
    raw_path = os.path.join(SIGNAL_CACHE_DIR, stem + "_raw.dat")
    raw = np.memmap(raw_path, dtype=np.float32, mode="w+", shape=(cursor, len(CHANNEL_COLS)))
    written = {key: 0 for key in sizes}
    for chunk in pd.read_csv(path, usecols=["ecg_id", *CHANNEL_COLS], dtype=dtype,
                             chunksize=CFG.SIGNAL_CSV_CHUNK_ROWS):
        for key, rows in chunk.groupby("ecg_id", sort=False):
            values = rows[CHANNEL_COLS].to_numpy(dtype=np.float32, copy=False)
            start = offsets[key] + written[key]
            raw[start:start + len(values)] = values
            written[key] += len(values)
    if written != sizes:
        raise RuntimeError(f"Incomplete signal CSV read: {name}")
    raw.flush()
    if source_fs == CFG.TARGET_FS:
        print(f"{name}: {len(sizes)} records, {cursor} samples, disk-backed at {source_fs} Hz")
        return {key: raw[offsets[key]:offsets[key] + n] for key, n in sizes.items()}, source_fs

    target_path = os.path.join(SIGNAL_CACHE_DIR, stem + "_target.dat")
    target = np.memmap(target_path, dtype=np.float32, mode="w+",
                       shape=(target_rows, len(CHANNEL_COLS)))
    result = {}
    pos = 0
    for key, n in tqdm(sizes.items(), desc=f"Resampling {name}"):
        converted = polyphase_resample_array(raw[offsets[key]:offsets[key] + n],
                                             source_fs, CFG.TARGET_FS)
        end = pos + len(converted)
        target[pos:end] = converted
        result[key] = target[pos:end]
        pos = end
    if pos != target_rows:
        raise RuntimeError(f"Resampling length mismatch for {name}: {pos} versus {target_rows}")
    target.flush()
    del raw
    gc.collect()
    os.remove(raw_path)
    print(f"{name}: {len(sizes)} records resampled {source_fs} -> {CFG.TARGET_FS} Hz on disk")
    return result, source_fs


def polyphase_resample_array(sig, fs, target_fs):
    fs = int(round(float(fs)))
    target_fs = int(round(float(target_fs)))
    if fs <= 0 or target_fs <= 0:
        raise ValueError(f"Invalid sampling frequency: fs={fs}, target={target_fs}")
    if fs == target_fs:
        return np.asarray(sig, dtype=np.float32)
    common = gcd(fs, target_fs)
    up, down = target_fs // common, fs // common
    out = scipy_signal.resample_poly(sig, up=up, down=down, axis=0)
    return np.asarray(out, dtype=np.float32)


def infer_ptbxl_source_fs(*sig_maps):
    lengths = []
    for sig_map in sig_maps:
        lengths.extend([v.shape[0] for v in sig_map.values()])
    if not lengths:
        raise RuntimeError("Cannot infer PTB-XL sampling frequency from empty signal maps.")
    # PTB-XL recordings are 10 s. A median length of 1000 or 5000 therefore
    # corresponds to 100 or 500 Hz, respectively.
    inferred = int(round(float(np.median(lengths)) / 10.0))
    if inferred not in {100, 250, 500, 1000}:
        raise RuntimeError(
            f"Inferred PTB-XL source frequency {inferred} Hz from median length "
            f"{np.median(lengths):.1f}; set/verify the source data before continuing."
        )
    return inferred


train_sig_map, train_fs = read_signal_map_chunked("train_signal.csv")
valid_sig_map, valid_fs = read_signal_map_chunked("valid_signal.csv")
test_sig_map, test_fs = read_signal_map_chunked("test_signal.csv")
if len({train_fs, valid_fs, test_fs}) != 1:
    raise RuntimeError(f"Different PTB-XL split sampling frequencies: {train_fs}, {valid_fs}, {test_fs}")
PTBXL_SOURCE_FS = train_fs
print("Inferred PTB-XL source frequency:", PTBXL_SOURCE_FS, "Hz")


def make_ptbxl_meta_df(meta_df, sig_map, split_name):
    df = meta_df.copy()
    df["ecg_id"] = df["ecg_id"].astype(str)
    df = df[df["ecg_id"].isin(sig_map.keys())].reset_index(drop=True)

    for c in LABEL_COLS:
        if c not in df.columns:
            raise RuntimeError(f"Missing label column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    y = df[LABEL_COLS].values.astype(int)
    df = df[(y.sum(axis=1) > 0)].reset_index(drop=True)

    df["source_dataset"] = "PTB-XL reformatted"
    df["split"] = split_name
    return df


train_meta = make_ptbxl_meta_df(train_meta, train_sig_map, "train")
valid_meta = make_ptbxl_meta_df(valid_meta, valid_sig_map, "valid")
test_meta  = make_ptbxl_meta_df(test_meta, test_sig_map, "test")


def resolve_patient_column(df):
    candidates = ["patient_id", "patient", "subject_id", "subject", "pid"]
    lower_to_original = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_to_original:
            return lower_to_original[c]
    return None


def canonical_ecg_key(value):
    """Normalize ECG identifiers from reformatted and original PTB-XL tables."""
    text = str(value).strip()
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return str(int(float(text)))
    return text


def validate_ptbxl_metadata_file(path):
    """Return True only for a plausible official PTB-XL metadata table."""
    try:
        probe = pd.read_csv(path, usecols=["ecg_id", "patient_id"], low_memory=False)
    except Exception:
        return False
    return (
        len(probe) >= 20000
        and probe["ecg_id"].notna().all()
        and probe["patient_id"].notna().mean() > 0.99
        and probe["ecg_id"].nunique() >= 20000
    )


def metadata_ecg_key_set(path):
    table = pd.read_csv(path, usecols=["ecg_id"], low_memory=False)
    return set(table["ecg_id"].dropna().map(canonical_ecg_key))


def metadata_coverage(path, required_ecg_keys):
    try:
        available = metadata_ecg_key_set(path)
    except Exception:
        return -1, 0
    matched = len(required_ecg_keys & available)
    return matched, len(required_ecg_keys) - matched


def download_official_ptbxl_metadata_candidates(required_ecg_keys=None):
    """Download metadata from all official versions needed for release matching."""
    if not CFG.ALLOW_OFFICIAL_METADATA_DOWNLOAD:
        return []

    errors = []
    downloaded = []
    for url in CFG.PTBXL_METADATA_URLS:
        version_match = re.search(r"/ptb-xl/([^/]+)/", url)
        version = version_match.group(1) if version_match else "unknown"
        destination = os.path.join(
            CFG.OUT_DIR,
            f"ptbxl_database_official_v{version.replace('.', '_')}.csv",
        )
        if os.path.isfile(destination) and validate_ptbxl_metadata_file(destination):
            print(f"Using cached official PTB-XL metadata v{version}:", destination)
            downloaded.append(destination)
            if required_ecg_keys and metadata_coverage(destination, required_ecg_keys)[1] == 0:
                break
            continue

        temporary = destination + ".part"
        try:
            print(f"Downloading official PTB-XL metadata v{version}:", url)
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "PTBXL-Reviewer-Reproducibility/1.0"},
            )
            with urllib.request.urlopen(request, timeout=120) as response, open(temporary, "wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)

            if not validate_ptbxl_metadata_file(temporary):
                raise RuntimeError("Downloaded file failed PTB-XL metadata validation.")
            os.replace(temporary, destination)
            print(f"Official PTB-XL metadata v{version} downloaded and validated:", destination)
            downloaded.append(destination)
            if required_ecg_keys and metadata_coverage(destination, required_ecg_keys)[1] == 0:
                break
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            if os.path.exists(temporary):
                try:
                    os.remove(temporary)
                except OSError:
                    pass

    if errors:
        print("Some official PTB-XL metadata download attempts failed:")
        for error in errors:
            print(" -", error)
    return downloaded


def find_original_ptbxl_metadata_csv(required_ecg_keys):
    """Select the local/official metadata release with maximum ECG-ID coverage."""
    candidates = []
    candidates.extend(glob.glob(os.path.join(PTBXL_PATH, "**", "ptbxl_database.csv"), recursive=True))
    candidates.extend(glob.glob("/kaggle/input/**/ptbxl_database.csv", recursive=True))
    candidates.extend(glob.glob(os.path.join(CFG.OUT_DIR, "ptbxl_database_official*.csv")))
    candidates = sorted(set(os.path.abspath(p) for p in candidates if os.path.isfile(p)))

    def score(path):
        low = path.lower().replace("_", "-")
        value = 0
        if os.path.commonpath([os.path.abspath(PTBXL_PATH), path]) == os.path.abspath(PTBXL_PATH):
            value += 100
        if "ptb-xl" in low or "ptbxl" in low:
            value += 20
        if "challenge-2020" in low or "challenge_2020" in low:
            value -= 1000
        return value

    candidates = [
        p for p in candidates
        if score(p) > -100 and validate_ptbxl_metadata_file(p)
    ]

    # If no local candidate has complete coverage, compare every official
    # release rather than accepting the first valid (but mismatched) file.
    local_coverages = [metadata_coverage(p, required_ecg_keys)[0] for p in candidates]
    if not local_coverages or max(local_coverages) < len(required_ecg_keys):
        candidates.extend(download_official_ptbxl_metadata_candidates(required_ecg_keys))
        candidates = sorted(set(candidates))

    audit_rows = []
    for path in candidates:
        if not validate_ptbxl_metadata_file(path):
            continue
        matched, missing = metadata_coverage(path, required_ecg_keys)
        audit_rows.append({
            "metadata_path": path,
            "matched_ecg_ids": int(matched),
            "missing_ecg_ids": int(missing),
            "coverage_percent": 100.0 * matched / max(1, len(required_ecg_keys)),
            "path_preference_score": score(path),
        })

    if not audit_rows:
        return None
    audit_df = pd.DataFrame(audit_rows).sort_values(
        ["matched_ecg_ids", "path_preference_score"], ascending=[False, False]
    ).reset_index(drop=True)
    audit_df.to_csv(
        os.path.join(CFG.OUT_DIR, "ptbxl_metadata_version_coverage_audit.csv"), index=False
    )
    selected = audit_df.loc[0, "metadata_path"]
    print("PTB-XL metadata release comparison:")
    print(audit_df.to_string(index=False))
    print("Selected metadata file:", selected)
    return selected


def attach_patient_ids_from_original_metadata(dataframes):
    """Recover patient_id through ecg_id from the official PTB-XL metadata."""
    required_ecg_keys = set()
    for df in dataframes.values():
        required_ecg_keys.update(df["ecg_id"].map(canonical_ecg_key).tolist())
    metadata_path = find_original_ptbxl_metadata_csv(required_ecg_keys)
    if metadata_path is None:
        raise RuntimeError(
            "The reformatted train_meta.csv, valid_meta.csv, and test_meta.csv files do not "
            "contain patient identifiers. ptbxl_database.csv was not found locally, and the "
            "official PhysioNet download failed. In Kaggle, turn Internet ON in Notebook "
            "Settings and rerun, or add the original PTB-XL dataset as a second input. The "
            "added dataset must contain ptbxl_database.csv. Patient-wise leakage cannot be "
            "verified from ecg_id alone, so the code will not use ecg_id as a substitute."
        )

    print("Recovering patient identifiers from:", metadata_path)
    source = pd.read_csv(metadata_path, low_memory=False)
    required = {"ecg_id", "patient_id"}
    if not required.issubset(source.columns):
        raise RuntimeError(
            f"{metadata_path} does not contain the required columns {sorted(required)}. "
            f"Available columns begin with: {source.columns.tolist()[:20]}"
        )

    source = source[["ecg_id", "patient_id"]].copy()
    source = source.dropna(subset=["ecg_id", "patient_id"])
    source["_ecg_key"] = source["ecg_id"].map(canonical_ecg_key)
    source["patient_id"] = source["patient_id"].map(canonical_ecg_key)

    conflicts = source.groupby("_ecg_key")["patient_id"].nunique()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        raise RuntimeError(
            "Conflicting patient identifiers were found for the same ecg_id in "
            f"{metadata_path}: {conflicts.index.tolist()[:10]}"
        )
    source = source.drop_duplicates("_ecg_key", keep="first")
    patient_lookup = source.set_index("_ecg_key")["patient_id"]

    recovery_rows = []
    for split_name, df in dataframes.items():
        df["_ecg_key"] = df["ecg_id"].map(canonical_ecg_key)
        df["patient_id_recovered"] = df["_ecg_key"].map(patient_lookup)
        missing = df["patient_id_recovered"].isna()
        recovery_rows.append({
            "split": split_name,
            "records": int(len(df)),
            "patient_ids_recovered": int((~missing).sum()),
            "patient_ids_missing": int(missing.sum()),
            "coverage_percent": float(100.0 * (~missing).mean()) if len(df) else 100.0,
            "metadata_source": metadata_path,
        })
        if missing.any():
            examples = df.loc[missing, "ecg_id"].astype(str).head(10).tolist()
            raise RuntimeError(
                f"Patient-ID recovery was incomplete for split '{split_name}': "
                f"{int(missing.sum())}/{len(df)} ECGs were unmatched. Example ecg_id values: "
                f"{examples}. Verify that the reformatted signals and ptbxl_database.csv are "
                "from the same PTB-XL release."
            )

    pd.DataFrame(recovery_rows).to_csv(
        os.path.join(CFG.OUT_DIR, "ptbxl_patient_id_recovery_audit.csv"), index=False
    )
    return metadata_path


PATIENT_COLS = {
    "train": resolve_patient_column(train_meta),
    "valid": resolve_patient_column(valid_meta),
    "test": resolve_patient_column(test_meta),
}
if any(v is None for v in PATIENT_COLS.values()):
    attach_patient_ids_from_original_metadata({
        "train": train_meta, "valid": valid_meta, "test": test_meta
    })
    PATIENT_COLS = {
        "train": "patient_id_recovered",
        "valid": "patient_id_recovered",
        "test": "patient_id_recovered",
    }

# Normalize to one canonical column used by bootstrapping and manifests.
for df, key in [(train_meta, "train"), (valid_meta, "valid"), (test_meta, "test")]:
    df["patient_id_audit"] = df[PATIENT_COLS[key]].map(canonical_ecg_key)

patient_sets = {
    "train": set(train_meta["patient_id_audit"]),
    "valid": set(valid_meta["patient_id_audit"]),
    "test": set(test_meta["patient_id_audit"]),
}
overlap_rows = []
for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
    overlap = sorted(patient_sets[a] & patient_sets[b])
    overlap_rows.append({
        "split_a": a,
        "split_b": b,
        "overlapping_patients": len(overlap),
        "example_patient_ids": ",".join(overlap[:10]),
    })
patient_overlap_df = pd.DataFrame(overlap_rows)
patient_overlap_df["audit_stage"] = "supplied_partitions"
patient_overlap_df.to_csv(
    os.path.join(CFG.OUT_DIR, "ptbxl_patient_overlap_audit_supplied.csv"), index=False
)
if patient_overlap_df["overlapping_patients"].sum() != 0:
    if not CFG.AUTO_REBUILD_LEAKING_SPLITS:
        raise RuntimeError(
            "Patient leakage detected across the supplied PTB-XL train/validation/test files. "
            "Set CFG.AUTO_REBUILD_LEAKING_SPLITS=True or rebuild them at patient level.\n" +
            patient_overlap_df.to_string(index=False)
        )

    print("Patient overlap was detected; rebuilding patient-wise 80/10/10 partitions.")
    all_meta_for_split = pd.concat([train_meta, valid_meta, test_meta], ignore_index=True)
    if all_meta_for_split["ecg_id"].duplicated().any():
        duplicated = all_meta_for_split.loc[
            all_meta_for_split["ecg_id"].duplicated(keep=False), "ecg_id"
        ].astype(str).unique().tolist()
        raise RuntimeError(
            "Duplicate ecg_id values occur across the supplied partitions, so the signals "
            f"cannot be reassigned safely. Examples: {duplicated[:10]}"
        )

    combined_sig_map = {}
    combined_sig_map.update(train_sig_map)
    combined_sig_map.update(valid_sig_map)
    combined_sig_map.update(test_sig_map)

    patient_table = (
        all_meta_for_split.groupby("patient_id_audit")[LABEL_COLS]
        .max().reset_index()
    )
    patient_ids_for_split = patient_table["patient_id_audit"].astype(str).to_numpy()
    patient_labels_for_split = patient_table[LABEL_COLS].to_numpy(dtype=int)
    holdout_fraction = CFG.PATIENTWISE_VALID_FRACTION + CFG.PATIENTWISE_TEST_FRACTION
    if not (0 < holdout_fraction < 1):
        raise ValueError("Validation plus test fractions must be between 0 and 1.")

    outer_split = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=holdout_fraction, random_state=RANDOM_STATE
    )
    dev_idx, holdout_idx = next(
        outer_split.split(patient_ids_for_split.reshape(-1, 1), patient_labels_for_split)
    )
    dev_patients = set(patient_ids_for_split[dev_idx])
    holdout_patients = patient_ids_for_split[holdout_idx]
    holdout_labels = patient_labels_for_split[holdout_idx]

    relative_test_fraction = CFG.PATIENTWISE_TEST_FRACTION / holdout_fraction
    holdout_split = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=relative_test_fraction, random_state=RANDOM_STATE + 1
    )
    valid_idx, test_idx = next(
        holdout_split.split(holdout_patients.reshape(-1, 1), holdout_labels)
    )
    valid_patients = set(holdout_patients[valid_idx])
    test_patients = set(holdout_patients[test_idx])

    train_meta = all_meta_for_split[
        all_meta_for_split["patient_id_audit"].isin(dev_patients)
    ].copy().reset_index(drop=True)
    valid_meta = all_meta_for_split[
        all_meta_for_split["patient_id_audit"].isin(valid_patients)
    ].copy().reset_index(drop=True)
    test_meta = all_meta_for_split[
        all_meta_for_split["patient_id_audit"].isin(test_patients)
    ].copy().reset_index(drop=True)
    train_meta["split"] = "train"
    valid_meta["split"] = "valid"
    test_meta["split"] = "test"

    train_sig_map = {str(i): combined_sig_map[str(i)] for i in train_meta["ecg_id"]}
    valid_sig_map = {str(i): combined_sig_map[str(i)] for i in valid_meta["ecg_id"]}
    test_sig_map = {str(i): combined_sig_map[str(i)] for i in test_meta["ecg_id"]}

    rebuilt_sets = {
        "train": set(train_meta["patient_id_audit"]),
        "valid": set(valid_meta["patient_id_audit"]),
        "test": set(test_meta["patient_id_audit"]),
    }
    rebuilt_rows = []
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        overlap = sorted(rebuilt_sets[a] & rebuilt_sets[b])
        rebuilt_rows.append({
            "split_a": a, "split_b": b,
            "overlapping_patients": len(overlap),
            "example_patient_ids": ",".join(overlap[:10]),
            "audit_stage": "rebuilt_partitions",
        })
    rebuilt_overlap_df = pd.DataFrame(rebuilt_rows)
    rebuilt_overlap_df.to_csv(
        os.path.join(CFG.OUT_DIR, "ptbxl_patient_overlap_audit_rebuilt.csv"), index=False
    )
    if rebuilt_overlap_df["overlapping_patients"].sum() != 0:
        raise AssertionError("Automatic patient-wise reconstruction failed its overlap audit.")

    split_summary_rows = []
    for name, frame in [("train", train_meta), ("valid", valid_meta), ("test", test_meta)]:
        split_summary_rows.append({
            "split": name,
            "patients": int(frame["patient_id_audit"].nunique()),
            "ecg_records": int(len(frame)),
            **{f"positive_{c}": int(frame[c].sum()) for c in LABEL_COLS},
        })
    pd.DataFrame(split_summary_rows).to_csv(
        os.path.join(CFG.OUT_DIR, "ptbxl_rebuilt_patientwise_split_summary.csv"), index=False
    )
    print(pd.DataFrame(split_summary_rows).to_string(index=False))
else:
    patient_overlap_df.rename(columns={"audit_stage": "audit_stage"}).to_csv(
        os.path.join(CFG.OUT_DIR, "ptbxl_patient_overlap_audit.csv"), index=False
    )

pd.concat([
    train_meta[["ecg_id", "patient_id_audit", "split"]],
    valid_meta[["ecg_id", "patient_id_audit", "split"]],
    test_meta[["ecg_id", "patient_id_audit", "split"]],
], ignore_index=True).to_csv(
    os.path.join(CFG.OUT_DIR, "ptbxl_patient_split_manifest.csv"), index=False
)

print("\nPTB-XL sizes:")
print("Train:", len(train_meta))
print("Valid:", len(valid_meta))
print("Test :", len(test_meta))


# ============================================================
# 4. WFDB EXTERNAL DATASET LOADING
# ============================================================

STANDARD_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"]


def find_wfdb_records(root, required_path_tokens=None):
    if root is None or not os.path.exists(root):
        return []

    hea_files = sorted(glob.glob(os.path.join(root, "**", "*.hea"), recursive=True))
    required_path_tokens = [str(t).lower() for t in (required_path_tokens or [])]
    if required_path_tokens:
        hea_files = [
            p for p in hea_files
            if any(token in p.lower().replace("\\", "/") for token in required_path_tokens)
        ]
    records = []

    for hea in hea_files:
        base = hea[:-4]
        mat = base + ".mat"
        dat = base + ".dat"
        if os.path.exists(mat) or os.path.exists(dat):
            records.append(base)

    return records


def parse_header_for_dx(record_base):
    hea_path = record_base + ".hea"

    with open(hea_path, "r", errors="ignore") as f:
        lines = [line.rstrip("\n") for line in f.readlines()]

    nonempty = [line.strip() for line in lines if line.strip()]
    if len(nonempty) == 0:
        raise RuntimeError(f"Empty header: {hea_path}")

    first = nonempty[0].split()
    record_name = os.path.basename(record_base)

    try:
        n_leads = int(first[1])
    except Exception:
        n_leads = 12

    try:
        fs = int(float(first[2]))
    except Exception:
        fs = CFG.TARGET_FS

    try:
        n_samples = int(first[3])
    except Exception:
        n_samples = 0

    sig_names = []
    for line in nonempty[1:1+n_leads]:
        parts = line.split()
        if len(parts) > 0:
            sig_names.append(parts[-1])

    dx_codes = []
    age = np.nan
    sex = "Unknown"
    comment_text = " ".join([line for line in nonempty if line.startswith("#")])

    for line in nonempty:
        low = line.lower().strip()

        if re.match(r"^#\s*dx\s*:", low):
            raw = line.split(":", 1)[1].strip()
            dx_codes = split_dx_field(raw)

        elif re.match(r"^#\s*age\s*:", low):
            try:
                age = float(line.split(":", 1)[1].strip())
            except Exception:
                age = np.nan

        elif re.match(r"^#\s*sex\s*:", low):
            sex = line.split(":", 1)[1].strip()

    labels, mapped_classes = map_dx_tokens_to_labels(dx_codes)

    # Disabled by default: free-text keyword labels are not equivalent to
    # clinically harmonized diagnostic codes.
    if CFG.ALLOW_TEXT_LABEL_FALLBACK and len(mapped_classes) == 0 and len(dx_codes) == 0:
        labels, mapped_classes = fallback_text_to_superclasses(comment_text)

    return {
        "ecg_id": record_name,
        "record_base": record_base,
        "hea_path": hea_path,
        "fs": fs,
        "n_leads": n_leads,
        "n_samples": n_samples,
        "sig_names": ",".join(sig_names),
        "dx_raw": ",".join(dx_codes),
        "mapped_classes": ",".join(mapped_classes),
        "age": age,
        "sex": sex,
        "has_formal_dx": int(len(dx_codes) > 0),
        **labels
    }


mapping_audit_rows = []


def build_external_meta(dataset_name, root, required_path_tokens=None, source_note=""):
    records = find_wfdb_records(root, required_path_tokens=required_path_tokens)
    print("\n" + "="*90)
    print(f"External dataset: {dataset_name}")
    print("Root:", root)
    print("WFDB records found:", len(records))
    print("="*90)

    if len(records) == 0:
        print(f"No WFDB records found for {dataset_name}. Skipping.")
        return None

    rows = []
    all_dx_counter = Counter()
    mapped_dx_counter = Counter()
    unmapped_counter = Counter()
    lead_counter = Counter()

    for rb in tqdm(records, desc=f"Parsing {dataset_name} headers"):
        try:
            row = parse_header_for_dx(rb)
        except Exception as e:
            continue

        lead_counter[int(row["n_leads"])] += 1

        dx_codes = split_dx_field(row["dx_raw"])
        for code in dx_codes:
            code = normalize_dx_token(code)
            all_dx_counter[code] += 1
            if code in SNOMED_TO_SUPERCLASS:
                mapped_dx_counter[code] += 1
            else:
                unmapped_counter[code] += 1

        if int(row["n_leads"]) >= 12:
            rows.append(row)

    if len(rows) == 0:
        print(f"No 12-lead records found for {dataset_name}. Skipping.")
        return None

    df = pd.DataFrame(rows)

    for c in LABEL_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    before = len(df)
    df = df[df[LABEL_COLS].sum(axis=1) > 0].reset_index(drop=True)
    after = len(df)

    print("Lead counts:", dict(lead_counter))
    print(f"Before superclass mapping filter: {before}")
    print(f"After superclass mapping filter : {after}")

    if after == 0:
        print(f"WARNING: {dataset_name} has no records mapped to NORM/MI/STTC/CD/HYP.")
        print("This dataset cannot be used for supervised external metrics under the current five-superclass label space.")
        return None

    print("\nMapped external class counts:")
    print(df[LABEL_COLS].sum())

    if len(unmapped_counter) > 0:
        print("\nTop unmapped diagnosis codes:")
        for code, count in unmapped_counter.most_common(15):
            print(code, count)

    for code, count in all_dx_counter.items():
        mapped_class = SNOMED_TO_SUPERCLASS.get(code, "")
        mapping_audit_rows.append({
            "source_dataset": dataset_name,
            "original_code": code,
            "occurrences": int(count),
            "mapped_superclass": mapped_class,
            "superclass_description": SUPERCLASS_DESCRIPTIONS.get(mapped_class, ""),
            "mapping_rule": MAPPING_RULES.get(
                mapped_class,
                "Excluded: no defensible mapping to the five PTB-XL diagnostic superclasses."
            ),
            "decision": "included" if mapped_class else "excluded",
        })

    if CFG.EXTERNAL_MAX_RECORDS_PER_DATASET is not None and len(df) > CFG.EXTERNAL_MAX_RECORDS_PER_DATASET:
        if CFG.EXTERNAL_RANDOM_SAMPLE:
            df = df.sample(
                n=CFG.EXTERNAL_MAX_RECORDS_PER_DATASET,
                random_state=RANDOM_STATE
            ).reset_index(drop=True)
        else:
            df = df.iloc[:CFG.EXTERNAL_MAX_RECORDS_PER_DATASET].reset_index(drop=True)

    df["source_dataset"] = dataset_name
    df["source_note"] = source_note
    df["records_discovered_in_source"] = int(len(records))
    df["records_with_12_leads_before_mapping"] = int(before)
    df["records_retained_after_mapping"] = int(len(df))
    df["split"] = "external"
    return df


external_meta_dict = {}

if CFG.RUN_EXTERNAL_DATASETS:
    for ds_name, root in EXTERNAL_PATHS.items():
        if root is None:
            print(f"\nDataset path not found for {ds_name}. Skipping.")
            continue

        spec = CFG.EXTERNAL_DATASETS[ds_name]
        meta = build_external_meta(
            ds_name,
            root,
            required_path_tokens=spec.get("required_path_tokens", []),
            source_note=spec.get("source_note", ""),
        )
        if meta is not None and len(meta) > 0:
            external_meta_dict[ds_name] = meta

print("\nLoaded external datasets:")
for k, df in external_meta_dict.items():
    print(k, df.shape, df[LABEL_COLS].sum().to_dict())

mapping_audit_df = pd.DataFrame(mapping_audit_rows)
if len(mapping_audit_df):
    mapping_audit_df = mapping_audit_df.sort_values(
        ["source_dataset", "decision", "occurrences"],
        ascending=[True, True, False]
    )
    mapping_audit_df.to_csv(
        os.path.join(CFG.OUT_DIR, "external_label_mapping_audit.csv"), index=False
    )

pd.DataFrame([
    {"superclass": c, "correct_description": SUPERCLASS_DESCRIPTIONS[c], "explicit_rule": MAPPING_RULES[c]}
    for c in LABEL_COLS
]).to_csv(os.path.join(CFG.OUT_DIR, "external_label_mapping_rules.csv"), index=False)

provenance_rows = []
for ds_name, df in external_meta_dict.items():
    fs_counts = df["fs"].value_counts().sort_index().to_dict()
    durations = df["n_samples"] / df["fs"].replace(0, np.nan)
    provenance_rows.append({
        "cohort": ds_name,
        "source_note": df["source_note"].iloc[0],
        "records_discovered": int(df["records_discovered_in_source"].iloc[0]),
        "records_with_12_leads_before_mapping": int(df["records_with_12_leads_before_mapping"].iloc[0]),
        "records_retained": int(len(df)),
        "sampling_frequency_counts": json.dumps({str(k): int(v) for k, v in fs_counts.items()}),
        "duration_seconds_min": float(np.nanmin(durations)),
        "duration_seconds_median": float(np.nanmedian(durations)),
        "duration_seconds_max": float(np.nanmax(durations)),
        **{f"positive_{c}": int(df[c].sum()) for c in LABEL_COLS},
    })
pd.DataFrame(provenance_rows).to_csv(
    os.path.join(CFG.OUT_DIR, "external_cohort_provenance_and_sample_sizes.csv"), index=False
)


# ============================================================
# 5. SIGNAL PREPROCESSING
# ============================================================

def resample_signal(sig, fs, target_fs):
    return polyphase_resample_array(sig, fs, target_fs)


def reorder_or_select_12_leads(signals, fields):
    if signals.shape[1] < 12:
        raise RuntimeError(f"Expected at least 12 leads, got {signals.shape[1]}")

    sig_names = fields.get("sig_name", None)
    if sig_names is None:
        return signals[:, :12].astype(np.float32)

    canonical = {lead.lower(): lead for lead in STANDARD_12_LEADS}
    name_to_idx = {str(name).strip().lower(): i for i, name in enumerate(sig_names)}

    if all(lead.lower() in name_to_idx for lead in STANDARD_12_LEADS):
        idx = [name_to_idx[lead.lower()] for lead in STANDARD_12_LEADS]
        return signals[:, idx].astype(np.float32)

    missing = [lead for lead in STANDARD_12_LEADS if lead.lower() not in name_to_idx]
    raise RuntimeError(f"Missing required standard leads {missing}; record excluded instead of positional guessing.")


def load_wfdb_signal(record_base, target_fs=500):
    signals, fields = wfdb.rdsamp(record_base)
    signals = signals.astype(np.float32)
    signals = reorder_or_select_12_leads(signals, fields)

    fs = int(fields.get("fs", target_fs))
    signals = resample_signal(signals, fs, target_fs)

    signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
    return signals.astype(np.float32)


def waveform_fingerprint(sig, fs=CFG.TARGET_FS, seconds=CFG.FINGERPRINT_SECONDS):
    """Scale-tolerant fingerprint of the first fixed-duration 12-lead segment."""
    n = min(len(sig), int(round(fs * seconds)))
    x = np.asarray(sig[:n], dtype=np.float32)
    if n < int(round(fs * seconds)):
        x = np.pad(x, ((0, int(round(fs * seconds)) - n), (0, 0)))
    med = np.median(x, axis=0, keepdims=True)
    mad = np.median(np.abs(x - med), axis=0, keepdims=True)
    x = (x - med) / np.maximum(mad, 1e-6)
    x = np.clip(np.round(x * 100.0), -32768, 32767).astype(np.int16)
    return hashlib.sha256(x.tobytes()).hexdigest()


def run_signal_overlap_audit():
    if not CFG.RUN_SIGNAL_FINGERPRINT_AUDIT:
        return

    seen = {}
    audit = []
    internal_maps = {
        "PTB-XL train": train_sig_map,
        "PTB-XL valid": valid_sig_map,
        "PTB-XL test": test_sig_map,
    }
    for source, sig_map in internal_maps.items():
        for ecg_id, sig in tqdm(sig_map.items(), desc=f"Fingerprinting {source}"):
            fp = waveform_fingerprint(sig)
            seen.setdefault(fp, (source, str(ecg_id)))

    for ds_name in list(external_meta_dict.keys()):
        df = external_meta_dict[ds_name].copy()
        keep = np.ones(len(df), dtype=bool)
        fingerprints = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Fingerprinting {ds_name}"):
            try:
                sig = load_wfdb_signal(row["record_base"], target_fs=CFG.TARGET_FS)
                fp = waveform_fingerprint(sig)
            except Exception as exc:
                fp = ""
                keep[i] = False
                audit.append({
                    "dataset": ds_name, "ecg_id": row["ecg_id"],
                    "fingerprint": "", "status": "excluded_load_error",
                    "matched_dataset": "", "matched_ecg_id": "", "detail": str(exc)[:300]
                })
                fingerprints.append(fp)
                continue

            fingerprints.append(fp)
            if fp in seen:
                matched_source, matched_id = seen[fp]
                keep[i] = False
                audit.append({
                    "dataset": ds_name, "ecg_id": row["ecg_id"],
                    "fingerprint": fp, "status": "excluded_duplicate",
                    "matched_dataset": matched_source, "matched_ecg_id": matched_id, "detail": ""
                })
            else:
                seen[fp] = (ds_name, str(row["ecg_id"]))
                audit.append({
                    "dataset": ds_name, "ecg_id": row["ecg_id"],
                    "fingerprint": fp, "status": "retained_unique",
                    "matched_dataset": "", "matched_ecg_id": "", "detail": ""
                })

        df["waveform_fingerprint"] = fingerprints
        external_meta_dict[ds_name] = df.loc[keep].reset_index(drop=True)

    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(os.path.join(CFG.OUT_DIR, "external_signal_overlap_audit.csv"), index=False)
    if len(audit_df):
        audit_df.groupby(["dataset", "status"]).size().rename("records").reset_index().to_csv(
            os.path.join(CFG.OUT_DIR, "external_signal_overlap_summary.csv"), index=False
        )


run_signal_overlap_audit()


def fit_scaler(sig_map, ids):
    # Streaming fit avoids materializing every ECG sample in RAM.
    sc = StandardScaler()
    for i in ids:
        i = str(i)
        sig = sig_map[i].reshape(-1, 12)
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        sc.partial_fit(sig)
    return sc


def apply_scaler(sig, scaler):
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    shp = sig.shape
    flat = sig.reshape(-1, 12)
    flat = scaler.transform(flat)
    return flat.reshape(shp).astype(np.float32)


def augment_batch(x):
    if torch.rand(1).item() < 0.50:
        scale = 0.9 + 0.2 * torch.rand(x.size(0), 1, 1, device=x.device)
        x = x * scale

    if torch.rand(1).item() < 0.35:
        x = x + 0.01 * torch.randn_like(x)

    if torch.rand(1).item() < 0.35:
        shift = int(torch.randint(low=-60, high=61, size=(1,)).item())
        x = torch.roll(x, shifts=shift, dims=1)

    if torch.rand(1).item() < 0.10:
        lead = int(torch.randint(low=0, high=12, size=(1,)).item())
        x[:, :, lead] = 0.0

    return x


# ============================================================
# 6. DATASET CLASSES
# ============================================================

class PTBXLWindowDataset(Dataset):
    def __init__(self, meta_df, sig_map, scaler, window=2000, stride=500):
        self.meta = meta_df.reset_index(drop=True)
        self.sig_map = sig_map
        self.scaler = scaler
        self.window = window
        self.stride = stride
        self.index = []

        for i in range(len(self.meta)):
            ecg_id = str(self.meta.loc[i, "ecg_id"])
            T = self.sig_map[ecg_id].shape[0]

            if T <= window:
                self.index.append((i, 0))
            else:
                for s in range(0, max(1, T - window + 1), stride):
                    self.index.append((i, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, s = self.index[k]
        ecg_id = str(self.meta.loc[i, "ecg_id"])
        y = self.meta.loc[i, LABEL_COLS].values.astype(np.float32)

        # Slice first, normalize second. The former code normalized the full ECG
        # again for every overlapping window (four times per 10-second record).
        sig = self.sig_map[ecg_id]
        T = sig.shape[0]

        if T >= self.window:
            x = apply_scaler(sig[s:s+self.window], self.scaler)
        else:
            sig = apply_scaler(sig, self.scaler)
            pad = np.zeros((self.window - T, 12), dtype=np.float32)
            x = np.vstack([sig, pad])

        return torch.from_numpy(x).float(), torch.from_numpy(y).float(), ecg_id


class WFDBExternalWindowDataset(Dataset):
    def __init__(self, meta_df, scaler, window=2000, stride=2000, target_fs=500):
        self.meta = meta_df.reset_index(drop=True)
        self.scaler = scaler
        self.window = window
        self.stride = stride
        self.target_fs = target_fs
        self.index = []

        for i in range(len(self.meta)):
            fs = int(self.meta.loc[i, "fs"])
            n_samples = int(self.meta.loc[i, "n_samples"])

            if n_samples <= 0:
                approx_T = window
            else:
                approx_T = int(round(n_samples * target_fs / max(1, fs)))

            if approx_T <= window:
                self.index.append((i, 0))
            else:
                for s in range(0, max(1, approx_T - window + 1), stride):
                    self.index.append((i, s))

    def __len__(self):
        return len(self.index)

    @lru_cache(maxsize=CFG.EXTERNAL_RECORD_CACHE_SIZE)
    def _load_record(self, row_index):
        """Read/resample a record once per worker instead of once per window."""
        record_base = self.meta.loc[row_index, "record_base"]
        return load_wfdb_signal(record_base, target_fs=self.target_fs)

    def __getitem__(self, k):
        i, s = self.index[k]
        row = self.meta.loc[i]
        ecg_id = str(row["ecg_id"])
        y = row[LABEL_COLS].values.astype(np.float32)

        sig = self._load_record(i)
        T = sig.shape[0]

        if T >= self.window:
            if s + self.window <= T:
                x = sig[s:s+self.window]
            else:
                x = sig[-self.window:]
            x = apply_scaler(x, self.scaler)
        else:
            sig = apply_scaler(sig, self.scaler)
            pad = np.zeros((self.window - T, 12), dtype=np.float32)
            x = np.vstack([sig, pad])

        return torch.from_numpy(x).float(), torch.from_numpy(y).float(), ecg_id


def aggregate_ecg(ecg_ids, probs, y_window):
    dfp = pd.DataFrame({"ecg_id": ecg_ids})

    for j, c in enumerate(LABEL_COLS):
        dfp[c] = probs[:, j]

    dfy = pd.DataFrame(y_window, columns=[f"y_{c}" for c in LABEL_COLS])
    dfy["ecg_id"] = ecg_ids

    p_ecg = dfp.groupby("ecg_id")[LABEL_COLS].mean().values.astype(np.float32)
    y_ecg = dfy.groupby("ecg_id")[[f"y_{c}" for c in LABEL_COLS]].max().values.astype(int)
    ids = list(dfp.groupby("ecg_id").groups.keys())

    return ids, y_ecg, p_ecg


# ============================================================
# 7. MODEL COMPONENTS
# ============================================================

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()

        for k, v in msd.items():
            if k not in self.shadow:
                continue

            if not torch.is_floating_point(v):
                self.shadow[k] = v.detach().clone()
                continue

            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


def asl_with_logits(logits, targets, gamma_pos=0.0, gamma_neg=3.0,
                    clip=0.05, eps=1e-7):
    """Asymmetric loss with float32 log operations, safe under AMP.

    The former eps=1e-8 becomes zero in float16. This can cause NaNs via
    zero * log(0), producing an empty CSV loss and collapsing later epochs.
    """
    logits = logits.float()
    targets = targets.float()
    p = torch.sigmoid(logits)
    p_neg = (p + clip).clamp(eps, 1.0 - eps) if clip else p.clamp(eps, 1.0 - eps)
    with torch.no_grad():
        w_pos = (1.0 - p).pow(gamma_pos) if gamma_pos else 1.0
        w_neg = p_neg.pow(gamma_neg) if gamma_neg else 1.0
    loss = -targets * w_pos * F.logsigmoid(logits) - (1.0 - targets) * w_neg * torch.log1p(-p_neg)
    if not torch.isfinite(loss).all():
        raise FloatingPointError("Non-finite asymmetric loss; inspect ECG scaling and logits.")
    return loss.mean()


class SE1D(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        hidden = max(8, ch // r)
        self.fc1 = nn.Conv1d(ch, hidden, 1)
        self.fc2 = nn.Conv1d(hidden, ch, 1)

    def forward(self, x):
        s = x.mean(dim=-1, keepdim=True)
        s = F.gelu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, k=9, drop=0.1, use_se=True):
        super().__init__()
        pad = k // 2

        self.c1 = nn.Conv1d(cin, cout, k, padding=pad, bias=False)
        self.b1 = nn.BatchNorm1d(cout)

        self.c2 = nn.Conv1d(cout, cout, k, padding=pad, bias=False)
        self.b2 = nn.BatchNorm1d(cout)

        self.se = SE1D(cout, r=8) if use_se else nn.Identity()
        self.drop = nn.Dropout(drop)

        if cin != cout:
            self.proj = nn.Sequential(
                nn.Conv1d(cin, cout, 1, bias=False),
                nn.BatchNorm1d(cout)
            )
        else:
            self.proj = None

    def forward(self, x):
        r = x if self.proj is None else self.proj(x)

        x = F.gelu(self.b1(self.c1(x)))
        x = self.drop(x)
        x = self.b2(self.c2(x))
        x = self.se(x)

        return F.gelu(x + r)


class ProposedECG_LabelTokenTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, depth=3, drop=0.10, use_label_tokens=True, use_se=True):
        super().__init__()

        self.patch = 50
        self.use_label_tokens = use_label_tokens

        self.stem = nn.Sequential(
            nn.Conv1d(12, d_model, 15, padding=7, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(drop)
        )

        self.b1 = ResBlock1D(d_model, d_model, k=9, drop=drop, use_se=use_se)
        self.b2 = ResBlock1D(d_model, d_model, k=9, drop=drop, use_se=use_se)
        self.b3 = ResBlock1D(d_model, d_model, k=9, drop=drop, use_se=use_se)

        self.patch_proj = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=self.patch,
            stride=self.patch,
            bias=False
        )

        if use_label_tokens:
            self.class_tokens = nn.Parameter(torch.randn(C, d_model) * 0.02)
            self.head = nn.Linear(d_model, 1)
        else:
            self.head = nn.Linear(d_model, C)

        self.pos_embed = nn.Parameter(torch.randn(1, 400, d_model) * 0.01)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=drop,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)

    def forward_features(self, x):
        h = x.transpose(1, 2)
        h = self.stem(h)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)

        if h.size(-1) % self.patch != 0:
            pad = self.patch - (h.size(-1) % self.patch)
            h = F.pad(h, (0, pad))

        pt = self.patch_proj(h).transpose(1, 2)
        B, N, D = pt.shape

        if N > self.pos_embed.size(1):
            raise RuntimeError(f"Number of patches {N} exceeds positional embedding length.")

        pt = pt + self.pos_embed[:, :N, :]

        if self.use_label_tokens:
            ct = self.class_tokens.unsqueeze(0).expand(B, -1, -1)
            tokens = torch.cat([ct, pt], dim=1)
            z = self.encoder(tokens)
            z = self.norm(z)
            label_features = z[:, :C, :]
            sample_features = label_features.mean(dim=1)
            return label_features, sample_features
        else:
            z = self.encoder(pt)
            z = self.norm(z)
            sample_features = z.mean(dim=1)
            return None, sample_features

    def forward(self, x):
        label_features, sample_features = self.forward_features(x)

        if self.use_label_tokens:
            logits = self.head(label_features).squeeze(-1)
        else:
            logits = self.head(sample_features)

        return logits


class BaselineCNN1D(nn.Module):
    def __init__(self, d_model=256, drop=0.10, use_se=False):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(12, d_model, 15, padding=7, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(drop)
        )

        self.b1 = ResBlock1D(d_model, d_model, k=9, drop=drop, use_se=use_se)
        self.b2 = ResBlock1D(d_model, d_model, k=9, drop=drop, use_se=use_se)
        self.b3 = ResBlock1D(d_model, d_model, k=9, drop=drop, use_se=use_se)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(d_model, C)

    def forward_features(self, x):
        h = x.transpose(1, 2)
        h = self.stem(h)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        z = self.pool(h).squeeze(-1)
        return None, z

    def forward(self, x):
        _, z = self.forward_features(x)
        return self.head(z)


class BaselineTransformerOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8, depth=3, drop=0.10):
        super().__init__()

        self.patch = 50
        self.patch_proj = nn.Conv1d(12, d_model, kernel_size=self.patch, stride=self.patch, bias=False)
        self.pos_embed = nn.Parameter(torch.randn(1, 400, d_model) * 0.01)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=drop,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, C)

    def forward_features(self, x):
        h = x.transpose(1, 2)

        if h.size(-1) % self.patch != 0:
            pad = self.patch - (h.size(-1) % self.patch)
            h = F.pad(h, (0, pad))

        pt = self.patch_proj(h).transpose(1, 2)
        B, N, D = pt.shape
        pt = pt + self.pos_embed[:, :N, :]

        z = self.encoder(pt)
        z = self.norm(z)
        sample_features = z.mean(dim=1)

        return None, sample_features

    def forward(self, x):
        _, z = self.forward_features(x)
        return self.head(z)


# ============================================================
# 8. METRICS
# ============================================================

def tune_thresholds(y_true, p):
    grid = np.linspace(CFG.THR_MIN, CFG.THR_MAX, CFG.THR_GRID)
    ths = np.full((C,), 0.5, dtype=np.float32)

    for j in range(C):
        if y_true[:, j].sum() == 0:
            continue

        best_f1 = -1
        best_t = 0.5

        for t in grid:
            pred = (p[:, j] >= t).astype(int)
            f = f1_score(y_true[:, j], pred, zero_division=0)

            if (f > best_f1 + 1e-12) or (
                abs(f - best_f1) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)
            ):
                best_f1 = f
                best_t = t

        ths[j] = best_t

    return ths


def metrics_all(y_true, p, ths=None):
    if ths is None:
        ths = np.full((C,), 0.5, dtype=np.float32)

    y_hat = (p >= ths.reshape(1, -1)).astype(int)

    aucs, aps, f1s = [], [], []

    for j in range(C):
        if len(np.unique(y_true[:, j])) < 2:
            aucs.append(np.nan)
            aps.append(np.nan)
        else:
            aucs.append(roc_auc_score(y_true[:, j], p[:, j]))
            aps.append(average_precision_score(y_true[:, j], p[:, j]))

        f1s.append(f1_score(y_true[:, j], y_hat[:, j], zero_division=0))

    return {
        "macro_auc": float(np.nanmean(aucs)),
        "macro_ap": float(np.nanmean(aps)),
        "macro_f1": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_hat, average="micro", zero_division=0)),
        "subset_acc": float(accuracy_score(y_true, y_hat)),
        "hamming_acc": float((y_true == y_hat).mean()),
        "jaccard": float(jaccard_score(y_true, y_hat, average="samples", zero_division=0)),
        "aucs": aucs,
        "aps": aps,
        "f1s": f1s,
        "ths": ths,
        "y_hat": y_hat
    }


def fixed_threshold_comparison(y_true, p, tuned_thresholds):
    """Reviewer-3 safeguard: ranking metrics must not change with thresholds."""
    tuned = metrics_all(y_true, p, tuned_thresholds)
    fixed = metrics_all(y_true, p, np.full(C, 0.5, dtype=np.float32))
    if not np.isclose(tuned["macro_auc"], fixed["macro_auc"], equal_nan=True):
        raise AssertionError("AUROC changed in threshold-only comparison; probabilities were not identical.")
    if not np.isclose(tuned["macro_ap"], fixed["macro_ap"], equal_nan=True):
        raise AssertionError("Average precision changed in threshold-only comparison; probabilities were not identical.")
    rows = []
    for name, m in [("Full model with validation-tuned thresholds", tuned),
                    ("Same full-model probabilities; fixed threshold = 0.5", fixed)]:
        rows.append({
            "Model variant": name,
            "Macro-AUROC": m["macro_auc"],
            "Macro-AP": m["macro_ap"],
            "Macro-F1": m["macro_f1"],
            "Micro-F1": m["micro_f1"],
            "Hamming Acc.": m["hamming_acc"],
            "Jaccard": m["jaccard"],
            "Subset Acc.": m["subset_acc"],
        })
    return pd.DataFrame(rows), tuned, fixed


def per_class_metric_table(y_true, p, thresholds, dataset_name):
    y_hat = (p >= np.asarray(thresholds).reshape(1, -1)).astype(int)
    rows = []
    for j, label in enumerate(LABEL_COLS):
        unique = np.unique(y_true[:, j])
        rows.append({
            "Dataset": dataset_name,
            "Class": label,
            "Positive support": int(y_true[:, j].sum()),
            "Negative support": int((1 - y_true[:, j]).sum()),
            "AUROC": roc_auc_score(y_true[:, j], p[:, j]) if len(unique) == 2 else np.nan,
            "AP": average_precision_score(y_true[:, j], p[:, j]) if len(unique) == 2 else np.nan,
            "F1": f1_score(y_true[:, j], y_hat[:, j], zero_division=0),
            "Threshold": float(thresholds[j]),
            "Metric status": "defined" if len(unique) == 2 else "AUROC/AP undefined: one observed class",
        })
    return pd.DataFrame(rows)


def paired_patient_bootstrap(
    ids, y_true, p_full, th_full, p_variant, th_variant,
    patient_lookup, variant_name, repeats=10000, seed=42
):
    """Paired cluster bootstrap; patients, not windows/ECGs, are sampled."""
    ids = [str(x) for x in ids]
    patients = np.asarray([str(patient_lookup[str(x)]) for x in ids])
    unique_patients = np.unique(patients)
    group_indices = {g: np.where(patients == g)[0] for g in unique_patients}
    rng = np.random.default_rng(seed)
    metric_keys = ["macro_auc", "macro_ap", "macro_f1", "micro_f1", "hamming_acc"]
    point_full = metrics_all(y_true, p_full, th_full)
    point_variant = metrics_all(y_true, p_variant, th_variant)
    deltas = {k: [] for k in metric_keys}

    for _ in tqdm(range(repeats), desc=f"Paired bootstrap: {variant_name}"):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        idx = np.concatenate([group_indices[g] for g in sampled])
        mf = metrics_all(y_true[idx], p_full[idx], th_full)
        mv = metrics_all(y_true[idx], p_variant[idx], th_variant)
        for k in metric_keys:
            d = mf[k] - mv[k]
            if np.isfinite(d):
                deltas[k].append(float(d))

    rows = []
    for k in metric_keys:
        arr = np.asarray(deltas[k], dtype=float)
        if len(arr) == 0:
            lo = hi = p_two = np.nan
        else:
            lo, hi = np.percentile(arr, [2.5, 97.5])
            p_two = min(1.0, 2.0 * min(np.mean(arr <= 0), np.mean(arr >= 0)))
        rows.append({
            "Comparison": f"Full proposed model - {variant_name}",
            "Metric": k,
            "Full model": point_full[k],
            "Variant": point_variant[k],
            "Difference": point_full[k] - point_variant[k],
            "95% CI lower": lo,
            "95% CI upper": hi,
            "Two-sided bootstrap p": p_two,
            "Bootstrap unit": "patient",
            "Replicates": int(repeats),
        })
    return pd.DataFrame(rows)


def print_metrics_block(name, m):
    print("\n" + "="*90)
    print(name)
    print("="*90)
    print(f"Macro-AUROC : {m['macro_auc']:.4f}")
    print(f"Macro-AP    : {m['macro_ap']:.4f}")
    print(f"Macro-F1    : {m['macro_f1']:.4f}")
    print(f"Micro-F1    : {m['micro_f1']:.4f}")
    print(f"Hamming Acc.: {m['hamming_acc']:.4f}")
    print(f"Jaccard     : {m['jaccard']:.4f}")
    print(f"Subset Acc. : {m['subset_acc']:.4f}")


def class_distribution_table(meta_df, split_name):
    Y = meta_df[LABEL_COLS].values.astype(int)
    counts = Y.sum(axis=0)
    n = len(meta_df)

    return pd.DataFrame({
        "Dataset/Split": split_name,
        "Class": LABEL_COLS,
        "Positive ECGs": counts,
        "Total ECGs": n,
        "Percentage (%)": np.round((counts / max(1, n)) * 100, 3)
    })


# ============================================================
# 9. PLOTTING
# ============================================================

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=CFG.FIG_DPI, bbox_inches="tight")
    print("Saved figure:", path)
    plt.show()


def plot_dataset_distribution(dist_df):
    plt.figure(figsize=(14, 6))

    classes = LABEL_COLS
    splits = dist_df["Dataset/Split"].unique().tolist()

    x = np.arange(len(classes))
    width = 0.8 / max(1, len(splits))

    for i, sp in enumerate(splits):
        sub = dist_df[dist_df["Dataset/Split"] == sp]
        vals = [sub[sub["Class"] == c]["Positive ECGs"].values[0] for c in classes]
        plt.bar(x + i*width - 0.4 + width/2, vals, width, label=sp)

    plt.xticks(x, classes)
    plt.ylabel("Positive ECGs")
    plt.xlabel("Diagnostic superclass")
    plt.title("Class Distribution Across Internal and External Datasets")
    plt.legend(fontsize=8)
    savefig(os.path.join(CFG.OUT_DIR, "figure_class_distribution_all_datasets_300dpi.png"))


def plot_multilabel_confusions(y_true, y_hat, title_prefix, out_name):
    mcm = multilabel_confusion_matrix(y_true, y_hat)

    fig, axes = plt.subplots(1, C, figsize=(4*C, 3))

    for j in range(C):
        ax = axes[j]
        mat = mcm[j]
        im = ax.imshow(mat)
        ax.set_title(f"{title_prefix} - {LABEL_COLS[j]}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(mat[r, c]), ha="center", va="center")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    savefig(os.path.join(CFG.OUT_DIR, out_name))


def plot_roc_pr_curves(y_true, p, dataset_name, prefix):
    plt.figure(figsize=(7, 5))

    for j, c in enumerate(LABEL_COLS):
        if len(np.unique(y_true[:, j])) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true[:, j], p[:, j])
        auc_val = roc_auc_score(y_true[:, j], p[:, j])
        plt.plot(fpr, tpr, label=f"{c} AUC={auc_val:.3f}")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - {dataset_name}")
    plt.legend()
    savefig(os.path.join(CFG.OUT_DIR, f"{prefix}_roc_curves_300dpi.png"))

    plt.figure(figsize=(7, 5))

    for j, c in enumerate(LABEL_COLS):
        if len(np.unique(y_true[:, j])) < 2:
            continue

        prec, rec, _ = precision_recall_curve(y_true[:, j], p[:, j])
        ap_val = average_precision_score(y_true[:, j], p[:, j])
        plt.plot(rec, prec, label=f"{c} AP={ap_val:.3f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves - {dataset_name}")
    plt.legend()
    savefig(os.path.join(CFG.OUT_DIR, f"{prefix}_pr_curves_300dpi.png"))


def plot_training_curves(hist_df, title, out_name):
    plt.figure(figsize=(8, 4))
    plt.plot(hist_df["epoch"], hist_df["loss"], marker="o", label="Train Loss")
    plt.plot(hist_df["epoch"], hist_df["val_macro_auc"], marker="o", label="Val Macro-AUROC")
    plt.plot(hist_df["epoch"], hist_df["val_micro_f1"], marker="o", label="Val Micro-F1")
    plt.plot(hist_df["epoch"], hist_df["val_hamming"], marker="o", label="Val Hamming Acc.")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.legend()
    savefig(os.path.join(CFG.OUT_DIR, out_name))


# ============================================================
# 10. TRAINING AND INFERENCE
# ============================================================

@torch.no_grad()
def infer_ecg_level(model, loader, ema=None):
    backup_state = None
    if ema is not None:
        backup_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)

    model.eval()
    all_ids, all_p, all_y = [], [], []

    for x, y, ecg_id in tqdm(loader, desc="Inference"):
        x = x.to(DEVICE)

        logits = model(x)
        p = torch.sigmoid(logits).detach().cpu().numpy()

        all_p.append(p)
        all_y.append(y.numpy())

        if isinstance(ecg_id, (list, tuple)):
            all_ids.extend([str(v) for v in ecg_id])
        else:
            all_ids.extend([str(v) for v in ecg_id])

    all_p = np.concatenate(all_p, axis=0)
    all_y = np.concatenate(all_y, axis=0).astype(int)

    ids, y_ecg, p_ecg = aggregate_ecg(all_ids, all_p, all_y)
    if backup_state is not None:
        model.load_state_dict(backup_state, strict=True)
    return ids, y_ecg, p_ecg


def cosine_lr(step, total_steps, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)

    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * t))


def train_model(
    model,
    tr_meta,
    tr_map,
    va_meta,
    va_map,
    scaler,
    model_name="model",
    epochs=10,
    lr=2e-4,
    use_ema=True,
    training_deadline_monotonic=None,
):
    tr_ds = PTBXLWindowDataset(
        tr_meta,
        tr_map,
        scaler,
        window=CFG.T_WINDOW,
        stride=CFG.STRIDE_TRAIN
    )

    va_ds = PTBXLWindowDataset(
        va_meta,
        va_map,
        scaler,
        window=CFG.T_WINDOW,
        stride=CFG.STRIDE_EVAL
    )

    tr_loader = DataLoader(
        tr_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        **dataloader_runtime_kwargs()
    )

    va_loader = DataLoader(
        va_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        **dataloader_runtime_kwargs()
    )

    model = model.to(DEVICE)
    ema = EMA(model, decay=CFG.EMA_DECAY) if use_ema else None

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=CFG.WD)
    amp_enabled = bool(CFG.AMP and torch.cuda.is_available())
    # PyTorch's usual initial FP16 scale (65536) may overflow ECG gradients in
    # the first mini-batch. Start lower and let dynamic scaling adapt.
    scaler_amp = torch.cuda.amp.GradScaler(
        enabled=amp_enabled, init_scale=1024.0, growth_interval=2000
    )
    consecutive_amp_overflows = 0
    total_amp_skipped_steps = 0

    total_steps = epochs * max(1, len(tr_loader))
    warmup_steps = CFG.WARMUP_EPOCHS * max(1, len(tr_loader))
    gstep = 0

    hist = []
    best = -1
    best_state = None
    best_shadow = None
    best_ths = None
    epochs_without_improvement = 0
    epoch_durations = []

    for ep in range(1, epochs + 1):
        # Predict whether another epoch would consume the finalization reserve.
        # At least one epoch is always allowed so a valid checkpoint exists.
        if epoch_durations:
            predicted_next = 1.20 * max(epoch_durations[-2:])
            available_seconds = runtime_seconds_remaining()
            if training_deadline_monotonic is not None:
                available_seconds = min(
                    available_seconds,
                    max(0.0, training_deadline_monotonic - time.monotonic()),
                )
            if available_seconds <= predicted_next:
                print(
                    f"RUNTIME BUDGET: stopping {model_name} before epoch {ep}; "
                    f"elapsed={elapsed_runtime_hours():.2f} h."
                )
                break

        epoch_started = time.monotonic()
        model.train()
        losses = []
        epoch_amp_skipped_steps = 0

        for x, y, _ in tqdm(tr_loader, desc=f"{model_name} epoch {ep}/{epochs}"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            x = augment_batch(x)
            if not torch.isfinite(x).all():
                raise FloatingPointError(f"{model_name}: non-finite ECG input in epoch {ep}")

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(x)
                loss = asl_with_logits(logits, y, gamma_pos=0.0, gamma_neg=3.0, clip=0.05)

            if not torch.isfinite(loss):
                if not amp_enabled:
                    raise FloatingPointError(f"{model_name}: non-finite FP32 loss in epoch {ep}")
                # An FP16 forward pass can also overflow. Retry this same
                # mini-batch in FP32 before declaring a data/model error.
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=False):
                    logits = model(x)
                    loss = asl_with_logits(logits, y, gamma_pos=0.0,
                                           gamma_neg=3.0, clip=0.05)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"{model_name}: non-finite loss even in FP32, epoch {ep}"
                    )
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                if not amp_enabled:
                    raise FloatingPointError(
                        f"{model_name}: non-finite FP32 gradient in epoch {ep}"
                    )
                # GradScaler marks inf/NaN grads during unscale_; step() then
                # skips this optimizer update and update() lowers the scale.
                scaler_amp.step(opt)
                scaler_amp.update()
                consecutive_amp_overflows += 1
                total_amp_skipped_steps += 1
                epoch_amp_skipped_steps += 1
                if consecutive_amp_overflows >= 8:
                    print(f"{model_name}: repeated AMP overflow; continuing in FP32")
                    amp_enabled = False
                    scaler_amp = torch.cuda.amp.GradScaler(enabled=False)
                continue
            scaler_amp.step(opt)
            scaler_amp.update()
            consecutive_amp_overflows = 0

            gstep += 1
            lr_now = cosine_lr(gstep, total_steps, lr, warmup_steps)

            for pg in opt.param_groups:
                pg["lr"] = lr_now

            if ema is not None:
                ema.update(model)

            losses.append(loss.item())

        if not losses or not np.isfinite(losses).all():
            raise FloatingPointError(f"{model_name}: missing or non-finite epoch loss {ep}")
        _, yv, pv = infer_ecg_level(model, va_loader, ema=ema)
        if not np.isfinite(pv).all():
            raise FloatingPointError(f"{model_name}: non-finite validation probabilities {ep}")
        ths = tune_thresholds(yv, pv)
        m = metrics_all(yv, pv, ths)

        score = 0.70 * m["macro_auc"] + 0.30 * m["macro_f1"]
        if not np.isfinite(score):
            raise FloatingPointError(f"{model_name}: non-finite checkpoint score {ep}")

        hist.append({
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "val_macro_auc": m["macro_auc"],
            "val_macro_ap": m["macro_ap"],
            "val_macro_f1": m["macro_f1"],
            "val_micro_f1": m["micro_f1"],
            "val_hamming": m["hamming_acc"],
            "score": score,
            "amp_skipped_steps": epoch_amp_skipped_steps,
            "amp_skipped_steps_total": total_amp_skipped_steps,
            "amp_enabled_after_epoch": bool(amp_enabled),
        })

        print(
            f"{model_name} | Epoch {ep:03d}/{epochs} | "
            f"Loss={np.mean(losses):.4f} | "
            f"Val AUC={m['macro_auc']:.4f} | "
            f"Val AP={m['macro_ap']:.4f} | "
            f"Val Macro-F1={m['macro_f1']:.4f} | "
            f"Val Micro-F1={m['micro_f1']:.4f}"
        )

        if score > best:
            best = score
            epochs_without_improvement = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if ema is not None:
                best_shadow = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}

            best_ths = ths.copy()
        else:
            epochs_without_improvement += 1

        epoch_durations.append(time.monotonic() - epoch_started)
        hist[-1]["epoch_seconds"] = epoch_durations[-1]
        hist[-1]["elapsed_runtime_hours"] = elapsed_runtime_hours()

        if (
            CFG.EARLY_STOPPING
            and ep >= CFG.EARLY_STOPPING_MIN_EPOCHS
            and epochs_without_improvement >= CFG.EARLY_STOPPING_PATIENCE
        ):
            print(
                f"EARLY STOPPING: {model_name} after epoch {ep}; "
                f"best validation score={best:.4f}."
            )
            break

    if best_state is None or best_ths is None:
        raise RuntimeError(f"{model_name}: no valid checkpoint was selected")
    if best_state is not None:
        model.load_state_dict(best_state)

    if ema is not None and best_shadow is not None:
        ema.shadow = {k: v.to(DEVICE) for k, v in best_shadow.items()}

    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(os.path.join(CFG.OUT_DIR, f"{model_name}_training_history.csv"), index=False)

    if model_name.startswith("CV_Fold_") and len(hist_df) < CFG.MIN_COMPLETED_CV_EPOCHS:
        raise RuntimeError(f"{model_name}: only {len(hist_df)} epochs completed; refusing incomplete CV.")
    return model, ema, best_ths, hist_df


def evaluate_ptbxl_test(model, ema, scaler, ths, model_name):
    test_ds = PTBXLWindowDataset(
        test_meta,
        test_sig_map,
        scaler,
        window=CFG.T_WINDOW,
        stride=CFG.STRIDE_EVAL
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        **dataloader_runtime_kwargs()
    )

    ids, y, p = infer_ecg_level(model, test_loader, ema=ema)
    m = metrics_all(y, p, ths)

    print_metrics_block(f"PTB-XL INTERNAL TEST: {model_name}", m)

    pred_df = pd.DataFrame({"ecg_id": ids})
    if "patient_id_audit" in test_meta.columns:
        patient_lookup = test_meta.set_index("ecg_id")["patient_id_audit"].astype(str).to_dict()
        pred_df["patient_id"] = pred_df["ecg_id"].map(patient_lookup)
    for j, c in enumerate(LABEL_COLS):
        pred_df[f"y_{c}"] = y[:, j]
        pred_df[f"p_{c}"] = p[:, j]
        pred_df[f"pred_{c}"] = m["y_hat"][:, j]

    pred_df.to_csv(
        os.path.join(CFG.OUT_DIR, f"{model_name}_ptbxl_predictions.csv"),
        index=False
    )

    per_class_metric_table(y, p, ths, "PTB-XL internal test").to_csv(
        os.path.join(CFG.OUT_DIR, f"{model_name}_ptbxl_per_class_metrics.csv"), index=False
    )

    return ids, y, p, m


def evaluate_external_dataset(model, ema, scaler, ths, dataset_name, meta_df):
    ds = WFDBExternalWindowDataset(
        meta_df,
        scaler,
        window=CFG.T_WINDOW,
        stride=CFG.STRIDE_EVAL,
        target_fs=CFG.TARGET_FS
    )

    loader = DataLoader(
        ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        **dataloader_runtime_kwargs()
    )

    ids, y, p = infer_ecg_level(model, loader, ema=ema)
    m = metrics_all(y, p, ths)

    safe_name = re.sub(r"[^A-Za-z0-9]+", "_", dataset_name).strip("_")

    print_metrics_block(f"EXTERNAL TEST: {dataset_name}", m)

    pred_df = pd.DataFrame({"ecg_id": ids})
    for j, c in enumerate(LABEL_COLS):
        pred_df[f"y_{c}"] = y[:, j]
        pred_df[f"p_{c}"] = p[:, j]
        pred_df[f"pred_{c}"] = m["y_hat"][:, j]

    pred_df.to_csv(
        os.path.join(CFG.OUT_DIR, f"external_{safe_name}_predictions.csv"),
        index=False
    )

    per_class_metric_table(y, p, ths, dataset_name).to_csv(
        os.path.join(CFG.OUT_DIR, f"external_{safe_name}_per_class_metrics.csv"), index=False
    )

    plot_multilabel_confusions(
        y,
        m["y_hat"],
        dataset_name,
        f"figure_external_{safe_name}_confusion_matrices_300dpi.png"
    )

    plot_roc_pr_curves(
        y,
        p,
        dataset_name,
        f"figure_external_{safe_name}"
    )

    return ids, y, p, m


# ============================================================
# 11. DATA DISTRIBUTION TABLE AND FIGURE
# ============================================================

dist_list = [
    class_distribution_table(train_meta, "PTB-XL Train"),
    class_distribution_table(valid_meta, "PTB-XL Valid"),
    class_distribution_table(test_meta, "PTB-XL Test"),
]

for ds_name, meta_df in external_meta_dict.items():
    dist_list.append(class_distribution_table(meta_df, ds_name))

dist_df = pd.concat(dist_list, ignore_index=True)
display(dist_df)
dist_df.to_csv(os.path.join(CFG.OUT_DIR, "dataset_class_distribution_all.csv"), index=False)
plot_dataset_distribution(dist_df)


# ============================================================
# 12. TRAIN PROPOSED MODEL
# ============================================================

signal_scaler = fit_scaler(train_sig_map, train_meta["ecg_id"].values)

proposed_model = ProposedECG_LabelTokenTransformer(
    d_model=CFG.D_MODEL,
    nhead=CFG.NHEAD,
    depth=CFG.TX_DEPTH,
    drop=CFG.DROPOUT,
    use_label_tokens=True,
    use_se=True
)

print("\n" + "="*90)
print("TRAINING PROPOSED CNN-SE LABEL-TOKEN TRANSFORMER MODEL")
print("="*90)

proposed_model, proposed_ema, proposed_ths, proposed_hist = train_model(
    model=proposed_model,
    tr_meta=train_meta,
    tr_map=train_sig_map,
    va_meta=valid_meta,
    va_map=valid_sig_map,
    scaler=signal_scaler,
    model_name="Proposed_CNN_SE_LabelTokenTransformer",
    epochs=CFG.EPOCHS,
    lr=CFG.LR,
    use_ema=True
)
proposed_epochs_trained = int(len(proposed_hist))

plot_training_curves(
    proposed_hist,
    "Proposed Model Training Curves",
    "figure_proposed_training_curves_300dpi.png"
)

print("\nOptimized PTB-XL validation thresholds:")
for c, t in zip(LABEL_COLS, proposed_ths):
    print(f"{c}: {t:.4f}")


# ============================================================
# 13. INTERNAL PTB-XL EVALUATION
# ============================================================

ptbxl_ids, y_ptbxl, p_ptbxl, m_ptbxl = evaluate_ptbxl_test(
    proposed_model,
    proposed_ema,
    signal_scaler,
    proposed_ths,
    "Proposed_CNN_SE_LabelTokenTransformer"
)

# Reviewer comment 3: this is a decision-threshold comparison only. It uses
# the exact same full-model probabilities, so AUROC and AP are asserted equal.
threshold_ablation_df, m_ptbxl_tuned, m_ptbxl_fixed05 = fixed_threshold_comparison(
    y_ptbxl, p_ptbxl, proposed_ths
)
threshold_ablation_df.to_csv(
    os.path.join(CFG.OUT_DIR, "threshold_only_ablation_same_probabilities.csv"), index=False
)
print("\nThreshold-only ablation (ranking metrics must be identical):")
display(threshold_ablation_df)

plot_multilabel_confusions(
    y_ptbxl,
    m_ptbxl["y_hat"],
    "PTB-XL Internal Test",
    "figure_ptbxl_internal_confusion_matrices_300dpi.png"
)

plot_roc_pr_curves(
    y_ptbxl,
    p_ptbxl,
    "PTB-XL Internal Test",
    "figure_ptbxl_internal"
)


# ============================================================
# 16C. OPTIONAL FIVE-FOLD PATIENT-WISE ROBUSTNESS ANALYSIS
# ============================================================
# This analysis is reported separately from the fixed train/validation/test
# experiment above. It must never be mixed into the held-out-test tables.

def run_patientwise_five_fold_cv():
    all_meta = pd.concat([train_meta, valid_meta, test_meta], ignore_index=True)
    all_sig_map = {}
    all_sig_map.update(train_sig_map)
    all_sig_map.update(valid_sig_map)
    all_sig_map.update(test_sig_map)

    if "patient_id_audit" not in all_meta.columns:
        raise RuntimeError("Five-fold CV requires patient_id_audit.")

    patient_table = all_meta.groupby("patient_id_audit")[LABEL_COLS].max().reset_index()
    patient_ids = patient_table["patient_id_audit"].astype(str).to_numpy()
    patient_y = patient_table[LABEL_COLS].to_numpy(dtype=int)
    outer = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fold_rows, oof_frames, split_rows = [], [], []
    for fold, (dev_idx, test_idx) in enumerate(outer.split(patient_ids, patient_y), start=1):
        dev_patients = patient_ids[dev_idx]
        dev_y = patient_y[dev_idx]
        test_patients = set(patient_ids[test_idx])

        inner = MultilabelStratifiedKFold(
            n_splits=5, shuffle=True, random_state=RANDOM_STATE + fold
        )
        inner_train_idx, inner_val_idx = next(inner.split(dev_patients, dev_y))
        train_patients = set(dev_patients[inner_train_idx])
        val_patients = set(dev_patients[inner_val_idx])

        if (train_patients & val_patients) or (train_patients & test_patients) or (val_patients & test_patients):
            raise AssertionError(f"Patient overlap detected in CV fold {fold}.")

        tr = all_meta[all_meta["patient_id_audit"].isin(train_patients)].copy().reset_index(drop=True)
        va = all_meta[all_meta["patient_id_audit"].isin(val_patients)].copy().reset_index(drop=True)
        te = all_meta[all_meta["patient_id_audit"].isin(test_patients)].copy().reset_index(drop=True)

        for split_name, frame in [("train", tr), ("validation", va), ("outer_test", te)]:
            split_rows.append({
                "fold": fold, "split": split_name,
                "patients": int(frame["patient_id_audit"].nunique()),
                "ecg_records": int(len(frame)),
                **{f"positive_{c}": int(frame[c].sum()) for c in LABEL_COLS},
            })

        fold_scaler = fit_scaler(all_sig_map, tr["ecg_id"].astype(str).to_numpy())
        seed_all(RANDOM_STATE + fold)
        fold_model = ProposedECG_LabelTokenTransformer(
            d_model=CFG.D_MODEL, nhead=CFG.NHEAD, depth=CFG.TX_DEPTH,
            drop=CFG.DROPOUT, use_label_tokens=True, use_se=True
        )
        remaining_folds = 6 - fold
        # Share the remaining training time across unprocessed folds. Twenty
        # percent of each share is retained for fold inference and file output.
        fold_training_deadline = (
            time.monotonic()
            + 0.80 * runtime_seconds_remaining() / max(1, remaining_folds)
        )
        fold_model, fold_ema, fold_ths, fold_hist = train_model(
            fold_model, tr, all_sig_map, va, all_sig_map, fold_scaler,
            model_name=f"CV_Fold_{fold}_Proposed",
            epochs=CFG.EPOCHS, lr=CFG.LR, use_ema=True,
            training_deadline_monotonic=fold_training_deadline,
        )

        te_ds = PTBXLWindowDataset(
            te, all_sig_map, fold_scaler,
            window=CFG.T_WINDOW, stride=CFG.STRIDE_EVAL
        )
        te_loader = DataLoader(
            te_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
            **dataloader_runtime_kwargs()
        )
        ids, y, p = infer_ecg_level(fold_model, te_loader, ema=fold_ema)
        m = metrics_all(y, p, fold_ths)
        fold_rows.append({
            "fold": fold, "Macro-AUROC": m["macro_auc"], "Macro-AP": m["macro_ap"],
            "Macro-F1": m["macro_f1"], "Micro-F1": m["micro_f1"],
            "Hamming Acc.": m["hamming_acc"], "Jaccard": m["jaccard"],
            "Subset Acc.": m["subset_acc"],
            "Epochs trained": int(len(fold_hist)),
        })
        frame = pd.DataFrame({"ecg_id": ids, "fold": fold})
        patient_lookup = te.set_index("ecg_id")["patient_id_audit"].astype(str).to_dict()
        frame["patient_id"] = frame["ecg_id"].map(patient_lookup)
        for j, c in enumerate(LABEL_COLS):
            frame[f"y_{c}"] = y[:, j]
            frame[f"p_{c}"] = p[:, j]
            frame[f"threshold_{c}"] = fold_ths[j]
        oof_frames.append(frame)

        del fold_model, fold_ema, fold_scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(os.path.join(CFG.OUT_DIR, "five_fold_patientwise_metrics.csv"), index=False)
    pd.DataFrame(split_rows).to_csv(
        os.path.join(CFG.OUT_DIR, "five_fold_patientwise_split_manifest.csv"), index=False
    )
    oof_df = pd.concat(oof_frames, ignore_index=True)
    if oof_df["ecg_id"].duplicated().any():
        raise AssertionError("An ECG appeared in more than one outer test fold.")
    oof_df.to_csv(os.path.join(CFG.OUT_DIR, "five_fold_patientwise_oof_predictions.csv"), index=False)

    y_oof = oof_df[[f"y_{c}" for c in LABEL_COLS]].to_numpy(dtype=int)
    p_oof = oof_df[[f"p_{c}" for c in LABEL_COLS]].to_numpy(dtype=float)
    # Each OOF row uses the threshold selected within its own outer-fold development data.
    pred_oof = np.zeros_like(y_oof)
    for j, c in enumerate(LABEL_COLS):
        pred_oof[:, j] = (
            oof_df[f"p_{c}"].to_numpy() >= oof_df[f"threshold_{c}"].to_numpy()
        ).astype(int)
    pooled = {
        "Macro-AUROC": np.nanmean([
            roc_auc_score(y_oof[:, j], p_oof[:, j]) if len(np.unique(y_oof[:, j])) == 2 else np.nan
            for j in range(C)
        ]),
        "Macro-AP": np.nanmean([
            average_precision_score(y_oof[:, j], p_oof[:, j]) if len(np.unique(y_oof[:, j])) == 2 else np.nan
            for j in range(C)
        ]),
        "Macro-F1": f1_score(y_oof, pred_oof, average="macro", zero_division=0),
        "Micro-F1": f1_score(y_oof, pred_oof, average="micro", zero_division=0),
        "Hamming Acc.": float((y_oof == pred_oof).mean()),
        "Jaccard": jaccard_score(y_oof, pred_oof, average="samples", zero_division=0),
        "Subset Acc.": accuracy_score(y_oof, pred_oof),
    }
    pd.DataFrame([pooled]).to_csv(
        os.path.join(CFG.OUT_DIR, "five_fold_patientwise_pooled_oof_metrics.csv"), index=False
    )
    return folds_df, oof_df, pooled


cv_epoch_seconds = (
    float(proposed_hist["epoch_seconds"].median())
    if "epoch_seconds" in proposed_hist and len(proposed_hist) else 15 * 60.0
)
cv_minimum_estimated_seconds = (
    5 * CFG.EPOCHS * cv_epoch_seconds * 1.10 + 10 * 60
)
if CFG.RUN_FIVE_FOLD_CV:
    CFG.RUN_FIVE_FOLD_CV = runtime_allows_optional_stage(
        "complete five-fold CV", cv_minimum_estimated_seconds
    )
if CFG.RUN_FIVE_FOLD_CV:
    cv_fold_metrics_df, cv_oof_df, cv_pooled_metrics = run_patientwise_five_fold_cv()
    print("\nFive-fold patient-wise CV is a separate robustness analysis:")
    display(cv_fold_metrics_df)
    display(pd.DataFrame([cv_pooled_metrics]))


# ============================================================
# 14. EXTERNAL DATASET EVALUATION
# ============================================================

external_eval_results = {}

if CFG.RUN_EXTERNAL_DATASETS:
    for ds_name, meta_df in external_meta_dict.items():
        # Reading and resampling thousands of WFDB recordings can dominate
        # runtime. Skip an entire cohort instead of publishing a partial cohort.
        if not runtime_allows_optional_stage(
            f"external cohort {ds_name}", estimated_seconds=30 * 60
        ):
            print(f"SKIPPED: {ds_name}; report it as unevaluated in this run.")
            continue
        ids_ext, y_ext, p_ext, m_ext = evaluate_external_dataset(
            model=proposed_model,
            ema=proposed_ema,
            scaler=signal_scaler,
            ths=proposed_ths,
            dataset_name=ds_name,
            meta_df=meta_df
        )

        external_eval_results[ds_name] = {
            "ids": ids_ext,
            "y": y_ext,
            "p": p_ext,
            "metrics": m_ext
        }


# ============================================================
# 15. INTERNAL VS EXTERNAL SUMMARY TABLE
# ============================================================

summary_rows = [
    {
        "Dataset": "PTB-XL reformatted",
        "Protocol": "Internal test",
        "Records evaluated": len(y_ptbxl),
        "Macro-AUROC": m_ptbxl["macro_auc"],
        "Macro-AP": m_ptbxl["macro_ap"],
        "Macro-F1": m_ptbxl["macro_f1"],
        "Micro-F1": m_ptbxl["micro_f1"],
        "Hamming Acc.": m_ptbxl["hamming_acc"],
        "Jaccard": m_ptbxl["jaccard"],
        "Subset Acc.": m_ptbxl["subset_acc"],
    }
]

for ds_name, obj in external_eval_results.items():
    m = obj["metrics"]
    y = obj["y"]

    summary_rows.append({
        "Dataset": ds_name,
        "Protocol": "Zero-shot external test using PTB-XL thresholds",
        "Records evaluated": len(y),
        "Macro-AUROC": m["macro_auc"],
        "Macro-AP": m["macro_ap"],
        "Macro-F1": m["macro_f1"],
        "Micro-F1": m["micro_f1"],
        "Hamming Acc.": m["hamming_acc"],
        "Jaccard": m["jaccard"],
        "Subset Acc.": m["subset_acc"],
    })

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(CFG.OUT_DIR, "internal_vs_all_external_summary.csv")
summary_df.to_csv(summary_path, index=False)

print("\nInternal vs External Summary:")
display(
    summary_df.style.format({
        "Macro-AUROC": "{:.4f}",
        "Macro-AP": "{:.4f}",
        "Macro-F1": "{:.4f}",
        "Micro-F1": "{:.4f}",
        "Hamming Acc.": "{:.4f}",
        "Jaccard": "{:.4f}",
        "Subset Acc.": "{:.4f}",
    })
)

print("Saved:", summary_path)

# Conservative, result-driven wording for Reviewer comment 9.
interpretation_lines = [
    "# External evaluation interpretation",
    "",
    "These source-defined evaluations measure zero-shot transfer using thresholds selected "
    "only from PTB-XL. They do not establish reliable multicentre clinical generalization.",
    "",
]
for _, row in summary_df.iterrows():
    if row["Protocol"].startswith("Zero-shot"):
        interpretation_lines.append(
            f"- {row['Dataset']}: Macro-AUROC={row['Macro-AUROC']:.4f}, "
            f"Macro-AP={row['Macro-AP']:.4f}, Macro-F1={row['Macro-F1']:.4f}, "
            f"Micro-F1={row['Micro-F1']:.4f}."
        )
interpretation_lines.extend([
    "",
    "Performance differences should be interpreted in relation to sampling conditions, "
    "recording duration, label compatibility, positive-class support, and prevalence shift. "
    "A high Hamming accuracy accompanied by low or zero F1 is not evidence of successful "
    "transfer because abundant negative labels can dominate Hamming accuracy.",
])
with open(os.path.join(CFG.OUT_DIR, "external_interpretation_cautious.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(interpretation_lines) + "\n")

plt.figure(figsize=(12, 5))
x = np.arange(len(summary_df))
plt.bar(x, summary_df["Macro-F1"].values)
plt.xticks(x, summary_df["Dataset"].values, rotation=25, ha="right")
plt.ylabel("Macro-F1")
plt.title("Internal PTB-XL vs External Dataset Evaluation")
savefig(os.path.join(CFG.OUT_DIR, "figure_internal_external_macro_f1_comparison_300dpi.png"))

plt.figure(figsize=(14, 5))
width = 0.18
metrics_to_plot = ["Macro-AUROC", "Macro-AP", "Macro-F1", "Micro-F1"]
x = np.arange(len(summary_df))

for i, metric in enumerate(metrics_to_plot):
    plt.bar(x + (i - 1.5) * width, summary_df[metric].values, width, label=metric)

plt.xticks(x, summary_df["Dataset"].values, rotation=25, ha="right")
plt.ylabel("Score")
plt.title("Internal and External Dataset Comparison Across Metrics")
plt.legend()
savefig(os.path.join(CFG.OUT_DIR, "figure_internal_external_multi_metric_comparison_300dpi.png"))


# ============================================================
# 16. UNIFIED PTB-XL BASELINE COMPARISON
# ============================================================

baseline_results = []
comparison_predictions = {
    "Full proposed model": {
        "ids": ptbxl_ids, "y": y_ptbxl, "p": p_ptbxl, "thresholds": proposed_ths,
        "epochs_trained": proposed_epochs_trained,
    }
}

baseline_results.append({
    "Model": "Proposed CNN-SE + Label-Token Transformer",
    "Dataset": "PTB-XL reformatted test",
    "Evaluation": "Same train/valid/test split, same threshold protocol",
    "Epochs trained": proposed_epochs_trained,
    "Macro-AUROC": m_ptbxl["macro_auc"],
    "Macro-AP": m_ptbxl["macro_ap"],
    "Macro-F1": m_ptbxl["macro_f1"],
    "Micro-F1": m_ptbxl["micro_f1"],
    "Hamming Acc.": m_ptbxl["hamming_acc"],
    "Jaccard": m_ptbxl["jaccard"],
    "Subset Acc.": m_ptbxl["subset_acc"]
})


def train_and_eval_baseline(model, name):
    print("\n" + "="*90)
    print(f"TRAINING BASELINE: {name}")
    print("="*90)

    seed_all(RANDOM_STATE)
    model, ema, ths, hist_df = train_model(
        model=model,
        tr_meta=train_meta,
        tr_map=train_sig_map,
        va_meta=valid_meta,
        va_map=valid_sig_map,
        scaler=signal_scaler,
        model_name=name,
        epochs=CFG.BASELINE_EPOCHS,
        lr=CFG.BASELINE_LR,
        use_ema=True
    )

    ids, y, p, m = evaluate_ptbxl_test(
        model,
        ema,
        signal_scaler,
        ths,
        name
    )

    row = {
        "Model": name,
        "Dataset": "PTB-XL reformatted test",
        "Evaluation": "Same train/valid/test split, same threshold protocol",
        "Macro-AUROC": m["macro_auc"],
        "Macro-AP": m["macro_ap"],
        "Macro-F1": m["macro_f1"],
        "Micro-F1": m["micro_f1"],
        "Hamming Acc.": m["hamming_acc"],
        "Jaccard": m["jaccard"],
        "Subset Acc.": m["subset_acc"]
    }

    comparison_predictions[name] = {
        "ids": ids, "y": y, "p": p, "thresholds": ths,
        "epochs_trained": int(len(hist_df)),
    }
    row["Epochs trained"] = int(len(hist_df))
    return row


if CFG.RUN_BASELINES:
    estimated = 4 * CFG.EPOCHS * cv_epoch_seconds * 1.20 + 15 * 60
    CFG.RUN_BASELINES = runtime_allows_optional_stage(
        "four complete PTB-XL baselines", estimated_seconds=estimated
    )
if CFG.RUN_BASELINES:
    baseline_specs = [
        ("Baseline_1D_CNN", BaselineCNN1D(d_model=CFG.D_MODEL, drop=CFG.DROPOUT, use_se=False)),
        ("Baseline_1D_CNN_SE", BaselineCNN1D(d_model=CFG.D_MODEL, drop=CFG.DROPOUT, use_se=True)),
        ("Baseline_Transformer_Only", BaselineTransformerOnly(
            d_model=CFG.D_MODEL,
            nhead=CFG.NHEAD,
            depth=CFG.TX_DEPTH,
            drop=CFG.DROPOUT
        )),
        ("Baseline_CNN_SE_Transformer_NoLabelTokens", ProposedECG_LabelTokenTransformer(
            d_model=CFG.D_MODEL,
            nhead=CFG.NHEAD,
            depth=CFG.TX_DEPTH,
            drop=CFG.DROPOUT,
            use_label_tokens=False,
            use_se=True
        )),
    ]

    for name, model_b in baseline_specs:
        row = train_and_eval_baseline(model_b, name)
        baseline_results.append(row)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

fair_df = pd.DataFrame(baseline_results)
fair_df = fair_df.sort_values("Macro-F1", ascending=False).reset_index(drop=True)

fair_path = os.path.join(CFG.OUT_DIR, "unified_ptbxl_fair_baseline_comparison.csv")
fair_df.to_csv(fair_path, index=False)

print("\nUnified PTB-XL Fair Baseline Comparison:")
display(
    fair_df.style.format({
        "Macro-AUROC": "{:.4f}",
        "Macro-AP": "{:.4f}",
        "Macro-F1": "{:.4f}",
        "Micro-F1": "{:.4f}",
        "Hamming Acc.": "{:.4f}",
        "Jaccard": "{:.4f}",
        "Subset Acc.": "{:.4f}",
    })
)

print("Saved:", fair_path)

plt.figure(figsize=(12, 5))
x = np.arange(len(fair_df))
plt.bar(x, fair_df["Macro-F1"].values)
plt.xticks(x, fair_df["Model"].values, rotation=30, ha="right")
plt.ylabel("Macro-F1")
plt.title("Unified PTB-XL Fair Comparison: Macro-F1")
savefig(os.path.join(CFG.OUT_DIR, "figure_unified_ptbxl_macro_f1_comparison_300dpi.png"))

plt.figure(figsize=(12, 5))
width = 0.2
metrics_to_plot = ["Macro-AUROC", "Macro-AP", "Macro-F1", "Micro-F1"]
x = np.arange(len(fair_df))

for i, metric in enumerate(metrics_to_plot):
    plt.bar(x + (i - 1.5) * width, fair_df[metric].values, width, label=metric)

plt.xticks(x, fair_df["Model"].values, rotation=30, ha="right")
plt.ylabel("Score")
plt.title("Unified PTB-XL Fair Comparison Across Metrics")
plt.legend()
savefig(os.path.join(CFG.OUT_DIR, "figure_unified_ptbxl_multi_metric_comparison_300dpi.png"))


# ============================================================
# 16B. COMPONENT ABLATIONS UNDER THE SAME FIXED PROTOCOL
# ============================================================

def train_ablation_variant(name, model, use_ema=True):
    seed_all(RANDOM_STATE)
    model, ema, ths, hist_df = train_model(
        model=model,
        tr_meta=train_meta,
        tr_map=train_sig_map,
        va_meta=valid_meta,
        va_map=valid_sig_map,
        scaler=signal_scaler,
        model_name=name,
        epochs=CFG.EPOCHS,
        lr=CFG.LR,
        use_ema=use_ema,
    )
    ids, y, p, m = evaluate_ptbxl_test(model, ema, signal_scaler, ths, name)
    if list(map(str, ids)) != list(map(str, ptbxl_ids)) or not np.array_equal(y, y_ptbxl):
        raise RuntimeError(f"Ablation {name} was not evaluated on the identical ordered test records.")
    comparison_predictions[name] = {
        "ids": ids, "y": y, "p": p, "thresholds": ths,
        "epochs_trained": int(len(hist_df)),
    }
    return m


if CFG.RUN_ABLATIONS:
    variants_needed = 2 + int(not CFG.RUN_BASELINES)
    estimated = variants_needed * CFG.EPOCHS * cv_epoch_seconds * 1.20 + 15 * 60
    CFG.RUN_ABLATIONS = runtime_allows_optional_stage(
        "complete architectural ablations", estimated_seconds=estimated
    )
if CFG.RUN_ABLATIONS:
    no_label_name = "Baseline_CNN_SE_Transformer_NoLabelTokens"
    if no_label_name not in comparison_predictions:
        train_ablation_variant(
            no_label_name,
            ProposedECG_LabelTokenTransformer(
                d_model=CFG.D_MODEL, nhead=CFG.NHEAD, depth=CFG.TX_DEPTH,
                drop=CFG.DROPOUT, use_label_tokens=False, use_se=True
            ),
            use_ema=True,
        )

    no_se_name = "Ablation_Without_SE"
    train_ablation_variant(
        no_se_name,
        ProposedECG_LabelTokenTransformer(
            d_model=CFG.D_MODEL, nhead=CFG.NHEAD, depth=CFG.TX_DEPTH,
            drop=CFG.DROPOUT, use_label_tokens=True, use_se=False
        ),
        use_ema=True,
    )

    no_ema_name = "Ablation_Without_EMA"
    train_ablation_variant(
        no_ema_name,
        ProposedECG_LabelTokenTransformer(
            d_model=CFG.D_MODEL, nhead=CFG.NHEAD, depth=CFG.TX_DEPTH,
            drop=CFG.DROPOUT, use_label_tokens=True, use_se=True
        ),
        use_ema=False,
    )

    ablation_rows = []
    ordered_variants = [
        ("Full proposed model", "Full proposed model"),
        ("Without SE block", no_se_name),
        ("Without label-token Transformer", no_label_name),
        ("Without EMA", no_ema_name),
    ]
    for display_name, key in ordered_variants:
        obj = comparison_predictions[key]
        m = metrics_all(obj["y"], obj["p"], obj["thresholds"])
        ablation_rows.append({
            "Model variant": display_name,
            "Macro-AUROC": m["macro_auc"], "Macro-AP": m["macro_ap"],
            "Macro-F1": m["macro_f1"], "Micro-F1": m["micro_f1"],
            "Hamming Acc.": m["hamming_acc"], "Jaccard": m["jaccard"],
            "Subset Acc.": m["subset_acc"],
            "Training epochs": int(obj.get("epochs_trained", CFG.EPOCHS)),
        })

    # Fixed threshold is not trained and uses the exact full-model probabilities.
    ablation_rows.append({
        "Model variant": "Same full model; fixed threshold = 0.5",
        "Macro-AUROC": m_ptbxl_fixed05["macro_auc"],
        "Macro-AP": m_ptbxl_fixed05["macro_ap"],
        "Macro-F1": m_ptbxl_fixed05["macro_f1"],
        "Micro-F1": m_ptbxl_fixed05["micro_f1"],
        "Hamming Acc.": m_ptbxl_fixed05["hamming_acc"],
        "Jaccard": m_ptbxl_fixed05["jaccard"],
        "Subset Acc.": m_ptbxl_fixed05["subset_acc"],
        "Training epochs": 0,
    })
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(os.path.join(CFG.OUT_DIR, "unified_ablation_table.csv"), index=False)
    display(ablation_df)

    if CFG.RUN_PAIRED_BOOTSTRAP and runtime_allows_optional_stage(
        "paired patient bootstrap", estimated_seconds=15 * 60
    ):
        if "patient_id_audit" not in test_meta.columns:
            raise RuntimeError("Patient identifiers are required for paired patient-level bootstrap.")
        patient_lookup = test_meta.set_index("ecg_id")["patient_id_audit"].astype(str).to_dict()
        bootstrap_tables = []
        for variant_key in [no_se_name, no_label_name, no_ema_name]:
            obj = comparison_predictions[variant_key]
            bootstrap_tables.append(paired_patient_bootstrap(
                ptbxl_ids, y_ptbxl, p_ptbxl, proposed_ths,
                obj["p"], obj["thresholds"], patient_lookup,
                variant_name=variant_key,
                repeats=CFG.BOOTSTRAP_REPEATS,
                seed=CFG.BOOTSTRAP_SEED,
            ))
        paired_stats_df = pd.concat(bootstrap_tables, ignore_index=True)
        paired_stats_df.to_csv(
            os.path.join(CFG.OUT_DIR, "paired_patient_bootstrap_ablation_significance.csv"),
            index=False,
        )
        display(paired_stats_df)


# ============================================================
# 17. t-SNE FEATURE-SPACE VISUALIZATION
# ============================================================

@torch.no_grad()
def extract_ecg_level_features(model, loader, ema=None):
    if ema is not None:
        ema.copy_to(model)

    model.eval()
    all_ids, all_y, all_f = [], [], []

    for x, y, ecg_id in tqdm(loader, desc="Extracting features"):
        x = x.to(DEVICE)

        if hasattr(model, "forward_features"):
            _, feat = model.forward_features(x)
        else:
            logits = model(x)
            feat = logits

        feat = feat.detach().cpu().numpy()

        all_f.append(feat)
        all_y.append(y.numpy())

        if isinstance(ecg_id, (list, tuple)):
            all_ids.extend([str(v) for v in ecg_id])
        else:
            all_ids.extend([str(v) for v in ecg_id])

    all_f = np.concatenate(all_f, axis=0)
    all_y = np.concatenate(all_y, axis=0).astype(int)

    dff = pd.DataFrame({"ecg_id": all_ids})
    for j in range(all_f.shape[1]):
        dff[f"f{j}"] = all_f[:, j]

    dfy = pd.DataFrame(all_y, columns=[f"y_{c}" for c in LABEL_COLS])
    dfy["ecg_id"] = all_ids

    feat_cols = [c for c in dff.columns if c.startswith("f")]

    f_ecg = dff.groupby("ecg_id")[feat_cols].mean()
    y_ecg = dfy.groupby("ecg_id")[[f"y_{c}" for c in LABEL_COLS]].max()

    ids = f_ecg.index.tolist()
    F_ecg = f_ecg.values.astype(np.float32)
    Y_ecg = y_ecg.values.astype(int)

    return ids, Y_ecg, F_ecg


def primary_label_from_multilabel(Y):
    labels = []

    for row in Y:
        idx = np.where(row == 1)[0]

        if len(idx) == 0:
            labels.append("None")
        elif len(idx) == 1:
            labels.append(LABEL_COLS[idx[0]])
        else:
            labels.append("Multi-label")

    return np.array(labels)


def run_tsne_and_plot(ids, Y, Fmat, dataset_name, prefix):
    n = len(ids)

    if n > CFG.TSNE_MAX_SAMPLES:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(n, size=CFG.TSNE_MAX_SAMPLES, replace=False)
        ids_use = [ids[i] for i in idx]
        Y_use = Y[idx]
        F_use = Fmat[idx]
    else:
        ids_use = ids
        Y_use = Y
        F_use = Fmat

    labels = primary_label_from_multilabel(Y_use)

    perplexity = min(30, max(5, len(ids_use)//20))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_STATE
    )

    Z = tsne.fit_transform(F_use)

    tsne_df = pd.DataFrame({
        "ecg_id": ids_use,
        "tsne_1": Z[:, 0],
        "tsne_2": Z[:, 1],
        "primary_label": labels
    })

    for j, c in enumerate(LABEL_COLS):
        tsne_df[f"y_{c}"] = Y_use[:, j]

    csv_path = os.path.join(CFG.OUT_DIR, f"{prefix}_tsne_coordinates.csv")
    tsne_df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    plt.figure(figsize=(8, 6))
    unique_labels = ["NORM", "MI", "STTC", "CD", "HYP", "Multi-label", "None"]

    for lab in unique_labels:
        sub = tsne_df[tsne_df["primary_label"] == lab]
        if len(sub) == 0:
            continue
        plt.scatter(sub["tsne_1"], sub["tsne_2"], s=12, alpha=0.75, label=lab)

    plt.title(f"t-SNE Visualization of Learned ECG Embeddings - {dataset_name}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(markerscale=2, fontsize=8)
    savefig(os.path.join(CFG.OUT_DIR, f"{prefix}_tsne_embeddings_300dpi.png"))

    return tsne_df


if CFG.RUN_TSNE and runtime_allows_optional_stage(
    "PTB-XL t-SNE and figure export", estimated_seconds=10 * 60
):
    ptbxl_test_ds = PTBXLWindowDataset(
        test_meta,
        test_sig_map,
        signal_scaler,
        window=CFG.T_WINDOW,
        stride=CFG.STRIDE_EVAL
    )

    ptbxl_test_loader = DataLoader(
        ptbxl_test_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        **dataloader_runtime_kwargs()
    )

    ids_feat, y_feat, f_feat = extract_ecg_level_features(
        proposed_model,
        ptbxl_test_loader,
        ema=proposed_ema
    )

    tsne_ptbxl_df = run_tsne_and_plot(
        ids_feat,
        y_feat,
        f_feat,
        "PTB-XL Internal Test",
        "figure_ptbxl_internal"
    )

    for ds_name, obj in external_eval_results.items() if CFG.RUN_EXTERNAL_TSNE else []:
        meta_df = external_meta_dict[ds_name]

        ext_ds = WFDBExternalWindowDataset(
            meta_df,
            signal_scaler,
            window=CFG.T_WINDOW,
            stride=CFG.STRIDE_EVAL,
            target_fs=CFG.TARGET_FS
        )

        ext_loader = DataLoader(
            ext_ds,
            batch_size=CFG.BATCH_SIZE,
            shuffle=False,
            **dataloader_runtime_kwargs()
        )

        ids_feat_ext, y_feat_ext, f_feat_ext = extract_ecg_level_features(
            proposed_model,
            ext_loader,
            ema=proposed_ema
        )

        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", ds_name).strip("_")

        run_tsne_and_plot(
            ids_feat_ext,
            y_feat_ext,
            f_feat_ext,
            ds_name,
            f"figure_external_{safe_name}"
        )


# ============================================================
# 18. SAVE MODEL AND FINAL ARTIFACTS
# ============================================================

def collect_reproducibility_manifest():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    gpu_vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if torch.cuda.is_available() else 0.0
    )
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_STATE,
        "device": DEVICE,
        "gpu": gpu_name,
        "gpu_total_vram_gb": round(float(gpu_vram_gb), 3),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "operating_system": platform.platform(),
        "target_sampling_frequency_hz": CFG.TARGET_FS,
        "inferred_ptbxl_source_frequency_hz": PTBXL_SOURCE_FS,
        "window_seconds": CFG.WINDOW_SECONDS,
        "window_samples": CFG.T_WINDOW,
        "training_stride_seconds": CFG.STRIDE_TRAIN_SECONDS,
        "evaluation_stride_seconds": CFG.STRIDE_EVAL_SECONDS,
        "batch_size": CFG.BATCH_SIZE,
        "maximum_epochs_per_trained_variant": CFG.EPOCHS,
        "early_stopping_enabled": CFG.EARLY_STOPPING,
        "early_stopping_patience": CFG.EARLY_STOPPING_PATIENCE,
        "early_stopping_minimum_epochs": CFG.EARLY_STOPPING_MIN_EPOCHS,
        "runtime_budget_hours": CFG.MAX_RUNTIME_HOURS,
        "finalization_reserve_minutes": CFG.FINALIZATION_RESERVE_MINUTES,
        "bootstrap_repeats": CFG.BOOTSTRAP_REPEATS,
        "external_tsne_enabled": CFG.RUN_EXTERNAL_TSNE,
        "optimizer": "AdamW",
        "learning_rate": CFG.LR,
        "weight_decay": CFG.WD,
        "warmup_epochs": CFG.WARMUP_EPOCHS,
        "scheduler": "linear warmup followed by cosine decay",
        "asymmetric_loss_gamma_positive": 0.0,
        "asymmetric_loss_gamma_negative": 3.0,
        "asymmetric_loss_clip": 0.05,
        "ema_decay": CFG.EMA_DECAY,
        "embedding_dimension": CFG.D_MODEL,
        "transformer_layers": CFG.TX_DEPTH,
        "attention_heads": CFG.NHEAD,
        "dropout": CFG.DROPOUT,
        "patch_samples": 50,
        "se_reduction_ratio": 8,
        "threshold_grid_minimum": CFG.THR_MIN,
        "threshold_grid_maximum": CFG.THR_MAX,
        "threshold_grid_points": CFG.THR_GRID,
        "threshold_tie_break": "closest to 0.5",
        "amplitude_scaling": "probability 0.50; multiplicative range [0.9, 1.1]",
        "gaussian_noise": "probability 0.35; standard deviation 0.01 after normalization",
        "temporal_shift": "probability 0.35; integer shift [-60, 60] samples",
        "lead_dropout": "probability 0.25; one randomly selected lead",
        "external_threshold_policy": "zero-shot: PTB-XL validation thresholds, no external recalibration",
        "code_archive_doi": CFG.ZENODO_DOI,
    }


with open(os.path.join(CFG.OUT_DIR, "reproducibility_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(collect_reproducibility_manifest(), f, indent=2)

split_protocol = {
    "primary_protocol": (
        "Provided PTB-XL train/validation/test files, accepted only after a zero-overlap "
        "patient audit. Normalization and windowing occur after assignment. Validation is "
        "used for checkpoint and threshold selection; test is used once for final evaluation."
    ),
    "five_fold_protocol": (
        "Separate robustness analysis over all PTB-XL patients. Each outer fold is patient-wise; "
        "an inner patient-wise split supplies validation patients. Pooled OOF values are not mixed "
        "with the fixed held-out-test result."
    ),
    "window_leakage_rule": "Every window inherits the split of its parent ECG and patient.",
    "external_protocol": (
        "Source-defined, deduplicated zero-shot cohorts; no external data are used for training, "
        "checkpoint selection, normalization fitting, or threshold selection."
    ),
}
with open(os.path.join(CFG.OUT_DIR, "evaluation_protocol_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(split_protocol, f, indent=2)

figure_statement = (
    "Statistical plots produced by this program are deterministic Matplotlib visualizations of "
    "computed results and are not generative-AI images. For manuscript workflow Figures 1-5, "
    "the authors must state the actual software and any AI assistance truthfully. Recommended "
    "final statement if redrawn manually: 'Figures 1-5 were recreated manually by the authors "
    "using [SOFTWARE]. No generative-AI or text-to-image system was used in the revised figures.'"
)
with open(os.path.join(CFG.OUT_DIR, "figure_provenance_statement_template.txt"), "w", encoding="utf-8") as f:
    f.write(figure_statement + "\n")

if CFG.ZENODO_DOI.startswith("TO_BE_"):
    print("WARNING: Deposit this code and its configuration in Zenodo and replace CFG.ZENODO_DOI before submission.")

model_path = os.path.join(CFG.OUT_DIR, "proposed_cnn_se_label_token_transformer_state_dict.pt")
torch.save(proposed_model.state_dict(), model_path)
print("Saved model:", model_path)

thr_path = os.path.join(CFG.OUT_DIR, "optimized_ptbxl_thresholds.csv")
pd.DataFrame({
    "Class": LABEL_COLS,
    "Threshold": proposed_ths
}).to_csv(thr_path, index=False)

print("Saved thresholds:", thr_path)

print("\nAll results saved in:", CFG.OUT_DIR)
print(f"Total notebook runtime: {elapsed_runtime_hours():.2f} hours")

print("\nMain output files:")
for f in sorted(glob.glob(os.path.join(CFG.OUT_DIR, "*"))):
    print(os.path.basename(f))

# ============================================================
# 19. PUBLICATION FIGURES FROM THIS RUN (NEVER FROM TARGET SCORES)
# ============================================================
def paper_figure(path):
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.OUT_DIR, path), dpi=CFG.FIG_DPI,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved", path)


def manuscript_workflow_figures():
    schemes = [
        (1, "ECG classification workflow", ["Patient-wise split", "Train-only scaling", "4-s ECG windows", "CNN + SE", "Label tokens", "ECG prediction"]),
        (2, "Residual 1D CNN morphology encoder", ["12 ECG leads", "Convolution stem", "Residual block 1", "Residual block 2", "Residual block 3"]),
        (3, "Squeeze-and-excitation recalibration", ["CNN features", "Global mean", "Bottleneck", "Sigmoid gates", "Feature weighting"]),
        (4, "Temporal patch and label-token encoder", ["Temporal patches", "Position embedding", "Five label tokens", "Transformer", "Five logits"]),
        (5, "Aggregation and evaluation", ["Window sigmoid", "ECG-level mean", "Validation thresholds", "Held-out decisions", "Metrics"]),
    ]
    for n, title, steps in schemes:
        fig, ax = plt.subplots(figsize=(max(11, 2.5*len(steps)), 2.7))
        ax.axis("off")
        for i, step in enumerate(steps):
            x = (i + 0.5)/len(steps)
            ax.text(x,.5,step,ha="center",va="center",transform=ax.transAxes,
                    fontsize=9,bbox=dict(boxstyle="round,pad=.6",fc="#edf4f9",ec="#397296"))
            if i + 1 < len(steps):
                ax.annotate("",xy=((i+1)/len(steps)+.025,.5),
                            xytext=((i+1)/len(steps)-.025,.5),xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="->",color="#397296"))
        ax.set_title(title,weight="bold")
        paper_figure(f"Figure_{n:02d}_workflow_300dpi.png")


def manuscript_cv_figures():
    if not CFG.RUN_FIVE_FOLD_CV:
        return
    path = os.path.join(CFG.OUT_DIR,"five_fold_patientwise_metrics.csv")
    if not os.path.exists(path):
        raise RuntimeError("Missing five-fold results; cannot make fold figures")
    df = pd.read_csv(path).sort_values("fold")
    if len(df)!=5 or (df["Epochs trained"] < CFG.MIN_COMPLETED_CV_EPOCHS).any():
        raise RuntimeError("Incomplete five-fold CV; refusing boxplot")
    histories=[]
    for fold in range(1,6):
        h=pd.read_csv(os.path.join(CFG.OUT_DIR,f"CV_Fold_{fold}_Proposed_training_history.csv"))
        if len(h)<CFG.MIN_COMPLETED_CV_EPOCHS or not np.isfinite(h["loss"]).all():
            raise RuntimeError(f"Fold {fold} has incomplete/non-finite training history")
        histories.append(h)
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    cols=[("val_macro_auc","Macro-AUROC"),("val_macro_ap","Macro-AP"),
          ("val_micro_f1","Micro-F1"),("val_hamming","Hamming accuracy")]
    for fold,h in enumerate(histories,1):
        for ax,(column,label) in zip(axes.flat,cols):
            ax.plot(h["epoch"],h[column],marker="o",markersize=2,
                    label=f"Fold {fold}")
            ax.set_title(label);ax.set_xlabel("Epoch");ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle("Five-fold validation dynamics",weight="bold")
    paper_figure("Figure_08_five_fold_training_dynamics_300dpi.png")
    plt.figure(figsize=(9,4))
    for fold,h in enumerate(histories,1):
        plt.plot(h["epoch"],h["loss"],label=f"Fold {fold}")
    plt.xlabel("Epoch");plt.ylabel("Finite training loss");plt.legend(ncol=5)
    plt.title("Five-fold training losses")
    paper_figure("Figure_09_five_fold_training_loss_300dpi.png")
    metrics=["Macro-AUROC","Micro-F1","Hamming Acc.","Subset Acc."]
    fig,axes=plt.subplots(1,4,figsize=(14,4))
    for ax,name in zip(axes,metrics):
        values=df[name].to_numpy(float)
        ax.boxplot(values,showmeans=True,widths=.4)
        ax.scatter(np.full(5,1.12),values,color="#2b7091",s=26,zorder=3)
        ax.set_title(name);ax.set_xticks([]);ax.set_ylim(max(0,values.min()-.06),min(1,values.max()+.06))
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("Held-out test metrics in five patient-wise folds",weight="bold")
    paper_figure("Figure_10_five_fold_boxplots_300dpi.png")


def manuscript_external_figures():
    if not external_eval_results:
        return
    names=list(external_eval_results)
    # Each panel uses the actual mapped labels, scores and predictions.
    for kind,n in [("roc",13),("pr",14)]:
        fig,axes=plt.subplots(2,2,figsize=(12,9))
        for ax,(name,obj) in zip(axes.flat,external_eval_results.items()):
            y,p=obj["y"],obj["p"]
            for j,c in enumerate(LABEL_COLS):
                if len(np.unique(y[:,j]))<2:continue
                if kind=="roc":
                    xx,yy,_=roc_curve(y[:,j],p[:,j]);value=roc_auc_score(y[:,j],p[:,j])
                else:
                    yy,xx,_=precision_recall_curve(y[:,j],p[:,j]);value=average_precision_score(y[:,j],p[:,j])
                ax.plot(xx,yy,label=f"{c} {value:.2f}")
            ax.set_title(name,fontsize=9);ax.legend(fontsize=7)
            ax.set(xlabel="FPR" if kind=="roc" else "Recall",
                   ylabel="TPR" if kind=="roc" else "Precision")
            ax.grid(alpha=.2)
        for ax in axes.flat[len(names):]:ax.axis("off")
        fig.suptitle(f"External one-vs-rest {kind.upper()} curves",weight="bold")
        paper_figure(f"Figure_{n:02d}_external_{kind}_300dpi.png")
    fig,axes=plt.subplots(len(names),C,figsize=(3*C,2.9*len(names)))
    axes=np.asarray(axes).reshape(len(names),C)
    for row,(name,obj) in enumerate(external_eval_results.items()):
        cm=multilabel_confusion_matrix(obj["y"],obj["metrics"]["y_hat"])
        for j,ax in enumerate(axes[row]):
            ax.imshow(cm[j],cmap="Blues");ax.set_title(f"{name}: {LABEL_COLS[j]}",fontsize=8)
            ax.set(xticks=[0,1],yticks=[0,1])
            for a in range(2):
                for b in range(2):ax.text(b,a,str(cm[j,a,b]),ha="center",va="center",fontsize=8)
    fig.suptitle("External one-vs-rest confusion matrices",weight="bold")
    paper_figure("Figure_12_external_confusions_300dpi.png")


def manuscript_comparison_figure(frame, filename, name_col, metrics):
    if frame is None or frame.empty:return
    fig,ax=plt.subplots(figsize=(12,5.8)); x=np.arange(len(frame));w=.8/len(metrics)
    for i,k in enumerate(metrics):
        ax.bar(x+(i-(len(metrics)-1)/2)*w,frame[k].to_numpy(float),w,label=k)
    ax.set_xticks(x,frame[name_col].astype(str),rotation=20,ha="right")
    ax.set_ylabel("Score");ax.set_ylim(0,1);ax.legend(ncol=len(metrics),fontsize=8)
    ax.set_title("Values computed in this run (one held-out test evaluation)")
    paper_figure(filename)


manuscript_workflow_figures()
manuscript_cv_figures()
manuscript_external_figures()
if CFG.RUN_BASELINES:
    manuscript_comparison_figure(fair_df,"Figure_15_PTBXL_baseline_comparison_300dpi.png",
        "Model",["Macro-AUROC","Macro-AP","Macro-F1","Micro-F1"])
if CFG.RUN_ABLATIONS:
    manuscript_comparison_figure(ablation_df,"Figure_16_PTBXL_ablation_300dpi.png",
        "Model variant",["Macro-AUROC","Macro-AP","Macro-F1","Micro-F1","Hamming Acc."])

# Reference values are for an explicit discrepancy audit, never used as plotted data.
reference={"Macro-AUROC":.9356,"Macro-AP":.8402,"Macro-F1":.7627,
           "Micro-F1":.7939,"Hamming Acc.":.8896}
observed={"Macro-AUROC":m_ptbxl["macro_auc"],"Macro-AP":m_ptbxl["macro_ap"],
          "Macro-F1":m_ptbxl["macro_f1"],"Micro-F1":m_ptbxl["micro_f1"],
          "Hamming Acc.":m_ptbxl["hamming_acc"]}
pd.DataFrame([{"metric":k,"Table_6_reported":reference[k],"this_run_heldout_test":observed[k],
               "difference":observed[k]-reference[k]} for k in reference]).to_csv(
    os.path.join(CFG.OUT_DIR,"manuscript_vs_this_run_audit.csv"),index=False)

figure_manifest = [
    (1,"Figure_01_workflow_300dpi.png","workflow schematic"),
    (2,"Figure_02_workflow_300dpi.png","residual CNN schematic"),
    (3,"Figure_03_workflow_300dpi.png","SE schematic"),
    (4,"Figure_04_workflow_300dpi.png","label-token schematic"),
    (5,"Figure_05_workflow_300dpi.png","aggregation and evaluation schematic"),
    (6,"figure_class_distribution_all_datasets_300dpi.png","measured class counts"),
    (7,"figure_ptbxl_internal_confusion_matrices_300dpi.png","held-out PTB-XL predictions"),
    (8,"Figure_08_five_fold_training_dynamics_300dpi.png","fold validation histories"),
    (9,"Figure_09_five_fold_training_loss_300dpi.png","fold training histories; check manuscript numbering"),
    (10,"Figure_10_five_fold_boxplots_300dpi.png","five observed held-out fold scores"),
    (11,"figure_ptbxl_internal_tsne_embeddings_300dpi.png","held-out ECG feature embeddings"),
    (12,"Figure_12_external_confusions_300dpi.png","external predictions"),
    (13,"Figure_13_external_roc_300dpi.png","external probabilities"),
    (14,"Figure_14_external_pr_300dpi.png","external probabilities"),
    (15,"Figure_15_PTBXL_baseline_comparison_300dpi.png","baseline comparison; single run"),
    (16,"Figure_16_PTBXL_ablation_300dpi.png","ablation comparison; single run, no invented SD"),
]
pd.DataFrame([{"Figure":i,"Output file":file,"Data basis":basis,
               "Exists":os.path.isfile(os.path.join(CFG.OUT_DIR,file))}
              for i,file,basis in figure_manifest]).to_csv(
    os.path.join(CFG.OUT_DIR,"manuscript_figure_manifest.csv"),index=False)
print("Figure manifest saved; inspect missing files before submission.")
