"""A_Patch zone classification with an R1-style max-pool head.

This A-specific branch reads both retained frozen-helper response streams,
appends patch geometry, and trains one patient-level max-pool MIL classifier.
B/B0/C/D use the classical zone pipeline instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import all_config as cfg
from e_ml import compute_metrics

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

N_SCALES = len(cfg.PATCH_SCALES)
N_ONEHOT = N_SCALES
STREAM_DIM = 5  # csPCa, PZ, TZ, csPCa*PZ, csPCa*TZ
CORE_DIM = STREAM_DIM * len(cfg.MAXPOOL_FEATURE_SETS)
MAXPOOL_INPUT_NAME = "+".join(cfg.MAXPOOL_FEATURE_SETS)
ZONE_THRESHOLD = 0.45


def _clean_id(value) -> str:
    return str(value).strip()


def _labels(path: Path) -> dict[str, int]:
    df = pd.read_excel(path, dtype={cfg.ID_COL: str})
    out: dict[str, int] = {}
    for _, row in df.iterrows():
        cs = str(row[cfg.CLIN_COL]).strip().lower()
        zone = str(row[cfg.ZONE_COL]).strip().upper()
        if cs in {"true", "1", "yes"} and zone in {"PZ", "TZ", "0", "1"}:
            out[_clean_id(row[cfg.ID_COL])] = 1 if zone in {"TZ", "1"} else 0
    return out


def _seeded_indices(case_id: str, scale_idx: np.ndarray, cap: int) -> np.ndarray:
    if cap <= 0 or len(scale_idx) <= cap:
        return np.arange(len(scale_idx))
    seed = int(hashlib.sha1(f"maxpool|{case_id}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    coarse = np.where(scale_idx > 0)[0]
    fine = np.where(scale_idx == 0)[0]
    if len(coarse) >= cap:
        return np.sort(rng.choice(coarse, cap, replace=False))
    room = cap - len(coarse)
    keep_fine = rng.choice(fine, min(room, len(fine)), replace=False)
    return np.sort(np.concatenate([coarse, keep_fine]))


class DetectionStore:
    def __init__(self, root: Path, labels_path: Path, bag_cap: int = cfg.MAXPOOL_BAG_CAP):
        self.root = Path(root)
        self.labels = _labels(labels_path)
        self.bag_cap = int(bag_cap)
        available_sets = []
        for feature_set in cfg.MAXPOOL_FEATURE_SETS:
            folder = self.root / feature_set
            available_sets.append({p.stem for p in folder.glob("*.npz")} if folder.exists() else set())
        self.available = set.intersection(*available_sets) if available_sets else set()
        self._bags: dict[str, np.ndarray] = {}

    def cases(self, case_ids: list[str] | None = None) -> list[str]:
        pool = sorted(self.available) if case_ids is None else [_clean_id(c) for c in case_ids]
        return [c for c in pool if c in self.available and c in self.labels]

    def label(self, case_id: str) -> int:
        return int(self.labels[case_id])

    def bag(self, case_id: str) -> np.ndarray:
        if case_id in self._bags:
            return self._bags[case_id]
        per_stream = []
        centers = shape = scale_idx = None
        for feature_set in cfg.MAXPOOL_FEATURE_SETS:
            with np.load(self.root / feature_set / f"{case_id}.npz") as data:
                scale_full = data["scale_idx"].astype(np.int64)
                idx = _seeded_indices(case_id, scale_full, self.bag_cap)
                stream_centers = data["centers"][idx].astype(np.float32)
                stream_shape = data["organ_shape"].astype(np.float32)
                stream_scores = data["scores"][idx].astype(np.float32)
                stream_scale_idx = scale_full[idx]
            if centers is None:
                centers, shape, scale_idx = stream_centers, stream_shape, stream_scale_idx
            elif (
                len(stream_scale_idx) != len(scale_idx)
                or not np.array_equal(stream_scale_idx, scale_idx)
                or not np.array_equal(stream_centers, centers)
            ):
                raise RuntimeError(f"Max-pool streams are not aligned for case {case_id}")
            cs, pz, tz = stream_scores[:, 0], stream_scores[:, 2], stream_scores[:, 3]
            per_stream.append(np.stack([cs, pz, tz, cs * pz, cs * tz], axis=1))
        core = np.concatenate(per_stream, axis=1)
        geometry = np.concatenate([
            centers / np.maximum(shape - 1.0, 1.0),
            np.eye(N_SCALES, dtype=np.float32)[scale_idx],
        ], axis=1)
        bag = np.nan_to_num(np.concatenate([core, geometry], axis=1), nan=0.0, posinf=0.0, neginf=0.0)
        self._bags[case_id] = bag.astype(np.float32)
        return self._bags[case_id]


class Standardizer:
    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, store: DetectionStore, case_ids: list[str], per_case: int = 3000):
        rng = np.random.default_rng(0)
        chunks = []
        for case_id in case_ids:
            x = store.bag(case_id)
            if len(x) > per_case:
                x = x[rng.choice(len(x), per_case, replace=False)]
            chunks.append(x)
        values = np.concatenate(chunks).astype(np.float64)
        mean, std = values.mean(axis=0), values.std(axis=0)
        std[std < 1e-8] = 1.0
        mean[-N_ONEHOT:] = 0.0
        std[-N_ONEHOT:] = 1.0
        self.mean_, self.std_ = mean.astype(np.float32), std.astype(np.float32)
        return self

    def apply(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean_) / self.std_).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean_, "std": self.std_}

    @classmethod
    def from_state(cls, state: dict):
        obj = cls()
        obj.mean_, obj.std_ = state["mean"], state["std"]
        return obj


class MaxPoolMIL(nn.Module):
    def __init__(self, in_dim: int = CORE_DIM + 3 + N_SCALES, hidden: int = 64, dropout: float = 0.20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(values)
        pooled = encoded.max(dim=0, keepdim=True).values
        return self.classifier(pooled).squeeze(1)


def _items(store: DetectionStore, ids: list[str], standardizer: Standardizer):
    return [(standardizer.apply(store.bag(c)), store.label(c)) for c in ids]


@torch.no_grad()
def predict(model: MaxPoolMIL, items, device: str) -> np.ndarray:
    model.eval()
    return np.asarray([
        torch.sigmoid(model(torch.from_numpy(x).float().to(device))).item()
        for x, _ in items
    ])


def train_model(train_items, val_items, device: str, seed: int, epochs: int = cfg.MAXPOOL_EPOCHS):
    torch.manual_seed(seed)
    model = MaxPoolMIL().to(device)
    labels = np.asarray([y for _, y in train_items])
    weight = float((labels == 0).sum() / max(1, int((labels == 1).sum())))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    tensors = [(torch.from_numpy(x).float().to(device), float(y)) for x, y in train_items]
    best_auc, best_state, stale = -1.0, None, 0
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        for step, index in enumerate(rng.permutation(len(tensors)), 1):
            x, y = tensors[index]
            loss = loss_fn(model(x), torch.tensor([y], device=device)) / 16
            loss.backward()
            if step % 16 == 0 or step == len(tensors):
                optimizer.step()
                optimizer.zero_grad()
        val_score = predict(model, val_items, device)
        val_true = np.asarray([y for _, y in val_items])
        auc = float(roc_auc_score(val_true, val_score)) if len(np.unique(val_true)) > 1 else -1.0
        if auc > best_auc + 1e-5:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.MAXPOOL_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def _split_payload(seed: int) -> dict:
    path = cfg.split_json_path("A", seed)
    if not path.exists():
        raise FileNotFoundError(f"Missing nested-CV split: {path}. Run the csPCa ML stage first.")
    return json.loads(path.read_text())


def _metric_row(y_true: np.ndarray, y_score: np.ndarray, threshold: float = ZONE_THRESHOLD) -> dict:
    return compute_metrics(y_true, y_score, (y_score >= threshold).astype(int))


def internal_result_path(seed: int) -> Path:
    return cfg.results_dir("A", seed) / "maxpool_zone.xlsx"


def internal_ready(seed: int) -> bool:
    path = internal_result_path(seed)
    if not path.exists():
        return False
    try:
        rows = pd.read_excel(path, sheet_name="fold_metrics")
        pd.read_excel(path, sheet_name="mean_outer")
        pd.read_excel(path, sheet_name="pooled")
    except Exception:
        return False
    got = {int(r["outer_fold"]) for _, r in rows.iterrows()}
    return set(range(1, 5)).issubset(got)


def run_internal(seed: int, epochs: int = cfg.MAXPOOL_EPOCHS, force: bool = False) -> Path:
    out_path = internal_result_path(seed)
    if not force and internal_ready(seed):
        print(f"[MAXPOOL] internal zone result exists: {out_path}")
        return out_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = _split_payload(seed)
    fold_rows, pred_rows = [], []
    started = time.time()
    store = DetectionStore(cfg.PATCH_DETECTION_DIR, cfg.DATASET_XLSX)
    for outer in split["outer_splits"]:
        fold = int(outer["outer_fold"])
        test_ids = store.cases(outer["test_case_ids"])
        member_scores = []
        for inner in outer["inner_splits"]:
            train_ids = store.cases(inner["train_case_ids"])
            val_ids = store.cases(inner["val_case_ids"])
            if not train_ids or not val_ids or len({store.label(c) for c in train_ids}) < 2:
                continue
            standardizer = Standardizer().fit(store, train_ids)
            model, _ = train_model(
                _items(store, train_ids, standardizer),
                _items(store, val_ids, standardizer),
                device, seed * 100 + int(inner["inner_fold"]), epochs,
            )
            member_scores.append(predict(model, _items(store, test_ids, standardizer), device))
        if not member_scores:
            raise RuntimeError(f"No valid max-pool inner models for outer fold {fold}")
        score = np.mean(member_scores, axis=0)
        truth = np.asarray([store.label(c) for c in test_ids])
        pred = (score >= ZONE_THRESHOLD).astype(int)
        fold_rows.append({"task": "zone", "head": "maxpool", "input_streams": MAXPOOL_INPUT_NAME,
                          "outer_fold": fold, "n": len(test_ids),
                          **compute_metrics(truth, score, pred)})
        pred_rows.extend({"task": "zone", "head": "maxpool", "input_streams": MAXPOOL_INPUT_NAME,
                          "outer_fold": fold, cfg.ID_COL: c, "y_true": int(y),
                          "y_score": float(s), "y_pred": int(p)}
                         for c, y, s, p in zip(test_ids, truth, score, pred))
        print(f"[MAXPOOL] outer={fold} n={len(test_ids)} auc={fold_rows[-1]['auc']:.4f}")
    folds = pd.DataFrame(fold_rows)
    preds = pd.DataFrame(pred_rows)
    pooled_metrics = compute_metrics(
        preds["y_true"].to_numpy(int),
        preds["y_score"].to_numpy(float),
        preds["y_pred"].to_numpy(int),
    )
    mean_outer = pd.DataFrame([{
        "task": "zone",
        "head": "maxpool",
        "input_streams": MAXPOOL_INPUT_NAME,
        "n_outer_folds": len(folds),
        **{f"{c}_mean": float(folds[c].mean()) for c in pooled_metrics},
        **{f"{c}_std": float(folds[c].std(ddof=0)) for c in pooled_metrics},
    }])
    pooled = pd.DataFrame([{
        "task": "zone",
        "head": "maxpool",
        "input_streams": MAXPOOL_INPUT_NAME,
        "n": int(len(preds)),
        **pooled_metrics,
    }])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        cfg.collapse_mean_std_columns(cfg.round_metrics(mean_outer)).to_excel(writer, sheet_name="mean_outer", index=False)
        cfg.round_metrics(pooled).to_excel(writer, sheet_name="pooled", index=False)
        cfg.round_metrics(folds).to_excel(writer, sheet_name="fold_metrics", index=False)
        cfg.round_metrics(preds).to_excel(writer, sheet_name="predictions", index=False)
    print(f"[MAXPOOL] saved {out_path} in {time.time() - started:.1f}s")
    return out_path


def model_path(seed: int) -> Path:
    return cfg.final_model_dir("A", seed) / "maxpool_zone.joblib"


def train_deployable(seed: int, epochs: int = cfg.MAXPOOL_EPOCHS, force: bool = False) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = model_path(seed)
    if out.exists() and not force:
        print(f"[MAXPOOL TRAIN] model exists: {out}")
        return
    store = DetectionStore(cfg.PATCH_DETECTION_DIR, cfg.DATASET_XLSX)
    case_ids = store.cases()
    labels = np.asarray([store.label(c) for c in case_ids])
    splitter = StratifiedKFold(n_splits=cfg.MAXPOOL_ENSEMBLE_MEMBERS, shuffle=True, random_state=seed)
    members = []
    for member, (train_idx, val_idx) in enumerate(splitter.split(case_ids, labels)):
        train_ids = [case_ids[i] for i in train_idx]
        val_ids = [case_ids[i] for i in val_idx]
        standardizer = Standardizer().fit(store, train_ids)
        model, val_auc = train_model(_items(store, train_ids, standardizer),
                                     _items(store, val_ids, standardizer), device,
                                     seed * 100 + member, epochs)
        members.append({"standardizer": standardizer.state(),
                        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "val_auc": val_auc})
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"task": "zone", "head": "maxpool", "input_streams": list(cfg.MAXPOOL_FEATURE_SETS),
                 "seed": seed, "members": members}, out)
    print(f"[MAXPOOL TRAIN] saved {out}")


def predict_external(seed: int, detection_root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(detection_root or cfg.external_detection_dir())
    bundle_path = model_path(seed)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing max-pool model: {bundle_path}")
    bundle = joblib.load(bundle_path)
    store = DetectionStore(root, cfg.P158_DATASET_XLSX)
    case_ids = store.cases()
    member_scores = []
    for member in bundle["members"]:
        standardizer = Standardizer.from_state(member["standardizer"])
        model = MaxPoolMIL().to(device)
        model.load_state_dict(member["state_dict"])
        member_scores.append(predict(model, _items(store, case_ids, standardizer), device))
    score = np.mean(member_scores, axis=0)
    truth = np.asarray([store.label(c) for c in case_ids])
    pred = (score >= ZONE_THRESHOLD).astype(int)
    metrics = compute_metrics(truth, score, pred)
    metric_rows = [{"experiment": "A_Patch", "task": "zone", "head": "maxpool",
                    "input_streams": MAXPOOL_INPUT_NAME, "n": len(case_ids), **metrics}]
    pred_rows = [{"experiment": "A_Patch", "task": "zone", "head": "maxpool",
                  "input_streams": MAXPOOL_INPUT_NAME, cfg.ID_COL: c, "y_true": int(y),
                  "y_score": float(s), "y_pred": int(p)}
                 for c, y, s, p in zip(case_ids, truth, score, pred)]
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="R1-style max-pool zone branch from 04_detection_by_helper.")
    parser.add_argument("--stage", choices=["internal", "train", "test"], default="internal")
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_STATE)
    parser.add_argument("--epochs", type=int, default=cfg.MAXPOOL_EPOCHS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "internal":
        run_internal(args.seed, args.epochs, args.force)
    elif args.stage == "train":
        train_deployable(args.seed, args.epochs, args.force)
    else:
        metrics, predictions = predict_external(args.seed)
        print(metrics.to_string(index=False))
        print(f"[MAXPOOL TEST] predictions={len(predictions)}")


if __name__ == "__main__":
    main()
