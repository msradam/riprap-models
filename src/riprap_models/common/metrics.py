"""Segmentation + regression metrics.

  IoU per class:  TP / (TP + FP + FN); ``ignore_index`` excluded entirely.
  mIoU:           ``macro`` = mean of per-class IoU; ``micro`` = sum-TP / (sum-TP + sum-FP + sum-FN).
  MAE / RMSE:     On surge residual (m). Persistence = last observed value held flat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentationScore:
    iou_per_class: dict[int, float]
    miou_macro: float
    miou_micro: float
    pixel_accuracy: float
    n_pixels: int


def confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int | None = None,
) -> np.ndarray:
    """Return an (num_classes, num_classes) integer confusion matrix.

    Rows are ground-truth class, columns are predicted class.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {pred.shape} != target {target.shape}")
    p = pred.reshape(-1).astype(np.int64)
    t = target.reshape(-1).astype(np.int64)
    if ignore_index is not None:
        keep = t != ignore_index
        p = p[keep]
        t = t[keep]
    # Clip predicted values that fall outside the label range. A model that
    # predicts class 5 against a 4-class task would otherwise blow the
    # bincount; we clamp and let mIoU reflect the mistake.
    p = np.clip(p, 0, num_classes - 1)
    t = np.clip(t, 0, num_classes - 1)
    idx = t * num_classes + p
    binc = np.bincount(idx, minlength=num_classes * num_classes)
    return binc.reshape(num_classes, num_classes)


def iou_from_confusion(cm: np.ndarray) -> dict[int, float]:
    iou: dict[int, float] = {}
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        denom = tp + fp + fn
        iou[c] = float(tp) / float(denom) if denom > 0 else float("nan")
    return iou


def segmentation_score(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int | None = None,
) -> SegmentationScore:
    cm = confusion_matrix(pred, target, num_classes, ignore_index)
    per_class = iou_from_confusion(cm)
    macro_vals = [v for v in per_class.values() if not np.isnan(v)]
    miou_macro = float(np.mean(macro_vals)) if macro_vals else float("nan")
    tp = float(np.trace(cm))
    fp_fn = float(cm.sum()) - tp  # for micro: TP / (TP + FP + FN) summed = trace / total
    miou_micro = tp / (tp + fp_fn) if (tp + fp_fn) > 0 else float("nan")
    pix_acc = tp / float(cm.sum()) if cm.sum() > 0 else float("nan")
    n_pix = int(cm.sum())
    return SegmentationScore(
        iou_per_class=per_class,
        miou_macro=miou_macro,
        miou_micro=miou_micro,
        pixel_accuracy=pix_acc,
        n_pixels=n_pix,
    )


# ---------- Regression -------------------------------------------------------


@dataclass
class RegressionScore:
    mae: float
    rmse: float
    n: int


def regression_score(pred: np.ndarray, target: np.ndarray) -> RegressionScore:
    if pred.shape != target.shape:
        raise ValueError(f"pred {pred.shape} != target {target.shape}")
    diff = pred.astype(np.float64) - target.astype(np.float64)
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    return RegressionScore(mae=mae, rmse=rmse, n=int(pred.size))


def persistence_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """Naive persistence: repeat the last observed value ``horizon`` times."""
    if history.size == 0:
        raise ValueError("history is empty")
    return np.full((horizon,), float(history[-1]), dtype=np.float64)
