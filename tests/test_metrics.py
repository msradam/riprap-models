"""Hand-computed expected values for the metric definitions."""

import math

import numpy as np
import pytest

from riprap_models.common.metrics import (
    confusion_matrix,
    iou_from_confusion,
    persistence_forecast,
    regression_score,
    segmentation_score,
)


def test_confusion_matrix_basic():
    # 2 classes, 4 pixels: [0,0,1,1] truth, [0,1,1,1] pred.
    # cm[0,0]=1 (TP class 0), cm[0,1]=1 (FN class 0 / FP class 1),
    # cm[1,0]=0, cm[1,1]=2 (TP class 1).
    pred = np.array([0, 1, 1, 1])
    target = np.array([0, 0, 1, 1])
    cm = confusion_matrix(pred, target, num_classes=2)
    assert cm.tolist() == [[1, 1], [0, 2]]


def test_iou_from_confusion_handcomputed():
    # cm above:
    #   class 0: TP=1, FP=0, FN=1 → IoU = 1/(1+0+1) = 0.5
    #   class 1: TP=2, FP=1, FN=0 → IoU = 2/(2+1+0) = 0.6666...
    cm = np.array([[1, 1], [0, 2]])
    iou = iou_from_confusion(cm)
    assert iou[0] == pytest.approx(0.5)
    assert iou[1] == pytest.approx(2 / 3)


def test_segmentation_score_macro_vs_micro():
    pred = np.array([0, 1, 1, 1])
    target = np.array([0, 0, 1, 1])
    s = segmentation_score(pred, target, num_classes=2)
    # macro = mean(0.5, 2/3)
    assert s.miou_macro == pytest.approx((0.5 + 2 / 3) / 2)
    # micro = sum-TP / total = (1+2)/4 = 0.75
    assert s.miou_micro == pytest.approx(0.75)
    assert s.pixel_accuracy == pytest.approx(0.75)
    assert s.n_pixels == 4


def test_segmentation_score_perfect():
    pred = np.array([0, 1, 2, 1, 0])
    target = pred.copy()
    s = segmentation_score(pred, target, num_classes=3)
    assert s.miou_macro == pytest.approx(1.0)
    assert s.miou_micro == pytest.approx(1.0)


def test_segmentation_ignore_index_excludes_pixels():
    pred = np.array([0, 0, 1, 1])
    target = np.array([0, 255, 1, 1])  # 255 is the standard ignore index
    s = segmentation_score(pred, target, num_classes=2, ignore_index=255)
    # Only 3 pixels participate, all correct.
    assert s.n_pixels == 3
    assert s.miou_macro == pytest.approx(1.0)


def test_segmentation_class_absent_yields_nan():
    # Class 1 never appears in target, never appears in pred → IoU undefined.
    pred = np.array([0, 0, 0])
    target = np.array([0, 0, 0])
    s = segmentation_score(pred, target, num_classes=2)
    assert math.isnan(s.iou_per_class[1])
    # Macro mean ignores NaN classes, so macro = IoU(class 0) = 1.0.
    assert s.miou_macro == pytest.approx(1.0)


def test_regression_score_handcomputed():
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.5, 2.5, 2.0])
    # diffs: -0.5, -0.5, 1.0 → MAE = (0.5+0.5+1.0)/3 = 2/3
    # squared: 0.25, 0.25, 1.0 → mean = 1.5/3 = 0.5 → RMSE = sqrt(0.5)
    r = regression_score(pred, target)
    assert r.mae == pytest.approx(2 / 3)
    assert r.rmse == pytest.approx(math.sqrt(0.5))
    assert r.n == 3


def test_persistence_forecast_repeats_last():
    h = np.array([1.0, 2.0, 3.0, 4.0])
    f = persistence_forecast(h, horizon=5)
    assert f.shape == (5,)
    assert (f == 4.0).all()


def test_persistence_forecast_empty_raises():
    with pytest.raises(ValueError):
        persistence_forecast(np.array([]), horizon=3)


def test_confusion_matrix_shape_mismatch_raises():
    with pytest.raises(ValueError):
        confusion_matrix(np.zeros(4), np.zeros(5), num_classes=2)
