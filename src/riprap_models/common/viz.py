"""Plot helpers shared across model report generators.

Kept minimal; matplotlib is an optional dep on the model extras. Each
function returns the output path so callers can embed it in a report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def side_by_side_segmentation(
    image: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    out_path: Path,
    title: str = "",
) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    if image.ndim == 3 and image.shape[-1] >= 3:
        # Show RGB-ish composite of the first three bands, normalized.
        rgb = image[..., :3]
        rgb = (rgb - rgb.min()) / (rgb.ptp() + 1e-9)
        axes[0].imshow(rgb)
    else:
        axes[0].imshow(image, cmap="gray")
    axes[0].set_title("input")
    axes[1].imshow(truth, cmap="tab10", vmin=0, vmax=10)
    axes[1].set_title("ground truth")
    axes[2].imshow(pred, cmap="tab10", vmin=0, vmax=10)
    axes[2].set_title("prediction")
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def time_series_forecast(
    history: np.ndarray,
    target: np.ndarray,
    forecast: np.ndarray,
    out_path: Path,
    title: str = "",
) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    h = np.arange(history.size)
    f = np.arange(history.size, history.size + forecast.size)
    ax.plot(h, history, label="history (observed surge residual, m)")
    ax.plot(f, target, label="actual", color="tab:green")
    ax.plot(f, forecast, label="forecast", color="tab:red", linestyle="--")
    ax.axvline(history.size, color="grey", alpha=0.3)
    ax.set_xlabel("6-min step")
    ax.set_ylabel("residual (m)")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
