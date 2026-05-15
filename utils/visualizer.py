"""
utils/visualizer.py
-------------------
Matplotlib helpers for the Fake News Detector.
"""

import logging
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config

logger = logging.getLogger(__name__)


def plot_confidence_gauge(fake_prob: float, real_prob: float) -> plt.Figure:
    """Render a horizontal stacked bar for fake vs real probability."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    fake_pct = fake_prob * 100
    real_pct = real_prob * 100

    ax.barh(0, fake_pct, color="#e74c3c", height=0.55, label=f"FAKE {fake_pct:.1f}%")
    ax.barh(
        0,
        real_pct,
        left=fake_pct,
        color="#2ecc71",
        height=0.55,
        label=f"REAL {real_pct:.1f}%",
    )

    real_threshold = config.REAL_THRESHOLD * 100
    fake_threshold = config.FAKE_THRESHOLD * 100
    ax.axvline(
        x=real_threshold,
        color="#f39c12",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
    )
    ax.axvline(
        x=fake_threshold,
        color="#e74c3c",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
    )
    ax.text(
        real_threshold,
        0.45,
        f"{real_threshold:.0f}%",
        color="#f39c12",
        fontsize=7,
        ha="center",
        va="bottom",
    )
    ax.text(
        fake_threshold,
        0.45,
        f"{fake_threshold:.0f}%",
        color="#e74c3c",
        fontsize=7,
        ha="center",
        va="bottom",
    )

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Confidence (%)", color="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.65),
        ncol=2,
        framealpha=0,
        labelcolor="white",
        fontsize=9,
    )
    ax.set_title("Text Prediction Confidence", color="white", fontsize=10, pad=6)
    fig.tight_layout()
    return fig


def plot_ela_comparison(original_img: Image.Image, ela_img: Image.Image) -> plt.Figure:
    """Render the original image beside its Error Level Analysis heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("#0e1117")

    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", color="white", fontsize=11)
    axes[0].axis("off")

    ela_array = np.array(ela_img.convert("L"))
    im = axes[1].imshow(ela_array, cmap="hot", vmin=0, vmax=255)
    axes[1].set_title(
        "ELA Heatmap\n(brighter = higher error level)",
        color="white",
        fontsize=11,
    )
    axes[1].axis("off")

    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)
    cbar.set_label("Error Level", color="white", fontsize=8)

    fig.suptitle(
        "Error Level Analysis - Image Authenticity Check",
        color="white",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_contributing_factors(factors: List[str]) -> plt.Figure:
    """Render a labelled factor chart for the final verdict."""
    if not factors:
        factors = ["No contributing factors available."]

    fig, ax = plt.subplots(figsize=(7, max(1.5, len(factors) * 0.7 + 0.5)))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.axis("off")

    ax.set_title("Contributing Factors", color="white", fontsize=11, pad=8)

    for i, factor in enumerate(factors):
        y_pos = 1.0 - (i / max(len(factors), 1)) * 0.85
        ax.text(
            0.02,
            y_pos,
            f"- {factor}",
            color="#ecf0f1",
            fontsize=9,
            transform=ax.transAxes,
            va="top",
            wrap=True,
        )

    fig.tight_layout()
    return fig
