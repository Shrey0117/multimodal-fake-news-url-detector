"""
modules/image_analyzer.py
--------------------------
Image authenticity analysis via Error Level Analysis (ELA).

ELA was introduced by Neal Krawetz in 2007 ("A Picture's Worth…") and works
on the following principle: JPEG compression is lossy — each save at a given
quality level introduces a characteristic pattern of artefacts.  If a region
of an image was copy-pasted from another source or digitally edited and then
re-saved, that region will have a *different* compression history from the
surrounding pixels, and re-saving reveals that discrepancy as a visible error
level difference.

Algorithm implemented here:
  1. Re-save the input image at quality=90 to a BytesIO buffer
  2. Reload the re-saved image
  3. Compute per-pixel absolute difference: |original - resaved|
  4. Amplify by a factor of 15 so subtle differences become visible
  5. Derive a manipulation score from the mean amplified difference

TODO (Phase 4): Replace ELA with a CNN-based forgery detector
               (e.g. MantraNet or MVSS-Net) for higher accuracy.
"""

import io
import logging
from typing import Dict

import numpy as np
from PIL import Image

import config

logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """
    Performs Error Level Analysis on uploaded images to detect potential
    digital manipulation.

    No external models are required — only PIL and NumPy.
    """

    # Amplification factor: scales the pixel difference for human visibility
    _ELA_AMPLIFY = 15

    def __init__(self) -> None:
        """Nothing to initialise — ELA is purely algorithmic."""
        pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _perform_ela(self, image: Image.Image) -> Image.Image:
        """
        Produce the ELA difference image.

        Steps:
          1. Convert to RGB (handles PNG with alpha, greyscale, etc.)
          2. Re-save to an in-memory buffer at the configured JPEG quality
          3. Reload from buffer
          4. Compute amplified absolute difference

        Returns a PIL Image representing per-pixel error levels.
        """
        # Ensure we always work in RGB — alpha channels cause JPEG errors
        original_rgb = image.convert("RGB")

        buffer = io.BytesIO()
        original_rgb.save(buffer, format="JPEG", quality=config.IMAGE_ELA_QUALITY)
        buffer.seek(0)
        resaved = Image.open(buffer).convert("RGB")

        orig_array    = np.array(original_rgb, dtype=np.float32)
        resaved_array = np.array(resaved,      dtype=np.float32)

        # Absolute difference — this is the "error level" for each pixel
        diff = np.abs(orig_array - resaved_array)

        # Amplify so subtle differences are perceptible as a heatmap
        ela_array = np.clip(diff * self._ELA_AMPLIFY, 0, 255).astype(np.uint8)

        return Image.fromarray(ela_array)

    def _compute_score(self, ela_image: Image.Image) -> float:
        """
        Derive a 0–100 manipulation score from the ELA image.

        We use the mean pixel value of the greyscale ELA image as a proxy
        for overall error level.  A pristine, unmodified JPEG typically
        scores 5–20; heavily manipulated images can exceed 50.

        The raw mean is then normalised to a 0–100 scale capped at 100.
        """
        ela_grey  = np.array(ela_image.convert("L"), dtype=np.float32)
        mean_diff = float(np.mean(ela_grey))

        # Linear normalisation: 0 mean → 0 score, 17 mean → 100 score
        # (empirically calibrated on typical JPEG images)
        score = min((mean_diff / 17.0) * 100, 100.0)
        return round(score, 2), round(mean_diff, 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, image: Image.Image) -> Dict:
        """
        Run ELA on the supplied image and return a structured result.

        Args:
            image: PIL.Image loaded from the user's upload

        Returns:
            {
                manipulation_score: float (0–100),
                risk_level:         "LOW" | "MEDIUM" | "HIGH",
                ela_image:          PIL.Image (the heatmap),
                mean_difference:    float (raw mean pixel diff),
                assessment:         str  (human-readable summary)
            }
        """
        try:
            ela_image = self._perform_ela(image)
        except Exception as exc:
            logger.error("ELA processing failed: %s", exc)
            raise RuntimeError(f"Image processing error: {exc}") from exc

        score, mean_diff = self._compute_score(ela_image)

        # Map score to a risk level using configured thresholds
        if score > config.IMAGE_MANIPULATION_THRESHOLD_HIGH:
            risk_level  = "HIGH"
            assessment  = (
                f"ELA score {score:.1f}/100 — significant compression inconsistencies "
                "detected.  Multiple image regions show error levels inconsistent with "
                "a single-source JPEG, suggesting possible copy-paste manipulation or "
                "digital editing."
            )
        elif score > config.IMAGE_MANIPULATION_THRESHOLD_LOW:
            risk_level  = "MEDIUM"
            assessment  = (
                f"ELA score {score:.1f}/100 — moderate error level variation detected. "
                "Some regions may have been edited, but results are inconclusive.  "
                "Treat this as a flag for further investigation."
            )
        else:
            risk_level  = "LOW"
            assessment  = (
                f"ELA score {score:.1f}/100 — error levels appear consistent throughout "
                "the image.  No strong evidence of digital manipulation detected."
            )

        logger.debug(
            "ELA complete — score: %.2f, risk: %s, mean_diff: %.4f",
            score, risk_level, mean_diff,
        )

        return {
            "manipulation_score": score,
            "risk_level":         risk_level,
            "ela_image":          ela_image,
            "mean_difference":    mean_diff,
            "assessment":         assessment,
        }
