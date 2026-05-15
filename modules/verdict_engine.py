"""
modules/verdict_engine.py
-------------------------
Combines text and optional image analysis into a scored verdict.
"""

import logging
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)


class VerdictEngine:
    """
    Combines multimodal analysis results into a unified verdict.

    The primary comprehensive score is a 0-100 credibility score, where higher
    means the article package looks more reliable based on the available
    modalities.
    """

    _COLOURS = {
        "LIKELY REAL": "#2ecc71",
        "UNCERTAIN": "#f39c12",
        "LIKELY FAKE": "#e74c3c",
    }

    def combine(
        self,
        text_result: Dict,
        image_result: Optional[Dict] = None,
    ) -> Dict:
        """
        Produce a final verdict from one or both modality results.

        Returns a backward-compatible dict with additional score fields:
            comprehensive_score: 0-100 credibility score
            risk_score:          0-100 blended fake/misinformation risk
            score_breakdown:     per-modality scores used in the decision
        """
        fake_prob = text_result["fake_probability"]
        factors: List[str] = []

        factors.append(
            f"Text model: {fake_prob * 100:.1f}% fake probability "
            f"({text_result['token_count']} tokens analysed"
            + (", input truncated to 512 tokens" if text_result.get("truncated") else "")
            + ")"
        )

        blended_fake_prob = fake_prob
        score_breakdown = {
            "text_fake_risk": round(fake_prob * 100, 1),
            "text_real_probability": round(text_result["real_probability"] * 100, 1),
            "image_manipulation_risk": None,
            "image_integrity_score": None,
            "final_fake_risk": None,
            "final_credibility_score": None,
        }

        if image_result is not None:
            image_risk = image_result["manipulation_score"] / 100.0
            blended_fake_prob = (
                config.TEXT_WEIGHT * fake_prob
                + config.IMAGE_WEIGHT * image_risk
            )
            score_breakdown["image_manipulation_risk"] = round(
                image_result["manipulation_score"], 1
            )
            score_breakdown["image_integrity_score"] = round(
                100 - image_result["manipulation_score"], 1
            )
            factors.append(
                f"Image ELA: manipulation score {image_result['manipulation_score']:.1f}/100 "
                f"(risk level: {image_result['risk_level']})"
            )
            factors.append(
                f"Fusion weights applied: text {config.TEXT_WEIGHT * 100:.0f}%, "
                f"image {config.IMAGE_WEIGHT * 100:.0f}%"
            )
            logger.debug(
                "Fusion: text_fake=%.3f image_risk=%.3f blended=%.3f",
                fake_prob,
                image_risk,
                blended_fake_prob,
            )

        risk_score = round(blended_fake_prob * 100, 1)
        comprehensive_score = round((1 - blended_fake_prob) * 100, 1)
        score_breakdown["final_fake_risk"] = risk_score
        score_breakdown["final_credibility_score"] = comprehensive_score

        if blended_fake_prob >= config.FAKE_THRESHOLD:
            verdict = "LIKELY FAKE"
            score_band = "High misinformation risk"
            explanation = (
                f"The combined analysis gives this article package a {risk_score:.1f}/100 "
                "misinformation risk score. Its language patterns are statistically "
                "similar to known misinformation"
                + (
                    ", and the associated image shows notable manipulation signals. "
                    if image_result and image_result["risk_level"] != "LOW"
                    else ". "
                )
                + "Cross-check the claims with authoritative sources before sharing."
            )
        elif blended_fake_prob <= config.REAL_THRESHOLD:
            verdict = "LIKELY REAL"
            score_band = "Higher credibility"
            explanation = (
                f"The combined analysis gives this article package a {comprehensive_score:.1f}/100 "
                "credibility score. Its language patterns align more closely with factual "
                "reporting"
                + (
                    ", and the image does not show strong manipulation signals. "
                    if image_result and image_result["risk_level"] == "LOW"
                    else ". "
                )
                + "Important claims should still be verified with primary sources."
            )
        else:
            verdict = "UNCERTAIN"
            score_band = "Needs verification"
            explanation = (
                f"The final fake-risk score is {risk_score:.1f}/100, which sits between "
                f"the real threshold ({config.REAL_THRESHOLD * 100:.0f}) and fake "
                f"threshold ({config.FAKE_THRESHOLD * 100:.0f}). The system cannot "
                "classify this confidently, so manual fact-checking is recommended."
            )

        factors.append(f"Final blended fake-risk score: {risk_score:.1f}/100")
        factors.append(f"Comprehensive credibility score: {comprehensive_score:.1f}/100")

        return {
            "verdict": verdict,
            "confidence": blended_fake_prob,
            "fake_probability": blended_fake_prob,
            "risk_score": risk_score,
            "comprehensive_score": comprehensive_score,
            "score_band": score_band,
            "score_breakdown": score_breakdown,
            "verdict_color": self._COLOURS[verdict],
            "explanation": explanation,
            "contributing_factors": factors,
        }
