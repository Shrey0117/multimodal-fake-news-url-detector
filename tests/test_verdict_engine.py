import unittest

from modules.verdict_engine import VerdictEngine


class VerdictEngineTest(unittest.TestCase):
    def test_comprehensive_score_uses_weighted_multimodal_risk(self):
        text_result = {
            "fake_probability": 0.8,
            "real_probability": 0.2,
            "token_count": 40,
            "truncated": False,
        }
        image_result = {
            "manipulation_score": 50.0,
            "risk_level": "MEDIUM",
        }

        result = VerdictEngine().combine(text_result, image_result)

        self.assertEqual(result["risk_score"], 71.0)
        self.assertEqual(result["comprehensive_score"], 29.0)
        self.assertEqual(result["score_breakdown"]["image_integrity_score"], 50.0)
        self.assertEqual(result["verdict"], "UNCERTAIN")


if __name__ == "__main__":
    unittest.main()
