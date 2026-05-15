"""
config.py
---------
Central configuration for the Fake News Detection System.
"""

# ---------------------------------------------------------------------------
# Text model configuration
# ---------------------------------------------------------------------------

# Primary article-level classifier. This checkpoint performed better than the
# original model during local validation on the bundled demo set.
MODEL_NAME = "jy46604790/Fake-News-Bert-Detect"

# Binary label layout for the primary classifier.
PRIMARY_MODEL_FAKE_LABEL_INDEX = 0
PRIMARY_MODEL_REAL_LABEL_INDEX = 1

# Fallback: valid DistilRoBERTa fill-mask checkpoint.
FALLBACK_MODEL_NAME = "distilroberta-base"

# ---------------------------------------------------------------------------
# Tokenizer / input limits
# ---------------------------------------------------------------------------

# RoBERTa-family models have a 512-token positional limit.
MAX_TOKEN_LENGTH = 512

# Minimum words required in the article body (enforced in preprocessor)
MIN_WORDS = 20

# Maximum raw characters accepted in the UI. URL extraction often returns full
# articles, so this version allows longer text before the model applies its
# 512-token transformer limit.
MAX_CHARS = 20_000

# ---------------------------------------------------------------------------
# Verdict thresholds
# A conservative fake threshold reduces false-positive "fake" verdicts.
# ---------------------------------------------------------------------------
FAKE_THRESHOLD = 0.95
REAL_THRESHOLD = 0.40

# ---------------------------------------------------------------------------
# Image / ELA configuration
# ---------------------------------------------------------------------------
IMAGE_ELA_QUALITY = 90
IMAGE_MANIPULATION_THRESHOLD_LOW = 35
IMAGE_MANIPULATION_THRESHOLD_HIGH = 60

# ---------------------------------------------------------------------------
# Fusion weights (must sum to 1.0)
# ---------------------------------------------------------------------------
TEXT_WEIGHT = 0.7
IMAGE_WEIGHT = 0.3

# ---------------------------------------------------------------------------
# UI strings
# ---------------------------------------------------------------------------
APP_TITLE = "Fake News Detector"
APP_SUBTITLE = "Single-submission article scoring with text classification and image forensics"

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
SAMPLE_ARTICLES = "data/sample_articles.json"
EDA_OUTPUT_DIR = "data/eda_outputs"

# ---------------------------------------------------------------------------
# Article URL extraction
# ---------------------------------------------------------------------------
URL_FETCH_TIMEOUT_SECONDS = 12
URL_HTML_MAX_BYTES = 4_000_000
URL_IMAGE_MAX_BYTES = 8_000_000
URL_IMAGE_CANDIDATE_LIMIT = 5
URL_FETCH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
