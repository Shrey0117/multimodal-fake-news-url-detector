"""
utils/preprocessor.py
---------------------
Text preprocessing and validation utilities for the Fake News Detection System.
"""

import logging
import re
from typing import Tuple

import config

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Cleans and validates raw article text.
    """

    def clean(self, text: str) -> str:
        """
        Remove noisy markup and normalize whitespace/punctuation.
        """
        text = re.sub(r"http\S+|www\.\S+|ftp\.\S+", "", text)
        text = re.sub(r"<[^>]+>", "", text)

        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2013", "-").replace("\u2014", "-")

        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def clean_title(self, title: str) -> str:
        """
        Light-touch cleaning for the headline field.
        """
        return self.clean(title)

    def is_valid(self, text: str) -> Tuple[bool, str]:
        """
        Validate article body text before it reaches the classifier.
        """
        if not text or not text.strip():
            return False, "Input is empty. Please paste an article."

        word_count = len(text.split())
        if word_count < config.MIN_WORDS:
            return (
                False,
                f"Text too short ({word_count} words). "
                f"Provide at least {config.MIN_WORDS} words for a reliable prediction.",
            )

        if len(text) > config.MAX_CHARS:
            return (
                False,
                f"Text too long ({len(text):,} characters). "
                f"Limit input to {config.MAX_CHARS:,} characters.",
            )

        return True, ""

    def split_title_body(self, raw_text: str) -> Tuple[str, str]:
        """
        Try to split pasted text into (headline, body).

        Heuristics:
          1. Short first line followed by more content -> treat first line as headline.
          2. Short first sentence followed by enough body text -> treat first sentence as headline.
          3. Otherwise return the whole text as body.
        """
        stripped = raw_text.strip()
        lines = stripped.split("\n")

        if len(lines) >= 2:
            first_line = lines[0].strip()
            if first_line and len(first_line.split()) <= 20:
                rest = "\n".join(lines[1:]).strip()
                if rest:
                    return first_line, rest

        sentences = re.split(r"(?<=[.!?])\s+", stripped)
        if len(sentences) >= 2:
            first_sentence = sentences[0].strip()
            if 3 <= len(first_sentence.split()) <= 20:
                rest = " ".join(sentences[1:]).strip()
                if len(rest.split()) >= config.MIN_WORDS:
                    return first_sentence, rest

        return "", stripped

    def get_stats(self, text: str) -> dict:
        """
        Return lightweight statistics for the UI.
        """
        words = text.split()
        sentences = [chunk for chunk in re.split(r"[.!?]+", text) if chunk.strip()]

        return {
            "word_count": len(words),
            "char_count": len(text),
            "sentence_count": len(sentences),
            "avg_word_length": (
                round(sum(len(word) for word in words) / len(words), 2) if words else 0
            ),
        }
