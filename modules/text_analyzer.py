"""
modules/text_analyzer.py
------------------------
Text-based fake-news classification with a cached Hugging Face classifier
and a lightweight fill-mask fallback.

Why this module was rewritten:
  - The previous primary checkpoint collapsed toward "FAKE" whenever a
    headline was included.
  - The previous fallback model ID was invalid, so the backup path could
    never load.
  - The new primary classifier performs materially better on the bundled
    sample set when we classify the article body directly.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Pipeline,
    pipeline,
)

import config

logger = logging.getLogger(__name__)

_FAKE_TOKENS = {
    "fake",
    "false",
    "fabricated",
    "misleading",
    "misinformation",
    "inaccurate",
    "untrue",
    "bogus",
    "fraudulent",
    "deceptive",
}
_REAL_TOKENS = {
    "real",
    "true",
    "genuine",
    "authentic",
    "accurate",
    "legitimate",
    "factual",
    "verified",
    "credible",
    "reliable",
}


class TextAnalyzer:
    """
    Wraps a binary sequence classifier with a fill-mask fallback.

    Public interface:
        analyze(text: str, title: str = "") -> Dict
    """

    def __init__(self) -> None:
        self._mode: str = "none"
        self._tokenizer = None
        self._model = None
        self._fallback_pipe: Optional[Pipeline] = None
        self._active_model_name: str = "none"
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load()

    def _resolve_model_source(self, model_name: str) -> Tuple[str, bool]:
        """
        Prefer a cached local snapshot when available.

        Loading from the cache path avoids repeated hub lookups on offline runs.
        Returns:
            (source, local_files_only)
        """
        try:
            local_path = snapshot_download(model_name, local_files_only=True)
            logger.info("Using cached snapshot for %s from %s", model_name, local_path)
            return local_path, True
        except Exception:
            logger.info("No cached snapshot found for %s; loading from Hub", model_name)
            return model_name, False

    def _load(self) -> None:
        """Try primary classifier first, then the fill-mask fallback."""
        try:
            source, local_only = self._resolve_model_source(config.MODEL_NAME)
            logger.info("Loading primary text classifier: %s", config.MODEL_NAME)
            self._tokenizer = AutoTokenizer.from_pretrained(
                source,
                local_files_only=local_only,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                source,
                local_files_only=local_only,
            )
            self._model.to(self._device)
            self._model.eval()
            self._mode = "classifier"
            self._active_model_name = config.MODEL_NAME
            logger.info("Primary classifier loaded successfully.")
            return
        except Exception as exc:
            logger.warning("Primary classifier failed: %s", exc)

        try:
            source, local_only = self._resolve_model_source(config.FALLBACK_MODEL_NAME)
            logger.info("Loading fallback fill-mask model: %s", config.FALLBACK_MODEL_NAME)
            self._fallback_pipe = pipeline(
                "fill-mask",
                model=source,
                tokenizer=source,
                top_k=50,
                device=0 if self._device.type == "cuda" else -1,
            )
            self._mode = "fill-mask"
            self._active_model_name = config.FALLBACK_MODEL_NAME
            logger.info("Fallback model loaded successfully.")
            return
        except Exception as exc:
            logger.warning("Fallback model failed: %s", exc)

        raise RuntimeError(
            "Both the primary article classifier and the fallback fill-mask model "
            "failed to load. Check your internet connection for the first download "
            "or ensure the Hugging Face cache is intact."
        )

    def _build_primary_input(
        self,
        title: str,
        content: str,
    ) -> Tuple[str, int, bool, str]:
        """
        Build the primary classifier input.

        The primary classifier is most stable when it reads the cleaned article
        body directly. Headlines are handled by the UI/preprocessor layer but
        are not concatenated into the classifier input.
        """
        body = content.strip()
        model_input = body
        input_strategy = "body_only"

        raw_ids = self._tokenizer(
            model_input,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        truncated = len(raw_ids) > config.MAX_TOKEN_LENGTH

        tokenized = self._tokenizer(
            model_input,
            max_length=config.MAX_TOKEN_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        token_count = int(tokenized["input_ids"].shape[1])

        return model_input, token_count, truncated, input_strategy

    def _infer_classifier(
        self,
        title: str,
        content: str,
    ) -> Tuple[float, float, int, bool, str, str]:
        """
        Run the primary classifier.

        Returns:
            fake_prob, real_prob, token_count, truncated, input_strategy, input_preview
        """
        model_input, token_count, truncated, input_strategy = self._build_primary_input(
            title,
            content,
        )

        encoded = self._tokenizer(
            model_input,
            max_length=config.MAX_TOKEN_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.no_grad():
            output = self._model(**encoded)

        probs = torch.nn.functional.softmax(output.logits, dim=1)[0]
        fake_prob = probs[config.PRIMARY_MODEL_FAKE_LABEL_INDEX].item()
        real_prob = probs[config.PRIMARY_MODEL_REAL_LABEL_INDEX].item()

        preview = model_input[:240]
        if len(model_input) > 240:
            preview += "..."

        logger.debug(
            "Classifier inference - fake=%.4f real=%.4f tokens=%d strategy=%s",
            fake_prob,
            real_prob,
            token_count,
            input_strategy,
        )

        return fake_prob, real_prob, token_count, truncated, input_strategy, preview

    def _infer_fill_mask(
        self,
        title: str,
        content: str,
    ) -> Tuple[float, float, int, bool, str, str]:
        """
        Use fill-mask prompting as a zero-shot binary classifier.
        """
        tok = self._fallback_pipe.tokenizer

        body = content.strip()
        article_text = body
        input_strategy = "body_only"

        prompt_prefix = f"This article is {tok.mask_token}. "
        prompt = f"{prompt_prefix}{article_text}"

        raw_ids = tok(prompt, add_special_tokens=True, truncation=False)["input_ids"]
        truncated = len(raw_ids) > config.MAX_TOKEN_LENGTH
        prompt_ids = tok(
            prompt,
            add_special_tokens=True,
            max_length=config.MAX_TOKEN_LENGTH,
            truncation=True,
        )["input_ids"]
        token_count = len(prompt_ids)

        trimmed_prompt = tok.decode(prompt_ids, skip_special_tokens=False)
        predictions = self._fallback_pipe(trimmed_prompt)

        fake_mass = 0.0
        real_mass = 0.0
        for pred in predictions:
            word = pred["token_str"].strip().lower()
            if word in _FAKE_TOKENS:
                fake_mass += pred["score"]
            elif word in _REAL_TOKENS:
                real_mass += pred["score"]

        total = fake_mass + real_mass
        if total < 1e-9:
            fake_prob, real_prob = 0.5, 0.5
        else:
            fake_prob = fake_mass / total
            real_prob = real_mass / total

        preview = article_text[:240]
        if len(article_text) > 240:
            preview += "..."

        logger.debug(
            "Fill-mask inference - fake=%.4f real=%.4f tokens=%d strategy=%s",
            fake_prob,
            real_prob,
            token_count,
            input_strategy,
        )

        return fake_prob, real_prob, token_count, truncated, input_strategy, preview

    @property
    def active_model(self) -> str:
        return self._active_model_name

    @property
    def mode(self) -> str:
        return self._mode

    def analyze(self, text: str, title: str = "") -> Dict:
        """
        Classify an article and return a structured result.
        """
        if not text or not text.strip():
            raise ValueError("Cannot analyse empty text.")

        try:
            if self._mode == "classifier":
                (
                    fake_prob,
                    real_prob,
                    token_count,
                    truncated,
                    input_strategy,
                    input_preview,
                ) = self._infer_classifier(title, text)
            else:
                (
                    fake_prob,
                    real_prob,
                    token_count,
                    truncated,
                    input_strategy,
                    input_preview,
                ) = self._infer_fill_mask(title, text)
        except Exception as exc:
            logger.error("Inference failed: %s", exc)
            raise RuntimeError(f"Model inference error: {exc}") from exc

        dominant_label = "FAKE" if fake_prob >= real_prob else "REAL"
        confidence = max(fake_prob, real_prob)

        return {
            "label": dominant_label,
            "confidence": confidence,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "token_count": token_count,
            "truncated": truncated,
            "model_used": self._active_model_name,
            "mode": self._mode,
            "input_strategy": input_strategy,
            "input_preview": input_preview,
        }
