"""
modules/article_parser.py
-------------------------
Turns one user article submission into separate text and image inputs.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError

from utils.preprocessor import TextPreprocessor


@dataclass
class ParsedArticle:
    """Normalized article content ready for modality-specific analyzers."""

    headline: str
    body: str
    image: Optional[Image.Image]
    image_name: str
    image_size: Optional[Tuple[int, int]]
    stats: Dict[str, float]
    warnings: List[str]

    @property
    def has_image(self) -> bool:
        return self.image is not None


class ArticleParser:
    """
    Parses a single multimedia article submission.

    The article text is cleaned and optionally split into headline/body, while
    the uploaded visual is decoded into a PIL image for image forensics.
    """

    def __init__(self, preprocessor: TextPreprocessor) -> None:
        self.preprocessor = preprocessor

    def parse(
        self,
        raw_text: str,
        headline: str = "",
        image_file=None,
        auto_detect_headline: bool = True,
    ) -> ParsedArticle:
        parsed_headline, body = self._parse_text(
            raw_text=raw_text,
            headline=headline,
            auto_detect_headline=auto_detect_headline,
        )

        valid, reason = self.preprocessor.is_valid(body)
        if not valid:
            raise ValueError(reason)

        image, image_name, image_size, warnings = self._parse_image(image_file)

        return ParsedArticle(
            headline=parsed_headline,
            body=body,
            image=image,
            image_name=image_name,
            image_size=image_size,
            stats=self.preprocessor.get_stats(body),
            warnings=warnings,
        )

    def _parse_text(
        self,
        raw_text: str,
        headline: str,
        auto_detect_headline: bool,
    ) -> Tuple[str, str]:
        clean_headline = self.preprocessor.clean_title(headline or "")
        body_source = (raw_text or "").strip()

        if auto_detect_headline and not clean_headline:
            detected_headline, detected_body = self.preprocessor.split_title_body(
                body_source
            )
            if detected_headline:
                clean_headline = self.preprocessor.clean_title(detected_headline)
                body_source = detected_body

        clean_body = self.preprocessor.clean(body_source)
        return clean_headline, clean_body

    def _parse_image(self, image_file) -> Tuple[Optional[Image.Image], str, Optional[Tuple[int, int]], List[str]]:
        if image_file is None:
            return (
                None,
                "",
                None,
                ["No article image was included, so the final score is text-only."],
            )

        try:
            image = Image.open(image_file)
            image.load()
        except UnidentifiedImageError as exc:
            raise ValueError(
                "Could not read the uploaded image. Use a JPG, JPEG, PNG, or WEBP file."
            ) from exc
        except Exception as exc:
            raise ValueError(f"Could not open image: {exc}") from exc

        image_name = getattr(image_file, "name", "uploaded image")
        image = image.convert("RGB")
        return image, image_name, image.size, []
