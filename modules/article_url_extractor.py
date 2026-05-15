"""
modules/article_url_extractor.py
--------------------------------
Fetches a news article URL and extracts likely article text and article images.

The extractor is intentionally heuristic and dependency-light. It avoids common
ad, navigation, subscription, footer, and sidebar regions before passing the
cleaned article package into the existing parser/analyzer pipeline.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError

import config


NOISE_KEYWORDS = {
    "ad",
    "ads",
    "advert",
    "advertisement",
    "banner",
    "breadcrumb",
    "cookie",
    "footer",
    "header",
    "login",
    "menu",
    "modal",
    "nav",
    "newsletter",
    "outbrain",
    "promo",
    "related",
    "share",
    "sidebar",
    "sponsor",
    "subscribe",
    "taboola",
    "tracking",
    "widget",
}

POSITIVE_IMAGE_KEYWORDS = {
    "article",
    "content",
    "featured",
    "hero",
    "lead",
    "main",
    "primary",
    "story",
}

IMAGE_NOISE_KEYWORDS = NOISE_KEYWORDS | {
    "avatar",
    "badge",
    "favicon",
    "icon",
    "logo",
    "pixel",
    "sprite",
}

TEXT_NOISE_PHRASES = (
    "accept cookies",
    "advertisement",
    "all rights reserved",
    "click here",
    "cookie policy",
    "newsletter",
    "privacy policy",
    "read more",
    "related articles",
    "sign in",
    "sign up",
    "subscribe",
    "terms of service",
)


@dataclass
class ExtractedImage:
    url: str
    alt: str = ""
    width: int = 0
    height: int = 0
    score: float = 0.0
    source: str = "html"


@dataclass
class ExtractedArticle:
    source_url: str
    title: str
    text: str
    images: List[ExtractedImage] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(re.findall(r"\b\w+\b", self.text))


class _ArticleHTMLParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.stack: List[Dict] = []
        self.meta: Dict[str, str] = {}
        self.title_chunks: List[str] = []
        self.capture_title = False
        self.capture: Optional[Dict] = None
        self.blocks: List[Tuple[str, str]] = []
        self.images: List[ExtractedImage] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        attrs_dict = {str(k).lower(): str(v or "") for k, v in attrs}

        if tag in {"script", "style", "noscript", "template", "svg"}:
            self.skip_depth += 1

        parent_noisy = any(frame.get("noisy") for frame in self.stack)
        noisy = parent_noisy or tag in {"aside", "footer", "form", "iframe", "nav"}
        attrs_text = self._attrs_text(attrs_dict)
        if self._contains_noise(attrs_text):
            noisy = True
        self.stack.append({"tag": tag, "attrs": attrs_dict, "noisy": noisy})

        if tag == "meta":
            self._handle_meta(attrs_dict)
        elif tag == "title":
            self.capture_title = True
        elif tag in {"p", "h1", "h2", "li"} and not noisy and not self.skip_depth:
            self.capture = {"tag": tag, "chunks": []}
        elif tag in {"img", "source"} and not noisy and not self.skip_depth:
            self._handle_image(attrs_dict, source="html")

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if self.capture and self.capture["tag"] == tag:
            text = self._clean_text(" ".join(self.capture["chunks"]))
            if self._is_useful_block(self.capture["tag"], text):
                self.blocks.append((self.capture["tag"], text))
            self.capture = None
        if tag == "title":
            self.capture_title = False
        if tag in {"script", "style", "noscript", "template", "svg"} and self.skip_depth:
            self.skip_depth -= 1
        if self.stack:
            self.stack.pop()

    def handle_data(self, data: str):
        if self.skip_depth:
            return
        if self.capture is not None:
            self.capture["chunks"].append(data)
        if self.capture_title:
            self.title_chunks.append(data)

    def _handle_meta(self, attrs: Dict[str, str]) -> None:
        key = (attrs.get("property") or attrs.get("name") or "").strip().lower()
        value = (attrs.get("content") or "").strip()
        if not key or not value:
            return
        if key in {
            "og:title",
            "twitter:title",
            "description",
            "og:description",
            "twitter:description",
            "article:published_time",
            "author",
        }:
            self.meta[key] = value
        if key in {"og:image", "twitter:image"}:
            self._add_image(value, "", 0, 0, source=key)

    def _handle_image(self, attrs: Dict[str, str], source: str) -> None:
        src = (
            attrs.get("src")
            or attrs.get("data-src")
            or attrs.get("data-original")
            or attrs.get("data-lazy-src")
            or self._best_srcset(attrs.get("srcset", ""))
        )
        if not src:
            return
        alt = attrs.get("alt", "")
        width = self._to_int(attrs.get("width", "0"))
        height = self._to_int(attrs.get("height", "0"))
        attrs_text = self._attrs_text(attrs)
        if self._is_noisy_image(src, alt, attrs_text, width, height):
            return
        self._add_image(src, alt, width, height, source=source, attrs_text=attrs_text)

    def _add_image(
        self,
        src: str,
        alt: str,
        width: int,
        height: int,
        source: str,
        attrs_text: str = "",
    ) -> None:
        url = urljoin(self.base_url, unescape(src.strip()))
        if self._is_noisy_image(url, alt, attrs_text, width, height):
            return
        score = 10.0
        if source in {"og:image", "twitter:image"}:
            score += 30
        if width and height:
            score += min((width * height) / 12000, 35)
        if any(word in attrs_text.lower() for word in POSITIVE_IMAGE_KEYWORDS):
            score += 12
        if alt and len(alt.split()) >= 3:
            score += 5
        self.images.append(
            ExtractedImage(url=url, alt=self._clean_text(alt), width=width, height=height, score=score, source=source)
        )

    @staticmethod
    def _attrs_text(attrs: Dict[str, str]) -> str:
        keys = ("id", "class", "role", "aria-label", "data-testid", "alt")
        return " ".join(attrs.get(key, "") for key in keys).lower()

    @staticmethod
    def _clean_text(text: str) -> str:
        text = unescape(text or "")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _contains_noise(text: str) -> bool:
        tokens = set(re.split(r"[^a-z0-9]+", text.lower()))
        return bool(tokens & NOISE_KEYWORDS)

    @staticmethod
    def _to_int(value: str) -> int:
        match = re.search(r"\d+", value or "")
        return int(match.group(0)) if match else 0

    @staticmethod
    def _best_srcset(srcset: str) -> str:
        if not srcset:
            return ""
        candidates = []
        for item in srcset.split(","):
            parts = item.strip().split()
            if not parts:
                continue
            url = parts[0]
            score = 0
            if len(parts) > 1:
                score = _ArticleHTMLParser._to_int(parts[1])
            candidates.append((score, url))
        return sorted(candidates, reverse=True)[0][1] if candidates else ""

    @staticmethod
    def _is_useful_block(tag: str, text: str) -> bool:
        if not text:
            return False
        lower = text.lower()
        if any(phrase in lower for phrase in TEXT_NOISE_PHRASES):
            return False
        word_count = len(re.findall(r"\b\w+\b", text))
        if tag in {"h1", "h2"}:
            return len(text) >= 8 and word_count >= 2
        if tag == "li":
            return len(text) >= 70 and word_count >= 10
        return len(text) >= 45 and word_count >= 8

    @staticmethod
    def _is_noisy_image(src: str, alt: str, attrs_text: str, width: int, height: int) -> bool:
        text = f"{src} {alt} {attrs_text}".lower()
        path = urlparse(src).path.lower()
        if path.endswith((".svg", ".ico")):
            return True
        tokens = set(re.split(r"[^a-z0-9]+", text))
        if tokens & IMAGE_NOISE_KEYWORDS:
            return True
        if width and height and (width < 180 or height < 120):
            return True
        return False


class ArticleURLExtractor:
    def fetch(self, url: str) -> ExtractedArticle:
        normalized_url = self._normalize_url(url)
        html = self._download_html(normalized_url)
        return self.extract_from_html(html, normalized_url)

    def extract_from_html(self, html: str, source_url: str) -> ExtractedArticle:
        parser = _ArticleHTMLParser(source_url)
        parser.feed(html)
        blocks = parser.blocks or self._fallback_blocks(html)

        raw_title = (
            self._first_h1(blocks)
            or parser.meta.get("og:title")
            or parser.meta.get("twitter:title")
            or " ".join(parser.title_chunks)
        )
        title = self._clean_title(raw_title)

        paragraphs = self._dedupe_blocks(blocks, title)
        text = "\n\n".join(paragraphs)

        images = self._rank_images(parser.images)
        warnings: List[str] = []
        if not title:
            warnings.append("No reliable headline was found in the page metadata.")
        if len(paragraphs) < 3:
            warnings.append("Only limited article text was extracted; this page may block scraping or use unusual markup.")
        if not images:
            warnings.append("No suitable article image was found after filtering ads, icons, and logos.")

        return ExtractedArticle(
            source_url=source_url,
            title=title,
            text=text,
            images=images,
            warnings=warnings,
        )

    def download_best_image(self, images: List[ExtractedImage]) -> Optional[io.BytesIO]:
        for image in images[: getattr(config, "URL_IMAGE_CANDIDATE_LIMIT", 5)]:
            try:
                return self.download_image(image)
            except ValueError:
                continue
        return None

    def download_image(self, image: ExtractedImage) -> io.BytesIO:
        request = Request(image.url, headers={"User-Agent": config.URL_FETCH_USER_AGENT})
        try:
            with urlopen(request, timeout=config.URL_FETCH_TIMEOUT_SECONDS) as response:
                content_type = response.headers.get("Content-Type", "")
                if content_type and not content_type.lower().startswith("image/"):
                    raise ValueError("URL did not return an image response.")
                data = response.read(config.URL_IMAGE_MAX_BYTES + 1)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise ValueError(f"Could not download article image: {exc}") from exc

        if len(data) > config.URL_IMAGE_MAX_BYTES:
            raise ValueError("Article image is too large to process safely.")

        try:
            probe = Image.open(io.BytesIO(data))
            probe.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Downloaded image could not be decoded.") from exc

        image_file = io.BytesIO(data)
        suffix = Path(urlparse(image.url).path).suffix or ".jpg"
        image_file.name = f"extracted_article_image{suffix}"
        image_file.seek(0)
        return image_file

    @staticmethod
    def _normalize_url(url: str) -> str:
        value = (url or "").strip()
        if not value:
            raise ValueError("Enter an article URL.")
        parsed = urlparse(value)
        if not parsed.scheme:
            value = "https://" + value
            parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Enter a valid http or https article URL.")
        return value

    @staticmethod
    def _download_html(url: str) -> str:
        request = Request(
            url,
            headers={
                "User-Agent": config.URL_FETCH_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        try:
            with urlopen(request, timeout=config.URL_FETCH_TIMEOUT_SECONDS) as response:
                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                    raise ValueError("The URL did not return an HTML article page.")
                raw = response.read(config.URL_HTML_MAX_BYTES + 1)
                charset = response.headers.get_content_charset() or "utf-8"
        except HTTPError as exc:
            if exc.code in {401, 403}:
                raise ValueError(
                    "This website blocked automated article fetching "
                    f"(HTTP {exc.code}). This is common on Reuters and some other "
                    "major news sites. The app will not bypass access controls; "
                    "open the article in your browser and use the manual paste/upload "
                    "tab for that source."
                ) from exc
            raise ValueError(f"Could not fetch article URL: HTTP {exc.code} {exc.reason}") from exc
        except (URLError, TimeoutError) as exc:
            raise ValueError(f"Could not fetch article URL: {exc}") from exc

        if len(raw) > config.URL_HTML_MAX_BYTES:
            raise ValueError("Article page is too large to process safely.")
        return raw.decode(charset, errors="replace")

    @staticmethod
    def _first_h1(blocks: List[Tuple[str, str]]) -> str:
        for tag, text in blocks:
            if tag == "h1":
                return text
        return ""

    @staticmethod
    def _clean_title(title: str) -> str:
        title = re.sub(r"\s+", " ", unescape(title or "")).strip()
        for separator in (" | ", " - ", " :: "):
            if separator in title:
                left, right = title.split(separator, 1)
                if len(left.split()) >= 3:
                    return left.strip()
                return right.strip()
        return title

    @staticmethod
    def _dedupe_blocks(blocks: List[Tuple[str, str]], title: str) -> List[str]:
        seen = set()
        result: List[str] = []
        title_norm = re.sub(r"\W+", "", title.lower())
        for tag, text in blocks:
            if tag in {"h1", "h2"}:
                continue
            norm = re.sub(r"\W+", "", text.lower())
            if not norm or norm in seen or norm == title_norm:
                continue
            seen.add(norm)
            result.append(text)
        return result

    @staticmethod
    def _rank_images(images: List[ExtractedImage]) -> List[ExtractedImage]:
        deduped: Dict[str, ExtractedImage] = {}
        for image in images:
            if image.url not in deduped or image.score > deduped[image.url].score:
                deduped[image.url] = image
        return sorted(deduped.values(), key=lambda item: item.score, reverse=True)

    @staticmethod
    def _fallback_blocks(html: str) -> List[Tuple[str, str]]:
        """Fallback for pages whose malformed markup prevents HTMLParser blocks."""
        cleaned = re.sub(
            r"<(script|style|noscript|template|svg)\b[^>]*>.*?</\1>",
            " ",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        cleaned = re.sub(
            r"<(aside|footer|form|header|iframe|nav)\b[^>]*>.*?</\1>",
            " ",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        blocks: List[Tuple[str, str]] = []
        for match in re.finditer(
            r"<(h1|h2|p|li)\b[^>]*>(.*?)</\1>",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            tag = match.group(1).lower()
            inner = re.sub(r"<br\s*/?>", " ", match.group(2), flags=re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", inner)
            text = re.sub(r"\s+", " ", unescape(text)).strip()
            if _ArticleHTMLParser._is_useful_block(tag, text):
                blocks.append((tag, text))
        return blocks
