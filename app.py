"""
app.py
------
Streamlit entry point for the multimodal Fake News Detector.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

import config
from modules.article_parser import ArticleParser, ParsedArticle
from modules.article_url_extractor import ArticleURLExtractor, ExtractedArticle
from modules.image_analyzer import ImageAnalyzer
from modules.text_analyzer import TextAnalyzer
from modules.verdict_engine import VerdictEngine
from utils.preprocessor import TextPreprocessor
from utils.visualizer import (
    plot_confidence_gauge,
    plot_contributing_factors,
    plot_ela_comparison,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


st.set_page_config(
    page_title=config.APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading text classifier...")
def load_text_analyzer() -> TextAnalyzer:
    return TextAnalyzer()


@st.cache_resource
def load_other_components():
    return ImageAnalyzer(), VerdictEngine(), TextPreprocessor()


@st.cache_resource
def load_url_extractor() -> ArticleURLExtractor:
    return ArticleURLExtractor()


@st.cache_data
def load_sample_articles():
    try:
        with open(config.SAMPLE_ARTICLES, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        logger.warning("Could not load sample articles: %s", exc)
        return []


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .score-band {
            border: 1px solid rgba(250, 250, 250, 0.12);
            border-left-width: 6px;
            border-radius: 8px;
            padding: 18px 20px;
            margin: 8px 0 18px;
            background: rgba(255, 255, 255, 0.035);
        }
        .score-label {
            color: rgba(250, 250, 250, 0.72);
            font-size: 0.9rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        .score-value {
            color: #fff;
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.05;
            margin-top: 4px;
        }
        .score-value span {
            color: rgba(250, 250, 250, 0.7);
            font-size: 1.05rem;
            font-weight: 600;
            margin-left: 4px;
        }
        .score-subtitle {
            color: rgba(250, 250, 250, 0.74);
            font-size: 1rem;
            margin-top: 8px;
        }
        .verdict-pill {
            display: inline-block;
            padding: 8px 18px;
            border-radius: 8px;
            color: #fff;
            font-size: 1.1rem;
            font-weight: 800;
            margin-top: 10px;
        }
        .model-pill {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 4px;
            color: #fff;
            font-size: 0.78rem;
            font-weight: 650;
            vertical-align: middle;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def verdict_badge(verdict: str, color: str) -> str:
    return (
        f'<span class="verdict-pill" style="background-color:{color};">'
        f"{verdict}</span>"
    )


def risk_badge(risk: str) -> str:
    colours = {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}
    return (
        f'<span class="verdict-pill" style="background-color:{colours.get(risk, "#888")}; '
        f'font-size:0.95rem; padding:6px 14px;">{risk} RISK</span>'
    )


def model_badge(mode: str) -> str:
    label = (
        "Article Classifier"
        if mode == "classifier"
        else "DistilRoBERTa Fill-Mask Fallback"
    )
    color = "#1a7f4b" if mode == "classifier" else "#b07d00"
    return f'<span class="model-pill" style="background-color:{color};">{label}</span>'


def render_sidebar(samples, text_analyzer: TextAnalyzer) -> None:
    with st.sidebar:
        st.markdown(f"## {config.APP_TITLE}")
        st.caption(config.APP_SUBTITLE)
        st.divider()

        mode_label = (
            "Article classifier"
            if text_analyzer.mode == "classifier"
            else "Fill-mask fallback"
        )
        st.success(f"Model active: **{mode_label}**")
        st.caption(f"`{text_analyzer.active_model}`")
        st.divider()

        with st.expander("Screening method", expanded=False):
            st.markdown(
                f"""
                | Signal | Weight |
                |---|---:|
                | Text fake-news classifier | {config.TEXT_WEIGHT * 100:.0f}% |
                | Image ELA manipulation score | {config.IMAGE_WEIGHT * 100:.0f}% |

                The final output is a decision-support score, not a verified fact-check.
                """
            )

        with st.expander("Model details", expanded=False):
            st.markdown(
                f"""
                | Property | Value |
                |---|---|
                | Primary model | `{config.MODEL_NAME}` |
                | Fallback model | `{config.FALLBACK_MODEL_NAME}` |
                | Active backend | `{text_analyzer.mode}` |
                | Hard token limit | **{config.MAX_TOKEN_LENGTH}** |
                | Fake threshold | `{config.FAKE_THRESHOLD * 100:.0f}%` |
                | Real threshold | `{config.REAL_THRESHOLD * 100:.0f}%` |
                """
            )

        st.divider()
        st.markdown("### Sample Article")

        if samples:
            options = {
                f"[{article['true_label']}] {article['title'][:55]}...": article
                for article in samples
            }
            selected_key = st.selectbox("Choose a sample", list(options.keys()), index=0)
            if st.button("Load Sample", use_container_width=True):
                article = options[selected_key]
                st.session_state["article_title_input"] = article["title"]
                st.session_state["article_text_input"] = article["content"]
                st.session_state["loaded_label"] = article["true_label"]
                st.success("Loaded into the article form.")
        else:
            st.warning("Sample articles file not found.")

        eda_dir = Path(config.EDA_OUTPUT_DIR)
        eda_plots = list(eda_dir.glob("*.png")) if eda_dir.exists() else []
        if eda_plots:
            st.divider()
            st.markdown("### EDA Outputs")
            for plot_path in eda_plots[:4]:
                st.image(str(plot_path), caption=plot_path.stem.replace("_", " ").title())


def render_article_form(parser: ArticleParser, url_extractor: ArticleURLExtractor):
    st.subheader("Article Package")
    url_tab, manual_tab = st.tabs(["Article URL", "Manual paste/upload"])

    with url_tab:
        parsed = render_url_article_form(parser, url_extractor)
        if parsed is not None:
            return parsed

    with manual_tab:
        return render_manual_article_form(parser)


def render_url_article_form(
    parser: ArticleParser,
    url_extractor: ArticleURLExtractor,
) -> Optional[ParsedArticle]:
    st.caption(
        "Paste a public article link. The extractor keeps likely article body text "
        "and large article images while filtering common ad, navigation, sidebar, "
        "newsletter, logo, and tracking content."
    )

    with st.form("article_url_form", clear_on_submit=False):
        article_url = st.text_input(
            "Article URL",
            key="article_url_input",
            placeholder="https://example.com/news/story",
        )
        use_extracted_image = st.checkbox(
            "Use best detected article image",
            value=True,
            help="The app downloads the highest-ranked non-ad image and passes it to the ELA image analyzer.",
        )
        submitted = st.form_submit_button(
            "Fetch and Analyze Article Link",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return None

    try:
        with st.spinner("Fetching and cleaning article page..."):
            extracted = url_extractor.fetch(article_url)
            image_file = (
                url_extractor.download_best_image(extracted.images)
                if use_extracted_image
                else None
            )
            parsed = parser.parse(
                raw_text=extracted.text,
                headline=extracted.title,
                image_file=image_file,
                auto_detect_headline=False,
            )
    except ValueError as exc:
        st.error(str(exc))
        return None

    st.session_state["url_extraction_summary"] = extracted
    if use_extracted_image and image_file is None and extracted.images:
        parsed.warnings.append(
            "Article images were detected, but none could be downloaded and decoded safely."
        )
    return parsed


def render_manual_article_form(parser: ArticleParser) -> Optional[ParsedArticle]:
    with st.form("article_package_form", clear_on_submit=False):
        headline = st.text_input(
            "Headline",
            key="article_title_input",
            placeholder="Optional headline",
        )
        article_text = st.text_area(
            "Article text",
            key="article_text_input",
            height=280,
            placeholder=f"Paste the complete article body here ({config.MIN_WORDS}+ words)",
        )
        article_image = st.file_uploader(
            "Article image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Attach the image published with this article for a full multimodal score.",
        )

        col_auto, col_submit = st.columns([1, 2])
        with col_auto:
            auto_split = st.checkbox(
                "Detect headline",
                value=(not bool(headline.strip())),
                help="When the headline is blank, use the first line or first sentence as the headline.",
            )
        with col_submit:
            submitted = st.form_submit_button(
                "Analyze Article Package",
                type="primary",
                use_container_width=True,
            )

    clear_clicked = st.button("Clear Manual Inputs", use_container_width=False)
    if clear_clicked:
        for key in ("article_title_input", "article_text_input", "loaded_label", "url_extraction_summary"):
            st.session_state.pop(key, None)
        st.rerun()

    if not submitted:
        return None

    try:
        st.session_state.pop("url_extraction_summary", None)
        return parser.parse(
            raw_text=article_text,
            headline=headline,
            image_file=article_image,
            auto_detect_headline=auto_split,
        )
    except ValueError as exc:
        st.error(str(exc))
        return None


def render_url_extraction_summary(extracted: ExtractedArticle) -> None:
    with st.expander("URL extraction details", expanded=True):
        st.markdown(f"**Source URL:** {extracted.source_url}")
        st.markdown(f"**Extracted headline:** {extracted.title or 'not found'}")
        st.metric("Extracted words", extracted.word_count)
        st.metric("Candidate article images", len(extracted.images))

        if extracted.images:
            top_images = extracted.images[:3]
            rows = [
                {
                    "rank": idx + 1,
                    "score": round(image.score, 1),
                    "source": image.source,
                    "size": f"{image.width} x {image.height}" if image.width and image.height else "unknown",
                    "url": image.url[:120] + ("..." if len(image.url) > 120 else ""),
                }
                for idx, image in enumerate(top_images)
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)

        for warning in extracted.warnings:
            st.warning(warning)


def render_parsed_summary(parsed: ParsedArticle) -> None:
    extracted = st.session_state.get("url_extraction_summary")
    if extracted:
        render_url_extraction_summary(extracted)

    with st.expander("Parsed input", expanded=False):
        col_text, col_image = st.columns(2)

        with col_text:
            st.markdown("#### Text")
            st.metric("Words", parsed.stats["word_count"])
            st.metric("Sentences", parsed.stats["sentence_count"])
            if parsed.headline:
                st.markdown(f"**Headline:** {parsed.headline}")
            else:
                st.markdown("**Headline:** none supplied")

        with col_image:
            st.markdown("#### Image")
            if parsed.has_image:
                width, height = parsed.image_size
                st.metric("Image size", f"{width} x {height}")
                st.caption(parsed.image_name)
                st.image(parsed.image, use_container_width=True)
            else:
                st.warning("No image was parsed from this submission.")

        if parsed.warnings:
            for warning in parsed.warnings:
                st.info(warning)


def render_score_report(verdict_result: Dict) -> None:
    color = verdict_result["verdict_color"]
    score = verdict_result["comprehensive_score"]
    risk = verdict_result["risk_score"]

    st.markdown(
        f"""
        <div class="score-band" style="border-left-color:{color};">
            <div class="score-label">Comprehensive Credibility Score</div>
            <div class="score-value">{score:.1f}<span>/100</span></div>
            <div class="score-subtitle">{verdict_result["score_band"]}</div>
            {verdict_badge(verdict_result["verdict"], color)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(int(max(0, min(100, score))), text=f"Credibility {score:.1f}/100")
    st.progress(int(max(0, min(100, risk))), text=f"Misinformation risk {risk:.1f}/100")


def render_metric_breakdown(verdict_result: Dict, text_result: Dict, image_result: Dict) -> None:
    scores = verdict_result["score_breakdown"]
    col_text, col_image, col_final = st.columns(3)

    with col_text:
        st.metric("Text Fake Risk", f"{scores['text_fake_risk']:.1f}/100")
        st.caption(f"Model confidence: {text_result['confidence'] * 100:.1f}%")

    with col_image:
        if image_result:
            st.metric(
                "Image Manipulation Risk",
                f"{scores['image_manipulation_risk']:.1f}/100",
            )
            st.markdown(risk_badge(image_result["risk_level"]), unsafe_allow_html=True)
        else:
            st.metric("Image Manipulation Risk", "N/A")
            st.caption("No image included")

    with col_final:
        st.metric("Final Fake Risk", f"{scores['final_fake_risk']:.1f}/100")
        st.caption(
            f"Text {config.TEXT_WEIGHT * 100:.0f}%"
            + (f" + image {config.IMAGE_WEIGHT * 100:.0f}%" if image_result else "")
        )


def analyze_article(
    parsed: ParsedArticle,
    text_analyzer: TextAnalyzer,
    image_analyzer: ImageAnalyzer,
    verdict_engine: VerdictEngine,
) -> None:
    with st.spinner("Running text and image analysis..."):
        try:
            text_result = text_analyzer.analyze(parsed.body, title=parsed.headline)
            image_result = (
                image_analyzer.analyze(parsed.image) if parsed.has_image else None
            )
            verdict_result = verdict_engine.combine(text_result, image_result)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            logger.exception("Article analysis error")
            return

    st.divider()
    st.markdown("### Comprehensive Result")
    st.markdown(model_badge(text_result["mode"]), unsafe_allow_html=True)
    render_score_report(verdict_result)
    st.markdown(f"**Assessment:** {verdict_result['explanation']}")

    if "loaded_label" in st.session_state:
        st.caption(f"Loaded sample label: `{st.session_state['loaded_label']}`")

    st.markdown("### Score Breakdown")
    render_metric_breakdown(verdict_result, text_result, image_result)

    st.markdown("### Text Evidence")
    gauge = plot_confidence_gauge(
        text_result["fake_probability"],
        text_result["real_probability"],
    )
    st.pyplot(gauge, use_container_width=True)

    with st.expander("Text model input", expanded=False):
        col_words, col_chars, col_sentences, col_tokens = st.columns(4)
        col_words.metric("Words", parsed.stats["word_count"])
        col_chars.metric("Characters", parsed.stats["char_count"])
        col_sentences.metric("Sentences", parsed.stats["sentence_count"])
        col_tokens.metric("Tokens", text_result["token_count"])

        used_pct = min(text_result["token_count"] / config.MAX_TOKEN_LENGTH, 1.0)
        st.progress(used_pct)
        if text_result["truncated"]:
            st.warning(
                f"Input exceeded {config.MAX_TOKEN_LENGTH} tokens, so only the first "
                "portion of the article was analysed."
            )
        else:
            st.success("Full text fit inside the model token limit.")

        st.markdown(f"**Input strategy:** `{text_result['input_strategy']}`")
        st.code(text_result["input_preview"], language=None)
        st.caption(f"Model: `{text_result['model_used']}`")

    if image_result:
        st.markdown("### Image Evidence")
        fig_ela = plot_ela_comparison(parsed.image, image_result["ela_image"])
        st.pyplot(fig_ela, use_container_width=True)
        st.markdown(f"**Image assessment:** {image_result['assessment']}")

        with st.expander("Image forensic details", expanded=False):
            st.markdown(
                f"""
                | Metric | Value |
                |---|---|
                | Manipulation score | `{image_result['manipulation_score']:.2f}` |
                | Mean pixel difference | `{image_result['mean_difference']:.4f}` |
                | Risk level | `{image_result['risk_level']}` |
                | ELA re-save quality | `{config.IMAGE_ELA_QUALITY}` |
                | Low threshold | `{config.IMAGE_MANIPULATION_THRESHOLD_LOW}` |
                | High threshold | `{config.IMAGE_MANIPULATION_THRESHOLD_HIGH}` |
                """
            )

    st.markdown("### Contributing Factors")
    factors = plot_contributing_factors(verdict_result["contributing_factors"])
    st.pyplot(factors, use_container_width=True)


def main() -> None:
    inject_styles()

    try:
        text_analyzer = load_text_analyzer()
    except RuntimeError as exc:
        st.error(
            f"**Model failed to load:** {exc}\n\n"
            "Troubleshooting:\n"
            "1. Make sure the first run has internet access so the model can download.\n"
            "2. Verify dependencies with `pip install -r requirements.txt`.\n"
            "3. Re-run the app if the initial model download was interrupted."
        )
        st.stop()

    image_analyzer, verdict_engine, preprocessor = load_other_components()
    url_extractor = load_url_extractor()
    parser = ArticleParser(preprocessor)
    samples = load_sample_articles()

    render_sidebar(samples, text_analyzer)

    st.title(config.APP_TITLE)
    st.caption("Analyze pasted articles or public article URLs with separate text and image scoring.")
    st.divider()

    parsed = render_article_form(parser, url_extractor)
    if parsed is None:
        return

    render_parsed_summary(parsed)
    analyze_article(parsed, text_analyzer, image_analyzer, verdict_engine)


if __name__ == "__main__":
    main()
