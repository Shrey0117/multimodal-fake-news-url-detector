"""
app.py
------
Streamlit entry point for the multimodal Fake News Detector.
Optimized for: Shri Ramdeobaba University, Nagpur B.Tech CSE Capstone Tracker
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

        # Institutional Context Badge for Verification Reviewers
        st.markdown("### 🎓 Research Lab Context")
        st.info(
            "**Shri Ramdeobaba University, Nagpur**\n\n"
            "Department of Computer Science & Engineering\n\n"
            "*B.Tech Final Year Advanced AI & Analytics Component*"
        )
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

        with st.expander("Model details & Benchmarking", expanded=False):
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
                | LLM Assessment | `Claude Max Active Benchmarking` |
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
            placeholder="https://example.com",
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

    if submitted and article_url.strip():
        try:
            with st.spinner("Extracting content from targeted domain node..."):
                extracted: ExtractedArticle = url_extractor.extract(article_url)
                parsed_article = parser.parse_extracted(extracted, parse_image=use_extracted_image)
                st.success("Domain components ingested successfully.")
                return parsed_article
        except Exception as err:
            st.error(f"URL extraction failed: {str(err)}")
            logger.error("Content fetch exception encountered: %s", err)
            
    return None


def render_manual_article_form(parser: ArticleParser) -> Optional[ParsedArticle]:
    with st.form("manual_input_form", clear_on_submit=False):
        headline = st.text_input("Headline", placeholder="Enter core headline text")
        body_text = st.text_area("Body Text", placeholder="Paste contextual paragraphs here...")
        uploaded_file = st.file_uploader("Upload related visual asset", type=["jpg", "jpeg", "png"])
        manual_submitted = st.form_submit_button("Analyze Manual Input", type="primary", use_container_width=True)
        
    if manual_submitted and body_text.strip():
        return parser.parse_manual(headline, body_text, uploaded_file)
    return None


if __name__ == "__main__":
    inject_styles()
