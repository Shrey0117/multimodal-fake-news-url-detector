# Fake News Detector

Streamlit app for multimodal misinformation screening. A user can either paste
an article package manually or provide a public article URL. For URLs, the app
extracts the likely headline, article body, and main article image while avoiding
common ad, navigation, sidebar, logo, newsletter, and tracking content. The
cleaned text and image are then passed into the same text, image, and verdict
pipeline.

## What It Does

- Parses the submitted article into headline, body text, and image.
- Fetches public article URLs and extracts likely article text and primary image.
- Filters common ad text, ad images, navigation links, logos, sidebars, and
  newsletter blocks before analysis.
- Scores the body text with a transformer-based fake-news classifier.
- Scores the image with Error Level Analysis (ELA) for manipulation signals.
- Combines both signals into a final 0-100 credibility score.
- Shows the text score, image score, final risk score, ELA heatmap, and the
  factors that drove the verdict.

## Project Layout

```text
fnd-v2/
|-- app.py
|-- config.py
|-- modules/
|   |-- article_parser.py
|   |-- article_url_extractor.py
|   |-- text_analyzer.py
|   |-- image_analyzer.py
|   `-- verdict_engine.py
|-- utils/
|   |-- preprocessor.py
|   `-- visualizer.py
|-- data/
|   |-- sample_articles.json
|   `-- eda_outputs/
`-- requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the text model from Hugging Face. Later runs reuse the
local cache.

## Models

| Component | Model / Technique |
|---|---|
| Primary text classifier | `jy46604790/Fake-News-Bert-Detect` |
| Text fallback | `distilroberta-base` with fill-mask prompting |
| Image analysis | Error Level Analysis (ELA) |

## Comprehensive Score

The final score is displayed as a 0-100 credibility score:

```text
text_fake_risk = text model fake probability * 100
image_manipulation_risk = ELA manipulation score
final_fake_risk = 0.7 * text_fake_risk + 0.3 * image_manipulation_risk
comprehensive_credibility_score = 100 - final_fake_risk
```

If no image is uploaded, the app still parses and scores the text, but the final
score is text-only.

## URL Article Intake

Open the **Article URL** tab, paste a public news/article link, and click
**Fetch and Analyze Article Link**. The extractor:

- reads the page title from article metadata or the main heading,
- keeps meaningful article paragraphs and drops obvious ad/sidebar/navigation text,
- ranks large non-ad images from `og:image`, `twitter:image`, and article `<img>`
  tags,
- downloads the highest-ranked decodable image and sends it to the ELA image
  analyzer.

Some sites block automated fetching, render text only after JavaScript, or hide
article images behind lazy-loading scripts. In those cases, use the manual
paste/upload tab as the fallback.

## Verdict Thresholds

| Condition | Verdict |
|---|---|
| Final fake risk >= 95 | `LIKELY FAKE` |
| Final fake risk <= 40 | `LIKELY REAL` |
| Otherwise | `UNCERTAIN` |

## Limitations

- This is a screening tool, not a substitute for real fact-checking.
- Text models can struggle with satire, opinion, niche domains, and brand-new
  events.
- ELA is stronger on JPEG-style manipulation than on screenshots, AI images, or
  heavily re-exported files.
- URL extraction is heuristic and cannot bypass paywalls, login walls, CAPTCHA
  pages, or JavaScript-only article rendering.
