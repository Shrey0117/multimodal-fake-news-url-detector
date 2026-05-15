import unittest

from modules.article_url_extractor import ArticleURLExtractor


class ArticleURLExtractorTest(unittest.TestCase):
    def test_extracts_article_text_and_skips_ad_blocks(self):
        html = """
        <html>
          <head>
            <meta property="og:title" content="Important Policy Story - Example News">
            <meta property="og:image" content="/images/story-hero.jpg">
          </head>
          <body>
            <nav>Home Politics Markets</nav>
            <article>
              <h1>Important Policy Story</h1>
              <p>The government announced a major policy update after weeks of public consultation and expert review.</p>
              <p>Officials said the new framework will be implemented in phases, with independent audits and public reporting.</p>
              <p>Researchers interviewed for the article said the policy details still require careful monitoring over time.</p>
              <img class="article hero" src="/images/story-hero.jpg" width="900" height="600" alt="Officials at press briefing">
            </article>
            <aside class="sidebar ad">
              <p>Advertisement subscribe now for unrelated offers and sponsored placements.</p>
              <img src="/ads/banner.jpg" width="728" height="90" alt="advertisement">
            </aside>
          </body>
        </html>
        """

        extracted = ArticleURLExtractor().extract_from_html(
            html,
            "https://news.example.com/politics/story",
        )

        self.assertEqual(extracted.title, "Important Policy Story")
        self.assertIn("major policy update", extracted.text)
        self.assertIn("independent audits", extracted.text)
        self.assertNotIn("Advertisement", extracted.text)
        self.assertTrue(extracted.images)
        self.assertEqual(
            extracted.images[0].url,
            "https://news.example.com/images/story-hero.jpg",
        )
        self.assertNotIn("banner", " ".join(image.url for image in extracted.images))

    def test_warns_when_no_article_image_survives_filtering(self):
        html = """
        <html>
          <head><title>Article Without Image</title></head>
          <body>
            <main>
              <h1>Article Without Image</h1>
              <p>This article contains enough meaningful body text to pass extraction without needing any sidebar content.</p>
              <p>The body includes a second paragraph so the extractor can return useful text for the text analyzer.</p>
              <p>A third paragraph confirms that article extraction is not dependent on image availability.</p>
              <img src="/logo.svg" width="80" height="80" alt="site logo">
            </main>
          </body>
        </html>
        """

        extracted = ArticleURLExtractor().extract_from_html(
            html,
            "https://news.example.com/story",
        )

        self.assertFalse(extracted.images)
        self.assertTrue(any("No suitable article image" in warning for warning in extracted.warnings))


if __name__ == "__main__":
    unittest.main()
