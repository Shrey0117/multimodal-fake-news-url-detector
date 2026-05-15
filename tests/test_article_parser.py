import io
import unittest

from PIL import Image

from modules.article_parser import ArticleParser
from utils.preprocessor import TextPreprocessor


class ArticleParserTest(unittest.TestCase):
    def test_parse_splits_headline_and_image(self):
        body = " ".join(f"word{i}" for i in range(30))
        raw_text = f"Clear Headline\n{body}"
        image_file = io.BytesIO()
        Image.new("RGB", (8, 6), "white").save(image_file, format="PNG")
        image_file.seek(0)
        image_file.name = "article.png"

        parsed = ArticleParser(TextPreprocessor()).parse(
            raw_text=raw_text,
            image_file=image_file,
        )

        self.assertEqual(parsed.headline, "Clear Headline")
        self.assertEqual(parsed.stats["word_count"], 30)
        self.assertTrue(parsed.has_image)
        self.assertEqual(parsed.image_name, "article.png")
        self.assertEqual(parsed.image_size, (8, 6))

    def test_parse_allows_text_only_with_warning(self):
        body = " ".join(f"word{i}" for i in range(25))

        parsed = ArticleParser(TextPreprocessor()).parse(raw_text=body)

        self.assertFalse(parsed.has_image)
        self.assertIn("text-only", parsed.warnings[0])


if __name__ == "__main__":
    unittest.main()
