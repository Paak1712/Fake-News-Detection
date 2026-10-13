import os
from datetime import datetime

# 1. ALWAYS DO THIS FIRST, BEFORE ANY OTHER IMPORTS
from scrapy.utils.reactor import install_reactor
install_reactor('twisted.internet.asyncioreactor.AsyncioSelectorReactor') 

import pandas as pd
import scrapy
from scrapy.crawler import CrawlerRunner
from crochet import setup, wait_for

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier


# Initialize Crochet AFTER all imports
setup()


# ==========================================
# 1. SCRAPY SPIDER: Automated Web Collector
# ==========================================
class NewsScraperSpider(scrapy.Spider):
    name = "news_spider"

    start_urls = [
        "https://quotes.toscrape.com/",
    ]

    custom_settings = {
        "FEED_FORMAT": "json",
        "FEED_URI": "scraped_articles.json",
        "CONCURRENT_REQUESTS": 8,
        "DOWNLOAD_DELAY": 0.5,
        "LOG_LEVEL": "ERROR",
        "FEED_OVERWRITE": True,
    }

    def parse(self, response):
        """Extract individual article links and navigate to them."""
        links = response.css("div.quote a::attr(href)").getall()

        for link in links:
            yield response.follow(link, callback=self.parse_article)

    def parse_article(self, response):
        """Extract text and metadata from the web page."""

        title = response.css("h1::text, title::text").get(
            default="No Title"
        ).strip()

        paragraphs = response.css("p::text").getall()

        text = " ".join(
            p.strip() for p in paragraphs if p.strip()
        )

        author = response.css(".author::text").get(
            default="Unknown"
        ).strip()

        # Feature metrics
        words = text.split()
        word_count = len(words)
        title_length = len(title)
        exclamation_count = (
            text.count("!") + title.count("!")
        )
        caps_word_count = sum(
            1 for word in words
            if word.isupper() and len(word) > 1
        )

        yield {
            "title": title,
            "text": text,
            "author": author,
            "word_count": word_count,
            "title_length": title_length,
            "exclamation_count": exclamation_count,
            "caps_word_count": caps_word_count,
            "scraped_at": datetime.now().isoformat(),
            "url": response.url,
        }


# ==========================================
# Run Scrapy safely with Crochet
# ==========================================
@wait_for(timeout=30.0)
def run_spider():
    runner = CrawlerRunner()
    return runner.crawl(NewsScraperSpider)


# ==========================================
# 2. DETECTOR MODULE
# ==========================================
class FakeNewsDetector:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_df=0.7
        )

        self.classifier = PassiveAggressiveClassifier(
            max_iter=50
        )

        # Common clickbait / fake news trigger keywords
        self.fake_keywords = [
            "shocking", "secret", "miracle", "cures", "unbelievable",
            "exposed", "conspiracy", "aliens", "hoax", "banned",
            "100%", "guaranteed", "you won't believe"
        ]

        self._train_fake_news_ml_model()

    def _train_fake_news_ml_model(self):
        """Train model on extended Real vs Fake news dataset."""

        sample_data = [
            # --- FAKE NEWS EXAMPLES ---
            ("BREAKING: Shocking secret alien discovery inside military base!", "FAKE"),
            ("Miracle remedy cures all known ailments instantly without doctors!", "FAKE"),
            ("EXPOSED: Secret government plot to control the weather leaked!", "FAKE"),
            ("You won't believe what this celebrity did to gain endless youth!", "FAKE"),
            ("Scientists prove 100% cure found using this simple kitchen trick!", "FAKE"),
            ("BANNED VIDEO: The secret truth they don't want you to know!", "FAKE"),
            ("Local resident finds hidden gold chest inside public park overnight!", "FAKE"),
            
            # --- REAL NEWS EXAMPLES ---
            ("Government announces new annual infrastructure tax plan.", "REAL"),
            ("Central bank releases updated financial economic statistics today.", "REAL"),
            ("Ministry of Health publishes new guidelines on seasonal flu vaccines.", "REAL"),
            ("City council approves budget allocation for local public transit expansion.", "REAL"),
            ("Researchers present new peer-reviewed findings at international physics conference.", "REAL"),
            ("Department of Agriculture releases quarterly crop yield estimates.", "REAL"),
            ("Global trade summit concludes with bilateral economic agreement.", "REAL")
        ]

        texts, labels = zip(*sample_data)

        X = self.vectorizer.fit_transform(texts)

        self.classifier.fit(X, labels)

    def compute_rule_score(self, row):
        """Calculate rule-based heuristic score with fake news indicators."""

        score = 0
        reasons = []

        content_to_check = (str(row["title"]) + " " + str(row["text"])).lower()

        # Rule 1: Short content body
        if row["word_count"] < 100:
            score += 1
            reasons.append("Short Content Body (< 100 words)")

        # Rule 2: Clickbait or long title length
        if row["title_length"] > 120:
            score += 1
            reasons.append("Clickbait Title Length (> 120 chars)")

        # Rule 3: Excessive punctuation or ALL CAPS
        if row["exclamation_count"] > 2 or row["caps_word_count"] > 3:
            score += 1
            reasons.append("High Punctuation / CAPS Spikes")

        # Rule 4: Missing or suspicious author
        if row["author"] in ["Unknown", "Admin", ""]:
            score += 1
            reasons.append("Missing/Anonymous Author Attribution")

        # Rule 5: Sensationalist Fake News Keywords
        triggered_words = [word for word in self.fake_keywords if word in content_to_check]
        if triggered_words:
            score += 1
            reasons.append(f"Sensationalist Keywords Detected ({', '.join(triggered_words)})")

        return score, reasons

    def analyze_scraped_data(self, json_file):
        """Analyze scraped JSON data."""

        if not os.path.exists(json_file):
            print(
                f"Error: Output dataset file "
                f"'{json_file}' not found."
            )
            return

        df = pd.read_json(json_file)

        if df.empty:
            print("No data extracted.")
            return

        print("\n" + "=" * 70)
        print("          FAKE NEWS DETECTION ANALYSIS REPORT")
        print("=" * 70)

        for idx, row in df.iterrows():

            # Rule-based score
            rule_score, flags = self.compute_rule_score(row)

            # Risk level
            if rule_score >= 3:
                risk_level = "HIGH RISK (Suspicious Pattern)"
            elif rule_score == 2:
                risk_level = "MODERATE RISK"
            else:
                risk_level = "LOW RISK"

            # ML prediction
            text_vec = self.vectorizer.transform(
                [str(row["title"]) + " " + str(row["text"])]
            )

            ml_pred = self.classifier.predict(
                text_vec
            )[0]

            # Display results
            print(
                f"\n[Article #{idx + 1}] "
                f"Title: {str(row['title'])[:60]}..."
            )

            print(
                f" ► Source URL     : {row['url']}"
            )

            print(
                f" ► Metadata       : "
                f"Author: {row['author']} | "
                f"Words: {row['word_count']}"
            )

            print(
                f" ► Heuristic Score: "
                f"{rule_score}/5 ({risk_level})"
            )

            print(
                f" ► Triggered Rules: "
                f"{', '.join(flags) if flags else 'None'}"
            )

            print(
                f" ► ML Classifier  : "
                f"Predicted as [{ml_pred}]"
            )

            print("-" * 70)


# ==========================================
# 3. PIPELINE EXECUTION
# ==========================================
if __name__ == "__main__":

    print("Step 1: Running Scrapy Web Crawler...")

    run_spider()

    print(
        "Step 2: Scraped successfully! "
        "Analyzing extracted data..."
    )

    detector = FakeNewsDetector()

    detector.analyze_scraped_data(
        "scraped_articles.json"
    )