# ─────────────────────────────────────────────────────────────
#  DAG 1 — News Headline Scraper
#  Scrapes headlines from BBC RSS feed every day at midnight
#  Saves raw CSV to data/raw/headlines_YYYY-MM-DD.csv
# ─────────────────────────────────────────────────────────────

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
import xml.etree.ElementTree as ET
import csv
import os

# ── 1. Define the DAG ─────────────────────────────────────────
with DAG(
    dag_id="news_scraper",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["scraper", "week2"],
    description="Scrapes BBC news headlines daily and saves to CSV",
) as dag:

    # ── 2. Task functions ─────────────────────────────────────

    def scrape_headlines(**context):
        """
        Fetches BBC News RSS feed and extracts headlines + descriptions.
        Saves to data/raw/headlines_YYYY-MM-DD.csv
        Returns the output file path for downstream tasks via XCom.
        """
        # BBC News RSS feed — free, no API key needed
        RSS_URL = "http://feeds.bbci.co.uk/news/rss.xml"

        print(f"Fetching RSS feed from: {RSS_URL}")
        response = requests.get(RSS_URL, timeout=30)
        response.raise_for_status()  # raises error if request failed

        # Parse the XML
        root = ET.fromstring(response.content)
        channel = root.find("channel")
        items = channel.findall("item")

        print(f"Found {len(items)} headlines")

        # Extract headline data
        headlines = []
        for item in items:
            title = item.findtext("title", default="").strip()
            description = item.findtext("description", default="").strip()
            pub_date = item.findtext("pubDate", default="").strip()
            link = item.findtext("link", default="").strip()

            if title:  # skip empty titles
                headlines.append({
                    "title": title,
                    "description": description,
                    "pub_date": pub_date,
                    "link": link,
                    "scraped_at": datetime.now().isoformat(),
                })

        # Save to CSV
        today = context["ds"]  # Airflow provides the run date as YYYY-MM-DD
        output_dir = "/opt/airflow/data/raw"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/headlines_{today}.csv"

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["title", "description", "pub_date", "link", "scraped_at"])
            writer.writeheader()
            writer.writerows(headlines)

        print(f"Saved {len(headlines)} headlines to {output_path}")

        # Return path so next task knows where the file is (XCom)
        return output_path


    def verify_output(**context):
        """
        Pulls the file path from the previous task via XCom.
        Verifies the file exists and has data.
        """
        # XCom pull — get the return value from scrape_headlines
        ti = context["ti"]
        output_path = ti.xcom_pull(task_ids="scrape_headlines")

        print(f"Verifying file: {output_path}")

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output file not found: {output_path}")

        # Count rows
        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        print(f"Verification passed — {len(rows)} headlines in file")
        print(f"First headline: {rows[0]['title']}")
        print(f"Last headline:  {rows[-1]['title']}")

        if len(rows) == 0:
            raise ValueError("Output file is empty — scraping may have failed")

        return len(rows)


    # ── 3. Define tasks ───────────────────────────────────────

    task_scrape = PythonOperator(
        task_id="scrape_headlines",
        python_callable=scrape_headlines,
    )

    task_verify = PythonOperator(
        task_id="verify_output",
        python_callable=verify_output,
    )

    # ── 4. Set dependencies ───────────────────────────────────
    # scrape first, then verify
    task_scrape >> task_verify