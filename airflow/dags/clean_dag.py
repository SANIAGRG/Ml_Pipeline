# ─────────────────────────────────────────────────────────────
#  DAG 2 — Headline Cleaner
#  Reads raw CSV → cleans text → saves to data/cleaned/
# ─────────────────────────────────────────────────────────────

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import csv
import os
import re
import html

with DAG(
    dag_id="headline_cleaner",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["cleaner", "week2"],
    description="Cleans raw headlines and saves to data/cleaned/",
) as dag:

    def clean_text(text):
        """
        Applies all cleaning steps to a single string.
        Each step fixes one specific problem found in raw data.
        """
        if not text:
            return ""

        # Step 1 — fix broken unicode (â€¦ → …)
        text = text.encode("utf-8", errors="ignore").decode("utf-8")

        # Step 2 — decode HTML entities (&amp; → & , &quot; → ")
        text = html.unescape(text)

        # Step 3 — remove URLs
        text = re.sub(r"http\S+", "", text)

        # Step 4 — remove content inside square brackets e.g. [Video]
        text = re.sub(r"\[.*?\]", "", text)

        # Step 5 — remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Step 6 — remove non-ASCII characters that slipped through
        text = text.encode("ascii", errors="ignore").decode("ascii")

        return text


    def clean_headlines(**context):
        """
        Reads raw CSV for today's date.
        Applies cleaning to title and description.
        Removes duplicates.
        Saves cleaned CSV to data/cleaned/
        """
        today = context["ds"]
        input_path  = f"/opt/airflow/data/raw/headlines_{today}.csv"
        output_dir  = "/opt/airflow/data/cleaned"
        output_path = f"{output_dir}/headlines_{today}.csv"

        os.makedirs(output_dir, exist_ok=True)

        # Read raw data
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Raw file not found: {input_path}")

        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            raw_rows = list(reader)

        print(f"Raw rows loaded: {len(raw_rows)}")

        # Clean each row
        cleaned_rows = []
        for row in raw_rows:
            cleaned = {
                "title":       clean_text(row["title"]),
                "description": clean_text(row["description"]),
                "pub_date":    row["pub_date"].strip(),
                "scraped_at":  row["scraped_at"].strip(),
            }
            # Skip rows with empty title after cleaning
            if cleaned["title"]:
                cleaned_rows.append(cleaned)

        print(f"Rows after cleaning: {len(cleaned_rows)}")

        # Remove duplicates based on title
        seen_titles = set()
        deduped_rows = []
        for row in cleaned_rows:
            if row["title"] not in seen_titles:
                seen_titles.add(row["title"])
                deduped_rows.append(row)

        print(f"Rows after deduplication: {len(deduped_rows)}")
        duplicates_removed = len(cleaned_rows) - len(deduped_rows)
        print(f"Duplicates removed: {duplicates_removed}")

        # Save cleaned data
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["title", "description", "pub_date", "scraped_at"]
            )
            writer.writeheader()
            writer.writerows(deduped_rows)

        print(f"Saved cleaned file to: {output_path}")
        print(f"Sample cleaned title: {deduped_rows[0]['title']}")

        return output_path


    def verify_cleaned(**context):
        """
        Verifies the cleaned file exists and has fewer
        or equal rows than raw (deduplication worked).
        """
        ti = context["ti"]
        output_path = ti.xcom_pull(task_ids="clean_headlines")

        with open(output_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        print(f"Cleaned file has {len(rows)} rows")
        print("First 3 cleaned titles:")
        for row in rows[:3]:
            print(f"  → {row['title']}")

        if len(rows) == 0:
            raise ValueError("Cleaned file is empty!")

        return len(rows)


    # ── Tasks ─────────────────────────────────────────────────
    task_clean = PythonOperator(
        task_id="clean_headlines",
        python_callable=clean_headlines,
    )

    task_verify = PythonOperator(
        task_id="verify_cleaned",
        python_callable=verify_cleaned,
    )

    task_clean >> task_verify