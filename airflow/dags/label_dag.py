# ─────────────────────────────────────────────────────────────
#  DAG 3 — Headline Labeler
#  Merges raw (has link) + cleaned (has clean text) data
#  Assigns category labels based on BBC URL path
#  Saves to data/labeled/headlines_YYYY-MM-DD.csv
# ─────────────────────────────────────────────────────────────

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import csv
import os

with DAG(
    dag_id="headline_labeler",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["labeler", "week4"],
    description="Labels headlines based on BBC URL category",
) as dag:

    def assign_label(link):
        """
        Looks at the URL path to determine category.
        This is 'weak labeling' - using existing metadata
        as a proxy for the label we actually want.
        """
        link = link.lower()

        if "/sport/" in link:
            return "sport"
        elif "/news/" in link:
            return "news"
        elif "/sounds/" in link:
            return "entertainment"
        else:
            return "other"


    def label_headlines(**context):
        """
        1. Read raw CSV (has 'link' column)
        2. Read cleaned CSV (has clean 'title'/'description')
        3. Build a lookup: original title -> link
        4. For each cleaned row, find its link and assign a label
        5. Save labeled CSV
        """
        today = context["ds"]

        raw_path     = f"/opt/airflow/data/raw/headlines_{today}.csv"
        cleaned_path = f"/opt/airflow/data/cleaned/headlines_{today}.csv"
        output_dir   = "/opt/airflow/data/labeled"
        output_path  = f"{output_dir}/headlines_{today}.csv"

        os.makedirs(output_dir, exist_ok=True)

        # ── Step 1: Read raw data, build title -> link lookup ──
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_rows = list(csv.DictReader(f))

        print(f"Raw rows: {len(raw_rows)}")

        # Build lookup using a normalized version of the title
        # (raw titles may have unicode issues, cleaned titles don't)
        link_lookup = {}
        for row in raw_rows:
            # Normalize the same way clean_dag does, just enough to match
            normalized_title = (
                row["title"]
                .encode("ascii", errors="ignore")
                .decode("ascii")
                .strip()
            )
            link_lookup[normalized_title] = row["link"]

        # ── Step 2: Read cleaned data ──────────────────────────
        with open(cleaned_path, "r", encoding="utf-8") as f:
            cleaned_rows = list(csv.DictReader(f))

        print(f"Cleaned rows: {len(cleaned_rows)}")

        # ── Step 3: Match + label ──────────────────────────────
        labeled_rows = []
        unmatched = 0

        for row in cleaned_rows:
            link = link_lookup.get(row["title"], "")

            if not link:
                unmatched += 1
                label = "other"  # couldn't find link, default label
            else:
                label = assign_label(link)

            labeled_rows.append({
                "title": row["title"],
                "description": row["description"],
                "pub_date": row["pub_date"],
                "label": label,
            })

        print(f"Unmatched titles (defaulted to 'other'): {unmatched}")

        # ── Step 4: Print label distribution ───────────────────
        label_counts = {}
        for row in labeled_rows:
            label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1

        print("Label distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}")

        # ── Step 5: Save ────────────────────────────────────────
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["title", "description", "pub_date", "label"]
            )
            writer.writeheader()
            writer.writerows(labeled_rows)

        print(f"Saved labeled file to: {output_path}")
        return output_path


    def verify_labels(**context):
        """
        Verifies the labeled file exists, has the label column,
        and that labels are from our known set.
        """
        ti = context["ti"]
        output_path = ti.xcom_pull(task_ids="label_headlines")

        with open(output_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if len(rows) == 0:
            raise ValueError("Labeled file is empty!")

        valid_labels = {"sport", "news", "entertainment", "other"}

        for row in rows:
            if row["label"] not in valid_labels:
                raise ValueError(f"Invalid label found: {row['label']}")

        print(f"Verified {len(rows)} labeled rows")
        print("Sample rows:")
        for row in rows[:5]:
            print(f"  [{row['label']:>13}] {row['title']}")

        return len(rows)


    # ── Tasks ─────────────────────────────────────────────────
    task_label = PythonOperator(
        task_id="label_headlines",
        python_callable=label_headlines,
    )

    task_verify = PythonOperator(
        task_id="verify_labels",
        python_callable=verify_labels,
    )

    task_label >> task_verify
