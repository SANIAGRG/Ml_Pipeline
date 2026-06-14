# ML Data Pipeline 

This project is a fully automated machine learning pipeline that scrapes live news headlines from BBC every day, cleans and labels them, trains a text classification model, tracks every experiment, and serves predictions through a REST API. The entire system runs locally on your machine using Docker — no cloud account or paid services required. It was built week by week as a hands-on learning project covering the full spectrum of modern ML engineering, from raw data ingestion all the way to model serving.

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        AIRFLOW SCHEDULER                        │
│                    (runs every night at midnight)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │         news_scraper DAG         │
          │   BBC RSS Feed → raw CSV         │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │        headline_cleaner DAG      │
          │   Fix unicode · dedupe · strip   │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │        headline_labeler DAG      │
          │   URL path → news/sport/ent.     │
          └────────────────┬────────────────┘
                           │
               ┌───────────▼───────────┐
               │     DVC + MinIO        │
               │  Version every dataset │
               └───────────┬───────────┘
                           │
               ┌───────────▼───────────┐
               │   DistilBERT Training  │
               │   HuggingFace Trainer  │
               └───────────┬───────────┘
                           │
          ┌────────────────▼────────────────┐
          │         MLflow Tracking          │
          │  Log params · metrics · model    │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │       FastAPI /predict           │
          │  POST text → { label, score }    │
          └─────────────────────────────────┘
```

Every night, an Airflow scheduler triggers a scraping task that pulls the latest BBC News RSS feed and saves the headlines as a CSV. A second task cleans the raw text — fixing broken unicode characters, removing duplicates, and stripping HTML artifacts. A third task labels each headline automatically using the BBC's own URL structure as a signal: articles under `/sport/` are labeled sport, `/news/` become news, and `/sounds/` become entertainment. This is a technique called weak supervision — using metadata that already exists as a free proxy for ground-truth labels, which is how many real ML teams bootstrap labeled datasets before investing in human annotation.

---

## Model Training and Experiment Tracking

Once labeled data is available, a DistilBERT model is fine-tuned on the headlines using the HuggingFace Transformers library. DistilBERT is a compressed version of Google's BERT — 40% smaller, 60% faster, and retaining 97% of the original accuracy — making it practical to train on a laptop CPU in a few minutes. Every training run is tracked with MLflow, which records the hyperparameters used (learning rate, epochs, batch size), the metrics produced (training loss, runtime), and the model artifacts. This makes it possible to compare runs side by side and always know exactly what data and config produced any given model.

---

## Tech Stack

```
┌──────────────┬─────────────────────────────────┬─────────────────────┐
│    Layer      │           Tool                  │      Role           │
├──────────────┼─────────────────────────────────┼─────────────────────┤
│ Orchestration │ Apache Airflow                  │ Schedule & monitor  │
│ Infrastructure│ Docker + Docker Compose         │ Run all services    │
│ Storage       │ MinIO (local S3)                │ Store data + models │
│ Versioning    │ DVC                             │ Version datasets    │
│ ML            │ HuggingFace Transformers        │ Fine-tune DistilBERT│
│ Tracking      │ MLflow                          │ Log experiments     │
│ Serving       │ FastAPI                         │ REST API inference  │
│ Database      │ Postgres                        │ Airflow metadata    │
└──────────────┴─────────────────────────────────┴─────────────────────┘
```

All datasets and trained models are versioned with DVC (Data Version Control), which works like Git but for large files. The actual data lives in MinIO — a locally hosted, S3-compatible object store — while Git only tracks tiny pointer files. This means anyone who clones the repo can run `dvc pull` and get the exact dataset that produced any model, without large files ever bloating the repository. The full infrastructure — Airflow, MinIO, MLflow, Postgres, and a FastAPI inference server — is defined in a single `docker-compose.yml` and starts with one command. Every tool in this stack is free and open source.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/SANIAGRG/Ml_Pipeline.git
cd Ml_Pipeline

# 2. Create your .env file with your own credentials (never commit this)

# 3. Start all services
docker compose up airflow-init   # first time only
docker compose up -d

# 4. Pull versioned data
pip install dvc dvc-s3
dvc pull

# 5. Train the model
pip install transformers torch scikit-learn datasets accelerate mlflow boto3
python scripts/train_mlflow.py

# 6. Open the UIs
# Airflow  → http://localhost:8080
# MinIO    → http://localhost:9001
# MLflow   → http://localhost:5000
# FastAPI  → http://localhost:8000/docs
```

---

## License

MIT — free to use, modify, and learn from.