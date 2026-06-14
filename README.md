# ML Data Pipeline — End to End

A production-style ML pipeline built entirely with free, open-source tools. Scrapes real news headlines daily, cleans and labels them automatically, fine-tunes a DistilBERT text classifier, tracks every experiment, and serves predictions via a REST API.

**Built as a learning project** — every week added one layer to the stack, documented in detail.

---

## What it does

```
BBC News RSS Feed
        ↓
Airflow scraper DAG          → data/raw/headlines_DATE.csv
        ↓
Airflow cleaner DAG          → data/cleaned/headlines_DATE.csv
        ↓
Airflow labeler DAG          → data/labeled/headlines_DATE.csv
        ↓  (weak labeling via URL path)
DVC + MinIO                  → versioned data stored in object storage
        ↓
DistilBERT fine-tuning       → models/headline-classifier/
        ↓  (HuggingFace Trainer)
MLflow experiment tracking   → params, metrics, artifacts logged
        ↓
FastAPI inference server     → POST /predict → { label, confidence }
```

---

## Tech stack

| Tool | Purpose | Why free |
|---|---|---|
| Apache Airflow | Pipeline scheduling and orchestration | Open source (Apache 2.0) |
| Docker + Docker Compose | Runs all services consistently | Free to use |
| HuggingFace Transformers | DistilBERT fine-tuning | Open source (Apache 2.0) |
| MLflow | Experiment tracking + model registry | Open source (Apache 2.0) |
| MinIO | S3-compatible object storage for DVC artifacts | Community Edition |
| DVC | Data version control | Open source (Apache 2.0) |
| FastAPI | Model serving via REST API | Open source (MIT) |
| Postgres | Airflow metadata database | Open source |

---

## Services and URLs

| Service | URL | Credentials |
|---|---|---|
| Airflow UI | http://localhost:8080 | admin / admin |
| MinIO UI | http://localhost:9001 | admin / password123 |
| MLflow UI | http://localhost:5000 | — |
| FastAPI docs | http://localhost:8000/docs | — |

---

## Project structure

```
ml-pipeline/
├── airflow/
│   ├── dags/
│   │   ├── scrape_dag.py        # DAG 1: scrapes BBC RSS feed daily
│   │   ├── clean_dag.py         # DAG 2: cleans raw headlines
│   │   └── label_dag.py         # DAG 3: assigns category labels
│   ├── logs/                    # Airflow task logs (gitignored)
│   └── plugins/
├── api/
│   ├── Dockerfile               # FastAPI container definition
│   ├── main.py                  # /predict and /health endpoints
│   └── requirements.txt
├── data/
│   ├── raw/                     # Scraped CSVs (DVC tracked)
│   ├── cleaned/                 # Cleaned CSVs (DVC tracked)
│   └── labeled/                 # Labeled CSVs (DVC tracked)
├── models/
│   └── headline-classifier/     # Trained DistilBERT (DVC tracked)
├── scripts/
│   ├── train.py                 # Basic training script (Week 5)
│   └── train_mlflow.py          # Training with MLflow tracking (Week 6)
├── notebooks/                   # Exploration and experiments
├── docker-compose.yml           # Defines all services
├── .env                         # Secrets and config (gitignored)
├── .dvc/                        # DVC config (points to MinIO)
└── .gitignore
```

---

## Quick start

### Prerequisites
- Docker Desktop installed and running
- Git
- Python 3.11+

### 1. Clone the repo

```bash
git clone https://github.com/SANIAGRG/Ml_Pipeline.git
cd Ml_Pipeline
```

### 2. Create the .env file

Create a `.env` file in the project root:

```
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=password123
MINIO_ENDPOINT=http://minio:9000

AIRFLOW_UID=50000
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__CORE__FERNET_KEY=your-fernet-key-here
AIRFLOW__WEBSERVER__SECRET_KEY=your-secret-key-here

MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_S3_ENDPOINT_URL=http://minio:9000
AWS_ACCESS_KEY_ID=admin
AWS_SECRET_ACCESS_KEY=password123

POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=airflow
```

### 3. Start all services

```bash
# First time only — initialize Airflow database
docker compose up airflow-init

# Start everything
docker compose up -d

# Verify all services are running
docker compose ps
```

### 4. Pull data from DVC (get the versioned datasets)

```bash
pip install dvc dvc-s3

# Configure DVC to point to your MinIO instance
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify minio access_key_id admin
dvc remote modify minio secret_access_key password123

# Pull all tracked data
dvc pull
```

### 5. Train the model

```bash
pip install transformers torch scikit-learn datasets accelerate mlflow boto3

# Basic training
python scripts/train.py

# Training with MLflow experiment tracking
python scripts/train_mlflow.py
```

### 6. Run the pipeline

Open **http://localhost:8080**, log in with `admin / admin`, and trigger the DAGs in order:
1. `news_scraper` — scrapes today's headlines
2. `headline_cleaner` — cleans the raw data
3. `headline_labeler` — assigns category labels

---

## How data versioning works

Data files are tracked with DVC and stored in MinIO. Git only stores tiny `.dvc` pointer files. This means:

- Collaborators run `dvc pull` to get the exact dataset used for any model
- No large files ever go into Git
- Every dataset version is linked to the Git commit that produced it

```bash
# After adding new data
dvc add data/labeled/headlines_2026-06-09.csv
dvc push
git add data/labeled/headlines_2026-06-09.csv.dvc
git commit -m "Add labeled data for June 9"
git push
```

---

## How experiment tracking works

Every training run is recorded in MLflow at **http://localhost:5000**.

Each run logs:
- **Parameters**: model name, epochs, batch size, learning rate, data file used
- **Metrics**: training loss, runtime, samples per second, label distribution
- **Artifacts**: model weights, tokenizer, training config

To compare runs, open MLflow UI → `headline-classifier` experiment → select multiple runs → Compare.

---

## The labeling approach (weak supervision)

Labels are assigned automatically using BBC's URL structure as a proxy:

| URL contains | Label |
|---|---|
| `/sport/` | sport |
| `/news/` | news |
| `/sounds/` | entertainment |
| anything else | other |

This is called **weak supervision** — using existing metadata as a free proxy for labels. It is accurate for this use case because BBC's editorial team already categorizes articles by URL. With more days of data, the model improves automatically.

---

## Model details

- **Architecture**: DistilBERT (distilbert-base-uncased) with a classification head
- **Task**: Multi-class text classification (news / sport / entertainment)
- **Training data**: BBC News RSS headlines, daily accumulation
- **Fine-tuning**: HuggingFace Trainer API, 5 epochs, learning rate 2e-5
- **Inference**: ~10ms per headline on CPU

---

## Stopping and starting

```bash
# Stop all services (data is preserved)
docker compose down

# Start again
docker compose up -d

# Full reset (deletes all data)
docker compose down -v
```

DAGs restart automatically when Docker comes back up. No manual re-registration needed.

---

## Week by week build log

| Week | What was built |
|---|---|
| Week 1 | Docker infrastructure: Postgres, MinIO, Airflow, MLflow, FastAPI placeholder |
| Week 2 | Two Airflow DAGs: scraper (BBC RSS → CSV) and cleaner (fix unicode, deduplicate) |
| Week 3 | Data versioning: DVC + MinIO, .gitignore, dvc add/push/pull workflow |
| Week 4 | Labeling DAG: weak supervision via URL path, label distribution logging |
| Week 5 | HuggingFace fine-tuning: DistilBERT trained on labeled headlines, model saved |
| Week 6 | MLflow tracking: params + metrics logged per run, model in registry |
| Week 7 | FastAPI serving: /predict endpoint loads model, returns label + confidence |
| Week 8 | Full integration: Airflow triggers retraining when new labeled data arrives |

---

## Skills demonstrated

- **Data engineering**: Airflow DAG authoring, task dependencies, XCom, scheduling
- **Data versioning**: DVC with S3-compatible remote, reproducible datasets
- **NLP / ML**: HuggingFace fine-tuning, tokenization, DistilBERT classification
- **Experiment tracking**: MLflow params, metrics, artifacts, model registry
- **API development**: FastAPI, Pydantic, async inference endpoint
- **DevOps**: Docker Compose multi-service orchestration, environment management
- **Version control**: Git + DVC for code and data respectively

---

## License

MIT — free to use, modify, and learn from.
