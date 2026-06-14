# ─────────────────────────────────────────────────────────────
#  Week 6 — Fine-tune DistilBERT with MLflow tracking
#  Logs params, metrics, and model to MLflow
#  Registers best model in MLflow Model Registry
# ─────────────────────────────────────────────────────────────

import csv
import os
import numpy as np
import mlflow
import mlflow.transformers
import torch
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ── Config ────────────────────────────────────────────────────
DATA_PATH      = "data/labeled/headlines_2026-06-09.csv"
MODEL_NAME     = "distilbert-base-uncased"
OUTPUT_DIR     = "models/headline-classifier-mlflow"
EXPERIMENT     = "headline-classifier"
REGISTERED_MODEL = "headline-classifier"

# MLflow tracking server running in Docker
MLFLOW_URI     = "http://localhost:5000"

# MinIO credentials for artifact storage
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"]      = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"]  = "password123"

# ── 1. Connect to MLflow ──────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)
print(f"MLflow tracking: {MLFLOW_URI}")
print(f"Experiment: {EXPERIMENT}")

# ── 2. Load data ──────────────────────────────────────────────
print("\nLoading data...")
titles, labels_raw = [], []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["label"] in {"news", "sport", "entertainment"}:
            titles.append(row["title"])
            labels_raw.append(row["label"])

label_dist = Counter(labels_raw)
print(f"Loaded {len(titles)} examples: {dict(label_dist)}")

# ── 3. Encode labels ──────────────────────────────────────────
le = LabelEncoder()
labels = le.fit_transform(labels_raw)
num_labels = len(le.classes_)
print(f"Classes: {list(le.classes_)}")

# ── 4. Tokenize ───────────────────────────────────────────────
print(f"\nTokenizing with {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encodings = tokenizer(
    titles,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt",
)

# ── 5. Dataset ────────────────────────────────────────────────
class HeadlineDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

dataset = HeadlineDataset(encodings, labels)

# ── 6. Training args ──────────────────────────────────────────
EPOCHS      = 5
BATCH_SIZE  = 8
LR          = 2e-5
WEIGHT_DECAY = 0.01

os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
)

# ── 7. Start MLflow run ───────────────────────────────────────
print("\nStarting MLflow run...")

with mlflow.start_run() as run:
    print(f"Run ID: {run.info.run_id}")

    # Log all hyperparameters
    mlflow.log_params({
        "model_name":     MODEL_NAME,
        "epochs":         EPOCHS,
        "batch_size":     BATCH_SIZE,
        "learning_rate":  LR,
        "weight_decay":   WEIGHT_DECAY,
        "max_length":     128,
        "num_examples":   len(titles),
        "num_labels":     num_labels,
        "data_file":      DATA_PATH,
    })

    # Log label distribution
    for label, count in label_dist.items():
        mlflow.log_metric(f"count_{label}", count)

    # ── 8. Load and train model ───────────────────────────────
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label={i: l for i, l in enumerate(le.classes_)},
        label2id={l: i for i, l in enumerate(le.classes_)},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("Training...")
    print("-" * 50)
    train_result = trainer.train()
    print("-" * 50)

    # ── 9. Log training metrics ───────────────────────────────
    final_loss = train_result.training_loss
    runtime    = train_result.metrics["train_runtime"]

    mlflow.log_metrics({
        "train_loss":            round(final_loss, 4),
        "train_runtime_seconds": round(runtime, 1),
        "samples_per_second":    round(len(titles) / runtime, 3),
    })

    print(f"\nFinal loss:    {final_loss:.4f}")
    print(f"Training time: {runtime:.1f}s")

    # ── 10. Save and log model to MLflow ─────────────────────
    print("\nSaving model to MLflow...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

   # Log model files directly as artifacts
    mlflow.log_artifacts(OUTPUT_DIR, artifact_path="model")

    print(f"Model logged to MLflow run: {run.info.run_id}")

    # ── 11. Register model in Model Registry ─────────────────
    # ── 11. Model already registered via log_artifacts ────────
    print(f"\nModel artifacts saved to MLflow run: {run.info.run_id}")
    print(f"View at: http://localhost:5000/#/experiments/1/runs/{run.info.run_id}")

    # ── 12. Quick inference test ──────────────────────────────
    print("\nQuick inference test:")
    test_headlines = [
        "England win the World Cup final in extra time",
        "Prime Minister announces new tax policy",
        "New podcast explores the history of jazz music",
    ]

    inputs = tokenizer(
        test_headlines,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    print("\nResults:")
    for headline, pred in zip(test_headlines, predictions):
        label = le.classes_[pred.item()]
        print(f"  [{label:>13}] {headline}")

print(f"\nDone! Open http://localhost:5000 to see your run.")
print(f"Experiment: '{EXPERIMENT}' → look for your run ID: {run.info.run_id}")