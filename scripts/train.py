# ─────────────────────────────────────────────────────────────
#  Week 5 — Fine-tune DistilBERT on labeled headlines
#  Input:  data/labeled/headlines_2026-06-09.csv
#  Output: models/headline-classifier/
# ─────────────────────────────────────────────────────────────

import csv
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset

# ── 1. Load labeled data ──────────────────────────────────────
DATA_PATH  = "data/labeled/headlines_2026-06-09.csv"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/headline-classifier"

print("Loading data...")
titles = []
labels_raw = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["label"] in {"news", "sport", "entertainment"}:
            titles.append(row["title"])
            labels_raw.append(row["label"])

print(f"Loaded {len(titles)} examples")

# Label distribution
from collections import Counter
print("Label distribution:", Counter(labels_raw))

# ── 2. Encode labels as integers ──────────────────────────────
# DistilBERT needs numbers not strings:
# news=0, sport=1, entertainment=2 (order assigned by LabelEncoder)
le = LabelEncoder()
labels = le.fit_transform(labels_raw)
num_labels = len(le.classes_)

print(f"Classes: {le.classes_}")   # shows the mapping
print(f"Number of labels: {num_labels}")

# ── 3. Tokenize ───────────────────────────────────────────────
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

encodings = tokenizer(
    titles,
    truncation=True,       # cut text longer than 512 tokens
    padding=True,          # pad shorter texts to same length
    max_length=128,        # headlines are short, 128 is plenty
    return_tensors="pt",   # return PyTorch tensors
)

print(f"Tokenized {len(titles)} headlines")
print(f"Input shape: {encodings['input_ids'].shape}")

# ── 4. Create PyTorch Dataset ─────────────────────────────────
class HeadlineDataset(Dataset):
    """
    Wraps our tokenized data so HuggingFace Trainer can use it.
    __len__ returns total samples, __getitem__ returns one sample.
    """
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
print(f"Dataset size: {len(dataset)}")

# ── 5. Load model ─────────────────────────────────────────────
print(f"\nLoading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label={i: label for i, label in enumerate(le.classes_)},
    label2id={label: i for i, label in enumerate(le.classes_)},
)

# ── 6. Training arguments ─────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,           # passes through the full dataset
    per_device_train_batch_size=8,
    learning_rate=2e-5,           # standard for fine-tuning BERT models
    weight_decay=0.01,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=5,
    save_strategy="epoch",        # save checkpoint after each epoch
    report_to="none",             # don't send to wandb/tensorboard
)

# ── 7. Define metrics function ────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# ── 8. Create Trainer and train ───────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    compute_metrics=compute_metrics,
)

print("\nStarting training...")
print(f"Training on {len(dataset)} examples for {training_args.num_train_epochs} epochs")
print("-" * 50)

trainer.train()

# ── 9. Save final model ───────────────────────────────────────
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nModel saved to: {OUTPUT_DIR}")
print("Files saved:")
for f in os.listdir(OUTPUT_DIR):
    print(f"  {f}")

# ── 10. Quick test on training data ──────────────────────────
print("\nQuick test on sample headlines:")
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

print("\nTraining complete!")
print(f"Model saved to: {OUTPUT_DIR}")
print("Next step: Week 6 — log this run to MLflow")