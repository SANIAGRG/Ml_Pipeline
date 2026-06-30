from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/headline-classifier")

app = FastAPI(title="Headline Classifier API", version="1.0.0")

tokenizer = None
model = None
id2label = None


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    all_scores: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.on_event("startup")
def load_model():
    global tokenizer, model, id2label

    print(f"Loading model from: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model path does not exist: {MODEL_PATH}")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    id2label = model.config.id2label

    print(f"Model loaded successfully")
    print(f"Labels: {list(id2label.values())}")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check server logs.",
        )

    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text field cannot be empty.",
        )

    inputs = tokenizer(
        request.text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )

    # DistilBERT doesn't accept token_type_ids - remove if present
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    predicted_idx = torch.argmax(probabilities).item()
    predicted_label = id2label[predicted_idx]
    confidence = probabilities[predicted_idx].item()

    all_scores = {
        id2label[i]: round(probabilities[i].item(), 4)
        for i in range(len(id2label))
    }

    return PredictResponse(
        label=predicted_label,
        confidence=round(confidence, 4),
        all_scores=all_scores,
    )


@app.get("/")
def root():
    return {
        "message": "Headline Classifier API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict (POST)",
    }