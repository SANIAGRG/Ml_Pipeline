# ─────────────────────────────────────────────────────────────
#  FastAPI Inference Server
#  Loads the trained DistilBERT model once at startup,
#  serves predictions via POST /predict
# ─────────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# ── Config ────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/headline-classifier")

app = FastAPI(
    title="Headline Classifier API",
    description="Classifies news headlines as news, sport, or entertainment",
    version="1.0.0",
)

# ── Global variables for the loaded model ───────────────────────
# Loaded once at startup, reused for every request
tokenizer = None
model = None
id2label = None


# ── Request / Response schemas ──────────────────────────────────
class PredictRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {"text": "England win the World Cup final in extra time"}
        }


class PredictResponse(BaseModel):
    label: str
    confidence: float
    all_scores: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ── Startup event: load model once ──────────────────────────────
@app.on_event("startup")
def load_model():
    global tokenizer, model, id2label

    print(f"Loading model from: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model path does not exist: {MODEL_PATH}")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()  # set to inference mode (disables dropout etc.)

    # id2label comes from the model's config (set during training)
    id2label = model.config.id2label

    print(f"Model loaded successfully")
    print(f"Labels: {list(id2label.values())}")


# ── Health check endpoint ────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    """
    Returns whether the server is alive and the model loaded correctly.
    Used by monitoring tools to check the service is healthy.
    """
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
    )


# ── Prediction endpoint ──────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Takes headline text, returns predicted category with confidence.
    """
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

    # Tokenize the input text
    inputs = tokenizer(
        request.text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )

    # Run inference (no_grad disables gradient tracking - faster, less memory)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    # Get the predicted class and its confidence
    predicted_idx = torch.argmax(probabilities).item()
    predicted_label = id2label[predicted_idx]
    confidence = probabilities[predicted_idx].item()

    # Build the full score breakdown for all classes
    all_scores = {
        id2label[i]: round(probabilities[i].item(), 4)
        for i in range(len(id2label))
    }

    return PredictResponse(
        label=predicted_label,
        confidence=round(confidence, 4),
        all_scores=all_scores,
    )


# ── Root endpoint ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Headline Classifier API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict (POST)",
    }