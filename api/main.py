from fastapi import FastAPI

app = FastAPI(title="ML Pipeline API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(text: str):
    return {"message": "Model not loaded yet — coming in Week 7", "input": text}
