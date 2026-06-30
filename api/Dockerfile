FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model files will be mounted as a volume (see docker-compose.yml)
ENV MODEL_PATH=/app/models/headline-classifier

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]