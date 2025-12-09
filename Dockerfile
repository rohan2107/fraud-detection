FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FRAUD_MODEL_PATH=/app/fraud_model.pkl \
    DATA_PATH=/app/data/creditcard.csv \
    LOG_LEVEL=INFO

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY tests/samples ./tests/samples
COPY fraud_model.pkl ./fraud_model.pkl

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]


