# ---- Builder Stage ----
# This stage installs all dependencies and creates the final requirements.txt
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install pip-tools
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pip-tools

# Copy requirements files
COPY requirements.in .
COPY requirements-dev.in .

# Generate the full requirements.txt from both files
# This creates a single, pinned requirements file for the final image
RUN pip-compile -o requirements.txt requirements.in requirements-dev.in


# ---- App Stage ----
# This is the final, minimal image that will be deployed
FROM python:3.11-slim AS app

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Set path for the model to be loaded from
    FRAUD_MODEL_PATH=/app/fraud_model.pkl \
    LOG_LEVEL=INFO

# Create a non-root user
RUN addgroup --system app && adduser --system --group app

# Copy compiled requirements and install only production dependencies
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src ./src

# Copy the model artifact
# In a real-world scenario, this would likely be downloaded from a model registry
# or S3 at runtime, not baked into the image.
COPY fraud_model.pkl ./fraud_model.pkl

# Change ownership of the app directory to the non-root user
RUN chown -R app:app /app

# Switch to the non-root user
USER app

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]


