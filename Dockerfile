# ============================================================================
# MailMind AI — container image (serves the FastAPI REST API by default).
# Build:  docker build -t mailmind-ai .
# Run  :  docker run -p 8000:8000 mailmind-ai
# ============================================================================
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System build deps (kept minimal; wheels cover most of the stack).
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching.
COPY pyproject.toml requirements.txt setup.py ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e . && pip install -r requirements.txt

# NLP corpora (VADER, punkt, WordNet, stopwords).
COPY scripts ./scripts
RUN python scripts/download_nltk.py

# Application code, the trained model and the dataset.
COPY app ./app
COPY data ./data
COPY models ./models

# Train at build time if no model artefact was copied in (keeps the image self-contained).
RUN test -f models/mailmind_classifier.joblib || python scripts/train_model.py

EXPOSE 8000
CMD ["uvicorn", "mailmind.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
