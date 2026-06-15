# 🚀 MailMind AI — Deployment & Operations Guide

This guide covers everything needed to install, reproduce, run, containerise and
operate **MailMind AI — Your Inbox, Intelligently Organized**. All commands are
copy-pasteable and assume you start from the repository root unless stated
otherwise.

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python**  | 3.9 or newer (declared in `pyproject.toml` as `requires-python = ">=3.9"`). Developed and tested on **Python 3.12**. |
| **pip**     | A recent `pip` (≥ 23) is recommended. Upgrade with `python -m pip install --upgrade pip`. |
| **OS**      | Linux, macOS or Windows. SQLite ships with the Python standard library, so no external database server is required. |
| **Disk**    | A few hundred MB for the virtual environment and NLTK data. |
| **Optional**| spaCy (richer keyword extraction) and Playwright (screenshot capture) are optional and the code degrades gracefully without them. |

No GPU is required — the production classifier is a CPU-friendly Logistic
Regression model.

---

## 2. Local Setup

### 2a. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

> On **Windows**, activate the environment with `.venv\Scripts\activate` instead of the
> `source` line.

### 2b. Install the package and dependencies

```bash
pip install -e .
pip install -r requirements.txt
```

`pip install -e .` installs MailMind in editable/development mode (from
`pyproject.toml`); `pip install -r requirements.txt` adds the full runtime + dev stack
(API, UI, visualisation, tests).

### 2c. Download the NLTK corpora

The NLP layer uses VADER (sentiment), the Punkt tokenizer and WordNet
(lemmatisation). Fetch them once:

```bash
python scripts/download_nltk.py
```

### 2d. Alternative — run without installing (`PYTHONPATH=src`)

The project uses a `src/` layout. If you prefer not to `pip install -e .`, you
can expose the package at runtime by prefixing commands with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python scripts/train_model.py
PYTHONPATH=src pytest -q
```

The provided shell launchers (`scripts/run_api.sh`, `scripts/run_ui.sh`) already
set `PYTHONPATH=src` for you, so they work whether or not the package is
installed. You still need the third-party dependencies from
`requirements.txt` either way.

---

## 3. Reproduce From Scratch

Run the following commands **in order** to regenerate the dataset, train the
model, verify the test suite, and produce the demo outputs. The whole sequence
is deterministic (`seed = 42`), so results are repeatable.

```bash
python scripts/generate_data.py
python scripts/train_model.py
PYTHONPATH=src pytest -q
python scripts/demo.py
python scripts/sample_outputs.py
```

In order, these steps: **(1)** build the 4,200-email synthetic dataset (700/category),
**(2)** train + evaluate and select the best model (Logistic Regression), **(3)** run the
44-test suite (all should pass), **(4)** run a quick end-to-end agent demo, and **(5)**
print the 8-email priority-sorted demo inbox.

What each step produces:

- **`generate_data.py`** → writes the dataset CSV under `data/` (stratified
  80/20 → 3,360 train / 840 test, 140 per class).
- **`train_model.py`** → fits TF-IDF + 8 engineered features, compares
  candidate models, and persists the best classifier and metrics into
  `models/` (`mailmind_classifier.joblib`, `metrics.json`,
  `model_comparison.csv`). Expected best result: **accuracy ≈ 0.9095,
  macro-F1 ≈ 0.9095** (Logistic Regression).
- **`pytest -q`** → exercises the data, ML, NLP, behavioural, context, agent,
  DB and API layers (44 tests).
- **`demo.py` / `sample_outputs.py`** → human-readable agent output sorted by
  priority score.

> **Tip:** if you ran `pip install -e .` you may drop the `PYTHONPATH=src`
> prefix from the `pytest` step.

---

## 4. Run the REST API

The HTTP service is a FastAPI app exposing the agent. Launch it with the helper
script (which sets `PYTHONPATH=src` and runs Uvicorn with hot-reload on port
8000):

```bash
bash scripts/run_api.sh
```

This is equivalent to `PYTHONPATH=src uvicorn mailmind.api.main:app --reload --port 8000`.
Interactive Swagger documentation is then available at:

- **http://localhost:8000/docs**

### Endpoints

| Method | Path             | Purpose |
|--------|------------------|---------|
| `GET`  | `/health`        | Liveness probe; returns version and the active classifier type. |
| `POST` | `/classify`      | Classify a single email into one of the six categories. |
| `POST` | `/analyze`       | Return NLP signals (keywords, sentiment, urgency, intent). |
| `POST` | `/process`       | Run the **full** agent pipeline over one email (classification → NLP → priority → actions). |
| `POST` | `/process_inbox` | Run the full pipeline over a batch, returned **sorted by priority**. |
| `POST` | `/feedback`      | Record a user action (replied / opened / ignored / deleted) so the agent adapts. |
| `GET`  | `/stats`         | Aggregate statistics about processed mail and feedback. |

### Example — full pipeline over one email (`/process`)

```bash
curl -s -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
        "subject": "URGENT: Production outage needs your approval",
        "body": "The payment service is down. Please approve the hotfix deploy now.",
        "sender": "oncall@company.com",
        "has_attachment": false,
        "num_links": 0
      }'
```

The response includes the predicted category, priority score and band, NLP
signals, an extractive summary, suggested actions and any urgent/VIP/spam flags.

> **Note:** the agent (and its SQLite DB) is built lazily on the first request,
> so the very first call may take a moment longer than subsequent ones.

---

## 5. Run the Dashboard

The Streamlit dashboard provides an interactive inbox, an email-detail panel and
an analytics view:

```bash
bash scripts/run_ui.sh
```

This is equivalent to `PYTHONPATH=src streamlit run app/streamlit_app.py`. Open the app at:

- **http://localhost:8501**

Click the **“Load demo inbox”** button to populate the dashboard with the
sample 8-email inbox; emails are triaged and shown sorted by priority, and
selecting a row opens its detailed insight (category, priority band, signals,
summary and suggested actions).

---

## 6. Docker

### 6a. Sample `Dockerfile` (API service)

Create a `Dockerfile` in the repository root:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# System certs + build basics (slim images are minimal).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project and install.
COPY . /app
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir -r requirements.txt \
    && python scripts/download_nltk.py

# Pre-build the model so the container is ready to serve.
RUN python scripts/generate_data.py && python scripts/train_model.py

EXPOSE 8000

# Bind to 0.0.0.0 so the port is reachable from outside the container.
CMD ["uvicorn", "mailmind.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6b. Build and run

```bash
docker build -t mailmind-ai .
docker run --rm -p 8000:8000 mailmind-ai
```

The API is then reachable at **http://localhost:8000/docs**, exactly as in the
local setup.

### 6c. Running the dashboard in a second container

The Streamlit UI can run as its own container from the same image — just
override the command and expose port 8501:

```bash
docker run --rm -p 8501:8501 mailmind-ai \
  streamlit run app/streamlit_app.py \
    --server.address 0.0.0.0 --server.port 8501
```

Browse to **http://localhost:8501**. For a single combined stack, define both
services (API on 8000, UI on 8501) in a `docker-compose.yml`; if you want the UI
container to talk to the API container, point it at the API service’s hostname
rather than `localhost`.

---

## 7. Configuration

All paths, labels, hyper-parameters and scoring weights are centralised in:

- **`src/mailmind/config.py`**

Key items defined there:

| Area | What lives in `config.py` |
|------|---------------------------|
| **Filesystem** | `ROOT_DIR`, `DATA_DIR`, `MODELS_DIR`, `DOCS_DIR`, `FIGURES_DIR` and concrete paths such as `DATASET_PATH`, `MODEL_PATH`, `METRICS_PATH`, `DB_PATH`. Paths are resolved relative to the repo root, so the project works from any working directory. |
| **Labels** | The six categories (`Important`, `Work`, `Personal`, `Social`, `Promotions`, `Spam`) and their UI colour map. |
| **Reproducibility** | `RANDOM_SEED = 42`. |
| **Weights** | Priority weights (category 0.34, urgency 0.24, sender 0.22, behaviour 0.12, freshness 0.08), category base-importance values, behavioural action weights and band thresholds. |

To re-tune the system, edit the relevant values in `config.py` and re-run
training (`python scripts/train_model.py`).

### Environment notes

- **`PYTHONPATH=src`** — only needed if you did *not* `pip install -e .`. The
  launcher scripts set it automatically.
- **SQLite** — the database file (`mailmind.db`) is created next to the repo
  root on first use; no server or connection string is required.
- **Ports** — the API defaults to `8000` and the dashboard to `8501`. Override
  with `--port` (Uvicorn) or `--server.port` (Streamlit) if those ports are
  busy.

---

## 8. Troubleshooting

| Symptom | Cause & Fix |
|---------|-------------|
| **`LookupError` / NLTK data not found** (VADER, punkt or WordNet) | The NLTK corpora are not downloaded. Run `python scripts/download_nltk.py`. |
| **“Model not found” / agent uses simple rules** | The trained classifier (`models/mailmind_classifier.joblib`) is missing. Run training first: `python scripts/train_model.py`. **Until a model exists, the agent automatically falls back to the rule-based `HeuristicClassifier`**, so the system still works (at lower accuracy ≈ 0.7214) — it just won’t use the ML model. |
| **`ModuleNotFoundError: mailmind`** | The package isn’t on the path. Either `pip install -e .` or prefix the command with `PYTHONPATH=src`. |
| **spaCy warnings / “spaCy not available”** | spaCy is **optional**. Keyword extraction falls back to the TF/positional heuristic. To enable the richer path, `pip install spacy` and download a model (e.g. `python -m spacy download en_core_web_sm`). |
| **Port already in use** | Start the API or UI on a different port (`uvicorn ... --port 8010`, `streamlit run ... --server.port 8510`). |
| **First API request is slow** | Expected — the agent and DB are constructed lazily on the first call, then cached for subsequent requests. |

---

## 9. Capturing the Dashboard Screenshots

The dashboard figures under `docs/screenshots/` (`dashboard_inbox.png`,
`dashboard_detail.png`, `dashboard_analytics.png`) are generated with
Playwright via `scripts/capture_screenshots.py`.

1. **Start the dashboard** in one terminal:

   ```bash
   bash scripts/run_ui.sh
   ```

2. **Install Playwright + Chromium** (one-off) in another terminal:

   ```bash
   pip install playwright
   python -m playwright install chromium
   ```

3. **Run the capture script** (it drives the running app, clicks “Load demo
   inbox”, and writes the PNGs into `docs/screenshots/`):

   ```bash
   PYTHONPATH=src python scripts/capture_screenshots.py
   ```

   To point at a different URL, pass `--url`:

   ```bash
   PYTHONPATH=src python scripts/capture_screenshots.py --url http://localhost:8501
   ```

The static evaluation figures (`confusion_matrix.png`, `per_class_metrics.png`,
`category_distribution.png`, `model_comparison.png`, `architecture.png`) are
produced by the training/visualisation scripts rather than Playwright and do not
require the dashboard to be running.
