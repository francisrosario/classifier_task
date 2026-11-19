# SMS Classification

Check if a SMS message is a **spam** or **ham** (a normal message).

1. **ML:** logistic regression (scikit-learn).
2. **LLM:** Google Gemma 3 12B on Amazon Bedrock.

Both are served as an API by one Docker container.

## Run it (Docker)

You need [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Building + Starting

```bash
cp .env.example .env          # The AWS key you were given, I've sent this via email for demonstration.
docker compose up --build     # Thr build takes about 2-10 minutes
```

### Rebuilding + Restarting

```bash
docker compose down           # stops and removes the running container
docker compose up --build     # rebuilds starts again
```

The APIs are ready when you see `Uvicorn running on http://0.0.0.0:8000`.

- Interactive API page: http://localhost:8000/docs You can try both endpoints by clicking 'Try Out' but click which classifier you wanted to try either /ml/classify or /llm/classify.
- Try both APIs from another terminal (needs Python with `httpx`, `pip install httpx==0.28.1`):

  ```bash
  python sample_client.py
  ```

## The two endpoints:

| Method | Path | Body | Answer |
|---|---|---|---|
| POST | `/ml/classify` | `{"sms": "..."}` | `{"label": "spam", "model": "tfidf-logreg"}` |
| POST | `/llm/classify` | `{"sms": "..."}` | `{"label": "ham", "model": "google.gemma-3-12b-it"}` |

Full details, errors and examples: [docs/API.md](docs/API.md).

## Task 1 results

| | ML (logistic regression) | LLM (Bedrock Gemma 3 12B) |
|---|---|---|
| Accuracy | **97.7%** | 93.6% |
| Spam precision | **0.908** | 0.668 |
| Spam recall | 0.908 | **0.985** |
| Spam F1 | **0.908** | 0.796 |
| Time per message | about 0.01 ms | about 0.5 s |

The ML model is the better choice when identifying spam / ham. The LLM catches every spam but it also flags normal messages as spam. 

## Run without Docker (development)

Python 3.11:

```bash
python -m venv .venv
.venv\Scripts\activate                       # Windows (Mac/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt
pytest
uvicorn api:app --port 8000 --env-file .env  # --env-file loads your AWS key from .env
```
