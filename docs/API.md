# API Reference

**Base URL:** `http://localhost:8000`  
**Swagger UI:** `http://localhost:8000/docs`

Both endpoints accept an SMS message and return its classification (`spam` or `ham`). You can test both endpoints my accesing swagger UI once you have completed the docker build and docker start.

### Request

```json
{
  "sms": "WINNER!! You have won a 900 prize. Call 09061701461 to claim"
}
```

The `sms` field is required and must contain 1–300 characters.

### 1. ML Classifier – `POST /ml/classify`

Uses logistic regression (`ml_classifier.py`).

### 2. LLM Classifier – `POST /llm/classify`

Uses Google Gemma 3 12B through Amazon Bedrock (`llm_classifier.py`).

### Response

Both endpoints return the same format:

```json
{
  "label": "spam",
  "model": "tfidf-logreg"
}
```

`label` is either `spam` or `ham`. `model` identifies the classifier used.
