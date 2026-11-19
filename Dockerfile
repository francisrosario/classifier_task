# One image with both classifiers. No training happens here: the ML model is copied in ready-made,
# and the LLM runs on Amazon Bedrock (AWS credentials are passed in at start, never stored here).
# Official Python image, pinned to an exact version and digest so every build gets the same base.
FROM python:3.11.17-slim@sha256:b7a9e5f6a93d6d4a31f129bbd4bdee69efccb6ded0fb230d78aafaf37af98377

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml_classifier.py llm_classifier.py api.py ./
COPY models/ models/

# Run as a normal user, not root.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
