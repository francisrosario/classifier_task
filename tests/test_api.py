from fastapi.testclient import TestClient

import api

client = TestClient(api.app)


def test_ml_and_llm_classifier_api():
    spam = client.post("/ml/classify", json={"sms": "WINNER! You won a FREE prize. Call 09061701461 to claim"})
    ham = client.post("/ml/classify", json={"sms": "Ok, see you at 6 for dinner"})
    assert spam.json() == {"label": "spam", "model": "tfidf-logreg"}
    assert ham.json()["label"] == "ham"
