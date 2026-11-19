import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

import llm_classifier
import ml_classifier

app = FastAPI()

ml_model = joblib.load(ml_classifier.MODEL_FILE)


class SMS(BaseModel):
    sms: str = Field(min_length=1, max_length=300,
                     examples=["WINNER!! Claim your 900 prize now, call 09061701461"])


class Tag(BaseModel):
    label: str = Field(examples=["spam"])
    model: str = Field(examples=["tfidf-logreg"])


@app.post("/ml/classify", response_model=Tag)
def classify_with_ml(request: SMS):
    label = ml_classifier.predict(ml_model, request.sms)
    return Tag(label=label, model="tfidf-logreg")


@app.post("/llm/classify", response_model=Tag)
def classify_with_llm(request: SMS):
    label = llm_classifier.predict(request.sms)
    return Tag(label=label, model=llm_classifier.MODEL_ID)
