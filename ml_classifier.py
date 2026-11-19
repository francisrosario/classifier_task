from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_FOLDER = Path(__file__).parent
DATA_FILE = PROJECT_FOLDER / "data" / "SMSSpamCollection"
MODEL_FILE = PROJECT_FOLDER / "models" / "ml_model.joblib"


def load_train_test():
    raw = pd.read_csv(DATA_FILE, sep="\t", header=None, names=["label", "text"], quoting=3)
    messages = raw.drop_duplicates(subset="text")

    return train_test_split(messages, test_size=0.2, stratify=messages["label"], random_state=42)


def train(train_data):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(train_data["text"], train_data["label"])
    return model


def predict(model, text):
    return model.predict([text])[0]


if __name__ == "__main__":
    train_data, test_data = load_train_test()
    model = train(train_data)

    predictions = model.predict(test_data["text"])
    print(classification_report(test_data["label"], predictions))

    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print("Saved the model to", MODEL_FILE)
