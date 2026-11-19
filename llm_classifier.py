import os
import time

import boto3
from sklearn.metrics import classification_report

from ml_classifier import load_train_test

MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "google.gemma-3-12b-it")
bedrock = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))

# PROMPT = """You sort SMS messages into spam or ham.
# spam = unwanted ads, prizes, competitions, paid services, requests to call or text a number.
# ham = normal personal messages between people, even if short, informal or full of slang.

# Examples:
# SMS: Hey, are we still meeting at 6? -> ham
# SMS: Ok lor, see u later -> ham
# SMS: Congratulations! You won a 500 voucher. Call 09061701461 to claim -> spam
# SMS: Get 50% off ringtones, text TONES to 80082 now -> spam

# Answer with one word: spam or ham.
# SMS: {text} ->"""

PROMPT = """You sort SMS messages into spam or ham.
spam = ads, sales, prizes, competitions, paid subscriptions, ringtones, dating/chat lines, or any message asking you to call a phone number or text a shortcode.
ham = personal conversations between friends/family, casual chat, slang, and normal everyday questions.

Examples:
SMS: Hey, are we still meeting at 6? -> ham
SMS: Ok lor, see u later -> ham
SMS: Call me when you reach home -> ham
SMS: Congratulations! You won a 500 voucher. Call 09061701461 to claim -> spam
SMS: Get 50% off ringtones, text TONES to 80082 now -> spam
SMS: Urgent! We are trying to contact you. Please call our customer service on 09066364589 -> spam
SMS: Hiya, bored at home. Text BABE to 88088 for fun chat, 1.50/msg -> spam
SMS: FREE entry into our weekly comp! Just txt WIN to 80086 -> spam

Answer with one word: spam or ham.
SMS: {text} ->"""


def predict(text):
    response = bedrock.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": PROMPT.format(text=text)}]}],
        inferenceConfig={"temperature": 0},
    )
    answer = response["output"]["message"]["content"][0]["text"].strip().lower()

    if answer.startswith("spam"):
        return "spam"
    if answer.startswith("ham"):
        return "ham"
    return "unparsed"


if __name__ == "__main__":
    train_data, test_data = load_train_test()

    start = time.time()
    predictions = [predict(text) for text in test_data["text"]]
    seconds = time.time() - start

    print(classification_report(test_data["label"], predictions, zero_division=0))
    print("Unparsed answers:", predictions.count("unparsed"))
    print(f"Time per message: {seconds / len(predictions):.2f} s")
