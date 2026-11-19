import sys

import httpx

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

EXAMPLES = [
    ("WINNER!! You have won a 900 prize. Call 09061701461 to claim", "spam"),
    ("Ok lar... see you at 6 for dinner", "ham"),
    ("FREE entry in a weekly competition to win FA Cup final tickets. Text FA to 87121", "spam"),
    ("URGENT! Your mobile number has been awarded a 2000 bonus. Call 09050000123 now", "spam"),
    ("Congratulations! You have been selected for a FREE holiday. Reply YES to 80082", "spam"),
    ("Your account will be closed today. Claim your 500 reward at www.prize-now.com", "spam"),
    ("Can you pick up some milk on your way home?", "ham"),
    ("Happy birthday! Hope you have a great day", "ham"),
    ("Running 10 mins late. Start free trial send an me an message.", "spam"),
]

for text, true_type in EXAMPLES:
    print(f"SMS: {text}")
    print(f"  true type: {true_type}")
    for route in ["/ml/classify", "/llm/classify"]:
        response = httpx.post(BASE_URL + route, json={"sms": text}, timeout=60)
        print(f"  {route:<14} {response.status_code}  {response.text}")
    print()
