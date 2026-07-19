"""
Simple Flask website for the Arabic paragraph classifier.

This loads the model/tokenizer/labels saved by train_model.py — it does NOT
retrain anything, so the page loads instantly and predictions are fast.

Run:
    python app.py
Then open:
    http://localhost:5000

Sarah Rmeity — Lebanese University
"""

import pickle

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocessing import preprocess

# ================================================================
# EDIT THESE THREE PATHS to point at your trained model files.
# They can be relative (e.g. "model.keras") or a full absolute path
# (e.g. "C:/Users/Sarah/Desktop/model.keras" or "/home/sarah/model.keras").
# All three are produced together by train_model.py.
# ================================================================
MODEL_PATH = "model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
LABELS_PATH = "labels.pkl"

app = Flask(__name__)

print("Loading model and preprocessing artifacts...")
model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(LABELS_PATH, "rb") as f:
    labels_data = pickle.load(f)

idx2cat = labels_data["idx2cat"]
MAXLEN = labels_data["maxlen"]
categories = [idx2cat[i] for i in range(len(idx2cat))]
print("Ready. Categories:", categories)


def classify(text):
    processed = preprocess(text)
    seq = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(seq, maxlen=MAXLEN, padding="post")
    probs = model.predict(padded, verbose=0)[0]

    scores = {idx2cat[i]: float(probs[i]) * 100 for i in range(len(categories))}
    ordered = sorted(scores.items(), key=lambda x: -x[1])
    best_category, best_pct = ordered[0]
    return best_category, best_pct, ordered


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("text", "").strip()
        if input_text:
            best_category, best_pct, ordered = classify(input_text)
            result = {
                "best_category": best_category,
                "best_pct": round(best_pct, 1),
                "all_scores": [(cat, round(pct, 1)) for cat, pct in ordered],
            }

    return render_template("index.html", result=result, input_text=input_text)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
