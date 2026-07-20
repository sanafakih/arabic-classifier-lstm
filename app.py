"""
Flask website for the Arabic classifier — uses the LiteRT (TFLite) runtime
to run your EXACT trained LSTM model (same architecture, same trained
weights), but WITHOUT ever importing full TensorFlow — that import alone
uses a lot of memory, which is what caused the free-hosting crash before.

Instead of TensorFlow's Tokenizer class, this uses a plain word-index
dictionary (tokenizer_data.pkl, produced by convert_to_tflite.py) and a
small hand-written version of the same tokenization logic Keras uses
internally, so results match your original model exactly.

Run:
    python app.py
Then open:
    http://localhost:5000

Sarah Rmeity — Lebanese University
"""

import os
import pickle

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from flask import Flask, render_template, request

from preprocessing import preprocess

MODEL_PATH = "model.tflite"
TOKENIZER_DATA_PATH = "tokenizer_data.pkl"
LABELS_PATH = "labels.pkl"

app = Flask(__name__)

# Same default filters Keras' Tokenizer uses internally.
_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
_TRANSLATE_MAP = str.maketrans({c: " " for c in _FILTERS})


def text_to_sequence(text, word_index, oov_index):
    """Re-implements Keras Tokenizer.texts_to_sequences for a single text,
    using only plain Python — no TensorFlow needed."""
    text = text.lower().translate(_TRANSLATE_MAP)
    words = [w for w in text.split(" ") if w]
    seq = []
    for w in words:
        idx = word_index.get(w)
        if idx is not None:
            seq.append(idx)
        elif oov_index is not None:
            seq.append(oov_index)
    return seq


def pad_sequence_post(seq, maxlen):
    """Re-implements Keras pad_sequences(padding='post', truncating='post')
    using plain numpy — no TensorFlow needed."""
    arr = np.zeros((1, maxlen), dtype=np.float32)
    trimmed = seq[:maxlen]
    arr[0, :len(trimmed)] = trimmed
    return arr


print("Loading TFLite model and preprocessing artifacts...")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(TOKENIZER_DATA_PATH, "rb") as f:
    tokenizer_data = pickle.load(f)

word_index = tokenizer_data["word_index"]
oov_token = tokenizer_data["oov_token"]
oov_index = word_index.get(oov_token) if oov_token else None

with open(LABELS_PATH, "rb") as f:
    labels_data = pickle.load(f)

idx2cat = labels_data["idx2cat"]
MAXLEN = labels_data["maxlen"]
categories = [idx2cat[i] for i in range(len(idx2cat))]
print("Ready. Categories:", categories)


def classify(text):
    processed = preprocess(text)
    seq = text_to_sequence(processed, word_index, oov_index)
    padded = pad_sequence_post(seq, MAXLEN).astype(input_details[0]["dtype"])

    interpreter.set_tensor(input_details[0]["index"], padded)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]["index"])[0]

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
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("PORT") is None
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
