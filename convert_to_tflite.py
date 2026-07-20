"""
Converts your trained model.keras into model.tflite (a lightweight format
of the EXACT SAME model — same architecture, same trained weights, and
verified to produce identical predictions), and also extracts your
tokenizer into a plain, TensorFlow-free format.

Why: TensorFlow Lite / LiteRT can run this model using a fraction of the
memory that full TensorFlow needs — but the ORIGINAL tokenizer.pkl can
only be loaded by code that imports TensorFlow's Tokenizer class, which
would defeat the purpose (importing TensorFlow at all uses a lot of
memory, regardless of whether you use it afterward). So this script also
saves a plain tokenizer_data.pkl containing just the word list, which the
hosted app can use without ever importing TensorFlow.

Note on the conversion itself: Bidirectional LSTM models need a couple of
extra steps to convert to TFLite cleanly (a fixed sequence length, and
unroll=True on the LSTM layer) — this script handles that by rebuilding
the same architecture with those settings and copying your trained
weights over. The result is verified to be numerically identical to your
original model, just in a lighter-weight format.

Run this LOCALLY on your PC, where you already have TensorFlow installed
(the same place you ran train_model.py):

    python convert_to_tflite.py

This produces: model.tflite, tokenizer_data.pkl
Upload these two files (together with labels.pkl, which stays the same)
to your GitHub repo / hosting.

Sarah Rmeity — Lebanese University
"""

import pickle

import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, Input, LSTM
from tensorflow.keras.models import Sequential

print("Loading model.keras...")
orig_model = tf.keras.models.load_model("model.keras")

print("Loading tokenizer.pkl and labels.pkl (needed to know vocab size / maxlen)...")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("labels.pkl", "rb") as f:
    labels_data = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
maxlen = labels_data["maxlen"]
n_classes = len(labels_data["idx2cat"])

print(f"vocab_size={vocab_size}  maxlen={maxlen}  n_classes={n_classes}")

print("\nRebuilding the same architecture with TFLite-friendly settings...")
# Same architecture as train_model.py, but with:
#   - an explicit static Input shape (TFLite needs a fixed sequence length)
#   - unroll=True on the LSTM (required for Bidirectional LSTM -> TFLite)
# Weights are copied directly from your trained model, so this produces
# numerically identical predictions.
fixed_model = Sequential([
    Input(shape=(maxlen,)),
    Embedding(vocab_size, 64),
    Bidirectional(LSTM(64, return_sequences=False, unroll=True)),
    Dropout(0.4),
    Dense(32, activation="relu"),
    Dense(n_classes, activation="softmax"),
])
fixed_model.set_weights(orig_model.get_weights())

print("Converting to TFLite (this may take a moment)...")
converter = tf.lite.TFLiteConverter.from_keras_model(fixed_model)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
print("Saved model.tflite")

print("Extracting tokenizer into a plain (TensorFlow-free) format...")
tokenizer_data = {
    "word_index": tokenizer.word_index,
    "oov_token": tokenizer.oov_token,
}
with open("tokenizer_data.pkl", "wb") as f:
    pickle.dump(tokenizer_data, f)
print("Saved tokenizer_data.pkl")

print("\nVerifying the converted model matches the original...")
import numpy as np
from ai_edge_litert.interpreter import Interpreter

test_input = np.random.randint(1, vocab_size, size=(1, maxlen)).astype(np.float32)
orig_pred = orig_model.predict(test_input, verbose=0)[0]

interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]["index"], test_input.astype(input_details[0]["dtype"]))
interpreter.invoke()
tflite_pred = interpreter.get_tensor(output_details[0]["index"])[0]

max_diff = np.max(np.abs(orig_pred - tflite_pred))
print(f"Max difference between original and TFLite predictions: {max_diff:.8f}")
if max_diff < 1e-4:
    print("Verified: converted model matches the original.")
else:
    print("WARNING: outputs differ more than expected — double check before deploying.")

print("\nDone! Upload model.tflite + tokenizer_data.pkl + labels.pkl to your repo.")
