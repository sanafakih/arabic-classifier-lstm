"""
Train the Arabic paragraph classifier and save everything the Flask app needs:
  - model.keras     (the trained LSTM model)
  - tokenizer.pkl   (the fitted Keras tokenizer)
  - labels.pkl      (category index <-> name mapping + max sequence length)

Run this once (or whenever you retrain on a new dataset):
    python train_model.py

Sarah Rmeity — Lebanese University
"""

import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from preprocessing import preprocess

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

EXCEL_PATH = "diverse_arabic_paragraphs_dataset.xlsx"
TEXT_COL = "الفقرة"
LABEL_COL = "المجال"


def main():
    print("Loading dataset...")
    df = pd.read_excel(EXCEL_PATH)
    df = df[[TEXT_COL, LABEL_COL]].dropna().reset_index(drop=True)
    df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)

    print("Dataset loaded:", df.shape)
    print(df[LABEL_COL].value_counts())

    print("\nPreprocessing all paragraphs (this can take a bit)...")
    df["processed"] = df[TEXT_COL].apply(preprocess)

    categories = sorted(df[LABEL_COL].unique())
    cat2idx = {c: i for i, c in enumerate(categories)}
    idx2cat = {i: c for c, i in cat2idx.items()}
    df["label_idx"] = df[LABEL_COL].map(cat2idx)

    print("\nCategories:", categories)

    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(df["processed"])
    sequences = tokenizer.texts_to_sequences(df["processed"])

    maxlen = max(len(s) for s in sequences)
    X = pad_sequences(sequences, maxlen=maxlen, padding="post")
    Y = tf.keras.utils.to_categorical(df["label_idx"], num_classes=len(categories))
    vocab_size = len(tokenizer.word_index) + 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=df["label_idx"]
    )

    print(
        "\nVocab size:", vocab_size,
        "| max sequence length:", maxlen,
        "| categories:", len(categories),
    )
    print("Train size:", len(X_train), "| Test size:", len(X_test))

    model = Sequential([
        Embedding(vocab_size, 64),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.4),
        Dense(32, activation="relu"),
        Dense(len(categories), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    print("\nTraining LSTM classifier...")
    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=16,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc * 100:.2f}%  |  Test loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=[idx2cat[i] for i in range(len(categories))]))

    print("Confusion matrix (rows=true, cols=predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index=[idx2cat[i] for i in range(len(categories))],
        columns=[idx2cat[i] for i in range(len(categories))],
    ))

    print("\nSaving model and preprocessing artifacts...")
    model.save("model.keras")
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    with open("labels.pkl", "wb") as f:
        pickle.dump({"idx2cat": idx2cat, "maxlen": maxlen}, f)

    print("Done! Saved: model.keras, tokenizer.pkl, labels.pkl")
    print("You can now run: python app.py")


if __name__ == "__main__":
    main()
