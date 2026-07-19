# Arabic Paragraph Classifier — Flask Website

A simple website version of the LSTM Arabic text classifier. Paste any
Arabic paragraph in and it predicts the field/category (law, medicine,
engineering, etc.) with a confidence breakdown.

## Project structure
```
arabic_classifier_app/
├── train_model.py       # trains the LSTM model, saves model.keras + tokenizer.pkl + labels.pkl
├── app.py                # Flask website that loads the saved model and serves predictions
├── preprocessing.py       # shared cleaning/stopwords/lemmatization/stemming (same as the notebook)
├── templates/
│   └── index.html         # the web page (Arabic RTL form)
├── requirements.txt
└── README.md
```

## Setup (run once)

1. Install Python 3.9+ if you don't have it.
2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   ```
3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Put `diverse_arabic_paragraphs_dataset.xlsx` in this same folder
   (next to `train_model.py`).

## Step 1 — Train the model

```
python train_model.py
```

This runs the same preprocessing + LSTM + early stopping pipeline you already
validated in Colab, then saves three files in this folder:
- `model.keras` — the trained model
- `tokenizer.pkl` — the fitted tokenizer
- `labels.pkl` — category names + max sequence length

You only need to do this once (or again whenever you update the dataset).

## Step 2 — Run the website

```
python app.py
```

Then open your browser to:
```
http://localhost:5000
```

Paste any Arabic paragraph into the text box and click "تصنيف النص" (Classify
Text) — it'll show the predicted category plus a confidence bar for every
category, just like the `predict()` function did in the notebook.

## Notes
- The website only loads the already-trained model — it does not retrain
  itself, so it starts up instantly and predictions are fast.
- If you retrain on a new/updated dataset, just re-run `train_model.py` — it
  overwrites the saved files, and the next time you start `app.py` it'll pick
  up the new model automatically.
