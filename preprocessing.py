"""
Shared Arabic text preprocessing.

This is the EXACT same cleaning / stopword-removal / lemmatization / stemming
pipeline used in the notebook (Sarah Rmeity — Lebanese University), pulled out
into its own file so both train_model.py and app.py use identical logic.
"""

import re
from nltk.stem.isri import ISRIStemmer
import qalsadi.lemmatizer as qlm

_isri = ISRIStemmer()
_lemmatizer = qlm.Lemmatizer()

ARABIC_STOPWORDS = set([
    "من", "في", "على", "الى", "عن", "مع", "يا", "هو", "هي", "هم", "هن", "هذا", "هذه", "هؤلاء",
    "ذلك", "تلك", "أو", "و", "ثم", "لكن", "إن", "أن", "لا", "لم", "لن", "ليست", "كان", "يكون",
    "كأن", "حتى", "إلا", "إذا", "اي", "أيضا", "بين", "قبل", "بعد", "دون", "غير", "كل", "قد", "مثل",
    "أين", "كيف", "متى", "لماذا", "ما", "هنا", "هناك", "التي", "الذي", "الذين", "اللاتي", "اللذين",
    "لأن", "أجل", "أكثر", "أولئك", "إذ", "إذن", "الك",
    "الكل", "اللواتي", "اللى", "الي", "ان", "اى", "بعض", "جميع", "سوف",
    "فوق", "كي", "لعل", "لكم",
])


def clean(text):
    """Basic text cleaning: keep Arabic characters and spaces only."""
    text = re.sub(r'[^؀-ۿ\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text):
    words = text.split()
    return " ".join(w for w in words if w not in ARABIC_STOPWORDS)


def lemmatize(word):
    try:
        lemma = _lemmatizer.lemmatize(word)
        return lemma if lemma else word
    except Exception:
        return word


def stem(word):
    try:
        stemmed = _isri.stem(word)
        return stemmed if stemmed else word
    except Exception:
        return word


def preprocess(text):
    """Returns the final processed string that feeds the model."""
    cleaned = clean(text)
    no_stop = remove_stopwords(cleaned)
    words = [w for w in no_stop.split() if len(w) > 1]
    lemmas = [lemmatize(w) for w in words]
    stems = [stem(w) for w in lemmas]
    return " ".join(stems)
