import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textstat import flesch_reading_ease, sentence_count

def basic_features(text: str) -> dict:
    txt = text or ""
    words = re.findall(r"\w+", txt)
    wc = len(words)
    try:
        sc = sentence_count(txt)
    except Exception:
        sc = max(1, txt.count(".") + txt.count("!") + txt.count("?"))
    try:
        fre = float(flesch_reading_ease(txt))
    except Exception:
        fre = 0.0
    return {"word_count": wc, "sentence_count": sc, "flesch_reading_ease": fre}

def top_keywords_from_tfidf(vectorizer: TfidfVectorizer, row_vector, top_k=5):
    # row_vector is 1 x V sparse
    if row_vector.nnz == 0:
        return []
    indices = row_vector.indices
    data = row_vector.data
    top_idx = data.argsort()[::-1][:top_k]
    vocab = vectorizer.get_feature_names_out()
    return [vocab[indices[i]] for i in top_idx]
