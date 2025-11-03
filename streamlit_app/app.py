"""import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

from utils.parser import extract_title_and_text, clean_text
from utils.features import basic_features, top_keywords_from_tfidf
from utils.scorer import thin_content, rule_label, int_to_label

st.set_page_config(page_title="SEO Content Quality & Duplicate Detector", layout="wide")
st.title("SEO Content Quality & Duplicate Detector")

@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open("models/quality_model.pkl", "rb"))
    except Exception:
        model = None
    try:
        feats = pd.read_csv("data/features.csv")
    except Exception:
        feats = None
    tfidf = None
    X = None
    texts = None
    if feats is not None:
        texts = feats["body_text"].fillna("")
        tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2)
        X = tfidf.fit_transform(texts)
    return model, feats, tfidf, X, texts

model, feats, tfidf, X, texts = load_assets()

url = st.text_input("Analyze a URL")
if st.button("Analyze") and url:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Assignment Demo)"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        title, body = extract_title_and_text(r.text)
        body_clean = clean_text(body)
        bf = basic_features(body_clean)

        # Duplicate detection (TF-IDF cosine vs corpus)
        dupes = []
        if tfidf is not None and X is not None and texts is not None:
            xq = tfidf.transform([body_clean])
            sims = cosine_similarity(xq, X).ravel()
            top_idx = np.argsort(sims)[::-1][:5]
            for i in top_idx:
                if sims[i] >= 0.80:
                    dupes.append({"url": feats.loc[i, "url"], "similarity": float(sims[i])})

        # Predict
        label_rule = rule_label(bf["word_count"], bf["flesch_reading_ease"])
        label_model = label_rule
        if model is not None:
            # Minimal set of features required
            cols = ["word_count", "sentence_count", "flesch_reading_ease"]
            Xq = [[bf[c] for c in cols]]
            try:
                pred = model.predict(Xq)[0]
                label_model = int_to_label(int(pred))
            except Exception:
                pass

        st.subheader("Results")
        st.json({
            "url": url,
            "title": title,
            "word_count": bf["word_count"],
            "readability": bf["flesch_reading_ease"],
            "quality_label_model": label_model,
            "quality_label_rule": label_rule,
            "is_thin": thin_content(bf["word_count"]),
            "similar_to": dupes
        })
    except Exception as e:
        st.error(f"Failed to analyze: {e}")


"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

from utils.parser import extract_title_and_text, clean_text
from utils.features import basic_features, top_keywords_from_tfidf
from utils.scorer import thin_content, rule_label, int_to_label

# Streamlit setup
st.set_page_config(page_title="SEO Content Quality & Duplicate Detector", layout="wide")
st.title("SEO Content Quality & Duplicate Detector")

@st.cache_resource
def load_assets():
    """Load model, dataset, and TF-IDF features"""
    # Load model
    try:
        model = pickle.load(open("models/quality_model.pkl", "rb"))
    except Exception as e:
        st.warning(f"⚠️ Model could not be loaded: {e}")
        model = None

    # Load feature dataset
    try:
        feats = pd.read_csv("data/features.csv")
    except Exception:
        feats = None

    # Prepare TF-IDF for duplicate detection
    tfidf = None
    X = None
    texts = None
    if feats is not None:
        texts = feats["body_text"].fillna("")
        tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
        X = tfidf.fit_transform(texts)

    return model, feats, tfidf, X, texts

# Load all assets
model, feats, tfidf, X, texts = load_assets()

# Display model load status
if model is None:
    st.warning("⚠️ Model not loaded — using **rule-based scoring** instead of ML predictions.")
else:
    st.success("✅ ML Model loaded successfully — using **ML-based content scoring**.")

# User input
url = st.text_input("Enter a URL to analyze:")

if st.button("Analyze") and url:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Assignment Demo)"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()

        # Extract and clean text
        title, body = extract_title_and_text(r.text)
        body_clean = clean_text(body)
        bf = basic_features(body_clean)

        # Duplicate detection
        dupes = []
        if tfidf is not None and X is not None and texts is not None:
            xq = tfidf.transform([body_clean])
            sims = cosine_similarity(xq, X).ravel()
            top_idx = np.argsort(sims)[::-1][:5]
            for i in top_idx:
                if sims[i] >= 0.80:
                    dupes.append({
                        "url": feats.loc[i, "url"],
                        "similarity": float(sims[i])
                    })

        # Predict quality label
        label_rule = rule_label(bf["word_count"], bf["flesch_reading_ease"])
        label_model = label_rule

        if model is not None:
            cols = ["word_count", "sentence_count", "flesch_reading_ease"]
            Xq = [[bf[c] for c in cols]]
            try:
                pred = model.predict(Xq)[0]
                label_model = int_to_label(int(pred))
            except Exception as e:
                st.error(f"Model prediction failed: {e}")

        # Show results
        st.subheader("Results")
        st.json({
            "url": url,
            "title": title,
            "word_count": bf["word_count"],
            "readability": bf["flesch_reading_ease"],
            "quality_label_model": label_model,
            "quality_label_rule": label_rule,
            "is_thin": thin_content(bf["word_count"]),
            "similar_to": dupes
        })

    except Exception as e:
        st.error(f"❌ Failed to analyze URL: {e}")
