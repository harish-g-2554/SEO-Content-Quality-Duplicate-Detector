# SEO Content Quality & Duplicate Detector

A complete, reproducible pipeline that parses web content, engineers NLP features, detects near-duplicate content, and scores overall SEO quality using machine learning.
Designed for quick experimentation and completion within 4 hours — focusing on clarity, modularity, and reproducibility.

---

## 🧠 Project Overview

This project builds an automated SEO content analysis system that:

* Extracts and cleans raw HTML content.
* Computes key readability and keyword-based NLP features.
* Detects near-duplicate content using TF-IDF cosine similarity.
* Classifies content quality into **Low**, **Medium**, or **High** using a RandomForest model.

---

## ⚙️ Setup

```bash
git clone https://github.com/harish-g-2554/seo-content-detector
cd seo-content-detector
pip install -r requirements.txt
jupyter notebook notebooks/seo_pipeline.ipynb
```

---

## 🚀 Quick Start

1. Place the provided dataset as `data/data.csv`.

   * **Primary dataset** columns: `url`, `html_content`.
   * **Alternative dataset** (URLs only): notebook will scrape automatically.
2. Open and run the Jupyter notebook end-to-end.

   * It generates:

     * `data/extracted_content.csv` → Clean text extraction
     * `data/features.csv` → NLP feature table
     * `data/duplicates.csv` → Near-duplicate URL pairs
     * `models/quality_model.pkl` → Trained model

---

## 💻 Streamlit Demo

```bash
streamlit run streamlit_app/app.py
```

> Make sure you’ve already run the notebook first (to generate `features.csv` and `quality_model.pkl`).
> The app accepts any URL and returns readability, quality score, and duplicate matches.

(Optional deployed version – add link if deployed to Streamlit Cloud)

---

## 🔑 Key Decisions

* **Parsing:** Used BeautifulSoup with semantic tag priority (`<main>`, `<article>`, `<section>`, `<p>`) for extracting main content.
* **Feature Engineering:** Combined readability (Flesch Reading Ease), word/sentence counts, and keyword density metrics.
* **Similarity Detection:** TF-IDF cosine similarity; threshold = **0.80** chosen for high-confidence near-duplicates.
* **Model Choice:** RandomForestClassifier (tuned) trained on 9 textual and readability features; compared against rule-based baseline.
* **Upsampling:** Applied minor upsampling to balance High-quality samples while avoiding overfitting.

---

## 📊 Results Summary

| Metric                    | Value    |
| ------------------------- | -------- |
| **Model Accuracy**        | **0.94** |
| **Baseline Accuracy**     | 0.28     |
| **Macro Avg (F1)**        | 0.92     |
| **Weighted Avg (F1)**     | 0.94     |
| **Total Pages Analyzed**  | 34       |
| **Duplicate Pairs Found** | 2        |
| **Thin Content Pages**    | 8 (≈23%) |

### 🔍 Confusion Matrix

| True \ Predicted | Low | Medium | High |
| ---------------- | --- | ------ | ---- |
| **Low**          | 15  | 0      | 0    |
| **Medium**       | 2   | 5      | 0    |
| **High**         | 0   | 0      | 12   |

### 💡 Top Features (by importance)

1. `content_depth_score` – 0.091
2. `flesch_reading_ease` – 0.078
3. `complex_word_ratio` – 0.044

✅ **Interpretation:**
The tuned RandomForest achieved **94% accuracy**, far outperforming the baseline (28%).
The confusion matrix shows excellent separation of “Low” and “High” classes, with minor overlap in “Medium.”
The model relies most on content depth, readability, and word complexity — strong indicators of SEO quality.

---

## ⚠️ Limitations

* **Readability metrics** may misinterpret pages with code, lists, or mixed languages.
* **TF-IDF embeddings** are lexical; semantic duplicates may go undetected (can be improved using Sentence-Transformer embeddings).
* **HTML parsing** is rule-based; dynamic or JavaScript-rendered content is ignored.

---

## 📁 Repository Structure

```
seo-content-detector/
├── data/
│   ├── data.csv
│   ├── extracted_content.csv
│   ├── features.csv
│   └── duplicates.csv
├── models/
│   └── quality_model.pkl
├── notebooks/
│   └── seo_pipeline.ipynb
├── streamlit_app/
│   ├── app.py
│   └── utils/
│       ├── parser.py
│       ├── features.py
│       └── scorer.py
├── requirements.txt
└── README.md
```

---

## 👨‍💻 Developed by

**Harish G**
*MSc Data Science*

---
