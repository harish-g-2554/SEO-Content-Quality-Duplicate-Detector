# SEO Content Quality & Duplicate Detector

A compact, reproducible pipeline that parses web content, engineers NLP features, detects near-duplicates, and scores content quality. Optimized to complete the core features in ~4 hours.

## Setup
```bash
git clone https://github.com/yourusername/seo-content-detector
cd seo-content-detector
pip install -r requirements.txt
jupyter notebook notebooks/seo_pipeline.ipynb
```

## Quick Start
1. Place the provided dataset as `data/data.csv`.  
   - **Primary dataset** columns: `url`, `html_content` (preferred).  
   - **Alternative dataset**: `url` only; the notebook will scrape.
2. Open the notebook and run all cells (it saves:  
   `data/extracted_content.csv`, `data/features.csv`, `data/duplicates.csv`, and a trained model under `models/quality_model.pkl`).

## (Optional) Streamlit Demo
```bash
streamlit run streamlit_app/app.py
```
> Make sure `data/features.csv` and `models/quality_model.pkl` exist (run the notebook first).

## Key Decisions
- **Parsing**: BeautifulSoup with semantic tag priority (`<main>`, `<article>`, `<section>`, `<p>`).  
- **Similarity**: TF‑IDF cosine similarity (fast, robust for near duplicates).  
- **Threshold**: Default 0.80; chosen empirically as a conservative match for near duplicates.  
- **Model**: RandomForest baseline over simple textual features; compared against rule‑only baseline.

## Results Summary (example)
- **Notebook** prints accuracy/F1 and confusion matrix.  
- **Duplicates**: Written to `data/duplicates.csv` with cosine similarity scores.  
- **Quality**: Labels (`Low/Medium/High`) saved with feature table.

## Limitations
- Readability formulas are heuristic; long code blocks or boilerplate can skew scores.  
- TF‑IDF embeddings ignore semantics; Sentence‑Transformer embeddings are optionally supported (heavier).  
- Simple main‑content extraction; not a full Mercury/Boilerpipe clone.
