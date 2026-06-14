# Book Genre Classifier & Cluster Analysis

Classifies and clusters text passages from books into 5 genres: **biography**, **fantasy**, **horror**, **romance**, **sci-fi**.

**Assignment 2 (Clustering):** Four feature representations (BOW, TF-IDF, LDA, Sentence Embeddings) are clustered with K-Means, EM (GMM), and Hierarchical clustering (plus SOM/Kohonen on the Embeddings feature set), then evaluated against the true genre labels.

---

## Run Order

Run notebooks top-to-bottom in this sequence:

| Step | Notebook                     | What it does |
|------|------------------------------|-------------|
| 1 | `BOW.ipynb`                  | Strips Gutenberg boilerplate from `KnownBooks/`, extracts genre-level n-gram vocabulary, saves `FeatureTrainingData/vocabulary.csv` |
| 2 | `Book partitioner.ipynb`     | Samples 200 × 400-word partitions from each of the 25 training books, saves `TestingData/partitions.csv` |
| 3 | `Genre classifier.ipynb`     | Trains all 6 models, runs 10-fold CV, plots confusion matrices, saves models to `models/` |
| 4 | `Predict New Book.ipynb`     | Predicts genre for up to 5 custom books — edit the `BOOKS` list to point to your files |
| 5 | `Batch Evaluation.ipynb`     | Downloads ~100 unseen books from Project Gutenberg into `UnknownBooks/`, evaluates all models |
| 6 | `claire-Clustering v2.ipynb` | Clusters the same partitions (no labels used during fitting) across 4 feature sets × 3 algorithms, evaluates with Silhouette, Kappa, and Coherence |

---

## Clustering Notebook — Assignment 2 Additions from Claire

`Clustering.ipynb` extends the original 3-feature × 3-algorithm grid (TF-IDF, LDA, Embeddings) with a 4th feature set and two additional evaluation/analysis techniques required by the assignment brief.

| Addition | What it does | Why |
|---|---|---|
| **BOW feature set** | Raw `CountVectorizer` output (term counts, no TF-IDF weighting), 2000 features, unigrams+bigrams | Brief requires "Transform to BOW *and* TF-IDF" |
| **Coherence score** | Manual PMI-based C_V-style coherence, computed from scratch with NumPy (avoids `gensim`, which fails to build on Python 3.14) | Brief requires Coherence alongside Kappa and Silhouette |
| **Collocation analysis** | Top-10 bigram collocations (by PMI) among misclassified instances, via NLTK | Brief requires error analysis using "top 10 frequent words and/or top collocations" |
| **Cosine-distance Hierarchical** | Re-runs the champion model (Hierarchical + Embeddings) with cosine distance / average linkage instead of Euclidean / Ward | Tests whether cosine similarity — generally preferred for semantic embeddings — improves results |
| **Per-genre purity** | Breaks down the champion model's accuracy genre-by-genre | Identifies which genres cluster well vs. poorly (mirrors A1's per-genre F1 analysis) |
| **Seed stability check** | Runs K-Means and EM across 5 random seeds per feature set, reports Kappa/Silhouette mean ± std | Hierarchical clustering is deterministic; this checks how reproducible the stochastic algorithms are |
| **SOM (Kohonen)** | Self-organizing map added as a 4th clustering algorithm, run on the Embeddings feature set | Brief requires comparing multiple clustering algorithms; tests a topology-preserving alternative to K-Means/EM/Hierarchical |

### Key Findings

**Champion model:** Hierarchical clustering on sentence embeddings (`all-MiniLM-L6-v2`), Kappa = 0.627 — unchanged even after adding BOW as a 4th feature set and SOM (Kohonen) as a 4th algorithm.

**Coherence vs. Kappa are independent measures.** BOW clusters had the highest coherence (0.80–0.83) — meaning their top words co-occur strongly — but only mediocre genre accuracy (Kappa 0.24–0.29). TF-IDF coherence ranged from −1.17 (K-Means/EM) to +1.65 (Hierarchical), showing that high "topic quality" does not imply alignment with true labels.

**Sci-fi is the hardest genre to cluster.** Per-genre purity for the champion model: romance 0.899, fantasy 0.765, biography 0.753, horror 0.697, **sci-fi 0.395**. The confusion matrix shows 538/1000 sci-fi partitions were grouped into the horror cluster — by far the largest single error pattern. This echoes the sci-fi generalisation issues seen in Assignment 1.

**Cosine distance dramatically hurts label alignment despite improving geometric separation.** Switching the champion model from Euclidean/Ward to cosine/average linkage *increased* Silhouette (0.034 → 0.097) but *collapsed* Kappa (0.627 → 0.003). Geometric "tightness" of clusters and their correspondence to true genre labels are not the same thing — optimizing for one can actively destroy the other.

**Stability:** LDA + K-Means is perfectly deterministic across random seeds (Kappa std = 0.000) but has a lower mean Kappa (0.449) than embeddings-based methods (~0.53, std ≈ 0.07). The overall champion, Hierarchical + Embeddings, is deterministic by construction — combining the best Kappa (0.627) with full reproducibility.

**SOM clustering produces balanced clusters but recovers only 3 of 5 genres.** Adding SOM (Kohonen) as a 4th algorithm on the Embeddings feature set yields a Silhouette score (0.026) similar to the other three algorithms (0.034–0.041), but a much lower Kappa (0.219 vs. 0.618–0.627). The five resulting clusters are reasonably sized (708, 587, 1534, 1705, 466 — none collapse to near-zero), but when mapped to majority genre labels, two clusters map to biography, one to fantasy, and two to sci-fi. No cluster maps to horror or romance, so every horror and romance partition is necessarily misclassified into one of the other three genres. This echoes the sci-fi/biography "catch-all" pattern seen in the per-genre purity breakdown above and in Assignment 1's generalisation testing — horror and romance appear to occupy embedding regions that overlap heavily with biography and sci-fi rather than forming their own distinct neighbourhood.

---

## Folder Structure

```
KnownBooks/          25 training books (.txt), 5 per genre
UnknownBooks/        Unseen test books, auto-downloaded by Batch Evaluation
TestingData/         partitions.csv (training partition data)
FeatureTrainingData/ vocabulary.csv, featurevector.csv, vectorizer.pkl
models/              Trained model .pkl files + model_distilbert/
images/              Output charts and confusion matrices
ngrams/              Per-genre top n-gram CSVs
```

---

## Dependencies

```bash
pip install scikit-learn pandas numpy matplotlib seaborn xgboost transformers accelerate joblib nltk sentence-transformers
```

**PyTorch** must be installed separately. For CPU only:
```bash
pip install torch
```
For NVIDIA GPU (recommended — ~5 min training vs ~30 min on CPU), replace `cu124` with your CUDA version (check with `nvidia-smi`):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

The DistilBERT training cell will print your GPU name and confirm `fp16: ENABLED` if CUDA is detected correctly.

**Note on `gensim`:** the standard `CoherenceModel` from `gensim` may fail to build on newer Python versions (e.g. 3.14) due to missing prebuilt wheels. `Clustering.ipynb` implements coherence manually with NumPy instead, so `gensim` is **not** required.
