# Book Genre Clustering

Clusters text passages from books into 5 genres — **biography**, **fantasy**, **horror**, **romance**, **sci-fi** — using six unsupervised clustering algorithms applied to six different feature representations.

---

## Run Order

### 25 Known Books (main pipeline)

Run notebooks in sequence:

| Step | Notebook | What it does |
|------|----------|-------------|
| 1 | `Book partitioner.ipynb` | Samples 200 × 400-word partitions from each of the 25 KnownBooks, saves `TestingData/partitions.csv` |
| 2 | `BOW.ipynb` | Strips Gutenberg boilerplate, extracts genre vocabulary and n-grams, then builds **6 feature representations** saved to `FeatureTrainingData/` |
| 3 | `Clustering.ipynb` | Loads all feature sets, applies all 6 clustering algorithms, evaluates with silhouette / kappa / coherence, error analysis, and top-20 words per cluster |

### Full Corpus (~98 books)

| Notebook | What it does |
|----------|-------------|
| `ClusterAllBooks.ipynb` | Standalone — loads books directly from `AllBooks/<genre>/`, partitions and features inline, runs the same 6 clustering algorithms |

---

## Feature Representations (`FeatureTrainingData/`)

| File | Description |
|------|-------------|
| `bow_features.csv` | Bag-of-Words count vectors (genre vocabulary, 1109 n-grams) |
| `tfidf_features.csv` | TF-IDF weighted n-gram vectors (same vocabulary) |
| `stylometric_features.csv` | 10 surface style signals: avg word/sentence length, type-token ratio, punctuation density, hapax ratio, syllables |
| `lmf_features.csv` | 9 POS-based lexical-morphological features: noun/verb/adj/adv ratios, lexical density, function-word ratio (NLTK POS tagger) |
| `sbert_features.npy` | 384-dimensional SBERT embeddings (all-MiniLM-L6-v2) — binary, fast load |
| `sbert_features.csv` | Same SBERT embeddings with partition metadata columns |

---

## Clustering Algorithms (`Clustering.ipynb`)

| Algorithm | Implementation | Feature sets |
|-----------|---------------|--------------|
| **K-Means** | `sklearn.cluster.KMeans` (k=5) | BOW, TF-IDF, Stylometric, LMF, Stylo+LMF, SBERT |
| **Divisive Hierarchical** | Bisecting K-Means (custom top-down) | TF-IDF, SBERT, Stylo+LMF |
| **LDA** | `sklearn.decomposition.LatentDirichletAllocation` | Raw count vectors |
| **GMM** | `sklearn.mixture.GaussianMixture` (full cov) | TF-IDF-PCA50, Stylometric, LMF, Stylo+LMF, SBERT-PCA50 |
| **SOM** | `minisom.MiniSom` (10×10 grid) + K-Means on BMU weights | SBERT-PCA50, Stylo+LMF, TF-IDF-PCA50 |
| **DEC** | Stacked autoencoder (TensorFlow/Keras) + K-Means in latent space | SBERT-PCA50, Stylo+LMF |

---

## Evaluations

| Metric | Applies to |
|--------|-----------|
| **Silhouette score** | All algorithms |
| **Cohen's Kappa** | All algorithms (clusters mapped to genres by majority vote) |
| **Topic Coherence (C_V)** | LDA only (via Gensim) |

Outputs include:
- Evaluation heatmaps (`images/clustering_evaluation_heatmap.png`)
- Top-15 configurations chart (`images/clustering_top15.png`)
- Confusion matrix for champion model (`images/clustering_confusion_matrix.png`)
- Per-genre error rates, top confusion pairs, and sample misclassified snippets
- Top 20 unigrams and bigrams per cluster

---

## Folder Structure

```
KnownBooks/          25 training books (.txt + _clean.txt), 5 per genre
AllBooks/            Full corpus (~98 books) in genre subdirectories
TestingData/         partitions.csv (from Book partitioner.ipynb)
FeatureTrainingData/ bow, tfidf, stylometric, lmf, sbert feature CSVs + vectorizer.pkl
images/              Output charts, heatmaps, confusion matrices
ngrams/              Per-genre top n-gram CSVs
```

---

## Dependencies

Core:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn nltk sentence-transformers
```

Optional (required for specific algorithms):
```bash
pip install minisom          # SOM
pip install gensim           # LDA topic coherence (C_V score)
pip install tensorflow       # DEC (deep embedded clustering)
```

NLTK data (downloaded automatically on first run):
- `stopwords`, `punkt_tab`, `names`, `averaged_perceptron_tagger_eng`
