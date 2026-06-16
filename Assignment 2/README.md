# Book Genre Clustering

Clusters text passages from books into 5 genres — **biography**, **fantasy**, **horror**, **romance**, **sci-fi** — using six unsupervised clustering algorithms applied to nine different feature representations.

---

## Run Order

### 25 Known Books (main pipeline)

| Step | Notebook | What it does |
|------|----------|-------------|
| 1 | `Book partitioner.ipynb` | Samples 200 × 400-word partitions from each of the 25 KnownBooks, saves `TestingData/partitions.csv` |
| 2 | `Feature Vector.ipynb` | Strips Gutenberg boilerplate, filters non-English tokens, extracts genre vocabulary, builds all feature representations, runs vocabulary analysis, generates `selected_stylo_lmf_features.json` |
| 3 | `Clustering.ipynb` | Loads all feature sets, applies all 6 algorithms, evaluates with silhouette / kappa / coherence, confusion matrices, cluster plots, error analysis |

### Full Corpus (~98 books)

| Notebook | What it does |
|----------|-------------|
| `ClusterAllBooks.ipynb` | Loads pre-computed AllBooks features from `Feature Vector.ipynb`, applies language filtering, runs the same 6 algorithms |

> Run the **AllBooks section** of `Feature Vector.ipynb` before `ClusterAllBooks.ipynb` to generate the `allbooks_*.csv/npy` files.

---

## Feature Representations

### KnownBooks — `FeatureTrainingData/` (5000 partitions)

| File | Description | Shape |
|------|-------------|-------|
| `bow_features.csv` | Bag-of-Words count vectors (curated vocabulary) | 5000 × 1026 |
| `tfidf_features.csv` | TF-IDF weighted n-gram vectors (same vocabulary) | 5000 × 1026 |
| `stylometric_features.csv` | 10 surface style signals: avg word/sentence length, type-token ratio, punctuation density, hapax ratio, avg syllables | 5000 × 10 |
| `lmf_features.csv` | 9 POS-based features: noun/verb/adj/adv ratios, lexical density, function-word ratio (NLTK POS tagger) | 5000 × 9 |
| `sbert_features.npy` | SBERT embeddings — `all-MiniLM-L6-v2`, binary fast load | 5000 × 384 |
| `sbert_features.csv` | SBERT embeddings with partition metadata | 5000 × 387 |
| `selected_stylo_lmf_features.json` | Top-15 Stylo+LMF feature names (mutual information + correlation redundancy filter) | 15 names |
| `vocabulary.csv` | Curated genre vocabulary after stop-word, name, and cross-genre filtering | 1026 terms |
| `vectorizer.pkl` | Fitted TF-IDF vectorizer — use `.transform()` on new text | — |

### AllBooks — `FeatureTrainingData/` (~19,400 partitions)

| File | Description |
|------|-------------|
| `allbooks_partitions.csv` | Cached partitions with language-detection filtering applied |
| `allbooks_bow_features.csv` | BOW using curated KnownBooks vocabulary |
| `allbooks_tfidf_features.csv` | TF-IDF using curated KnownBooks vocabulary |
| `allbooks_stylometric_features.csv` | Stylometric features |
| `allbooks_lmf_features.csv` | LMF features (slow to compute — ~30 min) |
| `allbooks_sbert_features.npy` | SBERT embeddings |

---

## Feature Sets in Clustering

| Feature Set | Description | Dim |
|-------------|-------------|-----|
| `BOW` | Raw n-gram counts (curated vocabulary) | 1026 |
| `TF-IDF` | TF-IDF weighted n-grams | 1026 |
| `TF-IDF-PCA50` | TF-IDF reduced to 50 PCA components | 50 |
| `Stylometric` | Writing-style metrics | 10 |
| `LMF` | Lexical-morphological POS ratios | 9 |
| `Stylo+LMF` | Stylometric and LMF concatenated | 19 |
| `Stylo+LMF-15` | Top-15 features selected by mutual information + redundancy filter | 15 |
| `SBERT` | Sentence-transformer semantic embeddings | 384 |
| `SBERT-PCA50` | SBERT reduced to 50 PCA components | 50 |
| `CountVec` | Raw counts for LDA topic model (5000 vocab) | 5000 |

---

## Clustering Algorithms

| Algorithm | Implementation | Notes |
|-----------|---------------|-------|
| **K-Means** | `sklearn.cluster.KMeans` (k=5, n_init=10) | Hard assignment; assumes spherical equal-size clusters |
| **Divisive Hierarchical** | Bisecting K-Means (custom top-down) | Repeatedly splits largest cluster; preserves global structure at each step |
| **LDA** | `sklearn.decomposition.LatentDirichletAllocation` | Requires integer CountVec (Dirichlet-Multinomial model); coherence via Gensim |
| **GMM** | `sklearn.mixture.GaussianMixture` (full cov, k=5) | Soft assignment; elliptical clusters; dim > 100 skipped (use PCA50 variants) |
| **SOM** | `minisom.MiniSom` (10×10 grid) + K-Means on BMU weights | Topology-preserving neural clustering |
| **DEC** | Stacked autoencoder (256→128→32) + K-Means in latent space | Learns compressed representation before clustering; requires TensorFlow |

---

## Vocabulary & Feature Analysis

`Feature Vector.ipynb` produces diagnostic plots saved to `images/`:

| Plot | File |
|------|------|
| Top 15 TF-IDF terms per genre | `vocab_tfidf_top_terms.png` |
| TF-IDF sparsity & active vocab per genre | `vocab_tfidf_sparsity.png` |
| Stylometric feature distributions (violin plots) | `vocab_stylometric_violin.png` |
| Stylometric + LMF feature correlation matrix | `vocab_stylo_lmf_correlation.png` |
| Feature selection — mutual information ranking | `vocab_feature_selection.png` |
| LMF features per genre (grouped bar chart) | `vocab_lmf_per_genre.png` |
| SBERT intra/inter-genre cosine similarity | `vocab_sbert_similarity.png` |
| SBERT embeddings — PCA 2D scatter | `vocab_sbert_pca2d.png` |
| Genre separability — silhouette scores per feature set | `vocab_silhouette_comparison.png` |
| N-gram genre overlap heatmaps (before/after filtering) | `ngram_genre_overlap.png` |

---

## Evaluations

| Metric | Description |
|--------|-------------|
| **Silhouette score** | Geometric cluster compactness — how much closer each point is to its own cluster vs the nearest other (−1 to +1) |
| **Cohen's Kappa** | Agreement between predicted clusters (mapped by majority vote) and true genre labels |
| **Topic Coherence (C_V)** | LDA only — semantic coherence of topic top-words via Gensim |

Clustering outputs saved to `images/`:

| File | Description |
|------|-------------|
| `clustering_evaluation_heatmap.png` | Kappa and Silhouette heatmaps across all algorithm × feature combinations |
| `clustering_top15.png` | Top-15 configurations ranked by Kappa and Silhouette |
| `clustering_top5_confusion.png` | 2×3 grid of confusion matrices for the top-5 configurations by Kappa |
| `clustering_confusion_matrix.png` | Champion model confusion matrix |
| `clustering_top5_cluster_plots.png` | PCA 2D scatter plots (true genre vs predicted) for top-5 |

AllBooks equivalents: `allbooks_evaluation_heatmap.png`, `allbooks_top5_confusion.png`, `allbooks_confusion_matrix.png`, `allbooks_top5_cluster_plots.png`

---

## Language Filtering

Non-English content is excluded using two layers:

1. **Token-level** (`tokenize()` in `Feature Vector.ipynb`): tokens not in the NLTK `words` corpus (234k English words) are silently dropped — non-English terms never enter the vocabulary.
2. **Book-level** (AllBooks loading): a heuristic samples 300 tokens from each book's first 5 partitions and requires ≥60% to be English — books below this threshold are excluded entirely.

---

## Folder Structure

```
KnownBooks/          25 training books (.txt + _clean.txt), 5 per genre
AllBooks/            Full corpus (~98 books) in genre subdirectories
TestingData/         partitions.csv (5000 rows from Book partitioner.ipynb)
FeatureTrainingData/ Feature CSVs, vectorizer.pkl, selected_stylo_lmf_features.json
images/              Charts, heatmaps, confusion matrices, vocabulary analysis plots
ngrams/              Per-genre top n-gram CSVs, removed_ngrams.csv
```

---

## Dependencies

Core:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn nltk sentence-transformers
```

Optional (required for specific algorithms or features):
```bash
pip install minisom          # SOM clustering
pip install gensim           # LDA topic coherence (C_V score)
pip install tensorflow       # DEC (deep embedded clustering)
```

NLTK data (auto-downloaded on first run):
`stopwords` · `punkt` · `names` · `words` · `averaged_perceptron_tagger_eng`
