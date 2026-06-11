# Book Genre Classifier

Classifies text passages from books into 5 genres: **biography**, **fantasy**, **horror**, **romance**, **sci-fi**.

Six models are trained and compared: Logistic Regression, SVM, Random Forest, Naive Bayes, XGBoost, and fine-tuned DistilBERT.

---

## Run Order

Run notebooks top-to-bottom in this sequence:

| Step | Notebook | What it does |
|------|----------|-------------|
| 1 | `BOW.ipynb` | Strips Gutenberg boilerplate from `KnownBooks/`, extracts genre-level n-gram vocabulary, saves `FeatureTrainingData/vocabulary.csv` |
| 2 | `Book partitioner.ipynb` | Samples 200 × 400-word partitions from each of the 25 training books, saves `TestingData/partitions.csv` |
| 3 | `Genre classifier.ipynb` | Trains all 6 models, runs 10-fold CV, plots confusion matrices, saves models to `models/` |
| 4 | `Predict New Book.ipynb` | Predicts genre for up to 5 custom books — edit the `BOOKS` list to point to your files |
| 5 | `Batch Evaluation.ipynb` | Downloads ~100 unseen books from Project Gutenberg into `UnknownBooks/`, evaluates all models |

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
pip install scikit-learn pandas numpy matplotlib seaborn xgboost transformers accelerate joblib nltk
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
