import numpy as np
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

GENRES   = ["biography", "fantasy", "horror", "romance", "sci-fi"]
ID2GENRE = {i: g for i, g in enumerate(GENRES)}


class DistilBertGenreClassifier:
    """Lazy-loading wrapper around a fine-tuned DistilBERT genre classifier.

    Accepts raw text strings (not TF-IDF features).
    """

    def __init__(self, model_dir='models/model_distilbert', max_length=512, batch_size=32):
        self.model_dir  = model_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self._tokenizer = None
        self._model     = None
        self._device    = None

    def _load(self):
        if self._model is None:
            self._device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_dir)
            self._model     = DistilBertForSequenceClassification.from_pretrained(self.model_dir)
            self._model.to(self._device).eval()

    def _logits(self, texts):
        self._load()
        all_logits = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            enc   = self._tokenizer(
                batch, truncation=True, padding=True,
                max_length=self.max_length, return_tensors='pt',
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                all_logits.append(self._model(**enc).logits.cpu())
        return torch.cat(all_logits, dim=0)

    def predict(self, texts):
        ids = torch.argmax(self._logits(texts), dim=-1).numpy()
        return np.array([ID2GENRE[i] for i in ids])

    def predict_proba(self, texts):
        return F.softmax(self._logits(texts), dim=-1).numpy()
