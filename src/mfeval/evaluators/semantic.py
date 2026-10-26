"""Level 2 (medium fidelity): semantic similarity to the reference answer.

The response and the reference are turned into embeddings and compared with cosine
similarity. This catches answers that pass every rule but say something unrelated or
unhelpful. It is weak on small factual errors: "312 km" and "213 km" look almost the
same to an embedding model.

Two embedders:
* "st"    – sentence-transformers (PyTorch), all-MiniLM-L6-v2 by default. Use this for real runs.
* "tfidf" – character n-gram TF-IDF from scikit-learn. No download, used in CI and tests.
"""

from __future__ import annotations

import time

import numpy as np

from ..schema import AssistantOutput, EvalResult, TestCase

DEFAULT_THRESHOLDS = {"st": 0.55, "tfidf": 0.35}


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        m = self.vectorizer.transform(texts).toarray()
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)


class SentenceTransformerEmbedder:
    name = "st"

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError(
                "sentence-transformers is not installed. Run `pip install -r requirements.txt` "
                "or use --embedder tfidf."
            ) from e
        self.model = SentenceTransformer(model)

    def fit(self, texts: list[str]) -> None:  # nothing to fit, kept for the same interface
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def make_embedder(name: str):
    if name == "st":
        return SentenceTransformerEmbedder()
    if name == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"unknown embedder: {name}")


def evaluate_batch(cases: list[TestCase], outputs: list[AssistantOutput], embedder,
                   threshold: float | None = None) -> list[EvalResult]:
    """Score all outputs at once (much faster than one by one)."""
    threshold = DEFAULT_THRESHOLDS.get(embedder.name, 0.5) if threshold is None else threshold
    refs = [c.reference for c in cases]
    resps = [o.response or "" for o in outputs]
    embedder.fit(refs + resps)

    start = time.perf_counter()
    ref_vecs = embedder.encode(refs)
    resp_vecs = embedder.encode(resps)
    sims = np.clip(np.sum(ref_vecs * resp_vecs, axis=1), 0.0, 1.0)
    per_item = (time.perf_counter() - start) / max(len(cases), 1)

    results = []
    for case, out, sim in zip(cases, outputs, sims):
        if out.error or not out.response:
            sim = 0.0
        results.append(EvalResult(case.id, "L2", bool(sim >= threshold), round(float(sim), 4),
                                  f"cosine={sim:.3f}", per_item))
    return results
