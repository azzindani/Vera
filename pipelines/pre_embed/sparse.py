"""BM25 sparse vectoriser · the lexical arm of hybrid search.

Hand-rolled rather than sklearn, for one reason: the vocabulary and weights ARE
the recipe. The corpus we inherited had tfidf vectors whose vocabulary was never
saved, which made them unusable for query scoring -- 20,000 opaque integers with
no term map. Everything here is written to a JSON artifact that travels with the
corpus.

BM25 over TF-IDF: term-frequency saturation and document-length normalisation,
so a long regulation does not dominate purely by being long.

Scoring is INNER PRODUCT, not cosine. Document vectors carry the BM25 weights;
the query vector carries IDF per matched term. Cosine would re-normalise the
length term that BM25 has already handled deliberately.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

# ! Part of the recipe. Doc and query MUST tokenise identically; if this regex
# changes, every vector in the corpus is invalidated.
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class Bm25Vectorizer:
    """Fit on a corpus, transform documents and queries into sparse vectors."""

    def __init__(self, vocab: dict[str, int], idf: dict[str, float], avgdl: float,
                 n_docs: int, dim: int, k1: float = K1, b: float = B):
        self.vocab = vocab
        self.idf = idf
        self.avgdl = avgdl
        self.n_docs = n_docs
        self.dim = dim
        self.k1 = k1
        self.b = b

    # -- fitting ------------------------------------------------------------

    @classmethod
    def fit(cls, docs, max_features: int = 20_000, k1: float = K1, b: float = B):
        """Build vocabulary and IDF from an iterable of texts.

        ! Fit on the FULL corpus even when only part of it will be embedded.
        Fitting is text-only CPU work; doing it once on everything means later
        batches need no re-fit and no back-fill, because vocabulary and IDF are
        already final.
        """
        df: Counter[str] = Counter()
        n_docs = 0
        total_len = 0

        for text in docs:
            toks = tokenize(text)
            if not toks:
                continue
            n_docs += 1
            total_len += len(toks)
            df.update(set(toks))

        # Top terms by document frequency, then sorted alphabetically so index
        # assignment is deterministic across runs.
        keep = sorted(t for t, _ in df.most_common(max_features))
        vocab = {t: i for i, t in enumerate(keep)}

        idf = {
            t: math.log(1.0 + (n_docs - df[t] + 0.5) / (df[t] + 0.5))
            for t in keep
        }
        avgdl = total_len / n_docs if n_docs else 0.0
        return cls(vocab, idf, avgdl, n_docs, len(keep), k1, b)

    # -- transforming -------------------------------------------------------

    def document(self, text: str) -> dict[int, float]:
        """BM25-weighted document vector, as {index: weight}."""
        toks = tokenize(text)
        if not toks:
            return {}
        tf = Counter(t for t in toks if t in self.vocab)
        dl = len(toks)
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else self.k1

        out: dict[int, float] = {}
        for term, f in tf.items():
            w = self.idf[term] * (f * (self.k1 + 1)) / (f + norm)
            if w > 0:
                out[self.vocab[term]] = w
        return out

    def query(self, text: str) -> dict[int, float]:
        """Query vector · presence-weighted, so the dot product IS the BM25 score."""
        toks = {t for t in tokenize(text) if t in self.vocab}
        return {self.vocab[t]: 1.0 for t in toks}

    # -- pgvector interop ---------------------------------------------------

    def to_sparsevec(self, vec: dict[int, float]) -> str:
        """pgvector sparsevec literal · '{idx:val,...}/dim', indices 1-based."""
        if not vec:
            return "{}/" + str(self.dim)
        body = ",".join(f"{i + 1}:{v:.6g}" for i, v in sorted(vec.items()))
        return "{" + body + "}/" + str(self.dim)

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> str:
        """Write the vectoriser and return its sha256 for corpus_meta."""
        payload = {
            "scheme": "bm25",
            "k1": self.k1,
            "b": self.b,
            "dim": self.dim,
            "n_docs": self.n_docs,
            "avgdl": self.avgdl,
            "token_pattern": TOKEN_RE.pattern,
            "lowercase": True,
            "vocab": self.vocab,
            "idf": self.idf,
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        Path(path).write_text(blob, encoding="utf-8")
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> "Bm25Vectorizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d["token_pattern"] != TOKEN_RE.pattern:
            raise ValueError(
                f"tokeniser drift: artifact used {d['token_pattern']!r}, "
                f"this code uses {TOKEN_RE.pattern!r} · vectors would not match"
            )
        return cls(d["vocab"], d["idf"], d["avgdl"], d["n_docs"], d["dim"],
                   d["k1"], d["b"])
