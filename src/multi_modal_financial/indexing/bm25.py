"""BM25 Okapi Sparse Search Engine with financial tokenization."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


_PAREN_CLEAN_PATTERN = re.compile(
    r"\(([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?(?:[MBKmbk%])?)\)"
)
_TOKEN_PATTERN = re.compile(
    r"\$?-?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?%?(?:[a-zA-Z])?|[a-zA-Z0-9_\-]+"
)


class BM25Index:
    """Okapi BM25 implementation tuned for financial vocabulary and numerical terms."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size: int = 0
        self.avgdl: float = 0.0
        self.doc_freqs: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.doc_len: list[int] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_ids: list[str] = []
        self.doc_id_to_idx: dict[str, int] = {}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenize preserving financial tickers, numbers, percentages, and currencies."""
        if not text:
            return []

        tokens: list[str] = []

        # Handle accounting parenthesized negatives, e.g. (1,234.5M) -> -1234.5m
        paren_clean = _PAREN_CLEAN_PATTERN.sub(r"-\1", text)

        # Regex captures:
        # - currencies & numbers: $12,500, -$45.2B, 14.5%
        # - alphanumeric tokens, tickers, acronyms: AAPL, FY2025, Q3, EBITDA
        raw_tokens = _TOKEN_PATTERN.findall(paren_clean)

        for tok in raw_tokens:
            t_low = tok.lower()
            if len(t_low) > 1 or t_low.isdigit():
                tokens.append(t_low)
                if "," in t_low:
                    tokens.append(t_low.replace(",", ""))
                # If token is a currency or percentage, also add raw number
                if t_low.startswith("$"):
                    num_part = t_low.lstrip("$")
                    if num_part:
                        tokens.append(num_part)
                        if "," in num_part:
                            tokens.append(num_part.replace(",", ""))
                elif t_low.endswith("%"):
                    num_part = t_low.rstrip("%")
                    if num_part:
                        tokens.append(num_part)
                        if "," in num_part:
                            tokens.append(num_part.replace(",", ""))

        return tokens

    def fit(self, doc_ids: list[str], documents: list[str]) -> BM25Index:
        """Fit index on collection of document texts."""
        self.doc_ids = list(doc_ids)
        self.doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        self.corpus_size = len(documents)
        self.doc_tokens = []
        self.doc_len = []
        self.doc_freqs = Counter()

        total_length = 0
        for text in documents:
            tokens = self.tokenize(text)
            self.doc_tokens.append(tokens)
            tok_len = len(tokens)
            self.doc_len.append(tok_len)
            total_length += tok_len

            unique_terms = set(tokens)
            for t in unique_terms:
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self.avgdl = (total_length / self.corpus_size) if self.corpus_size > 0 else 0.0
        self._compute_idf()
        return self

    def add_documents(self, doc_ids: list[str], documents: list[str]) -> BM25Index:
        """Incrementally add documents to existing index."""
        if self.corpus_size == 0:
            return self.fit(doc_ids, documents)

        for doc_id, text in zip(doc_ids, documents, strict=False):
            if doc_id in self.doc_id_to_idx:
                # Update existing document
                idx = self.doc_id_to_idx[doc_id]
                old_tokens = set(self.doc_tokens[idx])
                for t in old_tokens:
                    self.doc_freqs[t] = max(0, self.doc_freqs.get(t, 1) - 1)
                new_tokens = self.tokenize(text)
                self.doc_tokens[idx] = new_tokens
                self.doc_len[idx] = len(new_tokens)
                for t in set(new_tokens):
                    self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1
            else:
                idx = len(self.doc_ids)
                self.doc_ids.append(doc_id)
                self.doc_id_to_idx[doc_id] = idx
                tokens = self.tokenize(text)
                self.doc_tokens.append(tokens)
                self.doc_len.append(len(tokens))
                for t in set(tokens):
                    self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self.corpus_size = len(self.doc_ids)
        total_len = sum(self.doc_len)
        self.avgdl = (total_len / self.corpus_size) if self.corpus_size > 0 else 0.0
        self._compute_idf()
        return self

    def _compute_idf(self) -> None:
        """Compute IDF with standard Okapi smoothing."""
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            idf_val = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[term] = max(idf_val, 1e-4)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search query against BM25 index and return (doc_id, score) pairs."""
        if self.corpus_size == 0:
            return []

        q_tokens = self.tokenize(query)
        if not q_tokens:
            return []

        scores: list[float] = [0.0] * self.corpus_size

        for term in q_tokens:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for doc_idx, tokens in enumerate(self.doc_tokens):
                tf = tokens.count(term)
                if tf == 0:
                    continue
                doc_len = self.doc_len[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                numerator = tf * (self.k1 + 1.0)
                scores[doc_idx] += idf * (numerator / denom)

        ranked_indices = sorted(range(self.corpus_size), key=lambda i: scores[i], reverse=True)
        results = [
            (self.doc_ids[i], float(scores[i])) for i in ranked_indices[:top_k] if scores[i] > 0
        ]
        return results

    def get_doc_term_frequency(self, doc_id: str, term: str) -> int:
        """Return the count of a term within a specific indexed document."""
        idx = self.doc_id_to_idx.get(doc_id)
        if idx is None:
            return 0
        term_clean = term.lower()
        return self.doc_tokens[idx].count(term_clean)

    def corpus_statistics(self) -> dict[str, Any]:
        """Return summary metrics of the current index state."""
        return {
            "corpus_size": self.corpus_size,
            "vocabulary_size": len(self.doc_freqs),
            "avg_document_length": round(self.avgdl, 2),
            "k1": self.k1,
            "b": self.b,
        }
