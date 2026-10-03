"""BM25 keyword retrieval module for JIGNASA 2026 Internship Information Assistant.

Indexes Notice documents using BM25Okapi over raw notice text and structured metadata fields.
"""

from __future__ import annotations

import re
from typing import Sequence
from rank_bm25 import BM25Okapi

from schemas import Notice

# Token pattern that preserves technical terms such as C++, C#, .NET, Python, Node.js, B.Tech, etc.
TOKEN_PATTERN = re.compile(r"\.?[\w+#]+(?:[\.\-][\w+#]+)*")


def tokenize(text: str) -> list[str]:
    """Tokenize and normalize text while preserving technical terminology."""
    if not text:
        return []
    return TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """In-memory BM25 index over internship notices."""

    def __init__(self) -> None:
        self.notice_ids: list[str] = []
        self.index: BM25Okapi | None = None

    def _build_document_text(self, notice: Notice) -> str:
        """Combines raw text and key structured metadata fields into an indexable document."""
        parts: list[str] = []
        if notice.title:
            parts.append(notice.title)
        if notice.organization:
            parts.append(notice.organization)
        if notice.skills:
            parts.extend(notice.skills)
        if notice.branches:
            parts.extend(notice.branches)
        if notice.location:
            parts.append(notice.location)
        if notice.degree:
            parts.append(notice.degree)
        if notice.mode:
            parts.append(notice.mode)
        if notice.duration:
            parts.append(notice.duration)
        if notice.raw_text:
            parts.append(notice.raw_text)
        return " ".join(parts)

    def build(self, notices: Sequence[Notice]) -> None:
        """Builds a BM25 index over a list of Notice objects.

        Preserves only notice IDs internally to avoid duplicating Notice objects.
        """
        if not notices:
            self.notice_ids = []
            self.index = None
            return

        self.notice_ids = [n.notice_id for n in notices]
        tokenized_corpus = [
            tokenize(self._build_document_text(n)) for n in notices
        ]

        # Handle corpus where all documents yield empty tokens
        if not any(tokenized_corpus):
            self.index = None
            return

        self.index = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Searches the BM25 index and returns a ranked list of (notice_id, bm25_score) tuples.

        Gracefully handles empty queries, uninitialized index, and top_k variations.
        """
        if not self.notice_ids or top_k <= 0:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        if self.index is None:
            # Corpus was indexed but contained no valid tokens
            return [(nid, 0.0) for nid in self.notice_ids[:top_k]]

        scores = self.index.get_scores(tokens)
        results = [
            (nid, float(score))
            for nid, score in zip(self.notice_ids, scores)
        ]
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]
