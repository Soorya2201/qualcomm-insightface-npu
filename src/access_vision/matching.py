from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Match:
    allowed: bool
    person_id: str | None
    cosine_similarity: float


class AllowList:
    def __init__(self, names: list[str], embeddings: np.ndarray, threshold: float) -> None:
        self.names = names
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(names):
            raise ValueError("Embeddings must be a 2D matrix with one row per name")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms < 1e-12):
            raise ValueError("Database contains a zero embedding")
        self.embeddings = matrix / norms
        self.threshold = threshold

    @classmethod
    def load(
        cls, path: Path, threshold: float, expected_dimension: int | None = None
    ) -> "AllowList":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or not data:
            raise ValueError(f"Allow-list is empty or invalid: {path}")

        # The current format stores multiple templates per identity. A flat
        # vector from the earlier averaged format remains valid as one template.
        names: list[str] = []
        embeddings: list[np.ndarray] = []
        lengths: set[int] = set()
        for name, stored in data.items():
            try:
                templates = np.asarray(stored, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid embeddings for identity {name!r}") from exc
            if templates.ndim == 1:
                templates = templates.reshape(1, -1)
            if templates.ndim != 2 or templates.shape[0] == 0 or templates.shape[1] == 0:
                raise ValueError(f"Identity {name!r} has no valid embeddings")
            names.extend([str(name)] * templates.shape[0])
            embeddings.extend(templates)
            lengths.add(templates.shape[1])

        if len(lengths) != 1:
            raise ValueError(f"Allow-list contains inconsistent embedding dimensions: {lengths}")
        if expected_dimension is not None and lengths != {expected_dimension}:
            raise ValueError(
                f"Expected {expected_dimension} values per embedding, got {lengths}"
            )
        return cls(names, np.asarray(embeddings), threshold)

    def match(self, embedding: np.ndarray) -> Match:
        # np.asarray can return the caller's own buffer; the in-place divide
        # below would then mutate a stored template or a cached embedding.
        vector = np.array(embedding, dtype=np.float32).reshape(-1)
        expected_dimension = self.embeddings.shape[1]
        if vector.size != expected_dimension:
            raise ValueError(
                f"Expected a {expected_dimension}-value query embedding, got {vector.size}"
            )
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        similarities = self.embeddings @ vector
        index = int(np.argmax(similarities))
        score = float(similarities[index])
        return Match(score >= self.threshold, self.names[index] if score >= self.threshold else None, score)
