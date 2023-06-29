from typing import List

import numpy as np
from haystack import Document
from pinecone_text.hybrid import hybrid_convex_scale
from pinecone_text.sparse import SparseVector

from haystack_hybrid_embedding.vector import HybridValues, SpladeEmbeddingEncoder


class SparseDenseMixin:
    def __init__(self, sparse_encoder: SpladeEmbeddingEncoder, alpha: float = 0.8):
        self.sparse_encoder = sparse_encoder
        self.alpha = alpha

    def _hybrid_embed(
        self, dense: List[List[float]], sparse: List[SparseVector]
    ) -> np.ndarray:
        hybrid = [
            hybrid_convex_scale(d, s, self.alpha) for (d, s) in zip(dense, sparse)
        ]
        return np.stack([HybridValues.register(d, s) for (d, s) in hybrid])

    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        dense = super().embed_documents(documents).tolist()
        sparse = self.sparse_encoder.embed_documents(documents)
        return self._hybrid_embed(dense, sparse)

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        dense = super().embed_queries(queries).tolist()
        sparse = self.sparse_encoder.embed_queries(queries)
        return self._hybrid_embed(dense, sparse)
