from dataclasses import dataclass
from typing import ClassVar, List, Optional, cast

import numpy as np
from haystack import Document
from haystack.nodes.retriever._base_embedding_encoder import _BaseEmbeddingEncoder
from pinecone_text.sparse import SparseVector, SpladeEncoder


class SpladeEmbeddingEncoder(_BaseEmbeddingEncoder):
    def __init__(self, device: str = "cpu"):
        self.splade = SpladeEncoder(device=device)

    def embed_queries(self, queries: List[str]) -> List[SparseVector]:
        return cast(List[SparseVector], self.splade.encode_queries(queries))

    def embed_documents(self, docs: List[Document]) -> List[SparseVector]:
        return cast(
            List[SparseVector],
            self.splade.encode_documents([doc.content for doc in docs]),
        )


@dataclass
class HybridValues:
    registry: ClassVar[List["HybridValues"]] = []

    values: np.ndarray
    sparse_values: Optional[SparseVector]

    def __post_init__(self):
        self.registry.append(self)

    @classmethod
    def register(cls, values: List[float], sparse_values: SparseVector) -> np.ndarray:
        return cls(np.array(values), sparse_values).values

    @classmethod
    def pop(cls, values: List[float]) -> Optional["HybridValues"]:
        for i, hybrid in enumerate(cls.registry):
            if np.allclose(hybrid.values, values):
                cls.registry.pop(i)
                return hybrid
