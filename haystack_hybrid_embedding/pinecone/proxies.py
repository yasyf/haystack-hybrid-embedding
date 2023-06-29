from typing import Any, Optional

from haystack.document_stores.pinecone import PineconeDocumentStore
from wrapt import ObjectProxy

from haystack_hybrid_embedding.vector import HybridValues


class HybridIndex(ObjectProxy):
    def upsert(self, vectors: list, *args, **kwargs):
        vectors = [
            {
                "id": id,
                "values": hybrid.values.tolist() if hybrid else values,
                "metadata": meta,
                **({"sparse_values": hybrid.sparse_values} if hybrid else {}),
            }
            for (id, values, meta) in vectors
            for hybrid in [HybridValues.pop(values)]
        ]
        return self.__wrapped__.upsert(vectors, *args, **kwargs)

    def query(self, vector: Optional[Any] = None, *args, **kwargs):
        if isinstance(vector, HybridValues) and "sparse_vector" not in kwargs:
            kwargs["sparse_vector"] = vector.sparse_values
            vector = vector.values
        return self.__wrapped__.query(vector, *args, **kwargs)


class PineconeHybridDocumentStore(PineconeDocumentStore):
    def _create_index(self, *args, **kwargs):
        index = super()._create_index(*args, **kwargs)
        if not isinstance(index, HybridIndex):
            index = HybridIndex(index)
        return index
