import inspect
from typing import List, Optional, Union

import torch
from haystack.document_stores import BaseDocumentStore
from haystack.nodes import EmbeddingRetriever, MultihopEmbeddingRetriever

from haystack_hybrid_embedding.pinecone.mixin import SparseDenseMixin
from haystack_hybrid_embedding.vector import SpladeEmbeddingEncoder


class SparseDenseRetriever(SparseDenseMixin, EmbeddingRetriever):
    def __init__(
        self,
        embedding_model: str,
        sparse_encoder: SpladeEmbeddingEncoder,
        document_store: Optional[BaseDocumentStore] = None,
        model_version: Optional[str] = None,
        use_gpu: bool = True,
        batch_size: int = 32,
        max_seq_len: int = 512,
        model_format: Optional[str] = None,
        pooling_strategy: str = "reduce_mean",
        emb_extraction_layer: int = -1,
        top_k: int = 10,
        progress_bar: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        scale_score: bool = True,
        embed_meta_fields: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        azure_api_version: str = "2022-12-01",
        azure_base_url: Optional[str] = None,
        azure_deployment_name: Optional[str] = None,
        alpha: float = 0.8,
    ):
        super_kwargs = inspect.signature(EmbeddingRetriever).parameters
        kwargs = {k: v for k, v in locals().items() if k in super_kwargs}
        EmbeddingRetriever.__init__(self, **kwargs)

        SparseDenseMixin.__init__(self, sparse_encoder, alpha)


class SparseDenseMultihopRetriever(SparseDenseMixin, MultihopEmbeddingRetriever):
    def __init__(
        self,
        embedding_model: str,
        sparse_encoder: SpladeEmbeddingEncoder,
        document_store: Optional[BaseDocumentStore] = None,
        model_version: Optional[str] = None,
        num_iterations: int = 2,
        use_gpu: bool = True,
        batch_size: int = 32,
        max_seq_len: int = 512,
        model_format: str = "farm",
        pooling_strategy: str = "reduce_mean",
        emb_extraction_layer: int = -1,
        top_k: int = 10,
        progress_bar: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        scale_score: bool = True,
        embed_meta_fields: Optional[List[str]] = None,
        alpha: float = 0.8,
    ):
        super_kwargs = inspect.signature(EmbeddingRetriever).parameters
        kwargs = {k: v for k, v in locals().items() if k in super_kwargs}
        EmbeddingRetriever.__init__(self, **kwargs)

        SparseDenseMixin.__init__(self, sparse_encoder, alpha)
