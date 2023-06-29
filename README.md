# `haystack-hybrid-embedding`

Recently, Pinecone announced support for [Sparse-dense embeddings](https://docs.pinecone.io/docs/hybrid-search), allowing for hybrid vector search (both semantic and keyword search).

[`haystack`](https://haystack.deepset.ai/) is a fantastic NLP framework that does not yet support hybrid vectors for [`Retrievers`](https://docs.haystack.deepset.ai/docs/retriever).

This little library helps temporarily bridge the gap!

## Installation

```shell
$ pip install haystack-hybrid-embedding
```

## Usage

```python
from haystack_hybrid_embedding import SpladeEmbeddingEncoder
from haystack_hybrid_embedding.pinecone import PineconeHybridDocumentStore, SparseDenseRetriever

document_store = PineconeHybridDocumentStore(...)

retriever = SparseDenseRetriever(
  sparse_encoder=SpladeEmbeddingEncoder(),
  alpha=0.8,
  ...
)
```

### Replacing `EmbeddingRetriever`

Simply replace your imports of `PineconeDocumentStore` and `EmbeddingRetriever`/`MultihopEmbeddingRetriever`.



```diff
1,2c1,3
< from haystack.document_stores.pinecone import PineconeDocumentStore
< from haystack.nodes import EmbeddingRetriever, MultihopEmbeddingRetriever
---
> from haystack_hybrid_embedding import SpladeEmbeddingEncoder
> from haystack_hybrid_embedding.pinecone import PineconeHybridDocumentStore, SparseDenseRetriever, SparseDenseMultihopRetriever
>
4c5
< document_store = PineconeDocumentStore(...)
---
> document_store = PineconeHybridDocumentStore(...)
6c7
< retriever = EmbeddingRetriever(
---
> retriever = SparseDenseRetriever(
7a9,10
>     sparse_encoder=SpladeEmbeddingEncoder(),
>     alpha=0.8,
```

## Config

There are only two additional parameters exposed on `SparseDenseRetriever` over `EmbeddingRetriever`:

- `sparse_encoder` embeds both queries and documents into sparse vectors
- `alpha` controls the weighting between the sparse and dense vectors (`0` is all sparse, and `1` is all dense)
