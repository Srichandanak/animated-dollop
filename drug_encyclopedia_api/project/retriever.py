# retriever.py
import os
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# Cross-encoder model for reranking (small, fast)
_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """Tokenise chunk texts and build a BM25 index."""
    tokenised = [chunk["text"].lower().split() for chunk in chunks]
    return BM25Okapi(tokenised)


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60
) -> list[dict]:
    """
    Merges two ranked lists using Reciprocal Rank Fusion.
    Returns deduplicated list ordered by fused score (highest first).
    k=60 is the standard constant that dampens high-rank advantages.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, doc in enumerate(vector_results):
        key = doc["text"]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_results):
        key = doc["text"]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in ranked]


def hybrid_retrieve(
    query: str,
    collection,           # Chroma collection
    model,                # SentenceTransformer
    chunks: list[dict],   # all chunks (needed for BM25)
    bm25_index: BM25Okapi,
    source_filter: str,   # "prescription" or "homeopathic"
    top_k: int = 10,
    rerank_top_n: int = 3
) -> list[dict]:
    """
    Full pipeline: vector search + BM25 → RRF merge → cross-encoder rerank.
    Returns top rerank_top_n chunks with their text and metadata.
    """

    # 1. Vector search via Chroma
    query_embedding = model.encode([query]).tolist()
    chroma_results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where={"type": source_filter}  # filter by source at retrieval time
    )
    vector_docs = []
    if chroma_results["documents"] and chroma_results["documents"][0]:
        for text, meta in zip(chroma_results["documents"][0], chroma_results["metadatas"][0]):
            vector_docs.append({"text": text, **meta})

    # 2. BM25 keyword search
    tokenised_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenised_query)
    # Get indices of top_k BM25 results for the right source
    filtered = [
        (i, score) for i, (score, chunk) in enumerate(zip(bm25_scores, chunks))
        if chunk["type"] == source_filter
    ]
    filtered.sort(key=lambda x: x[1], reverse=True)
    bm25_docs = [chunks[i] for i, _ in filtered[:top_k]]

    # 3. RRF merge
    merged = reciprocal_rank_fusion(vector_docs, bm25_docs)

    # 4. Cross-encoder reranking
    if not merged:
        return []

    cross_enc = get_cross_encoder()
    pairs = [[query, doc["text"]] for doc in merged[:20]]  # rerank top 20 candidates
    ce_scores = cross_enc.predict(pairs)

    reranked = sorted(
        zip(merged[:20], ce_scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in reranked[:rerank_top_n]]