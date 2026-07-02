def fuse_results_rrf(tfidf_results, embedding_results, top_k=5, rrf_k=60):
    fused = {}

    for rank, result in enumerate(tfidf_results, start=1):
        chunk_id = result['chunk_id']
        fused[chunk_id] = result.copy()
        fused[chunk_id]['score'] = 1 / (rrf_k + rank)
        fused[chunk_id]['tfidf_score'] = result['score']
        fused[chunk_id]['tfidf_rank'] = rank

    for rank, result in enumerate(embedding_results, start=1):
        chunk_id = result['chunk_id']

        if chunk_id not in fused:
            fused[chunk_id] = result.copy()
            fused[chunk_id]['score'] = 0
        fused[chunk_id]['score'] += 1 / (rrf_k + rank)
        fused[chunk_id]['embedding_score'] = result['score']
        fused[chunk_id]['embedding_rank'] = rank

    fused_results = list(fused.values())
    fused_results = sorted(fused_results, key=lambda item: item["score"], reverse=True)
    return fused_results[:top_k]


def hybrid_search(query, tfidf_retriever, embedding_retriever, candidate_k=20, top_k=5, rrf_k=60):
    tfidf_results = tfidf_retriever(query, candidate_k)
    embedding_results = embedding_retriever(query, candidate_k)
    return fuse_results_rrf(tfidf_results, embedding_results, top_k, rrf_k)
