from retrieve import search, build_tfidf_index, load_chunks_jsonl
from retrieve_embeddings import search_by_embedding, load_jsonl, build_embedding_index, load_query_embedding_cache, get_or_create_query_embedding, embedding_model
import json
from retrieve_hybrid import hybrid_search

def load_eval_cases(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def deduplicate_preserving_order(doc_ids):
    unique_docs = []
    for doc_id in doc_ids:
        if doc_id not in unique_docs:
            unique_docs.append(doc_id)
    return unique_docs

def find_first_relevant_rank(retrieved_doc_ids, expected_doc_ids):
    rank_number = None
    for doc in retrieved_doc_ids:
        if doc in expected_doc_ids:
            rank_number = retrieved_doc_ids.index(doc) + 1
            break
    return rank_number


def evaluate_case(case, results):
    docs = [res['doc_id'] for res in results]
    unique_docs = deduplicate_preserving_order(docs)
    scores = [res['score'] for res in results]
    chunk_rank = find_first_relevant_rank(docs, case['expected_doc_ids'])
    doc_rank = find_first_relevant_rank(unique_docs, case['expected_doc_ids'])
    hit = doc_rank is not None
    rr = 1 / doc_rank if hit else 0
    return {'hit': hit, 
            'chunk_rank': chunk_rank,
            'doc_rank':doc_rank, 
            "reciprocal_rank": rr, 
            'retrieved_doc_ids':docs, 
            'scores':scores, 
            'unique_retrieved_doc_ids':unique_docs,
            }

def compute_metrics(eval_results, eval_cases):
    total_cases = len(eval_cases)
    hits = [eval_result['hit'] for eval_result in eval_results]
    reciprocal_ranks = [eval_result['reciprocal_rank'] for eval_result in eval_results]
    recall_at_5 = sum(hits) / total_cases
    recall_at_1 = sum([eval_result['doc_rank'] == 1 for eval_result in eval_results if eval_result['hit']]) / total_cases
    recall_at_3 = sum([eval_result['doc_rank'] <= 3 for eval_result in eval_results if eval_result['hit']]) / total_cases
    low_rank_cases = [[case['id'], case['query'], case['expected_doc_ids'],res['chunk_rank'], res['doc_rank'], res['retrieved_doc_ids'],res['unique_retrieved_doc_ids'], res['scores']] for res, case in zip(eval_results, eval_cases) if res['hit'] and res['doc_rank'] > 1] 
    mrr = sum(reciprocal_ranks) / total_cases
    misses = [{'id': case['id'],'query': case['query'], 'expected_doc_ids': case['expected_doc_ids'], 'retrieved_doc_ids': res['retrieved_doc_ids'], 'scores': res['scores']} for res, case in zip(eval_results, eval_cases) if not res['hit']]
    return {
        'metrics': {
            'total_cases': total_cases,
            'recall@1': recall_at_1,
            'recall@3': recall_at_3,
            'recall@5': recall_at_5,
            'mrr@5': mrr
        },
        'misses': misses,
        'low_rank_cases':[
            {"id": case[0],
            "query": case[1],
            "expected_doc_ids": case[2],
            "chunk_rank": case[3],
            "doc_rank": case[4],
            "retrieved_doc_ids": case[5],
            "unique_retrieved_doc_ids": case[6],
            "scores": case[7]}
            for case in low_rank_cases
        ]
    }


def evaluate_retriever(name, eval_cases, retriever, top_k=5):
    eval_results = []
    for case in eval_cases:
        results = retriever(case['query'], top_k)
        eval_result = evaluate_case(case, results)
        eval_results.append(eval_result)
    report = compute_metrics(eval_results, eval_cases)
    report['name'] = name
    return report


def print_report(report):
    print(report['name'])
    print(report['metrics'])
    print('Misses:', len(report['misses']))
    print('Low-rank cases:', len(report['low_rank_cases']))
    print("===================================")


def evaluate_hybrid_grid(eval_cases, tfidf_retriever, embedding_retriever, candidate_ks, rrf_ks, top_k=5):
    reports = []
    for candidate_k in candidate_ks:
        for rrf_k in rrf_ks:
            hybrid_retriever = lambda query, top_k: hybrid_search(
                query=query,
                tfidf_retriever=tfidf_retriever,
                embedding_retriever=embedding_retriever,
                candidate_k=candidate_k,
                top_k=top_k,
                rrf_k=rrf_k
            )
            report = evaluate_retriever(
                f"hybrid_rrf_candidate_{candidate_k}_rrf_{rrf_k}",
                eval_cases,
                hybrid_retriever,
                top_k=top_k
            )
            report['candidate_k'] = candidate_k
            report['rrf_k'] = rrf_k
            reports.append(report)
    return reports


if __name__ == "__main__":
    eval_cases = load_eval_cases('eval/retrieval_eval.json')
    chunks = load_chunks_jsonl('data/processed/chunks.jsonl')

    vectorizer, matrix = build_tfidf_index(chunks)
    tfidf_retriever = lambda query, top_k: search(query, chunks, vectorizer, matrix, top_k)

    embedding_records = load_jsonl('data/processed/chunk_embeddings.jsonl')
    candidate_chunks, candidate_vectors = build_embedding_index(chunks, embedding_records)
    query_cache_path = "eval/query_embeddings.jsonl"
    query_cache = load_query_embedding_cache(query_cache_path)

    def embedding_retriever(query, top_k):
        query_embedding = get_or_create_query_embedding(query=query, model=embedding_model, cache=query_cache, cache_path=query_cache_path)
        results = search_by_embedding(query_embedding=query_embedding, candidate_chunks=candidate_chunks, candidate_vectors=candidate_vectors, top_k=top_k)
        return results

    report = evaluate_retriever("tfidf", eval_cases, tfidf_retriever, top_k=5)
    report_embedding = evaluate_retriever('embedding', eval_cases, embedding_retriever, top_k=5)

    hybrid_retriever = lambda query, top_k: hybrid_search(
        query=query,
        tfidf_retriever=tfidf_retriever,
        embedding_retriever=embedding_retriever,
        candidate_k=30,
        top_k=top_k,
        rrf_k=10
    )
    hybrid_report = evaluate_retriever('hybrid_rrf_tuned', eval_cases, hybrid_retriever, top_k=5)

    print_report(report)
    print_report(report_embedding)
    print_report(hybrid_report)

    grid_reports = evaluate_hybrid_grid(
        eval_cases=eval_cases,
        tfidf_retriever=tfidf_retriever,
        embedding_retriever=embedding_retriever,
        candidate_ks=[10, 20, 30, 50],
        rrf_ks=[10, 30, 60, 100],
        top_k=5
    )
    grid_reports = sorted(
        grid_reports,
        key=lambda item: (
            item['metrics']['mrr@5'],
            item['metrics']['recall@5'],
            item['metrics']['recall@1'],
            -len(item['misses'])
        ),
        reverse=True
    )

    print("Hybrid tuning top configs")
    for report in grid_reports[:5]:
        print(
            f"candidate_k={report['candidate_k']}",
            f"rrf_k={report['rrf_k']}",
            report['metrics'],
            "Misses:",
            len(report['misses']),
            "Low-rank cases:",
            len(report['low_rank_cases'])
        )
