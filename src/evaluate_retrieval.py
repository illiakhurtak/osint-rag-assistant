from retrieve import search, build_tfidf_index, load_chunks_jsonl
import json

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

if __name__ == "__main__":
    eval_cases = load_eval_cases('eval/retrieval_eval.json')
    chunks = load_chunks_jsonl('data/processed/chunks.jsonl')
    vectorizer, matrix = build_tfidf_index(chunks)
    eval_results = []
    for case in eval_cases:
        search_result = search(query=case['query'], 
                        chunks=chunks,
                        vectorizer=vectorizer,
                        matrix=matrix,
                        top_k=5)
        eval_result = evaluate_case(case, search_result)
        eval_results.append(eval_result)
    total_cases = len(eval_cases)
    hits = [eval_result['hit'] for eval_result in eval_results]
    reciprocal_ranks = [eval_result['reciprocal_rank'] for eval_result in eval_results]
    recall_at_5 = sum(hits) / total_cases
    recall_at_1 = sum([eval_result['doc_rank'] == 1 for eval_result in eval_results if eval_result['hit']]) / total_cases
    recall_at_3 = sum([eval_result['doc_rank'] < 4 for eval_result in eval_results if eval_result['hit']]) / total_cases
    low_rank_cases = [[case['id'], case['query'], case['expected_doc_ids'],res['chunk_rank'], res['doc_rank'], res['retrieved_doc_ids'],res['unique_retrieved_doc_ids'], res['scores']] for res, case in zip(eval_results, eval_cases) if res['hit'] and res['doc_rank'] > 1] 
    mrr = sum(reciprocal_ranks) / total_cases
    misses = [[case['query'], case['expected_doc_ids'], res['retrieved_doc_ids'], res['scores']] for res, case in zip(eval_results, eval_cases) if not res['hit']]
    print('Total cases: ', total_cases)
    print('Recall@5: ',recall_at_5)
    print('Recall@3: ',recall_at_3)
    print('Recall@1: ',recall_at_1)
    print('MRR@5: ',mrr)
    print("Misses: ", misses)
    print('==========================================')
    print('Low-Rank Cases: ')
    print('==========================================')

    for case in low_rank_cases:
        print("ID:", case[0])
        print("QUERY:", case[1])
        print("EXPECTED:", case[2])
        print("CHUNK_RANK:", case[3])
        print("DOC_RANK:", case[4])
        print("RETRIEVED:", case[5])
        print("UNIQUE RETRIEVED", case[6])
        print("SCORES:", case[7])
        print('==========================================')


   