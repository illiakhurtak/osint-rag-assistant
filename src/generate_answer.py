from retrieve_hybrid import hybrid_search
from retrieve import load_chunks_jsonl, build_tfidf_index, search
from retrieve_embeddings import load_jsonl, build_embedding_index, load_query_embedding_cache, get_or_create_query_embedding, search_by_embedding, embedding_model
from context_builder import build_context
from openai import OpenAI

query_cache_path = 'eval/query_embeddings.jsonl'

client = OpenAI()

def build_retrievers():
    chunks = load_chunks_jsonl('data/processed/chunks.jsonl')
    vectorizer, matrix = build_tfidf_index(chunks)

    def tfidf_retriever(query, top_k):
        return search(query, chunks, vectorizer, matrix, top_k)
    
    embedding_records = load_jsonl('data/processed/chunk_embeddings.jsonl')
    candidate_chunks, candidate_vectors = build_embedding_index(chunks, embedding_records)
    query_cache = load_query_embedding_cache(query_cache_path)

    def embedding_retriever(query, top_k):
        query_embedding = get_or_create_query_embedding(query, model=embedding_model, cache=query_cache, cache_path=query_cache_path)
        return search_by_embedding(query_embedding, candidate_chunks, candidate_vectors, top_k)
    
    return tfidf_retriever, embedding_retriever

def retrieve_context(query):
    tfidf_retriever, embedding_retriever = build_retrievers()
    
    results = hybrid_search(
        query,
        tfidf_retriever,
        embedding_retriever,
        candidate_k=30,
        top_k=5,
        rrf_k=10
    )
    context = build_context(results, max_chunks=5)
    return context

def generate_answer(query):
    retrieved_context = retrieve_context(query)
    system_prompt = """
You are an OSINT RAG assistant.
Use only the provided context.
Do not use outside knowledge.
Every factual claim must be supported by citations like [1], [2].
If the context does not support an answer, say that evidence is insufficient.
Do not speculate.
If no source supports a claim, do not include that claim.
Set Human Review to Yes for active conflict, sanctions, cyber threats, critical infrastructure, defense procurement, named-entity allegations, or forecasting claims.
Return the answer in this structure:

Answer:
...

Citations:
- [1] ...
- [2] ...

Unknowns:
- ...

Confidence:
Low / Medium / High

Human Review:
Yes / No

Human Review Reason:
...
""".strip()

    user_prompt = f"""
Question:
{query}

Context:
{retrieved_context}
""".strip()
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
    )
    return response.output_text

if __name__ == "__main__":
    query = "How has the Russia-Ukraine war affected defense industrial base resilience?"
    print(generate_answer(query))