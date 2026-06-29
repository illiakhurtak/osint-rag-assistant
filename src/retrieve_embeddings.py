import json 
from openai import OpenAI
import os
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
embedding_model = 'text-embedding-3-small'

if not openai_api_key:
    raise ValueError("OPENAI KEY IS NOT SET")

client = OpenAI()

def load_jsonl(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            records.append(data)
    return records

def build_embedding_lookup(embedding_records):
    embeddings = dict()
    for elem in embedding_records:
        embeddings[elem['chunk_id']] = elem['embedding']
    return embeddings

def build_embedding_index(chunks, embedding_records):
    lookup = build_embedding_lookup(embedding_records)
    candidate_chunks=[]
    candidate_vectors=[]

    for chunk in chunks:
        chunk_id = chunk['chunk_id']

        if chunk_id not in lookup:
            continue
        candidate_chunks.append(chunk)
        candidate_vectors.append(lookup[chunk_id])

    return candidate_chunks, candidate_vectors

def embed_query(query, model):
    query = query.replace("\n", " ")
    response = client.embeddings.create(input=[query], model=model)
    return response.data[0].embedding

def search_embeddings(query, candidate_chunks, candidate_vectors, model, top_k=5):
    embedded_query = embed_query(query, model)
    scores = cosine_similarity([embedded_query], candidate_vectors)[0]
    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk = candidate_chunks[idx]
        score = scores[idx]
        results.append({
            "score": float(score),
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "source_org": chunk["source_org"],
            "url": chunk["url"],
            "topic_category": chunk["topic_category"],
            "text": chunk["text"]
        })


    return results
