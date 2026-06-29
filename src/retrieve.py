import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_chunks_jsonl(path):
    records = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            records.append(data)
    return records

def build_tfidf_index(chunks):
    texts = [chunk['text'] for chunk in chunks]

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

def search(query, chunks, vectorizer, matrix, top_k=5):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix)[0]

    top_indices = scores.argsort()[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        result = {
            "score": float(scores[idx]),
            "chunk_id": chunk['chunk_id'],
            "doc_id": chunk['doc_id'],
            "title": chunk['title'],
            "source_org": chunk['source_org'],
            "url": chunk['url'],
            "topic_category": chunk['topic_category'],
            "text": chunk['text']
        }
        results.append(result)
    return results


if __name__ == "__main__":
    chunks = load_chunks_jsonl("data/processed/chunks.jsonl")
    vectorizer, matrix = build_tfidf_index(chunks)