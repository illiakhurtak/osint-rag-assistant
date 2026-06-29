import os
from retrieve import load_chunks_jsonl
from openai import OpenAI
import json 
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
embedding_model = 'text-embedding-3-small'

if not openai_api_key:
    raise ValueError("OPENAI KEY IS NOT SET")

client = OpenAI()
file_path = "data/processed/chunk_embeddings.jsonl"

def get_embedding(text, model):
    text = text.replace("\n", " ")

    response = client.embeddings.create(
        input=[text],
        model=model
    )

    return response.data[0].embedding

def load_existing_chunk_ids(output_path):
    output_path = Path(output_path)
    if not output_path.is_file():
        return set()
    existing = set()
    with open(output_path, 'r', encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if chunk['embedding']:
                existing.add(chunk['chunk_id'])
    return existing

def embed_all_chunks(chunks):
    existing = load_existing_chunk_ids(file_path)
    with open(file_path, 'a', encoding="utf-8") as f:
        for chunk in chunks:
            if chunk['chunk_id'] in existing:
                print(f'Skipping {chunk['chunk_id']}')
                continue
            print(f'Embedding {chunk['chunk_id']}')
            chunk_record = {
                'chunk_id': chunk['chunk_id'],
                'doc_id': chunk['doc_id'],
                'model': embedding_model,
                'embedding': get_embedding(chunk['text'], model=embedding_model)
            }
            json_string = json.dumps(chunk_record,ensure_ascii=False)

            f.write(json_string + "\n")
            f.flush()


if __name__ == "__main__":
    embed_all_chunks(load_chunks_jsonl('data/processed/chunks.jsonl'))
    
    