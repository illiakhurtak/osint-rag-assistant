from pathlib import Path
import json
import os

HEADERS = [
    'TITLE:',
    'SOURCE_ORG:',
    'URL:', 
    'TOPIC_CATEGORY:',      
    'SUGGESTED_FILENAME:',
    'SUMMARY:',
    'WHY_USEFUL_FOR_RAG_OSINT_DEMO:',
    'USEFUL_PARAPHRASED_TEXT_FOR_INGESTION:',
]


def parse_document(text):
    sections = dict()
    current_header = None

    for line in text.splitlines():

        found_header = False
        for header in HEADERS:
            if line.startswith(header):

                found_header = True
                current_header = header[:-1]

                sections[current_header] = []
                value = line[len(header):].strip()
                if value:
                    sections[current_header].append(value)
                break
        
        if found_header == False and current_header != None:
            sections[current_header].append(line)
    
    clean_sections = {}
    for key, lines in sections.items():
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped:
                clean_lines.append(stripped)
                
        clean_sections[key] = "\n".join(clean_lines)

    return clean_sections

def clean_ingestion_text(text):
    if 'RAG demo notes' in text:
        return text.split('RAG demo notes')[0]
    else: return text.strip()

def build_rag_text(sections):
    title = sections.get('TITLE', '')
    summary = sections.get('SUMMARY', '')
    content = sections.get('USEFUL_PARAPHRASED_TEXT_FOR_INGESTION', '')
    content = clean_ingestion_text(content)
    return f'TITLE:\n{title}\n\nSUMMARY:\n{summary}\n\nCONTENT:\n{content}'


def chunk_text(text, chunk_size=1200, overlap=400):
    if overlap >= chunk_size:
        raise ValueError('overlap must be smaller that chunk size')
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)
        
        start = start + chunk_size - overlap

    return chunks 

def build_chunk_records(sections, chunks, filename):
    records = []

    title = sections.get("TITLE", "")
    source_org = sections.get("SOURCE_ORG", "")
    url = sections.get("URL", "")
    topic_category = sections.get("TOPIC_CATEGORY", "")
    
    doc_id = "_".join(filename.split('_')[:2])

    for i, chunk in enumerate(chunks):
        chunk_data = dict()
        chunk_id = f"{doc_id}_chunk_{i:03d}"
        chunk_data['chunk_id'] = chunk_id
        chunk_data['doc_id'] = doc_id
        chunk_data['filename'] = filename
        chunk_data['title'] = title
        chunk_data['source_org'] = source_org
        chunk_data['url'] = url
        chunk_data['topic_category'] = topic_category
        chunk_data['text'] = chunk
        records.append(chunk_data)
    return records


def process_all_documents(raw_dir):
    all_documents_paths = sorted(os.listdir(raw_dir))
    all_records = []

    for path in all_documents_paths:
        if not path.endswith(".txt"):
            continue

        with open(f"{raw_dir}/{path}", 'r', encoding='utf-8') as file:
            text = file.read()

        sections = parse_document(text)
        rag_text = build_rag_text(sections)
        chunks = chunk_text(rag_text)
        records = build_chunk_records(sections, chunks, path)
        all_records.extend(records)
    return all_records

def save_records_jsonl(records, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        for record in records:
            file.write(json.dumps(record,ensure_ascii=False)+'\n')

all_records = process_all_documents("data/raw")
save_records_jsonl(all_records, "data/processed/chunks.jsonl")
print("Total chunks:", len(all_records))
print("Saved to: data/processed/chunks.jsonl")