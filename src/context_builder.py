def deduplicate_by_doc_id(results):
    seen_doc_ids = set()
    deduped = []
    for res in results:
        doc_id = res['doc_id']

        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        deduped.append(res)
    return deduped
    
def build_context(results, max_chunks=5):
    deduped_result = deduplicate_by_doc_id(results)[:max_chunks]

    text_blocks = []

    for i, doc in enumerate(deduped_result, start=1):
        text_block = f"""
[{i}]
Title: {doc['title']}
Source: {doc['source_org']}
URL: {doc['url']}
Doc ID: {doc['doc_id']}
Chunk ID: {doc['chunk_id']}
Text:
{doc['text']}
""".strip()
        text_blocks.append(text_block)
    return "\n\n".join(text_blocks)


