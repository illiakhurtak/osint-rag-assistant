# OSINT RAG Assistant

Production Retrieval-Augmented Generation project for OSINT and geopolitical analysis.

This repository is my educational but engineering-focused implementation of a RAG system built from scratch. The goal is to  implement the core components behind a LLM assistant: document ingestion, parsing, chunking, metadata, sparse retrieval, dense retrieval, evaluation, and the next steps toward grounded answer generation and agentic workflows.

The current version is focused on the retrieval and evaluation layer: turning curated OSINT-style documents into searchable chunks, comparing TF-IDF retrieval against OpenAI embedding retrieval, and measuring quality with a validation dataset.

## Project Goal

The long-term goal is to build an OSINT RAG assistant that can answer questions about:

- geopolitics;
- critical infrastructure;
- semiconductor supply chains;
- sanctions and export controls;
- defense industrial base risks;
- cyber threats;
- influence operations and disinformation;
- OSINT methodology.

Answers should eventually be grounded in retrieved documents and include citations, confidence, unknowns, contradictions, and human-review signals.

## Current Status

Implemented:

- ingestion pipeline for semi-structured text documents;
- document parser for metadata fields;
- cleaning of ingestion text;
- fixed-size chunking with overlap;
- chunk-level metadata;
- JSONL corpus generation;
- TF-IDF retrieval baseline;
- OpenAI embedding generation;
- dense semantic retrieval with cosine similarity;
- retrieval validation dataset;
- retrieval metrics: Recall@1, Recall@3, Recall@5, MRR@5;
- miss and low-rank diagnostics.

Not implemented yet:

- hybrid retrieval;
- reranking;
- LLM answer generation;
- citation formatting;
- confidence estimation;
- contradiction detection;
- human-review routing;
- agentic workflow / tool use.

## Why This Project Matters

This project is designed around the responsibilities of an Agentic AI Engineer with focus on OSINT/geopolitical analysis:

- own document-processing pipelines;
- build retrieval systems with both sparse and dense methods;
- reason about chunking, embeddings, similarity, and ranking;
- evaluate retrieval quality instead of relying on manual inspection;
- prepare the foundation for grounded generation and agentic workflows.

## Dataset

The current corpus contains curated OSINT-style text documents covering topics such as semiconductor supply chains, critical infrastructure, cyber threats, maritime disruption, export controls, defense industrial base risk, influence operations, and Ukraine-related geopolitical risk.

Current corpus statistics:

- raw documents: 60;
- processed chunks: 171;
- embedding model: `text-embedding-3-small`;
- embedding dimensions: 1536;
- validation cases: 170.

The corpus is intentionally small enough to inspect manually, but large enough to expose real retrieval failure modes: semantic ambiguity, overlapping topics, source routing, multi-document questions, and low-rank relevant results.

## Architecture

Current pipeline:

```text
data/raw/*.txt
        |
        v
src/ingest.py
        |
        v
parse metadata + build RAG text
        |
        v
chunking with overlap
        |
        v
data/processed/chunks.jsonl
        |
        +--> src/retrieve.py              -> TF-IDF retrieval
        |
        +--> src/embed.py                 -> OpenAI embeddings
                 |
                 v
           data/processed/chunk_embeddings.jsonl
                 |
                 v
           src/retrieve_embeddings.py     -> embedding retrieval
        |
        v
src/evaluate_retrieval.py
        |
        v
retrieval metrics + miss diagnostics
```

## Retrieval Methods

### TF-IDF Baseline

The sparse retrieval baseline uses `TfidfVectorizer` with English stop words and 1-2 word ngrams.

This baseline is useful because it performs well on:

- exact terms;
- source names;
- policy names;
- entity-specific queries;
- keyword-heavy questions.

### Embedding Retrieval

Dense retrieval uses OpenAI embeddings with `text-embedding-3-small`.

This method is useful for:

- semantic paraphrases;
- conceptual questions;
- queries where the wording differs from the document;
- broader topic matching.

Both retrievers return a similar result schema:

```json
{
  "score": 0.0,
  "chunk_id": "doc_001_chunk_000",
  "doc_id": "doc_001",
  "title": "...",
  "source_org": "...",
  "url": "...",
  "topic_category": "...",
  "text": "..."
}
```

Keeping output contracts similar is important because it allows the same evaluation layer to compare different retrieval strategies.

## Evaluation

The evaluation dataset is stored in:

```text
eval/retrieval_eval.json
```

Each validation case contains:

- query;
- expected relevant document IDs;
- optionally related document IDs;
- topic;
- difficulty;
- test type;
- rationale.

The evaluator measures whether the expected document appears in the retrieved top-k results.

Metrics:

- `Recall@1`: relevant document is ranked first;
- `Recall@3`: relevant document appears in top 3;
- `Recall@5`: relevant document appears in top 5;
- `MRR@5`: mean reciprocal rank, rewarding higher-ranked relevant results.

## Current Results

Evaluation set:

- total cases: 170;
- top-k: 5.

TF-IDF baseline:

```text
Recall@5: 0.947
Recall@3: 0.918
Recall@1: 0.759
MRR@5:    0.837
```

Embedding retrieval:

```text
Recall@5: 0.959
Recall@3: 0.947
Recall@1: 0.735
MRR@5:    0.832
```

Interpretation:

- embeddings improve semantic recall in top-5 and top-3;
- TF-IDF is slightly stronger at ranking the exact expected document first;
- the next step is hybrid retrieval, combining sparse and dense scores.

This mirrors a common production RAG pattern: sparse retrieval catches exact terms, dense retrieval catches semantic meaning, and hybrid retrieval often improves robustness.

## Repository Structure

```text
.
├── data/
│   ├── raw/                       # source text documents
│   └── processed/
│       ├── chunks.jsonl            # processed chunk records
│       └── chunk_embeddings.jsonl  # generated embedding records
├── eval/
│   └── retrieval_eval.json         # retrieval validation dataset
├── src/
│   ├── ingest.py                   # parsing, cleaning, chunking, metadata
│   ├── retrieve.py                 # TF-IDF retrieval baseline
│   ├── embed.py                    # OpenAI embedding generation
│   ├── retrieve_embeddings.py      # dense retrieval over stored embeddings
│   └── evaluate_retrieval.py       # retrieval evaluation utilities
├── requirements.txt
├── sources.md
└── README.md
```

## Setup

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create an environment file:

```bash
cp .env.example .env
```

Then add your OpenAI API key:

```text
OPENAI_API_KEY=your_api_key_here
```

Do not commit `.env`.

## Usage

Generate processed chunks:

```bash
python src/ingest.py
```

Run TF-IDF retrieval evaluation:

```bash
PYTHONPATH=src python src/evaluate_retrieval.py
```

Generate embeddings:

```bash
PYTHONPATH=src python src/embed.py
```

Run embedding retrieval from Python:

```bash
PYTHONPATH=src python
```

## Engineering Notes

Important design decisions:

- JSONL is used as a simple inspectable storage format before introducing a vector database.
- TF-IDF is kept as a baseline because production systems need measurable baselines.
- Embeddings are stored separately from chunks to keep document text and vector artifacts decoupled.
- Retrieval outputs share a common schema to make evaluation reusable.
- Evaluation is document-level rather than only chunk-level because user-facing RAG answers usually need source-level grounding.

Known limitations:

- no vector database yet;
- no approximate nearest neighbor index;
- no batching for query embeddings;
- no hybrid score normalization yet;
- no reranking layer;
- no generation or citation layer yet.

## Roadmap

Next planned steps:

1. Add side-by-side evaluation for TF-IDF vs embedding retrieval.
2. Implement hybrid retrieval with score normalization.
3. Add reranking for top-k candidates.
4. Build grounded answer generation using retrieved context only.
5. Add citation extraction and answer-source mapping.
6. Add confidence and unknowns.
7. Add contradiction detection across retrieved sources.
8. Add human-review triggers for sensitive topics.
9. Design an agentic workflow for retrieval, answer drafting, verification, and fallback.
10. Introduce LangGraph after the core workflow is understood from scratch.

## Target Role Alignment

This project maps directly to LLM / RAG / Agentic AI engineering responsibilities:

- Document processing: ingestion, parsing, cleaning, chunking, metadata.
- RAG quality: sparse retrieval, dense retrieval, ranking diagnostics.
- Embeddings: OpenAI embedding generation, vector similarity, semantic search.
- Evaluation: validation dataset, Recall@k, MRR, miss analysis.
- Grounding foundation: chunk metadata, document IDs, source URLs, result contracts.
- Agentic roadmap: future retrieval-augmented workflow with verification and human-review routing.

The project is intentionally built in stages so each component can be understood, tested, and improved independently.