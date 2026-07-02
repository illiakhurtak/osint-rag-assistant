# OSINT RAG Assistant

Production Retrieval-Augmented Generation project for OSINT and geopolitical analysis.

This repository is my educational but engineering-focused implementation of a RAG system built from scratch. The goal is to implement the core components behind a LLM assistant: document ingestion, parsing, chunking, metadata, sparse retrieval, dense retrieval, hybrid retrieval, evaluation, and grounded answer generation.

The current version turns curated OSINT-style documents into searchable chunks, compares TF-IDF retrieval against OpenAI embedding retrieval, combines them with Hybrid RRF, evaluates retrieval quality with a validation dataset, and runs a first grounded answer-generation flow over retrieved context.

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

Answers are being designed to stay grounded in retrieved documents and include citations, confidence, unknowns, contradictions, and human-review signals.

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
- hybrid retrieval with Reciprocal Rank Fusion;
- query embedding cache for repeatable evaluation;
- context builder for prompt-ready grounded sources;
- first grounded answer generation layer with citations, unknowns, confidence, and human-review fields;
- retrieval validation dataset;
- retrieval metrics: Recall@1, Recall@3, Recall@5, MRR@5;
- miss and low-rank diagnostics.

Not implemented yet:

- reranking;
- contradiction detection;
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

### Hybrid Retrieval

Hybrid retrieval combines sparse TF-IDF retrieval and dense embedding retrieval using Reciprocal Rank Fusion (RRF).

The hybrid retriever:

- retrieves a wider candidate pool from both TF-IDF and embedding retrieval;
- fuses ranked results by `chunk_id`;
- assigns an RRF score based on each candidate's rank in each retriever;
- returns the same result schema as the other retrievers.

This avoids directly adding TF-IDF scores and embedding cosine scores, which live on different scales.

All retrievers return a similar result schema:

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

| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Misses | Low-rank cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF | 0.759 | 0.918 | 0.947 | 0.837 | 9 | 32 |
| Embedding | 0.729 | 0.947 | 0.959 | 0.829 | 7 | 39 |
| Hybrid RRF tuned (`candidate_k=30`, `rrf_k=10`) | 0.782 | 0.959 | 0.971 | 0.867 | 5 | 32 |

Hybrid RRF improves the headline ranking metrics over both standalone retrievers.

The current evaluator also includes a small tuning loop over:

```text
candidate_k = 10, 20, 30, 50
rrf_k       = 10, 30, 60, 100
```

Best observed configuration in this grid:

```text
candidate_k=30, rrf_k=10
Recall@1: 0.782
Recall@3: 0.959
Recall@5: 0.971
MRR@5:    0.867
Misses:   5
Low-rank: 32
```

Interpretation:

- TF-IDF is strong for exact terms, source names, and policy/entity-specific queries;
- embeddings improve semantic recall and reduce misses compared with sparse retrieval alone;
- Hybrid RRF combines both signals and improves Recall@1, Recall@5, MRR@5, and miss count.

This mirrors a common production RAG pattern: sparse retrieval catches exact terms, dense retrieval catches semantic meaning, and hybrid retrieval often improves robustness.

## Grounded Generation

The project now includes a first end-to-end RAG answer path:

```text
user query
    |
    v
Hybrid RRF retrieval
    |
    v
context builder
    |
    v
grounded LLM answer
```

The context builder formats retrieved chunks into numbered source blocks:

```text
[1]
Title: ...
Source: ...
URL: ...
Doc ID: ...
Chunk ID: ...
Text:
...
```

The generation prompt instructs the model to:

- use only the provided context;
- avoid outside knowledge and speculation;
- cite factual claims with source numbers such as `[1]`, `[2]`;
- state unknowns when evidence is insufficient;
- return confidence;
- flag human review for active conflict, sanctions, cyber threats, critical infrastructure, defense procurement, named-entity allegations, or forecasting claims.

Current answer structure:

```text
Answer:
...

Citations:
- [1] ...

Unknowns:
- ...

Confidence:
Low / Medium / High

Human Review:
Yes / No

Human Review Reason:
...
```

## Repository Structure

```text
.
├── data/
│   ├── raw/                       # source text documents
│   └── processed/
│       ├── chunks.jsonl            # processed chunk records
│       └── chunk_embeddings.jsonl  # generated embedding records
├── eval/
│   ├── retrieval_eval.json         # retrieval validation dataset
│   └── query_embeddings.jsonl      # cached eval-query embeddings
├── src/
│   ├── ingest.py                   # parsing, cleaning, chunking, metadata
│   ├── retrieve.py                 # TF-IDF retrieval baseline
│   ├── embed.py                    # OpenAI embedding generation
│   ├── retrieve_embeddings.py      # dense retrieval over stored embeddings
│   ├── retrieve_hybrid.py          # RRF hybrid retrieval
│   ├── context_builder.py          # prompt-ready context formatting
│   ├── generate_answer.py          # grounded answer generation demo
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

Run retrieval evaluation:

```bash
PYTHONPATH=src python src/evaluate_retrieval.py
```

This compares TF-IDF, embedding retrieval, and Hybrid RRF. It also prints the top hybrid tuning configurations.

Run grounded answer generation:

```bash
PYTHONPATH=src python src/generate_answer.py
```

This retrieves context with tuned Hybrid RRF, formats source blocks, and asks the LLM to answer with citations, unknowns, confidence, and human-review status.

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
- Eval-query embeddings are cached in JSONL so repeated evaluation does not repeatedly call the embedding API.
- Hybrid retrieval uses RRF first because sparse and dense scores are not directly comparable without normalization.
- Retrieval outputs share a common schema to make evaluation reusable.
- Context formatting is separated from generation so citation numbering and metadata are deterministic.
- Evaluation is document-level rather than only chunk-level because user-facing RAG answers usually need source-level grounding.

Known limitations:

- no vector database yet;
- no approximate nearest neighbor index;
- no batching for query embeddings;
- no learned or score-normalized hybrid retrieval yet;
- no reranking layer;
- no automated answer-quality evaluation yet.

## Roadmap

Next planned steps:

1. Add deeper miss analysis for hybrid retrieval failure cases.
2. Add reranking for top-k candidates.
3. Add answer-format validation for citations, unknowns, confidence, and human-review fields.
4. Add citation extraction and answer-source mapping.
5. Add contradiction detection across retrieved sources.
6. Add human-review policy tests for sensitive topics.
7. Design an agentic workflow for retrieval, answer drafting, verification, and fallback.
8. Introduce LangGraph after the core workflow is understood from scratch.

## Target Role Alignment

This project maps directly to LLM / RAG / Agentic AI engineering responsibilities:

- Document processing: ingestion, parsing, cleaning, chunking, metadata.
- RAG quality: sparse retrieval, dense retrieval, ranking diagnostics.
- Embeddings: OpenAI embedding generation, vector similarity, semantic search.
- Hybrid retrieval: RRF fusion, candidate-pool tuning, sparse/dense comparison.
- Evaluation: validation dataset, Recall@k, MRR, miss analysis, low-rank diagnostics.
- Grounding foundation: chunk metadata, document IDs, source URLs, result contracts, citation-ready context.
- Generation layer: grounded answer prompt with citations, unknowns, confidence, and human-review status.
- Agentic roadmap: future retrieval-augmented workflow with verification and human-review routing.

The project is intentionally built in stages so each component can be understood, tested, and improved independently.
