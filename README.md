# Advanced Retrieval Augmentation Skeleton

A robust library skeleton for building state-of-the-art Retrieval-Augmented Generation (RAG) pipelines. This project provides a multi-stage search and context enrichment pipeline, engineered for precise, cost-effective large language model (LLM) responses.

---

## 🔗 Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER QUERY (FastAPI)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  RETRIEVAL PIPELINE                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Multi-Vector│  │   MaxSim     │  │   Cascade    │          │
│  │   Embedding  │─▶│   Search     │─▶│   Reranking  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                           │                     │
│                                           ▼                     │
│                             ┌─────────────────────┐             │
│                             │ Parent Retrieval &  │             │
│                             │  Reverse Repacking  │             │
│                             └─────────────────────┘             │
└─────────────────────────────  ──┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM GENERATION                                │
│                  (Answer generation)                            │
└─────────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│              [ HALLUCINATION DETECTION]                       │
│         (HallBayes validation + Risk scoring)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

- **Hierarchical Chunking**: Documents split into parent (paragraph) and child (fine-grained) chunks for both local and token-level context.
- **Multi-Vector Embeddings**: Each chunk is represented by multiple embeddings. Configurable for ColBERT, ColPali, or similar models.
- **Multi-Stage Retrieval Cascade**:
  - Fast late-interaction search (MaxSim)
  - Sparse reranker with TILDE-v2
  - Deep semantic MonoT5 cross-encoder
  - LLM-based listwise reranker (RankLLaMA)
- **Context Enrichment**: Pulls full parent paragraphs for returned hits, deduplicates and merges context for the LLM.
- **Reverse Repacking**: Presents context in most-informative order for better LLM focus.
- **Cost-aware ranking**: More expensive reranker models are used only on narrowed-down candidates.
- **Cloud/Database Friendly**: Optimized for hybrid storage (OpenSearch, PostgreSQL) and can be adapted to Azure/GCP data solutions.
- **Integrated Hallucination Detection**: HallBayes module validates generation output and delivers risk scoring for answer trustworthiness.

---

## 🚦 End-to-End Flow

1. **User Query**
   - Entered via UI (e.g., FastAPI ➜ Gradio)
2. **Document Ingestion (Offline, one-time for new data)**
   - Split docs into parent and child chunks
   - Embed using ColBERT/ColPali/etc
   - Store:
     - Vectors and mappings in OpenSearch
     - Metadata/hierarchy in PostgreSQL
3. **Query Processing (Realtime)**
   - Convert user query into token-level embeddings
4. **Retrieval Cascade**
   - **MaxSim Late-Interaction Search**: Broadest candidate sweep (token-to-token matches)
   - **TILDE-v2 Sparse Reranking**: Fast, term-based elimination of irrelevant results
   - **MonoT5 Cross-Encoder**: Heavy semantic matching on top candidates
   - **RankLLaMA**: LLM-powered holistic final ordering
5. **Context Enrichment**
   - Retrieve parent paragraphs for final child chunks
   - Deduplicate and merge overlapping context
6. **Reverse Repacking & Answer Generation**
   - Order parent paragraphs for max LLM attention
   - Compose the input context and query for generation
7. **Hallucination Detection & Risk Scoring**
   - HallBayes module checks generated answer for hallucination risk

---

## 🏆 Why This Skeleton?

- **Higher Precision**: Token-level and multi-stage reranking dramatically boost retrieval precision.
- **Efficiency**: Costly models only run on a shrinking candidate pool; blazing-fast early stages.
- **Better Context**: Returns rich, full-paragraph context instead of isolated sentences or passages.
- **LLM-Optimized**: Presents context in the most salient order for generation.
- **Trustworthy Outputs**: Built-in HallBayes hallucination detection with risk scoring.
- **Composable**: Easily drop into your FastAPI/Gradio app or swap in other vector db / reranker combos.

---

## 🛠️ Plug into Your LLM Stack

To upgrade your RAG pipeline, replace/supplement these stages:
- Embedding: Use multi-vector models
- Search: Swap for MaxSim late interaction
- Reranking: Add multi-stage (sparse + cross-encoder + LLM)
- Post-Processing: Parent context retrieval & reverse repacking
- Trust Layer: Add HallBayes hallucination and risk detection

---

## Example Query

> **Q**: How do model-training costs change over time?
>
> **A**: Model-training costs typically drop exponentially over time due to hardware efficiency gains, algorithmic improvements, and…

---

## 📦 Data Transformations at a Glance
- **Documents**: Split into parent/child hierarchy
- **Embeddings**: Single vector → Multiple per chunk
- **Search**: Dot product → MaxSim
- **Reranking**: Single-stage → Three-stage
- **Context**: Individual → Full, merged parent context

---

## Get Started
- See `src/` and the referenced pipeline sections for a code walkthrough (coming soon!)

