# Advancing the Retrieval-Pipline with Example


---

## 1️⃣ User Query
```text
How do model-training costs change over time?
```
The user submits this question through the UI (FastAPI ➜ Gradio). The retrieval pipeline now springs into action.

---

## 2️⃣ Offline Document Processing (already completed)
*Executed once when documents are ingested — **before** any query arrives.*

1. **Hierarchical Chunking**
   * Parent chunks: 500-1000 tokens (paragraph-level context)
   * Child chunks: 50-150 tokens (fine-grained semantics)
2. **Multi-Vector Embedding**
   * **ColPali** – page patches → 1030 vectors × 128 dims
   * **ColBERT** – child-chunk tokens → *N* vectors × 128 dims
3. **Storage**
   * OpenSearch: All embeddings + child→parent mappings
   * PostgreSQL: Parent/child hierarchy + rich metadata

> **Outcome**: A search-ready corpus where every child chunk is represented by multiple vectors, while parent chunks hold the surrounding context.

---

## 3️⃣ Query Processing
*Happens instantly when the user sends the query.*

1. **Token-Level Query Embedding**  
   25-50 query tokens → vectors × 128 dims
2. Preserve raw query text for downstream neural rerankers.

---

## 4️⃣ Multi-Stage Retrieval Cascade
A precision funnel that narrows 1000+ candidates down to just a handful of highly relevant chunks.

| Stage | Technique | Candidates | Why it matters |
|-------|-----------|------------|----------------|
| **4.1** | **MaxSim Late-Interaction Search** | 1000 | Compares every query token to every document token — richer than dot-product similarity. |
| **4.2** | **TILDE-v2 Sparse Rerank** | 100 | Blazing-fast term-based filtering (≈ 20 ms) that removes obviously irrelevant hits. |
| **4.3** | **MonoT5 Cross-Encoder** | 20 | Deep semantic scoring using [Query ⊕ Chunk] input. |
| **4.4** | **RankLLaMA (Listwise LLM)** | 5 | Holistic reasoning across the top set to achieve perfect ordering. |

---

## 5️⃣ Context Enrichment
1. **Parent Retrieval** – For each of the 5 child chunks, fetch its parent paragraph from PostgreSQL.
2. **Deduplication & Merge** – Remove duplicates and merge overlapping parents ➜ typically 3-5 unique parent chunks remain.

---

## 6️⃣ Document Repacking
1. **Reverse Repack** – Order parents from *least* ➜ *most* relevant.  
   LLMs attend most to the beginning **and** end of the context window.
2. **Final Payload** – The reordered parent chunks + metadata are packaged and handed off to the Generation layer.

---

## Result
The LLM receives a **compact, context-rich, and optimally ordered** prompt that empowers it to answer:
> *“Model-training costs typically drop exponentially over time due to hardware efficiency gains, algorithmic improvements, and…”*

Compared to the original single-vector + one-shot rerank approach, the new pipeline:
* Finds **hard-to-surface nuggets** via token-level interactions.
* Uses a **cost-aware cascade** — cheapest models first, expensive ones only on a shrinking candidate set.
* Delivers **full-paragraph context** rather than isolated sentences.
* Optimizes document order to **maximise LLM attention**.

---

### 📈 Key Takeaways
1. **Accuracy ↑** – Multi-vector embeddings + four-stage ranking drastically improve precision.
2. **Latency ⚖️** – Smart staging (~45 ms median) keeps the system snappy.
3. **LLM-Readiness** – Reverse repacking ensures the most salient information is front-and-centre.

### ENHANCEMENT FITS IN YOUR CURRENT ARCHITECTURE

Looking at your original architecture:

- Replace "Embedding" step → Multi-vector embeddings (ColBERT/ColPali)
- Replace "Hybrid Search" → Late Interaction MaxSim search
- Replace single "Re-ranking" → Three-stage cascade (TILDE → MonoT5 → RankLLaMA)
- Add after "Top-K chunks" → Parent retrieval + Reverse repacking

**KEY DATA TRANSFORMATIONS**

- Documents: Split into parent/child hierarchy
- Embeddings: Single vector → Multiple vectors per chunk
- Search: Dot product → MaxSim scoring
- Reranking: Single step → Three-stage cascade
- Context: Individual chunks → Full parent contexts

