# RAG Evaluation Report — Template

## 1. Dataset and setup

The prototype was evaluated using the documents stored in `data/knowledge_base/`. The initial example knowledge base contains a short Markdown document explaining RAG concepts. Replace this with the assignment's target knowledge base and record the number of source files, document types, chunks, and chunking settings here.

- Number of source documents: **[fill in after ingestion]**
- Number of chunks: **[copy from ingest.py output]**
- Chunk size / overlap: **1,000 / 150 characters**
- Embedding model: **nomic-embed-text via Ollama**
- Vector index: **FAISS**
- Generation model: **qwen3.5:9b via Ollama**
- Retrieval setting: **top-k = 4 by default**

## 2. Evaluation method

The evaluation set is stored in `evaluation/questions.jsonl`. Each test case includes a question, an expected source document, and expected answer terms. `evaluate.py` reports source hit rate at 4 and expected-term coverage. These are simple proxy metrics: expected-term coverage does not prove that an answer is semantically correct, and source hit rate does not prove that the cited passage supports every claim.

For a stronger evaluation, manually label relevant chunks for each question and calculate Recall@k and Mean Reciprocal Rank (MRR). Review each generated answer for factual correctness, whether every citation exists in the retrieved context, whether the cited passage supports the claim, and whether the model abstains when evidence is missing.

## 3. Results

Copy actual measurements from `evaluation/results.json` after running `python evaluate.py`.

| Metric | Result |
|---|---:|
| Number of evaluation questions | **[fill in]** |
| Source hit rate @4 | **[fill in]** |
| Expected-term coverage | **[fill in]** |
| Manually reviewed answers | **[fill in]** |
| Supported citations / citations checked | **[fill in]** |

Do not report placeholder values as measured results. Save a copy of `evaluation/results.json` with your submission.

## 4. Qualitative failure cases

Record at least two real examples observed during testing.

1. **Retrieval miss:** [Question, expected source, what was retrieved, and why the expected evidence may have been missed.]
2. **Unsupported or incomplete answer:** [Question, model answer, relevant source passage, and whether the model cited it correctly.]
3. **Abstention test (recommended):** [Ask something not covered by the documents and record whether the model correctly says that the documents do not provide enough information.]

## 5. Hallucinations and the effect of retrieval

The prompt instructs the LLM to answer only from retrieved context, cite chunk IDs, and admit when evidence is insufficient. Retrieval can reduce unsupported answers by supplying relevant source text at generation time. However, RAG does not eliminate hallucinations: the retriever may return irrelevant chunks, relevant evidence may be absent, and the model may misinterpret evidence or produce an invalid citation. Compare model claims against the source passages during manual review before drawing conclusions.

## 6. Reflection (6 sentences)

The project helped me understand how document ingestion, chunking, embeddings, vector search, and text generation work together in a RAG pipeline. Using FAISS made it possible to build a local semantic index without relying on a hosted vector database. Ollama allowed the embedding model and language model to run locally. One challenge was [describe a real issue encountered, such as model setup, chunk boundaries, or retrieval quality]. The evaluation showed [insert your actual observed result or failure case], which demonstrated that a plausible answer is not necessarily a well-supported answer. As a next improvement, I would add [choose a real next step, such as hybrid search, reranking, stronger citation validation, or a larger manually labelled test set].
