# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation combines an information retrieval step with a language model. Instead of relying only on what the model learned during training, a RAG system searches a supplied knowledge base and provides relevant passages as context for answering a question.

A typical RAG pipeline has four stages: ingestion, indexing, retrieval, and generation.

During ingestion, files are loaded and their text is cleaned. The text is split into smaller chunks because an entire large document may be too long to embed or place into an LLM prompt. Chunk size and overlap are design choices. Overlap preserves some continuity between neighboring chunks, but excessive overlap increases index size and repeated evidence.

During indexing, an embedding model converts each chunk into a numeric vector. A vector store such as FAISS stores these vectors and supports similarity search. The query is embedded using the same embedding model so the system can find chunks close to the query in vector space.

During retrieval, the system selects the top-k chunks most similar to the user's question. A value of k that is too small may omit useful evidence; a value that is too large may add irrelevant context and consume the language model's context window.

During generation, the retrieved passages are inserted into a prompt. The model should answer using the supplied evidence, cite the passages supporting its claims, and say when the documents do not contain enough information. Retrieval can reduce hallucinations by grounding answers in source text, but it cannot eliminate hallucinations entirely.

Useful evaluation dimensions include retrieval Recall@k, Mean Reciprocal Rank (MRR), answer correctness, citation precision, and abstention quality. A small test set should contain questions, expected answers, and expected source documents or passages. Quantitative metrics should be complemented by manual inspection of incorrect or unsupported answers.

RAG limitations include poor source documents, retrieval misses, ambiguous questions, incorrect chunk boundaries, embedding mismatch, and a generator that ignores or misinterprets context. Potential improvements include better preprocessing, metadata filters, hybrid lexical/vector retrieval, reranking, and a larger human-reviewed evaluation set.
