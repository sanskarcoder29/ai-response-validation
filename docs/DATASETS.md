# Benchmark Datasets and Attribution

**Project:** Development of AI Response Validation System with Hallucination Detection Assistance  
**Milestone:** 1 — Foundation and Evaluation Understanding  
**Last updated:** 16 November 2026

---

## 1. Overview

The project uses two public benchmark datasets to prepare reference material for AI response validation and retrieval experiments:

1. TruthfulQA.
2. Stanford Question Answering Dataset (SQuAD).

The datasets are acquired from the Hugging Face Hub using the Hugging Face Dataset Viewer API and Python Requests.

Raw benchmark records, processed records, generated embeddings, and local vector databases are maintained in the project's local data directory. Large data files and API credentials are excluded from version control.

## 2. TruthfulQA

### 2.1 Dataset information

- **Dataset name:** TruthfulQA.
- **Hugging Face dataset ID:** `domenicrosati/TruthfulQA`.
- **Dataset page:** https://huggingface.co/datasets/domenicrosati/TruthfulQA
- **Configuration used in the project:** `default`.
- **Split used in the project:** `train`.
- **Records downloaded:** 817.
- **Records processed:** 817.
- **Records skipped:** 0.
- **Reference chunks generated:** 892.
- **Reference embeddings generated:** 892.

The dataset card describes TruthfulQA as a benchmark designed to measure whether language models produce truthful answers rather than repeating common misconceptions.

### 2.2 Purpose in this project

TruthfulQA is used to prepare truthfulness-oriented question and answer examples.

It is useful for studying:

- Misconception-based questions.
- Truthful and untruthful responses.
- Reference-supported answers.
- Factual accuracy and hallucination-related evaluation.

### 2.3 Dataset fields

The source dataset includes fields such as:

- `Type`: type of question.
- `Category`: category of the question.
- `Question`: natural-language question.
- `Best Answer`: best reference answer.
- `Correct Answers`: alternative correct answers.
- `Incorrect Answers`: known incorrect answers.
- `Source`: supporting source information.

The processing pipeline retains or represents relevant fields in the processed reference records. Incorrect answers are not indexed as positive reference evidence.

### 2.4 Dataset attribution and licensing

The Hugging Face dataset card declares the Apache License 2.0 (`Apache-2.0`). The dataset's source and licence information should be preserved when using or redistributing the benchmark data. This project documentation records the licence declaration shown by the dataset card; users should consult the original dataset card and licence text for applicable terms.

**Original research paper:**

Lin, S., Hilton, J., and Evans, O. (2021).  
“TruthfulQA: Measuring How Models Mimic Human Falsehoods.”  
https://arxiv.org/abs/2109.07958

---

## 3. SQuAD

### 3.1 Dataset information

- **Dataset name:** Stanford Question Answering Dataset (SQuAD).
- **Hugging Face dataset ID:** `rajpurkar/squad`.
- **Dataset page:** https://huggingface.co/datasets/rajpurkar/squad
- **Configuration used:** `plain_text`.
- **Split used:** `train`.
- **Available training records:** approximately 87,599.
- **Pilot records downloaded:** 500.
- **Records processed:** 500.
- **Records skipped:** 0.
- **Reference chunks generated:** 1,343.
- **Reference embeddings generated:** 1,343.

The full training split contains approximately 87,599 examples. The current project uses a 500-record pilot sample rather than the complete training split.

### 3.2 Purpose in this project

SQuAD is a reading-comprehension dataset containing questions, corresponding passages, and answer spans.

It is used for:

- Preparing question and context reference material.
- Testing passage retrieval.
- Preserving answer and context information.
- Exploring evidence-based question-answering validation.

### 3.3 Dataset fields

The `plain_text` configuration provides fields including:

- `id`: unique example identifier.
- `title`: article or passage title.
- `context`: reference passage.
- `question`: question associated with the passage.
- `answers.text`: one or more reference answer texts.
- `answers.answer_start`: character positions of answer spans in the context.

The preprocessing pipeline retains answer information and original character offsets. Chunk metadata records answer spans that fall fully within each chunk.

### 3.4 Dataset attribution and licensing

The Hugging Face dataset card declares the Creative Commons Attribution-ShareAlike 4.0 International licence (`CC BY-SA 4.0`). Appropriate attribution must be retained, and adaptations or redistributions must comply with the applicable licence terms. This project documentation records the licence declaration shown by the dataset card; users should consult the original dataset card and licence text for applicable terms.

The dataset licence is separate from the source-code licence of this project. Benchmark data must not be represented as MIT-licensed project code.

**Original research paper:**

Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P. (2016).  
“SQuAD: 100,000+ Questions for Machine Comprehension of Text.”  
Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing.  
https://aclanthology.org/D16-1264/

---

## 4. Data acquisition and processing status

The counts below describe the local project pilot, not the complete contents of the public benchmark datasets.

### 4.1 TruthfulQA

- Raw records downloaded: 817.
- Cleaned records: 817.
- Skipped records: 0.
- Chunks generated: 892.
- Embeddings generated: 892.
- Embeddings pending: 0.

### 4.2 SQuAD

- Available training records: approximately 87,599.
- Pilot records downloaded: 500.
- Cleaned records: 500.
- Skipped records: 0.
- Chunks generated: 1,343.
- Embeddings generated: 1,343.
- Embeddings pending: 0.

### 4.3 Combined processing status

- Total processed source records: 1,317.
- Total reference chunks: 2,235.
- Total generated embeddings: 2,235.
- Total pending embeddings: 0.
- Embedding model: `gemini-embedding-2`.
- Embedding dimensions: 3,072.

---

## 5. Chunking strategy

The processed reference records are converted into smaller text chunks before embedding generation.

Current configuration:

- **Chunk size:** 500 characters.
- **Chunk overlap:** 100 characters.

Overlap helps retain contextual continuity between neighbouring chunks.

Each chunk maintains identifying information such as:

- Chunk ID.
- Dataset name.
- Original record ID.
- Chunk index.
- Original reference metadata.

For SQuAD, metadata also preserves the article title, associated question, reference answers, and answer spans relative to the chunk where the answer is fully contained.

The combined chunk data is saved in:

`data/processed/reference_chunks.json`

The current file contains 2,235 chunks from the two benchmark datasets.

## 6. SQuAD duplicate-content audit

### 6.1 Audit results

The 500-record SQuAD pilot produces 1,343 chunks. An audit using whitespace-normalized, case-insensitive text identified:

- Total SQuAD chunks: 1,343.
- Unique normalized text chunks: 196.
- Duplicate groups: 194.
- Extra duplicate chunks: 1,147.

### 6.2 Reason for repeated chunks

Multiple SQuAD question records can share the same source context. The current chunking script processes each question record independently and assigns record-specific chunk IDs. Consequently, identical or repeated context chunks can appear multiple times with different IDs.

This is a duplication in the project's current chunking output. It does not mean that the original SQuAD benchmark has 1,147 invalid records.

### 6.3 Current mitigation

The retrieval module applies result-level duplicate filtering. It normalizes passage text by collapsing whitespace and ignoring case, and uses the dataset and normalized title as part of the duplicate key.

The highest-scoring occurrence is retained in the result list. Associated source record IDs, questions, and reference answers are collected in the result metadata.

A local verification returned five results containing five distinct normalized text passages.

### 6.4 Remaining limitation and future improvement

Result-level filtering does not delete duplicate rows from:

- `reference_chunks.json`.
- `reference_embeddings.jsonl`.
- `reference_vectors.db`.

A future source-level improvement is to chunk each unique SQuAD context once while preserving all associated questions, reference answers, and answer positions in metadata. This would reduce redundant embeddings and improve the diversity of the indexed reference collection.

---

## 7. Embedding generation and vector indexing

### 7.1 Embedding model

The project uses Google's Gemini Embedding 2 model.

- **Model:** `gemini-embedding-2`.
- **Expected embedding dimensions:** 3,072.

The embedding model converts reference passages and search queries into numerical vectors.

The reference-document formatter uses a title and passage-text structure:

`title: {title} | text: {text}`

The query embedder prepares question-answering queries using:

`task: question answering | query: {question}`

### 7.2 Embedding generation status

All 2,235 reference chunks have successfully received embeddings:

- TruthfulQA embeddings: 892.
- SQuAD embeddings: 1,343.
- Total embeddings: 2,235.
- Remaining embeddings: 0.

Generated embeddings are stored in:

`data/processed/reference_embeddings.jsonl`

The generation script saves each successful embedding immediately and skips previously completed valid chunk IDs. Its maximum new batch size is configured to 250 for future runs.

### 7.3 Vector storage

Reference vectors are stored in a local SQLite database:

`data/processed/reference_vectors.db`

The database contains:

- Total records: 2,235.
- TruthfulQA records: 892.
- SQuAD records: 1,343.
- Vector dimensions: 3,072.

The database stores embedding vectors, vector norms, chunk text, and associated reference metadata.

NumPy is used for cosine similarity calculations and ranking retrieved passages.

The database currently retains repeated SQuAD text under separate chunk IDs. The retrieval layer suppresses duplicate text in returned results; the stored source records remain unchanged.

---

## 8. Data acquisition and processing scripts

### TruthfulQA

- `backend/scripts/download_truthfulqa.py`
- `backend/scripts/preprocess_truthfulqa.py`

Project configuration:

- Dataset ID: `domenicrosati/TruthfulQA`.
- Configuration: `default`.
- Split: `train`.

### SQuAD

- `backend/scripts/download_squad.py`
- `backend/scripts/preprocess_squad.py`

Project configuration:

- Dataset ID: `rajpurkar/squad`.
- Configuration: `plain_text`.
- Split: `train`.
- Pilot sample: 500 records.

### Chunking, embedding, and indexing

- `backend/scripts/chunk_reference_data.py`
- `backend/scripts/generate_embeddings.py`
- `backend/scripts/build_vector_index.py`

These scripts prepare reference chunks, generate embeddings, and construct the local SQLite vector index.

### Retrieval modules

- `backend/retrieval/query_embedder.py`
- `backend/retrieval/retriever.py`
- `backend/retrieval/vector_search.py`
- `backend/retrieval/context_builder.py`

These modules generate query embeddings, search stored vectors, filter repeated results, and prepare evidence context.

---

## 9. Data storage and handling

The project uses the following local data organisation.

### Raw data

`data/raw/`

Contains acquired benchmark records.

### Processed data

`data/processed/`

Contains cleaned benchmark records, reference chunks, embeddings, and local vector indexes.

Important files include:

- `truthfulqa_raw.json`
- `truthfulqa_clean.json`
- `squad_raw.json`
- `squad_clean.json`
- `reference_chunks.json`
- `reference_embeddings.jsonl`
- `reference_vectors.db`

### Data handling principles

- Retain source dataset identifiers and attribution information.
- Preserve relevant metadata through preprocessing and chunking.
- Keep raw and processed benchmark data separate.
- Avoid committing large dataset files and generated vector databases to Git.
- Keep API credentials in local environment variables.
- Never commit `.env` files or expose API keys in screenshots.
- Do not represent benchmark datasets as universally reliable or comprehensive factual sources.
- Treat retrieved benchmark passages as candidate evidence that still requires evaluation.

The repository `.gitignore` is configured to exclude local data files and environment secrets.

---

## 10. Retrieval validation

### 10.1 Offline retrieval checks

The local tests verified:

- Loading stored vectors from SQLite.
- Cosine similarity calculation.
- Similarity ranking.
- Exact self-match for an existing stored embedding.
- Dataset filtering.
- Retrieval pipeline integration using a mocked query embedding.
- Evidence context generation and formatting.
- Evidence ID generation.
- Metadata preservation.
- Similarity threshold filtering.
- Context length validation.

### 10.2 Live semantic retrieval examples

Two initial natural-language retrieval examples were manually inspected after generating fresh query embeddings.

**TruthfulQA example**

- Query: “What happens to watermelon seeds after a person eats them?”
- Matching reference retrieved at Rank 1.
- Similarity score: `0.828731`.
- Dataset: TruthfulQA.

**SQuAD example**

- Query: “What is located immediately behind the Basilica of the Sacred Heart?”
- Relevant Grotto passage retrieved at Rank 2.
- Similarity score: `0.690037`.
- Dataset: SQuAD.

These examples show that the query embedding and local retrieval pipeline can retrieve relevant references for the tested questions. They are initial functional examples, not a statistically representative retrieval-quality evaluation. Similarity scores are ranking signals, not calibrated probabilities.

### 10.3 Retrieval quality limitations

- Only two live examples have been manually inspected.
- The SQuAD pilot contains repeated text, although result-level filtering is implemented.
- Similarity ranking alone does not establish factual support.
- A larger labelled test set and formal retrieval metrics should be used in future evaluation.

---

## 11. Dataset limitations

- TruthfulQA focuses on misconception-oriented questions and does not represent every factual domain.
- SQuAD is a reading-comprehension dataset based on Wikipedia passages and does not cover every type of real-world question.
- The current SQuAD implementation uses only a 500-record pilot sample.
- The SQuAD pilot contains repeated chunk text due to question records sharing contexts.
- Result-level duplicate filtering does not remove the underlying repeated records from the source chunks, embeddings, or vector database.
- A retrieved passage may be semantically similar without directly supporting the answer.
- Vector similarity alone does not establish factual correctness.
- Dataset reference answers and source passages should be treated as candidate evidence requiring evaluation.
- Dataset licensing and attribution requirements must be respected independently of the project's source-code licence.

---

## 12. Future work

The following activities remain for the reference knowledge base and its evaluation:

1. Perform broader semantic retrieval checks with questions from both datasets.
2. Build a labelled retrieval evaluation set and record relevance judgments.
3. Consider formal metrics such as precision at k, recall at k, and mean reciprocal rank where suitable labels are available.
4. Implement source-level SQuAD context deduplication while preserving question, answer, and answer-offset metadata.
5. Reuse existing embeddings where the exact embedding input remains unchanged; regenerate vectors only for changed or new inputs.
6. Add formal tests for duplicate filtering and aggregation of matched metadata.
7. Review the knowledge base design with the mentor before integrating it with automated judge agents.

Additional benchmark datasets may be introduced in later milestones after source quality, attribution, licensing, and retrieval requirements are reviewed.

---

## 13. References

**TruthfulQA**

- Hugging Face dataset: https://huggingface.co/datasets/domenicrosati/TruthfulQA
- Original paper: https://arxiv.org/abs/2109.07958

**SQuAD**

- Hugging Face dataset: https://huggingface.co/datasets/rajpurkar/squad
- Original paper: https://aclanthology.org/D16-1264/

---

**Dataset documentation status:** The project has acquired and preprocessed the TruthfulQA dataset and a 500-record SQuAD pilot, generated 2,235 chunks and 2,235 Gemini Embedding 2 vectors, and indexed the vectors in SQLite. Offline retrieval checks and two initial live semantic retrieval examples have been completed. Result-level duplicate filtering is implemented. Source-level SQuAD deduplication and broader retrieval-quality evaluation remain future improvements.
