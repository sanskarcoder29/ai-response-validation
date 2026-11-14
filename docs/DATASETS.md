# Benchmark Datasets and Attribution

**Project:** Development of AI Response Validation System with Hallucination Detection Assistance
**Milestone:** 1 — Foundation and Evaluation Understanding
**Last updated:** 14 November 2026

## 1. Overview

The project uses two public benchmark datasets to prepare reference material for AI response validation and retrieval experiments:

1. TruthfulQA
2. Stanford Question Answering Dataset (SQuAD)

The datasets are acquired from the Hugging Face Hub using the Hugging Face Dataset Viewer API and Python Requests.

Raw benchmark records, processed records, generated embeddings, and local vector databases are maintained in the project's local data directory.

---

## 2. TruthfulQA

### 2.1 Dataset information

- **Dataset name:** TruthfulQA
- **Hugging Face dataset ID:** `domenicrosati/TruthfulQA`
- **Dataset page:** https://huggingface.co/datasets/domenicrosati/TruthfulQA
- **Configuration used:** `default`
- **Split used:** `train`
- **Available records:** 817
- **Records downloaded:** 817
- **Records processed:** 817
- **Records skipped:** 0
- **Reference chunks generated:** 892

The Hugging Face dataset card explains that the benchmark contains 817 questions across 38 categories. It is designed to measure whether language models produce truthful answers instead of repeating common misconceptions.

### 2.2 Purpose in this project

TruthfulQA is used to prepare truthfulness-oriented question and answer examples.

It is useful for studying:

- Misconception-based questions.
- Truthful and untruthful responses.
- Reference-supported answers.
- Factual accuracy and hallucination-related evaluation.

### 2.3 Dataset fields

The dataset provides the following fields:

- `Type`: Type of question.
- `Category`: Category of the question.
- `Question`: Natural-language question.
- `Best Answer`: Best reference answer.
- `Correct Answers`: Alternative correct answers.
- `Incorrect Answers`: Known incorrect answers.
- `Source`: Supporting source information.

These fields are retained or represented in the processed reference records as appropriate.

### 2.4 Dataset attribution and licensing

The Hugging Face dataset card declares the Apache License 2.0 (Apache-2.0).

Dataset source and licence information should be preserved when using or redistributing the benchmark data.

**Original research paper:**

Lin, S., Hilton, J., and Evans, O. (2021).

"TruthfulQA: Measuring How Models Mimic Human Falsehoods."

https://arxiv.org/abs/2109.07958

---

## 3. SQuAD

### 3.1 Dataset information

- **Dataset name:** Stanford Question Answering Dataset (SQuAD)
- **Hugging Face dataset ID:** `rajpurkar/squad`
- **Dataset page:** https://huggingface.co/datasets/rajpurkar/squad
- **Configuration used:** `plain_text`
- **Split used:** `train`
- **Available training records:** 87,599
- **Records downloaded for the pilot:** 500
- **Records processed:** 500
- **Records skipped:** 0
- **Reference chunks generated:** 1,343

The full training split contains 87,599 examples. The current project uses a 500-record pilot sample rather than the complete training split.

### 3.2 Purpose in this project

SQuAD is a reading-comprehension dataset containing questions, corresponding passages, and answer spans.

It is used for:

- Preparing question and context reference material.
- Testing passage retrieval.
- Preserving answer and context information.
- Exploring evidence-based question-answering validation.

### 3.3 Dataset fields

The `plain_text` configuration provides:

- `id`: Unique example identifier.
- `title`: Article or passage title.
- `context`: Reference passage.
- `question`: Question associated with the passage.
- `answers.text`: One or more reference answer texts.
- `answers.answer_start`: Character positions of the answer spans in the context.

The answer offsets are retained with the answer information, and the context text is preserved to maintain its original answer positions.

### 3.4 Dataset attribution and licensing

The Hugging Face dataset card declares the Creative Commons Attribution-ShareAlike 4.0 International licence (CC BY-SA 4.0).

Appropriate attribution must be retained. Adaptations and redistributions must comply with the applicable licence terms.

The dataset licence is separate from the source-code licence of this project. The benchmark data must not be represented as MIT-licensed project code.

**Original research paper:**

Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P. (2016).

"SQuAD: 100,000+ Questions for Machine Comprehension of Text."

Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing.

https://aclanthology.org/D16-1264/

---

## 4. Data acquisition and processing status

The current acquisition and preprocessing results are summarised below.

### 4.1 TruthfulQA

- Raw records downloaded: 817
- Cleaned records: 817
- Skipped records: 0
- Chunks generated: 892
- Embeddings generated: 810
- Embeddings pending: 82

### 4.2 SQuAD

- Available training records: 87,599
- Pilot records downloaded: 500
- Cleaned records: 500
- Skipped records: 0
- Chunks generated: 1,343
- Embeddings generated: 0
- Embeddings pending: 1,343

### 4.3 Combined processing status

- Total processed records: 1,317
- Total reference chunks: 2,235
- Total generated embeddings: 810
- Total pending embeddings: 1,425

These figures describe the current local project state, not the complete contents of the public benchmark datasets.

---

## 5. Chunking strategy

The processed reference records are converted into smaller text chunks before embedding generation.

The current chunking configuration is:

- **Chunk size:** 500 characters.
- **Chunk overlap:** 100 characters.

Overlap helps retain contextual continuity between neighbouring chunks.

Each chunk maintains identifying information such as:

- Chunk ID.
- Dataset name.
- Original record ID.
- Chunk index.
- Original reference metadata.

The combined chunk data is saved in:

`data/processed/reference_chunks.json`

The current file contains 2,235 chunks from the two benchmark datasets.

---

## 6. Embedding generation and vector indexing

### 6.1 Embedding model

The project currently uses Google's Gemini Embedding 2 model.

- **Model:** `gemini-embedding-2`
- **Expected embedding dimensions:** 3072

The embedding model converts reference passages and search queries into numerical vectors.

### 6.2 Embedding generation status

Reference embedding generation has been completed partially.

Current progress:

- 810 TruthfulQA reference embeddings generated.
- 82 TruthfulQA embeddings pending.
- 1,343 SQuAD embeddings pending.

Bulk embedding generation is temporarily paused because the Gemini free-tier daily API quota is nearing its limit.

The remaining embeddings will be generated after quota availability is restored. The project will continue using the free tier without enabling paid billing.

### 6.3 Vector storage

The reference vectors are stored in a local SQLite database:

`data/processed/reference_vectors.db`

The database maintains the embedding vectors and associated reference metadata.

NumPy is used for cosine similarity calculations and ranking retrieved passages.

The current vector index contains 810 TruthfulQA embeddings. SQuAD embeddings have not yet been indexed.

---

## 7. Data acquisition scripts

The following scripts are used for benchmark acquisition and preprocessing.

### TruthfulQA

- `backend/scripts/download_truthfulqa.py`
- `backend/scripts/preprocess_truthfulqa.py`

The downloader uses:

- Dataset ID: `domenicrosati/TruthfulQA`
- Configuration: `default`
- Split: `train`

### SQuAD

- `backend/scripts/download_squad.py`
- `backend/scripts/preprocess_squad.py`

The downloader uses:

- Dataset ID: `rajpurkar/squad`
- Configuration: `plain_text`
- Split: `train`

The current pilot acquisition contains 500 SQuAD records.

### Chunking and embedding scripts

- `backend/scripts/chunk_reference_data.py`
- `backend/scripts/generate_embeddings.py`
- `backend/scripts/build_vector_index.py`

These scripts prepare reference chunks, generate embeddings, and construct the local SQLite vector index.

---

## 8. Data storage and handling

The project uses the following local data organisation.

### Raw data

`data/raw/`

Contains the acquired benchmark records.

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

File availability depends on the local processing and indexing stage.

### Data handling principles

- Retain source dataset identifiers and attribution information.
- Preserve relevant metadata through preprocessing and chunking.
- Keep raw and processed benchmark data separate.
- Avoid committing large dataset files and generated vector databases to Git.
- Keep API credentials in local environment variables.
- Never commit `.env` files or expose API keys in screenshots.
- Do not represent benchmark datasets as universally reliable or comprehensive factual sources.

The repository's `.gitignore` is configured to exclude local data files and environment secrets.

---

## 9. Dataset limitations

- TruthfulQA focuses on misconception-oriented questions and does not represent every factual domain.
- SQuAD is a reading-comprehension dataset based on Wikipedia passages and does not cover every type of real-world question.
- The current SQuAD implementation uses only a 500-record pilot sample.
- The current reference vector database contains only 810 TruthfulQA embeddings.
- The remaining TruthfulQA and SQuAD embeddings are pending.
- A retrieved passage may be semantically similar without directly supporting the answer.
- Vector similarity alone does not establish factual correctness.
- Dataset reference answers and source passages should be treated as candidate evidence requiring evaluation.
- Dataset licensing and attribution requirements must be respected independently of the project's source-code licence.

---

## 10. Future work

The following activities remain for the reference knowledge base:

1. Generate the remaining 82 TruthfulQA embeddings.
2. Generate embeddings for all 1,343 SQuAD pilot chunks.
3. Update the SQLite vector index.
4. Verify the complete vector counts and dataset distribution.
5. Generate fresh query embeddings for natural-language questions.
6. Inspect top-k retrieved passages from both datasets.
7. Record manual relevance observations and retrieval failures.
8. Evaluate retrieval quality before connecting it to automated claim-level evaluation.

Formal retrieval metrics and additional benchmark datasets may be introduced in later milestones.

---

## 11. References

**TruthfulQA**

- Hugging Face dataset: https://huggingface.co/datasets/domenicrosati/TruthfulQA
- Original paper: https://arxiv.org/abs/2109.07958

**SQuAD**

- Hugging Face dataset: https://huggingface.co/datasets/rajpurkar/squad
- Original paper: https://aclanthology.org/D16-1264/

---

**Dataset documentation status:** The actual downloader configurations, acquisition counts, preprocessing results, chunk counts, partial embedding progress, and local storage approach are documented. Remaining embedding generation and independent semantic retrieval validation are pending.
