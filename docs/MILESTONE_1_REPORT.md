# Milestone 1 — Foundation and Evaluation Understanding

**Project:** Development of AI Response Validation System with Hallucination Detection Assistance
**Milestone:** 1 — Foundation & Evaluation Understanding
**Last updated:** 14 November 2026
**Status:** Research, system design, input module, benchmark preprocessing, and partial reference retrieval implementation completed. Remaining embedding generation and independent semantic retrieval validation are in progress.

---

## 1. Problem statement

Large Language Models can generate fluent responses that are inaccurate, incomplete, irrelevant, or unsupported by the available evidence.

This project aims to develop a modular AI response validation system that accepts a question and an AI-generated response, retrieves relevant reference material, evaluates the response across defined quality dimensions, and eventually produces an explainable verdict supported by evidence.

The initial milestone focuses on understanding the evaluation problem, researching suitable techniques and datasets, designing the system architecture, and implementing the foundation required for subsequent evaluation modules.

---

## 2. Research summary

### LLM evaluation

LLM evaluation can combine reference-based metrics, reference-free model judging, and task-specific checks.

Reference-based evaluation compares a response with a known answer or labelled dataset. Reference-free evaluation uses a rubric or judge model to assess qualities such as relevance and clarity when an exact answer is unavailable.

Automated scores should be checked against human-reviewed examples because model judges can be inconsistent or biased.

### Hallucination detection and factuality

For this project, a hallucination is treated operationally as a substantive claim that is unsupported by, or contradicted by, the evidence available to the evaluator.

The proposed system should break a response into claims and classify each as:

- Supported
- Contradicted
- Insufficiently evidenced

Lack of evidence is not necessarily proof that a claim is false. Therefore, the system should distinguish between unsupported claims and claims directly contradicted by reliable evidence.

### Faithfulness and relevance

Faithfulness concerns whether the answer is supported by the retrieved context.

Relevance concerns whether the answer addresses the user's question.

A response can be relevant but unfaithful (on-topic but unsupported), or faithful to a retrieved passage but incomplete for the question.

Both dimensions are important for reliable AI response validation.

### RAG architecture

Retrieval-Augmented Generation (RAG) separates reference ingestion from query-time retrieval.

The proposed workflow includes:

1. Cleaning and preprocessing reference documents.
2. Splitting the reference material into manageable chunks.
3. Generating numerical embeddings for the chunks.
4. Storing embeddings and associated metadata.
5. Converting a user query into an embedding.
6. Retrieving semantically similar reference chunks.
7. Packaging retrieved passages as evidence for later evaluation.

Retrieval quality must be measured because poor retrieval can cause a good answer to look unsupported or can supply misleading evidence.

The current prototype implements reference preprocessing, chunking, embedding generation for a partial dataset, local vector storage, and cosine similarity search.

### LLM-as-a-Judge

An LLM-as-a-Judge approach prompts a model to assess an output against explicit criteria and return structured scores and explanations.

The proposed design uses separate judge roles for relevance, accuracy, hallucination detection, and completeness.

Judge outputs should use a fixed schema, include evidence identifiers, and be calibrated against human-labelled examples.

A judge model is itself fallible, so its verdict is an aid to review rather than ground truth.

**Implementation status:** Judge agents, automated scoring, and verdict generation are planned for later development. They are not implemented in the current Milestone 1 prototype.

### Framework review

- **RAGAS:** Useful for structured RAG evaluation. Its metric catalogue includes faithfulness, response relevancy, context precision, context recall, factual correctness, and other task-specific metrics.
- **TruLens:** Useful for instrumenting and evaluating LLM applications. Its documented concepts include context relevance, groundedness, and answer relevance.
- **Initial decision:** Study both frameworks, but begin with a small, explicit project rubric. Selected RAGAS or TruLens metrics can be evaluated later. Neither framework is currently integrated into the runtime.

---

## 3. Proposed system architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the system architecture diagram, data flow, component responsibilities, proposed data models, and orchestration design.

The overall proposed architecture is divided into:

1. User interface and backend API.
2. Submission validation and storage.
3. Reference dataset ingestion and preprocessing.
4. Embedding generation and vector-store indexing.
5. RAG retrieval and evidence packaging.
6. Evaluation orchestrator and specialised judge agents.
7. Score aggregation, verdict, and report presentation.

### Current Milestone 1 implementation

The implemented foundation currently includes:

- Streamlit-based user input interface.
- FastAPI backend.
- Pydantic request validation.
- SQLite submission storage.
- Benchmark dataset acquisition and preprocessing.
- Reference text chunking.
- Gemini embedding generation for a partial benchmark dataset.
- SQLite-based vector storage.
- NumPy-based cosine similarity search.
- Reference retrieval and evidence context construction.

The specialised judge agents, automated scoring, orchestration, and final verdict generation remain planned components.

---

## 4. Agent responsibilities

The following responsibilities describe the proposed evaluation architecture for future milestones.

- **Evaluation Orchestrator:** Coordinates the evaluation pipeline, assembles the submission and evidence package, invokes judges, and tracks execution status.
- **Relevance Judge Agent:** Checks whether the response answers the question without drifting off topic.
- **Accuracy Judge Agent:** Checks factual claims against the reference answer or trusted evidence.
- **Hallucination Detection Agent:** Identifies unsupported and contradicted claims and attaches evidence references where possible.
- **Completeness Judge Agent:** Checks whether important aspects of the question and reference are addressed.
- **Verdict Agent:** Aggregates structured outputs, applies agreed thresholds and evidence guardrails, and returns PASS, NEEDS_REVIEW, or FAIL.

**Current implementation status:** These agents and their automated evaluation responsibilities have not yet been implemented. Their interfaces and responsibilities are part of the proposed design.

---

## 5. Scoring dimensions

The proposed evaluation rubric uses four scores on a scale of 1–5:

- Relevance: 20%
- Accuracy: 30%
- Hallucination safety: 30%
- Completeness: 20%

Higher is better for every dimension.

Detailed definitions, provisional thresholds, claim labels, and calibration steps are documented in [`SCORING_RUBRIC.md`](SCORING_RUBRIC.md).

These weights and thresholds are provisional. They require mentor review and empirical calibration against human-reviewed examples before being used for actual automated evaluation.

**Current implementation status:** The scoring rubric is documented but automated score calculation and verdict generation are not implemented.

---

## 6. Evaluation Input Module

The current implementation provides the initial input and storage functionality.

### Implemented components

- Streamlit form for entering the required question and AI-generated response.
- Optional reference answer and pasted source/reference material.
- FastAPI backend for receiving submissions.
- Pydantic validation, including required-field checks and length limits.
- SQLite persistence with a generated submission ID and UTC timestamp.
- FastAPI API documentation through the `/docs` page.

### API endpoints

**GET `/health`**

Checks whether the backend service is running.

**POST `/submissions`**

Validates and stores an evaluation submission. The endpoint returns an accepted response after successful storage.

**POST `/knowledge-base/search`**

Accepts a search query and retrieves candidate reference passages from the local reference vector database.

Retrieved passages are candidate evidence and still require evaluation.

### Current boundary

The input module validates and stores submissions, and the retrieval endpoint provides reference passages.

It does not yet run evaluation agents, calculate hallucination scores, or produce an automated verdict.

The optional source material is currently provided as pasted text rather than a complete PDF or DOCX document-upload and parsing pipeline.

---

## 7. Reference Knowledge Base

### Data sources

The prototype uses two public benchmark datasets.

**1. TruthfulQA**

- Dataset: `domenicrosati/TruthfulQA`
- Split: `generation`
- Available records retrieved: 817
- Purpose: Provides misconception-oriented questions and reference answers for truthfulness-related evaluation.

**2. SQuAD**

- Dataset: `rajpurkar/squad`
- Configuration: `plain_text`
- Split: `train`
- Available training records: approximately 87,599
- Pilot sample acquired: 500 records
- Purpose: Provides question, context, and answer examples for reference retrieval and question-answering evaluation.

See [`DATASETS.md`](DATASETS.md) for source links, dataset attribution, declared licensing information, fields, and data-handling rules.

### Data acquisition and preprocessing

Benchmark data was acquired through the Hugging Face Dataset Viewer API using Python Requests.

The initial processing results are:

**TruthfulQA**

- Raw records downloaded: 817
- Processed records: 817
- Records skipped: 0
- Reference chunks generated: 892

**SQuAD**

- Pilot records downloaded: 500
- Processed records: 500
- Records skipped: 0
- Reference chunks generated: 1,343

The combined processed reference data contains 2,235 chunks.

### Chunking strategy

The current chunking configuration is:

- Chunk size: 500 characters
- Overlap: 100 characters

Chunk metadata preserves the dataset name, record identifier, chunk identifier, and chunk index, along with the original reference information.

### Embedding generation

The prototype uses Google's Gemini Embedding 2 model to convert reference chunks into numerical vectors.

The generated embedding dimension is 3,072.

Current embedding progress:

- TruthfulQA chunks: 892
- TruthfulQA embeddings generated: 810
- Remaining TruthfulQA embeddings: 82
- SQuAD chunks: 1,343
- SQuAD embeddings generated: 0
- Total generated embeddings: 810
- Total pending embeddings: 1,425

A single-sentence Gemini embedding test returned a successful HTTP 200 response and a vector containing 3,072 dimensions.

Bulk embedding generation is partially completed. Further requests are temporarily paused because of the current free-tier daily API quota. The remaining embeddings will be generated when the quota becomes available again, without enabling paid billing.

### Vector storage and retrieval

The current implementation uses a local SQLite database:

`data/processed/reference_vectors.db`

The database stores reference embeddings, vector norms, and associated metadata.

NumPy is used to calculate cosine similarity between a query vector and stored reference vectors.

The local retrieval implementation supports:

- Top-k result selection.
- Cosine similarity ranking.
- Optional dataset filtering.
- Preservation of reference text and metadata.
- Returning ranked reference passages with similarity scores.

The current vector database contains 810 indexed TruthfulQA embeddings. SQuAD embeddings are pending.

### Retrieval validation

Offline retrieval tests have verified:

- Loading stored vectors from SQLite.
- Cosine similarity calculation.
- Similarity ranking.
- Exact self-match for an existing stored embedding.
- Dataset filtering.
- Retrieval pipeline integration using a mocked query embedding.
- Evidence context generation and formatting.

These tests establish that the local vector-search and retrieval components operate correctly.

However, exact self-match and mocked query tests do not independently establish semantic retrieval quality for new natural-language questions.

Independent semantic retrieval validation using freshly generated query embeddings and manually inspected relevance examples remains pending.

---

## 8. Technology stack

The current implementation uses the following technologies.

- **Python 3.14.6:** Primary programming language used for the prototype.
- **FastAPI:** Backend API and endpoint implementation.
- **Pydantic:** Request validation and structured input schemas.
- **Streamlit:** User input interface.
- **SQLite:** Persistent storage for submissions and reference vectors.
- **NumPy:** Numerical vector operations and cosine similarity calculations.
- **Google Gemini Embedding 2:** Reference and query embedding generation.
- **Hugging Face Dataset Viewer API:** Benchmark dataset acquisition.
- **Requests:** HTTP communication with external APIs.
- **python-dotenv:** Loading local environment configuration.
- **pytest:** Automated schema validation tests.

RAGAS and TruLens have been reviewed as potential evaluation frameworks but are not currently integrated into the runtime.

ChromaDB was considered in an earlier ingestion implementation. The active retrieval architecture uses SQLite and NumPy instead.

The technology rationale and proposed future tools are documented in [`TECH_STACK.md`](TECH_STACK.md).

---

## 9. Local verification evidence

The following verification results have been observed during local testing.

### API and input module

- [x] FastAPI health check: HTTP 200 returned successfully.
- [x] Submission API: successful HTTP 201 submission was verified previously.
- [x] Blank required-field validation: passed.
- [x] Optional reference answer and source material validation: passed.
- [x] API response structure: verified through a mocked FastAPI endpoint test.
- [x] Python syntax compilation for backend and test files: completed without errors.

### Dataset ingestion and processing

- [x] TruthfulQA acquisition and preprocessing: 817 records processed successfully.
- [x] TruthfulQA chunk generation: 892 chunks generated.
- [x] SQuAD pilot acquisition and preprocessing: 500 records processed successfully.
- [x] SQuAD pilot chunk generation: 1,343 chunks generated.
- [x] Combined reference chunks: 2,235 chunks generated.
- [x] Sample Gemini embedding request: successful HTTP 200 response with 3,072 dimensions.
- [ ] Complete bulk embedding generation: pending.
- [ ] SQuAD embedding generation: pending.
- [ ] Remaining 82 TruthfulQA embeddings: pending.

### Retrieval and evidence processing

- [x] SQLite vector database construction: 810 embeddings indexed.
- [x] Offline cosine similarity retrieval: passed.
- [x] Exact self-match for a stored reference vector: passed.
- [x] TruthfulQA dataset filtering: passed.
- [x] Mock retrieval pipeline integration: passed.
- [x] Evidence context generation: passed.
- [x] Evidence ID generation: passed.
- [x] Metadata preservation: passed.
- [x] Similarity threshold filtering: passed.
- [x] Context length validation: passed.
- [ ] Independent semantic retrieval with freshly generated query embeddings: pending.
- [ ] Manual relevance evaluation using new questions from both datasets: pending.

### Automated tests

The formal pytest suite currently contains three test cases.

Command executed:

`python -m pytest -v`

Result:

`3 passed in 0.21s`

The three passing tests cover:

1. Acceptance of optional reference fields in a valid submission.
2. Rejection of blank required fields.
3. Validation of the bounded `top_k` value.

Additional standalone and mocked test scripts were executed successfully:

- `test_vector_search.py`
- `test_retrieval.py`
- `test_retriever_mock.py`
- `test_context_builder.py`
- Knowledge-base adapter mock test.
- FastAPI knowledge-base endpoint mock test.

The local checks reported successful results. The retriever mock script reported four passing checks, the context builder reported seven passing checks, and the offline retrieval script reported three passing checks.

These standalone script checks should not be confused with formal pytest test cases.

### Evidence screenshots to include before submission

Attach suitable screenshots showing:

1. Streamlit input form.
2. Successful submission response.
3. FastAPI health check.
4. Formal pytest result showing three passed tests.
5. Offline SQLite vector retrieval output.
6. Context builder test output.
7. Dataset and embedding progress, where useful.

Do not include API keys, access tokens, or `.env` contents in screenshots.

Replace any older ChromaDB retrieval screenshot requirements with evidence from the current SQLite-based retrieval implementation.

---

## 10. Risks and limitations

- Vector similarity is not factual verification. Retrieved chunks may be irrelevant, incomplete, or misleading.
- Benchmark examples do not cover every domain or current fact.
- The current vector index contains only 810 TruthfulQA embeddings. SQuAD embeddings and the remaining TruthfulQA embeddings are pending.
- Independent semantic retrieval quality has not yet been established using freshly generated query embeddings.
- Gemini embedding generation is subject to free-tier API quotas and availability.
- LLM judges may be biased, inconsistent, or prompt-sensitive.
- Accuracy cannot be assessed confidently when the reference is missing; an insufficient-evidence state should be supported.
- The current input interface accepts pasted reference material. Full document upload, extraction, and parsing are not yet implemented.
- Automated judge agents, scoring, orchestration, and verdict generation are planned future components.
- A larger SQuAD ingestion run may require additional processing time and local disk space.
- Dataset license terms are separate from the project's source-code license.

---

## 11. Next steps

### Remaining Milestone 1 activities

1. Complete the remaining 82 TruthfulQA embeddings when the Gemini free-tier quota becomes available.
2. Generate embeddings for the 1,343 SQuAD chunks.
3. Update the SQLite vector index with the additional embeddings.
4. Generate fresh query embeddings and test real query-time retrieval.
5. Inspect representative retrieval examples from both datasets and record their relevance.
6. Update the architecture, technology stack, and verification documentation with final results.
7. Review the architecture and scoring rubric with the mentor.

### Activities planned for subsequent milestones

1. Confirm the final LLM provider and model.
2. Design and implement claim extraction.
3. Implement the specialised judge-agent interfaces.
4. Connect retrieved RAG evidence to claim-level verification.
5. Build structured verdict generation and evaluation reports.
6. Add a labelled test set and compare automated scores against human review.

---

**Milestone 1 completion note:** The input and storage foundation, research documentation, benchmark preprocessing, and a partially populated local reference retrieval system are implemented. Final embedding completion and independent semantic retrieval validation remain pending. Automated response evaluation and hallucination scoring are reserved for subsequent development stages.
