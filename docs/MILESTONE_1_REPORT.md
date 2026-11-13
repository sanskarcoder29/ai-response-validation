# Milestone 1 — Foundation and Evaluation Understanding

**Project:** Development of AI Response Validation System with Hallucination Detection Assistance  
**Milestone:** 1 — Foundation & Evaluation Understanding  
**Status:** Initial design and local prototype scaffold; update this report with actual run results and screenshots before submission.

## 1. Problem statement

Large Language Models can generate fluent responses that are inaccurate, incomplete, irrelevant, or unsupported by the available evidence. This project proposes a modular evaluation system that accepts a question and AI-generated response, retrieves relevant reference material, evaluates the response across defined quality dimensions, and produces an explainable verdict with evidence.

## 2. Research summary

### LLM evaluation

LLM evaluation can combine reference-based metrics, reference-free model judging, and task-specific checks. Reference-based evaluation compares a response with a known answer or labelled dataset. Reference-free evaluation uses a rubric or judge model to assess qualities such as relevance and clarity when an exact answer is unavailable. Automated scores should be checked against human-reviewed examples because model judges can be inconsistent or biased.

### Hallucination detection and factuality

For this project, a hallucination is treated operationally as a substantive claim that is unsupported by, or contradicted by, the evidence available to the evaluator. The system should break a response into claims and classify each as supported, contradicted, or insufficiently evidenced. Lack of evidence is not necessarily proof that a claim is false.

### Faithfulness and relevance

Faithfulness concerns whether the answer is supported by the retrieved context. Relevance concerns whether the answer addresses the user's question. A response can be relevant but unfaithful (on-topic but unsupported), or faithful to a retrieved passage but incomplete for the question.

### RAG architecture

Retrieval-Augmented Generation (RAG) separates reference ingestion from query-time retrieval. Documents are cleaned and split into chunks, each chunk is embedded and stored with metadata, and a user query is embedded to retrieve semantically related chunks. The evaluator can then compare answer claims with retrieved passages. Retrieval quality must be measured because poor retrieval can cause a good answer to look unsupported or can supply misleading evidence.

### LLM-as-a-Judge

An LLM-as-a-Judge approach prompts a model to assess an output against explicit criteria and return structured scores and explanations. The proposed design uses separate judge roles for relevance, accuracy, hallucination detection, and completeness. Judge outputs should use a fixed schema, include evidence identifiers, and be calibrated against human-labelled examples. A judge model is itself fallible, so its verdict is an aid to review rather than ground truth.

### Framework review

- **RAGAS:** useful for structured RAG evaluation; its metric catalogue includes faithfulness, response relevancy, context precision, context recall, factual correctness, and other task-specific metrics.
- **TruLens:** useful for instrumenting and evaluating LLM applications; its documented concepts include the RAG triad of context relevance, groundedness, and answer relevance, along with evaluation and feedback functions.
- **Initial decision:** study both, but begin with a small, explicit project rubric and later compare selected RAGAS/TruLens metrics. Avoid introducing both frameworks into the runtime before the evaluation workflow is stable.

## 3. Proposed system architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the Mermaid diagram, data flow, component responsibilities, data models, and orchestration proposal.

The architecture is divided into:

1. User interface and backend API.
2. Submission validation and storage.
3. Reference dataset ingestion and preprocessing.
4. Embedding generation and vector-store indexing.
5. RAG retrieval and evidence packaging.
6. Evaluation orchestrator and specialised judge agents.
7. Score aggregation, verdict, and report presentation.

## 4. Agent responsibilities

- **Evaluation Orchestrator:** coordinates the pipeline, assembles the submission/evidence package, invokes judges, and tracks execution status.
- **Relevance Judge Agent:** checks whether the response answers the question without drifting off topic.
- **Accuracy Judge Agent:** checks factual claims against the reference answer or trusted evidence.
- **Hallucination Detection Agent:** identifies unsupported and contradicted claims and attaches evidence references where possible.
- **Completeness Judge Agent:** checks whether important aspects of the question/reference are addressed.
- **Verdict Agent:** aggregates structured outputs, applies agreed thresholds and evidence guardrails, and returns PASS, NEEDS_REVIEW, or FAIL.

## 5. Scoring dimensions

The proposed rubric uses four 1–5 scores: relevance (20%), accuracy (30%), hallucination safety (30%), and completeness (20%). Higher is better for every dimension. Detailed definitions, provisional thresholds, claim labels, and calibration steps are in [`SCORING_RUBRIC.md`](SCORING_RUBRIC.md). Thresholds require mentor review and empirical calibration.

## 6. Evaluation Input Module

The starter implementation provides:

- Streamlit form for the required question and AI-generated response.
- Optional reference answer and pasted source material.
- FastAPI Pydantic validation, including length limits and blank-input checks.
- SQLite persistence with a generated submission ID and UTC timestamp.
- API documentation through FastAPI's `/docs` page.

**Current boundary:** the module stores the submission; it does not yet run evaluation agents or claim to score the response.

## 7. Reference Knowledge Base

### Data sources

- TruthfulQA generation split: 817 examples on the Hub page, used for misconception/truthfulness-oriented QA examples.
- SQuAD training split: approximately 87.6K rows on the Hub page, used for question/context/answer retrieval examples.

See [`DATASETS.md`](DATASETS.md) for source links, declared dataset licenses, citations, fields, and data-handling rules.

### Ingestion plan

1. Load benchmark splits through Hugging Face `datasets`.
2. Select the configured pilot sample for SQuAD.
3. Normalise whitespace and remove empty content.
4. Build text records retaining question, answer, context, and source metadata.
5. Split long text into overlapping chunks (initial settings: 900 characters with 120-character overlap).
6. Generate local embeddings using ChromaDB's default embedding function.
7. Upsert chunk text and metadata into a persistent ChromaDB collection.
8. Run representative query searches and inspect whether top results contain relevant source context.

### Retrieval validation plan

Test a small set of questions from both datasets. For each query, inspect the top-k results and record whether the expected question/context or a semantically relevant passage appears. Track examples where retrieval fails, and adjust chunk size, overlap, or top-k only after inspecting the cause. A simple manual relevance check is acceptable for this initial milestone; formal retrieval metrics can be added later.

## 8. Technology stack

Python 3.11+, FastAPI, Pydantic, Streamlit, Hugging Face `datasets`, ChromaDB, SQLite, RAGAS, TruLens, and pytest. Rationale is documented in [`TECH_STACK.md`](TECH_STACK.md).

## 9. Local verification evidence — fill this in before submission

- API health check: [ ] passed / [ ] not yet tested
- Input form submission: [ ] passed / [ ] not yet tested
- Blank required-field validation: [ ] passed / [ ] not yet tested
- TruthfulQA ingestion: [ ] passed / [ ] not yet tested
- SQuAD pilot ingestion: [ ] passed / [ ] not yet tested
- Semantic retrieval examples inspected: [ ] passed / [ ] not yet tested
- Automated tests: [ ] passed / [ ] not yet tested

Add 2–3 screenshots: (1) the input form, (2) a successful API submission, and (3) ChromaDB retrieval results. Do not mark any item as passed until you have run it.

## 10. Risks and limitations

- Vector similarity is not factual verification; retrieved chunks may be irrelevant or incomplete.
- Benchmark examples do not cover every domain or current fact.
- LLM judges may be biased, inconsistent, or prompt-sensitive.
- Accuracy cannot be assessed confidently when the reference is missing; use an insufficient-evidence state.
- A larger SQuAD ingestion run may take time and local disk space.
- Dataset license terms are separate from the project's source-code license.

## 11. Next steps after Milestone 1

1. Review the architecture and scoring rubric with the mentor.
2. Confirm the LLM provider/model and whether judges are separate calls or one structured judge pipeline.
3. Implement the claim extraction and judge-agent interfaces.
4. Connect RAG evidence to claim-level verification.
5. Build the structured verdict and evaluation report.
6. Add a labelled test set and compare automated scores against human review.
