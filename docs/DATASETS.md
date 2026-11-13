# Benchmark Datasets and Attribution

The repository downloads datasets from the Hugging Face Hub at ingestion time. Raw datasets and generated vector indexes should not be committed to this repository.

## 1. TruthfulQA

- Hugging Face dataset: <https://huggingface.co/datasets/truthfulqa/truthful_qa>
- Configuration used: `generation`
- Split used: `validation`
- Dataset page reports 817 examples in this configuration.
- The Hub dataset page declares Apache-2.0. Re-check the current dataset card and terms before redistribution or commercial use.
- Purpose: test whether model answers repeat common misconceptions or produce untruthful answers.
- Fields used by the ingestion script: `question`, `best_answer`, `category`, and `source`.
- Paper: Lin, S., Hilton, J., and Evans, O. (2021), “TruthfulQA: Measuring How Models Mimic Human Falsehoods,” arXiv:2109.07958.

## 2. SQuAD

- Hugging Face dataset: <https://huggingface.co/datasets/rajpurkar/squad>
- Split used: `train`
- The Hub page reports approximately 87.6K training rows.
- The Hub dataset card declares CC BY-SA 4.0. Preserve attribution and comply with share-alike terms for adaptations of the dataset; do not represent the dataset as MIT-licensed project code.
- Purpose: reading-comprehension examples with questions, answer spans, and corresponding contexts.
- Fields used by the ingestion script: `id`, `title`, `context`, `question`, and `answers.text`.
- Paper: Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P. (2016), “SQuAD: 100,000+ Questions for Machine Comprehension of Text,” EMNLP.

## Data handling

- The initial command indexes all TruthfulQA generation examples and a configurable SQuAD sample (500 rows by default).
- Use `--max-squad-rows 0` to request the complete SQuAD training split after the pilot works.
- Each chunk stores dataset name, record ID, and relevant metadata for traceability.
- Dataset cache files, database files, and vector indexes are local-only and ignored by Git.
- Benchmark datasets are not a universal source of current facts. Retrieval results are candidate evidence and may be incomplete, outdated, or irrelevant; the evaluator must report evidence limitations.
