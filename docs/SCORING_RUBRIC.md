# Proposed Evaluation Scoring Rubric (For Mentor Review)

This is a starting proposal for architecture/design. Scores should be calibrated against labelled examples and reviewed with the mentor before being treated as validated performance measurements.

## Score scale

Use a 1–5 scale for each dimension. A higher score always indicates better response quality.

- **5 — Excellent:** Strongly satisfies the criterion; no material issue found.
- **4 — Good:** Mostly satisfies the criterion; only minor omissions or concerns.
- **3 — Moderate:** Partially satisfies the criterion; noticeable limitations exist.
- **2 — Poor:** Major issues, inaccuracies, unsupported statements, or omissions.
- **1 — Very poor:** Fails the criterion or contains serious contradictions/fabrications.
- **Not assessable:** Evidence or required context is insufficient. Do not convert this automatically to a low score.

## Dimensions and proposed weights

- **Relevance — 20%:** whether the response directly addresses the user's question and stays on topic.
- **Accuracy — 30%:** whether answer facts agree with a supplied reference answer or reliable reference information.
- **Hallucination safety — 30%:** whether substantive claims are supported by retrieved/source evidence and are not contradicted by it.
- **Completeness — 20%:** whether the response covers the important parts required to answer the question.

`Overall score = (Relevance × 0.20) + (Accuracy × 0.30) + (Hallucination safety × 0.30) + (Completeness × 0.20)`

All four component scores are 1–5. If a dimension is not assessable, the result must carry an `insufficient_evidence` flag and should not silently calculate a fully confident overall score. A later implementation may renormalise available weights, but that policy needs mentor approval.

## Distinguishing accuracy from hallucination safety

- **Accuracy** compares the response with a reference answer or well-supported factual target: is the information correct?
- **Hallucination safety** checks claim-level support: can each important claim be traced to the supplied or retrieved evidence, and is any claim contradicted by that evidence?

An answer may be accurate but poorly grounded when its sources are missing. It may also contain one correct statement and one fabricated extra claim; claim-level analysis should expose that issue.

## Proposed verdict thresholds

These are provisional engineering thresholds, not validated model-quality claims:

- **PASS:** Overall score ≥ 4.0, hallucination-safety score ≥ 4, and no critical contradicted claim.
- **NEEDS_REVIEW:** Overall score from 3.0 to below 4.0, an unassessable dimension, or weak/ambiguous evidence.
- **FAIL:** Overall score below 3.0, or a critical claim is contradicted by strong reference evidence.

A human reviewer should be able to inspect the evidence and override the automated suggestion. The system is an assistance tool, not a guarantee of truth.

## Claim-level hallucination labels (planned)

- `supported`: evidence directly supports the claim.
- `contradicted`: evidence conflicts with the claim.
- `insufficient_evidence`: the available context does not establish whether the claim is true.

## Calibration plan

1. Select representative benchmark examples from TruthfulQA and SQuAD.
2. Create a small manually reviewed set with expected labels and explanations.
3. Run the proposed rubric and compare model/judge scores with human labels.
4. Inspect disagreements, adjust prompts/criteria, and record rubric version.
5. Report limitations and avoid claiming accuracy without a measured evaluation set.
