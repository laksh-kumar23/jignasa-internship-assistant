# JIGNASA 2026: Internship Information Assistant

Internship Information Assistant is an intelligent decision-support system designed to help students discover and evaluate relevant internship opportunities based on their academic profile and search queries. By combining hybrid retrieval with deterministic eligibility filtering, grounded explanation generation, and citation validation, the system provides transparent, trustworthy recommendations while strictly adhering to compliance and application deadlines.

## Architecture Summary

```
Student Profile + Query
-> BM25 + Dense Retrieval
-> RRF
-> Top 30-50 candidates
-> Expired/Active split
-> Active reranking
-> Deterministic eligibility
-> Grounded LLM explanation
-> Citation validation
-> UI
```

## Core Policies & Constraints

- **Strict Expiration Policy:** Expired internships must never be recommended.
- **Data Completeness:** Missing fields must be explicitly flagged.
- **Advisory Disclaimer:** Recommendations do not guarantee selection.
