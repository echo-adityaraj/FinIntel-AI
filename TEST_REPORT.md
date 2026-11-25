# FinIntel AI 8.0.0 – Verification report

## Verification scope

The release was checked in two environments: an offline verification environment with the three original PDFs, and the project author's macOS environment with Python 3.11, Sentence Transformers, FAISS, Ollama and Docker Desktop. The local benchmark and deployment evidence were supplied as terminal and browser screenshots and saved metrics in the project archive. A fresh installation from the public GitHub repository has not yet been verified.

| Check | Observed result |
|---|---|
| Automated regression tests (offline verification environment) | 65 passed; archived run: 16.95 s |
| Original PDF ingestion | 3 documents, 339 pages, 1,720 passages |
| Offline PDF readiness and live API checks | Passed for four supported examples and one unsupported question |
| Full-mode macOS health endpoint | Healthy; release 8.0.0, PDF corpus, semantic retrieval, Ollama generation |
| Full-mode macOS evaluation | Completed on all 10 questions |
| Browser dashboard | Displayed saved full-mode metrics |
| Docker image build | Successful (`finintel-full`) |
| Docker-to-host Ollama model endpoint | Accessible |
| Docker container | Running (`finintel-ai`), bound to localhost:8000 |
| Docker HTTP requests | Home, health, metrics and question endpoint returned HTTP 200 |

The full Docker benchmark was not rerun inside the container. HTTP success confirms request processing, not semantic correctness. The 65-test count refers to the archived offline verification run, not a newly observed macOS test run. Model-selection unit tests use simulated responses; the actual local evaluation additionally exercised the live configuration.

## PDF benchmark

Seven answerable and three unsupported questions. Metrics below are taken from the saved JSON results, not inferred from HTTP status.

| Metric | TF-IDF / extractive | Hybrid / Ollama |
|---|---:|---:|
| Page Hit@1 | 3/7 | 4/7 |
| Page Hit@3 | 3/7 | 5/7 |
| Page Hit@5 | 5/7 | 7/7 |
| Page Hit@10 | 6/7 | 7/7 |
| Expected fragment match | 6/7 | 7/7 |
| Labelled citation page match | 5/7 | 6/7 |
| Answerable response rate | 6/7 | 7/7 |
| Unsupported refusal rate | 3/3 | 3/3 |
| Mean latency (seconds) | 0.0789848 | 3.0175004 |

The full-mode per-question results show that P06 (frontline staff skills) cited FG21/1 pages 17, 31 and 36 rather than the labelled page 4. This is a page-label mismatch, not automatically an incorrect answer. The detailed CSV records manual groundedness as unreviewed. The offline configuration falsely refused one answerable question and returned a citation outside the original page labels on another. The corpus and benchmark include development examples, so results do not establish independent generalisation or production readiness. The starter summary benchmark is separate.

## Reproduction

After dependency installation and `python -m scripts.download_documents`, run `python -m pytest -q`. Use `python -m scripts.run --mode pdf --check-only` for offline readiness or `python -m scripts.run --mode full --check-only` with Ollama running for full readiness. See the README for evaluation and Docker instructions. Record fresh results before claiming new verification or performance numbers.
