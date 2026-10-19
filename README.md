# Cynbit-Nexus AI

A Python proof of concept combining **hybrid retrieval**, **synthetic query-complexity routing**, **Monte Carlo engineering-effort estimation**, and a **Streamlit dashboard**.

> **Scope and limitations:** The included corpus and router examples are hand-authored synthetic data. Benchmark scores and simulated cost/latency values are illustrative, not independent validation or production guarantees. The grounding check is a lightweight lexical-overlap heuristic, not a complete hallucination detector. Do not use the estimates as client commitments without calibrating them against real delivery data.

## Capabilities

- **Hybrid retrieval:** custom Okapi BM25 lexical scoring plus TF-IDF/TruncatedSVD latent-semantic similarity, combined with Reciprocal Rank Fusion (RRF).
- **Query routing:** a sigmoid-calibrated logistic classifier over TF-IDF and structural query features; routes to illustrative fast/complex tiers.
- **Semantic cache:** process-local TF-IDF cosine-similarity cache for repeated or similar queries.
- **Effort estimation:** seeded Beta-PERT Monte Carlo sampling for P50, P80, and P95 effort quantiles, with a configurable compliance buffer.
- **Grounding audit:** lexical overlap checks against retrieved project context and a generated architecture summary.
- **Evaluation:** retrieval ablation, five-fold router evaluation, synthetic request economics, and a generated Markdown report.
- **Dashboard and CI:** Streamlit interface, pytest checks, and GitHub Actions on Python 3.11.

## Architecture

```text
Query / client requirement
       |
       +--> TF-IDF semantic cache ------------------> cached response (if threshold met)
       |
       +--> calibrated complexity router ----------> illustrative fast / complex tier
       |
       +--> BM25 + TF-IDF/TruncatedSVD retrieval
                    |
                    +--> Reciprocal Rank Fusion
                                |
                                +--> Beta-PERT effort simulation
                                +--> lexical grounding audit
                                            |
                                            +--> Streamlit diagnostics
```

## Project structure

```text
.
├── .github/workflows/ci.yml   # CI: compile and run tests
├── data/corpus.json           # Six synthetic project profiles and evaluation queries
├── src/
│   ├── retrieval.py            # BM25, latent semantic similarity, RRF
│   ├── router.py               # Synthetic classifier and semantic cache
│   └── simulator.py            # Monte Carlo estimates and lexical audit
├── tests/test_core.py          # Schema and core-behavior tests
├── app.py                      # Streamlit dashboard
├── evaluate.py                 # Reproducible synthetic evaluation
├── BENCHMARK_RESULTS.md        # Generated example benchmark output
├── requirements.txt
└── LICENSE
```

## Quick start

Python 3.11 is used by CI. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the tests:

```bash
pytest -q
```

Generate the benchmark report:

```bash
python evaluate.py
```

Launch the interactive dashboard:

```bash
streamlit run app.py
```

## How to interpret the output

- `Precision@1` and `MRR@3` are measured against the small, predefined query set in `data/corpus.json`.
- Router cross-validation uses generated synthetic examples with repeated templates. Perfect results on this data should not be interpreted as generalization to real customer traffic.
- Cost and latency are constants used in a simulation, not live model API measurements. No external LLM API is called.
- Monte Carlo intervals reflect the assumptions in the example subsystem-hour profiles; they are not a substitute for project-specific estimation.
- The grounding audit can return `WARN`; review retrieved context and recommendations rather than treating the status as a proof of factuality.

## Development notes

- Keep credentials and local secrets out of version control. The `.gitignore` excludes environment files and Streamlit secrets.
- Add representative, permissioned evaluation data before making performance claims.
- Run `python -m compileall -q src app.py evaluate.py` and `pytest -q` before submitting changes.

## License

MIT. See [LICENSE](LICENSE).
