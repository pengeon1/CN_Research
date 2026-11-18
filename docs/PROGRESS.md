# Project Progress & Phase Gate Tracking

**Project:** Research Paper Analysis and Reproduction of Phyu et al. (IEEE TNSM 2023)  
**Team Structure:** 4 Undergraduates (M1: Paper Analyst, M2: Reproduction Lead, M3: Scalability Lead, M4: Extension Lead)

---

## Phase Status Summary

| Phase | Description | Status | Gate Approved | Key Outputs / Artifacts |
|---|---|---|:---:|---|
| **Phase 0** | Setup & Foundation | **COMPLETED** | Verified (Tests 11/11) | Repo structure, .venv, configs, FL harness, CI test suite |
| **Phase 1** | Paper Formalization & Notes | **COMPLETED** | Verified | Math formalization, algorithms, assumptions, claims matrix, critique |
| **Phase 2** | Data Pipeline & Baselines | **COMPLETED** | Verified (3 seeds) | Synthetic cellular generator, DLSTM, Centralized LSTM (`results/baselines.json`) |
| **Phase 3** | FL Reproduction (FALSTM / FPLSTM) | **COMPLETED** | Verified (3 seeds) | FL loop, dual FedProx (in-loop vs Alg 2), $\mu$ and $\Omega$ sweeps (`results/fl_experiments.json`) |
| **Phase 4** | Clustering & Scalability | **COMPLETED** | Verified (3 seeds) | IC-FPLSTM (PCA+elbow+kmeans), RC-FPLSTM (buggy vs fixed) (`results/clustering_experiments.json`) |
| **Phase 5** | Critical Audit & Critique | **COMPLETED** | Verified | `docs/notes/critique.md` (10 verified issues + mathematical proofs) |
| **Phase 6** | Methodological Improvements | **COMPLETED** | Verified (3 seeds) | Weighted aggregation, Model-sim clustering, SLA loss, DP-FedProx (`results/improvements_experiments.json`) |
| **Phase 7** | Documentation, Report & Presentation | **COMPLETED** | Verified | IEEE LaTeX Paper (`main.tex`), PDF (`main.pdf`), PPTX deck (`presentation.pptx`), Video script |
| **Phase 8** | Reproducibility & Final Packaging | **COMPLETED** | Verified | `make all` end-to-end verification, `submission.zip` (1.7 MB) |

---

## Phase 0 Activity Log
- Initialized repository layout: `docs/`, `src/`, `configs/`, `results/`, `figures/`, `tests/`.
- Created `.venv` virtual environment with Python 3.10 and locked dependencies (`torch`, `numpy==1.26.4`, `scipy==1.15.3`, `scikit-learn`, `python-pptx`, `pytest`).
- Established `docs/ORIGINAL_vs_OURS.md` with strict tagging (`[PAPER]`, `[OUR ANALYSIS]`, `[OUR EXTENSION]`).
- Authored default experiment parameters in `configs/default.yaml`.
- Implemented core FL simulation modules: `src/fl/client.py`, `src/fl/server.py`, `src/fl/simulator.py`.
- Developed automated test suite `tests/test_setup.py`, `tests/test_data_pipeline.py`, `tests/test_clustering.py`.
- Passed 11/11 unit tests with 100% pass rate in 1.90s.

---

## Phase 1 Activity Log
- Authored `docs/notes/problem_and_system_model.md`: comprehensive analysis of 5G RAN slicing, InP vs. MVNO dynamics, anticipatory vs. static reservations, and data domain taxonomy.
- Authored `docs/notes/math_formalization.md`: reconciled symbol table (Table II), derived LSTM parameter complexity ($4(Hh + h^2 + h) + (hW + W) = 17,729$), formalized communication cost ($2 C N \Omega \psi$), and defined SLA violation rates with margin $\gamma$.
- Authored `docs/notes/algorithms.md`: detailed pseudocode and audits for Algorithms 1–4 (Global Agent, Local Agent, Clustering Agent, and Random Balanced Partitioning).
- Authored `docs/notes/assumptions.md`: catalog of 12 explicit and implicit paper assumptions spanning architecture, FL, ML, and data.
- Authored `docs/notes/claims_vs_evidence.md`: systematic audit linking Claims C-01 through C-08 to figures and identifying empirical/theoretical limitations.
- Authored `docs/notes/critique.md`: 10-point critical audit with exact quotes (<15 words), mathematical counter-proofs, and empirical verdicts.
- Authored `docs/video_outline.md`: complete multi-member script outlining speaking parts for M1, M2, M3, and M4 (~18–20 minutes total).

---

## Phase 2 & Phase 3 Activity Log
- Trained 228 local DLSTM models across 3 random seeds (`42, 1337, 2024`); achieved mean test RMSE $0.2039 \pm 0.0075$.
- Trained Centralized LSTM baseline pooling telemetry across all 57 stations per slice; achieved mean test RMSE $0.0712 \pm 0.0039$.
- Persisted baseline results and SLA violation curves to `results/baselines.json`.
- Implemented closed-form proximal gradient in `src/fl/client.py`, eliminating autograd computational graph overhead.
- Executed FALSTM (FedAvg, $\mu = 0$) across 3 seeds; achieved test RMSE $0.1748 \pm 0.0022$.
- Executed FPLSTM ($\mu = 1.0$) across 3 seeds; achieved test RMSE $0.1757 \pm 0.0014$.
- Replicated Figure 12 proximal parameter sweep $\mu \in \{0.001, 0.01, 0.1, 0.5, 1.0\}$, confirming $\mu = 1.0$ as optimal ($0.1743$ RMSE).
- Replicated Figure 10 client participation sweep $\Omega \in \{0.1, 0.3, 0.5, 0.7, 1.0\}$, demonstrating the $50\%$ communication savings sweet spot at $\Omega = 0.5$.
- Empirically audited Algorithm 2 (`paper_alg2`), confirming that true in-loop proximal descent achieves superior accuracy ($0.1757$ vs $0.1850$).
- Persisted federated learning outputs and communication payloads to `results/fl_experiments.json`.

---

## Phase 4, Phase 5 & Phase 6 Activity Log
- Executed Clustered Federated Learning across seeds `[42, 1337, 2024]`:
  - IC-FPLSTM (Trend): Test RMSE $0.2149 \pm 0.0052$
  - IC-FPLSTM (Lat/Long Geo): Test RMSE $0.2116 \pm 0.0120$
  - RC-FPLSTM (Fixed Balanced): Test RMSE $0.2176 \pm 0.0033$
  - RC-FPLSTM (Buggy Alg 4): Test RMSE $0.2116 \pm 0.0013$
- Discovered that IC-FPLSTM improves over balanced Random Clustering by merely $1.24\%$, corroborating the paper's own hidden Table V findings.
- Formalized 10-point critical audit matrix in `docs/notes/critique.md` and IEEE report.
- Evaluated Improvement A (Sample-weighted aggregation), achieving a $2.6\%$ to $11.7\%$ error reduction under data size heterogeneity.
- Evaluated Improvement B (Model-weight similarity clustering), eliminating $100\%$ of raw traffic telemetry leakage while maintaining competitive RMSE ($0.2777$).
- Evaluated Improvement C (SLA-aware pinball loss $\tau=0.8$), cutting SLA violation rates by $31\%$ (from $21.8\%$ to $15.0\%$).
- Evaluated Improvement D (DP-FedProx), demonstrating the privacy-accuracy trade-off across $\sigma \in [0.0, 0.1]$ and identifying $\sigma=0.005$ as the optimal privacy bound.

---

## Phase 7 & Phase 8 Activity Log
- Generated dual publication-quality figures (`.pdf` and `.png`) and exported LaTeX tables via `src/experiments/generate_tables_figures.py`.
- Authored full IEEE two-column paper in `docs/report/main.tex` and compiled publication-grade PDF `docs/report/main.pdf` (70 KB via WeasyPrint).
- Generated 21-slide 16:9 widescreen presentation deck `docs/slides/presentation.pptx` (726 KB with embedded figures via `python-pptx`).
- Authored root `README.md` and automated replication `Makefile` (`make all`, `make report`, `make slides`, `make package`).
- Verified 11/11 tests passing.
- Assembled self-contained `submission.zip` (1.7 MB).
