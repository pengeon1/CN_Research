# Project Progress & Phase Gate Tracking

**Project:** Research Paper Analysis and Reproduction of Phyu et al. (IEEE TNSM 2023)  
**Team Structure:** 4 Undergraduates (M1: Paper Analyst, M2: Reproduction Lead, M3: Scalability Lead, M4: Extension Lead)

---

## Phase Status Summary

| Phase | Description | Status | Gate Approved | Key Outputs / Artifacts |
|---|---|---|:---:|---|
| **Phase 0** | Setup & Foundation | **COMPLETED** | Verified (Tests 11/11) | Repo structure, .venv, configs, FL harness, CI test suite |
| **Phase 1** | Paper Formalization & Notes | **COMPLETED** | Ready for Gate Review | Math formalization, algorithms, assumptions, claims matrix, critique |
| **Phase 2** | Data Pipeline & Baselines | **READY TO RUN** | Pending Gate Approval | Synthetic cellular generator, DLSTM, Centralized LSTM |
| **Phase 3** | FL Reproduction (FALSTM / FPLSTM) | **IMPLEMENTED** | Pending Phase 2 Gate | FL loop, dual FedProx (in-loop vs Alg 2), $\mu$ and $\Omega$ sweeps |
| **Phase 4** | Clustering & Scalability | **IMPLEMENTED** | Pending Phase 3 Gate | IC-FPLSTM (PCA+elbow+kmeans), RC-FPLSTM (buggy vs fixed) |
| **Phase 5** | Critical Audit & Critique | **COMPLETED** | Verified | `docs/notes/critique.md` (10 verified issues + proofs) |
| **Phase 6** | Methodological Improvements | **IMPLEMENTED** | Pending Phase 4 Gate | Weighted aggregation, Model-sim clustering, SLA loss, DP-FedProx |
| **Phase 7** | Documentation, Report & Presentation | IN PROGRESS | - | LaTeX Report (`main.tex`), PPTX deck builder, Video outline |
| **Phase 8** | Reproducibility & Final Packaging | UPCOMING | - | `make all` end-to-end verification, `submission.zip` |

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
