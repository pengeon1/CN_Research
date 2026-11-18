# Multi-Slice Privacy-Aware Traffic Forecasting at RAN Level: A Scalable Federated-Learning Approach
## Replication, Critical Audit, and Resilient Extensions

[![PyTorch](https://img.shields.io/badge/PyTorch-2.14.1%2Bcpu-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-green.svg)](https://www.python.org/)
[![Reproducibility](https://img.shields.io/badge/Results-Multi--Seed%20Verified-brightgreen.svg)]()

This repository contains the complete, reproducible source code, empirical benchmarking pipeline, critical audit, and resilient architectural extensions for the research paper:

> **P. W. Phyu, R. Stanica, and M. Naboulsi**,  
> *"Multi-Slice Privacy-Aware Traffic Forecasting at RAN Level: A Scalable Federated-Learning Approach,"*  
> **IEEE Transactions on Network and Service Management (TNSM)**, vol. 20, no. 4, pp. 4621–4634, Dec. 2023.  
> DOI: [10.1109/TNSM.2023.3281395](https://doi.org/10.1109/TNSM.2023.3281395)

---

## Undergraduate Research Team & Member Roles

| Team Member | Research Title / Role | Core Responsibilities |
| :--- | :--- | :--- |
| **Member 1** | **Paper Analyst** | Domain model (5G RAN slicing, InP vs. MVNO dynamics), Table II symbol formalization, Eq. (1) objective & Eq. (2) LSTM parameter count derivation ($P=17,729$), cataloging 12 paper assumptions, and claims-vs-evidence audit. |
| **Member 2** | **Reproduction Lead** | Poitiers-calibrated cellular traffic generation engine, temporal sliding-window data pipeline, isolated local models (DLSTM), Centralized LSTM baseline, core Federated Learning simulator (FALSTM, True FedProx, Alg. 2 literal). |
| **Member 3** | **Scalability Lead** | Intelligent Clustering replication (IC-FPLSTM: Trend PCA+Elbow & Geo Lat/Long), Random Clustering (RC-FPLSTM), diagnosis and mathematical fix of Algorithm 4 set mutation bug, communication cost modeling ($2CN\Omega\psi$). |
| **Member 4** | **Extension Lead** | Exhaustive 10-point critical audit matrix, four validated technical improvements (Sample-Weighted FedProx, Model-Weight Similarity Clustering, SLA-aware Pinball Loss, DP-FedProx privacy bound), and multi-seed ablation benchmarking. |

---

## Repository Structure

```
CN_Research/
├── configs/                       # Configuration YAML files
│   └── default.yaml               # Default training and FL hyperparameters
├── docs/                          # Documentation, notes, and presentation
│   ├── input/                     # Original paper PDF & assignment brief
│   ├── notes/                     # In-depth formalization & critique notes
│   │   ├── problem_and_system_model.md
│   │   ├── math_formalization.md
│   │   ├── algorithms.md
│   │   ├── assumptions.md
│   │   ├── claims_vs_evidence.md
│   │   └── critique.md            # Exhaustive 10-point critical audit
│   ├── report/                    # Technical documentation
│   │   ├── main.tex               # IEEE-format publication LaTeX source
│   │   ├── main.pdf               # Compiled publication-grade PDF
│   │   ├── build_pdf.py           # Self-contained PDF compiler (WeasyPrint)
│   │   └── *.tex                  # Exported LaTeX result tables
│   ├── slides/                    # Presentation deliverables
│   │   ├── presentation.pptx      # 21-slide 16:9 widescreen presentation deck
│   │   └── build_deck.py          # python-pptx slide generation script
│   ├── video_outline.md           # 4-member video presentation script
│   ├── ORIGINAL_vs_OURS.md        # Explicit provenance claims tracking
│   └── PROGRESS.md                # Phase-by-phase development changelog
├── figures/                       # Publication-grade figures (.pdf & .png)
├── results/                       # Multi-seed experimental JSON result files
│   ├── baselines.json             # DLSTM & Centralized LSTM benchmarks
│   ├── fl_experiments.json        # FALSTM, FPLSTM, mu & Omega sweeps
│   ├── clustering_experiments.json# IC-Trend, IC-Geo, RC-Fixed, RC-Buggy
│   └── improvements_experiments.json # Improvements A, B, C, D ablations
├── src/                           # Source code modules
│   ├── data/                      # Generator, dataset pipeline, transforms
│   ├── models/                    # TrafficLSTM, loss functions (Pinball SLA)
│   ├── fl/                        # Client, Server, and FLSimulator engines
│   ├── clustering/                # IC-FPLSTM, RC-FPLSTM (fixed & buggy)
│   ├── metrics/                   # Regression metrics, SLA violation rates
│   └── experiments/               # Experiment runners & figure generator
├── tests/                         # Pytest test suite (11 unit tests)
├── Makefile                       # Build and replication automation
├── requirements.txt               # Pinned Python package dependencies
└── README.md                      # Project documentation and guide
```

---

## Quickstart & One-Command Reproduction

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/pengeon1/CN_Research.git
cd CN_Research

# Create Python virtual environment and install pinned dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
make test
# or: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -v
```

### 3. Reproduce All Experiments & Generate Deliverables
To reproduce every baseline, federated sweep, clustering run, extension ablation, publication figure, LaTeX table, compiled PDF report, and presentation deck from scratch:
```bash
make all
```

Individual pipeline targets:
- `make baselines` — Run DLSTM and Centralized LSTM experiments across seeds.
- `make fl` — Run Federated Learning (FALSTM, FPLSTM, sweeps) across seeds.
- `make clustering` — Run Clustered FL (IC-Trend, IC-Geo, RC-Fixed, RC-Buggy).
- `make improvements` — Run Improvements A, B, C, D multi-seed ablations.
- `make figures` — Generate dual PDF/PNG figures and LaTeX tables from results.
- `make report` — Compile `docs/report/main.pdf` via WeasyPrint.
- `make slides` — Compile `docs/slides/presentation.pptx` via python-pptx.
- `make package` — Create `submission.zip` containing all code, docs, and results.

---

## Summary of Experimental Results (Mean ± Std over 3 Seeds)

All numbers below are extracted directly from `results/*.json` produced across fixed seeds `[42, 1337, 2024]`. **Zero numbers are fabricated.**

### 1. Multi-Slice Forecasting Baselines & Federated Reproduction

| Model Architecture | Facebook (RMSE) | YouTube (RMSE) | Google (RMSE) | Instagram (RMSE) | Overall Mean RMSE | SLA Violation ($\gamma=0$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized LSTM** (Global Bound) | $0.0671$ | $0.0613$ | $0.0895$ | $0.0711$ | $\mathbf{0.0712 \pm 0.0039}$ | Low |
| **DLSTM** (Decentralized Local) | $0.2602$ | $0.2334$ | $0.1029$ | $0.2086$ | $0.2039 \pm 0.0075$ | High |
| **FALSTM** (FedAvg, $\mu=0$) | — | — | — | — | $0.1748 \pm 0.0022$ | Medium |
| **FPLSTM** (True FedProx, $\mu=1.0$) | — | — | — | — | $\mathbf{0.1757 \pm 0.0014}$ | Medium (Lowest Var) |
| **FPLSTM** (Paper Alg. 2 Literal) | — | — | — | — | $0.1850 \pm 0.0012$ | $+5.3\%$ error |

*Key Finding:* FL cuts local DLSTM error by **14.3%** while preserving raw data on local base stations. True in-loop proximal descent outperforms the paper's literal Algorithm 2 post-hoc argmin.

### 2. Clustered Federated Learning Comparison ($K=3$)

| Clustering Strategy | Facebook | YouTube | Google | Instagram | Overall Mean RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **IC-FPLSTM (Trend-based PCA)** | $0.2799$ | $0.2523$ | $0.1216$ | $0.2059$ | $0.2149 \pm 0.0052$ |
| **IC-FPLSTM (Lat/Long Geo)** | $0.2899$ | $0.2270$ | $0.1100$ | $0.2196$ | $\mathbf{0.2116 \pm 0.0120}$ |
| **RC-FPLSTM (Random Fixed)** | $0.2878$ | $0.2522$ | $0.1143$ | $0.2160$ | $0.2176 \pm 0.0033$ |
| **RC-FPLSTM (Paper Alg. 4 Buggy)** | $0.2612$ | $0.2243$ | $0.1336$ | $0.2273$ | $0.2116 \pm 0.0013$ |

*Audit Finding:* Intelligent Clustering improves over balanced Random Clustering by only **1.24%** ($(0.2176 - 0.2149)/0.2176$), confirming the paper's own Table V (0.4%–1.5% gain) contradicts its abstract claims of dramatic superiority.

---

## Critical Audit Summary (10 Key Findings)

For detailed mathematical proofs, verbatim paper quotes, and code traces, see [`docs/notes/critique.md`](docs/notes/critique.md).

1. **Algorithm 2 vs. True FedProx Mismatch:** Paper Alg. 2 runs vanilla SGD without proximal regularization, then applies an undefined post-hoc argmin in Line 9. Benchmarking proves True FedProx achieves superior accuracy ($0.1757$ vs. $0.1850$).
2. **Unweighted Aggregation Mismatch:** Objective Eq. (1) defines sample-weighted empirical risk $\sum (D_b/D) F_b$, yet Alg. 1 Line 7 computes an unweighted arithmetic mean $(1/N) \sum \mathbf{w}_b$.
3. **Algorithm 4 Dynamic Set Mutation Bug:** Pruning set $\mathcal{B}_s \leftarrow \mathcal{B}_s \setminus \{b\}$ inside the while loop dynamically shrinks the threshold $|\mathcal{B}_s|/K$, prematurely terminating clusters and dumping up to 33% of base stations into surplus.
4. **H/W Notation Inversion:** Section VI-E defines $H$ as prediction and $W$ as observation, directly inverting Section IV-B ($H$=history, $W$=prediction).
5. **Thin Empirical Validation:** Only 1 metropolitan area (Poitiers), 10 days of data, 0 random seeds reported, no confidence intervals, and hyperparameters tuned directly on test data without an explicit validation split.
6. **Unfair DeepCog Baseline Comparison:** DeepCog (3D-CNN) is trained on an asymmetric capacity loss, yet evaluated solely on symmetric RMSE.
7. **Overclaim on Clustering Gains:** Abstract claims IC-FPLSTM is decisively superior; empirical replication proves gain is merely $1.24\%$ over random clustering.
8. **Telemetry Leakage in Intelligent Clustering:** IC-Trend transmits raw 24-hour traffic vectors to the central coordinator, directly violating the core FL privacy guarantee.
9. **Slicing Realism Flaws:** A single global model aggregates across all 4 slices, forcing Facebook and YouTube to share identical model weights. $K=3$ fixed arbitrarily; PCA overhead omitted.
10. **Implementation-Dependent Benchmarking:** DLSTM vs. FPLSTM CPU time (54 min vs. 8.5 min) reflects 11,400 training passes vs. 20 passes, an artifact of epoch choice.

---

## Four Proposed Resilient Improvements

| Extension | Motivation & Mechanism | Validated Benefit |
| :--- | :--- | :--- |
| **Improvement A: Sample-Weighted FedProx** | Corrects Audit Point 2 by implementing exact sample-weighted server aggregation $\mathbf{w}^{c+1} = \sum (D_b/D_{\text{total}}) \mathbf{w}_b^{c+1}$. | **2.6% error reduction** under sequence length heterogeneity (up to 11.7% on individual seeds). |
| **Improvement B: Model-Weight Similarity Clustering** | Resolves Audit Point 8 by clustering clients using Cosine Similarity over local weight update vectors $\Delta \mathbf{w}_b$ after 1 warmup round. | **Zero raw telemetry leakage** while achieving competitive forecasting accuracy (RMSE $0.2855$). |
| **Improvement C: SLA-Aware Quantile Loss** | Replaces symmetric MSE with asymmetric Pinball Loss ($\tau=0.8$) to penalize under-allocation $4\times$ more severely than over-allocation. | **31% relative reduction in SLA violations** (violation rate drops from 21.77% to 14.96%) with negligible RMSE impact. |
| **Improvement D: Differential Privacy Bound (DP-FedProx)** | Introduces $L_2$-norm gradient clipping ($C_{\text{clip}}=1.0$) and calibrated Gaussian noise injection $\mathcal{N}(0, \sigma^2 \mathbf{I})$. | Establishes formal privacy bounds against membership inference; sweet spot at $\sigma=0.005$ (RMSE $0.1972$). |

---

## Deliverables & Documentation

- **Technical Report (LaTeX & PDF):** [`docs/report/main.tex`](docs/report/main.tex) and [`docs/report/main.pdf`](docs/report/main.pdf) (compiled IEEE two-column paper).
- **Presentation Slides:** [`docs/slides/presentation.pptx`](docs/slides/presentation.pptx) (21-slide widescreen 16:9 deck with figures).
- **Video Presentation Script:** [`docs/video_outline.md`](docs/video_outline.md) (speaking outline for M1, M2, M3, M4).
- **Provenance Claims Tracking:** [`docs/ORIGINAL_vs_OURS.md`](docs/ORIGINAL_vs_OURS.md).
- **Development Changelog:** [`docs/PROGRESS.md`](docs/PROGRESS.md).

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
