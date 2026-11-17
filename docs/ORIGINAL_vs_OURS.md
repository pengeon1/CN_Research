# Traceability Matrix: Original Paper vs. Our Work

This document maintains strict provenance for every claim, equation, algorithmic mechanism, experimental observation, and critique across this project, adhering to the assignment's Hard Rules.

## Tag Definitions
- `[PAPER]`: Direct claim, equation, parameter value, or empirical result asserted by Phyu et al. (IEEE TNSM 2023).
- `[OUR ANALYSIS]`: Critical evaluation, discrepancy audit, bug identification, theoretical derivation, or re-interpretation established by our team.
- `[OUR EXTENSION]`: Novel algorithmic improvement, corrected formulation, differential privacy integration, or custom experimental validation introduced by our team.

---

## Detailed Traceability Ledger

| ID | Item / Topic | Source / Context | Tag | Description & Evidence |
|:---|:---|:---|:---|:---|
| **CLM-01** | Multi-slice RAN traffic forecasting | Section I, III | `[PAPER]` | Proposes forecasting slice-level traffic at individual base stations to assist MVNOs with dynamic anticipatory resource reservations via InP MANO. |
| **CLM-02** | FPLSTM framework | Section IV-C, Alg 1 & 2 | `[PAPER]` | Integrates FedProx and 1D LSTM (64 hidden units) for 228 local agents (57 base stations $\times$ 4 slices). |
| **CLM-03** | Alg 2 vs. FedProx divergence | Alg 2, Line 9 | `[OUR ANALYSIS]` | Paper executes regular SGD for $E$ epochs without proximal gradient, then asserts post-hoc argmin. True FedProx (Li et al., MLSys 2020) requires proximal gradient $\mu(w - w_{global})$ inside local SGD steps. |
| **CLM-04** | Global aggregation discrepancy | Eq. (1) vs. Alg 1 Line 7 | `[OUR ANALYSIS]` | Eq. (1) states global loss is sample-weighted by $\rho_b^s = a_b^s / a$, but Alg 1 Line 7 uses an unweighted arithmetic mean $w^{c+1} = \frac{1}{\|N^c\|} \sum w_i$. |
| **CLM-05** | Alg 4 shrinking threshold bug | Alg 4, Lines 3 & 6 | `[OUR ANALYSIS]` | While loop checks $\|G_k\| < \|B_s\| / K$ while dynamically mutating $B_s \leftarrow B_s \setminus \{b\}$, causing premature termination and corrupting cluster balancing. |
| **CLM-06** | H vs. W notation inversion | Sec III-B vs. Sec VI-E | `[OUR ANALYSIS]` | Sec III-B defines $H$ as observation window and $W$ as prediction window. Sec VI-E narrative text swaps these definitions, although Tables IV & V revert to the original notation. |
| **CLM-07** | IC-FPLSTM & RC-FPLSTM | Section V, Alg 3 & 4 | `[PAPER]` | Clusters base stations using PCA (2 components) + elbow + k-means on traffic trend or lat/long; RC uses random balanced clustering. |
| **CLM-08** | Communication cost model | Section VI-C | `[PAPER]` | Expressed as $2 \cdot C \cdot N \cdot \Omega \cdot \psi$, comparing parameter exchange against centralized raw data transfer. |
| **CLM-09** | Synthetic Poitiers dataset | Data engineering | `[OUR EXTENSION]` | Due to proprietary Orange Poitiers dataset, calibrated synthetic cellular generator matches 57 base stations, 4 slices, diurnal curves, and burstiness profiles for 100% reproducible evaluation. |
| **CLM-10** | Dual FedProx implementation | src/fl/fedprox.py | `[OUR EXTENSION]` | Implementing both true in-loop FedProx and the paper's literal post-SGD Alg 2 to empirically quantify convergence and accuracy differences. |
| **CLM-11** | Corrected weighted aggregation | Improvement (A) | `[OUR EXTENSION]` | Enforces sample-weighted aggregation matching Eq. (1) and handles variable local dataset sizes. |
| **CLM-12** | Model-similarity clustering | Improvement (B) | `[OUR EXTENSION]` | Clusters base stations by model weight/gradient similarity rather than raw traffic trends, eliminating private telemetry exposure during clustering. |
| **CLM-13** | SLA-aware pinball loss | Improvement (C) | `[OUR EXTENSION]` | Asymmetric loss penalizing under-provisioning more heavily than over-provisioning, directly minimizing SLA violation rates. |
| **CLM-14** | Differential Privacy (DP-FedProx) | Improvement (D) | `[OUR EXTENSION]` | Adds gradient clipping and Gaussian noise to provide formal $(\epsilon, \delta)$-differential privacy bounds against model inversion attacks. |

