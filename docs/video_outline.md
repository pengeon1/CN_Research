# Video Presentation Script & Speaking Outline

**Target Paper:** *"Multi-Slice Privacy-Aware Traffic Forecasting at RAN Level: A Scalable Federated-Learning Approach"* (Phyu et al., IEEE TNSM 2023)  
**Assignment:** Computer Networks Research Paper Analysis  
**Team Structure:** 4 Undergraduates with Specialized Engineering Roles  
**Estimated Duration:** ~18–20 minutes (~4.5–5 minutes per member)

---

## Member 1: Paper Analyst & System Model Specialist (M1)

### Role & Ownership
Responsible for explaining the fundamental research problem, industrial motivation in 5G RAN slicing, mathematical modeling, algorithmic foundations, explicit/implicit assumptions, and the initial claims-vs-evidence audit.

### Speaking Script Outline (Slides 1 – 6)

#### 1. Title, Motivation & Cellular Slicing Landscape (0:00 – 1:30)
- **Introduction:** Introduce the team, course context, and the paper: Phyu et al. (*IEEE Transactions on Network and Service Management*, December 2023).
- **The Core Dilemma:** As 5G rolls out heterogeneous vertical services (eMBB, URLLC, mMTC), Mobile Virtual Network Operators (MVNOs) rent physical infrastructure from an Infrastructure Provider (InP).
- **Static vs. Anticipatory Provisioning:** Static over-provisioning wastes capital expenditure and radio resources. Anticipatory provisioning (predicting traffic for the upcoming window $W$) can boost RAN throughput by 35–40% (Bui & Widmer, 2018).
- **The Privacy Challenge:** Centralizing raw traffic at the InP discloses **Private Domain Data** (MVNO commercial strategy and market share) and **Personal Domain Data** (individual user habits in pico/femto small cells).

#### 2. System Model & Problem Dimensionality (1:30 – 3:00)
- **Problem Dimensions:**
  - *Centralized (3D) $\langle S, B, T \rangle$:* Exploits full spatial-temporal cross-slice correlations, but suffers from high complexity, high backhaul transmission, and total privacy exposure (e.g., DeepCog).
  - *2D-FL $\langle B_s, T \rangle$:* Groups stations per slice; still centralizes base stations within each MVNO.
  - *1D-FL $\langle T \rangle$ (Proposed FPLSTM):* Each local agent models a single 1D time series at base station $b$ for slice $s$. Only neural network weight matrices are exchanged with the InP global coordinator.
- **Mathematical Formulation:**
  - Define historical observation window $X_b^s(t, H) = \{v_b^s(t-H), \dots, v_b^s(t-1)\}$.
  - Define forecast horizon $\tilde{X}_b^s(t, W) = \{\tilde{v}_b^s(t), \dots, \tilde{v}_b^s(t+W-1)\}$.
  - Detail global objective Equation (1): $\min_{w_J} GL(w_J) = \sum_{s \in S} \sum_{b \in B_s} \rho_b^s LL(w_{J_b^s})$, where $\rho_b^s = a_b^s / a$ is relative sample weighting.

#### 3. Core Equations, LSTM Parameter Count & Assumptions (3:00 – 4:45)
- **LSTM Parameter Complexity:** Present Equation (2): $W = 4IH + 4H^2 + 3H + ZH$. Contrast Sak et al. (16,896 parameters for $H=64$) with PyTorch's actual dual-bias implementation ($17,217$ parameters).
- **Inventory of Assumptions:** Highlight key paper assumptions:
  1. Full geographic overlap ($B_s = B$ across all 4 slices).
  2. One-to-one mapping between single consumer apps (Facebook, YouTube, Google, Instagram) and network slices.
  3. Synchronous federated rounds with zero stragglers across heterogeneous base station hardware.
  4. Presumed privacy: exchanging raw model weights assumes no gradient inversion leakage.
- **Transition:** Hand over to Member 2 (Reproduction Lead) to present our data engineering and baseline reproduction.

---

## Member 2: Reproduction Lead & FL Core Engineer (M2)

### Role & Ownership
Responsible for the data pipeline, calibrated synthetic cellular generator, preprocessor scaling, non-collaborative DLSTM and centralized LSTM baselines, and faithful reproduction of FALSTM vs. FPLSTM.

### Speaking Script Outline (Slides 7 – 12)

#### 1. Data Engineering & Cellular Traffic Generator (4:45 – 6:15)
- **The Dataset Challenge:** The paper's Orange Poitiers dataset is private and unavailable.
- **Our Primary Solution:** Designed a documented, calibrated synthetic cellular generator matching Poitiers' exact setup:
  - 57 base stations with urban core and suburban coordinates.
  - 4 vertical application slices over 10 days at 10-minute intervals (1440 time slots per station).
  - Explicit physical calibration matching Section VI-E:
    - *Facebook:* Smooth, slowly varying diurnal pattern with moderate noise.
    - *Google:* Simpler diurnal shape, but highly bursty with sharp, frequent, intense traffic spikes.
    - *YouTube:* High volume, evening video streaming peaks with heavy-tailed bursts.
    - *Instagram:* Intermediate behavior combining photo/video feeds.
- **Leakage Prevention:** Chronological 80/20 train/test split; validation split carved strictly from training data; Min-Max scaling fitted strictly on training observations.

#### 2. Baseline Implementations & Reproduction (6:15 – 7:45)
- **DLSTM (Decentralized LSTM):** 228 local agents trained in complete isolation without global model sharing. Serves as our privacy-preserving lower bound.
- **Centralized LSTM:** Unified multi-series model trained on pooled data from all stations, serving as our non-private cloud benchmark.
- **Metrics Evaluated:** RMSE, MAE, MAPE, and SLA violation rate under safety margin $\gamma$.
- **Showcase Baseline Results:** Present our experimental baseline table and Figure 7 reproduction, noting that DLSTM performs respectably on stationary slices, but centralized training achieves lower error on bursty slices.

#### 3. Federated Learning Reproduction: FALSTM vs. FPLSTM (7:45 – 9:30)
- **Federated Simulation Architecture:** Explain our in-memory, deterministic FL simulator (`src/fl/simulator.py`).
- **The FedProx Mechanism:** Explain why standard FedAvg struggles with non-IID mobile traffic and how the proximal term $\frac{\mu}{2}\|w - w_J^c\|^2$ restricts client drift.
- **$\mu$ Parameter Sweep:** Present our re-creation of Figure 12 across $\mu \in [0.001, 1.0]$. Demonstrate that $\mu = 1.0$ provides the best regularization against client drift across heterogeneous slices.
- **The Dual Implementation Discovery:** Introduce the critical discovery that Algorithm 2 in the paper executes standard SGD for $E$ epochs without proximal gradient, asserting an argmin post-hoc. Contrast this with true FedProx (in-loop proximal gradient descent).
- **Transition:** Hand over to Member 3 (Scalability Lead) to discuss clustering, scalability, and communication cost modeling.

---

## Member 3: Scalability Lead & Clustering Architect (M3)

### Role & Ownership
Responsible for IC-FPLSTM (traffic trend and lat/long), RC-FPLSTM, the discovery and fix of the Algorithm 4 shrinking-threshold bug, client participation rate analysis, communication cost modeling, and CPU scaling.

### Speaking Script Outline (Slides 13 – 17)

#### 1. Scalability Bottlenecks in Ultra-Dense RANs (9:30 – 10:45)
- **The Problem:** In nation-wide networks with $10^5$ base stations, a single global coordinator suffers from:
  1. Massive computation and memory aggregation bottlenecks.
  2. High backhaul latency between distant edge stations and core clouds.
- **Clustering Paradigm (Section V, Figures 4 & 5):** Partition base stations into clusters, running localized global models per cluster at regional edge nodes.

#### 2. Information-Based Clustering (IC-FPLSTM) (10:45 – 12:00)
- **Algorithm 3 Walkthrough:**
  - *Traffic Trend:* High-dimensional traffic vectors reduced via PCA to 2 components. Optimal $K$ determined via the Elbow method on KMeans inertia curves.
  - *Geographical Lat/Long:* Base station coordinates clustered directly via Euclidean distance.
- **Empirical Findings:** Trend-based and Lat/Long clustering achieve ~0.4% to 1.5% RMSE improvement over unclustered FPLSTM, because intra-cluster models share correlated spatial-temporal patterns.
- **Privacy Trade-off:** Highlight that sharing traffic trends for clustering violates the InP data privacy boundary—a major paradox in the paper's design!

#### 3. Random Clustering & The Algorithm 4 Bug Fix (12:00 – 13:30)
- **RC-FPLSTM Concept:** Random balanced partitioning as a low-complexity benchmark.
- **The Algorithm 4 Shrinking-Threshold Bug:**
  - Line 3 checks `while |G_k| < |B_s| / K do`, but Line 6 modifies $B_s \leftarrow B_s \setminus \{b\}$.
  - As $B_s$ shrinks, the threshold decreases dynamically during execution, prematurely terminating cluster filling and dumping over 30% of base stations into the surplus branch!
- **Our Fixed Implementation:** Precompute invariant target size $\lfloor |B_s| / K \rfloor$ and distribute remainder round-robin.
- **Communication Cost Model ($2 \cdot C \cdot N \cdot \Omega \cdot \psi$):**
  - Walk through communication scaling vs. client participation $\Omega \in [0.1, 1.0]$ (Figure 10).
  - Reveal that $\Omega = 0.5$ cuts communication overhead by 50% with negligible accuracy degradation (<1%).
- **Transition:** Hand over to Member 4 (Extension Lead) for our critical audit, proposed improvements, and conclusions.

---

## Member 4: Extension Lead & Critical Auditor (M4)

### Role & Ownership
Responsible for synthesizing the 10-point critical audit, presenting our 4 methodological improvements (sample-weighted aggregation, model-similarity clustering, SLA-aware loss, and differential privacy), and summarizing conclusions.

### Speaking Script Outline (Slides 18 – 24)

#### 1. Comprehensive Critical Audit (13:30 – 15:00)
- **Walk through the 10 core audit findings:**
  1. *Algorithm 2 Mismatch:* Post-hoc argmin vs. in-loop proximal descent.
  2. *Equation 1 Discrepancy:* Sample-weighted loss formulation vs. unweighted arithmetic averaging in Algorithm 1.
  3. *Algorithm 4 Bug:* Shrinking threshold corrupts cluster balancing.
  4. *Notation Inversion:* $H$ and $W$ swapped between Section III-B and Section VI-E narrative.
  5. *Thin Evaluation:* Single city, 10 days, no confidence intervals, no random seed reporting.
  6. *Unfair DeepCog Comparison:* 3D-CNN with asymmetric capacity loss compared against 1D-LSTM on symmetric RMSE.
  7. *Overclaim in Conclusion:* Claiming IC-FPLSTM "outperforms" state-of-the-art when gain is only 0.4–1.5% (likely statistical noise).
  8. *No Formal Privacy Guarantee:* Weight exchanges remain vulnerable to gradient inversion; trend clustering exposes private traffic.
  9. *Single Global Model for Heterogeneous Apps:* Averaging Facebook weights with Google weights risks negative transfer.
  10. *CPU Time Artifact:* CPU hours difference reflects unoptimized 3D-CNN vs. 1D-LSTM, not an inherent FL property.

#### 2. Proposed Improvements & Experimental Validation (15:00 – 17:00)
- **Improvement A: Sample-Weighted Aggregation:**
  - Implemented $\sum \frac{a_i}{\sum a_j} w_i$ matching Equation (1). Evaluated under heterogeneous dataset lengths, showing faster convergence and lower variance.
- **Improvement B: Privacy-Preserving Model-Similarity Clustering:**
  - Clustered base stations using cosine similarity of initial model weight updates rather than raw traffic trends. Eliminates raw telemetry leakage while matching IC-FPLSTM accuracy.
- **Improvement C: SLA-Aware Asymmetric Quantile Loss:**
  - Replaced symmetric MSE with pinball loss ($\tau = 0.8$). Reduced SLA violations from 18.4% to 4.2% under $\gamma = 0$, directly aligning forecasting with network revenue requirements.
- **Improvement D: Differential Privacy (DP-FedProx):**
  - Added gradient clipping and Gaussian noise. Plotted the empirical privacy-accuracy Pareto curve, quantifying the accuracy trade-off for formal $(\epsilon, \delta)$-differential privacy.

#### 3. Summary of Contributions, Deliverables & Conclusion (17:00 – 18:30)
- **Team Deliverables Overview:**
  - Full reproducible codebase with unified `Makefile` (`make all`).
  - Comprehensive LaTeX report (`docs/report/main.pdf`) and 22-slide presentation (`docs/slides/presentation.pptx`).
  - Complete traceability matrix in `docs/ORIGINAL_vs_OURS.md`.
- **Final Verdict:** Phyu et al. present a promising vision for privacy-preserving RAN slicing, but contains theoretical discrepancies, algorithmic bugs, and privacy oversights. Our work successfully reproduces, audits, fixes, and extends the framework to provide true privacy and SLA resilience.

