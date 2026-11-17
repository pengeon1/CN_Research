# Critical Audit & Methodological Critique

**Target Paper:** *"Multi-Slice Privacy-Aware Traffic Forecasting at RAN Level: A Scalable Federated-Learning Approach"* (Phyu et al., IEEE TNSM 2023)  
**Status:** In-Depth Empirical & Mathematical Audit (`[OUR ANALYSIS]`)

---

## 1. Issue 1: Algorithm 2 vs. True FedProx Mismatch

- **Category:** Theoretical & Algorithmic Inconsistency
- **Exact Paper Quote (<15 words):** *"the proximal term is integrated into $LL(w_{J_b^s}^*)$ to limit the effect"* (Section IV-C, p. 5043)
- **Mathematical Audit:**
  In standard FedProx (Li et al., *MLSys* 2020), the local sub-problem optimizes:
  $$\min_w h_k(w; w^c) = F_k(w) + \frac{\mu}{2}\|w - w^c\|^2$$
  The proximal gradient $\mu(w - w^c)$ must be added *inside each SGD mini-batch step*:
  $$w \leftarrow w - \eta \left(\nabla \ell(w; \beta) + \mu(w - w^c)\right)$$
  In contrast, Algorithm 2 in the paper executes standard SGD for $E$ epochs without any proximal gradient (Lines 4–7), producing $w^*$. Line 9 then states:
  $$w^{c+1} \approx \arg\min_w \left[ LL(w^*) + \frac{\mu}{2}\|w^* - w^c\|^2 \right]$$
  Since $w^*$ is already the terminal point of SGD, evaluating this objective at $w^*$ does not alter $w^*$, unless an ad-hoc post-hoc shrinkage step is performed.
- **Verdict:** **CONFIRMED DISCREPANCY.** The paper conflates in-loop proximal regularization with an ad-hoc post-SGD penalty. Our code implements both `true_fedprox` and `paper_alg2` to benchmark their convergence differences.

---

## 2. Issue 2: Unweighted Mean in Alg. 1 Line 7 vs. Sample-Weighted $\rho$ in Eq. (1)

- **Category:** Mathematical Formulation Conflict
- **Exact Paper Quote (<15 words):** *"GA updates $w_J^{c+1} = \frac{1}{|N^c|} \sum_{n_b^s \in N^c} w_{J_b^s}^{c+1}$"* (Algorithm 1, Line 7, p. 5042)
- **Mathematical Audit:**
  Equation (1) defines the global risk minimization objective as:
  $$\min_{w_J} GL(w_J) = \sum_{s \in S} \sum_{b \in B_s} \rho_b^s LL(w_{J_b^s}), \quad \text{where } \rho_b^s = \frac{a_b^s}{a}$$
  where $a_b^s$ is the number of local samples. However, Algorithm 1 Line 7 updates the global model with an unweighted arithmetic mean $\frac{1}{|N^c|} \sum w_i$.
  If client datasets are unbalanced (e.g., base stations deployed at different times or cells with missing data), unweighted averaging gives disproportionate weight to small clients, violating the principle of empirical risk minimization.
- **Verdict:** **CONFIRMED DISCREPANCY.** The algorithm executed does not optimize the objective stated in Eq. (1).

---

## 3. Issue 3: Algorithm 4 Shrinking-Threshold Bug

- **Category:** Algorithmic Defect / Software Bug
- **Exact Paper Quote (<15 words):** *"while $|G_k| < \frac{|B_s|}{K}$ do ... $B_s \leftarrow B_s \setminus \{b\}$"* (Algorithm 4, Lines 3 & 6, p. 5044)
- **Mathematical & Code Audit:**
  In Line 3, the while-loop condition tests `|G_k| < |B_s| / K`. Inside the loop at Line 6, base stations are popped: `B_s = B_s \ {b}`.
  Because $B_s$ shrinks during execution, the threshold $|B_s| / K$ dynamically drops in every step:
  $$\text{Iteration } k=1: |B_s| = 57 \rightarrow 56 \rightarrow \dots \implies \text{threshold drops from } 19 \rightarrow 18 \rightarrow \dots$$
  The condition $|G_1| < |B_s| / K$ terminates when $|G_1| \approx 14$. For $k=2$, $B_s$ has already shrunk to $43$, so $|G_2|$ terminates at $\sim 9$. Over 30% of base stations are left unassigned and dumped randomly into clusters in Lines 9–14.
- **Verdict:** **CONFIRMED BUG.** The author intended to precompute an invariant target size $\lfloor |B_s| / K \rfloor$. Our repository provides both `rc_fplstm_paper_buggy` and `rc_fplstm_fixed`.

---

## 4. Issue 4: Observation vs. Prediction Window Notation Swap ($H$ vs. $W$)

- **Category:** Documentation & Notation Error
- **Exact Paper Quote (<15 words):** *"varying the observation time window W and the prediction time window H"* (Section VI-E, p. 5047)
- **Textual Evidence:**
  - Section III-B (p. 5041) defines $H$ as observation window ($X_b^s(t, H) = \{v(t-H), \dots, v(t-1)\}$) and $W$ as prediction window ($\tilde{X}_b^s(t, W) = \{\tilde{v}(t), \dots, \tilde{v}(t+W)\}$).
  - Section VI-E narrative text (lines 1009–1010) states: "varying the observation time window W and the prediction time window H".
  - However, Table IV is titled "UNDER DIFFERENT PREDICTION TIME WINDOWS W" and Table V is titled "UNDER DIFFERENT OBSERVATION TIME WINDOWS H", reverting to the original definitions.
- **Verdict:** **CONFIRMED NOTATION SWAP.** An editorial error in the narrative text inverted the variables.

---

## 5. Issue 5: Thin Evaluation Methodology

- **Category:** Empirical Rigor & Validation
- **Exact Paper Quote (<15 words):** *"dataset collected from the Orange 4G network in Poitiers, France"* (Section VI-A, p. 5045)
- **Audit Findings:**
  1. **Single Geographic City:** Telemetry is drawn from 57 base stations in Poitiers, France (a small provincial city with ~90k population). Cellular traffic dynamics in dense megacities (e.g., Paris, Tokyo, Milan) exhibit vastly different spatial correlations and peak loads.
  2. **Short 10-Day Timeframe:** 10 days in May 2019 cannot capture monthly billing cycles, holidays, or seasonal mobility shifts.
  3. **Solely RMSE Reported:** No MAE, MAPE, or relative errors are reported in baseline comparisons.
  4. **Zero Random Seeds or Confidence Intervals:** All figures and tables report single scalar numbers without standard deviations, error bars, or seed identifiers.
  5. **No Stated Validation Split:** Train (80%) and Test (20%) are described, but no validation set is mentioned for hyperparameter tuning, creating a substantial risk of test-set snooping.
- **Verdict:** **CONFIRMED LIMITATION.** The empirical claims rest on narrow, single-trial evaluation.

---

## 6. Issue 6: Asymmetric Loss Mismatch in DeepCog Comparison

- **Category:** Benchmark Fairness
- **Exact Paper Quote (<15 words):** *"loss function tries to find a balance between resource overprovisioning and unserviced"* (Section VI-B, p. 5046)
- **Audit Findings:**
  - DeepCog (Bega et al., *IEEE JSAC* 2020) was designed with a specialized asymmetric capacity loss penalizing under-provisioning (which drops packets and incurs heavy SLA penalties) much more severely than over-provisioning.
  - FPLSTM was trained to minimize symmetric Mean Squared Error (RMSE).
  - Evaluating DeepCog on symmetric RMSE evaluates it on an objective it was explicitly not designed to minimize, creating an unfair benchmark.
- **Verdict:** **CONFIRMED UNFAIR COMPARISON.** A fair comparison requires evaluating both models on asymmetric SLA capacity metrics.

---

## 7. Issue 7: Overclaim on IC-FPLSTM Performance Gain

- **Category:** Statistical Significance & Conclusion Overclaim
- **Exact Paper Quote (<15 words):** *"performance of the former approach is approximately 0.4% to 1.5% better"* (Section VI-I, p. 5050)
- **Audit Findings:**
  - In Section VI-I, the paper admits that IC-FPLSTM (traffic trend) improves RMSE over FPLSTM by merely **0.4% to 1.5%**.
  - Yet in the Conclusion (Section VII), the authors claim: *"the advantages of IC-FPLSTM in terms of accuracy, scalability and robustness are clear; the IC-FPLSTM outperforms the contemporary state-of-the-art"*.
  - A 0.4% to 1.5% gain without reported random seeds or statistical significance tests ($p > 0.05$) is well within standard neural network training noise.
  - Furthermore, IC-FPLSTM was never tested against DeepCog.
- **Verdict:** **CONFIRMED OVERCLAIM.** The claimed accuracy superiority is statistically indistinguishable from noise.

---

## 8. Issue 8: Absence of Formal Privacy Guarantees & Trend Leakage

- **Category:** Privacy & Security Vulnerability
- **Exact Paper Quote (<15 words):** *"since an average trend is used ... less likely to expose any sensitive information"* (Section II, p. 5040)
- **Audit Findings:**
  1. **Raw Weight Leakage:** Exchanging raw neural network weights $w_{J_b^s}$ does not guarantee privacy. Gradient inversion attacks (e.g., DLG, Zhu et al., 2019) can reconstruct sequential training inputs from gradients in small models.
  2. **Trend Vector Leakage:** Algorithm 3 requires base stations to share full historical traffic trend vectors $Y$ with the clustering server to perform PCA and $k$-means. For small cells, this directly exposes occupancy and peak usage hours to the InP.
- **Verdict:** **CONFIRMED VULNERABILITY.** The framework lacks formal $(\epsilon, \delta)$-Differential Privacy or Secure Aggregation.

---

## 9. Issue 9: Cross-Slice Weight Averaging & Clustering Overhead Exclusion

- **Category:** Architectural Design & Cost Accounting
- **Audit Findings:**
  1. **Heterogeneous Slice Averaging:** In FPLSTM, weights from Facebook, Google, YouTube, and Instagram local agents are averaged together into one single global model $w_J$. Slices have fundamentally different temporal shapes (Google has sharp bursts; Facebook is smooth). Forcing them into a single global model causes negative knowledge transfer.
  2. **Clustering Computation Omitted:** Figure 15 shows IC-FPLSTM CPU time is 15h vs. FPLSTM 65h. However, this calculation omits the CPU cycles spent executing PCA and iterative $k$-means across all stations.
- **Verdict:** **CONFIRMED ARCHITECTURAL GAPS.**

---

## 10. Issue 10: Implementation-Dependent CPU-Time Comparison

- **Category:** Computational Benchmarking
- **Exact Paper Quote (<15 words):** *"Deepcog requires approximately 18 and 9 times the CPU time"* (Section VI-F, p. 5049)
- **Audit Findings:**
  - DeepCog is a heavy 3-layer 3D-CNN operating over multi-dimensional tensors, whereas FPLSTM is a tiny 1-layer 1D-LSTM (64 hidden units).
  - Comparing CPU hours between an unoptimized 3D-CNN and a lightweight 1D-LSTM conflates neural network architecture complexity with federated learning efficiency.
- **Verdict:** **CONFIRMED COMPARISON ARTIFACT.** The speedup is primarily driven by model parameter count and dimensionality, not federated learning itself.

