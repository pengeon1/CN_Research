# Algorithmic Audit & Pseudocode Dissection

This document provides rigorous algorithmic specifications, line-by-line walkthroughs, and critical audits for Algorithms 1 through 4 presented in Phyu et al. (*IEEE TNSM* 2023).

---

## 1. Algorithm 1: Global Agent (GA)

```python
"""
Algorithm 1: Global Agent (GA)
Input: C (communication rounds), N (set of local agents), mu (proximal weight),
       w^0 (initial weights), Omega (selection rate)
Output: Converged global model w_J^C
"""
1:  Initialize w_J^0 = w^0, c = 0
2:  for each communication round c = 0, 1, ..., C - 1 do
3:      GA randomly selects Omega fraction of local agents N^c subset of N
4:      GA broadcasts w_J^c to all chosen local agents n_b^s in N^c
5:      /* Local Agents execute parallel training at base stations (Algorithm 2) */
6:      GA receives w_{J_b^s}^{c+1} from all participating agents in N^c
7:      GA updates global model:
            w_J^{c+1} = (1 / |N^c|) * sum_{n_b^s in N^c} w_{J_b^s}^{c+1}
8:      GA updates global_model(w_J^{c+1})  # No dataset required at GA
9:  end for
```

### Critical Audit of Algorithm 1
* **Discrepancy with Objective Eq. (1):** Line 7 computes a naive unweighted arithmetic mean. As established in [math_formalization.md](file:///home/piyush/CN_Research/docs/notes/math_formalization.md), this directly contradicts Equation (1), which defines global loss as $\sum \rho_b^s LL(w_{J_b^s})$ with sample weight $\rho_b^s = a_b^s / a$. If client datasets differ in size (e.g., base stations deployed at different times), unweighted averaging introduces client weight distortion.
* **Synchronous Assumption:** Line 6 assumes all selected agents return updates within round $c$. Stragglers or dropped connections in heterogeneous RAN environments are unaddressed at the coordinator level.

---

## 2. Algorithm 2: Local Agent (LA)

```python
"""
Algorithm 2: Local Agent (LA)
Input: X_b^s (local dataset), beta (batch size), E (local epochs),
       w_J^c (global weights), eta (learning rate), mu (proximal weight), c (round)
Output: w_{J_b^s}^{c+1} (updated local model weights)
"""
1:  Initialize w_{J_b^s}^c = w_J^c  # Adopt global model weights
2:  LA trains local_model(w_{J_b^s}^c, X_b^s)
3:  Batch <- Split X_b^s into batches of size beta
4:  for local epoch i = 1, ..., E do
5:      for each batch beta in Batch do
6:          w_{J_b^s}^* <- w_{J_b^s} - eta * nabla l(w_{J_b^s}, beta)
7:      end for
8:  end for
9:  Obtain local weight w_{J_b^s}^{c+1} approx argmin_{w} [ LL(w^*) + (mu / 2) * ||w^* - w_J^c||^2 ]
10: LA uploads w_{J_b^s}^{c+1} to GA
```

### Critical Audit of Algorithm 2: The FedProx Discrepancy
* **The Standard FedProx Definition (Li et al., MLSys 2020):**
  In true FedProx, the local objective explicitly adds the proximal penalty to the empirical loss *during* SGD:
  $$\min_w h_k(w; w^c) = F_k(w) + \frac{\mu}{2} \|w - w^c\|^2$$
  Each mini-batch gradient step is:
  $$w \leftarrow w - \eta \left( \nabla \ell(w; \beta) + \mu (w - w^c) \right)$$
* **What Algorithm 2 Literally Does:**
  Lines 4–8 execute standard Stochastic Gradient Descent for $E$ epochs without any proximal regularization, producing $w^*$. Line 9 then states that $w^{c+1} \approx \arg\min_w [ LL(w^*) + \frac{\mu}{2} \|w^* - w^c\|^2 ]$.
  If $w^*$ is already the terminal point of SGD, evaluating this objective at $w^*$ does not alter $w^*$ unless an extra optimization or proximal contraction step is executed.
* **Our Resolution in Code:**
  We implement:
  1. `true_fedprox`: Standard in-loop proximal gradient descent.
  2. `paper_alg2`: Literal post-SGD proximal contraction step:
     $$w^{c+1} = \frac{w^* + \mu w^c}{1 + \mu}$$
  This enables rigorous empirical comparison of convergence rates.

---

## 3. Algorithm 3: Information-Based Clustering (IC-FPLSTM)

```python
"""
Algorithm 3: IC-FPLSTM
Input: Y (set of base station feature vectors), Z (max iterations), trend (boolean)
Output: G = {G_1, ..., G_K} (set of clusters of base stations)
"""
1:  if trend == TRUE then
2:      Y = PCA(Y, n_components=2)  # Reduce high-dimensional traffic trends
3:  end if
4:  K = Elbow(Y)  # Determine optimal number of clusters K
5:  Centroids C = {c_1, ..., c_K} subset of Y  # Randomly initialize centroids
6:  for z = 1, ..., Z do
7:      G = {empty, ..., empty}  # K empty clusters
8:      for each base station y_b in Y do
9:          k* = argmin_k ||c_k - y_b||^2
10:         G_{k*} <- G_{k*} union {y_b}
11:     end for
12:     for k = 1, ..., K do
13:         c_k <- (1 / |G_k|) * sum_{y_b in G_k} y_b  # Update centroids
14:     end for
15: end for
```

### Critical Audit of Algorithm 3
* **Privacy Leakage in Trend Clustering:** If `trend == True`, base stations must share their historical traffic trend vectors $Y$ with the clustering entity to perform PCA and $k$-means. This directly compromises the paper's core privacy premise (safeguarding private domain data from the InP).
* **Elbow Heuristic Ambiguity:** Line 4 calls `Elbow(Y)`. The elbow method requires evaluating second-order differences or curvature on the inertia curve across $K \in [1, K_{\max}]$. In our implementation, we programmatically compute the elbow point using the maximum distance to the chord line (Kneedle algorithm).

---

## 4. Algorithm 4: Random Clustering (RC-FPLSTM) & The Shrinking Threshold Bug

```python
"""
Algorithm 4: RC-FPLSTM (Literal Paper Formulation)
Input: B_s (set of base stations), K (target number of clusters)
Output: G = {G_1, ..., G_K}
"""
1:  G = {empty, ..., empty}
2:  for k = 1, ..., K do
3:      while |G_k| < |B_s| / K do       # <--- FATAL BUG: |B_s| shrinks inside loop!
4:          b = Random(B_s)
5:          G_k <- G_k union {b}
6:          B_s <- B_s \ {b}             # <--- B_s is decremented here!
7:      end while
8:  end for
9:  if B_s != empty then
10:     for each b in B_s do
11:         k = Random(K)
12:         G_k <- G_k union {b}
13:     end for
14: end if
```

### The Shrinking Threshold Bug Explained:
1. Suppose $|B_s| = 57$ and $K = 3$. Target cluster size is $57 / 3 = 19$.
2. In iteration $k=1$:
   - Initially $|B_s| = 57$, threshold is $57 / 3 = 19$.
   - After picking 1 station, $|B_s| = 56$, threshold is $56 / 3 \approx 18.67$.
   - As elements are added to $G_1$, $|G_1|$ increases while $|B_s| / K$ simultaneously decreases!
   - They cross when $|G_1| \approx 14$, prematurely terminating the `while` loop!
3. In iteration $k=2$:
   - $|B_s|$ has already shrunk to $\sim 43$. The threshold $|B_s| / K$ is now $43 / 3 \approx 14.33$.
   - The loop terminates even earlier, adding only $\sim 9$ stations.
4. Consequence:
   - A massive surplus of unassigned base stations remains in $B_s$.
   - Lines 9–14 dump all remaining stations randomly into clusters, destroying balanced partitioning.

### Our Corrected Formulation:
```python
# Fixed RC-FPLSTM: Precompute invariant target cluster size
target_size = len(B_s) // K
remaining = list(B_s)
rng.shuffle(remaining)
G = [[] for _ in range(K)]
for k in range(K):
    G[k] = remaining[k * target_size : (k + 1) * target_size]
# Distribute remaining (surplus) elements round-robin
for idx, b in enumerate(remaining[K * target_size:]):
    G[idx % K].append(b)
```
In our codebase, we provide both `rc_fplstm_paper_buggy` and `rc_fplstm_fixed` to document and benchmark this flaw.

