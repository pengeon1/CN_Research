# Mathematical Formalization & Theoretical Analysis

This document provides complete mathematical definitions, symbol reconciliations, and parameter complexity derivations for Phyu et al. (*IEEE TNSM* 2023).

---

## 1. Reconciled Symbol Table

Reconciling Table II of the paper with standard Federated Learning notation:

| Symbol | Paper Definition | Standard FL Meaning | Domain / Dimensions |
|:---:|:---|:---|:---|
| $b \in B$ | Base station in heterogeneous set | Physical cellular site | $|B| = 57$ in empirical study |
| $s \in S$ | Sliced instance in set of slices | Vertical service / MVNO tenant | $|S| = 4$ (FB, YT, Google, IG) |
| $B_s \subseteq B$ | Base stations attached to slice $s$ | Client participation domain | Assumed $B_s = B$ in paper |
| $\mathcal{T}$ | Set of time intervals of interest | Discrete time horizon | 10-min bins, 144/day, 1440 total |
| $v_b^s(t)$ | Traffic volume at base station $b$, slice $s$, time $t$ | Scalar telemetry observation | $\mathbb{R}_{\ge 0}$ (normalized in $[0, 1]$) |
| $X_b^s$ | Historical traffic dataset at station $b$ for slice $s$ | Local client private dataset | $X_b^s \in \mathbb{R}^{|\mathcal{T}|}$ |
| $H$ | Observation window length | Number of historical lags (input) | Default $H=5$ (50 min); sweep: $[2, 20]$ |
| $W$ | Prediction window length | Forecasting horizon (output) | Default $W=1$ (10 min); sweep: $[1, 15]$ |
| $X_b^s(t, H)$ | Traffic trend vector for observation window $H$ | Feature input tensor to LSTM | $\mathbb{R}^H$ |
| $\tilde{X}_b^s(t, W)$ | Forecast traffic demand over prediction window $W$ | Model prediction output | $\mathbb{R}^W$ |
| $N$ | Set / total count of local agents | Total FL clients | $N = |S| \times |B| = 228$ |
| $N^c \subseteq N$ | Selected local agents in round $c$ | Sampled active clients in round $c$ | $|N^c| = \lceil \Omega \cdot N \rceil$ |
| $\Omega$ | Client selection fraction | Client sampling rate | $\Omega \in [0.1, 1.0]$, default $1.0$ |
| $C$ | Total global communication rounds | Outer federated iterations | $C = 20$ |
| $E$ | Number of local training epochs | Inner SGD epochs per round | $E = 200$ |
| $\beta$ | Local mini-batch size | Local SGD mini-batch size | $\beta = 16$ |
| $\eta$ | Learning rate | SGD step size | $\eta = 0.001$ |
| $\mu$ | Proximal regularization coefficient | FedProx penalty weight | $\mu \in [0.001, 1.0]$, default $1.0$ |
| $w_J^c$ | Weight matrix of global neural network $J$ at round $c$ | Global model parameters | $\mathbb{R}^{P_\psi}$ |
| $w_{J_b^s}^c$ | Weight matrix of local neural network at station $b$, slice $s$ | Local model parameters | $\mathbb{R}^{P_\psi}$ |
| $GL(\cdot)$ | Global objective loss function | Global population risk | $\mathbb{R}$ |
| $LL(\cdot)$ | Local loss function | Local empirical risk | $\mathbb{R}$ (MSE / RMSE) |
| $\rho_b^s$ | Relative sample weighting factor | Sample ratio $a_b^s / a$ | $\rho_b^s \ge 0, \sum \rho_b^s = 1$ |
| $\psi$ | Size of transferred object | Model size ($P_\psi \cdot \Upsilon$) | Bytes or Megabytes (MB) |
| $\gamma$ | Over-provisioning margin | Capacity safety margin factor | $\gamma \in [0.0, 0.5]$ |

---

## 2. Core Equations & Theoretical Formulations

### 2.1 Global Federated Objective — Equation (1)
The paper formalizes the global multi-slice optimization problem as:
$$\min_{w_J} GL(w_J) = \sum_{s \in S} \sum_{b \in B_s} \rho_b^s LL(w_{J_b^s})$$

where:
$$\rho_b^s = \frac{a_b^s}{a}, \quad a = \sum_{s \in S} \sum_{b \in B_s} a_b^s$$
and $a_b^s = |X_b^s|$ is the cardinality of local dataset $X_b^s$.

Local empirical loss:
$$LL(w_{J_b^s}) = \frac{1}{|X_b^s|} \sum_{i=1}^{|X_b^s|} \ell\left(f(X_{b,i}^s; w_{J_b^s}), y_{b,i}^s\right)$$

> **[OUR ANALYSIS] Theoretical Discrepancy:**
> Equation (1) strictly prescribes sample-weighted aggregation ($\rho_b^s$). However, Algorithm 1 Line 7 uses an unweighted average:
> $$w_J^{c+1} = \frac{1}{|N^c|} \sum_{n_b^s \in N^c} w_{J_b^s}^{c+1}$$
> In our experiments, we implement both the unweighted paper version and the corrected sample-weighted version to quantify the impact on heterogeneous local dataset sizes.

---

### 2.2 LSTM Parameter Count Derivation — Equation (2)
The paper quotes the LSTM parameter complexity formula (cites Sak et al., 2014):
$$W = 4 I H + 4 H^2 + 3 H + Z H \tag{2}$$
where $I$ is input units, $H$ is hidden units, and $Z$ is output units.

#### Dissection and Comparison with Standard PyTorch LSTM:
1. **Paper's Formulation (Sak et al., 2014):**
   - 4 internal gates (input, forget, cell, output): $4 \times (I \cdot H)$ input-to-hidden weights.
   - Recurrent hidden-to-hidden connections: $4 \times (H \cdot H) = 4 H^2$.
   - Peephole connections or tied gate biases: $3 H$.
   - Dense output projection: $Z \cdot H$.
   - For $I=1, H=64, Z=1$:
     $$W_{\text{paper}} = 4(1)(64) + 4(64^2) + 3(64) + (1)(64) = 256 + 16384 + 192 + 64 = 16,896\text{ parameters}$$

2. **Standard PyTorch LSTM (`nn.LSTM(1, 64)` + `nn.Linear(64, 1)`):**
   - Input weights `weight_ih_l0`: shape $(4H, I) \rightarrow 4 \times 64 \times 1 = 256$
   - Hidden weights `weight_hh_l0`: shape $(4H, H) \rightarrow 4 \times 64 \times 64 = 16,384$
   - Input bias `bias_ih_l0`: shape $(4H) \rightarrow 256$
   - Hidden bias `bias_hh_l0`: shape $(4H) \rightarrow 256$
   - Linear head `weight`: shape $(Z, H) \rightarrow 1 \times 64 = 64$
   - Linear head `bias`: shape $(Z) \rightarrow 1$
   - Total PyTorch parameters:
     $$P_{\text{PyTorch}} = 256 + 16384 + 256 + 256 + 64 + 1 = 17,217\text{ parameters}$$
   - Difference: PyTorch maintains two independent bias vectors for input and recurrent gates plus a bias in the linear readout layer, resulting in $17,217 - 16,896 = 321$ additional parameters.

---

### 2.3 Federated Communication Cost Model
Per Section VI-C, the total network communication overhead across $C$ rounds is:
$$\text{Cost}_{\text{FL}} = 2 \cdot C \cdot N \cdot \Omega \cdot \psi$$

where:
- Factor $2$ accounts for downlink (global model distribution) and uplink (local model updates).
- $\psi = P_\psi \cdot \Upsilon$, where $P_\psi$ is total trainable parameter count and $\Upsilon$ is floating-point precision (32 bits = 4 bytes).
- For $P_\psi = 17,217$, $\psi = 17,217 \times 4\text{ bytes} \approx 68.87\text{ KB}$.
- Across $C=20$ rounds, $N=228$ agents, $\Omega=1.0$:
  $$\text{Cost}_{\text{FL}} = 2 \times 20 \times 228 \times 1.0 \times 68,868\text{ bytes} \approx 628.08\text{ MB}$$

#### Comparison with Centralized Raw Data Transmission:
In a centralized setup (DeepCog), local agents transmit their full raw time-series datasets to the central server:
$$\text{Cost}_{\text{Centralized}} = N \cdot |\mathcal{T}| \cdot \Upsilon_{\text{data}}$$
For 10 days at 10-minute bins ($|\mathcal{T}| = 1440$ samples) with 32-bit floats:
$$\text{Cost}_{\text{Centralized}} = 228 \times 1440 \times 4\text{ bytes} \approx 1.31\text{ MB}$$

> **[OUR ANALYSIS] Critical Observation on Communication Framing:**
> Transmitting raw telemetry for a short 10-day dataset actually consumes **substantially less communication bandwidth** ($1.31\text{ MB}$) than 20 rounds of federated learning ($628\text{ MB}$)! Federated learning becomes communication-efficient only when deployed over continuous, long-term streaming horizons where raw data accumulates infinitely while model size remains constant. The paper's claim in Section VI-F that FPLSTM has a "constant communication cost" while DeepCog scales with dataset size is valid in the infinite time limit, but DeepCog transmission was framed unfavorably in short-term experimental reporting.

---

### 2.4 SLA Violation Rate Formulation
An SLA violation occurs when the forecasted allocation falls short of actual traffic demand. With over-provisioning margin $\gamma \ge 0$:
$$\text{Allocated Capacity } \hat{C}_b^s(t) = (1 + \gamma) \cdot \tilde{v}_b^s(t)$$
$$\text{SLA Violation Indicator: } \mathbb{I}_{\text{SLA}}(t) = \begin{cases} 1, & \text{if } v_b^s(t) > \hat{C}_b^s(t) \\ 0, & \text{otherwise} \end{cases}$$
$$\text{SLA Violation Rate}(\gamma) = \frac{1}{|\mathcal{T}_{\text{test}}|} \sum_{t \in \mathcal{T}_{\text{test}}} \mathbb{I}\left(v_b^s(t) > (1 + \gamma) \tilde{v}_b^s(t)\right)$$

