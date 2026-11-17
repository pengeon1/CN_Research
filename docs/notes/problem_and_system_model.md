# Problem Formulation & System Model Analysis

**Target Paper:** *"Multi-Slice Privacy-Aware Traffic Forecasting at RAN Level: A Scalable Federated-Learning Approach"* (Phyu, Stanica, Naboulsi, IEEE TNSM 2023)  
**Status:** Comprehensive Analysis Notes (`[PAPER]` vs. `[OUR ANALYSIS]`)

---

## 1. Problem Context & Business Stakeholders

In 5G and next-generation cellular networks, **Network Slicing** enables physical infrastructure sharing among multiple virtual operators, supporting diverse vertical use cases:
1. **eMBB (Enhanced Mobile Broadband):** High-throughput video streaming, social media, web downloads.
2. **URLLC (Ultra-Reliable Low-Latency Communication):** Autonomous driving, industrial automation, remote surgery.
3. **mMTC (Massive Machine-Type Communication):** Smart metering, IoT telemetry, sensor grids.

### Stakeholders and Roles
- **Infrastructure Provider (InP):** Owns and manages physical cellular sites (macro, micro, pico, femto base stations $b \in B$) and compute/networking resources. Hosts the Management and Orchestration (MANO) function.
- **Mobile Virtual Network Operators (MVNOs):** Slice tenants who lease virtualized RAN resources from the InP. Each MVNO offers specific services through a slice instance $s \in S$ deployed across a subset of base stations $B_s \subseteq B$.
- **End Users:** Mobile devices generating uplink/downlink traffic aggregated at base stations.

---

## 2. Dynamic Anticipatory Slicing & Motivation

### The Resource Provisioning Trade-off
- **Static Provisioning:** Resources are assigned according to peak demands. Straightforward to configure, but leads to massive resource underutilization during off-peak hours (waste of capital expenditure).
- **Dynamic Anticipatory Provisioning:** Resources at each base station are dynamically reconfigured in advance for each time window $W$ based on forecasted traffic demand.
  - Literature benchmark: Bui & Widmer (IEEE TMC 2018) demonstrated that a perfect anticipatory traffic predictor improves RAN throughput by **35% to 40%** compared to reactive non-anticipatory baselines.

### The Privacy Dilemma
To achieve anticipatory dynamic provisioning, the forecasting mechanism requires per-slice, per-station historical traffic. Centralizing this data at the InP introduces grave privacy vulnerabilities:
1. **Private Domain Data:** Aggregated traffic patterns at macro base stations disclose MVNO commercial strategy, user volume trends, market penetration, and business revenue streams.
2. **Personal Domain Data:** Small-cell (pico/femto) base stations covering single enterprises or private households expose individual user identities, living habits, daily trajectories, and lifestyle patterns.
3. **Communication Overhead:** Transmitting high-frequency raw time series across wide-area backhaul to a central data lake incurs massive communication costs and backhaul link congestion.

---

## 3. Dimensionality Scaling & Architectural Paradigm

The paper classifies multi-slice forecasting approaches according to problem dimensions (Section IV-A, Figure 2):

| Paradigm | Dimensions | Input Tensor | Description & Trade-offs |
|---|---|---|---|
| **Centralized (e.g., DeepCog)** | 3D | $\langle S, B, T \rangle$ | Global agent centrally processes traffic across all slices, base stations, and time steps. High system complexity, heavy wide-area communication cost, complete privacy exposure. |
| **2D Federated Learning (2DFL)** | 2D | $\langle B_s, T \rangle$ | Local agents per slice instance aggregate all base stations. Eliminates slice dimension $S$, but still requires base station telemetry to be centralized per MVNO. High communication overhead. |
| **1D Federated Learning (FPLSTM)** | 1D | $\langle T \rangle$ | Local agent per $(s, b)$ pair trains strictly on local 1D sequential time series $X_b^s$. Preserves data sovereignty, minimizes per-agent compute, and leverages the InP global agent solely for model weight aggregation. |

---

## 4. System Mathematical Formalism

- **Base Stations:** $B$, heterogeneous set of cellular sites ($|B| = 57$ in Poitiers evaluation).
- **Slice Instances:** $S$, set of vertical slices ($|S| = 4$: Facebook, YouTube, Google, Instagram).
- **Time Slotted:** $t \in \mathcal{T}$ with interval duration $\Delta t = 10\text{ minutes}$.
- **Traffic Volume:** $v_b^s(t)$, traffic volume at station $b$ for slice $s$ at time interval $t$.
- **Observation Window ($H$):**
  $$X_b^s(t, H) = \{v_b^s(t - H), v_b^s(t - H + 1), \dots, v_b^s(t - 1)\}$$
- **Prediction Window ($W$):**
  $$\tilde{X}_b^s(t, W) = \{\tilde{v}_b^s(t), \tilde{v}_b^s(t + 1), \dots, \tilde{v}_b^s(t + W - 1)\}$$
- **Global Objective:**
  $$\min_{w_J} GL(w_J) = \sum_{s \in S} \sum_{b \in B_s} \rho_b^s LL(w_{J_b^s})$$
  where $\rho_b^s = \frac{a_b^s}{\sum_{s' \in S} \sum_{b' \in B_{s'}} a_{b'}^{s'}}$ represents relative sample weighting.

