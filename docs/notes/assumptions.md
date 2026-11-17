# Comprehensive Inventory of Paper Assumptions

This document identifies, categorizes, and critically evaluates the implicit and explicit assumptions underpinning Phyu et al. (*IEEE TNSM* 2023).

---

## 1. Architectural & Slicing Assumptions

1. **One-to-One Slice-to-Application Mapping (`[PAPER]` Section VI-A):**
   * *Assumption:* Slices correspond directly to single consumer mobile applications (Facebook, YouTube, Google, Instagram).
   * *Critical Reality (`[OUR ANALYSIS]`):* 3GPP and ITU-T standards define network slices along broad QoS classes (eMBB, URLLC, mMTC) or specific enterprise verticals (e.g., smart healthcare, smart grid), rather than single third-party applications. Mapping individual apps to slices simplifies evaluation but ignores multi-tenant traffic multiplexing inside slices.

2. **Full Geographic Overlap (`[PAPER]` Section VI-A):**
   * *Assumption:* All 4 slices are deployed across all 57 base stations ($B_s = B, \forall s \in S$).
   * *Critical Reality (`[OUR ANALYSIS]`):* In realistic commercial deployments, specialized slices (e.g., connected vehicle URLLC along highways, industrial automation in factory districts) are deployed only on targeted subsets of base stations ($B_s \subset B$), creating heterogeneous client topology.

3. **Isolated Protocol Stacks & Perfect Virtualization (`[PAPER]` Section III-A):**
   * *Assumption:* Each slice operates an isolated virtual protocol stack with zero inter-slice interference in processing or buffer queuing at the base station.
   * *Critical Reality (`[OUR ANALYSIS]`):* In real Open RAN / vRAN deployments, physical CPU, memory bandwidth, and fronthaul/midhaul links are shared, introducing subtle resource contention.

---

## 2. Machine Learning & Federated Learning Assumptions

4. **Synchronous Round Completion (`[PAPER]` Section IV-C):**
   * *Assumption:* All selected local agents finish $E=200$ local epochs and transmit model updates simultaneously without dropouts or latency variance.
   * *Critical Reality (`[OUR ANALYSIS]`):* In heterogeneous RANs with varying base station CPU architectures (macro vs. pico/femto cells), stragglers delay synchronous aggregation rounds. Asynchronous federated schemes or deadlines are required.

5. **Local Station Compute Capacity (`[PAPER]` Section IV-C):**
   * *Assumption:* Every base station (including pico and femto small cells) possesses sufficient compute to train an LSTM model over 200 epochs per communication round across multiple slices.
   * *Critical Reality (`[OUR ANALYSIS]`):* Small cells often have microcontrollers or low-power embedded DSPs incapable of local deep learning backpropagation.

6. **Gradient Leakage Immunity (`[PAPER]` Section I, IV):**
   * *Assumption:* Exchanging neural network model weights $w_{J_b^s}$ completely guarantees data privacy because no raw samples are transmitted.
   * *Critical Reality (`[OUR ANALYSIS]`):* Modern privacy attacks (e.g., Deep Leakage from Gradients, model inversion, membership inference) can reconstruct private training sequences from shared weights in small models unless formal Differential Privacy (DP) or Secure Aggregation (SecAgg) is enforced.

7. **Traffic Trend Privacy in Clustering (`[PAPER]` Section V-B, Alg 3):**
   * *Assumption:* Sharing average traffic trends for PCA and $k$-means clustering does not leak private domain information.
   * *Critical Reality (`[OUR ANALYSIS]`):* In small cells, temporal traffic trend shapes expose peak occupancy hours and behavioral patterns to the clustering entity.

---

## 3. Data & Evaluation Assumptions

8. **Stationarity of User Habits (`[PAPER]` Section VI-A):**
   * *Assumption:* A 10-day dataset (8 days train, 2 days test) captures representative traffic dynamics.
   * *Critical Reality (`[OUR ANALYSIS]`):* Fails to capture seasonal effects, monthly billing cycles, holidays, or macro shifts in user mobility.

9. **Symmetric Loss Optimization for RAN Slicing (`[PAPER]` Section IV-B, VI-C):**
   * *Assumption:* Minimizing symmetric RMSE/MSE is the appropriate objective for network capacity forecasting.
   * *Critical Reality (`[OUR ANALYSIS]`):* In cellular networking, under-provisioning triggers severe SLA violations and packet drops, whereas over-provisioning merely results in slight energy or resource inefficiency. An asymmetric loss (e.g., pinball/quantile loss) is technically superior.

