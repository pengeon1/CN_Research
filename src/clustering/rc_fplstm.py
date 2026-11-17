"""
Random Clustering FPLSTM (RC-FPLSTM).
Implements both:
1. 'paper_buggy': Literal Algorithm 4 from Phyu et al. (IEEE TNSM 2023)
   with the shrinking threshold bug (Lines 3 & 6).
2. 'fixed': Corrected balanced random clustering.
"""

from typing import List, Dict, Any, Optional
import numpy as np


def rc_fplstm_paper_buggy(
    base_station_ids: List[int],
    k: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Literal implementation of Algorithm 4 from the paper:
    Notice Line 3: while |G_k| < |B_s| / K do
    and Line 6: B_s = B_s \ {b}
    Because B_s shrinks dynamically inside the loop, the threshold
    |B_s| / K decreases with every removed element, causing early loop exit
    and dumping the vast majority of base stations into the surplus branch!
    """
    rng = np.random.RandomState(random_state)
    b_s = list(base_station_ids)
    g: Dict[int, List[int]] = {i: [] for i in range(k)}

    # Lines 2-8
    for i in range(k):
        # In the paper: while |G_k| < |B_s| / K
        while len(b_s) > 0 and len(g[i]) < (len(b_s) / k):
            idx = rng.randint(0, len(b_s))
            b = b_s.pop(idx)
            g[i].append(b)

    # Lines 9-14: Surplus handling
    surplus_count = len(b_s)
    if len(b_s) > 0:
        for b in list(b_s):
            chosen_k = rng.randint(0, k)
            g[chosen_k].append(b)
        b_s.clear()

    cluster_sizes = {i: len(members) for i, members in g.items()}
    return {
        "variant": "paper_buggy",
        "num_clusters": k,
        "clusters": g,
        "cluster_sizes": cluster_sizes,
        "surplus_dumped": surplus_count,
    }


def rc_fplstm_fixed(
    base_station_ids: List[int],
    k: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Corrected balanced random clustering:
    Pre-computes invariant target cluster size: floor(|B_s| / K),
    and distributes remainder base stations uniformly.
    """
    rng = np.random.RandomState(random_state)
    shuffled = list(base_station_ids)
    rng.shuffle(shuffled)

    g: Dict[int, List[int]] = {i: [] for i in range(k)}
    target_size = len(shuffled) // k

    for i in range(k):
        g[i] = shuffled[i * target_size : (i + 1) * target_size]

    # Distribute surplus round-robin
    surplus = shuffled[k * target_size :]
    for idx, b in enumerate(surplus):
        g[idx % k].append(b)

    cluster_sizes = {i: len(members) for i, members in g.items()}
    return {
        "variant": "fixed",
        "num_clusters": k,
        "clusters": g,
        "cluster_sizes": cluster_sizes,
        "surplus_dumped": len(surplus),
    }

