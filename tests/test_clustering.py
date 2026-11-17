"""
Unit tests for IC-FPLSTM and RC-FPLSTM clustering algorithms.
"""

import numpy as np
import pytest
from src.clustering.ic_fplstm import ic_fplstm_cluster, find_elbow_k
from src.clustering.rc_fplstm import rc_fplstm_paper_buggy, rc_fplstm_fixed


def test_elbow_k_detection():
    """Verify elbow detection finds a reasonable K >= 2."""
    rng = np.random.RandomState(42)
    # 3 distinct 2D clusters
    c1 = rng.normal(loc=[-5, -5], scale=0.5, size=(20, 2))
    c2 = rng.normal(loc=[5, 5], scale=0.5, size=(20, 2))
    c3 = rng.normal(loc=[0, 0], scale=0.5, size=(20, 2))
    data = np.vstack([c1, c2, c3])

    k_opt, inertias = find_elbow_k(data, k_min=1, k_max=6, random_state=42)
    assert k_opt in [2, 3, 4]
    assert len(inertias) == 6


def test_ic_fplstm_clustering():
    """Verify IC-FPLSTM with PCA reduction and k-means."""
    rng = np.random.RandomState(42)
    # 30 stations with 50 time-slot trends
    feature_matrix = rng.normal(size=(30, 50))
    result = ic_fplstm_cluster(
        feature_matrix,
        trend=True,
        pca_components=2,
        num_clusters=3,
        random_state=42,
    )

    assert result["num_clusters"] == 3
    assert len(result["cluster_labels"]) == 30
    assert len(result["clusters"]) == 3
    total_assigned = sum(len(m) for m in result["clusters"].values())
    assert total_assigned == 30


def test_rc_fplstm_buggy_vs_fixed():
    """
    Demonstrates and asserts the Algorithm 4 shrinking-threshold bug:
    In paper_buggy, the while loop condition shrinks dynamically,
    leaving a large surplus of base stations to be dumped randomly.
    In fixed, all base stations are partitioned in a balanced manner.
    """
    station_ids = list(range(57))  # 57 base stations matching paper
    k = 3

    buggy_res = rc_fplstm_paper_buggy(station_ids, k=k, random_state=42)
    fixed_res = rc_fplstm_fixed(station_ids, k=k, random_state=42)

    # In buggy version, the while loop exits prematurely, creating a massive surplus
    assert buggy_res["surplus_dumped"] > 10, "Buggy version should produce a large surplus"

    # In fixed version, surplus is strictly 57 % 3 = 0
    assert fixed_res["surplus_dumped"] == 0
    for size in fixed_res["cluster_sizes"].values():
        assert size == 19, "Fixed RC-FPLSTM should produce exactly 19 stations per cluster"

