"""
Information-based Clustering FPLSTM (IC-FPLSTM).
Implements Algorithm 3 from Phyu et al. (IEEE TNSM 2023):
- PCA dimensionality reduction to 2 components for traffic trends
- Elbow method for finding optimal number of clusters K
- k-means clustering on traffic trends or geographical coordinates
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def find_elbow_k(
    data: np.ndarray,
    k_min: int = 1,
    k_max: int = 8,
    random_state: int = 42,
) -> Tuple[int, List[float]]:
    """
    Determines optimal K via the Elbow method on KMeans inertia curve.
    Uses maximum orthogonal distance to the chord connecting first and last points.
    """
    k_max = min(k_max, len(data))
    k_values = list(range(k_min, k_max + 1))
    inertias = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(data)
        inertias.append(float(kmeans.inertia_))

    if len(k_values) <= 2:
        return k_values[0], inertias

    # Geometric elbow detection (maximum distance from line between endpoints)
    p1 = np.array([k_values[0], inertias[0]])
    p2 = np.array([k_values[-1], inertias[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    distances = []
    for k, inert in zip(k_values, inertias):
        p = np.array([k, inert])
        dist = np.abs(np.cross(line_vec, p - p1)) / (line_len + 1e-9)
        distances.append(dist)

    best_idx = int(np.argmax(distances))
    best_k = k_values[best_idx]
    return max(2, best_k), inertias


def ic_fplstm_cluster(
    feature_matrix: np.ndarray,
    trend: bool = True,
    pca_components: int = 2,
    num_clusters: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes IC-FPLSTM clustering (Algorithm 3):
    feature_matrix: shape (num_stations, features)
      - if trend=True: features are historical traffic trends
      - if trend=False: features are (latitude, longitude) coordinates
    """
    data = feature_matrix.copy()

    # Step 1: PCA dimensionality reduction if traffic trend
    pca_model = None
    if trend:
        pca_model = PCA(n_components=pca_components, random_state=random_state)
        data = pca_model.fit_transform(data)

    # Step 2: Determine K via Elbow method if not pre-specified
    if num_clusters is None:
        k_opt, inertias = find_elbow_k(data, random_state=random_state)
    else:
        k_opt = num_clusters
        inertias = []

    # Step 3: k-means clustering
    kmeans = KMeans(n_clusters=k_opt, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(data)

    # Construct clusters dictionary: {cluster_id: [station_indices]}
    clusters: Dict[int, List[int]] = {k: [] for k in range(k_opt)}
    for station_idx, clabel in enumerate(cluster_labels):
        clusters[int(clabel)].append(station_idx)

    return {
        "num_clusters": k_opt,
        "cluster_labels": cluster_labels.tolist(),
        "clusters": clusters,
        "inertias": inertias,
        "reduced_data": data.tolist(),
    }
