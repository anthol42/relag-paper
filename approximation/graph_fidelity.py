"""How faithful is the HNSW proximity graph to the exact one, and does it matter?

For each dataset (at the comparison benchmark's thresholds and null-model gammas),
build the EXACT graph once and N_HNSW HNSW graphs (builds are not seedable, so each
differs), then run Leiden once on every graph. Metrics:

  recall              fraction of exact edges found by each HNSW build;
  agreement           ARI / AMI of each HNSW + Leiden partition vs exact + Leiden;
  split communities   exact communities whose members end up in more than one
                      community of an HNSW partition;
  leakage             80/20 splits (relag `partition`) of the exact and HNSW
                      partitions, and random splits: for each test sample, the exact
                      distance to its nearest train sample.

Usage:
    uv run python -m approximation.graph_fidelity
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rich import print
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from relag import exact_nearest_neighbors
from relag.core import (HNSWState, INWeightType, LeidenObjective, exact_edges,
                        find_communities, partition)

from distribution_shift.shift_probe import gamma_for
from src.cache import CacheStore
from src.datasets import DATASETS, load_dataset

DATASETS_TO_RUN = ["caco2_wang", "pgp_broccatelli", "dbaasp"]
N_HNSW = 10
TEST_RATIO = 0.2                       # as the comparison benchmark

RESULTS_PATH = Path(__file__).parent.parent / "results" / "approximation" / "graph_fidelity.json"


def edge_keys(store, n: int) -> np.ndarray:
    """Undirected edges as unique int64 keys i * n + j, i < j."""
    edges = np.array(store.edges(), dtype=float).reshape(-1, 3)
    i, j = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    return np.unique(np.minimum(i, j) * n + np.maximum(i, j))


def n_split_communities(exact_labels: list, hnsw_labels: list) -> int:
    """Number of exact communities whose members are in more than one HNSW community."""
    exact_of_pairs = [e for e, _ in set(zip(exact_labels, hnsw_labels))]
    _, n_hnsw_per_exact = np.unique(exact_of_pairs, return_counts=True)
    return int((n_hnsw_per_exact > 1).sum())


def nearest_train_distances(name: str, items: list, train, test) -> list[float]:
    cfg = DATASETS[name]
    hits = exact_nearest_neighbors(cfg.modality, [items[i] for i in test],
                                   [items[i] for i in train], 1, threads=0,
                                   progress=False, **cfg.kernel_params)
    return [round(float(h[0][1]), 4) for h in hits]


def evaluate(name: str, cache: CacheStore) -> dict:
    """Every metric for one dataset."""
    cfg = DATASETS[name]
    items, _ = load_dataset(name, cache)
    n, threshold = len(items), cfg.proximity_threshold
    gamma = gamma_for(name, cache)
    print(f"[bold]== {name}: {n:,} samples, threshold {threshold}, gamma {gamma:.3e} ==[/]")

    def communities(graph):
        return find_communities(graph, gamma=gamma, objective=LeidenObjective.CPM)

    def split_distances(labels, graph, seed):
        train, test = partition(labels, graph, test_ratio=TEST_RATIO, seed=seed, post_filtering=False)
        return nearest_train_distances(name, items, list(train), list(test))

    exact_store = exact_edges(cfg.modality, items, proximity_threshold=threshold, **cfg.kernel_params)
    exact_graph = exact_store.graph(inweight_type=INWeightType.Distance)
    exact_keys = edge_keys(exact_store, n)
    exact_labels = communities(exact_graph)

    builds = []
    for seed in range(N_HNSW):
        hnsw = HNSWState(cfg.modality, items, proximity_threshold=threshold, **cfg.kernel_params)
        hnsw.build(progress=True)
        store = hnsw.edges()
        graph = store.graph(inweight_type=INWeightType.Distance)
        labels = communities(graph)
        builds.append({
            "recall": float(np.isin(exact_keys, edge_keys(store, n)).mean()),
            "ari": float(adjusted_rand_score(exact_labels, labels)),
            "ami": float(adjusted_mutual_info_score(exact_labels, labels)),
            "n_split_communities": n_split_communities(exact_labels, labels),
            "nn_distances": split_distances(labels, graph, seed)})
        print(f"  build {seed}: recall {builds[-1]['recall']:.4f}, ARI {builds[-1]['ari']:.4f}")

    rng = np.random.default_rng(0)
    n_test = round(TEST_RATIO * n)
    random_splits = []
    for _ in range(N_HNSW):
        order = rng.permutation(n)
        random_splits.append(nearest_train_distances(name, items, order[n_test:], order[:n_test]))

    return {
        "dataset": name, "n": n, "threshold": threshold, "gamma": gamma,
        "n_exact_communities": len(set(exact_labels)),
        "builds": builds,
        "nn_distances": {
            "hnsw": [b.pop("nn_distances") for b in builds],
            "exact": [split_distances(exact_labels, exact_graph, seed) for seed in range(N_HNSW)],
            "random": random_splits},
    }


def main() -> None:
    cache = CacheStore()
    results = []
    for name in DATASETS_TO_RUN:
        results.append(evaluate(name, cache))
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps({
            "settings": {"n_hnsw": N_HNSW, "test_ratio": TEST_RATIO, "datasets": DATASETS_TO_RUN},
            "results": results}))
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
