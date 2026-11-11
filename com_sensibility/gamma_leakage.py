"""Leakage of relag splits made from the Leiden partitions of scaled gammas.

For each dataset (synthetic annotated set of repeat 0, DBAASP), builds the relag
split graph (HNSW `.edges()` at PARTITION_TAU, as `com_sensibility.modularity_vs_cpm`)
and runs CPM Leiden with the null-model gamma multiplied by each factor in
GAMMA_SCALES. Each partition is split with relag `partition` for N_SPLIT_SEEDS
seeds. Per split: leakage (% of test samples with a train sample closer than
PARTITION_TAU, looked up in the exact edges below it) and test size. ARI / AMI
are computed vs gamma x 1.

Usage:
    uv run python -m com_sensibility.gamma_leakage
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
from rich import print
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from relag.core import HNSWState, INWeightType, LeidenObjective, exact_edges, find_communities, partition

from distribution_shift.shift_probe import gamma_for
from src.cache import CacheStore
from src.datasets import DATASETS, load_dataset
from src.metrics import null_model
from synthetic.compare_splits import DATASET_KEY, N_NULL_PAIRS, SIGMA, dataset_seeds
from synthetic.dataset import SyntheticConfig, build_dataset
from synthetic.generation import atlas_profile

GAMMA_SCALES = [0.01, 0.1, 1.0, 10.0, 100.0]
PARTITION_TAU = 0.5
N_SPLIT_SEEDS = 5
TEST_RATIO = 0.2

RESULTS_PATH = Path(__file__).parent.parent / "results" / "com_sensibility" / "gamma_leakage.json"


def load_synthetic(cache: CacheStore) -> tuple[list[str], float]:
    """Annotated sequences of repeat 0 and the CPM gamma of `synthetic.compare_splits`."""
    profile = atlas_profile(cache, len_range=SyntheticConfig().len_range)
    annotated = build_dataset(SyntheticConfig(sigma=SIGMA), seed=dataset_seeds(0)[0], profile=profile)
    atlas, _ = load_dataset(DATASET_KEY, cache)
    null_cfg = dataclasses.replace(DATASETS[DATASET_KEY], proximity_threshold=PARTITION_TAU)
    return annotated.sequences, null_model(atlas, null_cfg, n_samples=N_NULL_PAIRS)


def load_dbaasp(cache: CacheStore) -> tuple[list[str], float]:
    """DBAASP sequences and the CPM gamma of the comparison benchmark."""
    items, _ = load_dataset("dbaasp", cache)
    return items, gamma_for("dbaasp", cache)


LOADERS = {"Synthetic": load_synthetic, "DBAASP": load_dbaasp}


def close_pairs(items: list[str]) -> np.ndarray:
    """Exact (i, j) pairs at distance below PARTITION_TAU, both orientations, shape (n_pairs, 2)."""
    cfg = DATASETS["dbaasp"]
    store = exact_edges(cfg.modality, items, proximity_threshold=PARTITION_TAU, **cfg.kernel_params)
    edges = np.array(store.edges(), dtype=float).reshape(-1, 3)
    pairs = edges[edges[:, 2] < PARTITION_TAU, :2].astype(np.int64)
    return np.concatenate([pairs, pairs[:, ::-1]])


def leakage(pairs: np.ndarray, n: int, train_idx, test_idx) -> float:
    """% of test samples with a train sample closer than PARTITION_TAU."""
    is_train, is_test = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    is_train[list(train_idx)], is_test[list(test_idx)] = True, True
    leaking = np.unique(pairs[is_test[pairs[:, 0]] & is_train[pairs[:, 1]], 0])
    return 100.0 * len(leaking) / len(test_idx)


def evaluate(name: str, cache: CacheStore) -> dict:
    items, base_gamma = LOADERS[name](cache)
    n = len(items)
    print(f"[bold]== {name}: {n:,} samples, gamma x 1 = {base_gamma:.3e} ==[/]")
    cfg = DATASETS["dbaasp"]
    hnsw = HNSWState(cfg.modality, items, proximity_threshold=PARTITION_TAU, **cfg.kernel_params)
    hnsw.build(progress=True)
    graph = hnsw.edges().graph(inweight_type=INWeightType.Distance)
    pairs = close_pairs(items)

    labels = [find_communities(graph, gamma=scale * base_gamma, objective=LeidenObjective.CPM)
              for scale in GAMMA_SCALES]
    base = labels[GAMMA_SCALES.index(1.0)]
    rows = []
    for scale, communities in zip(GAMMA_SCALES, labels):
        splits = [partition(communities, graph, test_ratio=TEST_RATIO, seed=seed,
                            post_filtering=False) for seed in range(N_SPLIT_SEEDS)]
        rows.append({"gamma_scale": scale, "gamma": scale * base_gamma,
                     "n_communities": len(set(communities)),
                     "ari": float(adjusted_rand_score(base, communities)),
                     "ami": float(adjusted_mutual_info_score(base, communities)),
                     "leakage[%]": [leakage(pairs, n, tr, te) for tr, te in splits],
                     "n_test": [len(te) for _, te in splits]})
        r = rows[-1]
        print(f"  gamma x {scale:<5g} communities={r['n_communities']:,}  ARI={r['ari']:.3f}  "
              f"leakage={np.mean(r['leakage[%]']):.2f} ± {np.std(r['leakage[%]'], ddof=1):.2f}%  "
              f"test={r['n_test']}")
    return {"dataset": name, "n": n, "rows": rows}


def main() -> None:
    cache = CacheStore()
    results = []
    for name in LOADERS:
        results.append(evaluate(name, cache))
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps({
            "settings": {"gamma_scales": GAMMA_SCALES, "tau": PARTITION_TAU,
                         "n_split_seeds": N_SPLIT_SEEDS, "test_ratio": TEST_RATIO},
            "results": results}))
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
