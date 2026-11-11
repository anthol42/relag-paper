"""Leiden objective on DBAASP: modularity vs CPM.

DBAASP counterpart of `synthetic.modularity_vs_cpm`. The HNSW graph at the
benchmark threshold is built once and partitioned with Leiden twice:

  cpm         CPM, gamma from the PeptideAtlas null model;
  modularity  modularity, gamma = MODULARITY_GAMMA.

For each repeat (split seed), each partition is split with relag `partition`,
and a random split is added. Per split: leakage (% of test samples with a train
sample closer than the threshold, looked up in the exact edges below it), test
PCC of the MLP of `synthetic.compare_splits` (DBAASP has no labelled production
set), and the number of communities.

Usage:
    uv run python -m com_sensibility.modularity_vs_cpm
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from rich import print

from relag.core import HNSWState, INWeightType, LeidenObjective, exact_edges, find_communities

from distribution_shift.shift_probe import embeddings_for, gamma_for
from src.cache import CacheStore
from src.datasets import DATASETS, load_dataset
from src.mlp import train_eval_mlp
from synthetic.compare_splits import (LEARNING_RATE, METRIC, pcc, random_split, relag_split,
                                      split_val)

DATASET = "dbaasp"
N_REPEATS = 10
MODULARITY_GAMMA = 1.0

RESULTS_PATH = Path(__file__).parent.parent / "results" / "com_sensibility" / "modularity_vs_cpm_dbaasp.json"


def close_pairs(items: list[str], threshold: float) -> np.ndarray:
    """Exact (i, j) pairs at distance below threshold, both orientations, shape (n_pairs, 2)."""
    cfg = DATASETS[DATASET]
    store = exact_edges(cfg.modality, items, proximity_threshold=threshold, **cfg.kernel_params)
    edges = np.array(store.edges(), dtype=float).reshape(-1, 3)
    pairs = edges[edges[:, 2] < threshold, :2].astype(np.int64)
    return np.concatenate([pairs, pairs[:, ::-1]])


def evaluate_split(pairs: np.ndarray, embeddings: torch.Tensor, labels: np.ndarray,
                   train_idx: list[int], test_idx: list[int], seed: int) -> dict:
    """Leakage and test PCC of one train / test split."""
    n = len(labels)
    is_train, is_test = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    is_train[train_idx], is_test[test_idx] = True, True
    leaking = np.unique(pairs[is_test[pairs[:, 0]] & is_train[pairs[:, 1]], 0])
    inner_train_idx, val_idx = split_val("relag", train_idx, seed, {})
    result = train_eval_mlp(embeddings, labels, inner_train_idx, val_idx, test_idx,
                            metric=METRIC, lr=LEARNING_RATE, seed=seed)
    return {"n_train": len(train_idx), "n_test": len(test_idx),
            "leakage[%]": 100.0 * len(leaking) / len(test_idx),
            "test_pcc": pcc(np.asarray(result["predictions"]), labels[test_idx])}


def main() -> None:
    cache = CacheStore()
    cfg = DATASETS[DATASET]
    items, labels = load_dataset(DATASET, cache)
    labels = np.asarray(labels, dtype=float)
    n, threshold = len(items), cfg.proximity_threshold
    print(f"[bold][orange2]=== Modularity vs CPM on {DATASET} ({n:,} samples, "
          f"threshold {threshold}) ===[/][/]")

    embeddings = torch.as_tensor(embeddings_for(DATASET, items, cache))
    cpm_gamma = gamma_for(DATASET, cache)
    print(f"  CPM gamma = {cpm_gamma:.3e}")

    hnsw = HNSWState(cfg.modality, items, proximity_threshold=threshold, **cfg.kernel_params)
    hnsw.build(progress=True)
    graph = hnsw.edges().graph(inweight_type=INWeightType.Distance)
    pairs = close_pairs(items, threshold)
    partitions = {
        "cpm": find_communities(graph, gamma=cpm_gamma, objective=LeidenObjective.CPM),
        "modularity": find_communities(graph, gamma=MODULARITY_GAMMA,
                                       objective=LeidenObjective.Modularity),
    }
    n_communities = {k: len(set(v)) for k, v in partitions.items()}
    print(f"  communities: {n_communities}")

    records = []
    for repeat in range(N_REPEATS):
        print(f"[bold]  -- repeat {repeat} --[/]")
        splits = {"random": random_split(n, repeat)}
        for objective, communities in partitions.items():
            splits[objective] = relag_split(n, repeat, communities=communities, graph=graph)
        record = {"repeat": repeat}
        for method, (train_idx, test_idx) in splits.items():
            record[method] = evaluate_split(pairs, embeddings, labels, train_idx, test_idx, repeat)
            if method in partitions:
                record[method]["n_communities"] = n_communities[method]
            r = record[method]
            print(f"    {method:<11} leakage={r['leakage[%]']:5.1f}%  test_pcc={r['test_pcc']:.3f}")
        records.append(record)
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps({
            "settings": {"dataset": DATASET, "tau": threshold, "cpm_gamma": cpm_gamma,
                         "modularity_gamma": MODULARITY_GAMMA, "n_repeats": N_REPEATS},
            "records": records}))
    print(f"  Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
