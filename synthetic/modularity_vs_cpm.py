"""Leiden objective: modularity vs CPM on the synthetic forest.

For each repeat (the annotated / production pair of `synthetic.compare_splits`),
build the HNSW graph at TAU once and partition it with Leiden twice:

  cpm         CPM, gamma from the null model (as `synthetic.compare_splits`);
  modularity  modularity, gamma = MODULARITY_GAMMA.

Each partition is split with relag `partition`, and a random split is added.
Per split: leakage (% of test samples whose nearest train sample is closer than
TAU), test / production PCC of the MLP of `synthetic.compare_splits`, and the
number of communities.

Usage:
    uv run python -m synthetic.modularity_vs_cpm
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from rich import print

from relag.core import HNSWState, INWeightType, LeidenObjective, find_communities

from src.cache import CacheStore
from src.datasets import DATASETS, load_dataset
from src.metrics import null_model
from synthetic.compare_splits import (DATASET_KEY, N_NULL_PAIRS, SIGMA, TAU, dataset_seeds,
                                      embed, nearest_train_distance, random_split,
                                      relag_split, score_model, split_val)
from synthetic.dataset import SyntheticConfig, build_dataset
from synthetic.generation import atlas_profile

N_REPEATS = 10
MODULARITY_GAMMA = 1.0

RESULTS_PATH = Path(__file__).parent.parent / "results" / "synthetic" / "modularity_vs_cpm.json"


def evaluate_split(sequences: list[str], embeddings: torch.Tensor, labels: np.ndarray,
                   train_idx: list[int], test_idx: list[int], production_idx: list[int],
                   seed: int) -> dict:
    """Leakage and test / production PCC of one train / test split."""
    distances = nearest_train_distance([sequences[i] for i in train_idx],
                                       [sequences[i] for i in test_idx])
    inner_train_idx, val_idx = split_val("relag", train_idx, seed, {})
    scores = score_model(embeddings, labels, inner_train_idx, val_idx, test_idx,
                         production_idx, seed)
    return {"n_train": len(train_idx), "n_test": len(test_idx),
            "leakage[%]": 100.0 * float((distances < TAU).mean()), **scores}


def run_repeat(repeat: int, cpm_gamma: float, profile, cache: CacheStore) -> dict:
    annotated_seed, production_seed = dataset_seeds(repeat)
    config = SyntheticConfig(sigma=SIGMA)
    annotated = build_dataset(config, seed=annotated_seed, profile=profile)
    production = build_dataset(config, seed=production_seed, profile=profile)
    print(f"[bold]  -- repeat {repeat} (datasets {annotated_seed}/{production_seed}) --[/]")

    name = f"synthetic_{config.n_peptides}_{config.n_families}"
    embeddings = torch.cat([
        embed(annotated.sequences, cache, f"{name}_annotated{annotated_seed}"),
        embed(production.sequences, cache, f"{name}_production{production_seed}"),
    ])
    n = len(annotated)
    production_idx = list(range(n, n + len(production)))
    labels = np.concatenate([annotated.labels, production.labels])

    cfg = DATASETS[DATASET_KEY]
    hnsw = HNSWState(cfg.modality, annotated.sequences, proximity_threshold=TAU,
                     **cfg.kernel_params)
    hnsw.build(progress=True)
    graph = hnsw.edges().graph(inweight_type=INWeightType.Distance)

    partitions = {
        "cpm": find_communities(graph, gamma=cpm_gamma, objective=LeidenObjective.CPM),
        "modularity": find_communities(graph, gamma=MODULARITY_GAMMA,
                                       objective=LeidenObjective.Modularity),
    }
    splits = {"random": random_split(n, repeat)}
    for objective, communities in partitions.items():
        splits[objective] = relag_split(n, repeat, communities=communities, graph=graph)

    record = {"repeat": repeat, "annotated_seed": annotated_seed,
              "production_seed": production_seed}
    for method, (train_idx, test_idx) in splits.items():
        record[method] = evaluate_split(annotated.sequences, embeddings, labels, train_idx,
                                        test_idx, production_idx, repeat)
        if method in partitions:
            record[method]["n_communities"] = len(set(partitions[method]))
        r = record[method]
        print(f"    {method:<11} leakage={r['leakage[%]']:5.1f}%  "
              f"test_pcc={r['test_pcc']:.3f}  prod_pcc={r['production_pcc']:.3f}")
    return record


def main() -> None:
    cache = CacheStore()
    profile = atlas_profile(cache, len_range=SyntheticConfig().len_range)

    print("[bold][orange2]=== Modularity vs CPM on the synthetic forest ===[/][/]")
    atlas_sequences, _ = load_dataset(DATASET_KEY, cache)
    null_cfg = dataclasses.replace(DATASETS[DATASET_KEY], proximity_threshold=TAU)
    cpm_gamma = null_model(atlas_sequences, null_cfg, n_samples=N_NULL_PAIRS)
    print(f"  CPM gamma(tau={TAU}) = {cpm_gamma:.3e}")

    records = []
    for repeat in range(N_REPEATS):
        records.append(run_repeat(repeat, cpm_gamma, profile, cache))
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps({
            "settings": {"tau": TAU, "cpm_gamma": cpm_gamma,
                         "modularity_gamma": MODULARITY_GAMMA, "sigma": SIGMA,
                         "n_repeats": N_REPEATS},
            "records": records}))
    print(f"  Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
