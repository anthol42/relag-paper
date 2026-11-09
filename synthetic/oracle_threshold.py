"""Is the threshold from the procedure close to the best one?

On the synthetic forest, production performance is known, so an oracle can pick
the threshold whose relag test score best matches production. For each repeat
(the annotated / production pair of `synthetic.compare_splits`) and each
threshold in THRESHOLDS:

  1. keep the HNSW edges below the threshold and run Leiden (CPM, null-model gamma);
  2. relag split, then train and score the MLP of `synthetic.compare_splits`;
  3. record delta = test PCC - production PCC.

The oracle threshold of a repeat minimises |delta|; compare it to TAU, the
threshold given by `synthetic.find_threshold`. A random split is scored once per
repeat as a reference.

Usage:
    uv run python -m synthetic.oracle_threshold
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from rich import print

from relag.core import (EdgeStore, HNSWState, INWeightType, LeidenObjective,
                        find_communities)

from src.cache import CacheStore
from src.datasets import DATASETS, load_dataset
from src.metrics import null_model_cdf, null_model_scores
from synthetic.compare_splits import (DATASET_KEY, N_NULL_PAIRS, SIGMA, TAU, dataset_seeds,
                                      embed, random_split, relag_split, score_model,
                                      split_val)
from synthetic.dataset import SyntheticConfig, build_dataset
from synthetic.generation import atlas_profile

THRESHOLDS = np.round(np.arange(0.10, 0.801, 0.05), 2).tolist()
N_REPEATS = 10

RESULTS_PATH = Path(__file__).parent.parent / "results" / "synthetic" / "oracle_threshold.json"


def delta(embeddings: torch.Tensor, labels: np.ndarray, train_idx: list[int],
          test_idx: list[int], production_idx: list[int], seed: int) -> dict:
    """Test and production PCC of the MLP trained on train_idx, and their difference."""
    inner_train_idx, val_idx = split_val("relag", train_idx, seed, {})
    scores = score_model(embeddings, labels, inner_train_idx, val_idx, test_idx,
                         production_idx, seed)
    return {**scores, "delta": scores["test_pcc"] - scores["production_pcc"],
            "n_test": len(test_idx)}


def run_repeat(repeat: int, gamma_cdf, profile, cache: CacheStore) -> dict:
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
    hnsw = HNSWState(cfg.modality, annotated.sequences, proximity_threshold=max(THRESHOLDS),
                     **cfg.kernel_params)
    hnsw.build(progress=True)
    edges = hnsw.edges().edges()

    by_threshold = []
    for threshold in THRESHOLDS:
        graph = EdgeStore(n, [e for e in edges if e[2] < threshold]).graph(
            inweight_type=INWeightType.Distance)
        communities = find_communities(graph, gamma=float(gamma_cdf(threshold)[0]),
                                       objective=LeidenObjective.CPM)
        try:
            train_idx, test_idx = relag_split(n, repeat, communities=communities, graph=graph)
            result = delta(embeddings, labels, train_idx, test_idx, production_idx, repeat)
        except Exception as error:
            result = {"error": f"{type(error).__name__}: {error}"}
        by_threshold.append({"threshold": threshold, **result})
        print(f"    tau={threshold:.2f}  " + (f"delta={result['delta']:+.3f}"
                                             if "delta" in result else f"[red]{result['error']}[/]"))

    train_idx, test_idx = random_split(n, repeat)
    return {"repeat": repeat, "annotated_seed": annotated_seed,
            "production_seed": production_seed, "relag": by_threshold,
            "random": delta(embeddings, labels, train_idx, test_idx, production_idx, repeat)}


def main() -> None:
    cache = CacheStore()
    profile = atlas_profile(cache, len_range=SyntheticConfig().len_range)

    print("[bold][orange2]=== Oracle threshold on the synthetic forest ===[/][/]")
    atlas_sequences, _ = load_dataset(DATASET_KEY, cache)
    gamma_cdf = null_model_cdf(null_model_scores(atlas_sequences, DATASETS[DATASET_KEY],
                                                 n_samples=N_NULL_PAIRS))

    records = []
    for repeat in range(N_REPEATS):
        records.append(run_repeat(repeat, gamma_cdf, profile, cache))
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps({
            "settings": {"procedure_tau": TAU, "thresholds": THRESHOLDS, "sigma": SIGMA,
                         "n_repeats": N_REPEATS},
            "records": records}))
    print(f"  Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
