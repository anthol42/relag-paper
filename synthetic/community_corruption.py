"""How does a relag split degrade when its communities are wrong?

For each repeat (the annotated / production pair of `synthetic.compare_splits`),
the CPM Leiden communities of the HNSW graph at TAU are corrupted by a chain of
random events. Each event is a split when there are fewer communities than at the
start, a merge when there are more, and either with probability 1/2 otherwise, so
the number of communities stays within one of the original:

  split  a community (drawn proportionally to its size) is cut into two parts of
         random members, at a uniform random size;
  merge  two communities (drawn uniformly) are joined.

The chain is cumulative: the partition at N_EVENTS[k] continues from the one at
N_EVENTS[k - 1]. At each level, the corrupted communities are split with relag
`partition`, and leakage (% of test samples with a train sample closer than TAU,
looked up in the exact edges below TAU, computed once per repeat) and test /
production PCC are recorded. A random split is scored once
per repeat as the reference the corrupted splits should converge to.

Usage:
    uv run python -m synthetic.community_corruption
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from rich import print

from relag.core import (HNSWState, INWeightType, LeidenObjective, exact_edges,
                        find_communities)

from src.cache import CacheStore
from src.datasets import DATASETS, load_dataset
from src.metrics import null_model
from synthetic.compare_splits import (DATASET_KEY, N_NULL_PAIRS, SIGMA, TAU, dataset_seeds,
                                      embed, random_split, relag_split, score_model,
                                      split_val)
from synthetic.dataset import SyntheticConfig, build_dataset
from synthetic.generation import atlas_profile

N_EVENTS = [0, 1, 10, 100, 1_000, 10_000, 100_000]
N_REPEATS = 3

RESULTS_PATH = Path(__file__).parent.parent / "results" / "synthetic" / "community_corruption.json"


class Corruptor:
    """Communities as member lists, corrupted in place by random split / merge events."""

    def __init__(self, labels: list[int], seed: int):
        self.rng = np.random.default_rng(seed)
        self.members: dict[int, list[int]] = {}
        for node, label in enumerate(labels):
            self.members.setdefault(label, []).append(node)
        self.community_of = np.asarray(labels)
        self.ids = list(self.members)
        self.next_id = max(self.ids) + 1
        self.target = len(self.ids)
        self.counts = {"split": 0, "merge": 0}

    def _remove(self, community: int) -> list[int]:
        position = self.ids.index(community)
        self.ids[position] = self.ids[-1]
        self.ids.pop()
        return self.members.pop(community)

    def _add(self, nodes: list[int]) -> None:
        self.members[self.next_id] = nodes
        self.community_of[nodes] = self.next_id
        self.ids.append(self.next_id)
        self.next_id += 1

    def split(self) -> bool:
        community = int(self.community_of[self.rng.integers(len(self.community_of))])
        if len(self.members[community]) < 2:
            return False
        nodes = self._remove(community)
        self.rng.shuffle(nodes)
        cut = int(self.rng.integers(1, len(nodes)))
        self._add(nodes[:cut])
        self._add(nodes[cut:])
        return True

    def merge(self) -> bool:
        if len(self.ids) < 2:
            return False
        a, b = self.rng.choice(len(self.ids), size=2, replace=False)
        first, second = self.ids[a], self.ids[b]
        self._add(self._remove(first) + self._remove(second))
        return True

    def run(self, n_events: int) -> None:
        """Apply n_events successful events."""
        done = 0
        while done < n_events:
            if len(self.ids) == self.target:
                kind = "split" if self.rng.random() < 0.5 else "merge"
            else:
                kind = "split" if len(self.ids) < self.target else "merge"
            if getattr(self, kind)():
                self.counts[kind] += 1
                done += 1

    def labels(self) -> list[int]:
        _, labels = np.unique(self.community_of, return_inverse=True)
        return labels.tolist()


def close_pairs(sequences: list[str]) -> np.ndarray:
    """Exact (i, j) pairs at distance below TAU, both orientations, shape (n_pairs, 2)."""
    cfg = DATASETS[DATASET_KEY]
    store = exact_edges(cfg.modality, sequences, proximity_threshold=TAU, **cfg.kernel_params)
    edges = np.array(store.edges(), dtype=float).reshape(-1, 3)
    pairs = edges[edges[:, 2] < TAU, :2].astype(np.int64)
    return np.concatenate([pairs, pairs[:, ::-1]])


def evaluate_split(pairs: np.ndarray, embeddings: torch.Tensor, labels: np.ndarray,
                   train_idx: list[int], test_idx: list[int], production_idx: list[int],
                   seed: int) -> dict:
    """Leakage and test / production PCC of one train / test split."""
    n = len(labels)
    is_train, is_test = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    is_train[train_idx], is_test[test_idx] = True, True
    leaking = np.unique(pairs[is_test[pairs[:, 0]] & is_train[pairs[:, 1]], 0])
    inner_train_idx, val_idx = split_val("relag", train_idx, seed, {})
    scores = score_model(embeddings, labels, inner_train_idx, val_idx, test_idx,
                         production_idx, seed)
    return {"n_train": len(train_idx), "n_test": len(test_idx),
            "leakage[%]": 100.0 * len(leaking) / len(test_idx), **scores}


def run_repeat(repeat: int, gamma: float, profile, cache: CacheStore) -> dict:
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
    pairs = close_pairs(annotated.sequences)
    corruptor = Corruptor(find_communities(graph, gamma=gamma, objective=LeidenObjective.CPM),
                          seed=repeat)

    levels, applied = [], 0
    for n_events in N_EVENTS:
        corruptor.run(n_events - applied)
        applied = n_events
        communities = corruptor.labels()
        try:
            train_idx, test_idx = relag_split(n, repeat, communities=communities, graph=graph)
            result = evaluate_split(pairs, embeddings, labels, train_idx,
                                    test_idx, production_idx, repeat)
        except Exception as error:
            result = {"error": f"{type(error).__name__}: {error}"}
        levels.append({"n_events": n_events, **dict(corruptor.counts),
                       "n_communities": len(set(communities)), **result})
        print(f"    events={n_events:>7,}  communities={levels[-1]['n_communities']:>6,}  " + (
            f"leakage={result['leakage[%]']:5.1f}%  test_pcc={result['test_pcc']:.3f}  "
            f"prod_pcc={result['production_pcc']:.3f}" if "error" not in result
            else f"[red]{result['error']}[/]"))

    train_idx, test_idx = random_split(n, repeat)
    return {"repeat": repeat, "annotated_seed": annotated_seed,
            "production_seed": production_seed, "relag": levels,
            "random": evaluate_split(pairs, embeddings, labels, train_idx,
                                     test_idx, production_idx, repeat)}


def main() -> None:
    cache = CacheStore()
    profile = atlas_profile(cache, len_range=SyntheticConfig().len_range)

    print("[bold][orange2]=== Community corruption on the synthetic forest ===[/][/]")
    atlas_sequences, _ = load_dataset(DATASET_KEY, cache)
    null_cfg = dataclasses.replace(DATASETS[DATASET_KEY], proximity_threshold=TAU)
    gamma = null_model(atlas_sequences, null_cfg, n_samples=N_NULL_PAIRS)
    print(f"  CPM gamma(tau={TAU}) = {gamma:.3e}")

    records = []
    for repeat in range(N_REPEATS):
        records.append(run_repeat(repeat, gamma, profile, cache))
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps({
            "settings": {"tau": TAU, "gamma": gamma, "n_events": N_EVENTS,
                         "sigma": SIGMA, "n_repeats": N_REPEATS},
            "records": records}))
    print(f"  Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
