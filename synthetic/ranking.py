"""Do relag test sets rank models better than random ones?

Reviewer question: a held-out set is used to *select* models. If it ranks
candidate models in the same order as their real-world (production) performance,
it is a good selection set. Here the ground truth is the ranking on the
production dataset -- an independent draw of the synthetic generative process,
so every family in it is unseen.

For each repeat and each split method (random, relag):
  1. split the annotated dataset into train / test (no validation set: no early stopping);
  2. train ONE MLP on ESM C embeddings of train, with no early stopping, and
     snapshot it at four checkpoints:
         32 steps   (barely trained, close to random)
         1 epoch
         10 epochs
         100 epochs (strongly overfit)
  3. score every checkpoint (PCC) on test and production, and rank the
     checkpoints on each set.

A split's test ranking is good if it matches the production ranking
of the same models; agreement is summarised by Kendall's tau.

Usage:
    uv run python -m synthetic.ranking
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rich import print
from scipy.stats import kendalltau, pearsonr
from torch.utils.data import DataLoader, TensorDataset

from src.cache import CacheStore
from src.datasets import DATASETS, load_dataset
from src.metrics import null_model
from src.mlp import MLP
from synthetic.compare_splits import (DATASET_KEY, N_NULL_PAIRS, SIGMA, TAU,
                                      build_context, dataset_seeds, embed, random_split,
                                      relag_split)
from synthetic.dataset import SyntheticConfig, build_dataset
from synthetic.generation import atlas_profile

# ─────────────────────────────── experiment knobs ─────────────────────────────

ENCODER = "esmc"
METHODS = {"random": random_split, "relag": relag_split}
# compare_splits still keys the relag context under its legacy name.
CONTEXT_KEY = {"random": "random", "relag": "refnd"}
N_REPEATS = 10
LEARNING_RATE = 5e-4
BATCH_SIZE = 64
CHECKPOINT_STEPS = 32
CHECKPOINT_EPOCHS = [1, 10, 100]
MODEL_NAMES = ["32_steps", "1_epoch", "10_epochs", "100_epochs"]

RESULTS_PATH = (Path(__file__).parent.parent / "results" / "synthetic"
                / f"ranking_tau{TAU:g}.json")


# ──────────────────────────────── training ───────────────────────────────────

def train_checkpoints(embeddings: torch.Tensor, labels: np.ndarray,
                      train_idx: list[int], seed: int) -> dict[str, nn.Module]:
    """One training run; snapshots after 32 steps and after 1, 10, 100 epochs."""
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = embeddings[train_idx].float().to(device)
    y = torch.tensor(labels[train_idx], dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    if len(loader) < CHECKPOINT_STEPS:
        raise ValueError(f"one epoch is {len(loader)} steps, fewer than "
                         f"{CHECKPOINT_STEPS}: checkpoints would not be ordered")

    model = MLP(x.shape[1], 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    snapshots: dict[str, nn.Module] = {}
    step = 0
    for epoch in range(1, max(CHECKPOINT_EPOCHS) + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb).squeeze(-1), yb).backward()
            optimizer.step()
            step += 1
            if step == CHECKPOINT_STEPS:
                snapshots["32_steps"] = copy.deepcopy(model).eval()
        if epoch in CHECKPOINT_EPOCHS:
            name = "1_epoch" if epoch == 1 else f"{epoch}_epochs"
            snapshots[name] = copy.deepcopy(model).eval()
    return {name: snapshots[name] for name in MODEL_NAMES}


def score(model: nn.Module, embeddings: torch.Tensor, labels: np.ndarray,
          idx: list[int]) -> float:
    device = next(model.parameters()).device
    with torch.no_grad():
        predictions = model(embeddings[idx].float().to(device)).squeeze(-1).cpu().numpy()
    return float(pearsonr(predictions, labels[idx])[0])


def ranking(scores: dict[str, float]) -> list[str]:
    """Model names, best (highest PCC) first."""
    return sorted(scores, key=scores.get, reverse=True)


def agreement(scores: dict[str, float], truth: dict[str, float]) -> float:
    names = list(truth)
    return float(kendalltau([scores[n] for n in names], [truth[n] for n in names])[0])


# ──────────────────────────────────── main ───────────────────────────────────

def run_repeat(repeat: int, gamma: float, profile, cache: CacheStore) -> list[dict]:
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
    production_idx = list(range(len(annotated), len(annotated) + len(production)))
    labels = np.concatenate([annotated.labels, production.labels])

    records = []
    for method, split_fn in METHODS.items():
        context = build_context(CONTEXT_KEY[method], annotated.sequences, annotated.families, gamma)
        train_idx, test_idx = split_fn(len(annotated), repeat, **context)
        models = train_checkpoints(embeddings, labels, train_idx, repeat)

        scores = {subset: {m: score(model, embeddings, labels, idx)
                           for m, model in models.items()}
                  for subset, idx in [("test", test_idx), ("production", production_idx)]}
        record = {
            "method": method, "repeat": repeat,
            "annotated_seed": annotated_seed, "production_seed": production_seed,
            "n_train": len(train_idx), "n_test": len(test_idx),
            "scores": scores,
            "ranking": {subset: ranking(s) for subset, s in scores.items()},
            "kendall_tau_vs_production": agreement(scores["test"], scores["production"]),
        }
        records.append(record)
        print(f"    {method:<7} test={record['ranking']['test']}  "
              f"prod={record['ranking']['production']}  "
              f"tau={record['kendall_tau_vs_production']:+.2f}")
    return records


def save(records: list[dict]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config = SyntheticConfig()
    summary = {}
    for method in METHODS:
        rows = [r for r in records if r["method"] == method]
        summary[method] = {
            "mean_kendall_tau": float(np.mean(
                [r["kendall_tau_vs_production"] for r in rows])),
            "exact_match_rate": float(np.mean(
                [r["ranking"]["test"] == r["ranking"]["production"] for r in rows])),
            "top1_match_rate": float(np.mean(
                [r["ranking"]["test"][0] == r["ranking"]["production"][0] for r in rows])),
        }
    RESULTS_PATH.write_text(json.dumps({
        "settings": {"tau": TAU, "sigma": SIGMA, "encoder": ENCODER,
                     "n_peptides": config.n_peptides, "n_families": config.n_families,
                     "methods": list(METHODS), "models": MODEL_NAMES,
                     "lr": LEARNING_RATE, "batch_size": BATCH_SIZE,
                     "n_repeats": N_REPEATS},
        "summary": summary,
        "records": records,
    }, indent=1))


def main() -> None:
    cache = CacheStore()
    profile = atlas_profile(cache, len_range=SyntheticConfig().len_range)

    print("[bold][orange2]=== Model ranking: random vs relag held-out sets ===[/][/]")
    atlas_sequences, _ = load_dataset(DATASET_KEY, cache)
    null_cfg = dataclasses.replace(DATASETS[DATASET_KEY], proximity_threshold=TAU)
    gamma = null_model(atlas_sequences, null_cfg, n_samples=N_NULL_PAIRS)
    print(f"  gamma(tau={TAU}) = {gamma:.3e}")

    records: list[dict] = []
    for repeat in range(N_REPEATS):
        records.extend(run_repeat(repeat, gamma, profile, cache))
        save(records)
    print(f"  Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
