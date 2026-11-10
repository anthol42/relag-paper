"""Distribution shift between the two halves of a split (adversarial validation).

Does a relag split separate the data into two different distributions? For each
benchmark dataset:

  1. label split: split the dataset 50/50 with the method under test (relag, or
     random as the no-shift control); each sample's label is the half it fell in;
  2. probe split: re-split the same samples 80/20 with relag. This one is ALWAYS
     relag, for both methods: with a random probe split, near-duplicates of
     probe-test samples would sit in probe-train carrying the same half label, and
     the probe could score high by recognising them instead of detecting a shift;
  3. train a linear probe (standardised logistic regression) on frozen embeddings
     (ESM C for proteins, ChemBERTa for molecules) to predict the half label, and
     score its balanced accuracy on the probe test set.

~0.5 means the two halves come from the same distribution; higher means a shift.
The relag graph is built once per dataset; each repeat re-draws both splits.

Usage:
    uv run python -m distribution_shift.shift_probe
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from rich import print
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from relag.core import (HNSWState, INWeightType, LeidenObjective, find_communities,
                        partition)

from src.cache import CacheStore
from src.datasets import DATASETS, load_dataset
from src.embeddings import _embed_esmc, _embed_hf
from src.metrics import null_model

# ─────────────────────────────── experiment knobs ─────────────────────────────

PROTEIN_DATASETS = ["dbaasp", "enzyme_topt"]
MOLECULE_DATASETS = ["cyp2c19_veith", "caco2_wang", "pgp_broccatelli", "ames",
                     "lipophilicity", "sr_are"]
ENCODERS = {"protein": "esmc_300m",
            "molecule": "seyonec/ChemBERTa-zinc-base-v1"}
METHODS = ["relag", "random"]
N_REPEATS = 5
LABEL_TEST_RATIO = 0.5
PROBE_TEST_RATIO = 0.2

# Null models, as in the comparison benchmark (comparison/scripts/relag_split/).
N_NULL_PAIRS_PROTEIN = 10_000_000       # PeptideAtlas null
N_NULL_MOLECULES, N_NULL_PAIRS_MOLECULE = 100_000, 2_000_000   # random-atom molecule null

RESULTS_PATH = Path(__file__).parent.parent / "results" / "distribution_shift" / "shift_probe.json"


# ─────────────────────────────── per-dataset setup ────────────────────────────

def gamma_for(name: str, cache: CacheStore) -> float:
    """CPM resolution: the benchmark's null-model P(random pair within threshold)."""
    cfg = DATASETS[name]
    if name in PROTEIN_DATASETS:
        atlas, _ = load_dataset("peptide_atlas", cache)
        null_cfg = dataclasses.replace(DATASETS["peptide_atlas"],
                                       proximity_threshold=cfg.proximity_threshold)
        return null_model(atlas, null_cfg, n_samples=N_NULL_PAIRS_PROTEIN)
    path = Path(__file__).parent.parent / "comparison/scripts/relag_split/null_model.py"
    spec = importlib.util.spec_from_file_location("comparison_null_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # dataclasses look the module up here
    spec.loader.exec_module(module)
    return module.random_molecule_null_gamma(
        module.NullCfg(modality=cfg.modality, proximity_threshold=cfg.proximity_threshold),
        n_molecules=N_NULL_MOLECULES, n_pairs=N_NULL_PAIRS_MOLECULE,
        cache_path=cache.root / f"mol_null_random_n{N_NULL_MOLECULES}_pairs{N_NULL_PAIRS_MOLECULE}_seed42.npy")


def embeddings_for(name: str, items: list, cache: CacheStore) -> np.ndarray:
    """Frozen embeddings, cached; molecules are embedded from their SMILES."""
    kind = "protein" if name in PROTEIN_DATASETS else "molecule"
    key = f"{name}_{ENCODERS[kind].split('/')[-1]}"
    cached = cache.get_embs(key)
    if cached is None:
        texts = items if kind == "protein" else cache.get_dataset(f"{name}_smiles")[0]
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cached = (_embed_esmc(list(texts), device) if kind == "protein"
                  else _embed_hf(ENCODERS[kind], list(texts), device))
        cache.store_embs(key, cached)
    return cached.numpy()


def relag_graph(name: str, items: list, cache: CacheStore):
    """HNSW distance graph + CPM communities (seed-independent)."""
    cfg = DATASETS[name]
    hnsw = HNSWState(cfg.modality, items, proximity_threshold=cfg.proximity_threshold,
                     **cfg.kernel_params)
    hnsw.build(progress=True)
    graph = hnsw.edges().graph(inweight_type=INWeightType.Distance)
    communities = find_communities(graph, gamma=gamma_for(name, cache),
                                   objective=LeidenObjective.CPM)
    return graph, communities


# ──────────────────────────────────── probe ──────────────────────────────────

def split(method: str, n: int, ratio: float, seed: int, graph, communities) -> np.ndarray:
    """Boolean mask of the `ratio` side of a relag or random split."""
    mask = np.zeros(n, dtype=bool)
    if method == "relag":
        _, test_idx = partition(communities, graph, test_ratio=ratio, seed=seed,
                                post_filtering=False)
        mask[list(test_idx)] = True
    else:
        mask[np.random.default_rng(seed).permutation(n)[:int(ratio * n)]] = True
    return mask


def probe(name: str, cache: CacheStore) -> list[dict]:
    """Every (method, repeat) record for one dataset."""
    items, _ = load_dataset(name, cache)
    print(f"[bold]== {name}: {len(items):,} samples ==[/]")
    X = embeddings_for(name, items, cache)
    graph, communities = relag_graph(name, items, cache)

    records = []
    for method in METHODS:
        for repeat in range(N_REPEATS):
            half = split(method, len(items), LABEL_TEST_RATIO, repeat, graph, communities)
            test = split("relag", len(items), PROBE_TEST_RATIO, 1000 + repeat, graph, communities)
            model = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=5000, class_weight="balanced"))
            model.fit(X[~test], half[~test])
            score = balanced_accuracy_score(half[test], model.predict(X[test]))
            records.append({"dataset": name, "method": method, "repeat": repeat,
                            "balanced_accuracy": float(score),
                            "label_fraction": float(half.mean()),
                            "probe_test_fraction": float(test.mean())})
            print(f"  {method:<7} repeat {repeat}: balanced acc = {score:.3f}  "
                  f"(half = {half.mean():.2f}, probe test = {test.mean():.2f})")
    return records


def main() -> None:
    cache = CacheStore()
    records: list[dict] = []
    for name in PROTEIN_DATASETS + MOLECULE_DATASETS:
        records += probe(name, cache)
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps({
            "settings": {"methods": METHODS, "n_repeats": N_REPEATS, "encoders": ENCODERS,
                         "label_test_ratio": LABEL_TEST_RATIO,
                         "probe_test_ratio": PROBE_TEST_RATIO,
                         "thresholds": {n: DATASETS[n].proximity_threshold
                                        for n in PROTEIN_DATASETS + MOLECULE_DATASETS}},
            "records": records}, indent=1))
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
