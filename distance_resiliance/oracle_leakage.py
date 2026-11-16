"""Oracle cosine threshold on the synthetic dataset: the one reproducing the reference leakage.

Reference split: relag on the global-identity HNSW `.edges()` graph at TAU_REF,
CPM gamma from the PeptideAtlas null model (as `com_sensibility.gamma_leakage`).
Alternative splits: relag on the ESM-C cosine HNSW graph filtered at each tau'
of TAU_GRID, CPM gamma(tau') from the cosine null of
`distance_resiliance.validate_matching`.

Every split (N_SPLIT_SEEDS seeds) is scored with the reference leakage: % of test
samples with a train sample at identity distance below TAU_REF. Also records the
test size and the number of communities.

Usage:
    uv run python -m distance_resiliance.oracle_leakage
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rich import print

from relag.core import HNSWState, INWeightType, LeidenObjective, find_communities, partition
from relag.kernels import KernelVariant

from com_sensibility.gamma_leakage import load_synthetic
from distance_resiliance.validate_matching import (TAU_REF, cosine_null, load_datasets,
                                                    reference_pairs)
from src.cache import CacheStore
from src.datasets import DATASETS
from src.metrics import null_model_cdf

TAU_GRID = np.round(np.linspace(0.004, 0.030, 27), 4).tolist()
N_SPLIT_SEEDS = 5
TEST_RATIO = 0.2

RESULTS_PATH = Path(__file__).parent.parent / "results" / "distance_resiliance" / "oracle_leakage.json"


def leakage(ref: np.ndarray, n: int, train_idx, test_idx) -> float:
    """% of test samples with a train sample at reference distance below TAU_REF."""
    i, j = ref // n, ref % n
    is_train, is_test = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    is_train[list(train_idx)], is_test[list(test_idx)] = True, True
    leaking = np.union1d(i[is_test[i] & is_train[j]], j[is_test[j] & is_train[i]])
    return 100.0 * len(leaking) / len(test_idx)


def split_stats(ref: np.ndarray, n: int, graph, gamma: float) -> dict:
    """Leakage and test size of N_SPLIT_SEEDS relag splits of one graph."""
    communities = find_communities(graph, gamma=gamma, objective=LeidenObjective.CPM)
    splits = [partition(communities, graph, test_ratio=TEST_RATIO, seed=s, post_filtering=False)
              for s in range(N_SPLIT_SEEDS)]
    return {"gamma": gamma, "n_communities": len(set(communities)),
            "leakage[%]": [leakage(ref, n, tr, te) for tr, te in splits],
            "n_test": [len(te) for _, te in splits]}


def show(label: str, r: dict) -> None:
    print(f"  {label:<14} communities={r['n_communities']:,}  "
          f"leakage={np.mean(r['leakage[%]']):.2f} ± {np.std(r['leakage[%]'], ddof=1):.2f}%  "
          f"test={int(np.mean(r['n_test']))}")


def main() -> None:
    cache = CacheStore()
    print("[bold][orange2]=== Oracle cosine threshold (synthetic) ===[/][/]")
    (annotated, production), embs = load_datasets(cache)
    sequences, ref_gamma = load_synthetic(cache)
    assert sequences == annotated.sequences
    n = len(sequences)
    ref = reference_pairs(sequences)

    cfg = DATASETS["peptide_atlas"]
    hnsw = HNSWState(cfg.modality, sequences, proximity_threshold=TAU_REF, **cfg.kernel_params)
    hnsw.build(progress=True)
    reference = split_stats(ref, n, hnsw.edges().graph(inweight_type=INWeightType.Distance), ref_gamma)
    show(f"identity {TAU_REF}", reference)

    gamma_cdf = null_model_cdf(cosine_null(embs[1], production.families))
    hnsw = HNSWState(KernelVariant.Cosine, list(embs[0]), proximity_threshold=max(TAU_GRID))
    hnsw.build(progress=True)
    edges = hnsw.edges()
    dist = np.fromiter((e[2] for e in edges), dtype=np.float32, count=len(edges))
    rows = []
    for tau in TAU_GRID:
        graph = edges[dist <= tau].graph(inweight_type=INWeightType.SimilarityComplement)
        rows.append({"tau": tau, **split_stats(ref, n, graph, float(gamma_cdf(np.array([tau]))[0]))})
        show(f"cosine {tau:.4f}", rows[-1])
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps({
            "settings": {"tau_ref": TAU_REF, "n_split_seeds": N_SPLIT_SEEDS, "test_ratio": TEST_RATIO},
            "reference": reference, "cosine": rows}))
    print(f"  Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
