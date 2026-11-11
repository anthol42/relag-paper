"""Null-model sensitivity of the threshold sweep on DBAASP.

DBAASP counterpart of `synthetic.null_sensitivity`. Reruns the sweep of
`threshold.peptides` (DBAASP vs PeptideAtlas, same cached HNSW layer-0 graph,
same null model) with gamma(tau) multiplied by each factor in GAMMA_SCALES, and
records the KS statistic / p-value per threshold, the KS-minimising threshold
and the acceptance plateau (KS p > KS_ALPHA). As in the threshold-selection
algorithm, each sweep stops at the first threshold where the KS statistic is
undefined (no pure-train or pure-production community).

At PARTITION_TAU, the Leiden partitions of the scaled gammas are also compared
pairwise (ARI / AMI), restricted to the DBAASP nodes.

Usage:
    uv run python -m com_sensibility.null_sensitivity
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rich import print
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from relag.core import INWeightType, LeidenObjective, find_communities

from src.cache import CacheStore
from src.datasets import load_dataset
from threshold.fit import sweep_thresholds
from threshold.peptides import build_hnsw, find_gamma_function

GAMMA_SCALES = [0.01, 0.1, 1.0, 10.0, 100.0]
THRESH_LO, THRESH_HI, N_SWEEP = 0.3, 0.7, 9
INWEIGHT = INWeightType.SimilarityComplement
KS_ALPHA = 0.05
PARTITION_TAU = 0.5

RESULTS_PATH = Path(__file__).parent.parent / "results" / "com_sensibility" / "null_sensitivity_dbaasp.json"


def truncate(results: list[dict]) -> list[dict]:
    """Sweep results up to (excluding) the first threshold with an undefined KS statistic."""
    for i, r in enumerate(results):
        if r["ks_stat"] is None:
            return results[:i]
    return results


def summarise(results: list[dict]) -> dict:
    """KS-minimising threshold and the acceptance plateau of one sweep."""
    tau = np.array([r["threshold"] for r in results])
    ks = np.array([r["ks_stat"] for r in results])
    p = np.array([r["ks_pvalue"] for r in results])
    accepted = tau[p > KS_ALPHA]
    return {"ks_min_tau": float(tau[np.argmin(ks)]), "ks_min": float(ks.min()),
            "plateau": [float(accepted.min()), float(accepted.max())] if len(accepted) else None}


def main() -> None:
    cache = CacheStore()
    print("[bold][orange2]=== Null-model sensitivity of the threshold sweep: DBAASP ===[/][/]")
    train, _ = load_dataset("dbaasp", cache)
    prod, _ = load_dataset("peptide_atlas", cache)
    graph = build_hnsw(train, prod, cache)
    train_nodes = np.arange(len(train), dtype=np.int32)
    prod_nodes = np.arange(len(train), len(train) + len(prod), dtype=np.int32)
    gamma_cdf = find_gamma_function(prod)

    sweeps = []
    for scale in GAMMA_SCALES:
        print(f"[bold]  -- gamma x {scale:g} --[/]")
        _, results = sweep_thresholds(graph, train_nodes, prod_nodes,
                                      lambda t, s=scale: s * gamma_cdf(t),
                                      THRESH_LO, THRESH_HI, N_SWEEP, inweight_type=INWEIGHT)
        results = truncate(results)
        sweeps.append({"gamma_scale": scale, **summarise(results), "results": results})
        print(f"  KS-min tau={sweeps[-1]['ks_min_tau']:g}  plateau={sweeps[-1]['plateau']}  "
              f"last tau={results[-1]['threshold']:g}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings = {"dataset": "dbaasp", "gamma_scales": GAMMA_SCALES, "ks_alpha": KS_ALPHA,
                "thresholds": [THRESH_LO, THRESH_HI, N_SWEEP]}
    RESULTS_PATH.write_text(json.dumps({"settings": settings, "sweeps": sweeps}))

    dist = np.fromiter((e[2] for e in graph), dtype=np.float32, count=len(graph))
    tau_graph = graph[dist <= PARTITION_TAU].graph(inweight_type=INWEIGHT)
    base_gamma = float(gamma_cdf(np.array([PARTITION_TAU]))[0])
    labels = [find_communities(tau_graph, gamma=scale * base_gamma,
                               objective=LeidenObjective.CPM)[:len(train)]
              for scale in GAMMA_SCALES]
    partitions = {
        "tau": PARTITION_TAU, "gamma": base_gamma,
        "n_communities": [len(set(l)) for l in labels],
        "ari": [[float(adjusted_rand_score(a, b)) for b in labels] for a in labels],
        "ami": [[float(adjusted_mutual_info_score(a, b)) for b in labels] for a in labels]}
    print(f"  tau={PARTITION_TAU}: communities {partitions['n_communities']}")

    RESULTS_PATH.write_text(json.dumps({"settings": settings, "sweeps": sweeps,
                                        "partitions": partitions}))
    print(f"  Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
