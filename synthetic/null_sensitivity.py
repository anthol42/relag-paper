"""How sensitive is the threshold procedure to the null model?

Reruns the sweep of `synthetic.find_threshold` (same datasets, same HNSW
layer-0 graph, same null model) with gamma(tau) multiplied by each factor in
GAMMA_SCALES, and records the KS statistic / p-value per threshold, the
KS-minimising threshold and the acceptance plateau (KS p > KS_ALPHA).

At PARTITION_TAU, the Leiden partitions of the scaled gammas are also compared
pairwise (ARI / AMI).

Usage:
    uv run python -m synthetic.null_sensitivity
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rich import print

from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from relag.core import INWeightType, LeidenObjective, find_communities

from src.cache import CacheStore
from synthetic.dataset import SyntheticConfig, build_dataset
from synthetic.find_threshold import build_hnsw, find_gamma_function, run_name
from threshold.fit import sweep_thresholds

GAMMA_SCALES = [0.01, 0.1, 1.0, 10.0, 100.0]
THRESH_LO, THRESH_HI, N_SWEEP = 0.0, 0.9, 19
KS_ALPHA = 0.05
PARTITION_TAU = 0.5
ANNOTATED_SEED, PRODUCTION_SEED = 0, 1

RESULTS_PATH = Path(__file__).parent.parent / "results" / "synthetic" / "null_sensitivity.json"


def summarise(results: list[dict]) -> dict:
    """KS-minimising threshold and the acceptance plateau of one sweep."""
    tau = np.array([r["threshold"] for r in results])
    ks = np.array([np.inf if r["ks_stat"] is None else r["ks_stat"] for r in results])
    p = np.array([0.0 if r["ks_pvalue"] is None else r["ks_pvalue"] for r in results])
    accepted = tau[p > KS_ALPHA]
    return {"ks_min_tau": float(tau[np.argmin(ks)]), "ks_min": float(ks.min()),
            "plateau": [float(accepted.min()), float(accepted.max())] if len(accepted) else None}


def main() -> None:
    config = SyntheticConfig()
    cache = CacheStore()
    print("[bold][orange2]=== Null-model sensitivity of the threshold sweep ===[/][/]")
    annotated = build_dataset(config, seed=ANNOTATED_SEED, cache=cache)
    production = build_dataset(config, seed=PRODUCTION_SEED, cache=cache)
    graph = build_hnsw(annotated.sequences, production.sequences, cache,
                       cache_key=f"threshold_sweep_{run_name(config)}")
    annotated_nodes = np.arange(len(annotated), dtype=np.int32)
    production_nodes = np.arange(len(annotated), len(annotated) + len(production), dtype=np.int32)
    gamma_cdf = find_gamma_function(production.sequences)

    sweeps = []
    for scale in GAMMA_SCALES:
        print(f"[bold]  -- gamma x {scale:g} --[/]")
        _, results = sweep_thresholds(graph, annotated_nodes, production_nodes,
                                      lambda t, s=scale: s * gamma_cdf(t),
                                      THRESH_LO, THRESH_HI, N_SWEEP,
                                      inweight_type=INWeightType.Distance)
        sweeps.append({"gamma_scale": scale, **summarise(results), "results": results})
        print(f"  KS-min tau={sweeps[-1]['ks_min_tau']:g}  plateau={sweeps[-1]['plateau']}")

    dist = np.fromiter((e[2] for e in graph), dtype=np.float32, count=len(graph))
    tau_graph = graph[dist <= PARTITION_TAU].graph(inweight_type=INWeightType.Distance)
    base_gamma = float(gamma_cdf(np.array([PARTITION_TAU]))[0])
    labels = [find_communities(tau_graph, gamma=scale * base_gamma, objective=LeidenObjective.CPM)
              for scale in GAMMA_SCALES]
    partitions = {
        "tau": PARTITION_TAU, "gamma": base_gamma,
        "n_communities": [len(set(l)) for l in labels],
        "ari": [[float(adjusted_rand_score(a, b)) for b in labels] for a in labels],
        "ami": [[float(adjusted_mutual_info_score(a, b)) for b in labels] for a in labels]}
    print(f"  tau={PARTITION_TAU}: communities {partitions['n_communities']}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "settings": {"gamma_scales": GAMMA_SCALES, "ks_alpha": KS_ALPHA,
                     "annotated_seed": ANNOTATED_SEED, "production_seed": PRODUCTION_SEED},
        "sweeps": sweeps, "partitions": partitions}))
    print(f"  Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
