"""Validation of production-free threshold matching on the synthetic dataset.

Reference distance: global alignment identity at TAU_REF (the threshold
`synthetic.find_threshold` selects against production). Alternative distance:
cosine between the ESM-C embeddings of `synthetic.compare_splits`.

Production threshold (ground truth): the threshold-selection sweep of
`threshold.fit` run with the alternative distance on the combined annotated +
production HNSW layer-0 graph, gamma(tau) from a null of real cosine distances
between production peptides of different families. Reports the KS-minimising
threshold and the acceptance plateau (KS p > KS_ALPHA) over [0, the
SWEEP_HI_QUANTILE quantile of the null].

Production-free thresholds, computed on the annotated set only from exact
all-pairs distances:
  quantile  the alternative threshold with as many pairs below it as the
            reference distance has below TAU_REF;
  jaccard   the alternative threshold maximising the Jaccard index between
            its edge set and the reference edge set at TAU_REF.

Usage:
    uv run python -m distance_resiliance.validate_matching
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rich import print

from relag.core import HNSWState, INWeightType, exact_edges
from relag.kernels import KernelVariant, zip_kernel

from src.cache import CacheStore
from src.datasets import DATASETS
from src.metrics import null_model_cdf
from synthetic.compare_splits import SIGMA, dataset_seeds, embed
from synthetic.dataset import SyntheticConfig, build_dataset
from synthetic.generation import atlas_profile
from threshold.fit import sweep_thresholds

TAU_REF = 0.5
SWEEP_HI_QUANTILE = 0.05
MAX_ALT_EDGES_FACTOR = 5
N_SWEEP = 25
N_NULL_PAIRS = 2_000_000
KS_ALPHA = 0.05
INWEIGHT = INWeightType.SimilarityComplement
CHUNK = 2_000

RESULTS_PATH = Path(__file__).parent.parent / "results" / "distance_resiliance" / "validate_matching.json"


def load_datasets(cache: CacheStore):
    """Annotated and production datasets of repeat 0 with their ESM-C embeddings."""
    config = SyntheticConfig(sigma=SIGMA)
    profile = atlas_profile(cache, len_range=config.len_range)
    seeds = dataset_seeds(0)
    name = f"synthetic_{config.n_peptides}_{config.n_families}"
    sets = [build_dataset(config, seed=s, profile=profile) for s in seeds]
    embs = [embed(d.sequences, cache, f"{name}_{kind}{s}").numpy().astype(np.float32)
            for d, kind, s in zip(sets, ["annotated", "production"], seeds)]
    return sets, embs


def reference_pairs(sequences: list[str]) -> np.ndarray:
    """Exact pairs with reference distance <= TAU_REF, encoded as i * n + j with i < j."""
    cfg = DATASETS["peptide_atlas"]
    store = exact_edges(cfg.modality, sequences, proximity_threshold=TAU_REF, **cfg.kernel_params)
    edges = np.array(store.edges(), dtype=float).reshape(-1, 3)
    edges = edges[edges[:, 2] <= TAU_REF]
    i, j = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    return np.unique(np.minimum(i, j) * len(sequences) + np.maximum(i, j))


def cosine_pairs(vectors: np.ndarray, max_pairs: int, max_dist: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """The max_pairs closest exact pairs by cosine distance (<= max_dist), encoded as
    i * n + j with i < j, sorted by distance."""
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    pairs, dists = [], []
    for start in range(0, len(unit), CHUNK):
        d = 1.0 - unit[start:start + CHUNK] @ unit.T
        i, j = np.nonzero(d <= max_dist)
        i += start
        keep = i < j
        pairs.append(i[keep].astype(np.int64) * len(unit) + j[keep])
        dists.append(d[i[keep] - start, j[keep]])
    pairs, dists = np.concatenate(pairs), np.concatenate(dists)
    order = np.argsort(dists, kind="stable")
    order = order[:max_pairs]
    return pairs[order], dists[order]


def matched_thresholds(ref: np.ndarray, pairs: np.ndarray, dists: np.ndarray) -> dict:
    """Quantile- and Jaccard-matched alternative thresholds, and the Jaccard curve."""
    in_ref = np.isin(pairs, ref)
    k = np.arange(1, len(pairs) + 1)
    intersection = np.cumsum(in_ref)
    jaccard = intersection / (len(ref) + k - intersection)
    best = int(np.argmax(jaccard))
    curve_idx = np.unique(np.linspace(0, len(pairs) - 1, 500).astype(int))
    return {"n_ref_edges": len(ref),
            "quantile_tau": float(dists[len(ref) - 1]),
            "jaccard_tau": float(dists[best]), "jaccard_max": float(jaccard[best]),
            "jaccard_at_quantile": float(jaccard[len(ref) - 1]),
            "jaccard_curve": {"tau": dists[curve_idx].tolist(), "jaccard": jaccard[curve_idx].tolist()}}


def cosine_null(vectors: np.ndarray, families: np.ndarray, seed: int = 42) -> np.ndarray:
    """Cosine distances between random production pairs of different families."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, len(vectors), size=N_NULL_PAIRS)
    b = rng.integers(0, len(vectors), size=N_NULL_PAIRS)
    same = families[a] == families[b]
    while same.any():
        b[same] = rng.integers(0, len(vectors), size=int(same.sum()))
        same = families[a] == families[b]
    scores = zip_kernel(KernelVariant.Cosine, list(vectors[a]), list(vectors[b]), n_threads=0, progress=True)
    return np.asarray(scores, dtype=np.float64)


def production_sweep(embs: list[np.ndarray], null: np.ndarray, hi: float) -> dict:
    """Threshold-selection sweep with the cosine distance, annotated vs production."""
    combined = list(embs[0]) + list(embs[1])
    hnsw = HNSWState(KernelVariant.Cosine, combined, proximity_threshold=0.0,
                     ef_construction=64, keep_all_edges=False, strict_ef=True)
    hnsw.build(progress=True)
    graph = hnsw.get_layer(0, weights=True, progress=True)
    n_ann = len(embs[0])
    train_nodes = np.arange(n_ann, dtype=np.int32)
    prod_nodes = np.arange(n_ann, n_ann + len(embs[1]), dtype=np.int32)
    _, results = sweep_thresholds(graph, train_nodes, prod_nodes, null_model_cdf(null),
                                  0.0, hi, N_SWEEP, inweight_type=INWEIGHT)
    valid = [r for r in results if r["ks_stat"] is not None]
    tau = np.array([r["threshold"] for r in valid])
    ks = np.array([r["ks_stat"] for r in valid])
    accepted = tau[np.array([r["ks_pvalue"] for r in valid]) > KS_ALPHA]
    return {"ks_min_tau": float(tau[np.argmin(ks)]), "ks_min": float(ks.min()),
            "plateau": [float(accepted.min()), float(accepted.max())] if len(accepted) else None,
            "results": results}


def main() -> None:
    cache = CacheStore()
    print("[bold][orange2]=== Threshold matching: identity -> ESM-C cosine (synthetic) ===[/][/]")
    (annotated, production), embs = load_datasets(cache)

    ref = reference_pairs(annotated.sequences)
    pairs, dists = cosine_pairs(embs[0], max_pairs=MAX_ALT_EDGES_FACTOR * len(ref))
    matching = matched_thresholds(ref, pairs, dists)
    print(f"  reference edges={matching['n_ref_edges']:,}  quantile tau'={matching['quantile_tau']:.4f}  "
          f"jaccard tau'={matching['jaccard_tau']:.4f} (J={matching['jaccard_max']:.3f})")

    null = cosine_null(embs[1], production.families)
    sweep_hi = float(np.quantile(null, SWEEP_HI_QUANTILE))
    sweep = production_sweep(embs, null, sweep_hi)
    print(f"  production KS-min tau'={sweep['ks_min_tau']:.4f}  plateau={sweep['plateau']}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "settings": {"tau_ref": TAU_REF, "n_sweep": N_SWEEP, "sweep_hi": sweep_hi, "sweep_hi_quantile": SWEEP_HI_QUANTILE,
                     "ks_alpha": KS_ALPHA, "n_null_pairs": N_NULL_PAIRS},
        "matching": matching, "production": sweep}))
    print(f"  Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
