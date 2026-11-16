"""Robustness of relag splits to the distance function, with an oracle threshold.

Per dataset, a reference distance (as in the benchmark) and an alternative one:

  enzyme_topt     global alignment identity  ->  ProtSpaM mismatch rate
  TDC molecules   Tanimoto on Morgan bits    ->  cosine on the same bits as vectors

Reference split: relag on the reference HNSW `.edges()` graph at the benchmark
threshold, CPM gamma from `gamma_for`. Alternative splits: relag on the
alternative HNSW graph filtered at each tau' of the dataset's grid, CPM
gamma(tau') from a null of the alternative distance (shuffled sequences for
ProtSpaM, the random-molecule fingerprints of the benchmark for cosine).

Every split (N_SPLIT_SEEDS seeds) is scored with the reference leakage: % of
test samples with a train sample at reference distance below the benchmark
threshold. The oracle tau' minimises |leakage(tau') - leakage(ref)|, with the
leakage pooled over the molecule datasets (leaking test samples / test samples
of every split), so all molecule datasets share one tau'; enzyme gets its own
(ties: smallest tau'). At the oracle: ARI / AMI vs the reference communities,
the downstream test score (MLP on the frozen embeddings) and the reference
distance from each test sample to its nearest train sample, for the reference,
oracle and random splits.

Usage:
    uv run python -m distance_resiliance.oracle_threshold [dataset ...]
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rich import print
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from relag.core import HNSWState, INWeightType, LeidenObjective, exact_edges, find_communities, partition
from relag.kernels import KernelVariant, zip_kernel
from relag.kernels.protspam import ProtSpamDistance
from relag.utils import SWPatternSet, SWSequence

from distribution_shift.shift_probe import embeddings_for, gamma_for
from src.cache import CacheStore
from src.datasets import DATASETS, load_dataset
from src.metrics import null_model_cdf
from src.mlp import train_eval_mlp
from synthetic.compare_splits import LEARNING_RATE, random_split, split_val

MOLECULES = ["caco2_wang", "pgp_broccatelli", "lipophilicity", "sr_are", "ames", "cyp2c19_veith"]
PROTEINS = ["enzyme_topt"]
GRIDS = {"cosine": np.round(np.arange(0.20, 0.701, 0.025), 3).tolist(),
         "protspam": np.round(np.arange(0.10, 0.801, 0.025), 3).tolist()}
PROTSPAM = {"n_patterns": 1, "weight": 3, "dont_care": 20,
            "significance_threshold": -1_000_000, "distance": ProtSpamDistance.MismatchRate}
N_SPLIT_SEEDS = 5
TEST_RATIO = 0.2
N_NULL_PAIRS = 1_000_000
NULL_SEED = 42

RESULTS_DIR = Path(__file__).parent.parent / "results" / "distance_resiliance"


def tanimoto_pairs(bits: np.ndarray, tau: float) -> np.ndarray:
    """Exact pairs with Tanimoto distance below tau, encoded as i * n + j with i < j."""
    x = bits.astype(np.float32)
    counts = x.sum(1)
    shared = x @ x.T
    dist = 1.0 - shared / np.maximum(counts[:, None] + counts[None, :] - shared, 1.0)
    i, j = np.nonzero(np.triu(dist < tau, k=1))
    return i.astype(np.int64) * len(bits) + j


def identity_pairs(sequences: list[str], tau: float) -> np.ndarray:
    """Exact pairs with global-identity distance below tau, encoded as i * n + j with i < j."""
    cfg = DATASETS["enzyme_topt"]
    store = exact_edges(cfg.modality, sequences, proximity_threshold=tau, **cfg.kernel_params)
    edges = np.array(store.edges(), dtype=float).reshape(-1, 3)
    edges = edges[edges[:, 2] < tau]
    i, j = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    return np.unique(np.minimum(i, j) * len(sequences) + np.maximum(i, j))


def nearest_train(name: str, items: list, cache: CacheStore, splits) -> list[list[float]]:
    """Per split: reference distance from each test sample to its nearest train sample."""
    if name in PROTEINS:
        cfg = DATASETS[name]
        store = exact_edges(cfg.modality, items, proximity_threshold=1.0, **cfg.kernel_params)
        edges = np.array(store.edges(), dtype=float).reshape(-1, 3)
        dist = np.ones((len(items), len(items)), dtype=np.float32)
        i, j = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
        dist[i, j] = dist[j, i] = edges[:, 2]
        rows = lambda test, train: dist[np.ix_(test, train)]
    else:
        x = np.asarray(cache.get_dataset(name)[0], dtype=np.float32)
        counts = x.sum(1)
        def rows(test, train):
            shared = x[test] @ x[train].T
            return 1.0 - shared / np.maximum(counts[test, None] + counts[None, train] - shared, 1.0)
    return [np.round(rows(np.asarray(te), np.asarray(tr)).min(1), 4).tolist() for tr, te in splits]


def random_molecule_bits() -> np.ndarray:
    """Morgan bits of the random-molecule null pool of the benchmark."""
    path = Path(__file__).parent.parent / "comparison/scripts/relag_split/null_model.py"
    spec = importlib.util.spec_from_file_location("comparison_null_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return np.stack([fp.to_np() for fp in module.generate_random_molecule_fps(100_000, NULL_SEED)])


def cosine_null(pool: np.ndarray) -> np.ndarray:
    """Cosine distances between random pairs of the null pool."""
    rng = np.random.default_rng(NULL_SEED)
    a, b = rng.integers(0, len(pool), N_NULL_PAIRS), rng.integers(0, len(pool), N_NULL_PAIRS)
    x = pool.astype(np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return 1.0 - np.einsum("ij,ij->i", x[a], x[b])


def protspam_null(sequences: list[str], patterns: SWPatternSet) -> np.ndarray:
    """ProtSpaM distances between random pairs of residue-shuffled sequences."""
    rng = np.random.default_rng(NULL_SEED)
    a, b = rng.integers(0, len(sequences), N_NULL_PAIRS), rng.integers(0, len(sequences), N_NULL_PAIRS)
    shuffled = ["".join(rng.permutation(list(s))) for s in sequences]
    encoded = [SWSequence(s, patterns) for s in shuffled]
    kw = {k: v for k, v in PROTSPAM.items() if k in ("significance_threshold", "distance")}
    scores = zip_kernel(KernelVariant.ProtSpam, [encoded[i] for i in a], [encoded[i] for i in b],
                        n_threads=0, progress=True, patterns=patterns, **kw)
    return np.asarray(scores, dtype=np.float64)


def alternative(name: str, items: list, cache: CacheStore) -> dict:
    """Alternative-distance HNSW inputs, kernel arguments, tau' grid and null scores."""
    if name in PROTEINS:
        patterns = SWPatternSet(PROTSPAM["n_patterns"], PROTSPAM["weight"], PROTSPAM["dont_care"])
        kw = {"patterns": patterns, "significance_threshold": PROTSPAM["significance_threshold"],
              "distance": PROTSPAM["distance"]}
        return {"label": "protspam", "variant": KernelVariant.ProtSpam,
                "data": [SWSequence(s, patterns) for s in items], "kwargs": kw,
                "grid": GRIDS["protspam"], "null": protspam_null(items, patterns)}
    bits = np.asarray(cache.get_dataset(name)[0], dtype=bool)
    return {"label": "cosine", "variant": KernelVariant.Cosine,
            "data": list(bits.astype(np.float32)), "kwargs": {},
            "grid": GRIDS["cosine"], "null": cosine_null(random_molecule_bits())}


def leakage(ref: np.ndarray, n: int, train_idx, test_idx) -> float:
    """% of test samples with a train sample at reference distance below the threshold."""
    i, j = ref // n, ref % n
    is_train, is_test = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    is_train[list(train_idx)], is_test[list(test_idx)] = True, True
    leaking = np.union1d(i[is_test[i] & is_train[j]], j[is_test[j] & is_train[i]])
    return 100.0 * len(leaking) / len(test_idx)


def splits_of(graph, gamma: float) -> tuple[list[int], list]:
    communities = find_communities(graph, gamma=gamma, objective=LeidenObjective.CPM)
    return communities, [partition(communities, graph, test_ratio=TEST_RATIO, seed=s, post_filtering=False)
                         for s in range(N_SPLIT_SEEDS)]


def stats(ref: np.ndarray, n: int, communities, splits, gamma: float) -> dict:
    return {"gamma": gamma, "n_communities": len(set(communities)),
            "leakage[%]": [leakage(ref, n, tr, te) for tr, te in splits],
            "n_test": [len(te) for _, te in splits]}


def downstream(embeddings: torch.Tensor, labels: np.ndarray, metric: str, splits) -> list[float]:
    """Test score of the MLP per split."""
    scores = []
    for seed, (train_idx, test_idx) in enumerate(splits):
        inner, val = split_val("relag", list(train_idx), seed, {})
        scores.append(train_eval_mlp(embeddings, labels, inner, val, list(test_idx),
                                     metric=metric, lr=LEARNING_RATE, seed=seed)["score"])
    return scores


def show(label: str, r: dict) -> None:
    print(f"  {label:<16} communities={r['n_communities']:,}  "
          f"leakage={np.mean(r['leakage[%]']):.2f} ± {np.std(r['leakage[%]'], ddof=1):.2f}%  "
          f"test={int(np.mean(r['n_test']))}")


def sweep_dataset(name: str, cache: CacheStore) -> dict:
    """Reference split and the alternative-distance sweep of one dataset."""
    cfg = DATASETS[name]
    items, labels = load_dataset(name, cache)
    n, tau_ref = len(items), cfg.proximity_threshold
    print(f"[bold]== {name}: {n:,} samples, reference tau={tau_ref} ==[/]")
    ref = (identity_pairs(items, tau_ref) if name in PROTEINS
           else tanimoto_pairs(np.asarray(cache.get_dataset(name)[0], dtype=bool), tau_ref))

    hnsw = HNSWState(cfg.modality, items, proximity_threshold=tau_ref, **cfg.kernel_params)
    hnsw.build(progress=True)
    ref_graph = hnsw.edges().graph(inweight_type=INWeightType.Distance)
    ref_gamma = gamma_for(name, cache)
    ref_comm, ref_splits = splits_of(ref_graph, ref_gamma)
    reference = stats(ref, n, ref_comm, ref_splits, ref_gamma)
    show("reference", reference)

    alt = alternative(name, items, cache)
    gamma_cdf = null_model_cdf(alt["null"])
    hnsw = HNSWState(alt["variant"], alt["data"], proximity_threshold=max(alt["grid"]), **alt["kwargs"])
    hnsw.build(progress=True)
    edges = hnsw.edges()
    dist = np.fromiter((e[2] for e in edges), dtype=np.float32, count=len(edges))
    sweep, kept = [], {}
    for tau in alt["grid"]:
        graph = edges[dist <= tau].graph(inweight_type=INWeightType.Distance)
        gamma = float(gamma_cdf(np.array([tau]))[0])
        kept[tau] = splits_of(graph, gamma)
        sweep.append({"tau": tau, **stats(ref, n, *kept[tau], gamma)})
        show(f"{alt['label']} {tau:.3f}", sweep[-1])
    return {"name": name, "items": items, "labels": labels, "ref": ref, "alt": alt["label"],
            "reference": reference, "ref_comm": ref_comm, "ref_splits": ref_splits,
            "sweep": sweep, "kept": kept}


def leaking_counts(r: dict) -> np.ndarray:
    """Leaking test samples per split seed."""
    return np.asarray(r["leakage[%]"]) * np.asarray(r["n_test"]) / 100.0


def pooled_oracle(states: list[dict]) -> float:
    """tau' whose leakage, pooled over the datasets, is closest to the pooled reference leakage."""
    def pooled(rows: list[dict]) -> float:
        return 100.0 * sum(leaking_counts(r).sum() for r in rows) / sum(np.sum(r["n_test"]) for r in rows)
    target = pooled([s["reference"] for s in states])
    grid = [r["tau"] for r in states[0]["sweep"]]
    curve = [pooled([s["sweep"][k] for s in states]) for k in range(len(grid))]
    for tau, value in zip(grid, curve):
        print(f"  pooled tau'={tau:.3f}  leakage={value:.2f}%  (reference {target:.2f}%)")
    return grid[int(np.argmin([abs(v - target) for v in curve]))]


def finalize(state: dict, tau: float, cache: CacheStore) -> dict:
    """ARI / AMI and downstream scores at tau'."""
    name, cfg = state["name"], DATASETS[state["name"]]
    n = len(state["items"])
    best = next(r for r in state["sweep"] if r["tau"] == tau)
    oracle_comm, oracle_splits = state["kept"][tau]
    embeddings = torch.as_tensor(embeddings_for(name, state["items"], cache))
    labels = np.asarray(state["labels"], dtype=float)
    random_splits = [random_split(n, s) for s in range(N_SPLIT_SEEDS)]
    reference = {**state["reference"],
                 "score": downstream(embeddings, labels, cfg.metric, state["ref_splits"])}
    oracle = {**best, "ari": float(adjusted_rand_score(state["ref_comm"], oracle_comm)),
              "ami": float(adjusted_mutual_info_score(state["ref_comm"], oracle_comm)),
              "score": downstream(embeddings, labels, cfg.metric, oracle_splits)}
    rand = {"leakage[%]": [leakage(state["ref"], n, tr, te) for tr, te in random_splits],
            "score": downstream(embeddings, labels, cfg.metric, random_splits)}
    for split, splits in ((reference, state["ref_splits"]), (oracle, oracle_splits), (rand, random_splits)):
        split["nn_dist"] = nearest_train(name, state["items"], cache, splits)
    print(f"  {name}: tau'={tau}  ARI={oracle['ari']:.3f}  AMI={oracle['ami']:.3f}  "
          f"{cfg.metric}: ref={np.mean(reference['score']):.3f}  oracle={np.mean(oracle['score']):.3f}  "
          f"random={np.mean(rand['score']):.3f}")
    return {"dataset": name, "n": n, "reference_tau": cfg.proximity_threshold,
            "alternative": state["alt"], "metric": cfg.metric, "reference": reference,
            "sweep": state["sweep"], "oracle": oracle, "random": rand}


def save(result: dict, suffix: str = "") -> None:
    path = RESULTS_DIR / f"oracle_threshold_{result['dataset']}{suffix}.json"
    path.write_text(json.dumps({"settings": {"n_split_seeds": N_SPLIT_SEEDS, "test_ratio": TEST_RATIO},
                                **result}))
    print(f"  Saved {path}")


def main() -> None:
    cache = CacheStore()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or PROTEINS + MOLECULES
    for name in [n for n in names if n in PROTEINS]:
        state = sweep_dataset(name, cache)
        save(finalize(state, pooled_oracle([state]), cache))
    molecules = [n for n in names if n in MOLECULES]
    if molecules:
        states = [sweep_dataset(name, cache) for name in molecules]
        tau = pooled_oracle(states)
        print(f"[bold]  shared molecule tau'={tau}[/]")
        for state in states:
            save(finalize(state, tau, cache), suffix="_pooled")


if __name__ == "__main__":
    main()
