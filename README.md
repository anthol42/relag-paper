# relag paper — figure reproduction

## Requirements

- [uv](https://docs.astral.sh/uv/) and Python 3.13
- `mmseqs2` on `PATH` (used by hestia)
- `apptainer` or `singularity` on `PATH` (DataSAIL splits)
- Kaggle credentials for the BELKA and tiny-imagenet downloads: `KAGGLE_API_TOKEN`,
  or `KAGGLE_USERNAME` + `KAGGLE_KEY`, in a `.env` file (you must also accept the
  BELKA competition rules on Kaggle)
- UniRef50 FASTA (`uniref50.fasta.gz`) for the UniRef / ProtSpaM figures (You can download from: [Uniprot](https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz))
- A GPU is optional (used for the comparison embeddings and training)

## Setup

```bash
uv sync
export UNIREF50_FASTA=/path/to/uniref50.fasta.gz
```

The comparison benchmark creates its own per-method venvs under
`$HOME/venvs/refnd_exp` (override with `RELAG_VENV_BASE`).

## Regenerating the data

Run from the repo root. Each script writes the result files read by one section
of `notebooks/comparison_figures.ipynb`; the committed results are already there,
so this is only needed to reproduce them.

```bash
# 1. Comparison benchmark (relag vs hestia / DataSAIL / random)
bash figure_scripts/comparison_perf.sh
bash figure_scripts/comparison_leakage.sh      # reuses the setup from comparison_perf.sh

# 2. Runtime scaling
bash figure_scripts/scaling_runtime.sh
bash figure_scripts/scaling_detailed_atlas.sh
bash figure_scripts/scaling_detailed_belka.sh

# 3. Threshold selection
bash figure_scripts/threshold_spread.sh
bash figure_scripts/uniref_ecdf.sh             # reuses the UniRef index from threshold_spread.sh
bash figure_scripts/protspam_vs_alignment.sh

# 4. Data starvation
bash figure_scripts/data_starvation_all.sh     # includes dbaasp (data_starvation_dbaasp.sh runs it alone)
bash figure_scripts/singleton_power.sh

# 5. Synthetic peptides
bash figure_scripts/synthetic_compare_splits.sh
bash figure_scripts/synthetic_threshold.sh
bash figure_scripts/synthetic_family_leakage.sh
```

## Making the figures

Run `notebooks/comparison_figures.ipynb` top to bottom. Figures are written to
`figs/` (the schematic figures in `figs/` are hand-made and not generated).
