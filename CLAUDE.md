# relag paper — figure reproduction

**ALWAYS USE UV RUN! NOT python3!!**
**ALWAYS USE THE MCP SERVER TO INTERACT WITH JUPYTER, NEVER WRITE TO DISK DIRECTLY!!**: Otherwise it will corrupt the system.

---

This repo only reproduces the paper's figures. See `README.md` for setup and run order.

## Structure

| Path | Role |
|------|------|
| `notebooks/comparison_figures.ipynb` | Makes every generated figure from the committed results |
| `figure_scripts/` | One shell script per notebook section; regenerates its result files |
| `results/`, `comparison/results/` | Committed result data read by the notebook |
| `figs/` | Figure output; the 8 committed SVGs are hand-made schematics |
| `relag/` | ReLaG source (Rust + PyO3 bindings, native deps vendored in `relag/vendor/`); built by `uv sync`, imported as `relag` |
| `src/` | Shared library (datasets, cache, embeddings, metrics, MLP), imported as `src.*` |
| `comparison/scripts/` | Comparison benchmark (relag vs hestia / DataSAIL / random): download, embeddings, splits, training, leakage. Runs in its own per-method venvs |
| `threshold/` | Threshold-selection sweeps (MNIST, molecules, peptides, UniRef) and ProtSpaM vs. alignment |
| `synthetic/` | Synthetic-peptide experiment (split comparison, threshold, family leakage) |
| `data_starvation/` | Data-starvation and singleton-power experiments |
| `scaling_benchmark.py`, `runtime_scripts/` | Runtime scaling benchmark (relag vs hestia) |
| `debug_scaling_relag.py` | Per-stage relag timing |
