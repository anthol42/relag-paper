# Sourced by threshold_spread.sh and uniref_ecdf.sh -- not run directly.
# Builds the ProtSpaM HNSW index over UniRef50 + the protein benchmarks and
# exports its layer-0 edge graph to $EDGESTORE. Resumable: every stage is
# skipped if its output file already exists.
#
# Needs: UNIREF50_FASTA (path to uniref50.fasta.gz)
# Optional: BENCHMARK_CACHE_DIR (default .cache/protein_benchmarks),
#           UNIREF_OUT (default .cache/uniref)

: "${UNIREF50_FASTA:?set UNIREF50_FASTA to the path of uniref50.fasta.gz}"
BENCHMARK_CACHE_DIR="${BENCHMARK_CACHE_DIR:-.cache/protein_benchmarks}"
UNIREF_OUT="${UNIREF_OUT:-.cache/uniref}"
EDGESTORE="$UNIREF_OUT/unirefxbenchmark_layer0.edgestr"

mkdir -p "$UNIREF_OUT" "$BENCHMARK_CACHE_DIR"

uv run python -m threshold.build_uniref_benchmark_index \
    --uniref50-fasta "$UNIREF50_FASTA" \
    --benchmark-cache-dir "$BENCHMARK_CACHE_DIR" \
    --uniref-sequences-out "$UNIREF_OUT/uniref50.swseq" \
    --benchmark-sequences-out "$UNIREF_OUT/benchmarks.swseq" \
    --offsets-out "$UNIREF_OUT/offsets.txt" \
    --patterns-out "$UNIREF_OUT/patterns.bin" \
    --uniref-index-out "$UNIREF_OUT/uniref50.hnsw" \
    --extended-index-out "$UNIREF_OUT/unirefxbenchmark.hnsw" \
    --edgestore-out "$EDGESTORE"
