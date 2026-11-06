#ifndef USALIGN_WRAPPER_H
#define USALIGN_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

/* ── File-path API ─────────────────────────────────────────────────────────── */

/* Align two PDB files and return TM-score + RMSD.
 *
 * tm1_out : TM-score normalized by length of pdb1  (0..1, 1 = identical)
 * tm2_out : TM-score normalized by length of pdb2  (0..1, 1 = identical)
 * rmsd_out: RMSD in Angstroms after superposition
 * fast    : non-zero → use fast alignment mode (~3-5x faster, slight accuracy loss)
 *
 * Returns 0 on success, 1 if a file cannot be parsed (< 3 residues),
 * 2 on any unhandled C++ exception.
 */
int usalign_align(
    const char* pdb1,
    const char* pdb2,
    float* tm1_out,
    float* tm2_out,
    float* rmsd_out,
    int*   n_aligned,
    int    fast
);

/* ── Pre-loaded structure API ──────────────────────────────────────────────── */

/* Opaque handle for a parsed PDB chain (coordinate arrays in memory). */
typedef struct PdbStructure PdbStructure;

/* Parse a PDB file into memory.
 * Returns NULL if the file cannot be opened or has fewer than 3 residues. */
PdbStructure* usalign_load_pdb(const char* path);

/* Free a PdbStructure allocated by usalign_load_pdb. Safe to call with NULL. */
void usalign_free_pdb(PdbStructure* s);

/* Align two pre-loaded structures.
 * fast: non-zero → use fast alignment mode (~3-5x faster, slight accuracy loss)
 * Returns 0 on success, 1 if either pointer is NULL, 2 on exception. */
int usalign_align_structures(
    const PdbStructure* s1,
    const PdbStructure* s2,
    float* tm1_out,
    float* tm2_out,
    float* rmsd_out,
    int*   n_aligned,
    int    fast
);

#ifdef __cplusplus
}
#endif

#endif /* USALIGN_WRAPPER_H */
