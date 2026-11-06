#include "wrapper.h"

// Suppress USalign's main() — we only want the library functions.
#define MAIN_GUARD
#include "usalign/basic_fun.h"
#include "usalign/TMalign.h"

// ── Internal struct definition ────────────────────────────────────────────────

struct PdbStructure {
    double** xa;
    char*    seqx;
    char*    secx;
    int      xlen;
    int      mol_type;  // -1 for protein, +1 for RNA
};

// ── Shared helpers ────────────────────────────────────────────────────────────

static const int    TER_OPT   = 3;       // stop at first TER record
static const int    INFMT_OPT = 0;       // PDB format
static const string ATOM_OPT  = " CA ";  // C-alpha atoms
static const bool   AUTOJUST  = true;
static const int    SPLIT_OPT = 0;
static const int    HET_OPT   = 0;

// Load and parse the first chain of a PDB file. Returns NULL on failure.
static PdbStructure* load_chain(const char* path)
{
    const vector<string> empty_vec;

    vector<vector<string>> pdb_lines;
    vector<string>         chain_ids;
    vector<int>            mol_vec;

    size_t nchains = get_PDB_lines(
        string(path), pdb_lines, chain_ids, mol_vec,
        TER_OPT, INFMT_OPT, ATOM_OPT, AUTOJUST,
        SPLIT_OPT, HET_OPT, empty_vec, empty_vec);

    if (!nchains || pdb_lines[0].size() < 3) return nullptr;

    int len = (int)pdb_lines[0].size();

    double** xa   = nullptr;
    char*    seqx = new char[len + 1];
    char*    secx = new char[len + 1];
    NewArray(&xa, len, 3);

    vector<string> resi_vec;
    len = read_PDB(pdb_lines[0], xa, seqx, resi_vec, 0);
    make_sec(xa, len, secx);

    PdbStructure* s  = new PdbStructure;
    s->xa       = xa;
    s->seqx     = seqx;
    s->secx     = secx;
    s->xlen     = len;
    s->mol_type = mol_vec[0];
    return s;
}

// Run TMalign and write results via output pointers.
static int run_tmalign(
    const PdbStructure* s1, const PdbStructure* s2,
    float* tm1_out, float* tm2_out, float* rmsd_out, int* n_aligned,
    bool fast)
{
    double t0[3], u0[3][3];
    double TM1, TM2, TM3, TM4, TM5;
    double d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out = 5.0;
    string seqM, seqxA, seqyA;
    vector<double> do_vec;
    double rmsd0 = 0.0, Liden = 0.0, TM_ali, rmsd_ali;
    int    L_ali = 0, n_ali = 0, n_ali8 = 0;
    bool   fast_opt = fast || (min(s1->xlen, s2->xlen) > 1500);
    int    mol_type = s1->mol_type + s2->mol_type;

    TMalign_main(
        s1->xa, s2->xa, s1->seqx, s2->seqx, s1->secx, s2->secx,
        t0, u0, TM1, TM2, TM3, TM4, TM5,
        d0_0, TM_0, d0A, d0B, d0u, d0a, d0_out,
        seqM, seqxA, seqyA, do_vec,
        rmsd0, L_ali, Liden, TM_ali, rmsd_ali, n_ali, n_ali8,
        s1->xlen, s2->xlen,
        vector<string>(), -1.0, 0.0,
        0, 0, false, false, fast_opt,
        mol_type);

    *tm1_out   = (float)TM1;
    *tm2_out   = (float)TM2;
    *rmsd_out  = (float)rmsd0;
    *n_aligned = n_ali8;
    return 0;
}

// ── Public C API ──────────────────────────────────────────────────────────────

extern "C" {

int usalign_align(
    const char* pdb1, const char* pdb2,
    float* tm1_out, float* tm2_out, float* rmsd_out, int* n_aligned,
    int fast)
{
    try {
        PdbStructure* s1 = load_chain(pdb1);
        if (!s1) return 1;
        PdbStructure* s2 = load_chain(pdb2);
        if (!s2) { usalign_free_pdb(s1); return 1; }

        int rc = run_tmalign(s1, s2, tm1_out, tm2_out, rmsd_out, n_aligned, (bool)fast);
        usalign_free_pdb(s1);
        usalign_free_pdb(s2);
        return rc;
    } catch (...) { return 2; }
}

PdbStructure* usalign_load_pdb(const char* path)
{
    try { return load_chain(path); } catch (...) { return nullptr; }
}

void usalign_free_pdb(PdbStructure* s)
{
    if (!s) return;
    DeleteArray(&s->xa, s->xlen);
    delete[] s->seqx;
    delete[] s->secx;
    delete s;
}

int usalign_align_structures(
    const PdbStructure* s1, const PdbStructure* s2,
    float* tm1_out, float* tm2_out, float* rmsd_out, int* n_aligned,
    int fast)
{
    if (!s1 || !s2) return 1;
    try { return run_tmalign(s1, s2, tm1_out, tm2_out, rmsd_out, n_aligned, (bool)fast); }
    catch (...) { return 2; }
}

} // extern "C"
