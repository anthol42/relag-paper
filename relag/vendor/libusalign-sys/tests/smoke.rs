//! Exercises libusalign-sys as an external consumer would: compile against it,
//! import the generated bindings, and run both public APIs end to end against
//! the sample structures bundled in the usalign submodule. This is a pipeline
//! smoke test, not a correctness suite for USalign's alignment algorithm.

use std::ffi::CString;
use std::path::Path;

fn sample_path(name: &str) -> CString {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("usalign")
        .join(name);
    CString::new(path.to_str().unwrap()).unwrap()
}

#[test]
fn align_from_file_paths() {
    let pdb1 = sample_path("PDB1.pdb");
    let pdb2 = sample_path("PDB2.pdb");

    let mut tm1 = 0.0f32;
    let mut tm2 = 0.0f32;
    let mut rmsd = 0.0f32;
    let mut n_aligned = 0i32;

    let rc = unsafe {
        libusalign_sys::usalign_align(
            pdb1.as_ptr(),
            pdb2.as_ptr(),
            &mut tm1,
            &mut tm2,
            &mut rmsd,
            &mut n_aligned,
            0,
        )
    };

    assert_eq!(rc, 0, "usalign_align should succeed on the bundled sample PDBs");
    assert!((0.0..=1.0).contains(&tm1), "tm1 out of range: {tm1}");
    assert!((0.0..=1.0).contains(&tm2), "tm2 out of range: {tm2}");
    assert!(rmsd >= 0.0, "rmsd should be non-negative: {rmsd}");
    assert!(n_aligned > 0, "expected at least one aligned residue");
}

#[test]
fn align_from_preloaded_structures() {
    let pdb1 = sample_path("PDB1.pdb");
    let pdb2 = sample_path("PDB2.pdb");

    unsafe {
        let s1 = libusalign_sys::usalign_load_pdb(pdb1.as_ptr());
        let s2 = libusalign_sys::usalign_load_pdb(pdb2.as_ptr());
        assert!(!s1.is_null(), "failed to load PDB1.pdb");
        assert!(!s2.is_null(), "failed to load PDB2.pdb");

        let mut tm1 = 0.0f32;
        let mut tm2 = 0.0f32;
        let mut rmsd = 0.0f32;
        let mut n_aligned = 0i32;

        let rc = libusalign_sys::usalign_align_structures(
            s1,
            s2,
            &mut tm1,
            &mut tm2,
            &mut rmsd,
            &mut n_aligned,
            1, // fast mode
        );

        libusalign_sys::usalign_free_pdb(s1);
        libusalign_sys::usalign_free_pdb(s2);

        assert_eq!(rc, 0, "usalign_align_structures should succeed");
        assert!(n_aligned > 0, "expected at least one aligned residue");
    }
}
