use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // Rerun when these change
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=wrapper.cpp");
    println!("cargo:rerun-if-changed=usalign/basic_fun.h");
    println!("cargo:rerun-if-changed=usalign/TMalign.h");

    let target = std::env::var("TARGET").unwrap();
    let is_msvc = target.contains("windows") && target.contains("msvc");

    // Compile wrapper.cpp (which includes USalign headers)
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file(manifest.join("wrapper.cpp"))
        .include(&manifest)
        .include(manifest.join("usalign"))
        .flag_if_supported("-O3")
        // suppress warnings from USalign's own headers
        .flag_if_supported("-Wno-unused-variable")
        .flag_if_supported("-Wno-unused-function")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-sign-compare")
        .flag_if_supported("-Wno-uninitialized")
        .flag_if_supported("-Wno-char-subscripts")
        .flag_if_supported("-Wno-unused-but-set-variable")
        .flag_if_supported("-Wno-unused-but-set-parameter")
        .flag_if_supported("-Wno-deprecated-declarations");

    if is_msvc {
        // MSVC's cl.exe rejects the variable-length arrays USalign relies on
        // (e.g. usalign/TMalign.h) since VLAs are only a GNU/Clang extension,
        // not standard C++. clang++ accepts them while still emitting object
        // code that conforms to the MSVC ABI (calling convention, name
        // mangling, exception model), so it links transparently with the
        // rest of an MSVC-targeted Rust build.
        build
            .compiler("clang++")
            .flag_if_supported("-Wno-vla-cxx-extension")
            // Statically link the MSVC C/C++ runtime (equivalent to cl.exe's
            // /MT) so consumers of this crate don't pick up a runtime
            // dependency on msvcp140.dll/vcruntime140.dll, which aren't
            // installed by default outside of a VC++ Redistributable. This
            // only pulls in libcmt/libcpmt via linker directives embedded in
            // the object file; it links fine (aside from a benign LNK4098
            // "conflicting defaultlib" warning) alongside the rest of a
            // normal dynamically-linked Rust build.
            .flag("-fms-runtime-lib=static");
    }

    build.compile("usalign");

    // Generate Rust bindings from wrapper.h
    let bindings = bindgen::Builder::default()
        .header(manifest.join("wrapper.h").to_str().unwrap())
        .allowlist_function("usalign_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed to generate bindings");

    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("could not write bindings.rs");
}
