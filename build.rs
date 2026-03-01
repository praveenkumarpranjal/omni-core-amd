fn main() {
    // All libraries are self-contained in our libs/ directory
    let project_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let libs_dir = format!("{}/libs", project_dir);

    // HIP runtime
    println!("cargo:rustc-link-search=native=/opt/rocm/lib");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    
    // rocBLAS for GEMM operations
    println!("cargo:rustc-link-lib=dylib=rocblas");

    // Our self-contained libraries
    println!("cargo:rustc-link-search=native={}", libs_dir);
    println!("cargo:rustc-link-lib=dylib=ggml_bridge");
    println!("cargo:rustc-link-lib=dylib=rope_fast");
    println!("cargo:rustc-link-lib=dylib=ggml-hip");
    println!("cargo:rustc-link-lib=dylib=ggml-base");

    // Runtime library path  
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", libs_dir);
    println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/rocm/lib");
}
