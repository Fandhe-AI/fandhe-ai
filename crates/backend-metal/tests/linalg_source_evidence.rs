//! イシュー #2676: `shaders/linalg.metal` の契約トークンを文字列で固定する証跡テスト
//! （Linux CI でも実行可能。MSL の実コンパイルは macOS 実機の `#[ignore]` テスト）。
//!
//! 固定するもの: 丸め契約（FMA 縮約 off・`precise::` 関数・`fma(` 不在）、4 カーネルの
//! 宣言名と `[[buffer(i)]]` 番号（`linalg_kernel.rs` と一致）、手動境界チェック（REQ-8）、
//! バリア規律、scale/ssq ノルム、threadgroup 幅定数。

const SRC: &str = include_str!("../src/shaders/linalg.metal");

fn kernel_body<'a>(name: &str) -> &'a str {
    let start = SRC
        .find(&format!("kernel void {name}("))
        .unwrap_or_else(|| panic!("カーネル {name} が見つからない"));
    let rest = &SRC[start..];
    let end = rest[1..]
        .find("\nkernel void ")
        .map(|p| p + 1)
        .unwrap_or(rest.len());
    &rest[..end]
}

#[test]
fn include_order_and_rounding_contract() {
    let inc = SRC.find("#include <metal_stdlib>").expect("include");
    let using = SRC.find("using namespace metal;").expect("using");
    assert!(inc < using);
    assert!(SRC.contains("#pragma clang fp contract(off)"));
    assert!(SRC.contains("precise::sqrt"));
    assert!(SRC.contains("precise::divide"));
    // コメント行を除いて `fma(` を使っていない。
    for line in SRC.lines() {
        if line.trim_start().starts_with("//") {
            continue;
        }
        assert!(!line.contains("fma("), "fma( は使わない: {line}");
    }
}

#[test]
fn kernels_and_buffer_indices_are_pinned() {
    let lu = kernel_body("linalg_lu_f32");
    for tok in [
        "device float* a [[buffer(0)]]",
        "device int* perm [[buffer(1)]]",
        "device int* status [[buffer(2)]]",
        "constant uint& n [[buffer(3)]]",
    ] {
        assert!(lu.contains(tok), "lu: {tok}");
    }
    let solve = kernel_body("linalg_lu_solve_f32");
    for tok in [
        "device const float* lu [[buffer(0)]]",
        "device const int* perm [[buffer(1)]]",
        "device const int* status [[buffer(2)]]",
        "device const float* b [[buffer(3)]]",
        "device float* x [[buffer(4)]]",
        "constant uint& n [[buffer(5)]]",
        "constant uint& k [[buffer(6)]]",
    ] {
        assert!(solve.contains(tok), "solve: {tok}");
    }
    let chol = kernel_body("linalg_cholesky_f32");
    for tok in [
        "device const float* a [[buffer(0)]]",
        "device float* l [[buffer(1)]]",
        "device int* status [[buffer(2)]]",
        "constant uint& n [[buffer(3)]]",
    ] {
        assert!(chol.contains(tok), "cholesky: {tok}");
    }
    let qr = kernel_body("linalg_qr_f32");
    for tok in [
        "device float* r [[buffer(0)]]",
        "device float* q [[buffer(1)]]",
        "device float* v [[buffer(2)]]",
        "device int* applied [[buffer(3)]]",
        "constant uint& m [[buffer(4)]]",
        "constant uint& n [[buffer(5)]]",
    ] {
        assert!(qr.contains(tok), "qr: {tok}");
    }
}

#[test]
fn bounds_checks_barriers_and_norm_are_present() {
    let lu = kernel_body("linalg_lu_f32");
    assert!(lu.contains("tg_pos != 0"));
    assert!(lu.contains("threadgroup_barrier"));
    assert!(lu.contains("i < n"));
    let solve = kernel_body("linalg_lu_solve_f32");
    assert!(solve.contains("gid >= k"));
    assert!(solve.contains("src < n"));
    let chol = kernel_body("linalg_cholesky_f32");
    assert!(chol.contains("threadgroup_barrier"));
    assert!(chol.contains("!isfinite(sum) || sum <= 0.0f"));
    let qr = kernel_body("linalg_qr_f32");
    assert!(qr.contains("scale_ssq_norm("));
    assert!(qr.contains("i < len"));
    assert!(SRC.contains("static inline float scale_ssq_norm("));
    assert!(SRC.contains("isnan(x)") && SRC.contains("isinf(x)"));
}

#[test]
fn threadgroup_width_constant_is_pinned() {
    assert_eq!(fandhe_ai_backend_metal::linalg::LINALG_TG_WIDTH, 256);
}
