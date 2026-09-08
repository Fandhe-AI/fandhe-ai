//! facade 経由 CPU matmul の bit 完全一致回帰（イシュー #1299）。
//!
//! `crates/backend-cpu/tests/gemm_output_alloc_bit_exact.rs` は
//! `CpuBackendOps::gemm` 単体を検証するのに対し、本ファイルは
//! `fandhe_ai::tape()`（公開 API の唯一の入口。`crates/facade/src/
//! lib.rs::tape`）→ `Var::matmul`（`crates/autodiff/src/var.rs`）→
//! `CpuBackendOps::gemm`（`crates/backend-cpu/src/ops.rs::zeroed_output`
//! 経由の出力確保）という **facade 経由の end-to-end** 経路が、出力
//! バッファ確保方式変更（イシュー #1299）の前後で bit 完全一致すること
//! を確認する。「変更前」は `fandhe_ai_backend_cpu::gemm_blis_parallel`
//! （カーネル本体・#1299 で不変）を `vec![0.0f32; m*n]` 出力へ直接呼んだ
//! 結果として再現する。
//!
//! **本番既定は `GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS = usize::MAX`（並列
//! 分岐は常に無効）**。M4 Max スモーク実測（#1299・`docs/perf/
//! cpu-matmul-fixed-cost-impl.md`）で N=2048 の後退を確認したため、DGX
//! Spark GB10 実機実測（#1301）が有効化可否を判断するまでの暫定値。
//! したがって本ファイルの `a_var.matmul(&b_var)` 呼び出しは（形状に
//! 依らず）常に逐次分岐を通る。並列分岐自体の bit 一致は
//! `crates/backend-cpu/src/ops.rs::zeroed_output_tests::
//! parallel_branch_output_matches_sequential_branch_through_kernel` が
//! 別途固定する。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::gemm_blis_parallel;
use fandhe_ai_tensor_core::Tensor;

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(len)
}

/// 「#1299 以前の逐次ゼロ確保」を明示的に再現する参照経路
/// （`crates/backend-cpu/tests/gemm_output_alloc_bit_exact.rs` と同型）。
fn gemm_pre_1299_reference(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    gemm_blis_parallel(a, b, &mut out, m, n, k).expect("gemm_blis_parallel must succeed");
    out
}

/// `fandhe_ai::tape()` 経由の `Var::matmul` 1 回が、参照経路と bit 完全
/// 一致することを 1 形状について検証する共通処理。
fn assert_facade_matmul_bit_exact(seed: u64, m: usize, k: usize, n: usize) {
    let a_data = random_matrix(seed, m * k);
    let b_data = random_matrix(seed.wrapping_add(1), k * n);
    let a_tensor = Tensor::new(a_data.clone(), &[m, k]).unwrap();
    let b_tensor = Tensor::new(b_data.clone(), &[k, n]).unwrap();

    // facade の唯一の公開入口（`fandhe_ai::tape()`）を通した matmul。
    let tape = fandhe_ai::tape();
    let a_var = tape.var(&a_tensor);
    let b_var = tape.var(&b_tensor);
    let c_var = a_var.matmul(&b_var).expect("matmul: shape 一致");
    let after = c_var.to_tensor();
    let after_bits: Vec<u32> = after
        .contiguous()
        .as_slice()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect();

    let before = gemm_pre_1299_reference(&a_data, &b_data, m, n, k);
    let before_bits: Vec<u32> = before.iter().map(|x| x.to_bits()).collect();

    assert_eq!(
        after.shape(),
        &[m, n],
        "m={m} k={k} n={n}: facade 経由 matmul の出力形状は不変のはず"
    );
    assert_eq!(
        after_bits, before_bits,
        "m={m} k={k} n={n}: facade 経由 matmul（zeroed_output 経由の出力 \
         確保）が #1299 以前の逐次ゼロ確保参照経路と bit 完全一致しない \
         はず"
    );
}

/// 小形状（`GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS` 未満。両経路とも従来の
/// 逐次ゼロ確保のみを通る）。
#[test]
fn small_shapes_facade_matmul_bit_exact() {
    for &(m, k, n) in &[
        (1usize, 1usize, 1usize),
        (4, 8, 4),
        (37, 65, 33),
        (128, 129, 96),
    ] {
        assert_facade_matmul_bit_exact(0x1000, m, k, n);
    }
}

/// 「旧設計時の閾値ちょうど・わずかに超える境界」相当（`k` を小さく
/// 抑えて debug でも短時間で完走させる）。本番既定 `usize::MAX` に
/// より実際には逐次経路を通る（ファイル冒頭コメント参照）。
#[test]
fn threshold_boundary_low_k_facade_matmul_bit_exact() {
    let min_elems = 1usize << 21; // 旧設計時の GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS 値。
    for &(m, n) in &[(2048usize, 1024usize), (2048, 1025)] {
        assert!(m * n >= min_elems, "テスト形状は閾値以上であるべき");
        assert_facade_matmul_bit_exact(0x4000 + m as u64 + n as u64, m, 3, n);
    }
}

/// 大形状正方（gate 対象形状。`#[ignore]`・release ビルドでの実機実行
/// 専用）。
#[test]
#[ignore = "release ビルドでの実機実行専用（大形状 GEMM は debug では低速）"]
fn large_square_shapes_facade_matmul_bit_exact() {
    for &n in &[512usize, 1024, 2048] {
        assert_facade_matmul_bit_exact(0x5000 + n as u64, n, n, n);
    }
}
