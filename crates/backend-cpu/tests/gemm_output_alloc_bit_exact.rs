//! `CpuBackendOps::gemm` の出力バッファ確保方式変更（イシュー #1299・
//! 設計 `docs/cpu-matmul-fixed-cost-design.md` §3.C 案 1a）前後の
//! bit 完全一致回帰テスト。
//!
//! 「変更後」は本番 `CpuBackendOps::gemm`（`crate::ops::zeroed_output`
//! 経由の出力確保）そのもの。「変更前」は呼び出し元で
//! `vec![0.0f32; m*n]`（#1299 以前の逐次ゼロ確保）を明示的に確保し、
//! カーネル本体（`gemm_blis_parallel`）へ直接渡した結果として再現する。
//! `gemm_blis_parallel` はカーネル本体そのもので #1299 が触っていない
//! ため、この 2 経路の唯一の差分は「出力バッファの確保方式」に絞られる。
//!
//! **本番既定は `GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS = usize::MAX`（並列
//! 分岐は常に無効）**。M4 Max スモーク実測（#1299・
//! `docs/perf/cpu-matmul-fixed-cost-impl.md`）で N=2048 の後退を確認した
//! 後、#1301 の両実機実測 → PR #1448 差し戻し → #1481 の独立再計測
//! （§20.7）で verdict=REJECT 確定 → #1482 で確定既定、という経緯を
//! 経ている。したがって本ファイルの `ops.gemm(..)` 呼び出しは（形状に
//! 依らず）常に `crate::ops::zeroed_output` の逐次分岐を通る。並列分岐
//! 自体の bit 一致は `crates/backend-cpu/src/ops.rs::zeroed_output_tests::
//! parallel_branch_output_matches_sequential_branch_through_kernel`
//! （クレート内テスト。明示的にしきい値 0 を渡して強制する）が別途
//! 固定する。
//!
//! 形状は次の 3 グループ:
//! - 小形状: 従来経路のみを通るため差が出ないことを確認
//! - 「閾値またぎ」相当（旧設計時の並列分岐境界形状。現在は本番既定が
//!   無効化のため両経路とも逐次を通るが、しきい値を再度有効化した際の
//!   回帰対象として形状自体は維持する）
//! - 大形状正方（512/1024/2048。`#[ignore]`・release 実機）:
//!   本番 gate 対象形状（`docs/perf/cpu-gemm-candle-gate-remeasurement.md`
//!   の `cpu={512,1024,2048}`）そのもの

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::{CpuBackendOps, gemm_blis_parallel};
use fandhe_ai_tensor_core::BackendOps;
use fandhe_ai_tensor_core::Tensor;

fn random_matrix(seed: u64, len: usize) -> Vec<f32> {
    Xorshift64Star::new(seed).fill_vec(len)
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).unwrap()
}

fn contiguous_slice(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().as_slice().unwrap().to_vec()
}

/// 「#1299 以前の逐次ゼロ確保」を明示的に再現する参照経路。
/// `gemm_blis_parallel`（カーネル本体。#1299 で変更していない）を
/// `vec![0.0f32; m*n]` 出力へ直接呼ぶ。
fn gemm_pre_1299_reference(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    gemm_blis_parallel(a, b, &mut out, m, n, k).expect("gemm_blis_parallel must succeed");
    out
}

/// 出力確保方式変更前後の bit 完全一致を 1 形状について検証する共通処理。
fn assert_output_alloc_bit_exact(seed: u64, m: usize, k: usize, n: usize) {
    let ops = CpuBackendOps::new();
    let a_data = random_matrix(seed, m * k);
    let b_data = random_matrix(seed.wrapping_add(1), k * n);
    let a = tensor(a_data.clone(), &[m, k]);
    let b = tensor(b_data.clone(), &[k, n]);

    // after: 本番 CpuBackendOps::gemm（zeroed_output 経由）。
    let after = ops.gemm(&a, &b).unwrap();
    let after_bits: Vec<u32> = contiguous_slice(&after)
        .iter()
        .map(|x| x.to_bits())
        .collect();

    // before: #1299 以前の逐次ゼロ確保を明示的に再現した参照経路。
    let before = gemm_pre_1299_reference(&a_data, &b_data, m, n, k);
    let before_bits: Vec<u32> = before.iter().map(|x| x.to_bits()).collect();

    assert_eq!(
        after.shape(),
        &[m, n],
        "m={m} k={k} n={n}: 出力形状は不変のはず"
    );
    assert_eq!(
        after_bits, before_bits,
        "m={m} k={k} n={n}: 出力確保方式変更後 (zeroed_output) が変更前 \
         (vec![0.0f32; m*n] 直接) と bit 完全一致しないはず（累積カーネル \
         契約はゼロ書き込みの並列度・順序に依存しない）"
    );
}

/// 小形状（明らかに `GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS` 未満。
/// 両経路とも従来の逐次ゼロ確保のみを通る）。
#[test]
fn small_shapes_bit_exact() {
    for &(m, k, n) in &[
        (1usize, 1usize, 1usize),
        (4, 8, 4),
        (37, 65, 33),
        (128, 129, 96),
    ] {
        assert_output_alloc_bit_exact(0x1000, m, k, n);
    }
}

/// `m == 1`（行ベクトル × 行列。`n` は「旧設計時の閾値」超相当・`k` は
/// 小さく抑えて debug ビルドでも短時間で完走させる。本番既定
/// `usize::MAX` により実際には両経路とも逐次を通る）。
#[test]
fn m_equals_one_large_n_bit_exact() {
    // n = 2^21 + 1024（閾値をわずかに超える要素数）・k は小さく抑える。
    let n = (1usize << 21) + 1024;
    assert_output_alloc_bit_exact(0x2000, 1, 4, n);
}

/// `n == 0`（空出力。`zeroed_output_with_threshold(0, ..)` は常に
/// 「未満」分岐で `vec![0.0f32; 0]` を返す契約を経由する）。
#[test]
fn n_equals_zero_bit_exact() {
    assert_output_alloc_bit_exact(0x3000, 4, 3, 0);
}

/// 「旧設計時の閾値ちょうど・わずかに超える境界」相当（`m*n` が
/// `2^21` 前後）。`k` を小さく抑えて debug ビルドでも短時間で完走
/// させる。本番既定 `usize::MAX` により実際には両経路とも逐次を通る
/// （並列分岐自体の bit 一致は `ops.rs::zeroed_output_tests::
/// parallel_branch_output_matches_sequential_branch_through_kernel`
/// が別途固定する）。
#[test]
fn threshold_boundary_low_k_bit_exact() {
    // 2^21 = 2,097,152（本番既定しきい値）。m*n をちょうど・+1 に合わせる。
    let min_elems = 1usize << 21;
    for &(m, n) in &[(2048usize, 1024usize), (2048, 1025)] {
        assert!(
            m * n >= min_elems,
            "テスト形状は閾値以上であるべき（m*n={}）",
            m * n
        );
        assert_output_alloc_bit_exact(0x4000 + m as u64 + n as u64, m, 3, n);
    }
    // 閾値未満（2047*1024 < 2^21）: 逐次経路のまま並列分岐へ入らないこと
    // も同じテストで確認する（境界のもう一方）。
    assert!((2047usize * 1024) < min_elems);
    assert_output_alloc_bit_exact(0x4100, 2047, 3, 1024);
}

/// 大形状正方（gate 対象形状。実機 release 実行専用。`docs/perf/
/// cpu-gemm-candle-gate-remeasurement.md` の `cpu={512,1024,2048}` と
/// 同一形状）。
#[test]
#[ignore = "release ビルドでの実機実行専用（大形状 GEMM は debug では低速）"]
fn large_square_shapes_bit_exact() {
    for &n in &[512usize, 1024, 2048] {
        assert_output_alloc_bit_exact(0x5000 + n as u64, n, n, n);
    }
}

/// 非正方の大形状（K 支配的形状。#1299 §5.1 の「非正方 1 形状」要求）。
#[test]
#[ignore = "release ビルドでの実機実行専用（大形状 GEMM は debug では低速）"]
fn large_non_square_shape_bit_exact() {
    // K 支配的（m,n が小さく k が大きい）非正方形状。出力サイズ自体は
    // 閾値未満（256*384 < 2^21）だが、カーネル実行時間の参考として
    // 大形状グループに含める。
    assert_output_alloc_bit_exact(0x6000, 256, 4096, 384);
}
