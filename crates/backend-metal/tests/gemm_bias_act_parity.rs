//! イシュー #605: elementwise 5 演算・`gemm_bias_act` epilogue 融合カーネル
//! の CPU-Metal 数値一致検証（CUDA 側 `backend-cuda::tests::
//! gemm_bias_act_parity`〈#599〉の Metal 対応版）。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する
//! （`gemm_naive_parity.rs`・`backend_ops_real_device.rs` と同方針。
//! `#![cfg(target_os = "macos")]` により Linux CI ではコンパイル対象外に
//! なり、`#[ignore]` により通常の `cargo test` からも除外される）。
//!
//! 判定式・許容誤差は再定義せず `backend_cpu::parity` を唯一の参照とする
//! （`.claude/rules/coding-rust.md`）。CUDA との数値一致は #604 softmax の
//! 先例に従い CPU 参照経由の推移的な担保とする（両実機を同時に持つ検証
//! 環境が存在しないため。両バックエンドとも同一 CPU 参照・同一複合判定で
//! 検証済み）。
//!
//! Linux CI での型検査（実機なしでもコンパイル可能性を担保）:
//!
//! ```sh
//! cargo check -p backend-metal --tests --target aarch64-apple-darwin
//! ```
//!
//! 実行コマンド（Apple Silicon 実機。`--release` 推奨）:
//!
//! ```sh
//! cargo test -p backend-metal --release --test gemm_bias_act_parity -- --ignored --nocapture
//! ```
//!
//! `BIAS_ACT_FUSED_LAUNCH_COUNT`（融合カーネル起動回数の可観測点）は
//! `pub(crate)` のためクレート境界外の本ファイルからは参照できない
//! （`crate::gemm::MetalGemm::pipeline_for_tile` ドキュメンテーション
//! コメントが記録する codex-review 是正〈`#[doc(hidden)] pub` を避け
//! `pub(crate)` を維持する〉方針に従う）。「フォールバック非経由」の確認は
//! `crates/backend-metal/src/gemm.rs` の `#[cfg(test)]` 内クレート内テスト
//! （`run_tiled_bias_act_f32_increments_fused_launch_counter`。macOS 実機
//! ・`#[ignore]`）に委ね、本ファイルは数値一致・経路選択（fused／
//! フォールバック）の結果 shape・値のみを検証する。

#![cfg(target_os = "macos")]

use backend_cpu::CpuBackendOps;
use backend_metal::MetalBackendOps;
use bench_harness::rng::Xorshift64Star;
use tensor_core::{Activation, BackendOps, Tensor};

/// CPU-Metal の `gemm_bias_act` 複合判定（REQ-2）と、Metal 上での融合 vs
/// 非融合合成（`gemm`→`add`→act）の bit 完全一致（`shaders/gemm.metal::
/// gemm_tiled_bias_act` ドキュメンテーションコメント「数値契約」参照）を
/// 検証する。
fn assert_gemm_bias_act_parity(
    seed_a: u64,
    seed_b: u64,
    seed_bias: u64,
    m: usize,
    n: usize,
    k: usize,
    act: Activation,
) {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let a_data = Xorshift64Star::new(seed_a).fill_vec(m * k);
    let b_data = Xorshift64Star::new(seed_b).fill_vec(k * n);
    let bias_data = Xorshift64Star::new(seed_bias).fill_vec(n);
    let a = Tensor::new(a_data, &[m, k]).expect("valid tensor");
    let b = Tensor::new(b_data, &[k, n]).expect("valid tensor");
    let bias = Tensor::new(bias_data, &[n]).expect("valid tensor");

    let cpu_result = cpu
        .gemm_bias_act(&a, &b, Some(&bias), act)
        .expect("cpu gemm_bias_act always succeeds");
    let metal_result = metal
        .gemm_bias_act(&a, &b, Some(&bias), act)
        .expect("MetalBackendOps::gemm_bias_act must succeed on Metal-equipped test runner");
    assert_eq!(metal_result.shape(), cpu_result.shape());
    backend_cpu::parity::assert_parity(
        &format!("gemm_bias_act cpu-metal parity m={m} n={n} k={k} act={act:?}"),
        metal_result.as_slice().expect("contiguous"),
        cpu_result.as_slice().expect("contiguous"),
    );

    // Metal 上での融合 vs 非融合合成の bit 完全一致（同一 `gemm_tiled` 系
    // アキュムレーションを経由するため。ただし融合経路は `gemm_tiled_bias_act`
    // を、非融合合成は `dispatch_auto`〈動的タイル選択〉経由の `gemm` を
    // 使うため小形状では simdgroup 系へ分岐しうる。ここでは複合判定〈REQ-2〉
    // で突き合わせ、bit 完全一致は前提としない〈CUDA 側は常に同一
    // `tiled_f32` を経由するため bit 完全一致が成立するが、Metal 側は
    // `dispatch_auto` の動的経路選択のため事情が異なる〉）。
    let mut composed = metal.gemm(&a, &b).expect("metal gemm must succeed");
    composed = metal.add(&composed, &bias).expect("metal add must succeed");
    if act == Activation::Relu {
        composed = metal.relu(&composed).expect("metal relu must succeed");
    }
    backend_cpu::parity::assert_parity(
        &format!("gemm_bias_act fused vs composed metal parity m={m} n={n} k={k} act={act:?}"),
        metal_result.as_slice().expect("contiguous"),
        composed.as_slice().expect("contiguous"),
    );
}

/// elementwise 5 演算（`add`／`mul`／`relu`／`exp`／`tanh`）の CPU-Metal
/// 数値一致（REQ-2 複合判定）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn elementwise_matches_cpu_across_ops() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let a_data = Xorshift64Star::new(11).fill_vec(37);
    let b_data = Xorshift64Star::new(12).fill_vec(37);
    let a = Tensor::new(a_data, &[37]).expect("valid tensor");
    let b = Tensor::new(b_data, &[37]).expect("valid tensor");

    let add_cpu = cpu.add(&a, &b).expect("cpu add");
    let add_metal = metal.add(&a, &b).expect("metal add");
    backend_cpu::parity::assert_parity(
        "elementwise add cpu-metal parity",
        add_metal.as_slice().expect("contiguous"),
        add_cpu.as_slice().expect("contiguous"),
    );

    let mul_cpu = cpu.mul(&a, &b).expect("cpu mul");
    let mul_metal = metal.mul(&a, &b).expect("metal mul");
    backend_cpu::parity::assert_parity(
        "elementwise mul cpu-metal parity",
        mul_metal.as_slice().expect("contiguous"),
        mul_cpu.as_slice().expect("contiguous"),
    );

    let relu_cpu = cpu.relu(&a).expect("cpu relu");
    let relu_metal = metal.relu(&a).expect("metal relu");
    backend_cpu::parity::assert_parity(
        "elementwise relu cpu-metal parity",
        relu_metal.as_slice().expect("contiguous"),
        relu_cpu.as_slice().expect("contiguous"),
    );

    let exp_cpu = cpu.exp(&a).expect("cpu exp");
    let exp_metal = metal.exp(&a).expect("metal exp");
    backend_cpu::parity::assert_parity(
        "elementwise exp cpu-metal parity",
        exp_metal.as_slice().expect("contiguous"),
        exp_cpu.as_slice().expect("contiguous"),
    );

    let tanh_cpu = cpu.tanh(&a).expect("cpu tanh");
    let tanh_metal = metal.tanh(&a).expect("metal tanh");
    backend_cpu::parity::assert_parity(
        "elementwise tanh cpu-metal parity",
        tanh_metal.as_slice().expect("contiguous"),
        tanh_cpu.as_slice().expect("contiguous"),
    );
}

/// `gemm_bias_act` 形状網羅（境界形状・K=4096 ストレス含む。融合経路）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_bias_act_matches_cpu_across_shapes_and_activations() {
    assert_gemm_bias_act_parity(501, 502, 503, 40, 32, 48, Activation::Relu);
    assert_gemm_bias_act_parity(504, 505, 506, 33, 29, 17, Activation::None);
    // threadgroup サイズ（16）の倍数でない境界形状。
    assert_gemm_bias_act_parity(507, 508, 509, 7, 13, 5, Activation::Relu);
    // K ストレスケース（PoC-v2-5 実測構成に対応）。
    assert_gemm_bias_act_parity(510, 511, 512, 64, 64, 4096, Activation::Relu);
}

/// bias 未指定（`None`）の融合経路。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_bias_act_without_bias_matches_cpu() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let a_data = Xorshift64Star::new(601).fill_vec(20 * 12);
    let b_data = Xorshift64Star::new(602).fill_vec(12 * 16);
    let a = Tensor::new(a_data, &[20, 12]).expect("valid tensor");
    let b = Tensor::new(b_data, &[12, 16]).expect("valid tensor");

    let cpu_result = cpu
        .gemm_bias_act(&a, &b, None, Activation::Relu)
        .expect("cpu gemm_bias_act always succeeds");
    let metal_result = metal
        .gemm_bias_act(&a, &b, None, Activation::Relu)
        .expect("metal gemm_bias_act (no bias) must succeed");
    backend_cpu::parity::assert_parity(
        "gemm_bias_act no-bias cpu-metal parity",
        metal_result.as_slice().expect("contiguous"),
        cpu_result.as_slice().expect("contiguous"),
    );
}

/// `k == 0` 縮退（ホスト側 epilogue のみ・GPU 起動なし）の CPU-Metal 一致。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_bias_act_k_zero_matches_cpu() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let a = Tensor::new(Vec::<f32>::new(), &[3, 0]).expect("valid tensor");
    let b = Tensor::new(Vec::<f32>::new(), &[0, 4]).expect("valid tensor");
    let bias = Tensor::new(vec![0.1, 0.2, -0.3, 0.4], &[4]).expect("valid tensor");

    let cpu_result = cpu
        .gemm_bias_act(&a, &b, Some(&bias), Activation::Relu)
        .expect("cpu gemm_bias_act always succeeds");
    let metal_result = metal
        .gemm_bias_act(&a, &b, Some(&bias), Activation::Relu)
        .expect("metal gemm_bias_act (k=0) must succeed");
    backend_cpu::parity::assert_parity(
        "gemm_bias_act k=0 cpu-metal parity",
        metal_result.as_slice().expect("contiguous"),
        cpu_result.as_slice().expect("contiguous"),
    );
}

/// bias 形状 `[1]`／`[1, n]`（ブロードキャスト可能だが `[n]` ちょうどでは
/// ないため非融合合成〈`ops::gemm_bias_act_route`〉へフォールバックする
/// 経路）の CPU-Metal 一致。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_bias_act_broadcast_fallback_matches_cpu() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let a_data = Xorshift64Star::new(701).fill_vec(10 * 6);
    let b_data = Xorshift64Star::new(702).fill_vec(6 * 8);
    let a = Tensor::new(a_data, &[10, 6]).expect("valid tensor");
    let b = Tensor::new(b_data, &[6, 8]).expect("valid tensor");

    for (shape, data) in [
        (vec![1usize], vec![0.5f32]),
        (vec![1usize, 8], Xorshift64Star::new(703).fill_vec(8)),
    ] {
        let bias = Tensor::new(data, &shape).expect("valid tensor");
        let cpu_result = cpu
            .gemm_bias_act(&a, &b, Some(&bias), Activation::Relu)
            .expect("cpu gemm_bias_act always succeeds");
        let metal_result = metal
            .gemm_bias_act(&a, &b, Some(&bias), Activation::Relu)
            .expect("metal gemm_bias_act (broadcast fallback) must succeed");
        backend_cpu::parity::assert_parity(
            &format!("gemm_bias_act broadcast-fallback cpu-metal parity shape={shape:?}"),
            metal_result.as_slice().expect("contiguous"),
            cpu_result.as_slice().expect("contiguous"),
        );
    }
}

/// bias 形状不整合（`[n]`／ブロードキャスト可能形状のいずれでもない）は
/// CPU・Metal とも同じ `BackendError::ShapeMismatch` を返すことを確認する
/// （バックエンド間で経路依存の挙動差を作らない契約。`ops.rs`
/// モジュール冒頭コメント参照）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_bias_act_rejects_incompatible_bias_shape_like_cpu() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("valid tensor");
    let bad_bias = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).expect("valid tensor");

    assert!(matches!(
        cpu.gemm_bias_act(&a, &b, Some(&bad_bias), Activation::None),
        Err(tensor_core::device::BackendError::ShapeMismatch(_))
    ));
    assert!(matches!(
        metal.gemm_bias_act(&a, &b, Some(&bad_bias), Activation::None),
        Err(tensor_core::device::BackendError::ShapeMismatch(_))
    ));
}

/// ブロードキャスト bias（`[1]`／`[1, n]`。非融合合成へフォールバックする
/// 経路）かつ `m`／`n`／`k` のいずれかがゼロの CPU-Metal 一致
/// （`gemm_bias_act_broadcast_fallback_matches_cpu` の非ゼロ形状に対し、
/// ゼロ次元専用の回帰。`self.gemm`〈`ZeroDimension` を拒否〉ではなく
/// CPU／CUDA と同じゼロ初期化結果を返す契約になったことを確認する。
/// Cursor Bugbot 指摘。PR #717 レビュースレッド
/// `discussion_r3795178880`）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_bias_act_broadcast_fallback_zero_dim_matches_cpu() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    // (m, n, k, bias_shape) のうち、いずれか 1 軸がゼロで
    // `gemm_bias_act_route` が `ComposedFallback` を選ぶ形状を網羅する。
    let cases: &[(usize, usize, usize, Vec<usize>)] = &[
        (0, 8, 6, vec![1]),
        (0, 8, 6, vec![1, 8]),
        (10, 0, 6, vec![1]),
        (10, 6, 0, vec![1]),
        (10, 6, 0, vec![1, 6]),
    ];

    for (m, n, k, bias_shape) in cases.iter().cloned() {
        let a =
            Tensor::new(Xorshift64Star::new(801).fill_vec(m * k), &[m, k]).expect("valid tensor");
        let b =
            Tensor::new(Xorshift64Star::new(802).fill_vec(k * n), &[k, n]).expect("valid tensor");
        let bias_len = bias_shape.iter().product();
        let bias = Tensor::new(Xorshift64Star::new(803).fill_vec(bias_len), &bias_shape)
            .expect("valid tensor");

        let cpu_result = cpu
            .gemm_bias_act(&a, &b, Some(&bias), Activation::Relu)
            .expect("cpu gemm_bias_act always succeeds");
        let metal_result = metal
            .gemm_bias_act(&a, &b, Some(&bias), Activation::Relu)
            .unwrap_or_else(|e| {
                panic!(
                    "metal gemm_bias_act (broadcast fallback, zero-dim m={m} n={n} k={k} \
                     bias_shape={bias_shape:?}) must succeed like cpu/cuda: {e}"
                )
            });
        assert_eq!(metal_result.shape(), cpu_result.shape());
        backend_cpu::parity::assert_parity(
            &format!(
                "gemm_bias_act broadcast-fallback zero-dim cpu-metal parity \
                 m={m} n={n} k={k} bias_shape={bias_shape:?}"
            ),
            metal_result.as_slice().expect("contiguous"),
            cpu_result.as_slice().expect("contiguous"),
        );
    }
}
