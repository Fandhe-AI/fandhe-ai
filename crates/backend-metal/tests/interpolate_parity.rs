//! イシュー #1757: `BackendOps::interpolate`（`torch.nn.functional.
//! interpolate(mode='nearest')` 相当）の CPU-Metal 数値一致検証
//! （CUDA 側 `interpolate_parity.rs` と同型の Metal 対応版）。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する
//! （`tests/gather_scatter_parity.rs` と同方針。`#![cfg(target_os =
//! "macos")]` により Linux CI ではコンパイル対象外になり、`#[ignore]`
//! により通常の `cargo test` からも除外される）。
//!
//! interpolate は算術を含まない純粋なコピー演算のため CPU-Metal 間で
//! **bit 完全一致**を検証する。`Var::interpolate` の backward
//! （scatter_add ベース VJP）が CPU テープと bit 一致することも
//! 確認する。
//!
//! Linux CI での型検査（実機なしでもコンパイル可能性を担保）:
//!
//! ```sh
//! cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin
//! ```
//!
//! 実行コマンド（Apple Silicon 実機。`--release` 推奨）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test interpolate_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_autodiff::Tape;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::{BackendOps, InterpolateMode, ShapeError, Tensor};

fn assert_interpolate_parity(seed: u64, in_shape: &[usize], size: &[usize]) {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let numel_in: usize = in_shape.iter().product();
    let input =
        Tensor::new(Xorshift64Star::new(seed).fill_vec(numel_in), in_shape).expect("valid tensor");

    let cpu_out = cpu
        .interpolate(&input, size, InterpolateMode::Nearest)
        .expect("cpu interpolate");
    let metal_out = metal
        .interpolate(&input, size, InterpolateMode::Nearest)
        .expect("metal interpolate");

    assert_eq!(cpu_out.shape(), metal_out.shape());
    let cpu_slice = cpu_out.as_slice().expect("contiguous");
    let metal_slice = metal_out.as_slice().expect("contiguous");
    assert_eq!(
        cpu_slice, metal_slice,
        "interpolate(in_shape={in_shape:?}, size={size:?}): CPU と Metal は bit 完全一致のはず"
    );

    // run-to-run で bit 同一（決定的カーネル）。
    let metal_out2 = metal
        .interpolate(&input, size, InterpolateMode::Nearest)
        .expect("metal interpolate rerun");
    assert_eq!(
        metal_out2.as_slice().expect("contiguous"),
        metal_slice,
        "interpolate: run-to-run で bit 同一のはず"
    );
}

/// 実機必須の形状網羅（受け入れ条件の本体）。
#[test]
#[ignore = "Apple Silicon 実機必須"]
fn interpolate_matches_cpu_across_shapes() {
    let cases: &[(&[usize], &[usize])] = &[
        (&[3], &[6]),          // 1-D 整数倍アップサンプル
        (&[8], &[3]),          // 1-D 非整数比ダウンサンプル
        (&[4], &[4]),          // 恒等サイズ
        (&[2, 3], &[7]),       // 先頭 batch 軸付き
        (&[2, 4, 4], &[9, 9]), // 2 軸空間（先頭 batch 軸付き）
        (&[1 << 12, 3], &[7]), // threadgroup 境界をまたぐ大きさ
        (&[5], &[1]),          // 極端ダウンサンプル
    ];
    let mut seed = 40_000u64;
    for &(in_shape, size) in cases {
        seed += 7;
        assert_interpolate_parity(seed, in_shape, size);
    }

    // 非 contiguous な input（transpose view）を渡しても contiguous 化
    // 後に一致する。
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();
    let base = Tensor::new((0..12).map(|v| v as f32).collect(), &[3, 4]).expect("valid tensor");
    let transposed = base.transpose(0, 1).expect("valid transpose");
    let cpu_out = cpu
        .interpolate(&transposed, &[6], InterpolateMode::Nearest)
        .expect("cpu interpolate");
    let metal_out = metal
        .interpolate(&transposed, &[6], InterpolateMode::Nearest)
        .expect("metal interpolate");
    assert_eq!(
        metal_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous")
    );

    // NaN／±inf の通過（算術を含まないコピー演算のため bit 単位で
    // そのまま伝播する）。
    let input_special = Tensor::new(vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0], &[4])
        .expect("valid tensor");
    let cpu_special = cpu
        .interpolate(&input_special, &[8], InterpolateMode::Nearest)
        .expect("cpu interpolate");
    let metal_special = metal
        .interpolate(&input_special, &[8], InterpolateMode::Nearest)
        .expect("metal interpolate");
    let cpu_special_slice = cpu_special.as_slice().expect("contiguous");
    let metal_special_slice = metal_special.as_slice().expect("contiguous");
    for (c, g) in cpu_special_slice.iter().zip(metal_special_slice.iter()) {
        if c.is_nan() {
            assert!(g.is_nan(), "NaN must remain NaN through interpolate");
        } else {
            assert_eq!(c.to_bits(), g.to_bits());
        }
    }
}

/// `Var::interpolate` の backward（scatter_add ベース VJP）が CPU
/// テープと bit 一致することを確認する。
#[test]
#[ignore = "Apple Silicon 実機必須"]
fn interpolate_backward_matches_cpu_tape() {
    let x = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).expect("valid tensor");

    let cpu_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let cpu_x = cpu_tape.var(&x);
    let cpu_out = cpu_x
        .interpolate(&[7], InterpolateMode::Nearest)
        .expect("cpu interpolate");
    let cpu_loss = cpu_out.sum(None).expect("cpu sum");
    let cpu_grads = cpu_tape.backward(&cpu_loss).expect("cpu backward");
    let cpu_dx = cpu_grads
        .get(&cpu_x)
        .expect("grad exists")
        .expect("reaches loss");

    let metal_tape = Tape::new_with_ops(Box::new(MetalBackendOps::new()));
    let metal_x = metal_tape.var(&x);
    let metal_out = metal_x
        .interpolate(&[7], InterpolateMode::Nearest)
        .expect("metal interpolate");
    let metal_loss = metal_out.sum(None).expect("metal sum");
    let metal_grads = metal_tape.backward(&metal_loss).expect("metal backward");
    let metal_dx = metal_grads
        .get(&metal_x)
        .expect("grad exists")
        .expect("reaches loss");

    assert_eq!(
        cpu_dx.as_slice().expect("contiguous"),
        metal_dx.as_slice().expect("contiguous"),
        "interpolate backward: CPU と Metal は bit 完全一致のはず"
    );
}

/// `MetalBackendOps::interpolate` が `interpolate_out_shape` による
/// shape 再検査を実装側でも行い、不一致を `BackendError::
/// ShapeMismatch` として fail-closed に拒否することを確認する
/// （`.claude/rules/security.md` A08）。
#[test]
#[ignore = "Apple Silicon 実機必須"]
fn backend_ops_interpolate_rejects_zero_spatial_axis() {
    let metal = MetalBackendOps::new();
    let input = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let err = metal
        .interpolate(&input, &[0], InterpolateMode::Nearest)
        .unwrap_err();
    assert!(matches!(
        err,
        fandhe_ai_tensor_core::device::BackendError::ShapeMismatch(
            ShapeError::ShapeMismatch { .. }
        )
    ));
}
