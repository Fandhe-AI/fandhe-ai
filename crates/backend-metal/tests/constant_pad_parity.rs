//! イシュー #1756: `BackendOps::pad`（`torch.nn.functional.pad
//! (mode='constant')` 相当）の CPU-Metal 数値一致検証（CUDA 側と同型の
//! Metal 対応版）。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する
//! （`tests/gather_scatter_parity.rs` と同方針。`#![cfg(target_os =
//! "macos")]` により Linux CI ではコンパイル対象外になり、`#[ignore]`
//! により通常の `cargo test` からも除外される）。
//!
//! pad は算術を含まない純粋なコピー演算のため CPU-Metal 間で **bit
//! 完全一致**を検証する（`value` が NaN の場合のみクラス一致）。
//! `Var::pad` の backward（narrow view 連鎖）が CPU テープと bit 一致
//! することも確認する。
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
//! cargo test -p fandhe-ai-backend-metal --release --test constant_pad_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_autodiff::Tape;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::{BackendOps, ShapeError, Tensor};

fn assert_pad_parity(seed: u64, in_shape: &[usize], pads: &[(usize, usize)], value: f32) {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let numel_in: usize = in_shape.iter().product();
    let input =
        Tensor::new(Xorshift64Star::new(seed).fill_vec(numel_in), in_shape).expect("valid tensor");

    let cpu_out = cpu.pad(&input, pads, value).expect("cpu pad");
    let metal_out = metal.pad(&input, pads, value).expect("metal pad");

    assert_eq!(cpu_out.shape(), metal_out.shape());
    let cpu_slice = cpu_out.as_slice().expect("contiguous");
    let metal_slice = metal_out.as_slice().expect("contiguous");
    assert_eq!(
        cpu_slice, metal_slice,
        "pad(in_shape={in_shape:?}, pads={pads:?}, value={value}): CPU と Metal は bit 完全一致のはず"
    );

    // run-to-run で bit 同一（決定的カーネル）。
    let metal_out2 = metal.pad(&input, pads, value).expect("metal pad rerun");
    assert_eq!(
        metal_out2.as_slice().expect("contiguous"),
        metal_slice,
        "pad: run-to-run で bit 同一のはず"
    );
}

/// `(in_shape, pads)` の組（clippy `type_complexity` 回避のためのエイリ
/// アス。本ファイル限定の局所定義）。
type ShapePads<'a> = (&'a [usize], &'a [(usize, usize)]);

/// 実機必須の形状網羅（受け入れ条件の本体）。
#[test]
#[ignore = "Apple Silicon 実機必須"]
fn pad_matches_cpu_across_shapes() {
    let shapes_pads: &[ShapePads] = &[
        (&[4], &[(1, 2)]),
        (&[2, 3], &[(1, 0), (0, 1)]),
        (&[2, 3], &[(0, 0), (0, 0)]), // 恒等
        (&[2, 3, 4], &[(1, 1), (0, 2), (2, 0)]),
        (&[1 << 12, 3], &[(1, 0), (0, 0)]), // threadgroup 境界をまたぐ大きさ
    ];
    let mut seed = 30_000u64;
    for &(in_shape, pads) in shapes_pads {
        seed += 7;
        assert_pad_parity(seed, in_shape, pads, -2.5);
    }

    // 非 contiguous な input（transpose view）を渡しても contiguous 化後
    // に一致する。
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();
    let base = Tensor::new((0..12).map(|v| v as f32).collect(), &[3, 4]).expect("valid tensor");
    let transposed = base.transpose(0, 1).expect("valid transpose");
    let pads = [(1usize, 0usize), (0usize, 1usize)];
    let cpu_out = cpu.pad(&transposed, &pads, 0.0).expect("cpu pad");
    let metal_out = metal.pad(&transposed, &pads, 0.0).expect("metal pad");
    assert_eq!(
        metal_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous")
    );

    // NaN／±inf の通過（入力）・`value` に NaN を使うクラス一致確認。
    let input_special = Tensor::new(
        vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0],
        &[2, 2],
    )
    .expect("valid tensor");
    let pads_special = [(1usize, 0usize), (0usize, 1usize)];
    let cpu_special = cpu
        .pad(&input_special, &pads_special, f32::NAN)
        .expect("cpu pad");
    let metal_special = metal
        .pad(&input_special, &pads_special, f32::NAN)
        .expect("metal pad");
    let cpu_special_slice = cpu_special.as_slice().expect("contiguous");
    let metal_special_slice = metal_special.as_slice().expect("contiguous");
    for (c, g) in cpu_special_slice.iter().zip(metal_special_slice.iter()) {
        if c.is_nan() {
            assert!(g.is_nan(), "NaN must remain NaN through pad");
        } else {
            assert_eq!(c, g);
        }
    }
}

/// 空入力 → 非空出力（value で埋まる）が CPU と Metal で一致する。
#[test]
#[ignore = "Apple Silicon 実機必須"]
fn pad_empty_input_matches_cpu() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();
    let empty_input = Tensor::new(Vec::<f32>::new(), &[0, 3]).expect("valid tensor");
    let cpu_out = cpu
        .pad(&empty_input, &[(2, 0), (0, 0)], 5.0)
        .expect("cpu pad on empty input");
    let metal_out = metal
        .pad(&empty_input, &[(2, 0), (0, 0)], 5.0)
        .expect("metal pad on empty input");
    assert_eq!(
        metal_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous")
    );
}

/// `Var::pad` の backward（narrow view 連鎖）が CPU テープと bit 一致
/// することを確認する。
#[test]
#[ignore = "Apple Silicon 実機必須"]
fn pad_backward_matches_cpu_tape() {
    let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");

    let cpu_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let cpu_x = cpu_tape.var(&x);
    let cpu_padded = cpu_x.pad(&[(1, 0), (0, 1)], -1.0).expect("cpu pad");
    let cpu_loss = cpu_padded.sum(None).expect("cpu sum");
    let cpu_grads = cpu_tape.backward(&cpu_loss).expect("cpu backward");
    let cpu_dx = cpu_grads
        .get(&cpu_x)
        .expect("grad exists")
        .expect("reaches loss");

    let metal_tape = Tape::new_with_ops(Box::new(MetalBackendOps::new()));
    let metal_x = metal_tape.var(&x);
    let metal_padded = metal_x.pad(&[(1, 0), (0, 1)], -1.0).expect("metal pad");
    let metal_loss = metal_padded.sum(None).expect("metal sum");
    let metal_grads = metal_tape.backward(&metal_loss).expect("metal backward");
    let metal_dx = metal_grads
        .get(&metal_x)
        .expect("grad exists")
        .expect("reaches loss");

    assert_eq!(
        cpu_dx.as_slice().expect("contiguous"),
        metal_dx.as_slice().expect("contiguous"),
        "pad backward: CPU と Metal は bit 完全一致のはず"
    );
}

/// `MetalBackendOps::pad` が `pad_out_shape` による shape 再検査を実装
/// 側でも行い、不一致を `BackendError::ShapeMismatch` として fail-closed
/// に拒否することを確認する（`.claude/rules/security.md` A08）。
#[test]
#[ignore = "Apple Silicon 実機必須"]
fn backend_ops_pad_rejects_rank_mismatch() {
    let metal = MetalBackendOps::new();
    let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let err = metal.pad(&input, &[(1, 0)], 0.0).unwrap_err();
    assert!(matches!(
        err,
        fandhe_ai_tensor_core::device::BackendError::ShapeMismatch(ShapeError::RankMismatch { .. })
    ));
}
