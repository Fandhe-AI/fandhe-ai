//! イシュー #1952: `crate::log_softmax_backward::MetalLogSoftmaxBackward`
//! （`log_softmax` backward）の CPU-Metal 数値一致検証。
//!
//! `MetalBackendOps::log_softmax_backward` への結線は同イシューで完了
//! 済み（`ops.rs`）。本ファイルは起動 API 直叩き（`run_f32`）と
//! `BackendOps` 経由の両方を検証する（`reduce_parity.rs` と同方針）。
//!
//! **数値契約**: `Σ_dim(g)` の縮約・`exp(y)` との乗算・`g` からの減算
//! （binary64 ソフトウェアエミュレーション連鎖）は CPU 参照実装
//! （`fandhe_ai_autodiff::grad::log_softmax_vjp_along` と同一式）に
//! 対し **REQ-2 統一複合判定**（相対誤差 1e-3 未満 または 絶対誤差
//! 1e-5 未満）で検証する（`exp(y)` 自体の丸めは Metal `precise::exp`
//! とホスト `f32::exp` で bit 一致が保証されないため。`shaders/
//! log_softmax_backward.metal`／`crate::log_softmax_backward_model`
//! doc 参照）。ただし `y=0`（`exp(0)=1.0` が厳密丸め）・`y=-inf`
//! （`exp(-inf)=0.0` が厳密丸め）の行では `exp` 自体の丸め差が
//! 生じないため **bit 完全一致**を主張し検証する。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する
//! （`reduce_parity.rs` と同方針。`#![cfg(target_os = "macos")]` に
//! より Linux CI ではコンパイル対象外・`#[ignore]` により通常の
//! `cargo test` からも除外される）。
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
//! cargo test -p fandhe-ai-backend-metal --release --test log_softmax_backward_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_backend_metal::{MetalBackendOps, MetalContext, MetalLogSoftmaxBackward};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn gen_data(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Xorshift64Star::new(seed);
    (0..n).map(|_| (rng.next_f32() - 0.5) * 8.0).collect()
}

/// `crates/autodiff/src/grad.rs::log_softmax_vjp_along` と同一式の
/// スタンドアロン CPU 参照実装（`autodiff` は Metal クレートへ依存
/// できないため複製する。`crates/autodiff/tests/
/// log_softmax_backward_dispatch.rs` の `common::log_softmax_vjp_along`
/// と同型の複製）。
fn cpu_log_softmax_vjp_along(y: &[f32], g: &[f32], shape: &[usize], dim: usize) -> Vec<f32> {
    let outer: usize = shape[..dim].iter().product();
    let axis_len = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let mut out = vec![0f32; y.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut sum_acc: f64 = 0.0;
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                sum_acc += g[idx] as f64;
            }
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                let d = g[idx] as f64 - (y[idx].exp() as f64) * sum_acc;
                out[idx] = d as f32;
            }
        }
    }
    out
}

/// `y`（CPU `eval` 相当。`torch.log_softmax` の実際の forward 値）を
/// `logits` から求める（`fandhe_ai_autodiff::eval::log_softmax_along`
/// を直接呼べないため——`autodiff` は Metal へ依存できない設計上の
/// 制約——数式的に同値な素朴な実装で代替する。オーバーフロー対策の
/// max 減算込み）。
fn log_softmax_forward(logits: &[f32], shape: &[usize], dim: usize) -> Vec<f32> {
    let outer: usize = shape[..dim].iter().product();
    let axis_len = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let mut out = vec![0f32; logits.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut max_v = f32::NEG_INFINITY;
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                if logits[idx] > max_v {
                    max_v = logits[idx];
                }
            }
            let mut sum_exp: f64 = 0.0;
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                sum_exp += ((logits[idx] - max_v) as f64).exp();
            }
            let log_sum_exp = sum_exp.ln() as f32;
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                out[idx] = (logits[idx] - max_v) - log_sum_exp;
            }
        }
    }
    out
}

/// `MetalLogSoftmaxBackward::run_f32` が CPU 参照実装と REQ-2 統一
/// 複合判定で一致することを複数 rank・複数 `dim`・現実的な入力
/// （ランダム logits → `log_softmax` forward → 適当な上流勾配）で
/// 確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn metal_log_softmax_backward_matches_cpu_composite_judgment() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let kernel =
        MetalLogSoftmaxBackward::new(&ctx).expect("MetalLogSoftmaxBackward::new に失敗した");

    let cases: &[(&[usize], usize)] = &[
        (&[5], 0),
        (&[4, 7], 1),
        (&[4, 7], 0),
        (&[3, 4, 5], 1),
        (&[3, 4, 5], 2),
        (&[3, 4, 5], 0),
        (&[2, 3, 4, 5], 2),
    ];

    for &(shape, dim) in cases {
        let numel: usize = shape.iter().product();
        let logits = gen_data(numel, 0x1952_0100 + numel as u64 + dim as u64);
        let y = log_softmax_forward(&logits, shape, dim);
        let g = gen_data(numel, 0x1952_0200 + numel as u64 + dim as u64);

        let outer: usize = shape[..dim].iter().product();
        let axis_len = shape[dim];
        let inner: usize = shape[dim + 1..].iter().product();

        let metal_out = kernel
            .run_f32(&ctx, &y, &g, outer, axis_len, inner)
            .expect("metal log_softmax_backward must succeed on Metal-equipped runner");
        let cpu_out = cpu_log_softmax_vjp_along(&y, &g, shape, dim);

        assert_parity(
            &format!("log_softmax_backward: shape={shape:?} dim={dim}"),
            &metal_out,
            &cpu_out,
        );

        // run-to-run 決定性。
        let metal_out2 = kernel
            .run_f32(&ctx, &y, &g, outer, axis_len, inner)
            .expect("metal log_softmax_backward run2");
        assert_eq!(
            metal_out2, metal_out,
            "shape={shape:?} dim={dim}: run-to-run で bit 同一のはず"
        );
    }
}

/// `y=0` 行（`exp(0)=1.0` が厳密丸め）では `exp` の丸め差が生じない
/// ため、CPU 参照実装と **bit 完全一致**することを確認する
/// （`crate::log_softmax_backward_model` モジュール doc「bit 一致を
/// 主張する範囲」の実機裏付け）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn metal_log_softmax_backward_matches_cpu_bit_exact_when_y_is_zero() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let kernel =
        MetalLogSoftmaxBackward::new(&ctx).expect("MetalLogSoftmaxBackward::new に失敗した");

    let shape: &[usize] = &[4, 5];
    let dim = 1;
    let y = vec![0.0f32; 20];
    let g = gen_data(20, 0x1952_0300);

    let outer: usize = shape[..dim].iter().product();
    let axis_len = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();

    let metal_out = kernel
        .run_f32(&ctx, &y, &g, outer, axis_len, inner)
        .expect("metal log_softmax_backward must succeed");
    let cpu_out = cpu_log_softmax_vjp_along(&y, &g, shape, dim);

    for (idx, (&m, &c)) in metal_out.iter().zip(cpu_out.iter()).enumerate() {
        assert_eq!(
            m.to_bits(),
            c.to_bits(),
            "idx={idx}: y=0 行は bit 完全一致のはず（metal={m:?}, cpu={c:?}）"
        );
    }
}

/// `y=-inf` 行（`exp(-inf)=0.0` が厳密丸め）でも同様に **bit 完全
/// 一致**することを確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn metal_log_softmax_backward_matches_cpu_bit_exact_when_y_is_neg_infinity() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let kernel =
        MetalLogSoftmaxBackward::new(&ctx).expect("MetalLogSoftmaxBackward::new に失敗した");

    let shape: &[usize] = &[3, 4];
    let dim = 1;
    let y = vec![f32::NEG_INFINITY; 12];
    let g = gen_data(12, 0x1952_0400);

    let outer: usize = shape[..dim].iter().product();
    let axis_len = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();

    let metal_out = kernel
        .run_f32(&ctx, &y, &g, outer, axis_len, inner)
        .expect("metal log_softmax_backward must succeed");
    // `exp(-inf)==0.0` のため `dx == g`（乗算項が厳密ゼロ）。
    for (idx, (&m, &expected)) in metal_out.iter().zip(g.iter()).enumerate() {
        assert_eq!(
            m.to_bits(),
            expected.to_bits(),
            "idx={idx}: y=-inf 行は dx==g のはず（metal={m:?}, g={expected:?}）"
        );
    }
}

/// `MetalBackendOps::log_softmax_backward`（`BackendOps` trait 経由。
/// `Op::LogSoftmax` の VJP が実際に到達する入口）が起動 API 直叩きと
/// 同じ結果を返すことを確認する（`Var::log_softmax` backward の実機
/// 到達確認・非最終軸の受理確認込み）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn backend_ops_log_softmax_backward_matches_direct_api_call() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let kernel =
        MetalLogSoftmaxBackward::new(&ctx).expect("MetalLogSoftmaxBackward::new に失敗した");
    let ops = MetalBackendOps;

    let shape: &[usize] = &[3, 4, 5];
    let dim = 0; // 非最終軸。
    let numel: usize = shape.iter().product();
    let logits = gen_data(numel, 0x1952_0500);
    let y = log_softmax_forward(&logits, shape, dim);
    let g = gen_data(numel, 0x1952_0600);

    let outer: usize = shape[..dim].iter().product();
    let axis_len = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();

    let direct = kernel
        .run_f32(&ctx, &y, &g, outer, axis_len, inner)
        .expect("direct run_f32 must succeed");

    let out_t = Tensor::new(y.clone(), shape).expect("tensor");
    let upstream_t = Tensor::new(g.clone(), shape).expect("tensor");
    let via_ops = ops
        .log_softmax_backward(&out_t, &upstream_t, dim)
        .expect("MetalBackendOps::log_softmax_backward must succeed");
    let via_ops_slice = via_ops.as_slice().expect("contiguous");

    assert_eq!(
        via_ops_slice, direct,
        "BackendOps 経由と直接呼び出しは同一結果のはず"
    );
}
