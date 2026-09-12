//! イシュー #1596: LayerNorm 順伝播カーネル（MSL・simdgroup 内
//! reduction・persistent threadgroup）の CPU-Metal 数値一致検証。
//!
//! `tests/rmsnorm_parity.rs` と同じ構成方針を踏襲する: Metal 実機
//! （Apple Silicon）依存のため `#![cfg(target_os = "macos")]` でファイル
//! 全体を macOS 限定にし、各テストに `#[ignore]` を付けて通常 CI では
//! 実行しない（`backend_ops_layer_norm_non_final_axis_is_unsupported`
//! は Metal デバイスに触れないため例外的に `#[ignore]` なし）。判定式・
//! 許容誤差は再定義せず `fandhe_ai_backend_cpu::parity` を唯一の参照と
//! する（`.claude/rules/coding-rust.md`）。
//!
//! 実行コマンド（Mac 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test layer_norm_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_backend_metal::{MetalContext, MetalLayerNorm};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

/// テスト専用 `f64` 参照実装（GPU の Neumaier + scale/ssq 方式・CPU の
/// `f64` 逐次和のいずれとも独立した実装で突き合わせることで、両実装
/// 共通のバグを検出できるようにする）。
fn f64_layer_norm_reference(
    x: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    eps: f32,
    rows: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    if hidden == 0 {
        return out;
    }
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let mean: f64 = row.iter().map(|&v| v as f64).sum::<f64>() / hidden as f64;
        let var: f64 = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / hidden as f64;
        let rstd = 1.0f64 / (var + eps as f64).sqrt();
        let out_row = &mut out[r * hidden..(r + 1) * hidden];
        for i in 0..hidden {
            let mut xhat = ((row[i] as f64 - mean) * rstd) as f32;
            if let Some(w) = w {
                xhat *= w[i];
            }
            if let Some(b) = b {
                xhat += b[i];
            }
            out_row[i] = xhat;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn assert_layer_norm_parity(
    ctx: &MetalContext,
    layer_norm: &MetalLayerNorm,
    seed_x: u64,
    seed_w: u64,
    seed_b: u64,
    rows: usize,
    hidden: usize,
    with_weight: bool,
    with_bias: bool,
    eps: f32,
) {
    let x_data = Xorshift64Star::new(seed_x).fill_vec(rows * hidden);
    let w_data = if with_weight {
        Some(Xorshift64Star::new(seed_w).fill_vec(hidden))
    } else {
        None
    };
    let b_data = if with_bias {
        Some(Xorshift64Star::new(seed_b).fill_vec(hidden))
    } else {
        None
    };

    let gpu_out = layer_norm
        .run_layer_norm_f32(
            ctx,
            &x_data,
            w_data.as_deref(),
            b_data.as_deref(),
            eps,
            rows,
            hidden,
        )
        .expect("MetalLayerNorm::run_layer_norm_f32 must succeed on Metal-equipped test runner");
    let expected = f64_layer_norm_reference(
        &x_data,
        w_data.as_deref(),
        b_data.as_deref(),
        eps,
        rows,
        hidden,
    );

    assert_parity(
        &format!(
            "layer_norm rows={rows} hidden={hidden} with_weight={with_weight} with_bias={with_bias} eps={eps}"
        ),
        &gpu_out,
        &expected,
    );
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn layer_norm_matches_f64_reference_across_shapes_and_affine_combinations() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let layer_norm = MetalLayerNorm::new(&ctx).expect("LayerNorm パイプラインの構築に失敗した");

    let hidden_cases: &[usize] = &[1, 3, 4, 5, 7, 8, 17, 33, 128, 4097];
    let rows_cases: &[usize] = &[1, 2, 5];
    let mut seed = 2000u64;
    for &hidden in hidden_cases {
        for &rows in rows_cases {
            for with_weight in [false, true] {
                for with_bias in [false, true] {
                    seed += 1;
                    assert_layer_norm_parity(
                        &ctx,
                        &layer_norm,
                        seed,
                        seed + 500,
                        seed + 900,
                        rows,
                        hidden,
                        with_weight,
                        with_bias,
                        1e-5,
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn layer_norm_extreme_values_no_nan_inf() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let layer_norm = MetalLayerNorm::new(&ctx).expect("LayerNorm パイプラインの構築に失敗した");

    let x = vec![1e30f32, -1e30, 1e30, -1e30, 1e-30, -1e-30, 0.0, 0.0];
    let out = layer_norm
        .run_layer_norm_f32(&ctx, &x, None, None, 1e-5, 2, 4)
        .expect("run_layer_norm_f32 must succeed");
    for &v in &out {
        assert!(v.is_finite(), "expected finite layer_norm output, got {v}");
    }
}

/// 極端な `eps`（`f32::MAX` 級。`ln_finalize_rstd` の疑似要素トリック
/// 〈`sqrt(eps)*sqrt(n)`〉が中間 overflow を避けることの実機確認）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn layer_norm_extreme_eps_does_not_overflow() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let layer_norm = MetalLayerNorm::new(&ctx).expect("LayerNorm パイプラインの構築に失敗した");

    let x_data = Xorshift64Star::new(3001).fill_vec(4 * 17);
    let out = layer_norm
        .run_layer_norm_f32(&ctx, &x_data, None, None, 1e30, 4, 17)
        .expect("run_layer_norm_f32 must succeed");
    for &v in &out {
        assert!(v.is_finite(), "expected finite layer_norm output, got {v}");
    }
}

/// NaN 伝播（行内に NaN が 1 つでもあれば行全体が NaN。`rmsnorm_parity.rs`
/// と同じ意味論契約）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn layer_norm_propagates_nan_for_row_with_nan_element() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let layer_norm = MetalLayerNorm::new(&ctx).expect("LayerNorm パイプラインの構築に失敗した");

    for hidden in [4usize, 65] {
        let mut x = vec![1.0f32; hidden];
        x[0] = f32::NAN;
        let out = layer_norm
            .run_layer_norm_f32(&ctx, &x, None, None, 1e-5, 1, hidden)
            .expect("run_layer_norm_f32 must succeed");
        assert!(
            out.iter().all(|v| v.is_nan()),
            "hidden={hidden}: NaN 要素を含む行の出力が NaN へ伝播していない: {out:?}"
        );
    }
}

// --- BackendOps::layer_norm 独立エントリ ---

/// `MetalBackendOps::layer_norm` が `MetalLayerNorm::run_layer_norm_f32`
/// と bit 同一であること（同一カーネルへのディスパッチであり別実装では
/// ないことの確認）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn backend_ops_layer_norm_is_bit_identical_to_metal_layer_norm() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let layer_norm = MetalLayerNorm::new(&ctx).expect("LayerNorm パイプラインの構築に失敗した");

    let rows = 3usize;
    let hidden = 17usize;
    let x_data = Xorshift64Star::new(6001).fill_vec(rows * hidden);
    let w_data = Xorshift64Star::new(6002).fill_vec(hidden);
    let b_data = Xorshift64Star::new(6003).fill_vec(hidden);
    let x = Tensor::new(x_data.clone(), &[rows, hidden]).expect("valid tensor");
    let w = Tensor::new(w_data.clone(), &[hidden]).expect("valid tensor");
    let b = Tensor::new(b_data.clone(), &[hidden]).expect("valid tensor");

    let metal = fandhe_ai_backend_metal::MetalBackendOps::new();
    let via_ops = metal
        .layer_norm(&x, Some(&w), Some(&b), 1e-5)
        .expect("BackendOps::layer_norm must succeed on Metal-equipped test runner");
    let via_kernel = layer_norm
        .run_layer_norm_f32(
            &ctx,
            &x_data,
            Some(&w_data),
            Some(&b_data),
            1e-5,
            rows,
            hidden,
        )
        .expect("MetalLayerNorm::run_layer_norm_f32 must succeed");

    assert_eq!(via_ops.shape(), &[rows, hidden]);
    assert_eq!(
        via_ops.as_slice().expect("contiguous"),
        via_kernel.as_slice()
    );
}

/// `MetalBackendOps::layer_norm` を CPU 参照実装と実機で直接
/// `assert_parity` 突合する（形状網羅）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn backend_ops_layer_norm_matches_cpu_reference_across_shapes() {
    let metal = fandhe_ai_backend_metal::MetalBackendOps::new();
    let cpu = fandhe_ai_backend_cpu::CpuBackendOps::new();

    let rows_cases: &[usize] = &[1, 3, 17];
    let hidden_cases: &[usize] = &[1, 31, 32, 33, 1024, 4097];
    let mut seed = 7000u64;
    for &rows in rows_cases {
        for &hidden in hidden_cases {
            seed += 1;
            let x_data = Xorshift64Star::new(seed).fill_vec(rows * hidden);
            let x = Tensor::new(x_data, &[rows, hidden]).expect("valid tensor");

            let gpu_out = metal
                .layer_norm(&x, None, None, 1e-5)
                .expect("BackendOps::layer_norm must succeed on Metal-equipped test runner");
            let cpu_out = cpu
                .layer_norm(&x, None, None, 1e-5)
                .expect("BackendOps::layer_norm must succeed on CPU");

            assert_eq!(gpu_out.shape(), &[rows, hidden]);
            assert_parity(
                &format!(
                    "BackendOps::layer_norm metal-cpu direct parity rows={rows} hidden={hidden}"
                ),
                gpu_out.as_slice().expect("contiguous"),
                cpu_out.as_slice().expect("contiguous"),
            );
        }
    }
}
