//! イシュー #604: 融合 RMSNorm 順伝播カーネル（MSL・warp 内 reduction・
//! persistent threadgroup）の CPU-Metal 数値一致検証。
//!
//! `tests/cpu_metal_parity.rs`（GEMM）と同じ構成方針を踏襲する: Metal 実機
//! （Apple Silicon）依存のため `#![cfg(target_os = "macos")]` でファイル
//! 全体を macOS 限定にし、各テストに `#[ignore]` を付けて通常 CI では
//! 実行しない。判定式・許容誤差は再定義せず `backend_cpu::parity` を唯一の
//! 参照とする（`.claude/rules/coding-rust.md`）。
//!
//! CPU 参照実装は本ファイル内のテスト専用関数（`f32::mul_add` 使用。CUDA
//! 側 `tests/rmsnorm_parity.rs::cpu_rmsnorm_reference` と同一意味論）で
//! ある。
//!
//! 実行コマンド（Mac 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p backend-metal --release --test rmsnorm_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use backend_cpu::parity::assert_parity;
use backend_metal::{MetalContext, MetalRmsNorm};
use bench_harness::rng::Xorshift64Star;

/// テスト専用 CPU 参照実装（`f32::mul_add` を使用し、GPU 側 `fma()` と
/// 丸め方針を揃える）。`out = x * rsqrt(mean(x^2, axis=-1) + eps) * w`
/// （`w` が `None` の場合は乗算をスキップ）。CUDA 側
/// `backend-cuda::tests::rmsnorm_parity::cpu_rmsnorm_reference` と同一
/// 意味論（実装計画 §6.2「CPU 参照は CUDA 側 `cpu_rmsnorm_reference` と
/// 同一意味論」）。
fn cpu_rmsnorm_reference(
    x: &[f32],
    w: Option<&[f32]>,
    eps: f32,
    rows: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    if hidden == 0 {
        return out;
    }
    let inv_n = 1.0f32 / hidden as f32;
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let mut acc = 0.0f32;
        for &v in row {
            acc = v.mul_add(v, acc);
        }
        let rstd = 1.0f32 / (acc.mul_add(inv_n, eps)).sqrt();
        let out_row = &mut out[r * hidden..(r + 1) * hidden];
        for i in 0..hidden {
            let mut normed = row[i] * rstd;
            if let Some(w) = w {
                normed *= w[i];
            }
            out_row[i] = normed;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn assert_rmsnorm_parity(
    ctx: &MetalContext,
    rmsnorm: &MetalRmsNorm,
    seed_x: u64,
    seed_w: u64,
    rows: usize,
    hidden: usize,
    with_weight: bool,
    eps: f32,
) {
    let x_data = Xorshift64Star::new(seed_x).fill_vec(rows * hidden);
    let w_data = if with_weight {
        Some(Xorshift64Star::new(seed_w).fill_vec(hidden))
    } else {
        None
    };

    let gpu_out = rmsnorm
        .run_rmsnorm_f32(ctx, &x_data, w_data.as_deref(), eps, rows, hidden)
        .expect("MetalRmsNorm::run_rmsnorm_f32 must succeed on Metal-equipped test runner");
    let cpu_out = cpu_rmsnorm_reference(&x_data, w_data.as_deref(), eps, rows, hidden);

    assert_eq!(gpu_out.len(), cpu_out.len());
    assert_parity(
        &format!(
            "rmsnorm cpu-metal parity rows={rows} hidden={hidden} with_weight={with_weight} \
             eps={eps}"
        ),
        &gpu_out,
        &cpu_out,
    );
}

/// 実機必須の形状網羅（受け入れ条件の本体。実装計画 §6.2「形状網羅」）。
///
/// hidden の網羅: 8（極小・4 の倍数）・9（極小・非倍数）・1024（1 パス
/// 中位）・4096（1 パス上限〈`ONEPASS_MAX_HIDDEN`〉ちょうど）・4097（2 パス
/// 強制・非倍数）・8192（2 パス）。rows の網羅: 1・3・33（persistent grid
/// を超えて行ループを複周回させる）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn rmsnorm_matches_cpu_across_shapes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let rmsnorm = MetalRmsNorm::new(&ctx).expect("RMSNorm パイプラインの構築に失敗した");

    let hiddens: &[usize] = &[8, 9, 1024, 4096, 4097, 8192];
    let rows_cases: &[usize] = &[1, 3, 33];

    let mut seed = 1000u64;
    for &hidden in hiddens {
        for &rows in rows_cases {
            for &with_weight in &[false, true] {
                seed += 1;
                let seed_w = seed + 500;
                assert_rmsnorm_parity(
                    &ctx,
                    &rmsnorm,
                    seed,
                    seed_w,
                    rows,
                    hidden,
                    with_weight,
                    1e-5,
                );
            }
        }
    }

    // eps=0.0（`run_fused` 経由の canonical プランと同じ eps。有限値の
    // 境界ケース）。
    assert_rmsnorm_parity(&ctx, &rmsnorm, 9001, 9002, 4, 256, false, 0.0);

    // hidden=1（行長 1。ベクトル化経路〈hidden % 4 == 0〉に入らない
    // 最小ケース）。
    assert_rmsnorm_parity(&ctx, &rmsnorm, 9101, 9102, 5, 1, false, 1e-5);
}

/// `run_fused`（`ops.rs::MetalBackendOps::run_fused`）経由の canonical
/// RMSNorm プラン実行を CPU per-op 合成（`mul → sum → rsqrt → broadcast →
/// mul`）と突き合わせる（実装計画 §6.2「run_fused 経路の実機テスト」）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn rmsnorm_run_fused_matches_cpu_composed() {
    use tensor_core::{BackendOps, DType, FusedOpKind, FusionPlan, Tensor};

    let hidden = 16usize;
    let x_data = Xorshift64Star::new(9101).fill_vec(hidden);
    let x = Tensor::new(x_data.clone(), &[hidden]).expect("valid tensor");

    let ops = vec![
        FusedOpKind::Input { leaf_index: 0 },
        FusedOpKind::Mul { lhs: 0, rhs: 0 },
        FusedOpKind::Sum {
            input: 1,
            axis: None,
        },
        FusedOpKind::Rsqrt { input: 2 },
        FusedOpKind::Broadcast {
            input: 3,
            axis: None,
        },
        FusedOpKind::Mul { lhs: 4, rhs: 0 },
    ];
    let plan = FusionPlan::from_ops(ops, vec![hidden], DType::F32, 1)
        .expect("canonical RMSNorm plan must construct");

    let metal = backend_metal::MetalBackendOps::new();
    let fused_out = metal
        .run_fused(&plan, &[&x])
        .expect("MetalBackendOps::run_fused must succeed on Metal-equipped test runner");

    let sq: Vec<f32> = x_data.iter().map(|v| v * v).collect();
    let sum: f32 = sq.iter().sum();
    let rstd = 1.0f32 / sum.sqrt();
    let composed: Vec<f32> = x_data.iter().map(|v| v * rstd).collect();

    assert_eq!(fused_out.shape(), &[hidden]);
    assert_parity(
        "rmsnorm run_fused vs cpu composed (canonical plan)",
        fused_out.as_slice().expect("contiguous"),
        &composed,
    );
}
