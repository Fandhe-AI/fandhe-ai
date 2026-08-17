//! イシュー #604: online softmax カーネル（MSL・warp 内 reduction・
//! persistent threadgroup）の CPU-Metal 数値一致検証。
//!
//! `tests/rmsnorm_parity.rs` と同じ構成方針: Metal 実機（Apple Silicon）
//! 依存のため `#![cfg(target_os = "macos")]` でファイル全体を macOS 限定に
//! し、各テストに `#[ignore]` を付けて通常 CI では実行しない。
//!
//! **CUDA との parity 状況（重要）**: CUDA 側の online softmax（#594・
//! G-7）は本イシュー時点で OPEN のため、CUDA 直接の parity 相手は存在
//! しない。CPU 参照実装（本ファイル内の素朴な `exp(x - max(x)) / sum`
//! 実装）に対する REQ-2 統一複合判定 green のみを本テストの受け入れ対象
//! とし、CUDA↔Metal 実機横断の突合は #594 実装後・実機ツリー #408 系での
//! 別途検証に委ねる（`softmax.rs` ドキュメンテーションコメント参照）。
//!
//! 実行コマンド（Mac 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p backend-metal --release --test softmax_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use backend_cpu::parity::assert_parity;
use backend_metal::{MetalContext, MetalSoftmax};
use bench_harness::rng::Xorshift64Star;

/// テスト専用 CPU 参照実装（素朴な `exp(x - max(x)) / sum(exp(x -
/// max(x)))`。カーネル側の `log2(e)` 事前スケール＋`exp2` 実装とは異なる
/// 数式だが、数学的に同一の softmax を計算するため REQ-2 統一複合判定で
/// 突き合わせられる）。
fn cpu_softmax_reference(x: &[f32], rows: usize, hidden: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    if hidden == 0 {
        return out;
    }
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| (v - m).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let out_row = &mut out[r * hidden..(r + 1) * hidden];
        for i in 0..hidden {
            out_row[i] = exps[i] / sum;
        }
    }
    out
}

fn assert_softmax_parity(
    ctx: &MetalContext,
    softmax: &MetalSoftmax,
    seed_x: u64,
    rows: usize,
    hidden: usize,
) {
    let x_data = Xorshift64Star::new(seed_x).fill_vec(rows * hidden);

    let gpu_out = softmax
        .run_softmax_f32(ctx, &x_data, rows, hidden)
        .expect("MetalSoftmax::run_softmax_f32 must succeed on Metal-equipped test runner");
    let cpu_out = cpu_softmax_reference(&x_data, rows, hidden);

    assert_eq!(gpu_out.len(), cpu_out.len());
    assert_parity(
        &format!("softmax cpu-metal parity rows={rows} hidden={hidden}"),
        &gpu_out,
        &cpu_out,
    );
}

/// 実機必須の形状網羅（受け入れ条件の本体。実装計画 §6.2「形状網羅」）。
///
/// hidden の網羅: 8（極小）・1024（1 パス中位）・4096（1 パス上限
/// ちょうど）・4097（2 パス強制）・8192（2 パス）。rows の網羅: 1・3・33
/// （persistent grid を超えて行ループを複周回させる）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn softmax_matches_cpu_across_shapes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let softmax = MetalSoftmax::new(&ctx).expect("softmax パイプラインの構築に失敗した");

    let hiddens: &[usize] = &[8, 1024, 4096, 4097, 8192];
    let rows_cases: &[usize] = &[1, 3, 33];

    let mut seed = 2000u64;
    for &hidden in hiddens {
        for &rows in rows_cases {
            seed += 1;
            assert_softmax_parity(&ctx, &softmax, seed, rows, hidden);
        }
    }

    // hidden=1（行長 1。単一要素の softmax は常に 1.0）。
    assert_softmax_parity(&ctx, &softmax, 2101, 5, 1);
}

/// softmax 極値安定性（実装計画 §6.2「softmax 極値安定性」。#594 記載の
/// 観点と同一: 全要素同値・大きな正値／負値）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn softmax_is_numerically_stable_at_extreme_values() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let softmax = MetalSoftmax::new(&ctx).expect("softmax パイプラインの構築に失敗した");

    let hidden = 128usize;

    // 全要素同値（softmax は一様分布 1/hidden になるはず）。
    let uniform = vec![42.0f32; hidden];
    let gpu_out = softmax
        .run_softmax_f32(&ctx, &uniform, 1, hidden)
        .expect("uniform softmax must succeed");
    let cpu_out = cpu_softmax_reference(&uniform, 1, hidden);
    assert_parity("softmax uniform values", &gpu_out, &cpu_out);

    // 大きな正値（naive exp(x) は overflow するが online max 減算で回避）。
    let mut large_positive = vec![0.0f32; hidden];
    for (i, v) in large_positive.iter_mut().enumerate() {
        *v = 1.0e4 + i as f32;
    }
    let gpu_out = softmax
        .run_softmax_f32(&ctx, &large_positive, 1, hidden)
        .expect("large positive softmax must succeed");
    let cpu_out = cpu_softmax_reference(&large_positive, 1, hidden);
    assert_parity("softmax large positive values", &gpu_out, &cpu_out);

    // 大きな負値（naive exp(x) は underflow して全要素 0/0 になりうるが
    // online max 減算で回避）。
    let mut large_negative = vec![0.0f32; hidden];
    for (i, v) in large_negative.iter_mut().enumerate() {
        *v = -1.0e4 - i as f32;
    }
    let gpu_out = softmax
        .run_softmax_f32(&ctx, &large_negative, 1, hidden)
        .expect("large negative softmax must succeed");
    let cpu_out = cpu_softmax_reference(&large_negative, 1, hidden);
    assert_parity("softmax large negative values", &gpu_out, &cpu_out);
}

/// `run_fused`（`ops.rs::MetalBackendOps::run_fused`）経由の canonical
/// softmax プラン実行を CPU 参照実装と突き合わせる（実装計画 §6.2
/// 「run_fused 経路の実機テスト」）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn softmax_run_fused_matches_cpu_reference() {
    use tensor_core::{BackendOps, DType, FusedOpKind, FusionPlan, Tensor};

    let hidden = 32usize;
    let x_data = Xorshift64Star::new(2201).fill_vec(hidden);
    let x = Tensor::new(x_data.clone(), &[hidden]).expect("valid tensor");

    // canonical softmax プラン（axis: None。leaf 0=x, 1=Max{None}(0),
    // 2=Broadcast{None}(1), 3=Sub(0,2), 4=Exp(3), 5=Sum{None}(4),
    // 6=Broadcast{None}(5), 7=Div(4,6)）。
    let ops = vec![
        FusedOpKind::Input { leaf_index: 0 },
        FusedOpKind::Max {
            input: 0,
            axis: None,
        },
        FusedOpKind::Broadcast {
            input: 1,
            axis: None,
        },
        FusedOpKind::Sub { lhs: 0, rhs: 2 },
        FusedOpKind::Exp { input: 3 },
        FusedOpKind::Sum {
            input: 4,
            axis: None,
        },
        FusedOpKind::Broadcast {
            input: 5,
            axis: None,
        },
        FusedOpKind::Div { lhs: 4, rhs: 6 },
    ];
    let plan = FusionPlan::from_ops(ops, vec![hidden], DType::F32, 1)
        .expect("canonical softmax plan must construct");

    let metal = backend_metal::MetalBackendOps::new();
    let fused_out = metal
        .run_fused(&plan, &[&x])
        .expect("MetalBackendOps::run_fused must succeed on Metal-equipped test runner");

    let composed = cpu_softmax_reference(&x_data, 1, hidden);

    assert_eq!(fused_out.shape(), &[hidden]);
    assert_parity(
        "softmax run_fused vs cpu reference (canonical plan)",
        fused_out.as_slice().expect("contiguous"),
        &composed,
    );
}
