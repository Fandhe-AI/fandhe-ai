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
//! cargo test -p fandhe-ai-backend-metal --release --test softmax_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_backend_metal::{MetalContext, MetalSoftmax};

/// テスト専用 CPU 参照実装（素朴な `exp(x - max(x)) / sum(exp(x -
/// max(x)))`。カーネル側の「最大値減算後に `log2(e)` を適用 + `exp2`」
/// 実装（`shaders/softmax.metal` ファイル冒頭コメント参照）とは異なる
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

/// softmax 事前スケーリングのオーバーフロー・sum 汚染回帰（PR #714
/// codex-review 指摘の 2 点を同時に検証する）:
///
/// 1. `x * log2(e)` を最大値減算より先に適用すると、有限だが `f32::MAX`
///    付近の入力で `+inf` へオーバーフローし後続の `exp2(inf - inf)` が
///    `NaN` になっていた（行 0: `f32::MAX` 付近の正の巨大値）。
/// 2. 範囲外レーンの寄与を sentinel の大小関係のみで暗黙に除外する設計は、
///    実データが `-f32::MAX` 付近の場合に sentinel と拮抗し sum を汚染
///    しうる（行 1: `-f32::MAX` 付近の負の巨大値）。
///
/// `shaders/softmax.metal` は最大値減算後に `log2(e)` を適用し、かつ
/// sum への寄与を `valid` フラグで明示的にゲートする構成へ是正済み
/// 〈ファイル冒頭コメント参照〉。本テストは ±`f32::MAX` 付近の有限値と、
/// hidden=37（32 の非倍数。SIMD 幅 32 の端数チャンクを強制する）の
/// 組み合わせで実機上の非 NaN・CPU 一致を検証する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn softmax_is_numerically_stable_near_f32_max_with_non_multiple_of_32_hidden() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let softmax = MetalSoftmax::new(&ctx).expect("softmax パイプラインの構築に失敗した");

    // hidden=37: SIMD 幅 32 の非倍数（最終チャンクが 5 要素の端数になる）。
    let hidden = 37usize;

    // 行 0: 先頭要素が f32::MAX、残りは緩やかに減少する有限大値。
    let mut near_max_positive = vec![0.0f32; hidden];
    near_max_positive[0] = f32::MAX;
    for (i, v) in near_max_positive.iter_mut().enumerate().skip(1) {
        *v = f32::MAX - (i as f32) * 1.0e30;
    }

    // 行 1: 先頭要素が -f32::MAX、残りは緩やかに増加する有限大負値。
    let mut near_max_negative = vec![0.0f32; hidden];
    near_max_negative[0] = -f32::MAX;
    for (i, v) in near_max_negative.iter_mut().enumerate().skip(1) {
        *v = -f32::MAX + (i as f32) * 1.0e30;
    }

    let mut x_data = Vec::with_capacity(2 * hidden);
    x_data.extend_from_slice(&near_max_positive);
    x_data.extend_from_slice(&near_max_negative);

    let gpu_out = softmax
        .run_softmax_f32(&ctx, &x_data, 2, hidden)
        .expect("f32::MAX 付近の有限入力でも run_softmax_f32 は成功しなければならない");

    assert!(
        gpu_out.iter().all(|v| v.is_finite()),
        "f32::MAX 付近の有限入力から NaN/inf が出力された: {gpu_out:?}"
    );

    let cpu_out = cpu_softmax_reference(&x_data, 2, hidden);
    assert_parity(
        "softmax near f32::MAX, hidden=37 (non-multiple of 32)",
        &gpu_out,
        &cpu_out,
    );
}

/// `run_fused`（`ops.rs::MetalBackendOps::run_fused`）経由の canonical
/// softmax プラン実行を CPU 参照実装と突き合わせる（実装計画 §6.2
/// 「run_fused 経路の実機テスト」）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn softmax_run_fused_matches_cpu_reference() {
    use fandhe_ai_tensor_core::{BackendOps, DType, FusedOpKind, FusionPlan, Tensor};

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

    let metal = fandhe_ai_backend_metal::MetalBackendOps::new();
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

/// CPU-Metal 直接突合（イシュー #607）: `fandhe_ai_backend_cpu::softmax::
/// run_softmax_f32`（NEON/rayon 参照実装。`f32::exp` ベース）を GPU 出力
/// と直接比較する。実機必須（`#[ignore]`。CI ではコンパイルのみ）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn softmax_matches_backend_cpu_directly() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let softmax = MetalSoftmax::new(&ctx).expect("softmax パイプラインの構築に失敗した");

    let rows = 3usize;
    let hidden = 4097usize; // NEON 端要素を含む。
    let x_data = Xorshift64Star::new(42_001).fill_vec(rows * hidden);

    let gpu_out = softmax
        .run_softmax_f32(&ctx, &x_data, rows, hidden)
        .expect("MetalSoftmax::run_softmax_f32 must succeed on Metal-equipped test runner");
    let cpu_out = fandhe_ai_backend_cpu::softmax::run_softmax_f32(&x_data, rows, hidden)
        .expect("fandhe_ai_backend_cpu::softmax::run_softmax_f32 must succeed");

    assert_parity(
        "softmax cpu(backend_cpu)-metal direct parity",
        &gpu_out,
        &cpu_out,
    );
}
