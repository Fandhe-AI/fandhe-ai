//! split-K 到達 11 形状（`docs/backend-metal-splitk-decision.md` §3・
//! `tests/gemm_splitk_parity.rs::TARGET_SHAPES` と同一）を対象に、
//! split-K 経路・classic 経路・CPU f32 参照実装
//! （[`fandhe_ai_backend_cpu::parity::matmul_reference_fma`]）それぞれの
//! 出力を、ホスト側 f64 真値（f32 入力を f64 へ昇格し、i 外側・k 中間・
//! j 内側の逐次加算で計算した厳密解）と突き合わせた誤差
//! （`max_abs`・`mean_abs`・`max_rel`）を記録する**診断専用テスト**
//! （イシュー #1549）。
//!
//! # 位置づけ（`gemm_splitk_parity.rs` との違い・判定基準を設けない理由）
//!
//! `gemm_splitk_parity.rs` は split-K 経路と CPU f32 参照実装（互いに
//! ホスト f32・逐次 FMA という共通の丸め方針を持つ 2 点）を
//! `assert_no_split_k_parity_regression`（実測ベースライン非後退方式。
//! `docs/backend-metal-splitk-parity-judgment-decision.md` §7 承認済み）
//! で判定する既存の受け入れテストであり、本テストはそれとは独立に
//! 「どちらが `f64` 厳密解に近いか」という**参考情報**を集めるための
//! ものである。本テストは **`assert_no_split_k_parity_regression` の
//! baseline・tolerance 定数（`RELATIVE_TOLERANCE`／
//! `ABSOLUTE_RESCUE_THRESHOLD`）・既存テストのいずれも変更しない**。
//! 誤差に対する閾値 assert は置かず、以下のみを assert する:
//!
//! - split-K 経路が実際に `SplitKRoute::Split`（`GemmRoute::SplitK`）へ
//!   到達したこと（フォールバックによる自明合格の排除）
//! - 出力長が `m * n` と一致すること
//! - 各経路の出力に `NaN` が含まれないこと
//!
//! 誤差の大小に関する「良い／悪い」の判断や、baseline 変更の提案は
//! 本テストのコードにもドキュメントにも含めない（`docs/perf/
//! metal-gemm-splitk-f64-truth.md` に事実のみを記録する）。
//!
//! # 入力生成の同一性
//!
//! `TARGET_SHAPES`・入力生成（`Xorshift64Star::new(seed).fill_vec(len)`・
//! シード `a: m*7+k+1`／`b: n*11+k+2`）は `tests/gemm_splitk_parity.rs`
//! と完全に同一にし、既存の記録済み baseline（`common::
//! splitk_parity_baseline::BASELINES`）と同じ入力に基づく誤差として
//! 紐付けられるようにする。転置パターンは NN（非転置）のみを対象とする
//! （`gemm_splitk_parity.rs` が検証する NT/TN/TT は本テストのスコープ
//! 外。理由は目的が「経路間の数値特性の参考比較」であり全パターンを
//! 網羅する必要がないため）。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する。`#[ignore]`
//! により通常の `cargo test` からは除外される
//! （`tests/gemm_splitk_parity.rs` と同じ方針）。
//!
//! 実機実行（Apple Silicon 必須）:
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --features internal-diagnostics --test gemm_splitk_f64_truth -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::parity::matmul_reference_fma;
use fandhe_ai_backend_metal::tile;
use fandhe_ai_backend_metal::{GemmRoute, MetalContext, MetalGemm};

/// `tests/gemm_splitk_parity.rs::TARGET_SHAPES` と同一の対象 11 形状。
const TARGET_SHAPES: &[(usize, usize, usize)] = &[
    (32, 32, 2048),
    (32, 32, 4096),
    (32, 32, 8192),
    (64, 64, 2048),
    (64, 64, 4096),
    (64, 64, 8192),
    (128, 128, 2048),
    (128, 128, 4096),
    (128, 128, 8192),
    (64, 64, 2056),
    (128, 128, 2064),
];

/// f32 入力を f64 へ昇格し、i 外側・k 中間・j 内側の逐次加算で求める
/// ホスト側厳密解（`crates/backend-cuda/tests/specialized_mma_f16_triage.rs::
/// exact_reference_f64` の f32 版）。`f32::mul_add` によるまとめ丸めを
/// 経由しない厳密な `f64` 累積のため、丸めの入り方が異なる split-K・
/// classic・CPU f32 参照の 3 経路いずれとも独立な基準点として使う。
fn exact_reference_f64(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        let a_row = &a[i * k..i * k + k];
        let c_row = &mut c[i * n..i * n + n];
        for (p, &a_ip) in a_row.iter().enumerate() {
            let a_ip = a_ip as f64;
            let b_row = &b[p * n..p * n + n];
            for j in 0..n {
                c_row[j] += a_ip * (b_row[j] as f64);
            }
        }
    }
    c
}

/// 経路 1 つ分の `f64` 真値に対する誤差集計（診断専用。判定には使わない）。
struct ErrorStats {
    max_abs: f64,
    mean_abs: f64,
    max_rel: f64,
}

/// `actual`（f32 出力）と `truth`（f64 真値）を突き合わせ、`max_abs`・
/// `mean_abs`・`max_rel` を `f64` で算出する。相対誤差の分母は真値
/// （`fandhe_ai_backend_cpu::parity::compare` とは異なる診断専用の指標
/// であり、REQ-2 統一複合判定の代替ではない）。
fn error_stats_vs_truth(actual: &[f32], truth: &[f64]) -> ErrorStats {
    assert_eq!(
        actual.len(),
        truth.len(),
        "actual と truth の長さが一致しない（内部バグ）"
    );
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    for (&a, &t) in actual.iter().zip(truth.iter()) {
        let a = a as f64;
        let abs = (a - t).abs();
        let rel = abs / t.abs().max(1e-12);
        max_abs = max_abs.max(abs);
        sum_abs += abs;
        max_rel = max_rel.max(rel);
    }
    ErrorStats {
        max_abs,
        mean_abs: sum_abs / (actual.len() as f64),
        max_rel,
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn split_k_classic_cpu_f32_error_vs_f64_truth_for_target_shapes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let split_k_gemm = MetalGemm::new_with_split_k_auto(&ctx, true)
        .expect("split-K 有効 GEMM パイプラインの構築に失敗した");
    let classic_gemm = MetalGemm::new_with_split_k_auto(&ctx, false)
        .expect("classic GEMM パイプラインの構築に失敗した");

    // 「split-K が classic より真値から遠い形状数」の集計（判定基準
    // ではなく事実の集計のみ）。
    let mut split_farther_than_classic_by_max_abs = 0usize;
    let mut split_farther_than_classic_by_mean_abs = 0usize;

    println!(
        "| m | n | k | route | max_abs | mean_abs | max_rel |\n\
         |---|---|---|-------|---------|----------|---------|"
    );

    for &(m, n, k) in TARGET_SHAPES {
        let a = Xorshift64Star::new(m as u64 * 7 + k as u64 + 1).fill_vec(m * k);
        let b = Xorshift64Star::new(n as u64 * 11 + k as u64 + 2).fill_vec(k * n);

        // split-K 経路。`should_split_k` が対象形状で `Some` を返す前提
        // （`docs/backend-metal-splitk-decision.md` §3）の確認も兼ねる。
        let _plan = tile::should_split_k(m, n, k).unwrap_or_else(|| {
            panic!(
                "should_split_k が None を返した（対象形状の前提が崩れている）: m={m}, n={n}, k={k}"
            )
        });
        let (split_out, split_route) = split_k_gemm
            .dispatch_auto_with_route(&ctx, &a, &b, m, n, k)
            .unwrap_or_else(|e| {
                panic!("split-K dispatch_auto_with_route failed (m={m}, n={n}, k={k}): {e}")
            });
        assert!(
            matches!(split_route, GemmRoute::SplitK(_)),
            "m={m}, n={n}, k={k}: split-K 経路が classic 経路へフォールバックした \
             （route={split_route:?}）。フォールバックによる自明合格を排除するため \
             split-K 経路への到達を要求する。"
        );

        // classic 経路（split_k_auto_enabled=false で構築したインスタンス）。
        let (classic_out, classic_route) = classic_gemm
            .dispatch_auto_with_route(&ctx, &a, &b, m, n, k)
            .unwrap_or_else(|e| {
                panic!("classic dispatch_auto_with_route failed (m={m}, n={n}, k={k}): {e}")
            });
        assert!(
            matches!(classic_route, GemmRoute::Classic(_)),
            "m={m}, n={n}, k={k}: classic 指定インスタンスなのに split-K 経路が選ばれた \
             （route={classic_route:?}）。"
        );

        // CPU f32 参照実装。
        let mut cpu_out = vec![0.0f32; m * n];
        matmul_reference_fma(&a, &b, &mut cpu_out, m, n, k)
            .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した");

        for (label, out) in [
            ("split_k", &split_out),
            ("classic", &classic_out),
            ("cpu_f32", &cpu_out),
        ] {
            assert_eq!(
                out.len(),
                m * n,
                "m={m}, n={n}, k={k}: {label} の出力長が m*n と一致しない"
            );
            assert!(
                out.iter().all(|v| !v.is_nan()),
                "m={m}, n={n}, k={k}: {label} の出力に NaN が含まれる"
            );
        }

        let truth = exact_reference_f64(&a, &b, m, n, k);
        let split_stats = error_stats_vs_truth(&split_out, &truth);
        let classic_stats = error_stats_vs_truth(&classic_out, &truth);
        let cpu_stats = error_stats_vs_truth(&cpu_out, &truth);

        if split_stats.max_abs > classic_stats.max_abs {
            split_farther_than_classic_by_max_abs += 1;
        }
        if split_stats.mean_abs > classic_stats.mean_abs {
            split_farther_than_classic_by_mean_abs += 1;
        }

        for (label, stats) in [
            ("split_k", &split_stats),
            ("classic", &classic_stats),
            ("cpu_f32", &cpu_stats),
        ] {
            println!(
                "| {m} | {n} | {k} | {label} | {:.10e} | {:.10e} | {:.10e} |",
                stats.max_abs, stats.mean_abs, stats.max_rel
            );
        }
    }

    println!(
        "\nsplit_farther_than_classic_by_max_abs={split_farther_than_classic_by_max_abs}/{}",
        TARGET_SHAPES.len()
    );
    println!(
        "split_farther_than_classic_by_mean_abs={split_farther_than_classic_by_mean_abs}/{}",
        TARGET_SHAPES.len()
    );
}
