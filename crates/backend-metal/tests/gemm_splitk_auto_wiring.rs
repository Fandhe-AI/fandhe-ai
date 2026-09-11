//! `MetalGemm::dispatch_auto`（本番 NN 経路の自動入口）への split-K
//! 本番結線（イシュー #1516）の実機受け入れテスト。
//!
//! # 位置づけ（`gemm_splitk_auto_entry_parity.rs` との違い）
//!
//! `gemm_splitk_auto_entry_parity.rs`（イシュー #1513）は
//! `MetalGemm::dispatch_split_k_strided_prepared`（split-K 専用の
//! ゲート付き明示入口）自体の正しさを検証する。本ファイルは、その 1 段
//! 上——`MetalBackendOps::gemm` が実際に呼ぶ本番 NN 入口
//! `MetalGemm::dispatch_auto` が `tile::should_split_k`／
//! `tile::select_route_for_device` の判定結果に基づき正しく分岐するか
//! （`MetalGemm::split_k_auto_enabled` インスタンスフィールド・実行時
//! トグル `crate::split_k_runtime::split_k_enabled()`）を検証する。
//!
//! **2026-09-11・イシュー #1516 で本番既定は `true`**（`docs/backend-
//! metal-splitk-decision.md` §5「本番結線（#1516）」参照: 性能面の正式
//! ADOPT 判定〈イシュー #1515 §10.4〉が M4 Max 実機 5 run で確定した
//! ことを受けた切替。#1547 でコンパイル時定数ゲートを撤去し
//! per-instance フィールドの固定値＋実行時トグルへ一本化した）。
//! よって `MetalGemm::new()`（本番既定コンストラクタ）
//! 自体が split-K 分岐を有効化した状態で構築される。本テストは
//! `MetalGemm::new_with_split_k_auto` で明示的に `true`／`false` を
//! 指定したインスタンスとの bit 一致・経路選択を検証する
//! （`gemm_fine_barrier_bit_match.rs` 等の A/B 自己検証テストと同型の
//! 設計）。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する。CI（GitHub
//! ホステッド・ubuntu-latest）では `#![cfg(target_os = "macos")]` により
//! コンパイル対象外になり、`#[ignore]` により通常の `cargo test` からも
//! 除外される。`internal-diagnostics` feature 限定の `MetalGemm::
//! dispatch_auto_with_route`（実際に split-K 経路まで到達したかを
//! フォールバックによる自明合格なしに検証するための診断入口）を直接
//! 呼ぶため `required-features` を要求する（`Cargo.toml` 参照）。
//!
//! 実機実行（Apple Silicon 必須）:
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --features internal-diagnostics --test gemm_splitk_auto_wiring -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

mod common;

use bench_harness::rng::Xorshift64Star;
use common::splitk_parity_baseline::{assert_no_split_k_parity_regression, find_baseline};
use fandhe_ai_backend_cpu::parity::{compare, matmul_reference_fma};
use fandhe_ai_backend_metal::split_k_runtime::{set_split_k_enabled, split_k_enabled};
use fandhe_ai_backend_metal::tile;
use fandhe_ai_backend_metal::{GemmRoute, MetalContext, MetalGemm};

/// 本ファイルの各テストはインスタンス単位ゲート（`MetalGemm::
/// split_k_auto_enabled`）の検証が目的であり、実行時トグル
/// （`crate::split_k_runtime`。イシュー #1545）の影響を受けないことを
/// 前提とする。本ガードは、本ファイルと同一プロセスで並列実行され
/// うる他テストバイナリの状態変化から独立させるための直列化・
/// 原状復帰 RAII ガード（`gemm_splitk_runtime_toggle.rs::
/// RuntimeFlagGuard` と同型）。各テスト冒頭で明示的に `true`（既定
/// 相当）へ固定し、本ファイルの意図（インスタンス単位ゲートの検証）を
/// 実行時トグルの状態に左右されないようにする。
struct RuntimeFlagGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    original: bool,
}

impl RuntimeFlagGuard {
    fn acquire_enabled() -> Self {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let lock = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let original = split_k_enabled();
        set_split_k_enabled(true);
        Self {
            _lock: lock,
            original,
        }
    }
}

impl Drop for RuntimeFlagGuard {
    fn drop(&mut self) {
        set_split_k_enabled(self.original);
    }
}

/// `to_bits()` 経由の bit 単位一致検証（`tests/gemm_splitk_bit_match.rs::
/// assert_bit_exact` と同じ理由: `assert_eq!` の `f32` 比較は `+0.0 == -0.0`
/// を区別できず符号ビットの差異を見逃しうるため）。
fn assert_bit_exact(actual: &[f32], expected: &[f32], context: &str) {
    let actual_bits: Vec<u32> = actual.iter().map(|v| v.to_bits()).collect();
    let expected_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        actual_bits, expected_bits,
        "{context}: dispatch_auto の出力がビット単位で一致しなかった。"
    );
}

/// 正方 4 形状（`docs/perf/gemm-optimization-baseline.md` の実測帯域）＋
/// 対照の非正方形状（`should_split_k` 対象外である `(256,256,2048)`）。
/// AC-1「正方 4 形状の出力 bit 同一」の対象。
const NON_TARGET_SHAPES: &[(usize, usize, usize)] = &[
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (256, 256, 2048),
    (64, 64, 63),
];

/// `tests/gemm_splitk_auto_entry_parity.rs::TARGET_SHAPES` と同一の
/// 承認済み 11 形状（`common::splitk_parity_baseline::BASELINES` が
/// 1 対 1 対応）。
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

/// AC-1: split-K 結線ゲートが既定 `true`（イシュー #1516・2026-09-11
/// 切替）であることを、`MetalGemm::new()`（本番コンストラクタ）と
/// `MetalGemm::new_with_split_k_auto(&ctx, true)`（明示 opt-in）が、
/// 対象形状・非対象形状の区別なく常に bit 同一の出力を返すことで
/// 直接検証する。ゲートの「既定値」が実際に `true` であることの機構的な
/// 証跡（`new()` が明示 `true` 指定と同じ経路を通ることの直接検証）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn wiring_default_is_bit_identical_to_explicit_on() {
    let _rt_guard = RuntimeFlagGuard::acquire_enabled();
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let base = MetalGemm::new(&ctx).expect("base GEMM パイプラインの構築に失敗した");
    let head = MetalGemm::new_with_split_k_auto(&ctx, true)
        .expect("head GEMM パイプラインの構築に失敗した");

    let shapes: Vec<(usize, usize, usize)> = NON_TARGET_SHAPES
        .iter()
        .chain(TARGET_SHAPES.iter())
        .copied()
        .collect();

    for (m, n, k) in shapes {
        let a = Xorshift64Star::new(m as u64 * 7 + k as u64 + 1).fill_vec(m * k);
        let b = Xorshift64Star::new(n as u64 * 11 + k as u64 + 2).fill_vec(k * n);

        let base_out = base
            .dispatch_auto(&ctx, &a, &b, m, n, k)
            .unwrap_or_else(|e| panic!("base dispatch_auto failed (m={m}, n={n}, k={k}): {e}"));
        let head_out = head
            .dispatch_auto(&ctx, &a, &b, m, n, k)
            .unwrap_or_else(|e| panic!("head dispatch_auto failed (m={m}, n={n}, k={k}): {e}"));

        assert_bit_exact(
            &head_out,
            &base_out,
            &format!("split_k_auto_enabled=true (default vs explicit) (m={m}, n={n}, k={k})"),
        );
    }
}

/// AC-1 補助: 明示的な opt-out（`new_with_split_k_auto(&ctx, false)`）は
/// 本番既定（`true`）とは無関係に、`should_split_k` が対象と判定する
/// 形状（`TARGET_SHAPES`）でも常に classic 経路（[`GemmRoute::Classic`]）
/// へ固定されることを確認する。ゲートを個別インスタンス単位で無効化
/// できる手段（実機診断・将来の A/B）が実際に classic 経路のみを通る
/// ことの直接検証（フォールバックによる自明合格の排除）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn wiring_explicit_off_forces_classic_route_for_targets() {
    let _rt_guard = RuntimeFlagGuard::acquire_enabled();
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let head = MetalGemm::new_with_split_k_auto(&ctx, false)
        .expect("head GEMM パイプラインの構築に失敗した");

    for &(m, n, k) in TARGET_SHAPES {
        let a = Xorshift64Star::new(m as u64 * 19 + k as u64 + 23).fill_vec(m * k);
        let b = Xorshift64Star::new(n as u64 * 29 + k as u64 + 31).fill_vec(k * n);

        let (_out, route) = head
            .dispatch_auto_with_route(&ctx, &a, &b, m, n, k)
            .unwrap_or_else(|e| {
                panic!("head dispatch_auto_with_route failed (m={m}, n={n}, k={k}): {e}")
            });

        assert!(
            matches!(route, GemmRoute::Classic(_)),
            "m={m}, n={n}, k={k}: split_k_auto_enabled=false のはずが GemmRoute::SplitK が \
             選ばれた（明示 opt-out がゲートを無視している疑いがある）。route={route:?}"
        );
    }
}

/// AC-1: 結線ゲート `true` の head インスタンスでも、`should_split_k` が
/// `None` を返す形状（正方 4 形状・対照形状）では `MetalGemm::new`
/// （classic 経路のみ）と bit 同一の出力を返し、かつ診断入口
/// `dispatch_auto_with_route` が実際に `GemmRoute::Classic` を返す
/// （フォールバックによる自明合格ではなく、経路選択そのものが正しく
/// classic を選んだことの直接検証）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn wiring_on_keeps_classic_bit_identical_for_non_eligible() {
    let _rt_guard = RuntimeFlagGuard::acquire_enabled();
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let base = MetalGemm::new(&ctx).expect("base GEMM パイプラインの構築に失敗した");
    let head = MetalGemm::new_with_split_k_auto(&ctx, true)
        .expect("head GEMM パイプラインの構築に失敗した");

    for &(m, n, k) in NON_TARGET_SHAPES {
        let a = Xorshift64Star::new(m as u64 * 13 + k as u64 + 3).fill_vec(m * k);
        let b = Xorshift64Star::new(n as u64 * 17 + k as u64 + 5).fill_vec(k * n);

        let base_out = base
            .dispatch_auto(&ctx, &a, &b, m, n, k)
            .unwrap_or_else(|e| panic!("base dispatch_auto failed (m={m}, n={n}, k={k}): {e}"));
        let (head_out, route) = head
            .dispatch_auto_with_route(&ctx, &a, &b, m, n, k)
            .unwrap_or_else(|e| {
                panic!("head dispatch_auto_with_route failed (m={m}, n={n}, k={k}): {e}")
            });

        assert!(
            matches!(route, GemmRoute::Classic(_)),
            "m={m}, n={n}, k={k}: split_k_auto_enabled=true でも should_split_k が None を \
             返す形状のはずが GemmRoute::SplitK が選ばれた（NON_TARGET_SHAPES の前提が \
             崩れている、または select_route_for_device の実装に誤りがある）。route={route:?}"
        );
        assert_bit_exact(
            &head_out,
            &base_out,
            &format!("split_k_auto_enabled=true, non-eligible (m={m}, n={n}, k={k})"),
        );
    }
}

/// AC-2: 結線ゲート `true` の head インスタンスは、承認済み 11 形状
/// （`should_split_k` が `Some` を返す形状）で実際に split-K 2 パス経路
/// （`GemmRoute::SplitK`）へ到達し、その出力が実測ベースライン非後退
/// 契約（`common::splitk_parity_baseline::assert_no_split_k_parity_regression`。
/// tolerance 定数は変更しない）を満たすことを確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn wiring_on_routes_split_k_for_targets_and_matches_baseline() {
    let _rt_guard = RuntimeFlagGuard::acquire_enabled();
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let head = MetalGemm::new_with_split_k_auto(&ctx, true)
        .expect("head GEMM パイプラインの構築に失敗した");

    for &(m, n, k) in TARGET_SHAPES {
        let baseline = find_baseline(m, n, k).unwrap_or_else(|| {
            panic!(
                "m={m}, n={n}, k={k} に対応する記録済みベースラインが見つからない \
                 （TARGET_SHAPES と common::splitk_parity_baseline::BASELINES の対応が \
                 崩れている）"
            )
        });

        let a = Xorshift64Star::new(m as u64 * 7 + k as u64 + 1).fill_vec(m * k);
        let b = Xorshift64Star::new(n as u64 * 11 + k as u64 + 2).fill_vec(k * n);
        let mut expected = vec![0.0f32; m * n];
        matmul_reference_fma(&a, &b, &mut expected, m, n, k)
            .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した");

        let (actual, route) = head
            .dispatch_auto_with_route(&ctx, &a, &b, m, n, k)
            .unwrap_or_else(|e| {
                panic!("head dispatch_auto_with_route failed (m={m}, n={n}, k={k}): {e}")
            });

        let expected_plan = tile::should_split_k(m, n, k).unwrap_or_else(|| {
            panic!(
                "should_split_k が None を返した（承認済み形状の前提が崩れている）: \
                 m={m}, n={n}, k={k}"
            )
        });
        assert!(
            matches!(route, GemmRoute::SplitK(plan) if plan == expected_plan),
            "m={m}, n={n}, k={k}: split_k_auto_enabled=true のはずが split-K 経路へ到達\
             しなかった（route={route:?}）。フォールバックによる自明合格の疑いがある。"
        );

        let report = compare(&actual, &expected)
            .unwrap_or_else(|e| panic!("parity compare failed (m={m}, n={n}, k={k}): {e}"));

        println!(
            "[auto-wiring] m={m} n={n} k={k} fail_count={}/{} max_abs_diff={:.10} \
             mean_abs_diff={:.10} max_rel_err={:.8}",
            report.fail_count,
            report.total,
            report.max_abs_diff,
            report.mean_abs_diff,
            report.max_rel_err
        );

        let context = format!("dispatch_auto split-K wiring parity (m={m}, n={n}, k={k})");
        assert_no_split_k_parity_regression(&context, &report, baseline);
    }
}
