//! split-K 実行時 opt-out トグル（`fandhe_ai_backend_metal::
//! split_k_runtime`。イシュー #1545）の実機受け入れテスト。
//!
//! # 位置づけ（`gemm_splitk_auto_wiring.rs` との違い）
//!
//! `gemm_splitk_auto_wiring.rs`（イシュー #1516）は、インスタンス単位
//! フィールド `MetalGemm::split_k_auto_enabled`（`new_with_split_k_auto`
//! で構築時に固定。`MetalGemm::new()` は既定 `true` 固定。#1547 で
//! コンパイル時定数ゲートを撤去し本フィールドへ一本化）による段の
//! ゲートを検証する。本ファイルは、その 1 段上——
//! `split_k_runtime::set_split_k_enabled`（プロセスワイドかつ**実行時
//! に**切り替え可能なゲート）が、同一の `MetalGemm` インスタンス
//! （`MetalGemm::new()`。本番既定コンストラクタ）に対して呼び出しごとの
//! 経路選択を実際に変えることを検証する（`docs/backend-metal-splitk-
//! decision.md` §5「実行時トグル」参照）。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する。CI（GitHub
//! ホステッド・ubuntu-latest）では `#![cfg(target_os = "macos")]` により
//! コンパイル対象外になり、`#[ignore]` により通常の `cargo test` からも
//! 除外される。`internal-diagnostics` feature 限定の `MetalGemm::
//! dispatch_auto_with_route` を直接呼ぶため `required-features` を要求
//! する（`Cargo.toml` 参照）。
//!
//! 実機実行（Apple Silicon 必須）:
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --features internal-diagnostics --test gemm_splitk_runtime_toggle -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_metal::split_k_runtime::{set_split_k_enabled, split_k_enabled};
use fandhe_ai_backend_metal::{GemmRoute, MetalContext, MetalGemm};
use std::sync::Mutex;

/// `to_bits()` 経由の bit 単位一致検証（`tests/gemm_splitk_bit_match.rs::
/// assert_bit_exact` と同じ理由）。
fn assert_bit_exact(actual: &[f32], expected: &[f32], context: &str) {
    let actual_bits: Vec<u32> = actual.iter().map(|v| v.to_bits()).collect();
    let expected_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        actual_bits, expected_bits,
        "{context}: dispatch_auto の出力がビット単位で一致しなかった。"
    );
}

/// `split_k_runtime` はプロセスグローバルなため、本ファイル内のテスト
/// （同一プロセスで並列実行されうる）を直列化・原状復帰する RAII
/// ガード。`crate::split_k_runtime::test_support` はクレート内部限定
/// （`pub(crate)`・`cfg(test)`）で統合テストからは到達できないため、
/// 本ファイル専用に同型の小さなガードを持つ（`gemm_splitk_auto_wiring.rs`
/// が `MetalGemm::new_with_split_k_auto` によるインスタンス単位の
/// フラグしか扱わず本種のプロセスグローバル直列化を要さないのとは
/// 対照的）。
struct RuntimeFlagGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    original: bool,
}

impl RuntimeFlagGuard {
    fn acquire() -> Self {
        static LOCK: Mutex<()> = Mutex::new(());
        let lock = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let original = split_k_enabled();
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

/// `tests/gemm_splitk_auto_wiring.rs::TARGET_SHAPES` と同一の承認済み
/// 11 形状（`should_split_k` が `Some` を返す対象）。
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

/// (a): `set_split_k_enabled(false)` の間は `MetalGemm::new()`（本番既定
/// コンストラクタ。実行時トグル以外はすべて既定値）の `dispatch_auto` が
/// `MetalGemm::new_with_split_k_auto(&ctx, false)`（インスタンス単位で
/// classic 固定）と対象形状すべてで bit 同一になり、かつ診断入口
/// `dispatch_auto_with_route` が実際に `GemmRoute::Classic` を返す
/// （フォールバックによる自明合格ではなく、経路選択そのものが実行時
/// トグルにより classic へ固定されたことの直接検証）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn runtime_off_forces_classic_route_and_matches_instance_off() {
    let _guard = RuntimeFlagGuard::acquire();
    set_split_k_enabled(false);
    assert!(!split_k_enabled());

    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let default_gemm = MetalGemm::new(&ctx).expect("default GEMM パイプラインの構築に失敗した");
    let instance_off_gemm = MetalGemm::new_with_split_k_auto(&ctx, false)
        .expect("instance-off GEMM パイプラインの構築に失敗した");

    for &(m, n, k) in TARGET_SHAPES {
        let a = Xorshift64Star::new(m as u64 * 7 + k as u64 + 1).fill_vec(m * k);
        let b = Xorshift64Star::new(n as u64 * 11 + k as u64 + 2).fill_vec(k * n);

        let (default_out, default_route) = default_gemm
            .dispatch_auto_with_route(&ctx, &a, &b, m, n, k)
            .unwrap_or_else(|e| {
                panic!("default dispatch_auto_with_route failed (m={m}, n={n}, k={k}): {e}")
            });
        assert!(
            matches!(default_route, GemmRoute::Classic(_)),
            "m={m}, n={n}, k={k}: 実行時トグル false のはずが GemmRoute::SplitK が \
             選ばれた（split_k_runtime がゲートを無視している疑いがある）。\
             route={default_route:?}"
        );

        let instance_off_out = instance_off_gemm
            .dispatch_auto(&ctx, &a, &b, m, n, k)
            .unwrap_or_else(|e| {
                panic!("instance-off dispatch_auto failed (m={m}, n={n}, k={k}): {e}")
            });

        assert_bit_exact(
            &default_out,
            &instance_off_out,
            &format!(
                "split_k_runtime=false vs instance split_k_auto_enabled=false (m={m}, n={n}, k={k})"
            ),
        );
    }
}

/// (b): `set_split_k_enabled(true)`（既定と同じ）の間は `MetalGemm::
/// new()` の `dispatch_auto` が対象形状で split-K 経路
/// （`GemmRoute::SplitK`）へ到達し、その出力が
/// `MetalGemm::new_with_split_k_auto(&ctx, true)` と bit 同一になる。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn runtime_on_routes_split_k_and_matches_instance_on() {
    let _guard = RuntimeFlagGuard::acquire();
    set_split_k_enabled(true);
    assert!(split_k_enabled());

    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let default_gemm = MetalGemm::new(&ctx).expect("default GEMM パイプラインの構築に失敗した");
    let instance_on_gemm = MetalGemm::new_with_split_k_auto(&ctx, true)
        .expect("instance-on GEMM パイプラインの構築に失敗した");

    for &(m, n, k) in TARGET_SHAPES {
        let a = Xorshift64Star::new(m as u64 * 13 + k as u64 + 3).fill_vec(m * k);
        let b = Xorshift64Star::new(n as u64 * 17 + k as u64 + 5).fill_vec(k * n);

        let (default_out, default_route) = default_gemm
            .dispatch_auto_with_route(&ctx, &a, &b, m, n, k)
            .unwrap_or_else(|e| {
                panic!("default dispatch_auto_with_route failed (m={m}, n={n}, k={k}): {e}")
            });
        assert!(
            matches!(default_route, GemmRoute::SplitK(_)),
            "m={m}, n={n}, k={k}: 実行時トグル true（既定）のはずが GemmRoute::Classic が \
             選ばれた（対象形状の前提が崩れている、または split_k_runtime の実装に誤りが \
             ある）。route={default_route:?}"
        );

        let instance_on_out = instance_on_gemm
            .dispatch_auto(&ctx, &a, &b, m, n, k)
            .unwrap_or_else(|e| {
                panic!("instance-on dispatch_auto failed (m={m}, n={n}, k={k}): {e}")
            });

        assert_bit_exact(
            &default_out,
            &instance_on_out,
            &format!(
                "split_k_runtime=true vs instance split_k_auto_enabled=true (m={m}, n={n}, k={k})"
            ),
        );
    }
}

/// (c): false→true の往復後、既定（実行時トグル未操作時の `true`）と
/// bit 同一の出力に戻ることを確認する（フラグが状態を引きずらないこと
/// の直接検証）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn runtime_toggle_round_trip_restores_default_behavior() {
    let _guard = RuntimeFlagGuard::acquire();

    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    let (m, n, k) = TARGET_SHAPES[0];
    let a = Xorshift64Star::new(m as u64 * 19 + k as u64 + 23).fill_vec(m * k);
    let b = Xorshift64Star::new(n as u64 * 29 + k as u64 + 31).fill_vec(k * n);

    set_split_k_enabled(true);
    let before_out = gemm
        .dispatch_auto(&ctx, &a, &b, m, n, k)
        .expect("before dispatch_auto に失敗した");

    set_split_k_enabled(false);
    let off_out = gemm
        .dispatch_auto(&ctx, &a, &b, m, n, k)
        .expect("off dispatch_auto に失敗した");

    set_split_k_enabled(true);
    let after_out = gemm
        .dispatch_auto(&ctx, &a, &b, m, n, k)
        .expect("after dispatch_auto に失敗した");

    assert_bit_exact(
        &after_out,
        &before_out,
        &format!("往復後の出力が往復前と一致しない (m={m}, n={n}, k={k})"),
    );

    // off 中は split-K（往復前後）とは異なる classic 経路の出力になる
    // はず（対象形状は split-K に到達する形状のため、classic と split-K
    // は現行の受け入れ判定方式上 bit 一致しないことがある
    // 〈baseline 非後退方式。厳密不一致の断定はしない〉。ここでは
    // off_out が意味のある値であること（NaN 等でない）の緩い健全性
    // チェックに留める。
    assert_eq!(off_out.len(), before_out.len());
    assert!(off_out.iter().all(|v| v.is_finite()));
}
