//! `backend-metal` naive GEMM（TASK-1.8b・#39）の受け入れ条件検証:
//! 「naive GEMM の数値が CPU 参照実装と複合判定（相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満。REQ-2）で一致する」。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する。CI（self-hosted・
//! Linux）では `#![cfg(target_os = "macos")]` によりコンパイル対象外に
//! なり、`#[ignore]` により通常の `cargo test` からも除外される
//! （実機依存テストの分離。`.claude/rules/coding-rust.md`。
//! `tests/device_smoke.rs`〈#38〉と同じ方針）。実行するには macOS 実機で
//! 以下を叩く:
//!
//! ```sh
//! cargo test -p backend-metal -- --ignored --nocapture
//! ```
//!
//! 本格的な実機 CI 整備は TASK-1.8e（#42）で行う。本テストはそれまでの間、
//! 受け入れ条件の手動検証手順を兼ねる。
//!
//! CPU 参照は `backend_cpu::parity::matmul_reference_fma`（FMA 契約の
//! 唯一の参照点）、判定は `backend_cpu::parity::assert_parity`（REQ-2
//! 統一複合判定の唯一の実体。閾値の独自定義・緩和は禁止。
//! `.claude/rules/security.md`）を使う。入力生成は
//! `bench_harness::rng::Xorshift64Star`（決定的シード）。

#![cfg(target_os = "macos")]

use backend_cpu::parity::{assert_parity, matmul_reference_fma};
use backend_metal::{MetalContext, MetalError, MetalGemm};
use bench_harness::rng::Xorshift64Star;

/// `(seed_a, seed_b, m, n, k)` の 1 ケースを実行し、CPU 参照実装との
/// 複合判定 PASS を確認する。
fn run_case(seed_a: u64, seed_b: u64, m: usize, n: usize, k: usize) {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("naive GEMM パイプラインの構築に失敗した");

    let a = Xorshift64Star::new(seed_a).fill_vec(m * k);
    let b = Xorshift64Star::new(seed_b).fill_vec(k * n);

    let mut expected = vec![0.0f32; m * n];
    matmul_reference_fma(&a, &b, &mut expected, m, n, k)
        .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した");

    let actual = gemm
        .dispatch(&ctx, &a, &b, m, n, k)
        .expect("Metal naive GEMM のディスパッチに失敗した");

    assert_parity(
        &format!("metal naive gemm m={m} n={n} k={k}"),
        &actual,
        &expected,
    );
}

/// threadgroup サイズ（16）の倍数でない形状。`shaders/gemm.metal` の
/// 手動境界チェック（`gid.y >= dims.m || gid.x >= dims.n`）が実際に
/// 効くケース（REQ-8）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn naive_matches_cpu_reference_non_multiple_of_threadgroup() {
    run_case(1, 2, 7, 13, 5);
    run_case(3, 4, 33, 65, 17);
}

/// 中規模形状（境界チェック検証用。1 threadgroup（16×16）を大きく超える
/// grid で末尾ブロックの境界処理を確認する）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn naive_matches_cpu_reference_medium_shape() {
    run_case(5, 6, 128, 96, 72);
}

/// K ストレスケース（PoC-v2-5 の FMA 契約実測ケースに対応。m=n=64,
/// k=4096。長い内積での丸め誤差蓄積が CPU 参照実装〈`f32::mul_add`〉と
/// 一致することを確認する）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn naive_matches_cpu_reference_k_stress() {
    run_case(7, 8, 64, 64, 4096);
}

/// 零次元・長さ不一致が FFI 呼び出し前に型付きエラーで拒否されることを
/// 確認する（OWASP A03 観点。`.claude/rules/security.md`）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dispatch_rejects_invalid_shapes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("naive GEMM パイプラインの構築に失敗した");

    let err = gemm.dispatch(&ctx, &[], &[], 0, 4, 3).unwrap_err();
    assert!(matches!(
        err,
        MetalError::ZeroDimension { m: 0, n: 4, k: 3 }
    ));

    let a = vec![0.0f32; 5]; // m*k=6 を期待
    let b = vec![0.0f32; 12];
    let err = gemm.dispatch(&ctx, &a, &b, 2, 4, 3).unwrap_err();
    assert!(matches!(
        err,
        MetalError::ALenMismatch {
            expected: 6,
            actual: 5
        }
    ));
}
