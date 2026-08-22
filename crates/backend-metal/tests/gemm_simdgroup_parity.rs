//! `backend-metal` tiled・simdgroup GEMM（TASK-1.8c・#40）の受け入れ条件検証:
//! 「tiled/simdgroup GEMM の数値が CPU 参照実装と複合判定（相対誤差 1e-3
//! 未満 または 絶対誤差 1e-5 未満。REQ-2）で一致する」。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する。CI（self-hosted・
//! Linux）では `#![cfg(target_os = "macos")]` によりコンパイル対象外に
//! なり、`#[ignore]` により通常の `cargo test` からも除外される
//! （実機依存テストの分離。`.claude/rules/coding-rust.md`。
//! `tests/gemm_naive_parity.rs`〈#39〉と同じ方針）。実行するには macOS 実機で
//! 以下を叩く:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal -- --ignored --nocapture
//! ```
//!
//! 実行手順・テスト一覧の正本は `docs/backend-metal-real-device-testing.md`
//! （TASK-1.8e・#42）を参照する。
//!
//! CPU 参照は `fandhe_ai_backend_cpu::parity::matmul_reference_fma`（FMA 契約の
//! 唯一の参照点）、判定は `fandhe_ai_backend_cpu::parity::assert_parity`（REQ-2
//! 統一複合判定の唯一の実体。閾値の独自定義・緩和は禁止。
//! `.claude/rules/security.md`）を使う。入力生成は
//! `bench_harness::rng::Xorshift64Star`（決定的シード）。

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::parity::{assert_parity, matmul_reference_fma};
use fandhe_ai_backend_metal::{GemmVariant, MetalContext, MetalError, MetalGemm};

/// `variant`・`(seed_a, seed_b, m, n, k)` の 1 ケースを実行し、CPU 参照
/// 実装との複合判定 PASS を確認する。
fn run_case(variant: GemmVariant, seed_a: u64, seed_b: u64, m: usize, n: usize, k: usize) {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    let a = Xorshift64Star::new(seed_a).fill_vec(m * k);
    let b = Xorshift64Star::new(seed_b).fill_vec(k * n);

    let mut expected = vec![0.0f32; m * n];
    matmul_reference_fma(&a, &b, &mut expected, m, n, k)
        .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した");

    let actual = gemm
        .dispatch_variant(&ctx, variant, &a, &b, m, n, k)
        .unwrap_or_else(|err| panic!("Metal {variant:?} GEMM のディスパッチに失敗した: {err}"));

    assert_parity(
        &format!("metal {variant:?} gemm m={m} n={n} k={k}"),
        &actual,
        &expected,
    );
}

/// threadgroup サイズ（16）・simdgroup タイル（8）いずれの倍数でもない
/// 形状。tiled は `shaders/gemm.metal` の手動境界チェック（`row < m &&
/// a_col < k` 等）、simdgroup は `crate::pad` によるパディング経路と
/// タイル原点の早期 return が実際に効くケース（REQ-8）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn tiled_matches_cpu_reference_non_multiple_of_tile() {
    run_case(GemmVariant::Tiled, 1, 2, 7, 13, 5);
    run_case(GemmVariant::Tiled, 3, 4, 33, 65, 17);
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn simdgroup_matches_cpu_reference_non_multiple_of_eight() {
    run_case(GemmVariant::Simdgroup, 1, 2, 7, 13, 5);
    run_case(GemmVariant::Simdgroup, 3, 4, 33, 65, 17);
}

/// 中規模形状（境界チェック検証用。1 threadgroup を大きく超える grid で
/// 末尾ブロックの境界処理を確認する）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn tiled_matches_cpu_reference_medium_shape() {
    run_case(GemmVariant::Tiled, 5, 6, 128, 96, 72);
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn simdgroup_matches_cpu_reference_medium_shape() {
    run_case(GemmVariant::Simdgroup, 5, 6, 128, 96, 72);
}

/// 縮退形状（m=1・n=1・k=1）。tiled は 1 threadgroup 内でほぼ全スレッドが
/// 手動境界チェックで早期 return する経路、simdgroup は `crate::pad` に
/// よる 8 の倍数へのパディングとタイル原点の早期 return が実際に効く
/// 経路を確認する（REQ-8。TASK-1.8e・#42 で追加）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn tiled_matches_cpu_reference_degenerate_shapes() {
    run_case(GemmVariant::Tiled, 9, 10, 1, 1, 1);
    run_case(GemmVariant::Tiled, 11, 12, 1, 8, 4);
    run_case(GemmVariant::Tiled, 13, 14, 8, 1, 4);
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn simdgroup_matches_cpu_reference_degenerate_shapes() {
    run_case(GemmVariant::Simdgroup, 9, 10, 1, 1, 1);
    run_case(GemmVariant::Simdgroup, 11, 12, 1, 8, 4);
    run_case(GemmVariant::Simdgroup, 13, 14, 8, 1, 4);
}

/// K ストレスケース（PoC-v2-5 の FMA 契約実測ケースに対応。m=n=64,
/// k=4096。長い内積での丸め誤差蓄積が CPU 参照実装〈`f32::mul_add`〉と
/// 一致することを確認する）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn tiled_matches_cpu_reference_k_stress() {
    run_case(GemmVariant::Tiled, 7, 8, 64, 64, 4096);
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn simdgroup_matches_cpu_reference_k_stress() {
    run_case(GemmVariant::Simdgroup, 7, 8, 64, 64, 4096);
}

/// 零次元・長さ不一致が FFI 呼び出し前に型付きエラーで拒否されることを
/// `dispatch_variant`（`Simdgroup` 経路）でも確認する（OWASP A03 観点。
/// `.claude/rules/security.md`）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dispatch_variant_rejects_invalid_shapes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    let err = gemm
        .dispatch_variant(&ctx, GemmVariant::Simdgroup, &[], &[], 0, 4, 3)
        .unwrap_err();
    assert!(matches!(
        err,
        MetalError::ZeroDimension { m: 0, n: 4, k: 3 }
    ));

    let a = vec![0.0f32; 5]; // m*k=6 を期待
    let b = vec![0.0f32; 12];
    let err = gemm
        .dispatch_variant(&ctx, GemmVariant::Simdgroup, &a, &b, 2, 4, 3)
        .unwrap_err();
    assert!(matches!(
        err,
        MetalError::ALenMismatch {
            expected: 6,
            actual: 5
        }
    ));
}
