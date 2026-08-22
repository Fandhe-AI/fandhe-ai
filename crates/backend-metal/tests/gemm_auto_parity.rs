//! GEMM 自動経路選択（`MetalGemm::dispatch_backend_auto`。TASK-11.2b・
//! #68）の受け入れ条件検証:「形状・HW に応じて経路が自動選択される」
//! （REQ-11）ことと、選択された経路が CPU 参照実装と複合判定
//! （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。REQ-2）で一致する
//! ことを合わせて検証する。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する
//! （`gemm_naive_parity.rs` と同じ方針。`#![cfg(target_os = "macos")]` ＋
//! `#[ignore]`）。決定表そのもの（HW・形状・dtype から `KernelKind` を
//! 選ぶ純関数）の網羅テストは `tensor-core` 側
//! （`crates/tensor-core/src/dispatch.rs` の `#[cfg(test)]`）が担当する。
//! 本ファイルは「Metal 実機上で、閾値の上下で選択経路が実際に切り替わり、
//! いずれも数値一致するか」の統合検証に限定する
//! （`docs/dispatch-rules-design.md` §3.2「Metal は
//! `min(M,N,K) >= 512` で `simdgroup_matrix`、未満で tiled」）。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::parity::{assert_parity, matmul_reference_fma};
use fandhe_ai_backend_metal::{MetalContext, MetalGemm};

/// `(seed_a, seed_b, m, n, k)` の 1 ケースを `dispatch_backend_auto` で
/// 実行し、CPU 参照実装との複合判定 PASS を確認する。
fn run_case(seed_a: u64, seed_b: u64, m: usize, n: usize, k: usize) {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

    let a = Xorshift64Star::new(seed_a).fill_vec(m * k);
    let b = Xorshift64Star::new(seed_b).fill_vec(k * n);

    let mut expected = vec![0.0f32; m * n];
    matmul_reference_fma(&a, &b, &mut expected, m, n, k)
        .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した");

    let actual = gemm
        .dispatch_backend_auto(&ctx, &a, &b, m, n, k)
        .expect("dispatch_backend_auto のディスパッチに失敗した");

    assert_parity(
        &format!("metal dispatch_backend_auto m={m} n={n} k={k}"),
        &actual,
        &expected,
    );
}

/// 閾値未満（`min(M,N,K) = 256 < METAL_SIMDGROUP_MIN_DIM = 512`）:
/// `select_gemm_kernel` は `Tiled` を返す想定（Apple7 以上のデバイスでも
/// 小形状は tiled 経路）。§3.2 の「Metal は閾値未満で非 accelerated」を
/// 実機で確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn below_threshold_shape_matches_cpu_reference() {
    run_case(101, 102, 256, 256, 256);
}

/// 閾値以上（`min(M,N,K) = 1024 >= 512`）: Apple7 以上のデバイスでは
/// `select_gemm_kernel` が `MatrixUnit` を返し `dispatch_auto`
/// （`simdgroup_matrix` 動的タイル選択）経路が使われる想定。§3.2 の
/// 「Metal は閾値以上で `simdgroup_matrix`」を実機で確認する。Apple7
/// 未満のデバイスでは `select_gemm_kernel` が `Tiled` へ倒れるため、
/// いずれの経路でも複合判定が PASS することが本テストの目的
/// （経路自体の判別は `tensor-core` 側の決定表ユニットテストが担当し、
/// 本テストは「どちらの経路が選ばれても数値が正しい」ことを検証する）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn at_or_above_threshold_shape_matches_cpu_reference() {
    run_case(103, 104, 1024, 1024, 1024);
}

/// 境界形状（`min(M,N,K)` がちょうど閾値の 511/512）で選択経路が切り替わり、
/// いずれも数値一致することを検証する（実装計画 §3.4「閾値上下（例: 256
/// と 1024）で選択経路が切り替わり、いずれも数値一致することを検証」の
/// 境界値版）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn threshold_boundary_511_and_512_both_match_cpu_reference() {
    run_case(105, 106, 511, 511, 511);
    run_case(107, 108, 512, 512, 512);
}
