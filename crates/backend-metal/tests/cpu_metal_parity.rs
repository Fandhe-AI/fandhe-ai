//! TASK-2.2c（#55）: CPU-Metal ペアの数値一致回帰テストスイート。
//!
//! REQ-2 統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」の
//! CPU-Metal ペア分を固定する。前提条件は 2 点:
//!
//! - (a) FMA 契約統一: CPU 参照は `backend_cpu::parity::matmul_reference_fma`
//!   （`f32::mul_add`）。GPU 側は Metal `simdgroup` 系命令の既定 FMA 契約。
//! - (b) Metal は `mathFloatingPointFunctions=Precise` 明示
//!   （`crate::pipeline::compile_options` で設定・`pipeline.rs` 内
//!   `#[cfg(test)]` で契約テスト化。本ファイルは実機での複合判定 PASS を
//!   固定する）。
//!
//! PoC-v2-5 では (a) 未適用条件で K=4096 ストレスケース
//! （512×512×4096）の CPU-Metal ペアが fail_cells=7/262144 で未達となり、
//! `mul_add` 差し替えで fail_cells=0 に解消することを確認実験で実測済み
//! （`docs/spec/03-poc/poc-v2-5-backend-numeric-parity/`）。(a) 適用後の
//! 再測定は「TASK-2.2 で確認すること」と spec に明記されており
//! （`docs/spec/04-requirements.md` REQ-2 受け入れ基準）、本ファイルの
//! `k4096_stress_poc_v2_5` ケースがその再測定に対応する。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する。CI（self-hosted・
//! Linux）では `#![cfg(target_os = "macos")]` によりコンパイル対象外に
//! なり、`#[ignore]` により通常の `cargo test` からも除外される
//! （実機依存テストの分離。`.claude/rules/coding-rust.md`。
//! `tests/gemm_naive_parity.rs`〈#39〉と同じ方針）。実行するには macOS
//! 実機で以下を叩く（K=4096 ストレスケースは debug では遅いため release
//! を推奨）:
//!
//! ```sh
//! cargo test -p backend-metal --release -- --ignored --nocapture
//! ```
//!
//! CPU 参照は `backend_cpu::parity::matmul_reference_fma`（FMA 契約の
//! 唯一の参照点）、判定は `backend_cpu::parity::{compare, assert_parity}`
//! （REQ-2 統一複合判定の唯一の実体。閾値の独自定義・緩和は禁止。
//! `.claude/rules/security.md`）を使う。入力生成は
//! `bench_harness::rng::Xorshift64Star`（決定的シード）。
//!
//! 実行手順・テスト一覧の正本は `docs/backend-metal-real-device-testing.md`
//! （TASK-1.8e・#42）を参照する。
//!
//! **`tests/gemm_naive_parity.rs`（#39）との関係**: #54（CPU-CUDA ペア・
//! PR #243）は `backend-cuda` 側の重複判定式を含む旧テストを削除し
//! `cpu_cuda_parity.rs` へ移管したが、本ファイルは `gemm_naive_parity.rs`
//! を削除・移管しない。同ファイルは既に唯一の判定ユーティリティ
//! （`backend_cpu::parity`）へ一本化済みで重複判定式を含まず、かつ
//! 「naive GEMM（#39）の受け入れ条件検証」という別目的を持つため
//! （本ファイルは TASK-2.2c の数値一致回帰テストという別目的）、削除する
//! 理由がない。境界形状ケースは意図的に `gemm_naive_parity.rs` とは
//! 異なる形状を選び、直接の重複を避けている。

#![cfg(target_os = "macos")]

use backend_cpu::parity::{assert_parity, compare, matmul_reference_fma};
use backend_metal::{MetalContext, MetalGemm};
use bench_harness::rng::Xorshift64Star;

/// `(seed_a, seed_b, m, n, k)` の 1 ケースを実行し、CPU 参照実装との
/// 複合判定 PASS を確認する。`tests/gemm_naive_parity.rs::run_case` と
/// 同型だが、本ファイルは TASK-2.2c（数値一致回帰テストそのもの）の
/// スコープとして独立させている（#39 の受け入れ条件検証とは目的が異なる。
/// 計画 #55 参照）。
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
        &format!("cpu-metal parity m={m} n={n} k={k}"),
        &actual,
        &expected,
    );
}

/// 基準形状（PoC-v2-5 基準形状。M=N=K=512）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn baseline_shape_512_matches_cpu_reference() {
    run_case(11, 12, 512, 512, 512);
}

/// K=4096 ストレスケース（M=N=512, K=4096。PoC-v2-5 で FMA 契約統一
/// 前は fail_cells=7/262144 で未達だった形状そのもの）。REQ-2 の
/// 「(a) 適用後の再測定は TASK-2.2 で確認」に対応する中核ケース。
/// 失敗時診断のため `compare` の `CompareReport` を直接検査し、分布統計を
/// メッセージへ出力する（`assert_parity` の整形メッセージと同等の情報を
/// このケースだけ明示的に持たせ、実機未達時に PoC-v2-5 との差分切り分けを
/// しやすくする）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn k4096_stress_poc_v2_5() {
    let m = 512;
    let n = 512;
    let k = 4096;

    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("naive GEMM パイプラインの構築に失敗した");

    let a = Xorshift64Star::new(21).fill_vec(m * k);
    let b = Xorshift64Star::new(22).fill_vec(k * n);

    let mut expected = vec![0.0f32; m * n];
    matmul_reference_fma(&a, &b, &mut expected, m, n, k)
        .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した");

    let actual = gemm
        .dispatch(&ctx, &a, &b, m, n, k)
        .expect("Metal naive GEMM のディスパッチに失敗した");

    let report = compare(&actual, &expected).expect("長さは m*n で一致するはず");
    assert!(
        report.passes(),
        "cpu-metal k4096 stress FAIL（fail_count={}/{}, max_abs_diff={:.3e}, \
         max_rel_err={:.3e}, mean_abs_diff={:.3e}, mean_rel_err={:.3e}, \
         p50_abs_diff={:.3e}, p99_abs_diff={:.3e}, p999_abs_diff={:.3e}。\
         PoC-v2-5 実測: FMA 契約統一前は fail_cells=7/262144）",
        report.fail_count,
        report.total,
        report.max_abs_diff,
        report.max_rel_err,
        report.mean_abs_diff,
        report.mean_rel_err,
        report.p50_abs_diff,
        report.p99_abs_diff,
        report.p999_abs_diff,
    );
}

/// threadgroup（16）非倍数・非正方の境界形状。シェーダ手動境界チェック
/// （REQ-8・`shaders/gemm.metal` の `gid.y >= dims.m || gid.x >= dims.n`）
/// が実際に効く経路の回帰。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn boundary_shapes_non_multiple_of_threadgroup() {
    // `gemm_naive_parity.rs` の (7,13,5)/(33,65,17)/(128,96,72) とは
    // 意図的に異なる形状を選び、直接の重複を避ける（ファイル冒頭コメント
    // 参照）。
    run_case(61, 62, 19, 41, 23);
    run_case(63, 64, 97, 130, 55);
}

/// 決定性テスト: 同一シードで 2 回 dispatch した結果が bit 完全一致する
/// こと。回帰テストとしての再現性を固定する（Metal 側の非同期実行・
/// スレッドスケジューリングの非決定性が出力へ混入しないことの確認）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dispatch_is_bit_deterministic_across_runs() {
    let m = 64;
    let n = 64;
    let k = 128;

    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("naive GEMM パイプラインの構築に失敗した");

    let a = Xorshift64Star::new(41).fill_vec(m * k);
    let b = Xorshift64Star::new(42).fill_vec(k * n);

    let first = gemm
        .dispatch(&ctx, &a, &b, m, n, k)
        .expect("1 回目の dispatch に失敗した");
    let second = gemm
        .dispatch(&ctx, &a, &b, m, n, k)
        .expect("2 回目の dispatch に失敗した");

    assert_eq!(
        first, second,
        "同一入力の 2 回 dispatch が bit 完全一致しない"
    );
}

/// falsification テスト: Metal 出力の 1 要素へ複合判定を確実に外れる
/// 摂動（+1.0）を注入し、`compare` が `fail_count > 0` を返すことを
/// 確認する（判定器が「常に PASS」に壊れていないことの確認。
/// `crates/backend-cpu/src/parity.rs` の falsification テストと
/// `tests/fma_contract.rs` の前例踏襲。PoC-v2-5 方針）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn falsification_injected_perturbation_is_detected() {
    let m = 32;
    let n = 32;
    let k = 32;

    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("naive GEMM パイプラインの構築に失敗した");

    let a = Xorshift64Star::new(51).fill_vec(m * k);
    let b = Xorshift64Star::new(52).fill_vec(k * n);

    let mut expected = vec![0.0f32; m * n];
    matmul_reference_fma(&a, &b, &mut expected, m, n, k)
        .expect("CPU 参照実装（matmul_reference_fma）の形状検証に失敗した");

    let mut actual = gemm
        .dispatch(&ctx, &a, &b, m, n, k)
        .expect("Metal naive GEMM のディスパッチに失敗した");

    // 複合判定を確実に外れる摂動を注入する（相対誤差・絶対誤差とも閾値超え）。
    actual[0] += 1.0;

    let report = compare(&actual, &expected).expect("長さは m*n で一致するはず");
    assert!(
        !report.passes(),
        "falsification: 摂動注入後も複合判定が PASS のままになっている（判定器の故障疑い）"
    );
    assert!(report.fail_count >= 1);
}
