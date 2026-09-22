//! イシュー #2070: `shaders/adam.metal` に REQ-8 境界検査・FP 縮約禁止
//! （`#pragma METAL fp contract(off)`）・意図した FMA 使用（`fma(`）・
//! 正確丸め平方根（`precise::sqrt`）が実在することを機械検査する証跡
//! テスト（`mse_source_evidence.rs` と同型）。
//!
//! `include_str!` によるビルド時文字列埋め込みへの contains 検査のみで
//! 完結するため、Metal 実機・`cfg(target_os = "macos")` を必要とせず
//! Linux CI 上でも green になる（本リポジトリで Metal 実機・数値一致を
//! 直接検証できる `#[ignore]` テスト〈`adam_device_parity.rs`〉と異なり、
//! 本テストは Linux 上での唯一の CUDA 側 `kernels_adam.rs::tests` に
//! 対応する証跡）。
//!
//! `ops.rs` に対する検査（`adam_source_evidence_locks_ops_wiring`）は
//! `ops.rs` 自体が `cfg(target_os = "macos")` 限定のため Linux では
//! 型検査すら走らない——カーネルは実装済みだが `ops.rs` の override
//! 結線が漏れ既定 `Unsupported` に落ちる回帰（#1730 追従イシューで
//! 実際に起きた事例）を Linux CI で検知する意図的な文字列検査であり、
//! hack ではない。

/// `crates/backend-metal/src/shaders/adam.metal` のソース全文。
const ADAM_METAL_SOURCE: &str = include_str!("../src/shaders/adam.metal");

/// `crates/backend-metal/src/ops.rs` のソース全文。`adam_model::
/// validate_adam_step_inputs` 相当の結線（本 PR では
/// `adam_model::validate_adam_step_shapes`／`adam_kernel_flags`）と
/// `adam_step_device_tracked` override の存在をロックする。
const OPS_SOURCE: &str = include_str!("../src/ops.rs");

/// コメント行（`//` 始まり）を除去したソースを返す（`#pragma` の出現
/// 順序検査がヘッダコメント中の言及に惑わされないようにするため）。
fn strip_line_comments(src: &str) -> String {
    src.lines()
        .map(|line| line.trim_start())
        .filter(|line| !line.starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// REQ-8: `adam_step_f32` が手動境界チェック `if (idx < numel)` を
/// 維持していることをロックする。コメント行除去
/// （`strip_line_comments`）後・カーネル本体（`kernel void
/// adam_step_f32` 以降）に限定して検査することで、ヘッダコメント中の
/// 同一文言の言及（本ファイル冒頭の説明コメント）だけで誤って
/// 成功しないようにする（境界チェック本体を削除する回帰を確実に
/// 検知するため）。
#[test]
fn adam_metal_source_has_bound_check() {
    let stripped = strip_line_comments(ADAM_METAL_SOURCE);
    let kernel_pos = stripped
        .find("kernel void adam_step_f32")
        .expect("`kernel void adam_step_f32` が見つかりません");
    let kernel_body = &stripped[kernel_pos..];
    assert!(
        kernel_body.contains("if (idx < numel)"),
        "adam_step_f32 の手動境界チェック `if (idx < numel)` が見つかりません"
    );
}

/// FP 縮約禁止契約: `#pragma METAL fp contract(off)` が `kernel void
/// adam_step_f32` より前（ファイルスコープ）に出現することをロックする
/// （MSL 仕様: 既定 `fast`、`MTLMathMode::Safe` でも縮約は `on` 止まり。
/// `coding-rust.md` の FMA 契約統一を守るための明示的無効化）。
#[test]
fn adam_metal_source_disables_fp_contract_before_kernel() {
    let stripped = strip_line_comments(ADAM_METAL_SOURCE);
    let pragma_pos = stripped
        .find("#pragma METAL fp contract(off)")
        .expect("`#pragma METAL fp contract(off)` が見つかりません");
    let kernel_pos = stripped
        .find("kernel void adam_step_f32")
        .expect("`kernel void adam_step_f32` が見つかりません");
    assert!(
        pragma_pos < kernel_pos,
        "`#pragma METAL fp contract(off)` は `adam_step_f32` の定義より前になければなりません"
    );
}

/// CPU 参照実装が明示的に `f32::mul_add` を使う 2 箇所（`m`／`v` の
/// 指数移動平均更新）に対応する `fma(` 呼び出しが 3 箇所以上あることを
/// ロックする（`g_eff` の coupled weight decay 分岐 + `m`／`v` 更新の
/// 3 箇所）。コメント行除去（`strip_line_comments`）後・カーネル本体
/// （`kernel void adam_step_f32` 以降）に限定して数えることで、ヘッダ
/// コメント中の `fma` 言及（本ファイル冒頭の説明コメント）を実呼び出し
/// として誤カウントしないようにする（実呼び出しが 1 個欠けても
/// ヘッダコメント分でしきい値 3 を満たしてしまう回帰を防ぐため）。
#[test]
fn adam_metal_source_uses_fma_at_least_three_times() {
    let stripped = strip_line_comments(ADAM_METAL_SOURCE);
    let kernel_pos = stripped
        .find("kernel void adam_step_f32")
        .expect("`kernel void adam_step_f32` が見つかりません");
    let kernel_body = &stripped[kernel_pos..];
    let count = kernel_body.matches("fma(").count();
    assert!(count >= 3, "fma( の出現数が想定より少ない: count={count}");
}

/// 正確丸め平方根 `precise::sqrt` を使い、近似 intrinsic（`rsqrt`／
/// `fast::`）を使わないことをロックする（`elementwise.metal`／
/// `bce.metal` の `precise::exp` と同じ「コンパイルオプションだけに
/// 委ねない」方針。イシュー #1105／#1893 の教訓の Metal 対応）。
#[test]
fn adam_metal_source_uses_precise_sqrt_without_fast_approximations() {
    assert!(
        ADAM_METAL_SOURCE.contains("precise::sqrt("),
        "precise::sqrt( が見つかりません"
    );
    assert!(
        !ADAM_METAL_SOURCE.contains("rsqrt"),
        "近似 intrinsic rsqrt は使わないはず"
    );
    assert!(
        !ADAM_METAL_SOURCE.contains("fast::"),
        "近似 intrinsic 名前空間 fast:: は使わないはず"
    );
}

/// 決定性: `atomic` 系命令を使わないことをロックする（`mse.metal` と
/// 同じ決定性契約）。
#[test]
fn adam_metal_source_has_no_atomics() {
    assert!(
        !ADAM_METAL_SOURCE.to_lowercase().contains("atomic"),
        "adam.metal に atomic 系命令が含まれています（決定性契約違反）"
    );
}

/// `ops.rs`（macOS 限定・Linux では型検査すら走らない）に `adam_model`
/// 経由の検証呼び出しと `adam_step_device_tracked` override が実在する
/// ことを文字列検査でロックする（本モジュール doc 参照。#1730 追従
/// イシューの教訓を踏襲した意図的な回帰検知）。
#[test]
fn adam_source_evidence_locks_ops_wiring() {
    assert!(
        OPS_SOURCE.contains("adam_model::validate_adam_step_shapes("),
        "ops.rs に adam_model::validate_adam_step_shapes( の呼び出しが見つかりません"
    );
    assert!(
        OPS_SOURCE.contains("adam_model::adam_kernel_flags("),
        "ops.rs に adam_model::adam_kernel_flags( の呼び出しが見つかりません"
    );
    assert!(
        OPS_SOURCE.contains("fn adam_step_device_tracked("),
        "ops.rs に adam_step_device_tracked の override が見つかりません"
    );
    assert!(
        OPS_SOURCE.contains("context_cache::cached_adam("),
        "ops.rs に context_cache::cached_adam( の呼び出しが見つかりません"
    );
}
