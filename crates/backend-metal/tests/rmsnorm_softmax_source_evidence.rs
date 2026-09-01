//! イシュー #604: 融合 RMSNorm 順伝播・online softmax カーネル（MSL）の
//! 文字列証跡テスト。`tests/shader_source_evidence.rs`（REQ-11 行列演算
//! ユニット証跡）と同方針: `include_str!` によるビルド時文字列埋め込みへの
//! contains 検査のみで完結するため、Metal 実機・`cfg(target_os = "macos")`
//! を必要とせず Linux CI（GitHub ホステッド）上でも green になる。
//!
//! `.claude/rules/coding-rust.md`「REQ-8: 性能下限・最適化の達成を理由に
//! 手動境界チェックを省略しない」の機械検証と、実装計画 §6.1 が要求する
//! アルゴリズム契約（`simd_shuffle_xor`・`threadgroup_barrier` 非使用・
//! `exp2` のみ使用・有限負値境界マスク）のロックを兼ねる。

/// `crates/backend-metal/src/shaders/rmsnorm.metal` のソース全文。
const RMSNORM_METAL_SOURCE: &str = include_str!("../src/shaders/rmsnorm.metal");

/// `crates/backend-metal/src/shaders/softmax.metal` のソース全文。
const SOFTMAX_METAL_SOURCE: &str = include_str!("../src/shaders/softmax.metal");

/// `row_kernel::ONEPASS_MAX_HIDDEN`（Rust 側）と MSL 側固定長配列の
/// 宣言サイズが一致することを、ソース文字列中の数値リテラルの突合で
/// ロックする（`crate::row_kernel` モジュール冒頭コメント「単一の真実源」
/// 参照）。
#[test]
fn onepass_max_hidden_matches_row_kernel_constant() {
    for src in [RMSNORM_METAL_SOURCE, SOFTMAX_METAL_SOURCE] {
        assert!(
            src.contains("4096u"),
            "固定長 threadgroup 配列の宣言サイズ（4096u）が見つかりません。\
             row_kernel::ONEPASS_MAX_HIDDEN（4096）と同期させること"
        );
    }
}

/// softmax 側の 5 段 butterfly reduction（`simd_shuffle_xor` 幅
/// 16/8/4/2/1 の展開済み 5 行）の使用をロックする（実装計画 §4.1
/// 「reduction は 5 段 butterfly」）。RMSNorm 側は縮約精度契約
/// （§9.8 追補・イシュー #1102）により Kahan 補償和のループ形式
/// （`rmsnorm_uses_five_stage_kahan_butterfly_reduction`）へ変更した
/// ため、本テストの対象から分離した。
#[test]
fn softmax_uses_five_stage_butterfly_reduction() {
    for width in ["16u", "8u", "4u", "2u", "1u"] {
        let needle = format!("simd_shuffle_xor(v, {width})");
        assert!(
            SOFTMAX_METAL_SOURCE.contains(&needle),
            "5 段 butterfly reduction の shuffle 幅 `{width}` が見つかりません: {needle}"
        );
    }
}

/// RMSNorm 側の 5 段 butterfly reduction は Kahan 補償和（`f64`
/// アキュムレータ相当。§9.8 追補・イシュー #1102）のレーン間結合へ
/// 変更されており、`offset` を 16u→1u へ 5 回半減させるループ
/// （softmax 側の展開済み 5 行とは異なる構造だが段数の契約は同じ）で
/// `sum`・補償項 `comp` の両方を `simd_shuffle_xor` することをロックする
/// （`comp` の shuffle が失われると `f64` 相当の精度契約が崩れるため
/// 特に重要）。
#[test]
fn rmsnorm_uses_five_stage_kahan_butterfly_reduction() {
    assert!(
        RMSNORM_METAL_SOURCE.contains("for (uint offset = 16u; offset > 0u; offset >>= 1u)"),
        "RMSNorm の Kahan butterfly ループ（16u→1u の 5 段半減）が見つかりません"
    );
    assert!(
        RMSNORM_METAL_SOURCE.contains("simd_shuffle_xor(sum, offset)"),
        "RMSNorm の Kahan butterfly が sum を shuffle していません"
    );
    assert!(
        RMSNORM_METAL_SOURCE.contains("simd_shuffle_xor(comp, offset)"),
        "RMSNorm の Kahan butterfly が補償項 comp を shuffle していません          （f64 相当の精度契約が壊れている可能性）"
    );
}

/// `threadgroup_barrier` を使わず `simdgroup_barrier(mem_flags::
/// mem_threadgroup)` のみを使うことをロックする（実装計画 §4.1「1
/// threadgroup = 1 simdgroup 固定」）。
#[test]
fn rmsnorm_and_softmax_do_not_use_threadgroup_barrier() {
    for src in [RMSNORM_METAL_SOURCE, SOFTMAX_METAL_SOURCE] {
        assert!(
            !src.contains("threadgroup_barrier"),
            "1 threadgroup = 1 simdgroup 固定のカーネルは threadgroup_barrier を使わない契約\
             （simdgroup_barrier のみ使用）"
        );
        assert!(
            src.contains("simdgroup_barrier(mem_flags::mem_threadgroup)"),
            "simdgroup_barrier(mem_flags::mem_threadgroup) が見つかりません"
        );
    }
}

/// softmax は `exp2` のみを使い `exp(` を使わないことをロックする
/// （実装計画 §4.3「`log2(e)` スケール + `exp2` のみ使用」）。
#[test]
fn softmax_uses_exp2_only_not_exp() {
    assert!(
        SOFTMAX_METAL_SOURCE.contains("exp2("),
        "softmax.metal に exp2( が見つかりません"
    );
    assert!(
        !SOFTMAX_METAL_SOURCE.contains("exp("),
        "softmax.metal は exp( を直接使わない契約（exp2 のみ使用）"
    );
}

/// softmax の境界外レーン sentinel が有限負値（`-INFINITY` を直接使わない）
/// であり、かつ sum への寄与が `valid` フラグで明示的にゲートされている
/// ことをロックする（実装計画 §4.3「境界マスク」・PR #714 codex-review
/// 是正: sentinel の大小関係のみに依存した暗黙除外は `-f32::MAX` 付近の
/// 有限入力で sum を汚染しうるため、`valid ? ... : 0.0` の明示ゲートへ
/// 是正した。`shaders/softmax.metal` ファイル冒頭コメント参照）。
#[test]
fn softmax_uses_finite_negative_sentinel_not_infinity_and_gates_sum_by_valid() {
    assert!(
        SOFTMAX_METAL_SOURCE.contains("SOFTMAX_NEG_FLT_MAX"),
        "softmax.metal に SOFTMAX_NEG_FLT_MAX が見つかりません"
    );
    assert!(
        !SOFTMAX_METAL_SOURCE.contains("INFINITY"),
        "softmax.metal は -INFINITY を直接使わない契約（有限負値 sentinel のみ使用）"
    );
    assert!(
        SOFTMAX_METAL_SOURCE.contains("valid ? exp2((xv - m_new) * SOFTMAX_LOG2E) : 0.0f"),
        "sum への寄与を valid フラグで明示的にゲートする式が見つかりません"
    );
}

/// ベクトル化ロード（`float4`）を適用する経路が `hidden % 4 == 0` の
/// 分岐に限定され、かつ手動境界チェック（`base + 3u < hidden`）を
/// 維持していることをロックする（REQ-8。`.claude/rules/coding-rust.md`
/// 「性能下限・最適化の達成を理由に手動境界チェックを省略しない」）。
#[test]
fn rmsnorm_vectorized_path_keeps_manual_bounds_check() {
    assert!(
        RMSNORM_METAL_SOURCE.contains("hidden % 4u == 0u"),
        "ベクトル化経路のゲート条件（hidden % 4u == 0u）が見つかりません"
    );
    assert!(
        RMSNORM_METAL_SOURCE.contains("base + 3u < hidden"),
        "ベクトル化経路の手動境界チェック（base + 3u < hidden）が見つかりません"
    );
}

/// ループ添字のオーバーフロー安全性: 行頭オフセット（`row_base`）が
/// `ulong` で宣言されていることをロックする（CUDA 側 PR #706 是正と
/// 同等の対策。実装計画 §4.1「REQ-8 境界検査」）。
#[test]
fn rmsnorm_and_softmax_row_base_is_declared_ulong() {
    for src in [RMSNORM_METAL_SOURCE, SOFTMAX_METAL_SOURCE] {
        assert!(
            src.contains("ulong row_base"),
            "row_base の ulong 宣言が見つかりません（オーバーフロー安全性）"
        );
    }
}

/// 縮約精度契約（§9.8 追補・イシュー #1102。ユーザー承認 2026-09-01）:
/// RMSNorm の二乗和累算は `f64` アキュムレータ相当の Kahan 補償和へ
/// 変更したため、単純な `fma()` 直接蓄積ではなく `rmsnorm_kahan_add`
/// ヘルパーの使用をロックする（`.claude/rules/coding-rust.md`「正規化
/// 統計・勾配の長軸縮約は f64 アキュムレータ（Metal は Kahan 補償和
/// f32）で統一する」契約）。正規化出力の積算（`v * rstd * wv` 等）自体
/// の FMA 契約は本テストの対象外（変更していない）。
#[test]
fn rmsnorm_uses_kahan_compensated_sum_for_accumulation() {
    assert!(
        RMSNORM_METAL_SOURCE.contains("rmsnorm_kahan_add(acc, acc_c, v.x * v.x)"),
        "RMSNorm の二乗和累算に Kahan 補償和ヘルパー（rmsnorm_kahan_add）が見つかりません"
    );
    assert!(
        !RMSNORM_METAL_SOURCE.contains("acc = fma(v.x, v.x, acc)"),
        "RMSNorm の二乗和累算が単純な fma() 直接蓄積へ後退している（Kahan 補償が失われる）"
    );
}

/// persistent threadgroup 方式（grid-stride ループ）の使用をロックする
/// （実装計画 §4.1「persistent threadgroup 方式」）。
#[test]
fn rmsnorm_and_softmax_use_persistent_threadgroup_loop() {
    for src in [RMSNORM_METAL_SOURCE, SOFTMAX_METAL_SOURCE] {
        assert!(
            src.contains("row += grid_size"),
            "persistent threadgroup の grid-stride ループが見つかりません"
        );
    }
}

/// 1 パス／2 パスの両カーネルエントリが定義されていることをロックする。
#[test]
fn rmsnorm_and_softmax_define_onepass_and_twopass_kernels() {
    for (name, prefix) in [("rmsnorm", "rmsnorm_f32"), ("softmax", "softmax_f32")] {
        let src = if name == "rmsnorm" {
            RMSNORM_METAL_SOURCE
        } else {
            SOFTMAX_METAL_SOURCE
        };
        for suffix in ["onepass", "twopass"] {
            let needle = format!("kernel void {prefix}_{suffix}(");
            assert!(
                src.contains(&needle),
                "{name} に `{needle}` が見つかりません"
            );
        }
    }
}
