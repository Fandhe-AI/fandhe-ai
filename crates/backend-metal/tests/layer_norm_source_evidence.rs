//! イシュー #1596: LayerNorm 順伝播カーネル（MSL）の文字列証跡テスト。
//! `tests/rmsnorm_softmax_source_evidence.rs` と同方針: `include_str!`
//! によるビルド時文字列埋め込みへの contains 検査のみで完結するため、
//! Metal 実機・`cfg(target_os = "macos")` を必要とせず Linux CI
//! （GitHub ホステッド）上でも green になる。
//!
//! `.claude/rules/coding-rust.md`「REQ-8: 性能下限・最適化の達成を理由に
//! 手動境界チェックを省略しない」の機械検証と、
//! `crates/backend-metal/src/shaders/layer_norm.metal` 冒頭コメントが
//! 明記するアルゴリズム契約（1 threadgroup = 1 simdgroup 固定・
//! `simd_shuffle_xor` 5 段 butterfly・`threadgroup_barrier` 非使用・
//! `ulong row_base` によるオーバーフロー安全な添字）のロックを兼ねる。

/// `crates/backend-metal/src/shaders/layer_norm.metal` のソース全文。
const LAYER_NORM_METAL_SOURCE: &str = include_str!("../src/shaders/layer_norm.metal");

/// REQ-8 境界検査の一環: 行アドレス計算に `ulong`（64-bit）を使い、
/// `rows * hidden` が `u32` の範囲を超える巨大形状でも添字計算が
/// オーバーフローしないことをロックする
/// （`rmsnorm.metal` と同じ対策。CUDA 側 PR #706 是正と同等）。
#[test]
fn row_base_uses_64bit_index_to_avoid_overflow() {
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("ulong row_base = (ulong)row * (ulong)hidden;"),
        "行アドレス計算が ulong（64-bit）を使っていません（オーバーフロー安全性の回帰）"
    );
}

/// 平均（`ln_reduce_sum`）・分散（`ln_reduce_ssq`）の両方が
/// `simd_shuffle_xor` を用いた 5 段 butterfly（`offset` を 16u→1u へ
/// 5 回半減させるループ）で reduction されることをロックする。
#[test]
fn both_reductions_use_five_stage_butterfly() {
    let occurrences = LAYER_NORM_METAL_SOURCE
        .matches("for (uint offset = 16u; offset > 0u; offset >>= 1u)")
        .count();
    assert_eq!(
        occurrences, 2,
        "5 段 butterfly ループ（16u→1u の 5 回半減）は平均・分散の 2 箇所に存在するはずだが \
         {occurrences} 箇所しか見つからなかった"
    );
}

/// 分散の二乗和 reduction が `scale`／`ssq`／補償項 `comp` の 3 つすべて
/// を `simd_shuffle_xor` することをロックする（`rmsnorm.metal` と同じ
/// overflow-safe な scale/ssq 方式。いずれかの shuffle が失われると
/// `f64` アキュムレータ相当の精度契約が崩れる）。
#[test]
fn variance_reduction_shuffles_scale_ssq_and_compensation() {
    for var_name in ["scale", "ssq", "comp"] {
        let needle = format!("simd_shuffle_xor({var_name}, offset)");
        assert!(
            LAYER_NORM_METAL_SOURCE.contains(&needle),
            "分散 reduction が `{var_name}` を shuffle していません（overflow-safe 精度契約 \
             が壊れている可能性）"
        );
    }
}

/// 平均の合計 reduction が `sum`／`comp` の Neumaier 補償和ペアを
/// `simd_shuffle_xor` することをロックする。
#[test]
fn mean_reduction_shuffles_sum_and_compensation() {
    for var_name in ["sum", "comp"] {
        let needle = format!("simd_shuffle_xor({var_name}, offset)");
        assert!(
            LAYER_NORM_METAL_SOURCE.contains(&needle),
            "平均 reduction が `{var_name}` を shuffle していません（Neumaier 補償和契約が \
             壊れている可能性）"
        );
    }
}

/// `threadgroup_barrier` を使わないことをロックする（1 threadgroup =
/// 1 simdgroup 固定・threadgroup memory を使わない設計。`rmsnorm.metal`
/// と同じ理由）。
#[test]
fn does_not_use_threadgroup_barrier() {
    assert!(
        !LAYER_NORM_METAL_SOURCE.contains("threadgroup_barrier("),
        "layer_norm.metal は threadgroup_barrier を使わない設計のはず（1 threadgroup = \
         1 simdgroup 固定の契約が崩れている可能性）"
    );
}

/// NaN／inf 伝播の明示処理（`isnan`／`isinf`）が分散計算の scale/ssq
/// ヘルパーに残っていることをロックする（`rmsnorm.metal` の同名契約と
/// 同じ理由。codex-review 指摘・PR #1120 の教訓を LayerNorm 側でも
/// 引き継ぐ）。
#[test]
fn variance_helpers_explicitly_handle_nan_and_inf() {
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("isnan(a)"),
        "ln_ssq_add に NaN 検出（isnan）が見つかりません"
    );
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("isinf(scale) && isinf(a)"),
        "ln_ssq_add に inf 同士の特殊分岐（isinf(scale) && isinf(a)）が見つかりません"
    );
}

/// persistent threadgroup 方式（`for (row = tg_id; row < rows; row +=
/// grid_size)`）を使うことをロックする（`grid_size` はホスト側
/// `row_kernel::derive_persistent_grid` が導出する単一の真実源。
/// `rmsnorm.metal` と同じ設計）。
#[test]
fn uses_persistent_threadgroup_loop() {
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("for (uint row = tg_id; row < rows; row += grid_size)"),
        "persistent threadgroup ループ（tg_id から grid_size ストライド）が見つかりません"
    );
}

/// 単一カーネル `layer_norm_f32` のバッファ引数が `x`／`w`／`b`／`out`
/// の 4 本（index 0〜3）であることをロックする（`ops.rs`／`layer_norm.rs`
/// のバッファ結線順序と MSL 側の引数宣言が食い違うと、コンパイルは
/// 通るがカーネルが誤った引数を読む黙示のバグになるため）。
#[test]
fn kernel_declares_four_buffer_arguments_in_expected_order() {
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("device const float* x [[buffer(0)]]"),
        "buffer(0) が x であることが見つかりません"
    );
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("device const float* w [[buffer(1)]]"),
        "buffer(1) が w であることが見つかりません"
    );
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("device const float* b [[buffer(2)]]"),
        "buffer(2) が b であることが見つかりません"
    );
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("device float* out [[buffer(3)]]"),
        "buffer(3) が out であることが見つかりません"
    );
}
