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

/// `maxabs`（行の 2 の冪スケール導出）・平均（`ln_reduce_kahan`）・
/// 分散（`ln_reduce_ssq`）の 3 箇所すべてが `simd_shuffle_xor` を用いた
/// 5 段 butterfly（`offset` を 16u→1u へ 5 回半減させるループ）で
/// reduction されることをロックする（codex-review 指摘を受けた Welford
/// → 2 の冪スケーリング + 2 段補償和への設計変更。`layer_norm.metal`
/// 冒頭コメント参照）。
#[test]
fn all_three_reductions_use_five_stage_butterfly() {
    let occurrences = LAYER_NORM_METAL_SOURCE
        .matches("for (uint offset = 16u; offset > 0u; offset >>= 1u)")
        .count();
    assert_eq!(
        occurrences, 3,
        "5 段 butterfly ループ（16u→1u の 5 回半減）は maxabs・平均・分散の 3 箇所に \
         存在するはずだが {occurrences} 箇所しか見つからなかった"
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

/// 平均の reduction が Neumaier 補償和の `(sum, comp)` ペアを
/// `simd_shuffle_xor` することをロックする（codex-review 指摘（平均を
/// 偏差計算前に丸めると精度が失われる・`meanB-meanA` の単純減算が
/// overflow しうる）を受け、Welford オンライン平均を破棄し「行内 2 の
/// 冪スケーリングした比スケール領域での Neumaier 補償和 → doubled-float
/// 拡張」設計へ変更した。`docs/norm-ops-design.md`・`layer_norm.metal`
/// 冒頭コメント参照）。
#[test]
fn mean_reduction_shuffles_sum_and_comp() {
    for var_name in ["sum", "comp"] {
        let needle = format!("simd_shuffle_xor({var_name}, offset)");
        assert!(
            LAYER_NORM_METAL_SOURCE.contains(&needle),
            "平均 reduction が `{var_name}` を shuffle していません（Neumaier 補償和の \
             overflow-safe 契約が壊れている可能性）"
        );
    }
}

/// 平均が `row_scale`（2 の冪。`ln_pow2_scale_from_maxabs`）で除した
/// 比スケール領域の値を Neumaier 補償和で蓄積し、`hidden` による厳密
/// 除算を FMA による doubled-float 拡張（`mean_hi`／`mean_lo`）で保持
/// することをロックする（codex-review 指摘の回帰防止: 平均を偏差計算前
/// に単一 `f32` へ丸める実装への逆戻りを検出する）。
///
/// PR #1671 codex-review 指摘（P1）の是正により、ホストが事前丸めした
/// `inv_n`〈`1/hidden`〉への乗算ベースの Dekker 分割から、`hidden` 自体
/// （`hidden_f`）への直接除算ベースの Dekker 型 div へ変更済み
/// （`inv_n` 自身の丸め誤差が `mean_lo` へ残存し `eps` 由来の極小
/// `scale` で増幅される問題の根治。`docs/backend-metal-splitk-decision.md`
/// と同様、ロック対象の期待文字列も実装変更と同じ PR 内で更新する）。
#[test]
fn mean_pass_uses_row_scale_and_doubled_float_extension() {
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("float ratio = x[row_base + idx] / row_scale;"),
        "平均パスが行の比スケール領域（x/row_scale）で縮約していません"
    );
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("float mean_hi = lane_sum / hidden_f;"),
        "平均パスが `hidden` による厳密除算で mean_hi を求めていません"
    );
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("float mean_div_r = fma(-mean_hi, hidden_f, lane_sum);"),
        "平均パスが FMA による doubled-float 拡張（mean_hi/mean_lo）を行っていません"
    );
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

/// `row_scale` の eps 対応拡張（PR #1671 スレッド 2 件目の是正）が
/// `ldexp` による最小限の右シフト（`LN_EPS_ELEM_SAFE_SHIFT`）を経由する
/// ことをロックする（`sqrt(eps)` の 2 の冪をそのまま採用する「正準」な
/// 実装への逆戻りを検出する。正準な実装は `x` の比が subnormal に潰れ
/// Apple GPU 実機で flush-to-zero される回帰を再導入する。冒頭コメント
/// 「`row_scale` の eps 対応拡張・weight 先乗算」参照）。
#[test]
fn eps_row_scale_extension_uses_minimal_ldexp_shift() {
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("constant int LN_EPS_ELEM_SAFE_SHIFT ="),
        "LN_EPS_ELEM_SAFE_SHIFT 定数が見つかりません"
    );
    assert!(
        LAYER_NORM_METAL_SOURCE.contains(
            "ldexp(ln_pow2_scale_from_maxabs(eps_pseudo_elem_scale), -LN_EPS_ELEM_SAFE_SHIFT)"
        ),
        "eps 側スケールが ldexp による最小限の右シフトを経由していません"
    );
}

/// パス 4 が `weight` の乗算順序（`scale` 除算の前か後か）を
/// `dev/scale` の subnormal リスクに応じて要素ごとに適応的に選ぶ
/// ことをロックする（PR #1671 スレッド 2 件目の是正。`eps` が `x` を
/// 極端に上回る行で中間値が subnormal に潰れる回帰・巨大 `weight` で
/// 無条件premultiplyがoverflowする回帰の両方を検出する。冒頭コメント
/// 「`row_scale` の eps 対応拡張・weight 乗算順序の適応的選択」参照）。
#[test]
fn pass4_selects_weight_multiply_order_adaptively() {
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("constant float LN_FLT_MIN_NORMAL ="),
        "LN_FLT_MIN_NORMAL 定数（subnormal 判定しきい値）が見つかりません"
    );
    assert!(
        LAYER_NORM_METAL_SOURCE.contains(
            "bool subnormal_risk =
                (scale > 0.0f) && (fabs(dev) < scale * LN_FLT_MIN_NORMAL);"
        ),
        "subnormal リスク判定（fabs(dev) < scale * LN_FLT_MIN_NORMAL）が見つかりません"
    );
    assert!(
        LAYER_NORM_METAL_SOURCE.contains(
            "float xhat_weighted = subnormal_risk
                ? (dev * wv) / scale
                : (dev / scale) * wv;"
        ),
        "weight 乗算順序の適応的選択（subnormal_risk ? (dev*wv)/scale : (dev/scale)*wv）が見つかりません"
    );
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("fma(xhat_weighted, norm, bv)"),
        "affine 最終段の fma 融合（fma(xhat_weighted, norm, bv)）が見つかりません"
    );
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("bool wv_finite = !isnan(wv) && !isinf(wv);"),
        "weight の有限性判定（wv_finite）が見つかりません"
    );
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("(wv_finite ? bv : fma(0.0f * wv, norm, bv))"),
        "ゼロ偏差短絡（weight 有限時は bv 直接返却・非有限時のみ fma 経由で NaN 伝播）が見つかりません"
    );
}
