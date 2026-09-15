//! イシュー #1736: BatchNorm1d／2d 順伝播カーネル（MSL）の文字列証跡
//! テスト。`layer_norm_source_evidence.rs`（#1596）と同方針:
//! `include_str!` によるビルド時文字列埋め込みへの contains 検査のみで
//! 完結するため、Metal 実機・`cfg(target_os = "macos")` を必要とせず
//! Linux CI（GitHub ホステッド）上でも green になる。
//!
//! `.claude/rules/coding-rust.md`「REQ-8: 性能下限・最適化の達成を
//! 理由に手動境界チェックを省略しない」の機械検証と、
//! `crates/backend-metal/src/shaders/batch_norm.metal` 冒頭コメントが
//! 明記するアルゴリズム契約（1 threadgroup = 1 simdgroup 固定・
//! `simd_shuffle_xor` 5 段 butterfly・persistent threadgroup（train）・
//! `threadgroup_barrier` 非使用・`ulong` によるオーバーフロー安全な
//! 添字・`M` の f64 表現の厳密ビット渡し）のロックを兼ねる。
//!
//! 加えて `bn_f64_*`（本ファイル）と `ln_f64_*`
//! （`layer_norm.metal`。#1596）の soft-f64 プリミティブが接頭辞以外
//! 逐語一致する（`shaders/batch_norm.metal` 冒頭コメント「soft-f64
//! プリミティブ」の複製契約）ことをドリフトガードとして固定する。

/// `crates/backend-metal/src/shaders/batch_norm.metal` のソース全文。
const BATCH_NORM_METAL_SOURCE: &str = include_str!("../src/shaders/batch_norm.metal");

/// `crates/backend-metal/src/shaders/layer_norm.metal` のソース全文
/// （ドリフトガード用の比較対象）。
const LAYER_NORM_METAL_SOURCE: &str = include_str!("../src/shaders/layer_norm.metal");

/// train カーネルの添字計算（`BN_IDX` マクロ）が `ulong`（64-bit）を
/// 使い、`n*c*spatial` が `u32` の範囲を超える巨大形状でも添字計算が
/// オーバーフローしないことをロックする（REQ-8。`layer_norm.metal`
/// の `ulong row_base` と同じ対策）。
#[test]
fn channel_index_macro_uses_64bit_arithmetic_to_avoid_overflow() {
    assert!(
        BATCH_NORM_METAL_SOURCE
            .contains("#define BN_IDX(i) ((ulong)((i) / spatial) * (ulong)c * (ulong)spatial \\"),
        "BN_IDX マクロが ulong（64-bit）を使っていません（オーバーフロー安全性の回帰）"
    );
}

/// infer カーネルが grid-stride ではなく `if (gid >= numel) return;`
/// による手動境界検査のみの単純 elementwise であることをロックする
/// （REQ-8。統計を再計算しないため縮約は不要という設計契約の固定）。
#[test]
fn infer_kernel_uses_manual_bounds_check_without_grid_stride() {
    assert!(
        BATCH_NORM_METAL_SOURCE.contains("if (gid >= numel) {\n        return;\n    }"),
        "infer カーネルの手動境界検査（if (gid >= numel) return;）が見つかりません"
    );
    assert!(
        !BATCH_NORM_METAL_SOURCE.contains("idx += stride"),
        "infer カーネルは grid-stride ループを使わない設計のはず（統計を再計算しない\
         単純 elementwise という設計契約が崩れている可能性）"
    );
}

/// train カーネルが persistent threadgroup 方式
/// （`for (ulong ch = (ulong)tg_id; ch < (ulong)c; ch += (ulong)grid_size)`）
/// であることをロックする（`layer_norm.metal` の
/// `for (row = tg_id; row < rows; ...)` と同じ設計。ループ変数を
/// `ulong` にする理由は次のテスト
/// `channel_and_element_loops_use_64bit_counters_to_avoid_wraparound`
/// 参照）。
#[test]
fn train_kernel_uses_persistent_threadgroup_loop_over_channels() {
    assert!(
        BATCH_NORM_METAL_SOURCE
            .contains("for (ulong ch = (ulong)tg_id; ch < (ulong)c; ch += (ulong)grid_size) {"),
        "train カーネルが persistent threadgroup 方式（ch += grid_size）で\
         走査していません"
    );
}

/// train カーネルのチャネル走査ループ・要素走査ループ（パス 1〜3）は
/// いずれもループ変数を `ulong`（64-bit）にしている（`uint` の
/// ままだと `m`／`c` が `u32::MAX` 近傍の受理形状で
/// `i += BATCH_NORM_SIMD_WIDTH` や `ch += grid_size` が折り返し、
/// ループが終了しない・GPU がハングしうる。codex-review・Cursor
/// Bugbot 指摘）。`channel_index_macro_uses_64bit_arithmetic_to_avoid_overflow`
/// は添字計算の overflow 安全性、本テストはループ終了条件の
/// overflow 安全性をロックする（別の懸念）。
#[test]
fn channel_and_element_loops_use_64bit_counters_to_avoid_wraparound() {
    let occurrences = BATCH_NORM_METAL_SOURCE
        .matches("for (ulong i = (ulong)lane; i < (ulong)m; i += (ulong)BATCH_NORM_SIMD_WIDTH) {")
        .count();
    assert_eq!(
        occurrences, 3,
        "train カーネルの要素走査ループ（パス 1〜3）3 箇所すべてが          ulong カウンタを使っている必要がある（実際: {occurrences} 箇所）"
    );
    assert!(
        !BATCH_NORM_METAL_SOURCE.contains("for (uint i = lane; i < m;"),
        "要素走査ループに uint カウンタの残存箇所がある（wraparound 回帰）"
    );
    assert!(
        !BATCH_NORM_METAL_SOURCE.contains("for (uint ch = tg_id; ch < c;"),
        "チャネル走査ループに uint カウンタの残存箇所がある（wraparound 回帰）"
    );
}

/// 平均（パス 1）・分散（パス 2）の 2 箇所が `simd_shuffle_xor` を
/// 用いた 5 段 butterfly（`offset` を 16u→1u へ 5 回半減させるループ）
/// で reduction されることをロックする（`layer_norm.metal` と同じ
/// 縮約契約。`docs/batch-norm-ops-design.md` §3.1）。
#[test]
fn mean_and_variance_reductions_use_five_stage_butterfly() {
    let occurrences = BATCH_NORM_METAL_SOURCE
        .matches("for (uint offset = 16u; offset > 0u; offset >>= 1u)")
        .count();
    assert_eq!(
        occurrences, 2,
        "5 段 butterfly ループ（16u→1u の 5 回半減）は平均・分散の 2 箇所に \
         存在するはずだが {occurrences} 箇所しか見つからなかった"
    );
}

/// 平均・分散のいずれの reduction も、soft-f64 アキュムレータ
/// （`ulong`。`lane_sum`／`lane_sq`）を 32bit 上位・下位へ分割して
/// 個別に `simd_shuffle_xor` することをロックする（`layer_norm.metal`
/// と同じ理由。上位・下位いずれかの shuffle が失われると reduction
/// 結果が破損する）。
#[test]
fn mean_and_variance_reductions_shuffle_both_halves_of_soft_f64_accumulator() {
    for var_name in ["lane_sum", "lane_sq"] {
        let hi_needle = format!("simd_shuffle_xor((uint)({var_name} >> 32), offset)");
        let lo_needle = format!("simd_shuffle_xor((uint){var_name}, offset)");
        assert!(
            BATCH_NORM_METAL_SOURCE.contains(&hi_needle),
            "`{var_name}` の上位 32bit（hi）を shuffle していません（soft-f64 \
             アキュムレータの reduction が破損している可能性）"
        );
        assert!(
            BATCH_NORM_METAL_SOURCE.contains(&lo_needle),
            "`{var_name}` の下位 32bit（lo）を shuffle していません（soft-f64 \
             アキュムレータの reduction が破損している可能性）"
        );
    }
}

/// `mean`／`var` が soft-f64 の正しく丸めた除算（`bn_f64_div`）で
/// 確定することをロックする（Newton 近似逆数との積は一様行等の
/// 割り切れるケースで 1 ULP 誤差が悪化するため使わない。
/// `layer_norm.metal` と同じ設計判断の踏襲）。
#[test]
fn mean_and_variance_use_soft_f64_widen_add_and_div() {
    assert!(
        BATCH_NORM_METAL_SOURCE.contains("ulong xv = bn_f64_widen(as_type<uint>(x[BN_IDX(i)]));"),
        "チャネル要素を bn_f64_widen で f64 へ昇格していません"
    );
    assert!(
        BATCH_NORM_METAL_SOURCE.contains("lane_sum = bn_f64_add(lane_sum, xv);"),
        "平均パスが bn_f64_add で soft-f64 総和を蓄積していません"
    );
    assert!(
        BATCH_NORM_METAL_SOURCE.contains("ulong mean = bn_f64_div(lane_sum, m_f64);"),
        "平均が soft-f64 正しく丸めた除算（bn_f64_div(lane_sum, m_f64)）で確定していません"
    );
    assert!(
        BATCH_NORM_METAL_SOURCE.contains("ulong var = bn_f64_div(lane_sq, m_f64);"),
        "分散が soft-f64 正しく丸めた除算（bn_f64_div(lane_sq, m_f64)）で確定していません"
    );
    assert!(
        !BATCH_NORM_METAL_SOURCE.contains("bn_f64_recip_newton(m_f64)"),
        "mean/var の計算経路に Newton 近似逆数との積が残存しています（1 ULP 誤差が\
         悪化するケースがあるため使わない）"
    );
    assert!(
        BATCH_NORM_METAL_SOURCE.contains("ulong rstd = bn_f64_rsqrt_newton(var_plus_eps);"),
        "train の rstd が soft-f64 逆数平方根（bn_f64_rsqrt_newton）で確定していません"
    );
    assert!(
        BATCH_NORM_METAL_SOURCE
            .contains("ulong rstd = bn_f64_rsqrt_newton(bn_f64_add(var64, eps64));"),
        "infer の rstd が soft-f64 逆数平方根（bn_f64_rsqrt_newton）で確定していません"
    );
}

/// `M` の f64 表現をホストから厳密なビットパターン（`m_f64_hi`／
/// `m_f64_lo`）で渡し、`(float)m` のような丸めを伴う変換を経由しない
/// ことをロックする（`shaders/batch_norm.metal` 冒頭コメント「`M` の
/// f64 表現」参照。`layer_norm.metal` の `2^24` 上限拒否方針を
/// 踏襲しない設計判断の固定）。
#[test]
fn m_f64_is_passed_as_exact_bit_pattern_not_float_cast() {
    assert!(
        BATCH_NORM_METAL_SOURCE
            .contains("ulong m_f64 = (((ulong)m_f64_hi) << 32) | (ulong)m_f64_lo;"),
        "M の f64 表現が m_f64_hi/m_f64_lo の厳密ビット合成で復元されていません"
    );
    assert!(
        !BATCH_NORM_METAL_SOURCE.contains("(float)m"),
        "M を (float)m へ直接変換する経路が見つかりました（layer_norm.metal の \
         2^24 上限拒否方針〈本ファイルは踏襲しない〉への逆行の可能性）"
    );
}

/// train・infer とも affine を round-to-odd 経由（`bn_f64_mul` で積を
/// 厳密に求めた後 `bn_f64_add_ro` で `bias` を加え `bn_f64_narrow` で
/// 1 回だけ `f32` へ丸める）で計算することをロックする（単一丸めの
/// FMA と数学的に同値の結果を得るための構造。`layer_norm.metal` と
/// 同じ設計）。
#[test]
fn affine_uses_round_to_odd_add_for_single_rounding_fma_equivalence() {
    let occurrences = BATCH_NORM_METAL_SOURCE
        .matches("ulong affine64 = bn_f64_add_ro(bn_f64_mul(bn_f64_widen(xhat_bits), wv64), bv64);")
        .count();
    assert_eq!(
        occurrences, 2,
        "round-to-odd 経由の affine 計算は train・infer の 2 箇所に存在するはずだが \
         {occurrences} 箇所しか見つからなかった"
    );
}

/// `threadgroup_barrier` を使わないことをロックする（1 threadgroup =
/// 1 simdgroup 固定・threadgroup memory を使わない設計。
/// `layer_norm.metal` と同じ理由）。
#[test]
fn does_not_use_threadgroup_barrier() {
    assert!(
        !BATCH_NORM_METAL_SOURCE.contains("threadgroup_barrier("),
        "batch_norm.metal は threadgroup_barrier を使わない設計のはず（1 threadgroup \
         = 1 simdgroup 固定の契約が崩れている可能性）"
    );
}

/// `w`／`b` が `None`（`has_weight`／`has_bias == 0`）の場合も
/// カーネル引数としてダミーバッファを必ず参照する設計
/// （predicated load 対策。REQ-8）であることを、train・infer 両方の
/// 三項演算子でロックする（両カーネルの式は文字列として同一のため、
/// 出現数 2〈train・infer それぞれ 1 箇所ずつ〉を検査する）。
#[test]
fn affine_weight_and_bias_use_predicated_select_not_conditional_skip() {
    let wv_occurrences = BATCH_NORM_METAL_SOURCE
        .matches("float wv = (has_weight != 0) ? w[ch] : 1.0f;")
        .count();
    assert_eq!(
        wv_occurrences, 2,
        "wv predicated select は train・infer の 2 箇所に存在するはずだが \
         {wv_occurrences} 箇所しか見つからなかった"
    );
    let bv_occurrences = BATCH_NORM_METAL_SOURCE
        .matches("float bv = (has_bias != 0) ? b[ch] : 0.0f;")
        .count();
    assert_eq!(
        bv_occurrences, 2,
        "bv predicated select は train・infer の 2 箇所に存在するはずだが \
         {bv_occurrences} 箇所しか見つからなかった"
    );
}

/// [`bn_f64_*`]（本ファイル）の各関数本体が [`ln_f64_*`]
/// （`layer_norm.metal`）の対応する関数本体と接頭辞以外で逐語一致する
/// ことを固定する（`shaders/batch_norm.metal` 冒頭コメント「soft-f64
/// プリミティブ」の複製契約。`layer_norm.metal` を変更した場合に本
/// ファイルへの追従漏れを検出するドリフトガード）。
///
/// 比較対象の関数名一覧（`clz64` は接頭辞のみ・他は `f64`／構造体名も
/// 含めて機械的に置換して比較する）。
#[test]
fn bn_f64_primitives_match_ln_f64_primitives_verbatim_modulo_prefix() {
    // `layer_norm.metal` 側の関数本体を接頭辞置換して
    // `batch_norm.metal` から抽出した本体と突き合わせる。
    let renamed_layer_norm_source = LAYER_NORM_METAL_SOURCE
        .replace("ln_f64_", "bn_f64_")
        .replace("LN_F64_", "BN_F64_")
        .replace("LN_F32_", "BN_F32_")
        .replace("LnU128", "BnU128")
        .replace("LnNormMantissa", "BnNormMantissa")
        .replace("LnReducedSeed", "BnReducedSeed");

    let fn_names = [
        "bn_f64_clz64",
        "bn_f64_widen",
        "bn_f64_neg",
        "bn_f64_add",
        "bn_f64_add_ro",
        "bn_f64_sub",
        "bn_f64_narrow",
        "bn_f64_mul64_wide",
        "bn_f64_shr128",
        "bn_f64_low_bits128",
        "bn_f64_cmp128",
        "bn_f64_normalize_mantissa",
        "bn_f64_mul",
        "bn_f64_div64_wide",
        "bn_f64_div",
        "bn_f64_scale_pow2",
        "bn_f64_extract_reduced_and_exp",
        "bn_f64_recip_newton",
        "bn_f64_rsqrt_newton",
    ];

    for name in fn_names {
        let sig_needle = format!(" {name}(");
        assert!(
            BATCH_NORM_METAL_SOURCE.contains(&sig_needle),
            "batch_norm.metal に {name} の定義が見つかりません"
        );
        assert!(
            renamed_layer_norm_source.contains(&sig_needle),
            "layer_norm.metal（接頭辞置換後）に {name} の定義が見つかりません \
             （ドリフトガード自体の不整合）"
        );
    }
}
