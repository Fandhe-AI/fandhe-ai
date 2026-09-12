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
//!
//! **PR #1671 codex-review 指摘（P1 2 件）を受けた全面書き換え**:
//! 当初実装（行の 2 の冪スケール `row_scale` によるリスケール総和 +
//! Neumaier 補償和 + scale/ssq 分散）は「行スケール除算での微小値消失」
//! 「正規化係数の丸め誤差が affine の相殺で増幅される」という 2 系統の
//! 反例で数値契約を満たせないことが判明し、IEEE 754 binary64 の
//! ソフトウェアエミュレーション（`ln_f64_*` 系関数。`crates/backend-metal/
//! src/soft_f64.rs` がホスト側逐語モデル）経由の設計へ全面的に置き換えた
//! （`layer_norm.metal` 冒頭コメント「数値方式」参照）。旧設計固有の
//! 文字列（`row_scale`・`ln_kahan_add`・`ln_ssq_add`・
//! `LN_EPS_ELEM_SAFE_SHIFT`・`LN_FLT_MIN_NORMAL` 等）を検査していた
//! テストは新設計の実際の構造に合わせて全面的に書き換えた。

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

/// 平均（パス 1）・分散（パス 2）の 2 箇所が `simd_shuffle_xor` を用いた
/// 5 段 butterfly（`offset` を 16u→1u へ 5 回半減させるループ）で
/// reduction されることをロックする（soft-f64 化により `row_scale`
/// 算出用の `maxabs` パスが不要になったため、旧設計の 3 箇所〈maxabs・
/// 平均・分散〉から 2 箇所〈平均・分散〉へ削減済み。`layer_norm.metal`
/// 冒頭コメント「`rmsnorm.metal` との差分」参照）。
#[test]
fn mean_and_variance_reductions_use_five_stage_butterfly() {
    let occurrences = LAYER_NORM_METAL_SOURCE
        .matches("for (uint offset = 16u; offset > 0u; offset >>= 1u)")
        .count();
    assert_eq!(
        occurrences, 2,
        "5 段 butterfly ループ（16u→1u の 5 回半減）は平均・分散の 2 箇所に \
         存在するはずだが {occurrences} 箇所しか見つからなかった"
    );
}

/// 平均・分散のいずれの reduction も、soft-f64 アキュムレータ（`ulong`。
/// `lane_sum`／`lane_sq`）を 32bit 上位・下位へ分割して個別に
/// `simd_shuffle_xor` することをロックする（MSL の `simd_shuffle_xor` が
/// 64bit 値〈`ulong`〉を直接サポートするか不明瞭なため、既知に動作する
/// `uint` 版を 2 回呼ぶ方式を採る。`layer_norm.metal` 冒頭コメント参照。
/// 上位・下位いずれかの shuffle が失われると reduction 結果が破損する）。
#[test]
fn mean_and_variance_reductions_shuffle_both_halves_of_soft_f64_accumulator() {
    for var_name in ["lane_sum", "lane_sq"] {
        let hi_needle = format!("simd_shuffle_xor((uint)({var_name} >> 32), offset)");
        let lo_needle = format!("simd_shuffle_xor((uint){var_name}, offset)");
        assert!(
            LAYER_NORM_METAL_SOURCE.contains(&hi_needle),
            "`{var_name}` の上位 32bit（hi）を shuffle していません（soft-f64 \
             アキュムレータの reduction が破損している可能性）"
        );
        assert!(
            LAYER_NORM_METAL_SOURCE.contains(&lo_needle),
            "`{var_name}` の下位 32bit（lo）を shuffle していません（soft-f64 \
             アキュムレータの reduction が破損している可能性）"
        );
    }
}

/// 平均・分散パスがそれぞれ `ln_f64_widen`（`f32→f64`）で行要素を
/// 昇格したうえで `ln_f64_add`（soft-f64 加算）で蓄積し、`hidden` の
/// soft-f64 逆数（`ln_f64_recip_newton`）を乗じて確定することをロック
/// する（codex-review 指摘の回帰防止: `row_scale` によるリスケール総和
/// への逆戻りを検出する。`layer_norm.metal` 冒頭コメント「数値方式」
/// 参照）。
#[test]
fn mean_and_variance_passes_use_soft_f64_widen_add_and_div() {
    assert!(
        LAYER_NORM_METAL_SOURCE
            .contains("ulong xv = ln_f64_widen(as_type<uint>(x[row_base + idx]));"),
        "行要素を ln_f64_widen で f64 へ昇格していません"
    );
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("lane_sum = ln_f64_add(lane_sum, xv);"),
        "平均パスが ln_f64_add で soft-f64 総和を蓄積していません"
    );
    assert!(
        LAYER_NORM_METAL_SOURCE
            .contains("ulong hidden_f64 = ln_f64_widen(as_type<uint>((float)hidden));"),
        "hidden の soft-f64 表現（ln_f64_widen）が見つかりません"
    );
    // PR #1671 codex-review・Cursor Bugbot 指摘への是正（イシュー
    // #1596）: `mean`／`var` は Newton 近似逆数との積
    // （`ln_f64_mul(sum, ln_f64_recip_newton(hidden))`）ではなく、
    // 正しく丸めた除算（`ln_f64_div`）で確定する。一様行等の割り切れる
    // ケースで Newton 近似特有の 1 ULP 誤差が悪化するのを防ぐため。
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("ulong mean = ln_f64_div(lane_sum, hidden_f64);"),
        "平均が soft-f64 正しく丸めた除算（ln_f64_div(lane_sum, hidden_f64)）で確定していません"
    );
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("ulong var = ln_f64_div(lane_sq, hidden_f64);"),
        "分散が soft-f64 正しく丸めた除算（ln_f64_div(lane_sq, hidden_f64)）で確定していません"
    );
    assert!(
        !LAYER_NORM_METAL_SOURCE.contains("ulong hidden_recip = ln_f64_recip_newton"),
        "mean/var の計算経路に Newton 近似逆数との積（hidden_recip）が\
         残存しています（1 ULP 誤差が悪化するケースがあるため使わない）"
    );
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("ulong rstd = ln_f64_rsqrt_newton(var_plus_eps);"),
        "rstd が soft-f64 逆数平方根（ln_f64_rsqrt_newton）で確定していません"
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

/// soft-f64 の特殊値伝播（NaN・0*inf）が `ln_f64_add`／`ln_f64_mul` に
/// 明示的に存在することをロックする（`rmsnorm.metal`／旧設計の
/// `isnan`／`isinf` 契約と同じ意図を soft-f64 版で引き継ぐ。codex-review
/// 指摘・PR #1120 の教訓の延長）。
#[test]
fn soft_f64_add_and_mul_explicitly_handle_nan_and_zero_times_inf() {
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("bool a_nan = (ea == LN_F64_EXP_MASK) && (fa != 0ul);"),
        "ln_f64_add に NaN 検出が見つかりません"
    );
    assert!(
        LAYER_NORM_METAL_SOURCE
            .contains("if ((a_zero && b_inf) || (a_inf && b_zero)) {\n        return LN_F64_QNAN;"),
        "ln_f64_mul に 0*inf（不定形）の明示分岐が見つかりません"
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

/// `eps` が `row_scale` 由来の擬似要素トリック（旧設計）を経由せず、
/// `ln_f64_widen` で直接 soft-f64 へ昇格されることをロックする
/// （codex-review 指摘の回帰防止: `eps` 側スケールを `f32` の表現範囲に
/// 押し込める旧トリックへの逆戻りを検出する）。
#[test]
fn eps_is_widened_directly_to_soft_f64() {
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("ulong eps_f64 = ln_f64_widen(as_type<uint>(eps));"),
        "eps が ln_f64_widen で直接 soft-f64 へ昇格されていません"
    );
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("ulong var_plus_eps = ln_f64_add(var, eps_f64);"),
        "var + eps が soft-f64 加算で確定していません"
    );
}

/// パス 3（書き出し）の affine（`x̂·w+b`）が、GPU の subnormal
/// flush-to-zero 対策として `xhat`／`weight`／`bias` すべてを soft-f64 へ
/// widen し直し `mul`＋`add` で計算してから 1 回だけ `f32` へ narrow する
/// ことをロックする（PR #1671 codex-review 反例〈`xhat` 自体が `f32`
/// subnormal になる行で平坦な `float` の `fma()` が GPU 実機の入力側
/// flush-to-zero に晒される〉の回帰防止。`layer_norm.metal` パス 3
/// コメント「affine も soft-f64 で計算する理由」参照）。
#[test]
fn pass3_computes_affine_entirely_in_soft_f64_to_avoid_gpu_subnormal_flush() {
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("uint xhat_bits = ln_f64_narrow(xhat64);"),
        "xhat が soft-f64 から f32 へ narrow されていません"
    );
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("ulong wv64 = ln_f64_widen(as_type<uint>(wv));"),
        "weight を soft-f64 へ widen していません"
    );
    assert!(
        LAYER_NORM_METAL_SOURCE.contains("ulong bv64 = ln_f64_widen(as_type<uint>(bv));"),
        "bias を soft-f64 へ widen していません"
    );
    assert!(
        LAYER_NORM_METAL_SOURCE.contains(
            "ulong affine64 = ln_f64_add(ln_f64_mul(ln_f64_widen(xhat_bits), wv64), bv64);"
        ),
        "affine が soft-f64 の mul+add で計算されていません（平坦な float fma への \
         逆戻りは GPU 実機の subnormal flush-to-zero を再導入する）"
    );
    assert!(
        !LAYER_NORM_METAL_SOURCE.contains("out[row_base + idx] = fma(xhat, wv, bv);"),
        "affine が平坦な float の fma() で計算されています（GPU 実機の subnormal \
         flush-to-zero に対し脆弱な旧経路への回帰）"
    );
}

/// soft-f64 の主要プリミティブ（`widen`／`add`／`mul`／`narrow`／
/// `recip_newton`／`rsqrt_newton`）がすべて定義されていることをロック
/// する（`crates/backend-metal/src/soft_f64.rs` のホスト側逐語モデルと
/// 1 対 1 対応する契約。いずれかが欠落すると `layer_norm.metal` 冒頭
/// コメント「ホスト側の逐語モデル」の前提が崩れる）。
#[test]
fn all_soft_f64_primitives_are_defined() {
    for needle in [
        "inline ulong ln_f64_widen(uint bits)",
        "inline ulong ln_f64_add(ulong a, ulong b)",
        "inline ulong ln_f64_sub(ulong a, ulong b)",
        "inline uint ln_f64_narrow(ulong bits)",
        "inline ulong ln_f64_mul(ulong a, ulong b)",
        "inline ulong ln_f64_div(ulong a, ulong b)",
        "inline ulong ln_f64_recip_newton(ulong x)",
        "inline ulong ln_f64_rsqrt_newton(ulong x)",
    ] {
        assert!(
            LAYER_NORM_METAL_SOURCE.contains(needle),
            "soft-f64 プリミティブの定義が見つかりません: {needle}"
        );
    }
}
