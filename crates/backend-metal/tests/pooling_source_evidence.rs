//! イシュー #1730: MaxPool／AvgPool／AdaptiveAvgPool（MSL）の文字列
//! 証跡テスト。`batch_norm_source_evidence.rs`（#1736）と同方針:
//! `include_str!` によるビルド時文字列埋め込みへの contains 検査のみで
//! 完結するため、Metal 実機・`cfg(target_os = "macos")` を必要とせず
//! Linux CI（GitHub ホステッド）上でも green になる。
//!
//! `.claude/rules/coding-rust.md`「REQ-8」の機械検証（手動境界検査・
//! grid-stride 不使用）と、`pool_f64_*`（本ファイル対象）と
//! `bn_f64_*`（`batch_norm.metal`。#1736）の soft-f64 プリミティブが
//! 接頭辞以外逐語一致することのドリフトガードを兼ねる。

/// `crates/backend-metal/src/shaders/pooling.metal` のソース全文。
const POOLING_METAL_SOURCE: &str = include_str!("../src/shaders/pooling.metal");

/// `crates/backend-metal/src/shaders/batch_norm.metal` のソース全文
/// （ドリフトガード用の比較対象）。
const BATCH_NORM_METAL_SOURCE: &str = include_str!("../src/shaders/batch_norm.metal");

/// 3 カーネルすべてが `if (gid >= dims.numel_out) { return; }` による
/// 手動境界検査を持ち、grid-stride ループ（`idx += stride` 等）を
/// 使わない単純 1 スレッド = 1 出力位置カーネルであることを固定する
/// （REQ-8。性能を理由に境界検査を省略しない）。
#[test]
fn all_kernels_use_manual_bounds_check_without_grid_stride() {
    let occurrences = POOLING_METAL_SOURCE
        .matches("if (gid >= dims.numel_out) {\n        return;\n    }")
        .count();
    assert_eq!(
        occurrences, 3,
        "3 カーネル（max_pool2d_f32／avg_pool2d_f32／adaptive_avg_pool2d_f32）すべてに \
         手動境界検査（if (gid >= dims.numel_out) return;）が必要です"
    );
    assert!(
        !POOLING_METAL_SOURCE.contains("+= stride"),
        "grid-stride ループは使わない設計のはず（1 スレッド = 1 出力位置）"
    );
}

/// 座標計算（`gid` からの `(n,c,oh,ow)` 分解・窓オフセット `ih`／
/// `iw`）が `long` で行われ、窓外（padding）位置を負値として自然に
/// 表現できることを固定する（`im2col.metal` と同型の REQ-8 対策。
/// `uint` のまま引き算すると wrap-around で誤って範囲内と判定され
/// うる）。
#[test]
fn kernels_use_signed_long_arithmetic_for_window_offsets() {
    assert!(
        POOLING_METAL_SOURCE.contains("long ih = oh * (long)dims.sh"),
        "max_pool2d_f32／avg_pool2d_f32 の ih 計算が long 演算でないようです"
    );
    assert!(
        POOLING_METAL_SOURCE.contains("long iw = ow * (long)dims.sw"),
        "max_pool2d_f32／avg_pool2d_f32 の iw 計算が long 演算でないようです"
    );
}

/// MaxPool の更新条件（「最初の有効タップで初期化」「以後は
/// `v > best || (isnan(v) && !isnan(best))` のときのみ更新」——タイは
/// 先勝ち・NaN は最初に出現した索引で確定する契約）を固定する。
#[test]
fn max_pool_update_rule_is_first_wins_tie_and_first_nan_sticky() {
    assert!(
        POOLING_METAL_SOURCE.contains("} else if (v > best || (isnan(v) && !isnan(best))) {"),
        "MaxPool の更新条件（先勝ちタイ・最初の NaN 固定）が見つかりません"
    );
    assert!(
        POOLING_METAL_SOURCE.contains("if (first) {"),
        "MaxPool の「最初の有効タップで初期化」ロジックが見つかりません"
    );
}

/// Avg 系カーネル（`avg_pool2d_f32`／`adaptive_avg_pool2d_f32`）が
/// soft-f64 プリミティブ（`pool_f64_widen`／`pool_f64_add`／
/// `pool_f64_div`／`pool_f64_narrow`）のみで窓内総和・除算を行い、
/// `threadgroup_barrier` を使わないことを固定する（単純逐次スキャン
/// でチャネル・スレッド間の同期が不要という設計契約）。
#[test]
fn avg_kernels_use_soft_f64_accumulation_and_no_barrier() {
    for needle in [
        "acc = pool_f64_add(acc, pool_f64_widen(as_type<uint>(v)));",
        "pool_f64_div(acc, pool_f64_from_uint(divisor))",
        "pool_f64_narrow(",
    ] {
        assert!(
            POOLING_METAL_SOURCE.contains(needle),
            "Avg 系カーネルの soft-f64 契約文字列 `{needle}` が見つかりません"
        );
    }
    assert!(
        !POOLING_METAL_SOURCE.contains("threadgroup_barrier"),
        "Avg 系カーネルは threadgroup_barrier を使わない設計のはず"
    );
}

/// `PoolDims` の MSL 宣言（18 フィールド・宣言順）を固定する
/// （`crate::pooling_model::PoolDims` の `#[repr(C)]` レイアウトと
/// 一致させる契約。`pooling_model.rs::
/// pool_dims_size_matches_msl_struct` が `size_of` を Linux 検証）。
#[test]
fn pool_dims_msl_struct_declares_fields_in_order() {
    let expected_order = [
        "uint n;",
        "uint c;",
        "uint h_in;",
        "uint w_in;",
        "uint h_out;",
        "uint w_out;",
        "uint kh;",
        "uint kw;",
        "uint sh;",
        "uint sw;",
        "uint ph;",
        "uint pw;",
        "uint dh;",
        "uint dw;",
        "uint count_include_pad;",
        "uint numel_out;",
        "uint plane_in;",
        "uint plane_out;",
    ];
    let struct_start = POOLING_METAL_SOURCE
        .find("struct PoolDims {")
        .expect("struct PoolDims の宣言が見つかりません");
    let struct_end = POOLING_METAL_SOURCE[struct_start..]
        .find("};")
        .map(|i| struct_start + i)
        .expect("struct PoolDims の終端 `};` が見つかりません");
    let body = &POOLING_METAL_SOURCE[struct_start..struct_end];

    let mut last_pos = 0usize;
    for field in expected_order {
        let pos = body[last_pos..]
            .find(field)
            .unwrap_or_else(|| panic!("フィールド `{field}` が期待順で見つかりません"));
        last_pos += pos + field.len();
    }
}

/// [`extract_fn_body`]: `name(` を含む行から波括弧の対応を辿り
/// 関数本体（`{`〜対応する `}`）を抽出する
/// （`batch_norm_source_evidence.rs::extract_fn_body` と同一実装）。
fn extract_fn_body<'a>(source: &'a str, name: &str) -> Option<&'a str> {
    let needle = format!(" {name}(");
    let sig_start = source.find(&needle)?;
    let brace_start = source[sig_start..].find('{').map(|i| sig_start + i)?;
    let mut depth: i32 = 0;
    for (offset, ch) in source[brace_start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let brace_end = brace_start + offset;
                    return Some(&source[brace_start..=brace_end]);
                }
            }
            _ => {}
        }
    }
    None
}

/// [`pool_f64_*`]（本ファイル）の各関数本体が [`bn_f64_*`]
/// （`batch_norm.metal`）の対応する関数本体と接頭辞以外で逐語一致する
/// ことを固定する（本ファイル冒頭コメント「soft-f64 プリミティブ」の
/// 複製契約。`batch_norm.metal` を変更した場合に本ファイルへの
/// 追従漏れを検出するドリフトガード。`batch_norm_source_evidence.rs::
/// bn_f64_primitives_match_ln_f64_primitives_verbatim_modulo_prefix`
/// と同型）。
#[test]
fn pool_f64_primitives_match_bn_f64_primitives_verbatim_modulo_prefix() {
    let renamed_batch_norm_source = BATCH_NORM_METAL_SOURCE
        .replace("bn_f64_", "pool_f64_")
        .replace("BN_F64_", "POOL_F64_")
        .replace("BN_F32_", "POOL_F32_")
        .replace("BnU128", "PoolU128")
        .replace("BnNormMantissa", "PoolNormMantissa");

    let fn_names = [
        "pool_f64_clz64",
        "pool_f64_widen",
        "pool_f64_add",
        "pool_f64_narrow",
        "pool_f64_mul64_wide",
        "pool_f64_normalize_mantissa",
        "pool_f64_div64_wide",
        "pool_f64_div",
    ];

    for name in fn_names {
        let pool_body = extract_fn_body(POOLING_METAL_SOURCE, name)
            .unwrap_or_else(|| panic!("pooling.metal に {name} の定義（本体）が見つかりません"));
        let bn_body = extract_fn_body(&renamed_batch_norm_source, name).unwrap_or_else(|| {
            panic!(
                "batch_norm.metal（接頭辞置換後）に {name} の定義（本体）が見つかりません \
                 （ドリフトガード自体の不整合。batch_norm.metal 側の関数名変更を確認）"
            )
        });
        assert_eq!(
            pool_body, bn_body,
            "{name} の関数本体が batch_norm.metal（接頭辞置換後）の対応する関数本体と \
             逐語一致しません（soft-f64 プリミティブの複製契約からのドリフト）"
        );
    }
}

/// `pool_f64_from_uint`（`pooling.metal` 固有・`batch_norm.metal` には
/// 存在しない u32→binary64 厳密変換ヘルパー）が定義されており、
/// `divisor`（Avg 系カーネルの除数）を soft-f64 除算へ渡す前に必ず
/// 経由することを固定する。
#[test]
fn pool_f64_from_uint_is_defined_and_used_for_divisor_conversion() {
    assert!(
        POOLING_METAL_SOURCE.contains("inline ulong pool_f64_from_uint(uint v) {"),
        "pool_f64_from_uint の定義が見つかりません"
    );
    let occurrences = POOLING_METAL_SOURCE
        .matches("pool_f64_from_uint(divisor)")
        .count();
    assert_eq!(
        occurrences, 2,
        "avg_pool2d_f32／adaptive_avg_pool2d_f32 の両方が divisor を \
         pool_f64_from_uint 経由で soft-f64 除算に渡す必要があります"
    );
}
