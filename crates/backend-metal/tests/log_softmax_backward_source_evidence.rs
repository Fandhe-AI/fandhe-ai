//! イシュー #1952: `log_softmax_bwd_lane_sum`／`log_softmax_bwd_apply_f32`
//! カーネル（MSL）の文字列証跡テスト。`reduce_source_evidence.rs` と
//! 同方針: `include_str!` によるビルド時文字列埋め込みへの contains
//! 検査のみで完結するため、Metal 実機・`cfg(target_os = "macos")` を
//! 必要とせず Linux CI（GitHub ホステッド）上でも green になる。
//!
//! `.claude/rules/coding-rust.md`「REQ-8: 手動境界チェックを省略しない」
//! の機械検証（`gid` 境界検査）・`precise::exp` の使用ロック（`exp` を
//! precise にするための必須事項。モジュール doc 参照）・`lane_sum` を
//! narrow していないことの検証・`crate::soft_f64`／`layer_norm.metal::
//! ln_f64_*` と bit 完全一致する契約の根拠となる binary64 ソフトウェア
//! エミュレーションプリミティブ（`lsb_f64_*`）の使用のロック・
//! `layer_norm.metal::ln_f64_*` との接頭辞置換後の関数本体逐語一致
//! （ドリフトガード。`reduce_source_evidence.rs::
//! red_f64_primitives_match_scan_f64_primitives_verbatim_modulo_prefix`
//! と同方針）を兼ねる。

/// `crates/backend-metal/src/shaders/log_softmax_backward.metal` の
/// ソース全文。
const LSB_METAL_SOURCE: &str = include_str!("../src/shaders/log_softmax_backward.metal");

/// `crates/backend-metal/src/shaders/layer_norm.metal` のソース全文
/// （接頭辞置換後の逐語一致比較対象。`ln_f64_mul` は本イシューより前に
/// 確立済みの soft-f64 精密乗算プリミティブ定義元）。
const LAYER_NORM_METAL_SOURCE: &str = include_str!("../src/shaders/layer_norm.metal");

/// `crate::log_softmax_backward::MetalLogSoftmaxBackward::new` はソース
/// 全文をそのまま `newLibraryWithSource_options_error` へ渡して実行時
/// コンパイルするため、`#include <metal_stdlib>`／`using namespace
/// metal;` の宣言が欠けていると `LibraryCompilation` エラーになる。
#[test]
fn source_includes_metal_stdlib_and_namespace() {
    assert!(
        LSB_METAL_SOURCE.contains("#include <metal_stdlib>"),
        "log_softmax_backward.metal に `#include <metal_stdlib>` が見つかりません"
    );
    assert!(
        LSB_METAL_SOURCE.contains("using namespace metal;"),
        "log_softmax_backward.metal に `using namespace metal;` が見つかりません"
    );
    let include_pos = LSB_METAL_SOURCE.find("#include <metal_stdlib>").unwrap();
    let using_pos = LSB_METAL_SOURCE.find("using namespace metal;").unwrap();
    assert!(
        include_pos < using_pos,
        "`#include <metal_stdlib>` は `using namespace metal;` より前に置く"
    );
}

/// `crate::log_softmax_backward::MetalLogSoftmaxBackward::new` が
/// `pipeline::make_pipeline` へ渡すカーネル名（2 種）と、各カーネルが
/// 期待するバッファ index の宣言を機械検証する（`crate::
/// log_softmax_backward::encode_*` の index 割当と一致することの
/// ロック）。
#[test]
fn kernel_names_and_buffer_indices_are_declared() {
    for kernel in ["log_softmax_bwd_lane_sum", "log_softmax_bwd_apply_f32"] {
        assert!(
            LSB_METAL_SOURCE.contains(&format!("kernel void {kernel}(")),
            "{kernel} カーネルの宣言が見つかりません"
        );
    }
    // 2 カーネル合算で buffer index 0〜6 が使われる（lane_sum: 0〜4・
    // apply: 0〜6）。
    for idx in 0..=6 {
        assert!(
            LSB_METAL_SOURCE.contains(&format!("[[buffer({idx})]]")),
            "buffer({idx}) の宣言が見つかりません"
        );
    }
}

/// REQ-8（境界検査規約）: `log_softmax_bwd_lane_sum` は `gid >= lanes`、
/// `log_softmax_bwd_apply_f32` は `gid >= numel` の境界検査を持つことを
/// 機械検証する（最適化を理由に省略しない契約のロック）。
#[test]
fn boundary_checks_are_present() {
    assert!(
        LSB_METAL_SOURCE.contains("if (gid >= lanes)"),
        "log_softmax_bwd_lane_sum に `if (gid >= lanes)` 境界検査が必要"
    );
    assert!(
        LSB_METAL_SOURCE.contains("if (gid >= numel)"),
        "log_softmax_bwd_apply_f32 に `if (gid >= numel)` 境界検査が必要"
    );
}

/// `log_softmax_bwd_apply_f32` が `precise::exp` を使うことを機械検証
/// する（`mathMode=Safe` だけでは超越関数は precise にならないため、
/// `bce.metal` と同様に明示が必須。モジュール doc「数値方式」参照）。
#[test]
fn apply_kernel_uses_precise_exp() {
    let apply_start = LSB_METAL_SOURCE
        .find("kernel void log_softmax_bwd_apply_f32(")
        .expect("log_softmax_bwd_apply_f32 declaration must exist");
    let apply_body = &LSB_METAL_SOURCE[apply_start..];
    assert!(
        apply_body.contains("precise::exp("),
        "log_softmax_bwd_apply_f32 は `precise::exp(` を使うはず"
    );
}

/// `log_softmax_bwd_lane_sum` が `lane_sum` を **narrow せず** `f64`
/// bit（`ulong`）のまま書くことを機械検証する（`reduce.metal::
/// reduce_sum_all_chunk_f32` の `partial` と同じ設計判断。narrow して
/// から `log_softmax_bwd_apply_f32` へ渡すと二重丸めで契約が崩れる）。
#[test]
fn lane_sum_kernel_does_not_narrow() {
    let lane_sum_start = LSB_METAL_SOURCE
        .find("kernel void log_softmax_bwd_lane_sum(")
        .expect("log_softmax_bwd_lane_sum declaration must exist");
    let apply_start = LSB_METAL_SOURCE
        .find("kernel void log_softmax_bwd_apply_f32(")
        .expect("log_softmax_bwd_apply_f32 declaration must exist");
    let lane_sum_body = &LSB_METAL_SOURCE[lane_sum_start..apply_start];
    assert!(
        lane_sum_body.contains("lsb_f64_widen") && lane_sum_body.contains("lsb_f64_add"),
        "log_softmax_bwd_lane_sum は lsb_f64_widen／lsb_f64_add を使うはず"
    );
    assert!(
        !lane_sum_body.contains("lsb_f64_narrow"),
        "log_softmax_bwd_lane_sum は narrow せず f64 bit のまま lane_sum へ書くはず"
    );

    let apply_body = &LSB_METAL_SOURCE[apply_start..];
    assert!(
        apply_body.contains("lsb_f64_widen")
            && apply_body.contains("lsb_f64_mul")
            && apply_body.contains("lsb_f64_sub")
            && apply_body.contains("lsb_f64_narrow"),
        "log_softmax_bwd_apply_f32 は lsb_f64_widen／lsb_f64_mul／lsb_f64_sub／lsb_f64_narrow を使うはず"
    );
}

/// 添字計算が `ulong`（64bit）で行われること（REQ-8 の一部。大規模
/// 形状での添字 overflow を防ぐ設計）を機械検証する。
#[test]
fn indices_are_64bit() {
    assert!(
        LSB_METAL_SOURCE.contains("(ulong)gid"),
        "gid からの添字計算は ulong キャストを経由するはず"
    );
}

/// `extract_fn_body`: 関数シグネチャ検索後の最初の `{` から波括弧対応で
/// 本体を抽出する（`reduce_source_evidence.rs::extract_fn_body` と同一
/// 実装。対応する閉じ波括弧が見つからない場合は `None`）。実際の定義は
/// 必ず `inline <戻り値型> {name}(` の形（行頭が `inline`）で始まる
/// 契約を利用し、コメント中の関数呼び出し例への誤ヒットを避ける。
fn extract_fn_body<'a>(source: &'a str, name: &str) -> Option<&'a str> {
    let needle = format!(" {name}(");
    let mut search_from = 0usize;
    let sig_start = loop {
        let rel = source[search_from..].find(&needle)?;
        let candidate = search_from + rel;
        let line_start = source[..candidate].rfind('\n').map(|i| i + 1).unwrap_or(0);
        if source[line_start..candidate]
            .trim_start()
            .starts_with("inline ")
        {
            break candidate;
        }
        search_from = candidate + needle.len();
    };
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

/// `lsb_f64_*`（本ファイル）の各関数本体が `ln_f64_*`（`layer_norm.
/// metal`）の対応する関数本体と接頭辞以外で逐語一致することを固定する
/// （`shaders/log_softmax_backward.metal` 冒頭コメント「binary64 の
/// ソフトウェアエミュレーション」の複製契約。`layer_norm.metal` を
/// 変更した場合に本ファイルへの追従漏れを検出するドリフトガード。
/// `reduce_source_evidence.rs::
/// red_f64_primitives_match_scan_f64_primitives_verbatim_modulo_prefix`
/// と同方針）。`log_softmax_backward.metal` は加算・減算・乗算を使う
/// ため `mul`（および `mul64_wide`／`shr128`／`low_bits128`／`cmp128`／
/// `normalize_mantissa` の精密乗算補助関数一式）まで比較対象に含める。
#[test]
fn lsb_f64_primitives_match_ln_f64_primitives_verbatim_modulo_prefix() {
    let renamed_ln_source = LAYER_NORM_METAL_SOURCE
        .replace("ln_f64_", "lsb_f64_")
        .replace("LN_F64_", "LSB_F64_")
        .replace("LN_F32_", "LSB_F32_")
        .replace("LnU128", "LsbU128")
        .replace("LnNormMantissa", "LsbNormMantissa");

    let fn_names = [
        "lsb_f64_clz64",
        "lsb_f64_widen",
        "lsb_f64_neg",
        "lsb_f64_add",
        "lsb_f64_sub",
        "lsb_f64_narrow",
        "lsb_f64_mul64_wide",
        "lsb_f64_shr128",
        "lsb_f64_low_bits128",
        "lsb_f64_cmp128",
        "lsb_f64_normalize_mantissa",
        "lsb_f64_mul",
    ];

    for name in fn_names {
        let lsb_body = extract_fn_body(LSB_METAL_SOURCE, name).unwrap_or_else(|| {
            panic!("log_softmax_backward.metal に {name} の定義（本体）が見つかりません")
        });
        let ln_body = extract_fn_body(&renamed_ln_source, name).unwrap_or_else(|| {
            panic!(
                "layer_norm.metal（接頭辞置換後）に {name} の定義（本体）が見つかりません \
                 （ドリフトガード自体の不整合）"
            )
        });
        assert_eq!(
            lsb_body, ln_body,
            "{name} の本体が layer_norm.metal（接頭辞置換後）と逐語一致しません（ドリフト検出）"
        );
    }
}
