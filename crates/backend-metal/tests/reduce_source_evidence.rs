//! イシュー #1895: `reduce_sum_all_chunk_f32`／`reduce_sum_all_finalize_f32`／
//! `reduce_sum_axis_f32` カーネル（MSL）の文字列証跡テスト。
//! `scan_source_evidence.rs` と同方針: `include_str!` によるビルド時
//! 文字列埋め込みへの contains 検査のみで完結するため、Metal 実機・
//! `cfg(target_os = "macos")` を必要とせず Linux CI（GitHub ホステッド）
//! 上でも green になる。
//!
//! `.claude/rules/coding-rust.md`「REQ-8: 性能下限・最適化の達成を理由に
//! 手動境界チェックを省略しない」の機械検証（`gid` 境界検査）と、
//! `crate::soft_f64`／`crate::reduce_model` と bit 完全一致する契約の
//! 根拠となる binary64 ソフトウェアエミュレーションアキュムレータ
//! （`red_f64_widen`／`red_f64_add`／`red_f64_narrow`）の使用のロック、
//! `crate::reduce_model::REDUCE_SUM_CHUNK` と MSL 側リテラルの一致検証、
//! `scan.metal::scan_f64_*` との接頭辞置換後の関数本体逐語一致（ドリフト
//! ガード。`batch_norm_source_evidence.rs::extract_fn_body` と同方針）を
//! 兼ねる。

/// `crates/backend-metal/src/shaders/reduce.metal` のソース全文。
const REDUCE_METAL_SOURCE: &str = include_str!("../src/shaders/reduce.metal");

/// `crates/backend-metal/src/shaders/scan.metal` のソース全文
/// （接頭辞置換後の逐語一致比較対象。`scan.metal` は本イシューより
/// 前に確立済みの soft-f64 プリミティブ定義元）。
const SCAN_METAL_SOURCE: &str = include_str!("../src/shaders/scan.metal");

/// `crate::reduce::MetalReduce::new` はソース全文をそのまま
/// `newLibraryWithSource_options_error` へ渡して実行時コンパイルする
/// ため、`#include <metal_stdlib>`／`using namespace metal;` の宣言が
/// 欠けていると `LibraryCompilation` エラーになる（`scan.metal`・
/// `unique.metal` と同じ構成が必須）。
#[test]
fn source_includes_metal_stdlib_and_namespace() {
    assert!(
        REDUCE_METAL_SOURCE.contains("#include <metal_stdlib>"),
        "reduce.metal に `#include <metal_stdlib>` が見つかりません"
    );
    assert!(
        REDUCE_METAL_SOURCE.contains("using namespace metal;"),
        "reduce.metal に `using namespace metal;` が見つかりません"
    );
    let include_pos = REDUCE_METAL_SOURCE.find("#include <metal_stdlib>").unwrap();
    let using_pos = REDUCE_METAL_SOURCE.find("using namespace metal;").unwrap();
    assert!(
        include_pos < using_pos,
        "`#include <metal_stdlib>` は `using namespace metal;` より前に置く"
    );
}

/// `crate::reduce::MetalReduce::new` が `pipeline::make_pipeline` へ渡す
/// カーネル名（3 種）と、各カーネルが期待するバッファ index の宣言を
/// 機械検証する（`crate::reduce::encode_*` の index 割当と一致する
/// ことのロック）。
#[test]
fn kernel_names_and_buffer_indices_are_declared() {
    for kernel in [
        "reduce_sum_all_chunk_f32",
        "reduce_sum_all_finalize_f32",
        "reduce_sum_axis_f32",
    ] {
        assert!(
            REDUCE_METAL_SOURCE.contains(&format!("kernel void {kernel}(")),
            "{kernel} カーネルの宣言が見つかりません"
        );
    }
    // 3 カーネル合算で buffer index 0〜4 が使われる（chunk: 0/1/2/3・
    // finalize: 0/1/2・axis: 0/1/2/3/4）。
    for idx in 0..=4 {
        assert!(
            REDUCE_METAL_SOURCE.contains(&format!("[[buffer({idx})]]")),
            "buffer({idx}) の宣言が見つかりません"
        );
    }
}

/// REQ-8（境界検査規約）: `reduce_sum_all_chunk_f32` は
/// `gid >= num_chunks`、`reduce_sum_all_finalize_f32` は `gid != 0u`、
/// `reduce_sum_axis_f32` は `gid >= lanes` の境界検査を持つことを
/// 機械検証する（最適化を理由に省略しない契約のロック）。
#[test]
fn boundary_checks_are_present() {
    assert_eq!(
        REDUCE_METAL_SOURCE
            .matches("if (gid >= num_chunks)")
            .count(),
        1,
        "reduce_sum_all_chunk_f32 に `if (gid >= num_chunks)` 境界検査が必要"
    );
    assert_eq!(
        REDUCE_METAL_SOURCE.matches("if (gid != 0u)").count(),
        1,
        "reduce_sum_all_finalize_f32 に `if (gid != 0u)` 境界検査が必要"
    );
    assert_eq!(
        REDUCE_METAL_SOURCE.matches("if (gid >= lanes)").count(),
        1,
        "reduce_sum_axis_f32 に `if (gid >= lanes)` 境界検査が必要"
    );
}

/// binary64 ソフトウェアエミュレーションアキュムレータ
/// （`red_f64_widen`／`red_f64_add`／`red_f64_narrow`）が各カーネル本体で
/// 使われていることを機械検証する（MSL は `double` 非対応のため、ホスト
/// `f64` 参照実装との bit 完全一致契約はこのソフトウェアエミュレーション
/// 経路に依存する。`.claude/rules/coding-rust.md`「勾配の長軸縮約」節と
/// 同じ設計）。
#[test]
fn kernels_use_soft_f64_accumulator() {
    for f in ["red_f64_widen", "red_f64_add", "red_f64_narrow"] {
        assert!(
            REDUCE_METAL_SOURCE.contains(f),
            "reduce.metal に `{f}`（binary64 ソフトウェアエミュレーション）が見つかりません"
        );
    }
    // 各カーネル本体区間で widen/add（chunk・axis）・add/narrow（finalize）
    // が使われることを個別に確認する。
    let chunk_start = REDUCE_METAL_SOURCE
        .find("kernel void reduce_sum_all_chunk_f32(")
        .expect("reduce_sum_all_chunk_f32 declaration must exist");
    let finalize_start = REDUCE_METAL_SOURCE
        .find("kernel void reduce_sum_all_finalize_f32(")
        .expect("reduce_sum_all_finalize_f32 declaration must exist");
    let axis_start = REDUCE_METAL_SOURCE
        .find("kernel void reduce_sum_axis_f32(")
        .expect("reduce_sum_axis_f32 declaration must exist");

    let chunk_body = &REDUCE_METAL_SOURCE[chunk_start..finalize_start];
    let finalize_body = &REDUCE_METAL_SOURCE[finalize_start..axis_start];
    let axis_body = &REDUCE_METAL_SOURCE[axis_start..];

    assert!(
        chunk_body.contains("red_f64_widen") && chunk_body.contains("red_f64_add"),
        "reduce_sum_all_chunk_f32 は red_f64_widen／red_f64_add を使うはず"
    );
    assert!(
        !chunk_body.contains("red_f64_narrow"),
        "reduce_sum_all_chunk_f32 は narrow せず f64 bit のまま partial へ書くはず"
    );
    assert!(
        finalize_body.contains("red_f64_add") && finalize_body.contains("red_f64_narrow"),
        "reduce_sum_all_finalize_f32 は red_f64_add／red_f64_narrow を使うはず"
    );
    assert!(
        axis_body.contains("red_f64_widen")
            && axis_body.contains("red_f64_add")
            && axis_body.contains("red_f64_narrow"),
        "reduce_sum_axis_f32 は red_f64_widen／red_f64_add／red_f64_narrow を使うはず"
    );
}

/// `crate::reduce_model::REDUCE_SUM_CHUNK`（Rust 側定数）と MSL 側
/// `#define REDUCE_SUM_CHUNK` リテラルの一致をロックする（ドリフト
/// 検出。`reduce_model.rs` 単体テストの境界感度テストと組み合わせて
/// 「Rust 側定数だけ変更してカーネル側を追従し忘れる」事故を防ぐ）。
#[test]
fn reduce_sum_chunk_constant_matches_rust_side() {
    let expected = format!(
        "#define REDUCE_SUM_CHUNK {}u",
        fandhe_ai_backend_metal::reduce_model::REDUCE_SUM_CHUNK
    );
    assert!(
        REDUCE_METAL_SOURCE.contains(&expected),
        "reduce.metal の `#define REDUCE_SUM_CHUNK` が Rust 側 `REDUCE_SUM_CHUNK`（{}）と一致しません（探索文字列: {expected:?}）",
        fandhe_ai_backend_metal::reduce_model::REDUCE_SUM_CHUNK
    );
}

/// `extract_fn_body`: 関数シグネチャ検索後の最初の `{` から波括弧対応で
/// 本体を抽出する（`batch_norm_source_evidence.rs::extract_fn_body` と
/// 同一実装。対応する閉じ波括弧が見つからない場合は `None`）。
///
/// `reduce.metal` の doc コメント（冒頭「数値方式」節）が
/// `` `acc = red_f64_add(acc, red_f64_widen(x[idx]))` `` のように
/// 関数呼び出し構文をそのまま引用するため、単純な `" {name}("` 検索
/// では実際の関数定義より前のコメント中の呼び出し例にヒットしうる
/// （`batch_norm_source_evidence.rs` の対象 2 ファイルにはこの引用
/// パターンが存在しないため同種の問題が顕在化しなかった）。実際の
/// 定義は必ず `inline <戻り値型> {name}(` の形（行頭が `inline`）で
/// 始まる契約を利用し、`\ninline` に続く最初の出現のみを対象とする。
fn extract_fn_body<'a>(source: &'a str, name: &str) -> Option<&'a str> {
    let needle = format!(" {name}(");
    let mut search_from = 0usize;
    let sig_start = loop {
        let rel = source[search_from..].find(&needle)?;
        let candidate = search_from + rel;
        // 直前が改行 + `inline` であれば実定義とみなす（`inline` の
        // 戻り値型トークンの直前に必ず改行がある契約は本ファイル・
        // `scan.metal` の関数群すべてで成立する）。
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

/// [`red_f64_*`]（本ファイル）の各関数本体が [`scan_f64_*`]
/// （`scan.metal`）の対応する関数本体と接頭辞以外で逐語一致することを
/// 固定する（`shaders/reduce.metal` 冒頭コメント「soft-f64 プリミティブ」
/// の複製契約。`scan.metal` を変更した場合に本ファイルへの追従漏れを
/// 検出するドリフトガード。`batch_norm_source_evidence.rs::
/// bn_f64_primitives_match_ln_f64_primitives_verbatim_modulo_prefix` と
/// 同方針）。`reduce.metal` は加算のみ使うため比較対象の関数は
/// `clz64`／`widen`／`add`／`narrow` の 4 つに限る（`mul` 等は含まない）。
#[test]
fn red_f64_primitives_match_scan_f64_primitives_verbatim_modulo_prefix() {
    let renamed_scan_source = SCAN_METAL_SOURCE
        .replace("scan_f64_", "red_f64_")
        .replace("SCAN_F64_", "RED_F64_")
        .replace("SCAN_F32_", "RED_F32_");

    let fn_names = [
        "red_f64_clz64",
        "red_f64_widen",
        "red_f64_add",
        "red_f64_narrow",
    ];

    for name in fn_names {
        let reduce_body = extract_fn_body(REDUCE_METAL_SOURCE, name)
            .unwrap_or_else(|| panic!("reduce.metal に {name} の定義（本体）が見つかりません"));
        let scan_body = extract_fn_body(&renamed_scan_source, name).unwrap_or_else(|| {
            panic!(
                "scan.metal（接頭辞置換後）に {name} の定義（本体）が見つかりません \
                 （ドリフトガード自体の不整合）"
            )
        });
        assert_eq!(
            reduce_body, scan_body,
            "{name} の本体が scan.metal（接頭辞置換後）と逐語一致しません（ドリフト検出）"
        );
    }
}

/// 添字計算が `ulong`（64bit）で行われること（`REQ-8` の一部。大規模
/// 形状での添字 overflow を防ぐ設計）・`threadgroup_barrier` を
/// 使わないこと（`scan.metal` と同じ「lane 逐次」構造の機構的確認）を
/// 機械検証する。
#[test]
fn indices_are_64bit_and_no_threadgroup_barrier() {
    assert!(
        REDUCE_METAL_SOURCE.contains("(ulong)gid"),
        "gid からの添字計算は ulong キャストを経由するはず"
    );
    assert!(
        !REDUCE_METAL_SOURCE.contains("threadgroup_barrier"),
        "reduce.metal は lane 逐次構造のため threadgroup_barrier を使わないはず"
    );
}
