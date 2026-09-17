//! イシュー #1953（親 #1947）: LayerNorm／RMSNorm backward カーネル
//! （MSL）の文字列証跡テスト。`tests/layer_norm_source_evidence.rs` と
//! 同方針: `include_str!` によるビルド時文字列埋め込みへの contains
//! 検査のみで完結するため、Metal 実機・`cfg(target_os = "macos")` を
//! 必要とせず Linux CI（GitHub ホステッド）上でも green になる。
//!
//! `.claude/rules/coding-rust.md`「REQ-8: 性能下限・最適化の達成を理由に
//! 手動境界チェックを省略しない」の機械検証と、
//! `crates/backend-metal/src/shaders/norm_backward.metal` 冒頭コメントが
//! 明記するカーネル構成（4 カーネル名・`nb_f64_*` soft-f64 関数・
//! `ulong row_base` によるオーバーフロー安全な添字・dw／db カーネルの
//! 列方向境界検査・dx カーネルの 5 段 butterfly reduction）のロックを
//! 兼ねる。

/// `crates/backend-metal/src/shaders/norm_backward.metal` のソース全文。
const NORM_BACKWARD_METAL_SOURCE: &str = include_str!("../src/shaders/norm_backward.metal");

/// 4 カーネルすべてが宣言されていることをロックする。
#[test]
fn all_four_kernels_are_declared() {
    for name in [
        "kernel void rmsnorm_bwd_dx_f32(",
        "kernel void rmsnorm_bwd_dw_f32(",
        "kernel void layer_norm_bwd_dx_f32(",
        "kernel void layer_norm_bwd_dwdb_f32(",
    ] {
        assert!(
            NORM_BACKWARD_METAL_SOURCE.contains(name),
            "カーネル宣言が見つかりません: {name}"
        );
    }
}

/// REQ-8 境界検査の一環: 行アドレス計算が `ulong`（64-bit）を使い
/// `rows * hidden` の乗算オーバーフローを避けることをロックする
/// （`layer_norm.metal`／`rmsnorm.metal` と同じ対策）。
#[test]
fn row_base_uses_64bit_index_to_avoid_overflow() {
    let occurrences = NORM_BACKWARD_METAL_SOURCE
        .matches("ulong row_base = (ulong)row * (ulong)hidden;")
        .count();
    assert_eq!(
        occurrences, 2,
        "dx カーネル（RMSNorm・LayerNorm）2 箇所で ulong row_base が使われるはずだが \
         {occurrences} 箇所しか見つからなかった"
    );
}

/// dw／db カーネル（1 スレッド = 1 列）が手動境界チェックを持つことを
/// ロックする（REQ-8。`.claude/rules/coding-rust.md`「カーネル実装の
/// 境界検査」節）。
#[test]
fn dw_and_dwdb_kernels_have_manual_bounds_check() {
    let occurrences = NORM_BACKWARD_METAL_SOURCE
        .matches("if (col >= hidden) {\n        return;\n    }")
        .count();
    assert_eq!(
        occurrences, 2,
        "dw／dwdb カーネル 2 箇所で列境界検査が使われるはずだが {occurrences} \
         箇所しか見つからなかった"
    );
}

/// dx カーネル（RMSNorm・LayerNorm）が `simd_shuffle_xor` を用いた
/// 5 段 butterfly（`offset` を 16u→1u へ 5 回半減させるループ）で
/// reduction することをロックする（RMSNorm dx は 1 パス〈二乗和〉+
/// 1 パス〈dot〉= 2 箇所、LayerNorm dx は 3 パス〈mean・var・
/// sum_dxhat+dot 同時ループ〉= 3 箇所で計 5 箇所）。
#[test]
fn dx_kernels_use_five_stage_butterfly_via_shuffle_helper() {
    let occurrences = NORM_BACKWARD_METAL_SOURCE
        .matches("for (ushort offset = 16u; offset > 0u; offset >>= 1u)")
        .count();
    assert_eq!(
        occurrences, 5,
        "5 段 butterfly ループ（16u→1u の 5 回半減）は dx カーネル計 5 箇所に \
         存在するはずだが {occurrences} 箇所しか見つからなかった"
    );
}

/// `nb_shuffle_xor_u64`（`ulong` を 32bit 上位・下位へ分割してシャッフル
/// するヘルパ）が定義され、reduction 呼び出し箇所（RMSNorm dx 2・
/// LayerNorm dx 4〈パス 1／2 が各 1・パス 3 が `sum_dxhat`／`dot` の
/// 2 アキュムレータを同一ループで縮約するため 2 呼び出し〉= 計 6）
/// から呼ばれることをロックする（`layer_norm.metal` の同型パターンと
/// 同じ理由: 単一の `simd_shuffle_xor` 呼び出しは `uint`〈32bit〉のみを
/// 扱う）。
#[test]
fn shuffle_helper_is_defined_and_used_by_all_reductions() {
    assert!(
        NORM_BACKWARD_METAL_SOURCE.contains("inline ulong nb_shuffle_xor_u64("),
        "nb_shuffle_xor_u64 ヘルパの定義が見つかりません"
    );
    let occurrences = NORM_BACKWARD_METAL_SOURCE
        .matches("nb_shuffle_xor_u64(")
        .count();
    // 定義 1 箇所 + 呼び出し 6 箇所（RMSNorm dx 2・LayerNorm dx 4）。
    assert_eq!(
        occurrences, 7,
        "nb_shuffle_xor_u64 の定義・呼び出し合計は 7 箇所のはずだが {occurrences} \
         箇所しか見つからなかった"
    );
}

/// `threadgroup_barrier` を使わない（persistent simdgroup 方式・
/// device メモリ再読のみで threadgroup memory を経由しないため不要。
/// `layer_norm.metal` と同じ設計）ことをロックする。
#[test]
fn kernels_do_not_use_threadgroup_barrier() {
    assert!(
        !NORM_BACKWARD_METAL_SOURCE.contains("threadgroup_barrier"),
        "threadgroup_barrier は使わない設計のはずだが検出された"
    );
}

/// `rsqrtf`／`INFINITY`／`#include`（`rsqrtf` は不使用・`INFINITY`／
/// `#include` は #1105／#1893 の教訓）を使わないことをロックする
/// （`kernels_norm_backward.rs` 冒頭コメント「REQ-8 境界検査」節と同じ
/// 教訓を Metal 側にも適用する。`<metal_stdlib>` の `#include` のみ許容
/// するため、`metal_stdlib` を除いた `#include` の非存在を検査する）。
#[test]
fn kernels_avoid_forbidden_directives() {
    assert!(
        !NORM_BACKWARD_METAL_SOURCE.contains("rsqrtf"),
        "rsqrtf は使わない設計のはずだが検出された"
    );
    assert!(
        !NORM_BACKWARD_METAL_SOURCE.contains("INFINITY"),
        "INFINITY マクロは使わない設計のはずだが検出された"
    );
    let include_count = NORM_BACKWARD_METAL_SOURCE.matches("#include").count();
    assert_eq!(
        include_count, 1,
        "#include は <metal_stdlib> の 1 箇所のみのはずだが {include_count} 箇所見つかった"
    );
    assert!(NORM_BACKWARD_METAL_SOURCE.contains("#include <metal_stdlib>"));
}

/// `nb_f64_*` soft-f64 プリミティブ（widen／add／sub／mul／div／narrow／
/// rsqrt_newton）がすべて定義されていることをロックする（`crate::soft_f64`
/// のホスト側逐語モデルとの 1 対 1 対応が本ファイル変更時に壊れていない
/// ことを検出するための最小限の存在検査）。
#[test]
fn nb_f64_primitives_are_all_defined() {
    for name in [
        "inline ulong nb_f64_widen(",
        "inline ulong nb_f64_neg(",
        "inline ulong nb_f64_add(",
        "inline ulong nb_f64_sub(",
        "inline uint nb_f64_narrow(",
        "inline ulong nb_f64_mul(",
        "inline ulong nb_f64_div(",
        "inline ulong nb_f64_rsqrt_newton(",
    ] {
        assert!(
            NORM_BACKWARD_METAL_SOURCE.contains(name),
            "soft-f64 プリミティブの定義が見つかりません: {name}"
        );
    }
}
