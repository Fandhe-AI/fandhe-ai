//! TASK-11.3（#70）: 行列演算ユニット活用（REQ-11）の証跡として、
//! `gemm.metal` に Apple GPU の行列専用命令（`simdgroup_matrix` API）が
//! 実在することを機械検査する。
//!
//! CUDA 側 `backend-cuda` の `kernels_wmma.rs`／`kernels_wmma_opt.rs`／
//! `kernels_mma.rs` 末尾 `#[cfg(test)]` にある「tensor core 命令実在検査」
//! と対になる証跡であり、`docs/matrix-unit-dispatch.md` の命令実在一覧表が
//! 参照する（同ドキュメントは主張のみを記述し、その主張の機械検証は
//! ここで担う。二重管理を避けるため実測値・命令リストの重複は書かない）。
//!
//! `include_str!` によるビルド時文字列埋め込みへの contains 検査のみで
//! 完結するため、Metal 実機・`cfg(target_os = "macos")` を必要とせず
//! Linux CI（self-hosted）上でも green になる。既存の `tile.rs`（本クレート
//! 同様の Linux 実行可テスト）と同じ位置づり。

/// `crates/backend-metal/src/shaders/gemm.metal` のソース全文。
/// このファイル自体が実行対象ではなく、文字列としての内容検査のみに使う。
const GEMM_METAL_SOURCE: &str = include_str!("../src/shaders/gemm.metal");

/// REQ-11 の証跡: `gemm_simdgroup`／`gemm_simdgroup_tiled` カーネルが
/// Apple GPU の行列専用命令（`simdgroup_float8x8` 型・
/// `simdgroup_load`／`simdgroup_multiply_accumulate`／`simdgroup_store`
/// 関数）を実際に使用していることをロックする。将来の書き換えで
/// これらの命令が誤って除去された場合に検出する（CUDA 側の
/// `wmma_f16_source_uses_wmma_instructions` と同方針）。
#[test]
fn gemm_metal_source_uses_simdgroup_matrix_instructions() {
    for needle in [
        "#include <metal_simdgroup_matrix>",
        "simdgroup_float8x8",
        "simdgroup_load",
        "simdgroup_multiply_accumulate",
        "simdgroup_store",
    ] {
        assert!(
            GEMM_METAL_SOURCE.contains(needle),
            "gemm.metal に行列演算ユニット命令 `{needle}` が見つかりません"
        );
    }
}

/// `gemm_simdgroup_f16` の定義域（`MM_T`/`ACC_T` の `typedef simdgroup_half8x8`
/// 開始位置から、次に定義される `gemm_simdgroup_tiled` 開始位置まで）を
/// 切り出す。
///
/// 開始位置を `kernel void gemm_simdgroup_f16(` ではなく直前の `typedef` 行に
/// 取る理由: 実際の行列型（`simdgroup_half8x8`）はカーネル本体内ではなく
/// この typedef 行にのみ現れる（本体は型エイリアス `MM_T`/`ACC_T` を使う。
/// L208-209）。`typedef` 行はこの f16 カーネル専用（f32 版 `gemm_simdgroup`
/// は L154-179 で既に完結しており typedef より前）であり、かつ f16 カーネル
/// 本体全体を包含するため、命令実在検査・REQ-8 境界チェック検査の両方に
/// 安全に使える。
///
/// 検索対象を `GEMM_METAL_SOURCE` 全文のままにすると、`gemm_simdgroup`
/// （f32 版。L154〜）が同じ `simdgroup_load`/`simdgroup_multiply_accumulate`/
/// `simdgroup_store` 命令を先に使っているため、f16 カーネル側からこれらの
/// 命令を誤って取り除いてもテストが green のまま通過する偽陽性が生じる
/// （PR #346 Bugbot 指摘）。
fn gemm_simdgroup_f16_kernel_body() -> &'static str {
    let kernel_start = GEMM_METAL_SOURCE
        .find("typedef simdgroup_half8x8 MM_T;")
        .expect("gemm_simdgroup_f16 の MM_T typedef が見つかりません");
    let next_kernel_start = GEMM_METAL_SOURCE[kernel_start..]
        .find("kernel void gemm_simdgroup_tiled(")
        .map(|offset| kernel_start + offset)
        .expect("gemm_simdgroup_tiled カーネル本体が見つかりません（次カーネル境界の特定に失敗）");
    &GEMM_METAL_SOURCE[kernel_start..next_kernel_start]
}

/// REQ-11・TASK-8.3b（#156）の証跡: `gemm_simdgroup_f16` が半精度
/// simdgroup 行列型（`simdgroup_half8x8`）と行列演算ユニット命令
/// （`simdgroup_load`/`simdgroup_multiply_accumulate`/`simdgroup_store`）を
/// 実際に使用していることをロックする（CUDA 側
/// `wmma_f16_source_uses_wmma_instructions` と同方針。
/// `docs/matrix-unit-dispatch.md` の命令実在一覧表が参照する）。
///
/// 検索範囲は `gemm_simdgroup_f16_kernel_body` で f16 カーネル本体に限定する
/// （全文検索だと f32 版 `gemm_simdgroup` の同一命令に一致してしまう偽陽性を防ぐ。
/// PR #346 Bugbot 指摘）。
#[test]
fn gemm_simdgroup_f16_source_uses_simdgroup_half_matrix_instructions() {
    let kernel_body = gemm_simdgroup_f16_kernel_body();
    for needle in [
        "simdgroup_half8x8",
        "simdgroup_load",
        "simdgroup_multiply_accumulate",
        "simdgroup_store",
    ] {
        assert!(
            kernel_body.contains(needle),
            "gemm.metal の gemm_simdgroup_f16 に `{needle}` が見つかりません"
        );
    }
}

/// REQ-8 の証跡: `gemm_simdgroup_f16` が手動境界チェック
/// （タイル原点が実効次元を超える場合の早期 return）を維持していることを
/// ロックする（`gemm_metal_source_uses_simdgroup_matrix_instructions` と
/// 対になる検査。性能上の下限・最適化の達成を理由に境界チェックを省略
/// しない方針 `.claude/rules/coding-rust.md` の機械検証）。
///
/// カーネル本体の切り出しは `gemm_simdgroup_f16` 開始位置から EOF までではなく、
/// 次に定義される `gemm_simdgroup_tiled` の開始位置までに限定する（
/// `gemm_simdgroup_f16_kernel_body` を共用）。EOF までスライスすると後続
/// カーネル内の同一境界ガード文字列にもマッチしてしまい、f16 カーネル側の
/// 境界チェックを誤って削除しても本テストが green のまま通過する偽陽性が
/// 生じるため（PR #346 Bugbot 指摘）。
#[test]
fn gemm_simdgroup_f16_source_retains_req8_boundary_guard() {
    let kernel_body = gemm_simdgroup_f16_kernel_body();
    assert!(
        kernel_body.contains("if (row0 >= dims.m || col0 >= dims.n)"),
        "gemm_simdgroup_f16 に REQ-8 手動境界チェックが見つかりません"
    );
}
