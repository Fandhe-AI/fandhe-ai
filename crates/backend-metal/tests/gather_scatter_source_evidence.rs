//! イシュー #1778: gather／scatter／scatter_add カーネル（MSL）の
//! 文字列証跡テスト。`tests/layer_norm_source_evidence.rs` と同方針:
//! `include_str!` によるビルド時文字列埋め込みへの contains 検査のみで
//! 完結するため、Metal 実機・`cfg(target_os = "macos")` を必要とせず
//! Linux CI（GitHub ホステッド）上でも green になる。
//!
//! `.claude/rules/coding-rust.md`「REQ-8: 性能下限・最適化の達成を理由に
//! 手動境界チェックを省略しない」の機械検証と、
//! `crates/backend-metal/src/shaders/gather_scatter.metal` 冒頭コメントが
//! 明記するアルゴリズム契約（出力定常方式・`scatter_add_f32` のみが
//! soft-f64 経路を通ること・`gemm.metal::bias_f64_*` との意図的な
//! ソース複製）のロックを兼ねる。

/// `crates/backend-metal/src/shaders/gather_scatter.metal` のソース全文。
const GATHER_SCATTER_METAL_SOURCE: &str = include_str!("../src/shaders/gather_scatter.metal");

/// `crates/backend-metal/src/soft_f64.rs` のソース全文（定数値の
/// ドリフト検出に使う）。
const SOFT_F64_SOURCE: &str = include_str!("../src/soft_f64.rs");

#[test]
fn kernel_names_and_buffer_order_are_declared() {
    assert!(
        GATHER_SCATTER_METAL_SOURCE.contains("kernel void gather_f32("),
        "gather_f32 カーネルの宣言が見つかりません"
    );
    assert!(
        GATHER_SCATTER_METAL_SOURCE.contains("kernel void scatter_overwrite_f32("),
        "scatter_overwrite_f32 カーネルの宣言が見つかりません"
    );
    assert!(
        GATHER_SCATTER_METAL_SOURCE.contains("kernel void scatter_add_f32("),
        "scatter_add_f32 カーネルの宣言が見つかりません"
    );
    // バッファ index の宣言順（`gather_scatter.rs` のエンコード関数と
    // 一致させる契約。冒頭コメント「バッファ配置」参照）。
    assert!(GATHER_SCATTER_METAL_SOURCE.contains("device const float* input [[buffer(0)]]"));
    assert!(GATHER_SCATTER_METAL_SOURCE.contains("device const int* index [[buffer(1)]]"));
}

/// `scatter_add_f32` のみが soft-f64 ヘルパ（`gs_f64_widen`／
/// `gs_f64_add`／`gs_f64_narrow`）を呼び、`gather_f32`／
/// `scatter_overwrite_f32` は呼ばない（丸めを伴わない純粋コピー・
/// 上書きのため。冒頭コメント「数値方式」参照）ことをロックする。
#[test]
fn only_scatter_add_uses_soft_f64_helpers() {
    let kernels: Vec<(&str, &str)> = split_kernels(GATHER_SCATTER_METAL_SOURCE);
    let mut checked_gather = false;
    let mut checked_overwrite = false;
    let mut checked_add = false;
    for (name, body) in kernels {
        let uses_soft_f64 = body.contains("gs_f64_widen")
            || body.contains("gs_f64_add(")
            || body.contains("gs_f64_narrow");
        match name {
            "gather_f32" => {
                assert!(
                    !uses_soft_f64,
                    "gather_f32 は soft-f64 ヘルパを使うべきではありません"
                );
                checked_gather = true;
            }
            "scatter_overwrite_f32" => {
                assert!(
                    !uses_soft_f64,
                    "scatter_overwrite_f32 は soft-f64 ヘルパを使うべきではありません"
                );
                checked_overwrite = true;
            }
            "scatter_add_f32" => {
                assert!(
                    uses_soft_f64,
                    "scatter_add_f32 は soft-f64 ヘルパ（gs_f64_widen/add/narrow）を使うはずです"
                );
                checked_add = true;
            }
            _ => {}
        }
    }
    assert!(
        checked_gather && checked_overwrite && checked_add,
        "3 カーネルすべてを検査できていません"
    );
}

/// `gs_f64_*` の soft-f64 定数値が `soft_f64.rs`（Rust 側逐語モデル）の
/// `F64_*`／`F32_*` 定数と同値であることをロックする（ドリフト検出。
/// `gemm.metal::bias_f64_*` と同型の設計判断）。
#[test]
fn gs_f64_constants_match_host_model_constants() {
    // Rust 側は `0x7FF8_0000_0000_0000` 形式、MSL 側は
    // `0x7FF8000000000000ul` 形式なのでアンダースコア除去済みの桁列で
    // 突き合わせる。
    let rust_qnan_f64 = SOFT_F64_SOURCE
        .lines()
        .find(|l| l.contains("const F64_QNAN"))
        .expect("F64_QNAN 定義が見つかりません");
    assert!(rust_qnan_f64.contains("0x7FF8_0000_0000_0000"));
    assert!(GATHER_SCATTER_METAL_SOURCE.contains("#define GS_F64_QNAN      0x7FF8000000000000ul"));

    let rust_inf_f64 = SOFT_F64_SOURCE
        .lines()
        .find(|l| l.contains("const F64_INF"))
        .expect("F64_INF 定義が見つかりません");
    assert!(rust_inf_f64.contains("0x7FF0_0000_0000_0000"));
    assert!(GATHER_SCATTER_METAL_SOURCE.contains("#define GS_F64_INF       0x7FF0000000000000ul"));

    assert!(GATHER_SCATTER_METAL_SOURCE.contains("#define GS_F32_QNAN      0x7FC00000u"));
    assert!(GATHER_SCATTER_METAL_SOURCE.contains("#define GS_F32_INF       0x7F800000u"));
}

/// REQ-8 境界検査: 3 カーネルすべてが `gid >= numel`（または
/// `numel_out`）の早期 return を持つ（末尾ブロックの余剰スレッド対策。
/// 手動境界チェックを省略しない）。
#[test]
fn all_kernels_have_grid_boundary_guard() {
    let occurrences_numel = GATHER_SCATTER_METAL_SOURCE
        .matches("if (gid >= numel) {")
        .count();
    let occurrences_numel_out = GATHER_SCATTER_METAL_SOURCE
        .matches("if (gid >= numel_out) {")
        .count();
    assert_eq!(
        occurrences_numel, 1,
        "gather_f32 の `gid >= numel` 境界検査が想定数と異なります"
    );
    assert_eq!(
        occurrences_numel_out, 2,
        "scatter 2 カーネルの `gid >= numel_out` 境界検査が想定数と異なります"
    );
}

/// gather の範囲外添字ガード（多層防御。`gather_scatter.rs` がホスト側
/// で事前検査済みだが、カーネル側にも防御的ガードを残す契約）をロック
/// する。
#[test]
fn gather_has_defensive_index_range_guard() {
    assert!(
        GATHER_SCATTER_METAL_SOURCE
            .contains("if (dim_idx_raw < 0 || (uint)dim_idx_raw >= dim_size) {"),
        "gather_f32 に防御的な範囲外添字ガードが見つかりません"
    );
}

/// 添字計算に `ulong`（64bit）を使うことをロックする（`u32` の中間
/// ストライド積オーバーフロー対策。REQ-8）。
#[test]
fn stride_arithmetic_uses_ulong() {
    assert!(GATHER_SCATTER_METAL_SOURCE.contains("thread ulong coords[GS_MAX_RANK]"));
    assert!(GATHER_SCATTER_METAL_SOURCE.contains(
        "inline ulong gs_ravel(thread const ulong* coords, constant uint* shape, uint rank)"
    ));
}

/// ソースを冒頭コメントの `kernel void <name>(` 宣言で雑に分割する
/// （厳密な MSL パーサではなく、本テストが必要とする粒度の簡易分割）。
fn split_kernels(source: &str) -> Vec<(&str, &str)> {
    let marker = "kernel void ";
    let mut result = Vec::new();
    let positions: Vec<usize> = {
        let mut v = Vec::new();
        let mut start = 0usize;
        while let Some(pos) = source[start..].find(marker) {
            v.push(start + pos);
            start = start + pos + marker.len();
        }
        v
    };
    for (i, &start) in positions.iter().enumerate() {
        let end = positions.get(i + 1).copied().unwrap_or(source.len());
        let chunk = &source[start..end];
        let name_start = marker.len();
        let name_end = chunk[name_start..]
            .find('(')
            .map(|p| name_start + p)
            .unwrap_or(chunk.len());
        let name = chunk[name_start..name_end].trim();
        result.push((name, chunk));
    }
    result
}
