//! A/B packing（BLIS 5-loop の pc/ic 層で使う連続バッファ生成）。
//!
//! [`super::microkernel`] の各カーネル（scalar／neon／avx2）は、A・B が
//! ストライドアクセス無しで読める「p-major」連続レイアウトであることを
//! 前提にする（キャッシュライン再利用・SIMD ロードの単純化。BLIS/GotoBLAS2
//! の 5-loop model と同じ設計判断）。本モジュールはその packing のみを
//! 担当し、安全な slice 操作（インデックスアクセス・範囲外は境界検査で
//! 弾かれる）で実装する（intrinsics は使わないため `unsafe` 不要）。
//!
//! ## レイアウト
//!
//! - A packing: 戻り値は `mr * kc_len` 要素、`ap[p * mr + i]` が
//!   `a[(row_start+i)*k_total + kc_start+p]` に対応する（p-major・i が
//!   連続方向）。`i >= mr_eff`（有効行数を超える）ぶんはゼロ padding。
//! - B packing: 戻り値は `kc_len * nr` 要素、`bp[p * nr + j]` が
//!   `b[(kc_start+p)*n_total + col_start+j]` に対応する（p-major・j が
//!   連続方向）。`j >= nr_eff` ぶんはゼロ padding。
//!
//! ## ゼロ padding が bit 完全一致契約（REQ-2）を崩さない理由
//!
//! padding は m／n 方向（有効レーンを超える i／j）にのみ発生し、K
//! 方向には発生しない（`kc_len` は呼び出し元が実際の残り K 幅として渡す
//! ため、常に有効値のみで構成される）。padding レーンは無限大等の非有限
//! 値を含まないゼロ埋めであり、有効レーンの蓄積とは独立した別レジスタ
//! ／別 `(i,j)` 出力先を占有するのみで、有効レーンの計算結果に一切影響
//! しない（マイクロカーネルはレーン間縮約を行わない設計。
//! `.claude/rules/coding-rust.md` FMA 契約統一節）。

/// A パネルを packing する。戻り値の長さは `mr * kc_len`。
///
/// `row_start + mr_eff <= a` の行数、`kc_start + kc_len <= k_total` を
/// 満たすことは呼び出し元（[`super`] の 5-loop ドライバ）の責務とする
/// （ドライバ側は `validate_dims` 済みの `m`・`k` 全体形状からブロック
/// 境界を計算するため、範囲外アクセスは発生しない設計）。
pub(super) fn pack_a(
    a: &[f32],
    k_total: usize,
    row_start: usize,
    mr: usize,
    mr_eff: usize,
    kc_start: usize,
    kc_len: usize,
) -> Vec<f32> {
    let mut ap = vec![0.0f32; mr * kc_len];
    for i in 0..mr_eff {
        let src_row =
            &a[(row_start + i) * k_total + kc_start..(row_start + i) * k_total + kc_start + kc_len];
        for (p, &val) in src_row.iter().enumerate() {
            ap[p * mr + i] = val;
        }
    }
    ap
}

/// B パネルを packing する。戻り値の長さは `kc_len * nr`。
///
/// `col_start + nr_eff <= n_total`、`kc_start + kc_len <= k_total`（B の
/// 行数）を満たすことは呼び出し元の責務とする（[`pack_a`] と同じ設計）。
pub(super) fn pack_b(
    b: &[f32],
    n_total: usize,
    kc_start: usize,
    kc_len: usize,
    col_start: usize,
    nr: usize,
    nr_eff: usize,
) -> Vec<f32> {
    let mut bp = vec![0.0f32; kc_len * nr];
    for p in 0..kc_len {
        let src_row =
            &b[(kc_start + p) * n_total + col_start..(kc_start + p) * n_total + col_start + nr_eff];
        bp[p * nr..p * nr + nr_eff].copy_from_slice(src_row);
    }
    bp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_a_full_tile_matches_source() {
        // 2 行 3 列（k_total=3）の a から mr=2(mr_eff=2)・kc_start=0・kc_len=3 を packing。
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ap = pack_a(&a, 3, 0, 2, 2, 0, 3);
        // p-major: [p0i0, p0i1, p1i0, p1i1, p2i0, p2i1]
        assert_eq!(ap, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn pack_a_edge_tile_zero_pads_unused_rows() {
        // mr=4 だが有効行は 2 行のみ（端タイル）。
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let ap = pack_a(&a, 2, 0, 4, 2, 0, 2);
        // p-major, mr=4: [p0i0,p0i1,p0i2,p0i3, p1i0,p1i1,p1i2,p1i3]
        assert_eq!(ap, vec![1.0, 3.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0]);
    }

    #[test]
    fn pack_b_full_tile_matches_source() {
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 rows x 3 cols
        let bp = pack_b(&b, 3, 0, 2, 0, 3, 3);
        assert_eq!(bp, b);
    }

    #[test]
    fn pack_b_edge_tile_zero_pads_unused_cols() {
        let b = vec![1.0, 2.0, 3.0, 4.0]; // 2 rows x 2 cols
        let bp = pack_b(&b, 2, 0, 2, 0, 4, 2);
        // p-major, nr=4: [p0j0,p0j1,0,0, p1j0,p1j1,0,0]
        assert_eq!(bp, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]);
    }
}
