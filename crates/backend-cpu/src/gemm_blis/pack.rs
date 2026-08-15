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
//! - A packing: 書き込み先 `dst` は `mr * kc_len` 要素（長さは関数入口の
//!   `assert_eq!` で検査。REQ-8 カーネル境界検査規約）、`dst[p * mr + i]`
//!   が `a[(row_start+i)*k_total + kc_start+p]` に対応する（p-major・i が
//!   連続方向）。`i >= mr_eff`（有効行数を超える）ぶんはゼロ padding。
//! - B packing: 書き込み先 `dst` は `kc_len * nr` 要素（同上）、
//!   `dst[p * nr + j]` が `b[(kc_start+p)*n_total + col_start+j]` に
//!   対応する（p-major・j が連続方向）。`j >= nr_eff` ぶんはゼロ padding。
//!
//! ## panel バッファへの直接書き込み（#554）
//!
//! 呼び出し元（[`super::gemm_blis_region`]）が確保した `a_panel`／
//! `b_panel` のサブスライスを `dst` として直接渡す設計とし、関数内部での
//! 中間 `Vec` 確保と呼び出し元での `copy_from_slice` による二段コピーを
//! 廃した（BLIS/GotoBLAS2・matrixmultiply 等の参照実装が採る「呼び出し側
//! 確保のバッファへ直接書き込む」packing 方式に合わせる。M=N=K=4096 で
//! A packing は概算 65,536 回・B packing は 8,192 回呼ばれるため、都度の
//! ヒープ確保＋二重コピーの回避が狙い）。引数は
//! `clippy::too_many_arguments`（7 引数上限）を安易な `#[allow]` で
//! 黙らせず、タイル形状パラメータを [`APackTile`]／[`BPackTile`] に
//! まとめることで回避している（`.claude/rules/coding-rust.md`）。
//!
//! ## ゼロ padding が bit 完全一致契約（REQ-2）を崩さない理由
//!
//! padding は m／n 方向（有効レーンを超える i／j）にのみ発生し、K
//! 方向には発生しない（`kc_len` は呼び出し元が実際の残り K 幅として渡す
//! ため、常に有効値のみで構成される）。padding レーンは無限大等の非有限
//! 値を含まないゼロ埋めであり、有効レーンの蓄積とは独立した別レジスタ
//! ／別 `(i,j)` 出力先を占有するのみで、有効レーンの計算結果に一切影響
//! しない（マイクロカーネルはレーン間縮約を行わない設計。
//! `.claude/rules/coding-rust.md` FMA 契約統一節）。padding のゼロ保証は
//! `dst` 側の事前初期化に依存しない（端タイルでは本関数が `dst.fill(0.0)`
//! を行ってから有効値を書くため、`dst` に未初期化相当のゴミ値が入って
//! いても padding レーンは常にゼロになる）。

/// [`pack_a`] のタイル形状パラメータ（引数数を抑えるための束ね。#554）。
pub(super) struct APackTile {
    pub k_total: usize,
    pub row_start: usize,
    pub mr: usize,
    pub mr_eff: usize,
    pub kc_start: usize,
    pub kc_len: usize,
}

/// [`pack_b`] のタイル形状パラメータ（引数数を抑えるための束ね。#554）。
pub(super) struct BPackTile {
    pub n_total: usize,
    pub kc_start: usize,
    pub kc_len: usize,
    pub col_start: usize,
    pub nr: usize,
    pub nr_eff: usize,
}

/// A パネルを `dst` へ直接 packing する（`dst` は呼び出し元が確保した
/// panel バッファのサブスライス。#554）。`dst` の長さは `mr * kc_len`
/// でなければならない。
///
/// `row_start + mr_eff <= a` の行数、`kc_start + kc_len <= k_total` を
/// 満たすことは呼び出し元（[`super`] の 5-loop ドライバ）の責務とする
/// （ドライバ側は `validate_dims` 済みの `m`・`k` 全体形状からブロック
/// 境界を計算するため、範囲外アクセスは発生しない設計）。`dst` の長さは
/// 関数入口の `assert_eq!` で検査する（REQ-8: 性能最適化を理由に手動
/// 境界チェックを省略しない）。
pub(super) fn pack_a(dst: &mut [f32], a: &[f32], tile: APackTile) {
    let APackTile {
        k_total,
        row_start,
        mr,
        mr_eff,
        kc_start,
        kc_len,
    } = tile;
    assert_eq!(
        dst.len(),
        mr * kc_len,
        "pack_a: dst 長さが mr*kc_len と不一致"
    );
    // 端タイル（mr_eff < mr）のみ padding レーンをゼロ初期化する。dst は
    // 呼び出し元の panel バッファ（再利用されうる）のスライスであり、
    // 事前状態に依存せず本関数がゼロを保証する（モジュールドキュメント
    // 「ゼロ padding が bit 完全一致契約を崩さない理由」節）。full タイル
    // の一般ケースでは全レーンを有効値で上書きするため memset は不要。
    if mr_eff < mr {
        dst.fill(0.0);
    }
    for i in 0..mr_eff {
        let src_row =
            &a[(row_start + i) * k_total + kc_start..(row_start + i) * k_total + kc_start + kc_len];
        for (p, &val) in src_row.iter().enumerate() {
            dst[p * mr + i] = val;
        }
    }
}

/// B パネルを `dst` へ直接 packing する（`dst` は呼び出し元が確保した
/// panel バッファのサブスライス。#554）。`dst` の長さは `kc_len * nr`
/// でなければならない。
///
/// `col_start + nr_eff <= n_total`、`kc_start + kc_len <= k_total`（B の
/// 行数）を満たすことは呼び出し元の責務とする（[`pack_a`] と同じ設計）。
pub(super) fn pack_b(dst: &mut [f32], b: &[f32], tile: BPackTile) {
    let BPackTile {
        n_total,
        kc_start,
        kc_len,
        col_start,
        nr,
        nr_eff,
    } = tile;
    assert_eq!(
        dst.len(),
        kc_len * nr,
        "pack_b: dst 長さが kc_len*nr と不一致"
    );
    // pack_a と同じ理由（dst 事前状態に依存しない padding ゼロ保証）。
    if nr_eff < nr {
        dst.fill(0.0);
    }
    for p in 0..kc_len {
        let src_row =
            &b[(kc_start + p) * n_total + col_start..(kc_start + p) * n_total + col_start + nr_eff];
        dst[p * nr..p * nr + nr_eff].copy_from_slice(src_row);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_a_full_tile_matches_source() {
        // 2 行 3 列（k_total=3）の a から mr=2(mr_eff=2)・kc_start=0・kc_len=3 を packing。
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = vec![0.0f32; 2 * 3];
        pack_a(
            &mut dst,
            &a,
            APackTile {
                k_total: 3,
                row_start: 0,
                mr: 2,
                mr_eff: 2,
                kc_start: 0,
                kc_len: 3,
            },
        );
        // p-major: [p0i0, p0i1, p1i0, p1i1, p2i0, p2i1]
        assert_eq!(dst, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn pack_a_edge_tile_zero_pads_unused_rows() {
        // mr=4 だが有効行は 2 行のみ（端タイル）。dst には事前にゴミ値を
        // 入れておき、padding ゼロ保証が dst の事前初期化に依存しないこと
        // を検証する（#554: dst.fill(0.0) の自己完結保証）。
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let mut dst = vec![9.9f32; 4 * 2];
        pack_a(
            &mut dst,
            &a,
            APackTile {
                k_total: 2,
                row_start: 0,
                mr: 4,
                mr_eff: 2,
                kc_start: 0,
                kc_len: 2,
            },
        );
        // p-major, mr=4: [p0i0,p0i1,p0i2,p0i3, p1i0,p1i1,p1i2,p1i3]
        assert_eq!(dst, vec![1.0, 3.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0]);
    }

    #[test]
    fn pack_b_full_tile_matches_source() {
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 rows x 3 cols
        let mut dst = vec![0.0f32; 2 * 3];
        pack_b(
            &mut dst,
            &b,
            BPackTile {
                n_total: 3,
                kc_start: 0,
                kc_len: 2,
                col_start: 0,
                nr: 3,
                nr_eff: 3,
            },
        );
        assert_eq!(dst, b);
    }

    #[test]
    fn pack_b_edge_tile_zero_pads_unused_cols() {
        // dst に事前にゴミ値を入れ、padding ゼロ保証の自己完結性を検証する（#554）。
        let b = vec![1.0, 2.0, 3.0, 4.0]; // 2 rows x 2 cols
        let mut dst = vec![9.9f32; 2 * 4];
        pack_b(
            &mut dst,
            &b,
            BPackTile {
                n_total: 2,
                kc_start: 0,
                kc_len: 2,
                col_start: 0,
                nr: 4,
                nr_eff: 2,
            },
        );
        // p-major, nr=4: [p0j0,p0j1,0,0, p1j0,p1j1,0,0]
        assert_eq!(dst, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]);
    }
}
