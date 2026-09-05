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
//! ## 転置格納からの直接 packing（#1213）
//!
//! VJP（`crates/autodiff/src/grad.rs`）の逆伝播 3 箇所（`matmul_vjp` の
//! d_input／d_weight・`Op::LinearResident` の d_weight）が渡す片側転置
//! オペランド（`transpose2d` の zero-copy view）を、[`super::ops`]（`ops.rs`
//! の判定ヘルパー。転置元クレートは `backend-cpu` に閉じる）が dense な
//! 転置格納（`strides == [1, shape[0]]`）と判定できた場合に限り、
//! `Tensor::contiguous()` による再パックコピーを経由せず本モジュールが
//! 直接 packing する。[`pack_a_from_transposed`]（A オペランドが転置格納
//! ＝ TN パターン）・[`pack_b_from_transposed`]（B オペランドが転置格納
//! ＝ NT パターン）を追加する。両方転置（TT）・一般 stride（`narrow` 後の
//! 転置等）は本イシューのスコープ外で、従来どおり `contiguous()` へ
//! フォールバックする（`docs/matmul-vjp-zero-copy-decision.md` §3.2）。
//!
//! `pack_a_from_transposed` は転置格納 `at`（論理形状 `[k_total, m_total]`
//! の行優先。元の A `[m_total, k_total]` を転置した view）から、`pack_b`
//! と同じ「p（K 方向）が外側行」の連続コピーで packing できる（`at` の
//! 行そのものが K 方向に並ぶため）。逆に `pack_b_from_transposed` は転置
//! 格納 `bt`（論理形状 `[n_total, k_total]` の行優先。元の B
//! `[k_total, n_total]` を転置した view）から、`pack_a` と同じ「i（N 方向）
//! が外側行」の gather で packing する。すなわち転置格納からの packing は
//! 「pack_a と pack_b の役割が入れ替わる」形で実装でき、新規カーネル・
//! 新規累積順序を一切導入しない（bit 完全一致契約は「`contiguous()` して
//! から既存 `pack_a`／`pack_b` で packing した場合と同一バイト列を書く」
//! ことで保たれる。`tests/gemm_transposed_parity.rs` で直接検証）。
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

/// [`pack_a_from_transposed`] のタイル形状パラメータ（#1213。[`APackTile`]
/// と対応するが、`k_total`／`m_total` は転置格納 `at`（論理形状
/// `[k_total, m_total]`）の行優先レイアウトそのものを表す）。
pub(super) struct ATPackTile {
    pub k_total: usize,
    pub m_total: usize,
    pub row_start: usize,
    pub mr: usize,
    pub mr_eff: usize,
    pub kc_start: usize,
    pub kc_len: usize,
}

/// [`pack_b_from_transposed`] のタイル形状パラメータ（#1213。[`BPackTile`]
/// と対応するが、`k_total`／`n_total` は転置格納 `bt`（論理形状
/// `[n_total, k_total]`）の行優先レイアウトそのものを表す）。
pub(super) struct BTPackTile {
    pub k_total: usize,
    pub n_total: usize,
    pub kc_start: usize,
    pub kc_len: usize,
    pub col_start: usize,
    pub nr: usize,
    pub nr_eff: usize,
}

/// A パネルを転置格納 `at`（論理形状 `[k_total, m_total]` の行優先。
/// 元の A `[m_total, k_total]` を転置した view）から直接 packing する
/// （#1213）。`at` の行そのものが K 方向に並ぶため、`dst[p*mr+i]` への
/// 書き込みは `at` の行 `kc_start+p` から `mr_eff` 要素を連続コピーする
/// だけでよい（[`pack_b`] と同型の実装。モジュールドキュメント
/// 「転置格納からの直接 packing」節参照）。`dst` の長さは `mr * kc_len`
/// でなければならない（[`pack_a`] と同じ境界検査規約。REQ-8）。
pub(super) fn pack_a_from_transposed(dst: &mut [f32], at: &[f32], tile: ATPackTile) {
    let ATPackTile {
        k_total,
        m_total,
        row_start,
        mr,
        mr_eff,
        kc_start,
        kc_len,
    } = tile;
    assert_eq!(
        dst.len(),
        mr * kc_len,
        "pack_a_from_transposed: dst 長さが mr*kc_len と不一致"
    );
    let _ = k_total; // 呼び出し元の意図（at の総行数）を明示するため保持。長さ検査は dst.len() で完結する。
    if mr_eff < mr {
        dst.fill(0.0);
    }
    for p in 0..kc_len {
        let src_row = &at
            [(kc_start + p) * m_total + row_start..(kc_start + p) * m_total + row_start + mr_eff];
        dst[p * mr..p * mr + mr_eff].copy_from_slice(src_row);
    }
}

/// B パネルを転置格納 `bt`（論理形状 `[n_total, k_total]` の行優先。
/// 元の B `[k_total, n_total]` を転置した view）から直接 packing する
/// （#1213）。`bt` の行が N 方向に並ぶため、`dst[p*nr+j]` への書き込みは
/// `bt` の行 `col_start+j` を p（K 方向）方向へ走査する gather になる
/// （[`pack_a`] と同型の実装）。`dst` の長さは `kc_len * nr` でなければ
/// ならない（[`pack_b`] と同じ境界検査規約。REQ-8）。
pub(super) fn pack_b_from_transposed(dst: &mut [f32], bt: &[f32], tile: BTPackTile) {
    let BTPackTile {
        k_total,
        n_total,
        kc_start,
        kc_len,
        col_start,
        nr,
        nr_eff,
    } = tile;
    let _ = n_total; // 呼び出し元の意図（bt の総行数）を明示するため保持。長さ検査は dst.len() で完結する。
    assert_eq!(
        dst.len(),
        kc_len * nr,
        "pack_b_from_transposed: dst 長さが kc_len*nr と不一致"
    );
    if nr_eff < nr {
        dst.fill(0.0);
    }
    for j in 0..nr_eff {
        let src_row = &bt
            [(col_start + j) * k_total + kc_start..(col_start + j) * k_total + kc_start + kc_len];
        for (p, &val) in src_row.iter().enumerate() {
            dst[p * nr + j] = val;
        }
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

    /// `at`（m_total x k_total を転置した行優先 [k_total, m_total]）から
    /// `pack_a_from_transposed` した結果が、`a`（[m_total, k_total]、
    /// `at` を素朴に転置コピーしたもの）から `pack_a` した結果と bit
    /// 完全一致することを検証する（#1213 の bit 完全一致契約の直接検証）。
    fn transpose_copy(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        // src は [cols, rows]（行優先）。戻り値は [rows, cols]（行優先）。
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..cols {
            for c in 0..rows {
                out[c * cols + r] = src[r * rows + c];
            }
        }
        out
    }

    #[test]
    fn pack_a_from_transposed_full_tile_matches_pack_a_of_contiguous_copy() {
        // at: [k_total=3, m_total=2] 行優先 → a（転置コピー）: [2,3]。
        let at = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = transpose_copy(&at, 2, 3);
        let mut dst_t = vec![0.0f32; 2 * 3];
        pack_a_from_transposed(
            &mut dst_t,
            &at,
            ATPackTile {
                k_total: 3,
                m_total: 2,
                row_start: 0,
                mr: 2,
                mr_eff: 2,
                kc_start: 0,
                kc_len: 3,
            },
        );
        let mut dst_ref = vec![0.0f32; 2 * 3];
        pack_a(
            &mut dst_ref,
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
        assert_eq!(dst_t, dst_ref);
    }

    #[test]
    fn pack_a_from_transposed_edge_tile_zero_pads_unused_rows() {
        let at = vec![1.0, 2.0, 3.0, 4.0]; // k_total=2, m_total=2
        let a = transpose_copy(&at, 2, 2);
        let mut dst_t = vec![9.9f32; 4 * 2];
        pack_a_from_transposed(
            &mut dst_t,
            &at,
            ATPackTile {
                k_total: 2,
                m_total: 2,
                row_start: 0,
                mr: 4,
                mr_eff: 2,
                kc_start: 0,
                kc_len: 2,
            },
        );
        let mut dst_ref = vec![9.9f32; 4 * 2];
        pack_a(
            &mut dst_ref,
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
        assert_eq!(dst_t, dst_ref);
    }

    #[test]
    fn pack_b_from_transposed_full_tile_matches_pack_b_of_contiguous_copy() {
        // bt: [n_total=3, k_total=2] 行優先 → b（転置コピー）: [2,3]。
        let bt = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = transpose_copy(&bt, 2, 3);
        let mut dst_t = vec![0.0f32; 2 * 3];
        pack_b_from_transposed(
            &mut dst_t,
            &bt,
            BTPackTile {
                k_total: 2,
                n_total: 3,
                kc_start: 0,
                kc_len: 2,
                col_start: 0,
                nr: 3,
                nr_eff: 3,
            },
        );
        let mut dst_ref = vec![0.0f32; 2 * 3];
        pack_b(
            &mut dst_ref,
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
        assert_eq!(dst_t, dst_ref);
    }

    #[test]
    fn pack_b_from_transposed_edge_tile_zero_pads_unused_cols() {
        let bt = vec![1.0, 2.0, 3.0, 4.0]; // n_total=2, k_total=2
        let b = transpose_copy(&bt, 2, 2);
        let mut dst_t = vec![9.9f32; 2 * 4];
        pack_b_from_transposed(
            &mut dst_t,
            &bt,
            BTPackTile {
                k_total: 2,
                n_total: 2,
                kc_start: 0,
                kc_len: 2,
                col_start: 0,
                nr: 4,
                nr_eff: 2,
            },
        );
        let mut dst_ref = vec![9.9f32; 2 * 4];
        pack_b(
            &mut dst_ref,
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
        assert_eq!(dst_t, dst_ref);
    }
}
