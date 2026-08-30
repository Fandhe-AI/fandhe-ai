//! GEMM 入口へ渡す 2 次元ビューの分類・先頭次元 collapse（イシュー #1040・
//! tensor-core への移設はイシュー #1046）。
//!
//! `backend-metal::gemm::MetalGemm` の strided 入口
//! （`dispatch_strided_bias_act_prepared`）が受け取る `MatrixLayout`
//! （転置有無・leading dimension）を、`Tensor::shape`/`strides` から
//! 純粋に導出する（`objc2` 等 FFI に一切触れない純粋関数のみで構成）。
//!
//! 元は `backend-metal` 専用モジュールだったが（#1040）、`autodiff::eval`
//! の `matmul`（VJP のホスト側転置コピー除去。#1046）が同じ分類ロジック
//! を必要としたため `tensor-core` へ移設した。`backend-metal` 側は
//! `crate::layout` を本モジュールの再エクスポートへ縮約している
//! （呼び出し元 `gemm.rs`・`ops.rs`・既存テストのパスは変更不要）。
//!
//! 背景: 学習ループの VJP（`crates/autodiff/src/grad.rs` の
//! `matmul_vjp`／`Op::LinearResident` 分岐）は `transpose2d(...)` で作った
//! 転置 view を渡すが、従来は `contiguous()` でホスト側の転置コピー
//! （repack）を経由していた。本モジュールは、行優先／列優先いずれの
//! 2 次元 view も `classify_2d` で判別し、転置コピーなしに下流（GPU
//! カーネル・`autodiff::eval::matmul` の添字計算）へ渡せる形（`ld`・
//! `transposed` フラグ）へ変換する。`[B, …, M, K]` のような先頭次元も
//! `collapse_leading_dims` で `[B*…*M, K]` へ畳み、rank-2 GEMM 入口を
//! そのまま再利用できるようにする（バッチ matmul の公開 API 化は別
//! イシュー。本モジュールは `facade` 非公開の内部ヘルパに留める。
//! `docs/compat-api-scope.md` §0）。

/// 転置パターン（NN/NT/TN/TT）。`#1037`（タイル構成のテーブル駆動選択）が
/// タイル表のキーとして利用できるよう `Copy + Eq + Hash` にしてある。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransposePattern {
    /// A・B ともに行優先 contiguous（既存 `gemm_simdgroup_tiled` 高速経路）。
    Nn,
    /// A は行優先、B は転置 view（列優先）。
    Nt,
    /// A は転置 view（列優先）、B は行優先。
    Tn,
    /// A・B ともに転置 view。
    Tt,
}

impl TransposePattern {
    /// A・B それぞれの `transposed` フラグから合成する。
    pub fn from_flags(trans_a: bool, trans_b: bool) -> Self {
        match (trans_a, trans_b) {
            (false, false) => TransposePattern::Nn,
            (false, true) => TransposePattern::Nt,
            (true, false) => TransposePattern::Tn,
            (true, true) => TransposePattern::Tt,
        }
    }
}

/// GEMM カーネルへ渡す 1 オペランド分の 2 次元レイアウト。
///
/// `rows`/`cols` は論理形状（転置前の意味論、すなわち呼び出し元が
/// 期待する `[rows, cols]` の行列としての形）。`ld`（leading dimension）は
/// 実データ上で 1 行（`transposed == false`）または 1 列
/// （`transposed == true`）分進めたときの要素ストライドで、
/// `shaders/gemm.metal::gemm_tiled_bias_act` の `GemmStrides` に
/// そのまま渡る。
///
/// 添字式（`crate::gemm` の `validate_strided_dims`・`gemm.metal` の
/// 添字計算と一致させる契約）:
/// - `transposed == false`（行優先）: 要素 `(r, c)` は `data[r * ld + c]`
/// - `transposed == true`（列優先 = 転置 view）: 要素 `(r, c)` は
///   `data[c * ld + r]`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixLayout {
    pub rows: usize,
    pub cols: usize,
    pub ld: usize,
    pub transposed: bool,
}

/// `shape`/`strides`（rank-2）を [`MatrixLayout`] へ分類する。
///
/// - 行優先 contiguous（`strides == [ld, 1]` かつ `ld >= cols`）→
///   `transposed = false`
/// - 列優先（転置 view。`strides == [1, ld]` かつ `ld >= rows`）→
///   `transposed = true`
/// - 上記いずれでもない（stride 0 のブロードキャスト・負 stride・
///   rank != 2 等）→ `None`（呼び出し元は従来の `contiguous()` へ
///   フォールバックする）
///
/// `rows == 0 || cols == 0`（空次元）は `ld` の下限検査を満たせないため
/// 一律 `None` とする（呼び出し元の 0 次元縮退分岐に委ねる）。
pub fn classify_2d(shape: &[usize], strides: &[isize]) -> Option<MatrixLayout> {
    if shape.len() != 2 || strides.len() != 2 {
        return None;
    }
    let (rows, cols) = (shape[0], shape[1]);
    if rows == 0 || cols == 0 {
        return None;
    }
    let (sr, sc) = (strides[0], strides[1]);
    if sr <= 0 || sc <= 0 {
        return None;
    }
    let (sr, sc) = (sr as usize, sc as usize);

    if sc == 1 && sr >= cols {
        return Some(MatrixLayout {
            rows,
            cols,
            ld: sr,
            transposed: false,
        });
    }
    if sr == 1 && sc >= rows {
        return Some(MatrixLayout {
            rows,
            cols,
            ld: sc,
            transposed: true,
        });
    }
    None
}

/// `[B0, …, M, K]`（rank >= 2）の先頭次元を行次元へ畳み、
/// `[B0*…*M, K]` の [`MatrixLayout`] を返す（candle の collapse 条件と
/// 同種: 各先頭軸の stride が「直後の軸の shape × stride」に一致し、
/// かつ末尾軸の stride が 1 であるときのみ畳める）。
///
/// - rank < 2、または末尾軸の stride != 1 の場合は `None`
/// - 各先頭軸（`shape[..rank-1]`）のうちいずれかで
///   `strides[i] != shape[i+1] as isize * strides[i+1]` が成り立たない
///   （collapse 不能な非連続 view）場合は `None`
/// - 要素数オーバーフロー（`checked_mul`）時は `None`
///
/// 戻り値の `ld = strides[rank - 2]`（畳んだ後の行 stride）、
/// `transposed = false`（collapse 結果は常に行優先解釈で表現する。
/// 末尾 2 軸自体が転置 view であるケースは `collapse_leading_dims` の
/// 対象外——呼び出し元が `classify_2d` で別途判定する）。
pub fn collapse_leading_dims(shape: &[usize], strides: &[isize]) -> Option<MatrixLayout> {
    let rank = shape.len();
    if rank < 2 || strides.len() != rank {
        return None;
    }
    let k = shape[rank - 1];
    if strides[rank - 1] != 1 {
        return None;
    }
    // 先頭 rank-1 軸すべて（バッチ軸 + 行軸）が「直後の軸の shape × stride」
    // と一致することを要求する（連続 view であることの必要十分条件。
    // candle `Layout::collapse` と同種の判定）。
    for i in 0..rank - 1 {
        let expected = (shape[i + 1] as isize).checked_mul(strides[i + 1])?;
        if strides[i] != expected {
            return None;
        }
    }
    let m: usize = shape[..rank - 1]
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))?;
    if m == 0 || k == 0 {
        return None;
    }
    let ld = strides[rank - 2];
    if ld <= 0 {
        return None;
    }
    Some(MatrixLayout {
        rows: m,
        cols: k,
        ld: ld as usize,
        transposed: false,
    })
}

/// [`MatrixLayout`] が要求する最小バッファ長（`offset` を含まない、
/// レイアウト自体が必要とする要素数）を返す。
///
/// - `transposed == false`: `(rows - 1) * ld + cols`
/// - `transposed == true`: `(cols - 1) * ld + rows`
///
/// いずれも `checked_mul`/`checked_add` を用い、オーバーフロー時は
/// `None` を返す（呼び出し元 `crate::gemm::validate_strided_dims` の
/// fail-closed なバッファ範囲検証に使う）。
pub fn required_span(layout: &MatrixLayout) -> Option<usize> {
    let (major, minor) = if layout.transposed {
        (layout.cols, layout.rows)
    } else {
        (layout.rows, layout.cols)
    };
    if major == 0 {
        return Some(0);
    }
    major
        .checked_sub(1)?
        .checked_mul(layout.ld)?
        .checked_add(minor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_row_major_contiguous() {
        let layout = classify_2d(&[3, 4], &[4, 1]).unwrap();
        assert_eq!(
            layout,
            MatrixLayout {
                rows: 3,
                cols: 4,
                ld: 4,
                transposed: false,
            }
        );
    }

    #[test]
    fn classify_row_major_padded_ld() {
        // ld > cols（末尾にパディング列を持つ行優先バッファ）も受理する。
        let layout = classify_2d(&[3, 4], &[8, 1]).unwrap();
        assert_eq!(layout.ld, 8);
        assert!(!layout.transposed);
    }

    #[test]
    fn classify_col_major_transposed() {
        // `transpose2d` された view: 元 [4,3] 行優先を transpose すると
        // shape=[3,4]・strides=[1,4]（列優先）になる。
        let layout = classify_2d(&[3, 4], &[1, 4]).unwrap();
        assert_eq!(
            layout,
            MatrixLayout {
                rows: 3,
                cols: 4,
                ld: 4,
                transposed: true,
            }
        );
    }

    #[test]
    fn classify_rejects_broadcast_zero_stride() {
        assert_eq!(classify_2d(&[3, 4], &[0, 1]), None);
    }

    #[test]
    fn classify_rejects_negative_stride() {
        assert_eq!(classify_2d(&[3, 4], &[-4, 1]), None);
    }

    #[test]
    fn classify_rejects_non_rank2() {
        assert_eq!(classify_2d(&[3, 4, 5], &[20, 5, 1]), None);
        assert_eq!(classify_2d(&[3], &[1]), None);
    }

    #[test]
    fn classify_rejects_zero_dim() {
        assert_eq!(classify_2d(&[0, 4], &[4, 1]), None);
        assert_eq!(classify_2d(&[3, 0], &[0, 1]), None);
    }

    #[test]
    fn classify_rejects_unsupported_stride_combo() {
        // どちらの軸も stride 1 でない・かつどちらも ld 条件を満たさない。
        assert_eq!(classify_2d(&[3, 4], &[5, 2]), None);
    }

    #[test]
    fn collapse_rank2_contiguous() {
        let layout = collapse_leading_dims(&[3, 4], &[4, 1]).unwrap();
        assert_eq!(
            layout,
            MatrixLayout {
                rows: 3,
                cols: 4,
                ld: 4,
                transposed: false,
            }
        );
    }

    #[test]
    fn collapse_rank3_batch() {
        // [B=2, M=3, K=4] 行優先 contiguous → [6, 4]
        let layout = collapse_leading_dims(&[2, 3, 4], &[12, 4, 1]).unwrap();
        assert_eq!(
            layout,
            MatrixLayout {
                rows: 6,
                cols: 4,
                ld: 4,
                transposed: false,
            }
        );
    }

    #[test]
    fn collapse_rank4_batch() {
        // [B0=2, B1=3, M=5, K=7] 行優先 contiguous → [30, 7]
        let shape = [2usize, 3, 5, 7];
        let strides = [3isize * 5 * 7, 5 * 7, 7, 1];
        let layout = collapse_leading_dims(&shape, &strides).unwrap();
        assert_eq!(layout.rows, 30);
        assert_eq!(layout.cols, 7);
        assert_eq!(layout.ld, 7);
        assert!(!layout.transposed);
    }

    #[test]
    fn collapse_rejects_non_contiguous_batch_gap() {
        // バッチ軸間に隙間があり collapse 条件（stride[i] == shape[i+1]*stride[i+1]）
        // を満たさない（narrow 等による view）。
        let layout = collapse_leading_dims(&[2, 3, 4], &[16, 4, 1]);
        assert_eq!(layout, None);
    }

    #[test]
    fn collapse_rejects_trailing_stride_not_one() {
        // 末尾軸が転置 view（stride != 1）の場合は collapse 対象外。
        assert_eq!(collapse_leading_dims(&[2, 3, 4], &[12, 1, 3]), None);
    }

    #[test]
    fn collapse_rejects_rank_below_2() {
        assert_eq!(collapse_leading_dims(&[4], &[1]), None);
        assert_eq!(collapse_leading_dims(&[], &[]), None);
    }

    #[test]
    fn collapse_rejects_zero_dim() {
        assert_eq!(collapse_leading_dims(&[0, 3, 4], &[12, 4, 1]), None);
        assert_eq!(collapse_leading_dims(&[2, 3, 0], &[0, 0, 1]), None);
    }

    #[test]
    fn collapse_rejects_element_count_overflow() {
        // shape 積が usize をオーバーフローするケース（`try_fold` の
        // `checked_mul` が `None` を返す）。
        let shape = [usize::MAX, 2, 4];
        let strides = [8isize, 4, 1];
        assert_eq!(collapse_leading_dims(&shape, &strides), None);
    }

    #[test]
    fn required_span_row_major() {
        let layout = MatrixLayout {
            rows: 3,
            cols: 4,
            ld: 8,
            transposed: false,
        };
        assert_eq!(required_span(&layout), Some(2 * 8 + 4));
    }

    #[test]
    fn required_span_transposed() {
        let layout = MatrixLayout {
            rows: 3,
            cols: 4,
            ld: 6,
            transposed: true,
        };
        assert_eq!(required_span(&layout), Some(3 * 6 + 3));
    }

    #[test]
    fn required_span_zero_major_is_zero() {
        let layout = MatrixLayout {
            rows: 0,
            cols: 4,
            ld: 4,
            transposed: false,
        };
        assert_eq!(required_span(&layout), Some(0));
    }

    #[test]
    fn required_span_overflow_is_none() {
        let layout = MatrixLayout {
            rows: usize::MAX,
            cols: 4,
            ld: usize::MAX,
            transposed: false,
        };
        assert_eq!(required_span(&layout), None);
    }

    /// NN/NT/TN/TT の 4 パターンで、`crate::gemm` の strided 添字式
    /// （`gemm.metal::gemm_tiled_bias_act` に実装する式と同一）を
    /// このモジュール内の純粋関数として再現し、素朴な稠密参照実装との
    /// 数値一致を検証する（advisor 指摘: Mac 実機なしでは検証できない
    /// MSL の添字ロジックを、Linux で先に固定するための回帰テスト）。
    fn a_at(data: &[f32], layout: &MatrixLayout, row: usize, kk: usize) -> f32 {
        if layout.transposed {
            data[kk * layout.ld + row]
        } else {
            data[row * layout.ld + kk]
        }
    }

    fn b_at(data: &[f32], layout: &MatrixLayout, kk: usize, col: usize) -> f32 {
        if layout.transposed {
            data[col * layout.ld + kk]
        } else {
            data[kk * layout.ld + col]
        }
    }

    fn strided_gemm_reference(
        a: &[f32],
        a_layout: &MatrixLayout,
        b: &[f32],
        b_layout: &MatrixLayout,
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc = a_at(a, a_layout, row, kk).mul_add(b_at(b, b_layout, kk, col), acc);
                }
                out[row * n + col] = acc;
            }
        }
        out
    }

    fn dense_reference(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc = a[row * k + kk].mul_add(b[kk * n + col], acc);
                }
                out[row * n + col] = acc;
            }
        }
        out
    }

    #[test]
    fn strided_reference_matches_dense_for_all_transpose_patterns() {
        let (m, n, k) = (3usize, 5usize, 4usize);
        let a_dense: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.5 - 1.0).collect();
        let b_dense: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.25 + 0.3).collect();
        let expected = dense_reference(&a_dense, &b_dense, m, n, k);

        // NN: A・B とも行優先そのまま。
        let a_nn = classify_2d(&[m, k], &[k as isize, 1]).unwrap();
        let b_nn = classify_2d(&[k, n], &[n as isize, 1]).unwrap();
        assert_eq!(
            strided_gemm_reference(&a_dense, &a_nn, &b_dense, &b_nn, m, n, k),
            expected
        );

        // TN: A は転置 view（元 [k,m] 行優先データを転置）。
        // a_dense を [m,k] 行優先として得るには、[k,m] 行優先バッファを
        // 転置 view（strides=[1,m]）で読む必要がある。
        let mut a_km = vec![0.0f32; k * m];
        for row in 0..m {
            for kk in 0..k {
                a_km[kk * m + row] = a_dense[row * k + kk];
            }
        }
        let a_tn = classify_2d(&[m, k], &[1, m as isize]).unwrap();
        assert_eq!(
            strided_gemm_reference(&a_km, &a_tn, &b_dense, &b_nn, m, n, k),
            expected
        );

        // NT: B は転置 view（元 [n,k] 行優先データを転置）。
        let mut b_nk = vec![0.0f32; n * k];
        for kk in 0..k {
            for col in 0..n {
                b_nk[col * k + kk] = b_dense[kk * n + col];
            }
        }
        let b_nt = classify_2d(&[k, n], &[1, k as isize]).unwrap();
        assert_eq!(
            strided_gemm_reference(&a_dense, &a_nn, &b_nk, &b_nt, m, n, k),
            expected
        );

        // TT: A・B とも転置 view。
        assert_eq!(
            strided_gemm_reference(&a_km, &a_tn, &b_nk, &b_nt, m, n, k),
            expected
        );
    }

    #[test]
    fn strided_reference_matches_dense_with_padded_ld() {
        // ld > 実次元（末尾にパディング列を持つ行優先バッファ）でも
        // 添字式が正しく実データのみを参照することを確認する。
        let (m, n, k) = (2usize, 3usize, 2usize);
        let ld_a = k + 3; // A の各行にパディング 3 要素
        let mut a_padded = vec![f32::NAN; m * ld_a];
        let a_dense: Vec<f32> = (0..m * k).map(|i| i as f32 + 1.0).collect();
        for row in 0..m {
            a_padded[row * ld_a..row * ld_a + k].copy_from_slice(&a_dense[row * k..row * k + k]);
        }
        let b_dense: Vec<f32> = (0..k * n).map(|i| i as f32 * 2.0).collect();

        let a_layout = classify_2d(&[m, k], &[ld_a as isize, 1]).unwrap();
        let b_layout = classify_2d(&[k, n], &[n as isize, 1]).unwrap();
        let expected = dense_reference(&a_dense, &b_dense, m, n, k);
        assert_eq!(
            strided_gemm_reference(&a_padded, &a_layout, &b_dense, &b_layout, m, n, k),
            expected
        );
    }

    #[test]
    fn transpose_pattern_from_flags() {
        assert_eq!(
            TransposePattern::from_flags(false, false),
            TransposePattern::Nn
        );
        assert_eq!(
            TransposePattern::from_flags(false, true),
            TransposePattern::Nt
        );
        assert_eq!(
            TransposePattern::from_flags(true, false),
            TransposePattern::Tn
        );
        assert_eq!(
            TransposePattern::from_flags(true, true),
            TransposePattern::Tt
        );
    }
}
