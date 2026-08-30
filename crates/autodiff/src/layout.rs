//! `eval::matmul` が転置 view をゼロコピーで扱うための 2 次元レイアウト
//! 分類（イシュー #1046）。クレート非公開（`facade` の公開 API 面には
//! 出さない。`docs/compat-api-scope.md` §0）。
//!
//! `crates/backend-metal/src/layout.rs` の `classify_2d`/`MatrixLayout`
//! と同一の分類規則（双子モジュール）——共有契約は
//! `crates/backend-metal/src/shaders/gemm.metal` の添字計算
//! （`gemm_tiled_bias_act`）であり、両モジュールはこのシェーダの
//! 添字式と一致させる。一度は `tensor-core::layout` へ集約したが
//! （#1046 初版）、`fandhe-ai-tensor-core`（crates.io 公開クレート）の
//! 公開面へ内部レイアウト型を露出させてしまう（codex-review P1・
//! AGENTS.md「内部表現の公開 API への漏出は P1」）ため、PR #1077 で
//! 各利用クレート内に閉じる方式へ差し戻した。`autodiff` が実際に
//! 使うのは `classify_2d`/`MatrixLayout` のみ（`collapse_leading_dims`・
//! `required_span`・`TransposePattern` は `backend-metal` 側のみが必要と
//! するため複製しない。`clippy -D warnings` の `dead_code` を避ける）。
//!
//! 背景: `crates/autodiff/src/grad.rs` の VJP（`matmul_vjp`・
//! `Op::LinearResident` 分岐）は `transpose2d(...)` で作った転置 view を
//! `eval::matmul` へ渡すが、従来は `contiguous()` でホスト側の転置
//! コピー（repack）を経由していた。本モジュールは、行優先／列優先
//! いずれの 2 次元 view も `classify_2d` で判別し、転置コピーなしに
//! `eval::matmul` の添字計算へ渡せる形（`ld`・`transposed` フラグ）へ
//! 変換する。

/// GEMM カーネルへ渡す 1 オペランド分の 2 次元レイアウト。
///
/// `rows`/`cols` は論理形状（転置前の意味論、すなわち呼び出し元が
/// 期待する `[rows, cols]` の行列としての形）。`ld`（leading dimension）は
/// 実データ上で 1 行（`transposed == false`）または 1 列
/// （`transposed == true`）分進めたときの要素ストライドである。
///
/// 添字式（`crate::eval::matmul` の添字計算・`backend-metal::layout`
/// と一致させる契約）:
/// - `transposed == false`（行優先）: 要素 `(r, c)` は `data[r * ld + c]`
/// - `transposed == true`（列優先 = 転置 view）: 要素 `(r, c)` は
///   `data[c * ld + r]`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MatrixLayout {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) ld: usize,
    pub(crate) transposed: bool,
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
pub(crate) fn classify_2d(shape: &[usize], strides: &[isize]) -> Option<MatrixLayout> {
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
}
