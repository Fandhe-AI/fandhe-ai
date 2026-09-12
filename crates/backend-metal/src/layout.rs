//! GEMM 入口へ渡す 2 次元ビューの分類・先頭次元 collapse（イシュー #1040）。
//!
//! `crate::gemm::MetalGemm` の strided 入口（`dispatch_strided_bias_act_prepared`）
//! が受け取る `MatrixLayout`（転置有無・leading dimension）を、
//! `fandhe_ai_tensor_core::Tensor::shape`/`strides` から純粋に導出する。
//! `objc2` 系 FFI に一切触れないため、`pad`/`tile` と同じ設計判断で
//! `cfg(target_os = "macos")` を付けず Linux（本実装環境・CI）でも
//! 単体テストが回るようにしてある（`lib.rs` の cfg 境界方針参照）。
//!
//! 背景: 学習ループの VJP（`crates/autodiff/src/grad.rs` の
//! `Op::LinearResident` 分岐）は `transpose2d(upstream)` で作った転置
//! view を `ops.rs::MetalBackendOps::gemm_resident_lhs` へ渡すが、従来は
//! `Tensor::contiguous()` でホスト側の転置コピー（repack）を経由していた。
//! 本モジュールは、行優先／列優先いずれの 2 次元 view も
//! `classify_2d` で判別し、転置コピーなしに GPU カーネルへ渡せる形
//! （`ld`・`transposed` フラグ）へ変換する。`[B, …, M, K]` のような
//! 先頭次元も `collapse_leading_dims` で `[B*…*M, K]` へ畳み、rank-2 GEMM
//! 入口をそのまま再利用できるようにする（バッチ matmul の公開 API 化は
//! 別イシュー。本モジュールは非公開の内部ヘルパに留める）。

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

/// `ops::MetalBackendOps::gemm_fp32_strict_into_with_bias_reduce_tracked`
/// が weight（GEMM 出力 `[m, n]`）と bias（縮約結果 `[n]`）を同一 `out`
/// バッファへ書く前に呼ぶ、範囲検証の純関数版（PR #1659 codex-review
/// P1 是正）。`ops.rs` は `cfg(target_os = "macos")` だが本関数は
/// `objc2` 系 FFI に触れない純粋な範囲計算のため、`pad`/`tile`/本
/// モジュールの他関数と同じ設計判断で cfg を付けず Linux（本実装
/// 環境・CI）でも単体テストが回るようにしてある。
///
/// 検証項目（いずれも `checked_mul`/`checked_add` でオーバーフロー
/// 安全）:
/// 1. `m * n` のオーバーフロー、`out_offset + m*n` のオーバーフロー・
///    `out_numel` 超過（weight 書き込み範囲）
/// 2. `bias` が `Some((bias_offset, bn))` の場合、`bn == n`・
///    `bias_offset + bn` のオーバーフロー・`out_numel` 超過（bias
///    書き込み範囲）
/// 3. **weight 範囲 `[out_offset, out_offset+m*n)` と bias 範囲
///    `[bias_offset, bias_offset+bn)` の重複禁止**（従来欠落していた
///    検証。例: `a` が単位行列・`out_offset=0`・`bias=Some((0, n))` は
///    weight・bias 両範囲が `[0, n)` で完全に重なり、bias の書き込みが
///    weight の一部を無言で上書きする。`m == 0` または `n == 0`
///    （空次元）は範囲自体が空集合になるため重複判定から除外する）。
///
/// 呼び出し元は返り値の `Err` をそのまま伝播すればよい（`BackendError::
/// InvalidArgument`）。
pub fn validate_gemm_bias_write_ranges(
    m: usize,
    n: usize,
    out_offset: usize,
    out_numel: usize,
    bias: Option<(usize, usize)>,
) -> Result<(), fandhe_ai_tensor_core::device::BackendError> {
    use fandhe_ai_tensor_core::device::BackendError;

    let mn = m.checked_mul(n).ok_or_else(|| {
        BackendError::InvalidArgument(
            "validate_gemm_bias_write_ranges: m * n overflowed usize".into(),
        )
    })?;
    let end = out_offset.checked_add(mn).ok_or_else(|| {
        BackendError::InvalidArgument(
            "validate_gemm_bias_write_ranges: out_offset + m * n overflowed usize".into(),
        )
    })?;
    if end > out_numel {
        return Err(BackendError::InvalidArgument(format!(
            "validate_gemm_bias_write_ranges: weight write range [{out_offset}, {end}) exceeds \
             out buffer length {out_numel}"
        )));
    }
    let Some((bias_offset, bn)) = bias else {
        return Ok(());
    };
    if bn != n {
        return Err(BackendError::InvalidArgument(format!(
            "validate_gemm_bias_write_ranges: bias n ({bn}) does not match GEMM n ({n})"
        )));
    }
    let bias_end = bias_offset.checked_add(bn).ok_or_else(|| {
        BackendError::InvalidArgument(
            "validate_gemm_bias_write_ranges: bias_offset + n overflowed usize".into(),
        )
    })?;
    if bias_end > out_numel {
        return Err(BackendError::InvalidArgument(format!(
            "validate_gemm_bias_write_ranges: bias write range [{bias_offset}, {bias_end}) \
             exceeds out buffer length {out_numel}"
        )));
    }
    // 半開区間 [out_offset, end) と [bias_offset, bias_end) の重複判定。
    // どちらかが空（mn == 0 または bn == 0）なら重複しようがない。
    if mn > 0 && bn > 0 && out_offset < bias_end && bias_offset < end {
        return Err(BackendError::InvalidArgument(format!(
            "validate_gemm_bias_write_ranges: weight write range [{out_offset}, {end}) \
             overlaps bias write range [{bias_offset}, {bias_end})"
        )));
    }
    Ok(())
}

/// bias 勾配（`Op::LinearResident` の VJP における `g` の行方向和）の
/// ホスト参照実装（イシュー #1566）。GPU カーネル
/// `shaders/gemm.metal::gemm_bias_grad_reduce_f32` の正しさを検証する
/// 基準、および `ops::MetalBackendOps::gemm_fp32_strict_into_with_
/// bias_reduce_tracked` の NN/TT・分類不能形状フォールバック経路
/// （GPU ディスパッチを経由しない）の両方から使う。
///
/// `autodiff::eval::reduce_bias_grad_rows`（`grad::reduce_to_shape` の
/// rank-2→rank-1 特殊ケース）と**アルゴリズム的に同一**（行 `0..rows`
/// 昇順・初期値 `0.0f32`・単純な `+=`）だが、`backend-metal` は
/// `autodiff` に依存できない（クレート依存方向: `autodiff` →
/// `backend-metal` の逆方向はない）ため独立実装する。両実装の bit
/// 完全一致は実機 `#[ignore]` テスト（`docs/backend-metal-command-
/// batching-design.md` §10 実装記録参照）で確認する。
///
/// `data` は `g` を [`classify_2d`] で分類した [`MatrixLayout`]
/// （`rows`/`cols`/`ld`/`transposed`）が示す添字式（本モジュール冒頭
/// doc「添字式」参照）に従って読む。
///
/// 呼び出し前に [`required_span`]（`checked_mul`/`checked_add` による
/// オーバーフロー安全な最小バッファ長算出。`crate::gemm::
/// validate_strided_dims` と同じ関数を再利用）で `data.len()` が
/// レイアウトの要求範囲を満たすことを検証し、満たさない場合は
/// `Err(BackendError::InvalidArgument)` を返す（PR #1659 codex-review
/// P1 是正: 従来は `debug_assert!` 契約違反検知＋release ゼロ埋め
/// フォールバックだったが、`pub fn` として任意の `data`/`layout` の
/// 組み合わせを受理しうる以上、境界外読み取り・添字乗算の
/// オーバーフローを事前検証で遮断する方が安全。検証を通過した後は
/// ループ中のあらゆる `idx` が `required_span` 未満に収まることが
/// `required_span` 自体の算出式から保証されるため、追加のオーバー
/// フローチェックなしに直接インデックスできる）。`.claude/rules/
/// coding-rust.md`「本番経路で `unwrap()`/`expect()` を使わない」は
/// 型付き `Result` エラーで満たす。
pub fn reduce_bias_grad_rows_host(
    data: &[f32],
    layout: &MatrixLayout,
) -> Result<Vec<f32>, fandhe_ai_tensor_core::device::BackendError> {
    let required = required_span(layout).ok_or_else(|| {
        fandhe_ai_tensor_core::device::BackendError::InvalidArgument(
            "reduce_bias_grad_rows_host: layout の添字計算が usize をオーバーフローする".into(),
        )
    })?;
    if data.len() < required {
        return Err(
            fandhe_ai_tensor_core::device::BackendError::InvalidArgument(format!(
                "reduce_bias_grad_rows_host: data の長さ {} が layout の要求範囲 {required} を満たさない",
                data.len()
            )),
        );
    }
    let (rows, cols, ld, transposed) = (layout.rows, layout.cols, layout.ld, layout.transposed);
    let mut out = vec![0f32; cols];
    for row in 0..rows {
        for (col, acc) in out.iter_mut().enumerate() {
            let idx = if transposed {
                col * ld + row
            } else {
                row * ld + col
            };
            *acc += data[idx];
        }
    }
    Ok(out)
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

    // イシュー #1566: `reduce_bias_grad_rows_host` の回帰テスト
    // （`autodiff::eval::reduce_bias_grad_rows` と同じ順序依存ケース。
    // クレート依存方向〈`backend-metal` は `autodiff` に依存できない〉
    // のため直接突合はできず、同一アルゴリズム〈行 0..rows 昇順・f32
    // 逐次 `+=`〉であることを手計算した期待値との一致で独立に確認する）。
    #[test]
    fn reduce_bias_grad_rows_host_contiguous_matches_manual_add_order() {
        // g: [3, 2]（行優先 contiguous）。列 0 は 1e8 + 1.0 + -1e8 が
        // 昇順加算だと桁落ちで 1.0 の寄与が失われる順序依存ケース
        // （`autodiff::eval` の同種テストと同じ意図）。
        let data = vec![1.0e8, 10.0, 1.0, 20.0, -1.0e8, 30.0];
        let layout = MatrixLayout {
            rows: 3,
            cols: 2,
            ld: 2,
            transposed: false,
        };
        let got = reduce_bias_grad_rows_host(&data, &layout).expect("valid layout/data in test");

        let mut expected_col0 = 0.0f32;
        expected_col0 += 1.0e8;
        expected_col0 += 1.0;
        expected_col0 += -1.0e8;
        let mut expected_col1 = 0.0f32;
        expected_col1 += 10.0;
        expected_col1 += 20.0;
        expected_col1 += 30.0;

        assert_eq!(got.len(), 2);
        assert_eq!(got[0].to_bits(), expected_col0.to_bits());
        assert_eq!(got[1].to_bits(), expected_col1.to_bits());
        assert_eq!(
            expected_col0, 0.0,
            "桁落ちにより 1.0 の寄与が失われることの確認"
        );
    }

    #[test]
    fn reduce_bias_grad_rows_host_transposed_view_matches_logical_shape() {
        // `g` が列優先 view（転置）の場合、`(row, col)` は
        // `data[col * ld + row]` で読む（本モジュール冒頭 doc「添字式」）。
        // 論理形状 [2, 3]（rows=2, cols=3）・実データは
        // `data[col*ld+row]`（ld=2）で `g[r][c] = data[c*2+r]` となる
        // 行列を手で構成する: g = [[1,3,5],[2,4,6]]。
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let layout = MatrixLayout {
            rows: 2,
            cols: 3,
            ld: 2,
            transposed: true,
        };
        let got = reduce_bias_grad_rows_host(&data, &layout).expect("valid layout/data in test");
        // 列ごとの和: col0=1+2=3・col1=3+4=7・col2=5+6=11。
        assert_eq!(got, vec![3.0f32, 7.0, 11.0]);
    }

    #[test]
    fn reduce_bias_grad_rows_host_preserves_nan_and_inf() {
        let data = vec![-0.0, f32::NAN, f32::INFINITY, 1.0, -0.0, f32::NEG_INFINITY];
        let layout = MatrixLayout {
            rows: 3,
            cols: 2,
            ld: 2,
            transposed: false,
        };
        let got = reduce_bias_grad_rows_host(&data, &layout).expect("valid layout/data in test");
        // row0=[-0.0, NaN]・row1=[+inf, 1.0]・row2=[-0.0, -inf]。
        // col0: -0.0 + (+inf) + -0.0 = +inf
        assert!(got[0].is_infinite() && got[0] > 0.0);
        // col1: NaN + 1.0 + -inf = NaN
        assert!(got[1].is_nan());
    }

    // PR #1659 codex-review P1 是正の回帰テスト: 空 `data` と最小 `MatrixLayout`
    // （codex-review 指摘の具体例そのもの）を渡すと、契約違反検知が debug
    // ビルドの panic ではなく型付き `Err(BackendError::InvalidArgument)` に
    // なることを確認する（`required_span` によるオーバーフロー安全な事前
    // 検証。本モジュール冒頭の `reduce_bias_grad_rows_host` doc 参照）。
    #[test]
    fn reduce_bias_grad_rows_host_rejects_insufficient_data_instead_of_panicking() {
        let layout = MatrixLayout {
            rows: 1,
            cols: 1,
            ld: 1,
            transposed: false,
        };
        let err = reduce_bias_grad_rows_host(&[], &layout)
            .expect_err("空 data は required_span 未満のため Err のはず");
        assert!(matches!(
            err,
            fandhe_ai_tensor_core::device::BackendError::InvalidArgument(_)
        ));
    }

    #[test]
    fn reduce_bias_grad_rows_host_rejects_overflowing_layout() {
        // `required_span` が `checked_mul`/`checked_add` でオーバーフローを
        // 検知するケース（`rows`/`ld` が usize::MAX 近傍）。
        let layout = MatrixLayout {
            rows: usize::MAX,
            cols: 2,
            ld: usize::MAX,
            transposed: false,
        };
        let err = reduce_bias_grad_rows_host(&[1.0, 2.0], &layout)
            .expect_err("layout の添字計算がオーバーフローするため Err のはず");
        assert!(matches!(
            err,
            fandhe_ai_tensor_core::device::BackendError::InvalidArgument(_)
        ));
    }

    // PR #1659 codex-review P1 是正の回帰テスト（`validate_gemm_bias_write_
    // ranges`）: weight・bias 書き込み範囲の重複検出。

    #[test]
    fn validate_gemm_bias_write_ranges_rejects_full_overlap() {
        // codex-review 指摘の具体例: a=単位行列(2x2)・b=[[1,2],[3,4]]・
        // out_offset=0・bias=Some((0,2))。m=n=2 なので weight は [0,4)・
        // bias は [0,2) で完全に重なる。
        let err = validate_gemm_bias_write_ranges(2, 2, 0, 8, Some((0, 2)))
            .expect_err("weight・bias 範囲が重複するため Err のはず");
        assert!(matches!(
            err,
            fandhe_ai_tensor_core::device::BackendError::InvalidArgument(_)
        ));
    }

    #[test]
    fn validate_gemm_bias_write_ranges_rejects_partial_overlap() {
        // weight [4, 8)・bias [6, 8) は末尾側で部分的に重なる。
        let err = validate_gemm_bias_write_ranges(2, 2, 4, 8, Some((6, 2)))
            .expect_err("weight・bias 範囲が部分的に重複するため Err のはず");
        assert!(matches!(
            err,
            fandhe_ai_tensor_core::device::BackendError::InvalidArgument(_)
        ));
    }

    #[test]
    fn validate_gemm_bias_write_ranges_accepts_disjoint_ranges() {
        // weight [0, 4)・bias [4, 6) は互いに素（隣接するだけで重ならない）。
        validate_gemm_bias_write_ranges(2, 2, 0, 6, Some((4, 2)))
            .expect("互いに素な範囲は受理されるはず");
    }

    #[test]
    fn validate_gemm_bias_write_ranges_accepts_zero_rows_without_false_overlap() {
        // m == 0（空次元）は weight 範囲が空集合になるため、bias 範囲が
        // 数値上重なって見えても実際には書き込みが発生せず重複ではない。
        validate_gemm_bias_write_ranges(0, 2, 0, 8, Some((0, 2)))
            .expect("m == 0 は空範囲のため重複判定の対象外のはず");
    }

    #[test]
    fn validate_gemm_bias_write_ranges_accepts_no_bias() {
        validate_gemm_bias_write_ranges(2, 2, 0, 4, None)
            .expect("bias なしなら weight 範囲検査のみ通れば受理されるはず");
    }

    #[test]
    fn validate_gemm_bias_write_ranges_rejects_out_of_bounds_weight() {
        let err = validate_gemm_bias_write_ranges(2, 2, 0, 3, None)
            .expect_err("out_numel を超える weight 範囲は Err のはず");
        assert!(matches!(
            err,
            fandhe_ai_tensor_core::device::BackendError::InvalidArgument(_)
        ));
    }

    #[test]
    fn validate_gemm_bias_write_ranges_rejects_bias_n_mismatch() {
        let err = validate_gemm_bias_write_ranges(2, 2, 0, 8, Some((4, 3)))
            .expect_err("bias の n が GEMM の n と不一致のため Err のはず");
        assert!(matches!(
            err,
            fandhe_ai_tensor_core::device::BackendError::InvalidArgument(_)
        ));
    }
}
