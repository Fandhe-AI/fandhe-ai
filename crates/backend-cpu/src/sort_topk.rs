//! sort／topk カーネル（`torch.sort`／`torch.topk` 相当。イシュー
//! #1733）。
//!
//! [`fandhe_ai_tensor_core::BackendOps::sort`]／[`BackendOps::topk`]
//! （`ops.rs`）の CPU 実装本体。呼び出し元（`ops.rs`）が `dim`／`k` を
//! [`fandhe_ai_tensor_core::sort_out_shape`]／
//! [`fandhe_ai_tensor_core::topk_out_shape`] で再検査してから本モジュール
//! へ委譲する契約のため、本モジュール自身は `dim` が `input` の rank
//! 範囲内であることを信頼し shape の再検査は行わない（`ops.rs` 側の
//! 二重検査が fail-closed 境界。`.claude/rules/security.md` A08。
//! `gather_scatter.rs` と同じ責務分担）。
//!
//! **順序契約**（[`fandhe_ai_tensor_core::BackendOps::sort`] doc の
//! 1〜4 を正とする。後続 GPU 実装〈イシュー #1741〉が満たすべき
//! bit／parity 契約でもある）:
//! 1. **安定性**: 同値（ties）は `descending` の値に関わらず元の
//!    `dim` 軸上の添字の昇順で並ぶ（[`sort_cmp`] が明示的なタイ
//!    ブレークで実装し、「昇順ソート結果を丸ごと reverse する」実装
//!    が同値の添字順を反転させてしまう罠を避ける）。
//! 2. **NaN**: NaN は任意の非 NaN より大きい・NaN 同士は同値
//!    （1 の安定性契約に従う）。
//! 3. **±0**: `partial_cmp` は `Equal` を返すため 1 により同値扱い
//!    （添字順）。
//! 4. **決定性**: 単一スレッド逐次実装で run-to-run bit 同一の出力を
//!    返す（並列化は将来の性能最適化 issue のスコープ・
//!    `.claude/rules/out-of-scope-tracking.md` 対象。並列化する場合も
//!    上記 1〜3 の観測結果を不変に保つこと）。
//!
//! `values` は算術演算を含まない `input` の要素の並べ替え（`gather`
//! と数学的に同一）であるため、本モジュールは `fandhe_ai_autodiff::
//! eval::sort`／`eval::topk`（ホスト参照実装）と**意図的に複製**する
//! （`autodiff` は具体バックエンドクレートへ依存できないため。
//! `crates/tensor-core/tests/architecture_boundaries.rs` が機械検査
//! する層境界。実装計画「意図的複製」節参照）。
//!
//! `Tensor::get`（境界チェック付き安全アクセス。REQ-8「境界検査を
//! 省略しない」）のみを用い、`unsafe`／`unwrap`／`expect` は本番経路で
//! 使わない（`.claude/rules/coding-rust.md`）。非 contiguous な
//! `input`（strided view）も `Tensor::get` で正しく読める
//! （`gather_scatter.rs` と同じ「境界チェック付きアクセスで strided
//! 入力を安全に読む」方針を踏襲する）。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

/// 線形添字（行優先）を `shape` の多次元添字へ展開する
/// （`gather_scatter.rs::unravel` と同型の独立実装。`gather_scatter.rs`
/// 側の関数は private のためモジュール間で共有しない）。
fn unravel(mut idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut out = vec![0usize; shape.len()];
    for (axis, &d) in shape.iter().enumerate().rev() {
        if d == 0 {
            out[axis] = 0;
            continue;
        }
        out[axis] = idx % d;
        idx /= d;
    }
    out
}

/// 行優先（C-order）ストライドを計算する（`unravel` と対）。
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// 多次元添字（行優先）を線形添字へ畳み込む（`unravel` の逆演算）。
/// 出力バッファ（`Tensor::new` へ渡す `Vec`。常に行優先で新規構築
/// する）への書き込み位置を求めるために使う——`input.get` は
/// `input` 自身の（非 contiguous かもしれない）stride を内部で解決
/// するため、出力側の位置計算とは独立に必要になる。
fn ravel(coords: &[usize], strides: &[usize]) -> usize {
    coords
        .iter()
        .zip(strides.iter())
        .map(|(&c, &s)| c * s)
        .sum()
}

/// shape の要素数積を `checked_mul` の畳み込みで検査する
/// （`gather_scatter.rs::checked_numel` と同型の独立実装。
/// `pub(crate)` でクレートを跨いで共有できないため複製する）。
fn checked_numel(shape: &[usize]) -> Result<usize, ShapeError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or(ShapeError::ElementCountOverflow)
}

/// `sort`／`topk` 共通の全順序比較（`(値, 元添字)` タプル）。
/// モジュール doc の順序契約 1〜3 を実装する
/// （`fandhe_ai_autodiff::eval::sort_cmp` と同一の意図的複製）。
fn sort_cmp(a: (f32, usize), b: (f32, usize), descending: bool) -> std::cmp::Ordering {
    fn value_cmp_ascending(x: f32, y: f32) -> std::cmp::Ordering {
        match (x.is_nan(), y.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
        }
    }
    let primary = if descending {
        value_cmp_ascending(b.0, a.0)
    } else {
        value_cmp_ascending(a.0, b.0)
    };
    primary.then(a.1.cmp(&b.1))
}

/// `dim` 軸に沿ったライン（`dim` 軸以外の添字を固定した要素列）を
/// `input` から読み出す共通ヘルパー（`sort`／`topk` で共有）。
/// `coords` は `dim` 軸の値を書き換えながら再利用する（呼び出し元が
/// 事前に確保した作業バッファ）。
fn read_line(
    input: &Tensor<f32>,
    coords: &mut [usize],
    dim: usize,
    dim_size: usize,
) -> Vec<(f32, usize)> {
    let mut line = Vec::with_capacity(dim_size);
    for idx in 0..dim_size {
        coords[dim] = idx;
        let value = input.get(coords);
        debug_assert!(
            value.is_some(),
            "sort_topk: input の走査ロジックにバグがあり範囲外になった（契約違反）"
        );
        line.push((value.unwrap_or(0.0), idx));
    }
    line
}

/// [`fandhe_ai_tensor_core::BackendOps::sort`] の CPU 実装本体
/// （イシュー #1733）。出力 shape（`values`／`index` とも）は
/// `input.shape()` と恒等。
pub fn sort(
    input: &Tensor<f32>,
    dim: usize,
    descending: bool,
) -> Result<(Tensor<f32>, Tensor<i32>), ShapeError> {
    let shape = input.shape().to_vec();
    if shape.contains(&0) {
        // 空出力早期リターン（`ストライド計算前に空出力を返す`という
        // `gather_scatter.rs::scatter` と同じ方針。以下の
        // `checked_numel`／`row_major_strides` は他軸のサイズ次第で
        // オーバーフローしうるため、空 shape はこれらの計算前に弾く）。
        return Ok((
            Tensor::new(Vec::new(), &shape)?,
            Tensor::new(Vec::new(), &shape)?,
        ));
    }
    let dim_size = shape[dim];
    let numel = checked_numel(&shape)?;
    let strides = row_major_strides(&shape);
    let mut out_vals = vec![0f32; numel];
    let mut out_idx = vec![0i32; numel];

    for flat in 0..numel {
        let mut coords = unravel(flat, &shape);
        // 各ライン（`dim` 軸以外の添字が共通の要素列）は、その先頭
        // （`dim` 軸添字 0）の位置に到達したときだけ 1 回処理する。
        if coords[dim] != 0 {
            continue;
        }
        let mut line = read_line(input, &mut coords, dim, dim_size);
        line.sort_by(|&a, &b| sort_cmp(a, b, descending));
        for (out_pos, &(val, orig_idx)) in line.iter().enumerate() {
            coords[dim] = out_pos;
            let flat_out = ravel(&coords, &strides);
            out_vals[flat_out] = val;
            out_idx[flat_out] = orig_idx as i32;
        }
        // clippy: `line` は明示的にドロップしなくても scope 終了で解放
        // されるが、次イテレーションの再確保を避けるためここでは特に
        // 何もしない（`Vec::with_capacity` を毎ライン再確保する単純な
        // 実装で十分——本モジュールは正しさ優先の参照実装）。
        drop(line);
    }
    Ok((
        Tensor::new(out_vals, &shape)?,
        Tensor::new(out_idx, &shape)?,
    ))
}

/// [`fandhe_ai_tensor_core::BackendOps::topk`] の CPU 実装本体
/// （イシュー #1733・`sorted=True` 固定）。`out_shape` は呼び出し元
/// （`ops.rs`）が [`fandhe_ai_tensor_core::topk_out_shape`] で検査・
/// 確定済みの出力 shape（`dim` 軸のみ `k` に置換）をそのまま渡す。
pub fn topk(
    input: &Tensor<f32>,
    dim: usize,
    k: usize,
    largest: bool,
    out_shape: &[usize],
) -> Result<(Tensor<f32>, Tensor<i32>), ShapeError> {
    if out_shape.contains(&0) {
        return Ok((
            Tensor::new(Vec::new(), out_shape)?,
            Tensor::new(Vec::new(), out_shape)?,
        ));
    }
    let in_shape = input.shape().to_vec();
    let dim_size = in_shape[dim];
    let in_numel = checked_numel(&in_shape)?;
    let out_numel = checked_numel(out_shape)?;
    let out_strides = row_major_strides(out_shape);
    let mut out_vals = vec![0f32; out_numel];
    let mut out_idx = vec![0i32; out_numel];

    for flat in 0..in_numel {
        let mut coords = unravel(flat, &in_shape);
        if coords[dim] != 0 {
            continue;
        }
        let mut line = read_line(input, &mut coords, dim, dim_size);
        line.sort_by(|&a, &b| sort_cmp(a, b, largest));
        for (out_pos, &(val, orig_idx)) in line.iter().take(k).enumerate() {
            coords[dim] = out_pos;
            let flat_out = ravel(&coords, &out_strides);
            out_vals[flat_out] = val;
            out_idx[flat_out] = orig_idx as i32;
        }
    }
    Ok((
        Tensor::new(out_vals, out_shape)?,
        Tensor::new(out_idx, out_shape)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
    }

    fn dense_i32(tensor: &Tensor<i32>) -> Vec<i32> {
        tensor
            .contiguous()
            .as_slice()
            .expect("test fixture: contiguous() 後は必ず as_slice() が Some")
            .to_vec()
    }

    fn dense_f32(tensor: &Tensor<f32>) -> Vec<f32> {
        tensor
            .contiguous()
            .as_slice()
            .expect("test fixture: contiguous() 後は必ず as_slice() が Some")
            .to_vec()
    }

    #[test]
    fn sort_ascending_basic() {
        let x = t(vec![3.0, 1.0, 4.0, 1.5], &[1, 4]);
        let (out, index) = sort(&x, 1, false).unwrap();
        assert_eq!(dense_f32(&out), vec![1.0, 1.5, 3.0, 4.0]);
        assert_eq!(dense_i32(&index), vec![1, 3, 0, 2]);
    }

    #[test]
    fn sort_descending_ties_preserve_index_ascending() {
        let x = t(vec![2.0, 1.0, 1.0, 2.0], &[1, 4]);
        let (out, index) = sort(&x, 1, true).unwrap();
        assert_eq!(dense_f32(&out), vec![2.0, 2.0, 1.0, 1.0]);
        // 単純な reverse() なら {3, 0} になってしまう箇所が {0, 3} で
        // 維持されることを確認する（順序契約 1）。
        assert_eq!(dense_i32(&index), vec![0, 3, 1, 2]);
    }

    #[test]
    fn sort_nan_is_largest_and_ties_among_nan() {
        let x = t(vec![f32::NAN, 1.0, f32::NAN, 0.0], &[1, 4]);
        let (out, index) = sort(&x, 1, false).unwrap();
        let vals = dense_f32(&out);
        assert_eq!(&vals[..2], &[0.0, 1.0]);
        assert!(vals[2].is_nan() && vals[3].is_nan());
        assert_eq!(dense_i32(&index), vec![3, 1, 0, 2]);
    }

    #[test]
    fn sort_negative_and_positive_zero_are_tied() {
        let x = t(vec![1.0, -0.0, 0.0], &[1, 3]);
        let (out, index) = sort(&x, 1, false).unwrap();
        assert_eq!(dense_f32(&out), vec![-0.0, 0.0, 1.0]);
        assert_eq!(dense_i32(&index), vec![1, 2, 0]);
    }

    #[test]
    fn sort_dim0_basic() {
        // shape [3, 1]・dim=0。
        let x = t(vec![3.0, 1.0, 2.0], &[3, 1]);
        let (out, index) = sort(&x, 0, false).unwrap();
        assert_eq!(dense_f32(&out), vec![1.0, 2.0, 3.0]);
        assert_eq!(dense_i32(&index), vec![1, 2, 0]);
    }

    #[test]
    fn sort_empty_dim_returns_empty() {
        let x = t(Vec::new(), &[1, 0]);
        let (out, index) = sort(&x, 1, false).unwrap();
        assert_eq!(out.shape(), &[1, 0]);
        assert_eq!(dense_f32(&out), Vec::<f32>::new());
        assert_eq!(dense_i32(&index), Vec::<i32>::new());
    }

    #[test]
    fn sort_non_contiguous_transposed_view() {
        // [2, 2] を transpose した非 contiguous view を dim=1 で sort。
        let x = t(vec![3.0, 1.0, 4.0, 1.5], &[2, 2]);
        let xt = x.transpose(0, 1).expect("transpose は常に成功する形状");
        let (out, index) = sort(&xt, 1, false).unwrap();
        // xt = [[3.0, 4.0], [1.0, 1.5]]（transpose 後の論理形状）。
        assert_eq!(dense_f32(&out), vec![3.0, 4.0, 1.0, 1.5]);
        assert_eq!(dense_i32(&index), vec![0, 1, 0, 1]);
    }

    #[test]
    fn topk_largest_basic() {
        let x = t(vec![3.0, 1.0, 4.0, 1.5], &[1, 4]);
        let (out, index) = topk(&x, 1, 2, true, &[1, 2]).unwrap();
        assert_eq!(dense_f32(&out), vec![4.0, 3.0]);
        assert_eq!(dense_i32(&index), vec![2, 0]);
    }

    #[test]
    fn topk_smallest_basic() {
        let x = t(vec![3.0, 1.0, 4.0, 1.5], &[1, 4]);
        let (out, index) = topk(&x, 1, 2, false, &[1, 2]).unwrap();
        assert_eq!(dense_f32(&out), vec![1.0, 1.5]);
        assert_eq!(dense_i32(&index), vec![1, 3]);
    }

    #[test]
    fn topk_k_equals_dim_size_matches_sort() {
        let x = t(vec![3.0, 1.0, 4.0, 1.5], &[1, 4]);
        let (topk_out, topk_index) = topk(&x, 1, 4, true, &[1, 4]).unwrap();
        let (sort_out, sort_index) = sort(&x, 1, true).unwrap();
        assert_eq!(dense_f32(&topk_out), dense_f32(&sort_out));
        assert_eq!(dense_i32(&topk_index), dense_i32(&sort_index));
    }

    #[test]
    fn topk_k_zero_returns_empty() {
        let x = t(vec![3.0, 1.0, 4.0, 1.5], &[1, 4]);
        let (out, index) = topk(&x, 1, 0, true, &[1, 0]).unwrap();
        assert_eq!(out.shape(), &[1, 0]);
        assert_eq!(dense_f32(&out), Vec::<f32>::new());
        assert_eq!(dense_i32(&index), Vec::<i32>::new());
    }

    #[test]
    fn topk_run_to_run_is_bit_identical() {
        let x = t(vec![3.0, 1.0, 4.0, 1.5, 2.0, 2.0], &[1, 6]);
        let (out1, index1) = topk(&x, 1, 3, true, &[1, 3]).unwrap();
        let (out2, index2) = topk(&x, 1, 3, true, &[1, 3]).unwrap();
        assert_eq!(
            dense_f32(&out1)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            dense_f32(&out2)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
        );
        assert_eq!(dense_i32(&index1), dense_i32(&index2));
    }
}
