//! 定数パディングカーネル（`torch.nn.functional.pad(mode='constant')`
//! 相当。イシュー #1756）。
//!
//! [`fandhe_ai_tensor_core::BackendOps::pad`]（`ops.rs`）の CPU 実装
//! 本体。呼び出し元（`ops.rs`）が `input.shape()`／`pads` を
//! [`fandhe_ai_tensor_core::pad_out_shape`] で再検査してから本モジュール
//! へ委譲する契約のため、本モジュール自身は呼び出し元が渡す `out_shape`
//! （検査・確定済み）をそのまま信頼し shape の再検査は行わない（`ops.rs`
//! 側の二重検査が fail-closed 境界。`.claude/rules/security.md` A08。
//! `gather_scatter.rs` モジュール doc と同型の契約）。
//!
//! 出力の各要素は「`input` 内部位置ならそのままコピー・パディング
//! 領域なら `value`」の 2 分岐のみで決まる純粋なコピー演算（算術を
//! 含まない）のため、数値契約は 3 バックエンド間 **bit 完全一致**
//! （`.claude/rules/coding-rust.md` 数値契約節参照）。`Tensor::get`
//! （境界チェック付き安全アクセス。REQ-8「境界検査を省略しない」）
//! のみを用い、`unsafe`／`unwrap`／`expect` は使わない
//! （`.claude/rules/coding-rust.md`）。非 contiguous な `input`
//! （strided view）も `Tensor::get` で正しく読める（`gather_scatter.rs`
//! と同じ方針）。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

/// 線形添字（行優先）を `shape` の多次元添字へ展開する
/// （`gather_scatter.rs::unravel` と同型の独立実装。モジュール間で
/// private 関数を共有しないため複製する）。
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

/// shape の要素数積を `checked_mul` の畳み込みで検査する
/// （`gather_scatter.rs::checked_numel` と同型。クレート内で
/// `pub(crate)` 共有できないため専用に複製する。理由は同モジュールの
/// doc を参照）。
fn checked_numel(shape: &[usize]) -> Result<usize, ShapeError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or(ShapeError::ElementCountOverflow)
}

/// [`fandhe_ai_tensor_core::BackendOps::pad`] の CPU 実装本体
/// （イシュー #1756）。`out_shape` は呼び出し元（`ops.rs`）が
/// [`fandhe_ai_tensor_core::pad_out_shape`] で検査・確定済みの出力
/// shape をそのまま渡す。`pads[axis] = (before, after)` は
/// `input.shape()` と同じ rank（呼び出し元の検査契約）。
///
/// **空テンソルの扱い**: `out_shape` がいずれかの軸で 0 を含む場合
/// （出力自体が空）は空 `Vec` を返す。`out_shape` は非空だが `input`
/// が空（`input.shape()` がいずれかの軸で 0 を含む）の場合は、出力を
/// 全域 `value` で埋めて返す（`gather` と異なり pad は「入力が空でも
/// 出力は非空になりうる」演算のため、`autodiff::eval::pad` と同じ
/// 順序でこの 2 ケースを扱う）。
pub fn pad(
    input: &Tensor<f32>,
    pads: &[(usize, usize)],
    value: f32,
    out_shape: &[usize],
) -> Result<Tensor<f32>, ShapeError> {
    let out_numel = checked_numel(out_shape)?;
    if out_numel == 0 {
        return Tensor::new(Vec::new(), out_shape);
    }
    let in_shape = input.shape();
    let in_is_empty = in_shape.contains(&0);

    let mut out = Vec::with_capacity(out_numel);
    for flat in 0..out_numel {
        let coords = unravel(flat, out_shape);
        if in_is_empty {
            out.push(value);
            continue;
        }
        let mut inside = true;
        let mut src_coords = Vec::with_capacity(coords.len());
        for (axis, &c) in coords.iter().enumerate() {
            let (before, _after) = pads[axis];
            match c.checked_sub(before) {
                Some(src_c) if src_c < in_shape[axis] => src_coords.push(src_c),
                _ => {
                    inside = false;
                    break;
                }
            }
        }
        if inside {
            let v = input.get(&src_coords);
            debug_assert!(
                v.is_some(),
                "pad: input の走査ロジックにバグがあり範囲外になった（契約違反。src_coords は \
                 直上で各軸 [0, in_shape[axis]) を検査済み）"
            );
            out.push(v.unwrap_or(value));
        } else {
            out.push(value);
        }
    }
    Tensor::new(out, out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_1d_basic() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        let pads = [(1usize, 2usize)];
        let out_shape = fandhe_ai_tensor_core::pad_out_shape(x.shape(), &pads).unwrap();
        let out = pad(&x, &pads, 0.0, &out_shape).unwrap();
        assert_eq!(out.shape(), &[6]);
        assert_eq!(
            out.contiguous().as_slice().unwrap(),
            &[0.0, 1.0, 2.0, 3.0, 0.0, 0.0]
        );
    }

    #[test]
    fn pad_2d_both_axes() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let pads = [(1usize, 0usize), (0usize, 1usize)];
        let out_shape = fandhe_ai_tensor_core::pad_out_shape(x.shape(), &pads).unwrap();
        let out = pad(&x, &pads, -1.0, &out_shape).unwrap();
        assert_eq!(out.shape(), &[3, 3]);
        assert_eq!(
            out.contiguous().as_slice().unwrap(),
            &[-1.0, -1.0, -1.0, 1.0, 2.0, -1.0, 3.0, 4.0, -1.0]
        );
    }

    #[test]
    fn pad_empty_input_fills_value() {
        let x = Tensor::new(Vec::<f32>::new(), &[0, 2]).unwrap();
        let pads = [(1usize, 0usize), (0usize, 0usize)];
        let out_shape = fandhe_ai_tensor_core::pad_out_shape(x.shape(), &pads).unwrap();
        let out = pad(&x, &pads, 7.0, &out_shape).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(out.contiguous().as_slice().unwrap(), &[7.0, 7.0]);
    }

    #[test]
    fn pad_empty_output_returns_empty() {
        let x = Tensor::new(vec![1.0f32, 2.0], &[2]).unwrap();
        let pads = [(0usize, 0usize)];
        // 出力 shape を人為的に空にするケース（input 自体は非空だが
        // 呼び出し元が既に空 out_shape を渡すケースの防御的確認）。
        let out_shape = [0usize];
        let out = pad(&x, &pads, 0.0, &out_shape).unwrap();
        assert_eq!(out.numel(), 0);
    }

    #[test]
    fn pad_noncontiguous_view_input() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        let pads = [(1usize, 0usize), (0usize, 0usize)];
        let out_shape = fandhe_ai_tensor_core::pad_out_shape(x.shape(), &pads).unwrap();
        let out = pad(&x, &pads, 0.0, &out_shape).unwrap();
        assert_eq!(out.shape(), &[4, 2]);
        assert_eq!(
            out.contiguous().as_slice().unwrap(),
            &[0.0, 0.0, 1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn pad_all_zero_pads_is_identity() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let pads = [(0usize, 0usize), (0usize, 0usize)];
        let out_shape = fandhe_ai_tensor_core::pad_out_shape(x.shape(), &pads).unwrap();
        let out = pad(&x, &pads, 0.0, &out_shape).unwrap();
        assert_eq!(
            out.contiguous().as_slice().unwrap(),
            x.contiguous().as_slice().unwrap()
        );
    }
}
