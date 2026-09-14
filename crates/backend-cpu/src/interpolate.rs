//! 最近傍リサンプリングカーネル（`torch.nn.functional.interpolate
//! (mode='nearest')` 相当。イシュー #1757）。
//!
//! [`fandhe_ai_tensor_core::BackendOps::interpolate`]（`ops.rs`）の
//! CPU 実装本体。呼び出し元（`ops.rs`）が `input.shape()`／`size` を
//! [`fandhe_ai_tensor_core::interpolate_out_shape`] で再検査してから
//! 本モジュールへ委譲する契約のため、本モジュール自身は呼び出し元が
//! 渡す `out_shape`（検査・確定済み）をそのまま信頼し shape の再検査は
//! 行わない（`constant_pad.rs`〈#1756 相当テンプレート〉・
//! `gather_scatter.rs` モジュール doc と同型の契約）。
//!
//! 出力の各要素は「対応する入力の単一要素をそのままコピーする」
//! 添字演算のみで決まる純粋なコピー演算（算術を含まない。添字式は
//! `src = (dst * in_size) / out_size`。整数除算のみで float を使わない
//! ため 3 バックエンド間で構造的に **bit 完全一致**）。`Tensor::get`
//! （境界チェック付き安全アクセス。REQ-8「境界検査を省略しない」）
//! のみを用い、`unsafe`／`unwrap`／`expect` は使わない
//! （`.claude/rules/coding-rust.md`）。非 contiguous な `input`
//! （strided view）も `Tensor::get` で正しく読める。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

/// 線形添字（行優先）を `shape` の多次元添字へ展開する
/// （`gather_scatter.rs::unravel`／`constant_pad.rs::unravel` と同型の
/// 独立実装。モジュール間で private 関数を共有しないため複製する）。
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
/// （`constant_pad.rs::checked_numel` と同型。クレート内で
/// `pub(crate)` 共有できないため専用に複製する。理由は同モジュールの
/// doc を参照）。
fn checked_numel(shape: &[usize]) -> Result<usize, ShapeError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or(ShapeError::ElementCountOverflow)
}

/// `dst`（`out_size` の範囲）に対応する `in_size` 側の添字を返す
/// （`fandhe_ai_autodiff::eval::nearest_src_coord` と同一の添字式の
/// 独立実装——`autodiff` は具体バックエンドクレートへ依存できないため
/// 重複実装する。`.claude/rules/coding-rust.md`）。`dst < out_size` より
/// 数学的に `src < in_size` が自動的に成立するが、REQ-8 の縦深防御
/// として `min(src, in_size - 1)` を明示的に取る。
///
/// 中間積 `dst * in_size` は `usize`（64bit 環境で最大約 1.8e19）を
/// 素朴な乗算で計算すると overflow しうる（例: `[1]` を
/// `broadcast_to` で `[1usize << 63]` へ拡張した view を `[3]` へ
/// 縮小する interpolate では `dst=2` の `dst * in_size` が `2^63 * 2`
/// を超える）。`u128`（最大 2^128 - 1）へ昇格して積・除算を行うこと
/// で、`dst`・`in_size` とも `usize::MAX` の場合でも `u128` の範囲に
/// 収まり overflow しない（本番経路 panic 禁止規約。イシュー #1834
/// codex-review P1 是正）。
fn nearest_src_coord(dst: usize, in_size: usize, out_size: usize) -> usize {
    if out_size == 0 || in_size == 0 {
        return 0;
    }
    let src = (dst as u128 * in_size as u128) / out_size as u128;
    // `src < in_size <= usize::MAX` が `u128` 除算の結果として保証
    // されるため（`dst < out_size` より `src < in_size`）、`as usize`
    // への縮小は安全（真の値が `usize` の範囲を超えることはない）。
    (src as usize).min(in_size - 1)
}

/// [`fandhe_ai_tensor_core::BackendOps::interpolate`]（`Nearest`）の
/// CPU 実装本体（イシュー #1757）。`out_shape` は呼び出し元
/// （`ops.rs`）が [`fandhe_ai_tensor_core::interpolate_out_shape`] で
/// 検査・確定済みの出力 shape をそのまま渡す。`spatial_start` は
/// `out_shape.len() - size.len()`（空間軸の開始位置。呼び出し元が
/// `size.len()` から導出して渡す）。
///
/// `out_shape` が要素数 0 を含む場合は空 `Vec` を返す（`interpolate_
/// out_shape` の契約により空間軸が 0 の shape は既に拒否済みだが、
/// 先頭の残り軸〈batch 等〉が 0 の場合はここで早期 return する）。
pub fn interpolate_nearest(
    input: &Tensor<f32>,
    spatial_start: usize,
    out_shape: &[usize],
) -> Result<Tensor<f32>, ShapeError> {
    let out_numel = checked_numel(out_shape)?;
    if out_numel == 0 {
        return Tensor::new(Vec::new(), out_shape);
    }
    let in_shape = input.shape();

    let mut out = Vec::with_capacity(out_numel);
    for flat in 0..out_numel {
        let coords = unravel(flat, out_shape);
        let mut src_coords = coords.clone();
        for axis in spatial_start..out_shape.len() {
            src_coords[axis] = nearest_src_coord(coords[axis], in_shape[axis], out_shape[axis]);
        }
        let v = input.get(&src_coords);
        debug_assert!(
            v.is_some(),
            "interpolate_nearest: 走査ロジックにバグがあり範囲外になった \
             （契約違反。src_coords は各軸 [0, in_shape[axis]) を検査済み）"
        );
        out.push(v.unwrap_or(0.0));
    }
    Tensor::new(out, out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn out_shape_for(in_shape: &[usize], size: &[usize]) -> Vec<usize> {
        fandhe_ai_tensor_core::interpolate_out_shape(in_shape, size)
            .expect("test fixture: 有効な shape")
    }

    #[test]
    fn interpolate_nearest_1d_upsample() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        let out_shape = out_shape_for(&[3], &[6]);
        let out = interpolate_nearest(&x, 0, &out_shape).unwrap();
        assert_eq!(out.shape(), &[6]);
        assert_eq!(
            out.contiguous().as_slice().unwrap(),
            &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        );
    }

    #[test]
    fn interpolate_nearest_1d_downsample_non_integer() {
        let x = Tensor::new((1..=8).map(|v| v as f32).collect(), &[8]).unwrap();
        let out_shape = out_shape_for(&[8], &[3]);
        let out = interpolate_nearest(&x, 0, &out_shape).unwrap();
        assert_eq!(out.shape(), &[3]);
        // src = (dst*8)/3: 0->0, 1->2, 2->5。
        assert_eq!(out.contiguous().as_slice().unwrap(), &[1.0, 3.0, 6.0]);
    }

    #[test]
    fn interpolate_nearest_2d_leading_batch_axis() {
        let x = Tensor::new(vec![1.0f32, 2.0, 10.0, 20.0], &[2, 2]).unwrap();
        let out_shape = out_shape_for(&[2, 2], &[4]);
        let out = interpolate_nearest(&x, 1, &out_shape).unwrap();
        assert_eq!(out.shape(), &[2, 4]);
        assert_eq!(
            out.contiguous().as_slice().unwrap(),
            &[1.0, 1.0, 2.0, 2.0, 10.0, 10.0, 20.0, 20.0]
        );
    }

    #[test]
    fn interpolate_nearest_identity_size_is_passthrough() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let out_shape = out_shape_for(&[4], &[4]);
        let out = interpolate_nearest(&x, 0, &out_shape).unwrap();
        assert_eq!(
            out.contiguous().as_slice().unwrap(),
            x.contiguous().as_slice().unwrap()
        );
    }

    #[test]
    fn interpolate_nearest_noncontiguous_view_input() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        // x.shape() == [3, 2]（転置 view）。末尾 1 軸のみ空間軸。
        let out_shape = out_shape_for(&[3, 2], &[4]);
        let out = interpolate_nearest(&x, 1, &out_shape).unwrap();
        assert_eq!(out.shape(), &[3, 4]);
        // 転置後の各行は [1,4]/[2,5]/[3,6]。各行 ×2 アップサンプル。
        assert_eq!(
            out.contiguous().as_slice().unwrap(),
            &[1.0, 1.0, 4.0, 4.0, 2.0, 2.0, 5.0, 5.0, 3.0, 3.0, 6.0, 6.0]
        );
    }

    #[test]
    fn interpolate_nearest_empty_leading_axis_returns_empty() {
        let x = Tensor::new(Vec::<f32>::new(), &[0, 3]).unwrap();
        let out_shape = out_shape_for(&[0, 3], &[6]);
        let out = interpolate_nearest(&x, 1, &out_shape).unwrap();
        assert_eq!(out.shape(), &[0, 6]);
        assert_eq!(out.numel(), 0);
    }

    // イシュー #1834 codex-review P1 是正: `dst * in_size` の中間積が
    // `usize` を overflow しないことの回帰テスト。

    #[test]
    fn nearest_src_coord_does_not_overflow_for_huge_in_size() {
        // `dst=2`・`in_size=2^63`・`out_size=3` は素朴な `usize` 乗算
        // （`dst * in_size = 2^64`）が overflow する組み合わせ（debug
        // ビルドでは overflow panic・release ビルドでは wrap して誤った
        // 添字を返す）。`u128` 昇格により overflow せず、数学的に正しい
        // `floor(2 * 2^63 / 3)` を返すことを確認する。
        let in_size = 1usize << 63;
        let src = nearest_src_coord(2, in_size, 3);
        let expected = ((2u128 * in_size as u128) / 3) as usize;
        assert_eq!(src, expected);
        assert!(src < in_size);
    }

    #[test]
    fn interpolate_nearest_broadcast_view_with_huge_in_size_does_not_overflow() {
        // 入力 `[1]` を `broadcast_to` で `[1usize << 63]` へ拡張した
        // stride 0 view を `[3]` へ縮小する（レビュー指摘のとおりの
        // 再現条件）。`Tensor::broadcast_to` 自体はデータを複製しない
        // ため、この巨大 in_shape でもテストは軽量に実行できる。
        let base = Tensor::new(vec![7.0f32], &[1]).unwrap();
        let huge_in_shape = 1usize << 63;
        let x = base.broadcast_to(&[huge_in_shape]).unwrap();
        let out_shape = out_shape_for(&[huge_in_shape], &[3]);
        let out = interpolate_nearest(&x, 0, &out_shape).unwrap();
        // 入力は全要素 7.0（broadcast）のため、出力もすべて 7.0。
        assert_eq!(out.contiguous().as_slice().unwrap(), &[7.0, 7.0, 7.0]);
    }
}
