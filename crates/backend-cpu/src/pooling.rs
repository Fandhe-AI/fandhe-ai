//! MaxPool／AvgPool／AdaptiveAvgPool（2d）の CPU 参照実装（イシュー
//! #1728・設計 `docs/pooling-ops-design.md`）。
//!
//! [`fandhe_ai_tensor_core::BackendOps::max_pool2d`]／`avg_pool2d`／
//! `adaptive_avg_pool2d`（`ops.rs`）の CPU 実装本体。呼び出し元
//! （`ops.rs`）が `input.shape()`／`params` を
//! [`fandhe_ai_tensor_core::pool2d_out_shape`]／
//! `adaptive_pool2d_out_shape` で再検査してから本モジュールへ委譲する
//! 契約のため、本モジュール自身は呼び出し元が渡す `out_shape`
//! （検査・確定済み）をそのまま信頼し shape の再検査は行わない
//! （`im2col.rs` モジュール doc と同型の契約。二重検査境界は `ops.rs`
//! 側が担う。`.claude/rules/security.md` A08）。
//!
//! 単一スレッド逐次実装（`gather_scatter.rs` の規律を踏襲。並列化は
//! out-of-scope。設計 doc §10「CPU: 参照実装」）。`Tensor::get`
//! （stride 対応・境界チェック付き安全アクセス）のみを用い、
//! `unsafe`／`unwrap`／`expect` は使わない。
//!
//! **数値契約**（設計 doc §5／§7）:
//! - [`max_pool2d`] は純粋な選択演算（丸めなし）のため 3 バックエンド
//!   **bit 完全一致**（値・索引とも）が受入基準。NaN 伝播・先勝ち
//!   タイ規則は同関数 doc comment を正とする。
//! - [`avg_pool2d`]／[`adaptive_avg_pool2d`] は窓内を row-major で
//!   `f64` アキュムレータへ逐次加算し最後に 1 回だけ `f32` へ
//!   downcast する（`.claude/rules/coding-rust.md` の勾配の長軸縮約
//!   規約と同型の数値契約）。

use fandhe_ai_tensor_core::{Pool2dParams, ShapeError, Tensor, adaptive_window};

/// 窓添字 `k` から入力座標を符号安全に逆算する（`im2col.rs::
/// im2col_input_pos` と同型。`pos = out_idx*stride + k*dilation -
/// padding` を `checked_*` のみで計算し、`padding` を超える減算は
/// アンダーフローとして `None` を返す）。
fn window_input_pos(
    out_idx: usize,
    stride: usize,
    k: usize,
    dilation: usize,
    padding: usize,
) -> Option<usize> {
    let base = out_idx.checked_mul(stride)?;
    let offset = k.checked_mul(dilation)?;
    let sum = base.checked_add(offset)?;
    sum.checked_sub(padding)
}

/// [`fandhe_ai_tensor_core::BackendOps::max_pool2d`] の CPU 実装本体
/// （イシュー #1728）。`out_shape` は呼び出し元（`ops.rs`）が
/// [`fandhe_ai_tensor_core::pool2d_out_shape`] で検査・確定済みの
/// `[N, C, Hout, Wout]` をそのまま渡す。
///
/// タイ規則・NaN 規則（設計 doc §5）: 窓内を `kh` 外側・`kw` 内側の
/// row-major で走査し `v > best || (v.is_nan() && !best.is_nan())` の
/// ときのみ更新する（先勝ち決定的・NaN 伝播・最初の NaN の索引が
/// 残る）。padding 位置は走査対象外（`pool2d_out_shape` の空窓拒否
/// 検査により全出力窓が少なくとも 1 つの有効入力位置を含むことが
/// 保証されているため、`best` が `None` のまま走査を終える経路は
/// 契約違反として `ShapeError::ElementCountOverflow` を返す。パニック
/// しない）。索引は `(n, c)` 平面内 flat 添字 `h·W + w`。
pub fn max_pool2d(
    input: &Tensor<f32>,
    params: &Pool2dParams,
    out_shape: &[usize],
) -> Result<(Tensor<f32>, Tensor<i32>), ShapeError> {
    let out_numel: usize = out_shape.iter().product();
    if out_numel == 0 {
        return Ok((
            Tensor::new(Vec::new(), out_shape)?,
            Tensor::new(Vec::new(), out_shape)?,
        ));
    }
    let in_shape = input.shape();
    let (h_in, w_in) = (in_shape[2], in_shape[3]);
    let (n_batch, c_ch, h_out, w_out) = (out_shape[0], out_shape[1], out_shape[2], out_shape[3]);
    let [kh, kw] = params.kernel_size();
    let [sh, sw] = params.stride();
    let [ph, pw] = params.padding();
    let [dh, dw] = params.dilation();

    let mut out_vals = vec![0f32; out_numel];
    let mut out_idx = vec![0i32; out_numel];
    for n in 0..n_batch {
        for c in 0..c_ch {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut best: Option<(f32, usize)> = None;
                    for kh_ in 0..kh {
                        let Some(h) = window_input_pos(oh, sh, kh_, dh, ph).filter(|&h| h < h_in)
                        else {
                            continue;
                        };
                        for kw_ in 0..kw {
                            let Some(w) =
                                window_input_pos(ow, sw, kw_, dw, pw).filter(|&w| w < w_in)
                            else {
                                continue;
                            };
                            let v = input.get(&[n, c, h, w]).ok_or_else(|| {
                                ShapeError::ShapeMismatch {
                                    lhs: vec![n, c, h, w],
                                    rhs: in_shape.to_vec(),
                                }
                            })?;
                            let flat = h * w_in + w;
                            best = Some(match best {
                                None => (v, flat),
                                Some((b, bi)) => {
                                    if v > b || (v.is_nan() && !b.is_nan()) {
                                        (v, flat)
                                    } else {
                                        (b, bi)
                                    }
                                }
                            });
                        }
                    }
                    // 契約違反（空窓）の安全側フォールバック: `pool2d_out_shape`
                    // の空窓拒否検査がすべての出力窓に対し成立している限り
                    // 到達しない分岐だが、呼び出し元の検査漏れがあっても
                    // panic せず型付きエラーで拒否する（REQ-8・A08）。
                    let (v, idx) = best.ok_or(ShapeError::ElementCountOverflow)?;
                    let out_pos = ((n * c_ch + c) * h_out + oh) * w_out + ow;
                    out_vals[out_pos] = v;
                    out_idx[out_pos] = i32::try_from(idx)
                        .map_err(|_| ShapeError::IndexRangeOverflow { index: idx })?;
                }
            }
        }
    }
    Ok((
        Tensor::new(out_vals, out_shape)?,
        Tensor::new(out_idx, out_shape)?,
    ))
}

/// [`fandhe_ai_tensor_core::BackendOps::avg_pool2d`] の CPU 実装本体
/// （イシュー #1728）。`out_shape` は呼び出し元が
/// [`fandhe_ai_tensor_core::pool2d_out_shape`] で検査・確定済みの
/// `[N, C, Hout, Wout]`。
///
/// `count_include_pad`: `true` は divisor 常に `kh·kw`（`ceil_mode=
/// false` 固定のため窓は padded 境界を超えない前提。設計 doc §5
/// 「AvgPool の divisor」）・`false` は有効要素数。窓内は row-major
/// で `f64` へ逐次加算し最後に 1 回 `f32` へ downcast（モジュール doc
/// 参照）。
pub fn avg_pool2d(
    input: &Tensor<f32>,
    params: &Pool2dParams,
    count_include_pad: bool,
    out_shape: &[usize],
) -> Result<Tensor<f32>, ShapeError> {
    let out_numel: usize = out_shape.iter().product();
    if out_numel == 0 {
        return Tensor::new(Vec::new(), out_shape);
    }
    let in_shape = input.shape();
    let (h_in, w_in) = (in_shape[2], in_shape[3]);
    let (n_batch, c_ch, h_out, w_out) = (out_shape[0], out_shape[1], out_shape[2], out_shape[3]);
    let [kh, kw] = params.kernel_size();
    let [sh, sw] = params.stride();
    let [ph, pw] = params.padding();
    // avg_pool2d は `Var::avg_pool2d` が常に dilation=[1,1] で
    // `Pool2dParams` を構築する契約（設計 doc §2「Avg 系は
    // dilation=[1,1] 固定」）だが、本関数自体は任意の dilation で
    // 正しく動作する一般実装のまま保つ（`Pool2dParams` に検査済みの
    // 値が入っている前提を再確認しない）。
    let [dh, dw] = params.dilation();

    let mut out = vec![0f32; out_numel];
    for n in 0..n_batch {
        for c in 0..c_ch {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc: f64 = 0.0;
                    let mut count: usize = 0;
                    for kh_ in 0..kh {
                        let Some(h) = window_input_pos(oh, sh, kh_, dh, ph).filter(|&h| h < h_in)
                        else {
                            continue;
                        };
                        for kw_ in 0..kw {
                            let Some(w) =
                                window_input_pos(ow, sw, kw_, dw, pw).filter(|&w| w < w_in)
                            else {
                                continue;
                            };
                            let v = input.get(&[n, c, h, w]).ok_or_else(|| {
                                ShapeError::ShapeMismatch {
                                    lhs: vec![n, c, h, w],
                                    rhs: in_shape.to_vec(),
                                }
                            })?;
                            acc += f64::from(v);
                            count += 1;
                        }
                    }
                    let divisor = if count_include_pad {
                        kh.checked_mul(kw).ok_or(ShapeError::ElementCountOverflow)?
                    } else {
                        count
                    };
                    // divisor == 0 は `pool2d_out_shape` の空窓拒否検査が
                    // 成立している限り到達しないが、`max_pool2d` と同じ
                    // 安全側フォールバックとして型付きエラーを返す。
                    if divisor == 0 {
                        return Err(ShapeError::ElementCountOverflow);
                    }
                    let v = (acc / divisor as f64) as f32;
                    let out_pos = ((n * c_ch + c) * h_out + oh) * w_out + ow;
                    out[out_pos] = v;
                }
            }
        }
    }
    Tensor::new(out, out_shape)
}

/// [`fandhe_ai_tensor_core::BackendOps::adaptive_avg_pool2d`] の CPU
/// 実装本体（イシュー #1728）。`out_shape` は呼び出し元が
/// [`fandhe_ai_tensor_core::adaptive_pool2d_out_shape`] で検査・確定
/// 済みの `[N, C, Hout, Wout]`。
///
/// 出力位置ごとの窓は [`adaptive_window`]（forward／VJP 共有の単一
/// 情報源。設計 doc §4）が定める `[start, end)`。divisor は常に実際の
/// 窓要素数（`count_include_pad=true` 相当）。縮約の数値契約は
/// [`avg_pool2d`] と同一。
pub fn adaptive_avg_pool2d(
    input: &Tensor<f32>,
    out_shape: &[usize],
) -> Result<Tensor<f32>, ShapeError> {
    let out_numel: usize = out_shape.iter().product();
    if out_numel == 0 {
        return Tensor::new(Vec::new(), out_shape);
    }
    let in_shape = input.shape();
    let (h_in, w_in) = (in_shape[2], in_shape[3]);
    let (n_batch, c_ch, h_out, w_out) = (out_shape[0], out_shape[1], out_shape[2], out_shape[3]);

    let mut out = vec![0f32; out_numel];
    for n in 0..n_batch {
        for c in 0..c_ch {
            for oh in 0..h_out {
                let (h_start, h_end) =
                    adaptive_window(oh, h_in, h_out).ok_or(ShapeError::ElementCountOverflow)?;
                for ow in 0..w_out {
                    let (w_start, w_end) =
                        adaptive_window(ow, w_in, w_out).ok_or(ShapeError::ElementCountOverflow)?;
                    let mut acc: f64 = 0.0;
                    let mut count: usize = 0;
                    for h in h_start..h_end {
                        for w in w_start..w_end {
                            let v = input.get(&[n, c, h, w]).ok_or_else(|| {
                                ShapeError::ShapeMismatch {
                                    lhs: vec![n, c, h, w],
                                    rhs: in_shape.to_vec(),
                                }
                            })?;
                            acc += f64::from(v);
                            count += 1;
                        }
                    }
                    // adaptive_pool2d_out_shape が H/W >= 1・output_size
                    // >= 1 を検査済みのため窓は常に非空。到達しない分岐
                    // だが安全側フォールバックとして型付きエラーを返す。
                    if count == 0 {
                        return Err(ShapeError::ElementCountOverflow);
                    }
                    let v = (acc / count as f64) as f32;
                    let out_pos = ((n * c_ch + c) * h_out + oh) * w_out + ow;
                    out[out_pos] = v;
                }
            }
        }
    }
    Tensor::new(out, out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_tensor_core::pool2d_out_shape;

    fn params(kernel: [usize; 2], stride: Option<[usize; 2]>, padding: [usize; 2]) -> Pool2dParams {
        Pool2dParams::new(kernel, stride, padding, [1, 1]).unwrap()
    }

    #[test]
    fn max_pool2d_tie_first_match_wins() {
        // 全要素同値の窓 -> 索引は窓先頭（flat 0）。
        let input = Tensor::new(vec![5.0; 16], &[1, 1, 4, 4]).unwrap();
        let p = params([2, 2], None, [0, 0]);
        let out_shape = pool2d_out_shape(&[1, 1, 4, 4], &p).unwrap();
        let (values, index) = max_pool2d(&input, &p, &out_shape).unwrap();
        assert_eq!(values.host_slice(), vec![5.0; 4]);
        // 窓 (0,0): 左上要素 flat=0。
        assert_eq!(index.get(&[0, 0, 0, 0]), Some(0));
    }

    #[test]
    fn max_pool2d_nan_propagates_and_keeps_first_nan_index() {
        // 窓内に NaN が 1 つあれば出力は NaN・索引はその NaN の flat 添字。
        let data = vec![1.0, f32::NAN, 2.0, 3.0];
        let input = Tensor::new(data, &[1, 1, 2, 2]).unwrap();
        let p = params([2, 2], None, [0, 0]);
        let out_shape = pool2d_out_shape(&[1, 1, 2, 2], &p).unwrap();
        let (values, index) = max_pool2d(&input, &p, &out_shape).unwrap();
        assert!(values.get(&[0, 0, 0, 0]).unwrap().is_nan());
        // row-major 走査で NaN は flat=1（(0,1)）に最初に現れる。
        assert_eq!(index.get(&[0, 0, 0, 0]), Some(1));
    }

    #[test]
    fn max_pool2d_padding_never_wins() {
        // padding 位置は走査対象外のため索引は常に [0, H*W) の実入力範囲。
        let input = Tensor::new(vec![-1.0, -2.0, -3.0, -4.0], &[1, 1, 2, 2]).unwrap();
        let p = params([2, 2], Some([1, 1]), [1, 1]);
        let out_shape = pool2d_out_shape(&[1, 1, 2, 2], &p).unwrap();
        let (_values, index) = max_pool2d(&input, &p, &out_shape).unwrap();
        for &i in index.host_slice().iter() {
            assert!((0..4).contains(&i), "index {i} は [0, H*W) の範囲外");
        }
    }

    #[test]
    fn max_pool2d_overlapping_windows_scan_order() {
        // stride < kernel の重なり窓でも各窓独立に先勝ち規則が適用される。
        let input = Tensor::new(vec![1.0, 3.0, 2.0, 5.0, 4.0, 0.0], &[1, 1, 2, 3]).unwrap();
        let p = params([2, 2], Some([1, 1]), [0, 0]);
        let out_shape = pool2d_out_shape(&[1, 1, 2, 3], &p).unwrap();
        let (values, _index) = max_pool2d(&input, &p, &out_shape).unwrap();
        // 窓 (0,0)=[1,3,5,4]->5, 窓 (0,1)=[3,2,4,0]->4
        assert_eq!(values.host_slice(), vec![5.0, 4.0]);
    }

    #[test]
    fn max_pool2d_batch_zero_yields_empty_output() {
        let input = Tensor::new(Vec::new(), &[0, 1, 4, 4]).unwrap();
        let p = params([2, 2], None, [0, 0]);
        let out_shape = pool2d_out_shape(&[0, 1, 4, 4], &p).unwrap();
        let (values, index) = max_pool2d(&input, &p, &out_shape).unwrap();
        assert_eq!(values.shape(), &[0, 1, 2, 2]);
        assert_eq!(index.shape(), &[0, 1, 2, 2]);
    }

    #[test]
    fn max_pool2d_non_contiguous_input_matches_contiguous() {
        // transpose(2,3) の非 contiguous view でも `Tensor::get` の
        // stride 対応読み取りにより contiguous() 版と bit 一致する。
        let input =
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 1, 2, 4]).unwrap();
        let transposed = input.transpose(2, 3).unwrap(); // [1,1,4,2]（非 contiguous）
        let contiguous = transposed.contiguous();
        let p = params([2, 2], None, [0, 0]);
        let out_shape = pool2d_out_shape(&[1, 1, 4, 2], &p).unwrap();
        let (v1, i1) = max_pool2d(&transposed, &p, &out_shape).unwrap();
        let (v2, i2) = max_pool2d(&contiguous, &p, &out_shape).unwrap();
        assert_eq!(v1.host_slice(), v2.host_slice());
        assert_eq!(i1.host_slice(), i2.host_slice());
    }

    #[test]
    fn avg_pool2d_f64_accumulation_fixed_order() {
        // 相殺列（f32 逐次和では丸め誤差が生じる代表例）を f64
        // アキュムレータで縮約することを確認する。
        let data = vec![1.0e8_f32, 1.0_f32, -1.0e8_f32, 1.0_f32];
        let input = Tensor::new(data, &[1, 1, 2, 2]).unwrap();
        let p = params([2, 2], None, [0, 0]);
        let out_shape = pool2d_out_shape(&[1, 1, 2, 2], &p).unwrap();
        let out = avg_pool2d(&input, &p, true, &out_shape).unwrap();
        let sum: f64 = 1.0e8 + 1.0 + -1.0e8 + 1.0;
        let expected = (sum / 4.0) as f32;
        assert_eq!(out.host_slice(), vec![expected]);
    }

    #[test]
    fn avg_pool2d_count_include_pad_true_uses_kernel_area() {
        let input = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], &[1, 1, 2, 2]).unwrap();
        let p = params([2, 2], Some([2, 2]), [1, 1]);
        let out_shape = pool2d_out_shape(&[1, 1, 2, 2], &p).unwrap();
        let out = avg_pool2d(&input, &p, true, &out_shape).unwrap();
        // 左上窓は入力の (0,0) 要素のみ有効・divisor=kh*kw=4。
        assert_eq!(out.get(&[0, 0, 0, 0]), Some(2.0 / 4.0));
    }

    #[test]
    fn avg_pool2d_count_include_pad_false_uses_valid_count() {
        let input = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], &[1, 1, 2, 2]).unwrap();
        let p = params([2, 2], Some([2, 2]), [1, 1]);
        let out_shape = pool2d_out_shape(&[1, 1, 2, 2], &p).unwrap();
        let out = avg_pool2d(&input, &p, false, &out_shape).unwrap();
        // 左上窓は入力の (0,0) 要素のみ有効・divisor=1。
        assert_eq!(out.get(&[0, 0, 0, 0]), Some(2.0));
    }

    #[test]
    fn avg_pool2d_overlapping_windows() {
        let input = Tensor::new(vec![1.0, 3.0, 2.0, 5.0, 4.0, 0.0], &[1, 1, 2, 3]).unwrap();
        let p = params([2, 2], Some([1, 1]), [0, 0]);
        let out_shape = pool2d_out_shape(&[1, 1, 2, 3], &p).unwrap();
        let out = avg_pool2d(&input, &p, true, &out_shape).unwrap();
        // 窓 (0,0)=[1,3,5,4] 平均 3.25, 窓 (0,1)=[3,2,4,0] 平均 2.25
        assert_eq!(out.host_slice(), vec![3.25, 2.25]);
    }

    #[test]
    fn adaptive_avg_pool2d_shrink() {
        let input = Tensor::new((1..=16).map(|v| v as f32).collect(), &[1, 1, 4, 4]).unwrap();
        let out_shape =
            fandhe_ai_tensor_core::adaptive_pool2d_out_shape(&[1, 1, 4, 4], [2, 2]).unwrap();
        let out = adaptive_avg_pool2d(&input, &out_shape).unwrap();
        // 左上窓 [0,2)x[0,2) = {1,2,5,6} 平均 3.5。
        assert_eq!(out.get(&[0, 0, 0, 0]), Some(3.5));
    }

    #[test]
    fn adaptive_avg_pool2d_expand() {
        // out > in（拡大側）: 各出力位置が単一入力要素を指す。
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
        let out_shape =
            fandhe_ai_tensor_core::adaptive_pool2d_out_shape(&[1, 1, 2, 2], [4, 4]).unwrap();
        let out = adaptive_avg_pool2d(&input, &out_shape).unwrap();
        assert_eq!(out.get(&[0, 0, 0, 0]), Some(1.0));
        assert_eq!(out.get(&[0, 0, 3, 3]), Some(4.0));
    }

    #[test]
    fn adaptive_avg_pool2d_non_divisible_matches_reference() {
        // in=7, out=2 の非割り切れ窓（[0,4)/[3,7) 相当）を手計算値と突合。
        let data: Vec<f32> = (0..7).map(|v| v as f32).collect();
        let input = Tensor::new(data.clone(), &[1, 1, 1, 7]).unwrap();
        let out_shape =
            fandhe_ai_tensor_core::adaptive_pool2d_out_shape(&[1, 1, 1, 7], [1, 2]).unwrap();
        let out = adaptive_avg_pool2d(&input, &out_shape).unwrap();
        let first: f64 = data[0..4].iter().map(|&v| v as f64).sum::<f64>() / 4.0;
        let second: f64 = data[3..7].iter().map(|&v| v as f64).sum::<f64>() / 4.0;
        assert_eq!(out.host_slice(), vec![first as f32, second as f32]);
    }
}
