//! Conv2d の im2col／col2im カーネル（`docs/conv-ops-design.md`。
//! イシュー #1764）。
//!
//! [`fandhe_ai_tensor_core::BackendOps::im2col`]／[`BackendOps::col2im`]
//! （`ops.rs`）の CPU 実装本体。呼び出し元（`ops.rs`）が
//! `input.shape()`／`params` を [`fandhe_ai_tensor_core::im2col_out_shape`]
//! で再検査してから本モジュールへ委譲する契約のため、本モジュール自身は
//! 呼び出し元が渡す `out_shape`（検査・確定済み）をそのまま信頼し shape
//! の再検査は行わない（`ops.rs` 側の二重検査が fail-closed 境界。
//! `.claude/rules/security.md` A08。`constant_pad.rs` モジュール doc と
//! 同型の契約）。
//!
//! `conv2d` 自身は override しない（既定 `Unsupported` のまま）。CPU の
//! forward は `autodiff::grad::conv2d_with_fallback` の段階的合成
//! （`im2col` → `CpuBackendOps::gemm_batched`〈BLIS〉→ `add`）を使う。
//!
//! **数値契約**: [`im2col`] は「`input` 内部位置ならそのままコピー・
//! padding 位置なら `0.0`」の 2 分岐のみで決まる純粋なコピー演算
//! （算術を含まない）のため **bit 完全一致**（`constant_pad.rs` と同型）。
//! [`col2im`] は重なり窓（`stride < dilation·(kernel−1)+1`）の加算順を
//! `(kh, kw)` row-major に固定し **`f64` アキュムレータへ逐次加算・
//! 最後に 1 回 `f32` へ downcast**する（`.claude/rules/coding-rust.md`
//! の勾配の長軸縮約規約。3 バックエンド bit 完全一致契約）。
//!
//! 座標計算（`h + p_h − kh·d_h` 等）は `usize` の通常減算では負値へ
//! アンダーフローしうる（`panic`／ラップアラウンド）ため、
//! `checked_mul`／`checked_add`／`checked_sub` のみを用いる
//! （設計 doc §6.2「訂正 2」。REQ-8「境界検査を省略しない」）。
//! `Tensor::get`（境界チェック付き安全アクセス）のみを用い、`unsafe`／
//! `unwrap`／`expect` は使わない。

use fandhe_ai_tensor_core::{Conv2dParams, ShapeError, Tensor, conv_out_len};

/// shape の要素数積を `checked_mul` の畳み込みで検査する
/// （`constant_pad.rs::checked_numel` と同型。クレート内で `pub(crate)`
/// 共有できないため専用に複製する。理由は同モジュールの doc を参照）。
fn checked_numel(shape: &[usize]) -> Result<usize, ShapeError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or(ShapeError::ElementCountOverflow)
}

/// im2col の走査で使う「窓添字 → 入力座標」の符号安全な逆変換
/// （col2im 側の「入力座標 → 窓添字」と対をなす。`Some(pos)` は
/// `pos < in_len` のときのみ有効入力位置、それ以外（`None` を含む）は
/// padding 位置として `0.0` を書く）。
///
/// `pos = out_idx * stride + k * dilation − padding`（`checked_*` のみ
/// で計算し、`padding` を超える減算はアンダーフローとして `None` を
/// 返す）。
fn im2col_input_pos(
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

/// col2im の走査で使う「入力座標 → 窓添字」の符号安全な逆変換
/// （設計 doc §6.2）。`numerator = pos + padding − k * dilation` を
/// `checked_*` のみで計算し、`stride` で割り切れ商が `out_len` 未満の
/// ときのみ `Some(out_idx)` を返す。それ以外（アンダーフロー・割り切れ
/// ない・範囲外）は寄与なし（`None`）として扱う。
fn col2im_out_idx(
    pos: usize,
    padding: usize,
    k: usize,
    dilation: usize,
    stride: usize,
    out_len: usize,
) -> Option<usize> {
    let offset = k.checked_mul(dilation)?;
    let pos_plus_p = pos.checked_add(padding)?;
    let numerator = pos_plus_p.checked_sub(offset)?;
    if numerator % stride != 0 {
        return None;
    }
    let out_idx = numerator / stride;
    if out_idx < out_len {
        Some(out_idx)
    } else {
        None
    }
}

/// [`fandhe_ai_tensor_core::BackendOps::im2col`] の CPU 実装本体
/// （イシュー #1764）。`out_shape` は呼び出し元（`ops.rs`）が
/// [`fandhe_ai_tensor_core::im2col_out_shape`] で検査・確定済みの
/// `[N, G, Cin_g·kH·kW, Hout·Wout]` をそのまま渡す。
///
/// `K_g` 軸は `(c_in_g, kh, kw)` の row-major・`P` 軸は `(oh, ow)` の
/// row-major で並べる（設計 doc §5.2）。
pub fn im2col(
    input: &Tensor<f32>,
    params: &Conv2dParams,
    out_shape: &[usize],
) -> Result<Tensor<f32>, ShapeError> {
    let out_numel = checked_numel(out_shape)?;
    if out_numel == 0 {
        return Tensor::new(Vec::new(), out_shape);
    }
    let in_shape = input.shape();
    let (n_batch, _cin, h_in, w_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
    let groups = out_shape[1];
    let k_g = out_shape[2];
    let p = out_shape[3];
    let cin_g = in_shape[1] / groups.max(1);
    let [kh_k, kw_k] = params.kernel_size();
    let [sh, sw] = params.stride();
    let [ph, pw] = params.padding();
    let [dh, dw] = params.dilation();
    let h_out = conv_out_len(h_in, kh_k, sh, ph, dh)?;
    let w_out = conv_out_len(w_in, kw_k, sw, pw, dw)?;
    debug_assert_eq!(
        h_out.checked_mul(w_out),
        Some(p),
        "im2col: out_shape の P 軸が conv_out_len から再計算した Hout*Wout と一致しない（契約違反）"
    );

    let mut out = vec![0f32; out_numel];
    for n in 0..n_batch {
        for g in 0..groups {
            for k_idx in 0..k_g {
                // k_idx を (c_g, kh, kw) の row-major へ展開
                // （kw が最内軸）。
                let kw_ = k_idx % kw_k;
                let rest = k_idx / kw_k;
                let kh_ = rest % kh_k;
                let c_g = rest / kh_k;
                let c = g * cin_g + c_g;
                for p_idx in 0..p {
                    let ow = p_idx % w_out;
                    let oh = p_idx / w_out;
                    let h_pos = im2col_input_pos(oh, sh, kh_, dh, ph);
                    let w_pos = im2col_input_pos(ow, sw, kw_, dw, pw);
                    let value = match (h_pos, w_pos) {
                        (Some(h), Some(w)) if h < h_in && w < w_in => {
                            input.get(&[n, c, h, w]).unwrap_or(0.0)
                        }
                        _ => 0.0,
                    };
                    let out_idx = ((n * groups + g) * k_g + k_idx) * p + p_idx;
                    out[out_idx] = value;
                }
            }
        }
    }
    Tensor::new(out, out_shape)
}

/// [`fandhe_ai_tensor_core::BackendOps::col2im`] の CPU 実装本体
/// （イシュー #1764）。`d_col: [N, G, K_g, P]` を `input_shape:
/// [N, Cin, H, W]` へ畳み戻す（[`im2col`] の随伴＝転置畳み込み）。
///
/// 入力位置定常（1 出力要素＝1 入力位置）の走査で `(kh, kw)` を
/// row-major に加算し、`f64` アキュムレータへ逐次加算して最後に 1 回
/// `f32` へ downcast する（モジュール doc の数値契約）。
pub fn col2im(
    d_col: &Tensor<f32>,
    input_shape: &[usize],
    params: &Conv2dParams,
) -> Result<Tensor<f32>, ShapeError> {
    let out_numel = checked_numel(input_shape)?;
    if out_numel == 0 {
        return Tensor::new(Vec::new(), input_shape);
    }
    let (n_batch, cin, h_in, w_in) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let d_col_shape = d_col.shape();
    let groups = d_col_shape[1];
    let p = d_col_shape[3];
    let cin_g = cin / groups.max(1);
    let [kh_k, kw_k] = params.kernel_size();
    let [sh, sw] = params.stride();
    let [ph, pw] = params.padding();
    let [dh, dw] = params.dilation();
    let h_out = conv_out_len(h_in, kh_k, sh, ph, dh)?;
    let w_out = conv_out_len(w_in, kw_k, sw, pw, dw)?;
    debug_assert_eq!(
        h_out.checked_mul(w_out),
        Some(p),
        "col2im: d_col の P 軸が conv_out_len から再計算した Hout*Wout と一致しない（契約違反）"
    );

    let mut out = vec![0f32; out_numel];
    for n in 0..n_batch {
        for c in 0..cin {
            let g = c / cin_g.max(1);
            let c_g = c % cin_g.max(1);
            for h in 0..h_in {
                for w in 0..w_in {
                    let mut acc: f64 = 0.0;
                    for kh_ in 0..kh_k {
                        let oh = match col2im_out_idx(h, ph, kh_, dh, sh, h_out) {
                            Some(v) => v,
                            None => continue,
                        };
                        for kw_ in 0..kw_k {
                            let ow = match col2im_out_idx(w, pw, kw_, dw, sw, w_out) {
                                Some(v) => v,
                                None => continue,
                            };
                            let k_idx = (c_g * kh_k + kh_) * kw_k + kw_;
                            let p_idx = oh * w_out + ow;
                            let v = d_col.get(&[n, g, k_idx, p_idx]).unwrap_or(0.0);
                            acc += f64::from(v);
                        }
                    }
                    let out_idx = ((n * cin + c) * h_in + h) * w_in + w;
                    out[out_idx] = acc as f32;
                }
            }
        }
    }
    Tensor::new(out, input_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_tensor_core::im2col_out_shape;

    fn params(
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Conv2dParams {
        Conv2dParams::new(kernel_size, stride, padding, dilation, groups).unwrap()
    }

    #[test]
    fn im2col_basic_no_pad_matches_naive_reference() {
        // 2x2 入力（1 チャンネル・1 バッチ）・kernel=2x2・stride=1・
        // padding=0 -> im2col は単一の 4 要素列。
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
        let p = params([2, 2], [1, 1], [0, 0], [1, 1], 1);
        let out_shape = im2col_out_shape(x.shape(), &p).unwrap();
        assert_eq!(out_shape, vec![1, 1, 4, 1]);
        let col = im2col(&x, &p, &out_shape).unwrap();
        // K_g row-major (kh, kw): (0,0)=1, (0,1)=2, (1,0)=3, (1,1)=4
        assert_eq!(col.contiguous().as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn im2col_with_padding_writes_zero_for_out_of_bounds() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
        let p = params([3, 3], [1, 1], [1, 1], [1, 1], 1);
        let out_shape = im2col_out_shape(x.shape(), &p).unwrap();
        // Hout=Wout=2 (in=2,k=3,s=1,p=1,d=1 -> (2+2-3-1)/1+1 = 1+1=2)
        assert_eq!(out_shape, vec![1, 1, 9, 4]);
        let col = im2col(&x, &p, &out_shape).unwrap();
        let data = col.contiguous().as_slice().unwrap().to_vec();
        // 窓 (oh=0, ow=0) は入力左上 3x3 で中心が x[0,0]=1、その他は
        // padding。K_g の (kh,kw) 順で P=0 列を読む。
        // 座標: 入力位置 h = oh + kh - 1, w = ow + kw - 1
        for kh in 0..3usize {
            for kw in 0..3usize {
                let k_idx = kh * 3 + kw;
                let expected = match (kh, kw) {
                    (1, 1) => 1.0,
                    (1, 2) => 2.0,
                    (2, 1) => 3.0,
                    (2, 2) => 4.0,
                    _ => 0.0,
                };
                assert_eq!(data[k_idx * 4], expected, "kh={kh} kw={kw}");
            }
        }
    }

    #[test]
    fn im2col_dilation_and_groups() {
        // groups=2, Cin=2 (Cin_g=1 each), Cout unrelated here (im2col
        // doesn't need weight). 2x2 spatial, kernel=1x1, dilation
        // irrelevant for k=1 (kh-1=0).
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 2, 2, 2]).unwrap();
        let p = params([1, 1], [1, 1], [0, 0], [1, 1], 2);
        let out_shape = im2col_out_shape(x.shape(), &p).unwrap();
        assert_eq!(out_shape, vec![1, 2, 1, 4]);
        let col = im2col(&x, &p, &out_shape).unwrap();
        let data = col.contiguous().as_slice().unwrap().to_vec();
        // group 0 -> channel 0 (1,2,3,4), group 1 -> channel 1 (5,6,7,8)
        assert_eq!(&data[0..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&data[4..8], &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn im2col_empty_output_returns_empty_tensor() {
        let x = Tensor::new(Vec::<f32>::new(), &[0, 1, 3, 3]).unwrap();
        let p = params([3, 3], [1, 1], [0, 0], [1, 1], 1);
        let out_shape = im2col_out_shape(x.shape(), &p).unwrap();
        assert_eq!(out_shape, vec![0, 1, 9, 1]);
        let col = im2col(&x, &p, &out_shape).unwrap();
        assert_eq!(col.numel(), 0);
    }

    #[test]
    fn col2im_is_adjoint_of_im2col_no_overlap() {
        // stride == kernel -> 窓が重ならないため col2im(im2col(x)) は
        // 恒等（padding 0）。
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
        let p = params([2, 2], [2, 2], [0, 0], [1, 1], 1);
        let out_shape = im2col_out_shape(x.shape(), &p).unwrap();
        let col = im2col(&x, &p, &out_shape).unwrap();
        let back = col2im(&col, x.shape(), &p).unwrap();
        assert_eq!(
            back.contiguous().as_slice().unwrap(),
            x.contiguous().as_slice().unwrap()
        );
    }

    #[test]
    fn col2im_overlapping_windows_sum_contributions() {
        // stride=1 < kernel=2 -> 窓が重なるため col2im は
        // im2col の随伴（転置畳み込み）として重複加算する。
        // 入力 1x1 の x=1、kernel=2x2 stride=1 padding=0 -> Hout=Wout=... 実際は
        // 3x3 入力で検証する方が分かりやすいため 3x3 を使う。
        let x = Tensor::new(
            vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            &[1, 1, 3, 3],
        )
        .unwrap();
        let p = params([2, 2], [1, 1], [0, 0], [1, 1], 1);
        let out_shape = im2col_out_shape(x.shape(), &p).unwrap();
        let col = im2col(&x, &p, &out_shape).unwrap();
        // upstream = col 自体（d_input の伴に相当する検証として、
        // upstream をすべて 1.0 にした場合の重なり回数を検証する）。
        let ones = Tensor::new(vec![1.0f32; col.numel()], col.shape()).unwrap();
        let back = col2im(&ones, x.shape(), &p).unwrap();
        // 中心セル (1,1) は 4 つの窓すべてに含まれるため寄与 4。
        // 角セル (0,0) は 1 つの窓のみに含まれるため寄与 1。
        // 辺セル (0,1) は 2 つの窓に含まれるため寄与 2。
        let back_c = back.contiguous();
        let data = back_c.as_slice().unwrap();
        assert_eq!(data[0 * 3 + 0], 1.0); // 角
        assert_eq!(data[0 * 3 + 1], 2.0); // 辺
        assert_eq!(data[1 * 3 + 1], 4.0); // 中心
    }

    #[test]
    fn col2im_coordinate_underflow_does_not_panic() {
        // 設計 doc §6.2「訂正 2」の回帰: H=3, kernel=3, padding=0,
        // dilation=1 では h + p_h - kh*d_h が負になる (h, kh) の組が
        // 存在する（例: h=0, kh=1 -> 0 + 0 - 1 = -1）。checked_sub に
        // よる早期スキップで panic せず正しく寄与なしとして扱われる
        // ことを確認する。
        let x: Tensor<f32> = Tensor::zeros(&[1, 1, 3, 3]).unwrap();
        let p = params([3, 3], [1, 1], [0, 0], [1, 1], 1);
        let out_shape = im2col_out_shape(x.shape(), &p).unwrap();
        let ones = Tensor::new(vec![1.0f32; out_shape.iter().product()], &out_shape).unwrap();
        let back = col2im(&ones, x.shape(), &p);
        assert!(back.is_ok());
    }
}
