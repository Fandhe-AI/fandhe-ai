//! ONNX `Conv`（2 次元畳み込み。opset-13 系）オペ（イシュー #2076・親
//! #2034）。
//!
//! `onnx::export_nn`（`crates/onnx-interop/src/onnx/export_nn.rs`）が
//! `fandhe_ai_autodiff::nn::Conv2d` を [`super::super::onnx::export_ops::
//! ExportOp::Conv`] へ写像する際の逆写像（interp 側）として、22 op の
//! allowlist（`export_ops::SUPPORTED_OP_TYPES`）と interp のディスパッチ表
//! （`onnx::interp::run`）が対称になるよう新設した 23 番目の op（`docs/
//! onnx-export-op-mapping.md` §2・§7）。
//!
//! `X: [N, Cin, H, W]`・`W: [Cout, Cin/group, kH, kW]`（PyTorch
//! `nn.Conv2d.weight` と同じレイアウト。`fandhe_ai_autodiff::nn::Conv2d`
//! と同一。`crates/autodiff/src/nn/conv.rs`）・`B`（省略可）: `[Cout]`。
//!
//! 形状検査は本クレート独自に二重実装せず、2 段の既存関数へ委譲する:
//! [`fandhe_ai_tensor_core::Conv2dParams::new`]（`kernel_size`／`stride`／
//! `dilation`／`groups` の 0・`2·padding` オーバーフロー拒否）と
//! [`fandhe_ai_tensor_core::conv2d_out_shape`]（`Cin % groups`・
//! `weight[1] * groups == Cin`・`Cout % groups`・`Cout >= groups`・出力
//! 要素数オーバーフロー拒否。PyTorch `check_shape_forward` 相当）。
//!
//! 数値契約: 本関数は素朴な直接ループ（`K_g` 軸を `(c_in_g, kh, kw)` の
//! row-major で走査。`fandhe_ai_autodiff::nn::Conv2d` の CPU forward
//! （`im2col` → `gemm_batched` → `add`。`autodiff/src/grad.rs`）と同じ
//! 走査順を採るが、結合順序（im2col+GEMM 合成 vs 直接ループ）が異なるため
//! bit 完全一致は主張しない（REQ-2 統一複合判定「相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満」で検証する。`.claude/rules/coding-rust.md`）。累積は
//! FMA 契約統一方針に従い `f32::mul_add` を用いる。
//!
//! 境界検査（REQ-8。`.claude/rules/coding-rust.md`）: パディングにより入力
//! 範囲外を参照しうる `ih`／`iw` は明示的な範囲検査（`0 <= ih < H`）で
//! ゼロパディングとして扱い、`unsafe` な無検査アクセスは行わない。

use fandhe_ai_tensor_core::{Conv2dParams, Tensor, conv2d_out_shape};

use super::error::OpError;

/// `Conv` の属性。ONNX Conv-13 仕様の `auto_pad`／`dilations`／`group`／
/// `kernel_shape`／`pads`／`strides`（`AttributeProto` から後続の decode 層
/// が変換する想定。本モジュール自体は decode 層に依存しないプレーンな
/// 構造体で受け取る。`gemm.rs`／`layer_norm.rs` と同じ設計）。
///
/// `kernel_shape`／`strides`／`dilations`／`pads` は ONNX 仕様上省略可
/// （空 `Vec` で「未指定」を表す。[`conv`] が未指定時の既定値を適用する）。
/// `auto_pad` は本クレートが `"NOTSET"`（または欠落相当の空文字列）のみ
/// 対応する（`SAME_UPPER`／`SAME_LOWER`／`VALID` は `docs/onnx-export-op-
/// mapping.md` §7 に記載のとおり未対応。`.claude/rules/security.md` A03
/// に従い黙って `NOTSET` 扱いにはせず拒否する）。
#[derive(Debug, Clone)]
pub struct ConvAttrs {
    /// `kernel_shape`（`[kH, kW]`）。空なら重み shape から導出するため
    /// 検査をスキップする（ONNX 仕様の「省略時は W から推論」相当）。
    pub kernel_shape: Vec<i64>,
    /// `strides`（`[sH, sW]`）。空なら `[1, 1]`。
    pub strides: Vec<i64>,
    /// `pads`（`[h_begin, w_begin, h_end, w_end]`。ONNX 仕様の軸順）。
    /// 空なら `[0, 0, 0, 0]`。本クレートは対称パディング（`h_begin ==
    /// h_end` かつ `w_begin == w_end`）のみ対応する（`Conv2dParams` が
    /// 軸ごとの対称パディングのみ表現できるため）。
    pub pads: Vec<i64>,
    /// `dilations`（`[dH, dW]`）。空なら `[1, 1]`。
    pub dilations: Vec<i64>,
    /// `group`。
    pub group: i64,
    /// `auto_pad`。空文字列または `"NOTSET"` のみ受理する。
    pub auto_pad: String,
}

impl Default for ConvAttrs {
    fn default() -> Self {
        ConvAttrs {
            kernel_shape: Vec::new(),
            strides: Vec::new(),
            pads: Vec::new(),
            dilations: Vec::new(),
            group: 1,
            auto_pad: String::new(),
        }
    }
}

/// `attrs` の `[usize; 2]` 属性（`strides`／`dilations`）を読む。空なら
/// `default`、長さ 2 以外または `i64 -> usize` 変換に失敗する要素があれば
/// [`OpError::InvalidConvAttribute`]。
fn parse_pair(
    name: &'static str,
    values: &[i64],
    default: [usize; 2],
) -> Result<[usize; 2], OpError> {
    if values.is_empty() {
        return Ok(default);
    }
    if values.len() != 2 {
        return Err(OpError::InvalidConvAttribute {
            reason: format!(
                "Conv: `{name}` の長さは 2 でなければならない（実際 {}）",
                values.len()
            ),
        });
    }
    let mut out = [0usize; 2];
    for (i, &v) in values.iter().enumerate() {
        out[i] = usize::try_from(v).map_err(|_| OpError::InvalidConvAttribute {
            reason: format!("Conv: `{name}[{i}]` は非負でなければならない（実際 {v}）"),
        })?;
    }
    Ok(out)
}

/// `pads` 属性（ONNX 軸順 `[h_begin, w_begin, h_end, w_end]`）を読み、
/// `Conv2dParams` が表現できる対称パディング `[pH, pW]` へ変換する。空
/// なら `[0, 0]`。非負性・長さ 4・対称性（`h_begin == h_end`・`w_begin ==
/// w_end`）を検査する。
fn parse_pads(values: &[i64]) -> Result<[usize; 2], OpError> {
    if values.is_empty() {
        return Ok([0, 0]);
    }
    if values.len() != 4 {
        return Err(OpError::InvalidConvAttribute {
            reason: format!(
                "Conv: `pads` の長さは 4 でなければならない（実際 {}）",
                values.len()
            ),
        });
    }
    let mut v = [0usize; 4];
    for (i, &x) in values.iter().enumerate() {
        v[i] = usize::try_from(x).map_err(|_| OpError::InvalidConvAttribute {
            reason: format!("Conv: `pads[{i}]` は非負でなければならない（実際 {x}）"),
        })?;
    }
    if v[0] != v[2] || v[1] != v[3] {
        return Err(OpError::InvalidConvAttribute {
            reason: format!(
                "Conv: `pads` は対称（h_begin == h_end かつ w_begin == w_end）のみ対応する（実際 {v:?}）"
            ),
        });
    }
    Ok([v[0], v[1]])
}

/// `Conv(X, W, [B])` を計算する。
///
/// 検証順序: `auto_pad` → `X`／`W` の rank（4）→ `strides`／`dilations`／
/// `pads` の属性検査（`parse_pair`／`parse_pads`。本モジュール内 private
/// 関数のため intra-doc link ではなくコードスパンで参照する）→ `kernel_shape`
/// 一致検査（指定されている場合）→ `group` の `i64 -> usize` 変換 →
/// [`Conv2dParams::new`]（0・オーバーフロー検査）→ [`conv2d_out_shape`]
/// （`Cin`／`Cout`／`groups` 整合・出力要素数検査）→ `B` の rank／長さ
/// 検査 → 直接ループでの計算。1 つでも失敗すれば以降の計算を行わない
/// （`.claude/rules/security.md` A08 の部分実行禁止と同じ規律）。
pub fn conv(
    x: &Tensor<f32>,
    w: &Tensor<f32>,
    b: Option<&Tensor<f32>>,
    attrs: &ConvAttrs,
) -> Result<Tensor<f32>, OpError> {
    if !attrs.auto_pad.is_empty() && attrs.auto_pad != "NOTSET" {
        return Err(OpError::InvalidConvAttribute {
            reason: format!(
                "Conv: `auto_pad` は \"NOTSET\"（または省略）のみ対応する（実際 \"{}\"）",
                attrs.auto_pad
            ),
        });
    }
    if x.rank() != 4 {
        return Err(OpError::RankMismatch {
            op: "Conv(X)",
            expected: 4,
            actual: x.rank(),
        });
    }
    if w.rank() != 4 {
        return Err(OpError::RankMismatch {
            op: "Conv(W)",
            expected: 4,
            actual: w.rank(),
        });
    }

    let w_shape = w.shape();
    let (cout, cin_g, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);

    if !attrs.kernel_shape.is_empty() {
        if attrs.kernel_shape.len() != 2 {
            return Err(OpError::InvalidConvAttribute {
                reason: format!(
                    "Conv: `kernel_shape` の長さは 2 でなければならない（実際 {}）",
                    attrs.kernel_shape.len()
                ),
            });
        }
        let given = parse_pair("kernel_shape", &attrs.kernel_shape, [kh, kw])?;
        if given != [kh, kw] {
            return Err(OpError::InvalidConvAttribute {
                reason: format!(
                    "Conv: `kernel_shape` {given:?} が重み shape [kH, kW]={:?} と一致しない",
                    [kh, kw]
                ),
            });
        }
    }

    let strides = parse_pair("strides", &attrs.strides, [1, 1])?;
    let dilations = parse_pair("dilations", &attrs.dilations, [1, 1])?;
    let pads = parse_pads(&attrs.pads)?;
    let group = usize::try_from(attrs.group).map_err(|_| OpError::InvalidConvAttribute {
        reason: format!(
            "Conv: `group` は 1 以上でなければならない（実際 {}）",
            attrs.group
        ),
    })?;

    let params = Conv2dParams::new([kh, kw], strides, pads, dilations, group).map_err(|e| {
        OpError::ConvParamsInvalid {
            reason: e.to_string(),
        }
    })?;

    let out_shape = conv2d_out_shape(x.shape(), w_shape, &params).map_err(OpError::from)?;
    let (n, _cin, _h, _w) = (out_shape[0], x.shape()[1], x.shape()[2], x.shape()[3]);
    let (out_cout, hout, wout) = (out_shape[1], out_shape[2], out_shape[3]);
    debug_assert_eq!(out_cout, cout);

    if let Some(bias) = b {
        if bias.rank() != 1 {
            return Err(OpError::RankMismatch {
                op: "Conv(B)",
                expected: 1,
                actual: bias.rank(),
            });
        }
        if bias.shape()[0] != cout {
            return Err(OpError::LengthMismatch {
                op: "Conv",
                name: "bias",
                expected: cout,
                actual: bias.shape()[0],
            });
        }
    }

    let x_c = x.contiguous();
    let w_c = w.contiguous();
    let x_slice = x_c
        .as_slice()
        .ok_or(OpError::NonContiguousInternal("Conv(X)"))?;
    let w_slice = w_c
        .as_slice()
        .ok_or(OpError::NonContiguousInternal("Conv(W)"))?;
    let bias_slice = match b {
        Some(bias) => {
            let bc = bias.contiguous();
            let data = bc
                .as_slice()
                .ok_or(OpError::NonContiguousInternal("Conv(B)"))?
                .to_vec();
            Some(data)
        }
        None => None,
    };

    let (cin_total, h, w_in) = (x.shape()[1], x.shape()[2], x.shape()[3]);
    let cout_g = cout / group;
    let [sh, sw] = strides;
    let [ph, pw] = pads;
    let [dh, dw] = dilations;

    let mut out = vec![0f32; n * cout * hout * wout];
    for ni in 0..n {
        for co in 0..cout {
            let g = co / cout_g;
            for oh in 0..hout {
                for ow in 0..wout {
                    let mut acc = 0f32;
                    for cg in 0..cin_g {
                        let cin = g * cin_g + cg;
                        // 境界検査（REQ-8）: パディング由来の範囲外参照は
                        // ゼロパディングとして扱い、`ih`／`iw` を明示的に
                        // `[0, h)`／`[0, w_in)` へ範囲検査してからのみ
                        // `x_slice` を読む（無検査アクセスをしない）。
                        //
                        // `strides`／`dilations`／`pads` は非信頼な ONNX
                        // 属性から `usize::MAX` 近傍まで受理されうる
                        // （`Conv2dParams::new` は非ゼロ・`2·padding` の
                        // オーバーフローのみ拒否し、stride／dilation の
                        // 上限は課さない）。このため `oh * sh + khi * dh`
                        // を無検査の `usize` 乗算・加算で計算してから
                        // `i64` へ `as` キャストすると、結果が `i64::MAX`
                        // を超える場合にキャストが負値へ折り返し、続く
                        // `- ph as i64` が debug build で panic、release
                        // build で誤った範囲判定になる（`.claude/rules/
                        // security.md` A03「外部入力の検証」）。
                        // `checked_mul`／`checked_add`／`checked_sub` の
                        // 連鎖で `usize` のまま検査し、表現不能（乗算・
                        // 加算オーバーフロー、または `ph`／`pw` 減算で
                        // 負になる＝パディング領域）な座標は範囲外として
                        // fail-closed に `continue`（ゼロパディング相当）
                        // する。符号付き整数への変換・`as` キャストは
                        // 一切行わない。
                        for khi in 0..kh {
                            let ih = match oh
                                .checked_mul(sh)
                                .and_then(|a| khi.checked_mul(dh).map(|b| (a, b)))
                                .and_then(|(a, b)| a.checked_add(b))
                                .and_then(|unpadded| unpadded.checked_sub(ph))
                            {
                                Some(v) if v < h => v,
                                _ => continue,
                            };
                            for kwi in 0..kw {
                                let iw = match ow
                                    .checked_mul(sw)
                                    .and_then(|a| kwi.checked_mul(dw).map(|b| (a, b)))
                                    .and_then(|(a, b)| a.checked_add(b))
                                    .and_then(|unpadded| unpadded.checked_sub(pw))
                                {
                                    Some(v) if v < w_in => v,
                                    _ => continue,
                                };
                                let x_idx = ((ni * cin_total + cin) * h + ih) * w_in + iw;
                                let w_idx = ((co * cin_g + cg) * kh + khi) * kw + kwi;
                                let x_val = *x_slice
                                    .get(x_idx)
                                    .ok_or(OpError::NonContiguousInternal("Conv(X read)"))?;
                                let w_val = *w_slice
                                    .get(w_idx)
                                    .ok_or(OpError::NonContiguousInternal("Conv(W read)"))?;
                                acc = w_val.mul_add(x_val, acc);
                            }
                        }
                    }
                    if let Some(bias) = &bias_slice {
                        acc += bias.get(co).copied().unwrap_or(0.0);
                    }
                    let out_idx = ((ni * cout + co) * hout + oh) * wout + ow;
                    if let Some(slot) = out.get_mut(out_idx) {
                        *slot = acc;
                    } else {
                        return Err(OpError::NonContiguousInternal("Conv(out write)"));
                    }
                }
            }
        }
    }

    Tensor::new(out, &out_shape).map_err(OpError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_1x1_kernel_no_padding() {
        // 1x1 kernel・stride 1・padding 0 は per-pixel の線形結合になる。
        // X: [1,1,2,2], W: [1,1,1,1] = [2.0] -> Y = 2*X。
        let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
        let w = Tensor::<f32>::new(vec![2.0], &[1, 1, 1, 1]).unwrap();
        let y = conv(&x, &w, None, &ConvAttrs::default()).unwrap();
        assert_eq!(y.shape(), &[1, 1, 2, 2]);
        assert_eq!(y.get(&[0, 0, 0, 0]).unwrap(), 2.0);
        assert_eq!(y.get(&[0, 0, 1, 1]).unwrap(), 8.0);
    }

    #[test]
    fn known_value_3x3_input_2x2_kernel() {
        // X: [1,1,3,3] = [[1,2,3],[4,5,6],[7,8,9]], W: [1,1,2,2] = [[1,0],[0,1]]
        // (対角のみ 1)。padding 0, stride 1 -> Y: [1,1,2,2].
        // Y[0,0] = X[0,0]*1 + X[0,1]*0 + X[1,0]*0 + X[1,1]*1 = 1 + 5 = 6
        // Y[0,1] = X[0,1] + X[1,2] = 2 + 6 = 8
        // Y[1,0] = X[1,0] + X[2,1] = 4 + 8 = 12
        // Y[1,1] = X[1,1] + X[2,2] = 5 + 9 = 14
        let x = Tensor::<f32>::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[1, 1, 3, 3],
        )
        .unwrap();
        let w = Tensor::<f32>::new(vec![1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]).unwrap();
        let y = conv(&x, &w, None, &ConvAttrs::default()).unwrap();
        assert_eq!(y.shape(), &[1, 1, 2, 2]);
        assert_eq!(y.get(&[0, 0, 0, 0]).unwrap(), 6.0);
        assert_eq!(y.get(&[0, 0, 0, 1]).unwrap(), 8.0);
        assert_eq!(y.get(&[0, 0, 1, 0]).unwrap(), 12.0);
        assert_eq!(y.get(&[0, 0, 1, 1]).unwrap(), 14.0);
    }

    #[test]
    fn asymmetric_kernel_catches_transposed_pads() {
        // `ph != pw`（非対称 kernel_shape を伴う入力）で pads 順を検証する。
        // X: [1,1,1,4] (H=1, W=4), kernel [1,3] (kH=1, kW=3), padding [0,1] (pH=0, pW=1).
        // pads(ONNX順) = [0,1,0,1] -> parse_pads は [pH=0, pW=1] へ変換される。
        let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4]).unwrap();
        let w = Tensor::<f32>::new(vec![1.0, 1.0, 1.0], &[1, 1, 1, 3]).unwrap();
        let attrs = ConvAttrs {
            pads: vec![0, 1, 0, 1],
            ..ConvAttrs::default()
        };
        let y = conv(&x, &w, None, &attrs).unwrap();
        // pW=1・kW=3・stride=1 -> Wout = (4 + 2*1 - 3) + 1 = 4
        assert_eq!(y.shape(), &[1, 1, 1, 4]);
        // padded input: [0, 1, 2, 3, 4, 0]
        // Y[0]=0+1+2=3, Y[1]=1+2+3=6, Y[2]=2+3+4=9, Y[3]=3+4+0=7
        assert_eq!(y.get(&[0, 0, 0, 0]).unwrap(), 3.0);
        assert_eq!(y.get(&[0, 0, 0, 1]).unwrap(), 6.0);
        assert_eq!(y.get(&[0, 0, 0, 2]).unwrap(), 9.0);
        assert_eq!(y.get(&[0, 0, 0, 3]).unwrap(), 7.0);
    }

    #[test]
    fn groups_split_channels() {
        // Cin=2, Cout=2, groups=2 -> 各グループは独立した 1x1 の Cin=1 -> Cout=1。
        let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 1, 2]).unwrap();
        // W: [Cout=2, Cin/g=1, 1, 1] = [10, 100]
        let w = Tensor::<f32>::new(vec![10.0, 100.0], &[2, 1, 1, 1]).unwrap();
        let attrs = ConvAttrs {
            group: 2,
            ..ConvAttrs::default()
        };
        let y = conv(&x, &w, None, &attrs).unwrap();
        assert_eq!(y.shape(), &[1, 2, 1, 2]);
        // channel 0 (group0): x=[1,2] * 10 = [10, 20]
        assert_eq!(y.get(&[0, 0, 0, 0]).unwrap(), 10.0);
        assert_eq!(y.get(&[0, 0, 0, 1]).unwrap(), 20.0);
        // channel 1 (group1): x=[3,4] * 100 = [300, 400]
        assert_eq!(y.get(&[0, 1, 0, 0]).unwrap(), 300.0);
        assert_eq!(y.get(&[0, 1, 0, 1]).unwrap(), 400.0);
    }

    #[test]
    fn bias_is_added_per_out_channel() {
        let x = Tensor::<f32>::new(vec![1.0], &[1, 1, 1, 1]).unwrap();
        let w = Tensor::<f32>::new(vec![2.0], &[1, 1, 1, 1]).unwrap();
        let bias = Tensor::<f32>::new(vec![5.0], &[1]).unwrap();
        let y = conv(&x, &w, Some(&bias), &ConvAttrs::default()).unwrap();
        assert_eq!(y.get(&[0, 0, 0, 0]).unwrap(), 2.0 * 1.0 + 5.0);
    }

    #[test]
    fn rank_mismatch_rejected() {
        let x = Tensor::<f32>::zeros(&[1, 1, 2]).unwrap();
        let w = Tensor::<f32>::zeros(&[1, 1, 1, 1]).unwrap();
        let err = conv(&x, &w, None, &ConvAttrs::default()).unwrap_err();
        assert!(matches!(
            err,
            OpError::RankMismatch {
                op: "Conv(X)",
                expected: 4,
                actual: 3,
            }
        ));
    }

    #[test]
    fn non_notset_auto_pad_rejected() {
        let x = Tensor::<f32>::zeros(&[1, 1, 2, 2]).unwrap();
        let w = Tensor::<f32>::zeros(&[1, 1, 1, 1]).unwrap();
        let attrs = ConvAttrs {
            auto_pad: "SAME_UPPER".to_string(),
            ..ConvAttrs::default()
        };
        let err = conv(&x, &w, None, &attrs).unwrap_err();
        assert!(matches!(err, OpError::InvalidConvAttribute { .. }));
    }

    #[test]
    fn asymmetric_pads_rejected() {
        let x = Tensor::<f32>::zeros(&[1, 1, 2, 2]).unwrap();
        let w = Tensor::<f32>::zeros(&[1, 1, 1, 1]).unwrap();
        let attrs = ConvAttrs {
            pads: vec![0, 0, 1, 0],
            ..ConvAttrs::default()
        };
        let err = conv(&x, &w, None, &attrs).unwrap_err();
        assert!(matches!(err, OpError::InvalidConvAttribute { .. }));
    }

    #[test]
    fn negative_pads_rejected() {
        let x = Tensor::<f32>::zeros(&[1, 1, 2, 2]).unwrap();
        let w = Tensor::<f32>::zeros(&[1, 1, 1, 1]).unwrap();
        let attrs = ConvAttrs {
            pads: vec![-1, 0, -1, 0],
            ..ConvAttrs::default()
        };
        let err = conv(&x, &w, None, &attrs).unwrap_err();
        assert!(matches!(err, OpError::InvalidConvAttribute { .. }));
    }

    #[test]
    fn kernel_shape_mismatch_rejected() {
        let x = Tensor::<f32>::zeros(&[1, 1, 3, 3]).unwrap();
        let w = Tensor::<f32>::zeros(&[1, 1, 2, 2]).unwrap();
        let attrs = ConvAttrs {
            kernel_shape: vec![3, 3],
            ..ConvAttrs::default()
        };
        let err = conv(&x, &w, None, &attrs).unwrap_err();
        assert!(matches!(err, OpError::InvalidConvAttribute { .. }));
    }

    #[test]
    fn cin_group_mismatch_rejected_via_conv2d_out_shape() {
        // Cin=3, groups=2 -> 3 % 2 != 0 -> conv2d_out_shape が Shape エラーを返す。
        let x = Tensor::<f32>::zeros(&[1, 3, 2, 2]).unwrap();
        let w = Tensor::<f32>::zeros(&[2, 1, 1, 1]).unwrap();
        let attrs = ConvAttrs {
            group: 2,
            ..ConvAttrs::default()
        };
        let err = conv(&x, &w, None, &attrs).unwrap_err();
        assert!(matches!(err, OpError::Shape(_)));
    }

    #[test]
    fn bias_length_mismatch_rejected() {
        let x = Tensor::<f32>::zeros(&[1, 1, 2, 2]).unwrap();
        let w = Tensor::<f32>::zeros(&[2, 1, 1, 1]).unwrap();
        let bias = Tensor::<f32>::zeros(&[3]).unwrap();
        let err = conv(&x, &w, Some(&bias), &ConvAttrs::default()).unwrap_err();
        assert!(matches!(err, OpError::LengthMismatch { op: "Conv", .. }));
    }

    // 回帰テスト（codex-review P0 指摘。PR #2220）: 非信頼な ONNX 属性
    // `pads`／`strides` から巨大な値を受理しても、`Conv2dParams::new`／
    // `conv2d_out_shape`（`conv_out_len`）の事前検査を通過してしまう
    // 境界値では、座標計算（旧実装の `oh * sh + khi * dh - ph`）が
    // debug build で panic しうることを固定値で検証する。
    //
    // 数値の選定根拠（`x: [1,1,3,3]`・`w: [1,1,2,2]`・
    // `pads = [2^62; 4]`・`strides = [i64::MAX; 2]`・`dilations = [1,1]`）:
    // - `p = 2^62` は `2 * padding = 2^63` が `usize`（64bit）の
    //   `checked_mul` を通過する（`Conv2dParams::new` は拒否しない）。
    // - `conv_out_len(in_len=3, k=2, s=i64::MAX, p=2^62, d=1)`:
    //   `numerator = (3 + 2*2^62) - (1*1) - 1 = 2^63 + 1`、
    //   `hout = numerator / (2^63 - 1) + 1 = 1 + 1 = 2`（`wout` も同じ）。
    //   出力要素数 `1*1*2*2=4` は `checked_numel_for` を通過する。
    // - 旧実装で `oh=1, khi=1` に到達すると
    //   `oh*sh + khi*dh = (2^63-1) + 1 = 2^63` を `as i64` すると
    //   ビット再解釈で `i64::MIN` になり、続く `- ph as i64`
    //   （`ph = 2^62`）が `i64::MIN - 2^62` で i64 の下限を突き抜けて
    //   debug build で subtract-overflow panic する。
    // - 本 PR の `checked_mul`／`checked_add`／`checked_sub` 連鎖は
    //   `unpadded = 2^63`・`unpadded.checked_sub(ph) = 2^62` を返し、
    //   `2^62 >= h(=3)` のため範囲外として `continue`（ゼロパディング
    //   扱い）する。したがって修正後は panic せず、全入力位置が
    //   パディング領域内になるため出力は全ゼロの `Ok` を返す
    //   （`W`／`B` はゼロ初期化のため `acc` も 0 のまま）。
    #[test]
    fn huge_pads_and_strides_no_longer_panic_and_yield_zero_output() {
        let x = Tensor::<f32>::zeros(&[1, 1, 3, 3]).unwrap();
        let w = Tensor::<f32>::zeros(&[1, 1, 2, 2]).unwrap();
        let huge_pad = 1i64 << 62;
        let attrs = ConvAttrs {
            pads: vec![huge_pad, huge_pad, huge_pad, huge_pad],
            strides: vec![i64::MAX, i64::MAX],
            dilations: vec![1, 1],
            ..ConvAttrs::default()
        };
        let out = conv(&x, &w, None, &attrs).expect(
            "巨大な pads/strides は範囲外として fail-closed に continue し、panic せず Ok を返す",
        );
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        let out_c = out.contiguous();
        let out_slice = out_c.as_slice().unwrap();
        assert!(
            out_slice.iter().all(|&v| v == 0.0),
            "全入力位置がパディング領域内のため出力は全ゼロのはず: {out_slice:?}"
        );
    }
}
