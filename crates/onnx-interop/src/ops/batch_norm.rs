//! ONNX `BatchNormalization`（推論モードのみ。イシュー #2200・親 #2185）オペ。
//!
//! ONNX `BatchNormalization-9`〜`-15` 系の必須 5 入力
//! `(X, scale, B, input_mean, input_var)` を推論モード（固定統計。学習
//! モードのバッチ統計再計算・running stats 更新は対象外）で計算する。
//! `training_mode` 属性（opset 14 以降）は `0` のみ受理し、それ以外は
//! [`OpError::InvalidBatchNormAttribute`] で拒否する（学習モードは
//! スコープ外。イシュー #2200 計画「対象外」節）。
//!
//! 数値契約（backend-cpu `run_batch_norm_infer_f32`・autodiff
//! `BatchNormCore` の eval forward と同じ逐語式。`.claude/rules/
//! coding-rust.md` の勾配長軸縮約とは別の「単純な eval 変換」であり、
//! `f64` アキュムレータでの縮約は伴わない——チャネルごとの `mean`／`var`
//! はすでに確定値として受け取るため）:
//! ```text
//! mean_c = mean[ch] as f64
//! rstd   = 1.0f64 / (var[ch] as f64 + eps as f64).sqrt()
//! xhat   = ((x[idx] as f64 - mean_c) * rstd) as f32
//! out    = xhat.mul_add(scale[ch], bias[ch])
//! ```
//! `idx` は `(n*C + ch)*spatial + sp`（`spatial = shape[2..]` の要素数積。
//! rank 2 なら `spatial = 1`）。本関数は `tensor_core::batch_norm_layout`
//! （rank 4 まで）に依存せず、ONNX BN が許す rank 2 以上を自前でレイアウト
//! 計算する。
//!
//! `ops::conv`・`ops::layer_normalization` と同じ「入力テンソル＋属性 →
//! 出力テンソル」の純粋関数方針に従い、ONNX proto デコード層・グラフ実行
//! エンジンには関与しない（`ops/mod.rs` モジュール doc 参照）。

use fandhe_ai_tensor_core::Tensor;

use super::error::OpError;

/// `BatchNormalization` の属性。ONNX 仕様の既定値（`epsilon = 1e-5`）を
/// [`Default`] に反映する。`momentum` は推論では使わないため保持しない
/// （型だけ検証して値を捨てる呼び出し側〈`onnx::interp::compute_batch_
/// normalization`〉の責務）。
#[derive(Debug, Clone, Copy)]
pub struct BatchNormAttrs {
    pub epsilon: f32,
}

impl Default for BatchNormAttrs {
    fn default() -> Self {
        BatchNormAttrs { epsilon: 1e-5 }
    }
}

/// `BatchNormalization(X, scale, B, input_mean, input_var)` を推論モードで
/// 計算する。
///
/// 検証順序（1 つでも失敗すれば以降の計算を行わない。`.claude/rules/
/// security.md` A08 の部分実行禁止と同じ規律）:
/// 1. `epsilon` が非有限なら [`OpError::InvalidEpsilon`]
/// 2. `epsilon < 0` なら [`OpError::InvalidBatchNormAttribute`]（`run_batch_
///    norm_infer_f32` の `validate_batch_norm_launch` と同じ条件で backend-cpu
///    と挙動を揃える。`LayerNormalization` は負値を許容するが、本オペは
///    ONNX `BatchNormalization` が学習・推論いずれも分散に非負を要求する
///    仕様〈PyTorch も `eps` を分散安定化定数として非負前提で扱う〉に従う）
/// 3. `x.rank() < 2` なら [`OpError::RankMismatch`]
/// 4. `scale`／`B`／`input_mean`／`input_var` の rank が 1 以外なら
///    [`OpError::RankMismatch`]、要素数が `C`（`x.shape()[1]`）と一致しな
///    ければ [`OpError::LengthMismatch`]（ブロードキャストは行わない）
/// 5. `input_var[ch] as f64 + epsilon as f64` が非負（`NaN` を含め、
///    `sum < 0.0 || sum.is_nan()` を拒否条件とする）でなければ
///    [`OpError::InvalidBatchNormAttribute`]。ONNX `BatchNormalization`
///    は `sqrt(var + epsilon)` を計算式とするため、判定は `var` 単体で
///    はなく `var + epsilon` に対して行う（`var` 単体の非負性で判定する
///    と、PyTorch エクスポートモデルでよく見られる「running variance が
///    浮動小数点誤差でわずかに負だが `epsilon` を加算すれば非負になる」
///    有効なケースまで fail-closed に拒否してしまう。Cursor Bugbot
///    指摘・イシュー #2200 PR #2312 レビュー）。`input_var` は ONNX
///    モデル（外部入力）由来の値であり、`var + epsilon` が負のまま
///    `sqrt` へ渡ると `rstd` が `NaN` に汚染され出力全体へ静かに伝播
///    する（OWASP A03。`.claude/rules/security.md`。`x` 自体の `NaN`
///    伝播〈`nan_propagates` テスト〉とは異なり、こちらは属性値の事前
///    検証で fail-closed に拒否する）
pub fn batch_normalization(
    x: &Tensor<f32>,
    scale: &Tensor<f32>,
    bias: &Tensor<f32>,
    mean: &Tensor<f32>,
    var: &Tensor<f32>,
    attrs: &BatchNormAttrs,
) -> Result<Tensor<f32>, OpError> {
    if !attrs.epsilon.is_finite() {
        return Err(OpError::InvalidEpsilon {
            op: "BatchNormalization",
            epsilon: attrs.epsilon,
        });
    }
    if attrs.epsilon < 0.0 {
        return Err(OpError::InvalidBatchNormAttribute {
            reason: format!("epsilon は非負でなければならない（実際 {}）", attrs.epsilon),
        });
    }

    if x.rank() < 2 {
        return Err(OpError::RankMismatch {
            op: "BatchNormalization(X)",
            expected: 2,
            actual: x.rank(),
        });
    }
    let c = x.shape()[1];

    for (name, t) in [
        ("scale", scale),
        ("B", bias),
        ("input_mean", mean),
        ("input_var", var),
    ] {
        if t.rank() != 1 {
            return Err(OpError::RankMismatch {
                op: "BatchNormalization",
                expected: 1,
                actual: t.rank(),
            });
        }
        if t.shape()[0] != c {
            return Err(OpError::LengthMismatch {
                op: "BatchNormalization",
                name,
                expected: c,
                actual: t.shape()[0],
            });
        }
    }

    let n = x.shape()[0];
    // `spatial` = shape[2..] の要素数積（rank 2 なら空スライスの積 = 1）。
    // 非信頼な shape 由来のオーバーフローを避けるため `checked_mul` で
    // 計算する（`Conv`／`GlobalAveragePool` と同じ方針。OWASP A03）。
    let spatial = x.shape()[2..]
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(OpError::Shape(
            fandhe_ai_tensor_core::ShapeError::ElementCountOverflow,
        ))?;

    if n == 0 || c == 0 || spatial == 0 {
        // いずれかの次元が 0 の場合は部分積を計算する前に空テンソルを
        // 返す（`tensor_core::batch_norm_layout` doc が警告する部分積
        // オーバーフローの罠を避ける。`ops::conv`／`global_average_pool`
        // と同じ早期 return 方針）。
        return Tensor::new(Vec::new(), x.shape()).map_err(OpError::from);
    }

    let xc = x.contiguous();
    let x_slice = xc
        .as_slice()
        .ok_or(OpError::NonContiguousInternal("BatchNormalization(X)"))?;

    let scale_c = scale.contiguous();
    let scale_slice = scale_c
        .as_slice()
        .ok_or(OpError::NonContiguousInternal("BatchNormalization(scale)"))?;
    let bias_c = bias.contiguous();
    let bias_slice = bias_c
        .as_slice()
        .ok_or(OpError::NonContiguousInternal("BatchNormalization(B)"))?;
    let mean_c = mean.contiguous();
    let mean_slice = mean_c.as_slice().ok_or(OpError::NonContiguousInternal(
        "BatchNormalization(input_mean)",
    ))?;
    let var_c = var.contiguous();
    let var_slice = var_c.as_slice().ok_or(OpError::NonContiguousInternal(
        "BatchNormalization(input_var)",
    ))?;

    // `input_var` は ONNX モデル（外部入力）由来の分散値。ONNX
    // `BatchNormalization` の計算式は `sqrt(var + epsilon)` であり、
    // `var` 単体ではなく `var + epsilon` の非負性を検証する（`var` 単体
    // で判定すると、PyTorch エクスポートモデルでよく見られる「running
    // variance が浮動小数点誤差でわずかに負だが epsilon を加算すれば
    // 非負になる」有効なケースまで fail-closed に拒否してしまう。
    // Cursor Bugbot 指摘・イシュー #2200 PR #2312 レビュー）。検証せず
    // `sqrt` へ渡すと `rstd` が `NaN` になり出力全体が静かに汚染される
    // （OWASP A03。`.claude/rules/security.md`）ため、計算ループへ入る前に
    // 全チャネルを検証する（`sum >= 0.0` は `NaN` に対して常に偽になる
    // ため、`var` が `NaN` の場合も `sum` の `NaN` 判定で同じ条件で拒否
    // できる）。`epsilon` は関数冒頭で既に有限・非負を検証済みのため、
    // ここでの非有限化要因は `var` 側の `NaN`／`inf` のみである。
    let eps_f64 = attrs.epsilon as f64;
    for (ch, &v) in var_slice.iter().enumerate() {
        let sum = v as f64 + eps_f64;
        // `sum < 0.0 || sum.is_nan()` は `!(sum >= 0.0)` と同値だが、
        // `clippy::neg_cmp_op_on_partial_ord` を避けつつ「負値・NaN の
        // どちらも拒否する」意図を明示する（`PartialOrd` の否定比較は
        // 非全順序型で直感に反する場合があるため、明示形を使う）。
        if sum < 0.0 || sum.is_nan() {
            return Err(OpError::InvalidBatchNormAttribute {
                reason: format!(
                    "input_var[{ch}] + epsilon は非負でなければならない（実際 var={v}, epsilon={}）",
                    attrs.epsilon
                ),
            });
        }
    }

    let numel = n * c * spatial;
    let mut out = vec![0f32; numel];
    for ch in 0..c {
        let mean_ch = mean_slice[ch] as f64;
        let rstd = 1.0f64 / (var_slice[ch] as f64 + attrs.epsilon as f64).sqrt();
        let sv = scale_slice[ch];
        let bv = bias_slice[ch];
        for ni in 0..n {
            let base = (ni * c + ch) * spatial;
            for sp in 0..spatial {
                let idx = base + sp;
                let xhat = ((x_slice[idx] as f64 - mean_ch) * rstd) as f32;
                out[idx] = xhat.mul_add(sv, bv);
            }
        }
    }

    Tensor::new(out, xc.shape()).map_err(OpError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 独立した素朴なループの参照実装（本体実装の `mean_ch`/`rstd` の
    /// チャネル外ループ切り出しとは異なる走査順で再計算し、実装バグの
    /// 見落としを防ぐ）。テスト専用のプレーンな引数列のため `ops/conv.rs`
    /// と同じ理由で `too_many_arguments` を許容する。
    #[allow(clippy::too_many_arguments)]
    fn reference_bn(
        x: &[f32],
        n: usize,
        c: usize,
        spatial: usize,
        scale: &[f32],
        bias: &[f32],
        mean: &[f32],
        var: &[f32],
        eps: f32,
    ) -> Vec<f32> {
        let mut out = vec![0f32; n * c * spatial];
        for ni in 0..n {
            for ch in 0..c {
                let rstd = 1.0f64 / (var[ch] as f64 + eps as f64).sqrt();
                for sp in 0..spatial {
                    let idx = (ni * c + ch) * spatial + sp;
                    let xhat = ((x[idx] as f64 - mean[ch] as f64) * rstd) as f32;
                    out[idx] = xhat * scale[ch] + bias[ch];
                }
            }
        }
        out
    }

    #[test]
    fn rank4_matches_reference() {
        // N=1, C=2, H=2, W=2
        let x: Vec<f32> = (0..8).map(|v| v as f32).collect();
        let xt = Tensor::<f32>::new(x.clone(), &[1, 2, 2, 2]).unwrap();
        let scale = Tensor::<f32>::new(vec![1.0, 2.0], &[2]).unwrap();
        let bias = Tensor::<f32>::new(vec![0.0, 1.0], &[2]).unwrap();
        let mean = Tensor::<f32>::new(vec![1.5, 5.5], &[2]).unwrap();
        let var = Tensor::<f32>::new(vec![1.25, 1.25], &[2]).unwrap();
        let attrs = BatchNormAttrs { epsilon: 1e-5 };
        let y = batch_normalization(&xt, &scale, &bias, &mean, &var, &attrs).unwrap();
        assert_eq!(y.shape(), &[1, 2, 2, 2]);

        let reference = reference_bn(
            &x,
            1,
            2,
            4,
            &[1.0, 2.0],
            &[0.0, 1.0],
            &[1.5, 5.5],
            &[1.25, 1.25],
            1e-5,
        );
        let yc = y.contiguous();
        assert_eq!(yc.as_slice().unwrap(), reference.as_slice());
    }

    #[test]
    fn rank3_and_rank2_supported() {
        // rank 3: [N=1, C=2, L=3]
        let x3: Vec<f32> = (0..6).map(|v| v as f32).collect();
        let xt3 = Tensor::<f32>::new(x3, &[1, 2, 3]).unwrap();
        let scale = Tensor::<f32>::new(vec![1.0, 1.0], &[2]).unwrap();
        let bias = Tensor::<f32>::new(vec![0.0, 0.0], &[2]).unwrap();
        let mean = Tensor::<f32>::new(vec![0.0, 0.0], &[2]).unwrap();
        let var = Tensor::<f32>::new(vec![1.0, 1.0], &[2]).unwrap();
        let attrs = BatchNormAttrs::default();
        let y3 = batch_normalization(&xt3, &scale, &bias, &mean, &var, &attrs).unwrap();
        assert_eq!(y3.shape(), &[1, 2, 3]);

        // rank 2: [N=2, C=2]
        let xt2 = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let y2 = batch_normalization(&xt2, &scale, &bias, &mean, &var, &attrs).unwrap();
        assert_eq!(y2.shape(), &[2, 2]);
    }

    #[test]
    fn empty_dim_yields_empty_output() {
        let xt = Tensor::<f32>::new(Vec::new(), &[0, 2, 2, 2]).unwrap();
        let scale = Tensor::<f32>::new(vec![1.0, 1.0], &[2]).unwrap();
        let bias = Tensor::<f32>::new(vec![0.0, 0.0], &[2]).unwrap();
        let mean = Tensor::<f32>::new(vec![0.0, 0.0], &[2]).unwrap();
        let var = Tensor::<f32>::new(vec![1.0, 1.0], &[2]).unwrap();
        let y = batch_normalization(&xt, &scale, &bias, &mean, &var, &BatchNormAttrs::default())
            .unwrap();
        assert_eq!(y.shape(), &[0, 2, 2, 2]);
        assert!(y.contiguous().as_slice().unwrap().is_empty());
    }

    #[test]
    fn negative_eps_rejected() {
        let xt = Tensor::<f32>::zeros(&[1, 2, 2, 2]).unwrap();
        let scale = Tensor::<f32>::zeros(&[2]).unwrap();
        let bias = Tensor::<f32>::zeros(&[2]).unwrap();
        let mean = Tensor::<f32>::zeros(&[2]).unwrap();
        let var = Tensor::<f32>::zeros(&[2]).unwrap();
        let attrs = BatchNormAttrs { epsilon: -1.0 };
        let err = batch_normalization(&xt, &scale, &bias, &mean, &var, &attrs).unwrap_err();
        assert!(matches!(err, OpError::InvalidBatchNormAttribute { .. }));
    }

    #[test]
    fn non_finite_eps_rejected() {
        let xt = Tensor::<f32>::zeros(&[1, 2, 2, 2]).unwrap();
        let scale = Tensor::<f32>::zeros(&[2]).unwrap();
        let bias = Tensor::<f32>::zeros(&[2]).unwrap();
        let mean = Tensor::<f32>::zeros(&[2]).unwrap();
        let var = Tensor::<f32>::zeros(&[2]).unwrap();
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let attrs = BatchNormAttrs { epsilon: bad };
            let err = batch_normalization(&xt, &scale, &bias, &mean, &var, &attrs).unwrap_err();
            assert!(matches!(err, OpError::InvalidEpsilon { .. }));
        }
    }

    #[test]
    fn rank_mismatch_rejected() {
        let xt = Tensor::<f32>::zeros(&[2]).unwrap();
        let scale = Tensor::<f32>::zeros(&[2]).unwrap();
        let bias = Tensor::<f32>::zeros(&[2]).unwrap();
        let mean = Tensor::<f32>::zeros(&[2]).unwrap();
        let var = Tensor::<f32>::zeros(&[2]).unwrap();
        let err = batch_normalization(&xt, &scale, &bias, &mean, &var, &BatchNormAttrs::default())
            .unwrap_err();
        assert!(matches!(
            err,
            OpError::RankMismatch {
                op: "BatchNormalization(X)",
                expected: 2,
                actual: 1,
            }
        ));
    }

    #[test]
    fn param_rank_mismatch_rejected() {
        let xt = Tensor::<f32>::zeros(&[1, 2, 2, 2]).unwrap();
        let scale = Tensor::<f32>::zeros(&[1, 2]).unwrap();
        let bias = Tensor::<f32>::zeros(&[2]).unwrap();
        let mean = Tensor::<f32>::zeros(&[2]).unwrap();
        let var = Tensor::<f32>::zeros(&[2]).unwrap();
        let err = batch_normalization(&xt, &scale, &bias, &mean, &var, &BatchNormAttrs::default())
            .unwrap_err();
        assert!(matches!(
            err,
            OpError::RankMismatch {
                op: "BatchNormalization",
                expected: 1,
                actual: 2,
            }
        ));
    }

    #[test]
    fn param_length_mismatch_rejected() {
        let xt = Tensor::<f32>::zeros(&[1, 2, 2, 2]).unwrap();
        let scale = Tensor::<f32>::zeros(&[3]).unwrap();
        let bias = Tensor::<f32>::zeros(&[2]).unwrap();
        let mean = Tensor::<f32>::zeros(&[2]).unwrap();
        let var = Tensor::<f32>::zeros(&[2]).unwrap();
        let err = batch_normalization(&xt, &scale, &bias, &mean, &var, &BatchNormAttrs::default())
            .unwrap_err();
        assert!(matches!(
            err,
            OpError::LengthMismatch {
                op: "BatchNormalization",
                name: "scale",
                expected: 2,
                actual: 3,
            }
        ));
    }

    #[test]
    fn negative_variance_rejected() {
        // `input_var` に負値を含む ONNX モデルは `sqrt` へそのまま渡すと
        // `NaN` が静かに伝播するため、計算前に fail-closed で拒否する
        // （codex-review 指摘・PR #2312。`crates/onnx-interop/src/ops/
        // batch_norm.rs` の検証順序 5.）。
        let xt = Tensor::<f32>::zeros(&[1, 2, 2, 2]).unwrap();
        let scale = Tensor::<f32>::new(vec![1.0, 1.0], &[2]).unwrap();
        let bias = Tensor::<f32>::new(vec![0.0, 0.0], &[2]).unwrap();
        let mean = Tensor::<f32>::new(vec![0.0, 0.0], &[2]).unwrap();
        let var = Tensor::<f32>::new(vec![1.0, -0.5], &[2]).unwrap();
        let err = batch_normalization(&xt, &scale, &bias, &mean, &var, &BatchNormAttrs::default())
            .unwrap_err();
        assert!(matches!(err, OpError::InvalidBatchNormAttribute { .. }));
    }

    #[test]
    fn variance_rescued_by_epsilon_accepted() {
        // `var` 単体はわずかに負だが `var + epsilon` は非負になる、PyTorch
        // エクスポートモデルで一般的な有効ケースを受理することを確認する
        // （Cursor Bugbot 指摘・イシュー #2200 PR #2312 レビュー。`var`
        // 単体の非負性で判定すると誤って拒否してしまう回帰の防止）。
        let xt = Tensor::<f32>::zeros(&[1, 2, 2, 2]).unwrap();
        let scale = Tensor::<f32>::new(vec![1.0, 1.0], &[2]).unwrap();
        let bias = Tensor::<f32>::new(vec![0.0, 0.0], &[2]).unwrap();
        let mean = Tensor::<f32>::new(vec![0.0, 0.0], &[2]).unwrap();
        // epsilon = 1e-5 に対し var = -1e-8 は `var + epsilon > 0` となる。
        let var = Tensor::<f32>::new(vec![1.0, -1e-8], &[2]).unwrap();
        let out = batch_normalization(&xt, &scale, &bias, &mean, &var, &BatchNormAttrs::default())
            .unwrap();
        assert!(out.as_slice().unwrap().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn variance_still_negative_after_epsilon_rejected() {
        // `var + epsilon` が依然として負の場合は引き続き fail-closed で
        // 拒否することを確認する（epsilon 加算で無条件に受理してしまう
        // 退行の防止）。
        let xt = Tensor::<f32>::zeros(&[1, 2, 2, 2]).unwrap();
        let scale = Tensor::<f32>::new(vec![1.0, 1.0], &[2]).unwrap();
        let bias = Tensor::<f32>::new(vec![0.0, 0.0], &[2]).unwrap();
        let mean = Tensor::<f32>::new(vec![0.0, 0.0], &[2]).unwrap();
        let var = Tensor::<f32>::new(vec![1.0, -0.5], &[2]).unwrap();
        let attrs = BatchNormAttrs { epsilon: 1e-5 };
        let err = batch_normalization(&xt, &scale, &bias, &mean, &var, &attrs).unwrap_err();
        assert!(matches!(err, OpError::InvalidBatchNormAttribute { .. }));
    }

    #[test]
    fn nan_variance_rejected() {
        // `NaN` は `v >= 0.0` が常に偽になるため負値と同じ経路で拒否される
        // ことを確認する（比較演算子の落とし穴〈`NaN < 0.0` も偽〉を突いた
        // 迂回を防ぐ）。
        let xt = Tensor::<f32>::zeros(&[1, 2, 2, 2]).unwrap();
        let scale = Tensor::<f32>::new(vec![1.0, 1.0], &[2]).unwrap();
        let bias = Tensor::<f32>::new(vec![0.0, 0.0], &[2]).unwrap();
        let mean = Tensor::<f32>::new(vec![0.0, 0.0], &[2]).unwrap();
        let var = Tensor::<f32>::new(vec![1.0, f32::NAN], &[2]).unwrap();
        let err = batch_normalization(&xt, &scale, &bias, &mean, &var, &BatchNormAttrs::default())
            .unwrap_err();
        assert!(matches!(err, OpError::InvalidBatchNormAttribute { .. }));
    }

    #[test]
    fn nan_propagates() {
        let xt = Tensor::<f32>::new(vec![f32::NAN, 1.0], &[1, 2]).unwrap();
        let scale = Tensor::<f32>::new(vec![1.0, 1.0], &[2]).unwrap();
        let bias = Tensor::<f32>::new(vec![0.0, 0.0], &[2]).unwrap();
        let mean = Tensor::<f32>::new(vec![0.0, 0.0], &[2]).unwrap();
        let var = Tensor::<f32>::new(vec![1.0, 1.0], &[2]).unwrap();
        let y = batch_normalization(&xt, &scale, &bias, &mean, &var, &BatchNormAttrs::default())
            .unwrap();
        assert!(y.get(&[0, 0]).unwrap().is_nan());
    }
}
