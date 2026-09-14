//! 基本活性化関数群（TASK-9.1b・#92）。
//!
//! `docs/spec/05-tasks.md` TASK-9.1（基本 NN モジュール〈Linear・活性化〉
//! の自作コア上再実装）のうち活性化部分を担当する。各構造体はフィール
//! ドを持たないユニット構造体で、`forward` は対応する `Var`
//! （`crate::var`）の演算メソッドを呼ぶだけの薄いラッパーに徹する
//! （REQ-9「互換 API 層は自作コアの上の薄いラッパーに徹する」）。
//!
//! **想定呼び出し元**: `compat::Sequential`（TASK-9.2・#94/#95）が
//! レイヤーの並びの一要素としてこれらを `forward` 経由で呼ぶ想定。
//! 本イシュー時点では共通 `Module` trait は未定義のため、各構造体は
//! 個別に `forward(&self, input: &Var<'t>) -> Var<'t>` を公開する
//! （trait 統一は Linear・#91 と合わせて #94/#95 側で設計する）。
//!
//! 当初のスコープは ReLU・Sigmoid・Tanh の 3 種に限定していた
//! （CrossEntropy 損失〈#191〉は log-softmax → NLL を個別オペ合成せず
//! 1 個の融合オペ〈`tape::Op::CrossEntropyLoss`〉として実装したため、
//! 独立した Softmax プリミティブは当時追加していなかった。`nn/loss.rs`
//! 冒頭 doc 参照）。イシュー #1594 で既存の行カーネル（`BackendOps::
//! softmax`／`log_softmax`）へ接続する独立した [`Softmax`]／
//! [`LogSoftmax`] を追加した（`CrossEntropyLoss` の内部 log-softmax
//! 〈`eval::softmax_along`〉はこれとは別実装のまま不変）。イシュー
//! #1713 で [`Gelu`]／[`GeluTanh`]／[`Softplus`] を追加した（`Var::gelu`／
//! `gelu_tanh`／`softplus` の薄いラッパー）。残る追加活性化（SiLU／
//! LeakyReLU／ELU／Hardswish 等。#1714）は後続イシューに委ねる。

use crate::error::AutodiffError;
use crate::var::Var;

/// ReLU（`max(x, 0)`）。`Var::relu` の薄いラッパー。
#[derive(Debug, Default, Clone, Copy)]
pub struct Relu;

impl Relu {
    pub fn forward<'t>(&self, input: &Var<'t>) -> Var<'t> {
        input.relu()
    }
}

/// シグモイド（`1 / (1 + exp(-x))`）。`Var::sigmoid` の薄いラッパー。
#[derive(Debug, Default, Clone, Copy)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn forward<'t>(&self, input: &Var<'t>) -> Var<'t> {
        input.sigmoid()
    }
}

/// 双曲線正接（`tanh(x)`）。`Var::tanh` の薄いラッパー。
#[derive(Debug, Default, Clone, Copy)]
pub struct Tanh;

impl Tanh {
    pub fn forward<'t>(&self, input: &Var<'t>) -> Var<'t> {
        input.tanh()
    }
}

/// GELU（誤差関数版）。`Var::gelu` の薄いラッパー（イシュー #1713）。
/// `Relu`/`Sigmoid`/`Tanh` と異なり `forward` は `Result` を返す
/// （`Var::gelu` の eager dispatch 契約が型付きエラーを返しうるため。
/// `Softmax`／`LogSoftmax` と同型）。
#[derive(Debug, Default, Clone, Copy)]
pub struct Gelu;

impl Gelu {
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.gelu()
    }
}

/// GELU（tanh 近似版）。`Var::gelu_tanh` の薄いラッパー（イシュー
/// #1713）。[`Gelu`] と同じ fallible 契約。
#[derive(Debug, Default, Clone, Copy)]
pub struct GeluTanh;

impl GeluTanh {
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.gelu_tanh()
    }
}

/// Softplus。`Var::softplus(beta, threshold)` の薄いラッパー（イシュー
/// #1713）。`Softmax`/`LogSoftmax` と同じ「`dim` を保持するフィールド」
/// の設計を踏襲し、`beta`／`threshold` を保持する。構築時に `Var::
/// softplus` と同じ検査（有限かつ `beta > 0`・`threshold` 有限）を行う
/// ため [`Softplus::new`] 自体が `Result` を返す（層構築の時点で早期に
/// 弾く。`nn/norm.rs::validate_eps` と同じ規律）。
#[derive(Debug, Clone, Copy)]
pub struct Softplus {
    beta: f32,
    threshold: f32,
}

impl Softplus {
    /// PyTorch `nn.Softplus` の既定値（`beta=1.0`・`threshold=20.0`）。
    pub fn new(beta: f32, threshold: f32) -> Result<Self, AutodiffError> {
        if !beta.is_finite() || beta <= 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Softplus::new: beta must be finite and positive, got {beta}"
            )));
        }
        if !threshold.is_finite() {
            return Err(AutodiffError::InvalidArgument(format!(
                "Softplus::new: threshold must be finite, got {threshold}"
            )));
        }
        Ok(Self { beta, threshold })
    }

    /// `self.beta`／`self.threshold` を用いて `Var::softplus` へ委譲する。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.softplus(self.beta, self.threshold)
    }

    /// `nn/module.rs::Module::forward_host` の `Softplus` 実装が
    /// `beta`／`threshold` を読み出すためのクレート内アクセサ
    /// （[`Softmax::dim`] と同じ理由。フィールド自体は非公開のまま）。
    pub(crate) fn beta(&self) -> f32 {
        self.beta
    }

    pub(crate) fn threshold(&self) -> f32 {
        self.threshold
    }
}

impl Default for Softplus {
    /// PyTorch `nn.Softplus` の既定値（`beta=1.0`・`threshold=20.0`）。
    /// `new` の検査（有限性・符号）を通る既知の定数のため、本番経路
    /// panic 禁止規約（`expect`／`unwrap` を避ける）に従いフィールドを
    /// 直接構築する。
    fn default() -> Self {
        Self {
            beta: 1.0,
            threshold: 20.0,
        }
    }
}

/// 行方向 softmax。`Var::softmax(dim)` の薄いラッパー（イシュー
/// #1594）。`Relu`/`Sigmoid`/`Tanh` と異なり `dim` を保持するフィールド
/// を持ち、`forward` は `dim` の軸範囲検査（`Var::softmax` 内部）により
/// `Result` を返す（構造的に失敗しうる）。
#[derive(Debug, Clone, Copy)]
pub struct Softmax {
    dim: usize,
}

impl Softmax {
    /// `dim`（softmax を適用する軸）を指定して構築する。
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// `self.dim` 軸に沿って softmax を適用する（`Var::softmax` へ委譲）。
    /// 戻り値 shape は `input` と同一（正規化は形状を変えない）。
    /// `self.dim >= input.rank()`（rank 0 を含む）は
    /// `AutodiffError::Shape` で fail-closed に拒否する。バックエンド
    /// 側 `Unsupported`（非最終軸）はホスト参照実装へ透過的に
    /// フォールバックし、それ以外のバックエンドエラーはそのまま
    /// 伝播する（`Var::softmax` の契約をそのまま引き継ぐ）。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.softmax(self.dim)
    }

    /// `nn/module.rs::Module::forward_host` の `Softmax` 実装が `dim` を
    /// 読み出すためのクレート内アクセサ（`dim` フィールド自体は
    /// カプセル化のため非公開のまま）。
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }
}

/// 行方向 log_softmax。`Var::log_softmax(dim)` の薄いラッパー（イシュー
/// #1594）。[`Softmax`] と同じ `dim` 保持・fallible 契約。
#[derive(Debug, Clone, Copy)]
pub struct LogSoftmax {
    dim: usize,
}

impl LogSoftmax {
    /// `dim`（log_softmax を適用する軸）を指定して構築する。
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// `self.dim` 軸に沿って log_softmax を適用する
    /// （`Var::log_softmax` へ委譲）。shape 不変・`self.dim` 範囲外の
    /// 拒否（`AutodiffError::Shape`）は [`Softmax::forward`] と同一。
    /// バックエンドが対応しない軸・演算（CUDA／Metal は現時点で
    /// log_softmax を最終軸含め未実装のため常にホストへフォール
    /// バックする）は `Unsupported` のときのみ透過的にホスト参照実装
    /// へ切り替わり、それ以外のバックエンドエラーは伝播する。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.log_softmax(self.dim)
    }

    /// [`Softmax::dim`] と同じ理由のクレート内アクセサ。
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    //! `nn::activation` 各構造体の `forward` が、対応する `Var` メソッド
    //! 直接呼び出しと同一の値・テープ記録を返すことを検証する
    //! （「薄いラッパー性」の担保。イシュー #92 実装計画 §5）。

    use super::*;
    use crate::eval::dense_vec;
    use crate::tape::Tape;

    #[test]
    fn relu_forward_matches_var_relu() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());
        let before = tape.len();

        let via_module = Relu.forward(&x);
        let via_var = x.relu();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        // `Tensor` は意図的に `PartialEq` を derive しないため
        // （`tensor-core::Tensor` のドキュメント参照）、稠密化した
        // データ列で値の一致を検証する。
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn sigmoid_forward_matches_var_sigmoid() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());
        let before = tape.len();

        let via_module = Sigmoid.forward(&x);
        let via_var = x.sigmoid();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        // `Tensor` は意図的に `PartialEq` を derive しないため
        // （`tensor-core::Tensor` のドキュメント参照）、稠密化した
        // データ列で値の一致を検証する。
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn tanh_forward_matches_var_tanh() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());
        let before = tape.len();

        let via_module = Tanh.forward(&x);
        let via_var = x.tanh();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        // `Tensor` は意図的に `PartialEq` を derive しないため
        // （`tensor-core::Tensor` のドキュメント参照）、稠密化した
        // データ列で値の一致を検証する。
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn gelu_forward_matches_var_gelu() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());
        let before = tape.len();

        let via_module = Gelu.forward(&x).unwrap();
        let via_var = x.gelu().unwrap();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn gelu_tanh_forward_matches_var_gelu_tanh() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());
        let before = tape.len();

        let via_module = GeluTanh.forward(&x).unwrap();
        let via_var = x.gelu_tanh().unwrap();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn softplus_forward_matches_var_softplus() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());
        let before = tape.len();

        let via_module = Softplus::new(1.0, 20.0).unwrap().forward(&x).unwrap();
        let via_var = x.softplus(1.0, 20.0).unwrap();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn softplus_new_rejects_non_positive_or_non_finite_beta_and_non_finite_threshold() {
        assert!(Softplus::new(0.0, 20.0).is_err());
        assert!(Softplus::new(-1.0, 20.0).is_err());
        assert!(Softplus::new(f32::NAN, 20.0).is_err());
        assert!(Softplus::new(f32::INFINITY, 20.0).is_err());
        assert!(Softplus::new(1.0, f32::NAN).is_err());
        assert!(Softplus::new(1.0, f32::INFINITY).is_err());
        assert!(Softplus::new(1.0, 20.0).is_ok());
    }

    #[test]
    fn softplus_default_matches_pytorch_defaults() {
        let s = Softplus::default();
        assert_eq!(s.beta(), 1.0);
        assert_eq!(s.threshold(), 20.0);
    }

    #[test]
    fn softmax_forward_matches_var_softmax() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape
            .var(&fandhe_ai_tensor_core::Tensor::new(vec![-1.0, 2.0, 0.5, 1.0], &[2, 2]).unwrap());
        let before = tape.len();

        let via_module = Softmax::new(1).forward(&x).unwrap();
        let via_var = x.softmax(1).unwrap();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn softmax_forward_rejects_axis_out_of_range() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());

        let result = Softmax::new(5).forward(&x);

        assert!(matches!(
            result,
            Err(crate::error::AutodiffError::Shape(
                fandhe_ai_tensor_core::ShapeError::AxisOutOfRange { axis: 5, rank: 1 }
            ))
        ));
    }

    #[test]
    fn log_softmax_forward_matches_var_log_softmax() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape
            .var(&fandhe_ai_tensor_core::Tensor::new(vec![-1.0, 2.0, 0.5, 1.0], &[2, 2]).unwrap());
        let before = tape.len();

        let via_module = LogSoftmax::new(1).forward(&x).unwrap();
        let via_var = x.log_softmax(1).unwrap();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }
}
