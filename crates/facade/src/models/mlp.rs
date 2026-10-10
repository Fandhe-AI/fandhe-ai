//! 参照モデル `Mlp`（PyTorch の定番 MLP 分類器。イシュー #2201 の examples 実装を
//! #2974 で `fandhe_ai::models` として公開したもの）の実体。
//!
//! 公開パスは `models/mod.rs` の `pub use` のみ（本モジュール自体は非公開）。
//! `compat::Sequential::add_*` だけを組み合わせた薄いラッパーで、新しいカーネルや
//! 保存経路は持たない。保存・読み込みは [`Mlp::sequential`] 経由で既存の
//! `compat::save_model`／`compat::load_model` を使う。
//!
//! # PyTorch 参照定義
//!
//! ```text
//! nn.Sequential(
//!     nn.Linear(784, 256), nn.ReLU(), nn.Dropout(p),
//!     nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p),
//!     nn.Linear(128, 10),
//! )
//! ```
//!
//! `Sequential` 上の index は隠れ層ごとに Linear／ReLU／Dropout の 3 層を消費する。
//! fandhe の `Linear.weight` は `[in, out]` で PyTorch の `[out, in]` とは転置の関係
//! （bias は同一）。PyTorch との層対応表は examples 側
//! （`crates/facade/examples/models/mlp.rs`）に置き、facade の公開面には含めない。

use crate::compat::Sequential;
use crate::{AutodiffError, Tape, Tensor, Var};

/// PyTorch 定番 MLP 分類器の参照実装（`compat::Sequential` のラッパー）。
///
/// facade は層を直接公開しないため（到達経路は `compat::Sequential::add_*` のみ）、
/// `Sequential` を内部に持つラッパーとして構成する。
pub struct Mlp {
    model: Sequential,
    dropout: f32,
}

/// [`Mlp::new`] が使う既定シード（決定的初期化。呼び出し元がシードを意識しなくても
/// 再現可能な学習曲線を得るため）。
const DEFAULT_SEED: u64 = 0x4D4C505F53454544; // "MLP_SEED" の ASCII 値。

impl Mlp {
    /// `input_dim` → `hidden_dims`（各層の後に ReLU・Dropout(`dropout`)）→ `output_dim`
    /// の MLP を構築する。既定シードで [`Mlp::with_seed`] へ委譲する。
    ///
    /// # Errors
    ///
    /// 次元に 0 を含む場合、または `dropout` が非有限・`[0, 1]` 範囲外の場合は
    /// `AutodiffError::InvalidArgument`。
    pub fn new(
        input_dim: usize,
        hidden_dims: &[usize],
        output_dim: usize,
        dropout: f32,
    ) -> Result<Self, AutodiffError> {
        Self::with_seed(input_dim, hidden_dims, output_dim, dropout, DEFAULT_SEED)
    }

    /// [`Mlp::new`] のシード指定版。各 `Linear` 層には `seed.wrapping_add(層番号)` を渡し、
    /// 隠れ層間で初期化系列が重複しないようにする。
    pub fn with_seed(
        input_dim: usize,
        hidden_dims: &[usize],
        output_dim: usize,
        dropout: f32,
        seed: u64,
    ) -> Result<Self, AutodiffError> {
        if input_dim == 0 || output_dim == 0 || hidden_dims.contains(&0) {
            return Err(AutodiffError::InvalidArgument(
                "Mlp::new: input_dim・hidden_dims の各要素・output_dim はいずれも 0 \
                 より大きい必要がある"
                    .to_string(),
            ));
        }
        // `hidden_dims` が空だとループ内の `add_dropout` が一度も呼ばれず `Dropout::new`
        // の検証を経由しないため、コンストラクタの入口で常に検証する（#2201 PR #2320）。
        if !dropout.is_finite() || !(0.0..=1.0).contains(&dropout) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Mlp::new: dropout must be finite and in [0, 1], got {dropout}"
            )));
        }

        let mut model = Sequential::new();
        let mut prev = input_dim;
        for (i, &h) in hidden_dims.iter().enumerate() {
            model = model
                .add_linear(prev, h, seed.wrapping_add(i as u64))?
                .add_relu()
                .add_dropout(dropout)?;
            prev = h;
        }
        model = model.add_linear(
            prev,
            output_dim,
            seed.wrapping_add(hidden_dims.len() as u64),
        )?;

        Ok(Mlp { model, dropout })
    }

    /// 外部 `Tape` 上で forward を計算する（`Sequential::forward` への薄い委譲。
    /// 学習ループ・grad check から使う）。
    pub fn forward<'t>(&self, tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.model.forward(tape, x)
    }

    /// 推論の入口（`Sequential::predict` への薄い委譲）。
    pub fn predict(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError> {
        self.model.predict(x)
    }

    /// 内部 `Sequential` への参照（`compile`／`fit`／`evaluate`／`named_parameters`／
    /// `save_model` 等の呼び出し元向け）。
    pub fn sequential(&self) -> &Sequential {
        &self.model
    }

    /// [`Mlp::sequential`] の可変版（`compile`・`train`・`eval` 用）。
    ///
    /// # 契約
    ///
    /// 層構成そのものの差し替えは想定しない。差し替えると examples 側の PyTorch 対応表
    /// 関数が実パラメータとの突き合わせで `Err` を返す。モード切替・学習系 API の
    /// 呼び出しに限定して使うこと。
    pub fn sequential_mut(&mut self) -> &mut Sequential {
        &mut self.model
    }

    /// 構成値（`dropout` 確率）を返す。
    pub fn dropout(&self) -> f32 {
        self.dropout
    }
}
