//! 参照モデル定義: `Mlp`（PyTorch の定番 MLP 分類器。イシュー #2201・親
//! #2190）。
//!
//! ## 位置づけ（重要）
//!
//! 本ファイルは `crates/facade/examples/reference_models.rs`（runnable
//! example。生成・forward・PyTorch 対応表の目視確認）と
//! `crates/facade/tests/example_mlp_mnist.rs`（統合テスト。
//! `#[path = "../examples/models/mlp.rs"] mod mlp;` で本ファイルを直接
//! 取り込む）の 2 箇所から**単独で完結する**ファイルとして読み込まれる
//! （`super::`／`crate::` による他ファイル参照はしない。テストごとに
//! 必要なファイルだけを取り込む方式のため、多少の重複は許容する
//! `docs/reference-models-decision.md` 参照）。
//!
//! `facade` の公開面（`crates/facade/src/`）は変更していない。
//! `compat::Sequential::add_*` だけを組み合わせた**利用者コード**であり、
//! `Mlp`／`LeNet` 型自体を `fandhe_ai::` から `pub use` する facade
//! 公開面拡張は、`docs/compat-api-scope.md` §5 の経路 2（未承認）に
//! 該当するため保留している（`docs/reference-models-decision.md`
//! 「保留事項」節）。
//!
//! ## PyTorch 参照定義
//!
//! ```text
//! nn.Sequential(
//!     nn.Linear(784, 256), nn.ReLU(), nn.Dropout(p),
//!     nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p),
//!     nn.Linear(128, 10),
//! )
//! ```
//!
//! `hidden_dims = [256, 128]` は親 #2190 が示す構成に対応する
//! （`Mlp::new(784, &[256, 128], 10, p)`）。
//!
//! ## 重みレイアウトの契約
//!
//! fandhe の `nn::Linear.weight` は `[in_features, out_features]`
//! （`crates/autodiff/src/nn/linear.rs` 冒頭 doc）で、PyTorch
//! `nn.Linear.weight` は `[out_features, in_features]` のため
//! **転置の関係**にある（[`Mlp::pytorch_param_map`] の
//! `transpose: true`）。bias は両者とも `[out_features]` で転置不要。

use fandhe_ai::compat::Sequential;
use fandhe_ai::{AutodiffError, Tape, Tensor, Var};

/// PyTorch チェックポイントとの層ごとの対応 1 件（[`Mlp::pytorch_param_map`]
/// の要素。イシュー #2201 の受け入れ条件 AC5「層ごとの重み対応を目視で
/// 確認できる」を満たす表現）。
#[derive(Debug, Clone)]
pub struct MlpParamMap {
    /// fandhe 側キー（`compat::Sequential::named_parameters` の形式。
    /// `"{index}.weight"` / `"{index}.bias"`）。
    pub fandhe_key: String,
    /// 対応する PyTorch `state_dict` キー（`nn.Sequential` の同じ index
    /// 規約のため fandhe 側と同一の index を使う）。
    pub pytorch_key: String,
    /// fandhe 側 shape（[`Mlp::new`] の構成値から導出。実パラメータの
    /// shape を読み返すと対応表の検証がトートロジーになるため、必ず
    /// 構成値から計算する）。
    pub fandhe_shape: Vec<usize>,
    /// 対応する PyTorch 側 shape（`weight` は `[out, in]`）。
    pub pytorch_shape: Vec<usize>,
    /// `true` の場合 `weight` は fandhe → PyTorch で転置が必要
    /// （`[in, out]` ↔ `[out, in]`）。
    pub transpose: bool,
}

/// PyTorch 定番 MLP 分類器の参照実装（`compat::Sequential` のラッパー）。
///
/// `facade` は層（`Linear`・`ReLU`・`Dropout`）を直接公開しないため
/// （到達経路は `compat::Sequential::add_*` のみ。`crates/facade/src/
/// nn/mod.rs` doc）、`Vec<Linear>` のような構造体フィールドでは組めない。
/// 代わりに `Sequential` を内部に持つラッパーとして構成する。
pub struct Mlp {
    model: Sequential,
    input_dim: usize,
    hidden_dims: Vec<usize>,
    output_dim: usize,
    dropout: f32,
}

/// [`Mlp::new`] が使う既定シード（決定的初期化。呼び出し元がシードを
/// 意識しなくても再現可能な学習曲線を得られるようにする）。
const DEFAULT_SEED: u64 = 0x4D4C505F53454544; // "MLP_SEED" の ASCII 値。

impl Mlp {
    /// `input_dim` → `hidden_dims`（各層の後に ReLU・Dropout(`dropout`)）
    /// → `output_dim` の MLP を構築する（AC1 の 4 引数形。既定シード
    /// [`DEFAULT_SEED`] で [`Mlp::with_seed`] へ委譲する）。
    pub fn new(
        input_dim: usize,
        hidden_dims: &[usize],
        output_dim: usize,
        dropout: f32,
    ) -> Result<Self, AutodiffError> {
        Self::with_seed(input_dim, hidden_dims, output_dim, dropout, DEFAULT_SEED)
    }

    /// [`Mlp::new`] のシード指定版。各 `Linear` 層には
    /// `seed.wrapping_add(層番号)` を渡し、隠れ層間で初期化系列が
    /// 重複しないようにする。
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
        // `hidden_dims` が空だとループ内の `add_dropout(dropout)` が
        // 一度も呼ばれず `Dropout::new` の検証（有限性・[0, 1] 範囲）を
        // 経由しないため、コンストラクタの入口で常に検証する
        // （codex-review 指摘・イシュー #2201 PR #2320）。
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

        Ok(Mlp {
            model,
            input_dim,
            hidden_dims: hidden_dims.to_vec(),
            output_dim,
            dropout,
        })
    }

    /// 外部 `Tape` 上で forward を計算する（`compat::Sequential::forward`
    /// への薄い委譲。学習ループ・grad check から使う）。
    pub fn forward<'t>(&self, tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.model.forward(tape, x)
    }

    /// 推論の入口（`compat::Sequential::predict` への薄い委譲）。
    pub fn predict(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError> {
        self.model.predict(x)
    }

    /// 内部 `Sequential` への参照（`compile`／`fit`／`evaluate`／`eval`／
    /// `train`／`named_parameters` 等、`compat::Sequential` の学習系
    /// API をそのまま呼びたい呼び出し元向け）。
    pub fn sequential(&self) -> &Sequential {
        &self.model
    }

    /// [`Mlp::sequential`] の可変版（`compile`・`train`・`eval` 用）。
    ///
    /// # 契約（重要）
    ///
    /// [`Mlp::pytorch_param_map`] は [`Mlp::new`]／[`Mlp::with_seed`] の
    /// 構成値から重み対応表を再計算する（実パラメータからの逆算による
    /// トートロジー化を避けるため）。この可変参照経由で内部
    /// `Sequential` の層構成そのものを差し替えると、対応表が実際の
    /// パラメータと不整合になりうる（codex-review 指摘・イシュー #2201
    /// PR #2320）。[`Mlp::pytorch_param_map`] は呼び出しのたびに
    /// `named_parameters()` と突き合わせて検証するため、差し替え後の
    /// 呼び出しは `Err` になる。本メソッドは `compile`・`train`・`eval`
    /// 等のモード切替・学習系 API 呼び出し用に限定して使うこと。
    pub fn sequential_mut(&mut self) -> &mut Sequential {
        &mut self.model
    }

    /// PyTorch 参照定義との層ごとの重み対応表（AC5）。`fandhe_shape` は
    /// [`Mlp::new`] の構成値（`input_dim`／`hidden_dims`／`output_dim`）
    /// から計算し、実パラメータの shape は読み返さない（対応表の検証を
    /// トートロジーにしないため）。そのうえで [`Sequential::named_parameters`]
    /// と突き合わせ、キー集合・shape が完全一致することを検証する
    /// （[`Mlp::sequential_mut`] 経由で内部構成が差し替えられていた
    /// 場合に不整合を検出するため。codex-review 指摘・イシュー #2201
    /// PR #2320）。
    pub fn pytorch_param_map(&self) -> Result<Vec<MlpParamMap>, AutodiffError> {
        let mut dims = Vec::with_capacity(self.hidden_dims.len() + 2);
        dims.push(self.input_dim);
        dims.extend(self.hidden_dims.iter().copied());
        dims.push(self.output_dim);

        let mut out = Vec::new();
        for (layer_no, w) in dims.windows(2).enumerate() {
            let (in_f, out_f) = (w[0], w[1]);
            // Sequential 上の index: 隠れ層ごとに Linear/ReLU/Dropout の
            // 3 層を消費する（`Mlp::with_seed` の構築順と一致させる）。
            let index = layer_no * 3;
            out.push(MlpParamMap {
                fandhe_key: format!("{index}.weight"),
                pytorch_key: format!("{index}.weight"),
                fandhe_shape: vec![in_f, out_f],
                pytorch_shape: vec![out_f, in_f],
                transpose: true,
            });
            out.push(MlpParamMap {
                fandhe_key: format!("{index}.bias"),
                pytorch_key: format!("{index}.bias"),
                fandhe_shape: vec![out_f],
                pytorch_shape: vec![out_f],
                transpose: false,
            });
        }

        let actual = self.model.named_parameters();
        if actual.len() != out.len() {
            return Err(AutodiffError::InvalidArgument(format!(
                "Mlp::pytorch_param_map: 対応表のエントリ数（{}）が実パラメータ数\
                 （{}）と一致しない（sequential_mut() 経由で内部構成が\
                 差し替えられた可能性がある）",
                out.len(),
                actual.len()
            )));
        }
        for entry in &out {
            let found = actual
                .iter()
                .find(|(key, _)| *key == entry.fandhe_key)
                .ok_or_else(|| {
                    AutodiffError::InvalidArgument(format!(
                        "Mlp::pytorch_param_map: キー '{}' が実パラメータに\
                         存在しない（sequential_mut() 経由で内部構成が\
                         差し替えられた可能性がある）",
                        entry.fandhe_key
                    ))
                })?;
            if found.1.shape() != entry.fandhe_shape.as_slice() {
                return Err(AutodiffError::InvalidArgument(format!(
                    "Mlp::pytorch_param_map: キー '{}' の shape が対応表\
                     （{:?}）と実パラメータ（{:?}）で不一致\
                     （sequential_mut() 経由で内部構成が差し替えられた\
                     可能性がある）",
                    entry.fandhe_key,
                    entry.fandhe_shape,
                    found.1.shape()
                )));
            }
        }

        Ok(out)
    }

    /// 構成値（`dropout` 確率）を返す（example・テストの表示用）。
    pub fn dropout(&self) -> f32 {
        self.dropout
    }
}
