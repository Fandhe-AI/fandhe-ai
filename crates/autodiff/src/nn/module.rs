//! 共通 `Module` trait（TASK-9.2a・#95）。
//!
//! `Sequential`（当時 `fandhe_ai_autodiff::compat::sequential`。TASK-9.4・#411 で
//! `fandhe_ai::compat::sequential` へ移設）がレイヤーの列を
//! `Vec<Box<dyn Module>>` として保持し、種類の異なる `nn` の部品
//! （`Linear`・活性化関数）を統一シグネチャで呼べるようにするための
//! 最小 trait。`nn/mod.rs` が「共通 `Module` trait の定義は
//! `compat::Sequential` 設計時に確定する」としていた点を本イシューで
//! 確定する。
//!
//! シグネチャに `tape: &'t Tape` を含める理由: `Linear` は
//! `Linear::bind(&tape)` でそのステップの葉ノードを毎回登録してから
//! でないと forward できない（`nn/linear.rs` の `Tape` ライフサイクル
//! 節参照）。活性化関数側は `tape` を使わないが、`Box<dyn Module>` を
//! 均一に扱うため同じ引数を受け取る。この「毎呼び出しで葉ノードを
//! 登録し直す」契約は推論・1 ステップ forward 用である。学習（勾配
//! 取得・パラメータ更新）は #294（`compat::Sequential::bind`・
//! `compat/sequential.rs` の `SequentialVars`）で対応済み: 下記
//! `as_linear`/`as_linear_mut` が `Sequential` から学習可能パラメータ
//! （`Linear`）を層順に取り出すためのダウンキャストフックを提供する。

use std::collections::{HashMap, HashSet};

use crate::error::AutodiffError;
use crate::eval;
use crate::nn::activation::{
    Elu, Gelu, GeluTanh, Hardswish, LeakyRelu, LogSoftmax, Relu, Sigmoid, Silu, Softmax, Softplus,
    Tanh,
};
use crate::nn::attention::MultiheadAttention;
use crate::nn::batch_norm::{
    BATCH_NORM_1D_RANKS, BATCH_NORM_2D_RANKS, BatchNorm1d, BatchNorm2d, BatchNormCore,
};
use crate::nn::conv::{Conv1d, Conv2d};
use crate::nn::embedding::Embedding;
use crate::nn::linear::Linear;
use crate::nn::norm::{LayerNorm, RmsNorm};
use crate::nn::pooling::{
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AvgPool1d, AvgPool2d, MaxPool1d, MaxPool2d,
};
use crate::tape::Tape;
use crate::var::Var;
use fandhe_ai_tensor_core::{
    BackendError, BackendOps, ShapeError, Tensor, adaptive_pool2d_out_shape, batch_norm_layout,
    broadcast_shape, gemm_out_shape, pool2d_out_shape, reduce_out_shape, require_same_shape,
    row_norm_layout,
};

/// [`Module::named_parameters`] の実装が、子 `Module`（`Linear` 等）を
/// 内包する複合層（`MultiheadAttention`・`Rnn`／`Lstm`／`Gru`）で名前へ
/// 接頭辞（`"q_proj."` 等）を連結するための共通ヘルパー（イシュー
/// #1758）。重複実装を避けるため `module.rs` 側に置き、`attention.rs`・
/// `rnn.rs` から使う。
pub(crate) fn prefixed<'a>(
    prefix: &str,
    inner: Vec<(String, &'a Tensor<f32>)>,
) -> Vec<(String, &'a Tensor<f32>)> {
    inner
        .into_iter()
        .map(|(name, tensor)| (format!("{prefix}.{name}"), tensor))
        .collect()
}

/// [`Module::set_parameter`] の実装が、子 `Module` を内包する複合層
/// （`MultiheadAttention`・`Rnn`／`Lstm`／`Gru`・`ModuleList`）で
/// `"{接頭辞}.{子の名前}"` から接頭辞を剥がして子へ再帰させるための
/// 共通ヘルパー（イシュー #1752）。[`prefixed`] の逆演算。区切りが
/// `'.'` でない、または接頭辞が一致しない場合は `None`（呼び出し元は
/// 「対応する子がない」として扱う）。
pub(crate) fn strip_child_prefix<'a>(name: &'a str, prefix: &str) -> Option<&'a str> {
    let rest = name.strip_prefix(prefix)?;
    rest.strip_prefix('.')
}

/// `nn` の部品（層・活性化関数）に共通の forward シグネチャ。
pub trait Module {
    /// このステップの `tape` 上で 1 回分の forward を計算する。
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError>;

    /// `tape`（葉ノード登録・演算記録）を経由せず、`ops` を直接呼んで
    /// 1 回分の forward をホスト常駐 `Tensor` で計算する（イシュー
    /// #1028・`docs/inference-forward-fixed-cost-design.md` §3.1「段階
    /// A」）。推論専用の tape 不要経路（`compat::Sequential::predict`
    /// が呼ぶ）が、[`Self::forward`] と同じ演算列・同じ丸め（bit
    /// 完全一致）を保ちつつ、`Tape::var` の葉クローン（`Linear` の
    /// `weight`/`bias` を毎呼び出しで clone する固定費。`nn/linear.rs`
    /// の `Linear::bind` 参照）とノード記録のアロケーションを回避する
    /// ために追加した。
    ///
    /// # デフォルト実装
    ///
    /// `fandhe-ai-autodiff` は crates.io 公開クレートであり
    /// （`docs/crates-io-naming-decision.md`）、本メソッドは非破壊拡張
    /// （デフォルトメソッド追加。外部実装者の既存 `impl Module` を壊さ
    /// ない）とする。既定は [`BackendError::Unsupported`] を返す
    /// fail-safe（本クレート内 15 実装〈`Linear`・`Relu`・`Sigmoid`・
    /// `Tanh`・`RmsNorm`・`LayerNorm`・`Softmax`・`LogSoftmax`・`Gelu`・
    /// `GeluTanh`・`Softplus`・`Silu`・`Hardswish`・`LeakyRelu`・`Elu`
    /// （イシュー #1714）〉はいずれも
    /// このデフォルトを
    /// オーバーライドする。呼び出し元
    /// が独自の `Module` 実装をこの経路で使う場合、`Unsupported` を
    /// フォールバックの合図として扱うこと）。
    fn forward_host(
        &self,
        _ops: &dyn BackendOps,
        _input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        Err(AutodiffError::Backend(BackendError::Unsupported(
            "Module::forward_host: default fail-safe (tape-free forward not implemented for \
             this Module)"
                .into(),
        )))
    }

    /// [`Self::forward_host`] が常に [`BackendError::Unsupported`] を
    /// 返す層かどうかを、実際には呼ばずに事前判定するフック（イシュー
    /// #1760・Cursor Bugbot 指摘是正: `compat::Sequential::predict` の
    /// tape 不要経路が層を順に `forward_host` していき、途中の層
    /// （`Embedding`／`MultiheadAttention` 等）で初めて `Unsupported`
    /// に当たって旧経路へフォールバックすると、それより手前の層で
    /// 既に発生した副作用——`Dropout` の RNG 消費・`BatchNorm` の
    /// running stats 更新（学習モード時。`RefCell` 越しに `&self` から
    /// 更新される）——が旧経路の再実行で二重に発生してしまう。
    ///
    /// 既定は `true`（[`Self::forward_host`] のデフォルト実装が
    /// fail-safe で `Unsupported` を返すのとは非対称だが、既定 `true`
    /// はこのクレート内の大多数の層——`forward_host` を実装済みの
    /// 層——の実態と一致する。`forward_host` を未実装のまま残す層
    /// （[`Embedding`]・[`MultiheadAttention`]）のみが `false` へ
    /// オーバーライドし、常に `Unsupported` を返すことを事前に
    /// 申告する）。
    ///
    /// [`crate::compat::Sequential::predict`]（`fandhe-ai-facade`）が
    /// 本メソッドで全層を事前判定し、1 層でも `false` を返す場合は
    /// tape 不要経路を**一切実行せず**旧経路（`Tape` 経由）から
    /// 開始する（部分的な副作用の発生自体を構造的に防ぐ）。動的な
    /// 入力依存で `Unsupported` を返す層（本クレート内には存在しない）
    /// は本メソッドの対象外——既定 `true` のまま実行時に
    /// `Unsupported` を返した場合の副作用二重化は本イシューの
    /// スコープ外として残る。
    fn supports_forward_host(&self) -> bool {
        true
    }

    /// 学習可能パラメータを持つ層（現状 `Linear` のみ）への読み取り
    /// アクセスフック。既定実装は `None`（活性化関数など無状態の層は
    /// オーバーライドしない）。`std::any::Any` による動的ダウンキャスト
    /// ではなくこの明示的フックを選ぶ理由: `compat` 層が対象とするレイヤー
    /// 集合は `docs/compat-api-scope.md` §1 で 3 種（Linear・
    /// ReLU/Sigmoid/Tanh）に閉じており、種類を増やすたびに `Any` の
    /// ダウンキャスト先を推測する曖昧さを避け、対応する層がここに
    /// 列挙されているかどうかで閉集合であることをコードとして保つため
    /// （#294。呼び出し元は `compat::Sequential::bind`/
    /// `trainable_parameters`/`apply_parameters`）。
    fn as_linear(&self) -> Option<&Linear> {
        None
    }

    /// [`Module::as_linear`] の可変版。`compat::Sequential::apply_parameters`
    /// が optimizer 更新後の `Tensor<f32>` を層へ書き戻す入口として使う。
    fn as_linear_mut(&mut self) -> Option<&mut Linear> {
        None
    }

    /// [`Module::as_linear`] と同型の明示フック（イシュー #1770・親
    /// #1645）。`compat::Sequential` の学習経路（`bind`／
    /// `trainable_parameters`／`apply_parameters` 等）が `Conv2d` 層を
    /// 認識するために使う。既定 `None`（他の層はオーバーライドしない）。
    fn as_conv2d(&self) -> Option<&Conv2d> {
        None
    }

    /// [`Module::as_conv2d`] の可変版（[`Module::as_linear_mut`] と同型）。
    fn as_conv2d_mut(&mut self) -> Option<&mut Conv2d> {
        None
    }

    /// [`Module::as_linear`] と同型の明示フック（イシュー #1770）。
    /// `Conv1d` 層向け。既定 `None`。
    fn as_conv1d(&self) -> Option<&Conv1d> {
        None
    }

    /// [`Module::as_conv1d`] の可変版。
    fn as_conv1d_mut(&mut self) -> Option<&mut Conv1d> {
        None
    }

    /// [`Module::as_linear`] と同型の明示フック（イシュー #1760・親
    /// #1618）。`compat::Sequential` の学習経路（`bind`／
    /// `trainable_parameters`／`apply_parameters` 等）が `LayerNorm` 層
    /// を認識するために使う。既定 `None`。
    fn as_layer_norm(&self) -> Option<&LayerNorm> {
        None
    }

    /// [`Module::as_layer_norm`] の可変版。
    fn as_layer_norm_mut(&mut self) -> Option<&mut LayerNorm> {
        None
    }

    /// [`Module::as_layer_norm`] と同型の明示フック（イシュー #1760）。
    /// `RmsNorm` 層向け。既定 `None`。
    fn as_rms_norm(&self) -> Option<&RmsNorm> {
        None
    }

    /// [`Module::as_rms_norm`] の可変版。
    fn as_rms_norm_mut(&mut self) -> Option<&mut RmsNorm> {
        None
    }

    /// [`Module::as_layer_norm`] と同型の明示フック（イシュー #1760）。
    /// `BatchNorm1d` 層向け。既定 `None`。
    fn as_batch_norm1d(&self) -> Option<&BatchNorm1d> {
        None
    }

    /// [`Module::as_batch_norm1d`] の可変版。
    fn as_batch_norm1d_mut(&mut self) -> Option<&mut BatchNorm1d> {
        None
    }

    /// [`Module::as_layer_norm`] と同型の明示フック（イシュー #1760）。
    /// `BatchNorm2d` 層向け。既定 `None`。
    fn as_batch_norm2d(&self) -> Option<&BatchNorm2d> {
        None
    }

    /// [`Module::as_batch_norm2d`] の可変版。
    fn as_batch_norm2d_mut(&mut self) -> Option<&mut BatchNorm2d> {
        None
    }

    /// [`Module::as_layer_norm`] と同型の明示フック（イシュー #1760）。
    /// `Embedding` 層向け（`nn/embedding.rs` モジュール doc「`Module`
    /// trait は実装しない（確定判断）」を本イシューで解消したことに
    /// 伴い追加）。既定 `None`。
    fn as_embedding(&self) -> Option<&Embedding> {
        None
    }

    /// [`Module::as_embedding`] の可変版。
    fn as_embedding_mut(&mut self) -> Option<&mut Embedding> {
        None
    }

    /// [`Module::as_layer_norm`] と同型の明示フック（イシュー #1760）。
    /// `MultiheadAttention` 層向け（`nn/attention.rs` モジュール doc
    /// 「`Module` trait との関係」で「`compat::Sequential::add_*` が
    /// 対応する層集合に含まれていない」としていた記述を本イシューで
    /// 解消したことに伴い追加）。既定 `None`。
    fn as_multihead_attention(&self) -> Option<&MultiheadAttention> {
        None
    }

    /// [`Module::as_multihead_attention`] の可変版。
    fn as_multihead_attention_mut(&mut self) -> Option<&mut MultiheadAttention> {
        None
    }

    /// この層が `ReLU` かどうか（イシュー #1044・`docs/kernel-fusion.md`
    /// §2.2「学習経路への結線」）。`as_linear` と同じ明示列挙方式
    /// （`docs/compat-api-scope.md` §1 の閉集合維持。`Any` ダウンキャスト
    /// は使わない）で、`fandhe_ai_facade::compat::sequential::Sequential`
    /// が forward 中に「次層が `ReLU` か」を先読みし、`Linear` 層を
    /// `LinearVars::forward_with_activation(input, Activation::Relu)`
    /// （1 ノード・1 カーネル起動）へ結線して `ReLU` 層自体をスキップ
    /// するかどうかを判定する。既定は `false`（`Linear`／
    /// `Sigmoid`／`Tanh` はオーバーライドしない。`Sigmoid`／`Tanh` は
    /// `BackendOps::gemm_bias_act` の `Activation` に対応する variant を
    /// 持たないため融合対象外）。
    fn as_relu(&self) -> bool {
        false
    }

    /// この層が Pooling（[`MaxPool2d`]／[`MaxPool1d`]／[`AvgPool2d`]／
    /// [`AvgPool1d`]／[`AdaptiveAvgPool2d`]／[`AdaptiveAvgPool1d`]）
    /// かどうか（イシュー #1957）。`as_relu` と同じ bool フック方式
    /// （`docs/compat-api-scope.md` §1 の閉集合維持。用途が
    /// `fandhe_ai_facade::compat::sequential::Sequential::
    /// contains_resident_unsupported_layer` の真偽判定のみであり
    /// 型付き参照を必要とする消費者が無いため、`as_conv2d` 等と異なり
    /// `Option<&T>` ではなく bool を返す）。既定は `false`
    /// （Pooling 6 型のみオーバーライドする）。
    fn is_pooling(&self) -> bool {
        false
    }

    /// `train()`／`eval()`（PyTorch `Module.training` 相当。イシュー
    /// #1758・`docs/spec/04-requirements.md` REQ-9 2026-09-12 追記
    /// Tier 1「Module の train／eval」）。**既定は no-op**。
    ///
    /// # 契約（無状態モジュールはモードを保持しない）
    ///
    /// 本クレート内実装（`Linear`・活性化関数群・`RmsNorm`／
    /// `LayerNorm`・`Softmax`／`LogSoftmax`・`MultiheadAttention`・
    /// `Rnn`／`Lstm`／`Gru`）はいずれも train／eval で挙動が変わらない
    /// ため、このデフォルト（no-op）のままオーバーライドしない。
    /// **モードの正はコンテナ**（`fandhe_ai_facade::compat::sequential::
    /// Sequential`・`crate::nn::container::ModuleList`／`Sequential`
    /// 〈イシュー #1759 で実装済み〉）**が保持するフラグ**である。
    /// [`crate::nn::Dropout`]（イシュー #1603）がこの契約に従い
    /// `set_training`／`training` を実際にオーバーライドする**最初の
    /// 実装**である（`nn/dropout.rs` モジュール doc 参照）。今後
    /// BatchNorm（#1608 配下）等のモード依存層を追加する際も、本
    /// メソッドと [`Module::training`] の両方を必ずオーバーライドし、
    /// 自層のフィールド（例: `Cell<bool>`）へ実際に保持すること
    /// （既定のまま放置すると、コンテナ側の `set_training` 呼び出しが
    /// 当該層へ伝播しても無視されてしまう）。
    fn set_training(&mut self, _training: bool) {}

    /// 現在のモード。**既定 `true`**（PyTorch `Module.training` の
    /// 初期値と揃える）。[`Module::set_training`] と同じ契約
    /// （無状態モジュールは保持しない・モード依存層は必ずオーバーライド
    /// する）に従う。
    fn training(&self) -> bool {
        true
    }

    /// この層（および子を持つ場合は子を含む）が公開する学習可能
    /// パラメータの「名前, 参照」列（PyTorch `Module.named_parameters()`
    /// 相当。イシュー #1758）。
    ///
    /// # 命名契約
    ///
    /// 名前の正は PyTorch の packed 命名（`in_proj_weight`・
    /// `weight_ih_l0` 等）ではなく、**本クレートの struct フィールド名／
    /// accessor 名**とする（`docs/compat-api-scope.md` §1.2 該当行・
    /// イシュー #1752〈state_dict／load_state_dict〉が直列化する
    /// compat 契約の一部となるため、実装ごとに一貫させる）。列挙順は
    /// 「登録順＝ weight → bias」（[`fandhe_ai_facade::compat::sequential::
    /// Sequential::trainable_parameters`] と共通の順序契約。同 struct
    /// を参照）。`Option` パラメータは `Some` のときのみ列挙する。
    /// 既定は空 `Vec`（無状態モジュール向け）。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        Vec::new()
    }

    /// [`Module::named_parameters`] が返す名前に対応するパラメータを
    /// 書き戻す（イシュー #1752・[`Module::load_state_dict`] の基盤）。
    ///
    /// # 契約
    ///
    /// - **shape 保存置換のみ**（`compat::Sequential::apply_parameters`
    ///   の #426 契約と同じ）。既存パラメータと `value.shape()` が
    ///   完全一致しなければ `AutodiffError::Shape`（`tensor-core::
    ///   ShapeError::ShapeMismatch` をラップ。本クレート内実装
    ///   〈`Linear`・`RmsNorm`・`LayerNorm`・`RnnCell`・`LstmCell`・
    ///   `GruCell`〉はいずれもこの variant を返す。codex-review 指摘・
    ///   PR #1875 是正: 旧 doc は誤って `InvalidArgument` と記載して
    ///   いた）。
    /// - `name` に該当するパラメータが存在しない場合（未知の名前、
    ///   または対象が `Option` で現在 `None` の場合を含む）は
    ///   `AutodiffError::InvalidArgument`（fail-closed。
    ///   `.claude/rules/security.md` A03）。
    /// - 非 contiguous テンソルの正規化は行わない（`apply_parameters`／
    ///   `Linear::from_parameters`／`Tape::var` と同じくそのまま保持
    ///   する）。
    ///
    /// # 既定実装
    ///
    /// [`Module::named_parameters`] と同じ非破壊拡張方針（デフォルト
    /// メソッド）。既定は `Err`（無状態モジュール向けの fail-safe）。
    ///
    /// # オーバーライド指針
    ///
    /// [`Module::named_parameters`] をオーバーライドする層は、
    /// **必ず本メソッドも対でオーバーライドすること**。片方だけだと
    /// [`Module::load_state_dict`] が該当パラメータの名前を認識できず
    /// `Err` になる。
    fn set_parameter(&mut self, name: &str, _value: Tensor<f32>) -> Result<(), AutodiffError> {
        Err(AutodiffError::InvalidArgument(format!(
            "Module::set_parameter: no parameter named `{name}` (default fail-safe; this \
             Module does not override set_parameter)"
        )))
    }

    /// [`Module::named_parameters`] のキー付きビュー（PyTorch
    /// `Module.state_dict()` 相当。イシュー #1752）。各値は `clone`
    /// される。順序契約は持たない（`HashMap` のため。列挙順の正は
    /// [`Module::named_parameters`] 側に残す）。
    fn state_dict(&self) -> HashMap<String, Tensor<f32>> {
        self.named_parameters()
            .into_iter()
            .map(|(name, tensor)| (name, tensor.clone()))
            .collect()
    }

    /// [`Module::state_dict`] の逆（PyTorch
    /// `Module.load_state_dict(state_dict, strict=True)` 相当。イシュー
    /// #1752）。**strict 限定**（PyTorch `strict=False` に相当する
    /// 部分ロードは未実装。必要になれば別 API として設計する）。
    ///
    /// # アトミック性（two-pass + ベストエフォート・ロールバック。
    /// `compat::Sequential::apply_parameters` と同型の意図・#294／#426。
    /// codex-review 指摘・PR #1875 是正）
    ///
    /// パス 1（検証のみ・無変更）で (a) `state` のキー集合と
    /// `named_parameters()` の名前集合の**完全一致**（欠落・余剰キーを
    /// それぞれ昇順で列挙し `AutodiffError::InvalidArgument`）、
    /// (b) 各キーの `shape()` 完全一致を検査する。ここまでは
    /// `apply_parameters` と同じく代入前の純粋な検証で、失敗しても
    /// 状態は一切変化しない。
    ///
    /// `apply_parameters` と異なり、本メソッドはここから先を
    /// 「置換前に全件を検証し尽くしてから代入する」形で完結**できない**:
    /// `apply_parameters` は `Linear::from_parameters` が返す**新しい
    /// `Linear` を丸ごと構築してから最後に代入する**ため代入自体が
    /// 失敗し得ないが、[`Module::set_parameter`] は任意の外部 `Module`
    /// 実装がオーバーライドしうる仮想呼び出しであり、パス 1 が検査した
    /// `named_parameters()` の名前・shape と実際に一致した書き戻しを
    /// 行うかは実装側の契約遵守に依存する（[`Module::set_parameter`]
    /// doc「オーバーライド指針」）。とくに [`crate::nn::container::
    /// ModuleList`] のような複合層に、`named_parameters` はオーバー
    /// ライドしたが `set_parameter` を対でオーバーライドし損ねた外部
    /// `Module`（既定実装のまま常に `Err` を返す）が混在する場合、
    /// パス 1 は通過するのにパス 2 の途中で `Err` になりうる。
    ///
    /// そこでパス 2 は次の手順でベストエフォートの原子性を担保する:
    ///
    /// 1. パス 1 で得た `named_parameters()` の現在値を丸ごと `clone`
    ///    して `snapshot`（ロールバック用の複製）を保持する。
    /// 2. `state` の走査順を **キー名の昇順**へ固定してから
    ///    [`Module::set_parameter`] を順に呼ぶ（`HashMap` の走査順の
    ///    まま適用すると、失敗時にどこまで適用済みかが実行のたびに
    ///    変わり再現できない）。
    /// 3. 途中の呼び出しが `Err` を返したら、**それまでに適用済みの
    ///    キーを逆順に** `snapshot` の値で `set_parameter` へ書き戻す
    ///    （後勝ちの上書きを避けるため適用順の逆順で戻す）。ロール
    ///    バックの各呼び出しは「直前に成功したのと同じ名前・同じ
    ///    shape」を渡すだけなので通常は成功する。
    /// 4. ロールバック自体が失敗した場合（外部実装が状態を持つ・
    ///    非決定的に失敗する等の想定外のケース）は、それを黙って
    ///    握り潰さず、**モデルが部分適用のまま残っている可能性**を
    ///    明示した `AutodiffError::InvalidArgument` を返す
    ///    （`.claude/rules/security.md` A08。fail-closed）。
    ///
    /// 以上により、**すべての `Module` 実装が「パス 1 で受理された
    /// `(name, shape)` の組を渡された `set_parameter` は、直後の再呼び
    /// 出しでも成功する」という契約に従う限り**、途中で失敗しても
    /// 呼び出し前の状態が維持される。この契約自体を破る実装（例:
    /// 副作用として一度きりしか成功しない `set_parameter`）に対しては
    /// 完全な原子性を構造的に保証できない（`Self: Clone` を要求せず
    /// 任意の `Module` トレイトオブジェクトに対応するための限界。
    /// `apply_parameters` が `Linear` という具象型に対してのみ実現
    /// できている「代入前に新オブジェクトを完成させる」方式は、
    /// `dyn Module` の汎用デフォルト実装としては再現できない）。
    ///
    /// ピークメモリは通常の約 2 倍（`state` + `snapshot`）になるが、
    /// [`Module::state_dict`] も同様に全パラメータを `clone` するため
    /// 既存契約からの追加コストではない。
    fn load_state_dict(
        &mut self,
        state: HashMap<String, Tensor<f32>>,
    ) -> Result<(), AutodiffError> {
        // パス 1（検証のみ）。ロールバック用に shape だけでなく値
        // そのものを丸ごと保持する（`&self` 借用の戻り値を所有権付きの
        // `(String, Tensor<f32>)` 列へ写し取ってから借用を解放する。
        // パス 2 が `&mut self` を要するため、`&Tensor` 借用を持ち越す
        // と借用検査に落ちる）。
        let snapshot: HashMap<String, Tensor<f32>> = self
            .named_parameters()
            .into_iter()
            .map(|(name, tensor)| (name, tensor.clone()))
            .collect();

        let expected_names: HashSet<&str> = snapshot.keys().map(String::as_str).collect();
        let state_names: HashSet<&str> = state.keys().map(|k| k.as_str()).collect();

        let mut missing: Vec<&str> = expected_names.difference(&state_names).copied().collect();
        missing.sort_unstable();
        if !missing.is_empty() {
            return Err(AutodiffError::InvalidArgument(format!(
                "Module::load_state_dict: missing keys: {missing:?}"
            )));
        }
        let mut unexpected: Vec<&str> = state_names.difference(&expected_names).copied().collect();
        unexpected.sort_unstable();
        if !unexpected.is_empty() {
            return Err(AutodiffError::InvalidArgument(format!(
                "Module::load_state_dict: unexpected keys: {unexpected:?}"
            )));
        }
        for (name, old_tensor) in &snapshot {
            // 直上の集合検査でキーは必ず存在するため `state.get` は
            // `Some` を返す（`unwrap`/`expect` は使わず `if let` で
            // 安全にアクセスする）。
            if let Some(tensor) = state.get(name.as_str())
                && tensor.shape() != old_tensor.shape()
            {
                return Err(AutodiffError::InvalidArgument(format!(
                    "Module::load_state_dict: shape mismatch for `{name}`: expected \
                     {:?}, got {:?}",
                    old_tensor.shape(),
                    tensor.shape()
                )));
            }
        }

        // パス 2（適用。ベストエフォート・ロールバック付き。メソッド
        // doc「アトミック性」節参照）。走査順をキー名の昇順へ固定する。
        let mut entries: Vec<(String, Tensor<f32>)> = state.into_iter().collect();
        entries.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));

        let mut applied: Vec<(String, Tensor<f32>)> = Vec::with_capacity(entries.len());
        for (name, new_value) in entries {
            match self.set_parameter(&name, new_value) {
                Ok(()) => {
                    // パス 1 の完全一致検査によりキーは必ず snapshot に
                    // 存在する（`unwrap`/`expect` は使わず `if let` で
                    // 安全に取り出す）。
                    if let Some(old_value) = snapshot.get(name.as_str()) {
                        applied.push((name, old_value.clone()));
                    }
                }
                Err(err) => {
                    // 適用済みの分を逆順（後勝ちで上書きされないよう）に
                    // 元の値へ戻す。
                    for (rollback_name, rollback_value) in applied.into_iter().rev() {
                        if let Err(rollback_err) =
                            self.set_parameter(&rollback_name, rollback_value)
                        {
                            return Err(AutodiffError::InvalidArgument(format!(
                                "Module::load_state_dict: failed to apply `{name}` ({err}), \
                                 and rollback of already-applied key `{rollback_name}` also \
                                 failed ({rollback_err}); the module may now be left in a \
                                 partially applied state"
                            )));
                        }
                    }
                    return Err(err);
                }
            }
        }
        Ok(())
    }
}

/// `Linear::bind(tape)` で当該ステップの葉ノードを登録してから
/// `LinearVars::forward` を呼ぶ（`nn/linear.rs` 参照）。
impl Module for Linear {
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.bind(tape).forward(input)
    }

    fn as_linear(&self) -> Option<&Linear> {
        Some(self)
    }

    fn as_linear_mut(&mut self) -> Option<&mut Linear> {
        Some(self)
    }

    /// 命名契約（`Module::named_parameters` doc §「命名契約」）:
    /// `weight`（常に）→ `bias`（`Some` の場合のみ）の順。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut out = vec![("weight".to_string(), self.weight())];
        if let Some(bias) = self.bias() {
            out.push(("bias".to_string(), bias));
        }
        out
    }

    /// [`Module::set_parameter`] の実装。`Linear::set_parameter`
    /// （`nn/linear.rs`）へ委譲する（イシュー #1752）。
    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        Linear::set_parameter(self, name, value)
    }

    /// [`Module::forward`]（`Linear::bind(tape).forward(input)`。
    /// `LinearVars::forward` が `input.matmul(&weight)` → `.add(&bias)`
    /// と非融合合成する）と **同一の演算列**（`ops.gemm` → `ops.add`）を
    /// 直接呼ぶ。融合カーネル（`ops.gemm_bias_act`）を使わない理由は
    /// bit-exactness 契約（`Module::forward_host` doc 参照）: 融合
    /// epilogue はカーネル内 tiling 次第で加算順序が変わりうるため、
    /// 旧経路と厳密に同じ累積順序を保証できるのは非融合合成のみ。
    /// この bit-exactness 契約は本 trait メソッド（汎用 `&dyn BackendOps`
    /// 向け）に対するもので、CUDA／Metal の融合オーバーライドは非融合
    /// 合成との bit 一致が未保証のため今後もここを融合へ切り替えない。
    ///
    /// **CPU 固定経路の例外（イシュー #1218・`docs/perf/
    /// cpu-infer-predict-profile.md`）**: CPU 融合カーネル
    /// （CPU バックエンドクレートの `CpuBackendOps::gemm_bias_act`）は epilogue
    /// を GEMM 完了後に適用するため非融合合成と bit 完全一致することが
    /// `crates/backend-cpu/tests/gemm_epilogue_parity.rs` で確認済み。
    /// `fandhe_ai_facade::compat::sequential::Sequential::predict` の
    /// CPU 固定 tape 不要経路はこの事実を根拠に `Linear::
    /// forward_host_with_activation`（`nn/linear.rs`。本メソッドとは別の
    /// inherent メソッド）を Linear→ReLU 限定で使う。本 trait メソッド
    /// 自体は汎用バックエンド向けのまま変更しない。
    ///
    /// **エラー型の一致契約（review 指摘）**: `Var::matmul`/`add`（tape
    /// 経路。`var.rs`）は shape 不整合を `gemm_out_shape`/
    /// `broadcast_shape` で `ops.gemm`/`ops.add` 呼び出し**前**に検査し
    /// `AutodiffError::Shape` として返す。本メソッド（tape 不要経路）が
    /// この事前検査を省いて `ops.gemm`/`ops.add` の `?` に任せると、同じ
    /// shape 不整合が `BackendError::ShapeMismatch` 経由の
    /// `AutodiffError::Backend` として返り、`compat::Sequential::predict`
    /// のフォールバック判定対象外の経路で `AutodiffError` の variant が
    /// 呼び出し元から見て変わってしまう（旧経路と新経路で同じ入力が
    /// 異なるエラー variant を返す）。それを避けるため、tape 経路と
    /// 同じ関数で同じ順序に事前検査してから `ops` を呼ぶ。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        gemm_out_shape(input.shape(), self.weight().shape())?;
        let y = ops.gemm(input, self.weight())?;
        match self.bias() {
            Some(bias) => {
                broadcast_shape(y.shape(), bias.shape())?;
                Ok(ops.add(&y, bias)?)
            }
            None => Ok(y),
        }
    }
}

/// `Conv2d::bind(tape).forward(input)`（`nn/conv.rs` 参照。イシュー
/// #1770）。
impl Module for Conv2d {
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.bind(tape).forward(input)
    }

    fn as_conv2d(&self) -> Option<&Conv2d> {
        Some(self)
    }

    fn as_conv2d_mut(&mut self) -> Option<&mut Conv2d> {
        Some(self)
    }

    /// 命名契約（`Module::named_parameters` doc §「命名契約」）:
    /// `weight`（常に）→ `bias`（`Some` の場合のみ）の順（`Linear` と
    /// 同型）。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut out = vec![("weight".to_string(), self.weight())];
        if let Some(bias) = self.bias() {
            out.push(("bias".to_string(), bias));
        }
        out
    }

    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        Conv2d::set_parameter(self, name, value)
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        Conv2d::forward_host(self, ops, input)
    }
}

/// `Conv1d::bind(tape).forward(input)`（`nn/conv.rs` 参照。イシュー
/// #1770）。
impl Module for Conv1d {
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.bind(tape).forward(input)
    }

    fn as_conv1d(&self) -> Option<&Conv1d> {
        Some(self)
    }

    fn as_conv1d_mut(&mut self) -> Option<&mut Conv1d> {
        Some(self)
    }

    /// 命名契約は `Conv2d` と同型（`weight` → `bias`）。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut out = vec![("weight".to_string(), self.weight())];
        if let Some(bias) = self.bias() {
            out.push(("bias".to_string(), bias));
        }
        out
    }

    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        Conv1d::set_parameter(self, name, value)
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        Conv1d::forward_host(self, ops, input)
    }
}

/// `Relu::forward`（shape 不変の単項演算のため構造的に失敗しえない）を
/// `Result` へ包むだけの委譲。`tape` は使わない（`nn/activation.rs`
/// 参照）。
impl Module for Relu {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(Relu::forward(self, input))
    }

    fn as_relu(&self) -> bool {
        true
    }

    /// `Var::relu()`（`nn::activation::Relu::forward`）が呼ぶ
    /// `tape.ops().relu(...)` と同一のディスパッチ（`ops.relu`）。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        Ok(ops.relu(input)?)
    }
}

/// `Sigmoid::forward` への委譲。`Relu` の実装と同じ理由で `tape` は
/// 使わない。
impl Module for Sigmoid {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(Sigmoid::forward(self, input))
    }

    /// `Var::sigmoid()` は `BackendOps` ディスパッチを経由せず
    /// `eval::sigmoid`（ホスト直接計算）を呼ぶ（`var.rs` 参照）。
    /// bit-exactness のため同じ `eval::sigmoid` を直接呼ぶ。
    fn forward_host(
        &self,
        _ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        Ok(eval::sigmoid(input))
    }
}

/// `Tanh::forward` への委譲。`Relu` の実装と同じ理由で `tape` は
/// 使わない。
impl Module for Tanh {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(Tanh::forward(self, input))
    }

    /// `Var::tanh()` も `Sigmoid` と同様 `eval::tanh`（ホスト直接計算）を
    /// 呼ぶ。bit-exactness のため同じ関数を直接呼ぶ。
    fn forward_host(
        &self,
        _ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        Ok(eval::tanh(input))
    }
}

/// `Gelu::forward` への委譲（イシュー #1713）。`Relu` と異なり `Var::gelu`
/// の eager dispatch 契約が型付きエラーを返しうるため `Softmax` と同じ
/// fallible 契約。
impl Module for Gelu {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Gelu::forward(self, input)
    }

    /// `Var::gelu()`（`scalar_unary` 経由）と同一のディスパッチ規律
    /// （`grad::scalar_unary_with_fallback`: バックエンド実装 →
    /// `Unsupported` のときのみホスト参照実装へフォールバック・戻り値
    /// shape 検証）を tape 不要経路で再現する。`Var::scalar_unary` が
    /// 呼ぶ同一関数をそのまま呼ぶため bit-exactness・判定迂回なしが
    /// 機構的に保証される。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        crate::grad::scalar_unary_with_fallback(
            ops,
            fandhe_ai_tensor_core::ScalarUnaryOp::Gelu,
            input,
        )
    }
}

/// `GeluTanh::forward` への委譲（イシュー #1713）。[`Gelu`] と同じ
/// fallible 契約・`forward_host` ディスパッチ規律。
impl Module for GeluTanh {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        GeluTanh::forward(self, input)
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        crate::grad::scalar_unary_with_fallback(
            ops,
            fandhe_ai_tensor_core::ScalarUnaryOp::GeluTanh,
            input,
        )
    }
}

/// `Softplus::forward` への委譲（イシュー #1713）。`beta`／`threshold`
/// の検査は [`Softplus::new`] が構築時に済ませているため、`forward`／
/// `forward_host` 自体は形状不整合以外では失敗しない。
impl Module for Softplus {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Softplus::forward(self, input)
    }

    /// [`Gelu::forward_host`] と同じディスパッチ規律。`self.beta()`／
    /// `self.threshold()`（クレート内アクセサ。`nn/activation.rs`
    /// 参照）で構築済みの検査済み値を読み出す。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        crate::grad::scalar_unary_with_fallback(
            ops,
            fandhe_ai_tensor_core::ScalarUnaryOp::Softplus {
                beta: self.beta(),
                threshold: self.threshold(),
            },
            input,
        )
    }
}

/// `MaxPool2d::forward` への委譲（イシュー #1728）。`(values, index)`
/// のうち `values` のみを返す（索引が必要な場合は `MaxPool2d::
/// forward` を直接呼ぶ。`nn/pooling.rs` モジュール doc 参照）。
impl Module for MaxPool2d {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let (values, _index) = MaxPool2d::forward(self, input)?;
        Ok(values)
    }

    /// `Var::max_pool2d` と同じ検査順序（`pool2d_out_shape` →
    /// `max_pool2d_with_fallback`）を tape 不要経路で再現する
    /// （`Var::max_pool2d` doc 参照）。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let out_shape =
            pool2d_out_shape(input.shape(), self.params()).map_err(AutodiffError::Shape)?;
        let (values, _index) =
            crate::grad::max_pool2d_with_fallback(ops, input, self.params(), &out_shape)?;
        Ok(values)
    }

    fn is_pooling(&self) -> bool {
        true
    }
}

/// `MaxPool1d::forward` への委譲。`[N,C,L]` を `[N,C,1,L]` へ reshape
/// して [`MaxPool2d`] の `forward_host` 経路（`H` 軸固定）を再利用し、
/// 出力を `[N,C,Lout]` へ戻す（`Var::max_pool1d` と同型。イシュー
/// #1728）。
impl Module for MaxPool1d {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let (values, _index) = MaxPool1d::forward(self, input)?;
        Ok(values)
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let in_shape = input.shape();
        if in_shape.len() != 3 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 3,
                actual: in_shape.len(),
            }));
        }
        let (n, c, l) = (in_shape[0], in_shape[1], in_shape[2]);
        let x4 = input
            .contiguous()
            .reshape(&[n, c, 1, l])
            .map_err(AutodiffError::Shape)?;
        let params2d = fandhe_ai_tensor_core::Pool2dParams::new(
            self.kernel_size_2d(),
            Some(self.stride_2d()),
            self.padding_2d(),
            self.dilation_2d(),
        )
        .map_err(AutodiffError::Backend)?;
        let out_shape4 =
            pool2d_out_shape(&[n, c, 1, l], &params2d).map_err(AutodiffError::Shape)?;
        let (values4, _index4) =
            crate::grad::max_pool2d_with_fallback(ops, &x4, &params2d, &out_shape4)?;
        let lout = out_shape4[3];
        values4.reshape(&[n, c, lout]).map_err(AutodiffError::Shape)
    }

    fn is_pooling(&self) -> bool {
        true
    }
}

/// `AvgPool2d::forward` への委譲（イシュー #1728）。
impl Module for AvgPool2d {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        AvgPool2d::forward(self, input)
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let out_shape =
            pool2d_out_shape(input.shape(), self.params()).map_err(AutodiffError::Shape)?;
        crate::grad::avg_pool2d_with_fallback(
            ops,
            input,
            self.params(),
            self.count_include_pad(),
            &out_shape,
        )
    }

    fn is_pooling(&self) -> bool {
        true
    }
}

/// `AvgPool1d::forward` への委譲（`MaxPool1d` の `forward_host` と
/// 同型の reshape 併合。イシュー #1728）。
impl Module for AvgPool1d {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        AvgPool1d::forward(self, input)
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let in_shape = input.shape();
        if in_shape.len() != 3 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 3,
                actual: in_shape.len(),
            }));
        }
        let (n, c, l) = (in_shape[0], in_shape[1], in_shape[2]);
        let x4 = input
            .contiguous()
            .reshape(&[n, c, 1, l])
            .map_err(AutodiffError::Shape)?;
        let params2d = fandhe_ai_tensor_core::Pool2dParams::new(
            self.kernel_size_2d(),
            Some(self.stride_2d()),
            self.padding_2d(),
            [1, 1],
        )
        .map_err(AutodiffError::Backend)?;
        let out_shape4 =
            pool2d_out_shape(&[n, c, 1, l], &params2d).map_err(AutodiffError::Shape)?;
        let values4 = crate::grad::avg_pool2d_with_fallback(
            ops,
            &x4,
            &params2d,
            self.count_include_pad(),
            &out_shape4,
        )?;
        let lout = out_shape4[3];
        values4.reshape(&[n, c, lout]).map_err(AutodiffError::Shape)
    }

    fn is_pooling(&self) -> bool {
        true
    }
}

/// `AdaptiveAvgPool2d::forward` への委譲（イシュー #1728）。
impl Module for AdaptiveAvgPool2d {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        AdaptiveAvgPool2d::forward(self, input)
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let out_shape = adaptive_pool2d_out_shape(input.shape(), self.output_size())
            .map_err(AutodiffError::Shape)?;
        crate::grad::adaptive_avg_pool2d_with_fallback(ops, input, self.output_size(), &out_shape)
    }

    fn is_pooling(&self) -> bool {
        true
    }
}

/// `AdaptiveAvgPool1d::forward` への委譲（`MaxPool1d` と同型の
/// reshape 併合。イシュー #1728）。
impl Module for AdaptiveAvgPool1d {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        AdaptiveAvgPool1d::forward(self, input)
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let in_shape = input.shape();
        if in_shape.len() != 3 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 3,
                actual: in_shape.len(),
            }));
        }
        let (n, c, l) = (in_shape[0], in_shape[1], in_shape[2]);
        let x4 = input
            .contiguous()
            .reshape(&[n, c, 1, l])
            .map_err(AutodiffError::Shape)?;
        let output_size4 = [1, self.output_size_1d()];
        let out_shape4 =
            adaptive_pool2d_out_shape(&[n, c, 1, l], output_size4).map_err(AutodiffError::Shape)?;
        let values4 =
            crate::grad::adaptive_avg_pool2d_with_fallback(ops, &x4, output_size4, &out_shape4)?;
        let lout = out_shape4[3];
        values4.reshape(&[n, c, lout]).map_err(AutodiffError::Shape)
    }

    fn is_pooling(&self) -> bool {
        true
    }
}

/// `Silu::forward` への委譲（イシュー #1714）。`Softmax` と異なり
/// `forward` 自体は shape 検査で失敗しないが、`Var::scalar_unary` の
/// eager 実体化契約（バックエンド dispatch が型付きエラーを返しうる）
/// により戻り値は `Result`（`Softmax`／`RmsNorm` と同じく `?` ではなく
/// そのまま返す）。
impl Module for Silu {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Silu::forward(self, input)
    }

    /// `Var::silu()`（`var.rs`）と同じディスパッチ（`grad::
    /// scalar_unary_with_fallback`）を `tape` 不要経路で再現する
    /// （`Var::scalar_unary` が呼ぶのと同一関数のため bit-exactness が
    /// 構造的に成立する。`.claude/rules/security.md` A08「判定迂回経路
    /// を作らない」規律）。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        crate::grad::scalar_unary_with_fallback(ops, self.op(), input)
    }
}

/// `Hardswish::forward` への委譲（イシュー #1714）。[`Silu`] と同じ
/// `forward_host` ディスパッチ規律。
impl Module for Hardswish {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Hardswish::forward(self, input)
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        crate::grad::scalar_unary_with_fallback(ops, self.op(), input)
    }
}

/// `LeakyRelu::forward` への委譲（イシュー #1714）。[`Silu`] と同じ
/// `forward_host` ディスパッチ規律。
impl Module for LeakyRelu {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        LeakyRelu::forward(self, input)
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        crate::grad::scalar_unary_with_fallback(ops, self.op(), input)
    }
}

/// `Elu::forward` への委譲（イシュー #1714）。[`Silu`] と同じ
/// `forward_host` ディスパッチ規律。
impl Module for Elu {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Elu::forward(self, input)
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        crate::grad::scalar_unary_with_fallback(ops, self.op(), input)
    }
}

/// `RmsNorm::bind(tape).forward(input)` への委譲（イシュー #1596）。
/// `Linear` と異なり `forward` 自体は fallible（`eps` 検査・`row_norm_
/// layout` の shape 検査で失敗しうる）ため、戻り値をそのまま返す。
impl Module for RmsNorm {
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.bind(tape).forward(input)
    }

    /// イシュー #1760: `compat::Sequential` の学習経路が `RmsNorm` 層を
    /// 認識するためのフック（`as_linear` と同型）。
    fn as_rms_norm(&self) -> Option<&RmsNorm> {
        Some(self)
    }

    /// [`Module::as_rms_norm`] の可変版。
    fn as_rms_norm_mut(&mut self) -> Option<&mut RmsNorm> {
        Some(self)
    }

    /// 命名契約（`Module::named_parameters` doc §「命名契約」）:
    /// `weight`（`Some` の場合のみ。affine なし構成は空）。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        self.weight()
            .map(|w| vec![("weight".to_string(), w)])
            .unwrap_or_default()
    }

    /// [`Module::set_parameter`] の実装。`RmsNorm::set_parameter`
    /// （`nn/norm.rs`）へ委譲する（イシュー #1752）。
    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        RmsNorm::set_parameter(self, name, value)
    }

    /// `Var::rms_norm`（`var.rs`）と同じディスパッチ規律を `tape` 不要
    /// 経路（`ops` を直接受け取る）で再現する: `row_norm_layout` で
    /// `hidden` を導出し `weight` の shape を検査してから `ops.rmsnorm`
    /// を試み、`Unsupported` のときのみ `eval::rmsnorm_rows` へ
    /// フォールバックする（`Var::rms_norm` と同じ判定迂回を作らない
    /// 規律。`.claude/rules/security.md` A08）。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let (rows, hidden) = row_norm_layout(input.shape())?;
        if let Some(w) = self.weight() {
            require_same_shape(w.shape(), &[hidden])?;
        }
        let value = match ops.rmsnorm(input, self.weight(), self.eps()) {
            Ok(v) => v,
            Err(BackendError::Unsupported(_)) => {
                let w_dense = self.weight().map(eval::dense_vec);
                eval::rmsnorm_rows(input, w_dense.as_deref(), self.eps(), rows, hidden)
            }
            Err(other) => return Err(AutodiffError::Backend(other)),
        };
        // `Var::rms_norm`（`var.rs`）と同じバックエンド契約検証
        // （`BackendOps::rmsnorm` doc「戻り値の shape は入力 `x` と
        // 恒等」）を tape 不要経路でも行う。省略すると `forward_host`
        // 経由の推論のみ不整合 shape を素通りさせてしまい、`Var::
        // rms_norm` と `forward_host` とで判定基準が食い違う
        // （`.claude/rules/security.md` A08 の判定迂回経路になる）。
        if value.shape() != input.shape() {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: value.shape().to_vec(),
                    rhs: input.shape().to_vec(),
                },
            )));
        }
        Ok(value)
    }
}

/// `LayerNorm::bind(tape).forward(input)` への委譲（イシュー #1596）。
/// `RmsNorm` と同じ fallible 契約・`forward_host` ディスパッチ規律。
impl Module for LayerNorm {
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.bind(tape).forward(input)
    }

    /// イシュー #1760: `compat::Sequential` の学習経路が `LayerNorm`
    /// 層を認識するためのフック（`as_linear` と同型）。
    fn as_layer_norm(&self) -> Option<&LayerNorm> {
        Some(self)
    }

    /// [`Module::as_layer_norm`] の可変版。
    fn as_layer_norm_mut(&mut self) -> Option<&mut LayerNorm> {
        Some(self)
    }

    /// 命名契約（`Module::named_parameters` doc §「命名契約」）:
    /// `weight`（`Some` の場合）→ `bias`（`Some` の場合）の順。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut out = Vec::new();
        if let Some(w) = self.weight() {
            out.push(("weight".to_string(), w));
        }
        if let Some(b) = self.bias() {
            out.push(("bias".to_string(), b));
        }
        out
    }

    /// [`Module::set_parameter`] の実装。`LayerNorm::set_parameter`
    /// （`nn/norm.rs`）へ委譲する（イシュー #1752）。
    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        LayerNorm::set_parameter(self, name, value)
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let (rows, hidden) = row_norm_layout(input.shape())?;
        if let Some(w) = self.weight() {
            require_same_shape(w.shape(), &[hidden])?;
        }
        if let Some(b) = self.bias() {
            require_same_shape(b.shape(), &[hidden])?;
        }
        let value = match ops.layer_norm(input, self.weight(), self.bias(), self.eps()) {
            Ok(v) => v,
            Err(BackendError::Unsupported(_)) => {
                let w_dense = self.weight().map(eval::dense_vec);
                let b_dense = self.bias().map(eval::dense_vec);
                eval::layer_norm_rows(
                    input,
                    w_dense.as_deref(),
                    b_dense.as_deref(),
                    self.eps(),
                    rows,
                    hidden,
                )
            }
            Err(other) => return Err(AutodiffError::Backend(other)),
        };
        // `RmsNorm::forward_host`（直上）と同じバックエンド契約検証。
        if value.shape() != input.shape() {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: value.shape().to_vec(),
                    rhs: input.shape().to_vec(),
                },
            )));
        }
        Ok(value)
    }
}

/// `BatchNorm1d`／`BatchNorm2d::forward_host`（両者共通）が呼ぶ
/// tape 不要経路の forward 本体（イシュー #1732・親 #1608）。`nn::
/// batch_norm::BatchNormVars::forward`（tape 経路）と同じ判定規律
/// （バックエンド → `Unsupported` のときのみホスト参照実装）を
/// `ops` 直接呼び出しで再現する（`RmsNorm::forward_host` と同じ
/// 理由）。train モードは呼ぶたび必ず running stats を更新する
/// （`BatchNormVars::forward` と同じ契約。`nn::batch_norm` モジュール
/// doc comment「running stats 更新契約」参照）。
fn batch_norm_forward_host(
    core: &BatchNormCore,
    ops: &dyn BackendOps,
    input: &Tensor<f32>,
) -> Result<Tensor<f32>, AutodiffError> {
    let (n, c, spatial) = batch_norm_layout(input.shape())?;
    // `weight`／`bias` が `Some` のときは直後の `require_same_shape`
    // が `w`／`b` の構築時 shape（`[num_features]`）経由で
    // `c == num_features` を間接検証するが、`without_affine`
    // （両方 `None`）構成ではその経路が無い。`core.update_running_stats`
    // （train 分岐。長さ `num_features` の running stats と `zip` する）
    // へ長さ `c` の batch 統計が渡る前に、ここで明示検査する
    // （codex-review P1・Cursor Bugbot 指摘。イシュー #1732 fix
    // ループ。`nn::batch_norm::BatchNormCore::forward_var` と同型の
    // 検査）。
    require_same_shape(&[c], &[core.num_features()])?;
    if let Some(w) = core.weight() {
        require_same_shape(w.shape(), &[c])?;
    }
    if let Some(b) = core.bias() {
        require_same_shape(b.shape(), &[c])?;
    }
    if core.training() {
        let m = n
            .checked_mul(spatial)
            .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
        if m <= 1 {
            return Err(AutodiffError::InvalidArgument(format!(
                "BatchNorm::forward_host: train モードはチャネルごとの要素数 \
                 M=n*spatial が 1 以下を許容しない（got n={n}, spatial={spatial}, M={m}）"
            )));
        }
        let out = crate::grad::batch_norm_train_with_fallback(
            ops,
            input,
            core.weight(),
            core.bias(),
            core.eps(),
            n,
            c,
            spatial,
        )?;
        core.update_running_stats(&out.batch_mean, &out.batch_var, m);
        Ok(out.output)
    } else {
        let running_mean = core.running_mean_ref();
        let running_var = core.running_var_ref();
        crate::grad::batch_norm_infer_with_fallback(
            ops,
            input,
            &running_mean,
            &running_var,
            core.weight(),
            core.bias(),
            core.eps(),
            n,
            c,
            spatial,
        )
    }
}

/// `BatchNorm1d::bind(tape).forward(input)` への委譲（イシュー
/// #1732・親 #1608）。本クレート内で初めて train／eval でモードにより
/// 挙動が変わる層のため `set_training`／`training` を明示
/// オーバーライドする（`Module::set_training` trait doc の「今後
/// BatchNorm 等を追加する際は必ずオーバーライドすること」を実装する。
/// `nn::batch_norm` モジュール doc comment 参照）。
impl Module for BatchNorm1d {
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.bind(tape).forward(input)
    }

    fn set_training(&mut self, training: bool) {
        self.core.set_training(training);
    }

    fn training(&self) -> bool {
        self.core.training()
    }

    /// イシュー #1760: `compat::Sequential` の学習経路が `BatchNorm1d`
    /// 層を認識するためのフック（`as_linear` と同型）。
    fn as_batch_norm1d(&self) -> Option<&BatchNorm1d> {
        Some(self)
    }

    /// [`Module::as_batch_norm1d`] の可変版。
    fn as_batch_norm1d_mut(&mut self) -> Option<&mut BatchNorm1d> {
        Some(self)
    }

    /// 命名契約（`Module::named_parameters` doc §「命名契約」）:
    /// `weight`（`Some` の場合）→ `bias`（`Some` の場合）の順。running
    /// stats は buffer であり学習可能パラメータではないため含めない
    /// （`nn::batch_norm` モジュール doc comment「`BatchNormCore` の
    /// 可視性」節参照）。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        self.core.named_parameters()
    }

    /// [`Module::set_parameter`] の実装。`BatchNormCore::set_parameter`
    /// （`nn/batch_norm.rs`）へ委譲する（`LayerNorm::set_parameter` と
    /// 同型。PR #1874 codex-review P1・Cursor Bugbot Medium 是正）。
    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        self.core.set_parameter(name, value)
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let rank = input.shape().len();
        if !BATCH_NORM_1D_RANKS.contains(&rank) {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: BATCH_NORM_1D_RANKS[0],
                actual: rank,
            }));
        }
        batch_norm_forward_host(&self.core, ops, input)
    }
}

/// `BatchNorm2d::bind(tape).forward(input)` への委譲（イシュー
/// #1732・親 #1608）。[`Module for BatchNorm1d`](trait.Module.html)
/// と同じ理由・同じ構造だが rank 限定契約のみ異なる（rank 4 のみ）。
impl Module for BatchNorm2d {
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.bind(tape).forward(input)
    }

    fn set_training(&mut self, training: bool) {
        self.core.set_training(training);
    }

    fn training(&self) -> bool {
        self.core.training()
    }

    /// イシュー #1760: `compat::Sequential` の学習経路が `BatchNorm2d`
    /// 層を認識するためのフック（`as_linear` と同型）。
    fn as_batch_norm2d(&self) -> Option<&BatchNorm2d> {
        Some(self)
    }

    /// [`Module::as_batch_norm2d`] の可変版。
    fn as_batch_norm2d_mut(&mut self) -> Option<&mut BatchNorm2d> {
        Some(self)
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        self.core.named_parameters()
    }

    /// [`Module::set_parameter`] の実装。
    /// [`Module for BatchNorm1d`](trait.Module.html) と同じ理由・
    /// 同じ委譲先。
    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        self.core.set_parameter(name, value)
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let rank = input.shape().len();
        if !BATCH_NORM_2D_RANKS.contains(&rank) {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: BATCH_NORM_2D_RANKS[0],
                actual: rank,
            }));
        }
        batch_norm_forward_host(&self.core, ops, input)
    }
}

/// `Softmax::forward` への委譲（イシュー #1594）。`Relu`/`Sigmoid`/
/// `Tanh` と異なり `forward` 自体が `dim` の軸範囲検査により失敗しうる
/// ため（fallible）、`?` で伝播するだけの `Relu` と違い戻り値をそのまま
/// 返す。
impl Module for Softmax {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Softmax::forward(self, input)
    }

    /// `Var::softmax()`（`var.rs`）と同じディスパッチ規律を `tape` 不要
    /// 経路（`ops` を直接受け取る）で再現する: `dim` を [`reduce_out_shape`]
    /// で事前検査してから `ops.softmax` を試み、`Unsupported` のときのみ
    /// `eval::softmax_along` へフォールバックする（`Var::softmax` と
    /// 同じ判定迂回を作らない規律。`.claude/rules/security.md` A08）。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let dim = self.dim();
        reduce_out_shape(input.shape(), Some(dim))?;
        let value = match ops.softmax(input, dim) {
            Ok(v) => v,
            Err(BackendError::Unsupported(_)) => eval::softmax_along(input, dim),
            Err(other) => return Err(AutodiffError::Backend(other)),
        };
        // `Var::softmax`（`var.rs`）と同じバックエンド契約検証
        // （`BackendOps::softmax` doc「戻り値 shape は入力と恒等」）を
        // tape 不要経路でも行う。ここを省略すると `forward_host`
        // 経由の推論のみ不整合 shape を素通りさせてしまい、`Var::
        // softmax` と `forward_host` とで判定基準が食い違う
        // （`.claude/rules/security.md` A08 の判定迂回経路になる）。
        if value.shape() != input.shape() {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                fandhe_ai_tensor_core::ShapeError::ShapeMismatch {
                    lhs: value.shape().to_vec(),
                    rhs: input.shape().to_vec(),
                },
            )));
        }
        Ok(value)
    }
}

/// `LogSoftmax::forward` への委譲（イシュー #1594）。`Softmax` と同じ
/// fallible 契約・`forward_host` ディスパッチ規律。
impl Module for LogSoftmax {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        LogSoftmax::forward(self, input)
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let dim = self.dim();
        reduce_out_shape(input.shape(), Some(dim))?;
        let value = match ops.log_softmax(input, dim) {
            Ok(v) => v,
            Err(BackendError::Unsupported(_)) => eval::log_softmax_along(input, dim),
            Err(other) => return Err(AutodiffError::Backend(other)),
        };
        // `Softmax::forward_host`（直上）と同じバックエンド契約検証。
        if value.shape() != input.shape() {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                fandhe_ai_tensor_core::ShapeError::ShapeMismatch {
                    lhs: value.shape().to_vec(),
                    rhs: input.shape().to_vec(),
                },
            )));
        }
        Ok(value)
    }
}

/// `Embedding::bind(tape).forward_from_var(input)` への委譲（イシュー
/// #1760）。`nn/embedding.rs` モジュール doc「`Module` trait は実装
/// しない（確定判断）」節が挙げていた 2 つの理由——(1) `Module::
/// forward` の f32 `Var` 契約と embedding の整数 id 契約が食い違う、
/// (2) `compat::Sequential` の学習経路が `as_linear` 系フックにしか
/// 反応しない——を本イシューで解消する: (1) は
/// `EmbeddingVars::forward_from_var`（`nn/embedding.rs`。f32 → 厳格な
/// i32 変換を挟む）で橋渡しし、(2) は [`Module::as_embedding`] フック
/// の追加で解消する。
impl Module for Embedding {
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.bind(tape).forward_from_var(input)
    }

    fn as_embedding(&self) -> Option<&Embedding> {
        Some(self)
    }

    fn as_embedding_mut(&mut self) -> Option<&mut Embedding> {
        Some(self)
    }

    /// `forward_host` を実装しないため既定 `Unsupported` のまま
    /// （trait doc 参照）。[`Module::supports_forward_host`] を `false`
    /// へオーバーライドし、`compat::Sequential::predict` の tape 不要
    /// 経路が本層で `Unsupported` に当たる前に全層を事前判定できる
    /// ようにする（イシュー #1760・Cursor Bugbot 指摘是正）。
    fn supports_forward_host(&self) -> bool {
        false
    }

    /// 命名契約（`Module::named_parameters` doc §「命名契約」）:
    /// `weight`（常に。`Embedding` は affine なし構成を持たないため
    /// `Linear`／`Conv*` と異なり必ず `Some` 相当で 1 件のみ）。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        vec![("weight".to_string(), self.weight())]
    }

    /// [`Module::set_parameter`] の実装。`Embedding::set_parameter`
    /// （`nn/embedding.rs`）へ委譲する。
    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        Embedding::set_parameter(self, name, value)
    }
}

#[cfg(test)]
mod tests {
    //! `Module::forward` が既存の直接呼び出し（`Linear::bind().forward()`・
    //! `Relu::forward()` 等）と同一の値・テープ記録を返すことを検証する
    //! （「薄いラッパー性」の担保）。

    use super::*;
    use crate::eval::dense_vec;
    use fandhe_ai_tensor_core::Tensor;

    #[test]
    fn linear_module_forward_matches_bind_forward() {
        let linear = Linear::new(3, 2, true, 42).expect("seed=42 は有効な構築引数");
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap());

        let via_module = <Linear as Module>::forward(&linear, &tape, &x).unwrap();
        let via_direct = linear.bind(&tape).forward(&x).unwrap();

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }

    #[test]
    fn relu_module_forward_matches_direct_forward() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());

        let via_module = <Relu as Module>::forward(&Relu, &tape, &x).unwrap();
        let via_direct = Relu.forward(&x);

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }

    #[test]
    fn sigmoid_module_forward_matches_direct_forward() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());

        let via_module = <Sigmoid as Module>::forward(&Sigmoid, &tape, &x).unwrap();
        let via_direct = Sigmoid.forward(&x);

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }

    #[test]
    fn tanh_module_forward_matches_direct_forward() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());

        let via_module = <Tanh as Module>::forward(&Tanh, &tape, &x).unwrap();
        let via_direct = Tanh.forward(&x);

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }

    #[test]
    fn softmax_module_forward_matches_direct_forward() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![-1.0, 2.0, 0.5, 1.0], &[2, 2]).unwrap());
        let softmax = Softmax::new(1);

        let via_module = <Softmax as Module>::forward(&softmax, &tape, &x).unwrap();
        let via_direct = softmax.forward(&x).unwrap();

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }

    #[test]
    fn log_softmax_module_forward_matches_direct_forward() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![-1.0, 2.0, 0.5, 1.0], &[2, 2]).unwrap());
        let log_softmax = LogSoftmax::new(1);

        let via_module = <LogSoftmax as Module>::forward(&log_softmax, &tape, &x).unwrap();
        let via_direct = log_softmax.forward(&x).unwrap();

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }

    /// `Module::forward_host`（tape 不要経路）と `Module::forward`
    /// （tape 経路）が同一値を返すことを確認する（`Linear` 等の既存
    /// 契約と同じ bit-exactness 期待。イシュー #1594）。
    #[test]
    fn softmax_forward_host_matches_tape_forward() {
        use crate::test_support::test_ops;

        let x = Tensor::new(vec![-1.0, 2.0, 0.5, 1.0], &[2, 2]).unwrap();
        let softmax = Softmax::new(1);

        let tape = Tape::new_with_ops(test_ops());
        let xv = tape.var(&x);
        let via_tape = <Softmax as Module>::forward(&softmax, &tape, &xv).unwrap();

        let via_host = softmax.forward_host(test_ops().as_ref(), &x).unwrap();

        assert_eq!(dense_vec(&via_tape.to_tensor()), dense_vec(&via_host));
    }

    #[test]
    fn log_softmax_forward_host_matches_tape_forward() {
        use crate::test_support::test_ops;

        let x = Tensor::new(vec![-1.0, 2.0, 0.5, 1.0], &[2, 2]).unwrap();
        let log_softmax = LogSoftmax::new(1);

        let tape = Tape::new_with_ops(test_ops());
        let xv = tape.var(&x);
        let via_tape = <LogSoftmax as Module>::forward(&log_softmax, &tape, &xv).unwrap();

        let via_host = log_softmax.forward_host(test_ops().as_ref(), &x).unwrap();

        assert_eq!(dense_vec(&via_tape.to_tensor()), dense_vec(&via_host));
    }

    #[test]
    fn softmax_forward_host_rejects_axis_out_of_range() {
        use crate::test_support::test_ops;

        let x = Tensor::new(vec![-1.0, 2.0], &[2]).unwrap();
        let softmax = Softmax::new(5);

        let result = softmax.forward_host(test_ops().as_ref(), &x);

        assert!(matches!(
            result,
            Err(AutodiffError::Shape(
                fandhe_ai_tensor_core::ShapeError::AxisOutOfRange { axis: 5, rank: 1 }
            ))
        ));
    }

    /// Cursor Bugbot 指摘（PR #1664）の回帰検証用モック: `softmax`／
    /// `log_softmax` が入力と異なる shape を返す不正なバックエンドを
    /// 模す（他のメソッドは非到達のため `unreachable!` でよい）。
    /// `Var::softmax`（`var.rs`）はこの契約違反を `ShapeMismatch` で
    /// 拒否するが、`Module::forward_host`（tape 不要推論経路）が同じ
    /// 検証を省略していると不整合 shape を素通りさせてしまう
    /// （`.claude/rules/security.md` A08 の判定迂回経路になる）。
    struct WrongShapeOps;

    impl BackendOps for WrongShapeOps {
        fn device(&self) -> fandhe_ai_tensor_core::Device {
            fandhe_ai_tensor_core::Device::Cpu
        }
        fn gemm(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("本テストでは gemm は呼ばれない")
        }
        fn add(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("本テストでは add は呼ばれない")
        }
        fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("本テストでは mul は呼ばれない")
        }
        fn relu(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("本テストでは relu は呼ばれない")
        }
        fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("本テストでは exp は呼ばれない")
        }
        fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("本テストでは tanh は呼ばれない")
        }
        fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("本テストでは sum は呼ばれない")
        }
        fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("本テストでは max は呼ばれない")
        }
        fn softmax(&self, x: &Tensor<f32>, _dim: usize) -> Result<Tensor<f32>, BackendError> {
            // 入力 shape をそのまま返さず要素数を減らした shape で返す
            // ことで、契約違反（`BackendOps::softmax` doc「戻り値 shape
            // は入力と恒等」）を意図的に起こす。
            let numel: usize = x.shape().iter().product();
            Ok(Tensor::new(vec![0.0f32; numel], &[numel]).unwrap())
        }
        fn log_softmax(&self, x: &Tensor<f32>, _dim: usize) -> Result<Tensor<f32>, BackendError> {
            let numel: usize = x.shape().iter().product();
            Ok(Tensor::new(vec![0.0f32; numel], &[numel]).unwrap())
        }
    }

    #[test]
    fn softmax_forward_host_rejects_wrong_shape_from_backend() {
        let x = Tensor::new(vec![-1.0, 2.0, 0.5, 1.0], &[2, 2]).unwrap();
        let softmax = Softmax::new(1);

        let result = softmax.forward_host(&WrongShapeOps, &x);

        assert!(matches!(
            result,
            Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
        ));
    }

    #[test]
    fn log_softmax_forward_host_rejects_wrong_shape_from_backend() {
        let x = Tensor::new(vec![-1.0, 2.0, 0.5, 1.0], &[2, 2]).unwrap();
        let log_softmax = LogSoftmax::new(1);

        let result = log_softmax.forward_host(&WrongShapeOps, &x);

        assert!(matches!(
            result,
            Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
        ));
    }

    /// `Silu`／`Hardswish`／`LeakyRelu`／`Elu` の `forward_host`（tape
    /// 不要経路）が `Module::forward`（tape 経路）と bit 完全一致する
    /// ことを検証する（イシュー #1714。`softmax_forward_host_matches_
    /// tape_forward` と同型——両経路とも同一の `grad::
    /// scalar_unary_with_fallback` を呼ぶため構造的に bit-exactness が
    /// 成立する）。
    #[test]
    fn silu_forward_host_matches_tape_forward() {
        use crate::test_support::test_ops;

        let x = Tensor::new(vec![-1.0, 2.0], &[2]).unwrap();
        let silu = Silu;

        let tape = Tape::new_with_ops(test_ops());
        let xv = tape.var(&x);
        let via_tape = <Silu as Module>::forward(&silu, &tape, &xv).unwrap();
        let via_host = silu.forward_host(test_ops().as_ref(), &x).unwrap();

        assert_eq!(dense_vec(&via_tape.to_tensor()), dense_vec(&via_host));
    }

    #[test]
    fn hardswish_forward_host_matches_tape_forward() {
        use crate::test_support::test_ops;

        let x = Tensor::new(vec![-4.0, 4.0], &[2]).unwrap();
        let hardswish = Hardswish;

        let tape = Tape::new_with_ops(test_ops());
        let xv = tape.var(&x);
        let via_tape = <Hardswish as Module>::forward(&hardswish, &tape, &xv).unwrap();
        let via_host = hardswish.forward_host(test_ops().as_ref(), &x).unwrap();

        assert_eq!(dense_vec(&via_tape.to_tensor()), dense_vec(&via_host));
    }

    #[test]
    fn leaky_relu_forward_host_matches_tape_forward() {
        use crate::test_support::test_ops;

        let x = Tensor::new(vec![-1.0, 2.0], &[2]).unwrap();
        let leaky_relu = LeakyRelu::new(0.2);

        let tape = Tape::new_with_ops(test_ops());
        let xv = tape.var(&x);
        let via_tape = <LeakyRelu as Module>::forward(&leaky_relu, &tape, &xv).unwrap();
        let via_host = leaky_relu.forward_host(test_ops().as_ref(), &x).unwrap();

        assert_eq!(dense_vec(&via_tape.to_tensor()), dense_vec(&via_host));
    }

    #[test]
    fn elu_forward_host_matches_tape_forward() {
        use crate::test_support::test_ops;

        let x = Tensor::new(vec![-1.0, 2.0], &[2]).unwrap();
        let elu = Elu::new(1.3);

        let tape = Tape::new_with_ops(test_ops());
        let xv = tape.var(&x);
        let via_tape = <Elu as Module>::forward(&elu, &tape, &xv).unwrap();
        let via_host = elu.forward_host(test_ops().as_ref(), &x).unwrap();

        assert_eq!(dense_vec(&via_tape.to_tensor()), dense_vec(&via_host));
    }

    #[test]
    fn batch_norm_without_affine_forward_host_rejects_channel_mismatch() {
        // `batch_norm_forward_host`（tape 不要経路）版の regression。
        // `nn::batch_norm::BatchNormCore::forward_var`（tape 経路）と
        // 同型の欠落——affine なし構成では `weight`／`bias` 経由の
        // `c == num_features` 間接検証が働かないため、明示検査を
        // `require_same_shape(&[c], &[core.num_features()])` で追加
        // 済み（codex-review P1 指摘・イシュー #1732 fix ループ）。
        // num_features=3 に対し c=2 の入力を渡し、panic せず型付き
        // エラーで拒否されることを確認する。
        use crate::nn::batch_norm::{BATCH_NORM_DEFAULT_EPS, BATCH_NORM_DEFAULT_MOMENTUM};
        use crate::test_support::test_ops;

        let bn =
            BatchNorm1d::without_affine(3, BATCH_NORM_DEFAULT_EPS, BATCH_NORM_DEFAULT_MOMENTUM)
                .unwrap();
        let x = Tensor::new(vec![1.0, 2.0, -1.0, 0.5], &[2, 2]).unwrap();
        let err = bn.forward_host(test_ops().as_ref(), &x).unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
        ));
        assert_eq!(bn.core.num_batches_tracked(), 0);
    }

    /// codex-review 指摘（PR #1875）の回帰テスト用モック: `named_parameters`
    /// はオーバーライドするが `set_parameter` は既定実装（常に `Err`）の
    /// まま残した「半分だけ実装した」外部 `Module`。`Module::set_parameter`
    /// doc「オーバーライド指針」に反する構成だが、[`ModuleList`] 等の
    /// 複合層に混在しうるケースとして `load_state_dict` のロールバック
    /// 経路を検証するために使う。
    struct HalfImplementedModule {
        param: Tensor<f32>,
    }

    impl Module for HalfImplementedModule {
        fn forward<'t>(&self, _tape: &'t Tape, _input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
            unreachable!("本テストでは forward は呼ばれない")
        }

        fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
            vec![("param".to_string(), &self.param)]
        }

        // `set_parameter` は意図的にオーバーライドしない（既定実装の
        // まま。常に `AutodiffError::InvalidArgument` を返す）。
    }

    use crate::nn::container::ModuleList;

    #[test]
    fn load_state_dict_rolls_back_earlier_success_when_later_key_fails() {
        // index 0: 正常に `set_parameter` へ応答する `Linear`。
        // index 1: `named_parameters` はあるが `set_parameter` が常に
        // `Err` を返す「半分だけ実装した」 Module（上記）。
        //
        // `state` のキーはパス 2 で昇順（"0.weight" < "1.param"）に
        // 適用されるため、index 0 が先に成功してから index 1 が失敗する
        // ——「後発キーの失敗が先発の成功済み変更を巻き戻す」経路を
        // 決定的に踏む。
        let mut list = ModuleList::new();
        list.push(Box::new(Linear::new(3, 2, true, 11).unwrap()));
        list.push(Box::new(HalfImplementedModule {
            param: Tensor::new(vec![1.0f32, 2.0], &[2]).unwrap(),
        }));

        let before = list.state_dict();
        let mut state = before.clone();
        state.insert(
            "0.weight".to_string(),
            Tensor::new(vec![9.0f32; 6], &[3, 2]).unwrap(),
        );
        state.insert(
            "1.param".to_string(),
            Tensor::new(vec![5.0f32, 6.0], &[2]).unwrap(),
        );

        let err = list
            .load_state_dict(state)
            .expect_err("HalfImplementedModule の set_parameter 既定失敗で Err のはず");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));

        let after = list.state_dict();
        for (key, tensor) in &before {
            assert_eq!(
                tensor.contiguous().as_slice().unwrap(),
                after[key].contiguous().as_slice().unwrap(),
                "ロールバック後も key `{key}` が呼び出し前の値と異なる \
                 （先発の `0.weight` 適用が巻き戻されていない可能性）"
            );
        }
    }

    #[test]
    fn load_state_dict_fails_immediately_when_first_key_fails() {
        // 単独の HalfImplementedModule（先発する成功例がない最小ケース）。
        let mut half = HalfImplementedModule {
            param: Tensor::new(vec![1.0f32, 2.0], &[2]).unwrap(),
        };
        let mut state = HashMap::new();
        state.insert(
            "param".to_string(),
            Tensor::new(vec![9.0f32, 9.0], &[2]).unwrap(),
        );
        let err = half
            .load_state_dict(state)
            .expect_err("set_parameter 既定失敗で Err のはず");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        assert_eq!(half.param.contiguous().as_slice().unwrap(), &[1.0f32, 2.0]);
    }

    // `BatchNorm1d`／`BatchNorm2d::set_parameter`（PR #1874 codex-review
    // P1・Cursor Bugbot Medium 是正・イシュー #1732）の回帰テスト。
    // `named_parameters` はオーバーライド済みだが `set_parameter` が
    // 既定実装（常に `Err`）のままだと、affine ありの層でも自身の
    // `state_dict()` を `load_state_dict()` へ渡すだけで失敗していた
    // （`Module::set_parameter` doc「オーバーライド指針」違反）。

    #[test]
    fn batch_norm1d_state_dict_round_trip_updates_weight_and_bias() {
        use crate::nn::batch_norm::{BATCH_NORM_DEFAULT_EPS, BATCH_NORM_DEFAULT_MOMENTUM};

        let mut bn =
            BatchNorm1d::new(3, BATCH_NORM_DEFAULT_EPS, BATCH_NORM_DEFAULT_MOMENTUM).unwrap();
        let mut state = bn.state_dict();
        assert_eq!(
            state.len(),
            2,
            "affine あり BatchNorm1d は weight／bias の 2 キー"
        );
        state.insert(
            "weight".to_string(),
            Tensor::new(vec![2.0f32, 3.0, 4.0], &[3]).unwrap(),
        );
        state.insert(
            "bias".to_string(),
            Tensor::new(vec![0.5f32, 0.6, 0.7], &[3]).unwrap(),
        );

        bn.load_state_dict(state)
            .expect("affine あり BatchNorm1d の state_dict 往復は成功するはず");

        assert_eq!(
            bn.weight().unwrap().contiguous().as_slice().unwrap(),
            &[2.0f32, 3.0, 4.0]
        );
        assert_eq!(
            bn.bias().unwrap().contiguous().as_slice().unwrap(),
            &[0.5f32, 0.6, 0.7]
        );
    }

    #[test]
    fn batch_norm2d_state_dict_round_trip_is_no_op_for_identity_state() {
        use crate::nn::batch_norm::{BATCH_NORM_DEFAULT_EPS, BATCH_NORM_DEFAULT_MOMENTUM};

        let mut bn =
            BatchNorm2d::new(4, BATCH_NORM_DEFAULT_EPS, BATCH_NORM_DEFAULT_MOMENTUM).unwrap();
        let before_weight = bn
            .weight()
            .unwrap()
            .contiguous()
            .as_slice()
            .unwrap()
            .to_vec();
        let before_bias = bn.bias().unwrap().contiguous().as_slice().unwrap().to_vec();

        bn.load_state_dict(bn.state_dict())
            .expect("自身の state_dict をそのまま load_state_dict へ渡すのは成功するはず");

        assert_eq!(
            bn.weight().unwrap().contiguous().as_slice().unwrap(),
            before_weight.as_slice()
        );
        assert_eq!(
            bn.bias().unwrap().contiguous().as_slice().unwrap(),
            before_bias.as_slice()
        );
    }

    #[test]
    fn batch_norm1d_load_state_dict_rejects_shape_mismatch_and_leaves_state_unchanged() {
        use crate::nn::batch_norm::{BATCH_NORM_DEFAULT_EPS, BATCH_NORM_DEFAULT_MOMENTUM};

        let mut bn =
            BatchNorm1d::new(3, BATCH_NORM_DEFAULT_EPS, BATCH_NORM_DEFAULT_MOMENTUM).unwrap();
        let before_weight = bn
            .weight()
            .unwrap()
            .contiguous()
            .as_slice()
            .unwrap()
            .to_vec();

        let mut state = bn.state_dict();
        state.insert(
            "weight".to_string(),
            Tensor::new(vec![1.0f32, 2.0], &[2]).unwrap(),
        );
        let err = bn
            .load_state_dict(state)
            .expect_err("shape 不一致は Err のはず");
        // `Module::load_state_dict` はパス 1（検証のみ）で shape 不一致を
        // 検出し `AutodiffError::InvalidArgument` を返す（`set_parameter`
        // 自体の `ShapeError::ShapeMismatch` へは到達しない。
        // `compat_sequential_state_dict.rs::
        // load_state_dict_rejects_shape_mismatch_and_leaves_model_unchanged`
        // と同じ契約）。
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        assert_eq!(
            bn.weight().unwrap().contiguous().as_slice().unwrap(),
            before_weight.as_slice(),
            "拒否後も weight が変化していない（アトミック性）"
        );
    }

    #[test]
    fn batch_norm1d_without_affine_load_state_dict_rejects_unknown_key() {
        use crate::nn::batch_norm::{BATCH_NORM_DEFAULT_EPS, BATCH_NORM_DEFAULT_MOMENTUM};

        let mut bn =
            BatchNorm1d::without_affine(3, BATCH_NORM_DEFAULT_EPS, BATCH_NORM_DEFAULT_MOMENTUM)
                .unwrap();
        assert!(
            bn.state_dict().is_empty(),
            "affine なしは named_parameters が空"
        );

        let mut state = HashMap::new();
        state.insert(
            "weight".to_string(),
            Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).unwrap(),
        );
        let err = bn
            .load_state_dict(state)
            .expect_err("affine なし構成では `weight` は未知キーのはず");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    /// `ModuleList`（`HalfImplementedModule` の回帰テストと同じ複合層
    /// パターン）内に `BatchNorm1d` を混在させても `state_dict`／
    /// `load_state_dict` の往復が成功することを確認する（facade
    /// `compat::Sequential` は現時点で `BatchNorm` 追加 API を持たない
    /// ため、`ModuleList` を複合層の代替として使う）。
    #[test]
    fn module_list_with_batch_norm_state_dict_round_trip() {
        use crate::nn::batch_norm::{BATCH_NORM_DEFAULT_EPS, BATCH_NORM_DEFAULT_MOMENTUM};

        let mut list = ModuleList::new();
        list.push(Box::new(Linear::new(3, 4, true, 21).unwrap()));
        list.push(Box::new(
            BatchNorm1d::new(4, BATCH_NORM_DEFAULT_EPS, BATCH_NORM_DEFAULT_MOMENTUM).unwrap(),
        ));

        let mut state = list.state_dict();
        assert!(state.contains_key("1.weight"));
        assert!(state.contains_key("1.bias"));
        state.insert(
            "1.weight".to_string(),
            Tensor::new(vec![9.0f32, 9.0, 9.0, 9.0], &[4]).unwrap(),
        );

        list.load_state_dict(state)
            .expect("Linear と BatchNorm1d 混在の ModuleList でも state_dict 往復は成功するはず");

        let after = list.state_dict();
        assert_eq!(
            after["1.weight"].contiguous().as_slice().unwrap(),
            &[9.0f32, 9.0, 9.0, 9.0]
        );
    }

    /// [`Module::is_pooling`] が Pooling 6 型でのみ `true` を返し、
    /// 他の層（`Relu`・`Linear`）では既定の `false` のままであること
    /// を確認する（イシュー #1957。`compat::Sequential::
    /// contains_resident_unsupported_layer` の判定入口として使う
    /// フックのため、閉集合であることをここで固定する）。
    #[test]
    fn is_pooling_true_only_for_pooling_layers() {
        let max_pool2d = MaxPool2d::new([2, 2], None, [0, 0], [1, 1]).unwrap();
        let max_pool1d = MaxPool1d::new(2, None, 0, 1).unwrap();
        let avg_pool2d = AvgPool2d::new([2, 2], None, [0, 0], true).unwrap();
        let avg_pool1d = AvgPool1d::new(2, None, 0, true).unwrap();
        let adaptive_avg_pool2d = AdaptiveAvgPool2d::new([2, 2]).unwrap();
        let adaptive_avg_pool1d = AdaptiveAvgPool1d::new(2).unwrap();

        assert!(Module::is_pooling(&max_pool2d));
        assert!(Module::is_pooling(&max_pool1d));
        assert!(Module::is_pooling(&avg_pool2d));
        assert!(Module::is_pooling(&avg_pool1d));
        assert!(Module::is_pooling(&adaptive_avg_pool2d));
        assert!(Module::is_pooling(&adaptive_avg_pool1d));

        assert!(!Module::is_pooling(&Relu));
        let linear = Linear::new(3, 2, true, 7).unwrap();
        assert!(!Module::is_pooling(&linear));
    }
}
