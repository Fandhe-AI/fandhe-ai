//! `MultiheadAttention` Module（イシュー #1640。親 #1605「MultiheadAttention」
//! の sub-issue (b)。`docs/spec/04-requirements.md` REQ-9 2026-09-12 追記
//! Tier 1・`docs/compat-api-scope.md` §1.2）。
//!
//! **設計方針（新規 `Op`／`BackendOps` メソッド／カーネルを追加しない。
//! `crate::einsum`・sub-issue (a)（イシュー #1639）と同型）**: in/out
//! projection・head 分割・attention 本体・head 結合のすべてを既存の
//! `Var` 演算（[`Var::matmul`]・[`Var::reshape`]・[`Var::permute`]・
//! [`Var::masked_fill`]・[`Var::softmax`]）と `nn::Linear`（4 層。q/k/v/out
//! projection）の合成として組み立てる。VJP は各構成演算の VJP 合成として
//! 自動的に成立し（`grad.rs` へ専用 VJP を追加しない）、バックエンド間
//! 数値一致もこれら既存演算の parity 契約にそのまま帰着する。「対応する
//! Op／`BackendOps` メソッドを追加する」という受入要件は、分解先の演算が
//! すでに CPU／CUDA／Metal 全てに実装済みであるため合成によって自動的に
//! 充足される。
//!
//! **sub-issue (a)（#1639・PR #1845）との関係（重要）**: 本イシュー着手
//! 時点で #1639 は未マージのため、`Var::scaled_dot_product_attention`
//! （`crate::attention::scaled_dot_product_attention`）を呼べない。
//! 本ファイル内に **private** な複製 [`sdpa_compose`] を置き、PR #1845 の
//! `crate::attention::scaled_dot_product_attention` と数式・mask 極性
//! （`true` = attend。PyTorch bool mask 規約）・causal 規約（top-left
//! aligned `j <= i`。非正方形状も対応）を完全に一致させる（`var.rs`／
//! `lib.rs` には一切触れないため #1639 との併行実装が衝突しない）。
//! **#1639 マージ後、[`sdpa_compose`] は `Var::scaled_dot_product_attention`
//! 呼び出しへ置き換える対象**（別 PR。追跡先は本イシューの PR 本文）。
//!
//! **mask 極性の注意**: 本モジュールが受理する `attn_mask` は PyTorch
//! `F.scaled_dot_product_attention` と同じ `true` = attend 規約であり、
//! PyTorch `nn.MultiheadAttention` の bool mask（`True` = blocked）とは
//! **極性が逆**である。
//!
//! **入出力契約（rank-3）**: `query: [B, L, E]`・`key`/`value: [B, S,
//! kdim]`/`[B, S, vdim]`（`batch_first=true`。既定）または `[L, B, E]`／
//! `[S, B, kdim]`／`[S, B, vdim]`（`batch_first=false`） → 出力は入力と
//! 同じ軸順（`[B, L, E]` または `[L, B, E]`）。イシュー #2163（親
//! #2131）で [`MultiheadAttentionConfig`]（`batch_first`・`kdim`・
//! `vdim`）・[`MultiheadAttentionVars::forward_with_key_padding_mask`]
//! （`key_padding_mask`）を追加し対象外を縮小した（「構築時オプション
//! （#2163）」「呼び出し時オプション（#2163）」節参照）。unbatched
//! `[L, E]`・`dropout_p`（MHA への結線は対象外。`Var::dropout` 自体は
//! #1603 で実装済み）・`need_weights`／attention weights の
//! 返却・`add_bias_kv`／`add_zero_attn`・packed `in_proj_weight`／
//! PyTorch `state_dict` 対応付け（#1616）は対象外のまま
//! （`out-of-scope-tracking.md`）。
//!
//! **構築時オプション（イシュー #2163・親 #2131）**: [`MultiheadAttentionConfig`]
//! で `batch_first`（既定 `true` = 現行挙動。PyTorch 既定 `False` とは
//! 異なる）・`kdim`／`vdim`（既定 `None` = `embed_dim`。`k_proj`／
//! `v_proj` の in_features を `embed_dim` から独立させる）を指定できる。
//! `MultiheadAttention::new`（シグネチャ不変）は内部で `from_config`
//! （kdim = vdim = embed_dim・batch_first = true 固定の config）へ委譲
//! するため、既存呼び出し元の初期値は bit 同一のまま変わらない。
//!
//! **呼び出し時オプション（イシュー #2163）**: [`MultiheadAttentionVars::
//! forward_with_key_padding_mask`] で `key_padding_mask: Tensor<bool>
//! [B, S]`（`true` = attend。本モジュールの `attn_mask` と同じ極性——
//! PyTorch `nn.MultiheadAttention` の `key_padding_mask`〈`True` = 無視〉
//! とは**逆**）を指定できる。既存 [`MultiheadAttentionVars::forward`]
//! は `key_padding_mask = None`・`batch_first = true` の既定経路として
//! 本メソッドへ委譲するのみで、テープに積むノード列は変更前と完全に
//! 同一（bit 同一保証。`forward` doc 参照）。
//!
//! **facade 公開は保留（承認事項）**: `MultiheadAttentionConfig` の
//! facade 再エクスポート・`compat::Sequential` のオプション付き構築
//! メソッドはいずれも未承認のため追加していない
//! （`crates/facade/src/lib.rs` の `MhaOptionsHoldDoctestGuard`・
//! `crates/facade/tests/api_surface.rs` の否定ガードで固定）。
//!
//! **`Module` trait との関係**: `impl Module for MultiheadAttention` の
//! `forward` は self-attention（`q=k=v=input`・mask なし・非 causal）を
//! 表す。`forward_host`（tape 不要推論経路）は trait 既定のまま
//! （`Unsupported` fail-safe）で本イシューでは実装しない（対象外）。
//! `compat::Sequential` 用の `as_linear`／`as_relu` フックはいずれも
//! trait 既定（オーバーライドしない）のままとする——`MultiheadAttention`
//! は `Linear` の直接な代替ではなく `ReLU` 融合対象でもないため、
//! それらのフックに誤って拾われることはない。学習可能パラメータの
//! 認識には専用の [`Module::as_multihead_attention`] フックを使う
//! （イシュー #1760・親 #1618 で `compat::Sequential::add_multihead_attention`
//! として結線済み）。
//!
//! **KV キャッシュ（イシュー #2084・親 #2059。設計正本
//! `docs/kv-cache-design.md`）**: [`KvCache`]・
//! [`MultiheadAttentionVars::forward_with_cache`]・[`StatefulAttention`]
//! を追加した（K-1 最小版）。新規 `Op`／`BackendOps`／カーネル／依存は
//! 追加せず、既存 `Var` 演算（[`Var::cat`]・[`Tape::var_no_grad`]・
//! [`project`]／[`split_heads`]／[`sdpa_compose`]）の合成のみで実装する
//! （`docs/kv-cache-design.md` §3.1 contract 確認表）。
//!
//! - **ホスト保持**: `KvCache` は `Tensor<f32>` としてホスト側に置き、
//!   `Tape` の外にある（`TapeNode::value` がホスト `OnceCell<Tensor<f32>>`
//!   である現行構造の制約。`docs/kv-cache-design.md` §0）。decode
//!   ループでの `Tape` 再作成／`Tape::reset` の影響を受けない。
//! - **mask 規則**（`is_causal` を引数に持たず内部で決定。decode で
//!   `is_causal=true` を渡すと top-left aligned `causal_blocked_mask`
//!   が先頭 key にしか attend しない誤答を返す落とし穴を構造的に塞ぐ）:
//!   `cache` が空（prefill）→ causal、`cache` 非空・`L_new == 1`
//!   （通常の decode）→ mask なし、`cache` 非空・`L_new > 1`（複数
//!   トークン追記）→ [`offset_allowed_mask`] を明示指定。
//! - **勾配の truncated 意味論**: `cache` は `Tape::var_no_grad` の葉
//!   として登録するため、過去ステップへは勾配が流れない。現ステップの
//!   射影パラメータへの勾配は通常どおり流れる。
//! - **原子的な更新**: 全段が成功した後にのみ `cache` へ書き戻す。
//!   途中でエラーになった場合、`cache` は変化しない。
//! - **K-3（デバイス常駐・リングバッファ）／`sdpa_compose` の
//!   `crate::attention::scaled_dot_product_attention` への置換・
//!   `TransformerEncoderLayer` の decode 版**はいずれも対象外
//!   （`docs/kv-cache-design.md` §7 スコープ外・§6 承認事項）。
//! - **facade 公開（K-2）は未承認のため保留**: `add_stateful_attention`・
//!   `StatefulAttention` 相当の facade `pub fn`／再エクスポートは
//!   追加していない（`crates/facade/tests/api_surface.rs` の否定
//!   ガードで固定。`docs/kv-cache-design.md` §6 承認事項 2）。

use fandhe_ai_tensor_core::{Activation, ScalarDType, ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::nn::init::{
    ATTN_K_SEED_SALT, ATTN_OUT_SEED_SALT, ATTN_Q_SEED_SALT, ATTN_V_SEED_SALT, derive_seed,
};
use crate::nn::linear::{Linear, LinearVars, linear_forward_low_precision};
use crate::nn::module::{Module, prefixed, strip_child_prefix};
use crate::tape::Tape;
use crate::var::Var;

/// `embed_dim`／`num_heads` の共通検証（`MultiheadAttention::new`／
/// `from_parameters`・`MultiheadAttentionVars::new` の全経路が通る
/// fail-closed ゲート）。`embed_dim % num_heads != 0` は PyTorch
/// `nn.MultiheadAttention` と同じ拒否条件（`head_dim = embed_dim /
/// num_heads` が割り切れない場合、head 分割 `reshape([B, L, H, Dh])`
/// 自体が要素数不一致で失敗するため、より分かりやすいメッセージで
/// 事前に拒否する）。
fn validate_embed_heads(embed_dim: usize, num_heads: usize) -> Result<(), AutodiffError> {
    if embed_dim == 0 {
        return Err(AutodiffError::InvalidArgument(
            "MultiheadAttention: embed_dim must be > 0".to_string(),
        ));
    }
    if num_heads == 0 {
        return Err(AutodiffError::InvalidArgument(
            "MultiheadAttention: num_heads must be > 0".to_string(),
        ));
    }
    if !embed_dim.is_multiple_of(num_heads) {
        return Err(AutodiffError::InvalidArgument(format!(
            "MultiheadAttention: embed_dim ({embed_dim}) must be divisible by num_heads \
             ({num_heads})"
        )));
    }
    Ok(())
}

/// `MultiheadAttention::from_parameters`／`from_parameters_with_config`
/// 向けの検証（`Tensor<f32>` を保持する `Linear` 本体を対象）。`q`／
/// `out` は正方 `[E, E]`、`k` は `[kdim, E]`、`v` は `[vdim, E]`
/// （`Linear` の weight レイアウトは `[in_features, out_features]`——
/// `project` の `x.reshape([bl, in]).matmul(weight)` 参照）であること・
/// bias が全て `Some`（各 `[E]`）または全て `None` であることを検査し、
/// `(E, kdim, vdim)` を返す（イシュー #2163 で `kdim`／`vdim` を weight
/// 形状から推論するよう緩和。`kdim = vdim = E` の入力に対する既存の
/// エラー値は変更しない）。`Linear::from_parameters` 自身は各層を独立に
/// 検証する（rank・zero-K・bias shape）ため、本関数はその上に「4 層を
/// またぐ整合性」のみを追加で検証する。
fn validate_linear_projections(
    q: &Linear,
    k: &Linear,
    v: &Linear,
    out: &Linear,
    num_heads: usize,
) -> Result<(usize, usize, usize), AutodiffError> {
    // `Linear::from_parameters`／`Linear::new` はいずれも weight を
    // rank 2 として構築する契約のため、ここでの rank 検査は本モジュール
    // 内での防御的な二重チェックに留まる（`Linear` の非公開フィールド
    // を直接いじる経路が存在しないため実運用では到達しないが、fail-
    // closed 方針として明示的に検査する）。
    let q_shape = q.weight().shape();
    if q_shape.len() != 2 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: q_shape.len(),
        }));
    }
    if q_shape[0] != q_shape[1] {
        return Err(AutodiffError::InvalidArgument(format!(
            "MultiheadAttention::from_parameters: q_proj weight must be square [E, E], got \
             {q_shape:?}"
        )));
    }
    let e = q_shape[0];
    let out_shape = out.weight().shape();
    if out_shape != [e, e] {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: out_shape.to_vec(),
            rhs: vec![e, e],
        }));
    }
    let k_shape = k.weight().shape();
    if k_shape.len() != 2 || k_shape[1] != e {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: k_shape.to_vec(),
            rhs: vec![k_shape.first().copied().unwrap_or(0), e],
        }));
    }
    let kdim = k_shape[0];
    let v_shape = v.weight().shape();
    if v_shape.len() != 2 || v_shape[1] != e {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: v_shape.to_vec(),
            rhs: vec![v_shape.first().copied().unwrap_or(0), e],
        }));
    }
    let vdim = v_shape[0];

    let bias_some = [
        q.bias().is_some(),
        k.bias().is_some(),
        v.bias().is_some(),
        out.bias().is_some(),
    ];
    if bias_some.iter().any(|&b| b) && !bias_some.iter().all(|&b| b) {
        return Err(AutodiffError::InvalidArgument(
            "MultiheadAttention::from_parameters: q_proj/k_proj/v_proj/out_proj biases must be \
             all Some or all None"
                .to_string(),
        ));
    }

    validate_embed_heads(e, num_heads)?;
    Ok((e, kdim, vdim))
}

/// `MultiheadAttentionVars::new`／`new_with_config` 向けの検証
/// （`Var`〈テープ登録済み〉を保持する `LinearVars` を対象）。
/// [`validate_linear_projections`] と同じ整合性（`q`／`out` は正方
/// `[E, E]`、`k` は `[kdim, E]`、`v` は `[vdim, E]`、bias 全 `Some`／全
/// `None`）に加え、4 層が同一 `Tape` に属することを検査する
/// （`Var::check_same_tape`）。`(E, kdim, vdim)` を返す（イシュー #2163
/// で `kdim`／`vdim` を weight 形状から推論するよう緩和。`kdim = vdim =
/// E` の入力に対する既存のエラー値は変更しない）。`kdim == 0`／
/// `vdim == 0`（`k`／`v` weight が `[0, E]`）は明示的に拒否する
/// （`LinearVars` は `weight`/`bias` が `pub` のため `Linear::
/// from_parameters` の zero-K 検証を経由せずに構築できてしまい、
/// `MultiheadAttention::from_config` 側の同検証（本ファイル
/// `from_config`）と非対称だった。PR #2279 レビュー指摘）。
fn validate_projection_vars<'t>(
    q: &LinearVars<'t>,
    k: &LinearVars<'t>,
    v: &LinearVars<'t>,
    out: &LinearVars<'t>,
    num_heads: usize,
) -> Result<(usize, usize, usize), AutodiffError> {
    q.weight.check_same_tape(&k.weight)?;
    q.weight.check_same_tape(&v.weight)?;
    q.weight.check_same_tape(&out.weight)?;

    let q_shape = q.weight.shape();
    if q_shape.len() != 2 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: q_shape.len(),
        }));
    }
    if q_shape[0] != q_shape[1] {
        return Err(AutodiffError::InvalidArgument(format!(
            "MultiheadAttentionVars::new: q weight must be square [E, E], got {q_shape:?}"
        )));
    }
    let e = q_shape[0];
    let out_shape = out.weight.shape();
    if out_shape.as_slice() != [e, e] {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: out_shape,
            rhs: vec![e, e],
        }));
    }
    let k_shape = k.weight.shape();
    if k_shape.len() != 2 || k_shape[1] != e {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: k_shape.clone(),
            rhs: vec![k_shape.first().copied().unwrap_or(0), e],
        }));
    }
    let kdim = k_shape[0];
    if kdim == 0 {
        return Err(AutodiffError::InvalidArgument(
            "MultiheadAttentionVars::new: k weight in_features (kdim) must be > 0".to_string(),
        ));
    }
    let v_shape = v.weight.shape();
    if v_shape.len() != 2 || v_shape[1] != e {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: v_shape.clone(),
            rhs: vec![v_shape.first().copied().unwrap_or(0), e],
        }));
    }
    let vdim = v_shape[0];
    if vdim == 0 {
        return Err(AutodiffError::InvalidArgument(
            "MultiheadAttentionVars::new: v weight in_features (vdim) must be > 0".to_string(),
        ));
    }

    let bias_some = [
        q.bias.is_some(),
        k.bias.is_some(),
        v.bias.is_some(),
        out.bias.is_some(),
    ];
    if bias_some.iter().any(|&b| b) && !bias_some.iter().all(|&b| b) {
        return Err(AutodiffError::InvalidArgument(
            "MultiheadAttentionVars::new: q/k/v/out biases must be all Some or all None"
                .to_string(),
        ));
    }
    for b in [&q.bias, &k.bias, &v.bias, &out.bias].into_iter().flatten() {
        // weight 側（q/k/v/out）は既に `check_same_tape` で相互検査済みのため、
        // bias は代表として q.weight とのみ検査すれば同一 Tape であることが
        // 推移的に保証される（codex-review 指摘: weight のみの検査では
        // 同形状の bias が別 Tape でも構築が成功し、後続の forward 内の
        // `add` まで TapeMismatch が遅延してしまう）。すべての projection
        // の bias は `[E]`（q/k/v/out 4 層とも out_features は常に E。
        // `kdim`/`vdim` は in_features 側のみ）。
        q.weight.check_same_tape(b)?;
        let bshape = b.shape();
        if bshape.as_slice() != [e] {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: bshape,
                rhs: vec![e],
            }));
        }
    }

    validate_embed_heads(e, num_heads)?;
    Ok((e, kdim, vdim))
}

/// `MultiheadAttention` の構築時オプション（イシュー #2163・親
/// #2131）。フィールドは private（リテラル構築を最初から不可能にし、
/// 将来のオプション追加も非破壊にする——`nn::conv::Conv2dConfig` 等の
/// 既存 `*Config` 型と同じビルダー方針）。`Default` は derive しない
/// （`embed_dim = 0`／`num_heads = 0` は不正値のため。[`MultiheadAttentionConfig::new`]
/// が唯一のコンストラクタ）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiheadAttentionConfig {
    embed_dim: usize,
    num_heads: usize,
    bias: bool,
    batch_first: bool,
    kdim: Option<usize>,
    vdim: Option<usize>,
}

impl MultiheadAttentionConfig {
    /// `bias = true`（PyTorch 既定・[`MultiheadAttention::new`] と同じ）・
    /// `batch_first = true`（= 現行挙動。PyTorch 既定 `False` とは異なる
    /// ことに注意）・`kdim`/`vdim = None`（= `embed_dim`）で初期化する。
    /// `with_*` メソッドで個別に上書きする（`embed_dim`／`num_heads`
    /// 自体の検証は [`MultiheadAttention::from_config`] が構築時に行う）。
    pub fn new(embed_dim: usize, num_heads: usize) -> MultiheadAttentionConfig {
        MultiheadAttentionConfig {
            embed_dim,
            num_heads,
            bias: true,
            batch_first: true,
            kdim: None,
            vdim: None,
        }
    }

    /// q/k/v/out projection の bias 有無を上書きする。
    pub fn with_bias(mut self, bias: bool) -> MultiheadAttentionConfig {
        self.bias = bias;
        self
    }

    /// `batch_first` を上書きする（`true`: `[B, Len, *]`・`false`:
    /// `[Len, B, *]`。[`MultiheadAttentionVars::forward_with_key_padding_mask`]
    /// 参照）。
    pub fn with_batch_first(mut self, batch_first: bool) -> MultiheadAttentionConfig {
        self.batch_first = batch_first;
        self
    }

    /// `k_proj` の in_features（`kdim`）を上書きする（既定 `embed_dim`）。
    pub fn with_kdim(mut self, kdim: usize) -> MultiheadAttentionConfig {
        self.kdim = Some(kdim);
        self
    }

    /// `v_proj` の in_features（`vdim`）を上書きする（既定 `embed_dim`）。
    pub fn with_vdim(mut self, vdim: usize) -> MultiheadAttentionConfig {
        self.vdim = Some(vdim);
        self
    }

    /// `new` に渡した埋め込み次元 `E`。
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// `new` に渡したヘッド数 `H`。
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// `with_bias` で設定した bias 有無（既定 `true`）。
    pub fn bias(&self) -> bool {
        self.bias
    }

    /// `with_batch_first` で設定した `batch_first`（既定 `true`）。
    pub fn batch_first(&self) -> bool {
        self.batch_first
    }

    /// `k_proj` の in_features（`with_kdim` 未設定なら `embed_dim` へ
    /// 解決した値を返す）。
    pub fn kdim(&self) -> usize {
        self.kdim.unwrap_or(self.embed_dim)
    }

    /// `v_proj` の in_features（`with_vdim` 未設定なら `embed_dim` へ
    /// 解決した値を返す）。
    pub fn vdim(&self) -> usize {
        self.vdim.unwrap_or(self.embed_dim)
    }
}

/// `MultiheadAttention` のパラメータ本体（`nn::Linear` 4 層：q/k/v/out
/// projection）。`nn::Linear`／`nn::rnn` と同じ「パラメータ本体（本
/// 構造体）とテープ上の `Var` を保持する `*Vars`（[`MultiheadAttentionVars`]）」
/// の分離方針を踏襲する。`kdim`／`vdim`／`batch_first`（イシュー
/// #2163）は private フィールドとして追加した（既存の 2 フィールド
/// `embed_dim`／`num_heads` と同じ理由で `pub` にしない。getter
/// 経由でのみ読める）。
pub struct MultiheadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    embed_dim: usize,
    num_heads: usize,
    kdim: usize,
    vdim: usize,
    batch_first: bool,
}

impl MultiheadAttention {
    /// 決定的シードで q/k/v/out の 4 `Linear` 層を構築する
    /// （`Linear::new` と同じ `U(-1/√E, 1/√E)` 一様初期化。PyTorch
    /// `nn.MultiheadAttention` の既定初期化〈xavier_uniform〉とは異なる
    /// ことに注意——本イシューは数値的な再現ではなく機能を対象とする）。
    /// 単一の呼び出し `seed` から `nn/init.rs` の 4 ソルト
    /// （`ATTN_Q_SEED_SALT`〜`ATTN_OUT_SEED_SALT`）で 4 系統の独立した
    /// `Linear::new` 呼び出しシードを導出する。
    ///
    /// イシュー #2163 で [`MultiheadAttentionConfig::new(embed_dim,
    /// num_heads).with_bias(bias)`]（`kdim = vdim = embed_dim`・
    /// `batch_first = true`）から [`Self::from_config`] へ委譲する形へ
    /// 変更した。シード導出は変更前と同一（同じソルト・同じ
    /// `Linear::new(embed_dim, embed_dim, bias, ..)` 呼び出し）のため、
    /// 返す 4 層の初期値は変更前と **bit 同一**（単体テスト
    /// `new_matches_from_config_bit_identical` で固定）。
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        bias: bool,
        seed: u64,
    ) -> Result<MultiheadAttention, AutodiffError> {
        Self::from_config(
            &MultiheadAttentionConfig::new(embed_dim, num_heads).with_bias(bias),
            seed,
        )
    }

    /// [`MultiheadAttentionConfig`] から構築する（イシュー #2163）。
    /// `q_proj`／`out_proj` は `[E, E]`、`k_proj` は `[kdim, E]`、
    /// `v_proj` は `[vdim, E]`（`Linear::new(in, out, ..)` の呼び出し順）
    /// で構築し、シード導出は [`Self::new`] と同一の 4 ソルトを使う。
    ///
    /// # Errors
    ///
    /// `embed_dim`／`kdim`／`vdim` のいずれかが 0、`num_heads` が 0、
    /// `embed_dim % num_heads != 0` のいずれかで
    /// `AutodiffError::InvalidArgument` を返す。
    pub fn from_config(
        config: &MultiheadAttentionConfig,
        seed: u64,
    ) -> Result<MultiheadAttention, AutodiffError> {
        let embed_dim = config.embed_dim();
        let num_heads = config.num_heads();
        let kdim = config.kdim();
        let vdim = config.vdim();
        validate_embed_heads(embed_dim, num_heads)?;
        if kdim == 0 {
            return Err(AutodiffError::InvalidArgument(
                "MultiheadAttentionConfig: kdim must be > 0".to_string(),
            ));
        }
        if vdim == 0 {
            return Err(AutodiffError::InvalidArgument(
                "MultiheadAttentionConfig: vdim must be > 0".to_string(),
            ));
        }
        let bias = config.bias();
        let q_proj = Linear::new(
            embed_dim,
            embed_dim,
            bias,
            derive_seed(seed, ATTN_Q_SEED_SALT),
        )?;
        let k_proj = Linear::new(kdim, embed_dim, bias, derive_seed(seed, ATTN_K_SEED_SALT))?;
        let v_proj = Linear::new(vdim, embed_dim, bias, derive_seed(seed, ATTN_V_SEED_SALT))?;
        let out_proj = Linear::new(
            embed_dim,
            embed_dim,
            bias,
            derive_seed(seed, ATTN_OUT_SEED_SALT),
        )?;
        Ok(MultiheadAttention {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            embed_dim,
            num_heads,
            kdim,
            vdim,
            batch_first: config.batch_first(),
        })
    }

    /// 明示的な 4 層（q/k/v/out projection）から構築する（テスト・
    /// safetensors 等の外部由来パラメータロード経路向けの入口。
    /// `Linear::from_parameters` と同じ位置づけ）。`clippy::
    /// too_many_arguments` を避けるため、8 個の生テンソルではなく
    /// 構築済みの [`Linear`] 4 層を受け取る（呼び出し元は
    /// `Linear::from_parameters(weight, bias)` で個別に構築する）。
    /// `kdim`／`vdim` は `k_proj`／`v_proj` の weight 形状から推論する
    /// （イシュー #2163。`validate_linear_projections` 参照）。
    /// `batch_first = true` 固定（`batch_first=false` を構築したい場合は
    /// [`Self::from_parameters_with_config`] を使う）。
    pub fn from_parameters(
        num_heads: usize,
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        out_proj: Linear,
    ) -> Result<MultiheadAttention, AutodiffError> {
        let (embed_dim, kdim, vdim) =
            validate_linear_projections(&q_proj, &k_proj, &v_proj, &out_proj, num_heads)?;
        Ok(MultiheadAttention {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            embed_dim,
            num_heads,
            kdim,
            vdim,
            batch_first: true,
        })
    }

    /// [`Self::from_parameters`] の config 版（イシュー #2163）。
    /// `config` の `embed_dim`／`num_heads`／`kdim`／`vdim`／`bias` と
    /// 4 層の実際の shape・bias 有無が一致することを検証する
    /// （`validate_linear_projections`〈非公開のためコードスパン表記で
    /// 参照しリンク化しない〉に加え、推論した `(E, kdim, vdim)` が
    /// `config` の値と一致するかの追加検査）。`batch_first` は
    /// `config` からそのまま設定する。
    ///
    /// # Errors
    ///
    /// [`Self::from_parameters`] と同じ shape 検証に加え、推論した
    /// `embed_dim`／`kdim`／`vdim` のいずれかが `config` の値と不一致、
    /// または `q_proj`（代表）の bias 有無が `config.bias()` と不一致
    /// のとき `AutodiffError::InvalidArgument` を返す。
    pub fn from_parameters_with_config(
        config: &MultiheadAttentionConfig,
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        out_proj: Linear,
    ) -> Result<MultiheadAttention, AutodiffError> {
        let (embed_dim, kdim, vdim) =
            validate_linear_projections(&q_proj, &k_proj, &v_proj, &out_proj, config.num_heads())?;
        if embed_dim != config.embed_dim() || kdim != config.kdim() || vdim != config.vdim() {
            return Err(AutodiffError::InvalidArgument(format!(
                "MultiheadAttention::from_parameters_with_config: 推論した (embed_dim={embed_dim}, \
                 kdim={kdim}, vdim={vdim}) が config (embed_dim={}, kdim={}, vdim={}) と不一致",
                config.embed_dim(),
                config.kdim(),
                config.vdim()
            )));
        }
        if q_proj.bias().is_some() != config.bias() {
            return Err(AutodiffError::InvalidArgument(format!(
                "MultiheadAttention::from_parameters_with_config: q_proj の bias 有無 ({}) が \
                 config.bias() ({}) と不一致",
                q_proj.bias().is_some(),
                config.bias()
            )));
        }
        Ok(MultiheadAttention {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            embed_dim,
            num_heads: config.num_heads(),
            kdim,
            vdim,
            batch_first: config.batch_first(),
        })
    }

    /// `new`／`from_parameters` に渡した埋め込み次元 `E`（`q_proj`／
    /// `out_proj` が正方 `[E, E]` であることを構築時に検証済み）。
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// `new`／`from_parameters` に渡したヘッド数 `H`（`E % H == 0`
    /// であることを構築時に検証済み）。
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// `embed_dim / num_heads`（`new`／`from_parameters` が
    /// `embed_dim % num_heads == 0` を検証済みのため整数除算で正確）。
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }

    /// `k_proj` の in_features（イシュー #2163。既定 `embed_dim`）。
    pub fn kdim(&self) -> usize {
        self.kdim
    }

    /// `v_proj` の in_features（イシュー #2163。既定 `embed_dim`）。
    pub fn vdim(&self) -> usize {
        self.vdim
    }

    /// 入出力の軸順（イシュー #2163。`true`: `[B, Len, *]`・既定。
    /// `false`: `[Len, B, *]`）。
    pub fn batch_first(&self) -> bool {
        self.batch_first
    }

    /// query projection（`nn::Linear`。`[E, E]`）への参照。
    pub fn q_proj(&self) -> &Linear {
        &self.q_proj
    }

    /// key projection（`nn::Linear`。`[kdim, E]`）への参照。
    pub fn k_proj(&self) -> &Linear {
        &self.k_proj
    }

    /// value projection（`nn::Linear`。`[E, E]`）への参照。
    pub fn v_proj(&self) -> &Linear {
        &self.v_proj
    }

    /// output projection（`nn::Linear`。`[E, E]`。結合後の全ヘッド出力
    /// を `E` 次元へ写す）への参照。
    pub fn out_proj(&self) -> &Linear {
        &self.out_proj
    }

    /// このステップの `tape` へ 4 層すべての `weight`／`bias` を葉
    /// ノードとして登録し、`forward` を呼べる [`MultiheadAttentionVars`]
    /// を返す（`Linear::bind` と同じ「毎ステップ作り直す」契約。
    /// `nn/linear.rs` の `Tape` ライフサイクル節参照）。`new`／
    /// `from_parameters` が構築時に不変条件（4 層とも正方 `[E, E]`・
    /// 同一 `E`・bias 全 `Some`／全 `None`・`E % H == 0`）を検証済みの
    /// ため、`bind` 自体は再検証しない。
    pub fn bind<'t>(&self, tape: &'t Tape) -> MultiheadAttentionVars<'t> {
        MultiheadAttentionVars {
            q: self.q_proj.bind(tape),
            k: self.k_proj.bind(tape),
            v: self.v_proj.bind(tape),
            out: self.out_proj.bind(tape),
            embed_dim: self.embed_dim,
            num_heads: self.num_heads,
            kdim: self.kdim,
            vdim: self.vdim,
            batch_first: self.batch_first,
        }
    }
}

/// `MultiheadAttention::bind` が返す、1 ステップ分のテープに登録済み
/// パラメータ。`q`／`k`／`v`／`out` を `pub` にする理由は `LinearVars`
/// と同じ（`Tape::backward` 後に `Gradients::get(&vars.q.weight)` 等で
/// 勾配を取り出すのは呼び出し側の責務）に加え、facade 横断 parity
/// テスト（`crates/facade/tests/`）が [`MultiheadAttentionVars::new`]
/// （下記）経由で `tape.var(&tensor)` から組み立てた `LinearVars` を
/// 直接渡せるようにするため（`facade` は本クレートの `pub(crate)` API
/// に触れられないため、この構成が facade テストから `MultiheadAttention`
/// 相当の forward を呼ぶ唯一の到達経路になる）。
pub struct MultiheadAttentionVars<'t> {
    pub q: LinearVars<'t>,
    pub k: LinearVars<'t>,
    pub v: LinearVars<'t>,
    pub out: LinearVars<'t>,
    embed_dim: usize,
    num_heads: usize,
    kdim: usize,
    vdim: usize,
    batch_first: bool,
}

impl<'t> MultiheadAttentionVars<'t> {
    /// `LinearVars` 4 個（q/k/v/out projection）から直接構築する。
    /// [`MultiheadAttention::bind`] を経由しない到達経路（facade 横断
    /// parity テスト向け。モジュール doc 参照）のため、`bind` が
    /// 省略していた不変条件の検証（`q`/`out` は正方 `[E, E]`・`k`/`v`
    /// は `[kdim/vdim, E]`・bias 全 `Some`／全 `None`・同一 `Tape`・
    /// `E % num_heads == 0`）を ここで行う（`validate_projection_vars`。
    /// 非公開のためコードスパン表記で参照しリンク化しない）。
    /// `batch_first = true` 固定（`false` を構築したい場合は
    /// [`Self::new_with_config`] を使う）。
    pub fn new(
        num_heads: usize,
        q: LinearVars<'t>,
        k: LinearVars<'t>,
        v: LinearVars<'t>,
        out: LinearVars<'t>,
    ) -> Result<MultiheadAttentionVars<'t>, AutodiffError> {
        let (embed_dim, kdim, vdim) = validate_projection_vars(&q, &k, &v, &out, num_heads)?;
        Ok(MultiheadAttentionVars {
            q,
            k,
            v,
            out,
            embed_dim,
            num_heads,
            kdim,
            vdim,
            batch_first: true,
        })
    }

    /// [`Self::new`] の config 版（イシュー #2163）。facade 横断 parity
    /// テスト（`crates/facade/tests/`）が `batch_first=false`・
    /// `kdim`/`vdim != embed_dim` を組み立てる唯一の到達経路
    /// （`MultiheadAttention::bind` は crate-internal のため）。
    /// `validate_projection_vars`〈非公開のためコードスパン表記で参照し
    /// リンク化しない〉に加え、推論した `(E, kdim, vdim)` が `config`
    /// の値と一致するかを追加検査する
    /// （[`MultiheadAttention::from_parameters_with_config`] と同型）。
    ///
    /// # Errors
    ///
    /// [`Self::new`] と同じ shape／tape 検証に加え、推論した
    /// `embed_dim`／`kdim`／`vdim` のいずれかが `config` の値と不一致、
    /// または `q`（代表）の bias 有無が `config.bias()` と不一致のとき
    /// `AutodiffError::InvalidArgument` を返す。
    pub fn new_with_config(
        config: &MultiheadAttentionConfig,
        q: LinearVars<'t>,
        k: LinearVars<'t>,
        v: LinearVars<'t>,
        out: LinearVars<'t>,
    ) -> Result<MultiheadAttentionVars<'t>, AutodiffError> {
        let (embed_dim, kdim, vdim) =
            validate_projection_vars(&q, &k, &v, &out, config.num_heads())?;
        if embed_dim != config.embed_dim() || kdim != config.kdim() || vdim != config.vdim() {
            return Err(AutodiffError::InvalidArgument(format!(
                "MultiheadAttentionVars::new_with_config: 推論した (embed_dim={embed_dim}, \
                 kdim={kdim}, vdim={vdim}) が config (embed_dim={}, kdim={}, vdim={}) と不一致",
                config.embed_dim(),
                config.kdim(),
                config.vdim()
            )));
        }
        // `validate_projection_vars` は 4 層の bias 有無が互いに一致する
        // （全 `Some`／全 `None`）ことのみ検証し、`config.bias()` との
        // 整合は見ない。`MultiheadAttention::from_parameters_with_config`
        // と契約を揃えるため、代表として `q` の bias 有無を
        // `config.bias()` と照合する（イシュー #2163 レビュー指摘）。
        if q.bias.is_some() != config.bias() {
            return Err(AutodiffError::InvalidArgument(format!(
                "MultiheadAttentionVars::new_with_config: q の bias 有無 ({}) が config.bias() \
                 ({}) と不一致",
                q.bias.is_some(),
                config.bias()
            )));
        }
        Ok(MultiheadAttentionVars {
            q,
            k,
            v,
            out,
            embed_dim,
            num_heads: config.num_heads(),
            kdim,
            vdim,
            batch_first: config.batch_first(),
        })
    }

    /// `new` に渡した埋め込み次元 `E`（`q`/`out` は正方 `[E, E]` かつ
    /// 同一 `Tape` であることを構築時に検証済み）。
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// `new` に渡したヘッド数 `H`（`E % H == 0` であることを構築時に
    /// 検証済み）。
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// `embed_dim / num_heads`（`new` が `embed_dim % num_heads == 0`
    /// を検証済みのため整数除算で正確）。
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }

    /// `k` の in_features（イシュー #2163。既定 `embed_dim`）。
    pub fn kdim(&self) -> usize {
        self.kdim
    }

    /// `v` の in_features（イシュー #2163。既定 `embed_dim`）。
    pub fn vdim(&self) -> usize {
        self.vdim
    }

    /// 入出力の軸順（イシュー #2163。`true`: `[B, Len, *]`・既定。
    /// `false`: `[Len, B, *]`）。
    pub fn batch_first(&self) -> bool {
        self.batch_first
    }

    /// `y = MultiheadAttention(query, key, value)`。`self.batch_first()`
    /// が `true`（既定）なら `query: [B, L, E]`・`key: [B, S, kdim]`・
    /// `value: [B, S, vdim]` → `[B, L, E]`、`false` なら `query: [L, B,
    /// E]`・`key: [S, B, kdim]`・`value: [S, B, vdim]` → `[L, B, E]`
    /// （モジュール doc「入出力契約」参照）。
    ///
    /// [`Self::forward_with_key_padding_mask`] へ `key_padding_mask =
    /// None` で委譲する薄いラッパー（イシュー #2163）。`batch_first ==
    /// true`（既定）のとき、テープに積むノード列は本イシュー着手前の
    /// `forward` 実装と **bit 同一**（`key_padding_mask.is_none()` かつ
    /// `batch_first` 分岐が transpose を挟まないため。単体テスト
    /// `forward_default_path_is_bit_identical_to_pre_2163` で固定）。
    ///
    /// # Errors
    ///
    /// [`Self::forward_with_key_padding_mask`] を参照。
    pub fn forward(
        &self,
        query: &Var<'t>,
        key: &Var<'t>,
        value: &Var<'t>,
        attn_mask: Option<&Tensor<bool>>,
        is_causal: bool,
    ) -> Result<Var<'t>, AutodiffError> {
        self.forward_with_key_padding_mask(query, key, value, attn_mask, None, is_causal)
    }

    /// [`Self::forward`] の拡張版（イシュー #2163）。`key_padding_mask:
    /// Option<&Tensor<bool>>`（`[B, S]`。`true` = attend。本モジュールの
    /// `attn_mask` と同じ極性——PyTorch `nn.MultiheadAttention` の
    /// `key_padding_mask`〈`True` = 無視〉とは**逆**）を追加で受け取る。
    ///
    /// 処理順序（`project`／`split_heads`／`sdpa_compose`／
    /// `combine_masks_with_key_padding` はいずれも非公開のためコード
    /// スパン表記で参照しリンク化しない）: ①テープ一致検査 → ②
    /// `batch_first == false` なら q/k/v を `transpose(0, 1)` で
    /// `[B, Len, *]` 化（`self.batch_first() == true` かつ
    /// `key_padding_mask.is_none()` なら本分岐に入らない——bit 同一
    /// 保証。[`Self::forward`] doc 参照）→ ③shape 検査（rank・`B`／
    /// `E`・`key` の最終軸が `kdim`・`value` の最終軸が `vdim`）→
    /// ④q/k/v projection（`project`） → ⑤head 分割（`split_heads`） →
    /// ⑥`attn_mask`／`key_padding_mask`／`is_causal` の合成
    /// （`key_padding_mask.is_some()` のときのみ
    /// `combine_masks_with_key_padding` を呼ぶ） → ⑦attention 本体
    /// （`sdpa_compose`） → ⑧head 結合 → ⑨out projection → ⑩
    /// `batch_first == false` なら出力を `transpose(0, 1)` で `[Len,
    /// B, E]` へ戻す。
    ///
    /// # Errors
    ///
    /// `query`／`key`／`value` の rank が 3 でない、バッチ次元 `B` が
    /// 不一致、`query` の最終軸が `embed_dim` と不一致、`key`／`value`
    /// の最終軸が `kdim`／`vdim` と不一致、`key`／`value` の `[B, S]`
    /// が食い違う、のいずれかで `AutodiffError::Shape` を返す。
    /// `key_padding_mask` の shape が `[B, S]` でない場合も
    /// `AutodiffError::Shape` を返す。`attn_mask` と `is_causal` の
    /// 同時指定・`attn_mask`／`key_padding_mask` の broadcast 不能・
    /// 全 masked 行は [`AutodiffError::InvalidArgument`]（`sdpa_compose`
    /// へ委譲、または本メソッドが `attn_mask + is_causal` を事前拒否）。
    /// テープ不一致は `check_same_tape`（`AutodiffError::TapeMismatch`）。
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_key_padding_mask(
        &self,
        query: &Var<'t>,
        key: &Var<'t>,
        value: &Var<'t>,
        attn_mask: Option<&Tensor<bool>>,
        key_padding_mask: Option<&Tensor<bool>>,
        is_causal: bool,
    ) -> Result<Var<'t>, AutodiffError> {
        query.check_same_tape(&self.q.weight)?;
        query.check_same_tape(key)?;
        query.check_same_tape(value)?;

        // `batch_first == false` の入力のみ `[Len, B, *]` -> `[B, Len,
        // *]` へ transpose する（zero-copy view）。`project` が
        // `NonContiguousReshape` 経路で `contiguous()` 再試行するため、
        // transpose 済み view をそのまま渡せる（モジュール doc「呼び出し
        // 時オプション」・`project` doc 参照）。既定経路（batch_first ==
        // true）は本分岐に入らないため bit 同一が保たれる。
        let (query, key, value): (std::borrow::Cow<'_, Var<'t>>, _, _) = if self.batch_first {
            (
                std::borrow::Cow::Borrowed(query),
                std::borrow::Cow::Borrowed(key),
                std::borrow::Cow::Borrowed(value),
            )
        } else {
            (
                std::borrow::Cow::Owned(query.transpose(0, 1)?),
                std::borrow::Cow::Owned(key.transpose(0, 1)?),
                std::borrow::Cow::Owned(value.transpose(0, 1)?),
            )
        };
        let query: &Var<'t> = &query;
        let key: &Var<'t> = &key;
        let value: &Var<'t> = &value;

        let q_shape = query.shape();
        let k_shape = key.shape();
        let v_shape = value.shape();
        if q_shape.len() != 3 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 3,
                actual: q_shape.len(),
            }));
        }
        if k_shape.len() != 3 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 3,
                actual: k_shape.len(),
            }));
        }
        if v_shape.len() != 3 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 3,
                actual: v_shape.len(),
            }));
        }

        let e = self.embed_dim;
        let kdim = self.kdim;
        let vdim = self.vdim;
        let (b, l) = (q_shape[0], q_shape[1]);
        if q_shape[2] != e {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: q_shape.clone(),
                rhs: vec![b, l, e],
            }));
        }
        if k_shape[0] != b || k_shape[2] != kdim {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: k_shape.clone(),
                rhs: vec![b, k_shape[1], kdim],
            }));
        }
        if v_shape[0] != b || v_shape[1] != k_shape[1] || v_shape[2] != vdim {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: v_shape.clone(),
                rhs: vec![b, k_shape[1], vdim],
            }));
        }
        let s = k_shape[1];

        let h = self.num_heads;
        // `validate_projection_vars`／`validate_embed_heads` が
        // `e % h == 0` かつ `e > 0`／`h > 0` を構築時に検証済みのため、
        // `dh` は常に 1 以上の整数（0 除算・`1/sqrt(0)` は生じない）。
        let dh = e / h;

        let q_proj = project(query, &self.q, b, l, e, e, None)?;
        let k_proj = project(key, &self.k, b, s, kdim, e, None)?;
        let v_proj = project(value, &self.v, b, s, vdim, e, None)?;

        let q_heads = split_heads(&q_proj, b, l, h, dh)?;
        let k_heads = split_heads(&k_proj, b, s, h, dh)?;
        let v_heads = split_heads(&v_proj, b, s, h, dh)?;

        let scale = 1.0f32 / (dh as f32).sqrt();
        let attn_out = match key_padding_mask {
            None => sdpa_compose(
                &q_heads, &k_heads, &v_heads, attn_mask, is_causal, scale, None,
            )?,
            Some(kpm) => {
                // `attn_mask + is_causal` は従来どおり事前拒否する
                // （`sdpa_compose` の排他検査と同一契約を、mask 合成前に
                // 前倒しで適用する）。
                if attn_mask.is_some() && is_causal {
                    return Err(AutodiffError::InvalidArgument(
                        "MultiheadAttentionVars::forward_with_key_padding_mask: attn_mask and \
                         is_causal are mutually exclusive"
                            .to_string(),
                    ));
                }
                let combined =
                    combine_masks_with_key_padding(attn_mask, kpm, is_causal, b, h, l, s)?;
                // `is_causal=false` で渡す: causal 分は `combined` へ
                // 既に織り込み済み（`sdpa_compose` の排他検査は上で
                // 前倒し済みのため通過する）。
                sdpa_compose(
                    &q_heads,
                    &k_heads,
                    &v_heads,
                    Some(&combined),
                    false,
                    scale,
                    None,
                )?
            }
        };

        // head 結合: [B, H, L, Dh] -> permute -> [B, L, H, Dh] ->
        // contiguous（permute 後は必ず非 contiguous になるため無条件）
        // -> reshape -> [B, L, E]。
        let merged = attn_out
            .permute(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape(&[b, l, e])?;

        let out = project(&merged, &self.out, b, l, e, e, None)?;
        if self.batch_first {
            Ok(out)
        } else {
            // `[B, L, E]` -> `[L, B, E]`。後続層の `reshape`（例えば
            // `project` の再呼び出し）が `NonContiguousReshape` で落ちない
            // よう即座に `contiguous()` する。
            out.transpose(0, 1)?.contiguous()
        }
    }
}

/// [`MultiheadAttentionVars::forward`] の opt-in 低精度版（イシュー
/// #2071・親 #1626／#1648）。q/k/v/out の 4 projection と SDPA 本体の
/// 2 回の matmul（`sdpa_compose` 内）を `dtype`（[`ScalarDType::F16`]／
/// [`ScalarDType::Bf16`]）で計算する（scale・transpose・mask・softmax
/// は PyTorch autocast の fp32 リストと同様 f32 のまま）。shape 検査・
/// 処理順序は [`MultiheadAttentionVars::forward`] と完全に同一
/// （`project`／`sdpa_compose` へ `Some(dtype)` を渡すのみの差分）。
///
/// **`MultiheadAttentionVars` へメソッドとして追加しない理由**:
/// `nn::linear::linear_forward_low_precision`・`nn::conv::
/// conv2d_forward_low_precision` と同じ理由（`pub q`／`k`／`v`／`out`
/// フィールドを持つ struct への破壊的変更を避けるため自由関数として
/// 配置する）。
///
/// **イシュー #2163 の fail-closed 化**: `vars` が非既定 config
/// （`batch_first == false` または `kdim`/`vdim != embed_dim`）を持つ
/// 場合、無言で batch-first 解釈するのではなく `InvalidArgument` で
/// 拒否する（低精度経路のオプション対応自体は本イシューの対象外。
/// `out-of-scope-tracking.md`）。
pub fn multihead_attention_forward_low_precision<'t>(
    vars: &MultiheadAttentionVars<'t>,
    query: &Var<'t>,
    key: &Var<'t>,
    value: &Var<'t>,
    attn_mask: Option<&Tensor<bool>>,
    is_causal: bool,
    dtype: ScalarDType,
) -> Result<Var<'t>, AutodiffError> {
    if !vars.batch_first || vars.kdim != vars.embed_dim || vars.vdim != vars.embed_dim {
        return Err(AutodiffError::InvalidArgument(
            "multihead_attention_forward_low_precision: batch_first=false・kdim/vdim != \
             embed_dim の MultiheadAttentionVars は対象外（イシュー #2163 スコープ外）"
                .to_string(),
        ));
    }
    query.check_same_tape(&vars.q.weight)?;
    query.check_same_tape(key)?;
    query.check_same_tape(value)?;

    let q_shape = query.shape();
    let k_shape = key.shape();
    let v_shape = value.shape();
    if q_shape.len() != 3 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 3,
            actual: q_shape.len(),
        }));
    }
    if k_shape.len() != 3 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 3,
            actual: k_shape.len(),
        }));
    }
    if v_shape.len() != 3 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 3,
            actual: v_shape.len(),
        }));
    }

    let e = vars.embed_dim;
    let (b, l) = (q_shape[0], q_shape[1]);
    if q_shape[2] != e {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: q_shape.clone(),
            rhs: vec![b, l, e],
        }));
    }
    if k_shape[0] != b || k_shape[2] != e {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: k_shape.clone(),
            rhs: vec![b, k_shape[1], e],
        }));
    }
    if v_shape != k_shape {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: v_shape.clone(),
            rhs: k_shape.clone(),
        }));
    }
    let s = k_shape[1];

    let h = vars.num_heads;
    let dh = e / h;

    let q_proj = project(query, &vars.q, b, l, e, e, Some(dtype))?;
    let k_proj = project(key, &vars.k, b, s, e, e, Some(dtype))?;
    let v_proj = project(value, &vars.v, b, s, e, e, Some(dtype))?;

    let q_heads = split_heads(&q_proj, b, l, h, dh)?;
    let k_heads = split_heads(&k_proj, b, s, h, dh)?;
    let v_heads = split_heads(&v_proj, b, s, h, dh)?;

    let scale = 1.0f32 / (dh as f32).sqrt();
    let attn_out = sdpa_compose(
        &q_heads,
        &k_heads,
        &v_heads,
        attn_mask,
        is_causal,
        scale,
        Some(dtype),
    )?;

    let merged = attn_out
        .permute(&[0, 2, 1, 3])?
        .contiguous()?
        .reshape(&[b, l, e])?;

    project(&merged, &vars.out, b, l, e, e, Some(dtype))
}

/// q/k/v/out projection の共通実装:
/// `y = x.reshape([B*Len, E]).matmul(weight) (+ bias)`（`LinearVars::
/// forward` へ委譲）を経て `[B, Len, E]` へ戻す。
///
/// `x` は `reshape([B*Len, E])` を直接試み、`ShapeError::
/// NonContiguousReshape`（`x` が非 contiguous な view——例えば
/// `MultiheadAttentionVars::forward` が `transpose` 済みの `query` を
/// そのまま渡した場合）のときのみ `contiguous()` を挟んで再試行する
/// （実装計画 D5。`Var::value()`／`is_contiguous()` を直接見て分岐する
/// より、reshape 自体の結果で判定する方が `RefCell` 借用の取り回しが
/// 単純かつ確実に安全）。
///
/// `pub(crate)` である理由（イシュー #2068）: `LinearVars::forward` は
/// rank 2 入力を要求する（`gemm_out_shape` の厳密検査）ため、
/// rank 3（`[B, Len, E]`）を扱う `nn` 層は同じ「flatten → forward →
/// unflatten」パターンを必要とする。`nn::transformer_encoder_layer`
/// の FFN 部分（`linear1`／`linear2`）が本関数を再利用する
/// （重複実装しない。`code-comment-style.md`「何を書かないか」）。
///
/// `in_features`／`out_features` を分離した理由（イシュー #2068）:
/// 本モジュール（`MultiheadAttention`）の呼び出し元は q/k/v/out の
/// 4 層すべてが正方 `[E, E]`（`validate_embed_heads` が構築時に検証
/// 済み）のため `in_features == out_features == e` で不変だが、
/// `nn::transformer_encoder_layer` の FFN 第 1 層（`d_model ->
/// dim_feedforward`）／第 2 層（`dim_feedforward -> d_model`）は非正方
/// のため、入力側の reshape（`in_features`）と出力側の reshape
/// （`out_features`）を独立に指定できる必要がある（単一の `e` を両方に
/// 使うと非正方 `proj` で `y.reshape(&[b, len, e])` が要素数不一致に
/// なる）。
/// `low_precision`（イシュー #2071）: `Some(dtype)` のとき `proj.forward`
/// （f32）の代わりに `nn::linear::linear_forward_low_precision`
/// （`TypedOps<f16/bf16>` 経由）へ委譲する。`None`（既定の呼び出し元は
/// すべて `None`）のときは従来どおり `proj.forward` を呼ぶため、
/// 既存呼び出し元は bit 同一のまま不変。
pub(crate) fn project<'t>(
    x: &Var<'t>,
    proj: &LinearVars<'t>,
    b: usize,
    len: usize,
    in_features: usize,
    out_features: usize,
    low_precision: Option<ScalarDType>,
) -> Result<Var<'t>, AutodiffError> {
    let bl = b.checked_mul(len).ok_or_else(|| {
        AutodiffError::InvalidArgument(format!(
            "MultiheadAttention: batch({b}) * seq_len({len}) overflowed usize"
        ))
    })?;
    let x_flat = match x.reshape(&[bl, in_features]) {
        Ok(flat) => flat,
        Err(AutodiffError::Shape(ShapeError::NonContiguousReshape)) => {
            x.contiguous()?.reshape(&[bl, in_features])?
        }
        Err(other) => return Err(other),
    };
    let y = match low_precision {
        Some(dtype) => linear_forward_low_precision(proj, &x_flat, Activation::None, dtype)?,
        None => proj.forward(&x_flat)?,
    };
    y.reshape(&[b, len, out_features])
}

/// head 分割: `[B, Len, E] -> reshape -> [B, Len, H, Dh] -> permute ->
/// [B, H, Len, Dh]`。`x` は [`project`] の出力（matmul／add の eager
/// 実体化結果であり常に contiguous）のため、reshape 前の contiguous
/// 化は不要（`project` と異なり `NonContiguousReshape` 経路を持たない）。
/// `permute` は zero-copy view のため、後続の [`sdpa_compose`]
/// （`matmul`／`transpose`）が非 contiguous な batch 次元を受け取る
/// ことになるが、`BackendOps::gemm_batched` の既定合成（
/// `normalize_batched_operand`）・各バックエンドのオーバーライドは
/// いずれも内部で `contiguous()` を経由してから 2 次元 GEMM へ渡すため
/// 正しく動作する（`tensor-core/src/backend_ops.rs::
/// normalize_batched_operand` doc 参照）。
fn split_heads<'t>(
    x: &Var<'t>,
    b: usize,
    len: usize,
    h: usize,
    dh: usize,
) -> Result<Var<'t>, AutodiffError> {
    x.reshape(&[b, len, h, dh])?.permute(&[0, 2, 1, 3])
}

/// `l * s`（causal mask の要素数）のオーバーフロー検査つき構築。
/// [`causal_blocked_mask`] からのみ呼ばれる。
fn checked_ls(l: usize, s: usize) -> Result<usize, AutodiffError> {
    l.checked_mul(s).ok_or_else(|| {
        AutodiffError::InvalidArgument(format!(
            "MultiheadAttention: l({l}) * s({s}) overflowed usize while building the causal mask"
        ))
    })
}

/// causal（top-left aligned。`j <= i` のみ attend を許可）ブロックマスク
/// `[l, s]` を構築する（PyTorch `_scaled_dot_product_attention_math`
/// 参照実装の `torch.ones(L, S, dtype=bool).tril(diagonal=0)` の否定と
/// 同一規約。#1639（PR #1845）の同名関数〈`crate::attention::
/// causal_blocked_mask`〉と数式・規約が完全に一致する——モジュール doc
/// 「sub-issue (a) との関係」参照）。`blocked[i][j] = j > i` であり、
/// 任意の `i >= 0` に対し `j = 0` は常に許可される（`0 > i` は `i >= 0`
/// では常に偽）ため、`l > 0 && s > 0` の下では全 masked 行は構造的に
/// 生じない（[`reject_fully_masked_rows`] は防御的に維持する）。
fn causal_blocked_mask(l: usize, s: usize) -> Result<Tensor<bool>, AutodiffError> {
    let capacity = checked_ls(l, s)?;
    let mut data = Vec::with_capacity(capacity);
    for i in 0..l {
        for j in 0..s {
            data.push(j > i);
        }
    }
    Tensor::new(data, &[l, s]).map_err(AutodiffError::Shape)
}

/// `blocked`（`[..., l, s]` へ broadcast 済み・`.contiguous()` 済みの
/// bool テンソル）の最終軸（`s` 列）を 1 行ずつ走査し、全要素が `true`
/// （全 key が masked）の行がないか検査する。全 masked 行は
/// `exp(-inf - max) / sum(...)` がバックエンド依存の不定値（`0/0`）を
/// 生みうるため、演算グラフへ `masked_fill` を記録する前に拒否する
/// （`ShapeError` ではなく `InvalidArgument`——値の組合せの問題であり
/// shape 自体は妥当なため。#1639 と同一規約）。
fn reject_fully_masked_rows(blocked: &Tensor<bool>) -> Result<(), AutodiffError> {
    let shape = blocked.shape();
    let rank = shape.len();
    if rank == 0 {
        return Ok(());
    }
    let s = shape[rank - 1];
    if s == 0 {
        // 列が 0 本の「行」に「全要素が masked」という主張は空虚
        // （vacuous truth で誤って fail させない）。
        return Ok(());
    }
    let contiguous = blocked.contiguous();
    let data = contiguous.as_slice().unwrap_or_default();
    if data.chunks_exact(s).any(|row| row.iter().all(|&b| b)) {
        return Err(AutodiffError::InvalidArgument(
            "MultiheadAttention: a row is fully masked by attn_mask/is_causal (no attendable \
             key), which would make softmax produce backend-dependent indeterminate values"
                .to_string(),
        ));
    }
    Ok(())
}

/// `attn_mask`（`true` = attend。PyTorch bool mask 規約）を `scores`
/// の shape（`[..., l, s]`）へ broadcast したうえで否定し、
/// [`Var::masked_fill`] へ渡す「block」テンソル（`true` = fill 対象）を
/// 作る（#1639 と同一規約）。`Tensor<bool>` に要素ごとの否定演算が
/// ないため、`as_slice` で読み出してから組み立てる（`Var::
/// where_cond`／`masked_fill` の broadcast → `as_slice` パターンと
/// 同型）。
fn negate_broadcast_mask(
    mask: &Tensor<bool>,
    scores_shape: &[usize],
) -> Result<Tensor<bool>, AutodiffError> {
    let allowed = mask
        .broadcast_to(scores_shape)
        .map_err(AutodiffError::Shape)?
        .contiguous();
    let allowed_data: Vec<bool> = allowed.as_slice().map(|s| s.to_vec()).unwrap_or_default();
    let blocked_data: Vec<bool> = allowed_data.iter().map(|&a| !a).collect();
    Tensor::new(blocked_data, scores_shape).map_err(AutodiffError::Shape)
}

/// `attn_mask`／`key_padding_mask`／`is_causal` を 1 本の「attend」極性
/// （`true` = attend。[`sdpa_compose`] の `attn_mask` 引数と同じ規約——
/// 呼び出し元がこの戻り値をそのまま `sdpa_compose(.., Some(&combined),
/// is_causal=false, ..)` へ渡す）テンソル `[B, H, L, S]` へ合成する
/// （イシュー #2163。[`MultiheadAttentionVars::
/// forward_with_key_padding_mask`] からのみ呼ばれる）。`key_padding_mask:
/// [B, S]` は行方向に broadcast（`kpm[b, j]` が全 `h`／`i` で共有）
/// するが、`Tensor::broadcast_to` は右詰め（末尾軸から）整列するため
/// `[B, S]` を直接 `[B, H, L, S]` へ broadcast すると `B` が `L` に
/// 整列してしまう（`kpm` は本来 `B`／`S` 軸に対応するため誤整列——
/// advisor 指摘）。このため `attn_mask` のみ `broadcast_to`
/// （`sdpa_compose` と同じ形状検証契約）で読み出し、`key_padding_mask`・
/// `is_causal` は明示的な添字ループで合成する（`B`／`S` 軸を取り違え
/// ない）。
///
/// 返す `allowed[b, h, i, j] = kpm[b, j] AND (attn_mask 指定時: attn_mask
/// を `[B, H, L, S]` へ broadcast した値) AND (is_causal 時: `j <= i`)`
/// を `[B, H, L, S]`（attend 極性）でそのまま返す。全体を `[B, H, L, S]`
/// で構築することで、`key_padding_mask` のみを指定したケースが
/// 「`[B, 1, 1, S]` の `attn_mask` を渡した場合」と同じ `[B, H, L, S]`
/// 形状へ帰着し、`sdpa_compose` 内で [`negate_broadcast_mask`] が生成
/// する block テンソルと bit 一致する（単体テスト
/// `key_padding_mask_matches_equivalent_attn_mask` で固定）。
///
/// `attn_mask` の broadcast 検証は `l`／`s` が 0 でも常に行う
/// （`sdpa_compose` の既存契約——モジュール doc「L == 0 または S == 0」
/// 節参照——と同じ理由）。`key_padding_mask` の `[B, S]` 厳密形状検証は
/// 常に行うが、`b`／`h`／`l`／`s` の要素数計算は `checked_mul` で
/// オーバーフローを検査する。
#[allow(clippy::too_many_arguments)]
fn combine_masks_with_key_padding(
    attn_mask: Option<&Tensor<bool>>,
    key_padding_mask: &Tensor<bool>,
    is_causal: bool,
    b: usize,
    h: usize,
    l: usize,
    s: usize,
) -> Result<Tensor<bool>, AutodiffError> {
    if key_padding_mask.shape() != [b, s] {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: key_padding_mask.shape().to_vec(),
            rhs: vec![b, s],
        }));
    }
    let kpm_data: Vec<bool> = key_padding_mask
        .contiguous()
        .as_slice()
        .map(|d| d.to_vec())
        .unwrap_or_default();

    let target_shape = [b, h, l, s];
    let attn_data: Option<Vec<bool>> = match attn_mask {
        Some(mask) => {
            let broadcast = mask
                .broadcast_to(&target_shape)
                .map_err(AutodiffError::Shape)?
                .contiguous();
            Some(broadcast.as_slice().map(|d| d.to_vec()).unwrap_or_default())
        }
        None => None,
    };

    let bh = b.checked_mul(h).ok_or_else(overflow_err)?;
    let bhl = bh.checked_mul(l).ok_or_else(overflow_err)?;
    let total = bhl.checked_mul(s).ok_or_else(overflow_err)?;
    let mut allowed = Vec::with_capacity(total);
    for b_idx in 0..b {
        for _h_idx in 0..h {
            for i in 0..l {
                for j in 0..s {
                    let kpm_ok = kpm_data.get(b_idx * s + j).copied().unwrap_or(false);
                    let idx = allowed.len();
                    let attn_ok = attn_data
                        .as_ref()
                        .map(|d| d.get(idx).copied().unwrap_or(false))
                        .unwrap_or(true);
                    let causal_ok = !is_causal || j <= i;
                    allowed.push(kpm_ok && attn_ok && causal_ok);
                }
            }
        }
    }
    Tensor::new(allowed, &target_shape).map_err(AutodiffError::Shape)
}

/// [`combine_masks_with_key_padding`] の要素数オーバーフロー検査で使う
/// 共通エラー構築（`checked_ls`／`project` の overflow ガードと同型）。
fn overflow_err() -> AutodiffError {
    AutodiffError::InvalidArgument(
        "MultiheadAttentionVars::forward_with_key_padding_mask: mask 要素数（B*H*L*S）が usize \
         をオーバーフローした"
            .to_string(),
    )
}

/// scaled dot product attention 本体（D1 フォールバック。モジュール
/// doc「sub-issue (a) との関係」参照）。`query`/`key`/`value` は
/// [`split_heads`] 済みの `[B, H, Len, Dh]`（一般には任意 rank≥2 の
/// バッチ形状）を受け取り、`[B, H, L, Dh]` を返す。`scale` は呼び出し元
/// （`MultiheadAttentionVars::forward`）が `1/sqrt(head_dim)` として
/// 確定済みの値を渡す（`head_dim >= 1` を構築時に保証済みのため常に
/// 有限・正）。
/// `low_precision`（イシュー #2071）: `Some(dtype)` のとき 2 回の
/// `matmul`（`scores = q_scaled @ k_t`・`out = weights @ value`）を
/// `Var::matmul_low_precision` へ切り替える。scale・transpose・mask・
/// softmax は PyTorch autocast の fp32 リスト（softmax）と同様 f32 の
/// まま（`None` の既存呼び出し元は bit 同一のまま不変）。
fn sdpa_compose<'t>(
    query: &Var<'t>,
    key: &Var<'t>,
    value: &Var<'t>,
    attn_mask: Option<&Tensor<bool>>,
    is_causal: bool,
    scale: f32,
    low_precision: Option<ScalarDType>,
) -> Result<Var<'t>, AutodiffError> {
    query.check_same_tape(key)?;
    query.check_same_tape(value)?;

    if attn_mask.is_some() && is_causal {
        return Err(AutodiffError::InvalidArgument(
            "MultiheadAttention: attn_mask and is_causal are mutually exclusive".to_string(),
        ));
    }

    // scale を q 側（scores より要素数が少ない）へ先に掛ける（丸め順序
    // の選択であり REQ-2 統一複合判定の範囲内。`amp::scale_loss` と同じ
    // 「スカラー Leaf を 1 つ登録して掛ける」パターン）。
    let scale_var = query.tape().var(&Tensor::scalar(scale));
    let q_scaled = query.mul(&scale_var)?;

    let k_rank = key.shape().len();
    if k_rank < 2 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: k_rank,
        }));
    }
    // zero-copy view（`Var::transpose`）。`matmul` の rhs としてそのまま
    // 渡せる（`normalize_batched_operand` が必要なら contiguous 化する。
    // `split_heads` doc 参照）。
    let k_t = key.transpose(k_rank - 2, k_rank - 1)?;

    // `[..., L, E] @ [..., E, S] -> [..., L, S]`。E 不一致・バッチ
    // broadcast 不能は `Var::matmul`／`matmul_low_precision`
    // （`matmul_out_shape`）が検査済み。
    let scores = match low_precision {
        Some(dtype) => q_scaled.matmul_low_precision(&k_t, dtype)?,
        None => q_scaled.matmul(&k_t)?,
    };
    let scores_shape = scores.shape();
    let rank = scores_shape.len();
    let l = scores_shape[rank - 2];
    let s = scores_shape[rank - 1];

    // L == 0 または S == 0 は空テンソル契約。「全 masked 行」の値検査
    // （`reject_fully_masked_rows`）は行自体が存在しないため意味を
    // なさず省略するが、`attn_mask` の broadcast 形状検証（不能形状の
    // `ShapeError` 化）は 0 サイズでも常に実施する（codex-review・
    // Cursor Bugbot 指摘の是正: L==0/S==0 で検証を丸ごとスキップすると
    // broadcast 不能な `attn_mask`（例: query=[2,0,4]・key/value=
    // [2,4,4] に mask=[3,5]）を渡しても受理されてしまい、`forward` が
    // 文書化する broadcast 不能時のエラー契約・`Var::
    // scaled_dot_product_attention`〈`crate::attention::
    // scaled_dot_product_attention`。同一指摘を PR #1845 codex-review
    // で是正済み〉との挙動整合が崩れる）。
    let scores = if is_causal {
        if l > 0 && s > 0 {
            let blocked = causal_blocked_mask(l, s)?;
            reject_fully_masked_rows(&blocked)?;
            // `masked_fill` 自身が `[l, s]` を `scores_shape` へ
            // broadcast する（`Var::masked_fill` doc 参照）。
            scores.masked_fill(&blocked, f32::NEG_INFINITY)?
        } else {
            // causal の block パターンは `l`／`s` のみから決定的に
            // 導出され外部形状を持たないため、0 サイズでは検証すべき
            // 追加形状が存在しない。
            scores
        }
    } else if let Some(mask) = attn_mask {
        // `negate_broadcast_mask` が `broadcast_to` による形状検証を
        // 行う（0 サイズでも常に呼ぶ）。値検査（全 masked 行）と実際の
        // `masked_fill` 適用は非空の場合のみ行う。
        let blocked = negate_broadcast_mask(mask, &scores_shape)?;
        if l > 0 && s > 0 {
            reject_fully_masked_rows(&blocked)?;
            scores.masked_fill(&blocked, f32::NEG_INFINITY)?
        } else {
            scores
        }
    } else {
        scores
    };

    // 最終軸（S）方向の softmax（CPU／CUDA／Metal いずれも行カーネルへ
    // 到達する契約。イシュー #1594）。
    let weights = scores.softmax(rank - 1)?;

    // `[..., L, S] @ [..., S, Dh] -> [..., L, Dh]`。S 不一致
    // （`key[-2] != value[-2]`）はここで `Var::matmul`／
    // `matmul_low_precision` が検査する（`split_heads` が q/k/v とも
    // 同じ `s` から head 分割するため本モジュール内では実際には発生
    // しない）。
    match low_precision {
        Some(dtype) => weights.matmul_low_precision(value, dtype),
        None => weights.matmul(value),
    }
}

/// KV キャッシュ（イシュー #2084・親 #2059。設計正本
/// `docs/kv-cache-design.md` §2 案 B）: 射影済みの K/V（`[B, S_cached,
/// E]`・contiguous）をホスト `Tensor<f32>` として保持する値型。
/// [`MultiheadAttentionVars::forward_with_cache`] が唯一の書き手であり、
/// `Tape` の外に置くため（`TapeNode::value` はホスト `OnceCell<Tensor<f32>>`。
/// `docs/kv-cache-design.md` §0）decode ループごとの `Tape` 再作成／
/// `Tape::reset` の影響を受けない。
///
/// **不変条件**: `k`／`v` は常に両方 `Some` か両方 `None`
/// （[`KvCache::new`]・[`KvCache::clear`] のみが状態を変える公開経路で
/// あり、外部から任意の `Tensor` を注入するセッターは設けない設計
/// ——`docs/kv-cache-design.md` §2「値型 `KvCache`」参照）。
///
/// **`clone()` は実データをコピーしない**: `Tensor<f32>` は内部
/// `storage: Arc<Storage<T>>` を `Arc` 共有する値型
/// （[`Var::to_tensor`] doc 参照）のため、`clone()` は `Arc` の
/// ポインタ複製のみで済む。
#[derive(Debug, Clone, Default)]
pub struct KvCache {
    k: Option<Tensor<f32>>,
    v: Option<Tensor<f32>>,
}

impl KvCache {
    /// 空キャッシュを作る（[`Default::default`] と同じ）。
    pub fn new() -> KvCache {
        KvCache::default()
    }

    /// キャッシュが空（`k`／`v` とも `None`）かどうか。
    pub fn is_empty(&self) -> bool {
        self.k.is_none()
    }

    /// キャッシュ済みの系列長 `S_cached`（空なら 0）。
    pub fn seq_len(&self) -> usize {
        self.k.as_ref().map(|k| k.shape()[1]).unwrap_or(0)
    }

    /// キャッシュ済みのバッチサイズ `B`（空なら `None`）。
    pub fn batch(&self) -> Option<usize> {
        self.k.as_ref().map(|k| k.shape()[0])
    }

    /// キャッシュ済みの埋め込み次元 `E`（空なら `None`）。
    pub fn embed_dim(&self) -> Option<usize> {
        self.k.as_ref().map(|k| k.shape()[2])
    }

    /// キャッシュを空に戻す（再 prefill 可能な状態へ再初期化）。
    pub fn clear(&mut self) {
        self.k = None;
        self.v = None;
    }

    /// キャッシュ済み K（`[B, S_cached, E]`）への参照。空なら `None`。
    pub fn k(&self) -> Option<&Tensor<f32>> {
        self.k.as_ref()
    }

    /// キャッシュ済み V（`[B, S_cached, E]`）への参照。空なら `None`。
    pub fn v(&self) -> Option<&Tensor<f32>> {
        self.v.as_ref()
    }
}

/// (c)（`L_new > 1` かつ cache 非空）の offset causal mask
/// `[l_new, s_total]` を構築する（`docs/kv-cache-design.md` §2
/// mask 規則 (c)）: `allowed[i][j] = j <= s_prev + i`（`true` = attend。
/// PyTorch bool mask 規約——[`sdpa_compose`] が受理する `attn_mask` と
/// 同じ極性）。[`causal_blocked_mask`] が `blocked[i][j] = j > i` の
/// 「block」極性（`s_prev == 0` のとき本関数の否定と一致——単体テスト
/// `offset_allowed_mask_matches_causal_blocked_mask_when_s_prev_is_zero`
/// で確認）を返すのに対し、本関数は `sdpa_compose` の `attn_mask`
/// 引数（`true` = attend）へ直接渡せる極性で返す。
fn offset_allowed_mask(
    l_new: usize,
    s_prev: usize,
    s_total: usize,
) -> Result<Tensor<bool>, AutodiffError> {
    let capacity = checked_ls(l_new, s_total)?;
    let mut data = Vec::with_capacity(capacity);
    for i in 0..l_new {
        // `s_prev + i` は `s_total = s_prev + l_new_kv`（呼び出し元が
        // `checked_add` で検査済み）以下であり `usize` オーバーフロー
        // しない。
        let boundary = s_prev + i;
        for j in 0..s_total {
            data.push(j <= boundary);
        }
    }
    Tensor::new(data, &[l_new, s_total]).map_err(AutodiffError::Shape)
}

impl<'t> MultiheadAttentionVars<'t> {
    /// KV キャッシュ付き forward（K-1 最小版。イシュー #2084・
    /// `docs/kv-cache-design.md` §2）。新規トークン分の q/k/v のみを
    /// 受け取り、射影 → `cache` との連結 → attention → `cache`
    /// 更新の順で処理する。`is_causal`／`attn_mask` 引数は持たない
    /// （decode で `is_causal=true` を誤って渡す落とし穴——top-left
    /// aligned `causal_blocked_mask` は `L=1` で先頭 key にしか
    /// attend しない——を構造的に塞ぐ fail-closed 設計。モジュール doc
    /// 「KV キャッシュ（#2084）」参照）。
    ///
    /// 入力: `query_new: [B, L_new, E]`・`key_new`/`value_new:
    /// [B, L_new_kv, E]`（self-attention の decode では 3 つとも同じ
    /// `Var` を渡す）。cross-attention 用途は対象外のため
    /// `L_new != L_new_kv` は `InvalidArgument` で拒否する（`cache` の
    /// offset 規則が `query_new` と `key_new` の系列長一致を前提と
    /// するため。`docs/kv-cache-design.md` §2 の (a)/(c) 規則参照）。
    ///
    /// mask 規則（`cache` の状態から内部で決定。`docs/kv-cache-design.md`
    /// §2）:
    /// - (a) `cache` が空（prefill）→ `is_causal=true` 相当
    /// - (b) `cache` が非空・`L_new == 1`（通常の decode）→ mask なし
    /// - (c) `cache` が非空・`L_new > 1`（複数トークン追記）→
    ///   `offset_allowed_mask`（本モジュール内 private 関数）を
    ///   `attn_mask` として渡す
    ///
    /// **勾配の扱い**: `cache` は [`Tape::var_no_grad`] で葉として
    /// 登録するため、過去ステップへは勾配が流れない（推論用の
    /// truncated 意味論。`docs/kv-cache-design.md` §2）。現ステップの
    /// 射影パラメータへの勾配は通常の `forward` と同様に流れる。
    ///
    /// **原子的な更新**: `cache` への書き戻しは全段が成功した後にのみ
    /// 行う。途中のいずれかの段でエラーになった場合、`cache` は
    /// 呼び出し前の状態のまま変化しない（エラー経路の単体テスト
    /// 群で確認）。
    ///
    /// **`Tape` の運用（呼び出し側の責務）**: `Var::cat`
    /// （`docs/kv-cache-design.md` §3.2）はステップごとに `Op::Concat`
    /// ノードを積むため、decode ループでは新しい `Tape` を作るか
    /// `Tape::reset` を使う運用を推奨する。`cache` 自体はホスト
    /// `Tensor<f32>` として `Tape` の外にあるため、この運用の影響を
    /// 受けない。
    ///
    /// # Errors
    ///
    /// `query_new`／`key_new`／`value_new` の rank が 3 でない、
    /// バッチ次元 `B` が不一致、最終軸が `embed_dim` と不一致、
    /// `key_new`/`value_new` の shape が食い違う、`cache` が非空で
    /// `batch`／`embed_dim` が不一致、`L_new != L_new_kv`、`S_prev +
    /// L_new_kv` が `usize` をオーバーフローする、のいずれかで
    /// `AutodiffError::Shape`／`InvalidArgument` を返す。テープ不一致は
    /// `check_same_tape`（`AutodiffError::TapeMismatch`）。
    pub fn forward_with_cache(
        &self,
        query_new: &Var<'t>,
        key_new: &Var<'t>,
        value_new: &Var<'t>,
        cache: &mut KvCache,
    ) -> Result<Var<'t>, AutodiffError> {
        // イシュー #2163 の fail-closed 化: 非既定 config（self-attention
        // 限定の K-1 設計が前提とする `kdim = vdim = embed_dim`・
        // `batch_first = true` から外れる）は `cache` を一切変更せず
        // 拒否する（先頭で検査するため原子性契約——モジュール doc
        // 「原子的な更新」——を破らない）。
        if !self.batch_first || self.kdim != self.embed_dim || self.vdim != self.embed_dim {
            return Err(AutodiffError::InvalidArgument(
                "MultiheadAttentionVars::forward_with_cache: batch_first=false・kdim/vdim != \
                 embed_dim は対象外（self-attention 限定の K-1 設計。イシュー #2163 スコープ外）"
                    .to_string(),
            ));
        }
        query_new.check_same_tape(&self.q.weight)?;
        query_new.check_same_tape(key_new)?;
        query_new.check_same_tape(value_new)?;

        let q_shape = query_new.shape();
        let k_shape = key_new.shape();
        let v_shape = value_new.shape();
        if q_shape.len() != 3 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 3,
                actual: q_shape.len(),
            }));
        }
        if k_shape.len() != 3 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 3,
                actual: k_shape.len(),
            }));
        }
        if v_shape.len() != 3 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 3,
                actual: v_shape.len(),
            }));
        }

        let e = self.embed_dim;
        let (b, l_new) = (q_shape[0], q_shape[1]);
        if q_shape[2] != e {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: q_shape.clone(),
                rhs: vec![b, l_new, e],
            }));
        }
        if k_shape[0] != b || k_shape[2] != e {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: k_shape.clone(),
                rhs: vec![b, k_shape[1], e],
            }));
        }
        if v_shape != k_shape {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: v_shape.clone(),
                rhs: k_shape.clone(),
            }));
        }
        let l_new_kv = k_shape[1];
        // self-attention の追記のみを対象とする（モジュール doc・
        // `docs/kv-cache-design.md` §2 参照。cross-attention 用の
        // 非対称なキャッシュ追記は最小版の対象外）。
        if l_new != l_new_kv {
            return Err(AutodiffError::InvalidArgument(format!(
                "MultiheadAttentionVars::forward_with_cache: query_new の系列長 \
                 ({l_new}) と key_new/value_new の系列長 ({l_new_kv}) は一致する必要がある \
                 （self-attention 限定。cross-attention は対象外）"
            )));
        }

        if !cache.is_empty() {
            if cache.batch() != Some(b) {
                return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                    lhs: vec![cache.batch().unwrap_or(0)],
                    rhs: vec![b],
                }));
            }
            if cache.embed_dim() != Some(e) {
                return Err(AutodiffError::InvalidArgument(format!(
                    "MultiheadAttentionVars::forward_with_cache: cache の embed_dim \
                     ({:?}) と入力の embed_dim ({e}) が不一致",
                    cache.embed_dim()
                )));
            }
        }
        let s_prev = cache.seq_len();
        let s_total = s_prev.checked_add(l_new_kv).ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "MultiheadAttentionVars::forward_with_cache: s_prev({s_prev}) + \
                 l_new_kv({l_new_kv}) が usize をオーバーフローした"
            ))
        })?;

        let q_proj = project(query_new, &self.q, b, l_new, e, e, None)?;
        let k_new_proj = project(key_new, &self.k, b, l_new_kv, e, e, None)?;
        let v_new_proj = project(value_new, &self.v, b, l_new_kv, e, e, None)?;

        // cache が非空なら `var_no_grad` で新規ステップの葉として登録し
        // 連結する（`docs/kv-cache-design.md` §2 手順③。過去ステップへ
        // 勾配は流れない — `Tape::var_no_grad` の契約どおり）。
        let (k_cat, v_cat) = match (cache.k(), cache.v()) {
            (Some(k_prev), Some(v_prev)) => {
                let k_prev_var = query_new.tape().var_no_grad(k_prev);
                let v_prev_var = query_new.tape().var_no_grad(v_prev);
                (
                    Var::cat(&[k_prev_var, k_new_proj], 1)?,
                    Var::cat(&[v_prev_var, v_new_proj], 1)?,
                )
            }
            _ => (k_new_proj, v_new_proj),
        };

        // `validate_projection_vars`／`validate_embed_heads` が
        // `e % h == 0` かつ `e > 0`／`h > 0` を構築時に検証済みのため、
        // `dh` は常に 1 以上の整数（[`MultiheadAttentionVars::forward`]
        // と同じ導出）。
        let h = self.num_heads;
        let dh = e / h;
        let q_heads = split_heads(&q_proj, b, l_new, h, dh)?;
        let k_heads = split_heads(&k_cat, b, s_total, h, dh)?;
        let v_heads = split_heads(&v_cat, b, s_total, h, dh)?;

        let scale = 1.0f32 / (dh as f32).sqrt();
        let attn_out = if s_prev == 0 {
            // (a) prefill: 全系列が新規のため causal。
            sdpa_compose(&q_heads, &k_heads, &v_heads, None, true, scale, None)?
        } else if l_new == 1 {
            // (b) 通常の decode: 新規 1 トークンは cache 済み全 key と
            // 自身に attend してよいため mask 不要。
            sdpa_compose(&q_heads, &k_heads, &v_heads, None, false, scale, None)?
        } else {
            // (c) 複数トークンの追記: offset causal mask を明示指定。
            let mask = offset_allowed_mask(l_new, s_prev, s_total)?;
            sdpa_compose(
                &q_heads,
                &k_heads,
                &v_heads,
                Some(&mask),
                false,
                scale,
                None,
            )?
        };

        let merged = attn_out
            .permute(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape(&[b, l_new, e])?;
        let out = project(&merged, &self.out, b, l_new, e, e, None)?;

        // 全段が成功した後にのみ cache を書き戻す（原子的な更新。
        // `to_tensor()` は `Arc` 共有のため算術を伴わない bit 完全一致
        // コピー——`docs/kv-cache-design.md` §2 手順⑧）。
        cache.k = Some(k_cat.to_tensor());
        cache.v = Some(v_cat.to_tensor());

        Ok(out)
    }
}

/// KV キャッシュを所有する薄い stateful ラッパー（イシュー #2084。
/// `docs/kv-cache-design.md` §2）。[`MultiheadAttention`]（パラメータ
/// 本体）と [`KvCache`] を保持し、`forward` は
/// [`MultiheadAttentionVars::forward_with_cache`] への 1 行委譲に
/// 徹する。
///
/// **`Module` trait は実装しない**: `Module::forward` は `&self` を
/// 取るため `cache` を更新できない。`RefCell` で内部可変にすると
/// 「状態を持たない forward」という `Module` の前提を壊すため、
/// あえて `Module` を実装しない（`compat::Sequential` への結線は
/// 行わない。`docs/kv-cache-design.md` §2「facade 到達経路」参照）。
pub struct StatefulAttention {
    mha: MultiheadAttention,
    cache: KvCache,
}

impl StatefulAttention {
    /// 空キャッシュで構築する。
    pub fn new(mha: MultiheadAttention) -> StatefulAttention {
        StatefulAttention {
            mha,
            cache: KvCache::new(),
        }
    }

    /// self-attention（`q = k = v = x_new`）として
    /// [`MultiheadAttentionVars::forward_with_cache`] を呼ぶ（1 行
    /// 委譲）。
    pub fn forward<'t>(
        &mut self,
        tape: &'t Tape,
        x_new: &Var<'t>,
    ) -> Result<Var<'t>, AutodiffError> {
        self.mha
            .bind(tape)
            .forward_with_cache(x_new, x_new, x_new, &mut self.cache)
    }

    /// キャッシュを空に戻す（[`KvCache::clear`] への委譲）。
    pub fn reset_cache(&mut self) {
        self.cache.clear();
    }

    /// 保持しているキャッシュへの参照。
    pub fn cache(&self) -> &KvCache {
        &self.cache
    }

    /// キャッシュ済みの系列長（[`KvCache::seq_len`] への委譲）。
    pub fn seq_len(&self) -> usize {
        self.cache.seq_len()
    }

    /// パラメータ本体への参照。
    pub fn mha(&self) -> &MultiheadAttention {
        &self.mha
    }
}

/// self-attention（`q = k = v = input`・mask なし・非 causal）として
/// `Module::forward` を定義する（モジュール doc「`Module` trait との
/// 関係」参照）。`forward_host`／`as_linear`／`as_relu` はいずれも
/// trait 既定のままオーバーライドしない。
impl Module for MultiheadAttention {
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.bind(tape).forward(input, input, input, None, false)
    }

    /// イシュー #1760（親 #1618）: `compat::Sequential` の学習経路が
    /// `MultiheadAttention` 層を認識するためのフック（`as_linear` と
    /// 同型）。モジュール doc「`Module` trait との関係」が当初挙げて
    /// いた「`compat::Sequential::add_*` が対応する層集合に含まれて
    /// いない」という前提は本イシューで解消した。
    fn as_multihead_attention(&self) -> Option<&MultiheadAttention> {
        Some(self)
    }

    /// [`Module::as_multihead_attention`] の可変版。
    fn as_multihead_attention_mut(&mut self) -> Option<&mut MultiheadAttention> {
        Some(self)
    }

    /// `forward_host` は trait 既定のまま（常に `Unsupported`）。
    /// [`Module::supports_forward_host`] を `false` へオーバーライド
    /// し、`compat::Sequential::predict` の tape 不要経路が本層で
    /// `Unsupported` に当たる前に全層を事前判定できるようにする
    /// （イシュー #1760・Cursor Bugbot 指摘是正: `Embedding` と同型）。
    fn supports_forward_host(&self) -> bool {
        false
    }

    /// 命名契約（`Module::named_parameters` doc §「命名契約」）:
    /// `q_proj.*` → `k_proj.*` → `v_proj.*` → `out_proj.*` の順で、各
    /// `Linear::named_parameters()`（`weight` → `bias`）に接頭辞を連結
    /// する（`module::prefixed` ヘルパー参照）。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut out = prefixed("q_proj", self.q_proj.named_parameters());
        out.extend(prefixed("k_proj", self.k_proj.named_parameters()));
        out.extend(prefixed("v_proj", self.v_proj.named_parameters()));
        out.extend(prefixed("out_proj", self.out_proj.named_parameters()));
        out
    }

    /// [`Module::set_parameter`] の実装（イシュー #1752）。
    /// `strip_child_prefix` で `q_proj.`／`k_proj.`／`v_proj.`／
    /// `out_proj.` のいずれかを剥がし、対応する `Linear::set_parameter`
    /// へ委譲する（`named_parameters` の接頭辞契約〈直上参照〉の
    /// 逆演算）。該当する接頭辞がない名前は未知名として拒否する。
    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        if let Some(rest) = strip_child_prefix(name, "q_proj") {
            return Linear::set_parameter(&mut self.q_proj, rest, value);
        }
        if let Some(rest) = strip_child_prefix(name, "k_proj") {
            return Linear::set_parameter(&mut self.k_proj, rest, value);
        }
        if let Some(rest) = strip_child_prefix(name, "v_proj") {
            return Linear::set_parameter(&mut self.v_proj, rest, value);
        }
        if let Some(rest) = strip_child_prefix(name, "out_proj") {
            return Linear::set_parameter(&mut self.out_proj, rest, value);
        }
        Err(AutodiffError::InvalidArgument(format!(
            "MultiheadAttention::set_parameter: no parameter named `{name}`"
        )))
    }

    /// [`Module::set_requires_grad`] の実装（イシュー #2137）。4 つの
    /// 子 `Linear`（`q_proj`／`k_proj`／`v_proj`／`out_proj`）すべてへ
    /// 伝播する。子はいずれも本クレート内の `Linear`（fail-closed 既定
    /// の対象外）のため実際には常に `Ok` を返すが、`Module` trait の
    /// 汎用契約に従い `?` で伝播する。
    fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
        Module::set_requires_grad(&mut self.q_proj, requires_grad)?;
        Module::set_requires_grad(&mut self.k_proj, requires_grad)?;
        Module::set_requires_grad(&mut self.v_proj, requires_grad)?;
        Module::set_requires_grad(&mut self.out_proj, requires_grad)?;
        Ok(())
    }

    /// 4 つの子 `Linear` はすべて private フィールドのため常に揃った
    /// 値を返す（`q_proj` の値を代表として返す）。
    fn requires_grad(&self) -> bool {
        Module::requires_grad(&self.q_proj)
    }

    /// [`Module::children`] の実装（イシュー #2134）。順序・名前は
    /// [`Self::named_parameters`] の接頭辞契約（`q_proj`→`k_proj`→
    /// `v_proj`→`out_proj`）と一致させる。
    fn children(&self) -> Vec<(String, &dyn Module)> {
        vec![
            ("q_proj".to_string(), &self.q_proj as &dyn Module),
            ("k_proj".to_string(), &self.k_proj as &dyn Module),
            ("v_proj".to_string(), &self.v_proj as &dyn Module),
            ("out_proj".to_string(), &self.out_proj as &dyn Module),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- offset_allowed_mask（イシュー #2084 の (c) mask 規則） ---

    #[test]
    fn offset_allowed_mask_matches_causal_blocked_mask_when_s_prev_is_zero() {
        // s_prev == 0 のとき、offset_allowed_mask（true=attend）は
        // causal_blocked_mask（true=block）の否定と一致するはず
        // （両者とも `j <= i` を境界とする causal 規約。モジュール doc
        // 参照）。
        for (l, s) in [(3, 3), (4, 2), (2, 4), (1, 1)] {
            let allowed = offset_allowed_mask(l, 0, s).unwrap();
            let blocked = causal_blocked_mask(l, s).unwrap();
            let allowed_data = allowed.as_slice().unwrap();
            let blocked_data = blocked.as_slice().unwrap();
            let negated: Vec<bool> = blocked_data.iter().map(|&b| !b).collect();
            assert_eq!(allowed_data, negated.as_slice(), "l={l} s={s}");
        }
    }

    #[test]
    fn offset_allowed_mask_with_nonzero_s_prev_shifts_boundary() {
        // s_prev=2・l_new=2・s_total=4 のとき、行 i の許可境界は
        // `s_prev + i`（i=0 -> j<=2・i=1 -> j<=3）。
        let m = offset_allowed_mask(2, 2, 4).unwrap();
        assert_eq!(m.shape(), &[2, 4]);
        let data = m.as_slice().unwrap();
        assert_eq!(
            data,
            &[
                true, true, true, false, // i=0: j<=2
                true, true, true, true, // i=1: j<=3
            ]
        );
    }

    #[test]
    fn offset_allowed_mask_never_produces_fully_masked_row() {
        // `j <= s_prev + i` は `s_prev + i >= 0` である限り `j = 0` を
        // 常に許可するため、s_total > 0 なら全 masked 行は構造的に
        // 生じない。`reject_fully_masked_rows` は「true=block」極性を
        // 期待するため、`offset_allowed_mask`（true=attend）を否定して
        // から渡す（[`negate_broadcast_mask`] と同じ極性変換）。
        for s_prev in 0..4 {
            for l in 1..4 {
                let s_total = s_prev + l;
                let allowed = offset_allowed_mask(l, s_prev, s_total).unwrap();
                let blocked_data: Vec<bool> =
                    allowed.as_slice().unwrap().iter().map(|&a| !a).collect();
                let blocked = Tensor::new(blocked_data, &[l, s_total]).unwrap();
                assert!(
                    reject_fully_masked_rows(&blocked).is_ok(),
                    "s_prev={s_prev} l={l}"
                );
            }
        }
    }

    #[test]
    fn offset_allowed_mask_rejects_overflow() {
        let err = offset_allowed_mask(usize::MAX, 1, usize::MAX).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    // --- KvCache（イシュー #2084） ---

    #[test]
    fn kv_cache_new_is_empty() {
        let cache = KvCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.batch(), None);
        assert_eq!(cache.embed_dim(), None);
        assert!(cache.k().is_none());
        assert!(cache.v().is_none());
    }

    #[test]
    fn kv_cache_clear_resets_to_empty() {
        let mut cache = KvCache::new();
        cache.k = Some(Tensor::new(vec![0.0f32; 2 * 3 * 4], &[2, 3, 4]).unwrap());
        cache.v = Some(Tensor::new(vec![0.0f32; 2 * 3 * 4], &[2, 3, 4]).unwrap());
        assert!(!cache.is_empty());
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn kv_cache_clone_shares_storage() {
        // `Tensor<f32>` は `Arc<Storage<T>>` を共有する値型のため、
        // `clone()` 後の as_slice ポインタは呼び出し元の同一 `Arc` を
        // 指す（実データコピーではないことの間接検証。`Tensor` は
        // `as_slice` が返す生ポインタを直接比較する API を持たない
        // ため、値の一致で代替する）。
        let mut cache = KvCache::new();
        cache.k = Some(Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 1, 4]).unwrap());
        cache.v = Some(Tensor::new(vec![5.0f32, 6.0, 7.0, 8.0], &[1, 1, 4]).unwrap());
        let cloned = cache.clone();
        assert_eq!(
            cloned.k().unwrap().as_slice().unwrap(),
            cache.k().unwrap().as_slice().unwrap()
        );
    }

    #[test]
    fn causal_blocked_mask_square_is_strict_upper_triangle() {
        let m = causal_blocked_mask(3, 3).unwrap();
        assert_eq!(m.shape(), &[3, 3]);
        let data = m.as_slice().unwrap();
        let expected = [
            false, true, true, // i=0: j=0 のみ許可
            false, false, true, // i=1: j=0,1 許可
            false, false, false, // i=2: 全許可
        ];
        assert_eq!(data, &expected);
    }

    #[test]
    fn causal_blocked_mask_rectangular_l_gt_s_last_rows_fully_open() {
        let m = causal_blocked_mask(4, 2).unwrap();
        let data = m.as_slice().unwrap();
        assert_eq!(
            data,
            &[false, true, false, false, false, false, false, false]
        );
    }

    #[test]
    fn causal_blocked_mask_rectangular_l_lt_s_later_columns_blocked() {
        let m = causal_blocked_mask(2, 4).unwrap();
        let data = m.as_slice().unwrap();
        assert_eq!(data, &[false, true, true, true, false, false, true, true]);
    }

    #[test]
    fn causal_blocked_mask_never_produces_fully_masked_row() {
        for l in 1..6 {
            for s in 1..6 {
                let m = causal_blocked_mask(l, s).unwrap();
                assert!(reject_fully_masked_rows(&m).is_ok(), "l={l} s={s}");
            }
        }
    }

    #[test]
    fn reject_fully_masked_rows_detects_all_true_row() {
        let m = Tensor::new(vec![false, true, false, true], &[2, 2]).unwrap();
        assert!(reject_fully_masked_rows(&m).is_ok());
        let m2 = Tensor::new(vec![true, true, false, true], &[2, 2]).unwrap();
        assert!(reject_fully_masked_rows(&m2).is_err());
    }

    #[test]
    fn reject_fully_masked_rows_zero_cols_is_vacuously_ok() {
        let m = Tensor::new(Vec::<bool>::new(), &[3, 0]).unwrap();
        assert!(reject_fully_masked_rows(&m).is_ok());
    }

    #[test]
    fn negate_broadcast_mask_broadcasts_and_negates() {
        let mask = Tensor::new(vec![true, false], &[1, 2]).unwrap();
        let blocked = negate_broadcast_mask(&mask, &[2, 2]).unwrap();
        assert_eq!(blocked.shape(), &[2, 2]);
        assert_eq!(blocked.as_slice().unwrap(), &[false, true, false, true]);
    }

    /// codex-review（P2）・Cursor Bugbot 指摘の回帰テスト（PR #1846）:
    /// `sdpa_compose` は `l == 0`（空系列）でも `attn_mask` の
    /// broadcast 形状検証を省略してはならない。query=[2,0,4]・
    /// key/value=[2,4,4]（scores shape は [2,0,4]）に対し、
    /// broadcast 不能な mask=[3,5] を渡すと `AutodiffError::Shape` を
    /// 返すことを確認する（`crate::attention::
    /// scaled_dot_product_attention` の同一契約と整合。全 masked 行の
    /// 値検査のみを省略し、形状検証自体は常に行う）。
    #[test]
    fn sdpa_compose_rejects_mask_not_broadcastable_even_with_zero_l() {
        let q = Tensor::new(Vec::<f32>::new(), &[2, 0, 4]).unwrap();
        let k = Tensor::new(vec![0.0_f32; 2 * 4 * 4], &[2, 4, 4]).unwrap();
        let v = Tensor::new(vec![0.0_f32; 2 * 4 * 4], &[2, 4, 4]).unwrap();
        let mask = Tensor::new(vec![true; 3 * 5], &[3, 5]).unwrap();

        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let (qv, kv, vv) = (tape.var(&q), tape.var(&k), tape.var(&v));

        let err = sdpa_compose(&qv, &kv, &vv, Some(&mask), false, 0.5, None).unwrap_err();
        assert!(
            matches!(err, AutodiffError::Shape(_)),
            "broadcast 不能な attn_mask は L==0 でも Shape エラーになるべき（実際: {err:?}）"
        );
    }

    /// 空系列（`l == 0`）で broadcast 可能な `attn_mask` は受理され、
    /// 値検査（全 masked 行）は省略されたまま空テンソルを返すことを
    /// 確認する（上記回帰テストの対照: 形状検証は必ず行うが、
    /// 妥当な形状なら空系列自体は従来どおり成功する）。
    #[test]
    fn sdpa_compose_zero_l_with_valid_attn_mask_returns_empty_tensor_without_panic() {
        let q = Tensor::new(Vec::<f32>::new(), &[2, 0, 4]).unwrap();
        let k = Tensor::new(vec![0.0_f32; 2 * 4 * 4], &[2, 4, 4]).unwrap();
        let v = Tensor::new(vec![0.0_f32; 2 * 4 * 4], &[2, 4, 4]).unwrap();
        // [1, 4] -> [2, 0, 4] へ broadcast 可能（全 true でも空系列なら
        // 全 masked 行検査自体が発生しない）。
        let mask = Tensor::new(vec![true; 4], &[1, 4]).unwrap();

        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let (qv, kv, vv) = (tape.var(&q), tape.var(&k), tape.var(&v));

        let out = sdpa_compose(&qv, &kv, &vv, Some(&mask), false, 0.5, None).unwrap();
        assert_eq!(out.shape(), &[2, 0, 4]);
    }

    #[test]
    fn validate_embed_heads_rejects_zero_and_non_divisible() {
        assert!(validate_embed_heads(0, 2).is_err());
        assert!(validate_embed_heads(4, 0).is_err());
        assert!(validate_embed_heads(5, 2).is_err());
        assert!(validate_embed_heads(4, 2).is_ok());
    }

    #[test]
    fn new_rejects_invalid_embed_heads() {
        assert!(MultiheadAttention::new(0, 2, true, 1).is_err());
        assert!(MultiheadAttention::new(4, 0, true, 1).is_err());
        assert!(MultiheadAttention::new(5, 2, true, 1).is_err());
    }

    /// codex-review 指摘（PR #1846）の回帰テスト: `MultiheadAttentionVars::
    /// new` は weight 側（q/k/v/out）の `check_same_tape` は行っていたが
    /// bias 側は形状のみ検査していたため、weight を tape A・同形状の bias
    /// を tape B に登録した `LinearVars` でも構築が成功してしまい、後続の
    /// `forward` 内の `add` まで `TapeMismatch` の検出が遅延していた。
    /// bias にも `q.weight.check_same_tape` を適用したことで、構築時点
    /// （`new` 自身）で `TapeMismatch` を返すことを確認する。
    #[test]
    fn new_rejects_bias_on_different_tape_than_weight() {
        let e = 4;
        let weight = Tensor::new(vec![0.0_f32; e * e], &[e, e]).unwrap();
        let bias = Tensor::new(vec![0.0_f32; e], &[e]).unwrap();

        let tape_weights = Tape::new_with_ops(crate::default_ops::naive_ops());
        let tape_bias = Tape::new_with_ops(crate::default_ops::naive_ops());

        // q/k/v/out の weight はすべて tape_weights 上に揃える。
        let mk_matching = || LinearVars {
            weight: tape_weights.var(&weight),
            bias: Some(tape_weights.var(&bias)),
        };
        let q = mk_matching();
        let k = mk_matching();
        let v = mk_matching();
        // out だけ bias を別 tape（tape_bias）へ登録し、weight は
        // tape_weights のまま揃える（weight 同士の `check_same_tape` は
        // 通過するが bias が食い違うケースを再現する）。
        let out = LinearVars {
            weight: tape_weights.var(&weight),
            bias: Some(tape_bias.var(&bias)),
        };

        match MultiheadAttentionVars::new(2, q, k, v, out) {
            Err(AutodiffError::TapeMismatch) => {}
            other => panic!(
                "bias の Tape 不一致は構築時点で TapeMismatch として検出されるべき（is_err={}）",
                other.is_err()
            ),
        }
    }

    /// Cursor Bugbot 指摘（PR #2279）の回帰テスト: `LinearVars` は
    /// `weight`/`bias` が `pub` のため `Linear::from_parameters` の
    /// zero-K 検証（`crates/autodiff/src/nn/linear.rs`）を経由せずに
    /// `[0, E]` 形状の weight を持つ `LinearVars` を構築できてしまう。
    /// `MultiheadAttentionVars::new`（`validate_projection_vars`）が
    /// `kdim == 0`／`vdim == 0` を明示的に拒否し、
    /// `MultiheadAttention::from_config` の同検証（`from_config_
    /// rejects_zero_kdim_or_vdim`）と対称であることを固定する。
    #[test]
    fn new_rejects_zero_kdim_or_vdim_on_vars_path() {
        let e = 4;
        let weight_e = Tensor::new(vec![0.0_f32; e * e], &[e, e]).unwrap();
        let bias_e = Tensor::new(vec![0.0_f32; e], &[e]).unwrap();
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());

        let mk_square = || LinearVars {
            weight: tape.var(&weight_e),
            bias: Some(tape.var(&bias_e)),
        };
        let mk_zero_kdim = || {
            // `[0, E]`: in_features（kdim）が 0 の縮退 weight。
            let zero_weight = Tensor::new(Vec::<f32>::new(), &[0, e]).unwrap();
            LinearVars {
                weight: tape.var(&zero_weight),
                bias: Some(tape.var(&bias_e)),
            }
        };

        // kdim == 0（k のみ縮退）。
        assert!(matches!(
            MultiheadAttentionVars::new(2, mk_square(), mk_zero_kdim(), mk_square(), mk_square()),
            Err(AutodiffError::InvalidArgument(_))
        ));
        // vdim == 0（v のみ縮退）。
        assert!(matches!(
            MultiheadAttentionVars::new(2, mk_square(), mk_square(), mk_zero_kdim(), mk_square()),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn new_produces_distinct_deterministic_projections() {
        let a = MultiheadAttention::new(4, 2, true, 7).unwrap();
        let b = MultiheadAttention::new(4, 2, true, 7).unwrap();
        assert_eq!(
            a.q_proj().weight().as_slice().unwrap(),
            b.q_proj().weight().as_slice().unwrap(),
            "同一シードなら q_proj の weight は決定的に一致するはず"
        );
        assert_ne!(
            a.q_proj().weight().as_slice().unwrap(),
            a.k_proj().weight().as_slice().unwrap(),
            "q_proj と k_proj の weight は互いに異なるはず（独立シード導出）"
        );
        assert_ne!(
            a.q_proj().weight().as_slice().unwrap(),
            a.v_proj().weight().as_slice().unwrap()
        );
        assert_ne!(
            a.q_proj().weight().as_slice().unwrap(),
            a.out_proj().weight().as_slice().unwrap()
        );
        assert_eq!(a.embed_dim(), 4);
        assert_eq!(a.num_heads(), 2);
        assert_eq!(a.head_dim(), 2);
    }

    #[test]
    fn from_parameters_rejects_non_square_weight() {
        use fandhe_ai_tensor_core::Tensor;
        let bad =
            Linear::from_parameters(Tensor::new(vec![0.1; 4 * 6], &[4, 6]).unwrap(), None).unwrap();
        let ok = Linear::new(4, 4, false, 1).unwrap();
        assert!(
            MultiheadAttention::from_parameters(
                2,
                bad,
                Linear::new(4, 4, false, 2).unwrap(),
                Linear::new(4, 4, false, 3).unwrap(),
                ok,
            )
            .is_err()
        );
    }

    #[test]
    fn from_parameters_rejects_mixed_bias_presence() {
        let with_bias = Linear::new(4, 4, true, 1).unwrap();
        let without_bias = Linear::new(4, 4, false, 2).unwrap();
        assert!(
            MultiheadAttention::from_parameters(
                2,
                with_bias,
                without_bias,
                Linear::new(4, 4, true, 3).unwrap(),
                Linear::new(4, 4, true, 4).unwrap(),
            )
            .is_err()
        );
    }

    #[test]
    fn from_parameters_rejects_embed_dim_not_divisible_by_num_heads() {
        let make = |seed| Linear::new(5, 5, false, seed).unwrap();
        assert!(
            MultiheadAttention::from_parameters(2, make(1), make(2), make(3), make(4)).is_err()
        );
    }

    #[test]
    fn from_parameters_accepts_consistent_projections() {
        let make = |seed| Linear::new(4, 4, true, seed).unwrap();
        let mha = MultiheadAttention::from_parameters(2, make(1), make(2), make(3), make(4))
            .expect("4 層とも正方 [4,4]・bias 全 Some・4 % 2 == 0 のため成功するはず");
        assert_eq!(mha.embed_dim(), 4);
        assert_eq!(mha.head_dim(), 2);
    }

    // `MultiheadAttention::set_parameter`（イシュー #1752）の単体
    // テスト。

    #[test]
    fn set_parameter_delegates_to_correct_projection() {
        let make = |seed| Linear::new(4, 4, true, seed).unwrap();
        let mut mha =
            MultiheadAttention::from_parameters(2, make(1), make(2), make(3), make(4)).unwrap();
        let new_weight = Tensor::new(vec![9.0f32; 16], &[4, 4]).unwrap();
        mha.set_parameter("k_proj.weight", new_weight.clone())
            .unwrap();
        assert_eq!(
            mha.k_proj().weight().contiguous().as_slice().unwrap(),
            new_weight.contiguous().as_slice().unwrap()
        );
        // 他の projection は不変。
        assert_ne!(
            mha.q_proj().weight().contiguous().as_slice().unwrap(),
            new_weight.contiguous().as_slice().unwrap()
        );
    }

    #[test]
    fn set_parameter_rejects_unknown_prefix() {
        let make = |seed| Linear::new(4, 4, true, seed).unwrap();
        let mut mha =
            MultiheadAttention::from_parameters(2, make(1), make(2), make(3), make(4)).unwrap();
        let err = mha
            .set_parameter("bogus.weight", mha.q_proj().weight().clone())
            .expect_err("未知の接頭辞は Err を返すはず");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    // === イシュー #2163（親 #2131）: MultiheadAttentionConfig（構築時
    // オプション: batch_first・kdim/vdim）・key_padding_mask ===

    #[test]
    fn config_defaults_match_pre_2163_behavior() {
        let cfg = MultiheadAttentionConfig::new(4, 2);
        assert_eq!(cfg.embed_dim(), 4);
        assert_eq!(cfg.num_heads(), 2);
        assert!(cfg.bias());
        assert!(cfg.batch_first());
        assert_eq!(cfg.kdim(), 4);
        assert_eq!(cfg.vdim(), 4);
    }

    #[test]
    fn config_with_methods_override_fields() {
        let cfg = MultiheadAttentionConfig::new(4, 2)
            .with_bias(false)
            .with_batch_first(false)
            .with_kdim(6)
            .with_vdim(8);
        assert!(!cfg.bias());
        assert!(!cfg.batch_first());
        assert_eq!(cfg.kdim(), 6);
        assert_eq!(cfg.vdim(), 8);
    }

    #[test]
    fn from_config_rejects_zero_kdim_or_vdim() {
        let cfg_k = MultiheadAttentionConfig::new(4, 2).with_kdim(0);
        assert!(MultiheadAttention::from_config(&cfg_k, 1).is_err());
        let cfg_v = MultiheadAttentionConfig::new(4, 2).with_vdim(0);
        assert!(MultiheadAttention::from_config(&cfg_v, 1).is_err());
    }

    /// [`MultiheadAttention::new`] が `from_config` へ委譲するように
    /// なった後も（イシュー #2163）、シード導出・4 層の初期値が変更前と
    /// **bit 同一**であることを固定する。
    #[test]
    fn new_matches_from_config_bit_identical() {
        let via_new = MultiheadAttention::new(6, 3, true, 42).unwrap();
        let cfg = MultiheadAttentionConfig::new(6, 3).with_bias(true);
        let via_config = MultiheadAttention::from_config(&cfg, 42).unwrap();
        for (a, b) in [
            (via_new.q_proj(), via_config.q_proj()),
            (via_new.k_proj(), via_config.k_proj()),
            (via_new.v_proj(), via_config.v_proj()),
            (via_new.out_proj(), via_config.out_proj()),
        ] {
            assert_eq!(
                a.weight().contiguous().as_slice().unwrap(),
                b.weight().contiguous().as_slice().unwrap()
            );
        }
        assert_eq!(via_new.kdim(), 6);
        assert_eq!(via_new.vdim(), 6);
        assert!(via_new.batch_first());
    }

    #[test]
    fn from_config_builds_asymmetric_kdim_vdim_projections() {
        let cfg = MultiheadAttentionConfig::new(4, 2)
            .with_kdim(6)
            .with_vdim(8);
        let mha = MultiheadAttention::from_config(&cfg, 1).unwrap();
        assert_eq!(mha.k_proj().weight().shape(), &[6, 4]);
        assert_eq!(mha.v_proj().weight().shape(), &[8, 4]);
        assert_eq!(mha.kdim(), 6);
        assert_eq!(mha.vdim(), 8);
    }

    #[test]
    fn from_parameters_infers_kdim_vdim_from_weight_shape() {
        let q = Linear::new(4, 4, true, 1).unwrap();
        let k = Linear::new(6, 4, true, 2).unwrap();
        let v = Linear::new(8, 4, true, 3).unwrap();
        let out = Linear::new(4, 4, true, 4).unwrap();
        let mha = MultiheadAttention::from_parameters(2, q, k, v, out).unwrap();
        assert_eq!(mha.kdim(), 6);
        assert_eq!(mha.vdim(), 8);
        assert!(mha.batch_first());
    }

    #[test]
    fn from_parameters_with_config_rejects_mismatched_kdim() {
        let cfg = MultiheadAttentionConfig::new(4, 2).with_kdim(6);
        let q = Linear::new(4, 4, true, 1).unwrap();
        let k = Linear::new(7, 4, true, 2).unwrap(); // kdim=7 != config.kdim()=6
        let v = Linear::new(4, 4, true, 3).unwrap();
        let out = Linear::new(4, 4, true, 4).unwrap();
        assert!(MultiheadAttention::from_parameters_with_config(&cfg, q, k, v, out).is_err());
    }

    #[test]
    fn from_parameters_with_config_accepts_consistent_batch_first_false() {
        let cfg = MultiheadAttentionConfig::new(4, 2).with_batch_first(false);
        let q = Linear::new(4, 4, true, 1).unwrap();
        let k = Linear::new(4, 4, true, 2).unwrap();
        let v = Linear::new(4, 4, true, 3).unwrap();
        let out = Linear::new(4, 4, true, 4).unwrap();
        let mha = MultiheadAttention::from_parameters_with_config(&cfg, q, k, v, out).unwrap();
        assert!(!mha.batch_first());
    }

    /// `new_with_config` 系テスト（[`new_with_config_matches_new_when_unchanged`]・
    /// [`new_with_config_rejects_mismatched_bias`] ほか）共通のヘルパー
    /// （テスト内クロージャは `&Tape` の借用ライフタイムを
    /// `LinearVars<'t>` の `'t` へ正しく単一化できないため `fn` にする）。
    /// bias 付き（`Some`）の `LinearVars` を返す。bias 無し版は
    /// [`fixture_linear_vars_no_bias`]。
    fn fixture_linear_vars<'t>(tape: &'t Tape, seed: i64, e: usize) -> LinearVars<'t> {
        LinearVars {
            weight: tape.var(
                &Tensor::new(
                    (0..e * e)
                        .map(|i| (seed + i as i64) as f32 * 0.01)
                        .collect(),
                    &[e, e],
                )
                .unwrap(),
            ),
            bias: Some(
                tape.var(
                    &Tensor::new(
                        (0..e).map(|i| (seed + i as i64) as f32 * 0.01).collect(),
                        &[e],
                    )
                    .unwrap(),
                ),
            ),
        }
    }

    /// [`fixture_linear_vars`] の bias 無し版（`new_with_config` の
    /// bias 不一致検証テスト専用）。
    fn fixture_linear_vars_no_bias<'t>(tape: &'t Tape, seed: i64, e: usize) -> LinearVars<'t> {
        LinearVars {
            weight: tape.var(
                &Tensor::new(
                    (0..e * e)
                        .map(|i| (seed + i as i64) as f32 * 0.01)
                        .collect(),
                    &[e, e],
                )
                .unwrap(),
            ),
            bias: None,
        }
    }

    /// レビュー指摘（イシュー #2163・PR #2279）: `new_with_config` は
    /// 次元一致のみでなく `config.bias()` と実パラメータの bias 有無も
    /// 検証する（`from_parameters_with_config` と同型の契約）。
    /// `with_bias(false)` の config に bias 付き 4 層を渡すと拒否される
    /// ことを固定する。
    #[test]
    fn new_with_config_rejects_mismatched_bias() {
        let e = 4;
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let q = fixture_linear_vars(&tape, 1, e); // bias = Some(..)
        let k = fixture_linear_vars(&tape, 2, e);
        let v = fixture_linear_vars(&tape, 3, e);
        let out = fixture_linear_vars(&tape, 4, e);
        let cfg = MultiheadAttentionConfig::new(e, 2).with_bias(false);
        assert!(MultiheadAttentionVars::new_with_config(&cfg, q, k, v, out).is_err());
    }

    /// bias 無し 4 層 + `with_bias(false)` config は一致するため成功する
    /// ことを固定する（上記の否定側だけでなく肯定側も検証する）。
    #[test]
    fn new_with_config_accepts_consistent_no_bias() {
        let e = 4;
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let q = fixture_linear_vars_no_bias(&tape, 1, e);
        let k = fixture_linear_vars_no_bias(&tape, 2, e);
        let v = fixture_linear_vars_no_bias(&tape, 3, e);
        let out = fixture_linear_vars_no_bias(&tape, 4, e);
        let cfg = MultiheadAttentionConfig::new(e, 2).with_bias(false);
        assert!(MultiheadAttentionVars::new_with_config(&cfg, q, k, v, out).is_ok());
    }

    #[test]
    fn new_with_config_matches_new_when_unchanged() {
        let e = 4;
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let q = fixture_linear_vars(&tape, 1, e);
        let k = fixture_linear_vars(&tape, 2, e);
        let v = fixture_linear_vars(&tape, 3, e);
        let out = fixture_linear_vars(&tape, 4, e);
        let cfg = MultiheadAttentionConfig::new(e, 2);
        let vars = MultiheadAttentionVars::new_with_config(&cfg, q, k, v, out).unwrap();
        assert_eq!(vars.embed_dim(), e);
        assert_eq!(vars.kdim(), e);
        assert_eq!(vars.vdim(), e);
        assert!(vars.batch_first());
    }

    /// [`MultiheadAttentionVars::forward`] の既定経路
    /// （`batch_first=true`・`key_padding_mask=None`）が
    /// `forward_with_key_padding_mask` へ委譲するようになった後も出力が
    /// 変更前と bit 同一であることを、`forward_with_key_padding_mask` を
    /// 直接呼んだ結果との一致として固定する（内部委譲のため本質的に
    /// 同一だが、リグレッション検知のために明示する）。
    #[test]
    fn forward_default_path_matches_forward_with_key_padding_mask_none() {
        let make = |seed| Linear::new(4, 4, true, seed).unwrap();
        let mha =
            MultiheadAttention::from_parameters(2, make(1), make(2), make(3), make(4)).unwrap();
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let vars = mha.bind(&tape);
        let x = Tensor::new(
            (0..2 * 3 * 4).map(|i| i as f32 * 0.01).collect(),
            &[2, 3, 4],
        )
        .unwrap();
        let xv = tape.var(&x);
        let out_a = vars
            .forward(&xv, &xv, &xv, None, false)
            .unwrap()
            .to_tensor();
        let out_b = vars
            .forward_with_key_padding_mask(&xv, &xv, &xv, None, None, false)
            .unwrap()
            .to_tensor();
        assert_eq!(out_a.as_slice().unwrap(), out_b.as_slice().unwrap());
    }

    /// `batch_first=false` の forward が「同じパラメータの
    /// `batch_first=true` 版へ transpose 済み入力を与え出力を transpose
    /// したもの」と bit 一致することを固定する（両者とも同一の合成
    /// 演算列に帰着するため）。
    #[test]
    fn batch_first_false_matches_transposed_batch_first_true() {
        let (b, l, e, h) = (2, 3, 4, 2);
        let cfg_bf_true = MultiheadAttentionConfig::new(e, h);
        let cfg_bf_false = MultiheadAttentionConfig::new(e, h).with_batch_first(false);

        // `Linear` は `Clone` を実装しないため、決定的シードから独立に
        // 2 組構築する（同一シード -> bit 同一な weight/bias。`new`
        // doc の決定性契約と同じ根拠）。
        let build = || {
            (
                Linear::new(e, e, true, 1).unwrap(),
                Linear::new(e, e, true, 2).unwrap(),
                Linear::new(e, e, true, 3).unwrap(),
                Linear::new(e, e, true, 4).unwrap(),
            )
        };
        let (q1, k1, v1, out1) = build();
        let (q2, k2, v2, out2) = build();

        let mha_true =
            MultiheadAttention::from_parameters_with_config(&cfg_bf_true, q1, k1, v1, out1)
                .unwrap();
        let mha_false =
            MultiheadAttention::from_parameters_with_config(&cfg_bf_false, q2, k2, v2, out2)
                .unwrap();

        let x = Tensor::new(
            (0..b * l * e).map(|i| (i as f32 * 0.017) - 1.0).collect(),
            &[b, l, e],
        )
        .unwrap();

        let tape_a = Tape::new_with_ops(crate::default_ops::naive_ops());
        let xv = tape_a.var(&x);
        let out_true = mha_true
            .bind(&tape_a)
            .forward(&xv, &xv, &xv, None, false)
            .unwrap()
            .to_tensor();

        // batch_first=false 側は [L, B, E] へ transpose した入力を渡し、
        // 出力を [B, L, E] へ戻して突合する。
        let x_lbe = x.permute(&[1, 0, 2]).unwrap().contiguous();
        let tape_b = Tape::new_with_ops(crate::default_ops::naive_ops());
        let xv_lbe = tape_b.var(&x_lbe);
        let out_false_lbe = mha_false
            .bind(&tape_b)
            .forward(&xv_lbe, &xv_lbe, &xv_lbe, None, false)
            .unwrap()
            .to_tensor();
        let out_false_ble = out_false_lbe.permute(&[1, 0, 2]).unwrap().contiguous();

        assert_eq!(
            out_true.as_slice().unwrap(),
            out_false_ble.as_slice().unwrap()
        );
    }

    /// `key_padding_mask` が「等価な `[B, 1, 1, S]` の `attn_mask`」と
    /// bit 一致することを固定する（`combine_masks_with_key_padding` の
    /// `[B, H, L, S]` 帰着設計。モジュール doc 参照）。
    #[test]
    fn key_padding_mask_matches_equivalent_attn_mask() {
        let (b, l, s, e, h) = (2, 3, 4, 4, 2);
        let make = |seed| Linear::new(e, e, true, seed).unwrap();
        let mha =
            MultiheadAttention::from_parameters(h, make(1), make(2), make(3), make(4)).unwrap();

        let q = Tensor::new(
            (0..b * l * e).map(|i| i as f32 * 0.01).collect(),
            &[b, l, e],
        )
        .unwrap();
        let k = Tensor::new(
            (0..b * s * e).map(|i| i as f32 * 0.02 - 0.5).collect(),
            &[b, s, e],
        )
        .unwrap();
        let v = Tensor::new(
            (0..b * s * e).map(|i| i as f32 * 0.03 - 0.3).collect(),
            &[b, s, e],
        )
        .unwrap();

        // kpm[b, j]: バッチ 0 は先頭 2 key のみ許可、バッチ 1 は末尾
        // 3 key のみ許可（s=4 のため全 masked 行にはならない）。
        let kpm_data = vec![true, true, false, false, false, true, true, true];
        let kpm = Tensor::new(kpm_data.clone(), &[b, s]).unwrap();
        // 等価な attn_mask: [B, 1, 1, S]（sdpa_compose が [B, H, L, S] へ
        // broadcast する）。
        let attn_mask_equiv = Tensor::new(kpm_data, &[b, 1, 1, s]).unwrap();

        let tape_a = Tape::new_with_ops(crate::default_ops::naive_ops());
        let vars_a = mha.bind(&tape_a);
        let (qa, ka, va) = (tape_a.var(&q), tape_a.var(&k), tape_a.var(&v));
        let out_kpm = vars_a
            .forward_with_key_padding_mask(&qa, &ka, &va, None, Some(&kpm), false)
            .unwrap()
            .to_tensor();

        let tape_b = Tape::new_with_ops(crate::default_ops::naive_ops());
        let vars_b = mha.bind(&tape_b);
        let (qb, kb, vb) = (tape_b.var(&q), tape_b.var(&k), tape_b.var(&v));
        let out_attn = vars_b
            .forward(&qb, &kb, &vb, Some(&attn_mask_equiv), false)
            .unwrap()
            .to_tensor();

        assert_eq!(out_kpm.as_slice().unwrap(), out_attn.as_slice().unwrap());
    }

    /// 全 `true` の `key_padding_mask` は「mask なし」と bit 一致する
    /// ことを固定する（既定経路との整合）。
    #[test]
    fn key_padding_mask_all_true_matches_no_mask() {
        let (b, l, s, e, h) = (2, 3, 4, 4, 2);
        let make = |seed| Linear::new(e, e, true, seed).unwrap();
        let mha =
            MultiheadAttention::from_parameters(h, make(1), make(2), make(3), make(4)).unwrap();
        let q = Tensor::new(
            (0..b * l * e).map(|i| i as f32 * 0.01).collect(),
            &[b, l, e],
        )
        .unwrap();
        let k = Tensor::new(
            (0..b * s * e).map(|i| i as f32 * 0.02).collect(),
            &[b, s, e],
        )
        .unwrap();
        let v = Tensor::new(
            (0..b * s * e).map(|i| i as f32 * 0.03).collect(),
            &[b, s, e],
        )
        .unwrap();
        let kpm = Tensor::new(vec![true; b * s], &[b, s]).unwrap();

        let tape_a = Tape::new_with_ops(crate::default_ops::naive_ops());
        let vars_a = mha.bind(&tape_a);
        let (qa, ka, va) = (tape_a.var(&q), tape_a.var(&k), tape_a.var(&v));
        let out_kpm = vars_a
            .forward_with_key_padding_mask(&qa, &ka, &va, None, Some(&kpm), false)
            .unwrap()
            .to_tensor();

        let tape_b = Tape::new_with_ops(crate::default_ops::naive_ops());
        let vars_b = mha.bind(&tape_b);
        let (qb, kb, vb) = (tape_b.var(&q), tape_b.var(&k), tape_b.var(&v));
        let out_none = vars_b
            .forward(&qb, &kb, &vb, None, false)
            .unwrap()
            .to_tensor();

        assert_eq!(out_kpm.as_slice().unwrap(), out_none.as_slice().unwrap());
    }

    /// `key_padding_mask + is_causal` が「kpm AND 下三角の明示
    /// `attn_mask`」と bit 一致することを固定する。
    #[test]
    fn key_padding_mask_with_causal_matches_explicit_combined_attn_mask() {
        let (b, l, s, e, h) = (1, 3, 3, 4, 2);
        let make = |seed| Linear::new(e, e, true, seed).unwrap();
        let mha =
            MultiheadAttention::from_parameters(h, make(1), make(2), make(3), make(4)).unwrap();
        let q = Tensor::new(
            (0..b * l * e).map(|i| i as f32 * 0.02).collect(),
            &[b, l, e],
        )
        .unwrap();
        let k = Tensor::new(
            (0..b * s * e).map(|i| i as f32 * 0.03).collect(),
            &[b, s, e],
        )
        .unwrap();
        let v = Tensor::new(
            (0..b * s * e).map(|i| i as f32 * 0.04).collect(),
            &[b, s, e],
        )
        .unwrap();
        // kpm: 末尾 key（j=2）を全バッチで無効化。
        let kpm = Tensor::new(vec![true, true, false], &[b, s]).unwrap();

        let tape_a = Tape::new_with_ops(crate::default_ops::naive_ops());
        let vars_a = mha.bind(&tape_a);
        let (qa, ka, va) = (tape_a.var(&q), tape_a.var(&k), tape_a.var(&v));
        let out_kpm = vars_a
            .forward_with_key_padding_mask(&qa, &ka, &va, None, Some(&kpm), true)
            .unwrap()
            .to_tensor();

        // 明示合成: causal（j<=i）AND kpm[j]。
        let mut combined = Vec::with_capacity(l * s);
        for i in 0..l {
            for j in 0..s {
                combined.push(j <= i && kpm.as_slice().unwrap()[j]);
            }
        }
        let combined_mask = Tensor::new(combined, &[l, s]).unwrap();

        let tape_b = Tape::new_with_ops(crate::default_ops::naive_ops());
        let vars_b = mha.bind(&tape_b);
        let (qb, kb, vb) = (tape_b.var(&q), tape_b.var(&k), tape_b.var(&v));
        let out_explicit = vars_b
            .forward(&qb, &kb, &vb, Some(&combined_mask), false)
            .unwrap()
            .to_tensor();

        assert_eq!(
            out_kpm.as_slice().unwrap(),
            out_explicit.as_slice().unwrap()
        );
    }

    #[test]
    fn forward_with_key_padding_mask_rejects_wrong_shape() {
        let (b, l, s, e, h) = (2, 3, 4, 4, 2);
        let make = |seed| Linear::new(e, e, true, seed).unwrap();
        let mha =
            MultiheadAttention::from_parameters(h, make(1), make(2), make(3), make(4)).unwrap();
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let vars = mha.bind(&tape);
        let q = tape.var(&Tensor::new(vec![0.0f32; b * l * e], &[b, l, e]).unwrap());
        let k = tape.var(&Tensor::new(vec![0.0f32; b * s * e], &[b, s, e]).unwrap());
        let v = tape.var(&Tensor::new(vec![0.0f32; b * s * e], &[b, s, e]).unwrap());
        // 誤った shape [S]（unbatched）。
        let bad_kpm = Tensor::new(vec![true; s], &[s]).unwrap();
        let err = vars
            .forward_with_key_padding_mask(&q, &k, &v, None, Some(&bad_kpm), false)
            .unwrap_err();
        assert!(matches!(err, AutodiffError::Shape(_)));
    }

    #[test]
    fn forward_with_key_padding_mask_rejects_attn_mask_and_causal_together() {
        let (b, l, s, e, h) = (1, 2, 2, 4, 2);
        let make = |seed| Linear::new(e, e, true, seed).unwrap();
        let mha =
            MultiheadAttention::from_parameters(h, make(1), make(2), make(3), make(4)).unwrap();
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let vars = mha.bind(&tape);
        let q = tape.var(&Tensor::new(vec![0.0f32; b * l * e], &[b, l, e]).unwrap());
        let k = tape.var(&Tensor::new(vec![0.0f32; b * s * e], &[b, s, e]).unwrap());
        let v = tape.var(&Tensor::new(vec![0.0f32; b * s * e], &[b, s, e]).unwrap());
        let kpm = Tensor::new(vec![true; b * s], &[b, s]).unwrap();
        let attn_mask = Tensor::new(vec![true; l * s], &[l, s]).unwrap();
        let err = vars
            .forward_with_key_padding_mask(&q, &k, &v, Some(&attn_mask), Some(&kpm), true)
            .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn forward_with_key_padding_mask_rejects_fully_masked_row() {
        let (b, l, s, e, h) = (1, 1, 2, 4, 2);
        let make = |seed| Linear::new(e, e, true, seed).unwrap();
        let mha =
            MultiheadAttention::from_parameters(h, make(1), make(2), make(3), make(4)).unwrap();
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let vars = mha.bind(&tape);
        let q = tape.var(&Tensor::new(vec![0.0f32; b * l * e], &[b, l, e]).unwrap());
        let k = tape.var(&Tensor::new(vec![0.0f32; b * s * e], &[b, s, e]).unwrap());
        let v = tape.var(&Tensor::new(vec![0.0f32; b * s * e], &[b, s, e]).unwrap());
        // 全 false（全 key を無視）。
        let kpm = Tensor::new(vec![false; b * s], &[b, s]).unwrap();
        let err = vars
            .forward_with_key_padding_mask(&q, &k, &v, None, Some(&kpm), false)
            .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn forward_with_key_padding_mask_rejects_key_final_axis_not_kdim() {
        let cfg = MultiheadAttentionConfig::new(4, 2).with_kdim(6);
        let mha = MultiheadAttention::from_config(&cfg, 1).unwrap();
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let vars = mha.bind(&tape);
        let q = tape.var(&Tensor::new(vec![0.0f32; 2 * 3 * 4], &[2, 3, 4]).unwrap());
        // key の最終軸が kdim(6) ではなく 4。
        let k = tape.var(&Tensor::new(vec![0.0f32; 2 * 3 * 4], &[2, 3, 4]).unwrap());
        let v = tape.var(&Tensor::new(vec![0.0f32; 2 * 3 * 6], &[2, 3, 6]).unwrap());
        let err = vars
            .forward_with_key_padding_mask(&q, &k, &v, None, None, false)
            .unwrap_err();
        assert!(matches!(err, AutodiffError::Shape(_)));
    }

    #[test]
    fn multihead_attention_forward_low_precision_rejects_non_default_config() {
        let cfg = MultiheadAttentionConfig::new(4, 2).with_batch_first(false);
        let mha = MultiheadAttention::from_config(&cfg, 1).unwrap();
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let vars = mha.bind(&tape);
        let x = tape.var(&Tensor::new(vec![0.0f32; 3 * 2 * 4], &[3, 2, 4]).unwrap());
        let err = multihead_attention_forward_low_precision(
            &vars,
            &x,
            &x,
            &x,
            None,
            false,
            ScalarDType::F16,
        )
        .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }
}
