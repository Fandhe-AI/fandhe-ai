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
//! **入出力契約（rank-3・batch_first 固定）**: `query: [B, L, E]`・
//! `key`/`value: [B, S, E]` → 出力 `[B, L, E]`。unbatched `[L, E]`・
//! `batch_first=false`・`kdim`/`vdim`・`key_padding_mask` 引数・
//! `dropout_p`（MHA への結線は対象外。`Var::dropout` 自体は #1603 で
//! 実装済み）・`need_weights`／attention weights の
//! 返却・`add_bias_kv`／`add_zero_attn`・packed `in_proj_weight`／
//! PyTorch `state_dict` 対応付け（#1616）は対象外
//! （`out-of-scope-tracking.md`）。
//!
//! **`Module` trait との関係**: `impl Module for MultiheadAttention` の
//! `forward` は self-attention（`q=k=v=input`・mask なし・非 causal）を
//! 表す。`forward_host`（tape 不要推論経路）は trait 既定のまま
//! （`Unsupported` fail-safe）で本イシューでは実装しない（対象外）。
//! `compat::Sequential` 用の `as_linear`／`as_relu` フックはいずれも
//! trait 既定（オーバーライドしない）のままとする——`MultiheadAttention`
//! は `Linear` の直接な代替ではなく、`compat::Sequential::add_*`
//! （#1616 系）が対応する層集合に含まれていないため、誤って学習可能
//! パラメータとして拾われたり `ReLU` 融合対象として先読みされたりする
//! ことはない。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::nn::init::{
    ATTN_K_SEED_SALT, ATTN_OUT_SEED_SALT, ATTN_Q_SEED_SALT, ATTN_V_SEED_SALT, derive_seed,
};
use crate::nn::linear::{Linear, LinearVars};
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

/// `MultiheadAttention::from_parameters` 向けの検証（`Tensor<f32>` を
/// 保持する `Linear` 本体を対象）。4 層の weight が全て正方 `[E, E]`
/// で同一 `E` を持つこと・bias が全て `Some`（各 `[E]`）または全て
/// `None` であることを検査し、`E`（`embed_dim`）を返す。`Linear::
/// from_parameters` 自身は各層を独立に検証する（rank・zero-K・bias
/// shape）ため、本関数はその上に「4 層をまたぐ整合性」のみを追加で
/// 検証する。
fn validate_linear_projections(
    q: &Linear,
    k: &Linear,
    v: &Linear,
    out: &Linear,
    num_heads: usize,
) -> Result<usize, AutodiffError> {
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
    for proj in [k, v, out] {
        let shape = proj.weight().shape();
        if shape != [e, e] {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: shape.to_vec(),
                rhs: vec![e, e],
            }));
        }
    }

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
    Ok(e)
}

/// `MultiheadAttentionVars::new` 向けの検証（`Var`〈テープ登録済み〉を
/// 保持する `LinearVars` を対象）。[`validate_linear_projections`] と
/// 同じ整合性（4 層とも正方 `[E, E]`・同一 `E`・bias 全 `Some`／全
/// `None`）に加え、4 層が同一 `Tape` に属することを検査する
/// （`Var::check_same_tape`）。
fn validate_projection_vars<'t>(
    q: &LinearVars<'t>,
    k: &LinearVars<'t>,
    v: &LinearVars<'t>,
    out: &LinearVars<'t>,
    num_heads: usize,
) -> Result<usize, AutodiffError> {
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
    for proj in [k, v, out] {
        let shape = proj.weight.shape();
        if shape.as_slice() != [e, e] {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: shape,
                rhs: vec![e, e],
            }));
        }
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
        // `add` まで TapeMismatch が遅延してしまう）。
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
    Ok(e)
}

/// `MultiheadAttention` のパラメータ本体（`nn::Linear` 4 層：q/k/v/out
/// projection）。`nn::Linear`／`nn::rnn` と同じ「パラメータ本体（本
/// 構造体）とテープ上の `Var` を保持する `*Vars`（[`MultiheadAttentionVars`]）」
/// の分離方針を踏襲する。
pub struct MultiheadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    embed_dim: usize,
    num_heads: usize,
}

impl MultiheadAttention {
    /// 決定的シードで q/k/v/out の 4 `Linear` 層を構築する
    /// （`Linear::new` と同じ `U(-1/√E, 1/√E)` 一様初期化。PyTorch
    /// `nn.MultiheadAttention` の既定初期化〈xavier_uniform〉とは異なる
    /// ことに注意——本イシューは数値的な再現ではなく機能を対象とする）。
    /// 単一の呼び出し `seed` から `nn/init.rs` の 4 ソルト
    /// （`ATTN_Q_SEED_SALT`〜`ATTN_OUT_SEED_SALT`）で 4 系統の独立した
    /// `Linear::new` 呼び出しシードを導出する。
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        bias: bool,
        seed: u64,
    ) -> Result<MultiheadAttention, AutodiffError> {
        validate_embed_heads(embed_dim, num_heads)?;
        let q_proj = Linear::new(
            embed_dim,
            embed_dim,
            bias,
            derive_seed(seed, ATTN_Q_SEED_SALT),
        )?;
        let k_proj = Linear::new(
            embed_dim,
            embed_dim,
            bias,
            derive_seed(seed, ATTN_K_SEED_SALT),
        )?;
        let v_proj = Linear::new(
            embed_dim,
            embed_dim,
            bias,
            derive_seed(seed, ATTN_V_SEED_SALT),
        )?;
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
        })
    }

    /// 明示的な 4 層（q/k/v/out projection）から構築する（テスト・
    /// safetensors 等の外部由来パラメータロード経路向けの入口。
    /// `Linear::from_parameters` と同じ位置づけ）。`clippy::
    /// too_many_arguments` を避けるため、8 個の生テンソルではなく
    /// 構築済みの [`Linear`] 4 層を受け取る（呼び出し元は
    /// `Linear::from_parameters(weight, bias)` で個別に構築する）。
    pub fn from_parameters(
        num_heads: usize,
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        out_proj: Linear,
    ) -> Result<MultiheadAttention, AutodiffError> {
        let embed_dim =
            validate_linear_projections(&q_proj, &k_proj, &v_proj, &out_proj, num_heads)?;
        Ok(MultiheadAttention {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            embed_dim,
            num_heads,
        })
    }

    /// `new`／`from_parameters` に渡した埋め込み次元 `E`（q/k/v/out
    /// 4 層とも正方 `[E, E]` であることを構築時に検証済み）。
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

    /// query projection（`nn::Linear`。`[E, E]`）への参照。
    pub fn q_proj(&self) -> &Linear {
        &self.q_proj
    }

    /// key projection（`nn::Linear`。`[E, E]`）への参照。
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
}

impl<'t> MultiheadAttentionVars<'t> {
    /// `LinearVars` 4 個（q/k/v/out projection）から直接構築する。
    /// [`MultiheadAttention::bind`] を経由しない到達経路（facade 横断
    /// parity テスト向け。モジュール doc 参照）のため、`bind` が
    /// 省略していた不変条件の検証（4 層とも正方 `[E, E]`・同一 `E`・
    /// bias 全 `Some`／全 `None`・同一 `Tape`・`E % num_heads == 0`）を
    /// ここで行う（`validate_projection_vars`。非公開のためコードスパン
    /// 表記で参照しリンク化しない）。
    pub fn new(
        num_heads: usize,
        q: LinearVars<'t>,
        k: LinearVars<'t>,
        v: LinearVars<'t>,
        out: LinearVars<'t>,
    ) -> Result<MultiheadAttentionVars<'t>, AutodiffError> {
        let embed_dim = validate_projection_vars(&q, &k, &v, &out, num_heads)?;
        Ok(MultiheadAttentionVars {
            q,
            k,
            v,
            out,
            embed_dim,
            num_heads,
        })
    }

    /// `new` に渡した埋め込み次元 `E`（q/k/v/out 4 層とも正方
    /// `[E, E]` かつ同一 `Tape` であることを構築時に検証済み）。
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

    /// `y = MultiheadAttention(query, key, value)`。`query: [B, L, E]`・
    /// `key`/`value: [B, S, E]` → `[B, L, E]`（モジュール doc「入出力
    /// 契約」参照）。
    ///
    /// 処理順序（`project`／`split_heads`／`sdpa_compose` はいずれも
    /// 非公開のためコードスパン表記で参照しリンク化しない）: ①shape
    /// 検査（rank・`B`／`E`・`key`/`value` の `[B, S, E]` 一致）→ ②q/k/v
    /// projection（`project`。`gemm_out_shape` を内包する `LinearVars::
    /// forward` に委譲）→ ③head 分割（`split_heads`。`reshape([B, Len,
    /// H, Dh])` → `permute([0, 2, 1, 3])`）→ ④attention 本体
    /// （`sdpa_compose`。scale・`attn_mask`／`is_causal`・softmax）→
    /// ⑤head 結合（`permute` →
    /// `contiguous` → `reshape`）→ ⑥out projection。
    ///
    /// # Errors
    ///
    /// `query`／`key`／`value` の rank が 3 でない、バッチ次元 `B` が
    /// 不一致、`query` の最終軸が `embed_dim` と不一致、`key`／`value`
    /// の shape が食い違う、のいずれかで `AutodiffError::Shape` を
    /// 返す。`attn_mask` と `is_causal` の同時指定・`attn_mask` の
    /// broadcast 不能・全 masked 行は `sdpa_compose`（[`AutodiffError::
    /// InvalidArgument`]）へ委譲する。テープ不一致は `check_same_tape`
    /// （`AutodiffError::TapeMismatch`）。
    pub fn forward(
        &self,
        query: &Var<'t>,
        key: &Var<'t>,
        value: &Var<'t>,
        attn_mask: Option<&Tensor<bool>>,
        is_causal: bool,
    ) -> Result<Var<'t>, AutodiffError> {
        query.check_same_tape(&self.q.weight)?;
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

        let e = self.embed_dim;
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

        let h = self.num_heads;
        // `validate_projection_vars`／`validate_embed_heads` が
        // `e % h == 0` かつ `e > 0`／`h > 0` を構築時に検証済みのため、
        // `dh` は常に 1 以上の整数（0 除算・`1/sqrt(0)` は生じない）。
        let dh = e / h;

        let q_proj = project(query, &self.q, b, l, e)?;
        let k_proj = project(key, &self.k, b, s, e)?;
        let v_proj = project(value, &self.v, b, s, e)?;

        let q_heads = split_heads(&q_proj, b, l, h, dh)?;
        let k_heads = split_heads(&k_proj, b, s, h, dh)?;
        let v_heads = split_heads(&v_proj, b, s, h, dh)?;

        let scale = 1.0f32 / (dh as f32).sqrt();
        let attn_out = sdpa_compose(&q_heads, &k_heads, &v_heads, attn_mask, is_causal, scale)?;

        // head 結合: [B, H, L, Dh] -> permute -> [B, L, H, Dh] ->
        // contiguous（permute 後は必ず非 contiguous になるため無条件）
        // -> reshape -> [B, L, E]。
        let merged = attn_out
            .permute(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape(&[b, l, e])?;

        project(&merged, &self.out, b, l, e)
    }
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
fn project<'t>(
    x: &Var<'t>,
    proj: &LinearVars<'t>,
    b: usize,
    len: usize,
    e: usize,
) -> Result<Var<'t>, AutodiffError> {
    let bl = b.checked_mul(len).ok_or_else(|| {
        AutodiffError::InvalidArgument(format!(
            "MultiheadAttention: batch({b}) * seq_len({len}) overflowed usize"
        ))
    })?;
    let x_flat = match x.reshape(&[bl, e]) {
        Ok(flat) => flat,
        Err(AutodiffError::Shape(ShapeError::NonContiguousReshape)) => {
            x.contiguous()?.reshape(&[bl, e])?
        }
        Err(other) => return Err(other),
    };
    let y = proj.forward(&x_flat)?;
    y.reshape(&[b, len, e])
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

/// scaled dot product attention 本体（D1 フォールバック。モジュール
/// doc「sub-issue (a) との関係」参照）。`query`/`key`/`value` は
/// [`split_heads`] 済みの `[B, H, Len, Dh]`（一般には任意 rank≥2 の
/// バッチ形状）を受け取り、`[B, H, L, Dh]` を返す。`scale` は呼び出し元
/// （`MultiheadAttentionVars::forward`）が `1/sqrt(head_dim)` として
/// 確定済みの値を渡す（`head_dim >= 1` を構築時に保証済みのため常に
/// 有限・正）。
fn sdpa_compose<'t>(
    query: &Var<'t>,
    key: &Var<'t>,
    value: &Var<'t>,
    attn_mask: Option<&Tensor<bool>>,
    is_causal: bool,
    scale: f32,
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
    // broadcast 不能は `Var::matmul`（`matmul_out_shape`）が検査済み。
    let scores = q_scaled.matmul(&k_t)?;
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
    // （`key[-2] != value[-2]`）はここで `Var::matmul` が検査する
    // （`split_heads` が q/k/v とも同じ `s` から head 分割するため本
    // モジュール内では実際には発生しない）。
    weights.matmul(value)
}

/// self-attention（`q = k = v = input`・mask なし・非 causal）として
/// `Module::forward` を定義する（モジュール doc「`Module` trait との
/// 関係」参照）。`forward_host`／`as_linear`／`as_relu` はいずれも
/// trait 既定のままオーバーライドしない。
impl Module for MultiheadAttention {
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.bind(tape).forward(input, input, input, None, false)
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
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let err = sdpa_compose(&qv, &kv, &vv, Some(&mask), false, 0.5).unwrap_err();
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

        let out = sdpa_compose(&qv, &kv, &vv, Some(&mask), false, 0.5).unwrap();
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
}
