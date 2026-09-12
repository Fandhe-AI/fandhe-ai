//! RNN／LSTM／GRU セル・Sequence レベル API（イシュー #1647・設計
//! `docs/autodiff-rnn-cell-tape-design.md` 決定 3・4・4a・6・9）。
//!
//! `nn::Linear`（`nn/linear.rs`）と同じ「パラメータ本体（`RnnCell`／
//! `LstmCell`／`GruCell`）とテープ上の `Var` を保持する `*CellVars`」の
//! 分離方針を踏襲する。Sequence レベル（`Rnn`／`Lstm`／`Gru`）は 1 個の
//! セルを T step 分共有して適用する薄いラッパーで、`bind` を 1 回だけ
//! 呼び重みを T step 間で共有することで BPTT を成立させる（決定 2・
//! `backward.rs::accumulate` の fan-in 蓄積に委ねる）。
//!
//! **`Module` trait との関係（決定 4a・9）**: `Rnn`／`Lstm`／`Gru` は
//! `Module` trait を実装するが、`forward(tape, input: &Var)` は常に
//! [`AutodiffError::InvalidArgument`] を返す。理由: `Module::forward`
//! は `input` を `Var` で受け取るため、時系列方向のスライス（決定 4
//! 候補 A: `Tensor` レベルで `Tensor::narrow` する）を行うには `input`
//! を `.value()`／`to_tensor()` で `Tensor` へ detach する経路しかなく、
//! それは前段層への逆伝播をサイレントに断ち切ってしまう（決定 4a (ii)
//! の問題）。学習経路は `x: &Tensor<f32>` を直接受け取る専用メソッド
//! [`Rnn::forward_seq`] 等を使う。`Module::forward_host`（tape 不要・
//! `predict` 経路）は本来の意味で実装する。
//!
//! **`[T,B,H]` 出力の非対称性（決定 4・#1598 依存）**: `forward_seq`
//! （tape 経路）は `Var::stack`（#1598・未実装）が無いため出力を
//! `Vec<Var<'t>>`（per-step。各 `[B,H]`）で返す。`forward_host`（tape
//! 不要）はテープを介さないため、本モジュール内で `Tensor` を直接
//! 連結し `[T,B,H]` を返せる（`stack_host_tensors`）。

use fandhe_ai_tensor_core::{BackendOps, ShapeError, Tensor, matmul_out_shape, require_same_shape};

use crate::error::AutodiffError;
use crate::nn::init::{
    BIAS_HH_SEED_SALT, BIAS_SEED_SALT, WEIGHT_HH_SEED_SALT, WEIGHT_SEED_SALT, derive_seed,
    try_uniform_init,
};
use crate::nn::module::Module;
use crate::tape::Tape;
use crate::var::{CellWeights, GateParams, Var};

/// `build_gate_params` の戻り値 `(weight_ih, weight_hh, bias_ih,
/// bias_hh)`。`clippy::type_complexity` 回避のための命名。
type GateParamTensors = (
    Tensor<f32>,
    Tensor<f32>,
    Option<Tensor<f32>>,
    Option<Tensor<f32>>,
);

/// `input_size`（`D`）・`hidden_size`（`H`）・`gates`（ゲート数。RNN=1・
/// GRU=3・LSTM=4）から `weight_ih: [D, G*H]`・`weight_hh: [H, G*H]`・
/// `bias_ih`／`bias_hh`（`Some` なら各 `[G*H]`）を構築する共通ヘルパー
/// （`RnnCell`／`LstmCell`／`GruCell::new` が共有する）。
///
/// 初期化範囲は PyTorch `nn.RNN`／`nn.LSTM`／`nn.GRU` の既定
/// （`U(-1/√H, 1/√H)`。`Linear` の `1/√in_features` とは異なり
/// `hidden_size` 基準）に整合させる（設計 doc 決定 5・9）。4 系統の
/// シード導出は `nn/init.rs` の 4 ソルト（`WEIGHT_SEED_SALT`・
/// `WEIGHT_HH_SEED_SALT`・`BIAS_SEED_SALT`・`BIAS_HH_SEED_SALT`）で
/// 互いに独立させる。
fn build_gate_params(
    input_size: usize,
    hidden_size: usize,
    gates: usize,
    bias: bool,
    seed: u64,
) -> Result<GateParamTensors, AutodiffError> {
    if input_size == 0 || hidden_size == 0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "input_size (D={input_size}) and hidden_size (H={hidden_size}) must both be > 0 \
             (1/sqrt(hidden_size) would be non-finite when H=0)"
        )));
    }
    let bound = 1.0 / (hidden_size as f32).sqrt();
    // 本番経路 panic 禁止（AGENTS.md）: `gates * hidden_size`・
    // `input_size * gh`・`hidden_size * gh` は公開コンストラクタから
    // ユーザー入力（`hidden_size`／`input_size`）をそのまま乗じるため、
    // 未検査のまま乗算すると大きな入力で overflow しうる（イシュー
    // #1647 codex-review P1 指摘）。`checked_mul` で検証し失敗を
    // 型付きエラーへ変換する。
    let gh = gates.checked_mul(hidden_size).ok_or_else(|| {
        AutodiffError::InvalidArgument(format!(
            "gates (={gates}) * hidden_size (={hidden_size}) overflowed usize"
        ))
    })?;
    let w_ih_len = input_size.checked_mul(gh).ok_or_else(|| {
        AutodiffError::InvalidArgument(format!(
            "input_size (={input_size}) * gates*hidden_size (={gh}) overflowed usize"
        ))
    })?;
    let w_hh_len = hidden_size.checked_mul(gh).ok_or_else(|| {
        AutodiffError::InvalidArgument(format!(
            "hidden_size (={hidden_size}) * gates*hidden_size (={gh}) overflowed usize"
        ))
    })?;

    let w_ih_seed = derive_seed(seed, WEIGHT_SEED_SALT);
    let weight_ih = Tensor::new(
        checked_uniform_init(w_ih_len, bound, w_ih_seed, "weight_ih")?,
        &[input_size, gh],
    )?;
    let w_hh_seed = derive_seed(seed, WEIGHT_HH_SEED_SALT);
    let weight_hh = Tensor::new(
        checked_uniform_init(w_hh_len, bound, w_hh_seed, "weight_hh")?,
        &[hidden_size, gh],
    )?;

    let (bias_ih, bias_hh) = if bias {
        let b_ih_seed = derive_seed(seed, BIAS_SEED_SALT);
        let b_hh_seed = derive_seed(seed, BIAS_HH_SEED_SALT);
        (
            Some(Tensor::new(
                checked_uniform_init(gh, bound, b_ih_seed, "bias_ih")?,
                &[gh],
            )?),
            Some(Tensor::new(
                checked_uniform_init(gh, bound, b_hh_seed, "bias_hh")?,
                &[gh],
            )?),
        )
    } else {
        (None, None)
    };

    Ok((weight_ih, weight_hh, bias_ih, bias_hh))
}

/// `try_uniform_init` の `Err`（`TryReserveError`）を
/// [`AutodiffError::InvalidArgument`] へ変換する `build_gate_params`
/// 共通ヘルパー。`field_name` はエラーメッセージにどのパラメータ
/// （`weight_ih`／`weight_hh`／`bias_ih`／`bias_hh`）の確保に失敗したか
/// を残すためのラベル（イシュー #1647 codex-review P1 指摘）。
fn checked_uniform_init(
    len: usize,
    bound: f32,
    seed: u64,
    field_name: &str,
) -> Result<Vec<f32>, AutodiffError> {
    try_uniform_init(len, bound, seed).map_err(|err| {
        AutodiffError::InvalidArgument(format!(
            "{field_name}: len={len} 要素分のバッファを確保できません: {err}"
        ))
    })
}

/// `gates * hidden`（ゲート幅）を `checked_mul` で検証する共通実装。
///
/// 本番経路 panic 禁止（AGENTS.md）: `validate_gate_params`／
/// `validate_cell_host_shapes` はいずれも入力由来の `hidden`（実体の
/// ある `Tensor` の shape 次元だが、対をなす軸の要素数が 0 の空
/// テンソルであれば任意の大きさを取りうる。例:
/// `weight_hh.shape() = [1usize << 62, 0]` は要素数 0 のまま合法）を
/// 使って `gates * hidden` を計算する。未検証のまま乗算すると
/// overflow して期待幅が小さい値へ周回し、本来 shape mismatch で
/// 拒否すべき不正な `LstmCell`／`GruCell` パラメータを誤って受理して
/// しまう（イシュー #1647 codex-review P1 指摘）。
fn checked_gate_width(gates: usize, hidden: usize) -> Result<usize, AutodiffError> {
    gates.checked_mul(hidden).ok_or_else(|| {
        AutodiffError::InvalidArgument(format!(
            "gates (={gates}) * hidden (={hidden}) overflowed usize"
        ))
    })
}

/// `from_parameters`（外部由来パラメータの入口。REQ-7 系 safetensors
/// ロード等を見据える）が計算前に行う shape 検証の共通実装（A03。
/// `nn::Linear::from_parameters` と同じ「壊れた shape を計算前に拒否
/// する」方針）。`gates` はゲート数（RNN=1・GRU=3・LSTM=4）。
fn validate_gate_params(
    weight_ih: &Tensor<f32>,
    weight_hh: &Tensor<f32>,
    bias_ih: Option<&Tensor<f32>>,
    bias_hh: Option<&Tensor<f32>>,
    gates: usize,
) -> Result<(), AutodiffError> {
    if weight_ih.rank() != 2 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: weight_ih.rank(),
        }));
    }
    if weight_hh.rank() != 2 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: weight_hh.rank(),
        }));
    }
    let d = weight_ih.shape()[0];
    let hidden = weight_hh.shape()[0];
    if d == 0 || hidden == 0 {
        return Err(AutodiffError::InvalidArgument(
            "weight_ih.shape()[0] (input_size) and weight_hh.shape()[0] (hidden_size) must both \
             be > 0"
                .to_string(),
        ));
    }
    let gh_ih = weight_ih.shape()[1];
    let gh_hh = weight_hh.shape()[1];
    let expected_gh = checked_gate_width(gates, hidden)?;
    if gh_ih != expected_gh || gh_hh != expected_gh {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: vec![d, gh_ih],
            rhs: vec![hidden, gh_hh],
        }));
    }
    if bias_ih.is_some() != bias_hh.is_some() {
        return Err(AutodiffError::InvalidArgument(
            "bias_ih and bias_hh must both be Some or both None".to_string(),
        ));
    }
    for b in [bias_ih, bias_hh].into_iter().flatten() {
        if b.shape() != [expected_gh] {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: b.shape().to_vec(),
                rhs: vec![expected_gh],
            }));
        }
    }
    Ok(())
}

/// `nn::rnn` 内の Sequence レベル `forward_host`（tape 不要経路）が
/// `[T,B,H]` を組み立てるための水平連結（`Var::stack`〈#1598〉が tape
/// 経路にまだ無いため、`Tensor` を直接扱えるこの経路限定で用意する）。
/// 各要素は `[B, H]`（`stack` 対象の shape が全て一致することを前提に
/// 呼び出し元が保証する）。
fn stack_host_tensors(
    steps: &[Tensor<f32>],
    b_dim: usize,
    hidden: usize,
) -> Result<Tensor<f32>, AutodiffError> {
    let mut data = Vec::with_capacity(steps.len() * b_dim * hidden);
    for step in steps {
        let c = step.contiguous();
        data.extend_from_slice(c.as_slice().unwrap_or(&[]));
    }
    // `steps` は本モジュール内部でのみ組み立てられる（各 `[B,H]` の
    // 演算結果）ため通常は要素数が一致するが、本番経路 panic 禁止
    // 方針（`.claude/rules/coding-rust.md`）に従い `.expect()`／
    // `debug_assert!(false)` の全ゼロフォールバックは使わず
    // `Tensor::new` の失敗をそのまま呼び出し元へ伝播する（イシュー
    // #1647 codex-review P1 指摘: 失敗を隠す全ゼロ出力は契約不整合を
    // 検出不能にする）。
    Ok(Tensor::new(data, &[steps.len(), b_dim, hidden])?)
}

/// `x: [T,B,D]` から step `t` の `[B,D]` をゼロコピー優先で切り出す
/// （`Tensor::narrow` の view を `contiguous()` してから `reshape`。
/// 非 contiguous な `x` を渡された場合も `contiguous()` が実体化して
/// 吸収するため、呼び出し元は `x` の contiguity を意識しなくてよい）。
fn slice_timestep(
    x: &Tensor<f32>,
    t: usize,
    b_dim: usize,
    d_dim: usize,
) -> Result<Tensor<f32>, AutodiffError> {
    let sliced = x.narrow(0, t, 1)?.contiguous();
    Ok(sliced.reshape(&[b_dim, d_dim])?)
}

/// `x: [T,B,D]` の rank・`T>0` 検査（`Rnn`／`Lstm`／`Gru` の
/// `forward_seq`／`forward_host` 共通の入口検査）。
fn validate_seq_input(
    x: &Tensor<f32>,
    op_name: &str,
) -> Result<(usize, usize, usize), AutodiffError> {
    let shape = x.shape();
    if shape.len() != 3 {
        return Err(AutodiffError::InvalidArgument(format!(
            "{op_name}: x must be rank-3 [T, B, D] (got rank {})",
            shape.len()
        )));
    }
    let (t_len, b_dim, d_dim) = (shape[0], shape[1], shape[2]);
    if t_len == 0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "{op_name}: T (sequence length) must be > 0"
        )));
    }
    Ok((t_len, b_dim, d_dim))
}

/// `outputs` バッファ（per-step の `Var`／`Tensor` を溜める `Vec`）を
/// `t_len` 要素分確保する。`validate_seq_input` は shape 由来の
/// `t_len` が `usize::MAX` 近辺でも通す（他の次元が 0 なら要素数 0 の
/// 有効なテンソルが構築できるため）。`Vec::with_capacity(t_len)` を
/// そのまま呼ぶと確保不能な `t_len` で本番経路が capacity overflow
/// panic する（`.claude/rules/coding-rust.md` 本番経路 panic 禁止。
/// イシュー #1647 codex-review P1 指摘）。`try_reserve_exact` で
/// 確保可否を先に確認し、失敗時は panic させず
/// [`AutodiffError::InvalidArgument`] へ変換して呼び出し元へ返す。
fn reserve_outputs<T>(t_len: usize, op_name: &str) -> Result<Vec<T>, AutodiffError> {
    let mut outputs = Vec::new();
    outputs.try_reserve_exact(t_len).map_err(|err| {
        AutodiffError::InvalidArgument(format!(
            "{op_name}: T={t_len} 分の出力バッファを確保できません: {err}"
        ))
    })?;
    Ok(outputs)
}

/// セル 1 step の入力形状を検証する（`Var::{rnn_cell,lstm_cell,
/// gru_cell}`〈`var.rs`〉が tape 経路で行う検証と同型の rank・batch
/// 数・hidden 次元・重み shape・bias shape チェック。`forward_host`
/// （tape 不要・`Module::forward_host`／`predict` 経路。決定 9）は
/// `Var` を経由せず `crate::var::{rnn_cell_forward_value,
/// lstm_cell_forward_values, gru_cell_forward_values}` を直接呼ぶ
/// ため、Var 経路が入口で行う検証を通らない（イシュー #1647
/// codex-review P1 指摘: 不正形状〈例 LSTM で `c_prev` のバッチ数が
/// `x`／`h_prev` と食い違う〉を素通しすると `eval::lstm_pointwise`
/// 等が範囲外参照して panic しうる。本番経路 panic 禁止。
/// `.claude/rules/coding-rust.md`）。`gates` はゲート数
/// （RNN=1・GRU=3・LSTM=4）。戻り値は `hidden`（`h_shape[1]`）。
#[allow(clippy::too_many_arguments)]
fn validate_cell_host_shapes(
    x_shape: &[usize],
    h_shape: &[usize],
    w_ih_shape: &[usize],
    w_hh_shape: &[usize],
    b_ih_shape: Option<&[usize]>,
    b_hh_shape: Option<&[usize]>,
    gates: usize,
    op_name: &str,
) -> Result<usize, AutodiffError> {
    for shape in [x_shape, h_shape, w_ih_shape, w_hh_shape] {
        if shape.len() != 2 {
            return Err(AutodiffError::InvalidArgument(format!(
                "{op_name}: all operands must be rank-2 (got shape {shape:?})"
            )));
        }
    }
    let hidden = h_shape[1];
    if x_shape[1] == 0 || hidden == 0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "{op_name}: input_size (D={}) and hidden_size (H={hidden}) must both be > 0",
            x_shape[1]
        )));
    }
    // バッチ数（軸 0）の不一致は matmul 自体は成立しうる（`x`／
    // `h_prev` の行数は matmul の非縮約軸のため独立）が、後続の
    // pointwise 段（`gates`・`c_prev`・`h_prev` を同一 `[B,H]` 前提で
    // 要素ごとに読む）が破綻するため、ここで明示的に検査する。
    if x_shape[0] != h_shape[0] {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: x_shape.to_vec(),
            rhs: h_shape.to_vec(),
        }));
    }
    let out_ih = matmul_out_shape(x_shape, w_ih_shape)
        .map_err(|_| AutodiffError::InvalidArgument(format!("{op_name}: x * w_ih が不整合")))?;
    let out_hh = matmul_out_shape(h_shape, w_hh_shape).map_err(|_| {
        AutodiffError::InvalidArgument(format!("{op_name}: h_prev * w_hh が不整合"))
    })?;
    require_same_shape(&out_ih, &out_hh)?;
    let expected_width = checked_gate_width(gates, hidden)?;
    if out_ih[1] != expected_width {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: out_ih,
            rhs: vec![h_shape[0], expected_width],
        }));
    }
    if let Some(b) = b_ih_shape {
        require_same_shape(b, &[expected_width])?;
    }
    if let Some(b) = b_hh_shape {
        require_same_shape(b, &[expected_width])?;
    }
    Ok(hidden)
}

/// `Module::forward` を明示的に無効化するための共通エラー（決定 4a
/// 項目 3）。
fn forward_not_supported(type_name: &str, seq_method: &str) -> AutodiffError {
    AutodiffError::InvalidArgument(format!(
        "{type_name}::forward: Module::forward (Var-based) is not supported for sequence \
         layers because it would silently detach the input from the tape; use \
         {type_name}::{seq_method} (tape) or Module::forward_host (tape-free) instead"
    ))
}

// =====================================================================
// RNN（tanh 版）
// =====================================================================

/// RNN（tanh 版。PyTorch `nn.RNNCell` 相当）セル 1 step のパラメータ
/// 本体。`weight_ih: [D, H]`・`weight_hh: [H, H]`・`bias_ih`／
/// `bias_hh`: 各 `[H]`（`Some` の場合。両方 `Some` か両方 `None`）。
#[derive(Debug)]
pub struct RnnCell {
    weight_ih: Tensor<f32>,
    weight_hh: Tensor<f32>,
    bias_ih: Option<Tensor<f32>>,
    bias_hh: Option<Tensor<f32>>,
}

impl RnnCell {
    /// 決定的シードで PyTorch `nn.RNNCell` 既定と同じ `U(-1/√H, 1/√H)`
    /// 初期化を行う。`input_size == 0`／`hidden_size == 0` は
    /// [`AutodiffError::InvalidArgument`]（zero-K ガード）。
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        bias: bool,
        seed: u64,
    ) -> Result<Self, AutodiffError> {
        let (weight_ih, weight_hh, bias_ih, bias_hh) =
            build_gate_params(input_size, hidden_size, 1, bias, seed)?;
        Ok(Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        })
    }

    /// 明示的な重み・bias から構築する（A03: 計算前に shape を検証する）。
    pub fn from_parameters(
        weight_ih: Tensor<f32>,
        weight_hh: Tensor<f32>,
        bias_ih: Option<Tensor<f32>>,
        bias_hh: Option<Tensor<f32>>,
    ) -> Result<Self, AutodiffError> {
        validate_gate_params(
            &weight_ih,
            &weight_hh,
            bias_ih.as_ref(),
            bias_hh.as_ref(),
            1,
        )?;
        Ok(Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        })
    }

    pub fn input_size(&self) -> usize {
        self.weight_ih.shape()[0]
    }

    pub fn hidden_size(&self) -> usize {
        self.weight_hh.shape()[0]
    }

    pub fn weight_ih(&self) -> &Tensor<f32> {
        &self.weight_ih
    }

    pub fn weight_hh(&self) -> &Tensor<f32> {
        &self.weight_hh
    }

    pub fn bias_ih(&self) -> Option<&Tensor<f32>> {
        self.bias_ih.as_ref()
    }

    pub fn bias_hh(&self) -> Option<&Tensor<f32>> {
        self.bias_hh.as_ref()
    }

    /// このステップの `tape` へ重み・bias を葉ノードとして登録する
    /// （`Linear::bind` と同じ per-step 再登録契約。決定 3）。
    pub fn bind<'t>(&self, tape: &'t Tape) -> RnnCellVars<'t> {
        RnnCellVars {
            weight_ih: tape.var(&self.weight_ih),
            weight_hh: tape.var(&self.weight_hh),
            bias_ih: self.bias_ih.as_ref().map(|b| tape.var(b)),
            bias_hh: self.bias_hh.as_ref().map(|b| tape.var(b)),
        }
    }

    /// tape 不要（ホスト常駐 `Tensor`）の forward 値計算。`var.rs` の
    /// 共有関数 `crate::var::rnn_cell_forward_value` へ委譲するため、
    /// `Var::rnn_cell`（tape 経路）と bit-exact に一致する（決定 9）。
    pub fn forward_host(
        &self,
        ops: &dyn BackendOps,
        x: &Tensor<f32>,
        h_prev: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        validate_cell_host_shapes(
            x.shape(),
            h_prev.shape(),
            self.weight_ih.shape(),
            self.weight_hh.shape(),
            self.bias_ih.as_ref().map(|b| b.shape()),
            self.bias_hh.as_ref().map(|b| b.shape()),
            1,
            "RnnCell::forward_host",
        )?;
        crate::var::rnn_cell_forward_value(
            ops,
            x,
            h_prev,
            &CellWeights {
                w_ih: &self.weight_ih,
                w_hh: &self.weight_hh,
                b_ih: self.bias_ih.as_ref(),
                b_hh: self.bias_hh.as_ref(),
            },
        )
    }
}

/// [`RnnCell::bind`] が返す、1 ステップ分のテープに登録済みパラメータ。
#[derive(Debug)]
pub struct RnnCellVars<'t> {
    pub weight_ih: Var<'t>,
    pub weight_hh: Var<'t>,
    pub bias_ih: Option<Var<'t>>,
    pub bias_hh: Option<Var<'t>>,
}

impl<'t> RnnCellVars<'t> {
    /// `h_t = tanh(x·W_ih + b_ih + h_prev·W_hh + b_hh)`
    /// （[`Var::rnn_cell`] へ委譲）。
    pub fn forward(&self, x: &Var<'t>, h_prev: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        x.rnn_cell(
            h_prev,
            GateParams {
                w_ih: &self.weight_ih,
                w_hh: &self.weight_hh,
                b_ih: self.bias_ih.as_ref(),
                b_hh: self.bias_hh.as_ref(),
            },
        )
    }
}

/// [`Rnn::forward_seq`]／[`Gru::forward_seq`] の戻り値。`outputs` は
/// 各 step の隠れ状態（`[B,H]`。決定 4「`Var::stack`〈#1598〉未実装の
/// ため per-step の `Vec` で返す」）、`h_n` は最終 step の隠れ状態
/// （`outputs` の最後の要素と同一）。
///
/// `params` は `forward_seq` が内部で `cell.bind(tape)` した、この
/// 呼び出しで実際に使われたテープ登録済みパラメータ（[`RnnCellVars`]
/// または [`GruCellVars`]。型パラメータ `P` で両セルに共用する）。
/// `forward_seq` がこれを返さず内部変数に閉じ込めていると、呼び出し
/// 元は `Gradients::get(&Var)` の引数となる `Var` を得る手段がなく
/// この計算で使われた重み・bias の勾配を取得できない（`forward_seq`
/// 呼び出しの前後に別途 `cell.bind(tape)` しても、それは別ノードとして
/// 登録される新しい葉であり `forward_seq` 内部の計算とは無関係な
/// ため `Gradients::get` は必ず `None` を返す）。これでは系列 API が
/// 学習経路として使い物にならない（イシュー #1647 codex-review P1
/// 指摘）。呼び出し元は `out.params.weight_ih` 等を
/// `grads.get(&out.params.weight_ih)` へ渡してパラメータ更新に使う。
#[derive(Debug)]
pub struct RnnSeqOutput<'t, P> {
    pub outputs: Vec<Var<'t>>,
    pub h_n: Var<'t>,
    pub params: P,
}

/// RNN（tanh 版）の時系列 Sequence レベル API（決定 2「展開
/// unrolled」）。1 個の [`RnnCell`] を T step 分共有して適用する。
#[derive(Debug)]
pub struct Rnn {
    cell: RnnCell,
}

impl Rnn {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        bias: bool,
        seed: u64,
    ) -> Result<Self, AutodiffError> {
        Ok(Self {
            cell: RnnCell::new(input_size, hidden_size, bias, seed)?,
        })
    }

    pub fn from_cell(cell: RnnCell) -> Self {
        Self { cell }
    }

    pub fn cell(&self) -> &RnnCell {
        &self.cell
    }

    /// 学習経路（tape 経路。決定 2・3・4）。`x: [T,B,D]` を `Tensor`
    /// レベルでスライスし、`bind` を **1 回**呼んで重みを T step 間で
    /// 共有する（BPTT は `backward.rs::accumulate` の fan-in 蓄積に
    /// 委ねる）。`h0` 省略時はゼロ（`[B,H]`）を葉登録する（決定 6）。
    pub fn forward_seq<'t>(
        &self,
        tape: &'t Tape,
        x: &Tensor<f32>,
        h0: Option<&Var<'t>>,
    ) -> Result<RnnSeqOutput<'t, RnnCellVars<'t>>, AutodiffError> {
        let (t_len, b_dim, d_dim) = validate_seq_input(x, "Rnn::forward_seq")?;
        let hidden = self.cell.hidden_size();
        let vars = self.cell.bind(tape);

        let mut h = match h0 {
            Some(v) => *v,
            None => tape.var(&Tensor::zeros(&[b_dim, hidden])?),
        };
        let mut outputs = reserve_outputs(t_len, "Rnn::forward_seq")?;
        for t in 0..t_len {
            let x_t_tensor = slice_timestep(x, t, b_dim, d_dim)?;
            let x_t = tape.var(&x_t_tensor);
            h = vars.forward(&x_t, &h)?;
            outputs.push(h);
        }
        Ok(RnnSeqOutput {
            outputs,
            h_n: h,
            params: vars,
        })
    }
}

impl Module for Rnn {
    fn forward<'t>(&self, _tape: &'t Tape, _input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Err(forward_not_supported("Rnn", "forward_seq"))
    }

    /// 推論経路（tape 不要。決定 9）。`x: [T,B,D]` → `[T,B,H]`。
    /// [`RnnCell::forward_host`] を T step 分逐次呼び、`h0` はゼロ固定
    /// （`forward_seq` の `h0` 引数は tape 経路限定。決定 6 のスコープは
    /// 学習経路のみで、推論経路のゼロ初期化は `Module::forward_host`
    /// の既存契約〈引数を追加しない〉と整合させる）。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let (t_len, b_dim, d_dim) = validate_seq_input(input, "Rnn::forward_host")?;
        let hidden = self.cell.hidden_size();
        let mut h = Tensor::zeros(&[b_dim, hidden])?;
        let mut outputs = reserve_outputs(t_len, "Rnn::forward_host")?;
        for t in 0..t_len {
            let x_t = slice_timestep(input, t, b_dim, d_dim)?;
            h = self.cell.forward_host(ops, &x_t, &h)?;
            outputs.push(h.clone());
        }
        stack_host_tensors(&outputs, b_dim, hidden)
    }
}

// =====================================================================
// LSTM
// =====================================================================

/// LSTM セル 1 step のパラメータ本体（決定 5・12。ゲート順 `i,f,g,o`）。
/// `weight_ih: [D, 4H]`・`weight_hh: [H, 4H]`・`bias_ih`／`bias_hh`:
/// 各 `[4H]`。
#[derive(Debug)]
pub struct LstmCell {
    weight_ih: Tensor<f32>,
    weight_hh: Tensor<f32>,
    bias_ih: Option<Tensor<f32>>,
    bias_hh: Option<Tensor<f32>>,
}

impl LstmCell {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        bias: bool,
        seed: u64,
    ) -> Result<Self, AutodiffError> {
        let (weight_ih, weight_hh, bias_ih, bias_hh) =
            build_gate_params(input_size, hidden_size, 4, bias, seed)?;
        Ok(Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        })
    }

    pub fn from_parameters(
        weight_ih: Tensor<f32>,
        weight_hh: Tensor<f32>,
        bias_ih: Option<Tensor<f32>>,
        bias_hh: Option<Tensor<f32>>,
    ) -> Result<Self, AutodiffError> {
        validate_gate_params(
            &weight_ih,
            &weight_hh,
            bias_ih.as_ref(),
            bias_hh.as_ref(),
            4,
        )?;
        Ok(Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        })
    }

    pub fn input_size(&self) -> usize {
        self.weight_ih.shape()[0]
    }

    pub fn hidden_size(&self) -> usize {
        self.weight_hh.shape()[0]
    }

    pub fn weight_ih(&self) -> &Tensor<f32> {
        &self.weight_ih
    }

    pub fn weight_hh(&self) -> &Tensor<f32> {
        &self.weight_hh
    }

    pub fn bias_ih(&self) -> Option<&Tensor<f32>> {
        self.bias_ih.as_ref()
    }

    pub fn bias_hh(&self) -> Option<&Tensor<f32>> {
        self.bias_hh.as_ref()
    }

    pub fn bind<'t>(&self, tape: &'t Tape) -> LstmCellVars<'t> {
        LstmCellVars {
            weight_ih: tape.var(&self.weight_ih),
            weight_hh: tape.var(&self.weight_hh),
            bias_ih: self.bias_ih.as_ref().map(|b| tape.var(b)),
            bias_hh: self.bias_hh.as_ref().map(|b| tape.var(b)),
        }
    }

    /// tape 不要の forward 値計算。`var.rs` の共有関数
    /// `crate::var::lstm_cell_forward_values` へ委譲する（決定 9）。
    /// 戻り値 `(h_t, c_t)`。
    pub fn forward_host(
        &self,
        ops: &dyn BackendOps,
        x: &Tensor<f32>,
        h_prev: &Tensor<f32>,
        c_prev: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>), AutodiffError> {
        validate_cell_host_shapes(
            x.shape(),
            h_prev.shape(),
            self.weight_ih.shape(),
            self.weight_hh.shape(),
            self.bias_ih.as_ref().map(|b| b.shape()),
            self.bias_hh.as_ref().map(|b| b.shape()),
            4,
            "LstmCell::forward_host",
        )?;
        // LSTM は `c_prev` を pointwise 段で `h_prev` と同じ `[B,H]`
        // 前提で要素ごとに読む（`eval::lstm_pointwise`）ため、matmul
        // 自体には現れない `c_prev` の shape 一致も明示的に検査する
        // （イシュー #1647 codex-review P1 指摘）。
        require_same_shape(h_prev.shape(), c_prev.shape())?;
        let out = crate::var::lstm_cell_forward_values(
            ops,
            x,
            h_prev,
            c_prev,
            &CellWeights {
                w_ih: &self.weight_ih,
                w_hh: &self.weight_hh,
                b_ih: self.bias_ih.as_ref(),
                b_hh: self.bias_hh.as_ref(),
            },
        )?;
        Ok((out.h, out.c))
    }
}

#[derive(Debug)]
pub struct LstmCellVars<'t> {
    pub weight_ih: Var<'t>,
    pub weight_hh: Var<'t>,
    pub bias_ih: Option<Var<'t>>,
    pub bias_hh: Option<Var<'t>>,
}

impl<'t> LstmCellVars<'t> {
    /// [`Var::lstm_cell`] へ委譲する。戻り値 `(h_t, c_t)`。
    pub fn forward(
        &self,
        x: &Var<'t>,
        h_prev: &Var<'t>,
        c_prev: &Var<'t>,
    ) -> Result<(Var<'t>, Var<'t>), AutodiffError> {
        x.lstm_cell(
            h_prev,
            c_prev,
            GateParams {
                w_ih: &self.weight_ih,
                w_hh: &self.weight_hh,
                b_ih: self.bias_ih.as_ref(),
                b_hh: self.bias_hh.as_ref(),
            },
        )
    }
}

/// [`Lstm::forward_seq`] の戻り値（決定 1b: LSTM は再帰出力 `h_t`／
/// `c_t` の 2 系統を持つため `RnnSeqOutput` とは別型）。`params` は
/// この呼び出しが内部で `cell.bind(tape)` した、実際に使われた
/// テープ登録済みパラメータ（[`RnnSeqOutput`] の doc comment 参照。
/// イシュー #1647 codex-review P1 指摘）。
#[derive(Debug)]
pub struct LstmSeqOutput<'t> {
    pub outputs: Vec<Var<'t>>,
    pub h_n: Var<'t>,
    pub c_n: Var<'t>,
    pub params: LstmCellVars<'t>,
}

/// LSTM の時系列 Sequence レベル API。
#[derive(Debug)]
pub struct Lstm {
    cell: LstmCell,
}

impl Lstm {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        bias: bool,
        seed: u64,
    ) -> Result<Self, AutodiffError> {
        Ok(Self {
            cell: LstmCell::new(input_size, hidden_size, bias, seed)?,
        })
    }

    pub fn from_cell(cell: LstmCell) -> Self {
        Self { cell }
    }

    pub fn cell(&self) -> &LstmCell {
        &self.cell
    }

    /// 学習経路。`h0`／`c0` 省略時はいずれもゼロを葉登録する（決定 6）。
    pub fn forward_seq<'t>(
        &self,
        tape: &'t Tape,
        x: &Tensor<f32>,
        h0: Option<&Var<'t>>,
        c0: Option<&Var<'t>>,
    ) -> Result<LstmSeqOutput<'t>, AutodiffError> {
        let (t_len, b_dim, d_dim) = validate_seq_input(x, "Lstm::forward_seq")?;
        let hidden = self.cell.hidden_size();
        let vars = self.cell.bind(tape);

        let mut h = match h0 {
            Some(v) => *v,
            None => tape.var(&Tensor::zeros(&[b_dim, hidden])?),
        };
        let mut c = match c0 {
            Some(v) => *v,
            None => tape.var(&Tensor::zeros(&[b_dim, hidden])?),
        };
        let mut outputs = reserve_outputs(t_len, "Lstm::forward_seq")?;
        for t in 0..t_len {
            let x_t_tensor = slice_timestep(x, t, b_dim, d_dim)?;
            let x_t = tape.var(&x_t_tensor);
            let (h_t, c_t) = vars.forward(&x_t, &h, &c)?;
            h = h_t;
            c = c_t;
            outputs.push(h);
        }
        Ok(LstmSeqOutput {
            outputs,
            h_n: h,
            c_n: c,
            params: vars,
        })
    }
}

impl Module for Lstm {
    fn forward<'t>(&self, _tape: &'t Tape, _input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Err(forward_not_supported("Lstm", "forward_seq"))
    }

    /// `x: [T,B,D]` → `[T,B,H]`（最終隠れ状態列。`c_n` は tape 不要
    /// 経路では返さない——決定 9 は推論経路の対象を隠れ状態出力のみと
    /// する）。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let (t_len, b_dim, d_dim) = validate_seq_input(input, "Lstm::forward_host")?;
        let hidden = self.cell.hidden_size();
        let mut h = Tensor::zeros(&[b_dim, hidden])?;
        let mut c = Tensor::zeros(&[b_dim, hidden])?;
        let mut outputs = reserve_outputs(t_len, "Lstm::forward_host")?;
        for t in 0..t_len {
            let x_t = slice_timestep(input, t, b_dim, d_dim)?;
            let (h_t, c_t) = self.cell.forward_host(ops, &x_t, &h, &c)?;
            h = h_t;
            c = c_t;
            outputs.push(h.clone());
        }
        stack_host_tensors(&outputs, b_dim, hidden)
    }
}

// =====================================================================
// GRU
// =====================================================================

/// GRU セル 1 step のパラメータ本体（決定 5・12。`reset_after=True`
/// 規約・ゲート順 `r,z,n`）。`weight_ih: [D, 3H]`・`weight_hh: [H, 3H]`・
/// `bias_ih`／`bias_hh`: 各 `[3H]`。
#[derive(Debug)]
pub struct GruCell {
    weight_ih: Tensor<f32>,
    weight_hh: Tensor<f32>,
    bias_ih: Option<Tensor<f32>>,
    bias_hh: Option<Tensor<f32>>,
}

impl GruCell {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        bias: bool,
        seed: u64,
    ) -> Result<Self, AutodiffError> {
        let (weight_ih, weight_hh, bias_ih, bias_hh) =
            build_gate_params(input_size, hidden_size, 3, bias, seed)?;
        Ok(Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        })
    }

    pub fn from_parameters(
        weight_ih: Tensor<f32>,
        weight_hh: Tensor<f32>,
        bias_ih: Option<Tensor<f32>>,
        bias_hh: Option<Tensor<f32>>,
    ) -> Result<Self, AutodiffError> {
        validate_gate_params(
            &weight_ih,
            &weight_hh,
            bias_ih.as_ref(),
            bias_hh.as_ref(),
            3,
        )?;
        Ok(Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        })
    }

    pub fn input_size(&self) -> usize {
        self.weight_ih.shape()[0]
    }

    pub fn hidden_size(&self) -> usize {
        self.weight_hh.shape()[0]
    }

    pub fn weight_ih(&self) -> &Tensor<f32> {
        &self.weight_ih
    }

    pub fn weight_hh(&self) -> &Tensor<f32> {
        &self.weight_hh
    }

    pub fn bias_ih(&self) -> Option<&Tensor<f32>> {
        self.bias_ih.as_ref()
    }

    pub fn bias_hh(&self) -> Option<&Tensor<f32>> {
        self.bias_hh.as_ref()
    }

    pub fn bind<'t>(&self, tape: &'t Tape) -> GruCellVars<'t> {
        GruCellVars {
            weight_ih: tape.var(&self.weight_ih),
            weight_hh: tape.var(&self.weight_hh),
            bias_ih: self.bias_ih.as_ref().map(|b| tape.var(b)),
            bias_hh: self.bias_hh.as_ref().map(|b| tape.var(b)),
        }
    }

    /// tape 不要の forward 値計算。`var.rs` の共有関数
    /// `crate::var::gru_cell_forward_values` へ委譲する（決定 9）。
    pub fn forward_host(
        &self,
        ops: &dyn BackendOps,
        x: &Tensor<f32>,
        h_prev: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        validate_cell_host_shapes(
            x.shape(),
            h_prev.shape(),
            self.weight_ih.shape(),
            self.weight_hh.shape(),
            self.bias_ih.as_ref().map(|b| b.shape()),
            self.bias_hh.as_ref().map(|b| b.shape()),
            3,
            "GruCell::forward_host",
        )?;
        let out = crate::var::gru_cell_forward_values(
            ops,
            x,
            h_prev,
            &CellWeights {
                w_ih: &self.weight_ih,
                w_hh: &self.weight_hh,
                b_ih: self.bias_ih.as_ref(),
                b_hh: self.bias_hh.as_ref(),
            },
        )?;
        Ok(out.h)
    }
}

#[derive(Debug)]
pub struct GruCellVars<'t> {
    pub weight_ih: Var<'t>,
    pub weight_hh: Var<'t>,
    pub bias_ih: Option<Var<'t>>,
    pub bias_hh: Option<Var<'t>>,
}

impl<'t> GruCellVars<'t> {
    /// [`Var::gru_cell`] へ委譲する。
    pub fn forward(&self, x: &Var<'t>, h_prev: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        x.gru_cell(
            h_prev,
            GateParams {
                w_ih: &self.weight_ih,
                w_hh: &self.weight_hh,
                b_ih: self.bias_ih.as_ref(),
                b_hh: self.bias_hh.as_ref(),
            },
        )
    }
}

/// GRU の時系列 Sequence レベル API。
#[derive(Debug)]
pub struct Gru {
    cell: GruCell,
}

impl Gru {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        bias: bool,
        seed: u64,
    ) -> Result<Self, AutodiffError> {
        Ok(Self {
            cell: GruCell::new(input_size, hidden_size, bias, seed)?,
        })
    }

    pub fn from_cell(cell: GruCell) -> Self {
        Self { cell }
    }

    pub fn cell(&self) -> &GruCell {
        &self.cell
    }

    /// 学習経路。`h0` 省略時はゼロを葉登録する（決定 6）。
    pub fn forward_seq<'t>(
        &self,
        tape: &'t Tape,
        x: &Tensor<f32>,
        h0: Option<&Var<'t>>,
    ) -> Result<RnnSeqOutput<'t, GruCellVars<'t>>, AutodiffError> {
        let (t_len, b_dim, d_dim) = validate_seq_input(x, "Gru::forward_seq")?;
        let hidden = self.cell.hidden_size();
        let vars = self.cell.bind(tape);

        let mut h = match h0 {
            Some(v) => *v,
            None => tape.var(&Tensor::zeros(&[b_dim, hidden])?),
        };
        let mut outputs = reserve_outputs(t_len, "Gru::forward_seq")?;
        for t in 0..t_len {
            let x_t_tensor = slice_timestep(x, t, b_dim, d_dim)?;
            let x_t = tape.var(&x_t_tensor);
            h = vars.forward(&x_t, &h)?;
            outputs.push(h);
        }
        Ok(RnnSeqOutput {
            outputs,
            h_n: h,
            params: vars,
        })
    }
}

impl Module for Gru {
    fn forward<'t>(&self, _tape: &'t Tape, _input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Err(forward_not_supported("Gru", "forward_seq"))
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let (t_len, b_dim, d_dim) = validate_seq_input(input, "Gru::forward_host")?;
        let hidden = self.cell.hidden_size();
        let mut h = Tensor::zeros(&[b_dim, hidden])?;
        let mut outputs = reserve_outputs(t_len, "Gru::forward_host")?;
        for t in 0..t_len {
            let x_t = slice_timestep(input, t, b_dim, d_dim)?;
            h = self.cell.forward_host(ops, &x_t, &h)?;
            outputs.push(h.clone());
        }
        stack_host_tensors(&outputs, b_dim, hidden)
    }
}
