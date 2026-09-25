//! RNN／LSTM／GRU の多層・双方向・層間 dropout（`RnnConfig`。イシュー
//! #2164・親 #2131）。
//!
//! `nn::rnn`（`rnn.rs`）が提供する単層・単方向の `Rnn`／`Lstm`／`Gru`
//! （イシュー #1955 で facade 公開済み）を土台に、PyTorch
//! `nn.RNN`／`nn.LSTM`／`nn.GRU` の `num_layers`・`bidirectional`・
//! `dropout`（層間）に相当する多層・双方向・層間 dropout を追加する。
//!
//! # なぜ `Rnn`／`Lstm`／`Gru` に `with_config` を足さないか（重要）
//!
//! `crates/facade/src/nn/rnn.rs` は既存 8 型（`Rnn`／`Lstm`／`Gru` を
//! 含む）を純再エクスポートしている（イシュー #1955）。既存型へ
//! inherent メソッド（例 `Rnn::with_config`）を追加すると、facade の
//! 公開面が引数型を facade から名指しできなくても `Default::default()`
//! の型推論経由で**自動的に**広がってしまう。本 issue は facade への
//! `RnnConfig` 公開を未承認事項として保留する（`docs/compat-api-scope.md`
//! §5 経路 2。`crates/facade/src/lib.rs::RnnConfigHoldDoctestGuard`
//! 参照）ため、多層化は既存型に触れず新しい型（[`StackedRnn`]／
//! [`StackedLstm`]／[`StackedGru`]）で提供する。`rnn.rs` の公開
//! シグネチャ・`RnnCell`／`LstmCell`／`GruCell`・`Rnn`／`Lstm`／`Gru`
//! 自体は一切変更しない。
//!
//! # Sequence レベルのスタックと決定 4a の関係
//!
//! `docs/autodiff-rnn-cell-tape-design.md` 決定 4a は「Sequence レベル
//! API（`Rnn::forward_seq` 等）同士を重ねると `Tensor` レベルの入力
//! スライスで前段への逆伝播が切れる」と整理し、セル単位で per-step に
//! 交互適用すれば勾配が連続することを示している（決定 4a (i)）。
//! 本モジュールはこの方針に従い、`RnnCell`／`LstmCell`／`GruCell::bind`
//! を各 `(layer, direction)` ごとに 1 回だけ呼び、層 1 以降の入力には
//! 前層の出力 `Var` を `Tensor` へ detach せずそのまま渡す
//! （`forward_seq` 内部で完結するため、利用者が手組みする必要がない）。
//! 双方向（決定 4a の対象外として列挙されていた項目）も本 issue で
//! 内部クレートに実装する。
//!
//! # 命名契約（PyTorch 互換の層・方向インデックス）
//!
//! [`Module::named_parameters`] は `l{layer}.`（forward 方向）／
//! `l{layer}_reverse.`（reverse 方向）を接頭辞として使う
//! （PyTorch `nn.LSTM` の `weight_ih_l0`／`weight_ih_l0_reverse` の
//! `_l{layer}[_reverse]` 部分に対応。ただし本クレートの命名契約
//! 〈`nn/module.rs::named_parameters` doc〉に従い struct フィールド名
//! ベースのため、PyTorch の packed `weight_ih_l0` とは文字列として
//! 一致しない——キー互換は対象外）。`set_parameter` はこの接頭辞を
//! パースして該当セルへ委譲する。

use fandhe_ai_tensor_core::Tensor;

use crate::error::AutodiffError;
use crate::nn::init::{RNN_STACK_SEED_SALT, derive_seed};
use crate::nn::module::Module;
use crate::nn::rnn::{
    GruCell, GruCellVars, LstmCell, LstmCellVars, RnnCell, RnnCellVars, forward_not_supported,
    reserve_outputs, slice_timestep, stack_host_tensors, validate_seq_input,
};
use crate::tape::Tape;
use crate::var::Var;
use fandhe_ai_tensor_core::BackendOps;

/// RNN／LSTM／GRU の多層・双方向・層間 dropout オプション（PyTorch
/// `nn.RNN(num_layers=, bidirectional=, dropout=)` 相当）。
///
/// `#[non_exhaustive]`: 将来 `proj_size`（LSTM）・`batch_first` 等を
/// 追加する余地を残す（実装計画 §6 のスコープ外事項）。
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct RnnConfig {
    num_layers: usize,
    bidirectional: bool,
    dropout: f32,
}

impl Default for RnnConfig {
    /// PyTorch `nn.RNN`／`nn.LSTM`／`nn.GRU` の既定値
    /// （`num_layers=1`・`bidirectional=False`・`dropout=0.0`）。
    fn default() -> Self {
        Self {
            num_layers: 1,
            bidirectional: false,
            dropout: 0.0,
        }
    }
}

impl RnnConfig {
    /// [`Self::default`] と同じ（`RnnConfig::new()` の明示形）。
    pub fn new() -> Self {
        Self::default()
    }

    /// 層数（`>= 1`。[`Self::validate`] が検査する）を設定する。
    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// 双方向にするかどうかを設定する。
    pub fn with_bidirectional(mut self, bidirectional: bool) -> Self {
        self.bidirectional = bidirectional;
        self
    }

    /// 層間 dropout の確率（`[0, 1]` かつ有限。[`Self::validate`] が
    /// 検査する。PyTorch と同じく**最終層の出力には適用されない**）を
    /// 設定する。
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// 層数。
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// 双方向かどうか。
    pub fn bidirectional(&self) -> bool {
        self.bidirectional
    }

    /// 層間 dropout の確率。
    pub fn dropout(&self) -> f32 {
        self.dropout
    }

    /// 方向数（双方向なら `2`・単方向なら `1`）。
    pub fn num_directions(&self) -> usize {
        if self.bidirectional { 2 } else { 1 }
    }

    /// 構築前検査（A03。本番経路 panic 禁止・`.claude/rules/
    /// coding-rust.md`）: `num_layers >= 1`、`dropout` が有限かつ
    /// `[0, 1]`。`num_layers == 1` かつ `dropout > 0` は PyTorch と
    /// 同じく受理する（最終層〈かつ唯一の層〉には dropout を適用
    /// しないため no-op になる。エラーにはしない）。
    pub fn validate(&self) -> Result<(), AutodiffError> {
        if self.num_layers == 0 {
            return Err(AutodiffError::InvalidArgument(
                "RnnConfig::validate: num_layers must be >= 1".to_string(),
            ));
        }
        if !self.dropout.is_finite() || !(0.0..=1.0).contains(&self.dropout) {
            return Err(AutodiffError::InvalidArgument(format!(
                "RnnConfig::validate: dropout must be finite and in [0, 1], got {}",
                self.dropout
            )));
        }
        Ok(())
    }
}

/// `(layer, direction)` の総数を `checked_mul` で検証し、[`RnnConfig`]
/// の妥当性検査と合わせて呼び出し元コンストラクタが共有する共通入口
/// （A03。本番経路 panic 禁止）。`hidden_size` を forward 方向の入力幅
/// （層 1 以降）として返す（`num_directions * hidden_size`。
/// `checked_mul` で overflow を検査する）。
fn validate_stack_config(
    config: &RnnConfig,
    hidden_size: usize,
) -> Result<(usize, usize), AutodiffError> {
    config.validate()?;
    let num_directions = config.num_directions();
    let total_cells = config
        .num_layers
        .checked_mul(num_directions)
        .ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "num_layers (={}) * num_directions (={num_directions}) overflowed usize",
                config.num_layers
            ))
        })?;
    let stacked_input_size = num_directions.checked_mul(hidden_size).ok_or_else(|| {
        AutodiffError::InvalidArgument(format!(
            "num_directions (={num_directions}) * hidden_size (={hidden_size}) overflowed usize"
        ))
    })?;
    Ok((total_cells, stacked_input_size))
}

/// `index = layer * num_directions + direction` を `checked_mul`／
/// `checked_add` で検証する共通ヘルパー（`validate_stack_config` が
/// 求めた `total_cells` の範囲内であることは呼び出し元ループが保証
/// するため、本関数は算術 overflow のみを検査する）。
fn cell_index(
    layer: usize,
    direction: usize,
    num_directions: usize,
) -> Result<usize, AutodiffError> {
    layer
        .checked_mul(num_directions)
        .and_then(|v| v.checked_add(direction))
        .ok_or_else(|| {
            AutodiffError::InvalidArgument(format!(
                "layer (={layer}) * num_directions (={num_directions}) + direction \
                 (={direction}) overflowed usize"
            ))
        })
}

/// index `k`（[`cell_index`] 参照）のセル構築シードを導出する
/// （`nn/init.rs::RNN_STACK_SEED_SALT` doc 参照）。`k == 0`（layer 0・
/// forward 方向）は `seed` をそのまま返す（既定 config での bit 一致
/// 契約）。
fn derive_cell_seed(seed: u64, k: usize) -> u64 {
    if k == 0 {
        seed
    } else {
        derive_seed(derive_seed(seed, RNN_STACK_SEED_SALT), k as u64)
    }
}

/// `h0`／`c0`（`Option<&[Var<'t>]>`）の長さ検証（`L*dirs` と一致する
/// こと）の共通実装。
fn validate_state_len(
    state: Option<&[Var<'_>]>,
    expected: usize,
    label: &str,
    op_name: &str,
) -> Result<(), AutodiffError> {
    if let Some(s) = state
        && s.len() != expected
    {
        return Err(AutodiffError::InvalidArgument(format!(
            "{op_name}: {label} has length {} but expected {expected} (num_layers * \
             num_directions)",
            s.len()
        )));
    }
    Ok(())
}

/// 層間 dropout（`layer < num_layers - 1` の出力にのみ適用。PyTorch
/// 同様、最終層の出力には適用しない）の tape 経路実装。RNG 消費順は
/// 「layer 昇順 → t 昇順」（呼び出し元ループの反復順そのまま）。
fn apply_interlayer_dropout<'t>(
    out: Var<'t>,
    layer: usize,
    num_layers: usize,
    dropout: f32,
    training: bool,
) -> Result<Var<'t>, AutodiffError> {
    if layer + 1 < num_layers {
        out.dropout(dropout, training)
    } else {
        Ok(out)
    }
}

/// [`StackedRnn`]／[`StackedGru`]・[`StackedLstm`] の `forward_seq` が
/// 返す隠れ状態出力（RNN／GRU 共用。`RnnSeqOutput`〈`rnn.rs`〉と同型の
/// 設計だが、多層・双方向のため `h_n`／`params` が `Vec`）。
///
/// - `outputs`: step ごとの出力（各 `[B, num_directions*H]`。双方向は
///   `Var::cat` で forward／reverse を結合済み）。
/// - `h_n`: 各 `(layer, direction)` の最終隠れ状態（長さ
///   `num_layers*num_directions`。並びは PyTorch の `h_n` と同じ
///   `index = layer*num_directions + direction`）。reverse 方向は
///   t=0 処理後の状態。
/// - `params`: この呼び出しで実際に `cell.bind(tape)` した、テープ
///   登録済みパラメータ（`RnnSeqOutput` doc と同じ理由で必須。
///   `grads.get(&out.params[k].weight_ih)` 等で使う）。
#[derive(Debug)]
#[non_exhaustive]
pub struct StackedRnnSeqOutput<'t, P> {
    pub outputs: Vec<Var<'t>>,
    pub h_n: Vec<Var<'t>>,
    pub params: Vec<P>,
}

/// [`StackedLstm`] の `forward_seq` が返す隠れ状態・セル状態出力
/// （[`StackedRnnSeqOutput`] に `c_n` を加えた LSTM 専用版。
/// `LstmSeqOutput`〈`rnn.rs`〉と同じ理由で別型にする）。
#[derive(Debug)]
#[non_exhaustive]
pub struct StackedLstmSeqOutput<'t> {
    pub outputs: Vec<Var<'t>>,
    pub h_n: Vec<Var<'t>>,
    pub c_n: Vec<Var<'t>>,
    pub params: Vec<LstmCellVars<'t>>,
}

/// `named_parameters` の接頭辞（`l{layer}` または `l{layer}_reverse`）
/// を構築する共通ヘルパー（3 型で共有）。
fn layer_prefix(layer: usize, direction: usize) -> String {
    if direction == 0 {
        format!("l{layer}")
    } else {
        format!("l{layer}_reverse")
    }
}

/// `named_parameters` の接頭辞をパースして `(layer, direction, rest)`
/// を返す（[`layer_prefix`] の逆演算。`set_parameter`／`cell` 委譲の
/// 共通実装）。`"l{layer}"`／`"l{layer}_reverse"` のいずれでもない、
/// または `layer` が `usize` として parse できない場合は `None`
/// （fail-closed。呼び出し元は未知名として扱う）。
fn parse_layer_prefix(name: &str, num_directions: usize) -> Option<(usize, usize, &str)> {
    let (head, rest) = name.split_once('.')?;
    let head = head.strip_prefix('l')?;
    let (layer_str, direction) = if num_directions > 1 {
        match head.strip_suffix("_reverse") {
            Some(l) => (l, 1),
            None => (head, 0),
        }
    } else {
        (head, 0)
    };
    let layer: usize = layer_str.parse().ok()?;
    Some((layer, direction, rest))
}

/// `RnnConfig::default()` で `Rnn::new` と bit 一致する多層・双方向
/// RNN（tanh 版）。モジュール doc「なぜ `Rnn` に `with_config` を
/// 足さないか」を参照。
#[derive(Debug)]
pub struct StackedRnn {
    /// `index = layer * config.num_directions() + direction`
    /// （PyTorch の `h_n` と同順）。
    cells: Vec<RnnCell>,
    config: RnnConfig,
    input_size: usize,
    hidden_size: usize,
    training: bool,
}

impl StackedRnn {
    /// `input_size`・`hidden_size`・`bias`・`seed`・`config` から構築
    /// する。エラー条件: [`RnnConfig::validate`] の失敗、
    /// `num_layers*num_directions`／`num_directions*hidden_size` の
    /// overflow、`cells` 確保失敗（`try_reserve_exact`）、個々の
    /// `RnnCell::new` の失敗（`input_size==0`／`hidden_size==0`）。
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        bias: bool,
        seed: u64,
        config: RnnConfig,
    ) -> Result<Self, AutodiffError> {
        let (total_cells, stacked_input_size) = validate_stack_config(&config, hidden_size)?;
        let num_directions = config.num_directions();
        let mut cells = Vec::new();
        cells.try_reserve_exact(total_cells).map_err(|err| {
            AutodiffError::InvalidArgument(format!(
                "StackedRnn::new: {total_cells} 個分のセルを確保できません: {err}"
            ))
        })?;
        for layer in 0..config.num_layers {
            let layer_input = if layer == 0 {
                input_size
            } else {
                stacked_input_size
            };
            for direction in 0..num_directions {
                let k = cell_index(layer, direction, num_directions)?;
                let cell_seed = derive_cell_seed(seed, k);
                cells.push(RnnCell::new(layer_input, hidden_size, bias, cell_seed)?);
            }
        }
        Ok(Self {
            cells,
            config,
            input_size,
            hidden_size,
            training: true,
        })
    }

    /// 構築時の [`RnnConfig`]。
    pub fn config(&self) -> &RnnConfig {
        &self.config
    }

    /// 層 0 の入力幅 `D`。
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// 隠れ状態幅 `H`。
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// `(layer, direction)` のセルへの参照。範囲外は `None`。
    pub fn cell(&self, layer: usize, direction: usize) -> Option<&RnnCell> {
        let k = cell_index(layer, direction, self.config.num_directions()).ok()?;
        self.cells.get(k)
    }

    /// 学習経路。`x: [T,B,D]`。`h0` は `Some` の場合、長さ
    /// `num_layers*num_directions`（`index = layer*num_directions +
    /// direction`）でなければならない。`None` はゼロ（`[B,H]`）を
    /// index ごとに葉登録する（`rnn.rs::Rnn::forward_seq` 決定 6 と
    /// 同じ）。
    pub fn forward_seq<'t>(
        &self,
        tape: &'t Tape,
        x: &Tensor<f32>,
        h0: Option<&[Var<'t>]>,
    ) -> Result<StackedRnnSeqOutput<'t, RnnCellVars<'t>>, AutodiffError> {
        let (t_len, b_dim, d_dim) =
            validate_seq_input(x, self.input_size, "StackedRnn::forward_seq")?;
        let num_directions = self.config.num_directions();
        let total_cells = self.cells.len();
        validate_state_len(h0, total_cells, "h0", "StackedRnn::forward_seq")?;

        // 層 0 の入力（forward・reverse 双方で共有する葉）。
        let mut layer_in: Vec<Var<'t>> = {
            let mut v = reserve_outputs(t_len, "StackedRnn::forward_seq")?;
            for t in 0..t_len {
                let x_t = slice_timestep(x, t, b_dim, d_dim)?;
                v.push(tape.var(&x_t));
            }
            v
        };

        let mut h_n = reserve_outputs(total_cells, "StackedRnn::forward_seq")?;
        h_n.resize(
            total_cells,
            tape.var(&Tensor::zeros(&[b_dim, self.hidden_size])?),
        );
        let mut params =
            reserve_outputs::<RnnCellVars<'t>>(total_cells, "StackedRnn::forward_seq")?;
        // `RnnCellVars` は `bind` の戻り値であり `Default` を持たない
        // ため、`h_n` のような `resize` 埋めではなく最終的に index 順で
        // 詰め替える（各 index を一度だけ埋める）。
        let mut params_slots: Vec<Option<RnnCellVars<'t>>> =
            (0..total_cells).map(|_| None).collect();

        for layer in 0..self.config.num_layers {
            let mut dir_outputs: Vec<Vec<Var<'t>>> = Vec::with_capacity(num_directions);
            for direction in 0..num_directions {
                let k = cell_index(layer, direction, num_directions)?;
                let vars = self.cells[k].bind(tape);
                let mut h = match h0 {
                    Some(s) => s[k],
                    None => tape.var(&Tensor::zeros(&[b_dim, self.hidden_size])?),
                };
                let mut step_outputs = reserve_outputs(t_len, "StackedRnn::forward_seq")?;
                let time_order: Box<dyn Iterator<Item = usize>> = if direction == 0 {
                    Box::new(0..t_len)
                } else {
                    Box::new((0..t_len).rev())
                };
                let mut ordered: Vec<(usize, Var<'t>)> = Vec::with_capacity(t_len);
                for t in time_order {
                    h = vars.forward(&layer_in[t], &h)?;
                    ordered.push((t, h));
                }
                ordered.sort_by_key(|(t, _)| *t);
                step_outputs.extend(ordered.into_iter().map(|(_, v)| v));
                h_n[k] = h;
                params_slots[k] = Some(vars);
                dir_outputs.push(step_outputs);
            }

            let mut layer_out = reserve_outputs(t_len, "StackedRnn::forward_seq")?;
            if num_directions == 1 {
                layer_out.extend(dir_outputs.into_iter().next().unwrap_or_default());
            } else {
                let fwd = &dir_outputs[0];
                let rev = &dir_outputs[1];
                for t in 0..t_len {
                    layer_out.push(Var::cat(&[fwd[t], rev[t]], 1)?);
                }
            }

            let mut dropped = reserve_outputs(t_len, "StackedRnn::forward_seq")?;
            for out in layer_out {
                dropped.push(apply_interlayer_dropout(
                    out,
                    layer,
                    self.config.num_layers,
                    self.config.dropout,
                    self.training,
                )?);
            }
            layer_in = dropped;
        }

        for slot in params_slots {
            match slot {
                Some(v) => params.push(v),
                None => {
                    return Err(AutodiffError::InvalidArgument(
                        "StackedRnn::forward_seq: internal error: not all cells were bound"
                            .to_string(),
                    ));
                }
            }
        }

        Ok(StackedRnnSeqOutput {
            outputs: layer_in,
            h_n,
            params,
        })
    }
}

impl Module for StackedRnn {
    fn forward<'t>(&self, _tape: &'t Tape, _input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Err(forward_not_supported("StackedRnn", "forward_seq"))
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn training(&self) -> bool {
        self.training
    }

    /// 命名契約: `l{layer}.`（forward）／`l{layer}_reverse.`（reverse）
    /// を接頭辞に、セル内は `weight_ih` → `weight_hh` → `bias_ih`?
    /// → `bias_hh`? の順（モジュール doc「命名契約」参照）。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let num_directions = self.config.num_directions();
        let mut out = Vec::new();
        for layer in 0..self.config.num_layers {
            for direction in 0..num_directions {
                let Ok(k) = cell_index(layer, direction, num_directions) else {
                    continue;
                };
                let Some(cell) = self.cells.get(k) else {
                    continue;
                };
                let prefix = layer_prefix(layer, direction);
                out.push((format!("{prefix}.weight_ih"), cell.weight_ih()));
                out.push((format!("{prefix}.weight_hh"), cell.weight_hh()));
                if let Some(b) = cell.bias_ih() {
                    out.push((format!("{prefix}.bias_ih"), b));
                }
                if let Some(b) = cell.bias_hh() {
                    out.push((format!("{prefix}.bias_hh"), b));
                }
            }
        }
        out
    }

    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        let num_directions = self.config.num_directions();
        let (layer, direction, rest) =
            parse_layer_prefix(name, num_directions).ok_or_else(|| {
                AutodiffError::InvalidArgument(format!(
                    "StackedRnn::set_parameter: no parameter named `{name}`"
                ))
            })?;
        let k = cell_index(layer, direction, num_directions)?;
        match self.cells.get_mut(k) {
            Some(cell) => cell.set_parameter(rest, value),
            None => Err(AutodiffError::InvalidArgument(format!(
                "StackedRnn::set_parameter: no parameter named `{name}` (layer {layer} \
                 direction {direction} out of range)"
            ))),
        }
    }

    fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
        for cell in &mut self.cells {
            cell.set_requires_grad(requires_grad);
        }
        Ok(())
    }

    fn requires_grad(&self) -> bool {
        self.cells.iter().all(|c| c.requires_grad())
    }

    /// 推論経路（tape 不要）。`h0` はゼロ固定（`rnn.rs` の既存契約と
    /// 同じ）。層間 dropout は `self.training && p > 0` のときのみ
    /// 適用する。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let (t_len, b_dim, d_dim) =
            validate_seq_input(input, self.input_size, "StackedRnn::forward_host")?;
        let num_directions = self.config.num_directions();

        let mut layer_in: Vec<Tensor<f32>> = reserve_outputs(t_len, "StackedRnn::forward_host")?;
        for t in 0..t_len {
            layer_in.push(slice_timestep(input, t, b_dim, d_dim)?);
        }

        for layer in 0..self.config.num_layers {
            let mut dir_outputs: Vec<Vec<Tensor<f32>>> = Vec::with_capacity(num_directions);
            for direction in 0..num_directions {
                let k = cell_index(layer, direction, num_directions)?;
                let cell = self.cells.get(k).ok_or_else(|| {
                    AutodiffError::InvalidArgument(
                        "StackedRnn::forward_host: internal error: missing cell".to_string(),
                    )
                })?;
                let mut h = Tensor::zeros(&[b_dim, self.hidden_size])?;
                let mut ordered: Vec<(usize, Tensor<f32>)> = Vec::with_capacity(t_len);
                let time_order: Box<dyn Iterator<Item = usize>> = if direction == 0 {
                    Box::new(0..t_len)
                } else {
                    Box::new((0..t_len).rev())
                };
                for t in time_order {
                    h = cell.forward_host(ops, &layer_in[t], &h)?;
                    ordered.push((t, h.clone()));
                }
                ordered.sort_by_key(|(t, _)| *t);
                dir_outputs.push(ordered.into_iter().map(|(_, v)| v).collect());
            }

            let mut layer_out: Vec<Tensor<f32>> =
                reserve_outputs(t_len, "StackedRnn::forward_host")?;
            if num_directions == 1 {
                layer_out.extend(dir_outputs.into_iter().next().unwrap_or_default());
            } else {
                let fwd = &dir_outputs[0];
                let rev = &dir_outputs[1];
                for t in 0..t_len {
                    let inputs: Vec<&Tensor<f32>> = vec![&fwd[t], &rev[t]];
                    let out_shape = fandhe_ai_tensor_core::concat_out_shape(
                        &[fwd[t].shape(), rev[t].shape()],
                        1,
                    )
                    .map_err(AutodiffError::Shape)?;
                    layer_out.push(crate::grad::concat_with_fallback(
                        ops, &inputs, 1, &out_shape,
                    )?);
                }
            }

            let apply_dropout =
                self.training && self.config.dropout > 0.0 && layer + 1 < self.config.num_layers;
            let mut dropped: Vec<Tensor<f32>> = reserve_outputs(t_len, "StackedRnn::forward_host")?;
            for out in layer_out {
                if apply_dropout {
                    let mask = crate::grad::dropout_mask(out.shape(), self.config.dropout)?;
                    dropped.push(crate::grad::dropout_with_fallback(ops, &out, &mask)?);
                } else {
                    dropped.push(out);
                }
            }
            layer_in = dropped;
        }

        stack_host_tensors(&layer_in, b_dim, num_directions * self.hidden_size)
    }
}

/// `RnnConfig::default()` で `Gru::new` と bit 一致する多層・双方向
/// GRU。実装形は [`StackedRnn`] と同型（状態が `h` のみのため。単一
/// state セルの共通部分をコメント付きで重複させる方針は実装計画
/// §3.6「内部の重複削減（任意）」の判断どおり見送った）。
#[derive(Debug)]
pub struct StackedGru {
    cells: Vec<GruCell>,
    config: RnnConfig,
    input_size: usize,
    hidden_size: usize,
    training: bool,
}

impl StackedGru {
    /// [`StackedRnn::new`] と同じ契約（セル型が `GruCell` になる点のみ
    /// 異なる）。
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        bias: bool,
        seed: u64,
        config: RnnConfig,
    ) -> Result<Self, AutodiffError> {
        let (total_cells, stacked_input_size) = validate_stack_config(&config, hidden_size)?;
        let num_directions = config.num_directions();
        let mut cells = Vec::new();
        cells.try_reserve_exact(total_cells).map_err(|err| {
            AutodiffError::InvalidArgument(format!(
                "StackedGru::new: {total_cells} 個分のセルを確保できません: {err}"
            ))
        })?;
        for layer in 0..config.num_layers {
            let layer_input = if layer == 0 {
                input_size
            } else {
                stacked_input_size
            };
            for direction in 0..num_directions {
                let k = cell_index(layer, direction, num_directions)?;
                let cell_seed = derive_cell_seed(seed, k);
                cells.push(GruCell::new(layer_input, hidden_size, bias, cell_seed)?);
            }
        }
        Ok(Self {
            cells,
            config,
            input_size,
            hidden_size,
            training: true,
        })
    }

    pub fn config(&self) -> &RnnConfig {
        &self.config
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn cell(&self, layer: usize, direction: usize) -> Option<&GruCell> {
        let k = cell_index(layer, direction, self.config.num_directions()).ok()?;
        self.cells.get(k)
    }

    /// [`StackedRnn::forward_seq`] と同じ契約（セル型が `GruCell` に
    /// なる点のみ異なる）。
    pub fn forward_seq<'t>(
        &self,
        tape: &'t Tape,
        x: &Tensor<f32>,
        h0: Option<&[Var<'t>]>,
    ) -> Result<StackedRnnSeqOutput<'t, GruCellVars<'t>>, AutodiffError> {
        let (t_len, b_dim, d_dim) =
            validate_seq_input(x, self.input_size, "StackedGru::forward_seq")?;
        let num_directions = self.config.num_directions();
        let total_cells = self.cells.len();
        validate_state_len(h0, total_cells, "h0", "StackedGru::forward_seq")?;

        let mut layer_in: Vec<Var<'t>> = {
            let mut v = reserve_outputs(t_len, "StackedGru::forward_seq")?;
            for t in 0..t_len {
                let x_t = slice_timestep(x, t, b_dim, d_dim)?;
                v.push(tape.var(&x_t));
            }
            v
        };

        let mut h_n = reserve_outputs(total_cells, "StackedGru::forward_seq")?;
        h_n.resize(
            total_cells,
            tape.var(&Tensor::zeros(&[b_dim, self.hidden_size])?),
        );
        let mut params =
            reserve_outputs::<GruCellVars<'t>>(total_cells, "StackedGru::forward_seq")?;
        let mut params_slots: Vec<Option<GruCellVars<'t>>> =
            (0..total_cells).map(|_| None).collect();

        for layer in 0..self.config.num_layers {
            let mut dir_outputs: Vec<Vec<Var<'t>>> = Vec::with_capacity(num_directions);
            for direction in 0..num_directions {
                let k = cell_index(layer, direction, num_directions)?;
                let vars = self.cells[k].bind(tape);
                let mut h = match h0 {
                    Some(s) => s[k],
                    None => tape.var(&Tensor::zeros(&[b_dim, self.hidden_size])?),
                };
                let time_order: Box<dyn Iterator<Item = usize>> = if direction == 0 {
                    Box::new(0..t_len)
                } else {
                    Box::new((0..t_len).rev())
                };
                let mut ordered: Vec<(usize, Var<'t>)> = Vec::with_capacity(t_len);
                for t in time_order {
                    h = vars.forward(&layer_in[t], &h)?;
                    ordered.push((t, h));
                }
                ordered.sort_by_key(|(t, _)| *t);
                let step_outputs: Vec<Var<'t>> = ordered.into_iter().map(|(_, v)| v).collect();
                h_n[k] = h;
                params_slots[k] = Some(vars);
                dir_outputs.push(step_outputs);
            }

            let mut layer_out = reserve_outputs(t_len, "StackedGru::forward_seq")?;
            if num_directions == 1 {
                layer_out.extend(dir_outputs.into_iter().next().unwrap_or_default());
            } else {
                let fwd = &dir_outputs[0];
                let rev = &dir_outputs[1];
                for t in 0..t_len {
                    layer_out.push(Var::cat(&[fwd[t], rev[t]], 1)?);
                }
            }

            let mut dropped = reserve_outputs(t_len, "StackedGru::forward_seq")?;
            for out in layer_out {
                dropped.push(apply_interlayer_dropout(
                    out,
                    layer,
                    self.config.num_layers,
                    self.config.dropout,
                    self.training,
                )?);
            }
            layer_in = dropped;
        }

        for slot in params_slots {
            match slot {
                Some(v) => params.push(v),
                None => {
                    return Err(AutodiffError::InvalidArgument(
                        "StackedGru::forward_seq: internal error: not all cells were bound"
                            .to_string(),
                    ));
                }
            }
        }

        Ok(StackedRnnSeqOutput {
            outputs: layer_in,
            h_n,
            params,
        })
    }
}

impl Module for StackedGru {
    fn forward<'t>(&self, _tape: &'t Tape, _input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Err(forward_not_supported("StackedGru", "forward_seq"))
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let num_directions = self.config.num_directions();
        let mut out = Vec::new();
        for layer in 0..self.config.num_layers {
            for direction in 0..num_directions {
                let Ok(k) = cell_index(layer, direction, num_directions) else {
                    continue;
                };
                let Some(cell) = self.cells.get(k) else {
                    continue;
                };
                let prefix = layer_prefix(layer, direction);
                out.push((format!("{prefix}.weight_ih"), cell.weight_ih()));
                out.push((format!("{prefix}.weight_hh"), cell.weight_hh()));
                if let Some(b) = cell.bias_ih() {
                    out.push((format!("{prefix}.bias_ih"), b));
                }
                if let Some(b) = cell.bias_hh() {
                    out.push((format!("{prefix}.bias_hh"), b));
                }
            }
        }
        out
    }

    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        let num_directions = self.config.num_directions();
        let (layer, direction, rest) =
            parse_layer_prefix(name, num_directions).ok_or_else(|| {
                AutodiffError::InvalidArgument(format!(
                    "StackedGru::set_parameter: no parameter named `{name}`"
                ))
            })?;
        let k = cell_index(layer, direction, num_directions)?;
        match self.cells.get_mut(k) {
            Some(cell) => cell.set_parameter(rest, value),
            None => Err(AutodiffError::InvalidArgument(format!(
                "StackedGru::set_parameter: no parameter named `{name}` (layer {layer} \
                 direction {direction} out of range)"
            ))),
        }
    }

    fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
        for cell in &mut self.cells {
            cell.set_requires_grad(requires_grad);
        }
        Ok(())
    }

    fn requires_grad(&self) -> bool {
        self.cells.iter().all(|c| c.requires_grad())
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let (t_len, b_dim, d_dim) =
            validate_seq_input(input, self.input_size, "StackedGru::forward_host")?;
        let num_directions = self.config.num_directions();

        let mut layer_in: Vec<Tensor<f32>> = reserve_outputs(t_len, "StackedGru::forward_host")?;
        for t in 0..t_len {
            layer_in.push(slice_timestep(input, t, b_dim, d_dim)?);
        }

        for layer in 0..self.config.num_layers {
            let mut dir_outputs: Vec<Vec<Tensor<f32>>> = Vec::with_capacity(num_directions);
            for direction in 0..num_directions {
                let k = cell_index(layer, direction, num_directions)?;
                let cell = self.cells.get(k).ok_or_else(|| {
                    AutodiffError::InvalidArgument(
                        "StackedGru::forward_host: internal error: missing cell".to_string(),
                    )
                })?;
                let mut h = Tensor::zeros(&[b_dim, self.hidden_size])?;
                let mut ordered: Vec<(usize, Tensor<f32>)> = Vec::with_capacity(t_len);
                let time_order: Box<dyn Iterator<Item = usize>> = if direction == 0 {
                    Box::new(0..t_len)
                } else {
                    Box::new((0..t_len).rev())
                };
                for t in time_order {
                    h = cell.forward_host(ops, &layer_in[t], &h)?;
                    ordered.push((t, h.clone()));
                }
                ordered.sort_by_key(|(t, _)| *t);
                dir_outputs.push(ordered.into_iter().map(|(_, v)| v).collect());
            }

            let mut layer_out: Vec<Tensor<f32>> =
                reserve_outputs(t_len, "StackedGru::forward_host")?;
            if num_directions == 1 {
                layer_out.extend(dir_outputs.into_iter().next().unwrap_or_default());
            } else {
                let fwd = &dir_outputs[0];
                let rev = &dir_outputs[1];
                for t in 0..t_len {
                    let inputs: Vec<&Tensor<f32>> = vec![&fwd[t], &rev[t]];
                    let out_shape = fandhe_ai_tensor_core::concat_out_shape(
                        &[fwd[t].shape(), rev[t].shape()],
                        1,
                    )
                    .map_err(AutodiffError::Shape)?;
                    layer_out.push(crate::grad::concat_with_fallback(
                        ops, &inputs, 1, &out_shape,
                    )?);
                }
            }

            let apply_dropout =
                self.training && self.config.dropout > 0.0 && layer + 1 < self.config.num_layers;
            let mut dropped: Vec<Tensor<f32>> = reserve_outputs(t_len, "StackedGru::forward_host")?;
            for out in layer_out {
                if apply_dropout {
                    let mask = crate::grad::dropout_mask(out.shape(), self.config.dropout)?;
                    dropped.push(crate::grad::dropout_with_fallback(ops, &out, &mask)?);
                } else {
                    dropped.push(out);
                }
            }
            layer_in = dropped;
        }

        stack_host_tensors(&layer_in, b_dim, num_directions * self.hidden_size)
    }
}

/// `RnnConfig::default()` で `Lstm::new` と bit 一致する多層・双方向
/// LSTM（`(h, c)` の 2 系統の状態を持つため [`StackedRnn`]／
/// [`StackedGru`] とは戻り値型・`h0`／`c0` 引数が異なる）。
#[derive(Debug)]
pub struct StackedLstm {
    cells: Vec<LstmCell>,
    config: RnnConfig,
    input_size: usize,
    hidden_size: usize,
    training: bool,
}

impl StackedLstm {
    /// [`StackedRnn::new`] と同じ契約（セル型が `LstmCell` になる点の
    /// み異なる）。
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        bias: bool,
        seed: u64,
        config: RnnConfig,
    ) -> Result<Self, AutodiffError> {
        let (total_cells, stacked_input_size) = validate_stack_config(&config, hidden_size)?;
        let num_directions = config.num_directions();
        let mut cells = Vec::new();
        cells.try_reserve_exact(total_cells).map_err(|err| {
            AutodiffError::InvalidArgument(format!(
                "StackedLstm::new: {total_cells} 個分のセルを確保できません: {err}"
            ))
        })?;
        for layer in 0..config.num_layers {
            let layer_input = if layer == 0 {
                input_size
            } else {
                stacked_input_size
            };
            for direction in 0..num_directions {
                let k = cell_index(layer, direction, num_directions)?;
                let cell_seed = derive_cell_seed(seed, k);
                cells.push(LstmCell::new(layer_input, hidden_size, bias, cell_seed)?);
            }
        }
        Ok(Self {
            cells,
            config,
            input_size,
            hidden_size,
            training: true,
        })
    }

    pub fn config(&self) -> &RnnConfig {
        &self.config
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn cell(&self, layer: usize, direction: usize) -> Option<&LstmCell> {
        let k = cell_index(layer, direction, self.config.num_directions()).ok()?;
        self.cells.get(k)
    }

    /// 学習経路。`h0`／`c0` はともに `Some` の場合、長さ
    /// `num_layers*num_directions` でなければならない
    /// （[`StackedRnn::forward_seq`] の `h0` と同じ契約が `c0` にも
    /// 適用される）。
    pub fn forward_seq<'t>(
        &self,
        tape: &'t Tape,
        x: &Tensor<f32>,
        h0: Option<&[Var<'t>]>,
        c0: Option<&[Var<'t>]>,
    ) -> Result<StackedLstmSeqOutput<'t>, AutodiffError> {
        let (t_len, b_dim, d_dim) =
            validate_seq_input(x, self.input_size, "StackedLstm::forward_seq")?;
        let num_directions = self.config.num_directions();
        let total_cells = self.cells.len();
        validate_state_len(h0, total_cells, "h0", "StackedLstm::forward_seq")?;
        validate_state_len(c0, total_cells, "c0", "StackedLstm::forward_seq")?;

        let mut layer_in: Vec<Var<'t>> = {
            let mut v = reserve_outputs(t_len, "StackedLstm::forward_seq")?;
            for t in 0..t_len {
                let x_t = slice_timestep(x, t, b_dim, d_dim)?;
                v.push(tape.var(&x_t));
            }
            v
        };

        let mut h_n = reserve_outputs(total_cells, "StackedLstm::forward_seq")?;
        h_n.resize(
            total_cells,
            tape.var(&Tensor::zeros(&[b_dim, self.hidden_size])?),
        );
        let mut c_n = reserve_outputs(total_cells, "StackedLstm::forward_seq")?;
        c_n.resize(
            total_cells,
            tape.var(&Tensor::zeros(&[b_dim, self.hidden_size])?),
        );
        let mut params =
            reserve_outputs::<LstmCellVars<'t>>(total_cells, "StackedLstm::forward_seq")?;
        let mut params_slots: Vec<Option<LstmCellVars<'t>>> =
            (0..total_cells).map(|_| None).collect();

        for layer in 0..self.config.num_layers {
            let mut dir_outputs: Vec<Vec<Var<'t>>> = Vec::with_capacity(num_directions);
            for direction in 0..num_directions {
                let k = cell_index(layer, direction, num_directions)?;
                let vars = self.cells[k].bind(tape);
                let mut h = match h0 {
                    Some(s) => s[k],
                    None => tape.var(&Tensor::zeros(&[b_dim, self.hidden_size])?),
                };
                let mut c = match c0 {
                    Some(s) => s[k],
                    None => tape.var(&Tensor::zeros(&[b_dim, self.hidden_size])?),
                };
                let time_order: Box<dyn Iterator<Item = usize>> = if direction == 0 {
                    Box::new(0..t_len)
                } else {
                    Box::new((0..t_len).rev())
                };
                let mut ordered: Vec<(usize, Var<'t>)> = Vec::with_capacity(t_len);
                for t in time_order {
                    let (h_t, c_t) = vars.forward(&layer_in[t], &h, &c)?;
                    h = h_t;
                    c = c_t;
                    ordered.push((t, h));
                }
                ordered.sort_by_key(|(t, _)| *t);
                let step_outputs: Vec<Var<'t>> = ordered.into_iter().map(|(_, v)| v).collect();
                h_n[k] = h;
                c_n[k] = c;
                params_slots[k] = Some(vars);
                dir_outputs.push(step_outputs);
            }

            let mut layer_out = reserve_outputs(t_len, "StackedLstm::forward_seq")?;
            if num_directions == 1 {
                layer_out.extend(dir_outputs.into_iter().next().unwrap_or_default());
            } else {
                let fwd = &dir_outputs[0];
                let rev = &dir_outputs[1];
                for t in 0..t_len {
                    layer_out.push(Var::cat(&[fwd[t], rev[t]], 1)?);
                }
            }

            let mut dropped = reserve_outputs(t_len, "StackedLstm::forward_seq")?;
            for out in layer_out {
                dropped.push(apply_interlayer_dropout(
                    out,
                    layer,
                    self.config.num_layers,
                    self.config.dropout,
                    self.training,
                )?);
            }
            layer_in = dropped;
        }

        for slot in params_slots {
            match slot {
                Some(v) => params.push(v),
                None => {
                    return Err(AutodiffError::InvalidArgument(
                        "StackedLstm::forward_seq: internal error: not all cells were bound"
                            .to_string(),
                    ));
                }
            }
        }

        Ok(StackedLstmSeqOutput {
            outputs: layer_in,
            h_n,
            c_n,
            params,
        })
    }
}

impl Module for StackedLstm {
    fn forward<'t>(&self, _tape: &'t Tape, _input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Err(forward_not_supported("StackedLstm", "forward_seq"))
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let num_directions = self.config.num_directions();
        let mut out = Vec::new();
        for layer in 0..self.config.num_layers {
            for direction in 0..num_directions {
                let Ok(k) = cell_index(layer, direction, num_directions) else {
                    continue;
                };
                let Some(cell) = self.cells.get(k) else {
                    continue;
                };
                let prefix = layer_prefix(layer, direction);
                out.push((format!("{prefix}.weight_ih"), cell.weight_ih()));
                out.push((format!("{prefix}.weight_hh"), cell.weight_hh()));
                if let Some(b) = cell.bias_ih() {
                    out.push((format!("{prefix}.bias_ih"), b));
                }
                if let Some(b) = cell.bias_hh() {
                    out.push((format!("{prefix}.bias_hh"), b));
                }
            }
        }
        out
    }

    fn set_parameter(&mut self, name: &str, value: Tensor<f32>) -> Result<(), AutodiffError> {
        let num_directions = self.config.num_directions();
        let (layer, direction, rest) =
            parse_layer_prefix(name, num_directions).ok_or_else(|| {
                AutodiffError::InvalidArgument(format!(
                    "StackedLstm::set_parameter: no parameter named `{name}`"
                ))
            })?;
        let k = cell_index(layer, direction, num_directions)?;
        match self.cells.get_mut(k) {
            Some(cell) => cell.set_parameter(rest, value),
            None => Err(AutodiffError::InvalidArgument(format!(
                "StackedLstm::set_parameter: no parameter named `{name}` (layer {layer} \
                 direction {direction} out of range)"
            ))),
        }
    }

    fn set_requires_grad(&mut self, requires_grad: bool) -> Result<(), AutodiffError> {
        for cell in &mut self.cells {
            cell.set_requires_grad(requires_grad);
        }
        Ok(())
    }

    fn requires_grad(&self) -> bool {
        self.cells.iter().all(|c| c.requires_grad())
    }

    /// 推論経路（tape 不要）。`c_n` は `rnn.rs::Lstm::forward_host`
    /// 決定 9 と同じ理由で返さない（隠れ状態出力のみ）。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let (t_len, b_dim, d_dim) =
            validate_seq_input(input, self.input_size, "StackedLstm::forward_host")?;
        let num_directions = self.config.num_directions();

        let mut layer_in: Vec<Tensor<f32>> = reserve_outputs(t_len, "StackedLstm::forward_host")?;
        for t in 0..t_len {
            layer_in.push(slice_timestep(input, t, b_dim, d_dim)?);
        }

        for layer in 0..self.config.num_layers {
            let mut dir_outputs: Vec<Vec<Tensor<f32>>> = Vec::with_capacity(num_directions);
            for direction in 0..num_directions {
                let k = cell_index(layer, direction, num_directions)?;
                let cell = self.cells.get(k).ok_or_else(|| {
                    AutodiffError::InvalidArgument(
                        "StackedLstm::forward_host: internal error: missing cell".to_string(),
                    )
                })?;
                let mut h = Tensor::zeros(&[b_dim, self.hidden_size])?;
                let mut c = Tensor::zeros(&[b_dim, self.hidden_size])?;
                let mut ordered: Vec<(usize, Tensor<f32>)> = Vec::with_capacity(t_len);
                let time_order: Box<dyn Iterator<Item = usize>> = if direction == 0 {
                    Box::new(0..t_len)
                } else {
                    Box::new((0..t_len).rev())
                };
                for t in time_order {
                    let (h_t, c_t) = cell.forward_host(ops, &layer_in[t], &h, &c)?;
                    h = h_t;
                    c = c_t;
                    ordered.push((t, h.clone()));
                }
                ordered.sort_by_key(|(t, _)| *t);
                dir_outputs.push(ordered.into_iter().map(|(_, v)| v).collect());
            }

            let mut layer_out: Vec<Tensor<f32>> =
                reserve_outputs(t_len, "StackedLstm::forward_host")?;
            if num_directions == 1 {
                layer_out.extend(dir_outputs.into_iter().next().unwrap_or_default());
            } else {
                let fwd = &dir_outputs[0];
                let rev = &dir_outputs[1];
                for t in 0..t_len {
                    let inputs: Vec<&Tensor<f32>> = vec![&fwd[t], &rev[t]];
                    let out_shape = fandhe_ai_tensor_core::concat_out_shape(
                        &[fwd[t].shape(), rev[t].shape()],
                        1,
                    )
                    .map_err(AutodiffError::Shape)?;
                    layer_out.push(crate::grad::concat_with_fallback(
                        ops, &inputs, 1, &out_shape,
                    )?);
                }
            }

            let apply_dropout =
                self.training && self.config.dropout > 0.0 && layer + 1 < self.config.num_layers;
            let mut dropped: Vec<Tensor<f32>> =
                reserve_outputs(t_len, "StackedLstm::forward_host")?;
            for out in layer_out {
                if apply_dropout {
                    let mask = crate::grad::dropout_mask(out.shape(), self.config.dropout)?;
                    dropped.push(crate::grad::dropout_with_fallback(ops, &out, &mask)?);
                } else {
                    dropped.push(out);
                }
            }
            layer_in = dropped;
        }

        stack_host_tensors(&layer_in, b_dim, num_directions * self.hidden_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_ops::NaiveOps;
    use crate::nn::rnn::{Gru, Lstm, Rnn};
    use crate::tape::Tape;

    #[test]
    fn rnn_config_default_matches_pytorch_defaults() {
        let c = RnnConfig::default();
        assert_eq!(c.num_layers(), 1);
        assert!(!c.bidirectional());
        assert_eq!(c.dropout(), 0.0);
        assert_eq!(c.num_directions(), 1);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn rnn_config_rejects_zero_layers() {
        let c = RnnConfig::new().with_num_layers(0);
        assert!(c.validate().is_err());
    }

    #[test]
    fn rnn_config_rejects_invalid_dropout() {
        assert!(RnnConfig::new().with_dropout(-0.1).validate().is_err());
        assert!(RnnConfig::new().with_dropout(1.1).validate().is_err());
        assert!(RnnConfig::new().with_dropout(f32::NAN).validate().is_err());
    }

    #[test]
    fn rnn_config_num_layers_one_with_dropout_is_ok_noop() {
        // PyTorch と同じく num_layers=1 かつ dropout>0 はエラーにしない
        // （最終層〈かつ唯一の層〉には適用されないため no-op）。
        let c = RnnConfig::new().with_dropout(0.5);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn stacked_rnn_default_config_bit_matches_rnn() {
        let seed = 42;
        let stacked = StackedRnn::new(3, 4, true, seed, RnnConfig::default()).unwrap();
        let plain = Rnn::new(3, 4, true, seed).unwrap();
        assert_eq!(
            stacked
                .cell(0, 0)
                .unwrap()
                .weight_ih()
                .contiguous()
                .as_slice()
                .unwrap(),
            plain.cell().weight_ih().contiguous().as_slice().unwrap()
        );
        assert_eq!(
            stacked
                .cell(0, 0)
                .unwrap()
                .weight_hh()
                .contiguous()
                .as_slice()
                .unwrap(),
            plain.cell().weight_hh().contiguous().as_slice().unwrap()
        );

        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]).unwrap();
        let out_stacked = stacked.forward_seq(&tape, &x, None).unwrap();
        let out_plain = plain.forward_seq(&tape, &x, None).unwrap();
        for (a, b) in out_stacked.outputs.iter().zip(out_plain.outputs.iter()) {
            let va = a.to_tensor();
            let vb = b.to_tensor();
            assert_eq!(
                va.contiguous().as_slice().unwrap(),
                vb.contiguous().as_slice().unwrap()
            );
        }
    }

    #[test]
    fn stacked_gru_default_config_bit_matches_gru() {
        let seed = 7;
        let stacked = StackedGru::new(2, 3, true, seed, RnnConfig::default()).unwrap();
        let plain = Gru::new(2, 3, true, seed).unwrap();
        assert_eq!(
            stacked
                .cell(0, 0)
                .unwrap()
                .weight_ih()
                .contiguous()
                .as_slice()
                .unwrap(),
            plain.cell().weight_ih().contiguous().as_slice().unwrap()
        );
    }

    #[test]
    fn stacked_lstm_default_config_bit_matches_lstm() {
        let seed = 99;
        let stacked = StackedLstm::new(2, 3, true, seed, RnnConfig::default()).unwrap();
        let plain = Lstm::new(2, 3, true, seed).unwrap();
        assert_eq!(
            stacked
                .cell(0, 0)
                .unwrap()
                .weight_ih()
                .contiguous()
                .as_slice()
                .unwrap(),
            plain.cell().weight_ih().contiguous().as_slice().unwrap()
        );
    }

    #[test]
    fn stacked_rnn_two_layers_gradient_is_continuous_through_layer0() {
        // 層 0 の重み勾配が非ゼロであることを確認する（決定 4a (i):
        // 層 1 以降の入力に前層の出力 `Var` をそのまま渡すため勾配が
        // 連続する）。
        let config = RnnConfig::new().with_num_layers(2);
        let stacked = StackedRnn::new(2, 3, true, 1, config).unwrap();
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = Tensor::new(vec![1.0, 0.5, -1.0, 0.3], &[2, 1, 2]).unwrap();
        let out = stacked.forward_seq(&tape, &x, None).unwrap();
        let loss = out.outputs.last().copied().unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let g0 = grads.get(&out.params[0].weight_ih).unwrap().unwrap();
        assert!(
            g0.contiguous()
                .as_slice()
                .unwrap()
                .iter()
                .any(|v| *v != 0.0)
        );
    }

    #[test]
    fn stacked_rnn_bidirectional_single_layer_shapes_and_order() {
        let config = RnnConfig::new().with_bidirectional(true);
        let stacked = StackedRnn::new(2, 3, true, 1, config).unwrap();
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = Tensor::new(vec![1.0, 0.5, -1.0, 0.3], &[2, 1, 2]).unwrap();
        let out = stacked.forward_seq(&tape, &x, None).unwrap();
        assert_eq!(out.h_n.len(), 2);
        for step in &out.outputs {
            assert_eq!(step.to_tensor().shape(), &[1, 6]);
        }
    }

    #[test]
    fn stacked_rnn_forward_host_matches_forward_seq_eval_mode() {
        let config = RnnConfig::new().with_num_layers(2).with_bidirectional(true);
        let mut stacked = StackedRnn::new(2, 3, true, 5, config).unwrap();
        stacked.set_training(false);
        let ops = NaiveOps;
        let x = Tensor::new(vec![1.0, 0.5, -1.0, 0.3, 0.2, -0.4, 0.1, 0.9], &[2, 2, 2]).unwrap();
        let host_out = stacked.forward_host(&ops, &x).unwrap();

        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let seq_out = stacked.forward_seq(&tape, &x, None).unwrap();
        let mut seq_data = Vec::new();
        for step in &seq_out.outputs {
            seq_data.extend_from_slice(step.to_tensor().contiguous().as_slice().unwrap());
        }
        assert_eq!(
            host_out.contiguous().as_slice().unwrap(),
            seq_data.as_slice()
        );
    }

    #[test]
    fn stacked_rnn_eval_mode_dropout_is_noop_and_does_not_consume_rng() {
        let config = RnnConfig::new().with_num_layers(2).with_dropout(0.9);
        let mut stacked = StackedRnn::new(2, 3, true, 3, config).unwrap();
        stacked.set_training(false);
        let ops = NaiveOps;
        let x = Tensor::new(vec![1.0, 0.5, -1.0, 0.3], &[2, 1, 2]).unwrap();
        let out_a = stacked.forward_host(&ops, &x).unwrap();
        let out_b = stacked.forward_host(&ops, &x).unwrap();
        assert_eq!(
            out_a.contiguous().as_slice().unwrap(),
            out_b.contiguous().as_slice().unwrap()
        );
    }

    #[test]
    fn stacked_rnn_new_rejects_zero_num_layers() {
        let config = RnnConfig::new().with_num_layers(0);
        assert!(StackedRnn::new(2, 3, true, 1, config).is_err());
    }

    #[test]
    fn stacked_rnn_forward_seq_rejects_h0_length_mismatch() {
        let config = RnnConfig::new().with_num_layers(2);
        let stacked = StackedRnn::new(2, 3, true, 1, config).unwrap();
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = Tensor::new(vec![1.0, 0.5, -1.0, 0.3], &[2, 1, 2]).unwrap();
        let bad_h0 = vec![tape.var(&Tensor::zeros(&[1, 3]).unwrap())];
        assert!(stacked.forward_seq(&tape, &x, Some(&bad_h0)).is_err());
    }

    #[test]
    fn stacked_rnn_named_parameters_uses_layer_and_direction_prefix() {
        let config = RnnConfig::new().with_num_layers(2).with_bidirectional(true);
        let stacked = StackedRnn::new(2, 3, true, 1, config).unwrap();
        let names: Vec<String> = stacked
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert!(names.iter().any(|n| n == "l0.weight_ih"));
        assert!(names.iter().any(|n| n == "l0_reverse.weight_ih"));
        assert!(names.iter().any(|n| n == "l1.weight_ih"));
        assert!(names.iter().any(|n| n == "l1_reverse.weight_ih"));
    }

    #[test]
    fn stacked_rnn_set_parameter_round_trips_via_layer_prefix() {
        let config = RnnConfig::new().with_num_layers(2);
        let mut stacked = StackedRnn::new(2, 3, true, 1, config).unwrap();
        // 層 1 の入力幅は `num_directions * hidden_size`（単方向なので
        // `1 * 3 = 3`）であり層 0 の `input_size`（2）とは異なる。
        let new_weight = Tensor::new(vec![9.0f32; 9], &[3, 3]).unwrap();
        stacked
            .set_parameter("l1.weight_ih", new_weight.clone())
            .unwrap();
        assert_eq!(
            stacked
                .cell(1, 0)
                .unwrap()
                .weight_ih()
                .contiguous()
                .as_slice()
                .unwrap(),
            new_weight.contiguous().as_slice().unwrap()
        );
    }

    #[test]
    fn stacked_rnn_set_parameter_rejects_unknown_layer() {
        let config = RnnConfig::new().with_num_layers(1);
        let mut stacked = StackedRnn::new(2, 3, true, 1, config).unwrap();
        let dummy = Tensor::new(vec![0.0f32; 6], &[2, 3]).unwrap();
        assert!(stacked.set_parameter("l5.weight_ih", dummy).is_err());
    }

    #[test]
    fn stacked_rnn_set_requires_grad_freezes_all_cells() {
        let config = RnnConfig::new().with_num_layers(2);
        let mut stacked = StackedRnn::new(2, 3, true, 1, config).unwrap();
        assert!(stacked.requires_grad());
        stacked.set_requires_grad(false).unwrap();
        assert!(!stacked.requires_grad());
        for layer in 0..2 {
            assert!(!stacked.cell(layer, 0).unwrap().requires_grad());
        }
    }

    #[test]
    fn stacked_lstm_forward_seq_returns_c_n_with_expected_len() {
        let config = RnnConfig::new().with_num_layers(2).with_bidirectional(true);
        let stacked = StackedLstm::new(2, 3, true, 1, config).unwrap();
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = Tensor::new(vec![1.0, 0.5, -1.0, 0.3], &[2, 1, 2]).unwrap();
        let out = stacked.forward_seq(&tape, &x, None, None).unwrap();
        assert_eq!(out.h_n.len(), 4);
        assert_eq!(out.c_n.len(), 4);
    }
}
