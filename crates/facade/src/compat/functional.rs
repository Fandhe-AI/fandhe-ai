//! Functional API（多入力・多出力グラフ）の内部実装（イシュー #2665・親 #2663・ルート #2499
//! Phase 4）。
//!
//! Keras Functional API 相当を facade に段階導入する計画の第 2 段で、**層ノードからなる DAG の
//! 構築**（[`FunctionalBuilder`]）と**挿入順（= トポロジカル順）の forward**
//! （[`FunctionalModel`]）を提供する。設計の正は `docs/facade-functional-api-decision.md`
//! （§3 グラフ構築・§4 forward・§10 公開形・§11 配置・§12 OWASP 観点）で、同記録の公開形は
//! **未承認**（承認依頼は #2677。公開は承認後の #2679）。
//!
//! # 配置と非公開の理由
//!
//! 本モジュールは `compat/mod.rs` で `#[cfg(test)] mod functional;`（非公開）として宣言し、
//! 型・メソッドはすべて `pub(crate)` に留める。出荷コードのどこからも呼ばれないため
//! `#[cfg(test)]` を外すと `dead_code` になり、`#[allow]` で黙らせない方針（
//! `.claude/rules/coding-rust.md`）とも衝突するため、先例の `predict_batches`（#2192。
//! `crate::inference::batch`）と同じ `#[cfg(test)]` 隔離方式を採る。公開面が増えていないことは
//! `lib.rs::FunctionalApiHoldDoctestGuard`（正のプローブ）と
//! `tests/api_surface.rs::facade_functional_api_stays_internal`（ソース走査）が固定する。
//!
//! **昇格手順（#2679）**: `#[cfg(test)]` を外す → 型を `pub` にする → `compat/mod.rs` で
//! `pub use` する → 保留ガードを正ガードへ反転する。
//!
//! # 層ノードの単位と第 1 段の範囲
//!
//! ノードの単位は `compat::Sequential` ブロック（所有渡し）。層語彙・保存・学習・`add_module`
//! を再利用でき、層語彙を二重に持たない（REQ-9「薄いラッパー」）。本段が扱うのは
//! 入力ノード・ブロックノード・fan-out（1 ノードを複数ブロックが消費）・多入力・多出力まで。
//! 結合ノード（Concatenate／Add 等）は #2666、`bind`／`trainable_parameters`／
//! `apply_parameters`／`compile`／`fit`／保存は #2667 の担当で、ノード種別は内部 enum とし
//! 入力添字を複数持てる形にして追加可能にしてある。shape の構築時推論は行わず（層側に推論 API
//! が無い）、不整合は forward 時に既存 `Var` 演算の型付きエラーで検出する。
//!
//! 数値は既存 `Var` 演算（`Sequential::forward`）の合成のみで、新規 `Op`・`BackendOps`
//! メソッド・VJP を追加しない。GPU は `crate::tape_for` で構築した tape 上でそのまま到達できる。
//! エラーはすべて `AutodiffError`（意味論エラーは `InvalidArgument`）で、本番経路に
//! `unwrap`／`expect`／添字 panic を置かない。`build`／`forward` は反復のみで再帰しない。

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use super::Sequential;
use crate::{AutodiffError, Tape, Tensor, Var};

#[cfg(test)]
mod tests;

/// ビルダー ID の採番器。他ビルダー由来のハンドルの取り違えを検出するためのプロセス内一意値。
static NEXT_BUILDER_ID: AtomicU64 = AtomicU64::new(1);

/// 意味論エラー（`InvalidArgument`）を作る。メッセージには添字・件数のみを入れる。
fn invalid(message: String) -> AutodiffError {
    AutodiffError::InvalidArgument(message)
}

/// `var` が `tape` に属さなければ `TapeMismatch` を返す。
///
/// 所属判定は `Var::is_on_tape`（ノードを積まない読み取り専用判定）で行うため、不一致で
/// 拒否しても `tape` のノード数・メモリは増えない。
fn ensure_on_tape(tape: &Tape, var: &Var<'_>) -> Result<(), AutodiffError> {
    if !var.is_on_tape(&tape.0) {
        return Err(AutodiffError::TapeMismatch);
    }
    Ok(())
}

/// [`FunctionalBuilder`] が返すノードハンドル（Keras の「シンボリックテンソル」相当の添字）。
///
/// 発行元ビルダーの ID と添字だけを持つ `Copy` 値で、他ビルダーのハンドルは `apply`／`build`
/// が拒否する。添字は発行元ビルダーのノード列への位置で、ノードは自分より前の添字しか参照
/// できないため、構築時点で DAG が保証され挿入順がそのままトポロジカル順になる。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Node {
    builder_id: u64,
    index: usize,
}

/// グラフ上のノード定義（非公開）。結合ノード（#2666）はここへ variant を足す。
enum NodeDef {
    /// 外部から値を受ける入力ノード（Keras `Input`）。
    Input,
    /// `Sequential` ブロックを入力ノード列へ適用するノード。現段階の入力は 1 件。
    Block {
        block: Box<Sequential>,
        inputs: Vec<usize>,
    },
}

/// DAG を組み立てるアリーナ型ビルダー（`docs/facade-functional-api-decision.md` §3 案 A）。
///
/// `input` → `apply` → `build` の順で呼ぶ。`build` が構造検証（fail-closed）を行って
/// [`FunctionalModel`] を返す。
pub(crate) struct FunctionalBuilder {
    id: u64,
    nodes: Vec<NodeDef>,
}

impl Default for FunctionalBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl FunctionalBuilder {
    /// 空のビルダーを作る（ID はプロセス内で一意）。
    pub(crate) fn new() -> Self {
        Self {
            id: NEXT_BUILDER_ID.fetch_add(1, Ordering::Relaxed),
            nodes: Vec::new(),
        }
    }

    /// ハンドルが自分のものか・範囲内かを検証して添字を返す。
    fn resolve(&self, node: Node, what: &str) -> Result<usize, AutodiffError> {
        if node.builder_id != self.id {
            return Err(invalid(format!(
                "{what}: 他のビルダーが発行したノードは使えない"
            )));
        }
        if node.index >= self.nodes.len() {
            return Err(invalid(format!(
                "{what}: ノード添字 {} が範囲外（ノード数 {}）",
                node.index,
                self.nodes.len()
            )));
        }
        Ok(node.index)
    }

    /// ノードを 1 つ末尾へ追加し、そのハンドルを返す。確保失敗は型付きエラー。
    fn push_node(&mut self, def: NodeDef) -> Result<Node, AutodiffError> {
        self.nodes
            .try_reserve(1)
            .map_err(|_| super::alloc_failed())?;
        let index = self.nodes.len();
        self.nodes.push(def);
        Ok(Node {
            builder_id: self.id,
            index,
        })
    }

    /// 入力ノードを追加する（Keras `Input`）。shape は宣言せず、forward 時の値で決まる。
    pub(crate) fn input(&mut self) -> Result<Node, AutodiffError> {
        self.push_node(NodeDef::Input)
    }

    /// `block` を `input` へ適用するノードを追加する（Keras の層呼び出し）。
    ///
    /// 拒否: 他ビルダーのハンドル・範囲外添字・`compile()` 済みブロック（optimizer 状態を
    /// モデル側が 1 つだけ持つため、ブロック側の状態を黙って捨てない）・層 0 個のブロック
    /// （保存形式の「層範囲は重複なし・隙間なし」と相性が悪く、後から許可するのは非破壊・
    /// 逆は破壊的なため安全側）。
    pub(crate) fn apply(&mut self, block: Sequential, input: Node) -> Result<Node, AutodiffError> {
        let src = self.resolve(input, "apply")?;
        if block.compiled.is_some() {
            return Err(invalid(
                "apply: compile() 済みの Sequential は使えない（compile は Functional モデル側で行う）"
                    .to_string(),
            ));
        }
        if block.layers().is_empty() {
            return Err(invalid(
                "apply: 層を 1 つも持たない Sequential は使えない".to_string(),
            ));
        }
        let mut inputs = Vec::new();
        inputs.try_reserve(1).map_err(|_| super::alloc_failed())?;
        inputs.push(src);
        self.push_node(NodeDef::Block {
            block: Box::new(block),
            inputs,
        })
    }

    /// グラフを検証して [`FunctionalModel`] へ確定する（Keras `Model(inputs, outputs)`）。
    ///
    /// 拒否（すべて型付き `Err`）: `inputs`／`outputs` が空・他ビルダーのハンドル・範囲外・
    /// 重複・`inputs` に入力ノード以外・`inputs` に載らない入力ノード（未束縛）・どの出力にも
    /// 寄与しないノード。入力ノードを出力へそのまま指定すること（素通し）は許可する。
    /// 到達性は出力から添字の降順に 1 回走査して伝播する（前方参照が構造的に無いため 1 パスで
    /// 足り、再帰を使わない）。
    pub(crate) fn build(
        self,
        inputs: &[Node],
        outputs: &[Node],
    ) -> Result<FunctionalModel, AutodiffError> {
        if inputs.is_empty() {
            return Err(invalid("build: inputs が空".to_string()));
        }
        if outputs.is_empty() {
            return Err(invalid("build: outputs が空".to_string()));
        }
        let node_count = self.nodes.len();

        let mut listed = vec![false; node_count];
        let mut input_indices = Vec::new();
        input_indices
            .try_reserve(inputs.len())
            .map_err(|_| super::alloc_failed())?;
        for node in inputs {
            let index = self.resolve(*node, "build inputs")?;
            if !matches!(self.nodes.get(index), Some(NodeDef::Input)) {
                return Err(invalid(format!(
                    "build inputs: ノード {index} は入力ノードではない"
                )));
            }
            let slot = listed
                .get_mut(index)
                .ok_or_else(|| invalid(format!("build inputs: ノード {index} が範囲外")))?;
            if *slot {
                return Err(invalid(format!("build inputs: ノード {index} が重複")));
            }
            *slot = true;
            input_indices.push(index);
        }
        for (index, def) in self.nodes.iter().enumerate() {
            let is_listed = listed.get(index).copied().unwrap_or(false);
            if matches!(def, NodeDef::Input) && !is_listed {
                return Err(invalid(format!(
                    "build: 入力ノード {index} が inputs に列挙されていない（未束縛）"
                )));
            }
        }

        let mut reached = vec![false; node_count];
        let mut output_indices = Vec::new();
        output_indices
            .try_reserve(outputs.len())
            .map_err(|_| super::alloc_failed())?;
        for node in outputs {
            let index = self.resolve(*node, "build outputs")?;
            let slot = reached
                .get_mut(index)
                .ok_or_else(|| invalid(format!("build outputs: ノード {index} が範囲外")))?;
            if *slot {
                return Err(invalid(format!("build outputs: ノード {index} が重複")));
            }
            *slot = true;
            output_indices.push(index);
        }
        for index in (0..node_count).rev() {
            if !reached.get(index).copied().unwrap_or(false) {
                continue;
            }
            if let Some(NodeDef::Block { inputs, .. }) = self.nodes.get(index) {
                for src in inputs {
                    if let Some(slot) = reached.get_mut(*src) {
                        *slot = true;
                    }
                }
            }
        }
        if let Some(dead) = reached.iter().position(|r| !*r) {
            return Err(invalid(format!(
                "build: ノード {dead} はどの出力にも寄与しない"
            )));
        }

        let mut layer_starts = Vec::new();
        layer_starts
            .try_reserve(node_count)
            .map_err(|_| super::alloc_failed())?;
        let mut next_layer = 0usize;
        for def in &self.nodes {
            layer_starts.push(next_layer);
            if let NodeDef::Block { block, .. } = def {
                next_layer = next_layer
                    .checked_add(block.layers().len())
                    .ok_or_else(|| {
                        invalid("build: 層の通し番号がオーバーフローする".to_string())
                    })?;
            }
        }

        Ok(FunctionalModel {
            nodes: self.nodes,
            inputs: input_indices,
            outputs: output_indices,
            layer_starts,
        })
    }
}

/// 検証済みの多入力・多出力グラフ（Keras `Model`）。
///
/// ノードを挿入順に反復で 1 回ずつ評価する。Dropout の RNG 消費・BatchNorm の running stats
/// 更新はノード挿入順に各 1 回で、fan-out しても上流ノードは再評価しない。パラメータキーは
/// 全ブロックを通した層の通し番号 `i` による `"{i}.{name}"`（`docs/facade-functional-api-
/// decision.md` §4。Block ノードの挿入順に層数を累積）。
pub(crate) struct FunctionalModel {
    nodes: Vec<NodeDef>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    /// ノードごとの通し番号の先頭（入力ノードは次のブロックの先頭と同値で未使用）。
    layer_starts: Vec<usize>,
}

/// ブロックのローカルキー `"{j}.{name}"` を最初の `.` でのみ分割し、通し番号キー
/// `"{layer_start + j}.{name}"` へ写す。名前側に `.` を含みうるため最初の 1 か所だけで分割する。
/// 通し番号キーの採番規則は本関数 1 か所に置く。
fn to_global_key(layer_start: usize, local: &str) -> Result<String, AutodiffError> {
    let (index, name) = local.split_once('.').ok_or_else(|| {
        invalid(format!(
            "パラメータキー {local:?} が \"{{index}}.{{name}}\" 形式でない"
        ))
    })?;
    let index: usize = index
        .parse()
        .map_err(|_| invalid(format!("パラメータキー {local:?} の層番号が数値でない")))?;
    let global = layer_start
        .checked_add(index)
        .ok_or_else(|| invalid("パラメータキーの通し番号がオーバーフローする".to_string()))?;
    Ok(format!("{global}.{name}"))
}

impl FunctionalModel {
    /// ブロックノードの `(ノード添字, ブロック)` を挿入順に列挙する。
    fn blocks(&self) -> impl Iterator<Item = (usize, &Sequential)> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, def)| match def {
                NodeDef::Block { block, .. } => Some((i, block.as_ref())),
                NodeDef::Input => None,
            })
    }

    /// 通し番号の先頭を引く（`build` が全ノード分を作るため常に存在する）。
    fn layer_start(&self, node_index: usize) -> Result<usize, AutodiffError> {
        self.layer_starts
            .get(node_index)
            .copied()
            .ok_or_else(|| invalid(format!("内部不整合: ノード {node_index} の層番号がない")))
    }

    /// `inputs`（`build` に渡した `inputs` の順）を入力ノードへ束ね、ノードを挿入順に 1 回ずつ
    /// 評価して `outputs`（`build` に渡した順）の値を返す。入力件数の不一致は `InvalidArgument`。
    /// shape の不整合は既存 `Var` 演算の型付きエラーに委ねる。
    pub(crate) fn forward<'t>(
        &self,
        tape: &'t Tape,
        inputs: &[Var<'t>],
    ) -> Result<Vec<Var<'t>>, AutodiffError> {
        if inputs.len() != self.inputs.len() {
            return Err(invalid(format!(
                "forward: 入力数 {} がモデルの入力数 {} と一致しない",
                inputs.len(),
                self.inputs.len()
            )));
        }
        // 素通しグラフ（入力をそのまま出力にする構成）ではブロック評価の演算が走らず、別 tape の
        // `Var` が成功として返りうる。入力を格納する前に全入力の所属 tape を検証する。
        for var in inputs {
            ensure_on_tape(tape, var)?;
        }
        let mut values: Vec<Option<Var<'t>>> = Vec::new();
        values
            .try_reserve_exact(self.nodes.len())
            .map_err(|_| super::alloc_failed())?;
        values.resize(self.nodes.len(), None);
        for (slot, var) in self.inputs.iter().zip(inputs) {
            *values
                .get_mut(*slot)
                .ok_or_else(|| invalid(format!("内部不整合: 入力ノード {slot} が範囲外")))? =
                Some(*var);
        }
        for (index, def) in self.nodes.iter().enumerate() {
            let NodeDef::Block { block, inputs } = def else {
                continue;
            };
            let [src] = inputs.as_slice() else {
                return Err(invalid(format!(
                    "内部不整合: ブロックノード {index} の入力数が 1 でない"
                )));
            };
            let x = values
                .get(*src)
                .copied()
                .flatten()
                .ok_or_else(|| invalid(format!("内部不整合: ノード {src} が未評価")))?;
            let y = block.forward(tape, &x)?;
            *values
                .get_mut(index)
                .ok_or_else(|| invalid(format!("内部不整合: ノード {index} が範囲外")))? = Some(y);
        }
        let mut outputs = Vec::new();
        outputs
            .try_reserve_exact(self.outputs.len())
            .map_err(|_| super::alloc_failed())?;
        for slot in &self.outputs {
            let value = values
                .get(*slot)
                .copied()
                .flatten()
                .ok_or_else(|| invalid(format!("内部不整合: 出力ノード {slot} が未評価")))?;
            outputs.push(value);
        }
        Ok(outputs)
    }

    /// 推論の入口。`crate::tape()`（既定 CPU）を 1 回構築して [`Self::forward`] を呼び、各出力を
    /// `Tensor` へ取り出す。`Sequential::predict` の tape 不要経路（Linear→ReLU 融合）は使わず
    /// 第 1 段は単純さを優先する。モードを暗黙に切り替えない（`Sequential::predict` と同じ。
    /// 決定的な推論は先に `eval()` を呼ぶ）。
    pub(crate) fn predict(
        &self,
        inputs: &[&Tensor<f32>],
    ) -> Result<Vec<Tensor<f32>>, AutodiffError> {
        let tape = crate::tape();
        let mut vars = Vec::new();
        vars.try_reserve_exact(inputs.len())
            .map_err(|_| super::alloc_failed())?;
        for tensor in inputs {
            vars.push(tape.var(tensor));
        }
        let outputs = self.forward(&tape, &vars)?;
        Ok(outputs.iter().map(Var::to_tensor).collect())
    }

    /// 全ブロックへモードを伝播する。モデル側に別フラグは持たない（ブロックと食い違う状態を
    /// 作らないため。`build` はブロックのモードを同期しない契約で、`Sequential::add_dropout` と
    /// 同じ）。
    pub(crate) fn set_training(&mut self, training: bool) {
        for def in &mut self.nodes {
            if let NodeDef::Block { block, .. } = def {
                block.set_training(training);
            }
        }
    }

    /// `set_training(true)` の別名。
    pub(crate) fn train(&mut self) {
        self.set_training(true);
    }

    /// `set_training(false)` の別名。
    pub(crate) fn eval(&mut self) {
        self.set_training(false);
    }

    /// 全ブロックが train のとき `true`（ブロック 0 個なら `true`）の導出値。
    pub(crate) fn training(&self) -> bool {
        self.blocks().all(|(_, block)| block.training())
    }

    /// 「通し番号キー, パラメータ参照」列。順序はブロックの挿入順・各ブロック内は
    /// `Sequential::named_parameters` と同じ。
    pub(crate) fn named_parameters(&self) -> Result<Vec<(String, &Tensor<f32>)>, AutodiffError> {
        let mut out = Vec::new();
        for (index, block) in self.blocks() {
            let start = self.layer_start(index)?;
            for (local, tensor) in block.named_parameters() {
                out.try_reserve(1).map_err(|_| super::alloc_failed())?;
                out.push((to_global_key(start, &local)?, tensor));
            }
        }
        Ok(out)
    }

    /// 通し番号キー付きのパラメータ辞書（`named_parameters` と同じキー集合）。
    pub(crate) fn state_dict(&self) -> Result<HashMap<String, Tensor<f32>>, AutodiffError> {
        let mut out = HashMap::new();
        for (index, block) in self.blocks() {
            let start = self.layer_start(index)?;
            for (local, tensor) in block.state_dict() {
                out.try_reserve(1).map_err(|_| super::alloc_failed())?;
                out.insert(to_global_key(start, &local)?, tensor);
            }
        }
        Ok(out)
    }

    /// [`Self::state_dict`] の逆（strict。`fandhe_ai_autodiff::nn::Module::load_state_dict` と同型の
    /// 「2 パス + ベストエフォート・ロールバック」契約）。
    ///
    /// 第 1 パスでキー集合の完全一致（未知キー・欠落キーを拒否）と各テンソルの shape 一致を
    /// 何も変更しないうちに検査する。第 2 パスでブロックごとにローカルキーへ戻して
    /// `Sequential::load_state_dict` へ委譲し、途中で失敗した場合は開始前のスナップショットで
    /// 適用済みブロックを巻き戻して元のエラーを返す。巻き戻し自体が失敗した場合は、
    /// 適用失敗と巻き戻し失敗の双方と部分適用の可能性を示す `InvalidArgument` を返す。
    pub(crate) fn load_state_dict(
        &mut self,
        mut state: HashMap<String, Tensor<f32>>,
    ) -> Result<(), AutodiffError> {
        // 第 1 パス: ブロックごとに (ローカルキー, 通し番号キー, 現在の shape) を引く。
        let mut plan: Vec<(usize, Vec<(String, String)>)> = Vec::new();
        let mut expected_keys = 0usize;
        for (index, block) in self.blocks() {
            let start = self.layer_start(index)?;
            let mut pairs = Vec::new();
            for (local, current) in block.state_dict() {
                let global = to_global_key(start, &local)?;
                let Some(given) = state.get(&global) else {
                    return Err(invalid(format!("load_state_dict: キー {global:?} が欠落")));
                };
                if given.shape() != current.shape() {
                    return Err(invalid(format!(
                        "load_state_dict: キー {global:?} の shape {:?} が期待 {:?} と一致しない",
                        given.shape(),
                        current.shape()
                    )));
                }
                expected_keys += 1;
                pairs.try_reserve(1).map_err(|_| super::alloc_failed())?;
                pairs.push((local, global));
            }
            plan.try_reserve(1).map_err(|_| super::alloc_failed())?;
            plan.push((index, pairs));
        }
        if state.len() != expected_keys {
            let unknown = state
                .keys()
                .find(|k| !plan.iter().any(|(_, p)| p.iter().any(|(_, g)| g == *k)));
            return Err(invalid(format!(
                "load_state_dict: 未知のキー {unknown:?} がある（期待 {expected_keys} 件・入力 {} 件）",
                state.len()
            )));
        }

        // 第 2 パス: スナップショットを取ってからブロック単位で適用する。
        let mut snapshots: Vec<HashMap<String, Tensor<f32>>> = Vec::new();
        snapshots
            .try_reserve_exact(plan.len())
            .map_err(|_| super::alloc_failed())?;
        for (index, _) in &plan {
            match self.nodes.get(*index) {
                Some(NodeDef::Block { block, .. }) => snapshots.push(block.state_dict()),
                _ => {
                    return Err(invalid(format!(
                        "内部不整合: ノード {index} がブロックでない"
                    )));
                }
            }
        }
        for (applied, (index, pairs)) in plan.iter().enumerate() {
            let mut local_state = HashMap::new();
            for (local, global) in pairs {
                let tensor = state.remove(global).ok_or_else(|| {
                    invalid(format!("内部不整合: キー {global:?} が第 2 パスで消えた"))
                })?;
                local_state.insert(local.clone(), tensor);
            }
            let result = match self.nodes.get_mut(*index) {
                Some(NodeDef::Block { block, .. }) => block.load_state_dict(local_state),
                _ => Err(invalid(format!(
                    "内部不整合: ノード {index} がブロックでない"
                ))),
            };
            if let Err(err) = result {
                // 巻き戻し失敗は握りつぶさず、部分適用の可能性を明示して fail-closed に返す
                // （`Module::load_state_dict` 契約・security.md A08）。
                let mut rollback_failures: Vec<String> = Vec::new();
                for ((done_index, _), snapshot) in plan.iter().zip(snapshots.iter()).take(applied) {
                    match self.nodes.get_mut(*done_index) {
                        Some(NodeDef::Block { block, .. }) => {
                            if let Err(rb) = block.load_state_dict(snapshot.clone()) {
                                rollback_failures.push(format!("ノード {done_index}: {rb}"));
                            }
                        }
                        _ => rollback_failures
                            .push(format!("ノード {done_index}: ブロックでないため復元不能")),
                    }
                }
                if !rollback_failures.is_empty() {
                    return Err(invalid(format!(
                        "load_state_dict: ノード {index} の適用に失敗（{err}）し、先行ブロックの巻き戻しにも失敗した（{}）。モデルが部分適用のまま残っている可能性がある",
                        rollback_failures.join("; ")
                    )));
                }
                return Err(err);
            }
        }
        Ok(())
    }
}
