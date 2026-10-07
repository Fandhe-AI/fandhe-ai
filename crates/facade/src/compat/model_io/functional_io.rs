//! Functional モデルのディレクトリ保存・復元（`save_functional_model`・`load_functional_model`。
//! イシュー #2667・親 #2663・ルート #2499 Phase 4）。
//!
//! 役割: `compat::functional::FunctionalModel`（多入力・多出力グラフ）を、既存の
//! `save_model`／`load_model`（`model_io.rs`）と同じ 2 ファイル構成
//! （`manifest.json` + `model.<gen>.safetensors`）で往復させる。形式の設計は
//! `docs/facade-functional-api-decision.md` §8（S1 案）・§18（#2667 実装記録）、ファイル I/O の
//! 脅威モデルは `docs/compat-model-io-decision.md` §12・§13 が正。イシュー #2679 で入口 2 本を
//! `fandhe_ai::compat` へ公開した（承認はルート #2499 のコメント。`compat/mod.rs` が
//! `pub use model_io::functional_io::{..}` で公開する。モジュール自体は非公開）。
//!
//! # 形式（`format = "fandhe-ai.compat.functional"`・`format_version = 1`）
//!
//! 最上位 object のキーは 13 個（既存 `Sequential` 形式の 10 個 + `nodes`・`inputs`・`outputs`）。
//! `layers` は**全ブロックを通した平坦な列**（safetensors キー `{i}.{name}` の通し番号名前空間を
//! `Sequential` 形式と同じに保つ）、`nodes` は
//! `{index, op, inputs:[..], layer_start, layer_len, params:{..}}` の平坦配列、`inputs` は入力ノードの
//! 添字列（`build` に渡した順。件数だけでは順序を復元できないため設計記録 §8 の「入力数」から改めた）、
//! `outputs` は出力ノードの添字列（同順）。`params` は `concatenate` の `{"dim": n}` だけで、他は `{}`。
//! 既存 `load_model` が拒否する（最上位キーの集合が異なるため `Manifest`）形式であり、逆も同様
//! （形式の相互排他）。
//!
//! # 非信頼入力の扱い（OWASP A03／A08）
//!
//! manifest は非信頼入力。既存の手書き厳格パーサ・`open_leaf_checked`・世代コミット書き込みを共有し
//! （別の I/O 経路を作らない）、**グラフの構造検証は safetensors を開く前に完了する**:
//! `nodes[].index` の連番・`op` の文字列 allowlist（未知は `UnsupportedModel`）・`inputs[j] < index`
//! （前方参照禁止。循環を構文で排除）・op ごとの入力件数と `layer_len`・`layer_start` が累積値と一致
//! （重複・隙間なし）・総和が `layers` 件数と一致・結合入力の重複なし・最上位 `inputs`／`outputs` の
//! 範囲と重複・全ノードが出力へ寄与すること。件数は配列を数えた結果だけを信じ、manifest の数値を
//! 確保量・ループ回数の根拠にしない。上限定数（`MAX_*`）は変更も新設もしない。
//!
//! # 第 1 段で保存・復元しないもの（fail-closed）
//!
//! `add_module` の利用者定義層・ブロック間でモードが食い違うモデル・`Optimizer::Lbfgs` と AMP を含む
//! compile 状態（`compile` が `Lbfgs` を拒否するため通常到達しない多層防御）は `UnsupportedModel`。
//! 保存は「書いたものは必ず読める」ため、書き込み前に manifest を描画して同じパーサで読み戻し、
//! 構成の一致を確認する（上限起因は `TooLarge`）。

use std::path::Path;

use fandhe_ai_tensor_core::Tensor;

use super::compiled::{CompiledMeta, OPTIMIZER_PREFIX, check_slot_shapes, parse_compiled};
use super::{
    CheckedState, Json, LoadedTensors, MANIFEST_FILE_NAME, MAX_ARRAY_LEN, MAX_LAYERS,
    MAX_MANIFEST_BYTES, ModelIoError, ParsedLayers, Parser, as_arr, as_str, as_u64, as_usize,
    build_model, collect_checked_state, exact_fields, is_valid_safetensors_file_name,
    manifest_error, map_leaf_error, parse_layers_and_keys, read_checked_tensors, render_key_shapes,
    render_params, save_platform_check, spec_kind,
};
use crate::compat::functional::{
    FunctionalBuilder, FunctionalModel, MergeKind, Node, NodeDef, to_global_key, to_local_key,
};
use crate::compat::sequential::{LayerSpec, Sequential};
use crate::compat::training::Optimizer;
use crate::fs_guard::{MAX_MODEL_FILE_BYTES, OpenedLeaf, open_leaf_checked};
use crate::interop::safetensors::save_safetensors_f32_to_bytes;

#[cfg(test)]
mod tests;

/// manifest の `format` 値（`Sequential` 形式の `fandhe-ai.compat.sequential` と別名にして相互排他にする）。
const FUNCTIONAL_FORMAT_NAME: &str = "fandhe-ai.compat.functional";
/// manifest の `format_version` 値。
const FUNCTIONAL_FORMAT_VERSION: u64 = 1;

/// グラフ上のノード種別（manifest の `op` 文字列 allowlist に対応）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeOp {
    Input,
    Block,
    Merge(MergeKind),
}

impl NodeOp {
    /// manifest 上の `op` 文字列。
    fn wire_name(self) -> &'static str {
        match self {
            NodeOp::Input => "input",
            NodeOp::Block => "block",
            NodeOp::Merge(MergeKind::Concatenate { .. }) => "concatenate",
            NodeOp::Merge(MergeKind::Add) => "add",
            NodeOp::Merge(MergeKind::Multiply) => "multiply",
            NodeOp::Merge(MergeKind::Average) => "average",
        }
    }
}

/// manifest の `nodes[]` 1 件（保存側が組み立て・復元側が検証して読む共通表現）。
#[derive(Debug, Clone, PartialEq, Eq)]
struct NodeRecord {
    op: NodeOp,
    inputs: Vec<usize>,
    layer_start: usize,
    layer_len: usize,
}

/// 検証を終えた保存対象（`dir` への副作用なしで組み立てる）。
struct PreparedFunctionalSave {
    training: bool,
    /// 全ブロックを通した平坦な層構成。
    specs: Vec<LayerSpec>,
    nodes: Vec<NodeRecord>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    parameter_keys: Vec<(String, Vec<usize>)>,
    buffer_keys: Vec<(String, Vec<usize>)>,
    compiled: Option<CompiledMeta>,
    safetensors: Vec<u8>,
}

/// 検証済みの manifest。
struct ParsedFunctional {
    training: bool,
    specs: Vec<LayerSpec>,
    nodes: Vec<NodeRecord>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    parameter_keys: Vec<(String, Vec<usize>)>,
    buffer_keys: Vec<(String, Vec<usize>)>,
    safetensors_file: String,
    safetensors_bytes: u64,
    compiled: Option<CompiledMeta>,
}

// ---------------------------------------------------------------------
// 保存
// ---------------------------------------------------------------------

/// `model` を `dir` へ保存する（世代コミット方式。手順・契約は `model_io.rs` の `save_model` と同じ）。
///
/// 検証・往復確認は `dir` へ触れる前にすべて完了し、失敗時は `dir` に何も残さない。非 unix は
/// `ErrorKind::Unsupported` で `dir` に触れず拒否する。
pub fn save_functional_model(
    model: &FunctionalModel,
    dir: impl AsRef<Path>,
) -> Result<(), ModelIoError> {
    save_platform_check()?;
    let prepared = prepare_functional_save(model)?;
    write_functional(dir.as_ref(), &prepared)
}

#[cfg(not(unix))]
fn write_functional(dir: &Path, prepared: &PreparedFunctionalSave) -> Result<(), ModelIoError> {
    let _ = (dir, prepared);
    Err(ModelIoError::Io(std::io::Error::from(
        std::io::ErrorKind::Unsupported,
    )))
}

#[cfg(unix)]
fn write_functional(dir: &Path, p: &PreparedFunctionalSave) -> Result<(), ModelIoError> {
    super::write_generation_with(
        dir,
        &p.safetensors,
        |file, bytes| render_functional_manifest(p, file, bytes),
        super::generation_id,
        super::tmp_manifest_name,
    )
}

/// 保存前の検証をすべて行い、書き込むバイト列と manifest の材料を返す（`dir` への副作用なし）。
fn prepare_functional_save(
    model: &FunctionalModel,
) -> Result<PreparedFunctionalSave, ModelIoError> {
    let snapshot = model
        .compile_state_snapshot()
        .map_err(ModelIoError::Autodiff)?;
    if let Some(snap) = &snapshot
        && (matches!(snap.optimizer, Optimizer::Lbfgs(_)) || snap.amp.is_some())
    {
        return Err(ModelIoError::UnsupportedModel {
            reason: "Functional モデルは Lbfgs・AMP を含む compile 状態を保存できません".into(),
        });
    }
    let compiled = snapshot
        .as_ref()
        .map(CompiledMeta::from_snapshot)
        .transpose()?;

    // 全ブロックを検査し、層構成・state を通し番号へ写して合成する。
    let mut specs: Vec<LayerSpec> = Vec::new();
    let mut state: std::collections::HashMap<String, Tensor<f32>> =
        std::collections::HashMap::new();
    let mut block_training: Option<bool> = None;
    let mut nodes: Vec<NodeRecord> = Vec::new();
    nodes
        .try_reserve_exact(model.nodes.len())
        .map_err(|_| ModelIoError::TooLarge {
            what: "ノード数",
            limit: MAX_ARRAY_LEN as u64,
        })?;
    for (index, def) in model.nodes.iter().enumerate() {
        let layer_start = specs.len();
        // 通し番号の先頭はモデル側の採番（`layer_starts`）と一致していなければならない。
        if model.layer_start(index).map_err(ModelIoError::Autodiff)? != layer_start {
            return Err(ModelIoError::UnsupportedModel {
                reason: format!("ノード {index} の層番号の先頭が内部の採番と一致しません"),
            });
        }
        match def {
            NodeDef::Input => nodes.push(NodeRecord {
                op: NodeOp::Input,
                inputs: Vec::new(),
                layer_start,
                layer_len: 0,
            }),
            NodeDef::Merge { kind, inputs } => nodes.push(NodeRecord {
                op: NodeOp::Merge(*kind),
                inputs: inputs.clone(),
                layer_start,
                layer_len: 0,
            }),
            NodeDef::Block { block, inputs } => {
                let layer_len = block.layers().len();
                // 総層数の上限は `Sequential` 形式と同じ（ブロック単位でも `collect_checked_state` が検査）。
                let total = layer_start
                    .checked_add(layer_len)
                    .filter(|n| *n <= MAX_LAYERS)
                    .ok_or(ModelIoError::TooLarge {
                        what: "層数",
                        limit: MAX_LAYERS as u64,
                    })?;
                match block_training {
                    None => block_training = Some(block.training()),
                    Some(t) if t != block.training() => {
                        return Err(ModelIoError::UnsupportedModel {
                            reason: format!(
                                "ブロック間でモード（training）が食い違っています（ノード {index}）。load は全ブロックをモデルのモードへ揃えるため、`train()`／`eval()` で揃えてから保存してください"
                            ),
                        });
                    }
                    Some(_) => {}
                }
                let CheckedState {
                    state: block_state, ..
                } = collect_checked_state(block)?;
                specs
                    .try_reserve(layer_len)
                    .map_err(|_| ModelIoError::TooLarge {
                        what: "層数",
                        limit: MAX_LAYERS as u64,
                    })?;
                specs.extend_from_slice(block.specs());
                debug_assert_eq!(specs.len(), total);
                for (key, tensor) in block_state {
                    let global =
                        to_global_key(layer_start, &key).map_err(ModelIoError::Autodiff)?;
                    if state.insert(global.clone(), tensor).is_some() {
                        return Err(ModelIoError::UnsupportedModel {
                            reason: format!("キー {global} がブロック間で衝突します"),
                        });
                    }
                }
                nodes.push(NodeRecord {
                    op: NodeOp::Block,
                    inputs: inputs.clone(),
                    layer_start,
                    layer_len,
                });
            }
        }
    }

    // 平坦化した層構成から期待キー・shape を導き、合成 state と完全一致を確認する。
    let expected = super::expected_parameter_keys(&specs);
    let expected_buffers = super::expected_buffer_keys(&specs);
    let keys_ok = state.len() == expected.len() + expected_buffers.len()
        && expected
            .iter()
            .chain(&expected_buffers)
            .all(|(k, shape)| state.get(k).is_some_and(|t| t.shape() == shape.as_slice()));
    if !keys_ok {
        return Err(ModelIoError::UnsupportedModel {
            reason: "state のキー・shape が層構成から導いた期待と一致しません".into(),
        });
    }

    if let Some(snap) = snapshot {
        // load 側と同じスロット整合ガードを保存側でも通す（「書いたものは必ず load できる」契約）。
        check_slot_shapes(&snap.optimizer_state, &model.trainable_parameters())?;
        for (key, tensor) in snap.optimizer_state {
            if state
                .insert(format!("{OPTIMIZER_PREFIX}{key}"), tensor)
                .is_some()
            {
                return Err(ModelIoError::UnsupportedModel {
                    reason: "optimizer 状態のキーがパラメータのキーと衝突しました".into(),
                });
            }
        }
    }
    let safetensors = save_safetensors_f32_to_bytes(&state, None)
        .map_err(|e| ModelIoError::Safetensors(e.to_string()))?;
    if safetensors.len() as u64 > MAX_MODEL_FILE_BYTES {
        return Err(ModelIoError::TooLarge {
            what: "model safetensors",
            limit: MAX_MODEL_FILE_BYTES,
        });
    }

    let prepared = PreparedFunctionalSave {
        // ブロック 0 個のモデルは `training()` が導出値 `true`。
        training: block_training.unwrap_or(true),
        specs,
        nodes,
        inputs: model.inputs.clone(),
        outputs: model.outputs.clone(),
        parameter_keys: expected,
        buffer_keys: expected_buffers,
        compiled,
        safetensors,
    };
    verify_functional_round_trip(&prepared)?;
    Ok(prepared)
}

/// 保存する構成を manifest へ描画し、load と同じ厳格パーサで読み戻して元の構成と一致することを
/// 確認する（「保存できたのに読めない」ファイルを作らない。`dir` への副作用の前に行う）。
fn verify_functional_round_trip(prepared: &PreparedFunctionalSave) -> Result<(), ModelIoError> {
    let probe_name = format!("model.{}.safetensors", "0".repeat(32));
    let text = render_functional_manifest(prepared, &probe_name, prepared.safetensors.len() as u64);
    if text.len() as u64 > MAX_MANIFEST_BYTES {
        return Err(ModelIoError::TooLarge {
            what: "manifest.json",
            limit: MAX_MANIFEST_BYTES,
        });
    }
    let parsed = match parse_functional_manifest(text.as_bytes()) {
        Ok(p) => p,
        Err(e @ ModelIoError::TooLarge { .. }) => return Err(e),
        Err(e) => {
            return Err(ModelIoError::UnsupportedModel {
                reason: format!("保存する構成を load が読み戻せません（{e}）"),
            });
        }
    };
    let same_specs = parsed.specs.len() == prepared.specs.len()
        && parsed
            .specs
            .iter()
            .zip(&prepared.specs)
            .all(|(a, b)| spec_kind(a) == spec_kind(b) && render_params(a) == render_params(b));
    let same_compiled = parsed
        .compiled
        .as_ref()
        .map(super::compiled::render_compiled)
        == prepared
            .compiled
            .as_ref()
            .map(super::compiled::render_compiled);
    if parsed.training != prepared.training
        || parsed.nodes != prepared.nodes
        || parsed.inputs != prepared.inputs
        || parsed.outputs != prepared.outputs
        || parsed.parameter_keys != prepared.parameter_keys
        || parsed.buffer_keys != prepared.buffer_keys
        || !same_specs
        || !same_compiled
    {
        return Err(ModelIoError::UnsupportedModel {
            reason: "保存する構成と読み戻した構成が一致しません".into(),
        });
    }
    Ok(())
}

/// 添字列を `[1,2,3]` 形式で描画する。
fn render_index_list(items: &[usize]) -> String {
    let parts: Vec<String> = items.iter().map(usize::to_string).collect();
    format!("[{}]", parts.join(","))
}

/// manifest を決定的な文字列にする（キー順固定・文字列はエスケープ不要なプログラム生成の ASCII のみ）。
fn render_functional_manifest(
    p: &PreparedFunctionalSave,
    safetensors_file: &str,
    safetensors_bytes: u64,
) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "{{\"format\":\"{FUNCTIONAL_FORMAT_NAME}\",\"format_version\":{FUNCTIONAL_FORMAT_VERSION},\"training\":{},\"num_layers\":{},\"layers\":[",
        p.training,
        p.specs.len()
    ));
    for (i, spec) in p.specs.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let kind = spec_kind(spec).unwrap_or("unsupported");
        let params = render_params(spec);
        s.push_str(&format!(
            "{{\"index\":{i},\"kind\":\"{kind}\",\"params\":{params}}}"
        ));
    }
    s.push_str("],\"nodes\":[");
    for (i, node) in p.nodes.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let params = match node.op {
            NodeOp::Merge(MergeKind::Concatenate { dim }) => format!("{{\"dim\":{dim}}}"),
            _ => "{}".to_string(),
        };
        s.push_str(&format!(
            "{{\"index\":{i},\"op\":\"{}\",\"inputs\":{},\"layer_start\":{},\"layer_len\":{},\"params\":{params}}}",
            node.op.wire_name(),
            render_index_list(&node.inputs),
            node.layer_start,
            node.layer_len
        ));
    }
    s.push_str("],\"inputs\":");
    s.push_str(&render_index_list(&p.inputs));
    s.push_str(",\"outputs\":");
    s.push_str(&render_index_list(&p.outputs));
    s.push_str(",\"parameter_keys\":");
    s.push_str(&render_key_shapes(&p.parameter_keys));
    s.push_str(",\"buffer_keys\":");
    s.push_str(&render_key_shapes(&p.buffer_keys));
    let compiled = p
        .compiled
        .as_ref()
        .map_or_else(|| "null".to_string(), super::compiled::render_compiled);
    s.push_str(&format!(
        ",\"safetensors_file\":\"{safetensors_file}\",\"safetensors_bytes\":{safetensors_bytes},\"compiled\":{compiled}}}"
    ));
    s
}

// ---------------------------------------------------------------------
// 復元
// ---------------------------------------------------------------------

/// `save_functional_model` が書いたディレクトリから `FunctionalModel` を復元する。
///
/// 重みと BatchNorm の running stats は safetensors の値を bit のまま設定し、`training` と compile
/// 状態も復元する。途中で失敗しても部分的に構築したモデルは返さない。
pub fn load_functional_model(dir: impl AsRef<Path>) -> Result<FunctionalModel, ModelIoError> {
    load_functional_with_limits(dir.as_ref(), MAX_MANIFEST_BYTES, MAX_MODEL_FILE_BYTES)
}

/// [`load_functional_model`] の本体。固定上限を引数化して、単体テストが巨大ファイルを作らずに上限超過の
/// 拒否を検証できるようにする（`load_from_dir_with_limits` と同型）。
fn load_functional_with_limits(
    dir: &Path,
    max_manifest: u64,
    max_model: u64,
) -> Result<FunctionalModel, ModelIoError> {
    let manifest_bytes = open_leaf_checked(&dir.join(MANIFEST_FILE_NAME), max_manifest)
        .and_then(OpenedLeaf::read_exact_len)
        .map_err(map_leaf_error("manifest.json"))?;
    // 構造検証（ここまでで safetensors には一切触れていない）。
    let manifest = parse_functional_manifest(&manifest_bytes)?;

    let LoadedTensors {
        mut tensors,
        optimizer_state,
    } = read_checked_tensors(
        dir,
        max_model,
        &manifest.safetensors_file,
        manifest.safetensors_bytes,
        &manifest.parameter_keys,
        &manifest.buffer_keys,
        manifest.compiled.as_ref(),
    )?;

    // 層番号 → 所有ブロックの序数（ブロックごとのローカルキー化に使う。反復 1 回で全キーを振り分ける）。
    let mut block_nodes: Vec<&NodeRecord> = Vec::new();
    let mut owner: Vec<usize> = vec![usize::MAX; manifest.specs.len()];
    for node in &manifest.nodes {
        if node.op == NodeOp::Block {
            let ordinal = block_nodes.len();
            block_nodes.push(node);
            for slot in owner.iter_mut().skip(node.layer_start).take(node.layer_len) {
                *slot = ordinal;
            }
        }
    }
    let mut block_params: Vec<std::collections::HashMap<String, Tensor<f32>>> =
        (0..block_nodes.len()).map(|_| Default::default()).collect();
    let mut block_buffers: Vec<std::collections::HashMap<String, Tensor<f32>>> =
        (0..block_nodes.len()).map(|_| Default::default()).collect();
    for (keys, maps) in [
        (&manifest.parameter_keys, &mut block_params),
        (&manifest.buffer_keys, &mut block_buffers),
    ] {
        for (key, _) in keys {
            let (ordinal, local) = localize_key(key, &owner, &block_nodes)?;
            let tensor = tensors.remove(key).ok_or_else(|| ModelIoError::Mismatch {
                message: "復元に必要なキーが safetensors に見つかりません".into(),
            })?;
            maps.get_mut(ordinal)
                .ok_or_else(|| ModelIoError::Mismatch {
                    message: "キーの所有ブロックが範囲外です".into(),
                })?
                .insert(local, tensor);
        }
    }
    if !tensors.is_empty() {
        return Err(ModelIoError::Mismatch {
            message: "safetensors に manifest が列挙していないキーがあります".into(),
        });
    }

    // グラフを manifest の順に再構築する（`build` が到達性・未束縛入力等を再検証する）。
    let mut builder = FunctionalBuilder::new();
    let mut handles: Vec<Node> = Vec::new();
    handles
        .try_reserve_exact(manifest.nodes.len())
        .map_err(|_| ModelIoError::TooLarge {
            what: "ノード数",
            limit: MAX_ARRAY_LEN as u64,
        })?;
    let mut next_block = 0usize;
    for (index, node) in manifest.nodes.iter().enumerate() {
        let sources: Vec<Node> = node
            .inputs
            .iter()
            .map(|i| {
                handles.get(*i).copied().ok_or_else(|| {
                    manifest_error(format!("nodes[{index}] の入力 {i} が未構築のノードです"))
                })
            })
            .collect::<Result<_, _>>()?;
        let handle = match node.op {
            NodeOp::Input => builder.input(),
            NodeOp::Block => {
                let ordinal = next_block;
                next_block += 1;
                let specs = manifest
                    .specs
                    .get(node.layer_start..node.layer_start + node.layer_len)
                    .ok_or_else(|| manifest_error("層範囲が layers の範囲外です"))?;
                let mut params =
                    std::mem::take(block_params.get_mut(ordinal).ok_or_else(|| {
                        manifest_error("ブロックのパラメータ集合が見つかりません")
                    })?);
                let mut buffers = std::mem::take(
                    block_buffers
                        .get_mut(ordinal)
                        .ok_or_else(|| manifest_error("ブロックの buffer 集合が見つかりません"))?,
                );
                let mut block: Sequential = build_model(specs, &mut buffers, &params)?;
                // 値を bit のまま設定する（strict。未知キー・欠落・shape 不一致は拒否される）。
                block
                    .load_state_dict(std::mem::take(&mut params))
                    .map_err(ModelIoError::Autodiff)?;
                let [src] = sources.as_slice() else {
                    return Err(manifest_error(format!(
                        "nodes[{index}] のブロックノードは入力 1 件が必要です"
                    )));
                };
                builder.apply(block, *src)
            }
            NodeOp::Merge(MergeKind::Concatenate { dim }) => builder.concatenate(&sources, dim),
            NodeOp::Merge(MergeKind::Add) => builder.add(&sources),
            NodeOp::Merge(MergeKind::Multiply) => builder.multiply(&sources),
            NodeOp::Merge(MergeKind::Average) => builder.average(&sources),
        }
        .map_err(ModelIoError::Autodiff)?;
        handles.push(handle);
    }
    let pick = |indices: &[usize], what: &str| -> Result<Vec<Node>, ModelIoError> {
        indices
            .iter()
            .map(|i| {
                handles
                    .get(*i)
                    .copied()
                    .ok_or_else(|| manifest_error(format!("{what} の添字 {i} が範囲外です")))
            })
            .collect()
    };
    let input_nodes = pick(&manifest.inputs, "inputs")?;
    let output_nodes = pick(&manifest.outputs, "outputs")?;
    let mut model = builder
        .build(&input_nodes, &output_nodes)
        .map_err(ModelIoError::Autodiff)?;
    model.set_training(manifest.training);

    if let Some(meta) = manifest.compiled {
        // optimizer の load はスロット内の整合しか見ないため、パラメータとの数・shape の照合は先に行う。
        // 復元は construct-before-assign（失敗時は部分状態を返さない）。
        check_slot_shapes(&optimizer_state, &model.trainable_parameters())?;
        model
            .restore_compile_state(meta.into_snapshot(optimizer_state))
            .map_err(ModelIoError::Autodiff)?;
    }
    Ok(model)
}

/// 通し番号キー `"{g}.{name}"` を所有ブロックの序数とローカルキーへ写す。
fn localize_key(
    key: &str,
    owner: &[usize],
    block_nodes: &[&NodeRecord],
) -> Result<(usize, String), ModelIoError> {
    let bad = || ModelIoError::Mismatch {
        message: "manifest のキーが層番号の形式 \"{index}.{name}\" ではありません".into(),
    };
    let (index, _) = key.split_once('.').ok_or_else(bad)?;
    let layer: usize = index.parse().map_err(|_| bad())?;
    let ordinal = *owner.get(layer).ok_or_else(bad)?;
    let node = block_nodes.get(ordinal).ok_or_else(bad)?;
    let local = to_local_key(node.layer_start, key).ok_or_else(bad)?;
    Ok((ordinal, local))
}

// ---------------------------------------------------------------------
// manifest の厳格パース・構造検証
// ---------------------------------------------------------------------

/// 添字列（非負整数の配列）を読む。
fn parse_index_list(value: &Json, ctx: &str) -> Result<Vec<usize>, ModelIoError> {
    let arr = as_arr(value, ctx)?;
    let mut out = Vec::new();
    out.try_reserve_exact(arr.len())
        .map_err(|_| ModelIoError::TooLarge {
            what: "JSON 配列の要素数",
            limit: MAX_ARRAY_LEN as u64,
        })?;
    for item in arr {
        out.push(as_usize(item, &format!("{ctx}[]"))?);
    }
    Ok(out)
}

/// manifest のバイト列を厳格に検証して [`ParsedFunctional`] にする。グラフの構造検証（`nodes`・
/// `inputs`・`outputs`）はここで完了し、safetensors には触れない。
fn parse_functional_manifest(bytes: &[u8]) -> Result<ParsedFunctional, ModelIoError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| manifest_error("manifest が UTF-8 ではありません"))?;
    let root = Parser {
        src: text.as_bytes(),
        pos: 0,
    }
    .document()?;
    let f = exact_fields(
        &root,
        "manifest",
        &[
            "format",
            "format_version",
            "training",
            "num_layers",
            "layers",
            "nodes",
            "inputs",
            "outputs",
            "parameter_keys",
            "buffer_keys",
            "safetensors_file",
            "safetensors_bytes",
            "compiled",
        ],
    )?;
    if as_str(f[0], "format")? != FUNCTIONAL_FORMAT_NAME {
        return Err(manifest_error("format が想定と異なります"));
    }
    if as_u64(f[1], "format_version")? != FUNCTIONAL_FORMAT_VERSION {
        return Err(manifest_error("format_version が未対応です"));
    }
    let Json::Bool(training) = f[2] else {
        return Err(manifest_error("training は bool である必要があります"));
    };
    let num_layers = as_usize(f[3], "num_layers")?;
    let layers = as_arr(f[4], "layers")?;
    let node_items = as_arr(f[5], "nodes")?;
    let inputs = parse_index_list(f[6], "inputs")?;
    let outputs = parse_index_list(f[7], "outputs")?;
    let parameter_keys = as_arr(f[8], "parameter_keys")?;
    let buffer_keys = as_arr(f[9], "buffer_keys")?;
    let safetensors_file = as_str(f[10], "safetensors_file")?;
    let safetensors_bytes = as_u64(f[11], "safetensors_bytes")?;

    // 層数は配列を 1 要素ずつ数えた結果だけを信じる（`num_layers` は一致確認のみ）。
    if layers.len() > MAX_LAYERS {
        return Err(ModelIoError::TooLarge {
            what: "層数",
            limit: MAX_LAYERS as u64,
        });
    }
    if num_layers != layers.len() {
        return Err(manifest_error(
            "num_layers が layers の要素数と一致しません",
        ));
    }
    if !is_valid_safetensors_file_name(safetensors_file) {
        return Err(manifest_error(
            "safetensors_file が model.<32 桁 16 進>.safetensors の形式ではありません",
        ));
    }

    let nodes = parse_nodes(node_items, layers.len())?;
    validate_graph(&nodes, &inputs, &outputs)?;
    let block_count = nodes.iter().filter(|n| n.op == NodeOp::Block).count();
    if block_count == 0 && !*training {
        // ブロック 0 個のモデルの `training()` は導出値 `true`（`false` は復元後に一致しない）。
        return Err(manifest_error(
            "ブロックを持たないグラフの training は true である必要があります",
        ));
    }

    // 旧形式（buffer_keys の欠落許容）は Functional 形式に存在しない。
    let ParsedLayers {
        specs,
        parameter_keys,
        buffer_keys,
    } = parse_layers_and_keys(layers, parameter_keys, buffer_keys, false)?;

    let compiled = parse_compiled(f[12])?;
    if let Some(meta) = &compiled
        && (matches!(meta.optimizer, Optimizer::Lbfgs(_)) || meta.amp.is_some())
    {
        return Err(ModelIoError::UnsupportedModel {
            reason: "Functional モデルは Lbfgs・AMP を含む compile 状態を復元できません".into(),
        });
    }

    Ok(ParsedFunctional {
        training: *training,
        specs,
        nodes,
        inputs,
        outputs,
        parameter_keys,
        buffer_keys,
        safetensors_file: safetensors_file.to_string(),
        safetensors_bytes,
        compiled,
    })
}

/// `nodes` 配列の各要素を検証して [`NodeRecord`] 列にする。`layer_count` は `layers` の件数。
///
/// 規則（すべて manifest の数値を確保量の根拠にしない）: `index` は連番・`op` は文字列 allowlist
/// （未知は `UnsupportedModel`）・`inputs[j] < index`（前方参照禁止）・`input` は入力 0 件／
/// `layer_len == 0`・`block` は入力 1 件／`layer_len >= 1`・結合は入力 2 件以上で重複なし／
/// `layer_len == 0`・`layer_start` は累積値と一致・総和が `layer_count` と一致。
fn parse_nodes(items: &[Json], layer_count: usize) -> Result<Vec<NodeRecord>, ModelIoError> {
    let mut nodes: Vec<NodeRecord> = Vec::new();
    nodes
        .try_reserve_exact(items.len())
        .map_err(|_| ModelIoError::TooLarge {
            what: "ノード数",
            limit: MAX_ARRAY_LEN as u64,
        })?;
    // 結合入力の重複検出に使う作業領域（要素数はノード数。線形時間）。
    let mut seen = vec![0usize; items.len()];
    let mut cumulative = 0usize;
    for (i, item) in items.iter().enumerate() {
        let nf = exact_fields(
            item,
            "nodes[]",
            &[
                "index",
                "op",
                "inputs",
                "layer_start",
                "layer_len",
                "params",
            ],
        )?;
        if as_usize(nf[0], "nodes[].index")? != i {
            return Err(manifest_error("nodes[].index が連番ではありません"));
        }
        let op_name = as_str(nf[1], "nodes[].op")?;
        let inputs = parse_index_list(nf[2], "nodes[].inputs")?;
        let layer_start = as_usize(nf[3], "nodes[].layer_start")?;
        let layer_len = as_usize(nf[4], "nodes[].layer_len")?;
        let params = nf[5];

        let op = match op_name {
            "input" | "block" | "add" | "multiply" | "average" => {
                exact_fields(params, "nodes[].params", &[])?;
                match op_name {
                    "input" => NodeOp::Input,
                    "block" => NodeOp::Block,
                    "add" => NodeOp::Merge(MergeKind::Add),
                    "multiply" => NodeOp::Merge(MergeKind::Multiply),
                    _ => NodeOp::Merge(MergeKind::Average),
                }
            }
            "concatenate" => {
                let pf = exact_fields(params, "nodes[].params", &["dim"])?;
                NodeOp::Merge(MergeKind::Concatenate {
                    dim: as_usize(pf[0], "nodes[].params.dim")?,
                })
            }
            _ => {
                return Err(ModelIoError::UnsupportedModel {
                    reason: format!("未対応のノード種別 {} です", super::clip(op_name)),
                });
            }
        };

        for src in &inputs {
            if *src >= i {
                return Err(manifest_error(format!(
                    "nodes[{i}] の入力 {src} が自ノード以前を参照しています（前方参照・自己参照は不可）"
                )));
            }
        }
        let (arity_ok, len_ok) = match op {
            NodeOp::Input => (inputs.is_empty(), layer_len == 0),
            NodeOp::Block => (inputs.len() == 1, layer_len >= 1),
            NodeOp::Merge(_) => (inputs.len() >= 2, layer_len == 0),
        };
        if !arity_ok {
            return Err(manifest_error(format!(
                "nodes[{i}] の入力件数が op {op_name} の規則に合いません"
            )));
        }
        if !len_ok {
            return Err(manifest_error(format!(
                "nodes[{i}] の layer_len が op {op_name} の規則に合いません"
            )));
        }
        if matches!(op, NodeOp::Merge(_)) {
            // 作業領域の値に「最後に見たノード添字 + 1」を書き、同一ノードの再出現を検出する。
            for src in &inputs {
                let slot = seen
                    .get_mut(*src)
                    .ok_or_else(|| manifest_error("nodes[].inputs が範囲外です"))?;
                if *slot == i + 1 {
                    return Err(manifest_error(format!(
                        "nodes[{i}] の入力 {src} が重複しています"
                    )));
                }
                *slot = i + 1;
            }
        }
        if layer_start != cumulative {
            return Err(manifest_error(format!(
                "nodes[{i}] の layer_start が累積層数と一致しません（層範囲は重複・隙間なし）"
            )));
        }
        cumulative = cumulative
            .checked_add(layer_len)
            .ok_or_else(|| manifest_error("layer_len の合計がオーバーフローします"))?;
        nodes.push(NodeRecord {
            op,
            inputs,
            layer_start,
            layer_len,
        });
    }
    if cumulative != layer_count {
        return Err(manifest_error(
            "ノードの層範囲の合計が layers の要素数と一致しません",
        ));
    }
    Ok(nodes)
}

/// グラフ全体の整合（`FunctionalBuilder::build` と同じ規則を safetensors を開く前に確認する）:
/// `inputs` は入力ノードだけを重複なく列挙し全入力ノードを含む・`outputs` は範囲内で重複なし・
/// 全ノードがいずれかの出力へ寄与する。
fn validate_graph(
    nodes: &[NodeRecord],
    inputs: &[usize],
    outputs: &[usize],
) -> Result<(), ModelIoError> {
    if inputs.is_empty() {
        return Err(manifest_error("inputs が空です"));
    }
    if outputs.is_empty() {
        return Err(manifest_error("outputs が空です"));
    }
    let mut listed = vec![false; nodes.len()];
    for index in inputs {
        let is_input = nodes.get(*index).is_some_and(|n| n.op == NodeOp::Input);
        if !is_input {
            return Err(manifest_error(format!(
                "inputs の添字 {index} が入力ノードではありません"
            )));
        }
        let slot = listed
            .get_mut(*index)
            .ok_or_else(|| manifest_error("inputs の添字が範囲外です"))?;
        if *slot {
            return Err(manifest_error(format!("inputs の添字 {index} が重複")));
        }
        *slot = true;
    }
    let input_nodes = nodes.iter().filter(|n| n.op == NodeOp::Input).count();
    if input_nodes != inputs.len() {
        return Err(manifest_error(
            "inputs に列挙されていない入力ノードがあります（未束縛）",
        ));
    }
    let mut reached = vec![false; nodes.len()];
    for index in outputs {
        let slot = reached
            .get_mut(*index)
            .ok_or_else(|| manifest_error(format!("outputs の添字 {index} が範囲外")))?;
        if *slot {
            return Err(manifest_error(format!("outputs の添字 {index} が重複")));
        }
        *slot = true;
    }
    for index in (0..nodes.len()).rev() {
        if !reached.get(index).copied().unwrap_or(false) {
            continue;
        }
        if let Some(node) = nodes.get(index) {
            for src in &node.inputs {
                if let Some(slot) = reached.get_mut(*src) {
                    *slot = true;
                }
            }
        }
    }
    if let Some(dead) = reached.iter().position(|r| !*r) {
        return Err(manifest_error(format!(
            "ノード {dead} はどの出力にも寄与しません"
        )));
    }
    Ok(())
}
