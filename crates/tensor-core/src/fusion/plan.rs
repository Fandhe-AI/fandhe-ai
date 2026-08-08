//! 融合プランの公開 DTO・`FusionSession`／`FusionValue`（TASK-12.1d・#164）。
//!
//! `docs/fusion-graph-design.md` §3.4 の型スケッチをそのまま実装する。
//! [`FusionPlan`] は `BackendOps::run_fused`（`backend_ops.rs`）の
//! シグネチャに現れる `pub` 型であり、`FusionGraph`／`FusionNode`／
//! `FusionOp`（`graph.rs`。`pub(crate)` のまま変更しない）を非公開
//! フィールドとして包む不透明ハンドルである（privacy 制約。同モジュール
//! 冒頭コメント参照）。読み取りは `impl FusionPlan` の `pub` DTO
//! アクセサ（`ops`／`output_shape`／`dtype`／`leaf_count`／`use_count`）
//! 経由のみで行う。
//!
//! 構築経路は 2 系統（設計書 §3.4）:
//! - [`FusionPlan::from_graph`]（`pub(crate)`）: `tensor-core` 内で
//!   `FusionGraph` から構築する経路（#162 の連鎖検出アルゴリズムが
//!   `tensor-core` 内で完結して使う将来のユースケースに備える）。
//! - [`FusionPlan::from_ops`]（`pub` + `#[doc(hidden)]`）: `autodiff`
//!   クレート専用の構築経路。`tensor-core` 内部の `pub(crate)` 型を
//!   一切経由せず、既に `pub` な DTO のみから直接構築する
//!   （REQ-12「利用者が明示的に融合を制御する API は提供しない」への
//!   抵触を避けるため `#[doc(hidden)]` を付す）。

use crate::Tensor;
use crate::backend_ops::BackendOps;
use crate::device::BackendError;
use crate::dispatch::DType;

use super::graph::{FusionGraph, FusionNodeId, FusionOp};

/// `FusionPlan` 内のノード位置を指す公開インデックス。内部の
/// `FusionNodeId`（`pub(crate)`、`graph.rs`）はそのまま公開できないため、
/// `FusionPlan` 内でのみ意味を持つ 0 起点の連番（発生順）として別の型を
/// 用意する（設計書 §3.4）。
pub type FusedNodeIndex = usize;

/// `FusionPlan::ops`（下記）が列挙する 1 ノード分の演算内容。内部
/// `pub(crate)` の `FusionOp`（`graph.rs`）と 1:1 対応するが、融合境界
/// ノード（`Gemm`／`Sum`／`Max`）は実体化境界のため融合対象区間そのもの
/// には含まれず列挙しない（設計書 §3.4）。
#[derive(Debug, Clone, Copy)]
pub enum FusedOpKind {
    /// 葉ノード（このプランへの外部入力）。`leaf_index` は `run_fused`
    /// の `leaves: &[&Tensor<f32>]` の添字と対応する。
    Input {
        leaf_index: FusedNodeIndex,
    },
    Add {
        lhs: FusedNodeIndex,
        rhs: FusedNodeIndex,
    },
    Mul {
        lhs: FusedNodeIndex,
        rhs: FusedNodeIndex,
    },
    Relu {
        input: FusedNodeIndex,
    },
    Exp {
        input: FusedNodeIndex,
    },
    Tanh {
        input: FusedNodeIndex,
    },
}

/// `run_fused`（`BackendOps` の非破壊拡張）へ渡す公開の不透明ハンドル。
/// 内部の融合 IR（`FusionGraph`／`FusionNode`／`FusionOp`。`pub(crate)`
/// のまま変更しない）をフィールドとして非公開のまま包む。
pub struct FusionPlan {
    graph: FusionGraph,
    /// このプランが表す出力ノード（`graph` 内の root）。`ops()`／
    /// `output_shape()`／`dtype()` はこのノードを起点に読む。
    root: FusionNodeId,
    /// このプランが要求する葉ノード数（`from_ops` 経由の構築では
    /// `leaves()` を持たない `FusionGraph`〈`autodiff` 由来〉のため、
    /// 別途保持する）。`from_graph` 経由の場合は `graph.leaves().len()`
    /// と一致する。
    leaf_count: usize,
}

impl FusionPlan {
    /// `graph` のうち `root` を出力とする部分グラフから融合対象区間を
    /// 切り出し、`FusionPlan` を構築する（`pub(crate)`。`tensor-core` 内で
    /// `FusionGraph` が既に存在する場合の構築経路。#162 の連鎖検出結果
    /// をそのまま包む用途）。
    pub(crate) fn from_graph(graph: &FusionGraph, root: FusionNodeId) -> FusionPlan {
        let leaf_count = graph
            .nodes_ref()
            .iter()
            .filter(|node| matches!(node.op, FusionOp::Input))
            .count();
        FusionPlan {
            graph: graph.clone(),
            root,
            leaf_count,
        }
    }

    /// `autodiff` クレート専用の構築経路（設計書 §3.4）。`tensor-core`
    /// 内部の `pub(crate)` 型を一切経由せず、既に `pub` な DTO のみから
    /// 直接 `FusionPlan` を組み立てる。`autodiff` は自身が保持する
    /// `TapeNode`／`Op` の遅延連鎖をこの `ops` へ変換して渡す
    /// （`Op::Relu`/`Add`/`Mul`/`Exp`/`Tanh` と `FusedOpKind` は 1:1
    /// 対応）。
    ///
    /// `#[doc(hidden)]`: この経路は `autodiff` という単一の内部利用者の
    /// ためのクレート間契約であり、利用者向けの融合制御 API ではない
    /// （REQ-12 への抵触を避けるため、`pub` API ドキュメントには現れない
    /// 内部専用シグネチャとして扱う）。
    ///
    /// `ops` が参照する `FusedNodeIndex`（`lhs`／`rhs`／`input`・
    /// `leaf_index`）は必ず自ノードより小さい既存ノードまたは葉を指す
    /// 契約とする。範囲外参照は `FusionGraph::push` と同じ検証（構築時に
    /// 検証を先行させる。`.claude/rules/security.md` A03 観点）で
    /// 拒否する。`leaf_count` は `leaves: &[&Tensor<f32>]`（`run_fused`）
    /// の長さとの一致契約を表す。
    #[doc(hidden)]
    pub fn from_ops(
        ops: Vec<FusedOpKind>,
        output_shape: Vec<usize>,
        dtype: DType,
        leaf_count: usize,
    ) -> Result<FusionPlan, BackendError> {
        let mut graph = FusionGraph::new();
        // 葉ノードを先に登録する（`FusedOpKind::Input { leaf_index }` が
        // 参照する `FusedNodeIndex` の並びを `graph` 内の `FusionNodeId`
        // と一致させるため。出力 shape は葉個々には持たせず、全体の
        // `output_shape` を root ノードにのみ持たせる契約とする——各葉の
        // 個別 shape は `run_fused` 実装側が `leaves` から直接読む）。
        let mut leaf_ids = Vec::with_capacity(leaf_count);
        for _ in 0..leaf_count {
            let id = graph
                .push(
                    FusionOp::Input,
                    super::graph::NodeMeta::new(output_shape.clone(), true, dtype),
                )
                .map_err(|err| {
                    BackendError::ShapeMismatch(crate::error::ShapeError::ElementCountMismatch {
                        expected: leaf_count,
                        actual: err.to_string().len().max(1),
                    })
                })?;
            leaf_ids.push(id);
        }

        // `FusedNodeIndex`（0 起点。`ops` 内での発生順）から `graph` 内
        // `FusionNodeId` への対応表。葉は `leaf_ids` の先頭
        // `leaf_count` 個で確定済みのため、非葉ノードの結果をここへ
        // 追記していく。
        let mut resolved: Vec<FusionNodeId> = leaf_ids.clone();
        let mut root = None;
        for (idx, kind) in ops.into_iter().enumerate() {
            let resolve = |i: FusedNodeIndex,
                           resolved: &[FusionNodeId]|
             -> Result<FusionNodeId, BackendError> {
                resolved.get(i).copied().ok_or(BackendError::ShapeMismatch(
                    crate::error::ShapeError::ElementCountMismatch {
                        expected: resolved.len(),
                        actual: i,
                    },
                ))
            };
            let node_id = match kind {
                FusedOpKind::Input { leaf_index } => resolve(leaf_index, &resolved)?,
                FusedOpKind::Add { lhs, rhs } => {
                    let a = resolve(lhs, &resolved)?;
                    let b = resolve(rhs, &resolved)?;
                    graph
                        .push(
                            FusionOp::Add(a, b),
                            super::graph::NodeMeta::new(output_shape.clone(), true, dtype),
                        )
                        .map_err(from_ops_err)?
                }
                FusedOpKind::Mul { lhs, rhs } => {
                    let a = resolve(lhs, &resolved)?;
                    let b = resolve(rhs, &resolved)?;
                    graph
                        .push(
                            FusionOp::Mul(a, b),
                            super::graph::NodeMeta::new(output_shape.clone(), true, dtype),
                        )
                        .map_err(from_ops_err)?
                }
                FusedOpKind::Relu { input } => {
                    let a = resolve(input, &resolved)?;
                    graph
                        .push(
                            FusionOp::Relu(a),
                            super::graph::NodeMeta::new(output_shape.clone(), true, dtype),
                        )
                        .map_err(from_ops_err)?
                }
                FusedOpKind::Exp { input } => {
                    let a = resolve(input, &resolved)?;
                    graph
                        .push(
                            FusionOp::Exp(a),
                            super::graph::NodeMeta::new(output_shape.clone(), true, dtype),
                        )
                        .map_err(from_ops_err)?
                }
                FusedOpKind::Tanh { input } => {
                    let a = resolve(input, &resolved)?;
                    graph
                        .push(
                            FusionOp::Tanh(a),
                            super::graph::NodeMeta::new(output_shape.clone(), true, dtype),
                        )
                        .map_err(from_ops_err)?
                }
            };
            // `resolved[leaf_count + idx]` が今追加したノードを指す
            // （`FusedNodeIndex` は葉に続けて非葉ノードも連番で参照する
            // 契約。`from_ops` 呼び出し元〈`autodiff`〉はこの並びで
            // `FusedOpKind` 列を構築する）。
            debug_assert_eq!(resolved.len(), leaf_count + idx);
            resolved.push(node_id);
            root = Some(node_id);
        }

        let root = root.unwrap_or_else(|| {
            // `ops` が空（葉 1 個をそのまま返す恒等プラン）の場合は
            // 最後の葉を root とする。空 `ops` は `autodiff` 側の走査
            // ロジックでは通常発生しないが、`FusionPlan` 単体としては
            // 妥当な最小プランのため防御的に扱う。
            leaf_ids.last().copied().unwrap_or(FusionNodeId(0))
        });

        Ok(FusionPlan {
            graph,
            root,
            leaf_count,
        })
    }

    /// 発生順（トポロジカル順）で `FusedOpKind` を列挙する。境界ノード
    /// （Gemm／Sum／Max）はプラン内に現れないため列挙対象外（本実装は
    /// 常に elementwise 5 演算のみを保持するため対象外ノード自体を
    /// 持たない）。葉ノード（`FusionOp::Input`）も `FusedOpKind::Input`
    /// として列挙する（`run_fused` 実装がプラン内の全ノードを発生順に
    /// 走査できるようにするため）。
    pub fn ops(&self) -> impl Iterator<Item = FusedOpKind> + '_ {
        let nodes = self.graph.nodes_ref();
        let leaf_index_of = |id: FusionNodeId| -> FusedNodeIndex {
            nodes[..id.0]
                .iter()
                .filter(|n| matches!(n.op, FusionOp::Input))
                .count()
        };
        nodes.iter().enumerate().filter_map(move |(idx, node)| {
            let this_idx = idx;
            match &node.op {
                FusionOp::Input => Some(FusedOpKind::Input {
                    leaf_index: leaf_index_of(FusionNodeId(this_idx)),
                }),
                FusionOp::Add(a, b) => Some(FusedOpKind::Add { lhs: a.0, rhs: b.0 }),
                FusionOp::Mul(a, b) => Some(FusedOpKind::Mul { lhs: a.0, rhs: b.0 }),
                FusionOp::Relu(a) => Some(FusedOpKind::Relu { input: a.0 }),
                FusionOp::Exp(a) => Some(FusedOpKind::Exp { input: a.0 }),
                FusionOp::Tanh(a) => Some(FusedOpKind::Tanh { input: a.0 }),
                FusionOp::Gemm(..) | FusionOp::Sum { .. } | FusionOp::Max { .. } => None,
            }
        })
    }

    /// このプランが表す出力テンソルの shape（root ノードの `NodeMeta.shape`）。
    pub fn output_shape(&self) -> &[usize] {
        self.graph
            .node(self.root)
            .map(|n| n.meta.shape.as_slice())
            .unwrap_or(&[])
    }

    /// このプランの dtype（root ノードの `NodeMeta.dtype`）。現状は常に
    /// `DType::F32`。
    pub fn dtype(&self) -> DType {
        self.graph
            .node(self.root)
            .map(|n| n.meta.dtype)
            .unwrap_or(DType::F32)
    }

    /// このプランが要求する葉ノード（外部入力）の個数。`run_fused` の
    /// `leaves: &[&Tensor<f32>]` の長さはこの値と一致する契約とする。
    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }

    /// 指定ノードの被参照数（プラン内からの被参照数のみ）。
    pub fn use_count(&self, node: FusedNodeIndex) -> usize {
        self.graph
            .node(FusionNodeId(node))
            .map(|n| n.use_count)
            .unwrap_or(0)
    }
}

fn from_ops_err(err: super::graph::FusionGraphError) -> BackendError {
    match err {
        super::graph::FusionGraphError::Shape(shape_err) => BackendError::ShapeMismatch(shape_err),
        super::graph::FusionGraphError::NodeIdOutOfRange { id, len } => {
            BackendError::ShapeMismatch(crate::error::ShapeError::ElementCountMismatch {
                expected: len,
                actual: id,
            })
        }
    }
}

/// 単一の fallible 呼び出しの実行スタック内だけで構築・破棄される、
/// 融合対象区間 1 本分のグラフビルダー（設計書 §3.4）。呼び出し元の
/// 関数フレームを越えて共有・保持されることはなく、`Tensor`／`Storage`
/// のどのフィールドにも格納されない。`Arc`／`Mutex`／`Send + Sync`
/// 境界は一切不要（`ops` は借用で足りる）。
///
/// `tensor-core` 内で `FusionGraph` が既に存在する場合（#162 の連鎖検出
/// アルゴリズムが `tensor-core` 内で完結して使う将来のユースケース）の
/// ための内部機構として残す。`autodiff::Tape` の実体化は本型を経由せず
/// `FusionPlan::from_ops` + `BackendOps::run_fused` を直接呼ぶ
/// （`tensor-core` → `autodiff` の逆依存を作れないため。設計書 §3.4）。
///
/// 葉ノードの実体 `Tensor<f32>` は `graph`（`FusionNode.meta` は shape
/// 等のメタデータのみを持ち実体は持たない。`graph.rs` 冒頭コメント参照）
/// ではなく `leaves`（本フィールド。`FusionOp::Input` の追加順に対応する
/// 発生順 `Vec`）に保持する。`FusionGraph::leaves()`（設計書 §3.4 の
/// スケッチが指す、`tensor-core` 内でのグラフ構築側がノード追加と同時に
/// 葉実体を記録する導線）は融合カーネル生成本体を担う #163 のスコープで
/// あり本イシュー（#164）時点では未実装のため、`FusionSession` 自身に
/// 葉実体を持たせる最小構成とする。
pub(crate) struct FusionSession<'ops> {
    graph: FusionGraph,
    leaves: Vec<Tensor<f32>>,
    ops: &'ops dyn BackendOps,
}

/// グラフ構築中に扱う 1 つの中間値。既に確定済みの `Tensor<f32>` か、
/// `session` 内にまだ実行していないノードとして積まれているか
/// （`Pending`）のいずれか。
pub(crate) enum FusionValue {
    Materialized(Tensor<f32>),
    Pending(FusionNodeId),
}

impl<'ops> FusionSession<'ops> {
    #[allow(dead_code)]
    pub(crate) fn new(ops: &'ops dyn BackendOps) -> Self {
        Self {
            graph: FusionGraph::new(),
            leaves: Vec::new(),
            ops,
        }
    }

    /// 外部入力（葉ノード）を登録し、その `FusionNodeId` を返す。
    #[allow(dead_code)]
    pub(crate) fn push_leaf(&mut self, tensor: Tensor<f32>, dtype: DType) -> FusionNodeId {
        let meta = super::graph::NodeMeta::new(tensor.shape().to_vec(), true, dtype);
        // `FusionOp::inputs()` は葉ノードに対し常に空 `Vec` を返す
        // （`graph.rs` 参照）ため、`push` の入力検証は構造的に失敗しない。
        // それでも本番経路 panic 禁止方針に従い、失敗時は直前のノード数
        // から採番される想定 ID をそのまま返す安全側フォールバックとする。
        let id = self
            .graph
            .push(FusionOp::Input, meta)
            .unwrap_or(FusionNodeId(self.graph.len()));
        self.leaves.push(tensor);
        id
    }

    /// §3.2 の実体化条件のいずれかに到達した時点、または呼び出し元の
    /// fallible 関数が自身の結果を返す直前に呼ぶ。`Materialized` は
    /// そのまま返し、`Pending` は `self.graph`／`self.ops` を使って
    /// 実際に計算する（`Unsupported` の場合は `graph` を発生順に辿り
    /// per-op メソッドへ逐次フォールバックする。設計書 §3.4）。
    #[allow(dead_code)]
    pub(crate) fn materialize(&self, value: FusionValue) -> Result<Tensor<f32>, BackendError> {
        match value {
            FusionValue::Materialized(t) => Ok(t),
            FusionValue::Pending(node) => {
                let plan = FusionPlan::from_graph(&self.graph, node);
                let leaves: Vec<&Tensor<f32>> = self.leaves.iter().collect();
                match self.ops.run_fused(&plan, &leaves) {
                    Ok(t) => Ok(t),
                    Err(BackendError::Unsupported(_)) => self.materialize_fallback(node),
                    Err(other) => Err(other),
                }
            }
        }
    }

    /// `run_fused` が `Unsupported` を返した場合の per-op 逐次フォール
    /// バック（設計書 §4 の fail-safe 方針）。
    fn materialize_fallback(&self, node: FusionNodeId) -> Result<Tensor<f32>, BackendError> {
        // ノード ID は発生順（`FusionGraph::push` の不変条件）のため、
        // 0 から node.0 まで昇順に辿れば入力が必ず先に計算済みになる。
        let mut computed: std::collections::HashMap<usize, Tensor<f32>> =
            std::collections::HashMap::new();
        let mut leaf_cursor = 0usize;
        for id in 0..=node.0 {
            let n = self
                .graph
                .node(FusionNodeId(id))
                .map_err(|_| BackendError::DeviceMismatch)?;
            let value = match &n.op {
                FusionOp::Input => {
                    let t = self
                        .leaves
                        .get(leaf_cursor)
                        .cloned()
                        .ok_or(BackendError::DeviceMismatch)?;
                    leaf_cursor += 1;
                    t
                }
                FusionOp::Add(a, b) => self.ops.add(&computed[&a.0], &computed[&b.0])?,
                FusionOp::Mul(a, b) => self.ops.mul(&computed[&a.0], &computed[&b.0])?,
                FusionOp::Relu(a) => self.ops.relu(&computed[&a.0])?,
                FusionOp::Exp(a) => self.ops.exp(&computed[&a.0])?,
                FusionOp::Tanh(a) => self.ops.tanh(&computed[&a.0])?,
                FusionOp::Gemm(..) | FusionOp::Sum { .. } | FusionOp::Max { .. } => {
                    return Err(BackendError::Unsupported(
                        "fusion fallback: boundary node inside segment".into(),
                    ));
                }
            };
            computed.insert(id, value);
        }
        computed.remove(&node.0).ok_or(BackendError::DeviceMismatch)
    }
}
