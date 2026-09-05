//! 動的テープ（Wengert list）本体。
//!
//! PoC-v2-2 が確定した「動的テープ式」記録方式
//! （`docs/spec/03-poc/poc-v2-2-autodiff/README.md:170`）を
//! `docs/public-api-design.md` §3.1 の productize 済み API 形状で実装する。
//! `Var`（`var.rs`）の演算メソッドが `Tape::push`／`push_lazy`（下記）を
//! 呼んで発生順に `TapeNode` を追記し、`Tape::backward`（`backward.rs`・
//! TASK-1.5c・#18）はこの記録を逆走査して勾配を計算する（`Op` が入力
//! `NodeId` を保持するため、逆走査の各ノードから入力ノードを直接辿れる）。
//!
//! **TASK-12.1d（#164）による拡張**: `Tape` は `ops: Box<dyn BackendOps +
//! Send>`（バックエンド実行の必須所有値）を保持し、`add`／`mul`／
//! `relu`／`exp`／`tanh`（elementwise 5 演算）は forward 値計算を
//! 即座に行わず `TapeNode::value`（`OnceCell`）を空のまま記録する
//! （遅延グラフの延長。`docs/fusion-graph-design.md` §3.5.1）。この
//! 遅延グラフを `matmul`／`sum`／`max`・`Tape::backward`・`Var::value`／
//! `to_tensor` が読み出す際に本ファイルの [`materialize_fallible`]／
//! [`materialize_non_fallible`] が実体化する（同 §3.5.2・§3.5.3）。
//! `FusionPlan::from_ops` + `BackendOps::run_fused`（`tensor-core`）を
//! 試み、`Unsupported` の場合は `ops` 自身の per-op メソッドへ逐次
//! フォールバックする。

use std::cell::{Cell, OnceCell, RefCell};
use std::sync::atomic::{AtomicU64, Ordering};

use fandhe_ai_tensor_core::{
    Activation, BackendError, BackendOps, DType, DeviceBufferView, FusedOpKind, FusionPlan,
    MAX_FUSED_CHAIN_LEN, Tensor,
};

use crate::error::AutodiffError;

/// テープの識別子。プロセス全体で単調増加するカウンタから発行する。
///
/// ポインタ等値（`ptr::eq`）ではなく専用 ID を用いる理由: スコープ末で
/// 破棄された `Tape` のメモリ領域は後続の `Tape::new_with_ops(ops)` に再利用され
/// うるため、ポインタ比較は別テープを同一と誤判定する（false positive）
/// 余地が残る（`docs/public-api-design.md` §3.1）。単調増加 ID は
/// プロセス生存中に衝突しない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TapeId(u64);

/// プロセス全体で共有する `TapeId` 発行カウンタ。`Tape::new_with_ops(ops)` からのみ
/// インクリメントされる（`fetch_add` は複数スレッドから並行に `Tape` を
/// 生成しても一意性を保つ）。
static NEXT_TAPE_ID: AtomicU64 = AtomicU64::new(0);

/// テープ内ノードの識別子。`nodes: Vec<TapeNode>` への添字を直接表す
/// newtype（`docs/public-api-design.md` §3.1 は `pub type` 相当だが、
/// 生の `usize` と取り違えないよう newtype 化する）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeId(pub(crate) usize);

/// テープへ記録される演算の種別。入力を `NodeId` で保持することで、
/// `Tape::backward`（`backward.rs`・#18・TASK-1.5c）が発生順とは逆順に
/// ノード列を走査し、各ノードの出力側勾配を入力ノードへ伝播できる。
///
/// クレート非公開: `Var`（`var.rs`）の演算メソッドのみが構築し、
/// 呼び出し側（autodiff クレートの利用者）には値と shape のみが
/// `Var::value`/`to_tensor` 経由で見える設計とする
/// （`docs/public-api-design.md` §3.2 の演算セットに 1:1 対応）。
///
/// 各 variant の入力 `NodeId`/`dim` フィールドは `grad.rs`（#17・
/// TASK-1.5b）の `vjp()` ディスパッチが読み出す（`op.clone()` して各
/// variant を分解し、入力側 `NodeId` へ勾配を割り当てる）。
/// `Tape::backward`（`backward.rs`・#18・TASK-1.5c）はノード列を逆走査
/// しながら `vjp()` を呼び、返った寄与を入力ノードへ蓄積する側。
///
/// **`Copy` を持たない理由**: `CrossEntropyLoss`（#191）の `targets` は
/// クラス添字（`Tensor<i32>`）を直接保持する非追跡ペイロードであり、
/// `Tensor<T>` は `Clone` のみ（`Copy` 不可）のため `Op` 全体も
/// `Clone` に留める（`grad.rs::vjp` は `op: &Op` を受け取り
/// `op.clone()` してから `match` する）。
#[derive(Debug, Clone)]
pub(crate) enum Op {
    /// `Tape::var()` が登録する非追跡入力（葉ノード）。逆伝播の起点
    /// にはなるが、それ自体の入力ノードは持たない。常に実体化済み
    /// （`push_eager`。TASK-12.1d・#164）。
    Leaf,
    /// 非 elementwise。常に実体化済み（`push_eager`）。
    MatMul(NodeId, NodeId),
    /// elementwise binary。遅延評価対象（`push_lazy`。TASK-12.1d・#164）。
    Add(NodeId, NodeId),
    /// elementwise binary。遅延評価対象。
    Mul(NodeId, NodeId),
    /// elementwise unary。遅延評価対象。
    Relu(NodeId),
    /// elementwise unary。遅延評価対象。
    Exp(NodeId),
    /// elementwise unary。遅延評価対象。
    Tanh(NodeId),
    /// シグモイド（`1 / (1 + exp(-x))`）。TASK-9.1b（#92）で
    /// `nn::activation::Sigmoid`（`nn/activation.rs`）から使う活性化
    /// プリミティブとして追加。`BackendOps` に対応メソッドがないため
    /// 融合対象外とし常に実体化済み（`push_eager`。TASK-12.1d・#164）。
    /// 数値安定形の forward は `eval.rs`（`eval::sigmoid`）、VJP は
    /// `grad.rs`（`out_value` 再利用方式）を参照。
    Sigmoid(NodeId),
    /// 非 elementwise。常に実体化済み。
    Sum { input: NodeId, dim: Option<usize> },
    /// 非 elementwise。常に実体化済み。
    Max { input: NodeId, dim: Option<usize> },
    /// 平均二乗誤差。`BackendOps` に対応メソッドがないため融合対象外
    /// とし常に実体化済み（`push_eager`）。`reduction` は #190
    /// （TASK-9.1c 相当・`nn::loss`）で mean/sum の両縮約に対応するため
    /// struct variant 化した（旧 `MseLoss(NodeId, NodeId)` は mean
    /// 固定だった）。`crate::var::Reduction` を再利用し、`grad.rs` の
    /// `vjp()` が `pred`/`target` への勾配スケールを縮約種別ごとに
    /// 分岐する。
    MseLoss {
        pred: NodeId,
        target: NodeId,
        reduction: crate::var::Reduction,
    },
    /// CrossEntropy 損失（log-sum-exp 安定化・クラス次元指定。#191・
    /// 親イシュー #189）。`BackendOps` に対応メソッドがないため融合対象外
    /// とし常に実体化済み（`push_eager`）。log-softmax → NLL を個別オペ
    /// 合成せず、`MseLoss` と同じく forward/backward を解析形で閉じられる
    /// 1 個の融合オペとして実装する（実装計画 §3.1。PyTorch
    /// `F.cross_entropy` も内部で同等の融合実装）。
    ///
    /// `targets`（クラス添字）は非追跡データのため `Var`/`NodeId` を
    /// 持たず、`Op` payload に直接 `Tensor<i32>` を埋め込む（勾配は
    /// `logits` の 1 系統のみ定義され、`targets` 側には流れない。
    /// `grad.rs::vjp` の `CrossEntropyLoss` 分岐参照）。`reduction` は
    /// `MseLoss` と同じ `crate::var::Reduction`（#190 定義）を再利用する
    /// （`nn::loss` 側に重複定義は置かない）。
    CrossEntropyLoss {
        logits: NodeId,
        targets: Tensor<i32>,
        class_dim: usize,
        reduction: crate::var::Reduction,
    },
    /// デバイス常駐パラメータの葉ノード（イシュー #1022・`docs/
    /// device-resident-update-design.md` §3.3e）。`Op::Leaf` と異なり
    /// **ホスト値を持たない**（`TapeNode::value` は常に空の `OnceCell`
    /// のまま。`Tape::push_resident_leaf` 参照）。
    ///
    /// `fandhe_ai_autodiff::optim::device_store::DeviceParamStore` の
    /// `register_resident_params`／`snapshot_resident_params`（forward
    /// 用）のみが構築し、外部へは不透明型
    /// `optim::device_store::ResidentLeaf`（`shape()` のみ公開）として
    /// しか見えない——`Var` として公開しないのは、`Var::value()`／
    /// `to_tensor()`（`var.rs`）が非 fallible な API であり、ホスト値を
    /// 持たない本ノードに誤って到達すると「panic させる」か「黙示的に
    /// ゼロを返す」のいずれかしか選べず、本イシューが求める fail-closed
    /// な型付きエラー化と両立しないため。
    ///
    /// `store_id`／`slot` は [`ResidentResolver::resident_buffer`] が
    /// `Op::LinearResident` の VJP から対応する `DeviceBuffer<f32>`
    /// （weight／bias の実体）を引くための鍵（`store_id` は
    /// `DeviceParamStore` ごとに一意、`slot` はそのストア内の位置）。
    /// `grad.rs::vjp` は本 variant を `Op::Leaf` と同じく「入力を持たない
    /// （contributions が空）」として扱う。
    ResidentLeaf { store_id: u64, slot: usize },
    /// デバイス常駐 `weight`（・`bias`）で forward した Linear 相当ノード
    /// （イシュー #1022）。`y = input.matmul(weight) (+ bias)` と数式は
    /// 同じだが、`weight`／`bias` はいずれも `Op::ResidentLeaf`（ホスト値
    /// を持たない）を指すため既存 `Op::MatMul`／`Op::Add` の合成では
    /// 表現できず専用 variant とする。
    ///
    /// **常に実体化済み**（`push_eager`）: `DeviceParamStore::
    /// linear_forward`（`optim::device_store`）が
    /// `BackendOps::gemm_resident_rhs`（`tensor-core`）で forward 値を
    /// 計算した直後に積む。融合対象外（elementwise 5 演算ではないため
    /// `push_lazy` を経由しない。`Op::is_lazy_elementwise` 参照）。
    ///
    /// **素の [`Tape::backward`]（resolver なし）では型付きエラー**:
    /// 本 variant の VJP（`grad.rs`）は `weight`（・`bias`）の
    /// `DeviceBuffer<f32>` を [`ResidentResolver`] 経由で取得する必要が
    /// あり、素の `backward` はこの解決手段を持たないため
    /// `AutodiffError::InvalidArgument`（「`DeviceParamStore::backward`
    /// を使え」を含むメッセージ）を返す（fail-closed。`docs/
    /// device-resident-update-design.md` §3.3e）。`DeviceParamStore::
    /// backward`（`optim::device_store`。自身が [`ResidentResolver`] を
    /// 実装する）を使うと正しく計算される。
    LinearResident {
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        /// bias 加算に続けて適用する epilogue activation（イシュー
        /// #1044）。`Activation::None` は既存の bias のみ融合と同じ
        /// 挙動（追加前は暗黙に `None` だった）。`DeviceParamStore::
        /// linear_forward_with_activation` が次層 `ReLU` 先読み時に
        /// `Activation::Relu` を渡す。VJP（`grad.rs`）は forward 記録値
        /// `out_value` から ReLU マスクを復元するため、この場合も
        /// 前活性化の再計算・追加ノードを必要としない。
        act: Activation,
    },
    /// ホスト常駐 `weight`（・`bias`）で forward した Linear+activation
    /// 融合ノード（イシュー #1044・`docs/kernel-fusion.md` §2.2「学習
    /// 経路への結線」）。`y = act(input.matmul(weight) (+ bias))` と
    /// 数式上は `Op::MatMul` → `Op::Add` → `Op::Relu` の合成と同じだが、
    /// `fandhe_ai_autodiff::nn::linear::LinearVars::forward_with_activation`
    /// が `BackendOps::gemm_bias_act`（epilogue 融合カーネル。3
    /// バックエンドとも既にオーバーライド済み）を直接呼んで 1 ノードで
    /// 記録する（`Sequential`〈`fandhe_ai_facade::compat::sequential`〉が
    /// `Linear` 層の直後が `ReLU` 層であることを先読みして本 variant を
    /// 選ぶ。次層が `ReLU` でなければ `act: Activation::None` で bias
    /// のみ融合する）。
    ///
    /// **常に実体化済み**（`push_eager`）: 非 elementwise（`MatMul` と
    /// 同じ扱い）のため融合対象外（`Op::is_lazy_elementwise` 参照）。
    ///
    /// **`Op::LinearResident` との違い**: 本 variant は `weight`／`bias`
    /// いずれもホスト常駐の通常ノード（`Op::Leaf` 等）を指す。デバイス
    /// 常駐オペランドは扱わないため `ResidentResolver` を必要とせず、
    /// 素の [`Tape::backward`] からも正しく計算できる。
    LinearAct {
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        act: Activation,
    },
    /// `Var::reshape` が記録する view ノード（イシュー #1047・親 #1043
    /// 「カーネル融合・autodiff 実行モデルの強化」）。出力 shape は
    /// `TapeNode.shape` に構造的に保持済みのため payload には持たない。
    ///
    /// **ホスト値は登録時のみ空**（`push_view` で `value: OnceCell::new()`
    /// のまま登録。burn-autodiff の `MemoryBound { retro_forward }`
    /// checkpointing に相当し、forward 出力を保持せず backward 時に親
    /// ノードから再導出する。`resolve_view`（下記）が実際の再導出
    /// ロジック）。初回実体化後は `materialize_fallible`／
    /// `materialize_non_fallible` が `resolve_view` の結果を自身の
    /// `OnceCell` へ `set`／`get_or_init` でキャッシュする——view
    /// ヘッダ（shape のみで値は伴わない軽量な結果）のキャッシュであり、
    /// 融合対象の中間 `Tensor` 実体化を避ける本イシューの利得とは
    /// 矛盾しない。`input` は push 前に層 1（`materialize_fallible`）で
    /// 実体化済み——これが `resolve_view` を infallible にできる理由
    /// （`Var::reshape` doc・`resolve_view` doc 参照）。
    ///
    /// **融合境界**: `is_lazy_elementwise() == false` のため
    /// `push_lazy`（elementwise 5 演算専用）を経由せず、
    /// `build_lazy_plan`／`fallback_per_op`／`eval_fallback` の 3 走査器
    /// からは常に「葉」（`_` 分岐）として扱われる（`docs/
    /// kernel-fusion.md` 表 4「transpose 混在連鎖は融合しない」が既に
    /// 確定済みのため後退ではない）。
    Reshape { input: NodeId },
    /// `Var::transpose` が記録する view ノード（イシュー #1047）。
    /// `dim0`／`dim1` は forward 時点で軸範囲検査済み（`Var::transpose`）。
    ///
    /// `Reshape` と同じく**ホスト値を持たない**・**融合境界**（同上）。
    /// VJP（`grad.rs`）は対合性（`transpose` を 2 回適用すると恒等）を
    /// 利用し、同じ `dim0`／`dim1` で upstream を transpose するだけで
    /// zero-copy に閉じる。
    Transpose {
        input: NodeId,
        dim0: usize,
        dim1: usize,
    },
}

/// [`Op::LinearResident`] の VJP（`grad.rs`）が `weight`／`bias` の
/// `Op::ResidentLeaf { store_id, slot }` から実際のデバイスバッファ範囲
/// を引くための解決インタフェース（イシュー #1022・#1023「R3」）。
///
/// `fandhe_ai_autodiff::optim::device_store::DeviceParamStore` が実装し、
/// `Tape::backward_with_resident` から `grad::vjp` へスレッドする。`Tape`
/// は `Send` を維持する必要があり（`tests/fusion_backend_integration.rs`
/// の `tape_is_send`）`DeviceBuffer<f32>`（`!Send` な `Box<dyn
/// BufferHandle>` を保持しうる）を `TapeNode`／`Tape` 自身へ持たせられ
/// ないため、バッファの所有は `DeviceParamStore` 側に留め、backward の
/// 呼び出し側から本トレイトオブジェクトとして一時的に借用する設計と
/// した（`docs/device-resident-update-design.md` §3.3e）。
///
/// 戻り値が `&DeviceBuffer<f32>` ではなく [`DeviceBufferView`] である
/// 理由: イシュー #1023「パラメータ横断の単一連結バッファ化」により
/// `DeviceParamStore` は全パラメータを 1 本の連結 `DeviceBuffer<f32>`
/// として保持するため、`slot` に対応する個々のパラメータは連結
/// バッファ内の要素オフセット範囲としてしか表現できない
/// （「R3: 要素オフセット付き常駐ビュー」設計）。
pub(crate) trait ResidentResolver {
    /// `store_id`（このリゾルバ自身が保持するストアと一致するはず）・
    /// `slot`（ストア内の位置）から対応するデバイスバッファ範囲を返す。
    /// 別ストアの葉が混入した場合・`slot` が範囲外の場合は
    /// `AutodiffError::InvalidArgument` で fail-closed に拒否する
    /// （`.claude/rules/security.md` A08。誤った勾配をデバイスへ適用
    /// させない）。
    fn resident_buffer(
        &self,
        store_id: u64,
        slot: usize,
    ) -> Result<DeviceBufferView<'_>, AutodiffError>;

    /// `Op::LinearResident` の VJP（`grad.rs`）が d_weight
    /// （`x^T @ g`）をデバイス常駐のまま書き込めるか試みる入口
    /// （イシュー #1212・`docs/device-resident-update-design.md` 追補）。
    ///
    /// 実装（`DeviceParamStore`）は `BackendOps::gemm_fp32_strict_into`
    /// （`tensor-core`。既定 `Unsupported`）を使い、成功すれば自身の
    /// grad staging バッファの `slot` に対応する範囲へ直接書き込んで
    /// `Ok(true)` を返す。バックエンドが `gemm_fp32_strict_into`／
    /// `MemoryOps` を実装しない場合は `Ok(false)`（呼び出し元は
    /// `ops.gemm_fp32_strict` によるホスト経路へフォールバックする）。
    /// `store_id`／`slot`／shape の不一致は
    /// [`AutodiffError::InvalidArgument`] で fail-closed に拒否する
    /// （`resident_buffer` と同じ方針。`.claude/rules/security.md` A08）。
    ///
    /// # デフォルト実装
    ///
    /// 常に `Ok(false)`（ホスト経路へフォールバック）。現時点で
    /// `DeviceParamStore` のみがオーバーライドする。
    fn fill_resident_weight_grad(
        &self,
        _ops: &dyn BackendOps,
        _store_id: u64,
        _slot: usize,
        _x_t: &Tensor<f32>,
        _g: &Tensor<f32>,
    ) -> Result<bool, AutodiffError> {
        Ok(false)
    }
}

impl Op {
    /// elementwise 5 演算（融合の直接対象。`add`/`mul`/`relu`/`exp`/
    /// `tanh`）かどうか（TASK-12.1d・#164。`docs/fusion-graph-design.md`
    /// §3.5.1「遅延を許容する演算とその場で実体化する演算の切り分け」）。
    fn is_lazy_elementwise(&self) -> bool {
        matches!(
            self,
            Op::Add(..) | Op::Mul(..) | Op::Relu(..) | Op::Exp(..) | Op::Tanh(..)
        )
    }

    /// view 系ノード（`reshape`/`transpose`。イシュー #1047）かどうか。
    /// `push_view` で登録時は `TapeNode::value` を空のまま保つノード
    /// 種別を指し、`materialize_fallible`／`materialize_non_fallible`
    /// はこの判定で `resolve_view`（下記）への分岐を選ぶ（初回実体化後は
    /// 同メソッドが `resolve_view` の結果を `value` へキャッシュする。
    /// `Op::Reshape` doc 参照）。
    pub(crate) fn is_view(&self) -> bool {
        matches!(self, Op::Reshape { .. } | Op::Transpose { .. })
    }
}

/// テープ上の 1 ノード。演算種別（`Op`）・構造的に確定する出力 shape・
/// 実体化済みの値（空は「未実体化」を表す。TASK-12.1d・#164）を保持する
/// （PoC-v2-2 の `TapeNode { op, value }` 構造を踏襲。f64→f32 化は
/// `docs/public-api-design.md` §3.1 の確定事項）。`op` は `grad.rs` の
/// `vjp()` ディスパッチが `Tape::backward`（`backward.rs`）の逆走査から
/// 読み出す。
///
/// `shape` は実体化なしに算出できる（`add`/`mul` は broadcast、
/// `matmul` は行列積、`sum`/`max` は縮約、`relu`/`exp`/`tanh` は恒等の
/// shape 計算式であり、いずれも入力の `shape` フィールドだけを読めば
/// 求まる。`var.rs` の既存 shape 検証ロジックは今日すでに
/// `.value().shape()` ではなく形状情報のみを消費している）。
///
/// `value: OnceCell<Tensor<f32>>` は `add`/`mul`/`relu`/`exp`/`tanh`
/// （elementwise 5 演算）に限り空のまま記録される（遅延グラフの延長。
/// `docs/fusion-graph-design.md` §3.5.1）。`matmul`/`sum`/`max`・
/// `Op::Leaf`・`Sigmoid`/`MseLoss`/`CrossEntropyLoss`（`BackendOps` に
/// 対応メソッドがない演算）は常に返る前に `OnceCell::from(...)` で
/// 即座に埋める。`OnceCell::get_or_init`／`set` はいずれも `&self`
/// （共有参照）で呼べるため、`RefCell<Vec<TapeNode>>` の `borrow()`
/// （共有借用）だけで埋められる。
///
/// `lazy_chain_size`（#404・設計書 §3.5.4。codex-review PR #406 の P1
/// 是正で `lazy_depth`〈最大値ベース〉から改名・再設計）: このノードを
/// 起点として `build_lazy_plan` が実際に収容する**未実体化 interior
/// ノード数の上界**。`push_eager`（実体化済みノード）は常に 0。
/// `push_lazy` は入力ノードの**実効サイズ**（入力が実体化済みなら 0、
/// 未実体化なら格納済み `lazy_chain_size`）の**総和** + 1 を格納する
/// （`Tape::push_lazy` のドキュメント参照）。
///
/// **最大値ではなく総和を使う理由**: fan-in（複数の未実体化枝を 1 個の
/// `Op::Add`/`Op::Mul` で合流させる形）では `build_lazy_plan` が両枝の
/// interior を合算して 1 個の `FusionPlan` に収容するため、「両入力の
/// 段数の最大値 + 1」（旧 `effective_depth`）は `build_lazy_plan` が
/// 実際に集める interior ノード数を過小評価しうる（例: 3 ノードの枝
/// 2 本を `add` で結合すると最大値ベースでは 4 だが実際の interior は
/// 7。codex-review PR #406 指摘の反例）。総和ベースであれば diamond 型
/// DAG（同一祖先を複数経路が共有する場合）で重複カウントし過大評価に
/// なりうるが、それは「早めに自己実体化する」方向の安全側の誤差
/// （融合機会をわずかに減らすのみで `MAX_FUSED_CHAIN_LEN` 契約は破らない。
/// [`Tape::effective_subtree_size`] のドキュメント参照）。
#[derive(Debug)]
pub(crate) struct TapeNode {
    pub(crate) op: Op,
    pub(crate) shape: Vec<usize>,
    pub(crate) value: OnceCell<Tensor<f32>>,
    pub(crate) lazy_chain_size: usize,
}

/// 演算を記録する Wengert list。`Var`（`var.rs`）上の演算のみがここに
/// 記録される。`fandhe_ai_tensor_core::Tensor<f32>` に対する演算はテープを構築
/// しないため、追跡の有無は型（`Tensor<f32>` か `Var` か）で表現される
/// （`docs/public-api-design.md` §3.1「型分離方式」）。
///
/// ノード列は `RefCell` で包み、`Var` からの内部可変性による追記を
/// 可能にする。PoC-v2-2 確定 API（`Tape::matmul(&mut self, ...)`）は
/// `&mut Tape` を要求するが、本実装では `Var` 単体を式中で連鎖させたい
/// （`a.matmul(&b)?.relu()` 等）ため、`RefCell` + 共有参照 `&'t Tape`
/// 方式（`docs/public-api-design.md` §3.1 が productize 時に許容する
/// 選択肢）を採用する。
///
/// **`ops` フィールド（TASK-12.1d・#164。必須所有値）**: `Tape::new_with_ops(ops)`
/// で構築したどの `Tape` でも常に埋まっている（`Option` を経由しない。
/// `docs/fusion-graph-design.md` §1「`None` に相当する値がそもそも
/// 存在しない」）。`Box<dyn BackendOps + Send>`（`Send` 境界あり）と
/// することで `Tape` の全フィールド（`id`・`nodes`・`ops`）が `Send` を
/// 満たし、`Tape: Send` の自動導出は後退しない（同文書 §3.4）。
/// `Sync` はいずれのフィールドにも要求しない（`RefCell`／`OnceCell` に
/// より元々 `!Sync`）。`BackendOps` は `Debug` をスーパートレイトに
/// 持たないため `#[derive(Debug)]` は撤去し、`ops` の中身を表示しない
/// 手書き `impl fmt::Debug` へ置き換える（下記）。
///
/// **学習ループでの運用（イシュー #1048 で確定）**: ステップごとに新しい
/// `Tape::new_with_ops(ops)` を生成・破棄する運用に加え、[`Tape::reset`]
/// でノード列を葉プレフィックスまで切り詰めて同一 `Tape` を再利用する
/// 運用にも対応する（`docs/public-api-design.md` §3.1.1。旧「未決事項」を
/// 確定へ更新）。`reset` は結果ノードの `Tensor<f32>`（デバイスバッファ）を
/// drop してプールへ返却するため、reuse GEMM・学習ループでのバッファ
/// 蓄積（framework-compare reuse ベンチ・#1048 発端）を解消する。
pub struct Tape {
    pub(crate) id: TapeId,
    pub(crate) nodes: RefCell<Vec<TapeNode>>,
    pub(crate) ops: Box<dyn BackendOps + Send>,
    /// [`Tape::reset`] が呼ばれるたびに +1 する世代カウンタ（#1048）。
    /// `Var<'t>`／`ResidentLeaf<'t>` は `&'t Tape` を静的に借用するため
    /// reset 後の stale 値はコンパイル時に排除されるが、`Tape` を借用
    /// しない値（[`crate::backward::Gradients`]・
    /// `optim::device_store::DeviceParamStore` の `pending`）は reset を
    /// またいで生存しうる。これらは本カウンタを記録しておき、参照時に
    /// 現在の `epoch()` と比較して不一致なら fail-closed に拒否する
    /// （`Gradients::get`・`DeviceParamStore::step` 参照）。
    epoch: Cell<u64>,
    /// 葉プレフィックス長（#1048）。最初の非葉ノード（`Op::Leaf`/
    /// `Op::ResidentLeaf` 以外）が push された時点の `nodes.len()` を
    /// [`Tape::freeze_leaf_prefix`] が 1 回だけ `Some` に固定する。
    /// `None` の間は「まだ演算が始まっていない」ことを表し、[`Tape::reset`]
    /// は現在の全ノードを保持する（`leaf_count`/`reset` 参照）。
    retained_leaf_len: Cell<Option<usize>>,
}

impl std::fmt::Debug for Tape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `ops`（`Box<dyn BackendOps + Send>`）の中身は `Debug` を実装
        // しない（`BackendOps` は `Debug` をスーパートレイトに持たない。
        // 外部実装を破壊しないための既存方針）ため表示しない。`Tape:
        // Debug` という公開契約自体は変更しない（実装手段が `derive`
        // から手書きへ変わるのみ。`docs/fusion-graph-design.md` §3.4）。
        f.debug_struct("Tape").finish_non_exhaustive()
    }
}

/// 無引数構築の compat 経路（codex-review 第 19〜22 波・PR #403 の P1
/// 是正で追加。`docs/public-api-design.md` §4.1「移行手順」追補）。
/// [`Tape::new`]（無引数版）に委譲する。
impl Default for Tape {
    fn default() -> Self {
        Tape::new()
    }
}

impl Tape {
    /// 無引数構築の compat 入口（codex-review 第 22 波・PR #403 の P1
    /// 是正で復元。既存呼び出し元のソース互換性を壊さないため、TASK-12.1d
    /// 導入前と同じ無引数シグネチャを維持する）。`ops` には
    /// `default_ops::naive_ops()`（`eval.rs` へ委譲する naive CPU 参照
    /// 実装。`backend-cpu` 等の具体バックエンドクレートには依存しない）を
    /// 使う。性能が必要な呼び出し元は [`Tape::new_with_ops`] へ最適化済み
    /// `BackendOps` を明示的に渡すこと（`default_ops` モジュール冒頭
    /// コメント参照）。
    pub fn new() -> Tape {
        Tape::new_with_ops(crate::default_ops::naive_ops())
    }

    /// バックエンドを明示指定してテープを生成する（TASK-12.1d・#164 で
    /// `Tape::new(ops)` として導入し、codex-review 第 22 波・PR #403 の
    /// P1 是正で `Tape::new_with_ops` へ改名した）。`ops` は
    /// このテープ上のすべてのバックエンド実行（融合実行・per-op
    /// フォールバック・`matmul`/`sum`/`max` の実行）に使われる必須所有値
    /// であり、`facade`（TASK-9.3・イシュー #410 で実装済み）の composition root が解決した
    /// 具体 `BackendOps` 実装、明示指定した `Device` の結線結果、または
    /// テスト用フィクスチャのいずれかを渡す（`docs/
    /// fusion-graph-design.md` §1・§3.4）。`TapeId` は `NEXT_TAPE_ID`
    /// から新規発行されるため、同時に存在する複数の `Tape` 間で衝突
    /// しない。
    ///
    /// **無引数構築が必要な場合**は [`Tape::new`]（または [`Tape::default`]）
    /// を使う（codex-review 第 19〜22 波・PR #403 の P1 是正で維持した
    /// compat 経路。`default_ops` モジュール参照）。
    pub fn new_with_ops(ops: Box<dyn BackendOps + Send>) -> Tape {
        Tape {
            id: TapeId(NEXT_TAPE_ID.fetch_add(1, Ordering::Relaxed)),
            nodes: RefCell::new(Vec::new()),
            ops,
            epoch: Cell::new(0),
            retained_leaf_len: Cell::new(None),
        }
    }

    /// 非追跡の `Tensor<f32>` を、テープ上の葉ノード（`Op::Leaf`）として
    /// 登録する。以後この `Var` を起点とする演算はすべて `self` へ記録
    /// される。葉ノードは常に実体化済み（`push_eager`）。
    pub fn var(&self, tensor: &Tensor<f32>) -> crate::var::Var<'_> {
        let id = self.push_eager(Op::Leaf, tensor.clone());
        crate::var::Var::from_raw(self, id)
    }

    /// 現在記録済みのノード数を返す。受け入れ条件（forward 実行時に
    /// テープへ演算が記録される）の検証・デバッグ用途の最小限のアクセサ
    /// （`tests/tape_recording.rs`）。
    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    /// テープにノードが 1 つも記録されていないか判定する。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 現在の世代番号（#1048。[`Tape::reset`] のたびに +1）。
    /// `Gradients::get`／`DeviceParamStore::step` が stale 値を fail-closed
    /// に拒否するための比較対象（`epoch` フィールド doc 参照）。
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch.get()
    }

    /// 葉プレフィックス長を、まだ固定されていなければ `current_len` で
    /// 1 回だけ固定する（#1048）。`push_eager`（`Op::Leaf` 以外）・
    /// `push_lazy`・`push_view`（いずれも非葉ノードのみを追記する経路）
    /// の**push 前**に、その時点の `nodes.len()`（＝これから追記する
    /// 非葉ノードの直前までの葉数）を渡して呼ぶ契約とする。
    /// `push_resident_leaf`（`Op::ResidentLeaf` は葉扱い。モジュール doc
    /// 「学習ループでの運用」参照）は呼ばない。
    fn freeze_leaf_prefix(&self, current_len: usize) {
        if self.retained_leaf_len.get().is_none() {
            self.retained_leaf_len.set(Some(current_len));
        }
    }

    /// [`Tape::reset`] 後も保持される葉の個数（#1048）。葉プレフィックスが
    /// まだ固定されていない（一度も演算が記録されていない）間は現在の
    /// 全ノード数を返す——この場合は登録済みの葉がすべて保持対象になる
    /// （`reset` のドキュメント参照）。
    pub fn leaf_count(&self) -> usize {
        self.retained_leaf_len
            .get()
            .unwrap_or_else(|| self.nodes.borrow().len())
    }

    /// 保持される葉 `index` 番目（登録順）の `Var` を再取得する
    /// （#1048）。`reset()` はノードの中身を保持したまま切り詰めるだけ
    /// なので、`leaf(index)` は既存ノードへの `NodeId` 再包装のみ
    /// （コピー・再計算なし・O(1)）。`index` が葉プレフィックス範囲外、
    /// または当該ノードが `Op::Leaf` でない（例: `Op::ResidentLeaf`。
    /// `Var::value()`/`to_tensor()` は非 fallible なため、ホスト値を
    /// 持たない `ResidentLeaf` を `Var` として返すと誤動作の温床になる。
    /// `tape::Op::ResidentLeaf` doc 参照）場合は `None` を返す
    /// （境界外・型不一致で panic しない。`.claude/rules/coding-rust.md`）。
    pub fn leaf(&self, index: usize) -> Option<crate::var::Var<'_>> {
        if index >= self.leaf_count() {
            return None;
        }
        let nodes = self.nodes.borrow();
        match nodes.get(index) {
            Some(node) if matches!(node.op, Op::Leaf) => {
                Some(crate::var::Var::from_raw(self, NodeId(index)))
            }
            _ => None,
        }
    }

    /// ノード列を葉プレフィックス（[`Tape::leaf_count`]）まで切り詰め、
    /// 次のステップで同一 `Tape` を再利用可能にする（#1048。イシュー
    /// 発端: framework-compare reuse GEMM が同一 tape へ matmul を
    /// 繰り返し呼ぶたびに結果ノードの `Tensor<f32>` が蓄積し続け解放
    /// されない問題）。
    ///
    /// **`&mut self` の理由**: `Var<'t>`（`var.rs`）・`ResidentLeaf<'t>`
    /// （`optim::device_store`）はいずれも `&'t Tape` を静的に借用する
    /// ため、`reset` が `&mut self` を要求すれば「reset 前に取得した
    /// 古い `Var`/`ResidentLeaf` を reset 後も使う」経路はコンパイル時に
    /// 排除される（借用検査で弾かれる）。実行時 stale 検査（世代番号）
    /// を要するのは `Tape` を借用しない値（`Gradients`・
    /// `DeviceParamStore::pending`）のみに限定できる（`epoch` フィールド
    /// doc・`Gradients::get`・`DeviceParamStore::step` 参照）。
    ///
    /// **保持される葉の範囲**: 最初の演算（非葉ノード）が記録される
    /// *前* に登録した葉のみが保持される（[`Tape::leaf_count`]）。
    /// forward 内で毎ステップ登録される葉（入力バッチ・
    /// `Op::ResidentLeaf`・`Sequential::bind` の per-step パラメータ葉）は
    /// 演算後に登録されるため葉プレフィックスに含まれず、reset のたびに
    /// 破棄される——これにより step ごとの葉が無限蓄積することもない。
    /// 一度も演算を記録していない `Tape`（`leaf_count` が未固定）の
    /// reset は全ノードを保持する no-op。
    ///
    /// **`&mut self` 経由で `RefCell` を介さず切り詰める**: `self.nodes
    /// .get_mut()` は `&mut Tape` から直接 `Vec` へアクセスするため、
    /// 実行中の借用（`Ref`/`RefMut`）が残っていても二重借用 panic には
    /// ならない（コンパイル時に排他が保証される）。`truncate` で
    /// drop された `TapeNode::value`（`Tensor<f32>` = `Arc<Storage>`）は
    /// 参照カウントが 0 になれば即座にデバイスバッファをプール
    /// （`SizeClassPool`）へ返却する——これが蓄積解消の実体。
    pub fn reset(&mut self) {
        let keep = self
            .retained_leaf_len
            .get()
            .unwrap_or_else(|| self.nodes.get_mut().len());
        self.nodes.get_mut().truncate(keep);
        self.epoch.set(self.epoch.get() + 1);
    }

    /// この `Tape` が保持する `ops`（バックエンド実行の必須所有値）への
    /// 借用。`backward.rs`／`grad.rs` が `materialize_fallible` を呼ぶ際
    /// に使う。
    pub(crate) fn ops(&self) -> &dyn BackendOps {
        self.ops.as_ref()
    }

    /// **非 elementwise・常に実体化済み**のノードを追記する（`matmul`/
    /// `sum`/`max`・`Op::Leaf`・`Sigmoid`/`MseLoss`/`CrossEntropyLoss`
    /// から呼ばれる。TASK-12.1d・#164 で `push` から改称）。`op` の入力側
    /// `Ref` を保持したまま呼ばれると `RefCell` の二重可変借用 panic に
    /// なるため、呼び出し元は値計算を終えて借用（`Ref`）を閉じてから
    /// 本関数を呼ぶ契約とする（`Var::value`/`to_tensor` のドキュメント
    /// 参照）。
    pub(crate) fn push_eager(&self, op: Op, value: Tensor<f32>) -> NodeId {
        let shape = value.shape().to_vec();
        let mut nodes = self.nodes.borrow_mut();
        // `Op::Leaf` のみ葉ノード（`push_resident_leaf` の `Op::ResidentLeaf`
        // と合わせて #1048 の葉プレフィックス判定対象）。それ以外
        // （`MatMul`/`Sum`/`Max`/`Sigmoid`/`MseLoss`/`CrossEntropyLoss`/
        // `LinearResident`）は演算ノードのため、これから追記する直前の
        // 長さで葉プレフィックスを固定する（`Tape::reset` doc 参照）。
        if !matches!(op, Op::Leaf) {
            self.freeze_leaf_prefix(nodes.len());
        }
        let id = NodeId(nodes.len());
        nodes.push(TapeNode {
            op,
            shape,
            value: OnceCell::from(value),
            lazy_chain_size: 0,
        });
        id
    }

    /// デバイス常駐パラメータの葉ノード（[`Op::ResidentLeaf`]）を追記する
    /// （イシュー #1022）。`push_eager` と異なり**ホスト値を渡さない**
    /// （`value: OnceCell::new()` のまま空で登録する）——これが本イシュー
    /// の中核である「forward のたびにパラメータをホストへ download
    /// しない」を実現する箇所そのもの。`shape` は呼び出し元
    /// （`DeviceParamStore`）が保持するデバイスバッファの shape（構造的に
    /// 既知。実体化不要）をそのまま渡す。
    pub(crate) fn push_resident_leaf(
        &self,
        shape: Vec<usize>,
        store_id: u64,
        slot: usize,
    ) -> NodeId {
        let mut nodes = self.nodes.borrow_mut();
        let id = NodeId(nodes.len());
        nodes.push(TapeNode {
            op: Op::ResidentLeaf { store_id, slot },
            shape,
            value: OnceCell::new(),
            lazy_chain_size: 0,
        });
        id
    }

    /// view 系ノード（[`Op::Reshape`]/[`Op::Transpose`]。イシュー #1047）を
    /// 追記する。`push_eager`（値渡し・即実体化）とも `push_lazy`
    /// （elementwise 5 演算限定・自己実体化契約あり）とも異なる第 3 の
    /// 登録経路——**ホスト値を一切渡さず**（`value: OnceCell::new()`）、
    /// `lazy_chain_size` は融合連鎖に参加しないため常に 0 とする。
    ///
    /// **呼び出し契約**: `op` が指す `input`（`Op::Reshape { input }`／
    /// `Op::Transpose { input, .. }`）は本関数を呼ぶ**前**に層 1
    /// （`materialize_fallible`）で実体化済みであること（`Var::reshape`／
    /// `Var::transpose` の実装がこの順序を守る）。view は既存バッファへの
    /// 別解釈でしかなく、参照先の実体が存在することを前提に
    /// [`resolve_view`] が infallible に動作できる設計だからである。
    pub(crate) fn push_view(&self, op: Op, shape: Vec<usize>) -> NodeId {
        debug_assert!(op.is_view());
        let mut nodes = self.nodes.borrow_mut();
        // view ノードは常に非葉（#1048。`Tape::reset` doc 参照）。
        self.freeze_leaf_prefix(nodes.len());
        let id = NodeId(nodes.len());
        nodes.push(TapeNode {
            op,
            shape,
            value: OnceCell::new(),
            lazy_chain_size: 0,
        });
        id
    }

    /// **fan-in 事前実体化（#404・codex-review PR #406 の P1 是正）**:
    /// `Var::add`/`mul`（二項 elementwise の 2 演算のみ。`relu`/`exp`/
    /// `tanh` は単項のため呼ぶ必要がない）が `push_lazy` を呼ぶ**前**に
    /// 必ず呼ぶ契約とする。2 本の未実体化枝を 1 個の演算で合流させる
    /// fan-in では、`push_lazy` 後の自己実体化（`at_limit`）だけでは
    /// 手遅れになりうる——たとえば `lazy_chain_size` 3 の枝を 2 本
    /// `add` で結合すると新規ノードの必要サイズは `3 + 3 + 1 = 7` で
    /// あり、この時点で自己実体化しても `build_lazy_plan` が集める
    /// interior は既に 7 ノードで `MAX_FUSED_CHAIN_LEN`（= 6）を超えて
    /// しまう（codex-review PR #406 指摘の反例）。
    ///
    /// そこで合流**前**に「2 入力の未実体化サイズ合計 + 1 が上限を
    /// 超えるか」を検査し、超える場合は**大きい方の枝のみ**を
    /// [`materialize_fallible`] でその場実体化する。小さい方の枝の
    /// `lazy_chain_size` は常に `MAX_FUSED_CHAIN_LEN − 1` 以下で有界
    /// （どの未実体化ノードも push 時点で上限未満だったものしか残ら
    /// ない、という [`Tape::push_lazy`] の不変条件）であるため、大きい
    /// 方を実体化（サイズ 0 化）すれば残りは「小さい方のサイズ + 1」
    /// ≤ `MAX_FUSED_CHAIN_LEN` に必ず収まり、両方を実体化する必要はない。
    pub(crate) fn pre_materialize_for_binary_merge(
        &self,
        a: NodeId,
        b: NodeId,
    ) -> Result<(), AutodiffError> {
        let nodes = self.nodes.borrow();
        let size_of = |id: NodeId| -> usize {
            let node = &nodes[id.0];
            if node.value.get().is_some() {
                0
            } else {
                node.lazy_chain_size
            }
        };
        let size_a = size_of(a);
        let size_b = size_of(b);
        if size_a + size_b + 1 > MAX_FUSED_CHAIN_LEN {
            let larger = if size_a >= size_b { a } else { b };
            materialize_fallible(&nodes, self.ops(), larger)?;
        }
        Ok(())
    }

    /// **elementwise 5 演算（`add`/`mul`/`relu`/`exp`/`tanh`）専用**の
    /// 遅延追記経路（TASK-12.1d・#164）。`value` を空の `OnceCell` の
    /// まま記録し、自身の出力を実体化せずに返す（4〜6 段連鎖を実現する
    /// 主要因。`docs/fusion-graph-design.md` §3.5.1）。`shape` は構造的に
    /// 確定済みの出力 shape（呼び出し元が `broadcast_shape` 等で計算
    /// 済み）を渡す。
    ///
    /// **連鎖長上限適用（#404・設計書 §3.5.4。codex-review PR #406 の
    /// P1 是正で最大値ベースから総和ベースへ再設計）**: 戻り値の第 2
    /// 要素は「新規ノードの `lazy_chain_size` が `MAX_FUSED_CHAIN_LEN`
    /// 以上に達したか」を表す。`true` の場合、呼び出し元（`Var::add`/
    /// `mul`/`relu`/`exp`/`tanh`）は返った `NodeId` を層 1（fallible。
    /// `add`/`mul`）または層 2（非 fallible。`relu`/`exp`/`tanh`）で
    /// **その場**実体化する契約（「上限到達時点でその場の演算を実体化
    /// してから連鎖を再開する」）。自己実体化により以後このノードは
    /// `value.get().is_some()` となるため、[`Tape::effective_subtree_size`]
    /// はこれを 0 として扱い、連鎖のリセットが状態更新なしで自然に成立
    /// する。
    pub(crate) fn push_lazy(&self, op: Op, shape: Vec<usize>) -> (NodeId, bool) {
        debug_assert!(op.is_lazy_elementwise());
        let mut nodes = self.nodes.borrow_mut();
        // elementwise 5 演算はいずれも非葉（#1048。`Tape::reset` doc 参照）。
        self.freeze_leaf_prefix(nodes.len());
        let size = Tape::effective_subtree_size(&nodes, &op);
        let at_limit = size >= MAX_FUSED_CHAIN_LEN;
        let id = NodeId(nodes.len());
        nodes.push(TapeNode {
            op,
            shape,
            value: OnceCell::new(),
            lazy_chain_size: size,
        });
        (id, at_limit)
    }

    /// `push_lazy` が新規ノードに割り当てる `lazy_chain_size` を、入力
    /// ノードの**実効サイズ**（実体化済みなら 0、未実体化なら格納済み
    /// `lazy_chain_size`）の**総和** + 1 として計算する（#404・設計書
    /// §3.5.4。codex-review PR #406 の P1 是正）。
    ///
    /// **総和（最大値ではない）が必須の理由**: `build_lazy_plan` は
    /// 「新規ノードから到達可能な未実体化ノード全体」を 1 個の
    /// `FusionPlan` に収容する。fan-in（`Op::Add`/`Op::Mul` が 2 本の
    /// 独立した未実体化枝を合流させる形）では interior 総数は両枝の
    /// ノード数の**合計**であり、最大値ではない。旧実装（`max_input_depth + 1`）はこの合流を考慮せず「入力のうち深い方の段数 + 1」しか見ないため、それぞれ 3 ノードの枝を 2 本 `add` で結合すると実際の interior は 7 ノードなのに旧指標は 4 のままとなり、`MAX_FUSED_CHAIN_LEN`（= 6）契約を静かに破っていた（codex-review PR #406 指摘の反例）。
    ///
    /// **過大評価は安全側**: diamond 型 DAG（同一祖先ノードを複数の
    /// 経路が共有する場合）では総和が実際の distinct interior 数を
    /// 超えて重複カウントしうるが、`build_lazy_plan` が実際に集める
    /// distinct interior 数は必ずこの総和以下になるため、上限判定は
    /// 常に安全側（早めの自己実体化）に倒れる。加えて、途中ノードが
    /// `push_lazy` 後に別経路（`Var::value` 等）で実体化されても格納済み
    /// `lazy_chain_size` はその場では更新しないため、後続ノードのサイズ
    /// 計算が実際より大きくなる場合があるが、これも同じく安全側の誤差
    /// であり融合機会をわずかに減らすのみで正しさには影響しない
    /// （実装計画 §2「連鎖長（段数）の定義と追跡」参照）。
    fn effective_subtree_size(nodes: &[TapeNode], op: &Op) -> usize {
        let input_size = |id: NodeId| -> usize {
            let node = &nodes[id.0];
            if node.value.get().is_some() {
                0
            } else {
                node.lazy_chain_size
            }
        };
        let total_input_size = match op {
            Op::Add(a, b) | Op::Mul(a, b) => input_size(*a) + input_size(*b),
            Op::Relu(a) | Op::Exp(a) | Op::Tanh(a) => input_size(*a),
            // `push_lazy` は `debug_assert!(op.is_lazy_elementwise())` に
            // より elementwise 5 演算専用。到達しない分岐だが `match` の
            // 網羅性のため安全側で 0 を返す（本番経路 panic 禁止方針）。
            _ => 0,
        };
        total_input_size + 1
    }
}

// =====================================================================
// materialize ヘルパー（TASK-12.1d・#164。`docs/fusion-graph-design.md`
// §3.5.2・§3.5.3）。
// =====================================================================

/// `id` を起点とする遅延部分グラフ（`add`/`mul`/`relu`/`exp`/`tanh` の
/// 未実体化ノードの連結成分）を後方走査し、`FusionPlan::from_ops` へ
/// 渡す `FusedOpKind` 列・葉ノード実体・出力 shape を組み立てる
/// （`docs/fusion-graph-design.md` §3.5.2 手順 1・2 の共通部分。層 1・
/// 層 2 のいずれからも呼ばれる）。
///
/// `id` 自身が既に実体化済み（`OnceCell::get().is_some()`）の場合は
/// 空プラン（`ops` 空・`leaf_count == 0`）を返す—呼び出し元
/// （[`materialize_fallible`]／[`materialize_non_fallible`]）はこの場合
/// `run_fused`/フォールバックを試みる前に早期リターンする。
///
/// **NodeId の単調増加が非巡回性を保証する**（`docs/
/// fusion-graph-design.md` §3.5.1「走査順が既に発生順トポロジカル順で
/// あること」）: `id.0` から `0` へ向けた降順の単純な線形スキャンだけで
/// 「参照済みノードをそれより手前で必ず処理し終えている」ことが保証
/// される。
///
/// **連鎖長上限は push 時点で適用済み（#404・設計書 §3.5.4。
/// codex-review PR #406 の P1 是正で fan-in 反例を修正）**:
/// `Tape::push_lazy` が新規ノードの `lazy_chain_size`
/// （[`Tape::effective_subtree_size`]。未実体化入力の**総和** + 1）を
/// 計算し、`MAX_FUSED_CHAIN_LEN`（= 6）に到達した時点で呼び出し元
/// （`Var::add`/`mul`/`relu`/`exp`/`tanh`）がそのノードをその場で自己
/// 実体化する（層 1／層 2。`var.rs` 参照）。
///
/// **二項演算（`add`/`mul`）は push 前の fan-in 事前実体化も必須**
/// （[`Tape::pre_materialize_for_binary_merge`]）: push 後の自己実体化
/// だけでは「2 本の未実体化枝を 1 回の演算で合流させた結果、新規ノード
/// 自身の `interior` が単独で `MAX_FUSED_CHAIN_LEN` を超える」ケース
/// （fan-in）を防げない——たとえば `lazy_chain_size` 3 の枝を 2 本
/// `add` で結合すると、push 後に自己実体化しても集める interior は
/// 既に 7 ノードになってしまう（codex-review PR #406 指摘の反例）。
/// そのため `Var::add`/`mul` は `push_lazy` を呼ぶ**前**に
/// `pre_materialize_for_binary_merge` で「合流後サイズが上限を超える
/// なら大きい方の枝を先に実体化する」ことで、push 時点で常に
/// `lazy_chain_size ≤ MAX_FUSED_CHAIN_LEN` に収まることを保証する。
///
/// 以上 2 段の適用により、本関数が呼ばれた時点で `id` を起点とする
/// 未実体化 elementwise 連結成分（`interior`）の distinct ノード数は
/// 常に `lazy_chain_size`（したがって `MAX_FUSED_CHAIN_LEN`）以下で
/// 有界であり（各ノードは push 時点で上限以下だったものしか未実体化の
/// まま残らない。diamond 型 DAG による重複カウントは上限判定を安全側
/// 〈早期実体化〉に倒すのみでこの上界を破らない）、下記 `node_index`
/// の線形探索（interior 数 n に対し構築コスト O(n²)）も定数上限で
/// 有界化されている。
fn build_lazy_plan(
    nodes: &[TapeNode],
    id: NodeId,
) -> Result<(FusionPlan, Vec<Tensor<f32>>, NodeId), BackendError> {
    use std::collections::HashMap;

    let mut reachable: HashMap<usize, ()> = HashMap::new();
    reachable.insert(id.0, ());
    let mut interior: Vec<usize> = Vec::new();
    let mut leaf_order: Vec<usize> = Vec::new();
    let mut leaf_index_of: HashMap<usize, usize> = HashMap::new();

    for cur in (0..=id.0).rev() {
        if !reachable.contains_key(&cur) {
            continue;
        }
        let node = &nodes[cur];
        if node.value.get().is_some() {
            // 実体化済み: このノードで走査を打ち切り、葉として扱う
            // （`Op::Leaf`・`matmul`/`sum`/`max`・`Sigmoid` 等の非
            // elementwise、または以前に別経路から実体化済みの
            // elementwise ノードのいずれも該当しうる）。
            leaf_index_of.entry(cur).or_insert_with(|| {
                let idx = leaf_order.len();
                leaf_order.push(cur);
                idx
            });
            continue;
        }
        match &node.op {
            Op::Add(a, b) | Op::Mul(a, b) => {
                interior.push(cur);
                reachable.insert(a.0, ());
                reachable.insert(b.0, ());
            }
            Op::Relu(a) | Op::Exp(a) | Op::Tanh(a) => {
                interior.push(cur);
                reachable.insert(a.0, ());
            }
            // 非 elementwise は `push_eager` により常に実体化済みの
            // はずであり、この分岐には構造上到達しない。それでも
            // 本番経路 panic 禁止のため、到達した場合は安全側で葉
            // として扱う（`OnceCell` 不変条件違反の防御的フォール）。
            _ => {
                leaf_index_of.entry(cur).or_insert_with(|| {
                    let idx = leaf_order.len();
                    leaf_order.push(cur);
                    idx
                });
            }
        }
    }
    interior.reverse(); // 発生順（トポロジカル順）へ揃える

    // push 時上限適用（#404。codex-review PR #406 の P1 是正で
    // `lazy_chain_size`〈総和ベース〉へ再設計）の不変条件を防御的に
    // 検査する（本番経路では panic させないため debug_assert に限る。
    // `.claude/rules/coding-rust.md` 本番経路 panic 禁止方針）。
    // fan-in を含むどの部分グラフでも distinct interior 数は
    // `lazy_chain_size`（総和ベース。diamond 型 DAG による重複カウントは
    // 安全側の過大評価）以下であり、二項演算は `pre_materialize_for_
    // binary_merge` の事前実体化により push 時点で `lazy_chain_size` が
    // 常に `MAX_FUSED_CHAIN_LEN` 以下に収まるため、interior は常に
    // `MAX_FUSED_CHAIN_LEN` 以下で有界となる。
    debug_assert!(
        interior.len() <= MAX_FUSED_CHAIN_LEN,
        "build_lazy_plan: interior が push 時上限適用（#404）の不変条件を超えた（契約違反）"
    );

    let leaf_count = leaf_order.len();
    let node_index = |n: usize, interior: &[usize]| -> usize {
        if let Some(&li) = leaf_index_of.get(&n) {
            li
        } else {
            // interior 内での位置（昇順走査中は必ず存在する契約）。
            let pos = interior.iter().position(|&x| x == n).unwrap_or(0);
            leaf_count + pos
        }
    };

    // `FusionPlan::from_ops`（`tensor-core::fusion::plan`）は `ops[0..
    // leaf_count)` に `FusedOpKind::Input { leaf_index }` が発生順（`0..
    // leaf_count` を一度ずつ）で並んでいることを構築時に検証する契約
    // （`plan.rs` の `from_ops` ドキュメント「検証」節）。`node_index` は
    // 葉を `0..leaf_count` の位置として扱うため、`ops` 自身にもその位置に
    // 対応する `Input` エントリを明示的に積む必要がある——省略すると
    // `ops[0]`（先頭の interior エントリ）が「まだ存在しない位置」を
    // 参照したとみなされ `FusionPlanError::IndexOutOfRange` で拒否される。
    let mut ops: Vec<FusedOpKind> = Vec::with_capacity(leaf_count + interior.len());
    for leaf_index in 0..leaf_count {
        ops.push(FusedOpKind::Input { leaf_index });
    }
    for &cur in &interior {
        let kind = match &nodes[cur].op {
            Op::Add(a, b) => FusedOpKind::Add {
                lhs: node_index(a.0, &interior),
                rhs: node_index(b.0, &interior),
            },
            Op::Mul(a, b) => FusedOpKind::Mul {
                lhs: node_index(a.0, &interior),
                rhs: node_index(b.0, &interior),
            },
            Op::Relu(a) => FusedOpKind::Relu {
                input: node_index(a.0, &interior),
            },
            Op::Exp(a) => FusedOpKind::Exp {
                input: node_index(a.0, &interior),
            },
            Op::Tanh(a) => FusedOpKind::Tanh {
                input: node_index(a.0, &interior),
            },
            _ => unreachable_op_kind(),
        };
        ops.push(kind);
    }

    let output_shape = nodes[id.0].shape.clone();
    let leaves: Vec<Tensor<f32>> = leaf_order
        .iter()
        .map(|&n| lazy_leaf_value(nodes, n))
        .collect();

    let plan = FusionPlan::from_ops(ops, output_shape, DType::F32, leaf_count)?;
    Ok((plan, leaves, id))
}

/// `node.shape` から全要素 `0.0` のテンソルを構築する安全側フォール
/// バック（PR #403 codex-review P1 是正）。`Tensor::from_shape_fill` は
/// `tensor-core::checked_numel` による要素数積オーバーフロー検査を
/// 経る `Result` 返り値になったため（`tensor.rs` の該当コメント参照）、
/// 本ファイルの契約違反時フォールバック 3 箇所（[`lazy_leaf_value`]・
/// [`fallback_per_op`]・`eval_fallback`）で共有する。渡す `shape` は
/// いずれも既存の `TapeNode` から読んだ値（過去に妥当な shape として
/// 構築済み）であり実運用でオーバーフローは起こらないが、`Err` 分岐は
/// `debug_assert!` で検知しつつ [`fandhe_ai_tensor_core::Tensor::scalar`]（真に
/// infallible）へ吸収し、本番経路 panic 禁止方針
/// （`.claude/rules/coding-rust.md`）を保つ。
fn safe_zeros(shape: &[usize]) -> Tensor<f32> {
    Tensor::from_shape_fill(shape, |_| 0.0).unwrap_or_else(|_| {
        debug_assert!(
            false,
            "safe_zeros: shape の要素数積がオーバーフローした（契約違反）"
        );
        Tensor::scalar(0.0)
    })
}

/// `interior` 収集ロジックが `_` 分岐（非 elementwise が未実体化のまま
/// 到達する契約違反）へ到達した場合の安全側フォールバック値。呼び出し元
/// （`build_lazy_plan`）はこの分岐へ実際には到達しない設計だが、
/// `match` の網羅性を保つために用意する（`.claude/rules/coding-rust.md`
/// 本番経路 panic 禁止方針。`unreachable!()` を直接書かない）。
fn unreachable_op_kind() -> FusedOpKind {
    debug_assert!(
        false,
        "build_lazy_plan: 非 elementwise ノードが未実体化のまま interior へ混入した（契約違反）"
    );
    FusedOpKind::Add { lhs: 0, rhs: 0 }
}

/// 実体化済みノードの値を読む（`build_lazy_plan`／`fallback_per_op`／
/// `eval_fallback` の葉収集共通ヘルパー）。
///
/// **view ノード対応（イシュー #1047）**: 3 走査器はいずれも
/// elementwise 5 演算の連結成分のみを `interior` として収集し、それ以外
/// （非 elementwise・未実体化）は `_` 分岐で「葉」として扱う
/// （`build_lazy_plan` 等のドキュメント参照）。`Op::Reshape`/
/// `Op::Transpose`（`push_view` で登録・`value` は常に空。`tape::Op`
/// doc 参照）がこの経路で葉として現れた場合、旧実装の
/// `debug_assert!(false) + ゼロ埋め`（「実体化済みのはずが未実体化
/// だった」契約違反フォールバック）に誤って落ちてしまう——view は
/// 契約上ホスト値を持たないのが正常な状態であり契約違反ではないため、
/// `resolve_view`（下記）で再導出する専用分岐を設ける。
///
/// それ以外（非 view で未実体化）は真の契約違反であり、`unwrap()`/
/// `expect()` は使わず `shape` から構築した安全側フォールバック（全要素
/// `0.0`）を返す（本番経路 panic 禁止方針）。
///
/// なお view ノードは `push_view` 登録時のみ `value` が空であり、
/// `materialize_fallible`／`materialize_non_fallible` を経由して一度
/// 実体化された後は同じ `OnceCell` に結果がキャッシュされるため、本関数
/// が `resolve_view` へ分岐するのは未実体化のまま到達した場合のみ
/// （`Op::Reshape` doc 参照）。
fn lazy_leaf_value(nodes: &[TapeNode], n: usize) -> Tensor<f32> {
    let node = &nodes[n];
    match node.value.get() {
        Some(t) => t.clone(),
        None if node.op.is_view() => resolve_view(nodes, NodeId(n)),
        None => {
            debug_assert!(
                false,
                "lazy_leaf_value: 実体化済みのはずのノードが未実体化だった（契約違反）"
            );
            safe_zeros(&node.shape)
        }
    }
}

/// view ノード（[`Op::Reshape`]/[`Op::Transpose`]。イシュー #1047）を
/// 入力側へ再帰的に辿り、最初に実体化済み（`value.get().is_some()`）な
/// ノードの値へ `reshape`/`transpose` を順に適用して再導出する
/// （burn-autodiff の `retro_forward` 相当）。`push_view` の呼び出し
/// 契約（`input` は push 前に層 1で実体化済み）により通常は 1 段の
/// 再帰で終わるが、view の view（例: `x.transpose(0,1)?.reshape(..)?`）
/// にも構造的に対応するため再帰実装とする。
///
/// **`Arc` 共有のみ・バッファは確保しない**: `Tensor::reshape`／
/// `transpose` はいずれも既存 `storage: Arc<Storage>` を `Arc::clone`
/// するだけの zero-copy view 演算（`tensor-core/src/tensor.rs`）で
/// あり、本関数はそれらを合成するだけなので新規ヒープ確保を伴わない
/// （`tests/view_zero_alloc.rs` が機械的に検証する）。
///
/// **infallible な理由**: `Var::reshape`/`transpose`（`var.rs`）が push
/// 時点で shape・軸範囲を検査済みのため、ここでの `reshape`/`transpose`
/// 呼び出しは構造的に失敗しえない。それでも本番経路 panic 禁止方針
/// （`.claude/rules/coding-rust.md`）のため、`Err` 分岐は
/// `debug_assert!` + 安全側フォールバック（全要素 `0.0`）に吸収する
/// （到達すれば forward 側の shape 検査ロジックにバグがある）。
fn resolve_view(nodes: &[TapeNode], id: NodeId) -> Tensor<f32> {
    let node = &nodes[id.0];
    if let Some(v) = node.value.get() {
        return v.clone();
    }
    match &node.op {
        Op::Reshape { input } => {
            let base = resolve_view(nodes, *input);
            base.reshape(&node.shape).unwrap_or_else(|_| {
                debug_assert!(
                    false,
                    "resolve_view: Op::Reshape の再導出が失敗した（forward 側の契約違反）"
                );
                safe_zeros(&node.shape)
            })
        }
        Op::Transpose { input, dim0, dim1 } => {
            let base = resolve_view(nodes, *input);
            base.transpose(*dim0, *dim1).unwrap_or_else(|_| {
                debug_assert!(
                    false,
                    "resolve_view: Op::Transpose の再導出が失敗した（forward 側の契約違反）"
                );
                safe_zeros(&node.shape)
            })
        }
        _ => {
            // `push_view` の呼び出し契約（`input` は事前実体化済み）に
            // より、非 view ノードが未実体化のままここへ渡ることはない。
            // 到達すれば契約違反であり、安全側フォールバックで吸収する。
            debug_assert!(
                false,
                "resolve_view: 非 view ノードへ到達した（契約違反。push_view の事前実体化契約が破られた）"
            );
            safe_zeros(&node.shape)
        }
    }
}

/// per-op フォールバック（`run_fused` が `Unsupported` を返した場合に
/// 遅延部分グラフを `ops` の既存 per-op メソッドで逐次再計算する。
/// `docs/fusion-graph-design.md` §3.4「fail-safe 方針」・§3.5.2 手順 3）。
/// `build_lazy_plan` と同じ後方走査を再実行し、今度は実際に値を計算する
/// （プランと違い、この経路は shape 検証済みの `ops` 呼び出しの結果を
/// そのまま使うため `FusionPlan` を経由しない）。
fn fallback_per_op(
    nodes: &[TapeNode],
    ops: &dyn BackendOps,
    id: NodeId,
) -> Result<Tensor<f32>, BackendError> {
    use std::collections::HashMap;

    // `build_lazy_plan` と同じ走査で interior（未実体化 elementwise の
    // 連結成分。発生順）を求める。
    let mut reachable: HashMap<usize, ()> = HashMap::new();
    reachable.insert(id.0, ());
    let mut interior: Vec<usize> = Vec::new();
    for cur in (0..=id.0).rev() {
        if !reachable.contains_key(&cur) {
            continue;
        }
        let node = &nodes[cur];
        if node.value.get().is_some() {
            continue;
        }
        match &node.op {
            Op::Add(a, b) | Op::Mul(a, b) => {
                interior.push(cur);
                reachable.insert(a.0, ());
                reachable.insert(b.0, ());
            }
            Op::Relu(a) | Op::Exp(a) | Op::Tanh(a) => {
                interior.push(cur);
                reachable.insert(a.0, ());
            }
            _ => {}
        }
    }
    interior.reverse();

    let mut computed: HashMap<usize, Tensor<f32>> = HashMap::new();
    let value_of = |n: usize, computed: &HashMap<usize, Tensor<f32>>| -> Tensor<f32> {
        if let Some(t) = nodes[n].value.get() {
            t.clone()
        } else {
            computed
                .get(&n)
                .cloned()
                .unwrap_or_else(|| lazy_leaf_value(nodes, n))
        }
    };
    for &cur in &interior {
        let result = match &nodes[cur].op {
            Op::Add(a, b) => ops.add(&value_of(a.0, &computed), &value_of(b.0, &computed))?,
            Op::Mul(a, b) => ops.mul(&value_of(a.0, &computed), &value_of(b.0, &computed))?,
            Op::Relu(a) => ops.relu(&value_of(a.0, &computed))?,
            Op::Exp(a) => ops.exp(&value_of(a.0, &computed))?,
            Op::Tanh(a) => ops.tanh(&value_of(a.0, &computed))?,
            _ => continue,
        };
        computed.insert(cur, result);
    }

    computed.get(&id.0).cloned().map(Ok).unwrap_or_else(|| {
        // interior が空（`id` が既に実体化済みだった呼び出し）の場合は
        // 呼び出し元が別途処理する契約のため、通常この分岐には来ない。
        Ok(nodes[id.0]
            .value
            .get()
            .cloned()
            .unwrap_or_else(|| safe_zeros(&nodes[id.0].shape)))
    })
}

/// 層 1（fallible 境界）: 後続の fallible `Var` 演算（`matmul`/`sum`/
/// `max`）・`Tape::backward` が forward 記録済みの未実体化ノードを
/// 読み出す際に**必ず**使用する（`docs/fusion-graph-design.md` §3.5.2）。
/// `Var::value`（層 2）は本関数を呼ばない。
///
/// `run_fused` の失敗のうち `BackendError::Unsupported`（能力不足）・
/// `BackendError::ShapeMismatch`（**Cursor Bugbot・PR #403 是正**:
/// `build_lazy_plan` は broadcast を伴う elementwise 連鎖（`[N,M] + [M]`
/// の bias add 等）も leaf shape を検査せず `FusionPlan` 化するため、
/// `run_fused` 実装（`backend-cpu::run_fused_elementwise` 等）が「全 leaf
/// と出力の shape が同一」という契約を検査して `ShapeMismatch` を返す
/// ケースが正当に発生しうる。この `ShapeMismatch` は `build_lazy_plan` が
/// 構築した `FusionPlan` 自体の shape 不整合ではない——各ノードの shape は
/// `Var::add`/`mul` がグラフ構築時点で `broadcast_shape` により検証済み
/// （このため `FusionPlan::from_ops` 自体は成功する）——のであり、単に
/// 「この融合バックエンド実装が broadcast 済み leaf の融合実行に対応
/// していない」という能力不足を意味する。`Unsupported` と区別せず同じ
/// フォールバック経路（層 2 の `materialize_non_fallible` が採る「エラー
/// 種別を区別しない」方針と同じ）に載せる）**以外**は eager フォール
/// バックで吸収せず型付きエラーのまま呼び出し元へ返す。上記 2 種別の
/// 場合のみ `ops` の per-op メソッドへ逐次フォールバックし、それも失敗
/// すれば `?` で伝播する（§3.5.2 手順 3・4。`eval.rs` への最終手段
/// フォールバックは層 2 限定であり層 1 では使わない）。
///
/// 実体化が完了したら実体化を要求した対象ノード自身の `OnceCell` にのみ
/// `set()` する（連鎖の途中ノードの `OnceCell` は空のまま残す。融合の
/// 利得〈中間 `Tensor` 実体化の除去〉を保つため）。`OnceCell::set` の
/// `Err`（fan-out に伴う二重到達）は通常分岐として扱い、`get()` で
/// 読み直した既存値を正として使う（`panic!`／`unwrap()`／`expect()` は
/// いずれも使わない）。
pub(crate) fn materialize_fallible<'a>(
    nodes: &'a [TapeNode],
    ops: &dyn BackendOps,
    id: NodeId,
) -> Result<&'a Tensor<f32>, AutodiffError> {
    if let Some(v) = nodes[id.0].value.get() {
        return Ok(v);
    }

    // view ノード（イシュー #1047）: `build_lazy_plan`（elementwise 5
    // 演算の連結成分専用）を経由せず、`resolve_view` で入力側から直接
    // 再導出する。結果（入力と `Arc` 共有の view ヘッダ）は他の実体化
    // 経路と同じく対象ノード自身の `OnceCell` にのみキャッシュする。
    if nodes[id.0].op.is_view() {
        let computed = resolve_view(nodes, id);
        match nodes[id.0].value.set(computed) {
            Ok(()) => {}
            Err(_rejected) => { /* fan-out 二重到達: 既存値を正として使う */ }
        }
        return nodes[id.0].value.get().ok_or_else(|| {
            AutodiffError::Backward(
                "materialize_fallible: OnceCell 不変条件違反（view 解決後の set 直後に get が None を返した）"
                    .into(),
            )
        });
    }

    let (plan, leaves, _root) = build_lazy_plan(nodes, id).map_err(AutodiffError::Backend)?;
    let leaf_refs: Vec<&Tensor<f32>> = leaves.iter().collect();
    let computed = match ops.run_fused(&plan, &leaf_refs) {
        Ok(t) => t,
        // `Unsupported`／`ShapeMismatch` はいずれも「この融合バックエンド
        // 実装がこの `FusionPlan` を融合実行する能力を持たない」ことを
        // 意味する（上記ドキュメンテーションコメント参照。Cursor
        // Bugbot・PR #403 是正）。per-op フォールバックへ委譲する。
        Err(BackendError::Unsupported(_)) | Err(BackendError::ShapeMismatch(_)) => {
            fallback_per_op(nodes, ops, id).map_err(AutodiffError::Backend)?
        }
        Err(other) => return Err(AutodiffError::Backend(other)),
    };

    match nodes[id.0].value.set(computed) {
        Ok(()) => {}
        Err(_rejected) => { /* fan-out 二重到達: 既存値を正として使う */ }
    }
    nodes[id.0].value.get().ok_or_else(|| {
        AutodiffError::Backward(
            "materialize_fallible: OnceCell 不変条件違反（set 直後に get が None を返した）".into(),
        )
    })
}

/// 層 2（非 fallible 境界）: `Var::value`／`Var::to_tensor`・
/// `Gradients::get` が使う（`docs/fusion-graph-design.md` §3.5.3）。
/// `matmul`/`sum`/`max`・`Tape::backward` は本関数を呼ばない
/// （[`materialize_fallible`] を使う）。
///
/// `ops.run_fused` が `Ok` 以外を返した場合はエラー種別を区別せず、
/// まず `ops` 自身の per-op メソッドへフォールバックし、それも失敗した
/// 場合に限り `autodiff` 自身の `eval.rs`（#164 で非 panic 構造へ改修
/// 済み）を最終手段として用いて再計算する。必ず `Tensor<f32>` を返し、
/// `panic!` も `Err` も返さない。
pub(crate) fn materialize_non_fallible<'a>(
    nodes: &'a [TapeNode],
    ops: &dyn BackendOps,
    id: NodeId,
) -> &'a Tensor<f32> {
    // `OnceCell::get_or_init`（設計書 §3.5.3 のスケッチどおり）は非
    // fallible なクロージャを取り `&self` のみで完結するため、層 1
    // （`materialize_fallible`）のような手動 set/get の 2 段階処理・
    // fan-out 二重到達の分岐が不要になる（`get_or_init` 自体が
    // 「既に初期化済みならそれを返す」を保証する）。クロージャ内は
    // 融合加速 → per-op フォールバック → `eval.rs` 最終手段の順に必ず
    // 値を返すため、この経路は構造的に失敗しない
    // （`docs/fusion-graph-design.md` §3.5.3 (i)〜(iii)）。
    nodes[id.0].value.get_or_init(|| {
        // view ノード（イシュー #1047）: 融合・per-op フォールバックの
        // 前に最優先で判定する（`resolve_view` は infallible なため、
        // ここで確実に値が決まる。層 1 の同分岐と対応）。
        if nodes[id.0].op.is_view() {
            return resolve_view(nodes, id);
        }
        build_lazy_plan(nodes, id)
            .ok()
            .and_then(|(plan, leaves, _root)| {
                let leaf_refs: Vec<&Tensor<f32>> = leaves.iter().collect();
                ops.run_fused(&plan, &leaf_refs).ok()
            })
            .or_else(|| fallback_per_op(nodes, ops, id).ok())
            .unwrap_or_else(|| eval_fallback(nodes, id))
    })
}

/// `materialize_non_fallible` の最終手段: `ops` の per-op メソッドも
/// 失敗した場合、`autodiff` 自身の `eval.rs`（#164 で非 panic 構造へ
/// 改修済み）を用いてトポロジカル順に逐次再計算する（`docs/
/// fusion-graph-design.md` §3.5.3。層 2 限定の例外）。
fn eval_fallback(nodes: &[TapeNode], id: NodeId) -> Tensor<f32> {
    use std::collections::HashMap;

    let mut reachable: HashMap<usize, ()> = HashMap::new();
    reachable.insert(id.0, ());
    let mut interior: Vec<usize> = Vec::new();
    for cur in (0..=id.0).rev() {
        if !reachable.contains_key(&cur) {
            continue;
        }
        let node = &nodes[cur];
        if node.value.get().is_some() {
            continue;
        }
        match &node.op {
            Op::Add(a, b) | Op::Mul(a, b) => {
                interior.push(cur);
                reachable.insert(a.0, ());
                reachable.insert(b.0, ());
            }
            Op::Relu(a) | Op::Exp(a) | Op::Tanh(a) => {
                interior.push(cur);
                reachable.insert(a.0, ());
            }
            _ => {}
        }
    }
    interior.reverse();

    let mut computed: HashMap<usize, Tensor<f32>> = HashMap::new();
    let value_of = |n: usize, computed: &HashMap<usize, Tensor<f32>>| -> Tensor<f32> {
        if let Some(t) = nodes[n].value.get() {
            t.clone()
        } else {
            computed
                .get(&n)
                .cloned()
                .unwrap_or_else(|| lazy_leaf_value(nodes, n))
        }
    };
    for &cur in &interior {
        let result = match &nodes[cur].op {
            Op::Add(a, b) => crate::eval::add(&value_of(a.0, &computed), &value_of(b.0, &computed)),
            Op::Mul(a, b) => crate::eval::mul(&value_of(a.0, &computed), &value_of(b.0, &computed)),
            Op::Relu(a) => crate::eval::relu(&value_of(a.0, &computed)),
            Op::Exp(a) => crate::eval::exp(&value_of(a.0, &computed)),
            Op::Tanh(a) => crate::eval::tanh(&value_of(a.0, &computed)),
            _ => continue,
        };
        computed.insert(cur, result);
    }

    computed
        .get(&id.0)
        .cloned()
        .unwrap_or_else(|| safe_zeros(&nodes[id.0].shape))
}
