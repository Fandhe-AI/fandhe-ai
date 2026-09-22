//! テープ上の 1 ノードを指す追跡対象値 `Var` と、その forward 演算群。
//!
//! `fandhe_ai_tensor_core::Tensor<f32>` に対する演算は一切テープを構築しない
//! （非追跡）。`Var` に対する演算のみが `Tape::push`（`tape.rs`）を
//! 経由してテープへ記録される。この「型分離」により、勾配追跡の
//! ON/OFF がコンパイル時に保証される（`docs/public-api-design.md`
//! §3.1「型分離方式」）。
//!
//! 各演算メソッドは
//! 「①クロステープ検査 → ②shape 検査 → ③forward 値計算（`eval.rs`）
//! → ④ノード記録（`Tape::push`）」の順で処理する。値計算の借用
//! （`Ref`）はスコープを閉じてから `push`（`borrow_mut`）を呼ぶ
//! （`RefCell` の二重可変借用 panic を避けるための実装規律。
//! `.claude/rules/coding-rust.md` の本番経路 panic 禁止方針）。

use std::cell::Ref;

use fandhe_ai_tensor_core::{
    Activation, BackendError, BackendOps, BatchNormTrainOutput, BceKind, CastElement,
    ChecksumReadout, Conv2dParams, Device, GemmChecksum, GruPointwiseOutput, HuberKind,
    InterpolateMode, KlDivTarget, LstmPointwiseOutput, MatrixNormOrd, MseReduction, Pool2dParams,
    ScalarBinaryOp, ScalarUnaryOp, ScatterReduce, ShapeError, Tensor, VectorNormOrd,
    adaptive_pool2d_out_shape, batch_norm_layout, broadcast_shape, concat_out_shape,
    conv_transpose2d_out_shape, conv2d_out_shape, flatten_out_shape, gather_out_shape,
    gemm_out_shape, interpolate_out_shape_for_mode, matmul_out_shape, one_hot_out_shape,
    pad_out_shape, pool2d_out_shape, reduce_out_shape, require_same_shape, row_norm_layout,
    scatter_out_shape, sort_out_shape, topk_out_shape,
};

use crate::error::AutodiffError;
use crate::eval;
use crate::grad::{
    ArgExtremum, adaptive_avg_pool2d_with_fallback, argext_with_fallback, avg_pool2d_with_fallback,
    batch_norm_infer_with_fallback, batch_norm_train_with_fallback, cast_from_f32_with_fallback,
    concat_with_fallback, conv_transpose2d_with_fallback, conv2d_with_fallback,
    gather_with_fallback, interpolate_with_fallback, max_pool2d_with_fallback, min_with_fallback,
    one_hot_with_fallback, pad_with_fallback, scalar_binary_with_fallback,
    scalar_unary_with_fallback, scatter_with_fallback, sort_with_fallback, topk_with_fallback,
    unique_with_fallback,
};
use crate::tape::{NodeId, Op, Tape, materialize_fallible, materialize_non_fallible};

/// `Var::mse_loss_with` の縮約種別（#190・TASK-9.1c 相当。親イシュー
/// #189「損失関数（MSE・CrossEntropy）の実装」）。PyTorch
/// `nn.MSELoss(reduction=...)` の `mean`/`sum` に対応する。
///
/// `#[non_exhaustive]` とする理由: 将来 `none`（要素ごと損失。PyTorch
/// `reduction='none'` 相当）を追加しうるが、本イシューでは #190 実装
/// 計画のスコープ外（out-of-scope-tracking.md 準拠でユーザー承認後に
/// 別途追加）としたため、追加時に呼び出し側の非網羅的 `match` を破壊
/// しないようにする。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Reduction {
    /// 全要素平均（`Σ(pred−target)² / n`）。`Var::mse_loss` の既定
    /// （PyTorch `nn.MSELoss` の既定 `reduction='mean'` と一致）。
    Mean,
    /// 全要素総和（`Σ(pred−target)²`）。
    Sum,
}

/// `crate::var::Reduction` → `fandhe_ai_tensor_core::MseReduction` の変換
/// （イシュー #1045）。`tensor-core` → `autodiff` の逆依存は作れないため
/// `MseReduction` は `Reduction` の再エクスポートではなく独立した型
/// （`backend_ops.rs::MseReduction` doc 参照）であり、`Var::mse_loss_with`
/// が `BackendOps::mse_loss`／`mse_loss_backward` を呼ぶ直前にここで変換
/// する。両者は `Mean`/`Sum` の 2 variant のみで意味論も同一のため
/// 単純な 1 対 1 写像。
impl From<Reduction> for MseReduction {
    fn from(value: Reduction) -> Self {
        match value {
            Reduction::Mean => MseReduction::Mean,
            Reduction::Sum => MseReduction::Sum,
        }
    }
}

/// `Op::MatMul` の forward 値計算（イシュー #1715）。`Var::matmul`・
/// `Tape::checkpoint` の再計算（`tape.rs` の `recompute_value`）が
/// 共有する単一の分岐点であり、rank 2 同士は [`BackendOps::gemm`]
/// （既存の bit 一致契約を保つため直接委譲）、rank≥3 を含む場合は
/// [`BackendOps::gemm_batched`]（NumPy 互換バッチブロードキャスト）へ
/// 分岐する。`ops` は `dyn BackendOps` として渡され、CPU／CUDA／Metal
/// いずれのバックエンドでも同一コードパスから呼び分けられる
/// （`tensor-core::backend_ops` の設計方針を踏襲）。
pub(crate) fn matmul_forward(
    ops: &dyn BackendOps,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
) -> Result<Tensor<f32>, BackendError> {
    if a.shape().len() == 2 && b.shape().len() == 2 {
        ops.gemm(a, b)
    } else {
        ops.gemm_batched(a, b)
    }
}

/// テープ上の 1 ノードを指す追跡対象値。値そのものではなく `NodeId` +
/// テープへの共有参照を保持し、演算のたびにテープへ新しいノードを
/// 追加する（`docs/public-api-design.md` §3.1）。
///
/// **クロステープ安全性**: ライフタイム `'t` の一致は同一 `Tape` を
/// 指す証明にはならない（同一スコープに複数の `Tape` が存在する場合、
/// それぞれの `Var<'t>` は同一の `'t` を持ちうる）。そのため二項演算
/// （`matmul`/`add`/`mul`/`mse_loss`）は入口で `self.tape.id` と相手側
/// `Var` が保持する `TapeId` の一致を実行時検査し、不一致なら
/// `AutodiffError::TapeMismatch` を返す。
#[derive(Debug, Clone, Copy)]
pub struct Var<'t> {
    tape: &'t Tape,
    id: NodeId,
}

impl<'t> Var<'t> {
    /// `Tape::var()` からのみ呼ばれる内部コンストラクタ。
    pub(crate) fn from_raw(tape: &'t Tape, id: NodeId) -> Var<'t> {
        Var { tape, id }
    }

    /// 追跡を外し、現在の値を非追跡の `Tensor<f32>` の借用として取り出す。
    ///
    /// **TASK-12.1d（#164）**: 対象ノードが未実体化（elementwise の遅延
    /// グラフの一部）であれば `materialize_non_fallible`（層 2。融合を
    /// 試み、失敗すれば `ops` の per-op メソッド → `eval.rs` の順に必ず
    /// 値を返す）経由で実体化する。`matmul`/`sum`/`max`・`Tape::backward`
    /// が使う層 1（`crate::tape::materialize_fallible`）とは異なる
    /// エラー処理契約を持つ（`docs/fusion-graph-design.md` §3.5.3）。
    ///
    /// **借用注意**: この `Ref` を保持したまま、同じ `Tape` に対して
    /// `borrow_mut()` を要する演算（`matmul`/`add` 等のノード追加）を
    /// 呼ぶと `RefCell` の二重可変借用で実行時 panic になる。値をその場
    /// の参照ではなく所有値として持ち出したい場合は `to_tensor()` を
    /// 使うこと（`docs/public-api-design.md` §3.1）。
    pub fn value(&self) -> Ref<'_, Tensor<f32>> {
        Ref::map(self.tape.nodes.borrow(), |nodes| {
            materialize_non_fallible(nodes, self.tape.ops(), self.id)
        })
    }

    /// `value()` の所有値版。`Tensor<f32>` へ複製して返すため `Ref` を
    /// 持ち越さず、直後に同じ `Tape` へノード追加演算を呼んでも借用
    /// エラー・panic が起きない。
    pub fn to_tensor(&self) -> Tensor<f32> {
        let nodes = self.tape.nodes.borrow();
        materialize_non_fallible(&nodes, self.tape.ops(), self.id).clone()
    }

    /// ホスト可視の値を借用で読み出す（イシュー #1335・`docs/public-api-
    /// design.md` §3.1「`VarHostView`」）。
    ///
    /// **P1 是正（codex-review 指摘・イシュー #1335）**: 当初実装は
    /// `Tape` の `RefCell` 借用（`Ref<'_, [f32]>`）をそのまま
    /// `VarHostView` へ持ち越しており、`host_view()` を保持したまま
    /// 同じ `Tape` へノード追加演算（`add`/`matmul` 等の `push_lazy`）を
    /// 呼ぶと `borrow_mut()` が実行時 panic した（本番経路 panic 禁止・
    /// `.claude/rules/coding-rust.md` 違反）。`value()` の既存制約と
    /// 同型ではあるが、新規公開 API がその制約をそのまま持ち込む理由には
    /// ならないため、`RefCell` 借用を関数内に閉じ込める形へ是正した。
    ///
    /// `Tensor<f32>` は内部 `storage: Arc<Storage<T>>` を `Arc` 共有する
    /// 値型（`tensor.rs` モジュール冒頭コメント「`Clone` は `Arc` の
    /// ポインタ複製のみで安価」参照）であるため、`materialize_non_fallible`
    /// が返す `&Tensor<f32>`（`Tape` の `RefCell` 借用が生存中のみ有効）
    /// を `Tensor::contiguous()`（contiguous な場合は内部で `self.clone()`
    /// する安価な `Arc` 複製、非 contiguous な場合のみ 1 回実体化）へ
    /// 通してから所有値として持ち出せば、いずれの分岐でもデータの
    /// 追加コピーなしに `RefCell` 借用（`nodes`）をこの関数のスコープ内
    /// で確実に解放できる。返す [`VarHostView`] は `Tape` の借用を一切
    /// 保持しないため、生存中に同じ `Var`／`Tape` への他の演算を呼んでも
    /// panic しない。
    pub fn host_view(&self) -> VarHostView {
        let tensor = {
            let nodes = self.tape.nodes.borrow();
            materialize_non_fallible(&nodes, self.tape.ops(), self.id).contiguous()
        };
        VarHostView { tensor }
    }

    /// 実体化なしに読める構造的な出力 shape（`TapeNode.shape`。
    /// TASK-12.1d・#164）。演算入口の shape 検査は本メソッドを使い、
    /// `value()`/`materialize_fallible` を呼ばない（`docs/
    /// fusion-graph-design.md` §3.5.1「shape 検証と実行を分離する」）。
    ///
    /// **`pub(crate)` 化（イシュー #1620）**: `crate::einsum` が
    /// 添字ごとの次元サイズ検査・presum／permute／reshape の計画算出に
    /// 使う唯一のクレート外モジュール呼び出し元。`Var` は facade から
    /// 再エクスポートされるが、本メソッド自体は facade へは公開しない
    /// （`pub` にすると `docs/compat-api-scope.md` §5 手続きの対象になる
    /// 新規公開 API を無断で追加してしまう）。
    pub(crate) fn shape(&self) -> Vec<usize> {
        self.tape.nodes.borrow()[self.id.0].shape.clone()
    }

    /// 演算入口で必ず shape 検査より前に呼ぶクロステープ検査
    /// （`docs/public-api-design.md` §3.1「クロステープ安全性」）。
    ///
    /// **`pub(crate)` 化（イシュー #1620）**: `crate::einsum::einsum`
    /// が複数オペランド間のクロステープ検査に使う（`shape()` と同じ
    /// 理由で `pub` にはしない）。
    pub(crate) fn check_same_tape(&self, other: &Var<'t>) -> Result<(), AutodiffError> {
        if self.tape.id != other.tape.id {
            return Err(AutodiffError::TapeMismatch);
        }
        Ok(())
    }

    /// この `Var` が属する `Tape` の識別子。`backward.rs`（TASK-1.5c・
    /// #18）は別モジュールのため `tape` フィールド（private）へ直接
    /// 触れられず、`Tape::backward`/`Gradients::get` のクロステープ検査
    /// （`check_same_tape` と同じ「入口で必ず shape・NodeId 解決より前に
    /// 検査する」契約）にこのアクセサを使う。
    pub(crate) fn tape_id(&self) -> crate::tape::TapeId {
        self.tape.id
    }

    /// この `Var` が属する `Tape` そのものへの参照（イシュー #1721）。
    /// `nn::optim::amp::scale_loss` が `loss` の値から新たなスカラー葉
    /// （`Tape::var(&Tensor::scalar(scale))`）を同じ `Tape` 上へ登録する
    /// のに使う（`amp` モジュールは `var.rs` の外にあり `tape` フィールド
    /// （private）へ直接触れられないため、`tape_id`/`tape_epoch` と同じ
    /// `pub(crate)` アクセサ方針でクレート内限定公開する）。
    pub(crate) fn tape(&self) -> &'t Tape {
        self.tape
    }

    /// この `Var` が指すテープ内ノードの識別子。`backward.rs` が
    /// `Gradients` から当該ノードの勾配を引くための添字として使う
    /// （`tape_id()` と同じくクレート内限定公開）。
    pub(crate) fn node_id(&self) -> NodeId {
        self.id
    }

    /// この `Var` が属する `Tape` の現在の世代番号（#1048）。`Gradients::get`
    /// が「この `Var` は reset 後の別世代のものか」を fail-closed に検査
    /// するための比較対象（`tape::Tape::epoch` doc 参照）。`Var<'t>` 自体は
    /// `&'t Tape` を静的に借用するため reset 後に stale な `Var` を作る
    /// ことはコンパイル時に排除されるが、`Gradients`（`Tape` を借用しない
    /// 値）は reset をまたいで生存しうるため、こちらは実行時検査が要る。
    pub(crate) fn tape_epoch(&self) -> u64 {
        self.tape.epoch()
    }

    /// この `Var` が指すノードの勾配追跡フラグ（イシュー #1748・
    /// `TapeNode::requires_grad` doc 参照）。`backward.rs::Gradients::get`
    /// が「対象ノードが構造的に勾配を持ちうるか」を判定するのに使う
    /// （`tape_id`/`tape_epoch` と同じ `pub(crate)` アクセサ方針）。
    pub(crate) fn requires_grad(&self) -> bool {
        self.tape.nodes.borrow()[self.id.0].requires_grad
    }

    /// 追跡を切り離し、現在の値を `requires_grad == false` の新しい
    /// 葉ノードとして同じ `Tape` へ登録する（イシュー #1748。PyTorch
    /// `Tensor.detach()` 相当。設計は `docs/
    /// autodiff-nograd-leaf-dinput-skip-decision.md` §5「案 B」）。
    ///
    /// **専用 `Op` を追加しない設計**: 「detach された `Var`」を
    /// 「`requires_grad == false` の通常の葉ノード」として表現する
    /// （`Tape::var_no_grad` と実体を共有する）。専用 `Op::Detach`
    /// variant を追加する案は、前方伝播（`Op::for_each_input` の網羅
    /// match）・`is_checkpoint_eligible`・`grad::vjp` の網羅 match に
    /// 3 箇所以上の更新を要するのに対し、本 issue が満たすべき契約
    /// （値を共有しつつ勾配経路を切る）は既存の葉ノード機構で過不足
    /// なく表現できるため採用しなかった。
    ///
    /// **値の共有**: `Tensor<f32>` は内部 `storage: Arc<Storage<T>>` を
    /// 共有する値型（immutable）のため、新しい葉ノードへ渡す
    /// `materialize_fallible` の戻り値の `.clone()` は `Arc` のポインタ
    /// 複製のみで実データのコピーは発生しない（PyTorch の `detach()` が
    /// storage を共有するのと同型）。
    ///
    /// **副作用**: 対象がまだ実体化されていない lazy elementwise 連鎖・
    /// view ノードの場合、本メソッド呼び出し時点でその実体化が走る
    /// （`materialize_fallible` 経由）。また checkpoint 解放済み
    /// （イシュー #1624）で再計算に失敗（poison）した値を `detach` する
    /// と `Err` を返す（stale／不正な値を新しい葉へ複製しない
    /// fail-closed 方針）。
    pub fn detach(&self) -> Result<Var<'t>, AutodiffError> {
        let value = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let id = self.tape.push_leaf(value, false);
        Ok(Var::from_raw(self.tape, id))
    }

    /// この `Var` が属する `Tape` のデバイス（イシュー #1614）。
    /// `Tape::device()`（`self.tape.ops.device()` への委譲）をそのまま
    /// 返す。
    pub fn device(&self) -> Device {
        self.tape.device()
    }

    /// PyTorch `Tensor.to(device)` 相当の同一デバイス検査（イシュー
    /// #1614）。`Var` は所属する 1 つの `Tape`（＝1 デバイス）に束縛
    /// されており（本ファイル冒頭「クロステープ安全性」）、同一テープ
    /// 内でデバイスを差し替える演算は表現できない。そのため本メソッドは
    /// 実際に値を転送する変換ではなく、**要求先デバイスが現在のデバイス
    /// と一致するかを検査する fail-fast の入口**として設計している:
    ///
    /// - `device == self.device()` の場合: 恒等（`*self` をそのまま
    ///   返す。`Var: Copy` のためテープへ新規ノードを追加しない）。
    /// - 不一致の場合: [`AutodiffError::DeviceMismatch`]。黙って別
    ///   デバイスへフォールバックしたり、暗黙にクロステープ複製（勾配
    ///   経路が切れる）したりしない。
    ///
    /// 別デバイスへ実際に値を転送したい場合は [`Self::to_tape`]（もしくは
    /// facade `Tape::transfer`）を使うこと——`to_tape` は新しい `Tape`
    /// 上へ値を複製した葉ノードとして登録するため、テープをまたぐ
    /// 明示的な操作であることが型シグネチャ（戻り値のライフタイムが
    /// 変わる）からも分かる。
    pub fn to(&self, device: Device) -> Result<Var<'t>, AutodiffError> {
        let actual = self.device();
        if device == actual {
            Ok(*self)
        } else {
            Err(AutodiffError::DeviceMismatch {
                requested: device,
                actual,
            })
        }
    }

    /// この `Var` の値を、別の `Tape`（`target`。同一デバイスでもよい）
    /// 上へ新しい葉ノードとして転送する（イシュー #1614。facade
    /// `Tape::transfer` の実体）。
    ///
    /// **`target` が同一 `Tape` の場合は恒等**（`Var::detach` と異なり
    /// 追跡は切り離さない——同一テープへの「転送」は no-op として
    /// 扱う）。この判定は `self.tape.nodes` の借用（`materialize_
    /// fallible` が要る `Ref`）より**前**に行う必要がある——後回しに
    /// すると、同一テープの場合に不変借用が生きたまま `target.
    /// push_leaf`（`borrow_mut`）を呼ぶことになり `RefCell` の二重
    /// 可変借用で実行時 panic になる（早期 return により到達不能に
    /// する）。
    ///
    /// 別テープの場合は `crate::tape::materialize_fallible`
    /// （`Var::detach` と同じ層 1 の実体化経路。lazy elementwise 連鎖・
    /// view ノードはここで確定し、checkpoint 解放済みで再計算に失敗
    /// した poison 値は `Err` で fail-closed に拒否する）で転送元
    /// デバイス上の値を取り出し、`Tensor<f32>` の `Arc` 共有複製
    /// （実データコピーなし）を `target.push_leaf` で新しい葉として
    /// 登録する。
    ///
    /// **数値契約**: 転送は算術を含まないホスト値の受け渡しのため、
    /// 転送元で実体化された値と転送後の葉の値は bit 完全一致する。
    /// **勾配契約**: 勾配はテープをまたがない（`Var::detach`／`Tape::
    /// var_no_grad` と同じ「非微分境界」）。転送先では通常の葉として
    /// `requires_grad()`（転送元の値を引き継ぐ）に従って勾配を受け取る
    /// ——転送元 `Var` 自身の勾配経路には一切影響しない。**`reset`
    /// 契約**: 転送先の `Tape` で既に非葉ノードが積まれている場合、
    /// この葉は以後の [`crate::tape::Tape::reset`] で truncate される
    /// （既存 `Tape::var` と同じ挙動）。
    pub fn to_tape<'u>(&self, target: &'u Tape) -> Result<Var<'u>, AutodiffError> {
        if target.id == self.tape.id {
            // 早期 return: 下の `self.tape.nodes.borrow()` より前に
            // 同一テープ判定を済ませることで、`target.push_leaf`
            // （= `self.tape.push_leaf`）が要求する可変借用との衝突を
            // 構造的に回避する（このブロックへ来た時点でまだ `nodes`
            // を借用していない）。
            //
            // SAFETY 相当の注記ではなく型の話: `target: &'u Tape` と
            // `self.tape: &'t Tape` が指す実体が同一であることは
            // `TapeId` の一致で保証されるが、ライフタイム `'t`／`'u`
            // は静的には無関係な可能性があるため、`self` から作った
            // `Var<'t>` をそのまま `Var<'u>` として返すことはできない。
            // `Var::from_raw(target, self.id)`（`target` 由来の
            // ライフタイム）で組み直す。
            return Ok(Var::from_raw(target, self.id));
        }
        let value = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let id = target.push_leaf(value, self.requires_grad());
        Ok(Var::from_raw(target, id))
    }

    /// `matmul`（rank≥2。バッチ次元は NumPy 互換ブロードキャスト。
    /// `docs/public-api-design.md` §3.2・spec REQ-9 2026-09-12 追記
    /// Tier 1「バッチ行列積」・`docs/compat-api-scope.md` §1.2。
    /// イシュー #1715 でバッチ次元対応へ拡張。rank 2 同士は従来どおり
    /// `ops.gemm` を呼ぶため既存の bit 一致契約〈`tape_matmul_cpu_bit_exact.rs`・
    /// repack カウンタテスト〉は不変）。
    ///
    /// **TASK-12.1d（#164）**: 非 elementwise のため常に実体化済みで
    /// 返る（`push_eager`）。実行は `eval.rs` 直接呼び出しから
    /// `matmul_forward`（`BackendOps` 経由。rank 2 は `ops.gemm`・
    /// rank≥3 を含む場合は `ops.gemm_batched` へ分岐）へ置き換えた
    /// （TASK-1.9「backend 経由実行への置き換え」・設計書 §3.5.2）。
    /// 入力が elementwise の遅延グラフであった場合は
    /// `materialize_fallible`（層 1）で自身の実行の一部として実体化
    /// する。
    pub fn matmul(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(other)?;
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        matmul_out_shape(&lhs_shape, &rhs_shape)?;
        let (lhs_val, rhs_val) = {
            let nodes = self.tape.nodes.borrow();
            let lhs_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let rhs_val = materialize_fallible(&nodes, self.tape.ops(), other.id)?.clone();
            (lhs_val, rhs_val)
        };
        let value = matmul_forward(self.tape.ops(), &lhs_val, &rhs_val)?;
        let id = self.tape.push_eager(Op::MatMul(self.id, other.id), value);
        Ok(Var::from_raw(self.tape, id))
    }

    /// [`Var::matmul`] の FP32 厳密版（rank 2 限定）。`ops.gemm`
    /// （CUDA `crate::precision` の TF32 opt-in フラグに従う）ではなく
    /// 常に `ops.gemm_fp32_strict` で forward 値を計算する点のみが
    /// 異なり、記録する `Op::MatMul` ノード自体・shape 検証・クロス
    /// テープ検査は [`Var::matmul`] と同一。
    ///
    /// [`create_graph::backward_create_graph`](crate::create_graph)
    /// の子テープ上 MatMul VJP（`da = g.matmul(&bᵀ)`・
    /// `db = aᵀ.matmul(&g)`）専用（codex-review 指摘。PR #2003）:
    /// 1 階 `grad.rs::matmul_vjp` は既に `ops.gemm_fp32_strict` を
    /// 使っており、CUDA TF32 opt-in（`set_cuda_gemm_precision`）が
    /// 有効な間もバックプロパゲーションだけは FP32 厳密のまま保つ
    /// 契約（`BackendOps::gemm_fp32_strict` doc 参照）。子テープ上で
    /// `Var::matmul`（`ops.gemm`）を使うと、この契約が二階微分の
    /// 記録経路でだけ破られ、勾配が TF32 相当まで精度低下する
    /// （REQ-2 の統一複合判定を外れうる）。rank 2 限定なのは
    /// [`create_graph::validate_ancestors`](crate::create_graph) が
    /// rank≥3 の `MatMul` を子テープ記録の対象から事前拒否しており、
    /// このメソッドの呼び出し元では常に rank 2 であるため
    /// （バッチ版 `gemm_batched_fp32_strict` は未使用）。
    pub(crate) fn matmul_fp32_strict(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(other)?;
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        matmul_out_shape(&lhs_shape, &rhs_shape)?;
        let (lhs_val, rhs_val) = {
            let nodes = self.tape.nodes.borrow();
            let lhs_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let rhs_val = materialize_fallible(&nodes, self.tape.ops(), other.id)?.clone();
            (lhs_val, rhs_val)
        };
        let value = self.tape.ops().gemm_fp32_strict(&lhs_val, &rhs_val)?;
        let id = self.tape.push_eager(Op::MatMul(self.id, other.id), value);
        // `TapeNode::fp32_strict` を立てる（codex-review 指摘。
        // PR #2003）: `push_eager` 自体は通常版・厳密版の呼び出し元を
        // 区別しないため、戻り値ノードへ限定してここで事後設定する。
        // これにより activation checkpointing（`release_checkpoint_
        // region`）が本ノードの forward 値を解放しなくなり、以後の
        // 再計算が非厳密な `matmul_forward`（`ops.gemm`）へ落ちる事故
        // （厳密精度契約が checkpoint 経由で失われる）を防ぐ。
        self.tape.nodes.borrow_mut()[id.0].fp32_strict = true;
        Ok(Var::from_raw(self.tape, id))
    }

    /// `C = self @ other` を計算しつつ、`C` の全要素和（checksum）を
    /// バックエンド側の `f64` reduction（`BackendOps::gemm_checksum`）で
    /// 求める（イシュー #1339）。framework-compare の gemm 計測窓が毎
    /// 反復行っていた「D2H → ホスト `f64` 逐次和」の 2 段を、GPU
    /// バックエンドでは「checksum（8 バイト）のみ読み戻す」経路へ置き
    /// 換えるための入口（`docs/perf/device-checksum-readback-ab.md`）。
    ///
    /// **[`Var::matmul`] との相違（tape への非記録）**: 本メソッドは
    /// `self.tape.push_eager` を呼ばず、戻り値の `GemmChecksum` は
    /// tape ノードを持たない生の計算結果として返す（backward の対象外。
    /// ベンチハーネスの計測専用入口という位置づけであり、学習経路
    /// （`Var::matmul` チェーン）とは独立している）。`readout` が
    /// [`ChecksumReadout::ChecksumOnly`] のときバックエンド実装は `C`
    /// をホストへ download しない契約（`BackendOps::gemm_checksum` doc
    /// 参照）。
    ///
    /// 検証手順は `matmul` と同じ「①クロステープ検査 → ②shape 検査 →
    /// ③入力実体化」までを行い、④のみ `ops().gemm` ではなく
    /// `ops().gemm_checksum` を呼ぶ。既定実装（CUDA／Metal は本イシュー
    /// 時点で未オーバーライド）が返す [`BackendError::Unsupported`] は
    /// そのまま呼び出し元（framework-compare ハーネス）へ伝播する
    /// （判定迂回経路を作らない。`.claude/rules/security.md` A08）。
    pub fn matmul_checksum(
        &self,
        other: &Var<'t>,
        readout: ChecksumReadout,
    ) -> Result<GemmChecksum, AutodiffError> {
        self.check_same_tape(other)?;
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        gemm_out_shape(&lhs_shape, &rhs_shape)?;
        let (lhs_val, rhs_val) = {
            let nodes = self.tape.nodes.borrow();
            let lhs_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let rhs_val = materialize_fallible(&nodes, self.tape.ops(), other.id)?.clone();
            (lhs_val, rhs_val)
        };
        Ok(self.tape.ops().gemm_checksum(&lhs_val, &rhs_val, readout)?)
    }

    /// `y = act(self.matmul(weight) (+ bias))` を 1 ノード
    /// （[`Op::LinearAct`]）として記録する（イシュー #1044・`docs/
    /// kernel-fusion.md` §2.2「学習経路への結線」）。
    /// `fandhe_ai_autodiff::nn::linear::LinearVars::forward_with_activation`
    /// が唯一の呼び出し元（`Var::matmul` と同じ「①クロステープ検査 →
    /// ②shape 検査 → ③forward 値計算 → ④ノード記録」の順で処理する
    /// 非 elementwise・常時実体化の演算）。
    ///
    /// `bias` の shape 検証は `broadcast_shape`（`out_shape` へブロード
    /// キャスト可能かの NumPy 互換判定）のみを行い、`[n]`（`weight` の
    /// 列数）と厳密一致しない bias（`[1, n]` 等）も含めてそのまま
    /// `BackendOps::gemm_bias_act` へ委譲する。**非融合合成へのフォール
    /// バックは本メソッド・呼び出し元（`LinearVars::
    /// forward_with_activation`）のどちらの責務でもなく、
    /// `BackendOps::gemm_bias_act` 自身の契約**（`tensor-core::
    /// backend_ops` の doc 参照。CPU／CUDA／Metal の融合カーネル実装は
    /// bias が `[n]` 厳密一致でない場合 `matmul` → `add`（NumPy 互換
    /// ブロードキャスト）→ activation の非融合合成へ内部的に
    /// フォールバックし、デフォルト実装も同じ合成のため、いずれの
    /// バックエンドでも `[n]` 以外の broadcast 可能な bias が
    /// `ShapeMismatch` になることはない）。本メソッドが呼び出し前に
    /// `broadcast_shape` で検証するのは「`gemm_bias_act` に委譲する前に
    /// ブロードキャスト不能な shape を早期に拒否する」ためであり、
    /// フォールバック経路の選択自体は行わない。
    pub(crate) fn linear_act(
        &self,
        weight: &Var<'t>,
        bias: Option<&Var<'t>>,
        act: Activation,
    ) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(weight)?;
        if let Some(b) = bias {
            self.check_same_tape(b)?;
        }
        let lhs_shape = self.shape();
        let rhs_shape = weight.shape();
        let out_shape = gemm_out_shape(&lhs_shape, &rhs_shape)?;
        if let Some(b) = bias {
            broadcast_shape(&out_shape, &b.shape())?;
        }
        let (lhs_val, rhs_val, bias_val) = {
            let nodes = self.tape.nodes.borrow();
            let lhs_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let rhs_val = materialize_fallible(&nodes, self.tape.ops(), weight.id)?.clone();
            let bias_val = match bias {
                Some(b) => Some(materialize_fallible(&nodes, self.tape.ops(), b.id)?.clone()),
                None => None,
            };
            (lhs_val, rhs_val, bias_val)
        };
        let value = self
            .tape
            .ops()
            .gemm_bias_act(&lhs_val, &rhs_val, bias_val.as_ref(), act)?;
        let id = self.tape.push_eager(
            Op::LinearAct {
                input: self.id,
                weight: weight.id,
                bias: bias.map(|b| b.id),
                act,
                compute_dtype: fandhe_ai_tensor_core::ScalarDType::F32,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// [`Self::linear_act`] の低精度版（イシュー #1960・親 #1626／
    /// #1648）。`fandhe_ai_tensor_core::linear_forward_low_precision`
    /// （`TypedOps<f16>`／`TypedOps<bf16>` 経由。CPU 昇格→降格ソフト
    /// ウェア変換方式）へ forward 値計算のみを委譲する opt-in 経路。
    /// `nn::linear::linear_forward_low_precision`（自由関数。
    /// `LinearVars` へのフィールド追加〈破壊的変更〉を避けるための
    /// 配置）の唯一の呼び出し元。
    ///
    /// `weight`／`bias`（f32 master 値）・backward（`grad::vjp` の
    /// `Op::LinearAct` 分岐。常に f32）は [`Self::linear_act`] と完全に
    /// 同一——低精度なのは forward の GEMM／bias 加算／activation の
    /// 計算過程のみで、テープに記録する `Tensor<f32>` 値自体は既存
    /// `LinearAct` と同じ f32 表現（`compute_dtype` フィールドで記録
    /// 経路のみを区別する）。①クロステープ検査 → ②shape 検査 →
    /// ③forward 値計算 → ④ノード記録の順序も [`Self::linear_act`] と
    /// 同一（`Var::matmul` 以来の共通パターン）。
    pub(crate) fn linear_act_low_precision(
        &self,
        weight: &Var<'t>,
        bias: Option<&Var<'t>>,
        act: Activation,
        dtype: fandhe_ai_tensor_core::ScalarDType,
    ) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(weight)?;
        if let Some(b) = bias {
            self.check_same_tape(b)?;
        }
        let lhs_shape = self.shape();
        let rhs_shape = weight.shape();
        let out_shape = gemm_out_shape(&lhs_shape, &rhs_shape)?;
        if let Some(b) = bias {
            // `broadcast_shape` の成功のみでは、`bias` が `out_shape`
            // （gemm の出力 shape）を「拡張」する場合（例: out_shape
            // `[1,1]`・bias `[2,1]`）を誤って受理してしまう。`grad::vjp`
            // の `Op::LinearAct` 分岐は upstream（実際の forward 出力
            // shape。add により拡張されていれば `[2,1]`）を `matmul_vjp`
            // へそのまま渡すため、`weight` の shape（`[2,1]`）との
            // 縮約次元が食い違い K 不一致で失敗する。勾配の形状契約を
            // 守るため、bias は `out_shape` へブロードキャスト「される」
            // 側（結果が `out_shape` と一致する）ことを検証し、拡張は
            // fail-closed に拒否する（codex-review 指摘・PR #2000）。
            let broadcast_result = broadcast_shape(&out_shape, &b.shape())?;
            if broadcast_result != out_shape {
                return Err(AutodiffError::Shape(ShapeError::BroadcastIncompatible {
                    lhs: out_shape,
                    rhs: b.shape().to_vec(),
                }));
            }
        }
        let (lhs_val, rhs_val, bias_val) = {
            let nodes = self.tape.nodes.borrow();
            let lhs_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let rhs_val = materialize_fallible(&nodes, self.tape.ops(), weight.id)?.clone();
            let bias_val = match bias {
                Some(b) => Some(materialize_fallible(&nodes, self.tape.ops(), b.id)?.clone()),
                None => None,
            };
            (lhs_val, rhs_val, bias_val)
        };
        let value = fandhe_ai_tensor_core::linear_forward_low_precision(
            self.tape.ops(),
            dtype,
            &lhs_val,
            &rhs_val,
            bias_val.as_ref(),
            act,
        )?;
        let id = self.tape.push_eager(
            Op::LinearAct {
                input: self.id,
                weight: weight.id,
                bias: bias.map(|b| b.id),
                act,
                compute_dtype: dtype,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// bias broadcast を含む要素ごとの加算（`docs/public-api-design.md`
    /// §3.2）。
    ///
    /// **TASK-12.1d（#164）**: elementwise 5 演算の 1 つ。shape 検証
    /// （①クロステープ検査・②shape 検査）のみ即時実行し、値計算
    /// （③）は実体化境界まで遅延させる（`push_lazy`。`Ok` を返すことは
    /// 「shape が妥当でノードが記録された」ことのみを意味し「加算が
    /// 計算済み」であることを意味しない。設計書 §3.5.1）。
    ///
    /// **連鎖長上限（#404・設計書 §3.5.4）**: `push_lazy` を呼ぶ**前**に
    /// `Tape::pre_materialize_for_binary_merge` で fan-in 事前実体化を
    /// 行う（2 本の未実体化枝を合流させた結果が単独で上限を超えるなら
    /// 大きい方の枝を先に実体化する。codex-review PR #406 の P1 是正。
    /// push 後の自己実体化だけでは fan-in を防げないため必須）。続けて
    /// `push_lazy` が返す `at_limit` が `true`（新規ノードの
    /// `lazy_chain_size` が `MAX_FUSED_CHAIN_LEN` に到達）の場合、層 1
    /// （`materialize_fallible`）でその場実体化する。**いずれの実体化
    /// も**発生した場合、`Ok` の意味は「shape が妥当でノードが記録され
    /// **かつバックエンド実行が成功した**」へ拡張される（同じ層 1 契約
    /// を持つ `matmul`/`sum`/`max` と同型の `Ok` 意味）。実体化失敗は
    /// （事前実体化・push 後の自己実体化のいずれも）`?` でそのまま伝播
    /// する。
    pub fn add(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(other)?;
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        let out_shape = broadcast_shape(&lhs_shape, &rhs_shape)?;
        // fan-in 事前実体化（#404・codex-review PR #406 の P1 是正）:
        // 2 本の未実体化枝を合流させる前に、合流後サイズが上限を超える
        // なら大きい方の枝を先に実体化する（`Tape::
        // pre_materialize_for_binary_merge` のドキュメント参照）。
        self.tape
            .pre_materialize_for_binary_merge(self.id, other.id)?;
        let (id, at_limit) = self.tape.push_lazy(Op::Add(self.id, other.id), out_shape);
        if at_limit {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), id)?;
        }
        Ok(Var::from_raw(self.tape, id))
    }

    /// ブロードキャスト付き要素ごとの乗算。elementwise 5 演算の 1 つ
    /// （`add` と同じ遅延契約・fan-in 事前実体化契約・連鎖長上限での
    /// 自己実体化契約。`Ok` の意味の拡張も `add` と同型。
    /// TASK-12.1d・#164・#404・codex-review PR #406）。
    pub fn mul(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(other)?;
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        let out_shape = broadcast_shape(&lhs_shape, &rhs_shape)?;
        // fan-in 事前実体化（`add` と同じ契約。#404・codex-review PR #406
        // の P1 是正）。
        self.tape
            .pre_materialize_for_binary_merge(self.id, other.id)?;
        let (id, at_limit) = self.tape.push_lazy(Op::Mul(self.id, other.id), out_shape);
        if at_limit {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), id)?;
        }
        Ok(Var::from_raw(self.tape, id))
    }

    /// `ScalarUnaryOp` 汎用 dispatch の `Var` 入口（イシュー #1634）。
    /// `pub(crate)`: 個別公開メソッド（[`Var::sqrt`]・[`Var::clamp`]・
    /// `log`／`log2`／`log10`／`sin`／`cos`／`tan`／`abs`／`neg`
    /// （イシュー #1710・#1711・#1712）・`gelu`／`gelu_tanh`／
    /// `softplus`（イシュー #1713）等。本ファイル下方）が薄い委譲で
    /// 公開する共通実装。残る活性化系（SiLU／LeakyReLU／ELU／
    /// Hardswish 等）の個別公開メソッドは #1714 が別途追加する。
    ///
    /// `where_cond`（`crate::grad::where_cond_with_fallback` 経由）と
    /// 同じ eager 実体化契約: ①入力値を層 1（[`materialize_fallible`]）
    /// で実体化し `RefCell` 借用を閉じる → ②`scalar_unary_with_fallback`
    /// （バックエンド実装 → `Unsupported` のときのみホスト参照実装
    /// フォールバック）→ ③`push_eager`。遅延融合（`push_lazy`）は
    /// 使わない（`Op::ScalarUnary` doc「eager」参照）。
    pub(crate) fn scalar_unary(&self, op: ScalarUnaryOp) -> Result<Var<'t>, AutodiffError> {
        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = scalar_unary_with_fallback(self.tape.ops(), op, &input_val)?;
        let id = self
            .tape
            .push_eager(Op::ScalarUnary { op, input: self.id }, value);
        Ok(Var::from_raw(self.tape, id))
    }

    /// `ScalarBinaryOp` 汎用 dispatch の `Var` 入口（イシュー #1634）。
    /// [`Var::scalar_unary`] の 2 項版で設計方針は同一
    /// （`pub(crate)`・eager・フォールバック契約）。`add`／`mul` と同じ
    /// NumPy 互換ブロードキャスト（`broadcast_shape`）。個別公開メソッド
    /// （[`Var::sub`]／[`Var::div`]／[`Var::pow`]／[`Var::gt`] 等 6 種の
    /// 比較演算。本ファイル下方）が呼ぶ共通実装（イシュー #1710・
    /// #1712）。
    pub(crate) fn scalar_binary(
        &self,
        other: &Var<'t>,
        op: ScalarBinaryOp,
    ) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(other)?;
        let out_shape = broadcast_shape(&self.shape(), &other.shape())?;
        let (a_val, b_val) = {
            let nodes = self.tape.nodes.borrow();
            let ops = self.tape.ops();
            let a_val = materialize_fallible(&nodes, ops, self.id)?.clone();
            let b_val = materialize_fallible(&nodes, ops, other.id)?.clone();
            (a_val, b_val)
        };
        let value = scalar_binary_with_fallback(self.tape.ops(), op, &a_val, &b_val, &out_shape)?;
        let id = self.tape.push_eager(
            Op::ScalarBinary {
                op,
                a: self.id,
                b: other.id,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 自然対数 `ln(x)`。`ScalarUnaryOp::Log` への薄い委譲
    /// （`Var::scalar_unary` 参照。イシュー #1711・親 #1593・#1592）。
    /// 定義域外（`x <= 0`）は IEEE のまま（`x == 0` は `-inf`・
    /// `x < 0` は `NaN`。panic しない）。導関数は `1/x`
    /// （`eval::scalar::unary_grad_factor`）。
    pub fn log(&self) -> Result<Var<'t>, AutodiffError> {
        self.scalar_unary(ScalarUnaryOp::Log)
    }

    /// 底 2 の対数 `log2(x)`。数値規約は [`Var::log`] と同じ
    /// （定義域外は IEEE のまま）。導関数は `1/(x·ln2)`。
    pub fn log2(&self) -> Result<Var<'t>, AutodiffError> {
        self.scalar_unary(ScalarUnaryOp::Log2)
    }

    /// 底 10 の対数 `log10(x)`。数値規約は [`Var::log`] と同じ
    /// （定義域外は IEEE のまま）。導関数は `1/(x·ln10)`。
    pub fn log10(&self) -> Result<Var<'t>, AutodiffError> {
        self.scalar_unary(ScalarUnaryOp::Log10)
    }

    /// 正弦 `sin(x)`。`ScalarUnaryOp::Sin` への薄い委譲。導関数は
    /// `cos(x)`（`eval::scalar::unary_grad_factor`）。
    pub fn sin(&self) -> Result<Var<'t>, AutodiffError> {
        self.scalar_unary(ScalarUnaryOp::Sin)
    }

    /// 余弦 `cos(x)`。導関数は `-sin(x)`。
    pub fn cos(&self) -> Result<Var<'t>, AutodiffError> {
        self.scalar_unary(ScalarUnaryOp::Cos)
    }

    /// 正接 `tan(x)`。極（`x = π/2 + kπ` 近傍）でのマスクは行わず
    /// IEEE のまま（`inf`／`NaN` が伝播しうる）。導関数は
    /// `1/cos(x)^2`。
    pub fn tan(&self) -> Result<Var<'t>, AutodiffError> {
        self.scalar_unary(ScalarUnaryOp::Tan)
    }

    /// 絶対値 `|x|`。劣勾配は `x == 0` で `0`（`sign(0) = 0`。
    /// `ScalarUnaryOp::Abs` doc・`eval::scalar::unary_grad_factor`
    /// 参照）。
    pub fn abs(&self) -> Result<Var<'t>, AutodiffError> {
        self.scalar_unary(ScalarUnaryOp::Abs)
    }

    /// 符号反転 `-x`。`-0.0` の符号ビット反転を含め IEEE のまま。
    /// 導関数は定数 `-1`。
    pub fn neg(&self) -> Result<Var<'t>, AutodiffError> {
        self.scalar_unary(ScalarUnaryOp::Neg)
    }

    /// SiLU／Swish（`x * sigmoid(x)`。PyTorch `torch.nn.functional.silu`
    /// 相当）。イシュー #1714（親 #1595）。
    ///
    /// `Var::scalar_unary`（[`ScalarUnaryOp::Silu`]）への薄い委譲
    /// （`sqrt`／`log` と同型——eager 実体化契約により `Result` を返す）。
    /// 導関数は `fandhe_ai_tensor_core::scalar_op::silu_grad`
    /// （`eval::scalar::unary_grad_factor`）。CUDA／Metal は超越関数
    /// （`exp`）を含むため REQ-2 統一複合判定（bit 同一は主張しない）。
    pub fn silu(&self) -> Result<Var<'t>, AutodiffError> {
        self.scalar_unary(ScalarUnaryOp::Silu)
    }

    /// Hardswish（`x * clamp(x + 3, 0, 6) / 6`。PyTorch
    /// `torch.nn.functional.hardswish` 相当）。イシュー #1714
    /// （親 #1595）。
    ///
    /// `Var::scalar_unary`（[`ScalarUnaryOp::Hardswish`]）への薄い委譲。
    /// 選択・算術のみのため CUDA／Metal ともホスト `f32` 演算と bit
    /// 同一になる想定（`kernels_scalar_op.rs`／`scalar_op_source.rs`
    /// モジュール doc 参照）。導関数は
    /// `fandhe_ai_tensor_core::scalar_op::hardswish_grad`。
    pub fn hardswish(&self) -> Result<Var<'t>, AutodiffError> {
        self.scalar_unary(ScalarUnaryOp::Hardswish)
    }

    /// Leaky ReLU（`x >= 0` なら `x`、それ以外は `negative_slope * x`。
    /// PyTorch `torch.nn.functional.leaky_relu` 相当）。イシュー #1714
    /// （親 #1595）。
    ///
    /// `Var::scalar_unary`（[`ScalarUnaryOp::LeakyRelu`]）への薄い委譲。
    /// `negative_slope` は検証せず IEEE のまま伝播する（`NaN` を渡せば
    /// `NaN` が出る。PyTorch と同様）。選択・乗算のみのため bit 同一に
    /// なる想定。導関数は `x >= 0` で `1`、それ以外は `negative_slope`。
    pub fn leaky_relu(&self, negative_slope: f32) -> Result<Var<'t>, AutodiffError> {
        self.scalar_unary(ScalarUnaryOp::LeakyRelu { negative_slope })
    }

    /// ELU（`x > 0` なら `x`、それ以外は `alpha * (exp(x) - 1)`。
    /// PyTorch `torch.nn.functional.elu` 相当）。イシュー #1714
    /// （親 #1595）。
    ///
    /// `Var::scalar_unary`（[`ScalarUnaryOp::Elu`]）への薄い委譲。
    /// `alpha` は検証せず IEEE のまま伝播する。CUDA／Metal は超越関数
    /// （`expm1f`／`metal::precise::exp`）を含むため REQ-2 統一複合
    /// 判定のみ（Metal の `expm1` 非対応・数値誤差は
    /// `scalar_op_source.rs` モジュール doc「`Elu` の `expm1` 非対応」
    /// 参照）。導関数は `x > 0` で `1`、それ以外は `alpha * exp(x)`。
    pub fn elu(&self, alpha: f32) -> Result<Var<'t>, AutodiffError> {
        self.scalar_unary(ScalarUnaryOp::Elu { alpha })
    }

    /// ブロードキャスト付き要素ごとの減算（`self − other`。PyTorch
    /// `torch.sub`／`-` 演算子相当）。イシュー #1710（親 #1593）。
    ///
    /// `Var::scalar_binary`（[`ScalarBinaryOp::Sub`]）への薄い委譲
    /// （`add`／`mul` と同じ NumPy 互換ブロードキャスト・eager 実体化
    /// 契約。`docs/scalar-op-dispatch-design.md` §7）。`facade` は
    /// `crates/autodiff::Var` を再エクスポートするのみで新規公開面は
    /// 追加しない（`docs/compat-api-scope.md` §5「Tier 1／Tier 2 列挙
    /// 済み機能は再適用不要」）。
    pub fn sub(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.scalar_binary(other, ScalarBinaryOp::Sub)
    }

    /// ブロードキャスト付き要素ごとの除算（`self ÷ other`。PyTorch
    /// `torch.div`／`/` 演算子相当）。イシュー #1710（親 #1593）。
    ///
    /// `Var::scalar_binary`（[`ScalarBinaryOp::Div`]）への薄い委譲。
    /// **数値規約（設計 §7 を変更せず踏襲）**: IEEE 754 のまま
    /// （0 除算は `inf`／`NaN` を返し panic しない）。`db =
    /// -(a/b)/b` の 0 除算・overflow・underflow 耐性は #1634／#1686
    /// の是正済みホスト参照実装（`eval::scalar::binary_partials`）に
    /// 従う。
    pub fn div(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.scalar_binary(other, ScalarBinaryOp::Div)
    }

    /// ブロードキャスト付き要素ごとの冪乗（`self.powf(other)`。PyTorch
    /// `torch.pow`／`**` 演算子相当。Var × Var の 2 項演算のみで、
    /// スカラー指数版〈`ScalarUnaryOp::PowScalar`〉は CUDA／Metal
    /// カーネル未実装のため本メソッドの対象外）。イシュー #1710
    /// （親 #1593）。
    ///
    /// `Var::scalar_binary`（[`ScalarBinaryOp::Pow`]）への薄い委譲。
    /// **数値規約（設計 §7）**: `da = b·a^(b−1)`・`db = y·ln(a)`
    /// （`a == 0` または `b == 0` は #1686 是正によりマスクされ勾配 0。
    /// `a < 0` かつ `b` が非整数のとき forward は IEEE `NaN` を返す
    /// PyTorch 同様の定義域規約）。
    pub fn pow(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.scalar_binary(other, ScalarBinaryOp::Pow)
    }

    /// 要素ごとの平方根（PyTorch `torch.sqrt` 相当）。イシュー #1710
    /// （親 #1593）。
    ///
    /// `Var::scalar_unary`（[`ScalarUnaryOp::Sqrt`]）への薄い委譲
    /// （`relu`／`exp`／`tanh` と異なり `Result` を返す——`scalar_unary`
    /// の eager 実体化契約〈①層 1 実体化 → ②バックエンド dispatch →
    /// ③`push_eager`〉が型付きエラーを返しうるため。`where_cond`／
    /// `softmax`／`add`／`mul` と同型）。**数値規約（設計 §7）**: 定義域
    /// 外（`x < 0`）は IEEE `NaN` を返し panic しない（PyTorch
    /// `torch.sqrt` と同じ）。導関数は `0.5 / y`（`eval::scalar::
    /// unary_grad_factor`）。
    pub fn sqrt(&self) -> Result<Var<'t>, AutodiffError> {
        self.scalar_unary(ScalarUnaryOp::Sqrt)
    }

    /// 要素ごとの範囲制限（PyTorch `torch.clamp` 相当）。イシュー #1712
    /// （親 #1593）。
    ///
    /// `Var::scalar_unary`（[`ScalarUnaryOp::Clamp`]）への薄い委譲。
    /// **数値規約（`eval::scalar::unary_grad_factor` を変更せず踏襲）**:
    /// 範囲内（境界値を含む）は勾配係数 1・範囲外は 0。`NaN` 入力は
    /// forward がそのまま `NaN` を伝播し勾配は 0。`min > max` は
    /// （PyTorch と同じく）常に `max` を返す定数関数として扱われ、
    /// 勾配は常に 0（panic しない）。
    pub fn clamp(&self, min: f32, max: f32) -> Result<Var<'t>, AutodiffError> {
        self.scalar_unary(ScalarUnaryOp::Clamp { min, max })
    }

    /// ブロードキャスト付き要素ごとの大なり比較（`self > other`。
    /// PyTorch `torch.gt`／`>` 演算子相当）。イシュー #1712（親 #1593）。
    ///
    /// 比較演算 6 種（`gt`／`ge`／`lt`／`le`／`eq`／`ne`）は共通の出力・
    /// 勾配規約を持つ: 出力は f32 の `0.0`／`1.0`（bool dtype 出力は
    /// #1613 の対象で本メソッドの対象外。`docs/scalar-op-dispatch-design.md`
    /// §3.2）。IEEE 754 準拠の比較（`NaN` を含む比較は `eq` を含め常に
    /// 偽・`ne` のみ真）。VJP は両入力とも常にゼロ勾配（比較演算は
    /// 局所的に階段関数のため微分不可能。`eval::scalar::binary_partials`
    /// が `(0.0, 0.0)` を返す設計を `grad.rs::vjp` がそのまま `Some`
    /// として伝播——寄与を省略しない）。`add`／`mul` と同じ NumPy
    /// 互換ブロードキャスト。
    pub fn gt(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.scalar_binary(other, ScalarBinaryOp::Gt)
    }

    /// ブロードキャスト付き要素ごとの以上比較（`self >= other`。
    /// PyTorch `torch.ge` 相当）。数値規約は [`Var::gt`] を参照。
    /// イシュー #1712（親 #1593）。
    pub fn ge(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.scalar_binary(other, ScalarBinaryOp::Ge)
    }

    /// ブロードキャスト付き要素ごとの小なり比較（`self < other`。
    /// PyTorch `torch.lt` 相当）。数値規約は [`Var::gt`] を参照。
    /// イシュー #1712（親 #1593）。
    pub fn lt(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.scalar_binary(other, ScalarBinaryOp::Lt)
    }

    /// ブロードキャスト付き要素ごとの以下比較（`self <= other`。
    /// PyTorch `torch.le` 相当）。数値規約は [`Var::gt`] を参照。
    /// イシュー #1712（親 #1593）。
    pub fn le(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.scalar_binary(other, ScalarBinaryOp::Le)
    }

    /// ブロードキャスト付き要素ごとの等価比較（`self == other`。
    /// PyTorch `torch.eq` 相当）。数値規約は [`Var::gt`] を参照（`NaN`
    /// 同士は偽）。イシュー #1712（親 #1593）。
    pub fn eq(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.scalar_binary(other, ScalarBinaryOp::Eq)
    }

    /// ブロードキャスト付き要素ごとの非等価比較（`self != other`。
    /// PyTorch `torch.ne` 相当）。数値規約は [`Var::gt`] を参照（`NaN`
    /// が絡む比較は常に真）。イシュー #1712（親 #1593）。
    pub fn ne(&self, other: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.scalar_binary(other, ScalarBinaryOp::Ne)
    }

    /// GELU（誤差関数版。PyTorch `F.gelu(x, approximate='none')`
    /// 相当）: `0.5 * x * (1 + erf(x / sqrt(2)))`。イシュー #1713
    /// （親 #1595）。
    ///
    /// `Var::scalar_unary`（[`ScalarUnaryOp::Gelu`]）への薄い委譲
    /// （`sqrt` と同型）。**数値規約**: `erf` は依存追加不可
    /// （`.claude/rules/deps-policy.md`）のため `f64` 精度の自作近似
    /// （Abramowitz–Stegun 7.1.26。`scalar_op.rs::erf_f64`）を使う。
    /// 導関数は `Φ(x) + x·φ(x)`（`eval::scalar::unary_grad_factor`・
    /// `scalar_op::gelu_erf_grad`）。CUDA（`erff`）・Metal（自作
    /// `scalar_erf_f32` prelude）はいずれも超越関数のため REQ-2 統一
    /// 複合判定のみで検証する（bit 同一は主張しない）。`facade` は
    /// `crates/autodiff::Var` を再エクスポートするのみで新規公開面は
    /// 追加しない（`docs/compat-api-scope.md` §5）。
    pub fn gelu(&self) -> Result<Var<'t>, AutodiffError> {
        self.scalar_unary(ScalarUnaryOp::Gelu)
    }

    /// GELU（tanh 近似版。PyTorch `F.gelu(x, approximate='tanh')`
    /// 相当）:
    /// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`。
    /// イシュー #1713（親 #1595）。
    ///
    /// [`Var::gelu`] と同じ委譲・公開面方針。導関数は
    /// `scalar_op::gelu_tanh_grad`（huge magnitude 入力でも `NaN` を
    /// 生まない是正済み実装。`scalar_op.rs` 参照）。
    pub fn gelu_tanh(&self) -> Result<Var<'t>, AutodiffError> {
        self.scalar_unary(ScalarUnaryOp::GeluTanh)
    }

    /// Softplus（PyTorch `F.softplus(x, beta, threshold)` 相当）:
    /// `x * beta > threshold` なら恒等（`x`）、それ以外は
    /// `(1 / beta) * ln(1 + exp(beta * x))`。イシュー #1713
    /// （親 #1595）。
    ///
    /// `beta`（有限かつ `> 0`。`apply` が `/ beta` を含むため
    /// `beta == 0` は `inf`／`NaN` を生む）・`threshold`（有限）を
    /// dispatch 前に検査し、違反時は `AutodiffError::InvalidArgument`
    /// を返す（`nn/norm.rs::validate_eps` と同じ fail-closed 規律）。
    /// 導関数は `sigmoid(beta * x)`（恒等領域では `1.0`。
    /// `eval::scalar::unary_grad_factor`）。CUDA／Metal は超越関数の
    /// ため REQ-2 統一複合判定のみで検証する。`facade` は
    /// `crates/autodiff::Var` を再エクスポートするのみで新規公開面は
    /// 追加しない（`docs/compat-api-scope.md` §5）。
    pub fn softplus(&self, beta: f32, threshold: f32) -> Result<Var<'t>, AutodiffError> {
        if !beta.is_finite() || beta <= 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Var::softplus: beta must be finite and positive, got {beta}"
            )));
        }
        if !threshold.is_finite() {
            return Err(AutodiffError::InvalidArgument(format!(
                "Var::softplus: threshold must be finite, got {threshold}"
            )));
        }
        self.scalar_unary(ScalarUnaryOp::Softplus { beta, threshold })
    }

    /// `dim` に沿った縮約和。`dim: None` は全軸縮約（スカラー）。
    /// 非 elementwise のため常に実体化済みで返る（`matmul` と同じ
    /// TASK-12.1d の置き換え方針。実行は `self.tape.ops().sum` 経由）。
    ///
    /// `BackendOps::sum` が `BackendError::Unsupported` を返した場合も
    /// ホスト参照実装へは**フォールバックせず**、`AutodiffError::Backend`
    /// としてそのまま伝播する（[`Self::cumsum`] のフォールバック規律とは
    /// 対になる契約。案 C・2026-09-17 ユーザー承認済み。
    /// `docs/backend-metal-reduce-sum-design.md` §10。CPU／CUDA／Metal の
    /// 本番 `sum` は実装済みで、`Unsupported` は Metal のカーネル引数上限
    /// 超過等の到達しにくい経路に限られる。ホスト側 `eval::sum` は素の
    /// f32 逐次和で CPU 参照実装〈f64 チャンク 2 段〉と bit 一致しないため
    /// silent fallback を許すと `mean`／`var`／`std` 等 `sum` に依存する
    /// 演算へ数値方式の非一貫性が波及する）。
    pub fn sum(&self, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        reduce_out_shape(&shape, dim)?;
        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = self.tape.ops().sum(&input_val, dim)?;
        let id = self.tape.push_eager(
            Op::Sum {
                input: self.id,
                dim,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// `dim` に沿った縮約最大値。`dim: None` は全軸縮約（スカラー）。
    /// `sum` と同じ置き換え方針（`self.tape.ops().max` 経由）。
    pub fn max(&self, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        reduce_out_shape(&shape, dim)?;
        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = self.tape.ops().max(&input_val, dim)?;
        let id = self.tape.push_eager(
            Op::Max {
                input: self.id,
                dim,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// `dim` に沿った縮約最小値（`torch.min(dim).values` 相当。
    /// イシュー #1720）。`dim: None` は全軸縮約（スカラー）。[`Var::max`]
    /// と対称だが、`BackendOps::min` はデフォルトメソッド
    /// （既定 `Unsupported`）のため `min_with_fallback` 経由で
    /// ホスト参照実装（`eval::min`）へフォールバックする
    /// （`Var::sort`／`argsort` と同じ非破壊拡張方針）。空縮約
    /// （要素数 0）は単位元を持たないため `AutodiffError::
    /// InvalidArgument` を返す（`min_with_fallback` doc 参照）。
    /// `keepdim`／多軸縮約は非対応（別イシューの対象。#1719 の
    /// `max_dims` と同型の `min_dims` は本メソッドの対象外）。
    pub fn min(&self, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        let out_shape = reduce_out_shape(&shape, dim)?;
        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = min_with_fallback(self.tape.ops(), &input_val, dim, &out_shape)?;
        let id = self.tape.push_eager(
            Op::Min {
                input: self.id,
                dim,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// `dim` に沿った最大値の添字（`torch.argmax(dim)` 相当。イシュー
    /// #1720）。**非微分演算**でテープにノードを追加しない
    /// （[`Var::argsort`] と同じ扱い）。戻り値は `Tensor<i32>`
    /// （shape は `min`／`max` と同じ縮約 shape）。タイは最初の添字・
    /// NaN は無視（[`fandhe_ai_tensor_core::BackendOps::argmax`] doc
    /// 参照）。空縮約はエラー。
    pub fn argmax(&self, dim: Option<usize>) -> Result<Tensor<i32>, AutodiffError> {
        let shape = self.shape();
        let out_shape = reduce_out_shape(&shape, dim)?;
        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        argext_with_fallback(
            self.tape.ops(),
            &input_val,
            dim,
            &out_shape,
            ArgExtremum::Max,
        )
    }

    /// `dim` に沿った最小値の添字（`torch.argmin(dim)` 相当。イシュー
    /// #1720）。[`Var::argmax`] の最小値版で、非微分・空縮約エラー等の
    /// 契約は同一（NaN 規約は [`Var::min`] と整合）。
    pub fn argmin(&self, dim: Option<usize>) -> Result<Tensor<i32>, AutodiffError> {
        let shape = self.shape();
        let out_shape = reduce_out_shape(&shape, dim)?;
        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        argext_with_fallback(
            self.tape.ops(),
            &input_val,
            dim,
            &out_shape,
            ArgExtremum::Min,
        )
    }

    /// `dim` に沿った縮約平均。`dim: None` は全軸縮約（スカラー）。
    /// `sum`／`max` と同じ置き換え方針だが、`BackendOps` を拡張せず
    /// `self.tape.ops().sum` の結果をホスト側で縮約対象要素数 `n` に
    /// より **1 回だけ除算**する合成実装（CPU バックエンド側の参照
    /// 実装〈`reduction::mean`〉の「sum の後に 1 回だけ除算する」丸め
    /// 規律と同じ。`tape::Op::Mean` doc 参照）。
    ///
    /// **`n == 0`（縮約対象の要素数が 0）は
    /// [`AutodiffError::InvalidArgument`] で拒否する**（PyTorch は
    /// `NaN` を返すが、`.claude/rules/coding-rust.md` の本番経路
    /// panic 禁止方針の趣旨に合わせ、値が定義されない縮約は安全側で
    /// 明示的に拒否する。CPU バックエンド側の参照実装
    /// `ReduceError::EmptyReduction` と同じ考え方）。
    ///
    /// **REQ-9 Tier 1「縮約（mean）」根拠**: イシュー #1719・親 #1601
    /// 「Phase 2（Tier 1）」。`fandhe_ai::compat::array`／
    /// `crates/autodiff::Var` を再エクスポートするのみで新規公開面は
    /// 追加しない（`docs/compat-api-scope.md` §5「Tier 1 に列挙済みの
    /// 機能の実装は本節の再適用を要しない」）。
    pub fn mean(&self, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        reduce_out_shape(&shape, dim)?;
        let n: usize = match dim {
            None => shape.iter().product(),
            Some(axis) => shape[axis],
        };
        if n == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Var::mean: 縮約対象の要素数が 0 です".to_string(),
            ));
        }
        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let sum_value = self.tape.ops().sum(&input_val, dim)?;
        let value = {
            let data: Vec<f32> = eval::dense_vec(&sum_value)
                .into_iter()
                .map(|v| v / n as f32)
                .collect();
            eval::build_tensor(data, sum_value.shape())
        };
        let id = self.tape.push_eager(
            Op::Mean {
                input: self.id,
                dim,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// `sum_dims`／`max_dims`／`mean_dims` 共通の骨格（`crate::
    /// reduce_dims::plan_reduce_dims` で検証・算出した計画に従って
    /// `merge_for_reduction` で単軸縮約可能な形へ整形し、`reduce`
    /// （呼び出し元が渡す `sum`／`max`／`mean` いずれかの単軸縮約）を
    /// 適用してから `keepdim: true` のときだけ `reshape` する）。単軸
    /// 縮約メソッドが 1 種類だけ異なる 3 メソッドの重複を避けるための
    /// private ヘルパ（イシュー #1719 レビュー指摘）。
    fn reduce_dims_with<F>(
        &self,
        dims: &[usize],
        keepdim: bool,
        reduce: F,
    ) -> Result<Var<'t>, AutodiffError>
    where
        F: FnOnce(Var<'t>, Option<usize>) -> Result<Var<'t>, AutodiffError>,
    {
        let shape = self.shape();
        let plan = crate::reduce_dims::plan_reduce_dims(&shape, dims)?;
        let (merged, axis) = crate::reduce_dims::merge_for_reduction(*self, &plan)?;
        let reduced = reduce(merged, axis)?;
        if keepdim {
            reduced.reshape(&plan.keepdim_shape)
        } else {
            Ok(reduced)
        }
    }

    /// 複数軸の縮約和（`torch.sum(dim=[...], keepdim=)` 相当。イシュー
    /// #1719）。単一軸なら `sum(Some(d))` と**bit 同一**（`crate::
    /// reduce_dims::merge_for_reduction` が単一軸・全軸を無加工で
    /// 既存単軸 API へ直接委譲するため）。それ以外は `permute` →
    /// `contiguous`（非 contiguous のときのみ）→ `reshape` で reduced
    /// 軸を 1 軸へ併合してから単軸 `sum` を呼ぶ（`crate::reduce_dims`
    /// モジュール doc「併合方式を選ぶ理由」参照。f64 アキュムレータの
    /// 1 パス蓄積を軸をまたいで維持するため、軸ごとの逐次縮約にはしない）。
    ///
    /// `dims` は空・範囲外・重複のいずれも
    /// [`AutodiffError`]（`crate::reduce_dims::plan_reduce_dims` が
    /// 検査）で拒否する。`keepdim: true` は縮約軸をサイズ 1 のまま
    /// 元の rank を保持する（PyTorch `keepdim=True` 相当）。
    pub fn sum_dims(&self, dims: &[usize], keepdim: bool) -> Result<Var<'t>, AutodiffError> {
        self.reduce_dims_with(dims, keepdim, |v, axis| v.sum(axis))
    }

    /// 複数軸の縮約最大値（forward 値は `torch.amax(dim=[...],
    /// keepdim=)` と同一。イシュー #1719）。`sum_dims` と同じ併合方式
    /// （`crate::reduce_dims`）を使うため、同値タイは縮約対象全要素を
    /// **1 回の `max_vjp` 呼び出し**で見る（併合順「kept 軸〈元の
    /// 順序〉→ reduced 軸〈昇順〉」で最初に現れる要素が先勝ちする。
    /// `grad.rs::max_vjp`（`extremum_first_match_vjp`）の「先勝ち
    /// 決定的」規約は、イシュー #1718 の確定により本メソッドも含めて
    /// **維持される**（`torch.amax` の均等分配とは勾配が異なる点は
    /// 意図的な設計判断。`docs/autodiff-amax-grad-distribution-
    /// decision.md` 参照）。`dims`／`keepdim` の契約は `sum_dims` と
    /// 同一。
    pub fn max_dims(&self, dims: &[usize], keepdim: bool) -> Result<Var<'t>, AutodiffError> {
        self.reduce_dims_with(dims, keepdim, |v, axis| v.max(axis))
    }

    /// 複数軸の縮約平均（`torch.mean(dim=[...], keepdim=)` 相当。
    /// イシュー #1719）。`sum_dims` と同じ併合（`crate::reduce_dims`）
    /// の結果へ `Var::mean` を適用するだけの合成（新規 Op は追加しない）。
    /// `dims`／`keepdim` の契約・`n == 0` 拒否は `mean`／`sum_dims` と同一。
    pub fn mean_dims(&self, dims: &[usize], keepdim: bool) -> Result<Var<'t>, AutodiffError> {
        self.reduce_dims_with(dims, keepdim, |v, axis| v.mean(axis))
    }

    /// 平均二乗誤差（`self` = 予測値、`target` = 正解値。全要素平均・
    /// PyTorch `nn.MSELoss` の既定 `reduction='mean'` 相当）。
    /// `mse_loss_with(target, Reduction::Mean)` への委譲（#190）。
    /// 既存呼び出し元（`nn::activation` 系テスト・`tests/backward.rs`
    /// 等）のシグネチャ・意味を変えないため本メソッドは維持する。
    pub fn mse_loss(&self, target: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.mse_loss_with(target, Reduction::Mean)
    }

    /// 平均二乗誤差（`self` = 予測値、`target` = 正解値）。`reduction`
    /// で mean/sum の縮約種別を選べる（#190。親イシュー #189「損失関数
    /// （MSE・CrossEntropy）の実装」）。`nn::loss::MseLoss`（`nn/loss.rs`）
    /// はこのメソッドを呼ぶだけの薄いラッパー（REQ-9）。
    ///
    /// **TASK-12.1d（#164）→ イシュー #1045 で更新**: 入力を層 1
    /// （`materialize_fallible`）で実体化したうえで、`self.tape.ops()`
    /// の `BackendOps::mse_loss`（CPU／CUDA／Metal の融合カーネル。
    /// `docs/kernel-fusion.md`）を試みる。`Err(BackendError::
    /// Unsupported(_))` のときのみ従来のホスト参照実装 `eval::mse_loss`
    /// へフォールバックし、それ以外のエラー（融合カーネルが実行時に
    /// 失敗した場合等）は伝播する（判定迂回経路を作らない。
    /// `.claude/rules/security.md` A08。`materialize_fallible` の
    /// `run_fused` フォールバック規律と同じ方針。`tape.rs:905` 参照）。
    /// `require_same_shape` が既に shape 一致を検査済みのため、
    /// バックエンド実装が返す `ShapeMismatch` は「バックエンド実装の
    /// 契約違反」を意味し、こちらも `Unsupported` 同様フォールバック
    /// せず伝播する（想定内の分岐で握り潰さない）。
    pub fn mse_loss_with(
        &self,
        target: &Var<'t>,
        reduction: Reduction,
    ) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(target)?;
        let lhs_shape = self.shape();
        let rhs_shape = target.shape();
        require_same_shape(&lhs_shape, &rhs_shape)?;
        let (pred_val, target_val) = {
            let nodes = self.tape.nodes.borrow();
            let pred_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let target_val = materialize_fallible(&nodes, self.tape.ops(), target.id)?.clone();
            (pred_val, target_val)
        };
        let value = match self
            .tape
            .ops()
            .mse_loss(&pred_val, &target_val, reduction.into())
        {
            Ok(v) => {
                // バックエンド実装の契約（`backend_ops.rs::BackendOps::
                // mse_loss` doc「戻り値は shape `[]`」）を検証する
                // （実装バグの黙認防止。`.claude/rules/security.md` A08）。
                if !v.shape().is_empty() {
                    return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                        fandhe_ai_tensor_core::ShapeError::ShapeMismatch {
                            lhs: v.shape().to_vec(),
                            rhs: Vec::new(),
                        },
                    )));
                }
                v
            }
            Err(BackendError::Unsupported(_)) => eval::mse_loss(&pred_val, &target_val, reduction),
            Err(other) => return Err(AutodiffError::Backend(other)),
        };
        let id = self.tape.push_eager(
            Op::MseLoss {
                pred: self.id,
                target: target.id,
                reduction,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// Huber 損失（`self` = 予測値、`target` = 正解値。PyTorch
    /// `nn.HuberLoss(delta)` 相当。イシュー #1739）。`|d| < delta`
    /// （`d = pred − target`）で二次（`0.5·d²`）、それ以外で線形
    /// （`delta·(|d| − 0.5·delta)`）となる区分的損失（`eval::
    /// huber_elem_loss` が意味論の正）。`delta` は有限かつ `> 0` を
    /// 要求する（PyTorch の `delta` 引数と同じ制約）。
    /// `huber_loss_impl(target, HuberKind::Huber, delta, reduction)`
    /// への委譲。
    pub fn huber_loss(
        &self,
        target: &Var<'t>,
        delta: f32,
        reduction: Reduction,
    ) -> Result<Var<'t>, AutodiffError> {
        self.huber_loss_impl(target, HuberKind::Huber, delta, reduction)
    }

    /// SmoothL1 損失（`self` = 予測値、`target` = 正解値。PyTorch
    /// `nn.SmoothL1Loss(beta)` 相当。イシュー #1739）。[`Self::huber_loss`]
    /// と同じ折れ点構造だが二次分岐が `0.5·d²/beta` に `beta` で
    /// スケールされる点のみ異なる（`beta = 1.0` のとき両者は一致する）。
    /// `beta = 0`（PyTorch では `nn.L1Loss` 相当に退化する特殊値）は
    /// 本メソッドの対象外——他の `delta`／`beta` 値と同じ「有限かつ
    /// `> 0`」検査により `AutodiffError::InvalidArgument` で拒否する。
    /// `huber_loss_impl(target, HuberKind::SmoothL1, beta, reduction)`
    /// への委譲。
    pub fn smooth_l1_loss(
        &self,
        target: &Var<'t>,
        beta: f32,
        reduction: Reduction,
    ) -> Result<Var<'t>, AutodiffError> {
        self.huber_loss_impl(target, HuberKind::SmoothL1, beta, reduction)
    }

    /// [`Self::huber_loss`]／[`Self::smooth_l1_loss`] の共通実装
    /// （イシュー #1739）。`mse_loss_with` と同じ演算メソッド規律
    /// （フォールバックは `Unsupported` のときのみ・それ以外のエラーは
    /// 伝播・バックエンド戻り値の shape 契約検証）に、`delta`
    /// （`SmoothL1` では `beta` と呼ぶが同じ引数）の検査を追加する
    /// 検査順序: ①`check_same_tape` → ②shape 一致
    /// （`require_same_shape`）→ ③`delta` が有限かつ `> 0`（違反は
    /// `AutodiffError::InvalidArgument`。`rms_norm` の `eps` 検査と同型）
    /// → ④実体化（層 1）→ ⑤`BackendOps::huber_loss` 試行 →
    /// ⑥バックエンド契約検証（戻り値 shape `[]`）→ ⑦ノード記録。
    fn huber_loss_impl(
        &self,
        target: &Var<'t>,
        kind: HuberKind,
        delta: f32,
        reduction: Reduction,
    ) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(target)?;
        let lhs_shape = self.shape();
        let rhs_shape = target.shape();
        require_same_shape(&lhs_shape, &rhs_shape)?;
        if !delta.is_finite() || delta <= 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Var::huber_loss_impl: delta must be finite and positive, got {delta}"
            )));
        }
        let (pred_val, target_val) = {
            let nodes = self.tape.nodes.borrow();
            let pred_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let target_val = materialize_fallible(&nodes, self.tape.ops(), target.id)?.clone();
            (pred_val, target_val)
        };
        let value =
            match self
                .tape
                .ops()
                .huber_loss(&pred_val, &target_val, kind, delta, reduction.into())
            {
                Ok(v) => {
                    // バックエンド実装の契約（`backend_ops.rs::BackendOps::
                    // huber_loss` doc「戻り値は shape `[]`」）を検証する
                    // （`mse_loss_with` と同じ理由）。
                    if !v.shape().is_empty() {
                        return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                            fandhe_ai_tensor_core::ShapeError::ShapeMismatch {
                                lhs: v.shape().to_vec(),
                                rhs: Vec::new(),
                            },
                        )));
                    }
                    v
                }
                Err(BackendError::Unsupported(_)) => {
                    eval::huber_loss(&pred_val, &target_val, kind, delta, reduction)
                }
                Err(other) => return Err(AutodiffError::Backend(other)),
            };
        let id = self.tape.push_eager(
            Op::HuberLoss {
                pred: self.id,
                target: target.id,
                kind,
                delta,
                reduction,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 二値交差エントロピー損失（`self` = 予測確率 `[0, 1]`、`target` =
    /// 正解ラベル。全要素平均・PyTorch `nn.BCELoss` 相当）。
    /// `bce_loss_impl(target, BceKind::Probabilities, reduction)` への
    /// 委譲（#1737。親イシュー #1609「損失関数の拡張」）。`nn::loss::
    /// BceLoss`（`nn/loss.rs`）はこのメソッドを呼ぶだけの薄いラッパー
    /// （REQ-9）。
    pub fn bce_loss(
        &self,
        target: &Var<'t>,
        reduction: Reduction,
    ) -> Result<Var<'t>, AutodiffError> {
        self.bce_loss_impl(target, BceKind::Probabilities, reduction)
    }

    /// 二値交差エントロピー損失（logits 入力版。`self` = 未正規化の
    /// logits、`target` = 正解ラベル。PyTorch `nn.BCEWithLogitsLoss`
    /// 相当）。sigmoid をカーネル内に内包した数値安定な合成式で計算
    /// する（`docs/compat-api-scope.md` §1.2 参照）。
    /// `bce_loss_impl(target, BceKind::Logits, reduction)` への委譲
    /// （#1737）。`self`（logits）は `[0, 1]` 範囲制約を受けない
    /// （[`Self::bce_loss`] と異なり範囲検査を行わない）。`nn::loss::
    /// BceWithLogitsLoss`（`nn/loss.rs`）はこのメソッドを呼ぶだけの
    /// 薄いラッパー（REQ-9）。
    pub fn bce_with_logits_loss(
        &self,
        target: &Var<'t>,
        reduction: Reduction,
    ) -> Result<Var<'t>, AutodiffError> {
        self.bce_loss_impl(target, BceKind::Logits, reduction)
    }

    /// [`Self::bce_loss`]／[`Self::bce_with_logits_loss`] 共通実装
    /// （#1737）。`mse_loss_with` と同じ演算メソッド規律（層 1
    /// 実体化 → `self.tape.ops()` の融合カーネルを試み `Unsupported`
    /// のときのみホスト参照実装 `eval::bce_loss` へフォールバック。
    /// それ以外のエラーは伝播し判定迂回経路を作らない。
    /// `.claude/rules/security.md` A08）に加え、[`BceKind::
    /// Probabilities`] のときのみ `input`／`target` 双方の値が
    /// `[0, 1]` 範囲内（NaN は範囲外として拒否）であることを
    /// 実体化直後・バックエンド呼び出し前にホスト側で検査する
    /// （`Self::cross_entropy_loss` の targets 範囲検査と同じ配置・
    /// 同じ `AutodiffError::InvalidArgument` 文言様式。`.claude/rules/
    /// security.md` A03）。
    fn bce_loss_impl(
        &self,
        target: &Var<'t>,
        kind: BceKind,
        reduction: Reduction,
    ) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(target)?;
        let lhs_shape = self.shape();
        let rhs_shape = target.shape();
        require_same_shape(&lhs_shape, &rhs_shape)?;
        let (input_val, target_val) = {
            let nodes = self.tape.nodes.borrow();
            let input_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let target_val = materialize_fallible(&nodes, self.tape.ops(), target.id)?.clone();
            (input_val, target_val)
        };
        if matches!(kind, BceKind::Probabilities) {
            for &v in eval::dense_vec(&input_val).iter() {
                if !(0.0..=1.0).contains(&v) {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "bce_loss: input 値 {v} が範囲 [0, 1] を外れている"
                    )));
                }
            }
            for &v in eval::dense_vec(&target_val).iter() {
                if !(0.0..=1.0).contains(&v) {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "bce_loss: target 値 {v} が範囲 [0, 1] を外れている"
                    )));
                }
            }
        }
        let value = match self
            .tape
            .ops()
            .bce_loss(&input_val, &target_val, kind, reduction.into())
        {
            Ok(v) => {
                // バックエンド実装の契約（`backend_ops.rs::BackendOps::
                // bce_loss` doc「戻り値は shape `[]`」）を検証する
                // （実装バグの黙認防止。`.claude/rules/security.md` A08）。
                if !v.shape().is_empty() {
                    return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                        fandhe_ai_tensor_core::ShapeError::ShapeMismatch {
                            lhs: v.shape().to_vec(),
                            rhs: Vec::new(),
                        },
                    )));
                }
                v
            }
            Err(BackendError::Unsupported(_)) => {
                eval::bce_loss(&input_val, &target_val, kind, reduction)
            }
            Err(other) => return Err(AutodiffError::Backend(other)),
        };
        let id = self.tape.push_eager(
            Op::BceLoss {
                input: self.id,
                target: target.id,
                kind,
                reduction,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }
    /// 行方向 RMSNorm（`x · rsqrt(mean(x²) + eps) · w`。`w` が `None`
    /// の場合は乗算をスキップ。イシュー #1596）。正規化軸は常に
    /// 最終軸（[`row_norm_layout`] が `(rows, hidden)` を導出する）。
    ///
    /// 検査順序（`mse_loss_with` と同じ演算メソッド規律。`weight` を
    /// 渡す場合のみクロステープ検査を追加）: ①`weight` があれば
    /// `check_same_tape` → ②`eps` が有限かつ非負であることを検査
    /// （違反は `AutodiffError::InvalidArgument`。`docs/norm-ops-design.md`
    /// 「`eps` 検査」節） → ③`row_norm_layout` で `self` の shape から
    /// `hidden` を導出し、`weight` の shape が `[hidden]` と厳密一致する
    /// ことを検査（`ShapeError::ShapeMismatch`） → ④実体化（層 1）
    /// → ⑤`self.tape.ops().rmsnorm` を試み `Unsupported` のときのみ
    /// `eval::rmsnorm_rows` へフォールバック（それ以外のエラーは伝播。
    /// 判定迂回経路を作らない。`.claude/rules/security.md` A08）
    /// → ⑥バックエンド契約検証（戻り値 shape が入力と恒等）
    /// → ⑦ノード記録。
    pub fn rms_norm(&self, weight: Option<&Var<'t>>, eps: f32) -> Result<Var<'t>, AutodiffError> {
        if let Some(w) = weight {
            self.check_same_tape(w)?;
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Var::rms_norm: eps must be finite and non-negative, got {eps}"
            )));
        }
        let x_shape = self.shape();
        let (_, hidden) = row_norm_layout(&x_shape)?;
        if let Some(w) = weight {
            require_same_shape(&w.shape(), &[hidden])?;
        }
        let (x_val, w_val) = {
            let nodes = self.tape.nodes.borrow();
            let x_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let w_val = match weight {
                Some(w) => Some(materialize_fallible(&nodes, self.tape.ops(), w.id)?.clone()),
                None => None,
            };
            (x_val, w_val)
        };
        let value = match self.tape.ops().rmsnorm(&x_val, w_val.as_ref(), eps) {
            Ok(v) => {
                // バックエンド実装の契約（`backend_ops.rs::BackendOps::
                // rmsnorm` doc「戻り値の shape は入力 `x` と恒等」）を
                // 検証する（実装バグの黙認防止。`.claude/rules/
                // security.md` A08）。
                if v.shape() != x_val.shape() {
                    return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                        ShapeError::ShapeMismatch {
                            lhs: v.shape().to_vec(),
                            rhs: x_val.shape().to_vec(),
                        },
                    )));
                }
                v
            }
            Err(BackendError::Unsupported(_)) => {
                let (rows, hidden) = row_norm_layout(&x_shape)?;
                // `as_slice()` は非 contiguous な入力（`weight` に転置
                // view 等が渡された場合）で `None` を返しうるため、
                // `dense_vec`（`contiguous()` 経由の稠密化。`eval.rs`）
                // を使う——`as_slice()` を直接使うと非 contiguous な
                // `weight` を誤って「重みなし」（乗算スキップ）として
                // 扱ってしまう（判定迂回経路。`.claude/rules/
                // security.md` A08）。
                let w_dense = w_val.as_ref().map(eval::dense_vec);
                eval::rmsnorm_rows(&x_val, w_dense.as_deref(), eps, rows, hidden)
            }
            Err(other) => return Err(AutodiffError::Backend(other)),
        };
        let id = self.tape.push_eager(
            Op::RmsNorm {
                input: self.id,
                weight: weight.map(|w| w.id),
                eps,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 行方向 LayerNorm（`(x − mean(x)) · rsqrt(var(x) + eps) · w + b`。
    /// `w`／`b` はそれぞれ `None` の場合は対応する演算をスキップ。
    /// 分散は biased（÷N）。イシュー #1596）。[`Self::rms_norm`] と
    /// 同じ最終軸限定契約・検査順序（`bias` も `weight` と同じ
    /// クロステープ検査・shape `[hidden]` 検査を受ける）。
    pub fn layer_norm(
        &self,
        weight: Option<&Var<'t>>,
        bias: Option<&Var<'t>>,
        eps: f32,
    ) -> Result<Var<'t>, AutodiffError> {
        if let Some(w) = weight {
            self.check_same_tape(w)?;
        }
        if let Some(b) = bias {
            self.check_same_tape(b)?;
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Var::layer_norm: eps must be finite and non-negative, got {eps}"
            )));
        }
        let x_shape = self.shape();
        let (_, hidden) = row_norm_layout(&x_shape)?;
        if let Some(w) = weight {
            require_same_shape(&w.shape(), &[hidden])?;
        }
        if let Some(b) = bias {
            require_same_shape(&b.shape(), &[hidden])?;
        }
        let (x_val, w_val, b_val) = {
            let nodes = self.tape.nodes.borrow();
            let x_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let w_val = match weight {
                Some(w) => Some(materialize_fallible(&nodes, self.tape.ops(), w.id)?.clone()),
                None => None,
            };
            let b_val = match bias {
                Some(b) => Some(materialize_fallible(&nodes, self.tape.ops(), b.id)?.clone()),
                None => None,
            };
            (x_val, w_val, b_val)
        };
        let value = match self
            .tape
            .ops()
            .layer_norm(&x_val, w_val.as_ref(), b_val.as_ref(), eps)
        {
            Ok(v) => {
                if v.shape() != x_val.shape() {
                    return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                        ShapeError::ShapeMismatch {
                            lhs: v.shape().to_vec(),
                            rhs: x_val.shape().to_vec(),
                        },
                    )));
                }
                v
            }
            Err(BackendError::Unsupported(_)) => {
                let (rows, hidden) = row_norm_layout(&x_shape)?;
                // `rms_norm` と同じ理由（直上コメント）で `dense_vec`
                // を使う（`as_slice()` は非 contiguous を無音で
                // 「なし」化してしまう）。
                let w_dense = w_val.as_ref().map(eval::dense_vec);
                let b_dense = b_val.as_ref().map(eval::dense_vec);
                eval::layer_norm_rows(
                    &x_val,
                    w_dense.as_deref(),
                    b_dense.as_deref(),
                    eps,
                    rows,
                    hidden,
                )
            }
            Err(other) => return Err(AutodiffError::Backend(other)),
        };
        let id = self.tape.push_eager(
            Op::LayerNorm {
                input: self.id,
                weight: weight.map(|w| w.id),
                bias: bias.map(|b| b.id),
                eps,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// BatchNorm1d／2d の train モード（バッチ統計。チャネル軸は常に
    /// dim 1。イシュー #1732・親 #1608・`docs/batch-norm-ops-design.md`）。
    /// [`Self::batch_norm_with_batch_stats`] の `.0`（output のみ）を
    /// 返す薄いラッパー。
    pub fn batch_norm(
        &self,
        weight: Option<&Var<'t>>,
        bias: Option<&Var<'t>>,
        eps: f32,
    ) -> Result<Var<'t>, AutodiffError> {
        Ok(self.batch_norm_with_batch_stats(weight, bias, eps)?.0)
    }

    /// [`Self::batch_norm`] の本体。バッチ統計（`batch_mean`／
    /// `batch_var`。biased ÷M）も `(output, batch_mean, batch_var)`
    /// タプルで返す——呼び出し元（`nn::BatchNorm1d`／`BatchNorm2d`）が
    /// running stats を更新するために必要（`Var::layer_norm` と異なり
    /// BatchNorm はバッチ統計を呼び出し元へ公開する必要がある）。
    ///
    /// 検査順序（[`Self::layer_norm`] と同じ演算メソッド規律）: ①
    /// `weight`／`bias` があれば `check_same_tape` → ②`eps` が有限かつ
    /// 非負であることを検査 → ③[`fandhe_ai_tensor_core::
    /// batch_norm_layout`] で `(n, c, spatial)` を導出 → ④`M = n*spatial`
    /// が 1 以下の場合 `AutodiffError::InvalidArgument`（unbiased 分散
    /// の `M-1` 除算で 0 除算になるため。PyTorch `torch.nn.functional.
    /// batch_norm` の `_verify_batch_size` と同じ拒否）→ ⑤`weight`／
    /// `bias` の shape が `[c]` と厳密一致することを検査 → ⑥実体化 →
    /// ⑦`batch_norm_train_with_fallback`（`grad.rs`。バックエンド →
    /// `Unsupported` のときのみホスト参照実装）→ ⑧ノード記録
    /// （`fixed_stats: None` が train モードを表す）。
    pub fn batch_norm_with_batch_stats(
        &self,
        weight: Option<&Var<'t>>,
        bias: Option<&Var<'t>>,
        eps: f32,
    ) -> Result<(Var<'t>, Tensor<f32>, Tensor<f32>), AutodiffError> {
        if let Some(w) = weight {
            self.check_same_tape(w)?;
        }
        if let Some(b) = bias {
            self.check_same_tape(b)?;
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Var::batch_norm: eps must be finite and non-negative, got {eps}"
            )));
        }
        let x_shape = self.shape();
        let (n, c, spatial) = batch_norm_layout(&x_shape)?;
        let m = n
            .checked_mul(spatial)
            .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
        if m <= 1 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Var::batch_norm: train モードはチャネルごとの要素数 M=n*spatial が \
                 1 以下を許容しない（unbiased 分散の M-1 除算に必要。got n={n}, \
                 spatial={spatial}, M={m}）"
            )));
        }
        if let Some(w) = weight {
            require_same_shape(&w.shape(), &[c])?;
        }
        if let Some(b) = bias {
            require_same_shape(&b.shape(), &[c])?;
        }
        let (x_val, w_val, b_val) = {
            let nodes = self.tape.nodes.borrow();
            let x_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let w_val = match weight {
                Some(w) => Some(materialize_fallible(&nodes, self.tape.ops(), w.id)?.clone()),
                None => None,
            };
            let b_val = match bias {
                Some(b) => Some(materialize_fallible(&nodes, self.tape.ops(), b.id)?.clone()),
                None => None,
            };
            (x_val, w_val, b_val)
        };
        let BatchNormTrainOutput {
            output,
            batch_mean,
            batch_var,
        } = batch_norm_train_with_fallback(
            self.tape.ops(),
            &x_val,
            w_val.as_ref(),
            b_val.as_ref(),
            eps,
            n,
            c,
            spatial,
        )?;
        let id = self.tape.push_eager(
            Op::BatchNorm {
                input: self.id,
                weight: weight.map(|w| w.id),
                bias: bias.map(|b| b.id),
                eps,
                fixed_stats: None,
            },
            output,
        );
        Ok((Var::from_raw(self.tape, id), batch_mean, batch_var))
    }

    /// BatchNorm1d／2d の eval モード（固定統計。`running_mean`／
    /// `running_var` は呼び出し元〈`nn::BatchNorm1d`／`BatchNorm2d`〉が
    /// 保持する running stats。イシュー #1732・親 #1608）。
    /// [`Self::batch_norm_with_batch_stats`] と同じ検査順序だが、`M`
    /// を導出・検査しない（バッチから統計を計算しないため `M<=1`
    /// 制約が不要）代わりに `running_mean`／`running_var` の shape
    /// `[c]` を検査する。
    pub fn batch_norm_infer(
        &self,
        weight: Option<&Var<'t>>,
        bias: Option<&Var<'t>>,
        running_mean: &Tensor<f32>,
        running_var: &Tensor<f32>,
        eps: f32,
    ) -> Result<Var<'t>, AutodiffError> {
        if let Some(w) = weight {
            self.check_same_tape(w)?;
        }
        if let Some(b) = bias {
            self.check_same_tape(b)?;
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Var::batch_norm_infer: eps must be finite and non-negative, got {eps}"
            )));
        }
        let x_shape = self.shape();
        let (n, c, spatial) = batch_norm_layout(&x_shape)?;
        require_same_shape(running_mean.shape(), &[c])?;
        require_same_shape(running_var.shape(), &[c])?;
        if let Some(w) = weight {
            require_same_shape(&w.shape(), &[c])?;
        }
        if let Some(b) = bias {
            require_same_shape(&b.shape(), &[c])?;
        }
        let (x_val, w_val, b_val) = {
            let nodes = self.tape.nodes.borrow();
            let x_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let w_val = match weight {
                Some(w) => Some(materialize_fallible(&nodes, self.tape.ops(), w.id)?.clone()),
                None => None,
            };
            let b_val = match bias {
                Some(b) => Some(materialize_fallible(&nodes, self.tape.ops(), b.id)?.clone()),
                None => None,
            };
            (x_val, w_val, b_val)
        };
        let value = batch_norm_infer_with_fallback(
            self.tape.ops(),
            &x_val,
            running_mean,
            running_var,
            w_val.as_ref(),
            b_val.as_ref(),
            eps,
            n,
            c,
            spatial,
        )?;
        let id = self.tape.push_eager(
            Op::BatchNorm {
                input: self.id,
                weight: weight.map(|w| w.id),
                bias: bias.map(|b| b.id),
                eps,
                fixed_stats: Some((running_mean.clone(), running_var.clone())),
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 行方向 softmax（`exp(x − max(x)) / Σexp(x − max(x))`。イシュー
    /// #1594）。`dim` は [`reduce_out_shape`] で範囲検査する（既存の
    /// `sum`/`max` と同じ軸検査ヘルパーを再利用。softmax は shape 不変
    /// のため戻り値 shape 自体には使わないが、`AxisOutOfRange` の検査
    /// 目的のみで呼ぶ）。
    ///
    /// `self.tape.ops().softmax`（`BackendOps::softmax`。CPU／CUDA／
    /// Metal の行カーネル。最終軸専用）を試み、`Err(BackendError::
    /// Unsupported(_))` のときのみホスト参照実装 `eval::softmax_along`
    /// （最終軸に限らず任意軸へ対応）へフォールバックする（それ以外の
    /// エラーは伝播する。判定迂回経路を作らない。`.claude/rules/
    /// security.md` A08。`mse_loss_with` と同じ規律）。
    pub fn softmax(&self, dim: usize) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        reduce_out_shape(&shape, Some(dim))?;
        let x_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = match self.tape.ops().softmax(&x_val, dim) {
            Ok(v) => {
                // バックエンド実装の契約（`backend_ops.rs::BackendOps::
                // softmax` doc「戻り値 shape は入力と恒等」）を検証する
                // （実装バグの黙認防止。`.claude/rules/security.md` A08）。
                if v.shape() != x_val.shape() {
                    return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                        ShapeError::ShapeMismatch {
                            lhs: v.shape().to_vec(),
                            rhs: x_val.shape().to_vec(),
                        },
                    )));
                }
                v
            }
            Err(BackendError::Unsupported(_)) => eval::softmax_along(&x_val, dim),
            Err(other) => return Err(AutodiffError::Backend(other)),
        };
        let id = self.tape.push_eager(
            Op::Softmax {
                input: self.id,
                dim,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 行方向 log_softmax（`x − m − ln(Σexp(x − m))`。イシュー #1594）。
    /// [`Self::softmax`] と同じ `dim` 検査・フォールバック規律
    /// （`BackendOps::log_softmax` → `Unsupported` のときのみ `eval::
    /// log_softmax_along` へフォールバック）。
    pub fn log_softmax(&self, dim: usize) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        reduce_out_shape(&shape, Some(dim))?;
        let x_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = match self.tape.ops().log_softmax(&x_val, dim) {
            Ok(v) => {
                if v.shape() != x_val.shape() {
                    return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                        ShapeError::ShapeMismatch {
                            lhs: v.shape().to_vec(),
                            rhs: x_val.shape().to_vec(),
                        },
                    )));
                }
                v
            }
            Err(BackendError::Unsupported(_)) => eval::log_softmax_along(&x_val, dim),
            Err(other) => return Err(AutodiffError::Backend(other)),
        };
        let id = self.tape.push_eager(
            Op::LogSoftmax {
                input: self.id,
                dim,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// `dim` 軸に沿った累積和（`torch.cumsum` 相当。イシュー #1731）。
    /// [`Self::softmax`] と同じ `dim` 検査・フォールバック規律
    /// （`BackendOps::cumsum` → `Unsupported` のときのみ `eval::
    /// cumsum_along` へフォールバック）。出力 shape は `self` と恒等
    /// （累積演算は shape を変えない）。
    pub fn cumsum(&self, dim: usize) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        reduce_out_shape(&shape, Some(dim))?;
        let x_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = match self.tape.ops().cumsum(&x_val, dim) {
            Ok(v) => {
                // バックエンド実装の契約（`backend_ops.rs::BackendOps::
                // cumsum` doc「戻り値 shape は入力と恒等」）を検証する
                // （実装バグの黙認防止。`.claude/rules/security.md` A08）。
                if v.shape() != x_val.shape() {
                    return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                        ShapeError::ShapeMismatch {
                            lhs: v.shape().to_vec(),
                            rhs: x_val.shape().to_vec(),
                        },
                    )));
                }
                v
            }
            Err(BackendError::Unsupported(_)) => eval::cumsum_along(&x_val, dim),
            Err(other) => return Err(AutodiffError::Backend(other)),
        };
        let id = self.tape.push_eager(
            Op::Cumsum {
                input: self.id,
                dim,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// `dim` 軸に沿った累積積（`torch.cumprod` 相当。イシュー #1731）。
    /// [`Self::cumsum`] と同じ `dim` 検査・フォールバック規律
    /// （`BackendOps::cumprod` → `Unsupported` のときのみ `eval::
    /// cumprod_along` へフォールバック）。
    pub fn cumprod(&self, dim: usize) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        reduce_out_shape(&shape, Some(dim))?;
        let x_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = match self.tape.ops().cumprod(&x_val, dim) {
            Ok(v) => {
                if v.shape() != x_val.shape() {
                    return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                        ShapeError::ShapeMismatch {
                            lhs: v.shape().to_vec(),
                            rhs: x_val.shape().to_vec(),
                        },
                    )));
                }
                v
            }
            Err(BackendError::Unsupported(_)) => eval::cumprod_along(&x_val, dim),
            Err(other) => return Err(AutodiffError::Backend(other)),
        };
        let id = self.tape.push_eager(
            Op::Cumprod {
                input: self.id,
                dim,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// CrossEntropy 損失（log-sum-exp 安定化・クラス次元指定。#191・
    /// 親イシュー #189）。`self` = logits（追跡対象）、`targets` = 正解
    /// クラス添字（非追跡・`Tensor<i32>`。勾配は定義されないため
    /// `Var` にしない。`tape::Op::CrossEntropyLoss` doc 参照）。
    ///
    /// 検査順序（本メソッド冒頭 doc の演算メソッド規律に、targets
    /// 範囲検査〈REQ-8 趣旨の境界外アクセス防止・A03 対策〉を追加）:
    /// ①`class_dim` 範囲・targets shape 一致（`reduce_out_shape` を
    /// 再利用。`class_dim >= rank` は `ShapeError::AxisOutOfRange`）
    /// → ②targets 全添字が `0 <= t < C`（違反は
    /// `AutodiffError::InvalidArgument`）→ ③実体化（層 1）→ ④forward
    /// 値計算（`eval::cross_entropy_loss`。`mse_loss_with` と同じく
    /// `BackendOps` に対応メソッドがないため融合対象外）→ ⑤ノード記録。
    pub fn cross_entropy_loss(
        &self,
        targets: &Tensor<i32>,
        class_dim: usize,
        reduction: Reduction,
    ) -> Result<Var<'t>, AutodiffError> {
        let logits_shape = self.shape();
        let expected_targets_shape = reduce_out_shape(&logits_shape, Some(class_dim))?;
        require_same_shape(targets.shape(), &expected_targets_shape)?;

        // `reduce_out_shape` が成功した時点で `class_dim < logits_shape.len()`
        // が保証されるため、この添字アクセスは安全（`.claude/rules/
        // coding-rust.md` REQ-8「境界検査を省略しない」の趣旨に沿い、
        // 検査済みの添字のみでアクセスする）。
        let num_classes = logits_shape[class_dim];
        for t in eval::dense_vec_i32(targets) {
            if t < 0 || (t as usize) >= num_classes {
                return Err(AutodiffError::InvalidArgument(format!(
                    "cross_entropy_loss: target 添字 {t} が範囲 [0, {num_classes}) を外れている"
                )));
            }
        }

        let logits_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = eval::cross_entropy_loss(&logits_val, targets, class_dim, reduction);
        let id = self.tape.push_eager(
            Op::CrossEntropyLoss {
                logits: self.id,
                targets: targets.clone(),
                class_dim,
                reduction,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 負対数尤度損失（`self` = log 確率、`targets` = 正解クラス添字。
    /// PyTorch `nn.NLLLoss` 相当。`log_softmax(x).nll_loss(t)` が
    /// `cross_entropy_loss(x, t)` と一致する。#1738。親イシュー #1609
    /// 「損失関数の拡張」）。`targets` は `Self::cross_entropy_loss` と
    /// 同じく非追跡（`Tensor<i32>`。勾配は定義されないため `Var` に
    /// しない。`tape::Op::NllLoss` doc 参照）。
    ///
    /// 検査順序（`Self::cross_entropy_loss` と同じ演算メソッド規律）:
    /// ①`class_dim` 範囲・targets shape 一致（`reduce_out_shape` を
    /// 再利用）→ ②targets 全添字が `0 <= t < C`（違反は
    /// `AutodiffError::InvalidArgument`。REQ-8 趣旨の境界外アクセス
    /// 防止・A03 対策）→ ③実体化（層 1）→ ④`self.tape.ops()` の融合
    /// カーネル（`BackendOps::nll_loss`）を試み `Unsupported` のときのみ
    /// ホスト参照実装（`eval::nll_loss`）へフォールバック（それ以外の
    /// エラーは伝播し判定迂回経路を作らない。`.claude/rules/security.md`
    /// A08）→ ⑤戻り値 shape `[]` の契約検証 → ⑥ノード記録。`nn::loss::
    /// NllLoss`（`nn/loss.rs`）はこのメソッドを呼ぶだけの薄いラッパー
    /// （REQ-9）。
    pub fn nll_loss(
        &self,
        targets: &Tensor<i32>,
        class_dim: usize,
        reduction: Reduction,
    ) -> Result<Var<'t>, AutodiffError> {
        let input_shape = self.shape();
        let expected_targets_shape = reduce_out_shape(&input_shape, Some(class_dim))?;
        require_same_shape(targets.shape(), &expected_targets_shape)?;

        // `reduce_out_shape` が成功した時点で `class_dim < input_shape.len()`
        // が保証されるため、この添字アクセスは安全（`Self::
        // cross_entropy_loss` と同型の境界検査規律）。
        let num_classes = input_shape[class_dim];
        for t in eval::dense_vec_i32(targets) {
            if t < 0 || (t as usize) >= num_classes {
                return Err(AutodiffError::InvalidArgument(format!(
                    "nll_loss: target 添字 {t} が範囲 [0, {num_classes}) を外れている"
                )));
            }
        }

        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = match self
            .tape
            .ops()
            .nll_loss(&input_val, targets, class_dim, reduction.into())
        {
            Ok(v) => {
                // バックエンド実装の契約（`backend_ops.rs::BackendOps::
                // nll_loss` doc「戻り値は shape `[]`」）を検証する
                // （実装バグの黙認防止。`.claude/rules/security.md` A08）。
                if !v.shape().is_empty() {
                    return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                        fandhe_ai_tensor_core::ShapeError::ShapeMismatch {
                            lhs: v.shape().to_vec(),
                            rhs: Vec::new(),
                        },
                    )));
                }
                v
            }
            Err(BackendError::Unsupported(_)) => {
                // `eval::nll_loss` は `ShapeError::ElementCountOverflow`
                // を返しうる（PR #1850 codex-review P1 是正: `outer`／
                // `inner` の部分積 overflow を `checked_mul` で拒否する
                // ようになった）。判定迂回経路を作らず伝播する
                // （`.claude/rules/security.md` A08）。
                eval::nll_loss(&input_val, targets, class_dim, reduction)
                    .map_err(AutodiffError::Shape)?
            }
            Err(other) => return Err(AutodiffError::Backend(other)),
        };
        let id = self.tape.push_eager(
            Op::NllLoss {
                input: self.id,
                targets: targets.clone(),
                class_dim,
                reduction,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// Kullback-Leibler ダイバージェンス損失（`self` = log 確率、
    /// `target` = 確率〈PyTorch 既定 `log_target=False`〉。PyTorch
    /// `nn.KLDivLoss` 相当。#1738）。`kl_div_loss_impl(target,
    /// KlDivTarget::Probabilities, reduction)` への委譲。`nn::loss::
    /// KlDivLoss`（`nn/loss.rs`）はこのメソッドを呼ぶだけの薄いラッパー
    /// （REQ-9）。
    pub fn kl_div_loss(
        &self,
        target: &Var<'t>,
        reduction: Reduction,
    ) -> Result<Var<'t>, AutodiffError> {
        self.kl_div_loss_impl(target, KlDivTarget::Probabilities, reduction)
    }

    /// Kullback-Leibler ダイバージェンス損失（対数確率 target 版。
    /// `self` = log 確率、`target` = 対数確率。PyTorch `nn.KLDivLoss(
    /// log_target=True)` 相当）。`kl_div_loss_impl(target,
    /// KlDivTarget::LogProbabilities, reduction)` への委譲（#1738）。
    /// `nn::loss::KlDivLoss::new_with_log_target`（`nn/loss.rs`）はこの
    /// メソッドを呼ぶだけの薄いラッパー（REQ-9）。
    pub fn kl_div_loss_with_log_target(
        &self,
        target: &Var<'t>,
        reduction: Reduction,
    ) -> Result<Var<'t>, AutodiffError> {
        self.kl_div_loss_impl(target, KlDivTarget::LogProbabilities, reduction)
    }

    /// [`Self::kl_div_loss`]／[`Self::kl_div_loss_with_log_target`] 共通
    /// 実装（#1738）。`Self::bce_loss_impl`（PR #1848・イシュー #1737）
    /// と同じ演算メソッド規律（層 1 実体化 → `self.tape.ops()` の融合
    /// カーネルを試み `Unsupported` のときのみホスト参照実装
    /// `eval::kl_div_loss` へフォールバック。それ以外のエラーは伝播し
    /// 判定迂回経路を作らない。`.claude/rules/security.md` A08）。値域
    /// 検査は行わない（shape 一致・クロステープのみ。`docs/
    /// compat-api-scope.md` §1.2 参照）。
    fn kl_div_loss_impl(
        &self,
        target: &Var<'t>,
        kind: KlDivTarget,
        reduction: Reduction,
    ) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(target)?;
        let lhs_shape = self.shape();
        let rhs_shape = target.shape();
        require_same_shape(&lhs_shape, &rhs_shape)?;
        let (input_val, target_val) = {
            let nodes = self.tape.nodes.borrow();
            let input_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let target_val = materialize_fallible(&nodes, self.tape.ops(), target.id)?.clone();
            (input_val, target_val)
        };
        let value =
            match self
                .tape
                .ops()
                .kl_div_loss(&input_val, &target_val, kind, reduction.into())
            {
                Ok(v) => {
                    // バックエンド実装の契約（`backend_ops.rs::BackendOps::
                    // kl_div_loss` doc「戻り値は shape `[]`」）を検証する
                    // （実装バグの黙認防止。`.claude/rules/security.md` A08）。
                    if !v.shape().is_empty() {
                        return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                            fandhe_ai_tensor_core::ShapeError::ShapeMismatch {
                                lhs: v.shape().to_vec(),
                                rhs: Vec::new(),
                            },
                        )));
                    }
                    v
                }
                Err(BackendError::Unsupported(_)) => {
                    eval::kl_div_loss(&input_val, &target_val, kind, reduction)
                }
                Err(other) => return Err(AutodiffError::Backend(other)),
            };
        let id = self.tape.push_eager(
            Op::KlDivLoss {
                input: self.id,
                target: target.id,
                kind,
                reduction,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// ReLU。shape を変えない要素ごとの演算のため構造的に失敗しえない
    /// （`docs/public-api-design.md` §3.2）。elementwise 5 演算の 1 つ
    /// （`add`/`mul` と同じ遅延契約。TASK-12.1d・#164）。
    ///
    /// **連鎖長上限（#404・設計書 §3.5.4）**: 非 fallible な単項演算
    /// のため、上限到達時は層 2（`materialize_non_fallible`）でその場
    /// 実体化する（`add`/`mul` の層 1 とは異なり、必ず値が入り
    /// panic／`Err` を返さない）。
    pub fn relu(&self) -> Var<'t> {
        let shape = self.shape();
        let (id, at_limit) = self.tape.push_lazy(Op::Relu(self.id), shape);
        if at_limit {
            let nodes = self.tape.nodes.borrow();
            materialize_non_fallible(&nodes, self.tape.ops(), id);
        }
        Var::from_raw(self.tape, id)
    }

    /// 要素ごとの指数関数。elementwise 5 演算の 1 つ（`relu` と同じ
    /// 遅延契約・連鎖長上限での自己実体化契約。#404）。
    pub fn exp(&self) -> Var<'t> {
        let shape = self.shape();
        let (id, at_limit) = self.tape.push_lazy(Op::Exp(self.id), shape);
        if at_limit {
            let nodes = self.tape.nodes.borrow();
            materialize_non_fallible(&nodes, self.tape.ops(), id);
        }
        Var::from_raw(self.tape, id)
    }

    /// 要素ごとの双曲線正接。elementwise 5 演算の 1 つ（`relu` と同じ
    /// 遅延契約・連鎖長上限での自己実体化契約。#404）。
    pub fn tanh(&self) -> Var<'t> {
        let shape = self.shape();
        let (id, at_limit) = self.tape.push_lazy(Op::Tanh(self.id), shape);
        if at_limit {
            let nodes = self.tape.nodes.borrow();
            materialize_non_fallible(&nodes, self.tape.ops(), id);
        }
        Var::from_raw(self.tape, id)
    }

    /// 要素ごとのシグモイド（`1 / (1 + exp(-x))`）。`relu`/`exp`/`tanh`
    /// と同じく shape 不変の単項演算のため構造的に失敗しえない
    /// （TASK-9.1b・#92。`nn::activation::Sigmoid` の薄いラッパーが
    /// このメソッドを呼ぶ）。forward は `eval::sigmoid`（数値安定形）
    /// を使う。
    ///
    /// **TASK-12.1d（#164）**: `BackendOps` に対応メソッドがないため
    /// 融合対象外とし、常に実体化済みで返る（`push_eager`）。入力読み
    /// 出しは非 fallible な本メソッド自身の契約に合わせ `value()`
    /// （層 2）経由とする（設計書 §3.5.1）。
    pub fn sigmoid(&self) -> Var<'t> {
        let value = eval::sigmoid(&self.value());
        let id = self.tape.push_eager(Op::Sigmoid(self.id), value);
        Var::from_raw(self.tape, id)
    }

    /// activation checkpointing（イシュー #1624・`docs/
    /// autodiff-checkpoint-design.md`）: `self` を区間の出力、`inputs`
    /// を区間の外側入力として、`inputs` より後・`self` 以前に push
    /// された再計算可能ノード（`Op::is_checkpoint_eligible()`）を解放
    /// する（`Tape::register_checkpoint` 経由）。`self` 自身は解放
    /// されない（呼び出し元が戻り値として保持し続けるため）。
    ///
    /// [`Tape::checkpoint`]（閉包版）の低儀式な代替入口——facade は
    /// `Tape` newtype への新規 `pub fn` 追加を承認事項として保留して
    /// いる一方、`Var` は素で再エクスポート済みのため、本メソッドが
    /// facade 経由でも到達可能な唯一の checkpoint 入口となる
    /// （`docs/compat-api-scope.md` §1.3）。
    ///
    /// `inputs` が空の場合、区間はテープ先頭（node id 0）から `self`
    /// までとする。`inputs` のいずれかが別 `Tape` に属する場合は
    /// `Err(TapeMismatch)`（クロステープ検査。`check_same_tape` doc
    /// 参照）。`self` の node id が `inputs` のどれよりも小さい（区間が
    /// 空）場合は no-op で `self` をそのまま返す。
    pub fn checkpoint_from(&self, inputs: &[&Var<'t>]) -> Result<Var<'t>, AutodiffError> {
        for &input in inputs {
            self.check_same_tape(input)?;
        }
        let lo = inputs.iter().map(|v| v.node_id().0 + 1).max().unwrap_or(0);
        let output_id = self.id.0;
        if output_id < lo {
            return Ok(Var::from_raw(self.tape, self.id));
        }
        self.tape.register_checkpoint(lo, output_id)?;
        Ok(Var::from_raw(self.tape, self.id))
    }

    /// 新しい shape へ再解釈する view 系ノード（イシュー #1047・親
    /// #1043）。`Tensor::reshape`（`tensor-core`）と同じく contiguous な
    /// 入力に限り zero-copy（案 A・エラー方式。`docs/spec/
    /// public-api-design.md` §2.2.1 は未決事項としていたが、本イシューは
    /// 自動運転モードのため安全側の案 A を踏襲する。案 B〈暗黙コピー〉
    /// への変更はユーザー承認事項として `tensor-core::Tensor::reshape`
    /// のドキュメント参照）。
    ///
    /// 検査順序（演算メソッド規律「①クロステープ検査 → ②shape 検査 →
    /// ③forward 値計算 → ④ノード記録」を view 系向けに具体化）:
    /// ①要素数一致（クロステープ検査は単項演算のため不要）→ ②層 1
    /// （`materialize_fallible`）で入力を実体化 → ③実体化値の
    /// `is_contiguous()` を検査（非 contiguous なら
    /// `ShapeError::NonContiguousReshape` を返し、暗黙コピーでバッファ
    /// 確保 0 の契約を破らない）→ ④`Tape::push_view` でホスト値を持たない
    /// ノードとして記録する（`tape::Op::Reshape` doc「forward のたびに
    /// バッファ確保しない」の中核）。
    pub fn reshape(&self, shape: &[usize]) -> Result<Var<'t>, AutodiffError> {
        let in_shape = self.shape();
        let in_numel: usize = in_shape.iter().product();
        // `checked_numel` 相当のオーバーフロー検査（`tensor-core` は
        // この関数を非公開にしているため、`autodiff` 側で `checked_mul`
        // を用いて自前実装する。REQ-8 趣旨の境界検査 A03 対策）。
        let out_numel = shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
        let out_numel = match out_numel {
            Some(n) => n,
            None => {
                return Err(AutodiffError::Shape(ShapeError::ElementCountOverflow));
            }
        };
        if out_numel != in_numel {
            return Err(AutodiffError::Shape(ShapeError::ElementCountMismatch {
                expected: out_numel,
                actual: in_numel,
            }));
        }

        // 入力を層 1 で実体化する（`Tape::push_view` の呼び出し契約:
        // `input` は push 前に必ず実体化済みであること）。`Ref` を保持
        // したまま `push_view`（`borrow_mut`）を呼ぶと `RefCell` の
        // 二重可変借用 panic になるため、`is_contiguous()` 検査まで
        // 完了してからスコープを閉じる。
        {
            let nodes = self.tape.nodes.borrow();
            let input_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?;
            if !input_val.is_contiguous() {
                return Err(AutodiffError::Shape(ShapeError::NonContiguousReshape));
            }
        }
        let id = self
            .tape
            .push_view(Op::Reshape { input: self.id }, shape.to_vec());
        Ok(Var::from_raw(self.tape, id))
    }

    /// 2 軸の転置（view 系ノード。イシュー #1047・親 #1043）。
    /// `Tensor::transpose`（`tensor-core`）と同じく常に zero-copy
    /// （strides の入れ替えのみ）。`dim0 == dim1` は恒等 view として
    /// 許容する（`Tensor::transpose` 自体が同じ挙動）。
    ///
    /// 検査順序: ①`dim0`／`dim1` が rank 範囲内 → ②`Tape::push_view` で
    /// ホスト値を持たないノードとして記録する（`reshape` と異なり
    /// transpose は非 contiguous 化しても失敗しない演算のため実体化前
    /// 検査は軸範囲のみで足りる。ただし `Tape::push_view` の呼び出し
    /// 契約〈`input` 事前実体化〉を満たすため、軸検査の後に層 1で入力を
    /// 実体化してから記録する）。
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Var<'t>, AutodiffError> {
        let in_shape = self.shape();
        let rank = in_shape.len();
        if dim0 >= rank {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: dim0,
                rank,
            }));
        }
        if dim1 >= rank {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: dim1,
                rank,
            }));
        }
        let mut out_shape = in_shape;
        out_shape.swap(dim0, dim1);

        // `Tape::push_view` の呼び出し契約（`input` は push 前に実体化
        // 済み）を満たす。`Ref` を保持したまま `push_view` を呼ばない
        // よう、実体化はスコープ内で完結させる。
        {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?;
        }
        let id = self.tape.push_view(
            Op::Transpose {
                input: self.id,
                dim0,
                dim1,
            },
            out_shape,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 任意軸並べ替え（view 系ノード。イシュー #1597）。`transpose` の
    /// 2 軸限定を補う N 階一般対応版（`Tensor::permute`／ONNX
    /// `Transpose` と同じ規約: `perm[k]` は出力軸 `k` が指す入力軸）。
    /// 常に zero-copy（strides の並べ替えのみ）。
    ///
    /// 検査順序: ①`perm` の長さが rank と一致（不一致は
    /// `ShapeError::RankMismatch`）→ ②各軸が範囲内（`AxisOutOfRange`）
    /// かつ重複なし（`DuplicateAxis`）→ ③層 1 で入力を実体化してから
    /// `Tape::push_view`（`transpose` と同じ理由で、実体化前検査は
    /// shape 情報のみで足りるため軸検査の後に行う）。
    ///
    /// `perm` が 2 軸のみを入れ替える順列（例 `[1, 0]`）の場合、出力
    /// strides は `Var::transpose(0, 1)` と同一になるため、CPU／Metal
    /// の NT/TN 転置入口（#1213／#1215）の高速経路は本メソッド経由でも
    /// 従来どおり到達する（strides 判定のため）。
    pub fn permute(&self, perm: &[usize]) -> Result<Var<'t>, AutodiffError> {
        let in_shape = self.shape();
        let rank = in_shape.len();
        if perm.len() != rank {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: rank,
                actual: perm.len(),
            }));
        }
        let mut seen = vec![false; rank];
        for &axis in perm {
            if axis >= rank {
                return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                    axis,
                    rank,
                }));
            }
            if seen[axis] {
                return Err(AutodiffError::Shape(ShapeError::DuplicateAxis { axis }));
            }
            seen[axis] = true;
        }
        let out_shape: Vec<usize> = perm.iter().map(|&p| in_shape[p]).collect();

        // `Tape::push_view` の呼び出し契約（`input` は push 前に実体化
        // 済み）を満たす（`transpose` と同じ規律）。
        {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?;
        }
        let id = self.tape.push_view(
            Op::Permute {
                input: self.id,
                perm: perm.to_vec(),
            },
            out_shape,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// `permute`／`transpose` 後の非 contiguous view を明示的に行優先
    /// 連続バッファへ実体化する eager ノード（イシュー #1620）。
    /// `Var::reshape` は非 contiguous 入力を暗黙コピーせず
    /// `ShapeError::NonContiguousReshape` で拒否する契約（案 A・
    /// `docs/public-api-design.md` §3.2）のため、`crate::einsum` が
    /// permute 後に reshape へ渡す前段の明示コピーとして使う。
    /// `Var::conv1d`（イシュー #1765）も同じ理由（`conv2d` は
    /// transpose 済み入力を受理するが `reshape` は非 contiguous を
    /// 拒否する非対称の解消）で reshape 直前の明示コピーに使う。
    ///
    /// 可視性は `pub(crate)` に留める——`Var` は facade から
    /// 再エクスポートされるため、`docs/compat-api-scope.md` の Tier
    /// 列挙にない `contiguous` を `pub` にすると同 §5 手続きの対象になる
    /// 新規公開 API を無断で追加してしまう（承認範囲は `Var::einsum`
    /// 自体のみ）。消費者は `crate::einsum`・`Var::conv1d`。
    ///
    /// 既に contiguous な場合は新規ノードを積まず `self` をそのまま
    /// 返す（`Tensor::contiguous` 自体は contiguous なら clone のみだが、
    /// tape ノードの増殖と VJP パススルー 1 段の追加コストを避ける。
    /// `crate::einsum::apply_permute`／`apply_reshape` の恒等スキップと
    /// 同じ方針）。
    pub(crate) fn contiguous(&self) -> Result<Var<'t>, AutodiffError> {
        let value = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        if value.is_contiguous() {
            return Ok(*self);
        }
        let out = value.contiguous();
        let id = self.tape.push_eager(Op::Contiguous { input: self.id }, out);
        Ok(Var::from_raw(self.tape, id))
    }

    /// einsum 記法（PyTorch `torch.einsum`／TF `tf.einsum` 相当）による
    /// 汎用縮約（イシュー #1620）。既存の `matmul`／`sum`／`permute`／
    /// `reshape`／`mul`（`crate::einsum` モジュール）への分解として
    /// 実装しており、新規カーネルは追加していない。VJP は分解先の
    /// 各演算の VJP 合成として自動的に成立する（`einsum` 専用の VJP を
    /// `grad.rs` に追加していない）。
    ///
    /// **受理範囲（v1・安全側）**: 添字は ASCII 英字のみ（空白は無視）。
    /// オペランドは 1〜2 個限定。ellipsis（`...`）・同一オペランド内の
    /// 添字重複（対角／trace）・出力添字の重複・batch 添字を伴う縮約
    /// （rank≥3 `matmul`〈#1600〉が未実装のため。例 `"bij,bjk->bik"`）は
    /// `AutodiffError::InvalidArgument` で拒否する。`->` 省略時は
    /// NumPy `einsum` 既定（入力に 1 回だけ現れる添字を ASCII 昇順）を
    /// 出力とみなす。詳細な受理範囲・分解アルゴリズムは
    /// `crate::einsum` モジュール doc を参照。
    ///
    /// **数値契約**: 縮約（`contract` が非空）を伴う場合は
    /// `Var::matmul` をそのまま呼ぶため、CUDA TF32 opt-in の挙動を
    /// 含めて `matmul` と同一（`.claude/rules/coding-rust.md` FMA
    /// 契約統一）。
    pub fn einsum(spec: &str, operands: &[&Var<'t>]) -> Result<Var<'t>, AutodiffError> {
        crate::einsum::einsum(spec, operands)
    }

    /// PyTorch `torch.nn.functional.scaled_dot_product_attention` 相当
    /// （イシュー #1639。親 #1605「MultiheadAttention」の sub-issue
    /// (a)。設計・対象外事項は `crate::attention`〈非公開モジュール〉の
    /// doc 参照）。
    ///
    /// `query: [..., L, E]`・`key: [..., S, E]`・`value: [..., S, Ev]`
    /// （バッチ次元 `...` は NumPy 互換ブロードキャスト。`Var::matmul`
    /// と同じ契約）を受け取り `[..., L, Ev]` を返す。
    ///
    /// `attn_mask`（`true` = attend。PyTorch bool mask 規約。`[..., L,
    /// S]` へ broadcast 可能な形状）と `is_causal`（top-left aligned の
    /// `j <= i` causal mask）は同時指定不可（[`AutodiffError::
    /// InvalidArgument`]）。`scale` は `None` のとき `1/sqrt(E)`
    /// （`E == 0` かつ `scale == None` は [`AutodiffError::
    /// InvalidArgument`]）。
    ///
    /// **対象外**: `dropout_p`（SDPA への結線は対象外。`Var::dropout`
    /// 自体は #1603 で実装済み）・`enable_gqa`・attention
    /// weights の返却・f16／bf16（#1626）。既存カーネルの合成のみで
    /// 実装しており、新規 `Op`／`BackendOps` メソッドは追加していない
    /// （CUDA／Metal 専用の融合 attention カーネルは対象外。
    /// `docs/kernel-fusion.md`）。
    ///
    /// # Errors
    ///
    /// `query`／`key` が rank < 2、`attn_mask` と `is_causal` の同時
    /// 指定、`attn_mask` が `[..., L, S]` へ broadcast 不能、いずれかの
    /// 行が全 key を masked にしてしまう、`scale`（既定値含む）が非
    /// 有限・0 以下、のいずれかで型付きエラーを返す。`E`／`S` の不一致・
    /// バッチ broadcast 不能・テープ不一致は内部で呼ぶ `matmul`／
    /// `transpose` の既存検査へ委譲する。
    pub fn scaled_dot_product_attention(
        query: &Var<'t>,
        key: &Var<'t>,
        value: &Var<'t>,
        attn_mask: Option<&fandhe_ai_tensor_core::Tensor<bool>>,
        is_causal: bool,
        scale: Option<f32>,
    ) -> Result<Var<'t>, AutodiffError> {
        crate::attention::scaled_dot_product_attention(
            query, key, value, attn_mask, is_causal, scale,
        )
    }

    /// RNN（tanh 版）セル 1 step（イシュー #1647・設計 `docs/autodiff-
    /// rnn-cell-tape-design.md` 決定 1・4・5）。
    /// `h_t = tanh(x·W_ih + b_ih + h_{t-1}·W_hh + b_hh)`。
    ///
    /// `self` が `x_t: [B, D]`、`h_prev: [B, H]`、`p.w_ih: [D, H]`、
    /// `p.w_hh: [H, H]`、`p.b_ih`／`p.b_hh` はいずれも `[H]`（両方
    /// `Some` か両方 `None`。片方のみは [`AutodiffError::InvalidArgument`]）。
    /// `D == 0`／`H == 0` は zero-K ガードとして拒否する。
    pub fn rnn_cell(
        &self,
        h_prev: &Var<'t>,
        p: GateParams<'_, 't>,
    ) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(h_prev)?;
        self.check_same_tape(p.w_ih)?;
        self.check_same_tape(p.w_hh)?;
        if let Some(b) = p.b_ih {
            self.check_same_tape(b)?;
        }
        if let Some(b) = p.b_hh {
            self.check_same_tape(b)?;
        }
        check_bias_pair(p.b_ih, p.b_hh)?;

        let x_shape = self.shape();
        let h_shape = h_prev.shape();
        let w_ih_shape = p.w_ih.shape();
        let w_hh_shape = p.w_hh.shape();
        require_rank2_all(&[&x_shape, &h_shape, &w_ih_shape, &w_hh_shape], "rnn_cell")?;
        let hidden = h_shape[1];
        require_positive_dims(x_shape[1], hidden, "rnn_cell")?;

        let out_ih = gemm_out_shape(&x_shape, &w_ih_shape)?;
        let out_hh = gemm_out_shape(&h_shape, &w_hh_shape)?;
        require_same_shape(&out_ih, &out_hh)?;
        if out_ih[1] != hidden {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: out_ih,
                rhs: vec![h_shape[0], hidden],
            }));
        }
        if let Some(b) = p.b_ih {
            require_same_shape(&b.shape(), &[hidden])?;
        }
        if let Some(b) = p.b_hh {
            require_same_shape(&b.shape(), &[hidden])?;
        }

        let (x_val, h_prev_val, w_ih_val, w_hh_val, b_ih_val, b_hh_val) = {
            let nodes = self.tape.nodes.borrow();
            let ops = self.tape.ops();
            let x_val = materialize_fallible(&nodes, ops, self.id)?.clone();
            let h_prev_val = materialize_fallible(&nodes, ops, h_prev.id)?.clone();
            let w_ih_val = materialize_fallible(&nodes, ops, p.w_ih.id)?.clone();
            let w_hh_val = materialize_fallible(&nodes, ops, p.w_hh.id)?.clone();
            let b_ih_val = optional_materialize(&nodes, ops, p.b_ih)?;
            let b_hh_val = optional_materialize(&nodes, ops, p.b_hh)?;
            (x_val, h_prev_val, w_ih_val, w_hh_val, b_ih_val, b_hh_val)
        };

        let value = rnn_cell_forward_value(
            self.tape.ops(),
            &x_val,
            &h_prev_val,
            &CellWeights {
                w_ih: &w_ih_val,
                w_hh: &w_hh_val,
                b_ih: b_ih_val.as_ref(),
                b_hh: b_hh_val.as_ref(),
            },
        )?;

        let id = self.tape.push_eager(
            Op::RnnCell {
                x: self.id,
                h_prev: h_prev.id,
                w_ih: p.w_ih.id,
                w_hh: p.w_hh.id,
                b_ih: p.b_ih.map(|b| b.id),
                b_hh: p.b_hh.map(|b| b.id),
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// `shape` へブロードキャストする view 系ノード（イシュー #1597）。
    /// `Tensor::broadcast_to`（NumPy `broadcast_to` 相当）と同じ規約:
    /// 拡張軸（元の軸長 1 が `shape` 側で 1 より大きい値に広がる軸）は
    /// stride 0 の view になり zero-copy。同一 shape への broadcast は
    /// 恒等 view として許容する。
    ///
    /// 検査順序: ①要素数オーバーフロー検査（`reshape` と同じ自前
    /// `checked_mul` 実装。`tensor-core` は `checked_numel` を非公開に
    /// しているため）→ ②`shape.len() < rank` または各軸が
    /// `src == dst || src == 1` を満たさない場合は
    /// `ShapeError::BroadcastIncompatible` → ③層 1 で入力を実体化して
    /// から `Tape::push_view`。
    ///
    /// VJP（`grad.rs`）は `Op::Add`／`Op::Mul` の暗黙ブロードキャストと
    /// 同じ `reduce_to_shape` 縮約を使うため、勾配バッファの確保を
    /// 伴う（`reshape`／`transpose`／`permute` の VJP は zero-copy だが
    /// 本メソッドの VJP は異なる。`tape::Op::BroadcastTo` doc 参照）。
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Var<'t>, AutodiffError> {
        let in_shape = self.shape();
        // `checked_numel` 相当のオーバーフロー検査（`reshape` と同じ
        // 自前実装。REQ-8 趣旨の境界検査 A03 対策）。
        let out_numel = shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
        if out_numel.is_none() {
            return Err(AutodiffError::Shape(ShapeError::ElementCountOverflow));
        }
        if shape.len() < in_shape.len() {
            return Err(AutodiffError::Shape(ShapeError::BroadcastIncompatible {
                lhs: in_shape,
                rhs: shape.to_vec(),
            }));
        }
        let offset_axes = shape.len() - in_shape.len();
        for (&src, &dst) in in_shape.iter().zip(&shape[offset_axes..]) {
            if src != dst && src != 1 {
                return Err(AutodiffError::Shape(ShapeError::BroadcastIncompatible {
                    lhs: in_shape,
                    rhs: shape.to_vec(),
                }));
            }
        }

        // `Tape::push_view` の呼び出し契約（`input` は push 前に実体化
        // 済み）を満たす。
        {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?;
        }
        let id = self
            .tape
            .push_view(Op::BroadcastTo { input: self.id }, shape.to_vec());
        Ok(Var::from_raw(self.tape, id))
    }

    /// [`Var::broadcast_to`] の PyTorch 名別名（`Tensor.expand`。イシュー
    /// #1597）。負値（-1 で当該軸を維持する PyTorch の記法）は
    /// `shape: &[usize]` の型上表現できないため非対応——呼び出し側は
    /// 維持したい軸の実サイズを明示的に渡すこと。
    pub fn expand(&self, shape: &[usize]) -> Result<Var<'t>, AutodiffError> {
        self.broadcast_to(shape)
    }

    /// 長さ 1 の軸を除去する view 系ノード（`Var::reshape` への委譲。
    /// イシュー #1597）。
    ///
    /// - `dim: None` — 長さ 1 の軸をすべて除去する（NumPy／PyTorch
    ///   `squeeze()` と同じ）。
    /// - `dim: Some(d)` — 軸 `d` が rank 範囲外なら
    ///   `ShapeError::AxisOutOfRange`。`d` が範囲内だが `shape[d] != 1`
    ///   の場合は **PyTorch 準拠の no-op**（shape を変えず `reshape` を
    ///   記録する。numpy／TensorFlow はここをエラーにするが、適合する
    ///   `ShapeError` variant が存在せず、crates.io 公開クレート
    ///   `tensor-core` の公開 enum への variant 追加は semver 可視の
    ///   変更になるため、本イシューでは PyTorch 方式を採用する）。
    ///
    /// 非 contiguous な入力（例: `permute`／`transpose`／`broadcast_to`
    /// の直後）に対する制約は `reshape` と同じ（`ShapeError::
    /// NonContiguousReshape`）。
    pub fn squeeze(&self, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
        let in_shape = self.shape();
        let rank = in_shape.len();
        let out_shape: Vec<usize> = match dim {
            None => in_shape.into_iter().filter(|&d| d != 1).collect(),
            Some(d) => {
                if d >= rank {
                    return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                        axis: d,
                        rank,
                    }));
                }
                if in_shape[d] != 1 {
                    // PyTorch 準拠 no-op（上記 doc 参照）。
                    in_shape
                } else {
                    let mut out = in_shape;
                    out.remove(d);
                    out
                }
            }
        };
        self.reshape(&out_shape)
    }

    /// 長さ 1 の軸を挿入する view 系ノード（`Var::reshape` への委譲。
    /// イシュー #1597）。`dim` は挿入後の rank（`rank + 1`）に対する
    /// 軸位置として扱うため有効範囲は `0..=rank`（PyTorch
    /// `unsqueeze` と同じ: 末尾への挿入 `dim == rank` を許容する）。
    /// 範囲外は `ShapeError::AxisOutOfRange { axis: dim, rank: rank + 1
    /// }`。
    ///
    /// 非 contiguous な入力に対する制約は `reshape` と同じ。
    pub fn unsqueeze(&self, dim: usize) -> Result<Var<'t>, AutodiffError> {
        let in_shape = self.shape();
        let rank = in_shape.len();
        if dim > rank {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: dim,
                rank: rank + 1,
            }));
        }
        let mut out_shape = in_shape;
        out_shape.insert(dim, 1);
        self.reshape(&out_shape)
    }

    /// `[start_dim, end_dim]`（両端含む）の連続する軸を 1 軸へ潰す
    /// view 系ノード（`Var::reshape` への委譲。イシュー #1597）。
    /// PyTorch `torch.flatten(start_dim, end_dim)` と同じ規約。
    ///
    /// `end_dim >= rank` または `start_dim > end_dim` は
    /// `ShapeError::AxisOutOfRange`（後者は `axis: start_dim` として
    /// 報告する）。rank 0（スカラー）は `(start_dim, end_dim) ==
    /// (0, 0)` のみ許容し `[1]` を返す（PyTorch と同じ）。潰す軸区間
    /// の部分積自体は `usize::MAX` を含むゼロ長軸混在形状で
    /// オーバーフローしうるため `checked_mul` で検査し、オーバー
    /// フロー時は `ShapeError::ElementCountOverflow` を返す（総要素数
    /// が `reshape` 側で検査済みでも部分積は別途検査が要る）。
    ///
    /// 非 contiguous な入力に対する制約は `reshape` と同じ。
    pub fn flatten(&self, start_dim: usize, end_dim: usize) -> Result<Var<'t>, AutodiffError> {
        // shape 検査・出力 shape の確定は `tensor-core::flatten_out_shape`
        // へ委譲する（イシュー #2065 でインライン実装を切り出し）。
        // `nn::Flatten::forward_host`（tape 不要経路。`compat::Sequential::
        // predict` が使う）も同じ関数を呼ぶことで、2 経路間の判定基準が
        // 食い違う「判定迂回経路」を作らない（`Softmax::forward_host` が
        // `reduce_out_shape` を tape 経路と共有する既存パターンの踏襲。
        // `.claude/rules/security.md` A08）。
        let out_shape = flatten_out_shape(&self.shape(), start_dim, end_dim)?;
        self.reshape(&out_shape)
    }

    /// 複数の `Var` を `dim` 軸で連結する（`torch.cat` 相当。イシュー
    /// #1598）。関連関数（`&self` を取らない）——`Var` は `Copy` の
    /// ため `&[Var<'t>]` で受ける。
    ///
    /// 検査順序: ①空リストは `vars[0]` に触れる前に
    /// `AutodiffError::InvalidArgument` → ②先頭要素基準の
    /// `check_same_tape`（`AutodiffError::TapeMismatch`）→ ③shape 検査
    /// （[`fandhe_ai_tensor_core::concat_out_shape`]。rank 不一致・
    /// `dim` 範囲外・`dim` 以外の軸不一致・要素数オーバーフローを
    /// 個別 variant で報告）→ ④全入力を層 1 で実体化 → ⑤
    /// `Tape::push_eager` で記録する（`Op::Concat` doc 参照）。
    ///
    /// forward 値は `grad::concat_with_fallback`（`ops.concat` →
    /// `Unsupported` のときのみ `eval::concat`）で計算する。VJP
    /// （`grad.rs`）は各入力へ `upstream.narrow` を zero-copy に分配
    /// する。1 要素リストも通常どおり `Op::Concat` ノードを記録する
    /// （恒等コピー）。
    pub fn cat(vars: &[Var<'t>], dim: usize) -> Result<Var<'t>, AutodiffError> {
        let first = match vars.first() {
            Some(v) => v,
            None => {
                return Err(AutodiffError::InvalidArgument(
                    "Var::cat: vars must not be empty".into(),
                ));
            }
        };
        for v in &vars[1..] {
            first.check_same_tape(v)?;
        }
        let shapes: Vec<Vec<usize>> = vars.iter().map(|v| v.shape()).collect();
        let shape_refs: Vec<&[usize]> = shapes.iter().map(|s| s.as_slice()).collect();
        let out_shape = concat_out_shape(&shape_refs, dim).map_err(AutodiffError::Shape)?;

        // `Tape::push_eager` に渡す forward 値を計算するため、全入力を
        // 層 1 で実体化してから所有値として持ち出す（`nodes` の
        // `RefCell` 借用を閉じてから `push_eager`〈`borrow_mut`〉を
        // 呼ぶ規律。モジュール冒頭コメント参照）。
        let materialized: Vec<Tensor<f32>> = {
            let nodes = first.tape.nodes.borrow();
            let ops = first.tape.ops();
            let mut out = Vec::with_capacity(vars.len());
            for v in vars {
                out.push(materialize_fallible(&nodes, ops, v.id)?.clone());
            }
            out
        };
        let refs: Vec<&Tensor<f32>> = materialized.iter().collect();
        let value = concat_with_fallback(first.tape.ops(), &refs, dim, &out_shape)?;

        let id = first.tape.push_eager(
            Op::Concat {
                inputs: vars.iter().map(|v| v.id).collect(),
                dim,
            },
            value,
        );
        Ok(Var::from_raw(first.tape, id))
    }

    /// 複数の `Var` を新規軸 `dim` で積み上げる（`torch.stack` 相当。
    /// イシュー #1598）。各要素を `unsqueeze(dim)` してから
    /// [`Self::cat`] する（PyTorch の定義そのもの）。関連関数。
    ///
    /// 検査順序: ①空リスト → `InvalidArgument`（`vars[0]` に触れる
    /// 前）→ ②先頭要素基準の `check_same_tape` → ③`dim <= rank`
    /// （`ShapeError::AxisOutOfRange { rank: rank + 1 }`）→ ④全要素の
    /// shape 完全一致（`ShapeError::ShapeMismatch`）→ ここまで全て
    /// 通過してから `unsqueeze` ノードを push する（失敗した `stack`
    /// がテープに中間ノードを残さないよう、`unsqueeze` を呼ぶ前に
    /// 全要素の contiguity も検査する）。
    ///
    /// `unsqueeze` は `reshape` へ委譲するため、**非 contiguous な
    /// 要素**（`permute`／`transpose` 直後）は
    /// `ShapeError::NonContiguousReshape` を返す（暗黙 `contiguous()`
    /// はしない。`Var::reshape` doc「案 A」と同じ規律）。
    pub fn stack(vars: &[Var<'t>], dim: usize) -> Result<Var<'t>, AutodiffError> {
        let first = match vars.first() {
            Some(v) => v,
            None => {
                return Err(AutodiffError::InvalidArgument(
                    "Var::stack: vars must not be empty".into(),
                ));
            }
        };
        for v in &vars[1..] {
            first.check_same_tape(v)?;
        }
        let in_shape = first.shape();
        let rank = in_shape.len();
        if dim > rank {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: dim,
                rank: rank + 1,
            }));
        }
        for v in &vars[1..] {
            let s = v.shape();
            if s != in_shape {
                return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                    lhs: in_shape.clone(),
                    rhs: s,
                }));
            }
        }
        for v in vars {
            let nodes = v.tape.nodes.borrow();
            let val = materialize_fallible(&nodes, v.tape.ops(), v.id)?;
            if !val.is_contiguous() {
                return Err(AutodiffError::Shape(ShapeError::NonContiguousReshape));
            }
        }
        let unsqueezed: Vec<Var<'t>> = vars
            .iter()
            .map(|v| v.unsqueeze(dim))
            .collect::<Result<_, _>>()?;
        Self::cat(&unsqueezed, dim)
    }

    /// `[start, start+len)` を切り出す zero-copy view（`torch.narrow`
    /// 相当。イシュー #1598・#1599「narrow」行の解消）。
    ///
    /// 検査順序: ①`dim < rank`（`ShapeError::AxisOutOfRange`）→
    /// ②`start + len <= shape[dim]`（`checked_add`。
    /// `ShapeError::NarrowOutOfBounds`）→ ③層 1 で入力を実体化してから
    /// `Tape::push_view`（`transpose`／`permute` と同じ規律）。
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Var<'t>, AutodiffError> {
        let in_shape = self.shape();
        let rank = in_shape.len();
        if dim >= rank {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: dim,
                rank,
            }));
        }
        let dim_size = in_shape[dim];
        let in_bounds = start.checked_add(len).is_some_and(|end| end <= dim_size);
        if !in_bounds {
            return Err(AutodiffError::Shape(ShapeError::NarrowOutOfBounds {
                dim,
                start,
                len,
                dim_size,
            }));
        }
        let mut out_shape = in_shape;
        out_shape[dim] = len;

        {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?;
        }
        let id = self.tape.push_view(
            Op::Narrow {
                input: self.id,
                dim,
                start,
                len,
            },
            out_shape,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 各軸を定数値 `value` で拡張する（`torch.nn.functional.pad
    /// (mode='constant')` 相当。イシュー #1756）。`pads[i] = (before,
    /// after)` は先頭次元から順に対応する（PyTorch `F.pad` の「末尾
    /// 次元から逆順の平坦リスト」とは異なる意図的な設計。負パディング
    /// （クロップ）は非対応——[`Self::narrow`] を使うこと。`reflect`／
    /// `replicate` モードも対象外。`docs/compat-feature-gap.md`
    /// 追補参照）。
    ///
    /// 検査順序: ①[`fandhe_ai_tensor_core::pad_out_shape`]（`pads.len()
    /// == rank`・各軸の `before`／`after` 加算オーバーフロー・
    /// 出力要素数積オーバーフローを検査し `out_shape` を確定。違反は
    /// `AutodiffError::Shape`）→ ②`self` を層 1 で実体化（`RefCell`
    /// 借用を閉じてから push。`Var::gather` と同じ「実体化してから
    /// フォールバックへ渡す」方針）→ ③`pad_with_fallback`
    /// （`ops.pad` → `Unsupported` のときのみホスト参照実装
    /// `eval::pad` へフォールバック）→ ④戻り shape 検証
    /// （`.claude/rules/security.md` A08）→ ⑤`push_eager`
    /// （非融合・常実体化。`value` は forward 記録値へ焼き込み済みの
    /// ため `Op::Pad` 自身は保持しない）。
    pub fn pad(&self, pads: &[(usize, usize)], value: f32) -> Result<Var<'t>, AutodiffError> {
        let in_shape = self.shape();
        let out_shape = pad_out_shape(&in_shape, pads).map_err(AutodiffError::Shape)?;

        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };

        let value_out = pad_with_fallback(self.tape.ops(), &input_val, pads, value, &out_shape)?;
        if value_out.shape() != out_shape {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: value_out.shape().to_vec(),
                    rhs: out_shape,
                },
            )));
        }
        let id = self.tape.push_eager(
            Op::Pad {
                input: self.id,
                pads: pads.to_vec(),
            },
            value_out,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 2 次元畳み込み（`torch.nn.functional.conv2d` 相当の
    /// cross-correlation。NCHW 固定。イシュー #1764・設計 `docs/
    /// conv-ops-design.md`）。`self`（`input`）: `[N, Cin, H, W]`・
    /// `weight`: `[Cout, Cin/groups, kH, kW]`（PyTorch 準拠。`nn::
    /// Linear` の `[in, out]` 格納とは異なる）・`bias`: `Some` なら
    /// `[Cout]`。
    ///
    /// 検査順序（設計 doc §3）: ①`weight.shape()[2..4]` から
    /// `kernel_size` を導出し [`Conv2dParams::new`]（`stride`／
    /// `dilation`／`groups` の 0・`2·padding` オーバーフローを
    /// `BackendError::InvalidArgument` → `AutodiffError::InvalidArgument`
    /// で拒否）→ ②[`fandhe_ai_tensor_core::conv2d_out_shape`] で
    /// `out_shape` を確定（rank・チャンネル整合・空間軸 `H`／`W = 0`
    /// 拒否・負分子拒否ゲート。`AutodiffError::Shape`）→ ③`self`／
    /// `weight`／`bias` を層 1 で実体化（`RefCell` 借用を閉じてから
    /// push）→ ④`conv2d_with_fallback`（`grad` 内非公開。`ops.conv2d` → im2col →
    /// `gemm_batched`〈常にバックエンド〉→ `add`〈bias〉の段階的合成）
    /// → ⑤戻り shape 検証（`.claude/rules/security.md` A08）→
    /// ⑥`push_eager`（非融合・常実体化。`col` は `Op::Conv2d` 自身に
    /// 保持しない——backward で再計算する）。
    pub fn conv2d(
        &self,
        weight: &Var<'t>,
        bias: Option<&Var<'t>>,
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(weight)?;
        if let Some(b) = bias {
            self.check_same_tape(b)?;
        }

        let weight_shape = weight.shape();
        if weight_shape.len() != 4 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 4,
                actual: weight_shape.len(),
            }));
        }
        let kernel_size = [weight_shape[2], weight_shape[3]];
        let params = Conv2dParams::new(kernel_size, stride, padding, dilation, groups)
            .map_err(AutodiffError::Backend)?;

        let in_shape = self.shape();
        let out_shape =
            conv2d_out_shape(&in_shape, &weight_shape, &params).map_err(AutodiffError::Shape)?;
        if let Some(b) = bias {
            let bias_shape = b.shape();
            let cout = weight_shape[0];
            if bias_shape != [cout] {
                return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                    lhs: bias_shape,
                    rhs: vec![cout],
                }));
            }
        }

        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let weight_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), weight.id)?.clone()
        };
        let bias_val = match bias {
            Some(b) => {
                let nodes = self.tape.nodes.borrow();
                Some(materialize_fallible(&nodes, self.tape.ops(), b.id)?.clone())
            }
            None => None,
        };

        let value_out = conv2d_with_fallback(
            self.tape.ops(),
            &input_val,
            &weight_val,
            bias_val.as_ref(),
            &params,
            &out_shape,
        )?;
        if value_out.shape() != out_shape {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: value_out.shape().to_vec(),
                    rhs: out_shape,
                },
            )));
        }
        let id = self.tape.push_eager(
            Op::Conv2d {
                input: self.id,
                weight: weight.id,
                bias: bias.map(|b| b.id),
                params,
            },
            value_out,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 2 次元転置畳み込み（`torch.nn.functional.conv_transpose2d`／
    /// `nn.ConvTranspose2d` 相当。NCHW 固定。イシュー #2067・設計
    /// `docs/conv-ops-design.md` §15）。`self`（`input`）: `[N, Cin,
    /// H, W]`・`weight`: `[Cin, Cout/groups, kH, kW]`（PyTorch
    /// `nn.ConvTranspose2d.weight` のレイアウト。[`Self::conv2d`] の
    /// `[Cout, Cin/groups, kH, kW]` と先頭 2 軸が逆な点に注意）・
    /// `bias`: `Some` なら `[Cout]`。
    ///
    /// **意図的な PyTorch 非互換**: `output_padding[i] < stride[i]`
    /// を要求する（PyTorch は `output_padding < max(stride,
    /// dilation)` まで許容）。3 バックエンド共通の `BackendOps::
    /// col2im` override 契約（`P` 軸＝`conv_out_len(Hout)·
    /// conv_out_len(Wout)`）と、`op >= stride` では
    /// `conv_out_len(Hout) ≠ H` になり不整合を起こすため
    /// （[`fandhe_ai_tensor_core::conv_transpose_out_len`] doc 参照。
    /// 緩和は `col2im` の `P` 軸契約を `conv_out_len` から切り離す
    /// 拡張が必要でスコープ外）。
    ///
    /// 検査順序: ①`check_same_tape` → ②weight rank 4 検査 →
    /// ③`weight.shape()[2..4]` から `kernel_size` を導出し
    /// [`Conv2dParams::new`]（`stride == 0` はここで拒否される。
    /// `output_padding` 検査より先に置くことで `stride=0` 入力が
    /// `output_padding` 起因の誤ったエラーで拒否されるのを防ぐ）→
    /// ④`output_padding[i] < stride[i]`（`AutodiffError::
    /// InvalidArgument`）→ ⑤[`conv_transpose2d_out_shape`] で
    /// `out_shape` を確定（rank・チャンネル整合・空間軸 `H`／`W = 0`
    /// 拒否・`op < s` ゲート込み）→ ⑥bias shape 検査 → ⑦`self`／
    /// `weight`／`bias` を実体化 → ⑧`conv_transpose2d_with_fallback`
    /// （`grad` 内非公開。§15「w_mat・x4・col2im」の段階的合成）→
    /// ⑨戻り shape 検証（`.claude/rules/security.md` A08）→
    /// ⑩`push_eager`（非融合・常実体化。`col2im`／`im2col` の中間
    /// 結果は保持せず backward で再計算する。[`Op::Conv2d`] と同型）。
    #[allow(clippy::too_many_arguments)] // PyTorch `F.conv_transpose2d` の全引数（output_padding 含む）を受理するため（`nn/conv.rs` の allow 方針を踏襲）。
    pub fn conv_transpose2d(
        &self,
        weight: &Var<'t>,
        bias: Option<&Var<'t>>,
        stride: [usize; 2],
        padding: [usize; 2],
        output_padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(weight)?;
        if let Some(b) = bias {
            self.check_same_tape(b)?;
        }

        let weight_shape = weight.shape();
        if weight_shape.len() != 4 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 4,
                actual: weight_shape.len(),
            }));
        }
        let kernel_size = [weight_shape[2], weight_shape[3]];
        let params = Conv2dParams::new(kernel_size, stride, padding, dilation, groups)
            .map_err(AutodiffError::Backend)?;
        if output_padding[0] >= stride[0] || output_padding[1] >= stride[1] {
            return Err(AutodiffError::InvalidArgument(format!(
                "Var::conv_transpose2d: output_padding ({output_padding:?}) must be < stride \
                 ({stride:?}) on each axis (col2im の P 軸契約による意図的な PyTorch 非互換。\
                 設計 docs/conv-ops-design.md §15)"
            )));
        }

        let in_shape = self.shape();
        let out_shape =
            conv_transpose2d_out_shape(&in_shape, &weight_shape, &params, output_padding)
                .map_err(AutodiffError::Shape)?;
        if let Some(b) = bias {
            let bias_shape = b.shape();
            let cout = out_shape[1];
            if bias_shape != [cout] {
                return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                    lhs: bias_shape,
                    rhs: vec![cout],
                }));
            }
        }

        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let weight_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), weight.id)?.clone()
        };
        let bias_val = match bias {
            Some(b) => {
                let nodes = self.tape.nodes.borrow();
                Some(materialize_fallible(&nodes, self.tape.ops(), b.id)?.clone())
            }
            None => None,
        };

        let value_out = conv_transpose2d_with_fallback(
            self.tape.ops(),
            &input_val,
            &weight_val,
            bias_val.as_ref(),
            &params,
            &out_shape,
        )?;
        if value_out.shape() != out_shape {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: value_out.shape().to_vec(),
                    rhs: out_shape,
                },
            )));
        }
        let id = self.tape.push_eager(
            Op::ConvTranspose2d {
                input: self.id,
                weight: weight.id,
                bias: bias.map(|b| b.id),
                params,
            },
            value_out,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 1 次元畳み込み（`torch.nn.functional.conv1d` 相当の
    /// cross-correlation）。[`Self::conv2d`] を `H` 軸固定
    /// （`kernel=1`・`stride=1`・`padding=0`・`dilation=1`）で呼び出す
    /// reshape 併合の薄いラッパーであり、新規 `Op`／`BackendOps`
    /// メソッド・VJP・バックエンドカーネルは追加しない（設計
    /// `docs/conv-ops-design.md` §2／§8・イシュー #1765）。`self`
    /// （`input`）: `[N, Cin, L]`・`weight`: `[Cout, Cin/groups, k]`・
    /// `bias`: `Some` なら `[Cout]`。
    ///
    /// 検査順序（`Var::reshape` は view ノードを tape へ push する
    /// ため、reshape より前にすべての引数検査を完了させ `Err` 経路で
    /// 孤児ノードを残さない）: ①`check_same_tape`（weight／bias。
    /// クロステープの weight を reshape すると相手 tape へ push して
    /// しまうため最初に行う）→ ②`self`／`weight` の rank 検査
    /// （rank 3 以外は `ShapeError::RankMismatch`）→ ③
    /// `Conv2dParams::new`（`H` 軸固定・`W` 軸に `stride`／
    /// `padding`／`dilation` を渡す。`stride`／`dilation`／`groups`
    /// の 0・`2 * padding` オーバーフローを拒否）→ ④合成した 4d
    /// shape に対する [`fandhe_ai_tensor_core::conv2d_out_shape`]
    /// （チャンネル整合・`L = 0` 拒否・負分子拒否ゲート。純粋な
    /// shape 計算で tape 非接触）→ ⑤bias shape 検査 → ⑥ここで
    /// 初めて `input`／`weight` を `[N, Cin, 1, L]`／
    /// `[Cout, Cin/groups, 1, k]` へ reshape し `conv2d` を呼ぶ
    /// （`Self::contiguous` 前段で `Var::reshape` の非 contiguous
    /// 拒否契約と `conv2d`〈transpose 済み入力も受理〉の間の非対称を
    /// 解消する）→ ⑦出力 `[N, Cout, 1, Lout]` を `[N, Cout, Lout]`
    /// へ reshape。
    ///
    /// `conv2d` 内部で再検査される項目（rank・チャンネル整合等）は
    /// fail-closed の二重検査として許容する。`conv2d` 側のバックエンド
    /// 失敗（`Unsupported` 以外）で view ノードが残る点は他の eager
    /// 演算と同じ振る舞い。CUDA 専用 im2col／col2im カーネルは
    /// #1766／#1767 で到達済み・Metal 専用 im2col／col2im カーネルも
    /// #1768 で到達済み（いずれも新規 `Op`／`BackendOps`／カーネル
    /// なしの reshape 併合のため、`conv2d` 側の CUDA／Metal override
    /// へそのまま委譲される。Metal 経路の 1d 形状テスト（model・
    /// ops・facade bit 一致）は #1769 で追加済み。CUDA／Metal 実機
    /// 実測は #1771 へ申し送り）。`nn::Conv1d` 層・
    /// `compat::Sequential::add_conv1d` は対象外（#1770 へ引き継ぐ）。
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d(
        &self,
        weight: &Var<'t>,
        bias: Option<&Var<'t>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(weight)?;
        if let Some(b) = bias {
            self.check_same_tape(b)?;
        }

        let in_shape = self.shape();
        if in_shape.len() != 3 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 3,
                actual: in_shape.len(),
            }));
        }
        let weight_shape = weight.shape();
        if weight_shape.len() != 3 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 3,
                actual: weight_shape.len(),
            }));
        }

        let (n, cin, l) = (in_shape[0], in_shape[1], in_shape[2]);
        let (cout, cin_g, k) = (weight_shape[0], weight_shape[1], weight_shape[2]);

        let params = Conv2dParams::new([1, k], [1, stride], [0, padding], [1, dilation], groups)
            .map_err(AutodiffError::Backend)?;

        let in_shape_4d = vec![n, cin, 1, l];
        let weight_shape_4d = vec![cout, cin_g, 1, k];
        let out_shape_4d = conv2d_out_shape(&in_shape_4d, &weight_shape_4d, &params)
            .map_err(AutodiffError::Shape)?;
        let lout = out_shape_4d[3];

        if let Some(b) = bias {
            let bias_shape = b.shape();
            if bias_shape != [cout] {
                return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                    lhs: bias_shape,
                    rhs: vec![cout],
                }));
            }
        }

        let x4 = self.contiguous()?.reshape(&[n, cin, 1, l])?;
        let w4 = weight.contiguous()?.reshape(&[cout, cin_g, 1, k])?;
        let out4 = x4.conv2d(&w4, bias, [1, stride], [0, padding], [1, dilation], groups)?;
        out4.reshape(&[n, cout, lout])
    }

    /// 2 次元 max pooling（`torch.nn.functional.max_pool2d` 相当。
    /// NCHW 固定。イシュー #1728・設計 `docs/pooling-ops-design.md`）。
    /// `self`（`input`）: `[N, C, H, W]`。戻り値は `(values, index)`
    /// で、`index`（`(n,c)` 平面内 flat 添字 `h·W+w`）は `Var::topk`
    /// と同様に追跡外の `Tensor<i32>` として公開する。
    ///
    /// 検査順序（設計 doc §3・§5）: ①`ceil_mode == true` を
    /// `AutodiffError::InvalidArgument` で拒否（v1 は `false` のみ）
    /// → ②[`Pool2dParams::new`]（`kernel_size`／`stride`／`dilation`
    /// の 0・padding 上限〈`padding <= kernel/2`〉超過を
    /// `BackendError::InvalidArgument` → `AutodiffError::Backend` で
    /// 拒否）→ ③[`pool2d_out_shape`]（rank・空間軸ゼロ拒否・負分子
    /// 拒否ゲート・空窓拒否。`AutodiffError::Shape`）→ ④
    /// `H·W <= i32::MAX`（索引は `i32` のため。`Var::sort`／`topk`
    /// にはない Max 固有の追加検査）→ ⑤`self` を層 1 で実体化
    /// （`RefCell` 借用を閉じてから push）→ ⑥
    /// `max_pool2d_with_fallback` → ⑦戻り shape 再検証
    /// （`.claude/rules/security.md` A08）→ ⑧`push_eager`
    /// （非融合・常実体化）。
    pub fn max_pool2d(
        &self,
        kernel_size: [usize; 2],
        stride: Option<[usize; 2]>,
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
        if ceil_mode {
            return Err(AutodiffError::InvalidArgument(
                "Var::max_pool2d: ceil_mode=true は v1 で未対応".into(),
            ));
        }
        let params = Pool2dParams::new(kernel_size, stride, padding, dilation)
            .map_err(AutodiffError::Backend)?;

        let in_shape = self.shape();
        let out_shape = pool2d_out_shape(&in_shape, &params).map_err(AutodiffError::Shape)?;
        let hw = in_shape[2]
            .checked_mul(in_shape[3])
            .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
        if hw > i32::MAX as usize {
            return Err(AutodiffError::Shape(ShapeError::IndexRangeOverflow {
                index: hw,
            }));
        }

        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };

        let (value, index) =
            max_pool2d_with_fallback(self.tape.ops(), &input_val, &params, &out_shape)?;
        if value.shape() != out_shape || index.shape() != out_shape {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: value.shape().to_vec(),
                    rhs: out_shape,
                },
            )));
        }
        let id = self.tape.push_eager(
            Op::MaxPool2d {
                input: self.id,
                index: index.clone(),
            },
            value,
        );
        Ok((Var::from_raw(self.tape, id), index))
    }

    /// 1 次元 max pooling。[`Self::max_pool2d`] を `H` 軸固定
    /// （`kernel=1`・`stride=1`・`padding=0`・`dilation=1`）で呼び出す
    /// reshape 併合の薄いラッパー（[`Self::conv1d`] と同型。イシュー
    /// #1728）。`self`（`input`）: `[N, C, L]`。索引は `H=1` のため
    /// flat `w` そのもの。
    pub fn max_pool1d(
        &self,
        kernel_size: usize,
        stride: Option<usize>,
        padding: usize,
        dilation: usize,
        ceil_mode: bool,
    ) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
        let in_shape = self.shape();
        if in_shape.len() != 3 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 3,
                actual: in_shape.len(),
            }));
        }
        let (n, c, l) = (in_shape[0], in_shape[1], in_shape[2]);
        // 検査を reshape より前に行う（`Var::conv1d` と同じ規律。
        // reshape 後にエラーを返すと孤立した view ノードがテープに
        // 残るため）。`max_pool2d` 側の再検査はフェイルクローズドの
        // 二重化として残す。
        if ceil_mode {
            return Err(AutodiffError::InvalidArgument(
                "Var::max_pool1d: ceil_mode=true は v1 で未対応".into(),
            ));
        }
        let params_1d = Pool2dParams::new(
            [1, kernel_size],
            stride.map(|s| [1, s]),
            [0, padding],
            [1, dilation],
        )
        .map_err(AutodiffError::Backend)?;
        pool2d_out_shape(&[n, c, 1, l], &params_1d).map_err(AutodiffError::Shape)?;

        let x4 = self.contiguous()?.reshape(&[n, c, 1, l])?;
        let (out4, index) = x4.max_pool2d(
            [1, kernel_size],
            stride.map(|s| [1, s]),
            [0, padding],
            [1, dilation],
            ceil_mode,
        )?;
        let out_shape4 = out4.shape();
        let lout = out_shape4[3];
        let out = out4.reshape(&[n, c, lout])?;
        let index = index.reshape(&[n, c, lout]).map_err(AutodiffError::Shape)?;
        Ok((out, index))
    }

    /// 2 次元 average pooling（`torch.nn.functional.avg_pool2d`
    /// 相当。NCHW 固定。イシュー #1728・設計 `docs/pooling-ops-
    /// design.md`）。`self`（`input`）: `[N, C, H, W]`。
    ///
    /// 検査順序: ①`ceil_mode == true` を `AutodiffError::
    /// InvalidArgument` で拒否 → ②[`Pool2dParams::new`]（`dilation`
    /// は常に `[1, 1]` 固定で構築する。設計 doc §2「Avg 系は
    /// `dilation=[1,1]` 固定」）→ ③[`pool2d_out_shape`] → ④実体化 →
    /// ⑤`avg_pool2d_with_fallback` → ⑥戻り shape 再検証 →
    /// ⑦`push_eager`。
    pub fn avg_pool2d(
        &self,
        kernel_size: [usize; 2],
        stride: Option<[usize; 2]>,
        padding: [usize; 2],
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Result<Var<'t>, AutodiffError> {
        if ceil_mode {
            return Err(AutodiffError::InvalidArgument(
                "Var::avg_pool2d: ceil_mode=true は v1 で未対応".into(),
            ));
        }
        let params = Pool2dParams::new(kernel_size, stride, padding, [1, 1])
            .map_err(AutodiffError::Backend)?;

        let in_shape = self.shape();
        let out_shape = pool2d_out_shape(&in_shape, &params).map_err(AutodiffError::Shape)?;

        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };

        let value_out = avg_pool2d_with_fallback(
            self.tape.ops(),
            &input_val,
            &params,
            count_include_pad,
            &out_shape,
        )?;
        if value_out.shape() != out_shape {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: value_out.shape().to_vec(),
                    rhs: out_shape,
                },
            )));
        }
        let id = self.tape.push_eager(
            Op::AvgPool2d {
                input: self.id,
                params,
                count_include_pad,
            },
            value_out,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 1 次元 average pooling。[`Self::avg_pool2d`] を `H` 軸固定で
    /// 呼び出す reshape 併合の薄いラッパー（[`Self::max_pool1d`] と
    /// 同型。イシュー #1728）。`self`（`input`）: `[N, C, L]`。
    pub fn avg_pool1d(
        &self,
        kernel_size: usize,
        stride: Option<usize>,
        padding: usize,
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Result<Var<'t>, AutodiffError> {
        let in_shape = self.shape();
        if in_shape.len() != 3 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 3,
                actual: in_shape.len(),
            }));
        }
        let (n, c, l) = (in_shape[0], in_shape[1], in_shape[2]);
        // 検査を reshape より前に行う（`Var::conv1d` と同じ規律。
        // `avg_pool2d` 側の再検査はフェイルクローズドの二重化として
        // 残す）。
        if ceil_mode {
            return Err(AutodiffError::InvalidArgument(
                "Var::avg_pool1d: ceil_mode=true は v1 で未対応".into(),
            ));
        }
        let params_1d = Pool2dParams::new(
            [1, kernel_size],
            stride.map(|s| [1, s]),
            [0, padding],
            [1, 1],
        )
        .map_err(AutodiffError::Backend)?;
        pool2d_out_shape(&[n, c, 1, l], &params_1d).map_err(AutodiffError::Shape)?;

        let x4 = self.contiguous()?.reshape(&[n, c, 1, l])?;
        let out4 = x4.avg_pool2d(
            [1, kernel_size],
            stride.map(|s| [1, s]),
            [0, padding],
            ceil_mode,
            count_include_pad,
        )?;
        let out_shape4 = out4.shape();
        let lout = out_shape4[3];
        out4.reshape(&[n, c, lout])
    }

    /// 2 次元 adaptive average pooling（`torch.nn.functional.
    /// adaptive_avg_pool2d` 相当。NCHW 固定。イシュー #1728・設計
    /// `docs/pooling-ops-design.md`）。`self`（`input`）:
    /// `[N, C, H, W]`・`output_size: [Hout, Wout]`。
    ///
    /// 検査順序: ①[`adaptive_pool2d_out_shape`]（rank・空間軸ゼロ
    /// 拒否・`output_size >= 1`）→ ②実体化 → ③
    /// `adaptive_avg_pool2d_with_fallback` → ④戻り shape 再検証 →
    /// ⑤`push_eager`。
    pub fn adaptive_avg_pool2d(&self, output_size: [usize; 2]) -> Result<Var<'t>, AutodiffError> {
        let in_shape = self.shape();
        let out_shape =
            adaptive_pool2d_out_shape(&in_shape, output_size).map_err(AutodiffError::Shape)?;

        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };

        let value_out = adaptive_avg_pool2d_with_fallback(
            self.tape.ops(),
            &input_val,
            output_size,
            &out_shape,
        )?;
        if value_out.shape() != out_shape {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: value_out.shape().to_vec(),
                    rhs: out_shape,
                },
            )));
        }
        let id = self
            .tape
            .push_eager(Op::AdaptiveAvgPool2d { input: self.id }, value_out);
        Ok(Var::from_raw(self.tape, id))
    }

    /// 1 次元 adaptive average pooling。[`Self::adaptive_avg_pool2d`]
    /// を `H` 軸固定（`output_size[0]=1`）で呼び出す reshape 併合の
    /// 薄いラッパー（イシュー #1728）。`self`（`input`）:
    /// `[N, C, L]`。
    pub fn adaptive_avg_pool1d(&self, output_size: usize) -> Result<Var<'t>, AutodiffError> {
        let in_shape = self.shape();
        if in_shape.len() != 3 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 3,
                actual: in_shape.len(),
            }));
        }
        let (n, c, l) = (in_shape[0], in_shape[1], in_shape[2]);
        // 検査を reshape より前に行う（`Var::conv1d` と同じ規律。
        // `adaptive_avg_pool2d` 側の再検査はフェイルクローズドの
        // 二重化として残す）。
        adaptive_pool2d_out_shape(&[n, c, 1, l], [1, output_size]).map_err(AutodiffError::Shape)?;

        let x4 = self.contiguous()?.reshape(&[n, c, 1, l])?;
        let out4 = x4.adaptive_avg_pool2d([1, output_size])?;
        let out_shape4 = out4.shape();
        let lout = out_shape4[3];
        out4.reshape(&[n, c, lout])
    }

    /// 指定した各長さ（`sizes`）で `dim` 軸を分割する（`torch.split`
    /// の list 形式相当。イシュー #1598）。各出力は [`Self::narrow`]
    /// （zero-copy view）。
    ///
    /// `sizes.iter().sum()`（`checked_add`）が `shape[dim]` と一致しな
    /// い場合 `ShapeError::ShapeMismatch { lhs: [shape[dim]], rhs:
    /// [sum] }` を返す。
    pub fn split_with_sizes(
        &self,
        sizes: &[usize],
        dim: usize,
    ) -> Result<Vec<Var<'t>>, AutodiffError> {
        let in_shape = self.shape();
        let rank = in_shape.len();
        if dim >= rank {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: dim,
                rank,
            }));
        }
        let dim_size = in_shape[dim];
        let mut sum: usize = 0;
        for &s in sizes {
            sum = sum
                .checked_add(s)
                .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
        }
        if sum != dim_size {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: vec![dim_size],
                rhs: vec![sum],
            }));
        }
        let mut out = Vec::with_capacity(sizes.len());
        let mut start = 0usize;
        for &len in sizes {
            out.push(self.narrow(dim, start, len)?);
            start += len;
        }
        Ok(out)
    }

    /// 先頭から `split_size` 刻みで `dim` 軸を分割する（`torch.split`
    /// の int 形式相当。末尾は端数。イシュー #1598）。
    ///
    /// `split_size == 0` は `shape[dim]` の値に依らず一律
    /// `AutodiffError::InvalidArgument`（PyTorch は `shape[dim] == 0`
    /// のとき許容するが、0 除算相当の分岐を持たない単純な契約を
    /// 優先する）。`shape[dim] == 0` かつ `split_size > 0` は PyTorch
    /// と同じく `[narrow(dim, 0, 0)]` の 1 要素を返す。
    pub fn split(&self, split_size: usize, dim: usize) -> Result<Vec<Var<'t>>, AutodiffError> {
        let in_shape = self.shape();
        let rank = in_shape.len();
        if dim >= rank {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: dim,
                rank,
            }));
        }
        if split_size == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Var::split: split_size must be nonzero".into(),
            ));
        }
        let dim_size = in_shape[dim];
        let mut sizes = Vec::new();
        if dim_size == 0 {
            sizes.push(0);
        } else {
            let mut remaining = dim_size;
            while remaining > 0 {
                let take = remaining.min(split_size);
                sizes.push(take);
                remaining -= take;
            }
        }
        self.split_with_sizes(&sizes, dim)
    }

    /// `dim` 軸を `chunks` 個以下に分割する（`torch.chunk` 相当。
    /// イシュー #1598）。`chunk_size = ceil(shape[dim] / chunks)`
    /// （`div_ceil`）を [`Self::split`] へ委譲する。
    ///
    /// `chunks == 0` は `AutodiffError::InvalidArgument`。
    /// `shape[dim] == 0` は `chunk_size` が 0 になり `split` の
    /// `split_size == 0` 規則と衝突するため、`split` へ委譲せず
    /// PyTorch と同じく `chunks` 個の空 `narrow(dim, 0, 0)` を返す
    /// （[`Self::split_with_sizes`] へ `[0; chunks]` を渡す）。
    pub fn chunk(&self, chunks: usize, dim: usize) -> Result<Vec<Var<'t>>, AutodiffError> {
        if chunks == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Var::chunk: chunks must be nonzero".into(),
            ));
        }
        let in_shape = self.shape();
        let rank = in_shape.len();
        if dim >= rank {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: dim,
                rank,
            }));
        }
        let dim_size = in_shape[dim];
        if dim_size == 0 {
            return self.split_with_sizes(&vec![0; chunks], dim);
        }
        let chunk_size = dim_size.div_ceil(chunks);
        self.split(chunk_size, dim)
    }

    /// 条件テンソル `cond` で `a`／`b` を要素選択する（`torch.where`
    /// 相当。イシュー #1637）。関連関数（`cond` が `Var` ではなく
    /// `&Tensor<bool>` のため。`Var::cat` と同様の位置づけ）。
    /// `#[doc(alias = "where")]` は Rust 予約語 `where` の代替名として
    /// 検索性を確保する目的。
    ///
    /// 出力 shape は `a`・`b`・`cond` **3 入力**の共通 broadcast 形状
    /// とする（`cond` 単独が軸を拡張するケース、例えば `a`／`b` が
    /// `[3]`・`cond` が `[2, 1]` で出力 `[2, 3]` になるケースを含む。
    /// codex-review 指摘・PR #1684）。
    ///
    /// 手順: ①`a.check_same_tape(b)` → ②`out_shape =
    /// broadcast_shape(broadcast_shape(a.shape, b.shape), cond.shape)`
    /// → ③`cond` を `out_shape` へ broadcast（不可なら
    /// `AutodiffError::Shape`）→ ④bool→f32 変換（`out_shape` ちょうど
    /// の contiguous テンソルへ 1 回だけ実体化。
    /// [`fandhe_ai_tensor_core::BackendOps::where_cond`] doc の f32
    /// マスク契約）→ ⑤`a`／`b` を層 1 で実体化（`RefCell` 借用を
    /// 閉じてから `push_eager` を呼ぶ規律。`Var::cat` と同型）→
    /// ⑥`ops.where_cond` → `Unsupported` のときのみホスト参照実装
    /// （`eval::where_cond`）へフォールバック → ⑦戻り shape 検証
    /// （`.claude/rules/security.md` A08）→ ⑧`push_eager`。
    ///
    /// backward（`where_vjp`）は `a`／`b` の勾配を `out_shape` から
    /// 元の `a_shape`／`b_shape`（`cond` を含まない）へ
    /// `reduce_to_shape` で縮約するため、`cond` 由来の拡張軸は
    /// `Op::Mul` 等の一般 broadcast VJP と同じ経路で正しく縮約される。
    #[doc(alias = "where")]
    pub fn where_cond(
        cond: &fandhe_ai_tensor_core::Tensor<bool>,
        a: &Var<'t>,
        b: &Var<'t>,
    ) -> Result<Var<'t>, AutodiffError> {
        a.check_same_tape(b)?;
        let ab_shape = broadcast_shape(&a.shape(), &b.shape()).map_err(AutodiffError::Shape)?;
        let out_shape = broadcast_shape(&ab_shape, cond.shape()).map_err(AutodiffError::Shape)?;
        let cond_bc = cond
            .broadcast_to(&out_shape)
            .map_err(AutodiffError::Shape)?
            .contiguous();
        let cond_mask: Vec<f32> = cond_bc
            .as_slice()
            .map(|s| s.iter().map(|&c| if c { 1.0 } else { 0.0 }).collect())
            .unwrap_or_default();
        let cond_f32 = eval::build_tensor(cond_mask, &out_shape);

        let (a_val, b_val) = {
            let nodes = a.tape.nodes.borrow();
            let ops = a.tape.ops();
            let a_val = materialize_fallible(&nodes, ops, a.id)?.clone();
            let b_val = materialize_fallible(&nodes, ops, b.id)?.clone();
            (a_val, b_val)
        };
        let a_bc = a_val
            .broadcast_to(&out_shape)
            .map_err(AutodiffError::Shape)?;
        let b_bc = b_val
            .broadcast_to(&out_shape)
            .map_err(AutodiffError::Shape)?;

        let value = crate::grad::where_cond_with_fallback(
            a.tape.ops(),
            &cond_f32,
            &a_bc,
            &b_bc,
            &out_shape,
        )?;
        if value.shape() != out_shape {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: value.shape().to_vec(),
                    rhs: out_shape.clone(),
                },
            )));
        }
        let id = a.tape.push_eager(
            Op::Where {
                cond: cond_f32,
                a: a.id,
                b: b.id,
            },
            value,
        );
        Ok(Var::from_raw(a.tape, id))
    }

    /// `mask` が真の位置を定数 `value` で置換する（`torch.masked_fill`
    /// 相当。イシュー #1637）。メソッド（`self` を書き換えず新しい
    /// `Var` を返す）。`mask` は `self` の shape へ broadcast 可能で
    /// あること（`Var::where_cond` と同じ bool→f32 変換規律）。
    pub fn masked_fill(
        &self,
        mask: &fandhe_ai_tensor_core::Tensor<bool>,
        value: f32,
    ) -> Result<Var<'t>, AutodiffError> {
        let out_shape = self.shape();
        let mask_bc = mask
            .broadcast_to(&out_shape)
            .map_err(AutodiffError::Shape)?
            .contiguous();
        let mask_data: Vec<f32> = mask_bc
            .as_slice()
            .map(|s| s.iter().map(|&m| if m { 1.0 } else { 0.0 }).collect())
            .unwrap_or_default();
        let mask_f32 = eval::build_tensor(mask_data, &out_shape);

        let x_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value_out =
            crate::grad::masked_fill_with_fallback(self.tape.ops(), &x_val, &mask_f32, value)?;
        if value_out.shape() != out_shape {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: value_out.shape().to_vec(),
                    rhs: out_shape,
                },
            )));
        }
        let id = self.tape.push_eager(
            Op::MaskedFill {
                input: self.id,
                mask: mask_f32,
            },
            value_out,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// dropout（`torch.nn.functional.dropout(input, p, training)`
    /// 相当。イシュー #1603）。要素ごと独立に確率 `p` で 0 へ落とし、
    /// 残す要素は `1/(1-p)` 倍する（inverted dropout。学習時と推論時で
    /// 期待値のスケールを揃える PyTorch と同じ規約）。
    ///
    /// # 早期リターン（`Op` を記録しない・RNG を消費しない）
    ///
    /// `training == false` または `p == 0.0` のときは `self` をそのまま
    /// 返す（`at::native::dropout` の `if (p == 0 || !train) return
    /// input;` と同じ。`crate::tensor_core::rng::with_global_rng` を
    /// 一切呼ばないため、eval モード下ではグローバル RNG の状態が
    /// dropout 呼び出しの有無で変化しない）。
    ///
    /// # マスク生成（イシュー #1602 のグローバル RNG 契約）
    ///
    /// `crate::grad::dropout_mask` が [`fandhe_ai_tensor_core::rng::
    /// rand`] を経由してホスト側だけでマスクを生成する（`BackendOps`
    /// を経由しない。`docs/rng-global-contract-design.md` §3.2）。
    ///
    /// # エラー
    ///
    /// `p` が非有限、または `[0, 1]` の範囲外の場合
    /// [`AutodiffError::InvalidArgument`]。
    pub fn dropout(&self, p: f32, training: bool) -> Result<Var<'t>, AutodiffError> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Var::dropout: p must be finite and in [0, 1], got {p}"
            )));
        }
        if !training || p == 0.0 {
            // 早期リターン: 新しいノードを積まず `self` の clone を
            // 返す（`self.id` をそのまま指す `Var` を作るだけで、
            // `Op` の記録もグローバル RNG の消費も一切行わない）。
            return Ok(Var::from_raw(self.tape, self.id));
        }
        let mask = crate::grad::dropout_mask(&self.shape(), p)?;
        self.dropout_with_mask(mask)
    }

    /// [`Self::dropout`] の内部本体（マスク固定入口。`pub(crate)`）。
    /// 現時点では [`Self::dropout`] が検査・早期リターン判定・マスク
    /// 生成の後に本メソッドへ委譲するだけの薄い入口であり、
    /// forward（マスク乗算）と backward（VJP）を `mask` の生成経路
    /// から分離する内部構造上の役割にとどまる（実際の VJP 解析解
    /// 検証は `crates/autodiff/tests/nn_dropout.rs` が `Var::mul` を
    /// 直接組み立てる別経路で行っており、本メソッドは経由しない）。
    /// `mask` を外部注入可能な形に分離してあるため、将来グローバル
    /// RNG に触れず固定マスクで forward／backward を検証するテストを
    /// 追加する余地として残している。
    pub(crate) fn dropout_with_mask(
        &self,
        mask: fandhe_ai_tensor_core::Tensor<f32>,
    ) -> Result<Var<'t>, AutodiffError> {
        let out_shape = self.shape();
        let x_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value_out = crate::grad::dropout_with_fallback(self.tape.ops(), &x_val, &mask)?;
        if value_out.shape() != out_shape {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: value_out.shape().to_vec(),
                    rhs: out_shape,
                },
            )));
        }
        let id = self.tape.push_eager(
            Op::Dropout {
                input: self.id,
                mask,
            },
            value_out,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// `dim` 軸に沿って `index` が指す要素を独立に読み出す
    /// （`torch.gather` 相当。イシュー #1776）。`index`（`Tensor<i32>`）
    /// は非追跡データ（`Var::cross_entropy_loss` の `targets` と同じ
    /// 「`Op` payload に直接埋め込む」設計。勾配は `index` 側には
    /// 流れない）。
    ///
    /// 検査順序: ①[`fandhe_ai_tensor_core::gather_out_shape`]（`dim`
    /// 範囲・`index` の rank・`dim` 以外の軸一致を検査し `out_shape`
    /// を確定。違反は `AutodiffError::Shape`）→ ②`index` 全添字が
    /// `0 <= idx < self.shape()[dim]`（違反は
    /// `AutodiffError::InvalidArgument`。`cross_entropy_loss` の
    /// targets 範囲検査と同じパターン）→ ③`index` を実体化
    /// （`contiguous()`。`Op::Where` の `cond` と同じ「1 回だけ
    /// 実体化し以後再計算しない」方針）→ ④`self` を層 1 で実体化 →
    /// ⑤`ops.gather` → `Unsupported` のときのみホスト参照実装
    /// （`eval::gather`）へフォールバック → ⑥戻り shape 検証
    /// （`.claude/rules/security.md` A08）→ ⑦`push_eager`。
    pub fn gather(&self, dim: usize, index: &Tensor<i32>) -> Result<Var<'t>, AutodiffError> {
        let in_shape = self.shape();
        let out_shape =
            gather_out_shape(&in_shape, index.shape(), dim).map_err(AutodiffError::Shape)?;

        // `gather_out_shape` が成功した時点で `dim < in_shape.len()` が
        // 保証されるため、この添字アクセスは安全（`.claude/rules/
        // coding-rust.md` REQ-8「境界検査を省略しない」の趣旨）。
        let dim_size = in_shape[dim];
        for v in eval::dense_vec_i32(index) {
            if v < 0 || (v as usize) >= dim_size {
                return Err(AutodiffError::InvalidArgument(format!(
                    "Var::gather: index 添字 {v} が範囲 [0, {dim_size}) を外れている"
                )));
            }
        }

        let index_c = index.contiguous();

        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };

        let value = gather_with_fallback(self.tape.ops(), &input_val, dim, &index_c, &out_shape)?;
        if value.shape() != out_shape {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: value.shape().to_vec(),
                    rhs: out_shape,
                },
            )));
        }
        let id = self.tape.push_eager(
            Op::Gather {
                input: self.id,
                dim,
                index: index_c,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 空間軸（末尾 `size.len()` 軸）を `size` へリサンプリングする
    /// （`torch.nn.functional.interpolate`／`tf.image.resize` 相当。
    /// イシュー #1757）。先頭の残り軸（batch／channel 等）は素通し。
    /// `mode` の意味論・数値契約は
    /// [`fandhe_ai_tensor_core::InterpolateMode`] doc を正とする
    /// （現状 [`fandhe_ai_tensor_core::InterpolateMode::Nearest`] の
    /// み。算術を含まない純粋なコピー演算のため forward は 3
    /// バックエンド間で構造的に bit 完全一致する）。
    ///
    /// 検査順序: ①[`fandhe_ai_tensor_core::interpolate_out_shape_for_mode`]
    /// （`size` の rank・空間軸の 0 サイズ・要素数オーバーフローを
    /// 検査し `out_shape` を確定。違反は `AutodiffError::Shape`）→
    /// ②`self` を層 1 で実体化 → ③`ops.interpolate` →
    /// `Unsupported` のときのみホスト参照実装
    /// （`eval::interpolate_nearest`）へフォールバック（それ以外の
    /// エラーは伝播する。判定迂回経路を作らない）→ ④戻り shape 検証
    /// （`.claude/rules/security.md` A08）→ ⑤`push_eager`。
    pub fn interpolate(
        &self,
        size: &[usize],
        mode: InterpolateMode,
    ) -> Result<Var<'t>, AutodiffError> {
        let in_shape = self.shape();
        let out_shape =
            interpolate_out_shape_for_mode(&in_shape, size, mode).map_err(AutodiffError::Shape)?;

        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };

        let value = interpolate_with_fallback(self.tape.ops(), &input_val, size, mode, &out_shape)?;
        if value.shape() != out_shape {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: value.shape().to_vec(),
                    rhs: out_shape,
                },
            )));
        }
        let id = self.tape.push_eager(
            Op::Interpolate {
                input: self.id,
                size: size.to_vec(),
                mode,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 1 次元 `index` で `dim` 軸を選択する（`torch.index_select`
    /// 相当。イシュー #1776）。専用 `Op` は持たず、`index`（`[k]`）を
    /// `self.shape()` と同 rank の stride-0 view（`reshape` して
    /// `dim` 軸のみ `k`・他軸 `1` にしたあと `broadcast_to` で
    /// `self.shape()` の `dim` 軸だけ `k` に置き換えた形へ拡張。
    /// いずれも zero-copy）へ拡張してから [`Self::gather`] へ委譲する
    /// （検査・VJP・Op 構築を 1 箇所に集約し重複を避ける設計判断。
    /// 実装計画「設計判断」§1 参照）。`index` は rank 1 でなければ
    /// `AutodiffError::Shape(ShapeError::RankMismatch)`。
    pub fn index_select(&self, dim: usize, index: &Tensor<i32>) -> Result<Var<'t>, AutodiffError> {
        let index_rank = index.shape().len();
        if index_rank != 1 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 1,
                actual: index_rank,
            }));
        }
        let in_shape = self.shape();
        let rank = in_shape.len();
        if dim >= rank {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: dim,
                rank,
            }));
        }
        let k = index.shape()[0];
        let mut expand_shape = vec![1usize; rank];
        expand_shape[dim] = k;
        // `reshape` は contiguous なテンソルしか受け付けない
        // （`ShapeError::NonContiguousReshape`）。`index`（strided な
        // 1 次元 view。例: 別の `narrow` の結果）を直接 `reshape` する
        // と `Var::gather` 側（`index.contiguous()` を経由する）とは
        // 非対称に失敗しうるため、ここでも先に実体化する（Bugbot
        // 指摘。イシュー #1776）。
        let index_c = index.contiguous();
        let index_reshaped = index_c
            .reshape(&expand_shape)
            .map_err(AutodiffError::Shape)?;
        let mut out_shape = in_shape;
        out_shape[dim] = k;
        let index_bc = index_reshaped
            .broadcast_to(&out_shape)
            .map_err(AutodiffError::Shape)?;
        self.gather(dim, &index_bc)
    }

    /// embedding テーブル（`self`。`[num_embeddings, embedding_dim]`）
    /// から `index` が指す行を抽出する（`nn::Embedding` の forward
    /// 本体。`torch.nn.functional.embedding` 相当。イシュー #1604）。
    /// `index` は非追跡データ（[`Self::gather`] の `index` と同じ
    /// 「`Op` payload に直接埋め込む」設計。勾配は `index` 側には
    /// 流れない）で、任意 rank（0 次元＝単一 id も含む）を受理する。
    /// `padding_idx`（`Some(p)`）を渡すと forward は行 `p` の現在値を
    /// そのまま返すが、勾配は本メソッドの `Op::Embedding` VJP（
    /// `grad.rs`）が行 `p` をゼロ上書きするため流れない（PyTorch
    /// `nn.Embedding(padding_idx=..)` の意味論）。
    ///
    /// 検査順序: ①`self.shape().len() == 2`（違反は
    /// `AutodiffError::Shape(ShapeError::RankMismatch)`）→
    /// ②`padding_idx < num_embeddings`（違反は
    /// `AutodiffError::InvalidArgument`）→ ③`index` 全添字が
    /// `0 <= id < num_embeddings`（違反は `AutodiffError::
    /// InvalidArgument`。[`Self::gather`] と同じ「バックエンド呼び出し
    /// 前に検査する」契約。`.claude/rules/security.md` A03）→ ④`index`
    /// を `[N, D]`（`N = index.numel()`）へ実体化（`contiguous()` →
    /// `reshape([N, 1])` → `broadcast_to([N, D])` → `contiguous()`。
    /// いずれも [`Self::index_select`] と同じ zero-copy な view 拡張
    /// 手順の最終段のみ実体化する）→ ⑤`self`（weight）を層 1
    /// （`materialize_fallible`）で実体化 → ⑥`gather_with_fallback`
    /// → ⑦戻り shape 検証 → ⑧`push_eager` →
    /// ⑨`index.rank() != 1` のときのみ、ノード shape（常に `[N, D]`）を
    /// `index.shape() ++ [D]` へ [`Self::reshape`] する（1-D ids は
    /// `[N] ++ [D] == [N, D]` で既に一致するため省略。rank-0 ids は
    /// `N == 1` から `[1, D]` を `[D]` へ縮める）。
    pub fn embedding(
        &self,
        index: &Tensor<i32>,
        padding_idx: Option<usize>,
    ) -> Result<Var<'t>, AutodiffError> {
        let weight_shape = self.shape();
        if weight_shape.len() != 2 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: weight_shape.len(),
            }));
        }
        let num_embeddings = weight_shape[0];
        let embedding_dim = weight_shape[1];

        if let Some(p) = padding_idx
            && p >= num_embeddings
        {
            return Err(AutodiffError::InvalidArgument(format!(
                "Var::embedding: padding_idx {p} が範囲 [0, {num_embeddings}) を外れている"
            )));
        }

        for v in eval::dense_vec_i32(index) {
            if v < 0 || (v as usize) >= num_embeddings {
                return Err(AutodiffError::InvalidArgument(format!(
                    "Var::embedding: index 添字 {v} が範囲 [0, {num_embeddings}) を外れている"
                )));
            }
        }

        let ids_rank = index.rank();
        let n = index.numel();
        let index_c = index.contiguous();
        let index_2d = index_c.reshape(&[n, 1]).map_err(AutodiffError::Shape)?;
        let index_nd = index_2d
            .broadcast_to(&[n, embedding_dim])
            .map_err(AutodiffError::Shape)?
            .contiguous();

        let weight_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };

        let out_shape = vec![n, embedding_dim];
        let value = gather_with_fallback(self.tape.ops(), &weight_val, 0, &index_nd, &out_shape)?;
        if value.shape() != out_shape.as_slice() {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: value.shape().to_vec(),
                    rhs: out_shape,
                },
            )));
        }
        let id = self.tape.push_eager(
            Op::Embedding {
                weight: self.id,
                index: index_nd,
                padding_idx,
            },
            value,
        );
        let out = Var::from_raw(self.tape, id);
        if ids_rank == 1 {
            Ok(out)
        } else {
            let mut final_shape = index.shape().to_vec();
            final_shape.push(embedding_dim);
            out.reshape(&final_shape)
        }
    }

    /// 一意値集合を返す（`torch.unique(input, sorted=True)` の values
    /// のみ。イシュー #1734・契約は
    /// [`fandhe_ai_tensor_core::BackendOps::unique`] doc を正とする）。
    ///
    /// **非微分演算**: unique は勾配を持たない（出力の各要素がどの
    /// 入力位置に由来するかは一意に定まらず、出力形状も入力値に
    /// 依存して動的に決まるため既存の `Var`〈tape ノード・静的
    /// shape〉には乗らない）。そのため本メソッドは新規 `Op` を tape に
    /// 記録せず（`push_eager` を呼ばない）、`self` を
    /// `materialize_fallible`（クレート内部ヘルパー）で実体化した値に
    /// 対して `unique_with_fallback` を適用した **detached な
    /// `Tensor<f32>`** を返す（`Var` ではない。
    /// `docs/unique-facade-exposure-decision.md` 参照）。
    pub fn unique(&self) -> Result<Tensor<f32>, AutodiffError> {
        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        unique_with_fallback(self.tape.ops(), &input_val)
    }

    /// `self` を `Tensor<T>` へ変換する（`torch.Tensor.to(dtype)` 相当
    /// の一方向。イシュー #1750。dtype 変換基盤の契約・数値表は
    /// `docs/tensor-core-cast-design.md` を正とする）。
    ///
    /// **非微分演算**: 出力が f32 以外の dtype への cast は勾配を
    /// 持たない（`Var::argmax`／`Var::unique` と同型の理由——`Tensor<T>`
    /// は `Var` の tape 表現に乗らない）ため、本メソッドは新規 `Op` を
    /// tape に記録せず（`push_eager` を呼ばない）、`self` を
    /// `materialize_fallible` で実体化した値に対して
    /// `cast_from_f32_with_fallback` を適用した **detached な
    /// `Tensor<T>`** を返す。
    ///
    /// `T = f32` を指定した場合も本メソッドは detached なコピーを
    /// 返すだけであり勾配は伝播しない——勾配を保ったまま f32 系の
    /// 恒等射を得たい場合は [`Var::to_f32`] を使うこと。
    pub fn cast<T: CastElement>(&self) -> Result<Tensor<T>, AutodiffError> {
        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        cast_from_f32_with_fallback(self.tape.ops(), &input_val)
    }

    /// f32 系の恒等射（イシュー #1750）。`self` は既に `Tensor<f32>`
    /// を表す `Var` であるため、`Var: Copy` により `self` をそのまま
    /// 返すだけで済み、勾配は通常どおり `self` の tape ノードへ伝播
    /// する（[`Var::cast`]`::<f32>()` が detached なコピーを返し勾配を
    /// 打ち切るのとは対照的）。
    pub fn to_f32(&self) -> Var<'t> {
        *self
    }

    /// `mask` が真の位置を上書きする（`torch.scatter` 相当。イシュー
    /// #1776）。共通実装 `Self::scatter_impl`（非公開）を
    /// [`fandhe_ai_tensor_core::ScatterReduce::Overwrite`] で呼ぶ。
    pub fn scatter(
        &self,
        dim: usize,
        index: &Tensor<i32>,
        src: &Var<'t>,
    ) -> Result<Var<'t>, AutodiffError> {
        self.scatter_impl(dim, index, src, ScatterReduce::Overwrite)
    }

    /// `index` が指す位置へ `src` を加算する（`torch.scatter_add`
    /// 相当。イシュー #1776）。共通実装 `Self::scatter_impl`（非公開）を
    /// [`fandhe_ai_tensor_core::ScatterReduce::Add`] で呼ぶ
    /// （決定的集約順序・`f64` アキュムレータ契約は
    /// [`fandhe_ai_tensor_core::ScatterReduce`] doc を正とする）。
    pub fn scatter_add(
        &self,
        dim: usize,
        index: &Tensor<i32>,
        src: &Var<'t>,
    ) -> Result<Var<'t>, AutodiffError> {
        self.scatter_impl(dim, index, src, ScatterReduce::Add)
    }

    /// [`Self::scatter`]／[`Self::scatter_add`] の共通実装（イシュー
    /// #1776）。`index`（`Tensor<i32>`）は [`Self::gather`] と同じ
    /// 非追跡データ、`src` は追跡対象（`self` と同一 `Tape` 上の
    /// `Var` であること・`check_same_tape` で検査）。
    ///
    /// 検査順序: ①`check_same_tape`（`AutodiffError::TapeMismatch`）
    /// → ②[`fandhe_ai_tensor_core::scatter_out_shape`]（`dim` 範囲・
    /// `index` の rank・`index.shape() == src.shape()`・`dim` 以外の
    /// 軸で `index.shape()[axis] <= self.shape()[axis]` を検査。
    /// 違反は `AutodiffError::Shape`）→ ③`index` 全添字が
    /// `0 <= idx < self.shape()[dim]`（違反は
    /// `AutodiffError::InvalidArgument`）→ ④`index` を実体化
    /// （`contiguous()`）→ ⑤`self`／`src` を層 1 で実体化 →
    /// ⑥`ops.scatter` → `Unsupported` のときのみホスト参照実装
    /// （`eval::scatter`。決定的集約契約に厳密に従う）へフォール
    /// バック → ⑦戻り shape 検証 → ⑧`push_eager`。
    fn scatter_impl(
        &self,
        dim: usize,
        index: &Tensor<i32>,
        src: &Var<'t>,
        reduce: ScatterReduce,
    ) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(src)?;
        let in_shape = self.shape();
        let src_shape = src.shape();
        let out_shape = scatter_out_shape(&in_shape, index.shape(), &src_shape, dim)
            .map_err(AutodiffError::Shape)?;

        // `scatter_out_shape` が成功した時点で `dim < in_shape.len()`
        // が保証されるため、この添字アクセスは安全。
        let dim_size = in_shape[dim];
        for v in eval::dense_vec_i32(index) {
            if v < 0 || (v as usize) >= dim_size {
                return Err(AutodiffError::InvalidArgument(format!(
                    "Var::scatter: index 添字 {v} が範囲 [0, {dim_size}) を外れている"
                )));
            }
        }

        let index_c = index.contiguous();

        let (input_val, src_val) = {
            let nodes = self.tape.nodes.borrow();
            let ops = self.tape.ops();
            let input_val = materialize_fallible(&nodes, ops, self.id)?.clone();
            let src_val = materialize_fallible(&nodes, ops, src.id)?.clone();
            (input_val, src_val)
        };

        let value = scatter_with_fallback(
            self.tape.ops(),
            &input_val,
            dim,
            &index_c,
            &src_val,
            reduce,
            &out_shape,
        )?;
        if value.shape() != out_shape {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: value.shape().to_vec(),
                    rhs: out_shape,
                },
            )));
        }
        let id = self.tape.push_eager(
            Op::Scatter {
                input: self.id,
                dim,
                index: index_c,
                src: src.id,
                reduce,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// `dim` 軸に沿って並べ替える（`torch.sort` 相当。イシュー
    /// #1733）。戻り値は `(values, index)` で、`values`（追跡対象。
    /// `Var`）は並べ替え後の値、`index`（非追跡・`Tensor<i32>`）は
    /// 元の `dim` 軸上の添字。同値（ties）・NaN・±0 の順序契約は
    /// [`fandhe_ai_tensor_core::BackendOps::sort`] doc の 1〜4 を正と
    /// する。
    ///
    /// 検査順序: ①[`fandhe_ai_tensor_core::sort_out_shape`]（`dim`
    /// 範囲を検査。違反は `AutodiffError::Shape`）→ ②`self` を層 1 で
    /// 実体化 → ③`ops.sort` → `Unsupported` のときのみホスト参照
    /// 実装（`eval::sort`）へフォールバック → ④戻り shape 検証
    /// （`.claude/rules/security.md` A08）→ ⑤`push_eager`。
    pub fn sort(
        &self,
        dim: usize,
        descending: bool,
    ) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
        let in_shape = self.shape();
        let out_shape = sort_out_shape(&in_shape, dim).map_err(AutodiffError::Shape)?;

        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };

        let (value, index) =
            sort_with_fallback(self.tape.ops(), &input_val, dim, descending, &out_shape)?;
        if value.shape() != out_shape || index.shape() != out_shape {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: value.shape().to_vec(),
                    rhs: out_shape,
                },
            )));
        }
        let id = self.tape.push_eager(
            Op::Sort {
                input: self.id,
                dim,
                index: index.clone(),
            },
            value,
        );
        Ok((Var::from_raw(self.tape, id), index))
    }

    /// `dim` 軸に沿って並べ替えた際の元添字のみを返す（`torch.argsort`
    /// 相当。イシュー #1733）。[`Self::sort`] と異なりテープへノードを
    /// **追加しない**（非微分演算。`index` は forward 値のみで勾配は
    /// 流れない）。`self` を実体化して [`Self::sort`] と同じ検査・
    /// フォールバック経路（`sort_out_shape` → `sort_with_fallback`）を
    /// 通し、`index` 出力のみを返す。
    pub fn argsort(&self, dim: usize, descending: bool) -> Result<Tensor<i32>, AutodiffError> {
        let in_shape = self.shape();
        let out_shape = sort_out_shape(&in_shape, dim).map_err(AutodiffError::Shape)?;

        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };

        let (_value, index) =
            sort_with_fallback(self.tape.ops(), &input_val, dim, descending, &out_shape)?;
        if index.shape() != out_shape {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: index.shape().to_vec(),
                    rhs: out_shape,
                },
            )));
        }
        Ok(index)
    }

    /// `dim` 軸に沿って上位（`largest=true`）／下位（`largest=false`）
    /// `k` 個を選ぶ（`torch.topk` 相当・`sorted=True` 固定。イシュー
    /// #1733）。戻り値は [`Self::sort`] と同じ `(values, index)` の形。
    ///
    /// 検査順序: ①[`fandhe_ai_tensor_core::topk_out_shape`]（`dim`
    /// 範囲・`k <= self.shape()[dim]` を検査。`k > dim_size` は
    /// `ShapeError::NarrowOutOfBounds`）→ ②`self` を層 1 で実体化 →
    /// ③`ops.topk` → `Unsupported` のときのみホスト参照実装
    /// （`eval::topk`）へフォールバック → ④戻り shape 検証 →
    /// ⑤`push_eager`。
    pub fn topk(
        &self,
        k: usize,
        dim: usize,
        largest: bool,
    ) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
        let in_shape = self.shape();
        let out_shape = topk_out_shape(&in_shape, dim, k).map_err(AutodiffError::Shape)?;

        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };

        let (value, index) =
            topk_with_fallback(self.tape.ops(), &input_val, dim, k, largest, &out_shape)?;
        if value.shape() != out_shape || index.shape() != out_shape {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: value.shape().to_vec(),
                    rhs: out_shape,
                },
            )));
        }
        let id = self.tape.push_eager(
            Op::Topk {
                input: self.id,
                dim,
                index: index.clone(),
            },
            value,
        );
        Ok((Var::from_raw(self.tape, id), index))
    }

    /// クラス id 列を one-hot 行列へ展開する（`torch.nn.functional.
    /// one_hot`／`tf.one_hot` 相当。**非微分演算**。イシュー #1755）。
    /// `self` はクラス id を f32 値として保持する追跡 `Var`（専用の
    /// integer `Var` 型は無いため、整数値を f32 で表現する既存方針を
    /// 踏襲。`Var::gather` の `index` 引数と異なり、本メソッドは
    /// `self` 自身がクラス id を保持する——「勾配計算グラフ上の入力」
    /// であることを明示するため `Op::OneHot` に `input: NodeId` として
    /// 記録し、VJP でゼロ勾配を流す対象にする）。出力 shape は
    /// `self.shape() ++ [num_classes]`
    /// （[`fandhe_ai_tensor_core::one_hot_out_shape`]）。
    ///
    /// 検査順序: ①`num_classes >= 1`（違反は `AutodiffError::
    /// InvalidArgument`。`num_classes == 0` 自体は
    /// `one_hot_out_shape` が `ShapeError::IndexOutOfRange` を返すが、
    /// 専用のわかりやすいメッセージを先に出す）→ ②
    /// [`fandhe_ai_tensor_core::one_hot_out_shape`] で出力 shape を
    /// 確定（違反は `AutodiffError::Shape`）→ ③`self` を層 1 実体化
    /// → ④全要素が有限・整数値（`fract() == 0.0`）・
    /// `0 <= v < num_classes` であることを検査（違反は
    /// `InvalidArgument`。[`Self::gather`] の `index` 範囲検査と同じ
    /// パターン）→ ⑤検証済み値を `Tensor<i32>`（`self.shape()` と同一
    /// shape の contiguous）へ変換 → ⑥`one_hot_with_fallback`
    /// （`Unsupported` のときのみホスト参照実装 `eval::one_hot` へ
    /// フォールバック）→ ⑦戻り shape 再検証（`.claude/rules/
    /// security.md` A08）→ ⑧`push_eager`。
    pub fn one_hot(&self, num_classes: usize) -> Result<Var<'t>, AutodiffError> {
        if num_classes == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Var::one_hot: num_classes は 1 以上でなければならない".into(),
            ));
        }
        let in_shape = self.shape();
        let out_shape = one_hot_out_shape(&in_shape, num_classes).map_err(AutodiffError::Shape)?;

        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };

        let raw = eval::dense_vec(&input_val);
        let mut index_data = Vec::with_capacity(raw.len());
        for v in raw {
            if !v.is_finite() {
                return Err(AutodiffError::InvalidArgument(format!(
                    "Var::one_hot: 非有限値 {v} はクラス id として使えない"
                )));
            }
            if v.fract() != 0.0 {
                return Err(AutodiffError::InvalidArgument(format!(
                    "Var::one_hot: 非整数値 {v} はクラス id として使えない"
                )));
            }
            // `v < 0.0` を先に検査するため、`v as usize`（f32 → usize
            // の `as` キャストは Rust 1.45 以降 saturating で UB は
            // 起きない）の評価に到達するのは非負値のみ（`Var::gather`
            // の index 範囲検査と同じ短絡順序）。
            if v < 0.0 || (v as usize) >= num_classes {
                return Err(AutodiffError::InvalidArgument(format!(
                    "Var::one_hot: クラス id {v} が範囲 [0, {num_classes}) を外れている"
                )));
            }
            // `v as i32` は `v` が `i32::MAX` を超える場合 saturating
            // キャストで `i32::MAX` へ丸まり（Rust 1.45 以降 UB は
            // 起きないが）値を破壊する。直前の範囲検査（`v < num_classes`）
            // は `num_classes: usize` が `i32::MAX` を超えうる（`usize`
            // は 64bit）ため、この破壊を防げない（例: `num_classes =
            // 4_000_000_000`・`v = 3_500_000_000.0` は範囲検査を通過
            // するが `i32` へは収まらない）。`index` テンソルの要素型が
            // `i32`（`gather`／`scatter` と共有する index 表現）である
            // 契約上、表現不能な値はキャスト前に明示的に拒否する
            // （codex-review 指摘。イシュー #1755）。
            //
            // `v > i32::MAX as f32` は誤り: `i32::MAX`（2147483647）は
            // f32（23bit 仮数部）で正確に表現できず最近接偶数丸めで
            // `2147483648.0`（2^31）へ切り上がる。このため `v ==
            // 2147483648.0` は等号非成立で検査を素通りし、続く
            // `v as i32` が `i32::MAX` へ saturating キャストされて
            // 別クラスを指してしまう（codex-review／Cursor Bugbot 再指摘。
            // イシュー #1755 PR #1827）。`f64`（52bit 仮数部）は
            // `i32::MAX` を含め全 f32 有限値・全 i32 値を正確に表現
            // できるため、比較前に `f64` へ昇格して丸め誤差なく判定する。
            if v as f64 > i32::MAX as f64 {
                return Err(AutodiffError::InvalidArgument(format!(
                    "Var::one_hot: クラス id {v} が i32 で表現できない"
                )));
            }
            index_data.push(v as i32);
        }
        let index = Tensor::new(index_data, &in_shape).map_err(AutodiffError::Shape)?;

        let value = one_hot_with_fallback(self.tape.ops(), &index, num_classes, &out_shape)?;
        if value.shape() != out_shape {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: value.shape().to_vec(),
                    rhs: out_shape,
                },
            )));
        }
        let id = self.tape.push_eager(Op::OneHot { input: self.id }, value);
        Ok(Var::from_raw(self.tape, id))
    }

    /// LSTM セル 1 step（イシュー #1647・設計 `docs/autodiff-rnn-cell-
    /// tape-design.md` 決定 1・1b・1c・4・5・12）。ゲート順は `i,f,g,o`
    /// （`p.w_ih`／`p.w_hh` は `[D, 4H]`／`[H, 4H]`、bias は `[4H]`）。
    /// 2 ノード（`Op::LstmCell` → `Op::LstmHidden`）を **この順で**
    /// push する（決定 1b の push 順序契約。backward の逆走査が
    /// `LstmHidden` を先に処理し `nodes[cell.0].op` を参照するため）。
    ///
    /// 戻り値は `(h_t, c_t)`。
    pub fn lstm_cell(
        &self,
        h_prev: &Var<'t>,
        c_prev: &Var<'t>,
        p: GateParams<'_, 't>,
    ) -> Result<(Var<'t>, Var<'t>), AutodiffError> {
        self.check_same_tape(h_prev)?;
        self.check_same_tape(c_prev)?;
        self.check_same_tape(p.w_ih)?;
        self.check_same_tape(p.w_hh)?;
        if let Some(b) = p.b_ih {
            self.check_same_tape(b)?;
        }
        if let Some(b) = p.b_hh {
            self.check_same_tape(b)?;
        }
        check_bias_pair(p.b_ih, p.b_hh)?;

        let x_shape = self.shape();
        let h_shape = h_prev.shape();
        let c_shape = c_prev.shape();
        let w_ih_shape = p.w_ih.shape();
        let w_hh_shape = p.w_hh.shape();
        require_rank2_all(
            &[&x_shape, &h_shape, &c_shape, &w_ih_shape, &w_hh_shape],
            "lstm_cell",
        )?;
        require_same_shape(&h_shape, &c_shape)?;
        let hidden = h_shape[1];
        require_positive_dims(x_shape[1], hidden, "lstm_cell")?;

        let out_ih = gemm_out_shape(&x_shape, &w_ih_shape)?;
        let out_hh = gemm_out_shape(&h_shape, &w_hh_shape)?;
        require_same_shape(&out_ih, &out_hh)?;
        let gate_width_4h = checked_gate_width(4, hidden)?;
        if out_ih[1] != gate_width_4h {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: out_ih,
                rhs: vec![h_shape[0], gate_width_4h],
            }));
        }
        if let Some(b) = p.b_ih {
            require_same_shape(&b.shape(), &[gate_width_4h])?;
        }
        if let Some(b) = p.b_hh {
            require_same_shape(&b.shape(), &[gate_width_4h])?;
        }

        let (x_val, h_prev_val, c_prev_val, w_ih_val, w_hh_val, b_ih_val, b_hh_val) = {
            let nodes = self.tape.nodes.borrow();
            let ops = self.tape.ops();
            let x_val = materialize_fallible(&nodes, ops, self.id)?.clone();
            let h_prev_val = materialize_fallible(&nodes, ops, h_prev.id)?.clone();
            let c_prev_val = materialize_fallible(&nodes, ops, c_prev.id)?.clone();
            let w_ih_val = materialize_fallible(&nodes, ops, p.w_ih.id)?.clone();
            let w_hh_val = materialize_fallible(&nodes, ops, p.w_hh.id)?.clone();
            let b_ih_val = optional_materialize(&nodes, ops, p.b_ih)?;
            let b_hh_val = optional_materialize(&nodes, ops, p.b_hh)?;
            (
                x_val, h_prev_val, c_prev_val, w_ih_val, w_hh_val, b_ih_val, b_hh_val,
            )
        };

        let out = lstm_cell_forward_values(
            self.tape.ops(),
            &x_val,
            &h_prev_val,
            &c_prev_val,
            &CellWeights {
                w_ih: &w_ih_val,
                w_hh: &w_hh_val,
                b_ih: b_ih_val.as_ref(),
                b_hh: b_hh_val.as_ref(),
            },
        )?;

        // 決定 1b: `gates` (`[B, 4H]`。列ブロック順 `i,f,g,o`) から
        // `LstmCell` payload（`i,f,g`。`[B, 3H]`）と `LstmHidden`
        // payload（`o`。`[B, H]`）を切り出す。`narrow` は zero-copy
        // view のため `contiguous()` で実体化してから非追跡 payload
        // として保持する（`Op` payload はホスト常駐の独立 `Tensor`
        // でなければならない。view のまま埋め込むと backward 時に
        // `resolve_view` が想定しない経路になる）。
        // `gate_width_4h` が overflow せず検証済み（上記）のため、
        // その内訳である `3 * hidden` も overflow しない
        // （`3 * hidden < 4 * hidden <= usize::MAX`）。
        let gate_width_3h = checked_gate_width(3, hidden)?;
        let gates_ifg = out
            .gates
            .narrow(1, 0, gate_width_3h)
            .map(|t| t.contiguous())?;
        let gate_o = out
            .gates
            .narrow(1, gate_width_3h, hidden)
            .map(|t| t.contiguous())?;

        let cell_id = self.tape.push_eager(
            Op::LstmCell {
                x: self.id,
                h_prev: h_prev.id,
                c_prev: c_prev.id,
                w_ih: p.w_ih.id,
                w_hh: p.w_hh.id,
                b_ih: p.b_ih.map(|b| b.id),
                b_hh: p.b_hh.map(|b| b.id),
                gates_ifg,
            },
            out.c,
        );
        let hidden_id = self.tape.push_eager(
            Op::LstmHidden {
                cell: cell_id,
                gate_o,
            },
            out.h,
        );
        Ok((
            Var::from_raw(self.tape, hidden_id),
            Var::from_raw(self.tape, cell_id),
        ))
    }

    /// GRU セル 1 step（イシュー #1647・設計 `docs/autodiff-rnn-cell-
    /// tape-design.md` 決定 1c・4・5・12。`reset_after=True` 規約）。
    /// ゲート順は `r,z,n`（`p.w_ih`／`p.w_hh` は `[D, 3H]`／`[H, 3H]`、
    /// bias は `[3H]`）。1 ノード（`Op::GruCell`）で表現する（GRU は
    /// LSTM と異なり単一出力 `h_t` のため 2 ノード分割は不要）。
    pub fn gru_cell(
        &self,
        h_prev: &Var<'t>,
        p: GateParams<'_, 't>,
    ) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(h_prev)?;
        self.check_same_tape(p.w_ih)?;
        self.check_same_tape(p.w_hh)?;
        if let Some(b) = p.b_ih {
            self.check_same_tape(b)?;
        }
        if let Some(b) = p.b_hh {
            self.check_same_tape(b)?;
        }
        check_bias_pair(p.b_ih, p.b_hh)?;

        let x_shape = self.shape();
        let h_shape = h_prev.shape();
        let w_ih_shape = p.w_ih.shape();
        let w_hh_shape = p.w_hh.shape();
        require_rank2_all(&[&x_shape, &h_shape, &w_ih_shape, &w_hh_shape], "gru_cell")?;
        let hidden = h_shape[1];
        require_positive_dims(x_shape[1], hidden, "gru_cell")?;

        let out_ih = gemm_out_shape(&x_shape, &w_ih_shape)?;
        let out_hh = gemm_out_shape(&h_shape, &w_hh_shape)?;
        require_same_shape(&out_ih, &out_hh)?;
        let gate_width_3h = checked_gate_width(3, hidden)?;
        if out_ih[1] != gate_width_3h {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: out_ih,
                rhs: vec![h_shape[0], gate_width_3h],
            }));
        }
        if let Some(b) = p.b_ih {
            require_same_shape(&b.shape(), &[gate_width_3h])?;
        }
        if let Some(b) = p.b_hh {
            require_same_shape(&b.shape(), &[gate_width_3h])?;
        }

        let (x_val, h_prev_val, w_ih_val, w_hh_val, b_ih_val, b_hh_val) = {
            let nodes = self.tape.nodes.borrow();
            let ops = self.tape.ops();
            let x_val = materialize_fallible(&nodes, ops, self.id)?.clone();
            let h_prev_val = materialize_fallible(&nodes, ops, h_prev.id)?.clone();
            let w_ih_val = materialize_fallible(&nodes, ops, p.w_ih.id)?.clone();
            let w_hh_val = materialize_fallible(&nodes, ops, p.w_hh.id)?.clone();
            let b_ih_val = optional_materialize(&nodes, ops, p.b_ih)?;
            let b_hh_val = optional_materialize(&nodes, ops, p.b_hh)?;
            (x_val, h_prev_val, w_ih_val, w_hh_val, b_ih_val, b_hh_val)
        };

        let out = gru_cell_forward_values(
            self.tape.ops(),
            &x_val,
            &h_prev_val,
            &CellWeights {
                w_ih: &w_ih_val,
                w_hh: &w_hh_val,
                b_ih: b_ih_val.as_ref(),
                b_hh: b_hh_val.as_ref(),
            },
        )?;

        let id = self.tape.push_eager(
            Op::GruCell {
                x: self.id,
                h_prev: h_prev.id,
                w_ih: p.w_ih.id,
                w_hh: p.w_hh.id,
                b_ih: p.b_ih.map(|b| b.id),
                b_hh: p.b_hh.map(|b| b.id),
                gates_rzn: out.gates,
                q: out.q,
            },
            out.h,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// `A^{-1}`（`A: [n,n]`）。イシュー #1621・親イシュー #1573
    /// 「Tier 2: 線形代数」・`docs/spec/04-requirements.md` REQ-9
    /// 2026-09-12 追記・`docs/autodiff-linalg-design.md`。
    ///
    /// 検査順序（`mse_loss_with` と同じ規律）: ①shape 検査（rank-2・
    /// 正方）→ ②入力実体化（層 1）→ ③`self.tape.ops().linalg_inv` を
    /// 試み `Err(BackendError::Unsupported(_))` のときのみ
    /// `eval::linalg::inv`（ホスト参照実装）へフォールバック（それ以外の
    /// エラー〈特異行列の `InvalidArgument` 等〉は伝播する。判定迂回
    /// 経路を作らない。`.claude/rules/security.md` A08）→ ④ノード記録。
    pub fn inv(&self) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        require_square(&shape, "Var::inv")?;
        let a_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = match self.tape.ops().linalg_inv(&a_val) {
            Ok(v) => {
                verify_shape(v.shape(), &shape)?;
                v
            }
            Err(BackendError::Unsupported(_)) => eval::linalg::inv(&a_val)?,
            Err(other) => return Err(unify_backend_error(other)),
        };
        let id = self.tape.push_eager(Op::Inv { input: self.id }, value);
        Ok(Var::from_raw(self.tape, id))
    }

    /// `A X = B` を解く（`self: [n,n]`・`b: [n,k]` → `[n,k]`）。
    /// イシュー #1621。`inv` と同じ二段フォールバック規律。
    pub fn solve(&self, b: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.check_same_tape(b)?;
        let a_shape = self.shape();
        let n = require_square(&a_shape, "Var::solve")?;
        let b_shape = b.shape();
        if b_shape.len() != 2 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: b_shape.len(),
            }));
        }
        if b_shape[0] != n {
            return Err(AutodiffError::InvalidArgument(format!(
                "Var::solve: a の行数 {n} と b の行数 {} が一致しない",
                b_shape[0]
            )));
        }
        let (a_val, b_val) = {
            let nodes = self.tape.nodes.borrow();
            let a_val = materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone();
            let b_val = materialize_fallible(&nodes, self.tape.ops(), b.id)?.clone();
            (a_val, b_val)
        };
        let expected_shape = vec![n, b_shape[1]];
        let value = match self.tape.ops().linalg_solve(&a_val, &b_val) {
            Ok(v) => {
                verify_shape(v.shape(), &expected_shape)?;
                v
            }
            Err(BackendError::Unsupported(_)) => eval::linalg::solve(&a_val, &b_val)?,
            Err(other) => return Err(unify_backend_error(other)),
        };
        let id = self.tape.push_eager(
            Op::Solve {
                a: self.id,
                b: b.id,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// `det(A)`（`A: [n,n]` → スカラー `[]`）。イシュー #1621。特異行列は
    /// forward で `0.0`（エラーにしない。`eval::linalg::det` doc・
    /// `torch.linalg.det` と同じ挙動）。
    pub fn det(&self) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        require_square(&shape, "Var::det")?;
        let a_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = match self.tape.ops().linalg_det(&a_val) {
            Ok(v) => {
                verify_shape(v.shape(), &[])?;
                v
            }
            Err(BackendError::Unsupported(_)) => eval::linalg::det(&a_val),
            Err(other) => return Err(unify_backend_error(other)),
        };
        let id = self.tape.push_eager(Op::Det { input: self.id }, value);
        Ok(Var::from_raw(self.tape, id))
    }

    /// Cholesky 分解（`A: [n,n]`〈対称正定値。下三角のみ読む〉→
    /// `L: [n,n]`〈下三角、`A = L Lᵀ`〉）。イシュー #1621。非正定値は
    /// `AutodiffError::InvalidArgument(_)`（CPU 本番経路・フォールバック
    /// とも `unify_backend_error` で同一 variant に統一済み。
    /// codex-review 指摘の是正: 以前は逆方向〈`Backend(InvalidArgument)`〉
    /// へ統一していたため、本番経路の数値エラーがドキュメント記載の
    /// variant と一致しなかった）。
    pub fn cholesky(&self) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        require_square(&shape, "Var::cholesky")?;
        let a_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = match self.tape.ops().linalg_cholesky(&a_val) {
            Ok(v) => {
                verify_shape(v.shape(), &shape)?;
                v
            }
            Err(BackendError::Unsupported(_)) => eval::linalg::cholesky(&a_val)?,
            Err(other) => return Err(unify_backend_error(other)),
        };
        let id = self.tape.push_eager(Op::Cholesky { input: self.id }, value);
        Ok(Var::from_raw(self.tape, id))
    }

    /// reduced QR 分解（`A: [m,n]` → [`QrVars`]。`k = min(m,n)`）。
    /// イシュー #1621。
    ///
    /// **多出力の扱い**（`docs/autodiff-linalg-design.md` §3.3）: テープは
    /// 1 ノード 1 出力のため `Q`／`R` を別ノード（`Op::QrQ`／
    /// `Op::QrR`）として積む。各ノードは兄弟ノードの forward 値を
    /// payload として保持し、VJP（`grad.rs`）はコタンジェントに線形な
    /// ことを利用して各出力ノードの部分寄与を返す
    /// （`Tape::backward` が入力ノードへ合算する）。
    pub fn qr(&self) -> Result<QrVars<'t>, AutodiffError> {
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: shape.len(),
            }));
        }
        let (m, n) = (shape[0], shape[1]);
        let k = m.min(n);
        let a_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let (q_val, r_val) = match self.tape.ops().linalg_qr(&a_val) {
            Ok(factors) => {
                verify_shape(factors.q.shape(), &[m, k])?;
                verify_shape(factors.r.shape(), &[k, n])?;
                (factors.q, factors.r)
            }
            Err(BackendError::Unsupported(_)) => eval::linalg::qr(&a_val),
            Err(other) => return Err(unify_backend_error(other)),
        };
        let q_id = self.tape.push_eager(
            Op::QrQ {
                input: self.id,
                r: r_val.clone(),
            },
            q_val.clone(),
        );
        let r_id = self.tape.push_eager(
            Op::QrR {
                input: self.id,
                q: q_val,
            },
            r_val,
        );
        Ok(QrVars {
            q: Var::from_raw(self.tape, q_id),
            r: Var::from_raw(self.tape, r_id),
        })
    }

    /// reduced SVD（`A: [m,n]` → [`SvdVars`]。`k = min(m,n)`）。
    /// イシュー #1621。`qr` と同じ多出力設計（`Op::SvdU`／
    /// `Op::SvdS`／`Op::SvdVh`）。反復が収束しない場合は
    /// `AutodiffError::InvalidArgument(_)`（CPU 本番経路・フォールバック
    /// とも `unify_backend_error` で同一 variant に統一済み。
    /// codex-review 指摘の是正）。
    pub fn svd(&self) -> Result<SvdVars<'t>, AutodiffError> {
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: shape.len(),
            }));
        }
        let (m, n) = (shape[0], shape[1]);
        let k = m.min(n);
        let a_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let (u_val, s_val, vh_val) = match self.tape.ops().linalg_svd(&a_val) {
            Ok(factors) => {
                verify_shape(factors.u.shape(), &[m, k])?;
                verify_shape(factors.s.shape(), &[k])?;
                verify_shape(factors.vh.shape(), &[k, n])?;
                (factors.u, factors.s, factors.vh)
            }
            Err(BackendError::Unsupported(_)) => eval::linalg::svd(&a_val)?,
            Err(other) => return Err(unify_backend_error(other)),
        };
        let u_id = self.tape.push_eager(
            Op::SvdU {
                input: self.id,
                s: s_val.clone(),
                vh: vh_val.clone(),
            },
            u_val.clone(),
        );
        let s_id = self.tape.push_eager(
            Op::SvdS {
                input: self.id,
                u: u_val.clone(),
                vh: vh_val.clone(),
            },
            s_val.clone(),
        );
        let vh_id = self.tape.push_eager(
            Op::SvdVh {
                input: self.id,
                u: u_val,
                s: s_val,
            },
            vh_val,
        );
        Ok(SvdVars {
            u: Var::from_raw(self.tape, u_id),
            s: Var::from_raw(self.tape, s_id),
            vh: Var::from_raw(self.tape, vh_id),
        })
    }

    /// 行列ノルム（`A: [m,n]`・`ord` → スカラー `[]`）。イシュー #1621。
    /// `ord` が [`MatrixNormOrd::Nuc`]／[`MatrixNormOrd::Spectral`] の
    /// 場合、フォールバック実装（`eval::linalg::matrix_norm`）内部で
    /// 特異値分解を用いる（`Var` 側で `svd` ノードを合成しない設計。
    /// `docs/autodiff-linalg-design.md` §3.2）。
    pub fn matrix_norm(&self, ord: MatrixNormOrd) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: shape.len(),
            }));
        }
        let a_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = match self.tape.ops().linalg_matrix_norm(&a_val, ord) {
            Ok(v) => {
                verify_shape(v.shape(), &[])?;
                v
            }
            Err(BackendError::Unsupported(_)) => eval::linalg::matrix_norm(&a_val, ord)?,
            Err(other) => return Err(unify_backend_error(other)),
        };
        let id = self.tape.push_eager(
            Op::MatrixNorm {
                input: self.id,
                ord,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 分散（`dim` に沿った縮約。`dim: None` は全軸縮約〈スカラー〉。
    /// `torch.var(dim, correction)` 相当。`correction`: `1` が不偏分散
    /// （既定相当）・`0` が母分散（`tf.math.reduce_variance` 相当）。
    /// イシュー #1723）。
    ///
    /// `sum`／`max`（`BackendOps` 必須メソッド）とは異なり
    /// `BackendOps::var` は既定 `Unsupported`（`matrix_norm` と同じ
    /// フォールバック契約）——バックエンド未実装のときのみ
    /// `eval::var_along`（ホスト参照実装）へ切り替える。
    ///
    /// **数値契約**: `f64` 二段計算（`.claude/rules/coding-rust.md`）。
    /// **エラー契約**: 縮約対象の要素数 `n`（`dim=Some(axis)` は
    /// `shape[axis]`・`dim=None` は全要素数）が `0` または
    /// `n <= correction` の場合は [`AutodiffError::InvalidArgument`]
    /// （`NaN`／`inf` を黙って返さない安全側の判断。PyTorch の
    /// `NaN`／`inf` 返却とは意図的に異なる。`docs/spec/` の対象外の
    /// 独自安全策）。
    pub fn var(&self, dim: Option<usize>, correction: usize) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        let out_shape = var_std_out_shape_checked(&shape, dim, correction, "Var::var")?;
        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = match self.tape.ops().var(&input_val, dim, correction) {
            Ok(v) => {
                verify_shape(v.shape(), &out_shape)?;
                v
            }
            Err(BackendError::Unsupported(_)) => {
                eval::var_along(&input_val, dim, correction, &out_shape)
            }
            Err(other) => return Err(unify_backend_error(other)),
        };
        let id = self.tape.push_eager(
            Op::Var {
                input: self.id,
                dim,
                correction,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// 標準偏差（`dim` に沿った縮約。`torch.std(dim, correction)`
    /// 相当。イシュー #1723・レビュー是正）。
    ///
    /// 当初 `Var::var(dim, correction)?.sqrt()`（新規 `Op` を設けない
    /// 合成）として実装していたが、`Op::Var` の forward 値が分散を
    /// `f32` へ downcast してから `Op::Sqrt` に渡すため、真の分散が
    /// `f32` の範囲（有限最大値 約 `3.4e38`）を超える極端な入力（例
    /// `[-1e20, 1e20]`。分散 `≈1e40`）で `std` 自体は `f32` で表現
    /// 可能（`≈1.41e20`）にもかかわらず `inf` になっていた
    /// （codex-review P2 指摘）。本メソッドは専用の `Op::Std`
    /// （`crate::tape::Op`）ノードを直接構築し、forward・backward
    /// とも `f64` の分散を経由してから最後に 1 回だけ `sqrt` を計算
    /// する（`eval::std_along`／`grad::std_vjp` 参照。`Var::var` と
    /// 異なり `BackendOps` に対応メソッドは設けない——常に
    /// `eval::std_along`〈ホスト参照実装〉を使う。将来デバイス側
    /// カーネルが必要になれば非破壊で追加できる）。
    ///
    /// **数値規約**: `std` は `f64` の分散から `f64` で `sqrt` した
    /// 値を 1 回だけ `f32` へ downcast する（`var == 0` の場合
    /// `std == 0`）。勾配（`Op::Std` の VJP）は `std == 0`（縮約対象
    /// が全て同値の定数列）の要素でゼロ勾配へ明示的にマスクする
    /// （`0.0 / 0.0` の `NaN` は伝播しない）。これは PyTorch
    /// `torch.std` backward（`std_backward`。`FunctionsManual.cpp`）が
    /// `masked_fill_(result == 0, 0)` で行うマスクと一致する規約
    /// （`grad::std_vjp` doc 参照）。**`Var::var(..).sqrt()`（新規
    /// `Op` を追加しない合成）とは異なる**点に注意——その合成では
    /// `Var::sqrt` の `y == 0` 規約により `0.0 / 0.0 = NaN` を返す。
    ///
    /// **エラー契約**: [`Var::var`] と同じ（縮約対象の要素数 `n` が
    /// `0` または `n <= correction` の場合は
    /// [`AutodiffError::InvalidArgument`]）。
    pub fn std(&self, dim: Option<usize>, correction: usize) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        let out_shape = var_std_out_shape_checked(&shape, dim, correction, "Var::std")?;
        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = eval::std_along(&input_val, dim, correction, &out_shape);
        let id = self.tape.push_eager(
            Op::Std {
                input: self.id,
                dim,
                correction,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }

    /// L1 ノルム（`Σ|x_i|`・`dim` に沿った縮約。`torch.norm(p=1)`
    /// 相当。イシュー #1723）。`Var::norm`（`pub(crate)`）への薄い
    /// 委譲。
    pub fn norm_l1(&self, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
        self.norm(VectorNormOrd::L1, dim)
    }

    /// L2 ノルム（`√Σx_i²`・`dim` に沿った縮約。`torch.norm(p=2)`
    /// 相当。イシュー #1723）。`Var::norm`（`pub(crate)`）への薄い
    /// 委譲。
    pub fn norm_l2(&self, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
        self.norm(VectorNormOrd::L2, dim)
    }

    /// [`Var::norm_l1`]／[`Var::norm_l2`] の共通実装（イシュー #1723）。
    /// `pub(crate)` 限定とする理由: `VectorNormOrd` を引数に取る汎用版
    /// （`norm(ord, dim)`）は spec `docs/compat-api-scope.md` §5「Tier 1
    /// 列挙済み機能は再適用不要」の対象が `var`／`std` に限られ `norm`
    /// は未列挙のため、facade への `VectorNormOrd` 型の新規公開面
    /// （enum 再エクスポート）を避け、`norm_l1`／`norm_l2` の 2 つの
    /// `pub fn`（enum を露出しない）のみを公開する（実装計画 §0「facade
    /// 公開面」参照）。
    ///
    /// `Var::var` と同じフォールバック契約
    /// （`BackendOps::vector_norm` → `Unsupported` のときのみ
    /// `eval::vector_norm_along`）。空縮約（`n == 0`）は
    /// [`AutodiffError::InvalidArgument`]。
    pub(crate) fn norm(
        &self,
        ord: VectorNormOrd,
        dim: Option<usize>,
    ) -> Result<Var<'t>, AutodiffError> {
        let shape = self.shape();
        let out_shape = reduce_out_shape(&shape, dim)?;
        let n = match dim {
            None => shape.iter().product(),
            Some(axis) => shape[axis],
        };
        if n == 0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Var::norm: 縮約対象の要素数が 0（dim={dim:?}）"
            )));
        }
        let input_val = {
            let nodes = self.tape.nodes.borrow();
            materialize_fallible(&nodes, self.tape.ops(), self.id)?.clone()
        };
        let value = match self.tape.ops().vector_norm(&input_val, ord, dim) {
            Ok(v) => {
                verify_shape(v.shape(), &out_shape)?;
                v
            }
            Err(BackendError::Unsupported(_)) => {
                eval::vector_norm_along(&input_val, ord, dim, &out_shape)
            }
            Err(other) => return Err(unify_backend_error(other)),
        };
        let id = self.tape.push_eager(
            Op::VectorNorm {
                input: self.id,
                ord,
                dim,
            },
            value,
        );
        Ok(Var::from_raw(self.tape, id))
    }
}

/// [`Var::rnn_cell`]／[`Var::lstm_cell`]／[`Var::gru_cell`] へ渡す
/// ゲートパラメータのまとめ（イシュー #1647・設計 `docs/autodiff-rnn-
/// cell-tape-design.md` 決定 5・10）。PyTorch のパラメータ順
/// （`weight_ih, weight_hh, bias_ih, bias_hh`）を踏襲する。
pub struct GateParams<'a, 't> {
    pub w_ih: &'a Var<'t>,
    pub w_hh: &'a Var<'t>,
    pub b_ih: Option<&'a Var<'t>>,
    pub b_hh: Option<&'a Var<'t>>,
}

/// `b_ih`／`b_hh` が両方 `Some` か両方 `None` であることを検査する
/// （決定 4 のセル API 契約。片方のみは bias 加算の意味論が定義され
/// ないため拒否する）。
fn check_bias_pair(b_ih: Option<&Var<'_>>, b_hh: Option<&Var<'_>>) -> Result<(), AutodiffError> {
    if b_ih.is_some() != b_hh.is_some() {
        return Err(AutodiffError::InvalidArgument(
            "gate cell: b_ih and b_hh must be both Some or both None".to_string(),
        ));
    }
    Ok(())
}

/// 全入力が rank-2 であることを検査する（セル API 共通の shape 検査
/// 冒頭）。
fn require_rank2_all(shapes: &[&[usize]], op_name: &str) -> Result<(), AutodiffError> {
    for shape in shapes {
        if shape.len() != 2 {
            return Err(AutodiffError::InvalidArgument(format!(
                "{op_name}: all operands must be rank-2 (got shape {shape:?})"
            )));
        }
    }
    Ok(())
}

/// `input_size`（`D`）・`hidden_size`（`H`）双方が 0 でないことを検査
/// する（zero-K ガード。決定 4）。
fn require_positive_dims(d: usize, hidden: usize, op_name: &str) -> Result<(), AutodiffError> {
    if d == 0 || hidden == 0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "{op_name}: input_size (D={d}) and hidden_size (H={hidden}) must both be > 0"
        )));
    }
    Ok(())
}

/// `gates * hidden`（ゲート幅）を `checked_mul` で検証する
/// （`nn::rnn::checked_gate_width` と同型）。
///
/// 本番経路 panic 禁止（AGENTS.md）: `lstm_cell`／`gru_cell` は
/// `h_prev.shape()[1]` から `hidden` を導出するが、`h_prev` が要素数
/// 0 の空 `Var`（例: `h_shape = [0, 1usize << 62]`）であれば
/// `require_positive_dims` の `hidden == 0` 検査を通過したまま
/// `hidden` が `usize::MAX` 近傍になりうる。`4 * hidden`／`3 * hidden`
/// を未検証のまま比較・`narrow` 幅へ使うと overflow により期待幅が
/// 周回し、不正な形状を誤って受理してしまう（イシュー #1647
/// codex-review P1 指摘）。
fn checked_gate_width(gates: usize, hidden: usize) -> Result<usize, AutodiffError> {
    gates.checked_mul(hidden).ok_or_else(|| {
        AutodiffError::InvalidArgument(format!(
            "gates (={gates}) * hidden (={hidden}) overflowed usize"
        ))
    })
}

/// `Option<&Var>` を `Option<Tensor<f32>>` へ実体化する共通ヘルパー
/// （`nodes` の借用スコープ内で使う）。
fn optional_materialize(
    nodes: &[crate::tape::TapeNode],
    ops: &dyn BackendOps,
    var: Option<&Var<'_>>,
) -> Result<Option<Tensor<f32>>, AutodiffError> {
    match var {
        Some(v) => Ok(Some(materialize_fallible(nodes, ops, v.id)?.clone())),
        None => Ok(None),
    }
}

// =====================================================================
// セル forward の値計算（イシュー #1647）。`Var::{rnn_cell,lstm_cell,
// gru_cell}`（tape 経路）と `nn::rnn`（`forward_host`。tape 不要経路）
// の両方から呼ばれる共有ロジックであり、同じ関数を通すことで両経路の
// forward が bit-exact に一致することを構造的に保証する
// （`docs/autodiff-rnn-cell-tape-design.md` 決定 9）。
// =====================================================================

/// ゲート演算（RNN／LSTM／GRU セル）の重み・bias をまとめた引数束
/// （`clippy::too_many_arguments` 回避。イシュー #1647）。
/// [`rnn_cell_forward_value`]／[`lstm_cell_forward_values`]／
/// [`gru_cell_forward_values`] が共通で受け取る。
pub(crate) struct CellWeights<'a> {
    pub w_ih: &'a Tensor<f32>,
    pub w_hh: &'a Tensor<f32>,
    pub b_ih: Option<&'a Tensor<f32>>,
    pub b_hh: Option<&'a Tensor<f32>>,
}

/// RNN（tanh 版）セルの forward 値計算。`BackendOps::gemm_bias_act` を
/// 2 回（`x·W_ih+b_ih`・`h_prev·W_hh+b_hh`）呼び、`add` → `tanh` で
/// 閉じる（決定 1「RNN は新規カーネル不要」）。
pub(crate) fn rnn_cell_forward_value(
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
    h_prev: &Tensor<f32>,
    w: &CellWeights<'_>,
) -> Result<Tensor<f32>, AutodiffError> {
    let pre_ih = ops.gemm_bias_act(x, w.w_ih, w.b_ih, Activation::None)?;
    let pre_hh = ops.gemm_bias_act(h_prev, w.w_hh, w.b_hh, Activation::None)?;
    let pre = ops.add(&pre_ih, &pre_hh)?;
    Ok(ops.tanh(&pre)?)
}

/// LSTM セルの forward 値計算。`pre = x·W_ih+b_ih + h_prev·W_hh+b_hh`
/// （`[B, 4H]`）を計算したのち `BackendOps::lstm_pointwise` へ渡す
/// （`Unsupported` のときのみ `eval::lstm_pointwise` へフォールバック。
/// A08: 判定迂回経路を作らない）。
pub(crate) fn lstm_cell_forward_values(
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
    h_prev: &Tensor<f32>,
    c_prev: &Tensor<f32>,
    w: &CellWeights<'_>,
) -> Result<LstmPointwiseOutput, AutodiffError> {
    let pre_ih = ops.gemm_bias_act(x, w.w_ih, w.b_ih, Activation::None)?;
    let pre_hh = ops.gemm_bias_act(h_prev, w.w_hh, w.b_hh, Activation::None)?;
    let pre = ops.add(&pre_ih, &pre_hh)?;
    match ops.lstm_pointwise(&pre, c_prev) {
        Ok(v) => Ok(v),
        Err(BackendError::Unsupported(_)) => Ok(eval::lstm_pointwise(&pre, c_prev)),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// GRU セルの forward 値計算。`pre_i = x·W_ih+b_ih`・
/// `pre_h = h_prev·W_hh+b_hh`（いずれも `[B, 3H]`。独立した 2 本の
/// GEMM のため 1 本に足し込まない点が LSTM と異なる）を計算したのち
/// `BackendOps::gru_pointwise` へ渡す（`Unsupported` のときのみ
/// `eval::gru_pointwise` へフォールバック）。
pub(crate) fn gru_cell_forward_values(
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
    h_prev: &Tensor<f32>,
    w: &CellWeights<'_>,
) -> Result<GruPointwiseOutput, AutodiffError> {
    let pre_i = ops.gemm_bias_act(x, w.w_ih, w.b_ih, Activation::None)?;
    let pre_h = ops.gemm_bias_act(h_prev, w.w_hh, w.b_hh, Activation::None)?;
    match ops.gru_pointwise(&pre_i, &pre_h, h_prev) {
        Ok(v) => Ok(v),
        Err(BackendError::Unsupported(_)) => Ok(eval::gru_pointwise(&pre_i, &pre_h, h_prev)),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// rank-2・正方であることを検査し、辺長 `n` を返す（線形代数演算
/// 共通のヘルパー。イシュー #1621）。非正方は `ShapeError` の既存
/// variant で意味的に表現できないため `AutodiffError::InvalidArgument`
/// とする（`cross_entropy_loss` の target 範囲検査と同方針）。
fn require_square(shape: &[usize], op_name: &str) -> Result<usize, AutodiffError> {
    if shape.len() != 2 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: shape.len(),
        }));
    }
    if shape[0] != shape[1] {
        return Err(AutodiffError::InvalidArgument(format!(
            "{op_name}: 正方行列（[n,n]）が必要（形状 {shape:?}）"
        )));
    }
    Ok(shape[0])
}

/// [`Var::var`]／[`Var::std`] 共通の出力 shape 算出・エラー検査
/// （イシュー #1723 レビュー是正で重複を統合）。`reduce_out_shape` で
/// 出力 shape を求めたうえで、縮約対象の要素数 `n`（`dim=Some(axis)`
/// は `shape[axis]`・`dim=None` は全要素数）が `0` または
/// `n <= correction` の場合は [`AutodiffError::InvalidArgument`] を
/// 返す（`NaN`／`inf` を黙って返さない安全側の判断。PyTorch の
/// `NaN`／`inf` 返却とは意図的に異なる。`docs/spec/` の対象外の
/// 独自安全策）。`op_name` はエラーメッセージに埋め込む呼び出し元の
/// 演算名（`"Var::var"`／`"Var::std"`）。
fn var_std_out_shape_checked(
    shape: &[usize],
    dim: Option<usize>,
    correction: usize,
    op_name: &str,
) -> Result<Vec<usize>, AutodiffError> {
    let out_shape = reduce_out_shape(shape, dim)?;
    let n = match dim {
        None => shape.iter().product(),
        Some(axis) => shape[axis],
    };
    if n == 0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "{op_name}: 縮約対象の要素数が 0（dim={dim:?}）"
        )));
    }
    if n <= correction {
        return Err(AutodiffError::InvalidArgument(format!(
            "{op_name}: 自由度不足（n={n} <= correction={correction}）"
        )));
    }
    Ok(out_shape)
}

/// バックエンド実装（`BackendOps::linalg_*`）の戻り値 shape が契約
/// （doc comment）どおりであることを検証する（`mse_loss_with` の
/// 「バックエンド実装の契約を検証する」規律と同型。実装バグの黙認
/// 防止。`.claude/rules/security.md` A08）。
fn verify_shape(actual: &[usize], expected: &[usize]) -> Result<(), AutodiffError> {
    if actual == expected {
        Ok(())
    } else {
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(
            ShapeError::ShapeMismatch {
                lhs: actual.to_vec(),
                rhs: expected.to_vec(),
            },
        )))
    }
}

/// CPU 本番経路（`BackendOps::linalg_*` が `Unsupported` 以外の `Err` を
/// 返した場合）の `BackendError` を、公開ドキュメント（`Var::inv` 等の
/// doc comment・`docs/autodiff-linalg-design.md` §3.5「エラー分類」）が
/// 約束する `AutodiffError` variant へ写像する。`BackendError::
/// InvalidArgument(_)`（特異行列・非正定値・非収束のいずれか）は
/// `eval::linalg`（ホスト参照実装。`Unsupported` 時のフォールバック
/// 経路）が返す `AutodiffError::InvalidArgument(_)` と同一 variant に
/// 揃える——以前は逆方向（フォールバック側を `Backend(InvalidArgument)`
/// へ包む）へ統一していたため、`fandhe_ai::tape()`（CPU 本番経路）を
/// 使う呼び出し元が公開ドキュメントどおり `AutodiffError::
/// InvalidArgument` を照合しても本番経路の数値エラーを捕捉できない
/// 不整合があった（codex-review 指摘。再試行はしない・エラー内容は
/// そのまま伝播するのみ）。`InvalidArgument` 以外の `BackendError`
/// （`Unsupported`〈本関数の呼び出し元では既に分岐済み〉・
/// `ShapeMismatch` 等）は意味を変えず `AutodiffError::Backend(_)` の
/// まま伝播する。
fn unify_backend_error(err: BackendError) -> AutodiffError {
    match err {
        BackendError::InvalidArgument(msg) => AutodiffError::InvalidArgument(msg),
        other => AutodiffError::Backend(other),
    }
}

/// [`Var::qr`] の戻り値。`Q`（`[m,k]`）・`R`（`[k,n]`）は別テープノード
/// （多出力設計。`Op::QrQ`／`Op::QrR` doc 参照）。
#[derive(Debug, Clone, Copy)]
pub struct QrVars<'t> {
    /// `[m, k]`（`k = min(m, n)`）。列直交。
    pub q: Var<'t>,
    /// `[k, n]`（`k = min(m, n)`）。上三角・対角非負。
    pub r: Var<'t>,
}

/// [`Var::svd`] の戻り値。`U`（`[m,k]`）・`S`（`[k]`）・`Vh`（`[k,n]`）は
/// 別テープノード（多出力設計。`Op::SvdU`／`Op::SvdS`／`Op::SvdVh` doc
/// 参照）。
#[derive(Debug, Clone, Copy)]
pub struct SvdVars<'t> {
    /// `[m, k]`（`k = min(m, n)`）。列直交。
    pub u: Var<'t>,
    /// `[k]`。特異値（降順・非負）。
    pub s: Var<'t>,
    /// `[k, n]`（`k = min(m, n)`）。行直交（`V^T`）。
    pub vh: Var<'t>,
}

/// [`Var::host_view`] が返す借用ビュー（イシュー #1335。P1 是正で
/// `Tape` の借用を保持しない設計へ変更——`Var::host_view` ドキュメント
/// コメント参照）。
///
/// `Deref<Target = [f32]>` でスライスとして使う。内部に保持する
/// `Tensor<f32>` は `host_view()` 構築時に既に
/// [`fandhe_ai_tensor_core::Tensor::contiguous`] 済み（[`Tensor::as_slice`]
/// が必ず `Some` を返す状態）であり、`Deref::deref` はその場で
/// borrow-checker 上の `&self` の借用として `&[f32]` を返すだけで
/// 追加コピーを伴わない。
///
/// **寿命契約**: `VarHostView` はライフタイムパラメータを持たず
/// `Tape`／`RefCell` の借用を一切保持しないため、生存中に同じ `Var`／
/// `Tape` へノード追加演算（`add`/`matmul` 等）を呼んでも panic しない
/// （[`Var::value`] の借用注意とは異なる）。
#[derive(Debug)]
pub struct VarHostView {
    tensor: Tensor<f32>,
}

impl std::ops::Deref for VarHostView {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        // `host_view()` が `contiguous()` 済みの `Tensor` のみを格納する
        // ため `as_slice()` は必ず `Some`。防御的に `unwrap_or(&[])`。
        self.tensor.as_slice().unwrap_or(&[])
    }
}

#[cfg(test)]
mod linear_act_tests {
    use super::*;
    use crate::tape::Tape;

    /// codex-review 指摘（PR #1079・discussion_r3889050931）の実測検証:
    /// `linear_act` は bias が `[n]`（`weight` の列数）と厳密一致しない
    /// broadcast 可能な shape（ここでは `[1, n]`）でも `ShapeMismatch` を
    /// 返さず、`matmul` → `add`（NumPy 互換ブロードキャスト）→ `relu` の
    /// 非融合合成と bit 一致する結果を返すことを確認する。フォール
    /// バックは `linear_act`／呼び出し元ではなく `BackendOps::
    /// gemm_bias_act` 自身の契約（`tensor-core::backend_ops` の doc・
    /// 各バックエンドの `ComposedFallback` 分岐）で行われる（本メソッド
    /// の doc コメント参照）。`Linear`（`nn::linear`）は `from_parameters`
    /// で bias を `[out_features]` 厳密一致にしか構築できないため、この
    /// broadcast bias 経路は `Linear` 経由では到達できない
    /// （`pub(crate)` の `linear_act` を直接呼ぶ本テストでのみ検証可能）。
    #[test]
    fn linear_act_accepts_broadcastable_bias_not_strictly_matching_out_features() {
        let tape = Tape::new();
        // input: [2, 2]、weight: [2, 3] → out: [2, 3]。
        let input = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap());
        let weight = tape.var(&Tensor::new(vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0], &[2, 3]).unwrap());
        // bias: `[3]`（out_features 厳密一致）ではなく `[1, 3]`
        // （broadcast 可能だが厳密一致ではない shape）。
        let bias = tape.var(&Tensor::new(vec![10.0, -5.0, 0.0], &[1, 3]).unwrap());

        let fused = input
            .linear_act(&weight, Some(&bias), Activation::Relu)
            .expect("broadcast bias は ShapeMismatch にならず成功するはず");

        let composed = input
            .matmul(&weight)
            .and_then(|y| y.add(&bias))
            .map(|y| y.relu())
            .expect("非融合合成（matmul→add→relu）も同じ broadcast bias で成功するはず");

        assert_eq!(
            fused.value().as_slice().unwrap(),
            composed.value().as_slice().unwrap(),
            "broadcast bias 経路は融合・非融合合成で bit 一致するはず"
        );
    }

    // `Op::LinearAct::compute_dtype`（イシュー #1960）: `linear_act`
    // （既存 F32 記録経路）と `linear_act_low_precision`（opt-in 低精度
    // 記録経路）が正しい `ScalarDType` を記録することを直接確認する。
    // `compute_dtype` は VJP からは読まれない純粋な記録専用フィールド
    // （`grad::vjp` の `Op::LinearAct` 分岐 doc 参照）のため、dead_code
    // 回避の便宜的な用途ではなく「forward 経路の識別子として正しく
    // 記録される」という本イシューの契約そのものを検証する。
    #[test]
    fn linear_act_records_f32_compute_dtype() {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(vec![1.0, 2.0], &[1, 2]).unwrap());
        let w = tape.var(&Tensor::new(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap());
        let out = x.linear_act(&w, None, Activation::None).unwrap();
        let nodes = tape.nodes.borrow();
        match &nodes[out.id.0].op {
            crate::tape::Op::LinearAct { compute_dtype, .. } => {
                assert_eq!(*compute_dtype, fandhe_ai_tensor_core::ScalarDType::F32);
            }
            other => panic!("expected Op::LinearAct, got {other:?}"),
        }
    }

    /// 低精度カーネル未実装のバックエンド（`Tape::new()` 既定の
    /// naive 参照実装は `typed_ops_f16`／`typed_ops_bf16` accessor が
    /// 既定 `None`）に対し、`linear_act_low_precision` が f32 へ静かに
    /// フォールバックせず `AutodiffError::Backend(BackendError::
    /// Unsupported(_))` を返すことを確認する（`.claude/rules/
    /// security.md` A04 の fail-closed 方針）。
    #[test]
    fn linear_act_low_precision_without_typed_ops_returns_unsupported() {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(vec![1.0, 2.0], &[1, 2]).unwrap());
        let w = tape.var(&Tensor::new(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap());
        let err = x
            .linear_act_low_precision(
                &w,
                None,
                Activation::None,
                fandhe_ai_tensor_core::ScalarDType::F16,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            crate::error::AutodiffError::Backend(fandhe_ai_tensor_core::BackendError::Unsupported(
                _
            ))
        ));
    }

    /// codex-review 指摘（PR #2000）の回帰テスト: `bias` が
    /// `gemm_out_shape` の結果（`out_shape`）を「拡張」するブロード
    /// キャスト（`input=[1,2]`・`weight=[2,1]`・`bias=[2,1]`。
    /// `out_shape` は `[1,1]` だが `bias` との NumPy 互換ブロードキャスト
    /// 結果は `[2,1]` で `out_shape` と一致しない）を、
    /// `typed_ops_f16()`／`typed_ops_bf16()` accessor が既定 `None` の
    /// `Tape::new()`（fail-closed 経路と同じ前提）でも shape 検査段階で
    /// 拒否することを確認する（`AutodiffError::Shape(_)`。accessor 探索
    /// より前に検査が完結する契約は `tensor-core::low_precision::
    /// linear_forward_low_precision` 側の同型テスト
    /// `expanding_bias_broadcast_is_rejected_before_accessor_lookup` と
    /// 対になる）。
    #[test]
    fn linear_act_low_precision_rejects_bias_that_expands_out_shape() {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(vec![1.0, 2.0], &[1, 2]).unwrap());
        let w = tape.var(&Tensor::new(vec![1.0, 1.0], &[2, 1]).unwrap());
        let bias = tape.var(&Tensor::new(vec![0.0, 0.0], &[2, 1]).unwrap());
        let err = x
            .linear_act_low_precision(
                &w,
                Some(&bias),
                Activation::None,
                fandhe_ai_tensor_core::ScalarDType::F16,
            )
            .unwrap_err();
        assert!(matches!(err, crate::error::AutodiffError::Shape(_)));
    }

    // 低精度 Linear forward（イシュー #1960）の成功経路テスト
    // （codex-review 指摘・PR #2000 discussion_r3889050931 対応）。
    // 上記 `..._without_typed_ops_returns_unsupported` は accessor 不在の
    // fail-closed 経路のみを検証しており、①丸めを伴う forward、②低精度
    // 出力による ReLU マスク、③f32 master 値を使う backward という
    // 成功経路の 3 契約（`docs/autodiff-low-precision-linear-design.md`
    // §1・§2）が未検証だった。以下はそれを埋める「bit 一致オラクル
    // テスト」（同 doc §1 が言及する検証手段の実体）。

    // `half::f16`／`half::bf16` は `fandhe_ai_tensor_core` の再エクスポート
    // 経由で参照する（本クレートの `half` への直接 Cargo 依存は追加しない。
    // codex-review 指摘対応・PR #2000 discussion 参照。`tensor-core::lib.rs`
    // の `pub use half::{bf16, f16};` doc comment 参照）。
    use fandhe_ai_tensor_core::{bf16, f16};

    /// `TypedOps<half::f16>`／`TypedOps<half::bf16>` の実装契約
    /// （`crates/backend-cpu/src/typed_f16.rs` doc:
    /// `f16::from_f32(BackendOps::op(upcast(x)))`）を再現するモック
    /// `BackendOps`。CPU バックエンドクレートへの新規依存を避けるため
    /// （`nn/linear.rs::ComputingMockOps` と同じ方針）、f32 側の
    /// `gemm`／`add`／`relu`（backward の `matmul_vjp` が経由する）も
    /// 素朴な実装を本構造体に複製する。
    struct ComputingLowPrecisionBackendOps;

    impl ComputingLowPrecisionBackendOps {
        fn gemm_f32(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
            // `matmul_vjp`（backward）は転置 view（`transpose2d`。非
            // contiguous な zero-copy view）をそのまま渡してくるため、
            // `nn/linear.rs::ComputingMockOps` と異なり本モックは
            // `.contiguous()` で実体化してから読む（本番 BackendOps は
            // stride 読みで対応するが、本テスト用の素朴な実装では
            // 簡略化する）。
            let a = a.contiguous();
            let b = b.contiguous();
            let (m, k) = (a.shape()[0], a.shape()[1]);
            let n = b.shape()[1];
            let a_data = a.as_slice().expect("test: a must be contiguous");
            let b_data = b.as_slice().expect("test: b must be contiguous");
            let mut out = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for p in 0..k {
                        acc = a_data[i * k + p].mul_add(b_data[p * n + j], acc);
                    }
                    out[i * n + j] = acc;
                }
            }
            Tensor::new(out, &[m, n]).unwrap()
        }

        fn add_f32(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
            let a_shape = a.shape().to_vec();
            let a_data = a.as_slice().expect("test: a must be contiguous");
            let b_data = b.as_slice().expect("test: b must be contiguous");
            let n = a_shape[1];
            let out: Vec<f32> = a_data
                .iter()
                .enumerate()
                .map(|(idx, x)| x + b_data[idx % n])
                .collect();
            Tensor::new(out, &a_shape).unwrap()
        }

        fn relu_f32(a: &Tensor<f32>) -> Tensor<f32> {
            let data = a.as_slice().expect("test: a must be contiguous");
            let out: Vec<f32> = data.iter().map(|x| x.max(0.0)).collect();
            Tensor::new(out, a.shape()).unwrap()
        }
    }

    impl BackendOps for ComputingLowPrecisionBackendOps {
        fn device(&self) -> Device {
            Device::Cpu
        }
        fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Ok(Self::gemm_f32(a, b))
        }
        fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Ok(Self::add_f32(a, b))
        }
        fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("test: mul".into()))
        }
        fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Ok(Self::relu_f32(a))
        }
        fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("test: exp".into()))
        }
        fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("test: tanh".into()))
        }
        fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            // 低精度 backward テスト（`linear_act_low_precision_backward_*`）
            // が `Var::sum(None)` で loss をスカラー化するためだけに使う
            // 全要素縮約の素朴な実装（`reduce_out_shape(shape, None) ==
            // []` に合わせ shape `[]` の 1 要素 Tensor を返す）。軸指定版
            // （`dim.is_some()`）は本テストでは未使用のため未実装のまま。
            match dim {
                None => {
                    let data = a.as_slice().expect("test: a must be contiguous");
                    let total: f32 = data.iter().sum();
                    Tensor::new(vec![total], &[]).map_err(BackendError::ShapeMismatch)
                }
                Some(_) => Err(BackendError::Unsupported("test: sum(dim)".into())),
            }
        }
        fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("test: max".into()))
        }
        fn typed_ops_f16(&self) -> Option<&dyn fandhe_ai_tensor_core::TypedOps<f16>> {
            Some(self)
        }
        fn typed_ops_bf16(&self) -> Option<&dyn fandhe_ai_tensor_core::TypedOps<bf16>> {
            Some(self)
        }
    }

    impl fandhe_ai_tensor_core::TypedOps<f16> for ComputingLowPrecisionBackendOps {
        fn gemm(&self, a: &Tensor<f16>, b: &Tensor<f16>) -> Result<Tensor<f16>, BackendError> {
            Ok(round_f16(&Self::gemm_f32(&upcast_f16(a), &upcast_f16(b))))
        }
        fn add(&self, a: &Tensor<f16>, b: &Tensor<f16>) -> Result<Tensor<f16>, BackendError> {
            Ok(round_f16(&Self::add_f32(&upcast_f16(a), &upcast_f16(b))))
        }
        fn mul(&self, _a: &Tensor<f16>, _b: &Tensor<f16>) -> Result<Tensor<f16>, BackendError> {
            Err(BackendError::Unsupported("test: mul".into()))
        }
        fn relu(&self, a: &Tensor<f16>) -> Result<Tensor<f16>, BackendError> {
            Ok(round_f16(&Self::relu_f32(&upcast_f16(a))))
        }
        fn exp(&self, _a: &Tensor<f16>) -> Result<Tensor<f16>, BackendError> {
            Err(BackendError::Unsupported("test: exp".into()))
        }
        fn tanh(&self, _a: &Tensor<f16>) -> Result<Tensor<f16>, BackendError> {
            Err(BackendError::Unsupported("test: tanh".into()))
        }
        fn sum(&self, _a: &Tensor<f16>, _dim: Option<usize>) -> Result<Tensor<f16>, BackendError> {
            Err(BackendError::Unsupported("test: sum".into()))
        }
        fn max(&self, _a: &Tensor<f16>, _dim: Option<usize>) -> Result<Tensor<f16>, BackendError> {
            Err(BackendError::Unsupported("test: max".into()))
        }
    }

    impl fandhe_ai_tensor_core::TypedOps<bf16> for ComputingLowPrecisionBackendOps {
        fn gemm(&self, a: &Tensor<bf16>, b: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
            Ok(round_bf16(&Self::gemm_f32(
                &upcast_bf16(a),
                &upcast_bf16(b),
            )))
        }
        fn add(&self, a: &Tensor<bf16>, b: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
            Ok(round_bf16(&Self::add_f32(&upcast_bf16(a), &upcast_bf16(b))))
        }
        fn mul(&self, _a: &Tensor<bf16>, _b: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
            Err(BackendError::Unsupported("test: mul".into()))
        }
        fn relu(&self, a: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
            Ok(round_bf16(&Self::relu_f32(&upcast_bf16(a))))
        }
        fn exp(&self, _a: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
            Err(BackendError::Unsupported("test: exp".into()))
        }
        fn tanh(&self, _a: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
            Err(BackendError::Unsupported("test: tanh".into()))
        }
        fn sum(
            &self,
            _a: &Tensor<bf16>,
            _dim: Option<usize>,
        ) -> Result<Tensor<bf16>, BackendError> {
            Err(BackendError::Unsupported("test: sum".into()))
        }
        fn max(
            &self,
            _a: &Tensor<bf16>,
            _dim: Option<usize>,
        ) -> Result<Tensor<bf16>, BackendError> {
            Err(BackendError::Unsupported("test: max".into()))
        }
    }

    fn upcast_f16(t: &Tensor<f16>) -> Tensor<f32> {
        let data: Vec<f32> = t
            .as_slice()
            .expect("test: contiguous")
            .iter()
            .map(|v| v.to_f32())
            .collect();
        Tensor::new(data, t.shape()).unwrap()
    }
    fn round_f16(t: &Tensor<f32>) -> Tensor<f16> {
        let data: Vec<f16> = t
            .as_slice()
            .expect("test: contiguous")
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        Tensor::new(data, t.shape()).unwrap()
    }
    fn upcast_bf16(t: &Tensor<bf16>) -> Tensor<f32> {
        let data: Vec<f32> = t
            .as_slice()
            .expect("test: contiguous")
            .iter()
            .map(|v| v.to_f32())
            .collect();
        Tensor::new(data, t.shape()).unwrap()
    }
    fn round_bf16(t: &Tensor<f32>) -> Tensor<bf16> {
        let data: Vec<bf16> = t
            .as_slice()
            .expect("test: contiguous")
            .iter()
            .map(|&v| bf16::from_f32(v))
            .collect();
        Tensor::new(data, t.shape()).unwrap()
    }

    /// forward（F16・bias・ReLU あり）が `TypedOps<f16>` の丸め契約
    /// （`downcast → gemm → add → relu → upcast`。各演算後に f16 へ
    /// 丸める）と bit 完全一致することを検証する。`x`／`w`／`bias` は
    /// f16 で厳密表現できない値（0.1・0.3 等）を含み、丸めが実際に
    /// パイプライン全体で適用されることを確認する。
    #[test]
    fn linear_act_low_precision_f16_forward_matches_rounding_oracle() {
        let tape = Tape::new_with_ops(Box::new(ComputingLowPrecisionBackendOps));
        let x_data = Tensor::new(vec![0.1f32, 0.2, -0.3, 0.4], &[2, 2]).unwrap();
        let w_data = Tensor::new(vec![0.3f32, 0.7, -0.5, 0.6], &[2, 2]).unwrap();
        let bias_data = Tensor::new(vec![0.05f32, -0.02], &[2]).unwrap();

        let x = tape.var(&x_data);
        let w = tape.var(&w_data);
        let bias = tape.var(&bias_data);

        let out = x
            .linear_act_low_precision(
                &w,
                Some(&bias),
                Activation::Relu,
                fandhe_ai_tensor_core::ScalarDType::F16,
            )
            .expect("f16 forward は mock TypedOps<f16> で成功するはず");

        // 独立オラクル: `linear_forward_typed`（`crates/tensor-core/src/
        // low_precision.rs`）と同じ手順を、Var 経由ではなく直接
        // `ComputingLowPrecisionBackendOps` の f32 ヘルパーと `half::f16`
        // の丸めのみで再現する。
        let x_f16 = round_f16(&x_data);
        let w_f16 = round_f16(&w_data);
        let bias_f16 = round_f16(&bias_data);
        let y = round_f16(&ComputingLowPrecisionBackendOps::gemm_f32(
            &upcast_f16(&x_f16),
            &upcast_f16(&w_f16),
        ));
        let y = round_f16(&ComputingLowPrecisionBackendOps::add_f32(
            &upcast_f16(&y),
            &upcast_f16(&bias_f16),
        ));
        let y = round_f16(&ComputingLowPrecisionBackendOps::relu_f32(&upcast_f16(&y)));
        let expected = upcast_f16(&y);

        assert_eq!(
            out.value().as_slice().unwrap(),
            expected.as_slice().unwrap(),
            "linear_act_low_precision(F16) の forward は TypedOps<f16> の \
             丸め契約（各演算後に f16 へ丸める）と bit 完全一致するはず"
        );
    }

    /// 上記 F16 版の Bf16 対（bias・活性化なしの単純ケース）。`typed_ops_bf16`
    /// accessor 経由の成功経路と丸め契約を確認する。
    #[test]
    fn linear_act_low_precision_bf16_forward_matches_rounding_oracle() {
        let tape = Tape::new_with_ops(Box::new(ComputingLowPrecisionBackendOps));
        let x_data = Tensor::new(vec![0.1f32, 0.2, -0.3, 0.4], &[2, 2]).unwrap();
        let w_data = Tensor::new(vec![0.3f32, 0.7, -0.5, 0.6], &[2, 2]).unwrap();

        let x = tape.var(&x_data);
        let w = tape.var(&w_data);

        let out = x
            .linear_act_low_precision(
                &w,
                None,
                Activation::None,
                fandhe_ai_tensor_core::ScalarDType::Bf16,
            )
            .expect("bf16 forward は mock TypedOps<bf16> で成功するはず");

        let x_bf16 = round_bf16(&x_data);
        let w_bf16 = round_bf16(&w_data);
        let y = round_bf16(&ComputingLowPrecisionBackendOps::gemm_f32(
            &upcast_bf16(&x_bf16),
            &upcast_bf16(&w_bf16),
        ));
        let expected = upcast_bf16(&y);

        assert_eq!(
            out.value().as_slice().unwrap(),
            expected.as_slice().unwrap(),
            "linear_act_low_precision(Bf16) の forward は TypedOps<bf16> の \
             丸め契約と bit 完全一致するはず"
        );
    }

    /// backward の ReLU マスクが低精度 forward 出力の符号（`out_value >
    /// 0.0`）で決まることを検証する（`grad::vjp` の `Op::LinearAct`
    /// 分岐）。`x`／`w` は f16 で厳密表現できる値（0.5・1.0 は 2 の冪）を
    /// 選び、丸めによる寄与を排除してマスク判定のみを分離して確認する。
    #[test]
    fn linear_act_low_precision_backward_relu_mask_uses_low_precision_output_sign() {
        let tape = Tape::new_with_ops(Box::new(ComputingLowPrecisionBackendOps));
        let x_data = Tensor::new(vec![0.5f32, 0.5, -0.5, -0.5], &[2, 2]).unwrap();
        let w_data = Tensor::new(vec![1.0f32, 1.0], &[2, 1]).unwrap();
        let x = tape.var(&x_data);
        let w = tape.var(&w_data);

        let out = x
            .linear_act_low_precision(
                &w,
                None,
                Activation::Relu,
                fandhe_ai_tensor_core::ScalarDType::F16,
            )
            .unwrap();
        // 1 行目: 0.5+0.5=1.0（relu 通過）／2 行目: -0.5-0.5=-1.0 →
        // relu(-1.0)=0.0（masked）。
        assert_eq!(out.value().as_slice().unwrap(), &[1.0f32, 0.0f32]);

        let loss = out.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let d_x = grads
            .get(&x)
            .unwrap()
            .expect("x へ backward が到達するはず");
        let d_w = grads
            .get(&w)
            .unwrap()
            .expect("w へ backward が到達するはず");

        // upstream（sum の勾配）は全要素 1.0 だが、relu マスクにより
        // masked 行（2 行目。低精度 forward 出力が 0.0 だった行）の
        // 寄与はゼロになる。
        assert_eq!(
            d_x.contiguous().as_slice().unwrap(),
            &[1.0f32, 1.0, 0.0, 0.0],
            "低精度 forward の ReLU マスクは backward 側の f32 VJP にも \
             正しく伝播するはず（2 行目は masked=0）"
        );
        assert_eq!(
            d_w.contiguous().as_slice().unwrap(),
            &[0.5f32, 0.5],
            "d_weight は masked 行（2 行目）を除いた寄与のみを持つはず \
             （1 行目 x=[0.5,0.5] の寄与のみ）"
        );
    }

    /// backward（`matmul_vjp`）が forward で丸めた低精度値ではなく f32
    /// master 値（`x`／`weight` そのもの）を使うことを検証する。`0.1`
    /// ／`0.2`／`0.3`／`0.4` は f16 で厳密表現できない値であり、backward
    /// が f16 丸め後の値を使っていれば本テストの期待値とは異なる結果に
    /// なる（`half::f16::from_f32(0.1).to_f32() != 0.1` を前提として
    /// 明示検査する）。
    #[test]
    fn linear_act_low_precision_backward_uses_f32_master_values_not_rounded() {
        assert_ne!(
            f16::from_f32(0.1f32).to_f32(),
            0.1f32,
            "本テストの前提（0.1 は f16 で厳密表現できない）が崩れている"
        );

        let tape = Tape::new_with_ops(Box::new(ComputingLowPrecisionBackendOps));
        let x_data = Tensor::new(vec![0.1f32, 0.2], &[1, 2]).unwrap();
        let w_data = Tensor::new(vec![0.3f32, 0.4], &[2, 1]).unwrap();
        let x = tape.var(&x_data);
        let w = tape.var(&w_data);

        let out = x
            .linear_act_low_precision(
                &w,
                None,
                Activation::None,
                fandhe_ai_tensor_core::ScalarDType::F16,
            )
            .unwrap();
        let loss = out.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let d_x = grads
            .get(&x)
            .unwrap()
            .expect("x へ backward が到達するはず");
        let d_w = grads
            .get(&w)
            .unwrap()
            .expect("w へ backward が到達するはず");

        // upstream（sum の勾配）は 1.0（out は [1,1] スカラー）。
        // d_input = g @ w^T = w の f32 master 値そのもの・
        // d_weight = x^T @ g = x の f32 master 値そのもの（k=1 の自明な
        // 乗算のため bit 完全一致で検証できる）。
        assert_eq!(
            d_x.contiguous().as_slice().unwrap(),
            &[0.3f32, 0.4],
            "d_input は w の f32 master 値（丸め前）と bit 完全一致するはず"
        );
        assert_eq!(
            d_w.contiguous().as_slice().unwrap(),
            &[0.1f32, 0.2],
            "d_weight は x の f32 master 値（丸め前）と bit 完全一致するはず"
        );
    }
}

#[cfg(test)]
mod host_view_tests {
    use super::*;
    use crate::tape::Tape;

    /// P1 是正の回帰テスト（イシュー #1335 codex-review 指摘）: `host_view()`
    /// が返す `VarHostView` を保持したまま同じ `Tape` へノード追加演算
    /// （`add`）を呼んでも `RefCell` の二重可変借用 panic が起きないこと
    /// を確認する。是正前は `Ref<'a, [f32]>` をそのまま保持していたため
    /// `x.add(&x)` の `push_lazy` 内 `borrow_mut()` が panic していた
    /// （指摘の再現手順そのもの）。
    #[test]
    fn host_view_does_not_panic_when_tape_op_follows() {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap());

        let view = x.host_view();
        // `view` 生存中に同じ `Tape` へノード追加演算を呼ぶ（是正前は panic）。
        let result = x.add(&x).expect("同一 shape の加算は成功するはず");
        drop(view);

        assert_eq!(
            result.to_tensor().as_slice().unwrap(),
            &[2.0, 4.0, 6.0, 8.0],
            "host_view() 生存中の add は通常どおりの結果を返すはず"
        );
    }

    /// 非 contiguous（`transpose` 後）な `Var` でも `host_view()` が
    /// `Tape` の借用を持ち越さないことを確認する（`contiguous()` 分岐の
    /// 回帰カバレッジ）。
    #[test]
    fn host_view_on_transposed_var_does_not_panic_when_tape_op_follows() {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap());
        let xt = x.transpose(0, 1).expect("transpose は 2 次元で成功する");

        let view = xt.host_view();
        let result = x
            .add(&x)
            .expect("transpose 済み view の生存中でも add は成功するはず");
        drop(view);

        assert_eq!(
            result.to_tensor().as_slice().unwrap(),
            &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        );
    }
}

#[cfg(test)]
mod batch_norm_empty_axis_huge_spatial_tests {
    use super::*;
    use crate::tape::Tape;

    /// Cursor Bugbot 指摘（PR #1874・イシュー #1732 fix ループ）の回帰
    /// テスト: rank-4 `[N=0, C, H=usize::MAX, W=2]` は `H*W` 単体では
    /// `usize` オーバーフローするが、`N=0` により `x` 全体は要素数 0 の
    /// 有効な空テンソルである。`batch_norm_layout`（`ops_shape.rs`）の
    /// 修正により `ElementCountOverflow` で誤って拒否されず、
    /// `batch_norm_infer`（eval モード）は空出力を返すことを確認する。
    #[test]
    fn batch_norm_infer_accepts_empty_leading_axis_with_huge_spatial() {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(Vec::<f32>::new(), &[0, 3, usize::MAX, 2]).unwrap());
        let mean = Tensor::new(vec![0.0f32; 3], &[3]).unwrap();
        let var = Tensor::new(vec![1.0f32; 3], &[3]).unwrap();

        let out = x
            .batch_norm_infer(None, None, &mean, &var, 1e-5)
            .expect("N=0 の空テンソルは ElementCountOverflow にならず成功するはず");
        assert_eq!(out.shape(), &[0, 3, usize::MAX, 2]);
        assert_eq!(out.to_tensor().as_slice().unwrap(), &[] as &[f32]);
    }

    /// `C=0` 版の同型ケース（`N`／空間軸は非ゼロかつ空間軸が巨大）。
    #[test]
    fn batch_norm_infer_accepts_empty_channel_axis_with_huge_spatial() {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(Vec::<f32>::new(), &[5, 0, usize::MAX, 2]).unwrap());
        let mean = Tensor::new(Vec::<f32>::new(), &[0]).unwrap();
        let var = Tensor::new(Vec::<f32>::new(), &[0]).unwrap();

        let out = x
            .batch_norm_infer(None, None, &mean, &var, 1e-5)
            .expect("C=0 の空テンソルは ElementCountOverflow にならず成功するはず");
        assert_eq!(out.shape(), &[5, 0, usize::MAX, 2]);
        assert_eq!(out.to_tensor().as_slice().unwrap(), &[] as &[f32]);
    }

    /// train モード（`batch_norm_with_batch_stats`）は同じ空テンソルに
    /// 対し `ElementCountOverflow` ではなく、`M=n*spatial<=1`
    /// （`n=0` の場合 `M=0`）を理由とする型付きエラーで拒否されること
    /// を確認する（`batch_norm_layout` 修正後は `spatial=0` となり
    /// `M<=1` 拒否経路へ正しく到達する）。
    #[test]
    fn batch_norm_train_rejects_empty_leading_axis_with_m_le_1_not_overflow() {
        let tape = Tape::new();
        let x = tape.var(&Tensor::new(Vec::<f32>::new(), &[0, 3, usize::MAX, 2]).unwrap());

        let err = x
            .batch_norm_with_batch_stats(None, None, 1e-5)
            .expect_err("train モードは M<=1 で拒否されるはず");
        assert!(
            matches!(err, AutodiffError::InvalidArgument(_)),
            "ElementCountOverflow ではなく M<=1 の InvalidArgument であるべき: {err:?}"
        );
    }

    // 4 軸すべてが非ゼロの場合に `ElementCountOverflow` が維持される
    // ことの回帰は `tensor-core::ops_shape::batch_norm_layout_tests::
    // rank4_nonempty_spatial_overflow_is_still_rejected` が担う
    // （`Var::batch_norm_infer` レベルでは `Tensor::new` 自体が
    // `checked_numel`〈shape 全体積〉で先に同じ `ElementCountOverflow`
    // を返すため、この層で意味のある `Tensor` を構築できない）。
}
