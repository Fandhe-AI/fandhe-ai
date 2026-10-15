//! 子テープ方式の高階微分（`create_graph`。イシュー #1942・親 #1940・
//! 設計 `docs/autodiff-higher-order-grad-decision.md` §7〜§9）。
//!
//! [`Tape::backward_create_graph`] は既存の 1 階 [`Tape::backward`]
//! （`backward.rs::backward_impl`。無変更）を呼んで 1 階勾配
//! （[`CreateGraphResult::first_order`]）を得たうえで、**呼び出し側が
//! あらかじめ構築した別インスタンスの空 `Tape`**（`child`）へ、
//! `loss` から到達する祖先ノードの forward 値を「写し」（mirror。
//! [`build_mirror`]）として登録し、その写しを使って VJP の各ステップを
//! `Var` 演算として記録する（[`build_cgrads`]）。子テープ上に記録された
//! 1 階勾配（[`CreateGraphResult::grad`] が返す `Var<'c>`）は通常の
//! `Var` と同じくさらに `child.backward(..)` で微分できるため、これが
//! 二階微分（grad of grad）の実体となる。
//!
//! **既存契約への影響（設計 doc §3「契約整理」）**: 1 階
//! `backward_impl`・`grad.rs`・REQ-2 統一複合判定・tolerance／baseline
//! は一切変更しない——本モジュールは `Tape::backward`（公開 API）を
//! 素のまま呼ぶのみで、`backward.rs` へのコード変更を伴わない（`grad.rs` は #3077 の
//! V3 用にホスト関数を `pub(crate)` 化し LogSoftmax の dispatch を共有
//! ヘルパーへ移しただけで 1 階の挙動は不変）。二階側の数値方式（本モジュールが子テープへ記録する `Var`
//! 演算列）は既存 VJP ヘルパー（`grad.rs`）とは独立の実装であり、
//! 一般には bit 同一を主張しない——正しさは有限差分突合
//! （`tests/create_graph.rs`）で検証する。ただし broadcast 縮約
//! （`reduce_to`）は例外で、`grad.rs::reduce_to_shape` の f32 逐次和
//! アルゴリズムを `Var::narrow`／`Var::add` の連鎖として逐語再現して
//! おり、1 階 VJP と数値方式（縮約順序・アキュムレータ精度）が一致
//! する（`reduce_to` のドキュメント参照。codex-review 指摘・PR
//! #1998 で是正——当初 `Var::sum_dims`〈CPU `sum` の `f64` アキュム
//! レータ縮約〉を使っていたため、桁落ちを伴う broadcast 入力で 1 階
//! 勾配と乖離し REQ-2 統一複合判定を満たさない具体例があった）。
//! `Op::Add` の bias パターン（`upstream: [m, n]` → `[n]`／`[1, n]`
//! の行方向縮約）に限っては [`reduce_bias_grad_var`] が
//! `grad.rs::reduce_bias_grad`（f64 相当のアキュムレータ。2026-09-12
//! ユーザー承認・`.claude/rules/coding-rust.md` の勾配長軸縮約契約）
//! と数値方式を揃える別経路を使う——`eval::reduce_bias_grad_rows`
//! （1 階 VJP と同一のホスト関数）でホスト側の値を直接計算し
//! `Tape::push_eager` で子テープへ登録する（`Var::sum` は
//! `child.ops()` の実装〈テスト用 `naive_ops()` は f32 逐次和〉に
//! 依存するため使わない。codex-review 指摘・PR #1998 是正）。
//!
//! REQ-12「利用者向け融合制御 API を提供しない」は、`create_graph` が
//! 二階の勾配グラフを構築するかどうかの選択であり `docs/
//! fusion-graph-design.md` §3.3 の融合境界制御（forward の elementwise
//! 遅延グラフ）とは別軸のため抵触しない（設計 doc §9「REQ-12 整理」。
//! 子テープに記録された `Var` 演算自体は通常どおり融合対象になりうる）。
//!
//! **facade 公開済み**（イシュー #2545・決定記録 §17）: facade は
//! `CreateGraphResult` を crate ルートで再エクスポートし、facade `Tape` に
//! 本メソッドへの薄い委譲 `Tape::backward_create_graph` を 1 件だけ持つ
//! （利用者は `fandhe_ai::tape()`／`tape_for(..)` で作った空の `Tape` を
//! 子テープとして渡す）。`TapeRef` 版・子テープ構築ヘルパー・HVP 専用 API
//! は公開していない（§17.3。`crates/facade/tests/api_surface.rs` の正ガード
//! が固定）。
//!
//! **対象スコープ（設計 doc §8。`Op::supports_create_graph` が判定する
//! 対象）**: `Leaf`・`Add`・`Mul`・`Relu`・`Exp`・`Tanh`・`Sigmoid`・
//! `Sum`・`Mean`・`Reshape`・`BroadcastTo`・`MatMul`（rank 2 × rank 2
//! 限定。#1943）の 12 variant に加え、イシュー #2062 で
//! `ScalarUnary`（#3081 で `Gelu`・`GeluTanh`・`Erf` も対象）・`ScalarBinary`（既知 13
//! variant すべて）・`Transpose`・`Permute`・`Narrow`・`Concat`・
//! `Contiguous`・`Where`・`RmsNorm`・`LayerNorm`（#3078。`create_graph_norm`
//! モジュール doc 参照）を拡張済み（[`scalar_unary_replayable`]／
//! [`scalar_binary_replayable`] 参照）。さらにイシュー #3080 で 2D 畳み込み族
//! （`Conv2d`・`ConvTranspose2d`・`Unfold`・`Fold`。実装は `conv` サブモジュール、
//! 決定記録 §23）を拡張済み。`F32` 以外の `compute_dtype` を持つ `Conv2d`
//! （低精度 forward）は [`validate_ancestors`] が明示拒否する。`Conv3d`・
//! `ConvTranspose3d` は引き続き対象外（#3080(b)。fail-closed）。イシュー #3083 で索引・累積族
//! （`Gather`・`Scatter`・`Embedding`・`Sort`・`Topk`・`Cummax`・`Cummin`・`Cumsum`。実装は
//! `index_cum` サブモジュール、決定記録 §25）を拡張済み。`IndexedScatterReduce`・`Cumprod`・
//! `Logcumsumexp` は引き続き対象外（fail-closed）。`MaskedFill`・`Pad` は引き続き対象外。
//! イシュー #3086（決定記録 §29）で FFT 4 件（`Rfft`・`Irfft`・`Fft`・`Ifft`）と線形代数 4 件
//! （`Inv`・`Solve`・`Det`・`SlogdetLogAbsDet`。実装は `linalg_fft` サブモジュール）を拡張済み。
//! `Stft`・`Istft`・`Cholesky`・`Pinv`・`Lstsq`・`MatrixNorm`・`Qr*`・`Svd*`・`Eigh*`・
//! `SlogdetSign`・`MatrixRank` の 15 件は fail-closed に固定した。
//! 損失 Op は `MseLoss`・`L1Loss`・`HuberLoss`・`CrossEntropyLoss`・`NllLoss` の 5 variant を
//! イシュー #3082（決定記録 §26。V1／V2 方式）で対象にした。残る損失は
//! `CrossEntropyLossWithOptions`（イシュー #2166。同型の理由で対象外）・
//! `CosineEmbeddingLoss`・`MarginRankingLoss`・`TripletMarginLoss`・
//! `PoissonNllLoss`（イシュー #2167。同型の理由で対象外）・
//! `CtcLoss`（イシュー #2168。同型の理由で対象外）・
//! `BceWithLogitsPosWeightLoss`・`HingeEmbeddingLoss`・`SoftMarginLoss`・
//! `GaussianNllLoss`（イシュー #2652。同型の理由で対象外）・
//! `MultiMarginLoss`・`MultiLabelMarginLoss`・`MultiLabelSoftMarginLoss`・
//! `SigmoidFocalLoss`（イシュー #2653。同型の理由で対象外）
//! は引き続き対象外（後続イシューへ引き継ぐ）。`BatchNorm`（train／eval。
//! イシュー #3079）は対象——方式は `create_graph_norm.rs` と決定記録 §28。それ以外の追跡対象 Op・rank≥3 の
//! `MatMul` へ到達した場合は `Err(AutodiffError::Backward)`
//! （fail-closed。後続イシューへ引き継ぐ）。`resident`／`fused` 経路
//! （`ResidentLeaf`／`LinearResident`／`LinearAct`）・checkpoint 済み
//! 親テープも同様に fail-closed で拒否する（[`validate_ancestors`]。
//! `child` へ一切書き込む前の入口で判定するため、途中失敗時も
//! `child` は無変更のまま保たれる）。
//!
//! **`MatMul` の数値契約（イシュー #1943・PR #2003 codex-review 指摘
//! で是正）**: 子テープの matmul VJP（[`build_cgrads`] の
//! `Op::MatMul` 腕）は [`Var::matmul_fp32_strict`]（`ops().
//! gemm_fp32_strict` 経由。`crate::var` 限定公開）を使い、1 階
//! `matmul_vjp`（`grad.rs`。同じく `ops.gemm_fp32_strict` 経由）と
//! 入口を揃えている——当初案の `Var::matmul`（`ops().gemm`）は CUDA
//! TF32 opt-in（`docs/cuda-tf32-optin-api-decision.md`）が有効な間、
//! 1 階 `matmul_vjp` が守る「バックプロパゲーションは常に FP32 厳密」
//! という契約を二階微分の記録経路でだけ破ってしまうため、`Var::
//! matmul_fp32_strict` へ切り替えた。CPU バックエンドは両者が同一
//! カーネルへ帰着するため bit 同一のまま不変。さらに、
//! `matmul_fp32_strict` が記録する `Op::MatMul` ノードは
//! `TapeNode::fp32_strict` フラグにより activation checkpointing
//! （`Tape::checkpoint`／`Var::checkpoint_from`）の解放対象からも
//! 除外される（`release_checkpoint_region` doc 参照）——`Op::MatMul`
//! variant 自体は forward 精度の情報を持たないため、解放後の再計算
//! （`recompute_value`）が非厳密な `matmul_forward`（`ops.gemm`）を
//! 使ってしまう事故を防ぐ。本モジュールは元々「子テープの数値方式は
//! 一般に bit 同一を主張せず、正しさは有限差分突合で検証する」立場
//! （本ファイル冒頭の既存契約節）を取っており、この整理はその範囲内
//! に収まる（tolerance／baseline は無変更）。

use fandhe_ai_tensor_core::{ScalarBinaryOp, ScalarUnaryOp, Tensor, batch_norm_layout};

use crate::backward::Gradients;
use crate::error::AutodiffError;
use crate::tape::{NodeId, Op, Tape, TapeId, TapeNode, materialize_fallible};
use crate::var::Var;

mod conv;
mod index_cum;
mod linalg_fft;

/// [`Op::supports_create_graph`] が `Op::ScalarUnary` へ委譲する判定
/// （イシュー #2062）。`replay_op`（forward 再生。`Var::scalar_unary`
/// への薄い委譲で全 variant 対応）とは異なり、[`build_cgrads`] の VJP
/// 合成は variant ごとに手書きのため、ここで対応済み variant のみを
/// 許可する。`Gelu`（誤差関数版）・`GeluTanh`・`Erf` はイシュー #3081
/// （決定記録 §21.5・§22）で対象へ追加した。`Gelu` の導関数は `Φ(x)` に
/// `erf` を要するため `Erf` を `Var::scalar_unary` で子テープへ記録し、
/// その `Erf` 自体も孫テープでの再生（閉包性）のため同時に対象にする。
/// `GeluTanh` は飽和域の死に枝を安全化して `NaN` を避ける。`ScalarUnaryOp` は `tensor-core` 側で
/// `#[non_exhaustive]` のため、末尾のワイルドカードは未知 variant を
/// 安全側の `false` へ倒す（`tape.rs::supports_create_graph` doc
/// 「例外」参照）。イシュー #2649 の `Selu`／`Celu`／`Softsign`／
/// `Hardsigmoid`／`LogSigmoid` も対象外で `false`（高階微分は型付きエラーで拒否）。
pub(crate) fn scalar_unary_replayable(op: ScalarUnaryOp) -> bool {
    matches!(
        op,
        ScalarUnaryOp::Neg
            | ScalarUnaryOp::Abs
            | ScalarUnaryOp::Sqrt
            | ScalarUnaryOp::Log
            | ScalarUnaryOp::Log2
            | ScalarUnaryOp::Log10
            | ScalarUnaryOp::Sin
            | ScalarUnaryOp::Cos
            | ScalarUnaryOp::Tan
            | ScalarUnaryOp::Relu
            | ScalarUnaryOp::Exp
            | ScalarUnaryOp::Tanh
            | ScalarUnaryOp::Sigmoid
            | ScalarUnaryOp::Silu
            | ScalarUnaryOp::Hardswish
            | ScalarUnaryOp::LeakyRelu { .. }
            | ScalarUnaryOp::Elu { .. }
            | ScalarUnaryOp::Softplus { .. }
            | ScalarUnaryOp::Clamp { .. }
            | ScalarUnaryOp::PowScalar { .. }
            | ScalarUnaryOp::Gelu
            | ScalarUnaryOp::GeluTanh
            | ScalarUnaryOp::Erf
    )
}

/// [`Op::supports_create_graph`] が `Op::ScalarBinary` へ委譲する判定
/// （イシュー #2062）。既知 13 variant（`Add`〜`Ne`。#2634 で追加した `Atan2` は
/// 対象外で `false`）はすべて `Var` 演算で VJP を
/// 合成できるため `true`（[`build_cgrads`] の `Op::ScalarBinary` 腕
/// 参照）。`ScalarBinaryOp` も `#[non_exhaustive]` のため末尾ワイルド
/// カードは未知 variant を `false` へ倒す。
pub(crate) fn scalar_binary_replayable(op: ScalarBinaryOp) -> bool {
    matches!(
        op,
        ScalarBinaryOp::Add
            | ScalarBinaryOp::Sub
            | ScalarBinaryOp::Mul
            | ScalarBinaryOp::Div
            | ScalarBinaryOp::Pow
            | ScalarBinaryOp::Maximum
            | ScalarBinaryOp::Minimum
            | ScalarBinaryOp::Gt
            | ScalarBinaryOp::Ge
            | ScalarBinaryOp::Lt
            | ScalarBinaryOp::Le
            | ScalarBinaryOp::Eq
            | ScalarBinaryOp::Ne
    )
}

/// [`Tape::backward_create_graph`] の戻り値。1 階勾配
/// （[`Self::first_order`]。既存 `backward_impl` の無変更な結果）に
/// 加え、子テープ `'c` 上に構築した「元テープの祖先ノードの写し」
/// （[`Self::child_var`]）と「その 1 階勾配（子テープ上の `Var`）」
/// （[`Self::grad`]）を保持する。
#[derive(Debug)]
pub struct CreateGraphResult<'c> {
    first_order: Gradients,
    parent_id: TapeId,
    parent_epoch: u64,
    /// 元テープの `NodeId.0` を添字とする、子テープ上の写し
    /// （祖先ノードのみ `Some`）。
    mirror: Vec<Option<Var<'c>>>,
    /// 同じ添字で、子テープ上に構築した 1 階勾配 `Var`（loss から
    /// 到達した対象ノードのみ `Some`）。
    cgrads: Vec<Option<Var<'c>>>,
}

impl<'c> CreateGraphResult<'c> {
    /// 既存 `Tape::backward` と同一の 1 階勾配（`Tensor<f32>` の
    /// 入れ物）。子テープ構築とは独立のパス（`Tape::backward` をその
    /// まま呼ぶのみ）で計算するため、単体で `backward` を呼んだ場合と
    /// bit 同一（設計 doc §9「1 階勾配の bit 同一性」選択肢 (i)。
    /// 推奨として本イシューで採用）。
    pub fn first_order(&self) -> &Gradients {
        &self.first_order
    }

    /// `parent_var`（元テープ上の `Var`）の 1 階勾配を、子テープ上の
    /// `Var<'c>` として返す（これをさらに `child.backward(..)` で
    /// 微分すれば二階勾配が得られる）。
    ///
    /// - `parent_var` が本 `CreateGraphResult` を生んだ親テープ・世代と
    ///   一致しない場合: `Err(TapeMismatch)`。
    /// - `parent_var` が `requires_grad == false`（`var_no_grad`／
    ///   `detach` の葉、またはそれのみを祖先に持つノード）の場合:
    ///   `Err(GradientTrackingDisabled)`（`Gradients::get` と同じ区別。
    ///   `docs/autodiff-nograd-leaf-dinput-skip-decision.md` §5）。
    /// - `loss` から未到達、または対象 Op が
    ///   `Op::supports_create_graph` を満たさない部分木の外側にある
    ///   場合: `Ok(None)`。
    pub fn grad(&self, parent_var: &Var<'_>) -> Result<Option<Var<'c>>, AutodiffError> {
        self.check(parent_var)?;
        Ok(self.cgrads.get(parent_var.node_id().0).copied().flatten())
    }

    /// `parent_var` の子テープ上の写し（forward 値の再生・または
    /// 定数葉化）を返す。`cg.grad(&cg.child_var(&x)?.unwrap())` の
    /// ように、子テープ上でさらに `backward` した [`Gradients`] から
    /// 二階勾配を読み出す際の鍵として使う。
    ///
    /// 検査は [`Self::grad`] と同一。
    pub fn child_var(&self, parent_var: &Var<'_>) -> Result<Option<Var<'c>>, AutodiffError> {
        self.check(parent_var)?;
        Ok(self.mirror.get(parent_var.node_id().0).copied().flatten())
    }

    /// 元テープの `NodeId.0` を添字とする子テープ上の写し一覧（祖先ノードのみ `Some`）。
    /// `functional_ops::jvp_create_graph` が 2 段の写し（親→子→孫）を合成するために使う
    /// crate 内部アクセサで、公開面は増えない（#3067・設計記録 §31.2）。
    pub(crate) fn mirror(&self) -> &[Option<Var<'c>>] {
        &self.mirror
    }

    fn check(&self, parent_var: &Var<'_>) -> Result<(), AutodiffError> {
        if parent_var.tape_id() != self.parent_id || parent_var.tape_epoch() != self.parent_epoch {
            return Err(AutodiffError::TapeMismatch);
        }
        if !parent_var.requires_grad() {
            return Err(AutodiffError::GradientTrackingDisabled);
        }
        Ok(())
    }
}

impl Tape {
    /// 子テープ方式の `create_graph`（PyTorch
    /// `torch.autograd.grad(..., create_graph=True)` 相当。イシュー
    /// #1942）。`loss` を起点に逆伝播しつつ、
    /// `Op::supports_create_graph` を満たす祖先ノードの VJP を
    /// `child`（呼び出し側があらかじめ [`Tape::new_with_ops`] 等で
    /// 構築した**空**の `Tape`）上へ `Var` 演算として記録する。
    ///
    /// **入口検査（順序固定）**:
    /// 1. `loss` が `self` に属さない → `Err(TapeMismatch)`。
    /// 2. `child` が `self` と同一テープ → `Err(Backward)`
    ///    （親と子を同一インスタンスにはできない）。
    /// 3. `child.device() != self.device()` → `Err(DeviceMismatch)`。
    /// 4. `child` が空でない → `Err(Backward)`（子テープの葉プレ
    ///    フィックス契約〈`Tape::reset` doc〉を素直に保つため、既存
    ///    ノードを持つテープの再利用は許さない）。
    /// 5. `self` に checkpoint 区間が登録済み → `Err(Backward)`
    ///    （`Tape::has_registered_checkpoints` doc 参照）。
    /// 6. `loss` から到達する祖先ノードを `collect_ancestors` で
    ///    走査し、`validate_ancestors` で resident／fused 経路・
    ///    未対応 Op・rank≥3 の `MatMul` を事前拒否する
    ///    （`Err(AutodiffError::Backward)`）。**素の [`Tape::backward`]
    ///    より前に行う**——resident グラフ（`Op::ResidentLeaf`／
    ///    `Op::LinearResident`）に対しては素の `backward` 自体が
    ///    `AutodiffError::InvalidArgument`（`DeviceParamStore::
    ///    backward` を使えという誤誘導的なメッセージ）を返してしまう
    ///    ため、先に構造的検査で正確な型付きエラーを返す
    ///    （イシュー #1943）。
    /// 7. 上記を満たせば既存 [`Tape::backward`] をそのまま呼んで 1 階
    ///    勾配を得る（`loss.requires_grad() == false` の拒否もここで
    ///    既存どおり発生する）。
    ///
    /// 4〜7 のいずれかで失敗した場合、`child` は空のまま残る（4 の事前検査により
    /// `child` の空性が確認済みであること、6 が `build_mirror`／`build_cgrads` より前に
    /// 走ることに加え、`build_mirror`／`build_cgrads` の途中失敗では積んだノードを
    /// `Tape::rollback_to` で取り除く）。
    pub fn backward_create_graph<'c>(
        &self,
        loss: &Var<'_>,
        child: &'c Tape,
    ) -> Result<CreateGraphResult<'c>, AutodiffError> {
        if loss.tape_id() != self.id {
            return Err(AutodiffError::TapeMismatch);
        }
        if child.id == self.id {
            return Err(AutodiffError::Backward(
                "create_graph: 子テープに親と同一の Tape は指定できない".into(),
            ));
        }
        if child.device() != self.device() {
            return Err(AutodiffError::DeviceMismatch {
                requested: child.device(),
                actual: self.device(),
            });
        }
        if !child.is_empty() {
            return Err(AutodiffError::Backward(
                "create_graph: 子テープは空である必要がある（既存ノードを持つ Tape は渡せない）"
                    .into(),
            ));
        }
        if self.has_registered_checkpoints() {
            return Err(AutodiffError::Backward(
                "create_graph: checkpoint 区間が登録済みの親テープでは create_graph は未対応（\
                 docs/autodiff-higher-order-grad-decision.md §8 参照）"
                    .into(),
            ));
        }

        // `ancestors` の収集・検査（[`validate_ancestors`]）は `self.
        // backward(loss)` より前に行う——resident グラフ（`Op::
        // ResidentLeaf`／`Op::LinearResident`）は素の `Tape::backward`
        // 自体が `AutodiffError::InvalidArgument`（`DeviceParamStore::
        // backward` を使えという誤誘導的なメッセージ）を返してしまう
        // ため、先に構造的な事前検査で「create_graph は resident 経路
        // 非対応」という型付き `Err(Backward)` を返す方が呼び出し側に
        // とって正確（イシュー #1943）。`collect_ancestors`／
        // `validate_ancestors` はいずれも `self`（構造の読み取りのみ）
        // に依存し `backward` の結果を必要としないため、順序を入れ替え
        // ても既存の 1 階勾配計算（[`Self::backward`]。無変更）には
        // 影響しない。
        let ancestors = collect_ancestors(self, loss.node_id());
        validate_ancestors(self, &ancestors)?;

        let first_order = self.backward(loss)?;

        let n = self.nodes.borrow().len();
        let mut mirror: Vec<Option<Var<'c>>> = vec![None; n];
        // `build_mirror`／`build_cgrads` は子テープへノードを積んだ後でも失敗しうる
        // （例: LayerNorm／RmsNorm の upstream 値は子テープ上で初めて分かり、構築の途中で
        // 拒否される。PR #3346 レビュー指摘）。失敗時は積んだノードをすべて取り除いて
        // `child` を空へ戻し、同じ `child` での再試行を可能にする（空性検査を通る）。
        let point = child.rollback_point();
        let built = build_mirror(self, child, &ancestors, &mut mirror)
            .and_then(|()| build_cgrads(self, child, &ancestors, &mirror, loss.node_id()));
        let cgrads = match built {
            Ok(c) => c,
            Err(e) => {
                mirror.clear();
                child.rollback_to(point);
                return Err(e);
            }
        };

        Ok(CreateGraphResult {
            first_order,
            parent_id: self.id,
            parent_epoch: self.epoch(),
            mirror,
            cgrads,
        })
    }
}

/// `root` から [`Op::for_each_input`] を辿って到達する祖先ノードの
/// `NodeId` を昇順（元テープの発生順。トポロジカル順と一致する——
/// 入力は常に自身より小さい `NodeId` を持つため。`Tape::push_*` の
/// 追記専用構造から導かれる不変条件）で返す。
///
/// **`requires_grad == false` のノードでは descend しない**（設計 doc
/// §8「`var_no_grad`／`detach` 葉との相互作用」）: 当該ノードは
/// [`build_mirror`] が forward 値をそのまま定数葉化するため、その入力
/// 側を祖先集合へ含める必要がない——むしろ含めると、対象外の Op
/// （resident 経路等）がたまたま `requires_grad == false` の部分木に
/// あるだけで `Err` になってしまい、PyTorch の「定数として扱われる
/// 演算は任意の Op でよい」という直感的な挙動を壊す。
fn collect_ancestors(tape: &Tape, root: NodeId) -> Vec<NodeId> {
    let nodes = tape.nodes.borrow();
    let n = nodes.len();
    let mut visited = vec![false; n];
    let mut stack = vec![root];
    let mut ids: Vec<NodeId> = Vec::new();
    while let Some(id) = stack.pop() {
        if visited[id.0] {
            continue;
        }
        visited[id.0] = true;
        ids.push(id);
        let node = &nodes[id.0];
        if node.requires_grad {
            node.op.for_each_input(|input_id| stack.push(input_id));
        }
    }
    ids.sort_unstable_by_key(|id| id.0);
    ids
}

/// [`Tape::backward_create_graph`] の入口検査 7（[`build_mirror`]／
/// [`build_cgrads`] が `child` へ一切書き込む前に呼ぶ純関数。イシュー
/// #1943）。`ancestors`（[`collect_ancestors`] の結果）を走査し、
/// 以下のいずれかに該当するノードがあれば即座に `Err` を返す:
///
/// (a) `Op::ResidentLeaf`／`Op::LinearResident`／`Op::LinearAct`
///     （`requires_grad` の値に関わらず。[`build_mirror`] 段 1 の
///     拒否と同一条件——値の実体化自体が成立しないため）。
/// (b) `requires_grad == true` かつ非葉（`Op::Leaf` でない）で
///     `Op::supports_create_graph() == false`。
/// (d) `requires_grad == true` で `TapeNode::low_precision` が立つノード
///     （Op を問わず。MatMul と elementwise 5 演算。イシュー #2628）。
/// (c) `requires_grad == true` の `Op::MatMul` で、いずれかの入力
///     shape の rank が 2 でない（子テープの matmul VJP は
///     rank 2 × rank 2 限定。[`Op::supports_create_graph`] の doc
///     参照）。
///
/// **多層防御**: [`build_mirror`]／[`build_cgrads`] 自身も同型の
/// 拒否分岐（(a) は段 1 の手前・(b) は段 2 の `supports_create_graph`
/// 検査）を保持したまま残す——本関数はそれらより前に実行され、失敗
/// 時に `child` へ一部ノードが残る事態（4 の事前検査が保証するのは
/// 「呼び出し開始時点で `child` が空」であって「呼び出し失敗時に
/// `child` が空のまま」ではない）を避けるための追加ゲートである。
fn validate_ancestors(parent: &Tape, ancestors: &[NodeId]) -> Result<(), AutodiffError> {
    let parent_nodes = parent.nodes.borrow();
    for &id in ancestors {
        let node = &parent_nodes[id.0];
        if matches!(
            node.op,
            Op::ResidentLeaf { .. } | Op::LinearResident { .. } | Op::LinearAct { .. }
        ) {
            return Err(AutodiffError::Backward(format!(
                "create_graph: resident／fused 経路の Op（NodeId({}))は非対応（\
                 docs/autodiff-higher-order-grad-decision.md §8 参照）",
                id.0
            )));
        }
        if !node.requires_grad || matches!(node.op, Op::Leaf) {
            continue;
        }
        // 低精度 forward ノード（`TapeNode::low_precision`。MatMul〈イシュー
        // #2071〉と elementwise 5 演算〈イシュー #2628〉）は Op を問わず
        // 拒否する。Op variant 自体は精度情報を持たないため、`replay_op`
        // （`fp32_strict` のみを読む）へ通すと子テープで静かに f32 として
        // 再生され、opt-in が精度について嘘をつく（fail-closed 方針。
        // `.claude/rules/security.md` A04。`crate::low_precision_ops`
        // モジュール doc 参照）。`requires_grad == false` のノードは上の
        // `continue` で対象外——`build_mirror` 段 1 が丸め済みの記録値を
        // そのまま定数葉にするだけで再生しないため精度は後退しない。
        if node.low_precision {
            return Err(AutodiffError::Backward(format!(
                "create_graph: 低精度 forward のノード（NodeId({}))は非対応（\
                 テープへ精度情報を保持しないため子テープへ replay すると静かに \
                 f32 精度へフォールバックする。イシュー #2071・#2628）",
                id.0
            )));
        }
        if !node.op.supports_create_graph() {
            return Err(AutodiffError::Backward(format!(
                "create_graph: 未対応の Op（NodeId({}))へ到達した（\
                 docs/autodiff-higher-order-grad-decision.md §8 の対象 Op のみ再生可能）",
                id.0
            )));
        }
        // FFT 4 件（イシュー #3086）: 未知の `FftNorm` は随伴の `norm` を決められないため、
        // 子テープへ書く前に拒否する（`ScatterReduce` の未知 variant と同じ多層防御）。
        if let Op::Rfft { norm, .. }
        | Op::Irfft { norm, .. }
        | Op::Fft { norm, .. }
        | Op::Ifft { norm, .. } = &node.op
        {
            linalg_fft::check_fft_norm(*norm)?;
        }
        // BatchNorm（イシュー #3079）: 空テンソル（`n`／`c`／`spatial` のいずれかが 0。
        // eval は `M<=1` も通る）は二階 VJP の統計構造（行 `narrow`・`mean`）が
        // 成立しないため、子テープへ書く前に fail-closed で拒否する。
        if let Op::BatchNorm { input, .. } = &node.op
            // 形状の積は `[usize::MAX, 2, 0]` で overflow して panic するため、
            // ゼロ軸の有無で判定する（本番経路 panic 禁止）。
            && parent_nodes[input.0].shape.contains(&0)
        {
            return Err(AutodiffError::Backward(format!(
                "create_graph: 空テンソルの BatchNorm（NodeId({}))は非対応（\
                 docs/autodiff-higher-order-grad-decision.md §28 参照）",
                id.0
            )));
        }
        // 低精度 `Conv2d`（イシュー #2071）は `TapeNode::low_precision` を立てず
        // `compute_dtype` で記録するため、ここで明示拒否する（イシュー #3080・
        // 決定記録 §21.3。子テープへ精度情報を持たず静かに f32 再生される
        // のを防ぐ fail-closed。`child` へ書き込む前の入口検査）。
        if let Op::Conv2d { compute_dtype, .. } = &node.op
            && *compute_dtype != fandhe_ai_tensor_core::ScalarDType::F32
        {
            return Err(AutodiffError::Backward(format!(
                "create_graph: 低精度 conv2d（{compute_dtype:?}。NodeId({}))は非対応（\
                 子テープへ精度情報を持たず静かに f32 再生される。イシュー #2071・#3080）",
                id.0
            )));
        }
        // `ScatterReduce` は `#[non_exhaustive]`。未知 variant は子テープへ
        // 書き込む前にここで拒否する（イシュー #3083。`child.is_empty()` を保つ）。
        if let Op::Scatter { reduce, .. } = &node.op
            && !matches!(
                reduce,
                fandhe_ai_tensor_core::ScatterReduce::Add
                    | fandhe_ai_tensor_core::ScatterReduce::Overwrite
            )
        {
            return Err(AutodiffError::Backward(format!(
                "create_graph: 未知の ScatterReduce variant {reduce:?}（NodeId({}))は非対応",
                id.0
            )));
        }
        // `Embedding` の `padding_idx` は forward で検査済みだが、cgrad が行マスクを
        // 構築するため入口で範囲を再確認する（境界検査を省略しない。REQ-8）。
        if let Op::Embedding {
            weight,
            padding_idx: Some(p),
            ..
        } = &node.op
        {
            let w_shape = &parent_nodes[weight.0].shape;
            if w_shape.len() != 2 || *p >= w_shape[0] {
                return Err(AutodiffError::Backward(format!(
                    "create_graph: Embedding の padding_idx={p} が weight shape {w_shape:?} \
                     の範囲外（NodeId({}))",
                    id.0
                )));
            }
        }
        if let Op::MatMul(a, b) = node.op {
            let a_rank = parent_nodes[a.0].shape.len();
            let b_rank = parent_nodes[b.0].shape.len();
            if a_rank != 2 || b_rank != 2 {
                return Err(AutodiffError::Backward(format!(
                    "create_graph: rank≥3 の MatMul（NodeId({}))は非対応（\
                     子テープの matmul VJP は rank 2 × rank 2 限定。\
                     docs/autodiff-higher-order-grad-decision.md §14 参照）",
                    id.0
                )));
            }
        }
        // RmsNorm／LayerNorm／BatchNorm の S1 構造が安全に扱えない入力（非有限・overflow 域）を
        // `child` へ書く前に拒否する（イシュー #3078・#3079。`create_graph_norm` モジュール doc）。
        crate::create_graph_norm::validate_norm_s1(&parent_nodes, parent.ops(), &node.op)?;
    }
    Ok(())
}

/// 祖先ノードの「写し」を子テープ上へ構築する（[`Tape::backward_
/// create_graph`] の本体その 1）。**2 段構成**（`Tape::reset` の葉
/// プレフィックス契約を子テープ側でも保つため。設計 doc §9「世代
/// 契約」）:
///
/// 1. 祖先すべてを昇順で走査し、葉相当（`Op::Leaf`、または
///    `requires_grad == false` の任意ノード）を先に子テープの葉として
///    登録する——`requires_grad == false` の任意ノードは、その Op が
///    `resident`／`fused`／未対応のいずれであっても、値を実体化して
///    子テープの `var_no_grad` 葉へ変換するだけで済む（replay 不要。
///    ただし resident／fused 経路〈`ResidentLeaf`／`LinearResident`／
///    `LinearAct`〉は値の実体化自体が成立しないため、`requires_grad`
///    の値に関わらず本段の**手前**で無条件に拒否する）。
/// 2. 残り（`requires_grad == true` かつ非葉）を昇順で走査し、
///    `Op::supports_create_graph` を満たす場合のみ `Var` 演算として
///    再生する。満たさない場合は `Err`（fail-closed。#1943 等へ引き継ぐ
///    未対応 Op）。
fn build_mirror<'c>(
    parent: &Tape,
    child: &'c Tape,
    ancestors: &[NodeId],
    mirror: &mut [Option<Var<'c>>],
) -> Result<(), AutodiffError> {
    let parent_nodes = parent.nodes.borrow();
    let parent_ops = parent.ops();

    // 段 1: resident／fused 経路を拒否しつつ、葉相当を先にすべて登録する。
    for &id in ancestors {
        let node = &parent_nodes[id.0];
        if matches!(
            node.op,
            Op::ResidentLeaf { .. } | Op::LinearResident { .. } | Op::LinearAct { .. }
        ) {
            return Err(AutodiffError::Backward(format!(
                "create_graph: resident／fused 経路の Op（NodeId({}))は非対応（\
                 docs/autodiff-higher-order-grad-decision.md §8 参照）",
                id.0
            )));
        }
        if !node.requires_grad {
            let value = materialize_fallible(&parent_nodes, parent_ops, id)?.clone();
            mirror[id.0] = Some(child.var_no_grad(&value));
            continue;
        }
        if matches!(node.op, Op::Leaf) {
            let value = materialize_fallible(&parent_nodes, parent_ops, id)?.clone();
            mirror[id.0] = Some(child.var(&value));
        }
    }

    // 段 2: 残り（requires_grad == true かつ非葉）を再生する。
    for &id in ancestors {
        if mirror[id.0].is_some() {
            continue;
        }
        let node = &parent_nodes[id.0];
        if !node.op.supports_create_graph() {
            return Err(AutodiffError::Backward(format!(
                "create_graph: 未対応の Op（NodeId({}))へ到達した（\
                 docs/autodiff-higher-order-grad-decision.md §8 の対象 Op のみ再生可能）",
                id.0
            )));
        }
        let replayed = replay_op(node.op.clone(), &node.shape, node.fp32_strict, mirror)?;
        mirror[id.0] = Some(replayed);
    }

    Ok(())
}

/// 子テープ上の写しを取得する（未構築なら内部不変条件違反として
/// `Err`。本番経路 panic 禁止方針〈`.claude/rules/coding-rust.md`〉に
/// 従い `unwrap`/`expect` は使わない）。[`collect_ancestors`]・
/// [`build_mirror`] の不変条件（祖先はトポロジカル順に登録されるため、
/// 非葉ノードの再生・逆走査の時点で入力側の写しは必ず存在する）が
/// 保たれている限り、この `Err` 分岐は実行時には到達しない。
fn get_mirror<'c>(mirror: &[Option<Var<'c>>], id: NodeId) -> Result<Var<'c>, AutodiffError> {
    mirror[id.0].ok_or_else(|| {
        AutodiffError::Backward(format!(
            "create_graph: 祖先ノード（NodeId({}))の写しが未構築（内部不変条件違反）",
            id.0
        ))
    })
}

/// 祖先ノード 1 個分の forward を子テープ上で再生する（[`build_mirror`]
/// 段 2 の本体）。呼び出し元は `op.supports_create_graph()` を確認済み
/// のため、ここでの `_ =>` 到達は契約違反（`AutodiffError::Backward`
/// で fail-closed に拒否する。本番経路 panic 禁止方針に従い `panic!`
/// ではなく型付きエラーとする）。
///
/// `shape` は再生対象ノード自身の `TapeNode::shape`（`Op::Reshape`／
/// `Op::BroadcastTo` の目的 shape はこの `shape` から読む——`Op` 自身は
/// 入力 `NodeId` のみを保持し目的 shape を持たないため）。
///
/// `fp32_strict` は再生対象ノード自身の `TapeNode::fp32_strict`
/// （PR #2003 codex-review 指摘で追加）。`Op::MatMul` variant 自体は
/// forward 精度の情報を持たないため、親テープ側で
/// `Var::matmul_fp32_strict` により記録されたノード（1 階
/// `matmul_vjp` の `d_input`／`d_weight` 等）を再生する際は、子テープ
/// 側でも `Var::matmul_fp32_strict` を使い、通常の `Var::matmul`
/// （`ops.gemm`。CUDA TF32 opt-in が有効な場合は非厳密）へ落ちて
/// 「バックプロパゲーションは常に FP32 厳密」という契約
/// （`.claude/rules/coding-rust.md`・`Var::matmul_fp32_strict` doc）を
/// 二階微分の記録経路でだけ破ることを防ぐ。
fn replay_op<'c>(
    op: Op,
    shape: &[usize],
    fp32_strict: bool,
    mirror: &[Option<Var<'c>>],
) -> Result<Var<'c>, AutodiffError> {
    match op {
        Op::Add(a, b) => get_mirror(mirror, a)?.add(&get_mirror(mirror, b)?),
        Op::Mul(a, b) => get_mirror(mirror, a)?.mul(&get_mirror(mirror, b)?),
        // rank 2 × rank 2 限定（`validate_ancestors` (c) が入口で
        // 事前検査済み）。親ノードが `fp32_strict` フラグを立てて
        // いれば（`Var::matmul_fp32_strict` 由来）子テープ側も
        // `matmul_fp32_strict` で再生し、`ops.gemm_fp32_strict`
        // 経由の厳密精度契約を引き継ぐ。フラグが立っていなければ
        // 通常の `Var::matmul`（`ops.gemm`）で forward と同一カーネル
        // を経由する。
        Op::MatMul(a, b) => {
            let lhs = get_mirror(mirror, a)?;
            let rhs = get_mirror(mirror, b)?;
            if fp32_strict {
                lhs.matmul_fp32_strict(&rhs)
            } else {
                lhs.matmul(&rhs)
            }
        }
        Op::Relu(a) => Ok(get_mirror(mirror, a)?.relu()),
        Op::Exp(a) => Ok(get_mirror(mirror, a)?.exp()),
        Op::Tanh(a) => Ok(get_mirror(mirror, a)?.tanh()),
        Op::Sigmoid(a) => Ok(get_mirror(mirror, a)?.sigmoid()),
        Op::Sum { input, dim } => get_mirror(mirror, input)?.sum(dim),
        Op::Mean { input, dim } => get_mirror(mirror, input)?.mean(dim),
        Op::Reshape { input } => get_mirror(mirror, input)?.contiguous()?.reshape(shape),
        Op::BroadcastTo { input } => get_mirror(mirror, input)?.broadcast_to(shape),
        // `Var::scalar_unary`／`scalar_binary`（`pub(crate)`。同一
        // crate 内のため呼べる）は forward 数式そのものを内包している
        // ため、対応済み variant（`scalar_unary_replayable`／
        // `scalar_binary_replayable` が事前検査済み）であれば variant
        // ごとの分岐なしに再生できる（イシュー #2062）。
        Op::ScalarUnary { op: sop, input } => get_mirror(mirror, input)?.scalar_unary(sop),
        Op::ScalarBinary { op: sop, a, b } => {
            get_mirror(mirror, a)?.scalar_binary(&get_mirror(mirror, b)?, sop)
        }
        Op::Transpose { input, dim0, dim1 } => get_mirror(mirror, input)?.transpose(dim0, dim1),
        Op::Permute { input, perm } => get_mirror(mirror, input)?.permute(&perm),
        Op::Narrow {
            input,
            dim,
            start,
            len,
        } => get_mirror(mirror, input)?.narrow(dim, start, len),
        Op::Concat { inputs, dim } => {
            let mirrored: Result<Vec<Var<'c>>, AutodiffError> =
                inputs.iter().map(|&id| get_mirror(mirror, id)).collect();
            Var::cat(&mirrored?, dim)
        }
        Op::Contiguous { input } => get_mirror(mirror, input)?.contiguous(),
        // forward と同じ dispatch（backend の行カーネル → `Unsupported`
        // ならホスト）を公開 `Var` メソッド経由で通す（イシュー #3077）。
        Op::Softmax { input, dim } => get_mirror(mirror, input)?.softmax(dim),
        Op::LogSoftmax { input, dim } => get_mirror(mirror, input)?.log_softmax(dim),
        // d_weight のホスト値ノード: 孫テープでも同じホスト関数で値を
        // 再計算する（f32 の構造側合成へ落とさない。§21.1(3)(5)）。
        Op::ConvWeightGradHost {
            structure,
            g,
            x,
            params,
            transposed,
        } => {
            let structure_m = get_mirror(mirror, structure)?;
            conv::host_weight_grad_node(
                structure_m.tape(),
                &structure_m,
                &get_mirror(mirror, g)?,
                &get_mirror(mirror, x)?,
                &params,
                transposed,
            )
        }
        // BatchNorm の dx／dw／db ホスト値ノード（イシュー #3079）: 孫テープでも
        // 1 階と同じホスト関数で値を再計算する（f64 縮約の補正を失わない）。
        Op::BatchNormGradHost {
            structure,
            g,
            x,
            weight,
            has_bias,
            eps,
            fixed_stats,
            kind,
        } => {
            let structure_m = get_mirror(mirror, structure)?;
            let w_m = weight.map(|w| get_mirror(mirror, w)).transpose()?;
            crate::create_graph_norm::host_grad_node(
                structure_m.tape(),
                &structure_m,
                &get_mirror(mirror, g)?,
                &get_mirror(mirror, x)?,
                w_m.as_ref(),
                has_bias,
                eps,
                fixed_stats,
                kind,
            )
        }
        // 2D 畳み込み族（イシュー #3080）。公開 `Var` メソッドへの委譲で、
        // forward と同じカーネルを通るため写しの値は親と一致する。
        Op::Conv2d {
            input,
            weight,
            bias,
            params,
            compute_dtype,
        } => {
            let bias_m = bias.map(|b| get_mirror(mirror, b)).transpose()?;
            conv::replay_conv2d(
                &get_mirror(mirror, input)?,
                &get_mirror(mirror, weight)?,
                bias_m.as_ref(),
                &params,
                compute_dtype,
            )
        }
        Op::ConvTranspose2d {
            input,
            weight,
            bias,
            params,
        } => {
            let bias_m = bias.map(|b| get_mirror(mirror, b)).transpose()?;
            conv::replay_conv_transpose2d(
                shape,
                &get_mirror(mirror, input)?,
                &get_mirror(mirror, weight)?,
                bias_m.as_ref(),
                &params,
            )
        }
        Op::Unfold { input, params } => conv::replay_unfold(&get_mirror(mirror, input)?, &params),
        Op::Fold { input, params } => {
            conv::replay_fold(shape, &get_mirror(mirror, input)?, &params)
        }
        // 索引・累積族（イシュー #3083。方式は `index_cum` モジュール doc 参照）。
        // `Sort`／`Topk`／`Cummax`／`Cummin` は値の選択のみのため `gather` で再生する。
        Op::Gather { input, dim, index }
        | Op::Sort { input, dim, index }
        | Op::Topk { input, dim, index }
        | Op::Cummax { input, dim, index }
        | Op::Cummin { input, dim, index } => get_mirror(mirror, input)?.gather(dim, &index),
        Op::Embedding { weight, index, .. } => get_mirror(mirror, weight)?.gather(0, &index),
        Op::Scatter {
            input,
            dim,
            index,
            src,
            reduce,
        } => index_cum::replay_scatter(
            &get_mirror(mirror, input)?,
            &get_mirror(mirror, src)?,
            dim,
            &index,
            reduce,
        ),
        Op::Cumsum { input, dim } => get_mirror(mirror, input)?.cumsum(dim),
        // `cond` は forward 時点（`Var::where_cond`）で `a`／`b` と
        // 同じ `out_shape` へ broadcast 済みの f32 マスク（`c != 0.0`
        // 判定契約。`Op::Where` doc 参照）として `Op` へ焼き込まれて
        // いるため、`Tensor<bool>`（`c != 0.0`）へ変換してそのまま
        // `Var::where_cond` へ渡せる（追加の broadcast は不要）。
        Op::Where { cond, a, b } => {
            let cond_bool = f32_mask_to_bool(&cond)?;
            Var::where_cond(&cond_bool, &get_mirror(mirror, a)?, &get_mirror(mirror, b)?)
        }
        // BatchNorm（イシュー #3079）: `fixed_stats` payload が eval の
        // 固定統計を保持しているため `Op` の拡張は不要。公開 `Var`
        // メソッド（forward と同じカーネル）へ委譲する。running stats の
        // 更新は `nn::BatchNormCore` だけが行い、`Var::` プリミティブは
        // 触れないため、再生で二重更新されない。
        Op::BatchNorm {
            input,
            weight,
            bias,
            eps,
            fixed_stats,
        } => {
            let x = get_mirror(mirror, input)?;
            let w = weight.map(|id| get_mirror(mirror, id)).transpose()?;
            let b = bias.map(|id| get_mirror(mirror, id)).transpose()?;
            match fixed_stats {
                None => x.batch_norm(w.as_ref(), b.as_ref(), eps),
                Some((mean, var)) => x.batch_norm_infer(w.as_ref(), b.as_ref(), &mean, &var, eps),
            }
        }
        Op::RmsNorm { .. } | Op::LayerNorm { .. } => {
            crate::create_graph_norm::replay_norm(&op, mirror)
        }
        // 損失 Op 第 1 弾（イシュー #3082・決定記録 §26）。公開 `Var` メソッドへ
        // 委譲し、forward の検証（shape・delta・targets 範囲）を迂回しない。
        Op::MseLoss {
            pred,
            target,
            reduction,
        } => get_mirror(mirror, pred)?.mse_loss_with(&get_mirror(mirror, target)?, reduction),
        Op::L1Loss {
            pred,
            target,
            reduction,
        } => get_mirror(mirror, pred)?.l1_loss(&get_mirror(mirror, target)?, reduction),
        Op::HuberLoss {
            pred,
            target,
            kind,
            delta,
            reduction,
        } => get_mirror(mirror, pred)?.huber_loss_impl(
            &get_mirror(mirror, target)?,
            kind,
            delta,
            reduction,
        ),
        Op::CrossEntropyLoss {
            logits,
            targets,
            class_dim,
            reduction,
        } => get_mirror(mirror, logits)?.cross_entropy_loss(&targets, class_dim, reduction),
        Op::NllLoss {
            input,
            targets,
            class_dim,
            reduction,
        } => get_mirror(mirror, input)?.nll_loss(&targets, class_dim, reduction),
        // FFT 4 件・線形代数 4 件（イシュー #3086。方式は `linalg_fft` モジュール doc 参照）。
        // forward と同じ dispatch を公開 `Var` メソッド経由で通す。`SlogdetLogAbsDet` の写しは
        // `slogdet` が積む未使用の `SlogdetSign` ノードを伴うが、どの根からも到達しないため
        // `collect_ancestors` の対象にならない。
        Op::Rfft {
            input,
            n,
            dim,
            norm,
        } => get_mirror(mirror, input)?.rfft(Some(n), Some(dim), norm),
        Op::Irfft {
            input,
            n,
            dim,
            norm,
        } => get_mirror(mirror, input)?.irfft(Some(n), Some(dim), norm),
        Op::Fft {
            input,
            n,
            dim,
            norm,
        } => get_mirror(mirror, input)?.fft(Some(n), Some(dim), norm),
        Op::Ifft {
            input,
            n,
            dim,
            norm,
        } => get_mirror(mirror, input)?.ifft(Some(n), Some(dim), norm),
        Op::Inv { input } => get_mirror(mirror, input)?.inv(),
        Op::Solve { a, b } => get_mirror(mirror, a)?.solve(&get_mirror(mirror, b)?),
        Op::Det { input } => get_mirror(mirror, input)?.det(),
        Op::SlogdetLogAbsDet { input } => Ok(get_mirror(mirror, input)?.slogdet()?.logabsdet),
        _ => Err(AutodiffError::Backward(
            "create_graph: replay_op: supports_create_graph() が true の未対応 Op（内部契約違反）"
                .into(),
        )),
    }
}

/// V3 の共通前処理: `v` の子テープ上の実体値を取り出す（借用は
/// 戻る前に解放する。`reduce_bias_grad_var` と同じ形）。
fn child_value(child: &Tape, v: &Var<'_>) -> Result<Tensor<f32>, AutodiffError> {
    let nodes = child.nodes.borrow();
    Ok(materialize_fallible(&nodes, child.ops(), v.node_id())?.clone())
}

/// V3（`docs/autodiff-higher-order-grad-decision.md` §21.1・§21.3・§23）の
/// 値と構造の結合。値 `host_value` は 1 階 VJP と同じホスト関数の結果、
/// 構造 `composed` は二階微分を流すための `Var` 演算グラフ。値を持つ
/// `Op::Contiguous`（恒等 VJP）ノードとして子テープへ登録する。
/// shape 不一致は内部不変条件違反として型付きエラーにする。
fn push_v3<'c>(
    child: &'c Tape,
    composed: &Var<'c>,
    host_value: Tensor<f32>,
) -> Result<Var<'c>, AutodiffError> {
    if host_value.shape() != composed.shape().as_slice() {
        return Err(AutodiffError::Backward(
            "create_graph: V3 のホスト値と構造グラフの shape が不一致（内部契約違反）".into(),
        ));
    }
    let nid = child.push_eager(
        Op::Contiguous {
            input: composed.node_id(),
        },
        host_value,
    );
    Ok(Var::from_raw(child, nid))
}

/// `dim` 軸を縮約して長さ 1 で復元し `shape` へ broadcast する。
fn reduce_dim_bcast<'c>(
    v: &Var<'c>,
    dim: usize,
    shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    v.sum(Some(dim))?
        .contiguous()?
        .unsqueeze(dim)?
        .broadcast_to(shape)
}

/// `Op::Softmax` の create_graph（V3。イシュー #3077）。値は 1 階と同じ
/// `grad::softmax_vjp_along`（f64 縮約）、構造は `y⊙g − y⊙Σ(g⊙y)`
/// （`Σy=1`・`y≥0` ゆえ中間値が `max|g|` 以下で overflow しない並べ方）。
///
/// 二階の逆伝播（codex-review 指摘・PR #3344）: `y` を 1 つのノードで
/// 3 箇所に使うと、`y` への寄与（`g`・`dot`・`g·(-y)` の各項）が同一ノードで
/// 蓄積され、個々は有限でも和が f32 範囲を超える（Softmax VJP の
/// 定数シフト不変性で本来は相殺される量）。そこで 3 つの使用ごとに入力から
/// `softmax` ミラーを別ノードとして立て、各寄与を別々に Softmax VJP へ
/// 通してから入力側で蓄積する（各項は `max|g|` 以下）。値は 1 階と同一の
/// ホスト関数が決めるため 1 階の bit 一致は不変。
fn softmax_cgrad_v3<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    id: NodeId,
    x_m: Var<'c>,
    g: &Var<'c>,
    dim: usize,
) -> Result<Var<'c>, AutodiffError> {
    let shape = parent_nodes[id.0].shape.clone();
    if shape.contains(&0) {
        return Ok(child.var_no_grad(&Tensor::zeros(&shape)?));
    }
    let y_a = x_m.softmax(dim)?;
    let y_b = x_m.softmax(dim)?;
    let y_c = x_m.softmax(dim)?;
    let out_value = materialize_fallible(parent_nodes, parent.ops(), id)?;
    let g_value = child_value(child, g)?;
    let t1 = y_a.mul(g)?;
    let dot = reduce_dim_bcast(&g.mul(&y_c)?, dim, &shape)?;
    let composed = t1.sub(&y_b.mul(&dot)?)?;
    let host = crate::grad::softmax_vjp_along(out_value, &g_value, dim);
    push_v3(child, &composed, host)
}

/// レーンごとのスケール `s = max|g|`（`dim` 軸を長さ 1 にした shape）。
/// `s == 0` と非有限（`g` に `inf`／`NaN`）のレーンは `1`（IEEE 伝播を
/// 構造側へ任せる。§24）。有限の小さい `max|g|` は下限
/// [`LANE_SCALE_FLOOR`] で持ち上げる（codex-review 指摘・PR #3344:
/// `g.div(&s)` の VJP は `1/s`・`-g/s²` を f32 で先に評価するため、
/// 非正規数級の `s` では二階微分の係数が `inf` になる。構造式は任意の
/// `s > 0` で数学的に同値なので下限を設けても 1 階値・二階の意味は
/// 不変）。
/// `lane_scale` の下限（`2^-20`）。`1/s² ≤ 2^40`・`|g|/s² ≤ 2^20/…` が
/// いずれも f32 で有限に収まる大きさ。
const LANE_SCALE_FLOOR: f32 = 9.536_743e-7;

fn lane_scale(g: &Tensor<f32>, dim: usize) -> Result<Tensor<f32>, AutodiffError> {
    let shape = g.shape().to_vec();
    let outer: usize = shape[..dim].iter().product();
    let axis_len = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let data = crate::eval::dense_vec(g);
    let mut s = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let mut m = 0f32;
            for a in 0..axis_len {
                let v = data[(o * axis_len + a) * inner + i].abs();
                // NaN は `>` が偽になるため明示的に非有限扱いへ倒す。
                if v.is_nan() {
                    m = f32::NAN;
                    break;
                }
                if v > m {
                    m = v;
                }
            }
            s.push(if m == 0.0 || !m.is_finite() {
                1.0
            } else {
                m.max(LANE_SCALE_FLOOR)
            });
        }
    }
    let mut sshape = shape;
    sshape[dim] = 1;
    Ok(Tensor::new(s, &sshape)?)
}

/// LogSoftmax の二階逆伝播で `exp` ミラーへ入る最大寄与
/// `q·s·|mean_gs| = n·p·s·|mean_gs|` をレーンごとにホストで評価し、
/// f32 範囲（約 3.4e38）に収まる上限 `SPLIT_LIMIT` を超える分だけ
/// ミラーの分割数を増やす（`1..=SPLIT_MAX`。通常は 1）。
const SPLIT_LIMIT: f64 = 2.0e38;
const SPLIT_MAX: usize = 64;

fn log_softmax_split_count(
    y: &Tensor<f32>,
    g: &Tensor<f32>,
    s_lane: &Tensor<f32>,
    dim: usize,
) -> usize {
    let shape = g.shape().to_vec();
    let outer: usize = shape[..dim].iter().product();
    let n = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let yd = crate::eval::dense_vec(y);
    let gd = crate::eval::dense_vec(g);
    let sd = crate::eval::dense_vec(s_lane);
    let mut worst = 0f64;
    for o in 0..outer {
        for i in 0..inner {
            let s = sd[o * inner + i] as f64;
            let mut sum = 0f64;
            let mut pmax = 0f64;
            for a in 0..n {
                let idx = (o * n + a) * inner + i;
                sum += gd[idx] as f64 / s;
                pmax = pmax.max((yd[idx] as f64).exp());
            }
            let mean = (sum / n as f64).abs();
            let m = n as f64 * pmax * s * mean;
            // 非有限は分割では救えない（IEEE 伝播に任せる）。
            if m.is_finite() {
                worst = worst.max(m);
            }
        }
    }
    ((worst / SPLIT_LIMIT).ceil() as usize).clamp(1, SPLIT_MAX)
}

/// `Op::LogSoftmax` の create_graph（V3。イシュー #3077）。値は 1 階と
/// 同じ `grad::log_softmax_vjp_dispatch`、構造は
/// `s ⊙ (g/s − exp(y) ⊙ Σ(g/s))`（`|g/s| ≤ 1` で縮約が overflow しない）。
fn log_softmax_cgrad_v3<'c>(
    parent: &Tape,
    child: &'c Tape,
    parent_nodes: &[TapeNode],
    x_m: Var<'c>,
    id: NodeId,
    g: &Var<'c>,
    dim: usize,
) -> Result<Var<'c>, AutodiffError> {
    let shape = parent_nodes[id.0].shape.clone();
    if shape.contains(&0) {
        return Ok(child.var_no_grad(&Tensor::zeros(&shape)?));
    }
    let out_value = materialize_fallible(parent_nodes, parent.ops(), id)?;
    let g_value = child_value(child, g)?;
    let s_lane = lane_scale(&g_value, dim)?;
    let pieces = log_softmax_split_count(out_value, &g_value, &s_lane, dim);
    // 二階の逆伝播の保護（codex-review 指摘・PR #3344。決定記録 §24.1b）:
    // `exp(y)·Σgs` の係数 `Σgs`（最大 `n`）が `exp` ノードへの寄与 `-s·Σgs`
    // を f32 範囲外へ押し出す。`q = exp(y + ln n) = n·p`・`mean_gs = Σgs / n`
    // と分け、`q` への寄与を `-s·mean_gs`（`≤ s`）に抑える。さらに `q` 自身
    // の VJP（`q ⊙ 寄与`、最大 `n·s`）が LogSoftmax VJP へ入る前に溢れる
    // ため、`q·mean_gs` を `pieces` 個の独立な `log_softmax` ミラー
    // （入力 `x_m` から別々に立てる）へ等分し、各ミラーへの寄与を別々に
    // LogSoftmax VJP へ通してから入力側で蓄積する（Softmax の `y` 3 分割と
    // 同型）。`pieces` はホストで最大寄与の上界から決める（通常は 1）。
    let n_lane = shape[dim] as f32;
    let ln_n = child.var_no_grad(&Tensor::full(&shape, n_lane.ln())?);
    let inv_nk = child.var_no_grad(&Tensor::full(&shape, 1.0 / (n_lane * pieces as f32))?);
    // `s ⊙ (gs − q·mean_gs) = g − q·mean_g` と恒等変形し、`s` を掛ける積を
    // 作らない（codex-review 指摘・PR #3344。入力依存の `g` でも `mean_gs_k`
    // 側へ `-s·q`（最大 `n·s`）が逆伝播しない）。`mean_g_k = Σ(g/(n·pieces))`
    // は各項 `≤ max|g|/n` で縮約が overflow しない。
    let mean_g_k = reduce_dim_bcast(&g.mul(&inv_nk)?, dim, &shape)?;
    let mut sub: Option<Var<'c>> = None;
    for _ in 0..pieces {
        let y_k = x_m.log_softmax(dim)?;
        let term = y_k.add(&ln_n)?.exp().mul(&mean_g_k)?;
        sub = Some(match sub {
            Some(acc) => acc.add(&term)?,
            None => term,
        });
    }
    let sub = sub.ok_or_else(|| AutodiffError::Backward("LogSoftmax 分割数が 0".into()))?;
    let composed = g.sub(&sub)?;
    let host = crate::grad::log_softmax_vjp_dispatch(child.ops(), out_value, &g_value, dim)?;
    push_v3(child, &composed, host)
}

/// [`Tape::backward_create_graph`] の逆走査（本体その 2。VJP を `Var`
/// 演算として子テープへ記録する）。`backward_impl`（`backward.rs`）と
/// 同型の「発生順とは逆順に走査し寄与を蓄積する」構造だが、蓄積先の
/// 値そのものが `Tensor<f32>` ではなく子テープ上の `Var<'c>`（さらに
/// 微分可能）である点が異なる。
fn build_cgrads<'c>(
    parent: &Tape,
    child: &'c Tape,
    ancestors: &[NodeId],
    mirror: &[Option<Var<'c>>],
    loss_id: NodeId,
) -> Result<Vec<Option<Var<'c>>>, AutodiffError> {
    let parent_nodes = parent.nodes.borrow();
    let parent_ops = parent.ops();
    let n = parent_nodes.len();
    let mut cgrads: Vec<Option<Var<'c>>> = vec![None; n];

    // シード: `Tape::backward`（`backward.rs::backward_impl`）と同じ
    // 「非スカラー loss は全要素 1 の暗黙の総和射影」意味論。
    let loss_shape = parent_nodes[loss_id.0].shape.clone();
    let seed_tensor = Tensor::full(&loss_shape, 1.0f32)?;
    cgrads[loss_id.0] = Some(child.var_no_grad(&seed_tensor));

    for &id in ancestors.iter().rev() {
        let Some(g) = cgrads[id.0] else {
            continue;
        };
        let node = &parent_nodes[id.0];
        match node.op.clone() {
            Op::Leaf => {}
            Op::Add(a, b) => {
                let a_shape = parent_nodes[a.0].shape.clone();
                let b_shape = parent_nodes[b.0].shape.clone();
                // codex-review 指摘（PR #1998）: `grad.rs::Op::Add` の
                // VJP は bias パターン（`upstream: [m, n]` →
                // `[n]`／`[1, n]` の行方向縮約）に限り `reduce_bias_
                // grad`（f64 相当アキュムレータ）へ委譲する
                // （`.claude/rules/coding-rust.md` の勾配長軸縮約
                // 契約・2026-09-12 ユーザー承認）。子テープ側も同じ
                // 分岐を [`reduce_bias_grad_var`] で再現し、bias
                // パターン以外は従来どおり [`reduce_to`]（`reduce_to_
                // shape` と同じ f32 逐次和）のまま維持する。
                let da = reduce_bias_grad_var(child, &g, &a_shape)?;
                let db = reduce_bias_grad_var(child, &g, &b_shape)?;
                accumulate(&parent_nodes, &mut cgrads, a, da)?;
                accumulate(&parent_nodes, &mut cgrads, b, db)?;
            }
            Op::Mul(a, b) => {
                let a_m = get_mirror(mirror, a)?;
                let b_m = get_mirror(mirror, b)?;
                let ga = g.mul(&b_m)?;
                let gb = g.mul(&a_m)?;
                let da = reduce_to(child, &ga, &a_m.shape())?;
                let db = reduce_to(child, &gb, &b_m.shape())?;
                accumulate(&parent_nodes, &mut cgrads, a, da)?;
                accumulate(&parent_nodes, &mut cgrads, b, db)?;
            }
            Op::MatMul(a, b) => {
                // 1 階 `grad.rs::matmul_vjp`（rank 2 経路）と同一の
                // オペランド順序（`da = gemm(g, bᵀ)`・
                // `db = gemm(aᵀ, g)`）を `transpose` と
                // `Var::matmul_fp32_strict`（codex-review 指摘。
                // PR #2003）の合成として子テープ上へ記録する。
                // `Var::matmul`（`ops.gemm`）ではなく
                // `matmul_fp32_strict`（`ops.gemm_fp32_strict`）を
                // 使うのは、CUDA TF32 opt-in（`set_cuda_gemm_precision`）
                // が有効な間も 1 階 `matmul_vjp` と同じく
                // バックプロパゲーションを FP32 厳密のまま保つため
                // （`Var::matmul_fp32_strict` doc 参照）。
                // `requires_grad == false` 側（`accumulate` が捨てる）
                // でも VJP 自体は記録して構わない——`transpose`／
                // `matmul_fp32_strict` は追加の副作用を持たないため
                // 無駄なノードが増えるだけで正しさに影響しない
                // （既存 `Op::Add`／`Mul` 腕と同じ方針）。
                let a_m = get_mirror(mirror, a)?;
                let b_m = get_mirror(mirror, b)?;
                let da = g.matmul_fp32_strict(&b_m.transpose(0, 1)?)?;
                let db = a_m.transpose(0, 1)?.matmul_fp32_strict(&g)?;
                accumulate(&parent_nodes, &mut cgrads, a, da)?;
                accumulate(&parent_nodes, &mut cgrads, b, db)?;
            }
            Op::Relu(a) => {
                // 劣勾配は x = 0 で 0（`grad.rs::Op::Relu` の VJP と同じ
                // 規約）。マスクは**入力側**の実測値（親テープ上で既に
                // 実体化済み）から作る——出力値ではなく入力値を見るのは
                // `grad.rs` の既存規約（`v > 0.0`）と一致させるため。
                let a_val = materialize_fallible(&parent_nodes, parent_ops, a)?;
                let mask = positive_mask(a_val)?;
                let zeros = child.var_no_grad(&Tensor::zeros(&g.shape())?);
                let da = Var::where_cond(&mask, &g, &zeros)?;
                accumulate(&parent_nodes, &mut cgrads, a, da)?;
            }
            Op::Exp(a) => {
                // d/dx exp(x) = exp(x)。子テープ上の写し（= exp(x) の
                // 子テープ上の再生値）を再利用し、`exp` を再計算しない
                // （`grad.rs::Op::Exp` の `out_value` 再利用と同じ方針）。
                let out_c = get_mirror(mirror, id)?;
                let da = g.mul(&out_c)?;
                accumulate(&parent_nodes, &mut cgrads, a, da)?;
            }
            Op::Tanh(a) => {
                // d/dx tanh(x) = 1 - tanh(x)^2。
                let out_c = get_mirror(mirror, id)?;
                let one = child.var_no_grad(&Tensor::full(&out_c.shape(), 1.0f32)?);
                let sq = out_c.mul(&out_c)?;
                let factor = one.sub(&sq)?;
                let da = g.mul(&factor)?;
                accumulate(&parent_nodes, &mut cgrads, a, da)?;
            }
            Op::Sigmoid(a) => {
                // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))。
                let out_c = get_mirror(mirror, id)?;
                let one = child.var_no_grad(&Tensor::full(&out_c.shape(), 1.0f32)?);
                let comp = one.sub(&out_c)?;
                let factor = out_c.mul(&comp)?;
                let da = g.mul(&factor)?;
                accumulate(&parent_nodes, &mut cgrads, a, da)?;
            }
            Op::Sum { input, dim } => {
                let input_shape = parent_nodes[input.0].shape.clone();
                let da = match dim {
                    None => g.broadcast_to(&input_shape)?,
                    Some(axis) => g
                        .contiguous()?
                        .unsqueeze(axis)?
                        .broadcast_to(&input_shape)?,
                };
                accumulate(&parent_nodes, &mut cgrads, input, da)?;
            }
            Op::Mean { input, dim } => {
                let input_shape = parent_nodes[input.0].shape.clone();
                let count: usize = match dim {
                    None => input_shape.iter().product(),
                    Some(axis) => input_shape[axis],
                };
                let scale = child.var_no_grad(&Tensor::scalar(1.0f32 / count as f32));
                let g_scaled = g.mul(&scale)?;
                let da = match dim {
                    None => g_scaled.broadcast_to(&input_shape)?,
                    Some(axis) => g_scaled
                        .contiguous()?
                        .unsqueeze(axis)?
                        .broadcast_to(&input_shape)?,
                };
                accumulate(&parent_nodes, &mut cgrads, input, da)?;
            }
            Op::Reshape { input } => {
                let input_shape = parent_nodes[input.0].shape.clone();
                let da = g.contiguous()?.reshape(&input_shape)?;
                accumulate(&parent_nodes, &mut cgrads, input, da)?;
            }
            Op::BroadcastTo { input } => {
                let input_shape = parent_nodes[input.0].shape.clone();
                let da = reduce_to(child, &g, &input_shape)?;
                accumulate(&parent_nodes, &mut cgrads, input, da)?;
            }
            Op::Transpose { input, dim0, dim1 } => {
                // 対合性（`transpose` を 2 回適用すると恒等）を使い、
                // 同じ軸で upstream を transpose するだけで閉じる
                // （`grad.rs::Op::Transpose` の VJP と同じ規約）。
                let da = g.transpose(dim0, dim1)?;
                accumulate(&parent_nodes, &mut cgrads, input, da)?;
            }
            Op::Permute { input, perm } => {
                // `grad.rs::inverse_permutation`（イシュー #2062 で
                // `pub(crate)` 化）を再利用し、逆置換の数式を二重管理
                // しない。
                let inv = crate::grad::inverse_permutation(&perm);
                let da = g.permute(&inv)?;
                accumulate(&parent_nodes, &mut cgrads, input, da)?;
            }
            Op::Narrow {
                input,
                dim,
                start,
                len,
            } => {
                // 「Split（Narrow）の VJP は Concat」の原則（`grad.rs::
                // Op::Narrow` の VJP と同型）。選択されなかった前後の
                // 区間を zero 定数で埋め、`Var::cat` で連結する。
                let input_shape = parent_nodes[input.0].shape.clone();
                let before_len = start;
                let after_len = input_shape[dim] - start - len;
                let mut pieces: Vec<Var<'c>> = Vec::with_capacity(3);
                if before_len > 0 {
                    let mut before_shape = input_shape.clone();
                    before_shape[dim] = before_len;
                    pieces.push(child.var_no_grad(&Tensor::zeros(&before_shape)?));
                }
                pieces.push(g);
                if after_len > 0 {
                    let mut after_shape = input_shape.clone();
                    after_shape[dim] = after_len;
                    pieces.push(child.var_no_grad(&Tensor::zeros(&after_shape)?));
                }
                let da = if pieces.len() == 1 {
                    g
                } else {
                    Var::cat(&pieces, dim)?
                };
                accumulate(&parent_nodes, &mut cgrads, input, da)?;
            }
            Op::Concat { inputs, dim } => {
                // 「Concat の VJP は Split（Narrow）」の原則（`grad.rs::
                // Op::Concat` の VJP と同型）。同一 `NodeId` の重複
                // （`cat(&[x, x])`）は `accumulate` が合算する。
                let mut off = 0usize;
                for input in inputs {
                    let len = parent_nodes[input.0].shape[dim];
                    let da = g.narrow(dim, off, len)?;
                    accumulate(&parent_nodes, &mut cgrads, input, da)?;
                    off += len;
                }
            }
            Op::Softmax { input, dim } => {
                let da = softmax_cgrad_v3(
                    parent,
                    child,
                    &parent_nodes,
                    id,
                    get_mirror(mirror, input)?,
                    &g,
                    dim,
                )?;
                accumulate(&parent_nodes, &mut cgrads, input, da)?;
            }
            Op::LogSoftmax { input, dim } => {
                let da = log_softmax_cgrad_v3(
                    parent,
                    child,
                    &parent_nodes,
                    get_mirror(mirror, input)?,
                    id,
                    &g,
                    dim,
                )?;
                accumulate(&parent_nodes, &mut cgrads, input, da)?;
            }
            Op::MseLoss {
                pred,
                target,
                reduction,
            } => {
                let pred_m = get_mirror(mirror, pred)?;
                let target_m = get_mirror(mirror, target)?;
                let shape = parent_nodes[pred.0].shape.clone();
                let dpred = if shape.contains(&0) {
                    child.var_no_grad(&Tensor::zeros(&shape)?)
                } else {
                    let n: usize = shape.iter().product();
                    let s_b = loss_scale_bcast(child, &g, reduction, true, n, &shape)?;
                    pred_m.sub(&target_m)?.mul(&s_b)?
                };
                let dtarget = dpred.neg()?;
                accumulate(&parent_nodes, &mut cgrads, pred, dpred)?;
                accumulate(&parent_nodes, &mut cgrads, target, dtarget)?;
            }
            Op::L1Loss {
                pred,
                target,
                reduction,
            } => {
                let pred_m = get_mirror(mirror, pred)?;
                let target_m = get_mirror(mirror, target)?;
                let shape = parent_nodes[pred.0].shape.clone();
                let dpred = if shape.contains(&0) {
                    child.var_no_grad(&Tensor::zeros(&shape)?)
                } else {
                    let n: usize = shape.iter().product();
                    let g_b = loss_scale_bcast(child, &g, reduction, false, n, &shape)?;
                    let p_val = materialize_fallible(&parent_nodes, parent_ops, pred)?;
                    let t_val = materialize_fallible(&parent_nodes, parent_ops, target)?;
                    let pos = pairwise_mask(p_val, t_val, |p, t| p - t > 0.0)?;
                    let neg = pairwise_mask(p_val, t_val, |p, t| p - t < 0.0)?;
                    let nan = pairwise_mask(p_val, t_val, |p, t| (p - t).is_nan())?;
                    let d_m = pred_m.sub(&target_m)?;
                    let zeros = child.var_no_grad(&Tensor::zeros(&shape)?);
                    // `d == 0` は区分定数の 0（V2。`g.mul(0/1 定数)` は使わない）。
                    // NaN は 1 階 `l1_grad_sign` と同じく `d_m` を通して伝播する。
                    let nan_branch = Var::where_cond(&nan, &d_m, &zeros)?;
                    let neg_branch = Var::where_cond(&neg, &g_b.neg()?, &nan_branch)?;
                    Var::where_cond(&pos, &g_b, &neg_branch)?
                };
                let dtarget = dpred.neg()?;
                accumulate(&parent_nodes, &mut cgrads, pred, dpred)?;
                accumulate(&parent_nodes, &mut cgrads, target, dtarget)?;
            }
            Op::HuberLoss {
                pred,
                target,
                kind,
                delta,
                reduction,
            } => {
                let pred_m = get_mirror(mirror, pred)?;
                let target_m = get_mirror(mirror, target)?;
                let shape = parent_nodes[pred.0].shape.clone();
                let smooth_l1 = match kind {
                    fandhe_ai_tensor_core::HuberKind::Huber => false,
                    fandhe_ai_tensor_core::HuberKind::SmoothL1 => true,
                    _ => {
                        return Err(AutodiffError::Backward(
                            "create_graph: HuberLoss: 未知の HuberKind は fail-closed".into(),
                        ));
                    }
                };
                let dpred = if shape.contains(&0) {
                    child.var_no_grad(&Tensor::zeros(&shape)?)
                } else {
                    let n: usize = shape.iter().product();
                    let g_b = loss_scale_bcast(child, &g, reduction, false, n, &shape)?;
                    let p_val = materialize_fallible(&parent_nodes, parent_ops, pred)?;
                    let t_val = materialize_fallible(&parent_nodes, parent_ops, target)?;
                    // 二次領域は `!(|d| >= delta)`（NaN の `d` は二次側へ入り、
                    // `d_m` 経由で NaN を伝播する。`eval::huber_elem_grad` と一致）。
                    let quad = pairwise_mask(p_val, t_val, |p, t| {
                        let a = (p - t).abs();
                        a < delta || a.is_nan()
                    })?;
                    let lin_data: Vec<f32> = {
                        let pc = p_val.contiguous();
                        let tc = t_val.contiguous();
                        let ps = pc.as_slice().unwrap_or(&[]);
                        let ts = tc.as_slice().unwrap_or(&[]);
                        ps.iter()
                            .zip(ts.iter())
                            .map(|(&p, &t)| {
                                let d = p - t;
                                if smooth_l1 {
                                    1.0f32.copysign(d)
                                } else {
                                    delta.copysign(d)
                                }
                            })
                            .collect()
                    };
                    let lin_const = child.var_no_grad(&Tensor::new(lin_data, &shape)?);
                    let d_m = pred_m.sub(&target_m)?;
                    // 線形領域（非選択分岐）の `d_m` は除算・乗算の前に 0 へマスクする。
                    // マスクせず `d_m / delta` を作ると、有限入力（例: pred=1e30・
                    // beta=1e-10）でも非選択側が inf になり、`where_cond` が二次側へ
                    // 流すゼロ余接と Mul／Div の VJP で積算されて NaN が二階微分へ
                    // 漏れる。NaN の `d` は `quad` に含まれるためマスク後も伝播する。
                    let zeros_q = child.var_no_grad(&Tensor::zeros(&shape)?);
                    let d_safe = Var::where_cond(&quad, &d_m, &zeros_q)?;
                    let quad_grad = if smooth_l1 {
                        let delta_c = child.var_no_grad(&Tensor::scalar(delta));
                        g_b.mul(&d_safe.div(&delta_c)?)?
                    } else {
                        g_b.mul(&d_safe)?
                    };
                    let lin_grad = g_b.mul(&lin_const)?;
                    Var::where_cond(&quad, &quad_grad, &lin_grad)?
                };
                let dtarget = dpred.neg()?;
                accumulate(&parent_nodes, &mut cgrads, pred, dpred)?;
                accumulate(&parent_nodes, &mut cgrads, target, dtarget)?;
            }
            Op::CrossEntropyLoss {
                logits,
                targets,
                class_dim,
                reduction,
            } => {
                let logits_m = get_mirror(mirror, logits)?;
                let shape = parent_nodes[logits.0].shape.clone();
                let dlogits = if shape.contains(&0) {
                    child.var_no_grad(&Tensor::zeros(&shape)?)
                } else {
                    let n = class_sample_count(&shape, class_dim);
                    let s_b = loss_scale_bcast(child, &g, reduction, false, n, &shape)?;
                    let onehot = child.var_no_grad(&onehot_f32(&targets, &shape, class_dim)?);
                    let y = logits_m.softmax(class_dim)?;
                    y.sub(&onehot)?.mul(&s_b)?
                };
                accumulate(&parent_nodes, &mut cgrads, logits, dlogits)?;
            }
            Op::NllLoss {
                input,
                targets,
                class_dim,
                reduction,
            } => {
                let shape = parent_nodes[input.0].shape.clone();
                let dinput = if shape.contains(&0) {
                    child.var_no_grad(&Tensor::zeros(&shape)?)
                } else {
                    let n = class_sample_count(&shape, class_dim);
                    let s_b = loss_scale_bcast(child, &g, reduction, false, n, &shape)?;
                    let mask = onehot_mask(&targets, &shape, class_dim)?;
                    let zeros = child.var_no_grad(&Tensor::zeros(&shape)?);
                    Var::where_cond(&mask, &s_b.neg()?, &zeros)?
                };
                accumulate(&parent_nodes, &mut cgrads, input, dinput)?;
            }
            Op::Contiguous { input } => {
                // 恒等（`grad.rs::Op::Contiguous` の VJP と同型）。
                accumulate(&parent_nodes, &mut cgrads, input, g)?;
            }
            Op::ConvWeightGradHost { structure, .. } | Op::BatchNormGradHost { structure, .. } => {
                // 恒等で `structure` のみへ（`grad.rs` の VJP と同型）。
                accumulate(&parent_nodes, &mut cgrads, structure, g)?;
            }
            Op::Where { cond, a, b } => {
                // `grad.rs::where_vjp` と同型: `cond` の真偽で `g` を
                // 通すか 0 にするかを選択し、`a`／`b` 元 shape へ縮約
                // する（`reduce_to`。`Op::Mul` 等と同じ broadcast 逆
                // 演算）。
                let a_shape = parent_nodes[a.0].shape.clone();
                let b_shape = parent_nodes[b.0].shape.clone();
                let mask = f32_mask_to_bool(&cond)?;
                let zeros = child.var_no_grad(&Tensor::zeros(&g.shape())?);
                let ga = Var::where_cond(&mask, &g, &zeros)?;
                let gb = Var::where_cond(&mask, &zeros, &g)?;
                let da = reduce_to(child, &ga, &a_shape)?;
                let db = reduce_to(child, &gb, &b_shape)?;
                accumulate(&parent_nodes, &mut cgrads, a, da)?;
                accumulate(&parent_nodes, &mut cgrads, b, db)?;
            }
            Op::BatchNorm {
                input,
                weight,
                bias,
                eps,
                fixed_stats,
            } => {
                // 方式（V3 値＋S1 構造・総括的な適用判定）は `create_graph_norm.rs`
                // と決定記録 §28 を参照。
                let x_host = materialize_fallible(&parent_nodes, parent_ops, input)?.clone();
                let (n_, c_, spatial_) = batch_norm_layout(x_host.shape())?;
                let inputs = crate::create_graph_norm::BatchNormCgradInputs {
                    x_mirror: get_mirror(mirror, input)?,
                    w_mirror: weight.map(|w| get_mirror(mirror, w)).transpose()?,
                    has_bias: bias.is_some(),
                    eps,
                    fixed_stats: fixed_stats.as_ref(),
                    x_host: &x_host,
                    need_dx: parent_nodes[input.0].requires_grad,
                    need_dw: weight.is_some_and(|w| parent_nodes[w.0].requires_grad),
                    need_db: bias.is_some_and(|b| parent_nodes[b.0].requires_grad),
                    n: n_,
                    c: c_,
                    spatial: spatial_,
                };
                let out = crate::create_graph_norm::batch_norm_cgrads(child, &inputs, &g)?;
                if let Some(dx) = out.dx {
                    accumulate(&parent_nodes, &mut cgrads, input, dx)?;
                }
                if let (Some(w), Some(dw)) = (weight, out.dw) {
                    accumulate(&parent_nodes, &mut cgrads, w, dw)?;
                }
                if let (Some(b), Some(db)) = (bias, out.db) {
                    accumulate(&parent_nodes, &mut cgrads, b, db)?;
                }
            }
            Op::ScalarUnary { op: sop, input } => {
                // `x_val`（materialize のクローン）・`one`（child.var_no_grad
                // の leaf ノード）は使う variant でのみ構築する（全 variant
                // 共通の先頭構築は使わない variant でも子テープへ不要な
                // ノードを積んでしまうため。review 指摘 2026-09-22）。
                let x_m = get_mirror(mirror, input)?;
                let da = match sop {
                    ScalarUnaryOp::Neg => g.neg()?,
                    ScalarUnaryOp::Abs => {
                        let x_val = materialize_fallible(&parent_nodes, parent_ops, input)?;
                        let pos = mask_from_pred(x_val, |v| v > 0.0)?;
                        let neg = mask_from_pred(x_val, |v| v < 0.0)?;
                        let zeros = child.var_no_grad(&Tensor::zeros(&g.shape())?);
                        let neg_branch = Var::where_cond(&neg, &g.neg()?, &zeros)?;
                        Var::where_cond(&pos, &g, &neg_branch)?
                    }
                    ScalarUnaryOp::Sqrt => {
                        let y_c = get_mirror(mirror, id)?;
                        let half = child.var_no_grad(&Tensor::scalar(0.5f32));
                        let factor = half.div(&y_c)?;
                        g.mul(&factor)?
                    }
                    ScalarUnaryOp::Log => {
                        let one = child.var_no_grad(&Tensor::scalar(1.0f32));
                        let factor = one.div(&x_m)?;
                        g.mul(&factor)?
                    }
                    ScalarUnaryOp::Log2 => {
                        let one = child.var_no_grad(&Tensor::scalar(1.0f32));
                        let ln2 = child.var_no_grad(&Tensor::scalar(std::f32::consts::LN_2));
                        let factor = one.div(&x_m.mul(&ln2)?)?;
                        g.mul(&factor)?
                    }
                    ScalarUnaryOp::Log10 => {
                        let one = child.var_no_grad(&Tensor::scalar(1.0f32));
                        let ln10 = child.var_no_grad(&Tensor::scalar(std::f32::consts::LN_10));
                        let factor = one.div(&x_m.mul(&ln10)?)?;
                        g.mul(&factor)?
                    }
                    ScalarUnaryOp::Sin => {
                        let factor = x_m.cos()?;
                        g.mul(&factor)?
                    }
                    ScalarUnaryOp::Cos => {
                        let factor = x_m.sin()?.neg()?;
                        g.mul(&factor)?
                    }
                    ScalarUnaryOp::Tan => {
                        let one = child.var_no_grad(&Tensor::scalar(1.0f32));
                        let c = x_m.cos()?;
                        let factor = one.div(&c.mul(&c)?)?;
                        g.mul(&factor)?
                    }
                    ScalarUnaryOp::Relu => {
                        let x_val = materialize_fallible(&parent_nodes, parent_ops, input)?;
                        let mask = positive_mask(x_val)?;
                        let zeros = child.var_no_grad(&Tensor::zeros(&g.shape())?);
                        Var::where_cond(&mask, &g, &zeros)?
                    }
                    ScalarUnaryOp::Exp => {
                        let out_c = get_mirror(mirror, id)?;
                        g.mul(&out_c)?
                    }
                    ScalarUnaryOp::Tanh => {
                        let out_c = get_mirror(mirror, id)?;
                        let onec = child.var_no_grad(&Tensor::full(&out_c.shape(), 1.0f32)?);
                        let sq = out_c.mul(&out_c)?;
                        let factor = onec.sub(&sq)?;
                        g.mul(&factor)?
                    }
                    ScalarUnaryOp::Sigmoid => {
                        let out_c = get_mirror(mirror, id)?;
                        let onec = child.var_no_grad(&Tensor::full(&out_c.shape(), 1.0f32)?);
                        let comp = onec.sub(&out_c)?;
                        let factor = out_c.mul(&comp)?;
                        g.mul(&factor)?
                    }
                    ScalarUnaryOp::Silu => {
                        let s = x_m.sigmoid();
                        let onec = child.var_no_grad(&Tensor::full(&x_m.shape(), 1.0f32)?);
                        let one_minus_s = onec.sub(&s)?;
                        let term = x_m.mul(&one_minus_s)?;
                        let inner = onec.add(&term)?;
                        let factor = s.mul(&inner)?;
                        g.mul(&factor)?
                    }
                    ScalarUnaryOp::Hardswish => {
                        let x_val = materialize_fallible(&parent_nodes, parent_ops, input)?;
                        let lo = mask_from_pred(x_val, |v| v <= -3.0)?;
                        let hi = mask_from_pred(x_val, |v| v >= 3.0)?;
                        let two = child.var_no_grad(&Tensor::scalar(2.0f32));
                        let three = child.var_no_grad(&Tensor::scalar(3.0f32));
                        let six = child.var_no_grad(&Tensor::scalar(6.0f32));
                        let mid = x_m.mul(&two)?.add(&three)?.div(&six)?;
                        let onec = child.var_no_grad(&Tensor::full(&x_m.shape(), 1.0f32)?);
                        let hi_branch = Var::where_cond(&hi, &onec, &mid)?;
                        let zeros = child.var_no_grad(&Tensor::zeros(&x_m.shape())?);
                        let factor = Var::where_cond(&lo, &zeros, &hi_branch)?;
                        g.mul(&factor)?
                    }
                    ScalarUnaryOp::LeakyRelu { negative_slope } => {
                        let x_val = materialize_fallible(&parent_nodes, parent_ops, input)?;
                        let mask = mask_from_pred(x_val, |v| v >= 0.0)?;
                        let slope = child.var_no_grad(&Tensor::scalar(negative_slope));
                        let else_branch = g.mul(&slope)?;
                        Var::where_cond(&mask, &g, &else_branch)?
                    }
                    ScalarUnaryOp::Elu { alpha } => {
                        let x_val = materialize_fallible(&parent_nodes, parent_ops, input)?;
                        let mask = mask_from_pred(x_val, |v| v > 0.0)?;
                        let alpha_c = child.var_no_grad(&Tensor::scalar(alpha));
                        let factor = x_m.exp().mul(&alpha_c)?;
                        let else_branch = g.mul(&factor)?;
                        Var::where_cond(&mask, &g, &else_branch)?
                    }
                    ScalarUnaryOp::Softplus { beta, threshold } => {
                        let x_val = materialize_fallible(&parent_nodes, parent_ops, input)?;
                        let mask = mask_from_pred(x_val, |v| v * beta > threshold)?;
                        let beta_c = child.var_no_grad(&Tensor::scalar(beta));
                        let factor = x_m.mul(&beta_c)?.sigmoid();
                        let else_branch = g.mul(&factor)?;
                        Var::where_cond(&mask, &g, &else_branch)?
                    }
                    ScalarUnaryOp::Clamp { min, max } => {
                        let x_val = materialize_fallible(&parent_nodes, parent_ops, input)?;
                        let mask = mask_from_pred(x_val, |v| {
                            !(v.is_nan() || min > max || v < min || v > max)
                        })?;
                        let zeros = child.var_no_grad(&Tensor::zeros(&g.shape())?);
                        Var::where_cond(&mask, &g, &zeros)?
                    }
                    ScalarUnaryOp::PowScalar { exponent } => {
                        if exponent == 0.0 {
                            child.var_no_grad(&Tensor::zeros(&g.shape())?)
                        } else {
                            let exp_c = child.var_no_grad(&Tensor::scalar(exponent));
                            let factor = x_m
                                .scalar_unary(ScalarUnaryOp::PowScalar {
                                    exponent: exponent - 1.0,
                                })?
                                .mul(&exp_c)?;
                            g.mul(&factor)?
                        }
                    }
                    ScalarUnaryOp::Erf => {
                        // erf'(x) = 2/√π · exp(-x²)（`erf_grad` と同順）。
                        let c = child.var_no_grad(&Tensor::scalar(
                            (2.0f64 * 0.564_189_583_547_756_3f64) as f32,
                        ));
                        let factor = x_m.mul(&x_m)?.neg()?.exp().mul(&c)?;
                        g.mul(&factor)?
                    }
                    ScalarUnaryOp::Gelu => {
                        // Φ(x) + x·φ(x)（`gelu_erf_grad` と同順）。Φ は子テープへ
                        // `Erf` として記録する（閉包性。#3081・§22）。
                        let inv_sqrt2 =
                            child.var_no_grad(&Tensor::scalar(std::f32::consts::FRAC_1_SQRT_2));
                        let half = child.var_no_grad(&Tensor::scalar(0.5f32));
                        let one = child.var_no_grad(&Tensor::scalar(1.0f32));
                        let neg_half = child.var_no_grad(&Tensor::scalar(-0.5f32));
                        let inv_sqrt_2pi = child.var_no_grad(&Tensor::scalar(0.398_942_3_f32));
                        let e = x_m.mul(&inv_sqrt2)?.scalar_unary(ScalarUnaryOp::Erf)?;
                        let cdf = one.add(&e)?.mul(&half)?;
                        let pdf = x_m.mul(&x_m)?.mul(&neg_half)?.exp().mul(&inv_sqrt_2pi)?;
                        let factor = cdf.add(&x_m.mul(&pdf)?)?;
                        g.mul(&factor)?
                    }
                    ScalarUnaryOp::GeluTanh => {
                        // 飽和域（1 - tanh² == 0）は host マスクで検出し、滑らかな式は
                        // 飽和要素を 0 に置換した x_safe で組む（double-where）。
                        // where_cond の死に枝へも 0 の余接が流れ、後続 Mul の VJP が
                        // `0·相方` を作るため、相方が inf だと 2 階へ NaN が混入する。
                        let x_val = materialize_fallible(&parent_nodes, parent_ops, input)?;
                        const C: f32 = 0.797_884_6;
                        const K: f32 = 0.044_715;
                        let t_host = |v: f32| (C * (v + K * v * v * v)).tanh();
                        let sat = mask_from_pred(x_val, |v| {
                            let t = t_host(v);
                            1.0 - t * t == 0.0
                        })?;
                        let contiguous = x_val.contiguous();
                        let limit_data: Vec<f32> = contiguous
                            .as_slice()
                            .map(|s| s.iter().map(|&v| 0.5 * (1.0 + t_host(v))).collect())
                            .unwrap_or_default();
                        let limit =
                            child.var_no_grad(&Tensor::new(limit_data, contiguous.shape())?);
                        let zeros = child.var_no_grad(&Tensor::zeros(&x_m.shape())?);
                        let x_safe = Var::where_cond(&sat, &zeros, &x_m)?;
                        let c = child.var_no_grad(&Tensor::scalar(C));
                        let k = child.var_no_grad(&Tensor::scalar(K));
                        let k3 = child.var_no_grad(&Tensor::scalar(3.0 * K));
                        let half = child.var_no_grad(&Tensor::scalar(0.5f32));
                        let one = child.var_no_grad(&Tensor::scalar(1.0f32));
                        let x2 = x_safe.mul(&x_safe)?;
                        let x3 = x2.mul(&x_safe)?;
                        let u = x_safe.add(&k.mul(&x3)?)?.mul(&c)?;
                        let t = u.tanh();
                        let sech2 = one.sub(&t.mul(&t)?)?;
                        let du = one.add(&k3.mul(&x2)?)?.mul(&c)?;
                        let first = one.add(&t)?.mul(&half)?;
                        let second = x_safe.mul(&sech2)?.mul(&du)?.mul(&half)?;
                        let smooth = first.add(&second)?;
                        let factor = Var::where_cond(&sat, &limit, &smooth)?;
                        g.mul(&factor)?
                    }
                    _ => {
                        return Err(AutodiffError::Backward(format!(
                            "create_graph: build_cgrads: 未対応の ScalarUnaryOp {sop:?}\
                             （内部契約違反。scalar_unary_replayable との判定不整合）"
                        )));
                    }
                };
                accumulate(&parent_nodes, &mut cgrads, input, da)?;
            }
            Op::ScalarBinary { op: sop, a, b } => {
                let a_shape = parent_nodes[a.0].shape.clone();
                let b_shape = parent_nodes[b.0].shape.clone();
                if sop.is_comparison() {
                    // 比較演算は区分定数で両入力の勾配が恒等的にゼロ
                    // （`grad.rs::Op::ScalarBinary` の比較演算分岐と
                    // 同型。`0 * upstream` を経由せず直接ゼロ定数を
                    // 返し `inf`／`NaN` 汚染を避ける）。
                    let da = child.var_no_grad(&Tensor::zeros(&a_shape)?);
                    let db = child.var_no_grad(&Tensor::zeros(&b_shape)?);
                    accumulate(&parent_nodes, &mut cgrads, a, da)?;
                    accumulate(&parent_nodes, &mut cgrads, b, db)?;
                } else {
                    let out_shape = parent_nodes[id.0].shape.clone();
                    let a_m = get_mirror(mirror, a)?;
                    let b_m = get_mirror(mirror, b)?;
                    // `a_bc`／`b_bc`（child tape 上への broadcast_to view
                    // ノード）は Add／Sub／Maximum／Minimum では使わない
                    // ため、使う分岐（Mul／Div／Pow）の内側でのみ構築
                    // する（review 指摘 2026-09-22。全分岐共通で先頭
                    // 構築すると使わない分岐のたびに未使用の子テープ
                    // ノードが 2 つ余分に積まれる）。
                    let (da_full, db_full) = match sop {
                        ScalarBinaryOp::Add => (g, g),
                        ScalarBinaryOp::Sub => (g, g.neg()?),
                        ScalarBinaryOp::Mul => {
                            let a_bc = a_m.broadcast_to(&out_shape)?;
                            let b_bc = b_m.broadcast_to(&out_shape)?;
                            (g.mul(&b_bc)?, g.mul(&a_bc)?)
                        }
                        ScalarBinaryOp::Div => {
                            let a_bc = a_m.broadcast_to(&out_shape)?;
                            let b_bc = b_m.broadcast_to(&out_shape)?;
                            let one = child.var_no_grad(&Tensor::scalar(1.0f32));
                            let factor_a = one.div(&b_bc)?;
                            let factor_b = a_bc.div(&b_bc)?.div(&b_bc)?.neg()?;
                            (g.mul(&factor_a)?, g.mul(&factor_b)?)
                        }
                        ScalarBinaryOp::Pow => {
                            let a_bc = a_m.broadcast_to(&out_shape)?;
                            let b_bc = b_m.broadcast_to(&out_shape)?;
                            let a_val = materialize_fallible(&parent_nodes, parent_ops, a)?;
                            let b_val = materialize_fallible(&parent_nodes, parent_ops, b)?;
                            let (a_bc_val, b_bc_val) =
                                a_val.broadcast_with(b_val).map_err(AutodiffError::Shape)?;
                            let mask_b0 = mask_from_pred(&b_bc_val, |v| v == 0.0)?;
                            let mask_a0 = mask_from_pred(&a_bc_val, |v| v == 0.0)?;
                            let zeros = child.var_no_grad(&Tensor::zeros(&out_shape)?);
                            let out_c = get_mirror(mirror, id)?;
                            let one = child.var_no_grad(&Tensor::scalar(1.0f32));
                            let b_minus_one = b_bc.sub(&one)?;
                            // `a_pow`（da 用）・`log(a)`（db 用）はいずれも
                            // `where_cond` で選ばれない側の枝として構築
                            // されるが、値自体は forward で即座に評価
                            // される（child tape は実値を持つ imperative
                            // グラフのため）。a==0 のとき `a^(負の指数)`
                            // ／`log(a)` は inf／-inf になり、この枝を
                            // Hessian（child tape を再度 backward する
                            // 経路）で微分すると、局所勾配が `1/a` 等の
                            // 発散値になる。`where_cond` の逆伝播は不選択
                            // 枝へ厳密に 0 を流すが、その 0 が発散した局所
                            // 勾配と掛け合わさり `0 * inf = NaN` へ汚染
                            // される（review 指摘 2026-09-22。db 側の
                            // `a_bc.log()` で顕在化。da 側の `a_bc.pow
                            // (b_minus_one)` も `a==0 ∧ b==0` で同型の
                            // 危険〈`0^(-1)` 等〉を持つため同じ安全化を
                            // 適用する）。危険になりうる入力を選択前に
                            // 安全値（`log(1)=0`／`1^e=1`）へ置換して
                            // から `log`／`pow` を取ることで、当該枝の
                            // 局所勾配自体を有限に保つ（最終的な forward
                            // 値は `where_cond` の選択で変わらない）。
                            let safe_a_for_pow = Var::where_cond(&mask_b0, &one, &a_bc)?;
                            let safe_a_for_log = Var::where_cond(&mask_a0, &one, &a_bc)?;
                            let a_pow = safe_a_for_pow.pow(&b_minus_one)?;
                            let da_factor = Var::where_cond(&mask_b0, &zeros, &b_bc.mul(&a_pow)?)?;
                            let db_factor = Var::where_cond(
                                &mask_a0,
                                &zeros,
                                &out_c.mul(&safe_a_for_log.log()?)?,
                            )?;
                            (g.mul(&da_factor)?, g.mul(&db_factor)?)
                        }
                        ScalarBinaryOp::Maximum | ScalarBinaryOp::Minimum => {
                            // Maximum／Minimum の勾配は mask 選択で `g` を
                            // 直接使う（`grad.rs` の 1 階側と同型）ため
                            // `a_bc`／`b_bc`（broadcast_to view ノード）は
                            // 不要。構築しない。
                            let a_val = materialize_fallible(&parent_nodes, parent_ops, a)?;
                            let b_val = materialize_fallible(&parent_nodes, parent_ops, b)?;
                            let (a_bc_val, b_bc_val) =
                                a_val.broadcast_with(b_val).map_err(AutodiffError::Shape)?;
                            let a_favored = matches!(sop, ScalarBinaryOp::Maximum);
                            let mask_a_wins = pairwise_mask(&a_bc_val, &b_bc_val, |x, y| {
                                !(x.is_nan() || y.is_nan()) && if a_favored { x > y } else { x < y }
                            })?;
                            let mask_b_wins = pairwise_mask(&a_bc_val, &b_bc_val, |x, y| {
                                !(x.is_nan() || y.is_nan()) && if a_favored { x < y } else { x > y }
                            })?;
                            let mask_tie = pairwise_mask(&a_bc_val, &b_bc_val, |x, y| {
                                !(x.is_nan() || y.is_nan()) && x == y
                            })?;
                            let zeros = child.var_no_grad(&Tensor::zeros(&out_shape)?);
                            let half = child.var_no_grad(&Tensor::scalar(0.5f32));
                            let half_g = g.mul(&half)?;
                            let tie_branch = Var::where_cond(&mask_tie, &half_g, &zeros)?;
                            let da = Var::where_cond(&mask_a_wins, &g, &tie_branch)?;
                            let db = Var::where_cond(&mask_b_wins, &g, &tie_branch)?;
                            (da, db)
                        }
                        _ => {
                            return Err(AutodiffError::Backward(format!(
                                "create_graph: build_cgrads: 未対応の ScalarBinaryOp {sop:?}\
                                 （内部契約違反。scalar_binary_replayable との判定不整合）"
                            )));
                        }
                    };
                    let da = reduce_bias_grad_var(child, &da_full, &a_shape)?;
                    let db = reduce_bias_grad_var(child, &db_full, &b_shape)?;
                    accumulate(&parent_nodes, &mut cgrads, a, da)?;
                    accumulate(&parent_nodes, &mut cgrads, b, db)?;
                }
            }
            Op::RmsNorm { input, weight, eps } => {
                let targets = crate::create_graph_norm::Targets {
                    input,
                    weight,
                    bias: None,
                };
                let contributions = crate::create_graph_norm::rms_norm_cgrads(
                    child,
                    &parent_nodes,
                    parent_ops,
                    mirror,
                    g,
                    &targets,
                    eps,
                )?;
                for (target, v) in contributions {
                    accumulate(&parent_nodes, &mut cgrads, target, v)?;
                }
            }
            Op::LayerNorm {
                input,
                weight,
                bias,
                eps,
            } => {
                let targets = crate::create_graph_norm::Targets {
                    input,
                    weight,
                    bias,
                };
                let contributions = crate::create_graph_norm::layer_norm_cgrads(
                    child,
                    &parent_nodes,
                    parent_ops,
                    mirror,
                    g,
                    &targets,
                    eps,
                )?;
                for (target, v) in contributions {
                    accumulate(&parent_nodes, &mut cgrads, target, v)?;
                }
            }
            // 2D 畳み込み族（イシュー #3080。方式は `conv` モジュール doc 参照）。
            Op::Conv2d {
                input,
                weight,
                bias,
                params,
                ..
            } => {
                let need = conv::ConvNeeds {
                    input: parent_nodes[input.0].requires_grad,
                    weight: parent_nodes[weight.0].requires_grad,
                    bias: bias.is_some_and(|b| parent_nodes[b.0].requires_grad),
                };
                let bias_shape = bias.map(|b| parent_nodes[b.0].shape.clone());
                let r = conv::conv2d_vjp(
                    child,
                    &g,
                    &get_mirror(mirror, input)?,
                    &get_mirror(mirror, weight)?,
                    bias_shape.as_deref(),
                    &params,
                    need,
                )?;
                if let Some(d) = r.d_input {
                    accumulate(&parent_nodes, &mut cgrads, input, d)?;
                }
                if let Some(d) = r.d_weight {
                    accumulate(&parent_nodes, &mut cgrads, weight, d)?;
                }
                if let (Some(b), Some(d)) = (bias, r.d_bias) {
                    accumulate(&parent_nodes, &mut cgrads, b, d)?;
                }
            }
            Op::ConvTranspose2d {
                input,
                weight,
                bias,
                params,
            } => {
                let need = conv::ConvNeeds {
                    input: parent_nodes[input.0].requires_grad,
                    weight: parent_nodes[weight.0].requires_grad,
                    bias: bias.is_some_and(|b| parent_nodes[b.0].requires_grad),
                };
                let bias_shape = bias.map(|b| parent_nodes[b.0].shape.clone());
                let r = conv::conv_transpose2d_vjp(
                    child,
                    &g,
                    &get_mirror(mirror, input)?,
                    &get_mirror(mirror, weight)?,
                    bias_shape.as_deref(),
                    &params,
                    need,
                )?;
                if let Some(d) = r.d_input {
                    accumulate(&parent_nodes, &mut cgrads, input, d)?;
                }
                if let Some(d) = r.d_weight {
                    accumulate(&parent_nodes, &mut cgrads, weight, d)?;
                }
                if let (Some(b), Some(d)) = (bias, r.d_bias) {
                    accumulate(&parent_nodes, &mut cgrads, b, d)?;
                }
            }
            Op::Unfold { input, params } => {
                let input_shape = parent_nodes[input.0].shape.clone();
                let d = conv::unfold_vjp(&g, &input_shape, &params)?;
                accumulate(&parent_nodes, &mut cgrads, input, d)?;
            }
            Op::Fold { input, params } => {
                let d = conv::fold_vjp(&g, &params)?;
                accumulate(&parent_nodes, &mut cgrads, input, d)?;
            }
            // 索引・累積族（イシュー #3083。方式は `index_cum` モジュール doc 参照）。
            Op::Gather { input, dim, index }
            | Op::Sort { input, dim, index }
            | Op::Topk { input, dim, index }
            | Op::Cummax { input, dim, index }
            | Op::Cummin { input, dim, index } => {
                if parent_nodes[input.0].requires_grad {
                    let input_shape = parent_nodes[input.0].shape.clone();
                    let d = index_cum::gather_like_vjp(child, &g, &input_shape, dim, &index)?;
                    accumulate(&parent_nodes, &mut cgrads, input, d)?;
                }
            }
            Op::Embedding {
                weight,
                index,
                padding_idx,
            } => {
                if parent_nodes[weight.0].requires_grad {
                    let weight_shape = parent_nodes[weight.0].shape.clone();
                    let d =
                        index_cum::embedding_vjp(child, &g, &weight_shape, &index, padding_idx)?;
                    accumulate(&parent_nodes, &mut cgrads, weight, d)?;
                }
            }
            Op::Scatter {
                input,
                dim,
                index,
                src,
                reduce,
            } => {
                let input_shape = parent_nodes[input.0].shape.clone();
                let src_shape = parent_nodes[src.0].shape.clone();
                let r = index_cum::scatter_vjp(
                    child,
                    &g,
                    &index_cum::ScatterSpec {
                        input_shape: &input_shape,
                        src_shape: &src_shape,
                        dim,
                        index: &index,
                        reduce,
                    },
                    parent_nodes[input.0].requires_grad,
                    parent_nodes[src.0].requires_grad,
                )?;
                if let Some(d) = r.d_input {
                    accumulate(&parent_nodes, &mut cgrads, input, d)?;
                }
                if let Some(d) = r.d_src {
                    accumulate(&parent_nodes, &mut cgrads, src, d)?;
                }
            }
            Op::Cumsum { input, dim } => {
                if parent_nodes[input.0].requires_grad {
                    let d = index_cum::cumsum_cgrad(child, &g, dim)?;
                    accumulate(&parent_nodes, &mut cgrads, input, d)?;
                }
            }
            // FFT 4 件（イシュー #3086。随伴は別の FFT＋長さ合わせ）。
            Op::Rfft {
                input,
                n,
                dim,
                norm,
            }
            | Op::Irfft {
                input,
                n,
                dim,
                norm,
            }
            | Op::Fft {
                input,
                n,
                dim,
                norm,
            }
            | Op::Ifft {
                input,
                n,
                dim,
                norm,
            } => {
                if parent_nodes[input.0].requires_grad {
                    let kind = match &node.op {
                        Op::Rfft { .. } => linalg_fft::FftKind::Rfft,
                        Op::Irfft { .. } => linalg_fft::FftKind::Irfft,
                        Op::Fft { .. } => linalg_fft::FftKind::Fft,
                        _ => linalg_fft::FftKind::Ifft,
                    };
                    let input_shape = parent_nodes[input.0].shape.clone();
                    let d = linalg_fft::fft_vjp(child, &g, kind, &input_shape, n, dim, norm)?;
                    accumulate(&parent_nodes, &mut cgrads, input, d)?;
                }
            }
            // 線形代数 4 件（イシュー #3086）。構造を積む前に f64 参照で総括的な適用判定を行い、
            // 外れる入力は `Err`（呼び出し側のロールバックで `child` は空へ戻る）。
            Op::Inv { input } => {
                if parent_nodes[input.0].requires_grad {
                    let a_val = materialize_fallible(&parent_nodes, parent_ops, input)?;
                    linalg_fft::guard_inv(a_val, &value_of(child, &g)?)?;
                    let x = get_mirror(mirror, id)?;
                    let a_shape = parent_nodes[input.0].shape.clone();
                    let d = linalg_fft::inv_cgrad(child, &g, &x, &a_shape)?;
                    accumulate(&parent_nodes, &mut cgrads, input, d)?;
                }
            }
            Op::Solve { a, b } => {
                if parent_nodes[a.0].requires_grad || parent_nodes[b.0].requires_grad {
                    let a_val = materialize_fallible(&parent_nodes, parent_ops, a)?;
                    let b_val = materialize_fallible(&parent_nodes, parent_ops, b)?;
                    linalg_fft::guard_solve(a_val, b_val, &value_of(child, &g)?)?;
                    let a_m = get_mirror(mirror, a)?;
                    let x = get_mirror(mirror, id)?;
                    let (da, db) = linalg_fft::solve_cgrad(child, &g, &a_m, &x)?;
                    accumulate(&parent_nodes, &mut cgrads, a, da)?;
                    accumulate(&parent_nodes, &mut cgrads, b, db)?;
                }
            }
            Op::Det { input } | Op::SlogdetLogAbsDet { input } => {
                if parent_nodes[input.0].requires_grad {
                    let with_det = matches!(node.op, Op::Det { .. });
                    let a_val = materialize_fallible(&parent_nodes, parent_ops, input)?;
                    linalg_fft::guard_det_like(a_val, &value_of(child, &g)?, with_det)?;
                    let a_m = get_mirror(mirror, input)?;
                    let scale = if with_det {
                        g.mul(&get_mirror(mirror, id)?)?
                    } else {
                        g
                    };
                    let d = linalg_fft::det_like_cgrad(child, &scale, &a_m)?;
                    accumulate(&parent_nodes, &mut cgrads, input, d)?;
                }
            }
            _ => {
                return Err(AutodiffError::Backward(
                    "create_graph: build_cgrads: supports_create_graph() が true の未対応 Op\
                     （内部契約違反）"
                        .into(),
                ));
            }
        }
    }

    Ok(cgrads)
}

/// 同一入力ノードへ複数経路から流入した勾配（子テープ上の `Var`）を
/// `Var::add` で合算する（`backward.rs::accumulate` の子テープ版）。
/// `target` が `requires_grad == false`（`docs/
/// autodiff-nograd-leaf-dinput-skip-decision.md` の契約と同型）なら
/// 寄与を捨てる——`backward_impl` が `requires_grad == false` の
/// ノードへの寄与を `accumulate` へ渡さず捨てるのと同じ意味論
/// （`target` の `mirror` は `var_no_grad` の定数葉であり、その勾配を
/// 蓄積する意味がないため）。
fn accumulate<'c>(
    parent_nodes: &[TapeNode],
    cgrads: &mut [Option<Var<'c>>],
    target: NodeId,
    contribution: Var<'c>,
) -> Result<(), AutodiffError> {
    if !parent_nodes[target.0].requires_grad {
        return Ok(());
    }
    match cgrads[target.0] {
        Some(existing) => {
            cgrads[target.0] = Some(existing.add(&contribution)?);
        }
        None => {
            cgrads[target.0] = Some(contribution);
        }
    }
    Ok(())
}

/// broadcast された `v` を `target_shape` へ縮約する（`Var::add`／
/// `mul` の VJP が使う `reduce_to_shape`〈`grad.rs`〉の `Var` 演算版）。
///
/// **数値方式は `grad.rs::reduce_to_shape` と同一のアルゴリズムを
/// `Var` 演算列として逐語再現する**（`Var::narrow` で縮約対象軸を
/// 長さ 1 のスライスへ分解し、`Var::add` で添字昇順に逐次加算する。
/// `reduce_to_shape` の二重ループ `for axis { for a in 0..axis_len {
/// reduced[...] += data[src] } }` と同じ縮約順序・同じ f32 逐次和で
/// あり、CPU 上の `Var::add` は要素ごとの単純加算〈f64 アキュムレータ
/// を挟まない〉のため 1 階 VJP の `reduce_to_shape` と bit 同一になる
/// ——当初実装〈`Var::sum_dims`。CPU `sum` の f64 アキュムレータ縮約〉
/// では、桁落ちを伴う broadcast 入力〈大きさの異なる値が完全に
/// 相殺するケース〉で 1 階勾配と乖離し REQ-2 統一複合判定を満たさない
/// 具体例が確認されたため是正した。codex-review 指摘（PR #1998）。
fn reduce_to<'c>(
    child: &'c Tape,
    v: &Var<'c>,
    target_shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    let v_shape = v.shape();
    if v_shape == target_shape {
        return Ok(*v);
    }
    debug_assert!(
        v_shape.len() >= target_shape.len(),
        "reduce_to: broadcast 後 shape の rank は入力 rank 以上のはず（契約違反）"
    );
    let rank_diff = v_shape.len().saturating_sub(target_shape.len());
    let mut padded_target = vec![1usize; rank_diff];
    padded_target.extend_from_slice(target_shape);

    let mut cur = *v;
    let mut cur_shape = v_shape;
    for axis in 0..cur_shape.len() {
        if padded_target[axis] == 1 && cur_shape[axis] != 1 {
            let axis_len = cur_shape[axis];
            if axis_len == 0 {
                // 合法な空テンソル（縮約対象の軸長が 0。例:
                // `x: [3]` を `broadcast_to(&[0, 3])` した結果を
                // 逆縮約する場合）。`reduce_to_shape`（1 階 VJP）の
                // 対応する二重ループは `for a in 0..axis_len {...}`
                // が 0 回実行されるため、ゼロ初期化された `reduced`
                // バッファがそのまま結果になる（codex-review P2
                // 是正・PR #1998）。`Var::narrow(axis, 0, 1)` は
                // `axis_len == 0` では範囲外（`NarrowOutOfBounds`）
                // になるため呼ばず、縮約後 shape を持つ明示的な
                // ゼロ定数へ直接差し替える。
                let mut reduced_shape = cur_shape.clone();
                reduced_shape[axis] = 1;
                cur = child.var_no_grad(&Tensor::zeros(&reduced_shape)?);
                cur_shape[axis] = 1;
                continue;
            }
            // `reduce_to_shape` の `for a in 0..axis_len { reduced[..] +=
            // data[src] }` を、添字昇順の `Var::narrow`＋`Var::add` の
            // 逐次連鎖として同じ順序で再現する（f32 逐次和・f64
            // アキュムレータなし）。
            let mut acc = cur.narrow(axis, 0, 1)?;
            for a in 1..axis_len {
                let slice = cur.narrow(axis, a, 1)?;
                acc = acc.add(&slice)?;
            }
            cur = acc;
            cur_shape[axis] = 1;
        }
    }
    if cur_shape == target_shape {
        Ok(cur)
    } else {
        // `cur_shape` は `padded_target`（先頭 `rank_diff` 個の 1 軸を
        // 含む）と一致しているはずであり、要素数は `target_shape` と
        // 同一のため `reshape` で先頭の 1 軸を落とせる。
        cur.contiguous()?.reshape(target_shape)
    }
}

/// 子テープ上の `v` の実体値を所有値として取り出す（`conv` モジュールの V3
/// ノード〈1 階と同じホスト関数の前方値〉の入力値取得に使う）。
fn value_of<'c>(child: &'c Tape, v: &Var<'c>) -> Result<Tensor<f32>, AutodiffError> {
    let nodes = child.nodes.borrow();
    Ok(materialize_fallible(&nodes, child.ops(), v.node_id())?.clone())
}

/// `Op::Add` の bias パターンに限り `grad.rs::reduce_bias_grad`（f64
/// 相当のアキュムレータで行方向を縮約する）と数値方式を揃える
/// `reduce_to` の薄いラッパー（codex-review 指摘・PR #1998 是正）。
///
/// **背景**: `.claude/rules/coding-rust.md` の「勾配の長軸縮約は `f64`
/// アキュムレータで統一する」契約は `Op::LinearResident`／
/// `Op::LinearAct` の bias フォールバックに加え、2026-09-12 ユーザー
/// 承認により `Op::Add`（`nn::Linear` 既定 forward 経路 `matmul → add`
/// が経由する）の bias パターンへも横展開済み（`grad.rs::
/// reduce_bias_grad`）。本モジュール（子テープ上の `Var` 演算列）の
/// `reduce_to` は 1 階 `reduce_to_shape`（`f32` 逐次和・bias 以外の
/// 一般的な broadcast 縮約）を逐語再現する設計のため、`Op::Add` を
/// 無条件に `reduce_to` へ委譲すると bias パターンでのみ 1 階勾配
/// （`reduce_bias_grad` 経由）と符号レベルで乖離する（例:
/// `x: [3, 1]`・`b: [1]`・`c = [1e8, 1, -1e8]: [3, 1]` に対する
/// `loss = ((x + b) * c).sum()` で `b` の 1 階勾配は `1` だが `f32`
/// 逐次和では丸め誤差により `0` になる）。
///
/// **判定条件は `reduce_bias_grad` と同一**（`g_shape.len() == 2` かつ
/// `target_shape` が「軸 0 方向の縮約」を表す形状——末尾次元が `g` の
/// 列数と一致し、それより前の全次元が `1`）。条件を満たさない場合は
/// 従来どおり [`reduce_to`] へそのまま委譲し挙動を変えない（bias
/// パターン限定の横展開であり、汎用 `Op::Add`・`reduce_to` 本体は
/// 不変）。
///
/// **数値方式の再現方法**: `Var::sum(Some(0))` は `child.ops().sum()`
/// （`BackendOps` 実装依存。テスト用 `naive_ops()` は `f32` 逐次和・
/// 本番 `backend-cpu` は `f64` アキュムレータの `axis_reduce_sum` など
/// バックエンドごとに異なる）を経由するため、`Var::sum` へは委譲
/// **しない**（当初案は `naive_ops()` を使う統合テストで 1 階 VJP と
/// 再度乖離する回帰があった。実装時の実測で判明）。代わりに
/// `eval::reduce_bias_grad_rows`（`grad.rs::reduce_bias_grad` が呼ぶ
/// のと**同一のホスト関数**。行 `0..m` を列ごとに `f64` で逐次加算し
/// 最後に 1 回 `f32` へ downcast。`m == 1` の短絡〈符号付きゼロ保持〉
/// も同関数内で処理される）を直接呼んでホスト側で縮約後の値を計算し、
/// [`Tape::push_eager`]（`pub(crate)`）で `Op::Sum { input: g.id, dim:
/// Some(0) }` として子テープへ登録する。`child.ops()` の実装（テスト
/// 用 `naive_ops()` を含む）に依存せず 1 階 VJP と常に一致する。
///
/// `Op` タグを `Op::Sum { dim: Some(0) }` のまま維持することは、
/// `child.backward(..)`（さらなる微分。設計上 3 階微分は対象外だが、
/// 2 階勾配 `cg.grad(&b)` 自身を通常の `Var` として扱うために必要）に
/// 対する整合性を壊さない——`grad.rs::Op::Sum` の VJP（上流勾配を
/// `input` の shape へ broadcast するのみ）は前方値の精度に一切
/// 依存しないため、値を host 側で計算し直しても VJP の正しさは
/// 影響を受けない。
fn reduce_bias_grad_var<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    target_shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    let g_shape = g.shape();
    let is_row_axis_reduction = g_shape.len() == 2
        && target_shape.last() == Some(&g_shape[1])
        && target_shape[..target_shape.len().saturating_sub(1)]
            .iter()
            .all(|&d| d == 1);
    if !is_row_axis_reduction {
        return reduce_to(child, g, target_shape);
    }
    let g_val = {
        let nodes = child.nodes.borrow();
        materialize_fallible(&nodes, child.ops(), g.node_id())?.clone()
    };
    let reduced = crate::eval::reduce_bias_grad_rows(&g_val);
    let value = Tensor::new(reduced, &[g_shape[1]])?;
    let id = child.push_eager(
        Op::Sum {
            input: g.node_id(),
            dim: Some(0),
        },
        value,
    );
    Var::from_raw(child, id).reshape(target_shape)
}

/// `t`（親テープ側の実測値）の各要素が正かどうかを表す `Tensor<bool>`
/// を構築する（`Op::Relu` の VJP マスク。`v > 0.0`。NaN はマスク
/// 不成立——`grad.rs::elementwise_mul_mask` が `Op::Relu` に適用する
/// 規約と同じ）。
fn positive_mask(t: &Tensor<f32>) -> Result<Tensor<bool>, AutodiffError> {
    mask_from_pred(t, |v| v > 0.0)
}

/// `t`（親テープ側の実測値）の各要素へ述語 `pred` を適用した
/// `Tensor<bool>` を構築する（[`positive_mask`] の一般化。イシュー
/// #2062。`ScalarUnaryOp` の区分定数な導関数——`Abs`・`LeakyRelu`・
/// `Elu`・`Softplus`・`Clamp`・`Hardswish`——が使う host マスク構築の
/// 単一情報源）。
fn mask_from_pred(
    t: &Tensor<f32>,
    pred: impl Fn(f32) -> bool,
) -> Result<Tensor<bool>, AutodiffError> {
    let contiguous = t.contiguous();
    let data: Vec<bool> = contiguous
        .as_slice()
        .map(|s| s.iter().map(|&v| pred(v)).collect())
        .unwrap_or_default();
    Tensor::new(data, contiguous.shape()).map_err(AutodiffError::from)
}

/// broadcast 後の 2 テンソル（同一 shape）へ 2 項述語 `pred` を適用した
/// `Tensor<bool>` を構築する（[`mask_from_pred`] の 2 項版。
/// `ScalarBinaryOp::Pow`（`a == 0`／`b == 0` ガード）・`Maximum`／
/// `Minimum`（勝ち／タイ／`NaN` 判定）が使う。`a`／`b` は呼び出し元が
/// 既に同一 shape へ broadcast 済みであること（`Tensor::broadcast_
/// with` の戻り値等）。
fn pairwise_mask(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    pred: impl Fn(f32, f32) -> bool,
) -> Result<Tensor<bool>, AutodiffError> {
    let a_c = a.contiguous();
    let b_c = b.contiguous();
    let data: Vec<bool> = match (a_c.as_slice(), b_c.as_slice()) {
        (Some(sa), Some(sb)) => sa
            .iter()
            .zip(sb.iter())
            .map(|(&x, &y)| pred(x, y))
            .collect(),
        _ => Vec::new(),
    };
    Tensor::new(data, a_c.shape()).map_err(AutodiffError::from)
}

/// 損失 Op（イシュー #3082・決定記録 §26）の上流勾配 `g`（出力 shape `[]`）
/// から、`shape`（入力の要素 shape）へ broadcast した scale を作る。演算順は
/// 1 階 VJP に合わせる（`mse_loss_scale`: Mean は `(g·2)/n`・Sum は `g·2`。
/// その他の損失: Mean は `g/n`・Sum は `g`）。`g` は子テープ上の `Var` のまま
/// 扱い、損失が根でない場合や孫テープ再生でも二階の勾配が `g` を通って流れる。
fn loss_scale_bcast<'c>(
    child: &'c Tape,
    g: &Var<'c>,
    reduction: crate::var::Reduction,
    mse: bool,
    n: usize,
    shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    let g2 = if mse {
        g.mul(&child.var_no_grad(&Tensor::scalar(2.0f32)))?
    } else {
        *g
    };
    let s = match reduction {
        crate::var::Reduction::Mean => g2.div(&child.var_no_grad(&Tensor::scalar(n as f32)))?,
        crate::var::Reduction::Sum => g2,
    };
    s.broadcast_to(shape)
}

/// クラス軸を除いたサンプル数 `outer · inner`（`CrossEntropyLoss`／`NllLoss` の
/// `Mean` の分母。1 階 `cross_entropy_loss_vjp`／`nll_loss_vjp` と同じ）。
fn class_sample_count(shape: &[usize], class_dim: usize) -> usize {
    let outer: usize = shape[..class_dim].iter().product();
    let inner: usize = shape[class_dim + 1..].iter().product();
    outer * inner
}

/// `targets`（クラス添字）の onehot 位置を `(o·C + c)·inner + i` の平坦添字で
/// 列挙する。範囲外の添字は 1 階 VJP と同じくスキップする（forward が検査済み。
/// panic・範囲外アクセスを起こさない）。
fn onehot_positions(targets: &Tensor<i32>, shape: &[usize], class_dim: usize) -> Vec<usize> {
    let outer: usize = shape[..class_dim].iter().product();
    let axis_len = shape[class_dim];
    let inner: usize = shape[class_dim + 1..].iter().product();
    let data = crate::eval::dense_vec_i32(targets);
    let mut out = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let Some(&t) = data.get(o * inner + i) else {
                continue;
            };
            if t >= 0 && (t as usize) < axis_len {
                out.push((o * axis_len + t as usize) * inner + i);
            }
        }
    }
    out
}

/// onehot（f32。`CrossEntropyLoss` の `softmax − onehot` 用の定数）。
fn onehot_f32(
    targets: &Tensor<i32>,
    shape: &[usize],
    class_dim: usize,
) -> Result<Tensor<f32>, AutodiffError> {
    let mut data = vec![0f32; shape.iter().product()];
    for idx in onehot_positions(targets, shape, class_dim) {
        data[idx] = 1.0;
    }
    Tensor::new(data, shape).map_err(AutodiffError::from)
}

/// onehot マスク（bool。`NllLoss` の target 位置選択用）。
fn onehot_mask(
    targets: &Tensor<i32>,
    shape: &[usize],
    class_dim: usize,
) -> Result<Tensor<bool>, AutodiffError> {
    let mut data = vec![false; shape.iter().product()];
    for idx in onehot_positions(targets, shape, class_dim) {
        data[idx] = true;
    }
    Tensor::new(data, shape).map_err(AutodiffError::from)
}

/// [`Op::Where`] が `Op` payload に保持する `cond`（forward 時点で
/// `out_shape` へ broadcast 済みの f32 マスク。`c != 0.0` 判定契約。
/// `tape.rs::Op::Where` doc 参照）を `Var::where_cond` が要求する
/// `Tensor<bool>` へ変換する（イシュー #2062）。
fn f32_mask_to_bool(cond: &Tensor<f32>) -> Result<Tensor<bool>, AutodiffError> {
    mask_from_pred(cond, |v| v != 0.0)
}

#[cfg(test)]
mod contiguous_tests {
    use super::*;

    /// `Op::Contiguous`（`Var::contiguous`）の create_graph 対応
    /// （イシュー #2062）。`Var::contiguous` は `pub(crate)` のため
    /// `crates/autodiff/tests/create_graph.rs`（別クレート扱いの統合
    /// テスト）から直接呼べず、本クレート内部の単体テストで検証する。
    ///
    /// `x: [2, 2]` を permute（非 contiguous view）してから
    /// `contiguous()` を挟み `sum(y * y)` を取る。`Op::Permute` と同じ
    /// 添字並べ替えのため、二次形式全体としては `sum(x * x)` と数値上
    /// 同値になる——`finite_diff_hessian` 相当の独立クロスチェックは
    /// 統合テスト側の `hessian_permute_quadratic_matches_finite_
    /// difference` が既に担っており、本テストは「`Op::Contiguous` の
    /// replay／VJP が恒等として正しく機能する」ことを対角 Hessian が
    /// `2` になることで確認する（`Tape::backward` 経由の 1 階勾配とも
    /// 突合する）。
    #[test]
    fn contiguous_replay_and_vjp_are_identity() {
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let child = Tape::new_with_ops(crate::default_ops::naive_ops());
        let x = tape.var(&Tensor::new(vec![1.0f32, -2.0, 0.5, 3.0], &[2, 2]).unwrap());
        let y = x.permute(&[1, 0]).unwrap().contiguous().unwrap();
        let loss = y.mul(&y).unwrap().sum(None).unwrap();

        let cg = tape
            .backward_create_graph(&loss, &child)
            .expect("backward_create_graph に失敗（Op::Contiguous は対象）");
        let gx = cg
            .grad(&x)
            .expect("grad: クロステープ検査は通るはず")
            .expect("x は loss に到達するはず（1 階勾配）");

        // 1 階勾配が `Tape::backward`（既存経路）と一致すること
        // （`d/dx sum(permute(x)^2) = 2x`）。
        let expected = cg.first_order().get(&x).unwrap().unwrap();
        assert_eq!(gx.to_tensor().as_slice(), expected.as_slice());

        // 二階微分（対角 Hessian = 2）。
        let cx = cg.child_var(&x).unwrap().unwrap();
        let row0 = child
            .backward(
                &gx.narrow(0, 0, 1)
                    .unwrap()
                    .narrow(1, 0, 1)
                    .unwrap()
                    .sum(None)
                    .unwrap(),
            )
            .unwrap();
        let d2 = row0.get(&cx).unwrap().unwrap();
        assert_eq!(d2.get(&[0, 0]).unwrap(), 2.0);
    }
}

/// `ScalarBinaryOp::Maximum`／`Minimum`（イシュー #2062）の
/// `create_graph` 対応を検証する単体テスト。`Var::scalar_binary` は
/// `pub(crate)` のため `crates/autodiff/tests/create_graph.rs`（別
/// クレート扱いの統合テスト）から直接呼べず、`contiguous_tests` と
/// 同じ理由で本クレート内部の単体テストに置く。
///
/// `build_cgrads` の Maximum／Minimum 分岐（`pairwise_mask` による
/// 勝ち／タイ判定。`NaN` はいずれのマスクにも一致せず勾配 0）を、
/// 「a 勝ち」「b 勝ち」「タイ」の 3 要素を 1 テンソルへ詰めて検証する。
/// `a = x^2`（曲率 2）・`b` は `x` に依存しない定数（曲率 0）とし、
/// 要素ごとに `a`/`b` の大小関係を作為的に作る：
/// - index 0: `a`（`x^2 = 9`）> `b`（`0.2`）→ a 勝ち。曲率は `a` の
///   曲率 `2` がそのまま伝播する
/// - index 1: `a`（`x^2 = 0.01`）< `b`（`30.0`）→ b 勝ち。`b` は `x`
///   非依存のため出力は `x` に無関係——曲率は恒等的に 0
/// - index 2: `a == b`（`x^2` の値をそのまま `b` の定数へコピーして
///   タイを固定的に作る）→ タイ分岐（勾配 `0.5` 分割）。`a` 側にのみ
///   曲率が伝播するため実効曲率は `0.5 * 2 = 1`
#[cfg(test)]
mod maximum_minimum_tests {
    use super::*;

    fn maximum_probe<'t>(tape: &'t Tape, x0: &[f32; 3]) -> (Tape, Var<'t>, Var<'t>) {
        // a = x^2（曲率 2）。b は x 非依存の定数（曲率 0）で、index ごと
        // に a 勝ち／b 勝ち／タイを作為的に配置する。
        let x = tape.var(&Tensor::new(x0.to_vec(), &[3]).unwrap());
        let a = x.mul(&x).unwrap();
        let b_data = vec![0.2f32, 30.0, x0[2] * x0[2]];
        let b = tape.var_no_grad(&Tensor::new(b_data, &[3]).unwrap());
        let loss = a
            .scalar_binary(&b, ScalarBinaryOp::Maximum)
            .unwrap()
            .sum(None)
            .unwrap();
        let child = Tape::new_with_ops(crate::default_ops::naive_ops());
        (child, x, loss)
    }

    #[test]
    fn hessian_maximum_win_lose_tie_matches_expected_curvature() {
        let x0 = [3.0f32, 0.1, -2.0];
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let (child, x, loss) = maximum_probe(&tape, &x0);

        let cg = tape
            .backward_create_graph(&loss, &child)
            .expect("backward_create_graph に失敗（Op::ScalarBinary::Maximum は対象）");
        let gx = cg.grad(&x).unwrap().unwrap();
        let cx = cg.child_var(&x).unwrap().unwrap();

        let mut diag = [0.0f32; 3];
        for (i, d) in diag.iter_mut().enumerate() {
            let gi = gx.narrow(0, i, 1).unwrap().sum(None).unwrap();
            let row = child.backward(&gi).unwrap();
            let d2 = row.get(&cx).unwrap().unwrap();
            *d = d2.get(&[i]).unwrap();
        }
        assert_eq!(diag[0], 2.0, "index0（a 勝ち）は a=x^2 の曲率 2 のはず");
        assert_eq!(
            diag[1], 0.0,
            "index1（b 勝ち。b は x 非依存）は曲率 0 のはず"
        );
        assert_eq!(diag[2], 1.0, "index2（タイ。0.5 分割）は 0.5*2=1 のはず");
    }

    #[test]
    fn hessian_minimum_win_lose_tie_matches_expected_curvature() {
        // Minimum は勝敗の向きが逆になるだけ（`a_favored = false`）。
        // a 勝ち（Minimum なので a < b）: index0 は a=x^2=0.01 < b=30
        // → a 勝ち。index1 は a=x^2=9 > b=0.2 → b 勝ち。index2 はタイ。
        let x0 = [0.1f32, 3.0, -2.0];
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let x = tape.var(&Tensor::new(x0.to_vec(), &[3]).unwrap());
        let a = x.mul(&x).unwrap();
        let b_data = vec![30.0f32, 0.2, x0[2] * x0[2]];
        let b = tape.var_no_grad(&Tensor::new(b_data, &[3]).unwrap());
        let loss = a
            .scalar_binary(&b, ScalarBinaryOp::Minimum)
            .unwrap()
            .sum(None)
            .unwrap();
        let child = Tape::new_with_ops(crate::default_ops::naive_ops());

        let cg = tape
            .backward_create_graph(&loss, &child)
            .expect("backward_create_graph に失敗（Op::ScalarBinary::Minimum は対象）");
        let gx = cg.grad(&x).unwrap().unwrap();
        let cx = cg.child_var(&x).unwrap().unwrap();

        let mut diag = [0.0f32; 3];
        for (i, d) in diag.iter_mut().enumerate() {
            let gi = gx.narrow(0, i, 1).unwrap().sum(None).unwrap();
            let row = child.backward(&gi).unwrap();
            let d2 = row.get(&cx).unwrap().unwrap();
            *d = d2.get(&[i]).unwrap();
        }
        assert_eq!(diag[0], 2.0, "index0（a 勝ち）は a=x^2 の曲率 2 のはず");
        assert_eq!(
            diag[1], 0.0,
            "index1（b 勝ち。b は x 非依存）は曲率 0 のはず"
        );
        assert_eq!(diag[2], 1.0, "index2（タイ。0.5 分割）は 0.5*2=1 のはず");
    }

    #[test]
    fn maximum_nan_element_grad_is_zero() {
        // `NaN` を含む要素は `pairwise_mask` のいずれのマスク（勝ち／
        // タイ）にも一致しない（`!(x.is_nan() || y.is_nan())` ガード）
        // ため、`Maximum` 自体の両入力への勾配は 0 になる（`eval/
        // scalar.rs` の 1 階契約 `binary_partials(Maximum, NaN, ..)`
        // と同型の意味論）。`a` は `x` の恒等写像（`a = x`。曲率 0）と
        // し、`x^2` 等の曲率を持つ合成にしない——`0 * NaN = NaN`
        // という IEEE 754 の連鎖律で「a=x^2 のような合成先の勾配が
        // NaN 汚染される」現象は本テストの検証対象（Maximum 自体の
        // ゼロ勾配契約）とは別の話のため混同しない。
        let tape = Tape::new_with_ops(crate::default_ops::naive_ops());
        let child = Tape::new_with_ops(crate::default_ops::naive_ops());
        let x = tape.var(&Tensor::new(vec![f32::NAN, 1.0, 2.0], &[3]).unwrap());
        let b = tape.var_no_grad(&Tensor::new(vec![1.0f32, 2.0, 0.5], &[3]).unwrap());
        let loss = x
            .scalar_binary(&b, ScalarBinaryOp::Maximum)
            .unwrap()
            .sum(None)
            .unwrap();

        let cg = tape
            .backward_create_graph(&loss, &child)
            .expect("backward_create_graph に失敗");
        let gx = cg.grad(&x).unwrap().unwrap();
        let g0 = gx.to_tensor().get(&[0]).expect("shape 範囲内のはず");
        assert_eq!(g0, 0.0, "NaN 要素の 1 階勾配は 0 のはず");

        // 子テープ上に記録した 1 階勾配（`gx`。`build_cgrads` が
        // `Var` 演算列として再構成した値）が、既存 1 階経路
        // （`Tape::backward`。`backward.rs::backward_impl` 無変更）の
        // 結果である `cg.first_order()` と bit 完全一致することを
        // 確認する（`contiguous_replay_and_vjp_are_identity` と同型の
        // 突合方式）。NaN 要素（index0）を含む全要素で一致することが、
        // `pairwise_mask` の NaN 除外ガードが 1 階側の `eval::scalar::
        // binary_partials` 契約と矛盾なく再現されていることの根拠になる。
        let expected = cg.first_order().get(&x).unwrap().unwrap();
        assert_eq!(
            gx.to_tensor().as_slice(),
            expected.as_slice(),
            "子テープ上の 1 階勾配が既存 Tape::backward 経路と一致しない"
        );
    }
}

/// イシュー #2649: 活性化 5 kind は `create_graph`（高階微分）の対象外で、
/// `scalar_unary_replayable` が `false`（型付きエラーで拒否・panic しない）を
/// 返し続けることを固定する。
#[cfg(test)]
mod new_2649_tests {
    use super::*;

    #[test]
    fn gelu_family_kinds_are_replayable() {
        // #3081: Gelu・GeluTanh と、Gelu の Φ を記録する Erf（閉包性）。
        assert!(scalar_unary_replayable(ScalarUnaryOp::Gelu));
        assert!(scalar_unary_replayable(ScalarUnaryOp::GeluTanh));
        assert!(scalar_unary_replayable(ScalarUnaryOp::Erf));
    }

    #[test]
    fn new_2649_kinds_are_not_replayable() {
        for op in [
            ScalarUnaryOp::Selu,
            ScalarUnaryOp::Celu { alpha: 1.0 },
            ScalarUnaryOp::Softsign,
            ScalarUnaryOp::Hardsigmoid,
            ScalarUnaryOp::LogSigmoid,
        ] {
            assert!(!scalar_unary_replayable(op), "{op:?}");
        }
    }
}
/// 低精度 `Op::Conv2d` の fail-closed（イシュー #3080・決定記録 §21.3）。
/// `Var::conv2d_low_precision` は `pub(crate)`、F16 の typed ops を持つ
/// テスト用バックエンド（`test_support`）も crate 内限定のため、統合テスト
/// ではなく本クレート内の単体テストで検証する。
#[cfg(test)]
mod conv_low_precision_tests {
    use super::*;
    use fandhe_ai_tensor_core::ScalarDType;

    #[test]
    fn low_precision_conv2d_is_rejected_and_child_stays_empty() {
        let tape = Tape::new_with_ops(crate::test_support::low_precision_test_ops());
        let child = Tape::new_with_ops(crate::default_ops::naive_ops());
        let x = tape.var(&Tensor::full(&[1, 1, 3, 3], 0.5f32).unwrap());
        let w = tape.var(&Tensor::full(&[1, 1, 2, 2], 0.25f32).unwrap());
        let y = x
            .conv2d_low_precision(&w, None, [1, 1], [0, 0], [1, 1], 1, ScalarDType::F16)
            .unwrap();
        let loss = y.sum(None).unwrap();
        let err = tape
            .backward_create_graph(&loss, &child)
            .expect_err("低精度 conv2d は fail-closed で拒否されるはず");
        assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
        assert!(child.is_empty(), "入口検査は child へ書き込む前に拒否する");
    }
}
