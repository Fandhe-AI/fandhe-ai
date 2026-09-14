//! カーネルディスパッチ機構（TASK-1.9c・#46）。
//!
//! 単一の計算記述（[`BackendOps`] を受け取る関数）から CPU／CUDA／Metal
//! いずれのバックエンドのカーネルへも呼び分けられるようにする入口。
//! `device`（TASK-1.9a・#44）と同じ依存逆転構成を踏襲する: trait 定義を
//! 3 バックエンドクレートが依存できる本クレートに置き、各バックエンド
//! クレート（`backend-cpu`／`backend-cuda`／`backend-metal`）側で実装する
//! （`tensor-core` → `backend-*` の逆依存は作らない）。
//!
//! シグネチャは `docs/public-api-design.md` §4.2 の `BackendOps` trait案を
//! 正本としつつ、以下の点で拡張・簡略化している（同文書「TASK-1.9 実装
//! イシューで本文書との突合を行うこと」に対応。突合結果は同文書にも
//! 注記する）:
//!
//! - **`DeviceBuffer`／`upload`／`download` を含めない**。§4.2 が示す
//!   デバイス常駐バッファ型・転送 API は TASK-1.9b（#45）の担当であり、
//!   本イシュー時点で `tensor-core`・3 バックエンドクレートいずれにも
//!   存在しない（実装開始時に `git fetch origin main` で確認済み）。
//!   本イシューの受け入れ条件は「同一コードで 3 バックエンドのカーネルが
//!   呼び分けられる」（機構的な呼び分け）であり、既存カーネル入口
//!   （CPU `gemm_blis_parallel`・CUDA `CudaGemm::run_tiled_f32`・Metal
//!   `MetalGemm::dispatch_auto`）がいずれもホスト常駐 `&[f32]` を受け取り
//!   内部で H2D／D2H 転送を完結させる契約であるため、`DeviceBuffer` なしで
//!   本受け入れ条件を満たせる。§4.2 の `DeviceBuffer` 版シグネチャへの
//!   移行（`upload`／`download` の追加）は #45 のマージ後、`BackendOps` の
//!   非破壊拡張（デフォルトメソッド追加等）として TASK-1.9d（#47）以降で
//!   検討する
//! - 各メソッドはホスト常駐 [`Tensor<f32>`](crate::Tensor) を受け取り
//!   [`Tensor<f32>`](crate::Tensor) を返す（§4.2 の `DeviceBuffer<f32>` を
//!   `Tensor<f32>` に読み替えた形）。CPU 実装は転送コストが発生しないため
//!   このままで問題なく、CUDA／Metal 実装は各メソッド内で
//!   `Tensor::as_slice` → カーネル呼び出し（内部で H2D／D2H）→
//!   `Tensor::new` で完結させる
//! - 未実装カーネル（CUDA／Metal の elementwise・reduction。TASK-1.9c 時点
//!   では両バックエンドとも GEMM カーネルのみ実装済み）は
//!   [`crate::device::BackendError::Unsupported`]（本イシューで追加した
//!   非破壊拡張 variant）を返す fail-safe 実装とする。GPU 側
//!   elementwise・reduction カーネルの実装自体は本イシューのスコープ外
//!   （out-of-scope-tracking.md 対象。引き継ぎ先はユーザー承認を得て別
//!   Issue で追跡する）
//!
//! ディスパッチ規則（形状・HW 判定による経路選択）は TASK-11.2b（#68）の
//! 担当でありスコープ外（`docs/dispatch-rules-design.md`。TASK-11.2a・
//! #67）。既定デバイス選択ロジック（CUDA 既定有効化の構成決定含む）も
//! ユーザー承認必須のためスコープ外（`device` モジュールと同方針）。
//! 3 バックエンド横断の統合テストは TASK-1.9d（#47）が本格的に担当し、
//! 本イシューは受け入れ条件検証に必要な最小限のテストに留める。

use crate::Tensor;
use crate::buffer::{DeviceBuffer, DeviceBufferView, MemoryOps};
use crate::device::{BackendError, Device};
use crate::dispatch_failure::DispatchFailureCell;
use crate::error::ShapeError;
use crate::fusion::FusionPlan;
use crate::pool_core::PoolStats;
use crate::scalar_op::{ScalarBinaryOp, ScalarUnaryOp};
use crate::tensor::{checked_numel, checked_numel_for};
use crate::typed_ops::TypedOps;
use half::{bf16, f16};

/// [`BackendOps::sgd_step_device`] の 1 ステップ分のハイパーパラメータ
/// （イシュー #935・`docs/device-resident-update-design.md` §3.1）。
///
/// `fandhe_ai_autodiff::optim::sgd::SgdConfig`（ホスト参照実装。`lr`／
/// `momentum`／`dampening`／`weight_decay`／`nesterov` の 5 フィールド）と
/// 同じ意味論のフィールドに `is_first_step` を加えたもの。`autodiff`
/// クレートは `tensor-core` へ依存する側（`tensor-core` → `autodiff` の
/// 逆依存は作らない）であるため、`SgdConfig` をここへ再エクスポートせず
/// 独立した型として定義する（`fandhe_ai_autodiff::optim::device_store::
/// DeviceParamStore::step` が `SgdConfig` から本型へ変換して渡す）。
///
/// `is_first_step`: PyTorch `torch.optim.SGD` の momentum 初期化規則
/// （`docs/spec` 由来。`fandhe_ai_autodiff::optim::sgd` モジュールコメント
/// 「Algorithm」節）は「初回 step は `b ← g`、2 回目以降は
/// `b ← μ·b + (1−τ)·g`」であり、この分岐はパラメータの値そのものではなく
/// 呼び出し元（`DeviceParamStore`）が保持するステップカウンタに依存する。
/// `SgdConfig` 自体は構築後不変（`fandhe_ai_autodiff::optim::sgd::SgdConfig`
///参照）だが、`is_first_step` はステップごとに変化するため `SgdConfig` の
/// フィールドではなく本型（呼び出しごとに構築する値）のフィールドとする。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SgdStepConfig {
    /// 学習率。
    pub lr: f32,
    /// momentum 係数 `μ`。`0.0` は momentum 無効（`velocity` 引数は
    /// 無視してよい）。
    pub momentum: f32,
    /// dampening `τ`。
    pub dampening: f32,
    /// weight decay `λ`（L2 正則化。`torch.optim.SGD` と同じく `p` に
    /// 係数を乗じて勾配へ加算する）。
    pub weight_decay: f32,
    /// nesterov momentum を使うか。
    pub nesterov: bool,
    /// このパラメータ列にとって最初の `step()` 呼び出しか（momentum
    /// バッファの初期化分岐。上記フィールドドキュメント参照）。
    pub is_first_step: bool,
}

/// GEMM epilogue で適用する activation 種別（TASK-12.1f・#203）。
///
/// [`BackendOps::gemm_bias_act`] の第 4 引数として渡す。CUTLASS 系実測
/// （epilogue 融合で平均 1.38〜1.45 倍。イシュー #203）が動機の
/// Linear+bias+ReLU 相当パターンを表現できれば TASK-12.1f の受け入れ
/// 条件を満たせるため、まず `Relu` のみを持つ。`#[non_exhaustive]` は
/// 公開 API 非破壊（ガードレール条件・`.claude/rules/security.md`）を
/// 保ちながら将来 `Gelu`／`Sigmoid` 等を追加できるようにするため
/// （呼び出し側の網羅的 match を破壊しない。`GemmError`・`ParityError`
/// と同方針）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// activation なし（bias 加算のみ、または恒等関数）。
    None,
    /// `max(x, 0)`。`BackendOps::relu` と同一の定義を epilogue 内で適用する。
    Relu,
}

/// [`BackendOps::binary_elementwise_device`] が適用する 2 項 elementwise
/// 演算の種別（イシュー #1584。`BackendOps::add`／`mul` と同一の演算を
/// [`DeviceBuffer`] 常駐のまま実行するための選択子）。
///
/// `#[non_exhaustive]`: 公開 API 非破壊（ガードレール条件・
/// `.claude/rules/security.md`）を保つため（`Activation`／`MseReduction`
/// と同方針）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryElementwiseOp {
    /// `a + b`（`BackendOps::add` と同一の定義）。
    Add,
    /// `a * b`（`BackendOps::mul` と同一の定義）。
    Mul,
}

/// [`BackendOps::unary_elementwise_device`] が適用する単項 elementwise
/// 演算の種別（イシュー #1584。`BackendOps::relu`／`exp`／`tanh` と同一の
/// 演算を [`DeviceBuffer`] 常駐のまま実行するための選択子）。
///
/// `#[non_exhaustive]`: `BinaryElementwiseOp` と同方針。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryElementwiseOp {
    /// `max(x, 0)`（`BackendOps::relu` と同一の定義）。
    Relu,
    /// `exp(x)`（`BackendOps::exp` と同一の定義）。
    Exp,
    /// `tanh(x)`（`BackendOps::tanh` と同一の定義）。
    Tanh,
}

/// [`BackendOps::mse_loss`]／[`BackendOps::mse_loss_backward`] の縮約種別
/// （イシュー #1045・親イシュー #1043「カーネル融合・autodiff 実行モデル
/// の強化」）。
///
/// `fandhe_ai_autodiff::var::Reduction`（`Mean`／`Sum`）と同一の意味論を
/// 持つが、`tensor-core` → `autodiff` の逆依存は作れない（本ファイル
/// 冒頭コメント・`SgdStepConfig` と同じ整理）ため独立した型として定義
/// する。`autodiff` 側で `impl From<Reduction> for MseReduction` を用意し
/// 変換する（`var.rs` 参照）。
///
/// `#[non_exhaustive]`: 公開 API 非破壊（ガードレール条件・
/// `.claude/rules/security.md`）を保つため（`Activation` と同方針）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MseReduction {
    /// 全要素平均（`Σ(pred−target)² / n`）。
    Mean,
    /// 全要素総和（`Σ(pred−target)²`）。
    Sum,
}

/// [`BackendOps::huber_loss`]／[`BackendOps::huber_loss_backward`] が
/// 計算する要素損失の種別（イシュー #1739）。`d = pred − target` として
/// `Huber`（PyTorch `nn.HuberLoss(delta)`）は二次分岐に `0.5·d²`・
/// 線形分岐に `delta·(|d| − 0.5·delta)` を使う。`SmoothL1`（PyTorch
/// `nn.SmoothL1Loss(beta)`）は同じ折れ点構造だが二次分岐が
/// `0.5·d²/beta` に `beta` でスケールされる点のみ異なる（`beta = 1.0`
/// のとき両者は一致する）。1 個の `Op::HuberLoss`／1 組のカーネルで両者を
/// 表現するための選択子（`delta`／`beta` はどちらも同じ `f32` 引数
/// スロットで渡す。呼び出し側の意味論は `delta`／`beta` という呼称の
/// 違いのみ）。
///
/// `#[non_exhaustive]`: 公開 API 非破壊（ガードレール条件・
/// `.claude/rules/security.md`）を保つため（`Activation`／`MseReduction`
/// と同方針）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HuberKind {
    /// PyTorch `nn.HuberLoss(delta)` 相当（二次分岐 `0.5·d²`）。
    Huber,
    /// PyTorch `nn.SmoothL1Loss(beta)` 相当（二次分岐 `0.5·d²/beta`）。
    SmoothL1,
}

/// [`BackendOps::bce_loss`]／[`BackendOps::bce_loss_backward`] の入力種別
/// （イシュー #1737。親イシュー #1609「損失関数の拡張」）。
///
/// `BCELoss`（PyTorch）は `input` を確率（`[0, 1]`）として扱うのに対し、
/// `BCEWithLogitsLoss` は `input` を未正規化の logits として受け取り
/// カーネル内で sigmoid を適用する（`log(sigmoid(x))` を素朴に
/// `ln(1/(1+exp(-x)))` で計算すると `x` が大きい負値のとき桁落ち・
/// overflow するため、数値安定な合成式（`x>=0`: `(1-y)·x + ln(1+exp(-x))`、
/// `x<0`: `-y·x + ln(1+exp(x))`。`max(x,0) - x·y + ln(1+exp(-|x|))` と
/// 数式として等価だが `x - x·y` の桁落ちを避けるため乗算のみで書く）
/// を使う。`docs/compat-api-scope.md` §1.2 損失行参照）。両者は forward
/// の要素式・backward の `dInput` 式が異なるため、[`MseReduction`] とは
/// 独立にこの入力種別で分岐する（縮約種別自体は [`MseReduction`] を
/// 共用する。`Mean`／`Sum` の意味論は BCE 系でも同一のため）。
///
/// `#[non_exhaustive]`: 公開 API 非破壊（ガードレール条件・
/// `.claude/rules/security.md`）を保つため（`Activation`／`MseReduction`
/// と同方針）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BceKind {
    /// `input` は確率（`[0, 1]`）。`BCELoss` 相当。
    Probabilities,
    /// `input` は logits（範囲制約なし）。`BCEWithLogitsLoss` 相当
    /// （sigmoid をカーネル内に内包し数値安定な合成式で計算する）。
    Logits,
}

/// [`BackendOps::scatter`] が行う縮約種別（イシュー #1776）。
/// `torch.scatter`（`Overwrite`）と `torch.scatter_add`（`Add`）の
/// 差異を 1 メソッドへ集約する（`Var::scatter`／`Var::scatter_add` は
/// それぞれ本 enum の異なる値で共通実装 `scatter_impl` を呼ぶ）。
///
/// **`Add` の決定的集約順序・精度契約（実装側が必ず守ること）**:
/// `index`（同 shape の `src`）を row-major（`Tensor::contiguous()` と
/// 同じ末尾軸最速の C-order）で走査し、同一出力位置への複数回の書き込み
/// は「この走査順で逐次加算」した結果とする。出力位置ごとに `f64`
/// アキュムレータを `input[pos] as f64` で初期化し、走査順に
/// `acc += src[p] as f64` を適用したうえで、走査完了後に**1 回だけ**
/// `as f32` へ downcast する（`.claude/rules/coding-rust.md`
/// 「勾配の長軸縮約」節・要素積を伴わない単純な行方向和と同じ精度
/// 規律の先取り適用。CUDA は `double` アキュムレータ、Metal は
/// `crates/backend-metal/src/soft_f64.rs`〈イシュー #1659 で導入済みの
/// binary64 逐次和ソフトウェアエミュレーション〉が同じ契約を満たす
/// 既存手段として使える）。`Overwrite` はこの走査順で「最後に処理された
/// 値」が残る単純代入のため `f64` 昇格は不要。
///
/// 本 issue（#1776）の CPU 参照実装（`backend-cpu::gather_scatter`）・
/// ホストフォールバック（`autodiff::eval::scatter`）はいずれも単一
/// スレッド逐次ループでこの契約を実装する（並列化は将来の性能最適化
/// issue のスコープ・`.claude/rules/out-of-scope-tracking.md` 対象）。
///
/// `#[non_exhaustive]`: 公開 API 非破壊（ガードレール条件・
/// `.claude/rules/security.md`）を保つため（`Activation`／`MseReduction`
/// と同方針）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterReduce {
    /// 上書き（`torch.scatter` 相当）。同一位置への複数回書き込みは
    /// row-major 走査順で最後に処理された値が残る。
    Overwrite,
    /// 加算（`torch.scatter_add` 相当）。上記の決定的集約契約
    /// （`f64` アキュムレータ・row-major 走査順の逐次加算）に従う。
    Add,
}

/// [`BackendOps::interpolate`] が適用するリサンプリング方式（イシュー
/// #1757）。現時点では `Nearest`（最近傍）のみ持つが、後続 #1762
/// （bilinear）が同じ足場（`Op::Interpolate`／`BackendOps::interpolate`）
/// を共有できるよう enum で受ける設計とする。
///
/// `#[non_exhaustive]`: 公開 API 非破壊（ガードレール条件・
/// `.claude/rules/security.md`）を保つため（`ScatterReduce`／
/// `Activation` と同方針）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterpolateMode {
    /// 最近傍（`torch.nn.functional.interpolate(mode='nearest')`／
    /// `tf.image.resize(method='nearest')` 相当）。添字式は
    /// `src = (dst * in_size) / out_size`（整数除算＝床。PyTorch
    /// `mode='nearest'` は `floor(dst * (in/out))` を `f32` で計算する
    /// ため極端な形状で 1 要素ずれうる差異がある——本 variant は
    /// float を一切使わない整数演算のみで、3 バックエンド間で
    /// **構造的に bit 完全一致**する。`nearest-exact`〈`floor((dst+0.5)
    /// *in/out)`〉は対象外・別 variant として扱う）。
    Nearest,
}

/// [`BackendOps::captured_segment_key`]／[`BackendOps::run_captured_sgd_step_segment`]
/// が扱う 1 個のデバイスバッファの識別子（イシュー #1349・親 #1348・
/// ルート #1341 → #1269）。
///
/// CUDA Graph capture は「同じアドレス・同じ要素数のバッファへ、同じ
/// カーネル引数で launch する」ことを再利用の前提とする（`docs/
/// backend-cuda-graph-step-capture-design.md` §4.4）。`addr` はバックエンド
/// 実装（`backend-cuda::ops::CudaBackendOps`）がバッファのハンドルから
/// 取り出す値で、`tensor-core` 自体はその由来（`cudarc::driver::
/// DevicePtr::device_ptr` 等）を知らない・関与しない（バックエンド非依存
/// の型として定義するため）。`numel == 0`（空バッファ）は `addr == 0` で
/// 表す契約とする（呼び出し元がゼロ要素バッファを capture 対象に含めた
/// 場合の識別に使う。実際に capture するかどうかの判断＝空 graph の回避
/// は呼び出し元〈`run_captured_sgd_step_segment` 実装〉の責務）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SegmentResource {
    /// バックエンド固有のバッファ識別子（CUDA では device pointer）。
    pub addr: u64,
    /// バッファの要素数。
    pub numel: usize,
}

/// capture 済み CUDA Graph の再利用可否を判定するキー（イシュー #1349）。
///
/// `generation` はバックエンドの poison 状態機械の世代
/// （`backend-cuda::context_cache::current_generation` 等）と一致させる
/// ことで、`invalidate` による回復（poison → 新世代）を跨いだ古い graph
/// を再利用しない（世代不一致は「別のデバイスコンテキストのグラフ」を
/// 意味し、キャッシュ側で evict する）。`config_key` は当該区間の
/// カーネル起動パラメータ（学習率等のハイパーパラメータ・`is_first_step`
/// 等の分岐フラグ）を呼び出し元が `u64` へ畳み込んだ値で、設定変更を
/// 検出して再 capture を促す。`resources` は区間が触れる全バッファの
/// [`SegmentResource`]（`Vec` の順序も含めて比較する。同じ集合でも順序が
/// 異なれば別キー＝別 graph として扱う。これは呼び出し元が毎回同じ順序で
/// 構築する契約であるため実害はなく、`Hash`/`Eq` の実装を単純に保つ
/// ための割り切り）。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SegmentKey {
    /// バックエンドの poison 状態機械の世代（世代不一致の古い graph を
    /// 再利用しないためのキー要素）。
    pub generation: u64,
    /// 当該区間のカーネル起動パラメータを畳み込んだ値（設定変更の検出）。
    pub config_key: u64,
    /// 当該区間が触れる全バッファの識別子（順序を含めて比較する）。
    pub resources: Vec<SegmentResource>,
}

/// [`BackendOps::run_captured_sgd_step_segment`] が実際に capture したか、
/// 既存の graph を再生（replay）しただけかを呼び出し元へ伝える（イシュー
/// #1349。呼び出し元の launch 回数計測・テストでの制御フロー検証に使う）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentRun {
    /// 新規に stream capture → instantiate → 初回 launch した。
    Captured,
    /// 既存のキャッシュ済み graph を launch（再生）した。
    Replayed,
}

/// [`BackendOps::gemm_checksum`] の読み戻しモード（イシュー #1339）。
///
/// framework-compare の gemm 計測窓では、GEMM 出力 `C = A@B` を毎反復
/// ホストへ D2H した上で「縮退検出用の全要素和（checksum）」をホスト側
/// `f64` 逐次和で求め直しており、`docs/perf/cuda-gemm-reuse-phase-
/// breakdown.md`・`metal-gemm-reuse-phase-breakdown.md` の実測でこの
/// `host_copy`＋`checksum` の 2 段がハーネス計測窓の 66〜75% を占める
/// ことが確定した（イシュー #1338 承認）。本 enum は「毎反復は checksum
/// のみ 8 バイト読み戻す」（`ChecksumOnly`）か「加えて `C` 自体もホストへ
/// download する」（`WithOutput`。末尾反復の parity 検証用）かを選ぶ。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChecksumReadout {
    /// checksum（8 バイト）のみ読み戻す。`output` は `None`。
    ChecksumOnly,
    /// checksum に加えて `C` 全体もホストへ download する。
    WithOutput,
}

/// [`BackendOps::gemm_checksum`] の戻り値。`checksum` は `C = A@B`
/// （論理領域 `m×n`）の全要素和を **`f64`（Metal は Neumaier 補償和で
/// `f64` 相当）アキュムレータ・固定順序**で求めた値（決定的。同一入力・
/// 同一カーネル選択であれば bit 決定的に同じ値を返す契約。CPU／CUDA／
/// Metal の各実装 doc 参照）。`output` は
/// [`ChecksumReadout::WithOutput`] のときのみ `Some`（[`BackendOps::gemm`]
/// と bit 同一の `Tensor<f32>`）。
#[derive(Debug, Clone)]
pub struct GemmChecksum {
    /// `C` の全要素和（f64 アキュムレータ・固定順序で決定的）。
    pub checksum: f64,
    /// [`ChecksumReadout::WithOutput`] のときのみ `Some`。
    pub output: Option<Tensor<f32>>,
}

/// [`BackendOps::linalg_qr`] の戻り値（イシュー #1621。`docs/
/// autodiff-linalg-design.md`）。`torch.linalg.qr(mode="reduced")` と
/// 同じ reduced QR（`A: [m,n]` → `q: [m,k]`・`r: [k,n]`、`k = min(m,n)`）。
/// `r` の対角は非負に正規化する（`fandhe_ai_autodiff::eval::linalg`
/// と本クレートの実装が同一符号規約を採る契約。設計文書 §3.5「符号・
/// ゲージ規約」）。フィールドは `pub`（`MseReduction`／`Activation` の
/// ような `#[non_exhaustive]` enum ではなく、バックエンド実装が値を
/// 直接構築する struct のため。`GemmChecksum` と同方針）。
#[derive(Debug, Clone)]
pub struct QrFactors {
    /// `[m, k]`（`k = min(m, n)`）。列直交（`QᵀQ ≈ I_k`）。
    pub q: Tensor<f32>,
    /// `[k, n]`（`k = min(m, n)`）。上三角・対角非負。
    pub r: Tensor<f32>,
}

/// [`BackendOps::linalg_svd`] の戻り値（イシュー #1621）。
/// `torch.linalg.svd(A, full_matrices=False)` と同じ reduced SVD
/// （`A: [m,n]` → `u: [m,k]`・`s: [k]`・`vh: [k,n]`、`k = min(m,n)`）。
/// `s` は降順（同値は安定ソート）に正規化する契約（設計文書 §3.5）。
#[derive(Debug, Clone)]
pub struct SvdFactors {
    /// `[m, k]`（`k = min(m, n)`）。列直交。
    pub u: Tensor<f32>,
    /// `[k]`。特異値（降順・非負）。
    pub s: Tensor<f32>,
    /// `[k, n]`（`k = min(m, n)`）。行直交（`V^T`）。
    pub vh: Tensor<f32>,
}

/// [`BackendOps::linalg_matrix_norm`] が計算する行列ノルムの種類
/// （イシュー #1621。`torch.linalg.matrix_norm` の `ord` 引数のうち
/// facade が対応する 5 種）。
///
/// `#[non_exhaustive]`: 公開 API 非破壊（ガードレール条件・
/// `.claude/rules/security.md`）を保つため（`Activation`／`MseReduction`
/// と同方針）。将来 `ord=p`（任意次数）・`dim` 指定版を追加しうる
/// （設計文書「スコープ外」節）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixNormOrd {
    /// Frobenius ノルム（`√Σ a_ij²`）。
    Fro,
    /// 最大絶対列和（`max_j Σ_i |a_ij|`）。
    One,
    /// 最大絶対行和（`max_i Σ_j |a_ij|`）。
    Inf,
    /// 核ノルム（特異値の総和）。
    Nuc,
    /// スペクトルノルム（最大特異値）。
    Spectral,
}

/// [`BackendOps::vector_norm`] が計算するベクトルノルムの種類
/// （イシュー #1723。`torch.norm`／`tf.norm` の L1／L2 相当。
/// [`MatrixNormOrd`] と対をなすがこちらは軸方向縮約〈`dim:
/// Option<usize>`〉を持つ点が異なるため別 enum とする）。
///
/// `#[non_exhaustive]`: 公開 API 非破壊（ガードレール条件・
/// `.claude/rules/security.md`）を保つため（`MatrixNormOrd` と同方針）。
/// 将来 `Lp`（任意次数）を追加しうる（実装計画 §8「スコープ外」節）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorNormOrd {
    /// L1 ノルム（`Σ |x_i|`）。
    L1,
    /// L2 ノルム（`√Σ x_i²`）。
    L2,
}

/// [`BackendOps::gru_backward`] の戻り値型エイリアス（イシュー #1647）。
/// `(d_pre_i, d_pre_h, dh_prev_direct)`（順に `[B, 3H]`・`[B, 3H]`・
/// `[B, H]`）。`clippy::type_complexity` 回避のための命名（doc は
/// `gru_backward` 側に集約する）。
pub type GruBackwardOutput = (Tensor<f32>, Tensor<f32>, Tensor<f32>);

/// 各バックエンド（CPU／CUDA／Metal）が実装するカーネル入口
/// （`docs/public-api-design.md` §4.2。差分はモジュール冒頭コメント参照）。
///
/// object-safe に設計している（`&dyn BackendOps` として扱える。
/// [`ops_for`] が複数バックエンドを横断して選択する際に使用する）。
/// v1 は PoC-v2-5 実測 API（`MetalOps`）のスコープに合わせて `f32` 固定
/// とする（f16 経路のジェネリック化は §4.2 6-8 のとおり保留）。
///
/// 公開 API はすべて safe。`unsafe` は各バックエンド実装内部の FFI 境界
/// （`cudarc`・`objc2` 系呼び出し）に閉じ込める
/// （`.claude/rules/coding-rust.md`）。
pub trait BackendOps {
    /// このインスタンスが対応する [`Device`]（呼び出し元がログ・
    /// エラーメッセージで識別するために使う）。
    fn device(&self) -> Device;

    /// このバックエンドの [`MemoryOps`]（確保・アップロード・ダウンロード）
    /// 実装への参照（イシュー #935・`docs/device-resident-update-design.md`
    /// §3.1）。
    ///
    /// # デフォルト実装（非破壊拡張）
    /// 既定は `None`（`MemoryOps` を持たない）。`BackendOps` を
    /// `MemoryOps` の supertrait にする案（`buffer.rs` モジュール冒頭
    /// コメント旧稿）は crates.io 公開済み trait への破壊的変更となる
    /// ため不採用と確定した（設計文書 §3.1）。本デフォルトメソッド追加は
    /// `gemm_bias_act`／`run_fused` と同じ非破壊拡張パターン（`BackendOps`
    /// を実装する外部クレートは何もしなくても既存実装のままコンパイル
    /// が通る）。
    ///
    /// `fandhe_ai_autodiff::optim::device_store::DeviceParamStore::new` が
    /// `tape.ops().memory_ops()` を呼び、`None` の場合は
    /// [`BackendError::Unsupported`] としてデバイス常駐パラメータ更新を
    /// 拒否する（fail-closed。「`memory_ops()` を呼ぶフォールバック合成は
    /// 設けない」という設計文書 §3.2 改訂の確定事項に従い、`Some` を返す
    /// バックエンドのみがこの経路をサポートする）。CPU／CUDA／Metal の 3
    /// バックエンドはいずれも本デフォルトを `Some(self)` へオーバーライド
    /// する（各バックエンドクレートの `ops.rs` 参照）。
    fn memory_ops(&self) -> Option<&dyn MemoryOps> {
        None
    }

    /// `f64` 演算本体（[`TypedOps<f64>`]）への capability accessor
    /// （イシュー #1687・`docs/backend-dtype-dispatch-design.md`）。
    ///
    /// # デフォルト実装（非破壊拡張）
    /// 既定は `None`（f64 演算未対応。fail-closed）。`memory_ops` と同じ
    /// 非破壊拡張パターンで、`BackendOps` を実装する外部クレートは何も
    /// しなくても既存実装のままコンパイルが通る。CPU 実装は #1697 が
    /// `Some(self)` へオーバーライドする。
    fn typed_ops_f64(&self) -> Option<&dyn TypedOps<f64>> {
        None
    }

    /// `half::f16` 演算本体（[`TypedOps<f16>`]）への capability accessor。
    /// 同上のデフォルト（`None`）。CPU 実装は #1698 が担当する。
    fn typed_ops_f16(&self) -> Option<&dyn TypedOps<f16>> {
        None
    }

    /// `half::bf16` 演算本体（[`TypedOps<bf16>`]）への capability
    /// accessor。同上のデフォルト（`None`）。CPU 実装は #1699 が担当する。
    fn typed_ops_bf16(&self) -> Option<&dyn TypedOps<bf16>> {
        None
    }

    /// SGD の 1 パラメータ分の更新をデバイス上で in-place に実行する
    /// （イシュー #935・`docs/device-resident-update-design.md` §3.2）。
    ///
    /// `param`／`grad`／`velocity`（momentum 有効時のみ）はいずれも
    /// このバックエンド自身が確保した [`DeviceBuffer<f32>`]
    /// （[`MemoryOps::alloc_zeroed`]／[`MemoryOps::upload`] の戻り値）を
    /// 要求する契約。呼び出し元（`fandhe_ai_autodiff::optim::device_store::
    /// DeviceParamStore::step`）は毎ステップ `grad` のみをアップロードし
    /// `param`／`velocity` は前ステップから使い回すことで、param の
    /// ホスト再アップロードを排除する（本イシューの受け入れ条件）。
    ///
    /// **呼び出し元は全パラメータを連結した単一バッファで 1 回だけ呼ぶ**
    /// （イシュー #1023「パラメータ横断の単一連結バッファ化」）。
    /// `param`／`grad`／`velocity` はいずれもパラメータ数だけ個別に渡す
    /// のではなく、`DeviceParamStore` が全パラメータを 1 本の shape
    /// `[total_numel]` バッファへ連結して常駐させ、`step()` ごとに本
    /// メソッド（`sgd_step_device_tracked` 経由）を 1 回だけ起動する。
    /// 本メソッド自体は要素単位で shape 非依存に定義されているため、
    /// この呼び出し規約変更はシグネチャ・カーネル実装（CPU／CUDA／
    /// Metal のいずれも）に一切変更を要求しない。
    ///
    /// 更新式は `fandhe_ai_autodiff::optim::sgd`（`Sgd::step` ホスト参照
    /// 実装）と同一の項順序（weight_decay → momentum〈`is_first_step` で
    /// `b ← g` 分岐〉→ nesterov → 減算）を 3 バックエンドで揃える契約
    /// （設計文書 §5.2）。カーネル境界検査は省略しない（REQ-8・
    /// `.claude/rules/coding-rust.md`）。
    ///
    /// # デフォルト実装（非破壊拡張）
    /// 既定は常に [`BackendError::Unsupported`] を返す fail-closed
    /// （`memory_ops()` を呼ぶフォールバック合成は設けない。設計文書
    /// §3.2 改訂）。CPU／CUDA／Metal はこのデフォルトを実カーネルで
    /// オーバーライドする。
    ///
    /// # エラー
    /// - `param`／`grad`／`velocity` のいずれかがこのバックエンドの
    ///   ハンドル型へダウンキャストできない・デバイスが一致しない →
    ///   [`BackendError::DeviceMismatch`]
    /// - shape が一致しない → [`BackendError::ShapeMismatch`]
    /// - `config.momentum != 0.0` なのに `velocity` が `None` →
    ///   [`BackendError::Unsupported`]
    fn sgd_step_device(
        &self,
        _param: &mut DeviceBuffer<f32>,
        _grad: &DeviceBuffer<f32>,
        _velocity: Option<&mut DeviceBuffer<f32>>,
        _config: &SgdStepConfig,
    ) -> Result<(), BackendError> {
        Err(BackendError::Unsupported(
            "sgd_step_device: default fail-safe (no in-place SGD kernel available)".into(),
        ))
    }

    /// [`BackendOps::sgd_step_device`] と同型だが、Metal のコマンド
    /// バッファ共有（イシュー #1017・`docs/backend-metal-command-
    /// batching-design.md`）向けに共有失敗トークン
    /// [`DispatchFailureCell`] を追加引数として受け取る非破壊拡張
    /// （`gemm_bias_act`／`run_fused` と同じ「デフォルトメソッド追加」
    /// パターン。`BackendOps` の SemVer 非破壊拡張）。
    ///
    /// # デフォルト実装
    /// 既定は `token` を無視して [`BackendOps::sgd_step_device`] へ
    /// そのまま委譲する。CPU は dispatch ごとに同期実行するため実行時
    /// エラーが呼び出し元に即座に返り、遅延失敗トークンを必要としない
    /// （このデフォルトのままでよい）。CUDA はイシュー #1013
    /// （`docs/backend-cuda-async-execution-design.md` §5）でカーネル
    /// 起動直後の都度 `synchronize()` を除去し非同期実行契約へ移行した
    /// が、本 `token`（`DispatchFailureCell`）は使わずオーバーライドも
    /// しない（このデフォルトのまま）。`backend-cuda::context_cache` は
    /// ordinal 単位の poison 状態機械（`begin_driver_call`／
    /// `observe_driver_result`／`observe_cuda_result`／`is_poisoned`。
    /// 単一ストリームの FIFO 順序保証を前提に sticky エラー観測時点で
    /// ordinal を poison する設計）を備え、PR #1064（イシュー #1013 の
    /// codex-review P0 指摘への対応）で `backend-cuda::ops`／
    /// `backend-cuda::memory` の `BackendOps`／`MemoryOps` 実装境界
    /// （`with_driver_call` ヘルパー）へ結線済みである
    /// （`docs/backend-cuda-async-execution-design.md` §12）。
    /// これにより、`sgd_step_device` 自身のカーネル起動が sticky な
    /// 実行時エラーを引き起こした場合、その ordinal 上で最初に
    /// `observe_cuda_result` が観測した時点（同一ステップの起動自体・
    /// 別の同一ステップ内 driver 呼び出し・または別テンソル演算の
    /// いずれか）で poison 化され、以降の `sgd_step_device` 呼び出しは
    /// `begin_driver_call` の拒否により `Err` を返す。`DeviceParamStore::
    /// step` は `sgd_step_device_tracked` が返す `Err` を常に
    /// `poisoned.store(true, ..)` へ変換する（`device_store.rs`
    /// `step` 実装参照）ため、この `Err` は必ず `StorePoisoned` への
    /// 自己遷移につながる。ただし検出は「次に同一 ordinal 上で
    /// driver 呼び出しが起きた時点」に限られ、poison からの**回復**
    /// （`context_cache::invalidate_with` の呼び出し）は #1062 へ
    /// 引き継いだままである。Metal のみ
    /// `backend-metal::ops::MetalBackendOps` がオーバーライドし、
    /// `MetalContext::encode` と**同一ロック区間で** `token` をバッチへ
    /// 登録する（encode と登録の間に別スレッドの `synchronize` が
    /// 割り込む競合を防ぐ。設計文書 §3.7 (2)）。
    ///
    /// `fandhe_ai_autodiff::optim::device_store::DeviceParamStore::step`
    /// が呼び出し元となり、自身が保持する `failure_token` を渡す
    /// （4 つの状態機械エントリ全てが `token.is_set()` を検査して
    /// 自己 poison する。`device_store.rs` モジュール冒頭コメント参照）。
    fn sgd_step_device_tracked(
        &self,
        param: &mut DeviceBuffer<f32>,
        grad: &DeviceBuffer<f32>,
        velocity: Option<&mut DeviceBuffer<f32>>,
        config: &SgdStepConfig,
        _token: &DispatchFailureCell,
    ) -> Result<(), BackendError> {
        self.sgd_step_device(param, grad, velocity, config)
    }

    /// 学習 step の一区間（イシュー #1349 では
    /// [`Self::sgd_step_device_tracked`] の update 区間のみ）を CUDA Graph
    /// で capture・再利用できるかを判定し、可能なら [`SegmentKey`] を返す
    /// （opt-in・既定 OFF。`docs/backend-cuda-graph-step-capture-design.md`
    /// §4.4）。
    ///
    /// `resources` に渡す各 [`DeviceBuffer<f32>`] は当該区間が読み書きする
    /// 全バッファ（例: `param`／`grad_staging`／`velocity`）を呼び出し元
    /// が**毎回同じ順序**で並べる契約（[`SegmentKey::resources`] の順序
    /// 込み比較）。`config_key` は当該区間のカーネル起動パラメータ
    /// （学習率等）を呼び出し元が `u64` へ畳み込んだ値。
    ///
    /// 呼び出し元（`fandhe_ai_autodiff::optim::device_store::
    /// DeviceParamStore::step`）は本メソッドが `Ok(Some(key))` を返した
    /// ときのみ [`Self::run_captured_sgd_step_segment`] を呼ぶ（`Ok(None)`
    /// は「このバックエンド・現在の設定では capture 非対応」を意味し、
    /// 呼び出し元は区間を直接実行する現行経路へフォールバックする）。
    ///
    /// # デフォルト実装（非破壊拡張）
    /// 既定は常に `Ok(None)`（graph 機構を持たないバックエンド・opt-in
    /// OFF の既定状態）。CUDA opt-in ON 時のみ
    /// `backend-cuda::ops::CudaBackendOps` がオーバーライドする。CPU・
    /// Metal はこのデフォルトのまま（graph 機構自体を持たない）。
    fn captured_segment_key(
        &self,
        _resources: &[&DeviceBuffer<f32>],
        _config_key: u64,
    ) -> Result<Option<SegmentKey>, BackendError> {
        Ok(None)
    }

    /// [`Self::captured_segment_key`] が返した `key` に対応する SGD 更新
    /// 区間（[`Self::sgd_step_device_tracked`] 相当）を capture（初回）
    /// または再生（2 回目以降）する（イシュー #1349）。
    ///
    /// **codex-review P0 指摘対応（任意クロージャの public 安全 API 化を
    /// 撤回）**: 旧稿は「`resources: &mut [&mut DeviceBuffer<f32>]` と
    /// 任意クロージャ `body: &mut dyn FnMut(&mut [&mut DeviceBuffer<f32>])`
    /// を受け取り、実装が `resources` のアドレスのみを再検証してから
    /// `body` を呼ぶ」形だった。この形には、`body` が **Rust クロージャの
    /// 環境キャプチャ経由で `resources` に含まれない外部の
    /// `DeviceBuffer<f32>` を直接触れる**（`resources` 引数を無視して
    /// クロージャがキャプチャした変数へ書き込む）という抜け道があり、
    /// 実装側の「`resources` のアドレス一致」再検証はその外部バッファを
    /// 一切カバーしない。CUDA Graph は capture 時点で触れた全アドレスを
    /// 焼き込むため、そのバッファが後で drop・再利用されると replay が
    /// 解放済み／別用途のアドレスを参照するメモリ安全性違反になりうる
    /// （`body` の型が `dyn FnMut` である限り、この抜け道を型システムで
    /// 塞ぐ手段がない）。本メソッドはこの型を公開 API から排除し、
    /// 「区間が触れる全リソースをメソッド自身の引数として直接受け取り、
    /// 区間本体（SGD 更新）もこのメソッドの実装が固定的に行う」——
    /// 任意クロージャを一切受け取らない操作記述に置き換える。これにより
    /// capture 中に触れうる `DeviceBuffer<f32>` は `param`／`grad`／
    /// `velocity` の 3 引数に限定され、実装のアドレス再検証がこの区間が
    /// 触れる全リソースを漏れなくカバーすることを型で保証する
    /// （`docs/backend-cuda-graph-step-capture-design.md` §4.4 追記）。
    ///
    /// `param`／`grad`／`velocity` は [`Self::captured_segment_key`] を
    /// 得たときと**同一の借用**で渡す契約: `SegmentKey` 自身はバッファの
    /// 所有権・借用を保持しない値型（`addr`／`numel` のみを畳み込んだ
    /// 識別子）であるため、呼び出し元が対応するバッファを drop した後に
    /// 本メソッドを呼ぶと、解放済み（または別バッファへ再利用済み）の
    /// アドレスを参照する古い graph を安全確認なしに再生してしまいうる
    /// （メモリ安全性違反）。実装は **replay 直前に** `param`／`grad`／
    /// `velocity` から導出した現在のアドレス集合が `key.resources`
    /// （capture 時点で刻印済み）と一致することを再検証し、不一致
    /// （呼び出し元の借用が `key` 発行時と異なる＝契約違反）なら
    /// [`BackendError::InvalidArgument`] で replay も新規 capture も
    /// 行わず拒否する（fail-closed）。
    ///
    /// `velocity` は [`Self::captured_segment_key`] の `resources` に
    /// velocity を含めた呼び出しと対でなければならない（`config.momentum
    /// != 0.0` かつ velocity 引数が異なる本メソッド・`captured_segment_key`
    /// を混在させない）。
    ///
    /// 初回（キャッシュミス）は stream capture を開始し、実装内部で
    /// [`Self::sgd_step_device_tracked`] 相当の更新を 1 回実行してから
    /// capture を終了・instantiate し、得られた graph をキーへ紐づけて
    /// キャッシュしたのち初回 launch する（[`SegmentRun::Captured`]）。
    /// 2 回目以降（キャッシュヒット）は更新を再実行せず、キャッシュ済み
    /// graph をそのまま launch する（[`SegmentRun::Replayed`]）。
    ///
    /// 区間本体が `Err` を返した場合、capture 自体は安全に終了させた
    /// うえでその `Err` を呼び出し元へ返す（capture 失敗の graph は
    /// キャッシュに残さない）。
    ///
    /// # デフォルト実装（非破壊拡張）
    /// 既定は常に [`BackendError::Unsupported`] を返す fail-closed。
    /// 呼び出し元は [`Self::captured_segment_key`] が `Some` を返した
    /// ときのみ本メソッドを呼ぶ契約であり、デフォルト実装のまま
    /// `Some` を返すバックエンドは存在しない（既定 `captured_segment_key`
    /// が常に `Ok(None)` を返すため、通常この `Err` へは到達しない。
    /// 到達した場合は呼び出し元・バックエンド実装間の契約違反であり、
    /// fail-closed に拒否する）。
    #[allow(clippy::too_many_arguments)]
    fn run_captured_sgd_step_segment(
        &self,
        _key: SegmentKey,
        _param: &mut DeviceBuffer<f32>,
        _grad: &DeviceBuffer<f32>,
        _velocity: Option<&mut DeviceBuffer<f32>>,
        _config: &SgdStepConfig,
        _token: &DispatchFailureCell,
    ) -> Result<SegmentRun, BackendError> {
        Err(BackendError::Unsupported(
            "run_captured_sgd_step_segment: default fail-safe (no CUDA Graph capture \
             mechanism available for this backend)"
                .into(),
        ))
    }

    /// 行列積 `C = A @ B` を計算する（`A: [m, k]`・`B: [k, n]` の 2 次元
    /// テンソルのみ受け付ける。shape 不整合は
    /// [`BackendError::ShapeMismatch`]）。
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;

    /// `gemm` と同じ行列積だが、**`crate::precision`（`backend-cuda`）の
    /// TF32 opt-in フラグ（`set_cuda_tf32_gemm_enabled`）の状態に関わらず
    /// 常に FP32 厳密で計算する**ことを契約するエントリ。
    ///
    /// `docs/cuda-tf32-optin-api-decision.md`・`backend-cuda::precision`
    /// モジュール冒頭コメントの契約「適用範囲は `CudaBackendOps::gemm`
    /// （素の公開 GEMM 入口）のみ。学習経路は本イシューのスコープ外の
    /// まま FP32 で動作する」を、`autodiff::grad`（VJP。イシュー #1211）
    /// のように **バックエンド非依存 `dyn BackendOps` 経由**で GEMM を
    /// 呼ぶ学習経路が満たすための入口。`ops.gemm(..)` を直接呼ぶと、
    /// CUDA では opt-in フラグが有効な間バックプロパゲーションが暗黙に
    /// TF32 化してしまう（codex-review 指摘。PR #1223）。
    ///
    /// 既定実装は `self.gemm(a, b)` に委譲する（TF32 の概念を持たない
    /// CPU・Metal はこれで契約を満たす）。TF32 opt-in を持つ
    /// `backend-cuda::CudaBackendOps` のみ、フラグを一切参照しない FP32
    /// 厳密経路（`run_tiled_f32`）へオーバーライドする。
    fn gemm_fp32_strict(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        self.gemm(a, b)
    }

    /// バッチ行列積（rank≥2。バッチ次元は NumPy 互換ブロードキャスト）を
    /// 計算する（イシュー #1715。spec REQ-9 2026-09-12 追記 Tier 1
    /// 「バッチ行列積」・`docs/compat-api-scope.md` §1.2）。
    ///
    /// `fandhe_ai_autodiff::Var::matmul`（rank≥3 を含む場合の分岐先）から
    /// 呼ばれる。shape 検査は [`crate::matmul_out_shape`] に委譲する
    /// （rank・内部次元・バッチブロードキャストの不整合は
    /// [`BackendError::ShapeMismatch`]）。
    ///
    /// # デフォルト実装（既定合成。非破壊拡張）
    ///
    /// rank 2 同士は [`Self::gemm`] へ直接委譲する（同一カーネル呼び出しで
    /// bit 同一を構造的に保証。`crates/backend-cpu` の `gemm_batched_parity`
    /// テストが検証する）。rank≥3 を含む場合は、各オペランドを
    /// `[batch..., m, k]`／`[batch..., k, n]` の contiguous 3 次元
    /// `[B, m, k]`／`[B, k, n]` へ正規化（バッチ shape が出力と異なる
    /// 〈broadcast が必要〉場合は `broadcast_to` → `contiguous()` →
    /// `reshape`、等しい場合は `contiguous()` → `reshape`。中間軸の
    /// broadcast〈stride 0〉を `reshape` へ直接渡すと
    /// `ShapeError::NonContiguousReshape` になるため、必ず `broadcast_to`
    /// で実体化してから `reshape` する）したうえで、出力バッチ `i` ごとに
    /// `narrow(0, i, 1).reshape([m, k])`（`Tensor::as_slice` は `offset`
    /// 起点で `numel` 分の窓を返す offset-aware 実装のため、narrow 後の
    /// view をそのまま [`Self::gemm`] へ渡せる）で [`Self::gemm`] を呼び、
    /// 結果を `[batch..., m, n]` の contiguous 出力へ順に書き込む
    /// （per-batch 2 次元 `gemm` と同一カーネル・同一累積順序のため
    /// bit 同一）。専用バッチカーネル（CUDA／Metal）へのオーバーライドは
    /// 後続イシュー（#1716／#1717）が担う。
    fn gemm_batched(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        default_gemm_batched(self, a, b, BatchedGemmKind::Standard)
    }

    /// [`Self::gemm_batched`] と同じバッチ行列積だが、各バッチの計算に
    /// [`Self::gemm_fp32_strict`] を使う（CUDA の TF32 opt-in フラグに
    /// 追従しない厳密 FP32 経路。イシュー #1715）。
    ///
    /// # デフォルト実装（既定合成。非破壊拡張）
    /// [`Self::gemm_batched`] と同一の正規化・バッチループ構成で、各
    /// バッチの計算のみ [`Self::gemm_fp32_strict`] に差し替える。
    /// TF32 の概念を持たない CPU・Metal では [`Self::gemm_batched`] と
    /// 同一結果になる。
    fn gemm_batched_fp32_strict(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        default_gemm_batched(self, a, b, BatchedGemmKind::Fp32Strict)
    }

    /// [`Self::gemm_fp32_strict`] と同じ行列積 `C = A @ B` を計算するが、
    /// 結果をホストへ戻さず**呼び出し元が渡す既存の [`DeviceBuffer<f32>`]
    /// の指定オフセットへ直接書き込む**（イシュー #1212・`docs/
    /// device-resident-update-design.md` 追補）。
    ///
    /// `fandhe_ai_autodiff::optim::device_store::DeviceParamStore` が
    /// `Op::LinearResident` の d_weight（`crate::grad::vjp` が
    /// `gemm_fp32_strict` で計算し、GPU バックエンドでは戻り値の
    /// `Tensor<f32>` 構築自体が D2H を伴っていた）を、自身が保持する
    /// grad staging バッファへデバイス常駐のまま直接書き込むための入口。
    /// D2H（本メソッドの戻り値）に続く `DeviceParamStore::step` 側の
    /// H2D（`MemoryOps::upload`）を 1 パラメータぶん丸ごと排除する。
    ///
    /// # 契約
    ///
    /// - `out[out_offset .. out_offset + m*n]` を `A @ B` の結果で**上書き**
    ///   する（累積ではない）。`out_offset + m*n` は呼び出し元・実装側の
    ///   両方で `checked_mul`/`checked_add` により `out.numel()` 以内で
    ///   あることを検査し、範囲外は [`BackendError::InvalidArgument`]
    ///   （REQ-8「シェーダ・カーネル側の手動境界チェックを省略しない」・
    ///   OWASP A03）。
    /// - 数値は同 shape の [`Self::gemm_fp32_strict`] と **bit 同一**
    ///   （同一カーネル選択・CUDA の TF32 opt-in フラグには追従しない）。
    /// - `out.device()` は `self.device()` と一致すること
    ///   （[`BackendError::DeviceMismatch`]）。
    /// - `a`／`b` は 2 次元のみ（[`BackendError::ShapeMismatch`]）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::mse_loss`] と同じ非破壊拡張パターン。既定は
    /// [`BackendError::Unsupported`] を返す fail-safe とし、`grad::vjp`
    /// の `Op::LinearResident` 分岐は `Unsupported` のときのみ既存の
    /// ホスト経路（`gemm_fp32_strict` を呼び戻り値をそのまま勾配として
    /// 使う）へフォールバックする（判定迂回を作らない。`.claude/rules/
    /// security.md` A08）。`backend-cpu::CpuBackendOps`（#1212）と
    /// `backend-metal::MetalBackendOps`（#1555。NT/TN 限定・encode-only）
    /// `backend-cuda::CudaBackendOps`（#1559。NT/TN 限定・GPU 側 smem
    /// 転置カーネル再利用）がオーバーライドする（3 バックエンドすべてが
    /// オーバーライド済み。詳細は `docs/perf/
    /// train-resident-grad-device-update.md`）。
    fn gemm_fp32_strict_into(
        &self,
        _a: &Tensor<f32>,
        _b: &Tensor<f32>,
        _out: &mut DeviceBuffer<f32>,
        _out_offset: usize,
    ) -> Result<(), BackendError> {
        Err(BackendError::Unsupported(
            "gemm_fp32_strict_into: default fail-safe (no in-place device GEMM kernel \
             available for this backend)"
                .into(),
        ))
    }

    /// [`Self::gemm_fp32_strict_into`] と同型だが、[`Self::
    /// sgd_step_device_tracked`] と同じく Metal のコマンドバッファ共有
    /// （イシュー #1017・`docs/backend-metal-command-batching-design.md`）
    /// 向けに共有失敗トークン [`DispatchFailureCell`] を追加引数として
    /// 受け取る非破壊拡張（`sgd_step_device`／`sgd_step_device_tracked`
    /// と同じ「デフォルトメソッド追加」パターン。`BackendOps` の SemVer
    /// 非破壊拡張）。
    ///
    /// # デフォルト実装
    /// 既定は `token` を無視して [`Self::gemm_fp32_strict_into`] へ
    /// そのまま委譲する。CPU は都度同期実行のため実行時エラーが
    /// 呼び出し元へ即座に返り、遅延失敗トークンを必要としない（この
    /// デフォルトのままでよい）。CUDA（`backend-cuda::ops::
    /// CudaBackendOps`。#1559）も同じ理由（`context_cache` の poison／
    /// 世代検査が各呼び出しごとに同期的に完結し、NT/TN 経路自体が
    /// 内部で `stream.synchronize()` する）でこのデフォルトのままで
    /// 良いが、トレイト doc の更新漏れを避けるため機能的に同一の明示
    /// オーバーライドを置いている（`ops.rs::CudaBackendOps::
    /// gemm_fp32_strict_into_tracked` のドキュメンテーションコメント
    /// 参照）。
    ///
    /// Metal のみ `backend-metal::ops::MetalBackendOps` がオーバーライド
    /// し、encode-only（待たない）で直接書き込む NT/TN 経路
    /// （`gemm_fp32_strict_into` doc「NT/TN 経路のみ encode-only にできる
    /// 理由」参照）で `MetalContext::encode` と**同一ロック区間で**
    /// `token` をバッチへ登録する（`sgd_step_device_tracked` doc と同じ
    /// 「encode と登録の間に別スレッドの `synchronize` が割り込む競合を
    /// 防ぐ」設計。codex-review 指摘・PR #1556: この登録がないと、
    /// 共有 `MetalContext` を使う別スレッドが先に `synchronize()` して
    /// GPU エラーを回収した場合、当該バッチは `committed` 列から drain
    /// 済みになり、呼び出し元（`DeviceParamStore`）自身の後続
    /// `download`／`upload_into` がエラーを observe できないまま成功
    /// してしまう——未完成または前回の勾配を正常値として読み出し・更新
    /// に使ってしまう fail-closed 違反を防ぐ）。
    ///
    /// `fandhe_ai_autodiff::optim::device_store::DeviceParamStore::
    /// fill_resident_weight_grad` が呼び出し元となり、自身が保持する
    /// `failure_token` を渡す（`sgd_step_device_tracked` doc「4 つの
    /// 状態機械エントリ」と同様、`step()` 冒頭の `failure_token.is_set()`
    /// 検査が自己 poison する）。
    fn gemm_fp32_strict_into_tracked(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        out: &mut DeviceBuffer<f32>,
        out_offset: usize,
        _token: &DispatchFailureCell,
    ) -> Result<(), BackendError> {
        self.gemm_fp32_strict_into(a, b, out, out_offset)
    }

    /// [`Self::gemm_fp32_strict_into_tracked`] と同じ `C = A @ B` を
    /// 計算するが、加えて `b`（`Op::LinearResident` の VJP では
    /// `d_weight = x_t @ g` の `g` そのもの）の**行方向の和**（bias 勾配。
    /// `autodiff::grad::reduce_to_shape` の rank-2→rank-1 特殊ケースと
    /// 同型の縮約〈shape の対応は同一だが蓄積方式は下記「# 引数」参照〉）
    /// を計算できる場合は `out` の別範囲へ同時に書き込むための非破壊
    /// 拡張（イシュー #1566・`docs/backend-metal-command-batching-
    /// design.md` §10）。
    ///
    /// `docs/perf/train-resident-grad-device-update.md`（#1212）で
    /// `d_weight` を resident staging へ直接書き込む経路が確立した後も、
    /// bias 勾配（`Op::LinearResident.bias`）は依然ホスト側
    /// `reduce_to_shape`（f32 逐次和）で計算し `MemoryOps::upload_into`
    /// で書き戻していた。この `upload_into` が防御的に呼ぶ
    /// `MetalContext::synchronize()` が `command_batching_bench` に残る
    /// 最後の同期点だった（`docs/backend-metal-command-batching-design.md`
    /// §10「案 A′」）。本メソッドは、既にアップロード済みの `g`
    /// （d_weight 計算に使う `b` 引数）を再利用して bias 勾配も同一
    /// ディスパッチ内で encode-only に計算することで、この同期点を
    /// 削減する経路を提供する。
    ///
    /// # 引数
    ///
    /// `bias` が `Some((bias_offset, n))` の場合、`out[bias_offset ..
    /// bias_offset + n]` へ `b` の行方向和（`b: [m, n]` の各列 `j` に
    /// ついて `sum_{i=0}^{m-1} b[i, j]`。走査順は行 `0..m` 昇順）を書き
    /// 込む。蓄積方式は `.claude/rules/coding-rust.md` の勾配長軸縮約
    /// `f64` アキュムレータ方針（2026-09-12 ユーザー承認 A）に従い、
    /// 実装はホスト `f64` アキュムレータ（`acc: f64 = 0.0` から行 `0..m`
    /// を昇順に加算し最後に 1 回 `as f32`）、または `double` 非対応の
    /// Metal では同じ演算列を IEEE 754 binary64 加算の 64bit 整数ソフト
    /// ウェアエミュレーションで再現するカーネル（`fandhe_ai_backend_
    /// metal::shaders::gemm_bias_grad_reduce_f32`。逐語モデルは
    /// `fandhe_ai_backend_metal::soft_f64`）を用いる。いずれも
    /// `fandhe_ai_backend_metal::layout::reduce_bias_grad_rows_host` と
    /// **bit 完全一致**する（NaN のみ payload がハードウェア依存のため
    /// クラス一致。`docs/backend-metal-command-batching-design.md`
    /// §10.14）。
    /// `bias_offset + n` は [`Self::gemm_fp32_strict_into`]
    /// の `out_offset + m*n` と同じ検査規約（`checked_add`・範囲外は
    /// [`BackendError::InvalidArgument`]。REQ-8・OWASP A03）を適用する。
    ///
    /// # 戻り値
    ///
    /// `Ok(bias_filled)`: weight（`out[out_offset..]`）は常に書き込まれる
    /// （成功時）。`bias_filled` は `bias` が `Some` のときに実際に
    /// `out[bias_offset..]` へも書き込めたかを示す——`bias` が `None` の
    /// ときは常に `false`。バックエンドが weight のみ対応し bias 縮約を
    /// 実装しない場合も `bias` を `Some` のまま `Ok(false)` を返してよい
    /// （呼び出し元はホスト `reduce_to_shape` へフォールバックする。
    /// weight/bias の対応可否は独立という契約——`autodiff::tape::
    /// ResidentResolver::fill_resident_weight_grad` doc 参照）。
    /// `Err`（`Unsupported` 含む）の場合は weight・bias いずれも `out` へ
    /// 書き込まれていないことを呼び出し元は仮定してよい（`gemm_fp32_
    /// strict_into` と同じ全体成功/失敗契約）。
    ///
    /// # デフォルト実装
    ///
    /// `bias` を無視して [`Self::gemm_fp32_strict_into_tracked`]
    /// （weight のみ）へ委譲し `Ok(false)` を返す（`CpuBackendOps`
    /// （#1212）・`CudaBackendOps`（既定 `Unsupported` のまま）は本メソッド
    /// を一切オーバーライドしない＝挙動変更ゼロ）。`backend-metal::ops::
    /// MetalBackendOps`（#1566）のみオーバーライドし、NT/TN
    /// （encode-only）経路では同一 `ctx.encode` 呼び出し内で bias 縮約
    /// も追加ディスパッチし、それ以外（NN/TT・分類不能形状）は
    /// ホスト経路フォールバック（`gemm_fp32_strict` → `upload_into` に
    /// 続けて bias もホスト計算 → `upload_into`）で `bias` の有無に
    /// 関わらず常に成功する（`gemm_fp32_strict_into_impl` の
    /// `resident_grad_capability` 汚染防止契約を維持する）。
    #[allow(clippy::too_many_arguments)]
    fn gemm_fp32_strict_into_with_bias_reduce_tracked(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        out: &mut DeviceBuffer<f32>,
        out_offset: usize,
        bias: Option<(usize, usize)>,
        token: &DispatchFailureCell,
    ) -> Result<bool, BackendError> {
        let _ = bias;
        self.gemm_fp32_strict_into_tracked(a, b, out, out_offset, token)?;
        Ok(false)
    }

    // elementwise（`docs/public-api-design.md` §4.2 と同じ 5 演算）
    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;
    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;
    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;
    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;
    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;

    /// `op(a)`（`op` は [`ScalarUnaryOp`]）をホスト常駐 `Tensor<f32>`
    /// 入出力で計算する（イシュー #1634・親 #1592）。
    ///
    /// 上記 5 演算固定の elementwise 面を、演算を 1 つ足すごとに trait
    /// メソッド追加が要らない形へ拡張する機構（`crate::scalar_op` モジュ
    /// ール doc「動機」参照）。`a` は broadcast 不要（単項）。数値契約
    /// （`NaN`／`inf` の伝播・`Relu` variant が既存 `Self::relu` と
    /// 異なる点）は [`ScalarUnaryOp::apply`] のドキュメントを正とする。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::linear_forward_device`] 等と同じ非破壊拡張パターン
    /// （`BackendOps` トレイトへのデフォルトメソッド追加。公開 API
    /// 非破壊はガードレール条件・`.claude/rules/security.md`）であり、
    /// 既定は [`BackendError::Unsupported`] を返す fail-safe とする。
    /// `backend-cpu`（`CpuBackendOps`）はこのデフォルトを汎用 `rayon`
    /// ループ（`backend-cpu::scalar_elementwise`）でオーバーライド
    /// する。`backend-cuda`／`backend-metal` は本イシュー時点では未
    /// 実装で既定のまま（親 #1592 の分担: #1635／#1636 が担当）。
    /// 呼び出し元（`fandhe_ai_autodiff::grad::scalar_unary_with_
    /// fallback`）は `Unsupported` を検出した場合ホスト参照実装
    /// （`ScalarUnaryOp::apply` の逐次適用）へフォールバックする契約。
    fn scalar_unary(
        &self,
        op: ScalarUnaryOp,
        a: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        let _ = (op, a);
        Err(BackendError::Unsupported(
            "scalar_unary: default fail-safe (no ScalarUnaryOp kernel available for this \
             backend; #1592 分担: CUDA/Metal は #1635/#1636)"
                .into(),
        ))
    }

    /// `op(a, b)`（`op` は [`ScalarBinaryOp`]）をホスト常駐 `Tensor<f32>`
    /// 入出力で計算する（イシュー #1634）。[`Self::scalar_unary`] の
    /// 2 項版で契約・デフォルト実装の設計方針は同一。`a`／`b` は
    /// `BackendOps::add`／`mul` と同じ NumPy 互換ブロードキャスト
    /// （broadcast 後の共通 shape で計算する）。
    fn scalar_binary(
        &self,
        op: ScalarBinaryOp,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        let _ = (op, a, b);
        Err(BackendError::Unsupported(
            "scalar_binary: default fail-safe (no ScalarBinaryOp kernel available for this \
             backend; #1592 分担: CUDA/Metal は #1635/#1636)"
                .into(),
        ))
    }

    // reduction（`docs/public-api-design.md` §4.2 と同じ 2 演算）
    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError>;
    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError>;

    /// `dim` 軸に沿った縮約最小値（`torch.min(dim)` 相当。イシュー
    /// #1720）。`dim: None` は全軸縮約（スカラー）。[`Self::max`] と
    /// 対称の意味論だが、`max` が必須メソッドなのに対し本メソッドは
    /// **デフォルトメソッド**として非破壊拡張する（`fandhe-ai-tensor-core`
    /// は crates.io 公開済みのため、既存の全 `BackendOps` 実装者
    /// 〈テスト用モック等〉を壊さない `sort`／`topk` と同じ拡張方針。
    /// 詳細は下記「デフォルト実装」節）。
    ///
    /// **数値契約**:
    /// - **NaN 非伝播**: `f32::min`（`fminf` と同じ）を用いる。
    ///   `+0.0`／`-0.0` は `partial_cmp` が `Equal` とみなすため実装
    ///   依存の一方が採用される。
    /// - **空縮約**: 縮約対象の要素数が 0 の場合は単位元を持たないため
    ///   エラーとする（`Self::max` の CPU 参照実装
    ///   〈`backend-cpu::reduction::max`〉と同じ `EmptyReduction`
    ///   方針。実装側は [`crate::ops_shape::reduce_out_shape`] で
    ///   `dim` を再検査したうえで空縮約を検出すること）。
    fn min(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "min: default fail-safe (no min reduction kernel available for this backend; \
             fall back to fandhe_ai_autodiff::eval::min)"
                .into(),
        ))
    }

    /// `dim` 軸に沿った最大値の添字（`torch.argmax(dim)` 相当。イシュー
    /// #1720）。戻り値は `Tensor<i32>`（[`Self::sort`] の `index` と同型）
    /// で shape は [`crate::ops_shape::reduce_out_shape`]（`min`／`max`
    /// と同じ縮約 shape）。**非微分演算**（テープにノードを追加しない。
    /// `Var::argsort` と同じ扱い）。
    ///
    /// **走査契約**（実装側が必ず守ること）:
    /// 1. **タイ**: 同値（`==`）の場合は `dim` 軸上の**最初の**添字を
    ///    返す（`v > best` のときのみ更新する昇順走査）。
    /// 2. **NaN**: [`Self::max`] の NaN 非伝播規約と整合させる——
    ///    `best` が NaN なら次の非 NaN 値で置換し、`v` が NaN なら
    ///    無視する（全要素 NaN の場合は添字 0）。
    /// 3. **決定性**: 逐次走査（縮約軸内は逐次・出力要素側のみ並列可）
    ///    で run-to-run bit 同一の添字を返す。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::sort`]／[`Self::topk`] と同じ非破壊拡張・fail-safe。
    /// 既定は [`BackendError::Unsupported`] を返し、`Var::argmax` は
    /// `Unsupported` のときのみホスト参照実装（`fandhe_ai_autodiff::
    /// eval::argmax`）へフォールバックする（それ以外のエラーは伝播
    /// する。判定迂回経路を作らない。`.claude/rules/security.md` A08）。
    /// 実装側でも `dim` を [`crate::ops_shape::reduce_out_shape`] で
    /// 再検査し、範囲外は [`BackendError::ShapeMismatch`] を返すこと
    /// （fail-closed）。空縮約は [`Self::min`] と同じくエラーとする。
    fn argmax(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<i32>, BackendError> {
        Err(BackendError::Unsupported(
            "argmax: default fail-safe (no argmax kernel available for this backend; fall \
             back to fandhe_ai_autodiff::eval::argmax)"
                .into(),
        ))
    }

    /// `dim` 軸に沿った最小値の添字（`torch.argmin(dim)` 相当。イシュー
    /// #1720）。[`Self::argmax`] の最小値版で、走査契約・空縮約の扱い
    /// （エラー）・デフォルト実装方針（既定 `Unsupported`・`Var::argmin`
    /// が `fandhe_ai_autodiff::eval::argmin` へフォールバック）は
    /// [`Self::argmax`] と対称（NaN 規約は [`Self::min`] と整合させ、
    /// `v < best` のときのみ更新する）。
    fn argmin(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<i32>, BackendError> {
        Err(BackendError::Unsupported(
            "argmin: default fail-safe (no argmin kernel available for this backend; fall \
             back to fandhe_ai_autodiff::eval::argmin)"
                .into(),
        ))
    }

    /// 平均二乗誤差 `reduction(Σ(pred−target)²)` の forward を 1 個の
    /// 融合カーネルで計算する（イシュー #1045・親イシュー #1043）。
    ///
    /// `pred`／`target` は同一 shape（呼び出し元が [`crate::ops_shape::
    /// require_same_shape`] で検証済み）。戻り値は shape `[]`（スカラー）。
    /// `numel == 0` は `Mean`／`Sum` とも `0.0`（`fandhe_ai_autodiff::eval::
    /// mse_loss` の既存契約と同じ。mean 側はゼロ除算回避、sum 側は空和が
    /// 数学的に 0 のため元々の定義と一致）。
    ///
    /// # デフォルト実装
    ///
    /// 本メソッドは `gemm_bias_act`・`sgd_step_device` と同じ非破壊拡張
    /// （デフォルトメソッド追加。公開 API 非破壊はガードレール条件・
    /// `.claude/rules/security.md`）であり、既定は
    /// [`BackendError::Unsupported`] を返す fail-safe とする。
    /// `fandhe_ai_autodiff::var::Var::mse_loss_with` は `Unsupported` の
    /// ときのみ従来のホスト参照実装（`eval::mse_loss`）へフォールバック
    /// し、それ以外のエラーは伝播する（判定迂回経路を作らない。
    /// `.claude/rules/security.md` A08）。CPU／CUDA／Metal の各実装は
    /// このデフォルトをカーネル内融合実装でオーバーライドする
    /// （`backend-cpu::mse`・`backend-cuda::mse`・`backend-metal::mse`
    /// 参照）。
    fn mse_loss(
        &self,
        _pred: &Tensor<f32>,
        _target: &Tensor<f32>,
        _reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "mse_loss: default fail-safe (no fused MSE forward kernel available)".into(),
        ))
    }

    /// 平均二乗誤差の backward（`dPred = scale·(pred−target)`）を 1 個の
    /// 融合カーネルで計算する（イシュー #1045）。
    ///
    /// `scale` は呼び出し元（`fandhe_ai_autodiff::grad::vjp` の
    /// `Op::MseLoss` 分岐）が上流勾配 `g`（スカラー）と `reduction` から
    /// 事前計算して渡す（`Mean` は `g·2/n`、`Sum` は `g·2`）。カーネル側は
    /// 縮約種別を意識せずこの `scale` を適用するだけでよい。
    ///
    /// `dTarget = −dPred` は常に成り立つ（`d/dtarget (pred−target)² =
    /// −2(pred−target)`）ため、本メソッドは `dPred` の 1 テンソルのみを
    /// 返す契約とする（`dTarget` を別テンソルとしてカーネルに計算・
    /// 転送させるのは無駄な allocation・D2H を増やすだけであり、融合
    /// カーネルで転送量を削減するという本イシューの目的と矛盾する）。
    /// 呼び出し元がホスト側で `dPred` を符号反転するだけで `dTarget` を
    /// 得る（`grad.rs` 参照）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::mse_loss`] と同じ非破壊拡張。既定は
    /// [`BackendError::Unsupported`] を返す fail-safe とし、`Var::
    /// mse_loss_with` の呼び出し元（`grad::vjp`）は `Unsupported` の
    /// ときのみ既存のホスト参照実装（`mse_loss_vjp`）へフォールバックする。
    fn mse_loss_backward(
        &self,
        _pred: &Tensor<f32>,
        _target: &Tensor<f32>,
        _scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "mse_loss_backward: default fail-safe (no fused MSE backward kernel available)".into(),
        ))
    }

    /// Huber／SmoothL1 損失（`d = pred − target` として `|d| < delta` で
    /// `0.5·d²`〈`SmoothL1` は `/delta` でスケール〉・それ以外で
    /// `delta·(|d| − 0.5·delta)`〈`SmoothL1` は `|d| − 0.5·delta`〉）の
    /// forward を 1 個の融合カーネルで計算する（イシュー #1739）。
    ///
    /// `pred`／`target` は同一 shape（呼び出し元が [`crate::ops_shape::
    /// require_same_shape`] で検証済み）。`delta` は呼び出し元
    /// （`fandhe_ai_autodiff::var::Var::huber_loss`／`smooth_l1_loss`）が
    /// 有限かつ `> 0` を検証済み。戻り値は shape `[]`（スカラー）。
    /// `numel == 0` は `Mean`／`Sum` とも `0.0`（[`Self::mse_loss`] と
    /// 同じ契約）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::mse_loss`] と同じ非破壊拡張（デフォルトメソッド追加）
    /// であり、既定は [`BackendError::Unsupported`] を返す fail-safe と
    /// する。`Var::huber_loss_impl` は `Unsupported` のときのみ従来の
    /// ホスト参照実装（`eval::huber_loss`）へフォールバックし、それ以外
    /// のエラーは伝播する（判定迂回経路を作らない。
    /// `.claude/rules/security.md` A08）。CPU／CUDA／Metal の各実装は
    /// このデフォルトをカーネル内融合実装でオーバーライドする
    /// （`backend-cpu::huber`・`backend-cuda::huber`・`backend-metal::huber`
    /// 参照）。
    fn huber_loss(
        &self,
        _pred: &Tensor<f32>,
        _target: &Tensor<f32>,
        _kind: HuberKind,
        _delta: f32,
        _reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "huber_loss: default fail-safe (no fused Huber/SmoothL1 forward kernel available)"
                .into(),
        ))
    }

    /// Huber／SmoothL1 損失の backward（`dPred = scale·grad_elem(d)`。
    /// `grad_elem(d)` は `|d| < delta` で `d`〈`SmoothL1` は `/delta`〉・
    /// それ以外で `copysign(delta, d)`〈`SmoothL1` は `copysign(1, d)`〉）
    /// を 1 個の融合カーネルで計算する（イシュー #1739）。
    ///
    /// `scale` は呼び出し元（`fandhe_ai_autodiff::grad::vjp` の
    /// `Op::HuberLoss` 分岐）が上流勾配 `g`（スカラー）と `reduction` から
    /// 事前計算して渡す（`Mean` は `g/n`、`Sum` は `g`。[`Self::
    /// mse_loss_backward`] と異なり係数 2 は付かない）。
    ///
    /// `dTarget = −dPred` は常に成り立つ（損失は `d = pred − target`
    /// のみに依存する）ため、[`Self::mse_loss_backward`] と同じ理由で
    /// 本メソッドは `dPred` の 1 テンソルのみを返す契約とする
    /// （呼び出し元がホスト側で符号反転して `dTarget` を得る）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::huber_loss`] と同じ非破壊拡張。既定は
    /// [`BackendError::Unsupported`] を返す fail-safe とし、`Var::
    /// huber_loss_impl` の呼び出し元（`grad::vjp`）は `Unsupported` の
    /// ときのみ既存のホスト参照実装（`huber_loss_vjp`）へフォールバック
    /// する。
    fn huber_loss_backward(
        &self,
        _pred: &Tensor<f32>,
        _target: &Tensor<f32>,
        _kind: HuberKind,
        _delta: f32,
        _scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "huber_loss_backward: default fail-safe (no fused Huber/SmoothL1 backward kernel \
             available)"
                .into(),
        ))
    }

    /// 二値交差エントロピー損失（`BCELoss`／`BCEWithLogitsLoss`。
    /// [`BceKind`] で分岐）の forward を 1 個の融合カーネルで計算する
    /// （イシュー #1737・親イシュー #1609）。
    ///
    /// `input`／`target` は同一 shape（呼び出し元が
    /// [`crate::ops_shape::require_same_shape`] で検証済み）。戻り値は
    /// shape `[]`（スカラー）。`numel == 0` は `Mean`／`Sum` とも `0.0`
    /// （[`Self::mse_loss`] と同じ空縮約契約）。
    ///
    /// `input ∈ [0, 1]`（[`BceKind::Probabilities`] のときのみ）の範囲
    /// 検査は呼び出し元（`fandhe_ai_autodiff::var::Var::bce_loss`）が
    /// 本メソッド呼び出し前にホスト側で行う契約であり、本メソッド自身は
    /// 範囲検査を行わない（カーネル内での分岐・エラー伝播コストを避ける
    /// ため。`.claude/rules/security.md` A03 の観点はホスト側検査で
    /// 満たす）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::mse_loss`] と同じ非破壊拡張。既定は
    /// [`BackendError::Unsupported`] を返す fail-safe とし、
    /// `Var::bce_loss`／`bce_with_logits_loss` は `Unsupported` の
    /// ときのみホスト参照実装（`eval::bce_loss`）へフォールバックする
    /// （それ以外のエラーは伝播する。判定迂回経路を作らない）。
    fn bce_loss(
        &self,
        _input: &Tensor<f32>,
        _target: &Tensor<f32>,
        _kind: BceKind,
        _reduction: MseReduction,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "bce_loss: default fail-safe (no fused BCE forward kernel available)".into(),
        ))
    }

    /// 二値交差エントロピー損失の backward（`dInput`）を 1 個の融合
    /// カーネルで計算する（イシュー #1737）。
    ///
    /// [`Self::mse_loss_backward`] と異なり `dTarget = -dInput` という
    /// 単純な符号反転関係が成り立たない（`dInput`／`dTarget` の式が
    /// 非対称。`docs/compat-api-scope.md` §1.2 参照）ため、本メソッドは
    /// `dInput` のみを返し、`dTarget` は呼び出し元（`grad::vjp`）が
    /// ホスト側で要素ごとに計算する（新規 GPU カーネル起動・D2H を
    /// 増やさないための設計判断。`mse_loss_backward` の dTarget 省略と
    /// 同じ動機）。
    ///
    /// `scale` は呼び出し元が上流勾配 `g`（スカラー）と `reduction` から
    /// 事前計算して渡す（`Mean` は `g/n`、`Sum` は `g`。二乗誤差の
    /// `2` 倍係数を持つ `mse_loss_backward` の `scale` とは異なる）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::bce_loss`] と同じ非破壊拡張。既定は
    /// [`BackendError::Unsupported`] を返す fail-safe とする。
    fn bce_loss_backward(
        &self,
        _input: &Tensor<f32>,
        _target: &Tensor<f32>,
        _kind: BceKind,
        _scale: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "bce_loss_backward: default fail-safe (no fused BCE backward kernel available)".into(),
        ))
    }

    /// 行方向 softmax（`exp(x - max(x)) / sum(exp(x - max(x)))`）の
    /// 独立エントリ（イシュー #1594）。既存の [`Self::run_fused`] 経由
    /// （`match_softmax_plan` の canonical プラン一致限定）とは別に、
    /// `fandhe_ai_autodiff::var::Var::softmax` が直接呼べる入口を提供する。
    ///
    /// **契約**: `dim` が `x` の最終軸のときのみ計算を試みてよい
    /// （[`crate::ops_shape::row_softmax_layout`] が非最終軸を `Ok(None)`
    /// として区別する契約に対応。行カーネルは最終軸専用のため、
    /// 非最終軸は本メソッドをオーバーライドしない実装でも
    /// [`BackendError::Unsupported`] を返す既定のままでよい）。戻り値の
    /// shape は入力 `x` と恒等（softmax は shape 不変）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::mse_loss`] と同じ非破壊拡張。既定は
    /// [`BackendError::Unsupported`] を返す fail-safe とし、`Var::
    /// softmax` は `Unsupported` のときのみホスト参照実装
    /// （`eval::softmax_along`）へフォールバックする（それ以外のエラーは
    /// 伝播する。判定迂回経路を作らない。`.claude/rules/security.md`
    /// A08）。CPU／CUDA／Metal はいずれもこのデフォルトを既存の融合
    /// softmax カーネル（`run_fused` の softmax 一致経路が使うものと同一
    /// のカーネル実体）でオーバーライドする。
    fn softmax(&self, _x: &Tensor<f32>, _dim: usize) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "softmax: default fail-safe (no fused softmax kernel available)".into(),
        ))
    }

    /// `dim` 軸に沿った累積和（`torch.cumsum` 相当。イシュー #1731）。
    /// `dim` 以外の軸の組（lane）ごとに `dim` の添字昇順で
    /// `out[i] = Σ_{j<=i} x[j]` を逐次計算する。出力 shape は入力と
    /// 恒等（累積演算は shape を変えない）。`dim` は呼び出し元
    /// （`fandhe_ai_autodiff::var::Var::cumsum`）が
    /// [`crate::reduce_out_shape`] で範囲検査済み。
    ///
    /// **数値契約**: lane 全体で `f64` アキュムレータを保持し、各
    /// ステップで `acc = acc + (x[i] as f64)` を計算したうえで、
    /// 出力要素 `out[i]` はその時点の `acc` を `f32` へ downcast した
    /// スナップショットとする（次ステップは downcast 後の `f32` を
    /// 読み戻さない。PyTorch CPU の `acc_type<float> = double` と同じ
    /// 読み方。`.claude/rules/coding-rust.md` の f64 アキュムレータ
    /// 方針を forward の scan へ拡張したもの）。この契約により
    /// `cumsum([1e8, 1.0, -1e8], 0) == [1e8, 1e8, 1.0]` となる（f32
    /// 逐次アキュムレータなら末尾が `0.0` になり丸め誤差が蓄積する）。
    /// NaN／inf は IEEE 754 のまま伝播する。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::softmax`] と同じ非破壊拡張・fail-safe。既定は
    /// [`BackendError::Unsupported`] を返し、`Var::cumsum` は
    /// `Unsupported` のときのみホスト参照実装
    /// （`fandhe_ai_autodiff::eval::cumsum_along`）へフォールバックする
    /// （それ以外のエラーは伝播する。判定迂回経路を作らない。
    /// `.claude/rules/security.md` A08）。**後続 GPU 実装（CUDA／
    /// Metal）への受け入れ契約**: 本 CPU 参照実装と bit 完全一致
    /// （REQ-2 の tolerance は使わない）。CUDA は native `double` の
    /// lane 逐次カーネル、Metal は `double` 非対応のため
    /// `crates/backend-metal/src/soft_f64.rs` と同型の binary64 逐次
    /// エミュレーションで達成できる。VJP はホスト側（`grad.rs`）に
    /// 留めるため、GPU 実装は forward のみでよい。
    fn cumsum(&self, _x: &Tensor<f32>, _dim: usize) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "cumsum: default fail-safe (no fused cumsum kernel available)".into(),
        ))
    }

    /// `dim` 軸に沿った累積積（`torch.cumprod` 相当。イシュー #1731）。
    /// [`Self::cumsum`] と同じ lane 構造・非破壊拡張・フォールバック
    /// 規律に従うが、アキュムレータは `1.0` から開始し各ステップで
    /// `acc = acc * (x[i] as f64)` を計算する。`f64` アキュムレータに
    /// より `f32` では即座に underflow する極小値の積も、途中の
    /// アキュムレータが有限のまま追跡され続ける（例:
    /// `cumprod([1e-30, 1e-30, 1e30], 0)` の `out[2]` は f32 逐次なら
    /// `0.0` になるが、f64 アキュムレータでは非零有限値になる）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::cumsum`] と同じ非破壊拡張・fail-safe・GPU 受け入れ契約。
    /// 既定は [`BackendError::Unsupported`] を返し、`Var::cumprod` は
    /// `Unsupported` のときのみホスト参照実装
    /// （`fandhe_ai_autodiff::eval::cumprod_along`）へフォールバック
    /// する。
    fn cumprod(&self, _x: &Tensor<f32>, _dim: usize) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "cumprod: default fail-safe (no fused cumprod kernel available)".into(),
        ))
    }

    /// 行方向 log_softmax（`x − m − ln(Σ exp(x − m))`。`m` は行 max）の
    /// 独立エントリ（イシュー #1594）。[`Self::softmax`] と同じ最終軸
    /// 限定契約・非破壊拡張・フォールバック規律に従う
    /// （`Var::log_softmax` は `Unsupported` のときのみ `eval::
    /// log_softmax_along` へフォールバックする）。
    ///
    /// **`ln(softmax(x))` にしない理由**: softmax の出力がアンダー
    /// フローで `0.0` になった要素で `ln(0.0) = -inf` を経由し数値精度を
    /// 落とすため、`x − m − ln(Σexp(x−m))` の解析形で計算する（PyTorch
    /// `F.log_softmax` と同じ安定化方針）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::softmax`] と同じ非破壊拡張・fail-safe。本イシュー時点で
    /// GPU 側（CUDA／Metal）に log_softmax 専用カーネルは存在しないため
    /// 両バックエンドともこの既定のまま（ホストフォールバックに委ねる）
    /// で、CPU のみ融合カーネル（`backend-cpu::softmax::
    /// run_log_softmax_f32`）でオーバーライドする。
    fn log_softmax(&self, _x: &Tensor<f32>, _dim: usize) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "log_softmax: default fail-safe (no fused log_softmax kernel available)".into(),
        ))
    }

    /// `inputs` を `dim` 軸で連結する（`torch.cat` 相当。イシュー
    /// #1598）。入力は strided view（`contiguous()` を経ずに渡されうる）
    /// でよく、出力は必ず contiguous・**bit 完全一致のコピー**（丸め
    /// なし。REQ-2 の複合判定より強い bit 同一を parity テストで要求
    /// する）。`dim` 以外の軸の shape 一致は呼び出し元
    /// （[`crate::ops_shape::concat_out_shape`]）で検査済みだが、実装側
    /// でも `inputs` の shape を再検査し、不一致は
    /// [`BackendError::ShapeMismatch`] を返すこと（fail-closed。
    /// 判定迂回経路を作らない。`.claude/rules/security.md` A08）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::softmax`] と同じ非破壊拡張・fail-safe。既定は
    /// [`BackendError::Unsupported`] を返し、`Var::cat`（`grad.rs::
    /// concat_with_fallback` 経由）は `Unsupported` のときのみホスト
    /// 参照実装（`eval::concat`）へフォールバックする（それ以外の
    /// エラーは伝播する）。
    fn concat(&self, _inputs: &[&Tensor<f32>], _dim: usize) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "concat: default fail-safe (no fused concat kernel available)".into(),
        ))
    }

    /// 条件テンソルによる要素選択（`torch.where` 相当。イシュー
    /// #1637）。`cond` は `a`／`b` と同じ shape へ broadcast 済みの
    /// **f32 マスク**として渡される（`Tensor<bool>` はデバイス転送
    /// 契約〈`MemoryOps` は f32 専用〉の対象外のため、呼び出し元
    /// `fandhe_ai_autodiff::var::Var::where_cond` が bool→f32 変換を
    /// 1 回だけ行い `out_shape` ちょうどの contiguous テンソルへ
    /// 実体化する）。真偽の判定契約は**3 バックエンド共通で
    /// `c != 0.0`**（CPU `Rust c != 0.0`・CUDA `c != 0.0f`・Metal MSL
    /// `c != 0.0f`。NaN マスクは呼び出し元で発生し得ないため考慮不要）。
    /// `cond`／`a`／`b` は全て同一 shape（`out_shape`）であること。
    /// 出力 shape は `a`（＝`b`＝`cond`）と恒等。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::concat`] と同じ非破壊拡張・fail-safe。既定は
    /// [`BackendError::Unsupported`] を返し、`Var::where_cond` は
    /// `Unsupported` のときのみホスト参照実装（`eval::where_cond`）へ
    /// フォールバックする（それ以外のエラーは伝播する。判定迂回経路を
    /// 作らない。`.claude/rules/security.md` A08）。実装側でも
    /// `cond`／`a`／`b` の shape を再検査し、不一致は
    /// [`BackendError::ShapeMismatch`] を返すこと（fail-closed）。
    fn where_cond(
        &self,
        _cond: &Tensor<f32>,
        _a: &Tensor<f32>,
        _b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "where_cond: default fail-safe (no fused where kernel available)".into(),
        ))
    }

    /// マスク位置を定数 `value` で置換する（`torch.masked_fill` 相当。
    /// イシュー #1637）。`mask` は `x` と同じ shape へ broadcast 済みの
    /// f32 マスク（[`Self::where_cond`] と同じ `c != 0.0` 判定契約・
    /// bool→f32 変換の位置づけ）。`mask` の要素が真（`!= 0.0`）の位置を
    /// `value` に置換し、それ以外は `x` の値をそのまま返す。出力 shape
    /// は `x` と恒等。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::where_cond`] と同じ非破壊拡張・fail-safe。既定は
    /// [`BackendError::Unsupported`] を返し、`Var::masked_fill` は
    /// `Unsupported` のときのみホスト参照実装（`eval::masked_fill`）へ
    /// フォールバックする。実装側でも `x`／`mask` の shape 一致を
    /// 再検査し、不一致は [`BackendError::ShapeMismatch`] を返すこと
    /// （fail-closed）。
    fn masked_fill(
        &self,
        _x: &Tensor<f32>,
        _mask: &Tensor<f32>,
        _value: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "masked_fill: default fail-safe (no fused masked_fill kernel available)".into(),
        ))
    }

    /// `dim` 軸に沿って `index` が指す要素を独立に読み出す
    /// （`torch.gather` 相当。イシュー #1776）。各出力位置 `p`
    /// （`p` は `index.shape()` 上の多次元添字）は
    /// `out[p] = input[p[0], .., index[p], .., p[rank-1]]`
    /// （`index[p]` を `dim` 軸の添字に差し替えた位置から読む）で
    /// 決まり、出力位置同士の書き込み衝突がないため決定的集約順序の
    /// 契約は不要（[`ScatterReduce`] doc 参照）。`index` の値は
    /// `[0, input.shape()[dim])` の範囲内であることを呼び出し元
    /// （`fandhe_ai_autodiff::var::Var::gather`）が forward 時点で
    /// 検査済み（本メソッドは値検査を行わない前提）。
    ///
    /// 出力 shape は `index.shape()` と恒等（`dim` 軸以外は
    /// `input.shape()` と一致する必要がある。
    /// [`crate::ops_shape::gather_out_shape`] 参照）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::masked_fill`] と同じ非破壊拡張・fail-safe。既定は
    /// [`BackendError::Unsupported`] を返し、`Var::gather` は
    /// `Unsupported` のときのみホスト参照実装
    /// （`fandhe_ai_autodiff::eval::gather`）へフォールバックする
    /// （それ以外のエラーは伝播する。判定迂回経路を作らない。
    /// `.claude/rules/security.md` A08）。実装側でも `input`／`index`
    /// の shape を [`crate::ops_shape::gather_out_shape`] で再検査し、
    /// 不一致は [`BackendError::ShapeMismatch`] を返すこと
    /// （fail-closed）。
    fn gather(
        &self,
        _input: &Tensor<f32>,
        _dim: usize,
        _index: &Tensor<i32>,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "gather: default fail-safe (no fused gather kernel available)".into(),
        ))
    }

    /// 各軸を `(before, after)` だけ定数値 `value` で拡張する
    /// （`torch.nn.functional.pad(mode='constant')` 相当。イシュー
    /// #1756）。出力の各要素は「`input` 内部位置ならそのままコピー・
    /// パディング領域なら `value`」の 2 分岐のみで決まる純粋なコピー
    /// 演算（算術を含まない）であり、`f64` アキュムレータ契約は非該当。
    /// バックエンド間数値一致は REQ-2 複合判定ではなく **bit 完全
    /// 一致**（`value` が NaN の場合のみクラス一致。`.claude/rules/
    /// coding-rust.md` 数値契約節参照）。
    ///
    /// `pads` は先頭次元から順に対応する（PyTorch `F.pad` の「末尾次元
    /// から逆順の平坦リスト」とは異なる意図的な設計。
    /// `docs/compat-feature-gap.md` 追補参照）。出力 shape は
    /// [`crate::ops_shape::pad_out_shape`] が定める。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::gather`] と同じ非破壊拡張・fail-safe。既定は
    /// [`BackendError::Unsupported`] を返し、`Var::pad` は
    /// `Unsupported` のときのみホスト参照実装
    /// （`fandhe_ai_autodiff::eval::pad`）へフォールバックする（それ
    /// 以外のエラーは伝播する。判定迂回経路を作らない。
    /// `.claude/rules/security.md` A08）。実装側でも `input.shape()`
    /// と `pads` を [`crate::ops_shape::pad_out_shape`] で再検査し、
    /// 不一致は [`BackendError::ShapeMismatch`] を返すこと
    /// （fail-closed）。
    fn pad(
        &self,
        _input: &Tensor<f32>,
        _pads: &[(usize, usize)],
        _value: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "pad: default fail-safe (no fused pad kernel available)".into(),
        ))
    }

    /// `dim` 軸に沿って `index` が指す位置へ `src` の値を書き込む
    /// （`torch.scatter`／`torch.scatter_add` 相当。`reduce` で選択。
    /// イシュー #1776）。出力 shape は `input.shape()` と恒等
    /// （scatter は shape を変えない）。`index`／`src` は同一 shape
    /// であることを要求する（本実装の簡略化。PyTorch の
    /// `index.size(d) <= src.size(d)` という緩い制約は対象外。
    /// [`crate::ops_shape::scatter_out_shape`] 参照）。
    ///
    /// 同一出力位置に複数回書き込まれる場合の意味論・数値契約は
    /// [`ScatterReduce`] のドキュメントを正とする（決定的集約順序・
    /// `Add` の `f64` アキュムレータ契約）。`index` の値は
    /// `[0, input.shape()[dim])` の範囲内であることを呼び出し元
    /// （`fandhe_ai_autodiff::var::Var::scatter`／`scatter_add`）が
    /// forward 時点で検査済み（本メソッドは値検査を行わない前提）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::gather`] と同じ非破壊拡張・fail-safe。既定は
    /// [`BackendError::Unsupported`] を返し、`Var::scatter`／
    /// `scatter_add` は `Unsupported` のときのみホスト参照実装
    /// （`fandhe_ai_autodiff::eval::scatter`。[`ScatterReduce`] の
    /// 決定的集約契約に厳密に従う）へフォールバックする。実装側でも
    /// `input`／`index`／`src` の shape を
    /// [`crate::ops_shape::scatter_out_shape`] で再検査し、不一致は
    /// [`BackendError::ShapeMismatch`] を返すこと（fail-closed）。
    fn scatter(
        &self,
        _input: &Tensor<f32>,
        _dim: usize,
        _index: &Tensor<i32>,
        _src: &Tensor<f32>,
        _reduce: ScatterReduce,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "scatter: default fail-safe (no fused scatter kernel available)".into(),
        ))
    }

    /// 空間軸（末尾 `size.len()` 軸）を `size` へリサンプリングする
    /// （`torch.nn.functional.interpolate`／`tf.image.resize` 相当。
    /// イシュー #1757）。先頭の残り軸（batch／channel 等）は素通し。
    /// 出力 shape は [`crate::ops_shape::interpolate_out_shape`] が
    /// 検査・確定する（`shape[..rank-k]` に `size` を連結した形）。
    ///
    /// [`InterpolateMode::Nearest`] の添字式・数値契約は同 variant の
    /// doc を正とする——float を使わない整数演算のみのため forward は
    /// **3 バックエンド間で構造的に bit 完全一致**する（`mode` の
    /// 未知 variant〈将来 #1762 の bilinear 追加等〉は実装側が
    /// [`BackendError::Unsupported`] を返す契約とする）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::scatter`] と同じ非破壊拡張・fail-safe。既定は
    /// [`BackendError::Unsupported`] を返し、`Var::interpolate` は
    /// `Unsupported` のときのみホスト参照実装
    /// （`fandhe_ai_autodiff::eval::interpolate_nearest`）へ
    /// フォールバックする（それ以外のエラーは伝播する。判定迂回経路を
    /// 作らない。`.claude/rules/security.md` A08）。実装側でも
    /// `input`／`size` を [`crate::ops_shape::interpolate_out_shape`]
    /// で再検査し、不一致は [`BackendError::ShapeMismatch`] を返す
    /// こと（fail-closed）。
    fn interpolate(
        &self,
        _input: &Tensor<f32>,
        _size: &[usize],
        _mode: InterpolateMode,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "interpolate: default fail-safe (no fused interpolate kernel available)".into(),
        ))
    }

    /// `dim` 軸に沿って `input` を並べ替える（`torch.sort` 相当。
    /// イシュー #1733）。戻り値は `(values, index)` で、`values` は
    /// 並べ替え後の値、`index` は元の（`dim` 軸上の）添字（PyTorch の
    /// `torch.sort` の第 2 戻り値と同じ意味論。`values[.., i, ..] ==
    /// input[.., index[.., i, ..], ..]`）。出力 shape は両方とも
    /// `input.shape()` と恒等（[`crate::ops_shape::sort_out_shape`]
    /// 参照）。
    ///
    /// **順序契約（実装側が必ず守ること。後続 GPU 実装〈イシュー
    /// #1741〉の bit 一致契約の正）**:
    /// 1. **安定性**: 同値（ties）は `descending` の値に関わらず元の
    ///    `dim` 軸上の添字の昇順で並ぶ（`descending` でも「値の降順・
    ///    同値内は添字昇順」——単純な昇順ソート結果の `reverse()` は
    ///    同値の添字順を反転させてしまうため禁止）。
    /// 2. **NaN**: `f32::partial_cmp` が `None` を返す比較は
    ///    「NaN は任意の非 NaN より大きい・NaN 同士は同値（1 の安定性
    ///    契約に従う）」として扱う（PyTorch `torch.sort` と同じ）。
    /// 3. **±0**: `partial_cmp` は `-0.0` と `0.0` を `Equal` とみなす
    ///    ため、1 の安定性契約により同値扱い（添字順）となる。
    /// 4. **決定性**: 単一スレッド逐次実装とし run-to-run で bit
    ///    同一の出力を返す（並列化する場合も本契約の観測結果を
    ///    不変に保つこと。`.claude/rules/out-of-scope-tracking.md`
    ///    対象・並列化自体は本イシューのスコープ外）。
    ///
    /// `values` は算術演算を含まない `input` の要素の並べ替え（`gather`
    /// と数学的に同一）であるため、FMA 契約・`f64` アキュムレータ契約
    /// は非該当。CPU vs GPU の数値一致判定は REQ-2 複合判定ではなく
    /// 「上記 1〜3 の index 契約の完全再現＋`values` の bit 一致」で
    /// 行う。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::scatter`] と同じ非破壊拡張・fail-safe。既定は
    /// [`BackendError::Unsupported`] を返し、`Var::sort`／`argsort` は
    /// `Unsupported` のときのみホスト参照実装（`fandhe_ai_autodiff::
    /// eval::sort`）へフォールバックする（それ以外のエラーは伝播する。
    /// 判定迂回経路を作らない。`.claude/rules/security.md` A08）。
    /// 実装側でも `dim` を [`crate::ops_shape::sort_out_shape`] で
    /// 再検査し、範囲外は [`BackendError::ShapeMismatch`] を返すこと
    /// （fail-closed）。
    fn sort(
        &self,
        _input: &Tensor<f32>,
        _dim: usize,
        _descending: bool,
    ) -> Result<(Tensor<f32>, Tensor<i32>), BackendError> {
        Err(BackendError::Unsupported(
            "sort: default fail-safe (no fused sort kernel available)".into(),
        ))
    }

    /// `dim` 軸に沿って上位（または下位）`k` 個を選ぶ（`torch.topk`
    /// 相当。`sorted=True` 固定。イシュー #1733）。戻り値は
    /// [`Self::sort`] と同じ `(values, index)` の形で、`largest=true`
    /// なら降順 sort の先頭 `k`、`largest=false` なら昇順 sort の
    /// 先頭 `k` に等しい（同値・NaN の扱いは [`Self::sort`] の順序
    /// 契約 1〜4 にそのまま従う）。出力 shape は `dim` 軸のみ `k` に
    /// 置換した shape（[`crate::ops_shape::topk_out_shape`] 参照）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::sort`] と同じ非破壊拡張・fail-safe。既定は
    /// [`BackendError::Unsupported`] を返し、`Var::topk` は
    /// `Unsupported` のときのみホスト参照実装（`fandhe_ai_autodiff::
    /// eval::topk`）へフォールバックする。実装側でも `dim`／`k` を
    /// [`crate::ops_shape::topk_out_shape`] で再検査し、`k` が対象軸
    /// のサイズを超える場合は [`BackendError::ShapeMismatch`] を
    /// 返すこと（fail-closed）。
    fn topk(
        &self,
        _input: &Tensor<f32>,
        _dim: usize,
        _k: usize,
        _largest: bool,
    ) -> Result<(Tensor<f32>, Tensor<i32>), BackendError> {
        Err(BackendError::Unsupported(
            "topk: default fail-safe (no fused topk kernel available)".into(),
        ))
    }

    /// クラス id 列（整数値の `index`）を one-hot 行列へ展開する
    /// （`torch.nn.functional.one_hot`／`tf.one_hot` 相当。**非微分演算**。
    /// イシュー #1755）。出力 shape は `index.shape()` へ末尾軸として
    /// `num_classes` を付加した形（[`crate::ops_shape::one_hot_out_shape`]
    /// 参照）で、`out[.., c] = 1.0 if index[..] == c else 0.0`。
    ///
    /// `index` の値は `[0, num_classes)` の範囲内であることを呼び出し元
    /// （`fandhe_ai_autodiff::var::Var::one_hot`）が forward 時点で検査
    /// 済み（本メソッドは値検査を行わない前提だが、[`Self::gather`] 等と
    /// 同様に実装側でも独立に値域を再検査し縦深防御とすることが望ましい。
    /// `.claude/rules/security.md` A08）。
    ///
    /// **非微分演算の契約**: `one_hot` は整数クラス id から離散的な
    /// 0/1 行列を作るため勾配を持たない。`fandhe_ai_autodiff::Op::OneHot`
    /// の VJP（`crates/autodiff/src/grad.rs::vjp`）は入力へ明示的にゼロ
    /// 勾配を返す（寄与なしではなく「ゼロ勾配が流れる」ことを
    /// `Gradients::get` で観測可能にするため）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::gather`] と同じ非破壊拡張・fail-safe。既定は
    /// [`BackendError::Unsupported`] を返し、`Var::one_hot` は
    /// `Unsupported` のときのみホスト参照実装
    /// （`fandhe_ai_autodiff::eval::one_hot`）へフォールバックする
    /// （それ以外のエラーは伝播する。判定迂回経路を作らない）。実装側
    /// でも戻り shape を [`crate::ops_shape::one_hot_out_shape`] で
    /// 再検査し、不一致は [`BackendError::ShapeMismatch`] を返すこと
    /// （fail-closed）。
    fn one_hot(
        &self,
        _index: &Tensor<i32>,
        _num_classes: usize,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "one_hot: default fail-safe (no fused one_hot kernel available)".into(),
        ))
    }

    /// 入力を平坦化しソートしたうえで重複を除去した一意値集合を返す
    /// （`torch.unique(input, sorted=True)` の values のみ。
    /// `return_inverse`／`return_counts`／`dim` 指定は対象外。イシュー
    /// #1734・`docs/unique-facade-exposure-decision.md`）。
    ///
    /// # 契約
    ///
    /// - **意味論**: 入力を row-major で平坦化 → ソート → 隣接重複除去
    ///   → rank 1・contiguous な `Tensor<f32>`（shape `[m]`、
    ///   `0 <= m <= numel`）を返す。空入力（`numel == 0`）は shape `[0]`。
    /// - **順序キー**: IEEE 754 totalOrder（Rust `f32::total_cmp`）。
    ///   `-NaN < -inf < … < -0.0 < +0.0 < … < +inf < +NaN`。NaN 同士は
    ///   符号・payload の bit 順で一意に順序が定まる。
    /// - **重複判定述語**: `a == b`（IEEE 比較。PyTorch 互換）。
    ///   `-0.0` と `+0.0` は同一とみなされ 1 要素へ集約され、totalOrder
    ///   の先頭側である **`-0.0` が代表として残る**。NaN は
    ///   `NaN != NaN` のため **すべて保持**される（bit パターンが同一でも
    ///   集約しない）。
    /// - **出力不変条件**（呼び出し元が事後検査に使う）: (1) rank 1、
    ///   (2) `len <= numel`、(3) 隣接ペア `(a, b)` すべてについて
    ///   `a.total_cmp(&b) != Ordering::Greater` かつ `!(a == b)`
    ///   （同一 bit の NaN が隣接しうるため「厳密増加」ではなく
    ///   「非減少 ＋ 隣接 `==` なし」）。
    /// - **数値契約**: 選択演算（丸めなし）。3 バックエンドの出力は
    ///   互いに bit 完全一致（NaN payload 含む）。REQ-2 複合判定は
    ///   用いない。
    /// - 入力は strided view（非 contiguous）でもよく、各実装が
    ///   `Tensor::contiguous()`／`host_slice()` で稠密化してから処理
    ///   する契約とする。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::gather`] と同じ非破壊拡張・fail-safe。既定は
    /// [`BackendError::Unsupported`] を返し、`Var::unique`（`autodiff`）
    /// は `Unsupported` のときのみホスト参照実装（`eval::unique`）へ
    /// フォールバックする。出力は非微分（勾配なし）のため `Op` を
    /// tape に記録せず、`Var::unique` は detached な `Tensor<f32>` を
    /// 返す（出力形状が入力値に依存して動的に決まるため、静的 shape
    /// 前提の `Var`／`DeviceBuffer` 常駐チェーンには乗らない設計判断。
    /// `docs/unique-facade-exposure-decision.md` 参照）。
    fn unique(&self, _x: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "unique: default fail-safe (no fused unique kernel available)".into(),
        ))
    }

    /// GEMM の epilogue（bias 加算・activation）を融合した
    /// `act(A @ B + bias)` を計算する（TASK-12.1f・#203）。
    ///
    /// `bias` は `[n]`（`B` の列数）の 1 次元テンソルで、`A @ B: [m, n]` の
    /// 各行へブロードキャスト加算される（`None` の場合は bias 加算を
    /// 省略する）。`act` は bias 加算後に適用する
    /// （[`Activation::None`] なら恒等関数）。
    ///
    /// # デフォルト実装（非融合合成）
    ///
    /// 本メソッドは **デフォルトメソッド**として追加している（`BackendOps`
    /// の非破壊拡張。公開 API 非破壊はガードレール条件・
    /// `.claude/rules/security.md`）。デフォルト実装は `gemm` →
    /// （`bias` があれば）`add`（行方向ブロードキャスト。
    /// `docs/public-api-design.md` §4.2 のブロードキャスト規約に従い
    /// `[n]` を `[1, n]` として `[m, n]` へ揃える）→ `act` に応じた
    /// activation メソッド呼び出しの 3 段合成である。CPU バックエンドは
    /// [`crate`] を利用する `backend-cpu::ops::CpuBackendOps` がこの
    /// デフォルトを **カーネル内融合実装でオーバーライド**し、中間
    /// `Tensor` 2 個の割当・GEMM 結果の再読み出しパスを削減する
    /// （CUTLASS 系実測で epilogue 融合が平均 1.38〜1.45 倍。動機は
    /// イシュー #203）。CUDA はイシュー #599 で
    /// `backend-cuda::ops::CudaBackendOps::gemm_bias_act` が本デフォルトを
    /// **カーネル内融合実装でオーバーライド**した（CPU と同じ「bias が
    /// `None` または `[n]` 厳密一致なら融合、それ以外は非融合合成へ
    /// フォールバック」という分岐条件。`backend-cuda::ops::
    /// gemm_bias_act_route` 参照）。Metal はイシュー #605 で
    /// `backend-metal::ops::MetalBackendOps::gemm_bias_act` が本デフォルトを
    /// **カーネル内融合実装でオーバーライド**した（CPU／CUDA と同じ「bias
    /// が `None` または `[n]` 厳密一致なら融合、それ以外は非融合合成へ
    /// フォールバック」という分岐条件。`backend-metal::ops::
    /// gemm_bias_act_route` 参照）。CPU／CUDA／Metal の 3 バックエンドが
    /// すべて融合カーネルでオーバーライド済みとなった。
    ///
    /// `bias` の shape が `[n]` の場合（CPU バックエンドでは融合カーネルの
    /// 対応範囲）はそのまま計算する。`[n]` でない場合は `add` の NumPy
    /// 互換ブロードキャスト判定へ委譲し、`out: [m, n]` へブロードキャスト
    /// **不能**な場合にのみ [`BackendError::ShapeMismatch`] を返す
    /// （`[1]`・`[1, n]`・`[m, n]` 等ブロードキャスト可能な shape は
    /// 成功する。CPU／CUDA／Metal で同一の意味論。#203 Review 指摘）。
    fn gemm_bias_act(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        bias: Option<&Tensor<f32>>,
        act: Activation,
    ) -> Result<Tensor<f32>, BackendError> {
        let mut out = self.gemm(a, b)?;
        if let Some(bias) = bias {
            out = self.add(&out, bias)?;
        }
        out = match act {
            Activation::None => out,
            Activation::Relu => self.relu(&out)?,
        };
        Ok(out)
    }

    /// デバイス常駐 `w`（・`bias`）のまま `y = a @ w (+ bias)` を計算する
    /// （イシュー #1022・#1023「R3」・`docs/device-resident-update-design.md`
    /// §3.3e）。
    ///
    /// `fandhe_ai_autodiff::optim::device_store::DeviceParamStore::linear_forward`
    /// が学習ループの forward で使う。`a`（ホスト常駐）は毎ステップ変化
    /// する活性化値、`w`（デバイス常駐）は学習対象パラメータであり、
    /// `sgd_step_device` と同じく **本メソッドが `w`／`bias` を
    /// ホストへ download しない**ことが受け入れ条件の中核（本イシューが
    /// 排除する対象は「forward のたびにパラメータをホストへ落とす」
    /// D2H であり、`a`・戻り値の D2H は含まない。`docs/device-resident-
    /// update-design.md` §1.2 の解釈）。
    ///
    /// `w`／`bias` は [`DeviceBufferView`]（イシュー #1023「パラメータ
    /// 横断の単一連結バッファ化」後、`DeviceParamStore` が全パラメータを
    /// 1 本の連結 `DeviceBuffer<f32>` として保持するため、個々の
    /// パラメータは連結バッファ内の要素オフセット範囲としてしか
    /// 表現できない。「R3: 要素オフセット付き常駐ビュー」設計。
    /// `docs/device-resident-update-design.md` 追補参照）で渡す。実装は
    /// `view.offset()..view.offset() + view.numel()` の範囲のみを
    /// `view.shape()` の重みとして扱う契約（この範囲チェック自体は
    /// [`DeviceBufferView::new`] が構築時に行うため、本メソッドの実装は
    /// 追加のオフセット境界検査を要しないが、カーネル側の手動境界検査
    /// 〈REQ-8〉は従来どおり省略しない）。
    ///
    /// `bias` は `Some` の場合 `[n]`（`w` の列数）への行方向複製のみ
    /// 対応する（[`BackendOps::gemm_bias_act`] の融合カーネルと同じ厳密
    /// 一致契約。ブロードキャスト全般は非対応）。`k`（`a` の列数 = `w`
    /// の行数）が 0 の呼び出しは `sgd_step_device` と同様に呼び出し元
    /// （`fandhe_ai_autodiff::nn::linear::Linear::new` が `in_features == 0`
    /// を構築時に拒否する）の契約により実運用では到達しない。
    ///
    /// # デフォルト実装
    ///
    /// 本メソッドは `sgd_step_device`／`gemm_bias_act` と同じ非破壊拡張
    /// （デフォルトメソッド追加。公開 API 非破壊はガードレール条件・
    /// `.claude/rules/security.md`）であり、既定は
    /// [`BackendError::Unsupported`] を返す fail-safe とする（デバイス
    /// 常駐オペランドを扱えないバックエンドが誤って黙示のホスト
    /// フォールバック〈`w` を download してから `gemm_bias_act` へ委譲する
    /// 等〉を行い、D2H 排除という受け入れ条件を静かに破ることを防ぐため。
    /// `download` してよいなら本メソッドを呼ぶ意味がない）。CPU／CUDA／
    /// Metal の各実装はこのデフォルトをカーネル呼び出しでオーバーライド
    /// する（`backend-cpu::ops::CpuBackendOps`・`backend-cuda::ops::
    /// CudaBackendOps`・`backend-metal::ops::MetalBackendOps` 参照）。
    fn gemm_resident_rhs(
        &self,
        _a: &Tensor<f32>,
        _w: DeviceBufferView<'_>,
        _bias: Option<DeviceBufferView<'_>>,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "gemm_resident_rhs: default fail-safe (no resident-operand GEMM kernel available)"
                .into(),
        ))
    }

    /// [`Self::gemm_resident_rhs`] の activation 融合版（イシュー #1044・
    /// `docs/kernel-fusion.md` §2.2）。学習 forward の `Linear` 層に続く
    /// `ReLU` 層をこの epilogue へ折り込み、層 1 個あたりのカーネル
    /// 起動数を 2（gemm+bias／relu）から 1（gemm+bias+act）へ減らす。
    /// 呼び出し元は `fandhe_ai_autodiff::optim::device_store::
    /// DeviceParamStore::linear_forward_with_activation`（次層が
    /// `ReLU` の場合のみ `Activation::Relu` を渡し、それ以外は
    /// `Activation::None` を渡す。bias のみの融合は既存
    /// `gemm_resident_rhs` と同じ）。
    ///
    /// `bias` は `Some` の場合 `[n]`（`w` の列数）への行方向複製のみ
    /// 対応する（[`Self::gemm_resident_rhs`] と同じ厳密一致契約）。
    ///
    /// # デフォルト実装（非破壊拡張）
    ///
    /// `gemm_bias_act`（`gemm` → `add` → `act` の 3 段合成）と同型の
    /// フェイルセーフ合成: [`Self::gemm_resident_rhs`]（bias 融合のみ・
    /// `act` なし）を呼んだ後、`act == Relu` なら結果へ
    /// `self.relu`（ホスト常駐 `Tensor` に対する elementwise。実体化済み
    /// のためこの合成は追加のデバイス常駐制約を破らない）を適用する。
    /// `gemm_resident_rhs` 自体が `Unsupported` を返すバックエンド
    /// （本メソッドをオーバーライドしていないバックエンド）では、この
    /// デフォルトも同じ `Unsupported` を透過的に伝播する。CPU
    /// バックエンド（`backend-cpu::ops::CpuBackendOps`）はこのデフォルトを
    /// カーネル内融合実装（`gemm_blis_bias_act_parallel` へ `act` を
    /// 直接渡す）でオーバーライドする。CUDA／Metal は本イシューの
    /// スコープ外（実機検証環境が必要。`launch_tiled_bias_act_f32_
    /// resident`／`dispatch_strided_bias_act_prepared` は既に `act_relu`
    /// を受け取れるため、後続イシューでの結線は型検査のみで済む）。
    fn gemm_resident_rhs_act(
        &self,
        a: &Tensor<f32>,
        w: DeviceBufferView<'_>,
        bias: Option<DeviceBufferView<'_>>,
        act: Activation,
    ) -> Result<Tensor<f32>, BackendError> {
        let out = self.gemm_resident_rhs(a, w, bias)?;
        match act {
            Activation::None => Ok(out),
            Activation::Relu => self.relu(&out),
        }
    }

    /// デバイス常駐 `w` のまま `c = w @ b` を計算する（イシュー #1022・
    /// #1023「R3」）。
    ///
    /// `DeviceParamStore` の resident backward（`Op::LinearResident` の
    /// VJP。`fandhe_ai_autodiff::grad`）が `d_input^T = w @ g^T` を計算する
    /// ために使う（`w: [k, n]`・`b: [n, m]` → `c: [k, m]`。呼び出し元が
    /// `c` を転置して `d_input: [m, k]` を得る）。[`Self::gemm_resident_rhs`]
    /// と対になる「常駐オペランドが左辺」の形（`w` が左、`b` がホスト
    /// 常駐の右辺）。`w` が [`DeviceBufferView`] を取る理由は
    /// [`Self::gemm_resident_rhs`] と同じ。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::gemm_resident_rhs`] と同じ理由・同じ fail-safe 方針
    /// （[`BackendError::Unsupported`]）のデフォルトメソッド。
    fn gemm_resident_lhs(
        &self,
        _w: DeviceBufferView<'_>,
        _b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "gemm_resident_lhs: default fail-safe (no resident-operand GEMM kernel available)"
                .into(),
        ))
    }

    /// `a`（デバイス常駐）・`w`（デバイス常駐）・`bias`（デバイス常駐・
    /// 任意）から `y = act(a @ w + bias)` を、入力・出力いずれもホストへ
    /// 実体化せずに計算する（イシュー #1028・`docs/inference-forward-
    /// fixed-cost-design.md` §3.2）。
    ///
    /// [`Self::gemm_resident_rhs`] は `a`・戻り値がホスト常駐 `Tensor<f32>`
    /// であり、多層 MLP の推論チェーンでは層ごとに D2H（戻り値）→ H2D
    /// （次層の `a`）が発生する（`docs/backend-cuda-async-execution-
    /// design.md` §2.3 が指摘する「ホスト `Tensor` を返す `BackendOps` API
    /// は戻り値の D2H が構造的な同期点」の具体例）。本メソッドは `a`・
    /// 戻り値をいずれも [`DeviceBuffer`] のまま扱うことで、推論チェーンの
    /// 同期点を最終出力の 1 回（呼び出し元が明示的に `download` する
    /// 箇所）へ集約できるようにする。
    ///
    /// `act` は bias 加算後の elementwise 適用（[`Activation::Relu`] は
    /// `max(0, x)` で数値的に恒等な後段適用。[`Self::gemm_bias_act`] と
    /// 同じ契約）。`bias` は `Some` の場合 `[n]`（`w` の列数）への行方向
    /// 複製のみ対応する（[`Self::gemm_resident_rhs`] と同じ厳密一致契約。
    /// ブロードキャスト全般は非対応）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::gemm_resident_rhs`]・[`Self::gemm_resident_lhs`] と同じ
    /// 非破壊拡張（デフォルトメソッド追加）であり、既定は
    /// [`BackendError::Unsupported`] を返す fail-safe とする（デバイス
    /// 常駐の入出力を扱えないバックエンドが誤って黙示のホスト
    /// フォールバック〈`a` を download して `gemm_bias_act` へ委譲し
    /// 結果を再 upload する等〉を行い、「入出力とも D2H/H2D しない」
    /// という受け入れ条件を静かに破ることを防ぐため。呼び出し元
    /// （`fandhe_ai_autodiff::optim::device_store` の推論ヘルパー）は
    /// `Unsupported` を検出した場合、層構成全体を [`Self::gemm_bias_act`]
    /// ベースの per-op 経路へフォールバックする契約とする）。CPU 実装
    /// （`backend-cpu::ops::CpuBackendOps`）はこのデフォルトをカーネル
    /// 呼び出しでオーバーライドする。CUDA（`backend-cuda::ops::
    /// CudaBackendOps`）・Metal（`backend-metal::ops::MetalBackendOps`）
    /// は #1216 で実装済み（融合カーネル `launch_tiled_bias_act_f32_
    /// resident`／`encode_strided_bias_act_prepared`〈`dispatch_
    /// strided_bias_act_prepared` の encode-only 版。`ctx.synchronize()`
    /// を呼ばずコマンドバッファへ積むのみで待たない〉を再利用し、
    /// `a`／戻り値もデバイス常駐のまま扱う。実機実測は `docs/perf/
    /// linear-forward-device-gpu.md`）。
    fn linear_forward_device(
        &self,
        _a: &DeviceBuffer<f32>,
        _w: DeviceBufferView<'_>,
        _bias: Option<DeviceBufferView<'_>>,
        _act: Activation,
    ) -> Result<DeviceBuffer<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "linear_forward_device: default fail-safe (no device-resident chained forward \
             kernel available)"
                .into(),
        ))
    }

    /// [`Self::linear_forward_device`] と同型だが、Metal のコマンドバッファ
    /// 共有（イシュー #1017）向けに共有失敗トークン [`DispatchFailureCell`]
    /// を追加引数として受け取る非破壊拡張（`sgd_step_device_tracked`／
    /// `gemm_fp32_strict_into_tracked` と同じ「デフォルトメソッド追加」
    /// パターン）。
    ///
    /// # デフォルト実装
    /// 既定は `token` を無視して [`Self::linear_forward_device`] へそのまま
    /// 委譲する。CPU は同期実行のため実行時エラーが即座に返り遅延失敗
    /// トークンを必要としない。CUDA はストリーム順序契約
    /// （`docs/backend-cuda-async-execution-design.md`）に従い本 `token`
    /// を使わずオーバーライドもしない（`sgd_step_device_tracked` と同じ
    /// 判断。設計文書 `docs/inference-chain-single-sync-design.md` 決定 5）。
    /// Metal のオーバーライド（`backend-metal::ops::MetalBackendOps`）は
    /// イシュー #1688 codex-review 指摘対応で追加済み: `token` を
    /// `gemm::MetalGemm::encode_strided_bias_act_prepared_with_c_offset`
    /// へ渡し、`encode` と同一ロック区間でバッチへ登録させる（`gemm_
    /// fp32_strict_into_tracked` と同型）。
    fn linear_forward_device_tracked(
        &self,
        a: &DeviceBuffer<f32>,
        w: DeviceBufferView<'_>,
        bias: Option<DeviceBufferView<'_>>,
        act: Activation,
        _token: &DispatchFailureCell,
    ) -> Result<DeviceBuffer<f32>, BackendError> {
        self.linear_forward_device(a, w, bias, act)
    }

    /// `a op b`（`op` は [`BinaryElementwiseOp`]）を `a`／`b`／戻り値
    /// いずれも [`DeviceBuffer`] 常駐のまま計算する（イシュー #1584）。
    /// `linear_forward_device` と同じ動機（`docs/inference-forward-
    /// fixed-cost-design.md` §2.3 の「ホスト `Tensor` を返す `BackendOps`
    /// API は戻り値の D2H が構造的な同期点」）で、H2D／D2H・同期を伴わず
    /// ストリーム／コマンドバッファへ積むだけの経路を提供し、呼び出し元
    /// が複数の elementwise 演算を連鎖させたうえで最後に 1 回だけ
    /// `download` できるようにする。
    ///
    /// `a`・`b` は shape 完全一致限定（ブロードキャスト非対応。不一致は
    /// `BackendError::ShapeMismatch` で fail-closed）。数値は対応する
    /// ホスト版（`BackendOps::add`／`mul`）と同一カーネルにより bit 同一
    /// となる契約。
    ///
    /// # 同期契約はバックエンドごとに異なる（イシュー #1675 codex-review
    /// 指摘）
    ///
    /// 上記「ストリーム／コマンドバッファへ積むだけで待たない」は
    /// `backend-cuda`（ストリーム順序実行。`docs/backend-cuda-async-
    /// execution-design.md`）の同期契約であり、**全バックエンド共通の
    /// トレイト契約ではない**。`backend-metal` の実装（`elementwise::
    /// MetalElementwise::dispatch_binary_resident`）は `MetalContext::
    /// dispatch_sync` 経由のため、呼び出しごとに 1 回 `waitUntilCompleted`
    /// する（`docs/backend-metal-command-batching-design.md`。呼び出し元
    /// が複数演算を連鎖させても同期点は 1 回に集約されない）。「同期点を
    /// 呼び出し元の `download` へ集約する」設計を前提にする呼び出し側は
    /// バックエンドごとの実装 doc（各 `fn binary_elementwise_device`／
    /// `unary_elementwise_device` オーバーライド）を確認すること。
    ///
    /// # デフォルト実装
    ///
    /// `linear_forward_device` と同じ非破壊拡張パターン（`BackendOps`
    /// トレイトへのデフォルトメソッド追加。公開 API 非破壊はガードレール
    /// 条件・`.claude/rules/security.md`）であり、既定は
    /// [`BackendError::Unsupported`] を返す fail-safe とする（デバイス
    /// 常駐の入出力を扱えないバックエンドが黙示のホストフォールバック
    /// 〈`a`／`b` を download して `add`／`mul` へ委譲し結果を再 upload
    /// する等〉を行い、「入出力とも D2H/H2D しない」という受け入れ条件を
    /// 静かに破ることを防ぐため）。`backend-cpu`・`backend-cuda`・
    /// `backend-metal` の各実装はこのデフォルトをオーバーライドする。
    fn binary_elementwise_device(
        &self,
        _op: BinaryElementwiseOp,
        _a: &DeviceBuffer<f32>,
        _b: &DeviceBuffer<f32>,
    ) -> Result<DeviceBuffer<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "binary_elementwise_device: default fail-safe (no device-resident elementwise \
             kernel available)"
                .into(),
        ))
    }

    /// `op(a)`（`op` は [`UnaryElementwiseOp`]）を `a`・戻り値いずれも
    /// [`DeviceBuffer`] 常駐のまま計算する（イシュー #1584）。
    /// [`Self::binary_elementwise_device`] の単項版で契約・デフォルト
    /// 実装の設計方針は同一（数値は対応するホスト版〈`BackendOps::
    /// relu`／`exp`／`tanh`〉と同一カーネルにより bit 同一）。
    ///
    /// # デフォルト実装
    /// [`Self::binary_elementwise_device`] と同じ非破壊拡張パターン。
    /// 既定は [`BackendError::Unsupported`]。
    fn unary_elementwise_device(
        &self,
        _op: UnaryElementwiseOp,
        _a: &DeviceBuffer<f32>,
    ) -> Result<DeviceBuffer<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "unary_elementwise_device: default fail-safe (no device-resident elementwise \
             kernel available)"
                .into(),
        ))
    }

    /// 融合グラフ（#162 が検出した elementwise 連鎖・#163 が生成する
    /// カーネル）を 1 回のカーネル呼び出しで実行する（TASK-12.1d・#164）。
    ///
    /// `gemm_bias_act` と同型の非破壊拡張（デフォルトメソッド追加）。
    /// デフォルト実装は `BackendError::Unsupported` を返す fail-safe
    /// （既存 elementwise・reduction 未実装カーネルと同じ設計）であり、
    /// `fandhe_ai_autodiff::Tape` の実体化経路（`materialize_fallible`／
    /// `materialize_non_fallible`。`crates/autodiff/src/tape.rs`）は
    /// `Unsupported` を検出した場合に `leaves` を使わず `self`（同じ
    /// `ops`）の per-op メソッド（`add`／`mul`／`relu`／`exp`／`tanh`）へ
    /// 逐次フォールバックする契約（`docs/fusion-graph-design.md` §3.4・
    /// §3.5.2・§3.5.3）。CPU 融合実行の提供元は `backend-cpu` 側の
    /// `run_fused` オーバーライド（#163 のスコープ。本イシュー〈#164〉
    /// 時点では #163 が未マージのため、CPU 側も本デフォルト実装のまま
    /// フォールバックする）。CUDA／Metal は融合カーネル生成が未実装の間
    /// このデフォルトへフォールバックする。
    fn run_fused(
        &self,
        _plan: &FusionPlan,
        _leaves: &[&Tensor<f32>],
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "run_fused: default fail-safe (no fusion kernel available)".into(),
        ))
    }

    /// REQ-14 の明示解放 API（イシュー #1018 ツリー・#1019 設計・#1020
    /// CUDA 実装・#1021 Metal 実装）。このバックエンドのデバイスメモリ
    /// プールがアイドル保持しているバッファを全て解放する。CUDA で
    /// `has_async_alloc()` が真の環境では、自作プール層の解放に加え
    /// driver 側 memory pool のトリム（`cuMemPoolTrimTo(0)` 相当）・
    /// 2 回の対象 stream 同期を内部で行う（`docs/device-memory-pool-
    /// design.md` §3.6 (2) の 4 フェーズ）。Metal は driver トリムを
    /// 持たないためフェーズが少ない（同 doc §3.6 (2)「バックエンド別の
    /// 該当フェーズ」表参照）。
    ///
    /// `crate::pool::PooledMemory::release_all_pooled`（`MemoryOps`
    /// デコレータ側の解放 API）とは別経路であり、本メソッドはホット
    /// パス確保（`backend_ops::BackendOps` 経由の GEMM／elementwise／
    /// softmax カーネル）が使う `SizeClassPool`（`pool_core.rs`）を
    /// 対象とする。
    ///
    /// # デフォルト実装（非破壊拡張）
    /// 既定は `Ok(())`（プールを持たないバックエンドは解放対象なし。
    /// fail-open ではなく「対象が存在しないため自明に成功」という
    /// 意味）。CPU バックエンドは常にこのデフォルトのまま（本イシューの
    /// 対象外。#1026）。CUDA（`backend-cuda::CudaBackendOps`）／Metal
    /// は同 doc §3.6 (2) の契約で実カーネルへオーバーライドする
    /// （`crates/backend-cuda/src/ops.rs`・`crates/backend-metal/src/
    /// ops.rs`）。
    ///
    /// # エラー
    /// `Err` は同 doc §3.6 (2)「バックエンド別の該当フェーズ」表が定める
    /// フェーズのいずれかの失敗を表す。実際に到達しうる `Err` の種別・
    /// 個数はバックエンドごとに異なるため（例: Metal はフェーズ (ii) が
    /// 失敗しない設計のため実質的にフェーズ (i) 失敗の 1 種類のみへ
    /// 到達しうる。CPU は本メソッドを常にデフォルト実装のまま使うため
    /// 到達しない）、本 doc comment では数を明記しない（正本は同 doc
    /// §3.6 (2) の表）。黙殺・panic は禁止する（fail-closed。
    /// `.claude/rules/coding-rust.md`）。
    fn release_cached_device_memory(&self) -> Result<(), BackendError> {
        Ok(())
    }

    /// デバイスメモリプールの統計スナップショット（診断用。イシュー
    /// #1020・#1021）。[`PoolStats`]（POD。内部ハンドル表現を一切
    /// 含まない）のみを返す。
    ///
    /// # デフォルト実装（非破壊拡張）
    /// 既定は `None`（プールを持たないバックエンド。`backend-cpu`）。
    /// `backend-cuda`・`backend-metal` は `Some(stats)` を返す
    /// オーバーライドを持つ。
    fn device_memory_pool_stats(&self) -> Option<PoolStats> {
        None
    }

    /// `C = A @ B` を [`Self::gemm`] と**同一カーネル・同一選択ロジック**
    /// で計算し（`output` を返す場合は `gemm` と bit 同一）、`C` の論理
    /// 領域 `m×n` 全要素和（checksum）を `f64` アキュムレータ（Metal は
    /// Neumaier 補償和で `f64` 相当）・固定順序で決定的に求める（イシュー
    /// #1339・親イシュー #1338）。
    ///
    /// `readout` が [`ChecksumReadout::ChecksumOnly`] のとき、GPU
    /// バックエンド（CUDA／Metal）は `C` をホストへ download せず
    /// checksum（8 バイト）のみ読み戻す契約とする（framework-compare の
    /// 毎反復計測窓から `host_copy` を排除する目的。`docs/perf/
    /// device-checksum-readback-ab.md` 参照）。[`ChecksumReadout::
    /// WithOutput`] のときは続けて `C` も download し
    /// [`GemmChecksum::output`] へ格納する。
    ///
    /// # デフォルト実装
    ///
    /// 本メソッドは `gemm_bias_act`・`linear_forward_device` と同じ
    /// 非破壊拡張パターン（`BackendOps` トレイトへのデフォルトメソッド
    /// 追加。公開 API 非破壊はガードレール条件・`.claude/rules/
    /// security.md`）であり、既定は [`BackendError::Unsupported`] を
    /// 返す fail-safe とする。`fandhe_ai_autodiff::var::Var::
    /// matmul_checksum` は `Unsupported` を透過し呼び出し元（framework-
    /// compare のハーネス）が判定する契約とする（判定迂回経路を作らない。
    /// `.claude/rules/security.md` A08）。`backend-cpu`・`backend-cuda`・
    /// `backend-metal` の各実装はこのデフォルトをオーバーライドする。
    fn gemm_checksum(
        &self,
        _a: &Tensor<f32>,
        _b: &Tensor<f32>,
        _readout: ChecksumReadout,
    ) -> Result<GemmChecksum, BackendError> {
        Err(BackendError::Unsupported(
            "gemm_checksum: default fail-safe (no device-side reduction kernel available)".into(),
        ))
    }

    /// 行方向 RMSNorm（`x · rsqrt(mean(x²) + eps) · w`。`w` が `None` の
    /// 場合は乗算をスキップ。イシュー #1596）の独立エントリ。既存の
    /// [`Self::run_fused`] 経由（`match_rmsnorm_plan` の canonical プラン
    /// 一致限定・`mean` 化なし・`eps` なし・`weight` なし）とは別に、
    /// `fandhe_ai_autodiff::var::Var::rms_norm` が直接呼べる入口を
    /// 提供する（`docs/compat-feature-gap.md` §2.7）。
    ///
    /// **契約**: 正規化軸は常に最終軸（[`crate::ops_shape::
    /// row_norm_layout`] が `(rows, hidden)` を導出する）。`weight` を
    /// 渡す場合は shape `[hidden]` を要求する（呼び出し元
    /// `Var::rms_norm` が事前検査する）。戻り値の shape は入力 `x` と
    /// 恒等（正規化は形状を変えない）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::mse_loss`] と同じ非破壊拡張。既定は
    /// [`BackendError::Unsupported`] を返す fail-safe とし、`Var::
    /// rms_norm` は `Unsupported` のときのみホスト参照実装
    /// （`eval::rmsnorm_rows`）へフォールバックする（それ以外のエラーは
    /// 伝播する。判定迂回経路を作らない。`.claude/rules/security.md`
    /// A08）。CPU／CUDA／Metal はいずれもこのデフォルトを既存の
    /// RMSNorm 行カーネル（`run_fused` の RMSNorm 一致経路が使うものと
    /// 同一のカーネル実体）でオーバーライドする。
    fn rmsnorm(
        &self,
        _x: &Tensor<f32>,
        _weight: Option<&Tensor<f32>>,
        _eps: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "rmsnorm: default fail-safe (no fused RMSNorm kernel available)".into(),
        ))
    }

    /// 行方向 LayerNorm（`(x − mean(x)) · rsqrt(var(x) + eps) · w + b`。
    /// `w`／`b` はそれぞれ `None` の場合は対応する演算をスキップ。
    /// 分散は biased（÷N）。イシュー #1596）の独立エントリ。[`Self::
    /// rmsnorm`] と同じ最終軸限定契約・非破壊拡張・フォールバック規律
    /// に従う（`Var::layer_norm` は `Unsupported` のときのみ
    /// `eval::layer_norm_rows` へフォールバックする）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::rmsnorm`] と同じ非破壊拡張・fail-safe。CPU／CUDA／Metal
    /// はいずれも本イシューで新設する専用カーネルでこのデフォルトを
    /// オーバーライドする（既存の融合 `run_fused` canonical プランには
    /// LayerNorm 一致経路を追加しない。LayerNorm は本エントリ経由でのみ
    /// 到達する）。
    fn layer_norm(
        &self,
        _x: &Tensor<f32>,
        _weight: Option<&Tensor<f32>>,
        _bias: Option<&Tensor<f32>>,
        _eps: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "layer_norm: default fail-safe (no fused LayerNorm kernel available)".into(),
        ))
    }

    /// LSTM セルの pointwise 段（イシュー #1647・設計 `docs/autodiff-
    /// rnn-cell-tape-design.md` 決定 1・決定 1b・決定 5）。融合 GEMM
    /// `pre = x·W_ih + b_ih + h_prev·W_hh + b_hh`（`[B,4H]`。列ブロック
    /// 順 `i,f,g,o`）を受け取り、4 ゲートの活性化・セル状態更新・隠れ
    /// 状態を 1 呼び出しで計算する。
    ///
    /// `pre` は `[B, 4H]`（`H` は `c_prev` の列数から導出）、`c_prev` は
    /// `[B, H]`。戻り値 `gates` は活性化後の `i,f,g,o`（`[B, 4H]`。
    /// `Op::LstmCell`／`Op::LstmHidden` の VJP が backward で読む
    /// payload そのもの）、`c` は新セル状態 `[B, H]`、`h` は新隠れ状態
    /// `[B, H]`。
    ///
    /// `c = f·c_prev + i·g`（FMA 契約統一・`.claude/rules/coding-rust.md`）、
    /// `h = o·tanh(c)`。
    ///
    /// `fandhe_ai_autodiff::var::Var::lstm_cell` から呼ばれ、
    /// [`BackendError::Unsupported`] のときのみホスト参照実装
    /// （`fandhe_ai_autodiff::eval::lstm_pointwise`）へフォールバックする
    /// （A08。判定迂回経路を作らない）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::gemm_bias_act`] と同じ非破壊拡張パターン。既定は
    /// [`BackendError::Unsupported`] を返す fail-safe。CPU／CUDA／Metal
    /// の各実装がこのデフォルトをオーバーライドする。
    fn lstm_pointwise(
        &self,
        _pre: &Tensor<f32>,
        _c_prev: &Tensor<f32>,
    ) -> Result<LstmPointwiseOutput, BackendError> {
        Err(BackendError::Unsupported(
            "lstm_pointwise: default fail-safe (no fused LSTM pointwise kernel available)".into(),
        ))
    }

    /// `Op::LstmHidden` の VJP 補助（決定 1b・決定 1b 追記）。`c`
    /// （現在のセル状態）・`gate_o`（forward 記録済みの o ゲート値）・
    /// `dh`（上流勾配）から、o ゲートの pre-activation 勾配
    /// `d_pre_o = dh·tanh(c)·o·(1−o)` と、`cell`（`Op::LstmCell`）へ
    /// 伝播するセル状態勾配 `dc = dh·o·(1−tanh(c)²)` を計算する。
    ///
    /// `c`／`gate_o`／`dh` はいずれも `[B, H]`。戻り値 `(d_pre_o, dc)`
    /// も `[B, H]`。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::lstm_pointwise`] と同じ fail-safe パターン。
    fn lstm_hidden_backward(
        &self,
        _c: &Tensor<f32>,
        _gate_o: &Tensor<f32>,
        _dh: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>), BackendError> {
        Err(BackendError::Unsupported(
            "lstm_hidden_backward: default fail-safe (no fused LSTM hidden backward kernel available)"
                .into(),
        ))
    }

    /// `Op::LstmCell` の VJP 補助（決定 1b）。forward 記録済みの
    /// `gates_ifg`（`i,f,g` の活性化後値。`[B, 3H]`）・`c_prev`
    /// （`[B, H]`）・上流のセル状態勾配 `dc`（`[B, H]`。`Op::LstmHidden`
    /// からの寄与と次 step の `dc_prev` の fan-in 合算済み）から、
    /// `i,f,g` 3 ゲートの pre-activation 勾配 `d_pre_ifg`（`[B, 3H]`）と
    /// 前セル状態への勾配 `dc_prev = dc·f`（`[B, H]`）を計算する。
    ///
    /// `d_pre_i = dc·g·i·(1−i)`、`d_pre_f = dc·c_prev·f·(1−f)`、
    /// `d_pre_g = dc·i·(1−g²)`。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::lstm_pointwise`] と同じ fail-safe パターン。
    fn lstm_cell_backward(
        &self,
        _gates_ifg: &Tensor<f32>,
        _c_prev: &Tensor<f32>,
        _dc: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>), BackendError> {
        Err(BackendError::Unsupported(
            "lstm_cell_backward: default fail-safe (no fused LSTM cell backward kernel available)"
                .into(),
        ))
    }

    /// GRU セルの pointwise 段（決定 1c・決定 5。`reset_after=True`
    /// 規約）。`pre_i = x·W_ih + b_ih`・`pre_h = h_prev·W_hh + b_hh`
    /// （いずれも `[B, 3H]`。列ブロック順 `r,z,n`）と前隠れ状態
    /// `h_prev`（`[B, H]`）を受け取り、`r,z` ゲート・`n`（新候補）・
    /// 新隠れ状態を計算する。
    ///
    /// `r = σ(pre_i_r + pre_h_r)`、`z = σ(pre_i_z + pre_h_z)`、
    /// `q = pre_h_n`（decision 1c: GEMM 再計算を避けるため payload に
    /// 保持する再帰側アフィン値）、`n = tanh(r·q + pre_i_n)`、
    /// `h = z·h_prev + (1−z)·n`。
    ///
    /// 戻り値 `gates` は活性化後の `r,z,n`（`[B, 3H]`）、`q` は再帰側
    /// アフィン値（`[B, H]`。決定 1c）、`h` は新隠れ状態（`[B, H]`）。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::lstm_pointwise`] と同じ fail-safe パターン。
    fn gru_pointwise(
        &self,
        _pre_i: &Tensor<f32>,
        _pre_h: &Tensor<f32>,
        _h_prev: &Tensor<f32>,
    ) -> Result<GruPointwiseOutput, BackendError> {
        Err(BackendError::Unsupported(
            "gru_pointwise: default fail-safe (no fused GRU pointwise kernel available)".into(),
        ))
    }

    /// `Op::GruCell` の VJP 補助。forward 記録済みの `gates_rzn`
    /// （`[B, 3H]`）・`q`（決定 1c の再帰側アフィン値。`[B, H]`）・
    /// `h_prev`（`[B, H]`）・上流勾配 `dh`（`[B, H]`）から、`W_ih` 側
    /// pre-activation 勾配 `d_pre_i`（`[B, 3H]`）・`W_hh` 側
    /// pre-activation 勾配 `d_pre_h`（`[B, 3H]`）・`h_prev` への直接
    /// 勾配 `dh_prev_direct = dh·z`（`[B, H]`）を計算する。
    ///
    /// `dn = dh·(1−z)`、`dz = dh·(h_prev−n)`、`d_pre_n = dn·(1−n²)`、
    /// `dr = d_pre_n·q`、`d_pre_r = dr·r·(1−r)`、
    /// `d_pre_z = dz·z·(1−z)`。`d_pre_i = [d_pre_r, d_pre_z, d_pre_n]`、
    /// `d_pre_h = [d_pre_r, d_pre_z, d_pre_n·r]`（`n` の `q` に対する
    /// 偏微分が `r` であるため、`W_hh` 側の n 列ブロックのみ追加で `r`
    /// を乗じる）。呼び出し元（`grad::vjp`）は `d_pre_h` を用いて
    /// `dh_prev = dh_prev_direct + d_pre_h·W_hhᵀ` を合成する。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::lstm_pointwise`] と同じ fail-safe パターン。
    fn gru_backward(
        &self,
        _gates_rzn: &Tensor<f32>,
        _q: &Tensor<f32>,
        _h_prev: &Tensor<f32>,
        _dh: &Tensor<f32>,
    ) -> Result<GruBackwardOutput, BackendError> {
        Err(BackendError::Unsupported(
            "gru_backward: default fail-safe (no fused GRU backward kernel available)".into(),
        ))
    }

    /// `A^{-1}`（`A: [n,n]`）。イシュー #1621・`docs/autodiff-linalg-design.md`。
    ///
    /// # デフォルト実装
    /// `gemm_checksum` と同じ非破壊拡張パターン。既定は
    /// [`BackendError::Unsupported`]（GPU バックエンドは本イシュー時点で
    /// 未実装。`fandhe_ai_autodiff::var::Var::inv` がこの `Unsupported`
    /// のみをホスト参照実装 `eval::linalg::inv` へフォールバックし、
    /// それ以外のエラー〈特異行列の `InvalidArgument` 等〉は伝播する）。
    /// `A` が特異（ピボットが厳密 0）の場合は `BackendError::
    /// InvalidArgument` を返す契約とする。
    fn linalg_inv(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "linalg_inv: default fail-safe (no device-side linear-algebra kernel available)".into(),
        ))
    }

    /// `A X = B` を解く（`A: [n,n]`・`B: [n,k]` → `X: [n,k]`）。
    /// イシュー #1621。
    ///
    /// # デフォルト実装
    /// [`Self::linalg_inv`] と同じ非破壊拡張・フォールバック契約。
    /// `A` が特異の場合は [`BackendError::InvalidArgument`]。
    fn linalg_solve(
        &self,
        _a: &Tensor<f32>,
        _b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "linalg_solve: default fail-safe (no device-side linear-algebra kernel available)"
                .into(),
        ))
    }

    /// `det(A)`（`A: [n,n]` → スカラー `[]`）。イシュー #1621。
    ///
    /// # デフォルト実装
    /// [`Self::linalg_inv`] と同じ非破壊拡張・フォールバック契約。
    /// 特異行列は `0.0` を返す（`torch.linalg.det` と同じくエラーに
    /// しない。設計文書 §3.5「エラー分類」）。
    fn linalg_det(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "linalg_det: default fail-safe (no device-side linear-algebra kernel available)".into(),
        ))
    }

    /// Cholesky 分解（`A: [n,n]`〈対称正定値。下三角のみ読む〉→
    /// `L: [n,n]`〈下三角、`A = L Lᵀ`〉）。イシュー #1621。
    ///
    /// # デフォルト実装
    /// [`Self::linalg_inv`] と同じ非破壊拡張・フォールバック契約。
    /// 非正定値（対角が非有限または非正）の場合は
    /// [`BackendError::InvalidArgument`]。
    fn linalg_cholesky(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "linalg_cholesky: default fail-safe (no device-side linear-algebra kernel available)"
                .into(),
        ))
    }

    /// reduced QR 分解（`A: [m,n]` → [`QrFactors`]）。イシュー #1621。
    ///
    /// # デフォルト実装
    /// [`Self::linalg_inv`] と同じ非破壊拡張・フォールバック契約。
    fn linalg_qr(&self, _a: &Tensor<f32>) -> Result<QrFactors, BackendError> {
        Err(BackendError::Unsupported(
            "linalg_qr: default fail-safe (no device-side linear-algebra kernel available)".into(),
        ))
    }

    /// reduced SVD（`A: [m,n]` → [`SvdFactors`]）。イシュー #1621。
    ///
    /// # デフォルト実装
    /// [`Self::linalg_inv`] と同じ非破壊拡張・フォールバック契約。
    /// 反復が収束しない場合は [`BackendError::InvalidArgument`]。
    fn linalg_svd(&self, _a: &Tensor<f32>) -> Result<SvdFactors, BackendError> {
        Err(BackendError::Unsupported(
            "linalg_svd: default fail-safe (no device-side linear-algebra kernel available)".into(),
        ))
    }

    /// 行列ノルム（`A: [m,n]`・`ord` → スカラー `[]`）。イシュー #1621。
    /// `ord` が [`MatrixNormOrd::Nuc`]／[`MatrixNormOrd::Spectral`] の
    /// 場合、実装内部で特異値分解を用いてよい（`Var` 側で `svd` ノードを
    /// 別途合成しない。設計文書 §3.2）。
    ///
    /// # デフォルト実装
    /// [`Self::linalg_inv`] と同じ非破壊拡張・フォールバック契約。
    fn linalg_matrix_norm(
        &self,
        _a: &Tensor<f32>,
        _ord: MatrixNormOrd,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "linalg_matrix_norm: default fail-safe (no device-side linear-algebra kernel \
             available)"
                .into(),
        ))
    }

    /// 分散（`Var: [.., dim_len, ..] → reduce_out_shape(dim)`。イシュー
    /// #1723。`torch.var(dim, correction)`／`tf.math.reduce_variance`
    /// 相当）。`dim=None` は全要素縮約（スカラー出力）。`correction`
    /// （`0` = 母分散・`1` = 不偏分散。`torch.var` の `correction`
    /// 引数と同じ意味）。
    ///
    /// # 数値契約
    /// [`Self::sum`]（`.claude/rules/coding-rust.md`「正規化統計・勾配の
    /// 長軸縮約は `f64` アキュムレータで統一する」契約）と同じく、
    /// 縮約対象の各要素を `f64` へ昇格してから平均・二乗和を計算し、
    /// **最後に 1 回だけ** `f32` へ downcast する。
    ///
    /// # エラー契約
    /// 縮約対象の要素数 `n`（`dim=Some(axis)` は `shape[axis]`・
    /// `dim=None` は全要素数）が `0` の場合、または `n <= correction`
    /// （自由度が非正になり `NaN`／`inf` を黙って返してしまう。PyTorch
    /// は `NaN`／`inf` をそのまま返す点と意図的に異なる安全側の判断）
    /// の場合は [`BackendError::InvalidArgument`] を返す。
    ///
    /// # デフォルト実装
    /// [`Self::linalg_inv`] と同じ非破壊拡張・フォールバック契約。
    fn var(
        &self,
        _a: &Tensor<f32>,
        _dim: Option<usize>,
        _correction: usize,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "var: default fail-safe (no device-side reduction kernel available)".into(),
        ))
    }

    /// ベクトルノルム（`ord` は [`VectorNormOrd`]。イシュー #1723。
    /// `torch.norm`／`tf.norm` の L1／L2 相当）。`dim=None` は全要素
    /// 縮約（スカラー出力）。
    ///
    /// # 数値契約
    /// [`Self::var`] と同じ `f64` 二段計算契約（要素を `f64` へ昇格して
    /// から縮約し、最後に 1 回だけ `f32` へ downcast する）。
    ///
    /// # エラー契約
    /// 縮約対象の要素数が `0` の場合は [`BackendError::InvalidArgument`]
    /// を返す（`L1`／`L2` いずれも空縮約は数学的には単位元 `0.0` を
    /// 持つが、[`Self::var`] と対称な「空縮約は明示エラー」の方針を
    /// 揃える。実装計画 §3.3）。
    ///
    /// # デフォルト実装
    /// [`Self::linalg_inv`] と同じ非破壊拡張・フォールバック契約。
    fn vector_norm(
        &self,
        _a: &Tensor<f32>,
        _ord: VectorNormOrd,
        _dim: Option<usize>,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "vector_norm: default fail-safe (no device-side reduction kernel available)".into(),
        ))
    }
}

/// `default_gemm_batched`（CUDA／Metal が経由する既定合成実装）と
/// `crates/backend-cpu::CpuBackendOps::gemm_batched`（専用オーバーライド
/// 実装）が共有する、バッチ GEMM 出力バッファの要素数を確保前に検証する
/// ヘルパー（PR #1810 codex-review P1 是正・イシュー #1715）。
///
/// `batch_len * m * n` の素朴な乗算は、結果が `usize` の範囲を超える
/// 巨大なバッチ次元では debug ビルドで overflow-checks によりパニック
/// し、release ビルドでは静かにラップして誤ったバッファサイズを生む
/// （本番経路 panic／未定義挙動禁止規約 `.claude/rules/coding-rust.md`
/// に反する）。加えて要素数が `usize` の範囲に収まっても `f32` 要素込み
/// のバイトサイズが `Vec` の allocation 上限（`isize::MAX` バイト）を
/// 超えると `Vec::with_capacity` が capacity overflow でパニックする
/// （`crate::tensor::checked_numel_for` と同じ理由だが、同関数は
/// `pub(crate)` でクレート外〈`backend-cpu` 等〉から呼べないため、本関数
/// を新設して両者が共有する単一情報源とする）。`ShapeError::
/// ElementCountOverflow` は両方のケースを表す型付きエラーとして
/// `checked_numel_for` と同じ規約で再利用する。
pub fn checked_gemm_batched_output_len(
    batch_len: usize,
    m: usize,
    n: usize,
) -> Result<usize, ShapeError> {
    let mn = m.checked_mul(n).ok_or(ShapeError::ElementCountOverflow)?;
    let total = batch_len
        .checked_mul(mn)
        .ok_or(ShapeError::ElementCountOverflow)?;
    let bytes = total
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(ShapeError::ElementCountOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(ShapeError::ElementCountOverflow);
    }
    Ok(total)
}

/// [`BackendOps::gemm_batched`] の既定合成実装（`default_gemm_batched`
/// を `BatchedGemmKind::Standard` で呼ぶ薄いラッパー）を、
/// [`BackendOps::gemm_batched`] をオーバーライド済みのバックエンドから
/// でも明示的に呼べるようにする公開入口（イシュー #1716）。
///
/// `default_gemm_batched`・`BatchedGemmKind` 自体は private のため、
/// `crates/backend-cuda` のように精度モード（TF32 opt-in）ごとに経路を
/// 分岐する必要があるオーバーライドは、この関数を経由して「既定合成
/// （per-batch `T::gemm` 呼び出し）」へ明示的に戻す。**一般利用の公開
/// API ではない**（バックエンドオーバーライド実装、および実機 bit 同一
/// テストのオラクルからの利用を想定した internal-facing な公開関数。
/// `docs/compat-api-scope.md` の対象外）。
///
/// 挙動は [`BackendOps::gemm_batched`] の既定実装のドキュメンテーション
/// コメントと同一（rank 2 は `T::gemm` へ直接委譲・rank≥3 は正規化＋
/// バッチループ）。
pub fn gemm_batched_via_per_batch_gemm<T: BackendOps + ?Sized>(
    ops: &T,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
) -> Result<Tensor<f32>, BackendError> {
    default_gemm_batched(ops, a, b, BatchedGemmKind::Standard)
}

/// [`gemm_batched_via_per_batch_gemm`] の `T::gemm_fp32_strict` 版
/// （[`BackendOps::gemm_batched_fp32_strict`] の既定合成実装。イシュー
/// #1716）。位置づけ・公開範囲の注意は上記と同一。
pub fn gemm_batched_via_per_batch_gemm_fp32_strict<T: BackendOps + ?Sized>(
    ops: &T,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
) -> Result<Tensor<f32>, BackendError> {
    default_gemm_batched(ops, a, b, BatchedGemmKind::Fp32Strict)
}

/// [`BackendOps::gemm_batched`]／[`BackendOps::gemm_batched_fp32_strict`]
/// の既定合成実装が各バッチにどちらの 2 次元カーネルへ委譲するかを表す
/// （イシュー #1715）。
enum BatchedGemmKind {
    /// [`BackendOps::gemm`] へ委譲する。
    Standard,
    /// [`BackendOps::gemm_fp32_strict`] へ委譲する。
    Fp32Strict,
}

/// [`BackendOps::gemm_batched`]／[`BackendOps::gemm_batched_fp32_strict`]
/// の既定合成実装本体（イシュー #1715）。`ops` の 2 次元カーネル
/// （`kind` で選択）をバッチループで呼び出す形で汎用にバッチ行列積を
/// 実現する。CUDA／Metal の専用バッチカーネル（後続イシュー）は
/// このデフォルトメソッドをオーバーライドして本関数を経由しなくなる。
fn default_gemm_batched<T: BackendOps + ?Sized>(
    ops: &T,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    kind: BatchedGemmKind,
) -> Result<Tensor<f32>, BackendError> {
    let plan = crate::ops_shape::batched_matmul_plan(a.shape(), b.shape())
        .map_err(BackendError::ShapeMismatch)?;

    let dispatch = |x: &Tensor<f32>, y: &Tensor<f32>| match kind {
        BatchedGemmKind::Standard => ops.gemm(x, y),
        BatchedGemmKind::Fp32Strict => ops.gemm_fp32_strict(x, y),
    };

    // rank 2 同士（`plan.batch_shape` が空）は 2 次元カーネルへ直接
    // 委譲する。バッチをほどく正規化・ループを経由しないため、
    // rank 2 入力に対しては `gemm`/`gemm_fp32_strict` 単体呼び出しと
    // bit 完全一致することが構造的に保証される。
    if plan.batch_shape().is_empty() {
        return dispatch(a, b);
    }

    let m = plan.m();
    let k = plan.k();
    let n = plan.n();
    let batch_len: usize = plan.batch_shape().iter().product();

    // 出力バッファ（`out_data`）を確保する前に、出力全体の shape
    // （`plan.out_shape()` == `batch_shape ++ [m, n]`）が
    // `Vec::<f32>::with_capacity` の allocation 上限（バイトサイズが
    // `isize::MAX` 以内）に収まるか検査する（PR #1810 codex-review
    // P1 是正）。`plan` 構築時の `checked_numel`（`ops_shape.rs`）は
    // 要素数積が `usize` の範囲に収まるかしか見ないため、要素数積は
    // 収まるがバイトサイズは超える shape（例えば batch 次元が
    // `isize::MAX as usize / 4 + 1` で m = n = 1）を通過させてしまい、
    // 後続の `Vec::with_capacity(batch_len * m * n)` が capacity
    // overflow で panic する（本番経路 panic 禁止方針
    // `.claude/rules/coding-rust.md` に反する）。
    checked_numel_for::<f32>(&plan.out_shape()).map_err(BackendError::ShapeMismatch)?;

    // 出力要素数を確保前に検証する（PR #1810 codex-review P1 是正）。
    // 以前は `Vec::with_capacity(batch_len * m * n)` を素朴な乗算で
    // 呼んでいたため、`batch_len * m * n` が `usize` の範囲を超える
    // 巨大なバッチ次元では debug ビルドで overflow-checks によりパニック
    // し、release ビルドでは静かにラップして誤ったバッファサイズを生む
    // （本番経路 panic／未定義挙動禁止規約 `.claude/rules/coding-rust.md`
    // に反する）。加えて要素数が `usize` の範囲に収まっても `f32` 要素
    // 込みのバイトサイズが `Vec` の allocation 上限（`isize::MAX`
    // バイト）を超えると `Vec::with_capacity` が capacity overflow で
    // パニックする（`crate::tensor::checked_numel_for` と同じ理由。
    // 同関数は `pub(crate)` でクレート外〈`backend-cpu` 等〉から呼べない
    // ため、[`checked_gemm_batched_output_len`] を新設して両者が共有する
    // 単一情報源とする）。
    let total =
        checked_gemm_batched_output_len(batch_len, m, n).map_err(BackendError::ShapeMismatch)?;
    // 出力要素数が 0（`m == 0` または `n == 0`）なら結果は空テンソルで
    // 確定するため、バッチループへ入らずに返す。空の `k`／`m` 軸を持つ
    // 入力は実データなしで巨大な `batch_len`（例: `isize::MAX / 4 + 1`）
    // を構成できるので、no-op GEMM を `batch_len` 回繰り返すと実質
    // ハングする（PR #1810 Cursor Bugbot Medium 是正）。
    // 出力要素数の検査と 0 判定はオペランドの正規化（broadcast の
    // 実体化）より前に置く（PR #1810 codex-review P2 是正）。空の結果
    // に対して巨大な入力コピー（例: a = [1, 1, 1]・b = [B, 1, 0] で
    // a を B 要素へ実体化）を行わないため。
    if total == 0 {
        return Tensor::new(Vec::new(), &plan.out_shape()).map_err(BackendError::ShapeMismatch);
    }

    // 各オペランドを `[B, m, k]`／`[B, k, n]` の contiguous 3 次元へ
    // 正規化する（`docs/compat-api-scope.md` §1.2 実装記録・イシュー
    // #1715 実装計画 §2.2 の「正規化規則」）。
    let a_norm = normalize_batched_operand(a, plan.batch_shape(), m, k)?;
    let b_norm = normalize_batched_operand(b, plan.batch_shape(), k, n)?;

    let mut out_data: Vec<f32> = Vec::with_capacity(total);
    for i in 0..batch_len {
        let a_i = a_norm
            .narrow(0, i, 1)
            .and_then(|t| t.reshape(&[m, k]))
            .map_err(BackendError::ShapeMismatch)?;
        let b_i = b_norm
            .narrow(0, i, 1)
            .and_then(|t| t.reshape(&[k, n]))
            .map_err(BackendError::ShapeMismatch)?;
        let c_i = dispatch(&a_i, &b_i)?;
        // `gemm`/`gemm_fp32_strict` は shape `[m, n]` の新規確保
        // テンソルを返す契約（既存 2 次元カーネル入口の契約そのもの）
        // であり、常に contiguous のため `as_slice` は必ず `Some` を
        // 返す。`None`（契約違反）は fail-closed で拒否する。
        let c_slice = c_i.as_slice().ok_or_else(|| {
            BackendError::InvalidArgument(
                "gemm_batched: per-batch gemm returned a non-contiguous tensor \
                 (contract violation)"
                    .into(),
            )
        })?;
        out_data.extend_from_slice(c_slice);
    }

    let out_shape = plan.out_shape();
    Tensor::new(out_data, &out_shape).map_err(BackendError::ShapeMismatch)
}

/// `default_gemm_batched` が呼ぶ、オペランドをバッチ次元付き
/// contiguous 3 次元 `[B, rows, cols]` へ正規化するヘルパー
/// （イシュー #1715）。`pub` にして `crates/backend-cpu` の
/// `CpuBackendOps::gemm_batched` オーバーライドからも再利用できるように
/// する（既定合成実装と同じ正規化規則を 2 重管理しない）。呼び出し元
/// （同上）はいずれも `batched_matmul_plan`（`ops_shape.rs`）が
/// 検査済みの rank≥2 shape から `[m, k]`/`[k, n]` を渡すが、本関数は
/// `pub` 公開面であり呼び出し元の契約に依存せず自前で rank を検証する
/// （PR #1810 codex-review 指摘。rank<2 のテンソルを渡すと
/// `operand_rank - 2` が `usize` 減算アンダーフローし、デバッグビルドでは
/// panic・リリースビルドでは範囲外スライス参照になる。
/// `.claude/rules/coding-rust.md` の本番経路 panic 禁止方針）。
///
/// - `operand` の rank が 2 未満の場合 `BackendError::ShapeMismatch`
///   （`ShapeError::RankMismatch { expected: 2, actual }`）を返す。
/// - `out_batch_shape` の要素数積が `usize` の範囲でオーバーフローする
///   場合も同様に `BackendError::ShapeMismatch`
///   （`ShapeError::ElementCountOverflow`）を返す（`checked_numel` 経由。
///   PR #1810 codex-review 指摘。呼び出し元の契約に依存せず自前で検証
///   する理由は上記 rank 検証と同じ）。
///
/// `operand` 自身のバッチ shape（先頭 rank−2 軸）が出力バッチ shape
/// `out_batch_shape` と**異なる**場合（broadcast が必要。中間軸の
/// broadcast を含む）は `broadcast_to(out_batch_shape ++ [rows, cols])`
/// で実体化してから `contiguous()` → `reshape` する。中間軸の
/// broadcast（stride 0）を `reshape` へ直接渡すと
/// `ShapeError::NonContiguousReshape` になるため、必ずこの順序で行う。
/// **等しい**場合は `contiguous()`（既に contiguous なら `Arc` 共有で
/// コピーなし）→ `reshape` のみで足りる。
pub fn normalize_batched_operand(
    operand: &Tensor<f32>,
    out_batch_shape: &[usize],
    rows: usize,
    cols: usize,
) -> Result<Tensor<f32>, BackendError> {
    let operand_rank = operand.shape().len();
    if operand_rank < 2 {
        return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
            expected: 2,
            actual: operand_rank,
        }));
    }
    let operand_batch_shape = &operand.shape()[..operand_rank - 2];

    // 末尾 2 軸（`[rows, cols]`）が要求と一致することを実体化前に検証
    // する（PR #1810 codex-review P1 是正）。本関数は `pub` 公開面の
    // ため呼び出し元の `plan` との整合を前提にできない。不一致のまま
    // 下記の `contiguous()` へ進むと、`full_shape` のバイトサイズ検査
    // （`out_batch_shape ++ [rows, cols]`）が実際の `operand.shape()`
    // を見ていないため、例えば 1 要素を `[H, 1, 1]`（H = isize::MAX /
    // 4 + 1）へ broadcast した view に `rows = cols = 0` を渡すと検査
    // を通過し、`contiguous()` が H 個の f32 を確保しようとして
    // capacity overflow で panic する（本番経路 panic 禁止方針
    // `.claude/rules/coding-rust.md` に反する）。
    let operand_tail = &operand.shape()[operand_rank - 2..];
    if operand_tail != [rows, cols] {
        return Err(BackendError::ShapeMismatch(ShapeError::MatmulDimMismatch {
            lhs: operand.shape().to_vec(),
            rhs: vec![rows, cols],
        }));
    }

    // `out_batch_shape` の要素数積を `checked_numel`（`tensor.rs` と共有）で
    // 検査する（PR #1810 codex-review 指摘。本関数は `pub` 公開面のため、
    // 呼び出し元が渡す `out_batch_shape` の健全性を前提にせず自前で
    // オーバーフローを検査する。未検証の `iter().product()` のままだと
    // 例えば `[usize::MAX, 2]` のような値でオーバーフローチェック有効
    // ビルドで乗算が panic し `.claude/rules/coding-rust.md` の本番経路
    // panic 禁止方針に反する）。
    let flat_len = checked_numel(out_batch_shape).map_err(BackendError::ShapeMismatch)?;
    let mut flat_shape = Vec::with_capacity(out_batch_shape.len() + 2);

    // 実体化後の完全な形状（`out_batch_shape ++ [rows, cols]`）を
    // `checked_numel_for::<f32>` で事前検査する（PR #1810 codex-review
    // P1 是正）。上記の `checked_numel` は要素数積が `usize` の範囲に
    // 収まるかしか見ないため、要素数積は収まるがバイトサイズ
    // （`numel * size_of::<f32>()`）が `Vec` の allocation 上限
    // （`isize::MAX` バイト）を超える shape（例えば
    // `out_batch_shape = [isize::MAX as usize / 4 + 1]`・
    // `rows = cols = 1`）を通過させてしまい、下記の `contiguous()`／
    // `broadcast_to().contiguous()` が内部の `Vec::with_capacity` で
    // capacity overflow により panic する（本番経路 panic 禁止方針
    // `.claude/rules/coding-rust.md` に反する）。`operand` 自身は既に
    // 検証済みの `Tensor` のため equal 分岐（`operand.contiguous()`）
    // 自体は安全だが、broadcast 分岐との分岐前に一括で検査すること
    // で分岐間の重複を避ける。
    let mut full_shape = Vec::with_capacity(out_batch_shape.len() + 2);
    full_shape.extend_from_slice(out_batch_shape);
    full_shape.push(rows);
    full_shape.push(cols);
    checked_numel_for::<f32>(&full_shape).map_err(BackendError::ShapeMismatch)?;

    let normalized = if operand_batch_shape == out_batch_shape {
        operand.contiguous()
    } else {
        operand
            .broadcast_to(&full_shape)
            .map_err(BackendError::ShapeMismatch)?
            .contiguous()
    };

    flat_shape.push(flat_len);
    flat_shape.push(rows);
    flat_shape.push(cols);
    normalized
        .reshape(&flat_shape)
        .map_err(BackendError::ShapeMismatch)
}

/// [`BackendOps::lstm_pointwise`] の戻り値（イシュー #1647）。
///
/// `gates` は活性化後の `i,f,g,o`（`[B, 4H]`。`Op::LstmCell`／
/// `Op::LstmHidden` の VJP が backward で参照する payload そのもの）、
/// `c` は新セル状態、`h` は新隠れ状態（いずれも `[B, H]`）。他クレート
/// （`backend-cpu`／`backend-cuda`／`backend-metal`）が構築するため
/// `#[non_exhaustive]` は付けない（フィールド追加は破壊的変更として
/// 扱う）。
#[derive(Debug, Clone)]
pub struct LstmPointwiseOutput {
    pub gates: Tensor<f32>,
    pub c: Tensor<f32>,
    pub h: Tensor<f32>,
}

/// [`BackendOps::gru_pointwise`] の戻り値（イシュー #1647）。
///
/// `gates` は活性化後の `r,z,n`（`[B, 3H]`）、`q` は決定 1c の再帰側
/// アフィン値（`[B, H]`。GEMM 再計算なしで `∂n/∂r` を復元するための
/// payload）、`h` は新隠れ状態（`[B, H]`）。
#[derive(Debug, Clone)]
pub struct GruPointwiseOutput {
    pub gates: Tensor<f32>,
    pub q: Tensor<f32>,
    pub h: Tensor<f32>,
}

/// 複数の `&dyn BackendOps` を横断して `device` に一致する実装を選択する。
///
/// `device::select_from`（TASK-1.9a）と同型の注入式ディスパッチ:
/// `tensor-core` は `backend-cpu`／`backend-cuda`／`backend-metal` を直接
/// 参照できないため、呼び出し側（結線を担う上位クレート・テスト）が
/// `ops` を注入する。本関数こそが受け入れ条件「同一コードで 3 バック
/// エンドのカーネルが呼び分けられる」の直接の実装であり、`device` の
/// variant にのみ基づいて対応実装を返す（形状・HW ヒューリスティクスは
/// 一切持ち込まない。TASK-11.2b・#68 のスコープ）。
///
/// 対応する実装が `ops` に含まれない場合は
/// [`BackendError::DeviceUnavailable`] を返す（`device::select_from` と
/// 同じエラー variant・同じ意味論。「対応 provider／ops 未登録」を表す）。
pub fn ops_for<'a>(
    ops: &[&'a dyn BackendOps],
    device: Device,
) -> Result<&'a dyn BackendOps, BackendError> {
    ops.iter()
        .find(|candidate| candidate.device() == device)
        .copied()
        .ok_or_else(|| {
            BackendError::DeviceUnavailable(format!(
                "no BackendOps registered for device {device:?}"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::BufferHandle;
    use std::any::Any;

    /// テスト専用のモック `BackendOps`。実バックエンドに依存せず
    /// `ops_for` の選択ロジックを検証するために `tensor-core` 内で定義
    /// する（実バックエンドの検証は各バックエンドクレートの結合テスト
    /// で行う。`device` モジュールの `MockProvider` と同じ位置付け）。
    struct MockOps(Device);

    impl BackendOps for MockOps {
        fn device(&self) -> Device {
            self.0
        }

        fn gemm(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: gemm".into()))
        }

        fn add(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: add".into()))
        }

        fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: mul".into()))
        }

        fn relu(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: relu".into()))
        }

        fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: exp".into()))
        }

        fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: tanh".into()))
        }

        fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: sum".into()))
        }

        fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: max".into()))
        }
    }

    /// [`TypedOps<f64>`] の positive-path テスト用スタブ（イシュー #1687）。
    /// 全メソッドが `Unsupported` を返すだけだが、`&Self → &dyn
    /// TypedOps<f64>` の coercion・dyn 呼び出しが実際に機能することを
    /// 検証する目的のため、値の正しさではなく呼び出しの到達を確認する。
    struct TypedF64StubOps;

    impl TypedOps<f64> for TypedF64StubOps {
        fn gemm(&self, _a: &Tensor<f64>, _b: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
            Err(BackendError::Unsupported("stub: gemm".into()))
        }

        fn add(&self, _a: &Tensor<f64>, _b: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
            Err(BackendError::Unsupported("stub: add".into()))
        }

        fn mul(&self, _a: &Tensor<f64>, _b: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
            Err(BackendError::Unsupported("stub: mul".into()))
        }

        fn relu(&self, _a: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
            Err(BackendError::Unsupported("stub: relu".into()))
        }

        fn exp(&self, _a: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
            Err(BackendError::Unsupported("stub: exp".into()))
        }

        fn tanh(&self, _a: &Tensor<f64>) -> Result<Tensor<f64>, BackendError> {
            Err(BackendError::Unsupported("stub: tanh".into()))
        }

        fn sum(&self, _a: &Tensor<f64>, _dim: Option<usize>) -> Result<Tensor<f64>, BackendError> {
            Err(BackendError::Unsupported("stub: sum".into()))
        }

        fn max(&self, _a: &Tensor<f64>, _dim: Option<usize>) -> Result<Tensor<f64>, BackendError> {
            Err(BackendError::Unsupported("stub: max".into()))
        }
    }

    /// `typed_ops_f64` を `Some(self.0)` へオーバーライドする `BackendOps`
    /// 実装（イシュー #1687）。`&dyn BackendOps` 経由で `TypedOps<f64>` の
    /// 具象実装へ実際に到達できることを検証するために使う（#1697 が
    /// CPU 実装で使うのと同じオーバーライドパターン）。
    struct OpsWithTypedF64(TypedF64StubOps);

    impl BackendOps for OpsWithTypedF64 {
        fn device(&self) -> Device {
            Device::Cpu
        }

        fn typed_ops_f64(&self) -> Option<&dyn TypedOps<f64>> {
            Some(&self.0)
        }

        fn gemm(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: gemm".into()))
        }

        fn add(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: add".into()))
        }

        fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: mul".into()))
        }

        fn relu(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: relu".into()))
        }

        fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: exp".into()))
        }

        fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: tanh".into()))
        }

        fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: sum".into()))
        }

        fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: max".into()))
        }
    }

    /// `gemm_bias_act` のデフォルト実装（非融合合成）を数値検証するための
    /// naive 計算モック。`MockOps`（常に `Unsupported`）と異なり `gemm`／
    /// `add`／`relu` を実際に計算する（行方向ブロードキャストのみ対応する
    /// 簡易 `add`。テスト用途のため `Tensor::get`／strided view には
    /// 対応しない）。
    struct ComputingMockOps;

    impl BackendOps for ComputingMockOps {
        fn device(&self) -> Device {
            Device::Cpu
        }

        fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
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
            Tensor::new(out, &[m, n]).map_err(BackendError::ShapeMismatch)
        }

        fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            // テストで使う形状のみ対応: `a: [m, n]`・`b: [n]`（行方向
            // ブロードキャスト）または同一 shape。
            let a_shape = a.shape().to_vec();
            let a_data = a.as_slice().expect("test: a must be contiguous");
            let b_data = b.as_slice().expect("test: b must be contiguous");
            let out = if b.shape() == a.shape() {
                a_data
                    .iter()
                    .zip(b_data)
                    .map(|(x, y)| x + y)
                    .collect::<Vec<_>>()
            } else if b.shape().len() == 1 && a_shape.len() == 2 && b.shape()[0] == a_shape[1] {
                let n = a_shape[1];
                a_data
                    .iter()
                    .enumerate()
                    .map(|(idx, x)| x + b_data[idx % n])
                    .collect::<Vec<_>>()
            } else {
                return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
                    expected: a_shape.len(),
                    actual: b.shape().len(),
                }));
            };
            Tensor::new(out, &a_shape).map_err(BackendError::ShapeMismatch)
        }

        fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("computing mock: mul".into()))
        }

        fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            let data = a.as_slice().expect("test: a must be contiguous");
            let out = data.iter().map(|x| x.max(0.0)).collect::<Vec<_>>();
            Tensor::new(out, a.shape()).map_err(BackendError::ShapeMismatch)
        }

        fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("computing mock: exp".into()))
        }

        fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("computing mock: tanh".into()))
        }

        fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("computing mock: sum".into()))
        }

        fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("computing mock: max".into()))
        }
    }

    /// object-safe であることの型検査を兼ねる（`Box<dyn BackendOps>` が
    /// 構築できることをコンパイル時に確認する）。
    fn assert_object_safe(_ops: &dyn BackendOps) {}

    #[test]
    fn gemm_bias_act_default_matches_manual_composition() {
        let ops = ComputingMockOps;
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let bias = Tensor::new(vec![-100.0, 1.0], &[2]).unwrap();

        // A@B = [[19, 22], [43, 50]] → + bias [-100, 1] → [[-81, 23], [-57, 51]]
        // → relu → [[0, 23], [0, 51]]
        let out = ops
            .gemm_bias_act(&a, &b, Some(&bias), Activation::Relu)
            .expect("gemm_bias_act should succeed");
        assert_eq!(out.as_slice().unwrap(), &[0.0, 23.0, 0.0, 51.0]);
    }

    #[test]
    fn gemm_bias_act_default_no_bias_no_act_matches_gemm() {
        let ops = ComputingMockOps;
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let plain_gemm = ops.gemm(&a, &b).unwrap();
        let fused = ops
            .gemm_bias_act(&a, &b, None, Activation::None)
            .expect("gemm_bias_act should succeed");
        assert_eq!(
            plain_gemm.as_slice().unwrap(),
            fused.as_slice().unwrap(),
            "bias=None・act=None は gemm と同一結果のはず"
        );
    }

    #[test]
    fn gemm_bias_act_default_propagates_unsupported_from_composed_ops() {
        // `MockOps` は `gemm` 自体が `Unsupported` を返すため、
        // デフォルト実装が最初のステップのエラーをそのまま伝播することを
        // 検証する（GPU バックエンドが GEMM 自体未実装の場合の fail-safe。
        // elementwise 未実装〈`add`/`relu` が `Unsupported`〉の伝播は
        // `backend-cuda`/`backend-metal` の結合テスト側で検証する）。
        let ops = MockOps(Device::Cpu);
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let result = ops.gemm_bias_act(&a, &b, None, Activation::Relu);
        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    #[test]
    fn ops_for_dispatches_to_matching_device() {
        let cpu = MockOps(Device::Cpu);
        let cuda = MockOps(Device::Cuda(0));
        let ops: Vec<&dyn BackendOps> = vec![&cpu, &cuda];

        let selected = ops_for(&ops, Device::Cuda(0)).expect("cuda ops registered");
        assert_eq!(selected.device(), Device::Cuda(0));
        assert_object_safe(selected);

        let selected = ops_for(&ops, Device::Cpu).expect("cpu ops registered");
        assert_eq!(selected.device(), Device::Cpu);
    }

    #[test]
    fn ops_for_missing_device_returns_device_unavailable() {
        let cpu = MockOps(Device::Cpu);
        let ops: Vec<&dyn BackendOps> = vec![&cpu];

        // `ops_for` の `Ok` 側は `&dyn BackendOps` を含み `Debug` を実装
        // しないため `expect_err` は使わず、`is_err`／`matches!` で
        // `Err` 経路のみ検査する。
        let result = ops_for(&ops, Device::Cuda(0));
        assert!(result.is_err());
        assert!(matches!(result, Err(BackendError::DeviceUnavailable(_))));
    }

    #[test]
    fn unsupported_error_carries_shape_error_independently() {
        // `BackendError::Unsupported` が既存 variant（`ShapeMismatch` 等）と
        // 独立して構築・表示できることを確認する（非破壊追加の検証）。
        let err = BackendError::Unsupported("elementwise add on cuda".into());
        assert!(err.to_string().contains("elementwise add on cuda"));

        let shape_err = BackendError::ShapeMismatch(ShapeError::RankMismatch {
            expected: 2,
            actual: 1,
        });
        assert!(!shape_err.to_string().is_empty());
    }

    #[test]
    fn run_fused_default_returns_unsupported() {
        // `run_fused`（TASK-12.1d・#164）のデフォルト実装は `Unsupported`
        // を返す fail-safe（`gemm_bias_act` 等の既存 elementwise・
        // reduction 未実装カーネルと同型の設計。backend_ops.rs 冒頭コメ
        // ント参照）。`MockOps` はこのデフォルトを override しない。
        let ops = MockOps(Device::Cpu);
        // `from_ops`（`fusion::plan`。TASK-12.1c・#163）は「`Input` エント
        // リのみで elementwise ノードが 1 個も無い」プランを
        // `FusionPlanError::NoElementwiseNode` として拒否する契約
        // （融合する意味が無いため。`plan.rs` ドキュメント参照）ため、本
        // テストは最小の elementwise ノード（`Relu`）を 1 個含む有効な
        // プランを使う。
        let plan = crate::fusion::FusionPlan::from_ops(
            vec![
                crate::fusion::FusedOpKind::Input { leaf_index: 0 },
                crate::fusion::FusedOpKind::Relu { input: 0 },
            ],
            vec![4],
            crate::dispatch::DType::F32,
            1,
        )
        .expect("from_ops should succeed for a minimal single-op plan");
        let leaf = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let leaves: Vec<&Tensor<f32>> = vec![&leaf];
        let result = ops.run_fused(&plan, &leaves);
        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// テスト専用の最小 `BufferHandle`（イシュー #1017・
    /// `sgd_step_device_tracked_default_delegates_to_sgd_step_device`
    /// が `DeviceBuffer<f32>` を構築するためだけに使う。データの実体は
    /// 持たず downcast のためだけの空ハンドル）。
    #[derive(Debug)]
    struct EmptyHandle;

    impl BufferHandle for EmptyHandle {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    fn empty_device_buffer(device: Device) -> DeviceBuffer<f32> {
        DeviceBuffer::new(device, vec![1], Box::new(EmptyHandle))
    }

    /// [`BackendOps::sgd_step_device_tracked`] のデフォルト実装が
    /// `token` を無視して [`BackendOps::sgd_step_device`] へそのまま
    /// 委譲することを確認する（イシュー #1017 の非破壊拡張ガード。
    /// `MockOps` はいずれのメソッドもオーバーライドしていないため、
    /// 両者が同一の `Unsupported` メッセージを返すことで委譲を検証する）。
    #[test]
    fn sgd_step_device_tracked_default_delegates_to_sgd_step_device() {
        let ops = MockOps(Device::Cpu);
        let mut param = empty_device_buffer(Device::Cpu);
        let grad = empty_device_buffer(Device::Cpu);
        let config = SgdStepConfig {
            lr: 0.1,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            is_first_step: true,
        };
        let token = DispatchFailureCell::new();

        let direct = ops.sgd_step_device(&mut param, &grad, None, &config);
        let tracked = ops.sgd_step_device_tracked(&mut param, &grad, None, &config, &token);

        match (direct, tracked) {
            (Err(BackendError::Unsupported(a)), Err(BackendError::Unsupported(b))) => {
                assert_eq!(a, b);
            }
            other => panic!("expected both to return the same Unsupported error: {other:?}"),
        }
        // デフォルト委譲は token に一切触れない。
        assert!(!token.is_set());
    }

    /// [`BackendOps::linear_forward_device`] の既定実装が fail-safe
    /// （[`BackendError::Unsupported`]）を返すことを確認する（イシュー
    /// #1028）。デバイス常駐の入出力を扱えないバックエンド（`MockOps`）
    /// が黙示のホストフォールバックへ落ちず、明示的に拒否することが
    /// 受け入れ条件の中核（`docs/inference-forward-fixed-cost-design.md`
    /// §3.2 のフォールバック契約）。
    #[test]
    fn linear_forward_device_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let a = empty_device_buffer(Device::Cpu);
        let w_buf = empty_device_buffer(Device::Cpu);
        let w_view = DeviceBufferView::new(&w_buf, 0, &[1]).unwrap();

        let result = ops.linear_forward_device(&a, w_view, None, Activation::None);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::linear_forward_device_tracked`] のデフォルト実装が
    /// `token` を無視して [`BackendOps::linear_forward_device`] へそのまま
    /// 委譲することを確認する（イシュー #1688。
    /// `sgd_step_device_tracked_default_delegates_to_sgd_step_device` と
    /// 同型のガード）。
    #[test]
    fn linear_forward_device_tracked_default_delegates_to_linear_forward_device() {
        let ops = MockOps(Device::Cpu);
        let a = empty_device_buffer(Device::Cpu);
        let w_buf = empty_device_buffer(Device::Cpu);
        let w_view = DeviceBufferView::new(&w_buf, 0, &[1]).unwrap();
        let token = DispatchFailureCell::new();

        let direct = ops.linear_forward_device(&a, w_view, None, Activation::None);
        let tracked = ops.linear_forward_device_tracked(&a, w_view, None, Activation::None, &token);

        match (direct, tracked) {
            (Err(BackendError::Unsupported(a)), Err(BackendError::Unsupported(b))) => {
                assert_eq!(a, b);
            }
            other => panic!("expected both to return the same Unsupported error: {other:?}"),
        }
        // デフォルト委譲は token に一切触れない。
        assert!(!token.is_set());
    }

    /// [`BackendOps::scalar_unary`] の既定実装が fail-safe
    /// （[`BackendError::Unsupported`]）を返すことを確認する（イシュー
    /// #1634。CUDA／Metal は本イシュー時点で未実装のため既定のまま
    /// （親 #1592 の分担）であることのガード）。
    #[test]
    fn scalar_unary_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let a = Tensor::new(vec![1.0_f32, 2.0, 3.0], &[3]).unwrap();

        let result = ops.scalar_unary(crate::scalar_op::ScalarUnaryOp::Relu, &a);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::scalar_binary`] の既定実装が fail-safe
    /// （[`BackendError::Unsupported`]）を返すことを確認する（イシュー
    /// #1634。`scalar_unary_default_is_unsupported` と同型のガード）。
    #[test]
    fn scalar_binary_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let a = Tensor::new(vec![1.0_f32, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::new(vec![4.0_f32, 5.0, 6.0], &[3]).unwrap();

        let result = ops.scalar_binary(crate::scalar_op::ScalarBinaryOp::Add, &a, &b);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::binary_elementwise_device`] の既定実装が fail-safe
    /// （[`BackendError::Unsupported`]）を返すことを確認する（イシュー
    /// #1584。`linear_forward_device_default_is_unsupported` と同型の
    /// ガード）。
    #[test]
    fn binary_elementwise_device_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let a = empty_device_buffer(Device::Cpu);
        let b = empty_device_buffer(Device::Cpu);

        let result = ops.binary_elementwise_device(BinaryElementwiseOp::Add, &a, &b);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::unary_elementwise_device`] の既定実装が fail-safe
    /// （[`BackendError::Unsupported`]）を返すことを確認する（イシュー
    /// #1584。`binary_elementwise_device_default_is_unsupported` と同型）。
    #[test]
    fn unary_elementwise_device_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let a = empty_device_buffer(Device::Cpu);

        let result = ops.unary_elementwise_device(UnaryElementwiseOp::Relu, &a);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::mse_loss`]／[`BackendOps::mse_loss_backward`] の
    /// 既定実装が両方とも fail-safe（[`BackendError::Unsupported`]）を
    /// 返すことを確認する（イシュー #1045。`run_fused_default_returns_
    /// unsupported`・`linear_forward_device_default_is_unsupported` と
    /// 同型のガード）。`MockOps` はいずれのメソッドもオーバーライドして
    /// いないため、融合カーネル未実装のバックエンドが黙示のホスト
    /// フォールバックへ落ちず明示的に拒否することが受け入れ条件の中核。
    #[test]
    fn mse_loss_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let pred = Tensor::new(vec![1.0, 2.0], &[2]).unwrap();
        let target = Tensor::new(vec![0.0, 0.0], &[2]).unwrap();

        let forward = ops.mse_loss(&pred, &target, MseReduction::Mean);
        let backward = ops.mse_loss_backward(&pred, &target, 1.0);

        assert!(matches!(forward, Err(BackendError::Unsupported(_))));
        assert!(matches!(backward, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::huber_loss`]／[`BackendOps::huber_loss_backward`] の
    /// 既定実装が fail-safe（[`BackendError::Unsupported`]）を返すことを
    /// 確認する（イシュー #1739。`mse_loss_default_is_unsupported` と
    /// 同型）。
    #[test]
    fn huber_loss_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let pred = Tensor::new(vec![1.0, 2.0], &[2]).unwrap();
        let target = Tensor::new(vec![0.0, 0.0], &[2]).unwrap();

        let forward = ops.huber_loss(&pred, &target, HuberKind::Huber, 1.0, MseReduction::Mean);
        let backward = ops.huber_loss_backward(&pred, &target, HuberKind::Huber, 1.0, 1.0);

        assert!(matches!(forward, Err(BackendError::Unsupported(_))));
        assert!(matches!(backward, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::bce_loss`]／[`BackendOps::bce_loss_backward`] の
    /// 既定実装が両方とも fail-safe（[`BackendError::Unsupported`]）を
    /// 返すことを確認する（イシュー #1737。
    /// `mse_loss_default_is_unsupported` と同型のガード）。
    #[test]
    fn bce_loss_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let input = Tensor::new(vec![0.3, 0.7], &[2]).unwrap();
        let target = Tensor::new(vec![0.0, 1.0], &[2]).unwrap();

        let forward = ops.bce_loss(&input, &target, BceKind::Probabilities, MseReduction::Mean);
        let backward = ops.bce_loss_backward(&input, &target, BceKind::Probabilities, 1.0);

        assert!(matches!(forward, Err(BackendError::Unsupported(_))));
        assert!(matches!(backward, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::softmax`] の既定実装が fail-safe
    /// （[`BackendError::Unsupported`]）を返すことを確認する
    /// （イシュー #1594。`mse_loss_default_is_unsupported` と同型）。
    #[test]
    fn softmax_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let x = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        let result = ops.softmax(&x, 0);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::cumsum`] の既定実装が fail-safe を返すことを確認
    /// する（イシュー #1731。`softmax_default_is_unsupported` と同型）。
    #[test]
    fn cumsum_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let x = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        let result = ops.cumsum(&x, 0);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::cumprod`] の既定実装が fail-safe を返すことを確認
    /// する（イシュー #1731）。
    #[test]
    fn cumprod_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let x = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        let result = ops.cumprod(&x, 0);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::log_softmax`] の既定実装が fail-safe を返すことを
    /// 確認する（イシュー #1594）。
    #[test]
    fn log_softmax_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let x = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        let result = ops.log_softmax(&x, 0);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::concat`] の既定実装が fail-safe を返すことを
    /// 確認する（イシュー #1598）。
    #[test]
    fn concat_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let a = Tensor::new(vec![1.0, 2.0], &[2]).unwrap();
        let b = Tensor::new(vec![3.0, 4.0], &[2]).unwrap();

        let result = ops.concat(&[&a, &b], 0);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::where_cond`] の既定実装が fail-safe を返すことを
    /// 確認する（イシュー #1637）。
    #[test]
    fn where_cond_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let cond = Tensor::new(vec![1.0, 0.0], &[2]).unwrap();
        let a = Tensor::new(vec![1.0, 2.0], &[2]).unwrap();
        let b = Tensor::new(vec![3.0, 4.0], &[2]).unwrap();

        let result = ops.where_cond(&cond, &a, &b);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::masked_fill`] の既定実装が fail-safe を返すことを
    /// 確認する（イシュー #1637）。
    #[test]
    fn masked_fill_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let x = Tensor::new(vec![1.0, 2.0], &[2]).unwrap();
        let mask = Tensor::new(vec![1.0, 0.0], &[2]).unwrap();

        let result = ops.masked_fill(&x, &mask, -1.0);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::gather`] の既定実装が非破壊拡張の fail-safe 契約
    /// （`Unsupported`）を満たすことを確認する（イシュー #1776）。
    #[test]
    fn gather_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let index = Tensor::<i32>::new(vec![0, 0, 1, 0], &[2, 2]).unwrap();

        let result = ops.gather(&input, 1, &index);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::pad`] の既定実装が非破壊拡張の fail-safe 契約
    /// （`Unsupported`）を満たすことを確認する（イシュー #1756）。
    #[test]
    fn pad_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let result = ops.pad(&input, &[(1, 0), (0, 1)], 0.0);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::scatter`] の既定実装が非破壊拡張の fail-safe 契約
    /// （`Unsupported`）を満たすことを確認する（イシュー #1776）。
    #[test]
    fn scatter_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let index = Tensor::<i32>::new(vec![0, 0, 1, 0], &[2, 2]).unwrap();
        let src = Tensor::new(vec![9.0, 9.0, 9.0, 9.0], &[2, 2]).unwrap();

        let result = ops.scatter(&input, 1, &index, &src, ScatterReduce::Overwrite);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::interpolate`] の既定実装が非破壊拡張の fail-safe
    /// 契約（`Unsupported`）を満たすことを確認する（イシュー #1757）。
    #[test]
    fn interpolate_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let result = ops.interpolate(&input, &[4, 4], InterpolateMode::Nearest);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::sort`] の既定実装が非破壊拡張の fail-safe 契約
    /// （`Unsupported`）を満たすことを確認する（イシュー #1733）。
    #[test]
    fn sort_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let input = Tensor::new(vec![3.0, 1.0, 2.0, 4.0], &[2, 2]).unwrap();

        let result = ops.sort(&input, 1, false);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::topk`] の既定実装が非破壊拡張の fail-safe 契約
    /// （`Unsupported`）を満たすことを確認する（イシュー #1733）。
    #[test]
    fn topk_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let input = Tensor::new(vec![3.0, 1.0, 2.0, 4.0], &[2, 2]).unwrap();

        let result = ops.topk(&input, 1, 1, true);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::min`] の既定実装が非破壊拡張の fail-safe 契約
    /// （`Unsupported`）を満たすことを確認する（イシュー #1720）。
    #[test]
    fn min_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let input = Tensor::new(vec![3.0, 1.0, 2.0, 4.0], &[2, 2]).unwrap();

        let result = ops.min(&input, Some(1));

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::argmax`] の既定実装が非破壊拡張の fail-safe 契約
    /// （`Unsupported`）を満たすことを確認する（イシュー #1720）。
    #[test]
    fn argmax_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let input = Tensor::new(vec![3.0, 1.0, 2.0, 4.0], &[2, 2]).unwrap();

        let result = ops.argmax(&input, Some(1));

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::argmin`] の既定実装が非破壊拡張の fail-safe 契約
    /// （`Unsupported`）を満たすことを確認する（イシュー #1720）。
    #[test]
    fn argmin_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let input = Tensor::new(vec![3.0, 1.0, 2.0, 4.0], &[2, 2]).unwrap();

        let result = ops.argmin(&input, Some(1));

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::one_hot`] の既定実装が非破壊拡張の fail-safe 契約
    /// （`Unsupported`）を満たすことを確認する（イシュー #1755）。
    #[test]
    fn one_hot_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let index = Tensor::<i32>::new(vec![0, 2, 1, 1], &[2, 2]).unwrap();

        let result = ops.one_hot(&index, 3);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::unique`] の既定実装が非破壊拡張の fail-safe 契約
    /// （`Unsupported`）を満たすことを確認する（イシュー #1734）。
    #[test]
    fn unique_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let input = Tensor::new(vec![3.0, 1.0, 2.0, 1.0], &[2, 2]).unwrap();

        let result = ops.unique(&input);
        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::captured_segment_key`]／[`BackendOps::
    /// run_captured_sgd_step_segment`] の既定実装が非破壊拡張の
    /// fail-safe 契約（前者は `Ok(None)`・後者は `Err(Unsupported)`）を
    /// 満たすことを確認する（イシュー #1349）。`MockOps` は CUDA Graph
    /// 機構を持たないため、graph 非対応バックエンド・opt-in OFF の既定
    /// 状態を模す。
    #[test]
    fn captured_segment_key_default_is_none() {
        let ops = MockOps(Device::Cpu);
        let buf = empty_device_buffer(Device::Cpu);
        let key = ops.captured_segment_key(&[&buf], 0);
        assert!(matches!(key, Ok(None)));
    }

    /// [`BackendOps::run_captured_sgd_step_segment`] の既定実装は区間
    /// 本体（SGD 更新）を一切実行せずに `Unsupported` を返す（呼び出し元
    /// の契約「`captured_segment_key` が `Some` を返したときのみ呼ぶ」が
    /// 守られていれば到達しない経路だが、契約違反時も二重実行等の副作用
    /// を起こさないことを固定する）。`param` が変化しないことで「本体
    /// 未実行」を確認する（codex-review P0 指摘対応で `body` クロージャ
    /// を廃したため、旧テストの `call_count` の代わりに副作用の不在を
    /// 直接観測する）。
    #[test]
    fn run_captured_sgd_step_segment_default_is_unsupported_and_has_no_effect() {
        let ops = MockOps(Device::Cpu);
        let key = SegmentKey {
            generation: 0,
            config_key: 0,
            resources: vec![SegmentResource { addr: 0, numel: 0 }],
        };
        let mut param = empty_device_buffer(Device::Cpu);
        let grad = empty_device_buffer(Device::Cpu);
        let config = SgdStepConfig {
            lr: 0.1,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            is_first_step: true,
        };
        let token = DispatchFailureCell::new();
        let result =
            ops.run_captured_sgd_step_segment(key, &mut param, &grad, None, &config, &token);
        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// `&dyn BackendOps` 経由でも新規デフォルトメソッドを呼べる
    /// （object-safety が壊れていない）ことを確認する（`run_captured_
    /// sgd_step_segment` が object-safe な形で trait に追加できている
    /// ことの回帰ガード）。
    #[test]
    fn captured_segment_methods_are_object_safe() {
        let ops = MockOps(Device::Cpu);
        let dyn_ops: &dyn BackendOps = &ops;
        let buf = empty_device_buffer(Device::Cpu);
        assert!(matches!(dyn_ops.captured_segment_key(&[&buf], 0), Ok(None)));
    }
    /// [`BackendOps::gemm_checksum`] の既定実装が fail-safe
    /// （[`BackendError::Unsupported`]）を返すことを確認する（イシュー
    /// #1339。`mse_loss_default_is_unsupported` と同型のガード）。
    #[test]
    fn gemm_checksum_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let a = Tensor::new(vec![1.0, 2.0], &[1, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0], &[2, 1]).unwrap();

        let result = ops.gemm_checksum(&a, &b, ChecksumReadout::ChecksumOnly);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::rmsnorm`] の既定実装が fail-safe
    /// （[`BackendError::Unsupported`]）を返すことを確認する（イシュー
    /// #1596。`mse_loss_default_is_unsupported` と同型）。
    #[test]
    fn rmsnorm_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let x = Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();

        let result = ops.rmsnorm(&x, None, 1e-6);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// [`BackendOps::layer_norm`] の既定実装が fail-safe を返すことを
    /// 確認する（イシュー #1596）。
    #[test]
    fn layer_norm_default_is_unsupported() {
        let ops = MockOps(Device::Cpu);
        let x = Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();

        let result = ops.layer_norm(&x, None, None, 1e-5);

        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// RNN／LSTM／GRU セル演算（イシュー #1647）の 5 メソッドすべてが
    /// 既定実装で `BackendError::Unsupported` を返すことを確認する
    /// （`gemm_checksum_default_is_unsupported` と同型のガード。
    /// `fandhe_ai_autodiff::var::{rnn_cell,lstm_cell,gru_cell}` はこの
    /// 契約に依存してホスト参照実装〈`eval.rs`〉へフォールバックする）。
    #[test]
    fn rnn_cell_ops_default_are_unsupported() {
        let ops = MockOps(Device::Cpu);
        let b = 2usize;
        let hidden = 3usize;
        let pre4h = Tensor::new(vec![0.0f32; b * 4 * hidden], &[b, 4 * hidden]).unwrap();
        let pre3h = Tensor::new(vec![0.0f32; b * 3 * hidden], &[b, 3 * hidden]).unwrap();
        let bh = Tensor::new(vec![0.0f32; b * hidden], &[b, hidden]).unwrap();

        assert!(matches!(
            ops.lstm_pointwise(&pre4h, &bh),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.lstm_hidden_backward(&bh, &bh, &bh),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.lstm_cell_backward(&pre3h, &bh, &bh),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.gru_pointwise(&pre3h, &pre3h, &bh),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.gru_backward(&pre3h, &bh, &bh, &bh),
            Err(BackendError::Unsupported(_))
        ));
    }

    /// [`BackendOps::linalg_*`]（7 メソッド）の既定実装がいずれも
    /// fail-safe（[`BackendError::Unsupported`]）を返すことを確認する
    /// （イシュー #1621。`gemm_checksum_default_is_unsupported` と同型の
    /// 回帰ガード）。
    #[test]
    fn linalg_defaults_are_unsupported() {
        let ops = MockOps(Device::Cpu);
        let a = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0], &[2, 1]).unwrap();

        assert!(matches!(
            ops.linalg_inv(&a),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.linalg_solve(&a, &b),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.linalg_det(&a),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.linalg_cholesky(&a),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.linalg_qr(&a),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.linalg_svd(&a),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.linalg_matrix_norm(&a, MatrixNormOrd::Fro),
            Err(BackendError::Unsupported(_))
        ));
    }

    /// [`BackendOps::var`]／[`BackendOps::vector_norm`] の既定実装が
    /// いずれも fail-safe（[`BackendError::Unsupported`]）を返すことを
    /// 確認する（イシュー #1723。`linalg_defaults_are_unsupported` と
    /// 同型の回帰ガード）。
    #[test]
    fn var_vector_norm_defaults_are_unsupported() {
        let ops = MockOps(Device::Cpu);
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        assert!(matches!(
            ops.var(&a, None, 1),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.vector_norm(&a, VectorNormOrd::L1, None),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            ops.vector_norm(&a, VectorNormOrd::L2, Some(0)),
            Err(BackendError::Unsupported(_))
        ));
    }

    /// `typed_ops_f64`／`typed_ops_f16`／`typed_ops_bf16` の 3 accessor が
    /// いずれも既定で `None`（f64/f16/bf16 演算未対応）を返すことを確認
    /// する（イシュー #1687・非破壊拡張の fail-closed 既定値回帰ガード）。
    #[test]
    fn typed_ops_accessors_default_to_none() {
        let ops = MockOps(Device::Cpu);
        assert!(ops.typed_ops_f64().is_none());
        assert!(ops.typed_ops_f16().is_none());
        assert!(ops.typed_ops_bf16().is_none());
    }

    /// `Box<dyn BackendOps + Send>` が成立し続けることを直接検証する
    /// （`fandhe_ai_autodiff::Tape.ops` と同じ型。イシュー #1687の
    /// `typed_ops_*` accessor 追加が object safety・`Send` 境界の両方を
    /// 壊していないことの回帰ガード）。
    fn assert_object_safe_send(_ops: Box<dyn BackendOps + Send>) {}

    #[test]
    fn backend_ops_boxed_with_send_is_object_safe() {
        assert_object_safe_send(Box::new(MockOps(Device::Cpu)) as Box<dyn BackendOps + Send>);
    }

    /// `typed_ops_f64` accessor が `&dyn BackendOps` 経由で具象
    /// `TypedOps<f64>` 実装まで到達することを検証する（イシュー #1687）。
    /// `&Self → &dyn TypedOps<f64>` の coercion・`TypedOps<f64>` の dyn
    /// 互換性・`dyn BackendOps` からの accessor 経由呼び出しの 3 点が
    /// 実際にコンパイル・実行できることの機械検証（#1697 が同じ
    /// オーバーライドパターンを使う前提の裏付け）。
    #[test]
    fn typed_ops_f64_accessor_reaches_concrete_impl_through_dyn_backend_ops() {
        let stub = OpsWithTypedF64(TypedF64StubOps);
        let ops: &dyn BackendOps = &stub;

        let a = Tensor::new(vec![1.0f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![5.0f64, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let result = ops
            .typed_ops_f64()
            .expect("typed_ops_f64 should be Some for OpsWithTypedF64")
            .gemm(&a, &b);
        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// `gemm_batched`／`gemm_batched_fp32_strict` の既定合成実装
    /// （イシュー #1715）を検証するための、実際に 2 次元 GEMM を計算する
    /// テスト用 `BackendOps`。`MockOps` は `gemm` が常に `Unsupported` を
    /// 返すため既定合成の per-batch 委譲を検証できず、本構造体を別途
    /// 用意する（naive 三重ループ。`f32::mul_add` で CPU 参照実装の FMA
    /// 契約〈`.claude/rules/coding-rust.md`〉に揃える）。
    struct NaiveMockOps;

    impl NaiveMockOps {
        fn naive_gemm_2d(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            let out_shape = crate::ops_shape::gemm_out_shape(a.shape(), b.shape())
                .map_err(BackendError::ShapeMismatch)?;
            let (m, k, n) = (a.shape()[0], a.shape()[1], b.shape()[1]);
            let a_s = a.as_slice().expect("gemm input must be contiguous in test");
            let b_s = b.as_slice().expect("gemm input must be contiguous in test");
            let mut out = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for kk in 0..k {
                        acc = a_s[i * k + kk].mul_add(b_s[kk * n + j], acc);
                    }
                    out[i * n + j] = acc;
                }
            }
            Tensor::new(out, &out_shape).map_err(BackendError::ShapeMismatch)
        }
    }

    impl BackendOps for NaiveMockOps {
        fn device(&self) -> Device {
            Device::Cpu
        }

        fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Self::naive_gemm_2d(a, b)
        }

        fn add(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("naive mock: add".into()))
        }

        fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("naive mock: mul".into()))
        }

        fn relu(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("naive mock: relu".into()))
        }

        fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("naive mock: exp".into()))
        }

        fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("naive mock: tanh".into()))
        }

        fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("naive mock: sum".into()))
        }

        fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("naive mock: max".into()))
        }
    }

    #[test]
    fn gemm_batched_rank2_delegates_to_gemm() {
        let ops = NaiveMockOps;
        let a = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = Tensor::new((1..=12).map(|x| x as f32).collect(), &[3, 4]).unwrap();
        let batched = ops.gemm_batched(&a, &b).unwrap();
        let direct = ops.gemm(&a, &b).unwrap();
        assert_eq!(batched.shape(), direct.shape());
        assert_eq!(batched.as_slice().unwrap(), direct.as_slice().unwrap());
    }

    #[test]
    fn gemm_batched_matches_per_batch_gemm() {
        // B=2・m=2・k=3・n=2 のバッチ行列積が、各バッチを個別に
        // `gemm`（2 次元）へ渡した結果と要素単位で完全一致することを
        // 検証する（既定合成実装の bit 同一契約）。
        let ops = NaiveMockOps;
        let a = Tensor::new((0..12).map(|x| x as f32).collect(), &[2, 2, 3]).unwrap();
        let b = Tensor::new((0..12).map(|x| x as f32).collect(), &[2, 3, 2]).unwrap();

        let batched = ops.gemm_batched(&a, &b).unwrap();
        assert_eq!(batched.shape(), &[2, 2, 2]);

        for i in 0..2 {
            let a_i = a.narrow(0, i, 1).unwrap().reshape(&[2, 3]).unwrap();
            let b_i = b.narrow(0, i, 1).unwrap().reshape(&[3, 2]).unwrap();
            let expected = ops.gemm(&a_i, &b_i).unwrap();
            let got = batched.narrow(0, i, 1).unwrap().reshape(&[2, 2]).unwrap();
            assert_eq!(got.as_slice().unwrap(), expected.as_slice().unwrap());
        }
    }

    #[test]
    fn gemm_batched_broadcasts_lhs_batch() {
        // lhs=[1,2,3]（バッチ 1）・rhs=[2,3,2]（バッチ 2）。
        // lhs はバッチ 0 の内容が両方の出力バッチへ繰り返し使われる。
        let ops = NaiveMockOps;
        let a = Tensor::new(vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], &[1, 2, 3]).unwrap();
        let b = Tensor::new((0..12).map(|x| x as f32).collect(), &[2, 3, 2]).unwrap();

        let batched = ops.gemm_batched(&a, &b).unwrap();
        assert_eq!(batched.shape(), &[2, 2, 2]);

        let a_2d = a.reshape(&[2, 3]).unwrap();
        for i in 0..2 {
            let b_i = b.narrow(0, i, 1).unwrap().reshape(&[3, 2]).unwrap();
            let expected = ops.gemm(&a_2d, &b_i).unwrap();
            let got = batched.narrow(0, i, 1).unwrap().reshape(&[2, 2]).unwrap();
            assert_eq!(got.as_slice().unwrap(), expected.as_slice().unwrap());
        }
    }

    #[test]
    fn gemm_batched_shape_mismatch_is_shape_mismatch_error() {
        let ops = NaiveMockOps;
        let a = Tensor::new((0..12).map(|x| x as f32).collect(), &[2, 2, 3]).unwrap();
        let b = Tensor::new((0..16).map(|x| x as f32).collect(), &[2, 4, 2]).unwrap();
        let err = ops.gemm_batched(&a, &b).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }

    #[test]
    fn gemm_batched_zero_batch_returns_empty_output() {
        let ops = NaiveMockOps;
        let a = Tensor::new(Vec::<f32>::new(), &[0, 2, 3]).unwrap();
        let b = Tensor::new(Vec::<f32>::new(), &[0, 3, 2]).unwrap();
        let out = ops.gemm_batched(&a, &b).unwrap();
        assert_eq!(out.shape(), &[0, 2, 2]);
        assert_eq!(out.numel(), 0);
    }

    #[test]
    fn gemm_batched_fp32_strict_matches_gemm_batched_for_backend_without_tf32() {
        // TF32 の概念を持たない CPU 相当のバックエンドでは
        // `gemm_batched_fp32_strict` は `gemm_batched` と同一結果になる
        // （既定 `gemm_fp32_strict` が `gemm` へ委譲するため）。
        let ops = NaiveMockOps;
        let a = Tensor::new((0..12).map(|x| x as f32).collect(), &[2, 2, 3]).unwrap();
        let b = Tensor::new((0..12).map(|x| x as f32).collect(), &[2, 3, 2]).unwrap();
        let standard = ops.gemm_batched(&a, &b).unwrap();
        let strict = ops.gemm_batched_fp32_strict(&a, &b).unwrap();
        assert_eq!(standard.as_slice().unwrap(), strict.as_slice().unwrap());
    }

    #[test]
    fn normalize_batched_operand_rejects_rank_below_2() {
        // PR #1810 codex-review 指摘: `pub` 公開面である
        // `normalize_batched_operand` は呼び出し元の契約（rank≥2 の shape
        // のみ渡す）に依存せず、rank 0（スカラー）・rank 1 の入力を自前で
        // 検出して `Err` を返さなければならない（`operand_rank - 2` の
        // `usize` 減算アンダーフローによる panic・範囲外スライス参照を
        // 防ぐため）。
        let scalar = Tensor::new(vec![1.0f32], &[]).unwrap();
        let err = normalize_batched_operand(&scalar, &[2], 3, 4).unwrap_err();
        assert!(matches!(
            err,
            BackendError::ShapeMismatch(ShapeError::RankMismatch {
                expected: 2,
                actual: 0
            })
        ));

        let rank1 = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        let err = normalize_batched_operand(&rank1, &[2], 3, 4).unwrap_err();
        assert!(matches!(
            err,
            BackendError::ShapeMismatch(ShapeError::RankMismatch {
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn normalize_batched_operand_rejects_batch_shape_element_count_overflow() {
        // PR #1810 codex-review 指摘: `out_batch_shape` の要素数積を
        // 未検証の `iter().product()` で計算すると、正常な rank≥2 の
        // `operand` を渡していても `out_batch_shape` に
        // `[usize::MAX, 2]` のような値が混入した場合、オーバーフロー
        // チェック有効ビルドでは `broadcast_to`／`reshape` へ到達する
        // 前に乗算自体が panic する。`checked_numel` 経由の事前検査で
        // `Err(ShapeError::ElementCountOverflow)` を返すことを確認する。
        let operand = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let err = normalize_batched_operand(&operand, &[usize::MAX, 2], 2, 2).unwrap_err();
        assert!(matches!(
            err,
            BackendError::ShapeMismatch(ShapeError::ElementCountOverflow)
        ));
    }

    #[test]
    fn normalize_batched_operand_rejects_byte_size_overflow_without_element_count_overflow() {
        // PR #1810 codex-review P1 指摘: `out_batch_shape =
        // [isize::MAX as usize / 4 + 1]`・rows = cols = 1 は、要素数積
        // （`checked_numel`。`usize` の範囲）自体はオーバーフローしない
        // が、f32 としてのバイトサイズ（`numel * 4`）は `isize::MAX` を
        // 超えるため `Vec::<f32>::with_capacity` が capacity overflow
        // で panic しうる。`checked_numel_for::<f32>` による事前検査で
        // panic せず `Err(ShapeError::ElementCountOverflow)` を返す
        // ことを確認する（broadcast 分岐に到達する rank≥2 operand で
        // 再現）。
        let huge = isize::MAX as usize / 4 + 1;
        let operand = Tensor::new(vec![1.0f32], &[1, 1]).unwrap();
        let err = normalize_batched_operand(&operand, &[huge], 1, 1).unwrap_err();
        assert!(matches!(
            err,
            BackendError::ShapeMismatch(ShapeError::ElementCountOverflow)
        ));
    }

    #[test]
    fn normalize_batched_operand_rejects_trailing_dims_mismatch_before_materialization() {
        // PR #1810 codex-review P1 指摘: 1 要素を `[H, 1, 1]`
        // （H = isize::MAX / 4 + 1）へ broadcast した view（実データは
        // 1 要素のまま）に `rows = cols = 0` を渡すと、`full_shape =
        // [H, 0, 0]` のバイトサイズ検査は通過するが、末尾 2 軸が一致
        // しないまま `contiguous()` へ進むと H 個の f32 を確保しようと
        // して panic する。末尾 2 軸の不一致を実体化前に
        // `MatmulDimMismatch` として拒否することを確認する。
        let huge = isize::MAX as usize / 4 + 1;
        let one = Tensor::new(vec![1.0f32], &[1, 1, 1]).unwrap();
        let view = one.broadcast_to(&[huge, 1, 1]).unwrap();
        let err = normalize_batched_operand(&view, &[huge], 0, 0).unwrap_err();
        assert!(matches!(
            err,
            BackendError::ShapeMismatch(ShapeError::MatmulDimMismatch { .. })
        ));
    }

    /// [`checked_gemm_batched_output_len`] 単体の境界値検証（PR #1810
    /// codex-review P1 是正）。`batch_len * m * n` 自体は `usize` の
    /// 範囲に収まっても、`f32` 要素込みのバイトサイズが `Vec` の
    /// allocation 上限（`isize::MAX` バイト）を超える場合は
    /// `ElementCountOverflow` を返す（実際の確保を試みない）。
    #[test]
    fn checked_gemm_batched_output_len_rejects_byte_size_overflow() {
        let huge_batch = isize::MAX as usize / 4 + 1;
        let err = checked_gemm_batched_output_len(huge_batch, 1, 1).unwrap_err();
        assert_eq!(err, ShapeError::ElementCountOverflow);
    }

    #[test]
    fn checked_gemm_batched_output_len_rejects_usize_overflow() {
        let err = checked_gemm_batched_output_len(usize::MAX, 2, 2).unwrap_err();
        assert_eq!(err, ShapeError::ElementCountOverflow);
    }

    #[test]
    fn checked_gemm_batched_output_len_accepts_in_range_size() {
        assert_eq!(checked_gemm_batched_output_len(3, 4, 5).unwrap(), 60);
    }

    /// [`default_gemm_batched`]（CUDA／Metal が経由する既定合成実装）
    /// 経由でも、出力バイトサイズが `isize::MAX` を超える巨大なバッチ
    /// 次元は実際の確保を試みずに `BackendError::ShapeMismatch` を
    /// 返す（`k = 0` の shape でテスト自体は実データを確保しない）。
    #[test]
    fn gemm_batched_huge_batch_output_bytes_overflow_is_shape_mismatch_not_panic() {
        let huge_batch = isize::MAX as usize / 4 + 1;
        let ops = NaiveMockOps;
        let a = Tensor::new(Vec::<f32>::new(), &[huge_batch, 1, 0]).unwrap();
        let b = Tensor::new(Vec::<f32>::new(), &[huge_batch, 0, 1]).unwrap();
        let err = ops.gemm_batched(&a, &b).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }
}
