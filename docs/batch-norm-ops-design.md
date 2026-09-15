# BatchNorm1d／2d の CPU 実装・設計記録（イシュー #1732・親 #1608）

## 0. 背景

`docs/compat-feature-gap.md` §2.7（対象 HEAD `097bff19`）が特定したギャップ:

- `nn.BatchNorm2d`: なし（`rmsnorm` はあるが統計対象軸・running stats が
  異なる）

本 issue は 3 バックエンド分解（#1732 CPU → #1735 CUDA → #1736 Metal）の
先頭であり、`BackendOps` 契約・shape 契約・縮約順序・VJP・`nn` 層（running
stats・train／eval）を確定する役割を持つ。CUDA／Metal は既定
`Unsupported`（ホスト参照実装へフォールバック）のまま兄弟 issue へ引き継ぐ。

本 issue は本クレート内で**初めて train／eval でモードにより挙動が変わる
層**を実装する（`docs/spec/04-requirements.md` REQ-9 2026-09-12 追記
Tier 1「Module の train／eval」〈#1758〉で用意した機構の最初の実利用者）。

## 1. レイアウト契約（`tensor-core::batch_norm_layout`）

`fandhe_ai_tensor_core::batch_norm_layout(shape) -> Result<(n, c, spatial),
ShapeError>`（`crates/tensor-core/src/ops_shape.rs`）が `row_norm_layout`
（#1596）の BatchNorm 版として `(n, c, spatial)` を導出する:

- rank 2 `[N, C]` → `spatial = 1`（`BatchNorm1d` の非空間入力）
- rank 3 `[N, C, L]` → `spatial = L`（`BatchNorm1d` の空間入力）
- rank 4 `[N, C, H, W]` → `spatial = H*W`（`BatchNorm2d`。`checked_numel` で
  `usize` 乗算オーバーフローを検出し `ShapeError::ElementCountOverflow`）
- rank 0・1・5 以上は `ShapeError::RankMismatch`

チャネル軸は常に dim 1（NCHW／NCL 固定。`docs/conv-ops-design.md` と同じ
レイアウト契約）。`batch_norm_layout` 自体は rank 2〜4 を一様に受理し、
`BatchNorm1d`（rank 2/3 限定）・`BatchNorm2d`（rank 4 限定）の追加検査は
`nn::batch_norm` 側（`BatchNormVars::forward`／`Module::forward_host`）が
行う。`M = n * spatial`（チャネルごとの縮約要素数）の導出は呼び出し側の
責務とする。

## 2. `BackendOps` 契約（非破壊拡張・既定 `Unsupported`）

`crates/tensor-core/src/backend_ops.rs` に 2 メソッドを追加（`layer_norm`
と同じデフォルトメソッド方式）:

```rust
fn batch_norm_train(
    &self,
    x: &Tensor<f32>,
    weight: Option<&Tensor<f32>>,
    bias: Option<&Tensor<f32>>,
    eps: f32,
) -> Result<BatchNormTrainOutput, BackendError>;   // 既定 Unsupported

fn batch_norm_infer(
    &self,
    x: &Tensor<f32>,
    mean: &Tensor<f32>,
    var: &Tensor<f32>,
    weight: Option<&Tensor<f32>>,
    bias: Option<&Tensor<f32>>,
    eps: f32,
) -> Result<Tensor<f32>, BackendError>;             // 既定 Unsupported
```

`BatchNormTrainOutput { output, batch_mean, batch_var }`（`LstmPointwiseOutput`
と同型の `pub struct`。`tensor-core::lib.rs` で再エクスポート）。

**2 メソッドに分ける理由**: 統計計算と適用を別呼び出しにすると `mean` を
一度 `f32` へ丸めてから `x̂` を計算することになり、LayerNorm で
codex-review が指摘した「`mean` の早期丸めが `x̂` を歪める」問題
（`crates/autodiff/src/eval.rs::row_ln_stats` doc 参照）を再発させる。
train 側は `output`（正規化・affine 適用後の値）に加え `batch_mean`／
`batch_var`（biased ÷M。shape `[c]`）を返し、呼び出し元（`nn::BatchNorm1d`／
`BatchNorm2d`）が running stats の更新に使う。

## 3. 数値契約（`.claude/rules/coding-rust.md`）

### 3.1 縮約順序

チャネル `c` ごとに `M = n*spatial` 要素を縮約する。局所添字 `i in 0..M` を
`batch = i / spatial`・`sp = i % spatial` へ分解し、実データ添字
`batch*(c_total*spatial) + c*spatial + sp` へ写像する（NCHW／NCL 平坦化
データに対するチャネル方向ストライドアクセス）。

**縮約順序は LayerNorm（#1596）が確立した「32 レーンのストライドアクセス +
offset 16→8→4→2→1 の butterfly」**（`warp_reduce_f64`。`crates/backend-cpu/
src/layer_norm.rs`・`crates/autodiff/src/eval.rs` の同名関数）**をチャネル
ごとの `M` 要素に対して適用する**（`i mod 32` でレーン割り当て）。これを
**CUDA（1 warp = 1 channel）／Metal（1 simdgroup = 1 channel）が再現すべき
契約**として本 doc に明記する（相殺入力での順序差が REQ-2 判定を超えた
LayerNorm の P1 事故〈PR #1671〉を再発させないため。単純逐次和は採らない）。

本実装（`crates/autodiff/src/eval.rs::channel_bn_stats`・`crates/backend-cpu/
src/batch_norm.rs::channel_stats`）はこの縮約順序を chunk 単位で逐語複製
している（2 系統。LayerNorm も同型の複製を持つ）。

### 3.2 統計計算

- 平均は `Σx / M`（直接除算。事前丸めした逆数との積ではない）
- 分散は二パス `Σ(x−μ)² / M`（`f64::mul_add`。`E[x²]−μ²` は使わない）
- `rstd = 1/sqrt(var+eps)` を `f64` で保持し、`x̂ = ((x as f64 − μ) * rstd)
  as f32` の 1 回だけ downcast する

### 3.3 affine 適用

`w`／`b` とも `Some` のときのみ `f32::mul_add`（FMA 契約統一）。片方
`None` はその演算をスキップする（LayerNorm の `layer_norm_row` と同じ 4
分岐。`eval::apply_affine`／`backend-cpu::batch_norm::apply_affine`）。

### 3.4 VJP

`x̂`・`dx̂ = dy·w`。

- **train モード**: `dx = rstd·(dx̂ − mean_M(dx̂) − x̂·mean_M(dx̂·x̂))`
  （`mean_M` はチャネル `c` の `M` 要素にわたる平均。LayerNorm の
  `layer_norm_vjp_rows` のチャネル版）。VJP は入力を `materialize_fallible`
  で実体化し `f64` 統計を §3.1 と同一順序で再計算する（forward 記録値や
  `f32` に丸めた統計を使わない。`Op::LayerNorm` の VJP と同じ理由）
- **eval モード**: `mean`／`var` が定数（`fixed_stats` payload）のため
  `dx = rstd·dx̂` のみ（補正項なし。`mean`／`var` 自体への逆伝播は不要
  ——固定統計ノードは `Op::BatchNorm` の入力グラフに現れない）

`dw_c = Σ_M dy·x̂`・`db_c = Σ_M dy`（train／eval 共通。要素を `f32` で確定
してから `f64` へ蓄積する縮約精度契約）。

実装: `crates/autodiff/src/grad.rs::batch_norm_vjp_channels`。

### 3.5 CPU bit 完全一致契約

CPU カーネル（`backend-cpu::batch_norm`）とホスト参照実装（`autodiff::
eval::batch_norm_train_channels`／`batch_norm_infer_channels`）は **bit
完全一致**する（`crates/backend-cpu/tests/batch_norm_parity.rs::
backend_ops_batch_norm_train_is_bit_identical_to_run_batch_norm_train_f32`
で固定）。REQ-2 複合判定は naive `f64` 参照実装との突合（`assert_parity`）・
将来の GPU 側との突合に用いる。

### 3.6 running stats 更新契約（`nn` 層）

`running = (1−momentum)·running + momentum·batch_stat`（`f64` で計算し 1 回
downcast）。`running_var` には **unbiased** 分散（`batch_var · M/(M−1)`）を
用いる一方、出力の正規化自体は biased 分散（PyTorch `torch.nn.functional.
batch_norm` 互換）。train モードの forward は呼ぶたび必ず running stats を
更新する（PyTorch の functional と同じ契約。eval モードは更新しない）。

train モードは `M <= 1` を `AutodiffError::InvalidArgument` で拒否する
（unbiased 分散の `M−1` 除算による 0 除算防止。PyTorch
`torch.nn.functional.batch_norm` の `_verify_batch_size` と同じ拒否）。

## 4. autodiff（`Op`・`Var`・VJP）

- `Op::BatchNorm { input, weight: Option<NodeId>, bias: Option<NodeId>, eps:
  f32, fixed_stats: Option<(Tensor<f32>, Tensor<f32>)> }`（`crates/autodiff/
  src/tape.rs`）。`fixed_stats = None` が train（バッチ統計）・
  `Some((mean, var))` が eval（固定統計）を表す。`Op::MaskedFill { mask }`
  と同型の「payload に非追跡 `Tensor<f32>` を保持する eager 実体化演算」
  （`mean`／`var` 自体は学習対象ではなく `nn::BatchNorm1d`／`BatchNorm2d`
  が外部で保持する running stats のスナップショットであるため、`NodeId`
  ではなく値そのものを持つ）
  - `Op::is_checkpoint_eligible` → `false`（`Op::LayerNorm` と同列。
    `docs/autodiff-checkpoint-design.md` §8 のスコープ外事項）
  - `Op::for_each_input` → `input`・`weight`（Some）・`bias`（Some）
- `Var::batch_norm(weight, bias, eps)`（train。`batch_norm_with_batch_stats`
  の `.0` を返す薄いラッパー）
- `Var::batch_norm_with_batch_stats(weight, bias, eps) ->
  Result<(Var, Tensor<f32>, Tensor<f32>), AutodiffError>`（train 本体。
  `(output, batch_mean, batch_var)` をタプルで返す——呼び出し元が running
  stats を更新するために必要。新規 `pub` 型を追加せず `Var`／`Tensor<f32>`
  のタプルで表現している）
- `Var::batch_norm_infer(weight, bias, running_mean, running_var, eps)`
  （eval。固定統計）
- 共通ヘルパー `grad::batch_norm_train_with_fallback`／
  `batch_norm_infer_with_fallback`（`conv2d_with_fallback` と同型:
  バックエンド → `Unsupported` のときのみホスト参照実装へ。`Var` メソッド
  と `Module::forward_host` の両方が同一ヘルパーを呼ぶことで tape 経路／
  tape 不要経路の bit 一致を構造的に担保する）

## 5. `nn` 層（`crates/autodiff/src/nn/batch_norm.rs`）

- `BatchNormCore`（クレート内公開の共通パラメータ本体。`BatchNorm1d`／
  `BatchNorm2d` が `core` フィールドとして保持する）: `weight`／`bias`
  （`Option<Tensor<f32>>`）・`running_mean`／`running_var`
  （`RefCell<Tensor<f32>>`。`Module::forward`／`forward_host` が `&self` の
  ため train モードでの更新に内部可変性が必要——`nn::optim::device_store`
  の `RefCell<Option<GradStaging>>` が先例）・`num_batches_tracked`
  （`Cell<u64>`）・`training: bool`（plain フィールド。`Module::
  set_training` は `&mut self`）
- `BatchNorm1d::new`／`without_affine`／`from_parameters`・`BatchNorm2d`
  も同型（`nn::LayerNorm` と同じコンストラクタ 3 種パターン）
- accessor（`weight()`／`bias()`／`running_mean()`／`running_var()`／
  `num_batches_tracked()`／`eps()`／`momentum()`）: `running_mean`／
  `running_var` は `Ref` の漏出を避けるため clone した `Tensor<f32>` を
  返す
- `bind(tape) -> BatchNormVars<'t, '_>`（2 lifetime: `'t` は tape の
  ライフタイム・`'a` は `&BatchNormCore` の借用ライフタイム）。
  `BatchNormVars::forward` が rank 検査
  （`BATCH_NORM_1D_RANKS=[2,3]`／`BATCH_NORM_2D_RANKS=[4]`）後、
  `self.core.training` に応じて train／eval へ委譲する
- `impl Module for BatchNorm1d`／`BatchNorm2d`（`crates/autodiff/src/nn/
  module.rs`）: `forward`／`forward_host`（共通ヘルパー
  `batch_norm_forward_host` に集約）・`set_training`／`training`（**本
  クレート内で初めてオーバーライドする**。`Module::set_training` trait
  doc の「今後 BatchNorm 等を追加する際は必ずオーバーライドすること」を
  実装する）・`named_parameters`（`weight`→`bias` の順。**running stats
  は buffer であり `named_parameters` に含めない**）

`momentum=None`（累積移動平均）・`track_running_stats=false` は本 issue
では非対応（§7「対象外」参照）。

## 6. facade

`crates/facade/src/**` へのコード変更なし。`Var::batch_norm*` は既存
`pub use fandhe_ai_autodiff::Var` 経由で到達する（LayerNorm・softmax と
同型）。`nn::BatchNorm1d`／`BatchNorm2d` は facade に `nn` 再エクスポート
が無いため `fandhe_ai_autodiff::nn` 経由（`nn::LayerNorm` と同じ扱い）。
`compat::Sequential::add_batch_norm*` は #1618 系（Keras 風層追加）の
スコープとし対象外（`add_layer_norm` と同じ判断。`docs/norm-ops-design.md`
§7）。

## 7. 対象外（CUDA／Metal 兄弟 issue への申し送り含む）

- **Metal（#1736）カーネル**: 縮約順序契約（§3.1）は本 issue が確定
  済み。`BackendOps::batch_norm_train`／`batch_norm_infer` は既定
  `Unsupported` のまま本 issue では変更しない（`Var::batch_norm*` は
  自動的にホスト参照実装へフォールバックするため既存の `#[ignore]`
  テストに影響しない）。**CUDA は #1735 で実装済み（§9 参照）**
- `momentum=None`（累積移動平均）
- `track_running_stats=false`
- `named_buffers`／state_dict 直列化（running stats を buffer として
  永続化する仕組み。#1616 の課題）
- `compat::Sequential::add_batch_norm*`（#1618 系）
- NEON ベクトル化・チャネル数が少ない場合の `M` 方向並列化（性能課題）
- rank 5（BatchNorm3d）・channels-last レイアウト

## 8. 実装記録

- `crates/tensor-core/src/ops_shape.rs::batch_norm_layout`
- `crates/tensor-core/src/backend_ops.rs::{BatchNormTrainOutput,
  BackendOps::batch_norm_train, BackendOps::batch_norm_infer}`
- `crates/autodiff/src/eval.rs::{channel_bn_stats, apply_affine,
  batch_norm_train_channels, batch_norm_infer_channels}`
- `crates/autodiff/src/tape.rs::Op::BatchNorm`
- `crates/autodiff/src/grad.rs::{batch_norm_vjp_channels,
  batch_norm_train_with_fallback, batch_norm_infer_with_fallback}`
- `crates/autodiff/src/var.rs::{Var::batch_norm, batch_norm_with_batch_stats,
  batch_norm_infer}`
- `crates/autodiff/src/nn/batch_norm.rs`（新設。`BatchNormCore`・
  `BatchNorm1d`・`BatchNorm2d`・`BatchNormVars`）
- `crates/autodiff/src/nn/module.rs::{impl Module for BatchNorm1d,
  impl Module for BatchNorm2d, batch_norm_forward_host}`
- `crates/backend-cpu/src/batch_norm.rs`（新設。`run_batch_norm_train_f32`・
  `run_batch_norm_infer_f32`）
- `crates/backend-cpu/src/ops.rs::{CpuBackendOps::batch_norm_train,
  CpuBackendOps::batch_norm_infer}`

テスト: `crates/tensor-core/src/{ops_shape,backend_ops}.rs`（クレート内・
既定 `Unsupported` ガード）・`crates/autodiff/src/{eval,grad,nn/batch_norm}.rs`
（クレート内。channel 統計手計算突合・数値微分突合・running stats 更新）・
`crates/autodiff/tests/{batch_norm,nn_module_mode}.rs`（統合。統計オラクル
突合〈`Var::mean_dims`／`Var::var`〉・`nn::Sequential` 経由のモード伝播・
`forward`／`forward_host` bit 一致）・`crates/backend-cpu/tests/
batch_norm_parity.rs`（REQ-2 突合・並列閾値跨ぎ・非 contiguous 入力）・
`crates/facade/tests/batch_norm_backend_parity.rs`（facade 公開面のみを
import する CPU parity・`#[ignore]` Metal／CUDA スキャフォールド）。

CUDA／Metal 実機は本 issue の対象外（`BackendOps` 既定 `Unsupported` に
よりホストフォールバックへ到達するため）。兄弟 #1735／#1736 が実機 parity
を担う。

## 9. CUDA 実装記録（#1735）

§3.1「CUDA（1 warp = 1 channel）が再現すべき契約」（`warp_reduce_f64`
butterfly 縮約・二パス分散・`double` アキュムレータ）を逐語実装した
NVRTC カーネル 2 本（train／infer）を追加し、`CudaBackendOps::
batch_norm_train`／`batch_norm_infer` を新設カーネル経由へオーバー
ライドした（`Var::batch_norm*` の判定規律は不変）。

### 9.1 設計判断

- **カーネル分割**: train は `1 CTA = 1 warp（32 レーン）= 1 channel`・
  `grid_dim = c`（`kernels_layer_norm.rs` と同じ 1 対 1 マッピング。
  persistent block は採用しない）。infer は統計を再計算しないため
  grid-stride の単純 elementwise（`block_dim = 256`）とし、train と
  異なるカーネル・異なる block 幅を用いる（`kernels_batch_norm.rs`
  冒頭コメント参照）
- **`i32` 上限超過**（`CudaError::BatchNormSizeLimitExceeded`）は
  `BackendError::Unsupported` へ写像し `fandhe_ai_autodiff::grad::
  batch_norm_train_with_fallback`／`batch_norm_infer_with_fallback`
  のホストフォールバックへ委ねる（`map_im2col_error`／
  `map_unique_error` と同じ設計判断。ハード fail にしない）
- **内部契約違反**（`CudaError::InvalidBatchNormShape`。`weight`／
  `bias`／`mean`／`var` の長さ不一致・`eps` 検査）は `ops.rs` 側で
  事前検査せず `CudaBatchNorm` 内部の `validate_batch_norm_launch` が
  唯一の検査点であるため、CPU 側 `CpuBackendOps::batch_norm_train`
  と同じ `BackendError::KernelLaunchFailed` へ写像する（判定迂回経路
  を作らない）
- **`w`／`b` が `None` の場合のダミーバッファ**: `layer_norm.rs` の
  Cursor Bugbot 指摘（predicated load による OOB 読み出し回避）と
  同じ理由で `c` 要素ゼロ初期化バッファを渡す
- **判定契約**: CUDA vs CPU は REQ-2 統一複合判定
  （`fandhe_ai_backend_cpu::parity::assert_parity`）を正式判定とする
  （bit 一致は主張・assert しない）

### 9.2 実装ファイル

- `crates/backend-cuda/src/kernels_batch_norm.rs`（新設。
  `BATCH_NORM_TRAIN_F32`／`BATCH_NORM_INFER_F32` の NVRTC ソース文字列）
- `crates/backend-cuda/src/batch_norm.rs`（新設。`CudaBatchNorm`・
  `validate_batch_norm_launch`・`run_batch_norm_train_f32`／
  `run_batch_norm_infer_f32`）
- `crates/backend-cuda/src/error.rs::{CudaError::InvalidBatchNormShape,
  CudaError::BatchNormSizeLimitExceeded}`
- `crates/backend-cuda/src/context_cache.rs::cached_batch_norm`
- `crates/backend-cuda/src/lib.rs`（`mod batch_norm;`・
  `mod kernels_batch_norm;`・`pub use batch_norm::CudaBatchNorm;`）
- `crates/backend-cuda/src/ops.rs::{map_batch_norm_error,
  CudaBackendOps::batch_norm_train, CudaBackendOps::batch_norm_infer}`

テスト: `crates/backend-cuda/src/batch_norm.rs`（クレート内。
`validate_batch_norm_launch` 単体テスト・GPU 不要）・
`crates/backend-cuda/tests/batch_norm_parity.rs`（環境適応スモーク・
実機必須の形状網羅〈rank 2／3／4・warp 幅端数・極値・NaN 伝播・
決定性・`CpuBackendOps` 直接突合〉）・`crates/facade/tests/
batch_norm_backend_parity.rs`（`cuda_batch_norm_train_forward_matches_cpu`・
`cuda_batch_norm_infer_forward_matches_cpu`・
`cuda_batch_norm_train_backward_matches_cpu`〈CUDA forward + ホスト
VJP の結線確認〉・`cuda_batch_norm_train_forward_rank4_matches_cpu`）。

**facade 新規公開面なし**（`crates/facade/src/**` は無変更。`Var::
batch_norm*` は既存再エクスポート経由）。

### 9.3 対象外（本 issue のスコープ外）

- CUDA persistent grid／occupancy 最適化・`M` 方向並列化・`float4`
  ベクトル化（性能課題。`c` が少なく `M` が巨大な形状では 1 warp = 1
  channel は並列度不足だが、LayerNorm CUDA と同じく「正しい新設」を
  優先する）
- GPU backward カーネル（VJP は #1732 でホスト側に 3 バックエンド
  共通実装済み。本 issue では forward カーネルのみを追加）
- デバイス常駐入出力（`DeviceBuffer` 経由の `batch_norm`）
- **GB10 実機実測**: 本エージェント実行環境に DGX Spark GB10 実機
  への到達手段がなく未実施のまま `docs/perf/logs/cuda-batch-norm-1735/`
  へ申し送る
