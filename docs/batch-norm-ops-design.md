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

- **CUDA（#1735）／Metal（#1736）カーネル**: 縮約順序契約（§3.1）は本
  issue が確定済み。両バックエンドとも `BackendOps::batch_norm_train`／
  `batch_norm_infer` は既定 `Unsupported` のまま本 issue では変更しない
  （`Var::batch_norm*` は自動的にホスト参照実装へフォールバックするため
  既存の `#[ignore]` テストに影響しない）。**CUDA は #1735、Metal は
  #1736 でそれぞれ実装済み（§9 参照）**
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

## 9. Metal 実装記録（#1736）

### 9.1 数値方式: soft-f64（issue 題名の Neumaier＋scale/ssq は不採用）

issue 題名は「Neumaier 補償和＋scale/ssq 方式」を指定していたが、実装は
**soft-f64（IEEE 754 binary64 の 64bit 整数ソフトウェアエミュレーション）
方式**を採用した。理由: 同一の計算列（mean → 二パス var → rstd →
affine）を持つ Metal LayerNorm（#1596）が当初 Neumaier＋scale/ssq で
実装されていたが、PR #1671 codex-review 指摘の 2 系統の反例（行スケール
除算での subnormal FTZ 消失・`rstd` の `f32` 丸めが affine の相殺で
増幅される問題）により REQ-2 統一複合判定を満たせず soft-f64 へ全面
置換された経緯がある（`crates/backend-metal/src/shaders/layer_norm.metal`
冒頭コメント「数値方式」参照）。BatchNorm の統計計算は LayerNorm と
同型の反例を抱えるため、実装時点から soft-f64 方式を採用した。
`.claude/rules/coding-rust.md`「正規化統計の二乗和」節は Neumaier＋
scale/ssq を Metal の f64 相当実装形として挙げるが禁止規定ではなく、
soft-f64 は「f64 相当の精度を保つ」契約をより強く満たす。

### 9.2 カーネル構成

`crates/backend-metal/src/shaders/batch_norm.metal`（新設）に 2 カーネル
を実装:

- **`batch_norm_train_f32`**: 1 threadgroup = 1 simdgroup（32 レーン）
  = 1 チャネル、persistent threadgroup 方式
  （`for (ch = tg_id; ch < c; ch += grid_size)`。`layer_norm.metal` の
  行ループをチャネルへ置換）。パス 1（平均。soft-f64 総和 + 5 段
  butterfly reduction + `bn_f64_div`）→ パス 2（二パス分散）→ パス 3
  （書き出し。round-to-odd 経由 affine）の 3 段走査。`mean_out`／
  `var_out`（`BatchNormTrainOutput::batch_mean`／`batch_var` 用）は
  lane 0 のみが書き出す
- **`batch_norm_infer_f32`**: 統計を再計算しないため grid-stride 不要の
  単純 elementwise（`if (gid >= numel) return;` の手動境界検査）
- soft-f64 プリミティブ（`bn_f64_*`）は `layer_norm.metal::ln_f64_*` の
  接頭辞置換による逐語複製（MSL は翻訳単位を共有できないため意図的な
  重複。`tests/batch_norm_source_evidence.rs::
  bn_f64_primitives_match_ln_f64_primitives_verbatim_modulo_prefix` が
  ドリフトを検出する）
- **`M`（チャネルごとの縮約要素数 `n*spatial`）の f64 表現**は
  `layer_norm.metal` の `validate_hidden_exact_f32`（`2^24` 超を起動前
  拒否）方式を踏襲しない——BatchNorm2d の `M` は実用形状で容易に
  `2^24` を超えるため。代わりにホスト側（`batch_norm.rs::
  run_batch_norm_train_f32`）が `(m as f64).to_bits()` を計算し、上位
  ／下位 32bit の 2 引数（`m_f64_hi`／`m_f64_lo`）としてカーネルへ渡す

### 9.3 判定契約

CPU との bit 一致は主張しない。`batch_mean` は CPU（`warp_reduce_f64`
逐次蓄積 → butterfly → 正しく丸めた除算）と同じ縮約順序・演算のため
bit 一致する見込みだが、CPU の分散は `f64` FMA・ハードウェア `sqrt` で
あるのに対し soft-f64 は `mul`+`add`（二重丸め）・Newton-Raphson
`rsqrt` のため `var`／`rstd`／出力は bit 一致しない。よって本カーネル
全体は REQ-2 統一複合判定（`fandhe_ai_backend_cpu::parity::
assert_parity`）で CPU 参照実装と検証する。

### 9.4 実装ファイル

- `crates/backend-metal/src/shaders/batch_norm.metal`（新設）
- `crates/backend-metal/src/batch_norm_model.rs`（新設・`cfg` なし。
  `validate_batch_norm_launch`・`channel_index`・`m_f64_bits`・
  ホスト側 soft-f64 逐語モデル `batch_norm_train_host_model`／
  `batch_norm_infer_host_model`。`crate::soft_f64` の既存プリミティブ
  を呼ぶのみで新規 soft-f64 実装は持たない）
- `crates/backend-metal/src/batch_norm.rs`（新設・`cfg(macos)`。
  `MetalBatchNorm::new`／`run_batch_norm_train_f32`／
  `run_batch_norm_infer_f32`）
- `crates/backend-metal/src/error.rs::{MetalError::
  InvalidBatchNormShape, MetalError::BatchNormSizeLimitExceeded}`
- `crates/backend-metal/src/context_cache.rs::cached_batch_norm`
- `crates/backend-metal/src/ops.rs::{map_batch_norm_error,
  MetalBackendOps::batch_norm_train, MetalBackendOps::batch_norm_infer}`
- `crates/backend-metal/src/lib.rs`（`pub mod batch_norm_model;`・
  `pub mod batch_norm;`〈cfg macos〉・`pub use batch_norm::
  MetalBatchNorm;`）

### 9.5 エラー写像

`MetalError::BatchNormSizeLimitExceeded`（`n`／`c`／`spatial`／`m`／
`numel` のいずれかがカーネル引数 `uint`〈`u32::MAX`〉上限を超過）
**のみ** `BackendError::Unsupported` へ写像しホストフォールバックへ
委ねる。`MetalError::InvalidBatchNormShape`（`weight`／`bias`／
`mean`／`var` の長さ不一致・`eps` 検査等）は CPU 側
`CpuBackendOps::batch_norm_train` と同じ `KernelLaunchFailed` へ揃える
（`ops.rs::map_batch_norm_error`。`map_im2col_error` と同じ設計判断・
`backend-cuda::ops::map_batch_norm_error` と対になる）。

### 9.6 テスト

- `crates/backend-metal/src/batch_norm_model.rs`（クレート内単体
  テスト。Linux 実行可能。検証関数・`channel_index`・`m_f64_bits`・
  ホスト soft-f64 モデルと `fandhe_ai_backend_cpu::
  run_batch_norm_train_f32`／`run_batch_norm_infer_f32` の REQ-2 突合
  を含む 13 件。全 green——soft-f64 モデルの正しさをデバイス非依存で
  裏付ける）
- `crates/backend-metal/tests/batch_norm_source_evidence.rs`
  （Linux 実行可能。11 件。`BN_IDX` の `ulong` 演算・infer の手動
  境界検査・train の persistent threadgroup ループ・5 段 butterfly・
  soft-f64 widen/add/div の使用・`M` の厳密ビット渡し・round-to-odd
  affine・`threadgroup_barrier` 不使用・predicated select・
  `bn_f64_*` ↔ `ln_f64_*` ドリフトガードを固定）
- `crates/backend-metal/tests/batch_norm_parity.rs`（macOS 限定・
  `#[ignore]`。形状網羅〈rank 2／3／4 相当・`M` が 32 の倍数でない・
  `M=1`〜大形状・affine 4 分岐〉の `f64` naive 参照との突合・極端な
  値〈`1e18`。内部 soft-f64 計算自体は `2e20` 級でも有限のまま完結
  するが、`var` を最終的に `f32` へ narrow する時点で正当に `+inf`
  になりうるため、本テストの意図〈NaN／中間 overflow の不在確認〉に
  沿う規模へ調整済み。codex-review 指摘〉での有限性・NaN の該当
  チャネル限定伝播・run-to-run 決定性・`MetalBackendOps` vs
  `CpuBackendOps` 直接突合・非 contiguous 入力・weight 長さ不一致
  拒否・空軸早期 return）
- `crates/facade/tests/batch_norm_backend_parity.rs`（既存の
  `metal_batch_norm_train_forward_matches_cpu` のみ。当初は
  `metal_batch_norm_infer_forward_matches_cpu`／
  `metal_batch_norm_train_backward_matches_cpu`／
  `metal_batch_norm_train_forward_rank4_matches_cpu` も本 issue で
  追加したと記載していたが、実際にこの 3 件は同ファイル内の
  `cuda_batch_norm_infer_forward_matches_cpu`／
  `cuda_batch_norm_train_backward_matches_cpu`／
  `cuda_batch_norm_train_forward_rank4_matches_cpu`（CUDA 版。
  兄弟 issue #1735 で追加）であり、Metal 版の対応テストは存在しない
  ——記載と実装の不一致だったため訂正する。codex-review 指摘。Metal
  側の infer／backward／rank4 テスト追加は対象外のまま §9.7 へ
  引き継ぐ）

facade 新規公開面なし（`crates/facade/src/**` 無変更。`Var::
batch_norm*` は既存 `pub use Var` 経由）。M4 Max 実機実測は本エージェント
実行環境に Apple Silicon 実機がないため未実施のまま
`docs/perf/logs/metal-batch-norm-1736/README.md` へ申し送る。

### 9.6a M4 Max 実機実測（2026-09-16）

`docs/perf/logs/metal-batch-norm-1736/README.md` の手順で実測した（origin/main
`3e43bbd0`・共有負荷下・ログは同ディレクトリ `batch_norm_parity.log`・
`batch_norm_backend_parity.log`。`make test-ignored-metal` 相当の非後退確認は
`docs/perf/logs/metal-realdevice-phase2-2026-09-16/make-test-ignored-metal-nofailfast.log`）。

- `crates/backend-metal/tests/batch_norm_parity.rs`（`--ignored`）: **9 pass / 0 fail**
  （`batch_norm_train_matches_f64_reference_across_shapes_and_affine_combinations`・
  `batch_norm_infer_matches_f64_reference`・`metal_backend_ops_batch_norm_train_matches_cpu_backend_ops_req2`・
  `batch_norm_train_is_deterministic_across_runs`〈run-to-run bit 一致〉・NaN 伝播・
  極端値・非 contiguous 入力・空軸・重み長不一致の各テスト）
- `crates/facade/tests/batch_norm_backend_parity.rs::metal_batch_norm_train_forward_matches_cpu`:
  **1 pass / 0 fail**（facade 経路の REQ-2 統一複合判定）
- 既存 `#[ignore]` 群（`layer_norm_parity` 18 pass・`rmsnorm_parity` 9 pass・
  `softmax_parity` 7 pass 等）は非後退

判定: PASS（§9.3 判定契約どおり REQ-2 統一複合判定で fail 0 件・決定性成立。
tolerance／baseline 変更なし）。

### 9.7 対象外（本 issue でも変更しない）

§7 と同じ（GPU backward・性能最適化・`momentum=None`・
`track_running_stats=false`・rank 5・channels-last 等）。加えて
デバイス常駐入出力（`DeviceBuffer` 経由の `batch_norm`）も対象外。
`crates/facade/tests/batch_norm_backend_parity.rs` への Metal 版
`metal_batch_norm_infer_forward_matches_cpu`／
`metal_batch_norm_train_backward_matches_cpu`／
`metal_batch_norm_train_forward_rank4_matches_cpu` の追加（§9.6
訂正参照）も本 issue のスコープ外のまま後続 issue へ引き継ぐ。
## 10. CUDA 実装記録（#1735）

§3.1「CUDA（1 warp = 1 channel）が再現すべき契約」（`warp_reduce_f64`
butterfly 縮約・二パス分散・`double` アキュムレータ）を逐語実装した
NVRTC カーネル 2 本（train／infer）を追加し、`CudaBackendOps::
batch_norm_train`／`batch_norm_infer` を新設カーネル経由へオーバー
ライドした（`Var::batch_norm*` の判定規律は不変）。

### 10.1 設計判断

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

### 10.2 実装ファイル

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

### 10.3 対象外（本 issue のスコープ外）

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
