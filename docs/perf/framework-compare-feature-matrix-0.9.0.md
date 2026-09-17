# フレームワーク横並びスコアボード「役割・機能の対応表」v0.9.0 再判定

イシュー #1938。フレームワーク横並びスコアボード（claude.ai Artifact）に掲載されている
「役割・機能の対応表」（dtype／自動微分／演算の範囲／NN 層／最適化・学習ループ／
バックエンド・ハード／相互運用／事前学習済みモデル／推論・サービングの 9 行）を
crates.io `fandhe-ai =0.9.0` の公開面基準で再判定し、各行の判定根拠（facade 公開
API 名・実装イシュー番号、または残る穴の追跡先イシュー番号）を記録する。

## 1. 基準面の証明

判定基準は HEAD の実装状況ではなく、crates.io に公開済みの `fandhe-ai =0.9.0`
（= タグ `v0.9.0` 時点の `crates/facade`）の公開面とする。

```
$ git diff --stat v0.9.0..HEAD -- crates
（出力なし）
```

タグ `v0.9.0` から本 doc 作成時点の HEAD まで `crates/` 配下に差分は無い
（`v0.9.0` 以降の変更は `scripts/`・`docs/` のみ）。したがって HEAD の
`crates/facade/src/` をそのまま crates.io 0.9.0 の公開面として判定に用いる。

## 2. 判定規則

- **到達性**: `Var` の `pub fn` は `crates/facade/src/lib.rs` が `Var` を丸ごと
  `pub use` しているため facade 公開として扱う。`nn::*` の層構造体は
  `compat::Sequential::add_*`・`fandhe_ai::optim`・`fandhe_ai::data` 等
  facade から個別に到達できるもののみ公開扱いとし、到達手段がない場合は
  「リポ内」とする。非公開クレート（`onnx-interop` 等）限定の機能も「リポ内」
  （根拠: `crates/facade/tests/api_surface.rs` の
  `facade_does_not_depend_on_unpublished_onnx_interop`／
  `facade_sources_do_not_reference_onnx_interop` が facade の
  `onnx-interop` 非依存を機械的に固定している）。
- **判定値**: ある／部分的／リポ内／ない の 4 値。1 行内で項目により判定が
  割れる場合は行判定を「部分的」とし、根拠欄で内訳を分ける。
- **0.8.0 時点の判定列**: 親イシュー #1937 の対応表に記載された値を出典として
  用いる（0.8.0 時点のスコアボード原本は本セッションから参照不能のため）。

## 3. 対応表

| 行 | 0.8.0 時点（#1937 記載値） | 0.9.0 判定 | 根拠（ある: API 名 + 実装イシュー／リポ内: イシュー／残る穴: 子イシュー） |
|---|---|---|---|
| dtype | 部分的（f32 固定 dispatch＋一部拡張） | 部分的 | ある: `Var::cast`・`CastDType`／`CastElement`（facade 再エクスポート済み。`crates/facade/src/lib.rs:192`）による f32⇄f64/i32/i64/bool 変換（#1750／#1751）。リポ内: `TypedOps<f64/f16/bf16>`（`backend-*::typed_*`）は facade 非公開（#1648〜#1651・#1697〜#1699・#1703〜#1706）。残る穴: 型付き dispatch の facade 公開自体は #1939（親 #1937 Phase 4）。出典: `docs/backend-dtype-dispatch-design.md`・`docs/tensor-core-cast-design.md` |
| 自動微分 | 部分的（VJP 一般構成・高階微分/custom Function 除外） | 部分的 | ある: `Tape::var_no_grad`／`Var::detach`（#1748）・`Tape::backward_accumulate`（`retain_graph` 相当。#1749）・`Var::checkpoint_from`（activation checkpointing。#1624）。リポ内/対象外: 高階微分（grad of grad）は段階 0 で非対応確定（`docs/autodiff-higher-order-grad-decision.md`）、custom autograd Function も段階 0（`docs/autodiff-custom-function-decision.md`）。残る穴: 高階微分は #1940（子 #1941〜#1943）、custom Function は #1944（子 #1945・#1946） |
| 演算の範囲 | 部分的（多数の欠落: softmax/GELU/Conv/Embedding 等） | ある（ほぼ網羅） | ある: `Var` 再エクスポート経由で softmax／log_softmax／GELU 系／Conv1d・2d／pooling（max/avg/adaptive）／Embedding／MultiheadAttention（`scaled_dot_product_attention`）／einsum／where・masked_fill／gather・scatter・scatter_add・index_select／sort・argsort・topk／cumsum・cumprod／one_hot／interpolate／BCE・NLL・KLDiv・Huber・SmoothL1 損失／pad／unique／min・argmax・argmin 等（各実装イシューは `docs/compat-api-scope.md` §1.2／§1.3 を参照）。残る穴（未実装または facade 非公開）: 詳細は #1947（子 #1948〜#1953）。出典: `docs/compat-api-scope.md`・`docs/compat-feature-gap.md` 追補 |
| NN 層 | 部分的（Linear/ReLU 等の基本層のみ） | 部分的 | ある: `compat::Sequential::add_linear`／`add_relu`／`add_sigmoid`／`add_tanh`／`add_silu`／`add_hardswish`／`add_leaky_relu`／`add_elu`／`add_dropout`／`add_conv2d`／`add_conv1d`／`add_layer_norm`／`add_rms_norm`／`add_batch_norm1d`／`add_batch_norm2d`／`add_embedding`／`add_multihead_attention`（`crates/facade/src/compat/sequential.rs` 実測。実装イシュー #1595・#1603・#1608・#1613・#1639・#1640・#1760・#1764／#1769）。リポ内: RNN／LSTM／GRU（`fandhe_ai_autodiff::nn::rnn`。#1647）は facade から到達する `add_*`／再エクスポートが無く未公開。Pooling 層（`nn::MaxPool2d` 等。#1727〜#1730）も `compat::Sequential` に `add_*` が無く未公開（`Var::max_pool2d` 等の素の演算は facade 到達可能）。残る穴: #1954（子 #1955〜#1957） |
| 最適化・学習ループ | 部分的（Adam 系のみ・LR scheduler/AMP 欠落） | ある（主要項目網羅） | ある: `fandhe_ai::optim` 再エクスポートに `Sgd`／`Adam`／`AdamW`／`RmsProp`／`Adagrad`／`Lamb`／`GradScaler`（AMP）／`ConstantLr`／`StepLr`／`CosineAnnealingLr`／`ExponentialLr`／`LinearWarmupLr`／`ReduceLrOnPlateau`／`OneCycleLr`／`clip_grad_norm`／`clip_grad_value` が揃う（`crates/facade/src/optim.rs` 実測）。`compat::{Loss, Optimizer, FitConfig, History}`・`Sequential::{compile, fit, fit_with_callbacks, evaluate}`・`Callback`（EarlyStopping／ModelCheckpoint）も実装済み（#1761・#1763）。残る穴（metrics・`validation_data` の高度な連携等の細部）: #1958（子 #1959〜#1961） |
| バックエンド・ハード | 部分的（CPU/CUDA/Metal のみ・分散/AMD 対象外） | 部分的 | ある: `Device`（facade 再エクスポート。`crates/facade/src/lib.rs:144`）・`tape`／`tape_for(Device)`（同 `:395,407`）により CPU/CUDA/Metal 3 バックエンドをユーザーが明示選択できる（`Device::Cpu`／`Device::Cuda(ordinal)`／`Device::Metal`）。リポ内（対象外方針）: 複数 GPU（DataParallel／DDP）・量子化（int8）・AMD(ROCm) 追加は段階 0（現時点非対応）で確定済み（`docs/facade-multi-gpu-ddp-decision.md`・`docs/backend-int8-quantization-decision.md`・`docs/backend-abstraction-amd-readiness-decision.md`）。facade に `nccl`／`ddp`／`all_reduce` 相当の公開面は無し（grep で不在確認）。残る穴: #1964・#1965 |
| 相互運用 | ない（ONNX/safetensors 未公開） | リポ内（承認待ち） | ONNX import（`onnx-interop::onnx::interp`）・export（`onnx-interop::onnx::export`。#1772〜#1774 で本体実装済み）・safetensors save/load（`onnx-interop::st_load`／`st_save`）はいずれも facade 公開の設計案は整理済みだが publish 承認未取得のため段階 0 のまま（`docs/facade-onnx-import-exposure-decision.md`・`docs/facade-onnx-export-exposure-decision.md`・`docs/facade-safetensors-exposure-decision.md`）。`crates/facade/tests/api_surface.rs` の guard テストが facade の `onnx-interop` 非依存を機械的に固定（実測: `facade/Cargo.toml` に onnx 依存なし）。`state_dict`／`load_state_dict`（safetensors ではなく `HashMap<String, Tensor<f32>>` 形式）は facade 公開済み（#1752）。残る穴: #1963 |
| 事前学習済みモデル | ない | リポ内（承認待ち） | ONNX import が非公開のため学習済みモデルの取り込み手段が facade に無い（相互運用行と同一の段階 0 決定に従属）。残る穴: #1963・#1965 |
| 推論・サービング | 部分的（forward のみ・チェーン単一同期化前） | ある | ある: `Sequential::predict`（tape 不要経路）／`Sequential::predict_resident`（`DeviceParamStore::predict_device_chain` 優先経路。単一同期化。CPU/CUDA/Metal 3 バックエンドとも実装・実機実測済み: #1580 M4 Max・#1689 GB10）。汎用 JIT／グラフ最適化拡張（`torch.compile` 相当）は非目標と整理済み（`docs/autodiff-graph-optimization-scope-decision.md`）。残る穴（サービング周辺の細部）: #1962。出典: `docs/inference-chain-single-sync-design.md` |

## 4. 親表 #1937 との差分（grep で確定した事実のみ）

- `CastDType`／`CastElement`・`Var::cast` は既に facade 再エクスポート済み
  （`crates/facade/src/lib.rs:192`）。親 #1937 の記述が「cast 非公開」の
  ままであれば本 doc の実測が正。
- `compat::Sequential::add_conv2d`／`add_conv1d` は既に実装・公開済み
  （`crates/facade/src/compat/sequential.rs:244,273`）。
- `fandhe_ai::optim::OneCycleLr` は既に facade 再エクスポート済み
  （`crates/facade/src/optim.rs:217`）。LR scheduler 群（Cosine／Exponential／
  LinearWarmup／Plateau／OneCycle）は 0.9.0 時点で全て揃っている。
- RNN／LSTM／GRU・Pooling 層（`nn::MaxPool2d` 等）は本体クレートに実装済み
  だが、`compat::Sequential` に対応する `add_*` が無いため facade からは
  依然到達不能（「NN 層」行が「部分的」のまま残る理由）。

## 5. スコアボード（Artifact）反映状況

本イシューの受入基準 3（スコアボード生成スクリプトの対応表節を更新し再公開する）
について:

- スコアボード生成スクリプトの実体は本リポジトリ内に存在しない
  （`git ls-files | grep -i scoreboard` 等で未検出）。
- 0.9.0 スコアボード Artifact は本実装セッションのアカウントから
  `Artifact` action=list（scope=all）で確認したところ一件も見つからず、
  読み取り・再公開のいずれも実行できない（private 既定のまま他セッションが
  所有していると推定される）。
- 上記の理由により、本 PR では新規 Artifact の作成・既存 Artifact への
  in-place 反映は行わない（出典 URL の分裂を避けるため）。§3 の対応表を
  確定内容とし、Artifact 所有セッションでの反映をイシュー #1938 のコメント
  で申し送る。

## 6. 変更していないもの

`crates/` 全体・`docs/spec/`・tolerance／baseline・依存関係・CI 設定・
`docs/compat-api-scope.md`／`docs/compat-feature-gap.md` は変更していない。
