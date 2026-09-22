# `fit()` の分類 metrics 対応（accuracy・precision・recall・F1・confusion matrix）の設計判断記録

イシュー #2072（親 #2059。ルート #2058）。`docs/compat-callbacks-design.md`
§8・`docs/compat-fit-evaluate-design.md` が「引き続き対象外」としていた
metrics（accuracy 等）を実装した。

## 1. 目的・スコープ

`compat::Sequential::fit_with_callbacks`（#1763）は loss（`History::loss`／
`val_loss`）と lr しか記録せず、分類性能の多元的評価（accuracy・
precision・recall・F1・confusion matrix）が存在しなかった。本イシューは
これを Keras `fit(..., metrics=[...])` 相当として埋め、
`ModelCheckpoint`／`EarlyStopping`／`LrSchedule` が metrics を監視でき
るようにする。

### 対象外（Issue 記載 + 本実装で確定した追加分）

- カスタム metrics（trait object 拡張）・TensorBoard／W&B 統合・
  threshold 調整による最適化
- 学習セット側の metrics（Keras の `accuracy`。validation のみ対象）
- binary／micro／weighted 平均（macro 平均のみ実装）・`f32` target
  （`Loss::Mse`・sigmoid 出力）に対する閾値ベースの 2 値 metrics
- `evaluate_with_metrics`（受入基準に列挙されていない公開面のため
  追加しない。フォローアップ候補）
- `MonitorMode` の自動推定（Keras `mode='auto'`）

## 2. 公開 API

```rust
pub enum Metrics { Accuracy, Precision, Recall, F1, ConfusionMatrix } // #[non_exhaustive]
pub struct MetricsResult {                                            // #[non_exhaustive]
    pub num_classes: usize,
    pub accuracy: Option<f32>,
    pub precision: Option<f32>,
    pub recall: Option<f32>,
    pub f1: Option<f32>,
    pub confusion_matrix: Option<Vec<u64>>,
}
impl MetricsResult {
    pub fn compute(metrics: &[Metrics], logits: &Tensor<f32>, target: &Tensor<i32>)
        -> Result<MetricsResult, AutodiffError>;
}

impl Sequential {
    pub fn fit_with_metrics<T: FitTarget>(
        &mut self, x, y, config, validation, callbacks, metrics: &[Metrics],
    ) -> Result<History, AutodiffError>;
}

pub enum Monitor { Loss, ValLoss, ValMetric(Metrics) } // #[non_exhaustive]（既存 enum への追加）

pub struct History {
    pub loss: Vec<f32>, pub val_loss: Vec<f32>, pub lr: Vec<f32>,
    pub val_metrics: Vec<MetricsResult>,                            // 新設
}
```

facade 新規公開面はこの 4 件（`Metrics`・`MetricsResult`・
`Sequential::fit_with_metrics`・`Monitor::ValMetric`）に限る
（`History::val_metrics` は既存 `#[non_exhaustive]` struct へのフィール
ド追加のため非破壊）。`crates/facade/tests/api_surface.rs::
metrics_types_are_reachable_via_facade_only` で機械固定した。

## 3. `fit_with_metrics` を追加入口にした理由（既存シグネチャ非変更）

`fit_with_callbacks`（7 引数）は crates.io `fandhe-ai =0.9.0` として
出荷済みの公開 API のため、破壊的変更（引数追加）を避け、代わりに
`fit_with_metrics`（8 引数。`metrics: &[Metrics]` を追加）という新規
公開メソッドを設けた。`fit_with_callbacks` は従来どおり内部で
`fit_with_callbacks_named(..., &[])` へ委譲するため、両者の演算列・
`History` は完全に同一のまま不変（`fit_with_metrics_empty_matches_
fit_with_callbacks_bit_exact` で検証）。

内部の非公開ディスパッチ関数（`fit_with_callbacks_named`／`run_fit`）
は `metrics: &[Metrics]` を追加してどちらの公開入口からも共有する。
引数が 8〜9 個になり `clippy::too_many_arguments`（既定閾値 7）に抵触
するため `#[allow(clippy::too_many_arguments)]` を維持する——`run_fit`
が既に同じ理由でこの allow を持っており（呼び出し元は
`fit_with_callbacks_named` の 1 箇所のみ）、公開 API の破壊よりも
私的関数の引数個数超過を許容するほうが安全側と判断した。

## 4. `nn::Metrics` ではなく `compat::Metrics` に配置した理由

Issue 記載の想定パスは `nn::Metrics` だったが、以下の理由で
`compat::Metrics`（`crates/facade/src/compat/metrics.rs`）へ配置した:

- `crates/facade/src/nn/mod.rs` のモジュール doc が `nn` を rnn（`Rnn`／
  `Lstm`／`Gru` 等の純再エクスポート）限定の契約として固定している
  （`docs/compat-api-scope.md` §5「適用記録（経路 2。イシュー #1955）」）
- 関連する公開面（`Loss`・`History`・`Monitor`・`Callback`）はいずれも
  既に `compat` に配置されている——metrics も同じ利用者（`Sequential::
  fit_with_metrics` の呼び出し元）が同じ import 文から到達できる方が
  一貫する

## 5. 数値定義（macro 平均・zero-division・`f64` 導出）

`crates/facade/src/compat/metrics.rs::ConfusionAccumulator` が `u64` の
混同行列（行優先 `[C*C]`。`row` = 正解クラス・`col` = 予測クラス）を
バッチ横断で蓄積し、`finish()` で以下を導出する（いずれも `f64` で
計算し最後に 1 回だけ `f32` へ downcast）:

- **accuracy** = `trace(confusion) / total`
- **precision／recall（macro 平均）**: クラスごとの `tp/(tp+fp)`／
  `tp/(tp+fn)`（分母 0 は sklearn `zero_division=0` と同じく 0 として
  寄与）を算術平均する
- **F1（macro F1）**: クラスごとの `2·P_c·R_c/(P_c+R_c)`（分母 0 は 0）
  を求めてから算術平均する——**macro precision／recall の調和平均では
  ない**（両者は一般に異なる値になる。`macro_f1_is_mean_of_per_class_
  f1_not_harmonic_mean_of_macro_pr` テストで判別）
- **confusion_matrix**: 蓄積した `Vec<u64>` をそのまま返す（要求した
  場合のみ）

予測クラスは常に `Var::argmax(Some(1))`（非微分・タイは先頭添字・NaN
無視。イシュー #1720）で求め、独自の tie-break ロジックは持たない。

## 6. バックエンド非依存性（受入基準 F）と CPU／CUDA／Metal 契約

`MetricsResult::compute`／`ConfusionAccumulator` が扱う入力は予測クラス・
正解クラスという**整数添字**のみであり、算術はホスト側 `f64`（最後に
1 回だけ `f32` へ downcast）で行う。バックエンド依存性は forward 出力
（`logits`。`Var::argmax` の入力）のみに閉じるため、metrics 算術自体は
CPU／CUDA／Metal で **bit 同一**になる契約——`.claude/rules/coding-
rust.md` REQ-2 の複合判定（相対誤差／絶対誤差）ではなく `assert_eq!`
契約である。`crates/facade/tests/compat_sequential_metrics_backend_
parity.rs`（`#[ignore]`）がこの契約を検証する。

**事前登録判定規則**（実機実測で不一致が出た場合の切り分け手順）:
logits の近接タイ（parity レベルの forward 差で `Var::argmax` の結果が
反転する）が原因である可能性が高い。metrics 算術自体はバックエンド
非依存のため、原因が argmax 反転であることを確認できた場合でも、判定
規則・tolerance・baseline は変更しない。

## 7. `fit` の演算列への非侵襲性（loss 系列 bit 不変）

`run_evaluate` は `run_evaluate_with_metrics(x, y, batch_size, loss,
metrics, method) -> Result<(f32, Option<MetricsResult>), AutodiffError>`
へ拡張し、既存 `run_evaluate`（`evaluate()` 用）は `&[]` で委譲する薄い
関数へ置き換えた。loss の演算列（`forward → T::loss_for →
to_tensor().get(&[])`）は `metrics` の有無に関わらず完全に不変——
`pred.argmax(Some(1))` の追加計算は `metrics` が非空のときのみ、loss
計算の**後**に行う（`metrics_computation_does_not_perturb_loss_or_
params` で `loss`／`val_loss`／`lr`／パラメータの bit 完全一致を検証）。

`num_classes` は `Var::shape`（`pub(crate)`。facade から到達不可）では
なく `pred.to_tensor().shape()`（公開 API。`loss_var` 計算で `pred` は
既に実体化済みのため追加の演算コストはメモリコピー相当のみ）で読む。

## 8. `Monitor::ValMetric` の fail-closed 事前検査

`callbacks` が `Monitor::ValMetric(m)` を監視する場合、`fit_with_
callbacks_named` の事前検査（train／eval モード変更前）で以下を拒否
する——`Monitor::value_at` が黙って `None` を返し callback がサイレント
にスキップされる穴を塞ぐため:

1. `m == Metrics::ConfusionMatrix`（非スカラーのため監視値として
   定義できない）
2. `!metrics.contains(&m)`（`fit_with_metrics` に渡した `metrics`
   スライスに `m` が含まれない）

`Callback::monitor(&self) -> Option<Monitor>` を新設し（`EarlyStopping`／
`ModelCheckpoint`／`LrSchedule` それぞれの非公開 `monitor` フィールドへ
同一モジュール内から直接アクセスする薄いアクセサ）、上記検査を
`callbacks.rs` の外（`training.rs`）から行えるようにした。

`MonitorMode` の既定は `Min`（loss 系向け）のまま——accuracy 等
（大きいほど良い指標）を `ValMetric` で監視する場合は呼び出し側が
`.mode(MonitorMode::Max)` を明示する必要がある（自動推定はしない）。

## 9. 正しさの検証

- `crates/facade/src/compat/metrics.rs`（unit tests。9 件）: accuracy／
  macro precision・recall／macro F1 と調和平均の判別／confusion matrix
  のレイアウト・選択的計算／fail-closed 5 パターン／`Var::argmax` 経由
  の end-to-end
- `crates/facade/tests/compat_sequential_metrics.rs`（統合テスト。9 件）:
  - `fit_with_metrics_empty_matches_fit_with_callbacks_bit_exact`:
    `metrics = &[]` が `fit_with_callbacks` と `History`・パラメータとも
    bit 完全一致
  - `metrics_computation_does_not_perturb_loss_or_params`: metrics
    あり／なしで `loss`／`val_loss`／`lr`／パラメータが bit 完全一致
  - `val_metrics_matches_independent_predict_and_compute_per_epoch`:
    `val_metrics.len() == epochs`・各要素が「1 epoch ずつ `fit_with_
    metrics` を回し、eval モードの `predict` + 独立の `MetricsResult::
    compute`」と完全一致（`fit(1)+fit(1) == fit(2)` 契約を利用した
    非トートロジー検証）
  - `model_checkpoint_tracks_best_val_accuracy_with_max_mode`:
    `ModelCheckpoint(Monitor::ValMetric(Accuracy), MonitorMode::Max)`
    が accuracy 最大 epoch を追跡
  - `early_stopping_and_lr_schedule_are_driven_by_val_metric`:
    `EarlyStopping`／`LrSchedule` も `Monitor::ValMetric` で駆動できる
  - fail-closed 4 パターン: `metrics` 非空 × `validation=None`／
    `Loss::Mse`（`Tensor<f32>` target）／`ValMetric(m)` で `m ∉
    metrics`／`ValMetric(ConfusionMatrix)`
- `crates/facade/tests/compat_sequential_metrics_backend_parity.rs`
  （`#[ignore]`）: CUDA／Metal 実機 bit 同一検証（§6 参照。未実測のまま
  `docs/perf/logs/compat-sequential-metrics-2072/` へ申し送り）
- `crates/facade/tests/api_surface.rs::metrics_types_are_reachable_via_
  facade_only`: 公開面が facade のみ import で到達可能なことを固定

## 10. セキュリティ考慮（OWASP Top 10）

- **A03 インジェクション／入力検証**: `MetricsResult::compute` と
  validation 経路で logits rank・target shape・添字範囲・バッチ間 `C`
  一致を事前検証し `InvalidArgument` で fail-closed。`Vec` 確保は
  `try_reserve_exact`・`C*C` は `checked_mul`（巨大クラス数での
  panic／OOM 回避）。本番経路で `unwrap`／`expect` を使わない
- **A04 安全でない設計**: `ValMetric(m)` の `m ∉ metrics`／非スカラー
  監視を事前検査で拒否し、callback がサイレントにスキップされる経路を
  作らない
- **A05 設定ミス**: `MonitorMode` 既定 `Min` のまま accuracy を監視する
  誤設定は doc で明示（自動推定による暗黙挙動を導入しない）
- **A06 脆弱・古いコンポーネント**: 依存追加なし（`Cargo.toml`／
  `Cargo.lock` 不変）
- **A08 整合性**: tolerance／baseline／ガードレール閾値／`docs/spec/`
  不変。facade 公開面は §2 の 4 件に限定し `api_surface.rs` で機械固定。
  `unsafe` 追加なし

## 11. 承認事項の整理

facade 公開面の拡張（`Metrics`・`MetricsResult`・`Sequential::
fit_with_metrics`・`Monitor::ValMetric` の 4 件）は `docs/compat-api-
scope.md` §5「適用記録（経路 2。イシュー #2072）」に記録済み。依存
追加・`unsafe`・spec 提案・tolerance／baseline 変更は発生していない。
