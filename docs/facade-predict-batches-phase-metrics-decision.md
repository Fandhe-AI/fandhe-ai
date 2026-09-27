# バッチ推論 API・phase 計測公開面（`predict_batches`）設計判断記録

イシュー #2192（親 #2131「PyTorch／TF 置き換えの API 網羅（対応表の
行内深掘り）」）。`docs/compat-train-step-hook-decision.md`（#2184）・
`docs/compat-model-io-decision.md`（#2188）と同型の記録。

## §0 結論

`DataLoader` を全バッチ反復して `Sequential::predict` を呼び出す内部
実装（`Sequential::run_loader_inference`・`crate::inference::batch`）
と、その内訳（データロード・tape 構築・forward・デバイス転送の各
フェーズの累計時間・呼び出し回数）を計測する内部機構
（`InferencePhase`・`PhaseRecorder`・`TimingPhaseRecorder`・
`InferencePhaseStats`・thread-local アキュムレータ）を実装した。
`Sequential::predict`（既存公開 API）は本体を
`predict_recorded<R: PhaseRecorder>` へ移設し、`NoopPhaseRecorder`
（計測オーバーヘッドなし）を渡すだけの薄い委譲にした。

**facade 公開面は追加していない（承認待ち）**: イシュー #2192・親
#2131 のいずれにも所有者の承認コメントはない（着手前に `gh issue
view --json comments` で確認済み・コメント 0 件）。親 #2131 の
「facade 公開面の拡張は設計判断記録 → 承認 → 実装の 2 段」規則
（先例 #2177・#2179・#2180・#2171・#2173・#2176・#2178・#2184・
#2188・#2198）に従い、`pub fn predict_batches`・`pub struct
PhaseMetrics`（`get_phase_metrics`／`current_phase_metrics`／
`reset_phase_metrics` を含む計測アクセサ一式）・`pub mod inference`
の新設は追加せず保留する。現状は `#[cfg(test)] fn Sequential::
run_loader_inference` 経由でのみ到達でき、通常経路（公開 API のみ）
ではバッチ反復推論・phase 計測は使えない。

## §1 背景・要件（原文はイシュー本文。ここでは構造化要約のみを記す）

- R1: `Sequential::predict_batches(&DataLoader) -> Vec<Tensor<f32>>`
  相当。`DataLoader` による batch 読み込み・`predict` の自動反復
- R2: `get_phase_metrics() -> PhaseMetrics { tape_build_us,
  forward_us, device_transfer_us, ... }` 型と accessor（プロファイラ
  統合。tape_build・forward・device_transfer 等の phase 計測を facade
  から読み取り可能にする）
- R3: バッチごとの出力が単体 `predict` した結果と一致すること
  （checksum・forward 出力の一致確認）
- R4: phase metrics が before/after で比較可能（時間単位・呼び出し
  回数）な構造であること

**契約（不変条件）**: tolerance・baseline・`Cargo.toml`／
`Cargo.lock`・ガードレール閾値・`docs/spec/` は変更していない。新規
`unsafe` は入れていない。新規 `Op`／`BackendOps`／カーネルも追加して
いない（ホスト側の計測・反復ロジックのみ）。crates.io 出荷済み
`fandhe-ai =0.9.0` の公開 API は破壊していない。

**スコープ外**（§6 参照）: GPU 側の fine-grained profiling（CUDA
event・Metal capture）・distributed inference（parallel batch）・
`crates/bench-harness/`・`scripts/bench/` 側の集計ツール・facade 公開
面の新設（§5 参照）。

## §2 設計

### 2.1 `InferencePhase` の 4 分類と計測範囲

| phase | 計測対象 | 常時有効か |
|---|---|---|
| `DataLoad` | `DataLoader::iter`／`Batches::next` の呼び出し（バッチの切り出し） | `run_loader_inference`（`#[cfg(test)]` 限定）専用 |
| `TapeBuild` | tape 経路（`predict_via_tape`）の `crate::tape()` | `Sequential::predict` の tape 経路でも計上（`NoopPhaseRecorder` 経由のため実 stats には残らない） |
| `Forward` | 層の forward 実行（tape 不要経路は `forward_host` 連鎖、tape 経路は `tape.var(..)`・`forward`・`to_tensor()`） | 同上 |
| `DeviceTransfer` | デバイス常駐経路（`predict_resident` 等）のホスト⇔デバイス転送 | `run_loader_inference` 専用。CPU 固定の `predict` 経路では発生しないため常に `calls == 0`（既定の推論経路がデバイス常駐 API を経由しないため。デバイス常駐経路への `run_loader_inference` の対応は本イシューのスコープ外） |

`DataLoad`／`DeviceTransfer` は `#[cfg(test)]` を付けて隔離する
（`Sequential::run_loader_inference` 自体が `#[cfg(test)]` 限定のため、
通常ビルドではこれらの variant に到達するコードが存在せず
`dead_code` lint に抵触するため）。`TapeBuild`／`Forward` は
`Sequential::predict`（常時公開）が `NoopPhaseRecorder` 経由で参照
するため無条件で存在する。

### 2.2 単一本体化（`predict_recorded`）

`Sequential::predict` の本体を `predict_recorded<R: PhaseRecorder>`
（`pub(crate)`）へ移設し、`predict` 自体は `NoopPhaseRecorder` を
渡すだけの 1 行にした。`Sequential::run_loader_inference` は
`TimingPhaseRecorder` を渡してバッチ推論の内訳を集計する。単一本体
にすることで、両呼び出し元の分岐論理（tape 要否判定・フォール
バック構造）が乖離しないことを構造で保証する。

**フォールバック時の計測**: tape 不要経路が `Unsupported` を返し
旧経路（tape 経路）へフォールバックした場合、`Forward` は実際に
行った試行の回数として 2 回計上される（1 回目は `Unsupported` に
終わった tape 不要経路の試行、2 回目はフォールバック先）。この重複
計上は仕様であり是正しない（フォールバックの発生自体が稀な経路
異常であり、`calls` を「実際に実行した回数」の単純な意味論に保つ
ことを優先した）。

### 2.3 thread-local アキュムレータ（プロセス全体ではなくスレッド単位）

`INFERENCE_PHASE_STATS`（`thread_local!`）をスレッド単位にした理由:
ロック不要でテスト間の干渉を避け、「このスレッドが
`run_loader_inference` で計測した累計」という単純な意味論にするため。
承認時にプロセス全体集計（`Mutex`／`AtomicU64` 等）へ変えるかは §5
の確認事項とする。

`run_loader_inference` は呼び出し 1 回分の計測値
（`TimingPhaseRecorder::into_stats()`）を、途中で失敗した場合も含めて
`merge_inference_phase_stats` で thread-local へ飽和加算する（それ
までに計測した分を失わない）。`PhaseStat::add`／`saturating_sub`／
`merged` はすべて飽和演算（`Duration::saturating_add`・
`u64::saturating_add`）で、`Duration::MAX`・`u64::MAX` 付近でも
panic しない。

### 2.4 R4（before/after 比較）: `InferencePhaseStats::since`

`InferencePhaseStats::since(&self, before: &Self) -> Self` が
`self`（after）と `before` の飽和差分を返す。呼び出し側
（bench-harness 等、承認後）は `inference_phase_stats_snapshot()` を
呼び出し前後で 2 回取得し `after.since(&before)` でその区間の計測値
のみを取り出せる。`total_micros()`（`u128`。`Duration::as_micros` と
同精度）で浮動小数へ変換しやすい形にしている。

### 2.5 `shuffle=true` の拒否

`run_loader_inference` は `loader.config().shuffle == true` を
`AutodiffError::InvalidArgument` で fail-closed に拒否する。理由は
2 つ: (1) 出力順がデータセット順と対応しなくなる、(2)
`Batches::next` がグローバル RNG を消費し呼び出し元の RNG 状態を
暗黙に変える（`fandhe_ai_tensor_core::data` モジュール doc
「shuffle=false はグローバル RNG を消費しない」契約の消費者側への
波及）。拒否経路自体は RNG を一切呼ばないため、拒否してもグローバル
RNG は消費しない（テスト
`run_loader_inference_rejects_shuffle_without_consuming_rng` で固定）。

### 2.6 ラベル無視（`LoaderInferenceInput` トレイト）

`DataLoader<D>::iter()` が生成するバッチ型から推論対象の
`Tensor<f32>` を取り出す `LoaderInferenceInput` トレイトを設け、
`Tensor<f32>`・`(Tensor<f32>, B)`・`(Tensor<f32>, B, C)` の 3 形を
サポートする。ラベル付きデータセットではラベルを無視する（Keras
`model.predict(dataset)` が特徴量のみを使うのと同じ扱い）。

### 2.7 出力 `Vec` の確保（信頼できない `Dataset::len()`）

`Dataset::len()` は実装者が任意の値（`usize::MAX` を含む）を返し
うる非信頼入力として扱う（`.claude/rules/security.md` A03）。
`Vec::with_capacity` は capacity overflow で panic するため使わず、
`Vec::new()` から `try_reserve(1)` しつつ逐次 `push` する
（`compat/training.rs` の `try_reserve_exact` パターンとは異なり、
バッチ数の事前見積り自体を信頼しない設計）。

## §3 Keras／PyTorch との差分

Keras `model.predict(dataset)`・PyTorch `DataLoader` を使った推論
ループは、いずれも呼び出し側がバッチを反復しつつ推論結果を蓄積する
命令的な形を取る。本実装（`run_loader_inference`）は同じ反復・蓄積
構造を持ちつつ、`training` フラグを暗黙に変更しない（`Sequential::
predict` と同じ既存契約を維持する。バッチ単体の `predict` と bit
一致させることが受入条件のため。呼び出し側が必要なら先に
`Sequential::eval` を呼ぶ）点で PyTorch の `model.eval()` 明示呼び出し
規約に近い。

## §4 数値一致と受入条件の実測

- **(a)(b)（R3）**: バッチごとの出力が、同じ行を単体 `predict` した
  結果と bit 完全一致すること（端数ありでも成立。N=7・batch_size=3
  の 3 バッチ構成。
  `run_loader_inference_matches_predict_per_batch_bit_exact`）
- **(c)**: `(x, y)` タプルのデータセットでラベルが無視されること
  （タプル版データローダーと素の特徴量版データローダーの出力が
  一致。`run_loader_inference_ignores_labels_in_tuple_dataset`）
- **(d)**: `drop_last=true` で端数バッチが捨てられること（N=7・
  batch_size=3 で 2 バッチのみ。`run_loader_inference_respects_
  drop_last`）
- **(e)**: `shuffle=true` を `InvalidArgument` で拒否し、拒否前後で
  グローバル RNG の出力列が変わらないこと
  （`run_loader_inference_rejects_shuffle_without_consuming_rng`）
- **(f)**: 空の `loader`（0 件データセット）が `Ok(vec![])` を返し
  `batches() == 0` になること
  （`run_loader_inference_on_empty_loader_returns_empty_vec`）
- **(g)（R2・R4）**: phase 計測——tape 不要経路で `Forward.calls ==
  batches`・`DataLoad.calls == batches + 1`（反復終了を告げる最後の
  `None` も 1 回の `DataLoad` 計測に含まれる）・`TapeBuild.calls ==
  0`・`DeviceTransfer.calls == 0`（N=5・batch_size=2 の 3 バッチ構成。
  `run_loader_inference_records_phase_stats_for_tape_free_path`）
- `PhaseStat::add`・`saturating_sub`・`InferencePhaseStats::since`・
  `merge`・thread-local の分離（別スレッドの累計が漏れ伝わらない
  こと）は `crate::inference::batch::recorded::tests`（7 テスト）で
  個別に固定済み

CUDA／Metal 固有の処理は追加していない（ホスト側の反復・計測ロジック
だけの変更）ため、実機 parity の `#[ignore]` テストも
`docs/perf/logs/` への申し送りも発生しない。

## §5 承認事項（未承認のため保留）

facade 公開面の新設は次が未承認。`crates/facade/src/
lib.rs::PredictBatchesHoldDoctestGuard`（正のプローブ doctest）・
`crates/facade/tests/api_surface.rs` の 4 テスト
（`predict_batches_hold_doctest_globs_all_pub_modules`・
`predict_batches_hold_doctest_probe_body_matches_fixed_contract`・
`facade_does_not_reexport_or_declare_predict_batches_items`（その自己
テスト含む）・`workspace_declares_predict_batches_fn_names_nowhere`）
が機械的に固定する:

1. `Sequential::predict_batches(&self, loader: &DataLoader<D>) ->
   Result<Vec<Tensor<f32>>, AutodiffError>`（`D: Dataset, D::Batch:
   LoaderInferenceInput` の generic 境界を facade 公開面へどう
   出すか——`DataLoader<D>` の `D` を型パラメータのまま公開するか、
   `dyn` 境界へ変えるかは要検討）
2. `pub struct PhaseMetrics`: `tape_build_us`・`forward_us`・
   `device_transfer_us`・`data_load_us`（承認時に追加するか）・
   `batches`・`samples` 等のフィールド構成
3. `pub fn get_phase_metrics() -> PhaseMetrics`（プロセスまたは
   スレッド単位の累計 snapshot）・`current_phase_metrics()`（`get_
   phase_metrics` との呼称重複の要整理）・`reset_phase_metrics()`
4. `pub mod inference` への昇格（現状は非公開 `mod inference`）。
   昇格時に `batch` サブモジュール名を facade 公開面にどう出すか
   （`inference::batch::*` をそのまま出すか、`inference::` 直下へ
   フラット化するか）
5. **プロセス全体集計 vs スレッド単位集計**（§2.3）: 現状の
   thread-local 方式のまま公開するか、`bench-harness` 等の別スレッド
   計測を集約できるプロセス全体方式へ変えるか

**承認後の作業手順**: 公開型・入口の追加、テストを `#[cfg(test)]
mod recorded::tests`（`crates/facade/src/inference/batch.rs`）から
`crates/facade/tests/inference_predict_batches.rs`（外部統合テスト
クレート。イシュー本文が指す `crates/facade/tests/inference_*.rs`）
へ移設、`run_loader_inference`／`recorded` モジュールの `#[cfg(test)]`
属性を撤去し `pub` へ昇格、`PredictBatchesHoldDoctestGuard` と対応
する 4 テストの削除（正のガード・facade テストへ置き換え）。

承認を得た日が来たら、上記を実施する。

## §6 スコープ外（out-of-scope-tracking）

- facade 公開面の新設（§5 参照）
- GPU 側の fine-grained profiling（CUDA event・Metal capture）
- distributed inference（parallel batch）
- `crates/bench-harness/`・`scripts/bench/` 側の phase metrics 集計・
  分析ツール（facade 公開面が確定してから着手する）
- デバイス常駐経路（`predict_resident` 等）への `run_loader_inference`
  の対応（`DeviceTransfer` phase は型として用意済みだが、現状の
  `run_loader_inference` は CPU 固定の `predict_recorded` のみを呼ぶ
  ため到達しない）

新規 Issue の起票はユーザー承認後に行う。

## §7 検証コマンドと結果

```
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p fandhe-ai --lib inference::batch
cargo test -p fandhe-ai --lib compat::sequential
cargo test -p fandhe-ai --test api_surface predict_batches
cargo test -p fandhe-ai --doc
cargo test -p fandhe-ai
cargo build --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
```

全て green（2026-09-27 実測。ローカル worktree）。
