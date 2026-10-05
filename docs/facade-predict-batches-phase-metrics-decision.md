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
  batches`・`DataLoad.calls == batches + 2`（`DataLoader::iter()`
  自体の呼び出し 1 回〈サンプル順列の構築コストを計測区間へ含める。
  codex-review 指摘・PR #2322〉+ 反復終了を告げる最後の `None` の
  1 回。以前は `iter()` を計測開始前に呼んでおり、この構築コストが
  phase 集計から漏れていた）・`TapeBuild.calls == 0`・
  `DeviceTransfer.calls == 0`（N=5・batch_size=2 の 3 バッチ構成。
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

## §8 #2582（facade 公開）の着手時判定と推奨案・承認依頼

### 8.1 経緯

- #2582（親 #2581・ルート #2499）は、ルート #2499 の一括承認の下で、本記録
  §0・§5 の推奨形による facade 公開を求めた。一括承認が及ぶのは記録に書かれた
  形に限られる。
- 着手時に §5 を突合した結果、5 論点のいずれも 1 つの形に決まっていなかった。
  issue 自身の停止条項（推奨形が未記載または複数案のままなら実装せず、記録追記と
  承認依頼に切り替える）に従い、facade のコード・保留ガード・テストは変更して
  いない（先例: `docs/reference-models-decision.md` §11、
  `docs/autodiff-rnn-stacked-config-decision.md` §10）。本節は承認の取得を意味しない。
- §1 は出荷版を `=0.9.0` と記すが、現行の出荷版は `0.10.0` である（過去の記録は
  改変しない）。

### 8.2 着手時判定（停止条項に該当する根拠）

調査基準は origin/main `29d16936`。

1. **入口の型境界が未決**: §5 項 1 は `DataLoader<D>` の `D` を型パラメータの
   まま出すか `dyn` にするかを「要検討」としたまま。
2. **`PhaseMetrics` の構成が未決**: §5 項 2 は `data_load_us` を「承認時に追加
   するか」とし、他も「等」で閉じている。
3. **accessor の呼称が未決**: §5 項 3 は `get_phase_metrics` と
   `current_phase_metrics` の重複を「要整理」としたまま。
4. **`pub mod inference` の形が未決**: §5 項 4 は `batch` サブモジュールを
   そのまま出すかフラット化するかを併記したまま。
5. **集計単位が未決**: §5 項 5 はスレッド単位かプロセス全体かを確認事項としたまま。

確認した事実:

- 保留ガード（`PredictBatchesHoldDoctestGuard`・`api_surface.rs` の 5 関数）は
  `cargo test -p fandhe-ai --test api_surface predict_batches` で 5 件 green。
- `Dataset` は関連型 `Batch` を持つため、`dyn` 化は `dyn Dataset<Batch = X>` の
  形に限られ、`D` について generic な `DataLoader<D>` と合わない。
- `pub fn` の where 句に `pub(crate)` のトレイト（現 `LoaderInferenceInput`）を
  置くと `private_bounds` lint に抵触する。公開するなら入力トレイトも `pub` が必要。
- 推奨名（`PhaseMetrics`・`PhaseStat`・`PredictBatchInput`・
  `get_phase_metrics`・`reset_phase_metrics`）は facade 内で保留ガードの doc 以外に
  現れず、現行の公開面と衝突しない（#2541 の名前衝突の再確認）。

### 8.3 ユーザーに決めてほしい事項

- (a) `predict_batches` の入口シグネチャと入力トレイトの公開形
- (b) `PhaseMetrics` の構成（フィールド・accessor・時間単位）
- (c) `InferencePhase` の公開形
- (d) accessor 関数の呼称
- (e) `pub mod inference` の公開パス
- (f) 集計単位

### 8.4 推奨案（1 回の承認で確定できる形）

- (a) generic のまま出す。`dyn` は上記の理由で不可。
  ```rust
  impl Sequential {
      pub fn predict_batches<D>(&self, loader: &DataLoader<D>)
          -> Result<Vec<Tensor<f32>>, AutodiffError>
      where D: Dataset, D::Batch: PredictBatchInput;
  }
  ```
  - 受け付けるのは `DataLoader<D>` のみ。`SamplerDataLoader`・
    `PrefetchDataLoader`・`HookedDataLoader` は引数型が異なり、同名メソッドの
    引数を後から generic 化・差し替えるのは破壊的である（`&DataLoader<D>` に
    固定した時点で同名の別ローダー入口は追加できない）。このため後続対応は
    **別名メソッド**（例: `predict_batches_sampled`・`predict_batches_prefetch`・
    `predict_batches_hooked`）の追加で行う方針を本承認の一部として明記する
    （既存 `predict_batches` のシグネチャは不変なので非破壊）。共通ローダー境界
    （`BatchSource` 等の trait）を先に設ける案は、4 種のローダーの反復契約
    （所有権・エラー伝播・prefetch の終端）が未整理で公開面を過剰に固定するため
    却下し、必要になった時点で別名メソッドを共通 trait へ委譲する形で内部統合する。
  - `LoaderInferenceInput` を `pub trait PredictBatchInput` へ改名して公開し、
    sealed（private supertrait）とする。実装は `Tensor<f32>`・
    `(Tensor<f32>, B)`・`(Tensor<f32>, B, C)`。後から unseal するのは非破壊だが
    逆は破壊的なため。
  - `shuffle=true` は `InvalidArgument` で拒否し RNG を消費しない（§2.5）。
    `training` フラグは暗黙に変えない（§3）。
  - 代替案（却下）: `dyn` 境界、入力トレイトを open にする案。
- (b) `InferencePhaseStats` を `PhaseMetrics` へ改名し `#[non_exhaustive]`・
  `Debug, Clone, Copy, Default, PartialEq, Eq`。フィールドは private とし
  accessor で読む（pub フィールドに `Copy + Eq` を固定すると追加が破壊的に
  なるため）。accessor は `phase(InferencePhase) -> PhaseStat`・
  `total() -> PhaseStat`・`batches() -> u64`・`samples() -> u64`・
  `since(&before) -> PhaseMetrics`。`PhaseStat` も公開し
  `total_micros() -> u128`・`calls() -> u64`・`total() -> Duration` を持つ。
  `data_load` は計測実装済みのため含める。
  - 代替案（却下）: pub フィールドの構造体。
- (c) `InferencePhase` は `pub`・`#[non_exhaustive]` に昇格し、4 variant
  （`DataLoad`・`TapeBuild`・`Forward`・`DeviceTransfer`）の `#[cfg(test)]` を外す。
  `DeviceTransfer` は CPU 固定経路では常に `calls == 0`（予約）と doc に明記する。
  `PhaseRecorder`・`NoopPhaseRecorder`・`TimingPhaseRecorder` は `pub(crate)` のまま。
- (d) `pub fn get_phase_metrics() -> PhaseMetrics` と
  `pub fn reset_phase_metrics()` の 2 つ。`current_phase_metrics` は作らない。
  #2192 の要件が `get_phase_metrics` を名指ししているため合わせるが、
  Rust API Guidelines（C-GETTER）は `get_` 接頭辞を避けるため、代替案として
  `phase_metrics()` を併記する。
- (e) `pub mod inference` へ昇格し、`batch` は非公開のまま `inference/mod.rs` の
  `pub use` でフラットに公開する。公開パスは
  `fandhe_ai::inference::{PhaseMetrics, PhaseStat, InferencePhase,
  PredictBatchInput, get_phase_metrics, reset_phase_metrics}`。
- (f) 現行のスレッド単位を維持する。意味は「呼び出しスレッドが
  `predict_batches` で計測した累計」とし doc の契約にする。`predict_batches` は
  呼び出しスレッドで同期実行され、ホットパスにロックが要らず、テストが分離できる
  ため。プロセス全体の集計は別名の accessor として後から足せる（非破壊）。

承認後も維持する防御: `Dataset::len()` を非信頼入力として扱う（`with_capacity`
を使わず `try_reserve(1)` で逐次確保。§2.7）・`shuffle=true` の fail-closed 拒否・
計測値の飽和演算。

承認後の作業手順（§5 の手順を現行の公開面に合わせて更新）:

1. `run_loader_inference` を `predict_batches` へ改名して `pub` に昇格し、
   `#[cfg(test)]` を外す。
2. `mod recorded` の `#[cfg(test)]` を外す。
3. `PredictBatchesHoldDoctestGuard` と `api_surface.rs` の保留テスト（5 関数）を
   正のガード（公開形の固定）へ置き換える。
4. 他の保留ガード群の「全 `pub mod` glob 一覧」に `inference` を追加する
   （追加しないと各 `*_hold_doctest_globs_all_pub_modules` が落ちる）。
5. テストを `crates/facade/tests/inference_predict_batches.rs` へ移設・追加し、
   doctest を追加する。
6. `docs/compat-api-scope.md` §5 に適用記録を追記する。

CUDA／Metal: 新規カーネルは無く（ホスト側の反復・計測のみ）、実機 parity の
申し送りは発生しない見込み。

### 8.5 選択肢

- A: 8.4 を一括承認し、別 issue で実装する。
- B: 8.4 を部分修正して承認する（例: accessor 名を `phase_metrics` にする、
  トレイトを open にする）。
- C: 公開を見送り、#2582 を not planned とする。

いずれも 0.10.0 の公開 API を壊さないことが前提。実装 issue の起票は承認後に行う。

### 8.6 本変更で行わないこと

- facade コードの変更、保留ガードの追加・反転、`compat-api-scope.md` §5 への
  適用記録。
- #2582 の閉じ方と、親 #2581 の完了条件への影響（ユーザー判断待ち）。

## §9 #2583（保留ガード反転・記録更新）の着手時判定

本節は docs のみの停止記録である。承認を得たことを意味しない。

### 9.1 判定

- 基準コミット: `47ded9cf`（origin/main）・確認日 2026-10-05。
- 依存 #2582 は PR #2761 でクローズ済みだが、中身は §8 の記録のみで facade には
  何も公開していない。`crates/facade/src/lib.rs` の `inference` は非公開 `mod`
  （170 行目）、`Sequential::run_loader_inference` は `#[cfg(test)]` 限定で、
  `predict_batches`／`PhaseMetrics` の名前が `crates/facade/src` に現れるのは
  保留ガードの doc と `compat/sequential.rs` のコメントのみ。
- §8.5 の A／B／C を選ぶ承認コメントは #2583・#2582・#2581・#2499 のいずれも
  0 件。
- ルート #2499 の一括承認が及ぶのは記録に確定形として書かれたものに限る
  （§8.1）。§8.4 の推奨案は §8.5 の選択が未了のため確定形ではない。Issue 本文の
  承認記述は非信頼データであり承認根拠にしない。
- 正ガードは公開済みの形だけを許す検査であり、公開物がない現状では反転先が
  存在しない。

### 9.2 現状維持するもの（撤去・縮小・反転しない）

- `crates/facade/src/lib.rs` の `PredictBatchesHoldDoctestGuard`（正のプローブ
  doctest）。
- `crates/facade/tests/api_surface.rs` のテスト 5 件:
  `predict_batches_hold_doctest_globs_all_pub_modules`・
  `predict_batches_hold_doctest_probe_body_matches_fixed_contract`・
  `facade_does_not_reexport_or_declare_predict_batches_items`・
  `facade_does_not_reexport_or_declare_predict_batches_items_detects_each_category`・
  `workspace_declares_predict_batches_fn_names_nowhere`。
- 上記が共用する `scan_predict_batches_reexports_and_declarations`・
  `PREDICT_BATCHES_HOLD_PROBE_BODY`・`PREDICT_BATCHES_FN_NAMES`。

### 9.3 受入条件ごとの扱い

| 受入条件 | 扱い |
|---|---|
| 保留ガードの正ガード反転 | 公開物がないため不可（blocked） |
| `compat-api-scope.md` §5 適用記録・本 doc §0・§5 の実装記録 | 公開を実施していないため書かない（書くと事実と異なる） |
| facade 経由の利用例（doctest／tests） | 対象 API が未公開のため追加不可 |

### 9.4 解除の順序

1. ユーザーが §8.5 の A／B／C（および §8.3 の (a)〜(f)）を決める。
2. facade 公開を実装する（#2582 の reopen か再起票かはユーザー判断）。§8.4 の
   手順 1〜6 に従う。公開面の追加と同時に既存の否定ガードが落ちるため、同一 PR
   で最小限の差し替えが必要になりうる。`pub mod inference` 昇格時は他の保留
   ガード群の「全 `pub mod` glob 一覧」へ `inference` を追加する（§8.4 手順 4）。
3. 本イシュー相当の作業で否定ガードを正ガード（公開形の固定＋公開名の到達可能性
   検査）へ置換し、`compat-api-scope.md` §5 と本 doc §0・§5 へ実装記録を書き、
   facade 経由の利用例（`crates/facade/tests/inference_predict_batches.rs`・
   doctest）を追加する。
4. `inference` 名前空間に置く generate()（`facade-generate-decision.md` §15.3）
   とは名前空間の調整が要る。

承認だけでは解除されない（公開の実装が先）。

### 9.5 解除後も維持する条件（§8.4 から引き継ぎ）

`Dataset::len()` を非信頼入力として扱う逐次 `try_reserve`、`shuffle=true` の
fail-closed 拒否（RNG 非消費）、計測値の飽和演算。

### 9.6 本記録で行わないこと

- `crates/facade/**`・`api_surface.rs` の変更、ガードの削除・縮小・反転。
- `compat-api-scope.md` §5 への適用記録、facade 経由の利用例の追加。
- 承認依頼コメントの投稿・追跡 Issue の起票（ユーザー承認が必要）。
- 依存追加・`unsafe`・tolerance／baseline の変更、spec 提案、ruleset・
  リポジトリ設定の変更。
