# DataLoader マルチワーカー prefetch の設計判断記録（#2183）

イシュー #2183「DataLoader マルチワーカー prefetch（rayon）の実装と性能検証」。親 #2131（Phase 5「PyTorch／TF 置き換えの API 網羅」5-A 基盤）。

本ドキュメントは `tensor-core::data` 側実装（PyTorch `torch.utils.data.DataLoader(num_workers=...)` 相当）と、facade 公開面拡張・`Sequential::fit` 結線（いずれも承認事項）の切り分け、および決定性契約（seed のみで決まる出力順序）を記録する。`crates/facade/src/data.rs`（純再エクスポート。無変更）・`crates/facade/src/compat/training.rs`（`FitConfig` 無変更）・`CLAUDE.md`・`docs/spec/`（正本 submodule）・`Cargo.toml`／`Cargo.lock`（全クレート）・tolerance／baseline・ガードレール閾値・CI／hooks は変更しない。

## 0. 結論・段階

- **内部クレート側（`tensor-core::data`）は実装済み**（本 PR）。`PrefetchConfig`（`num_workers`／`prefetch_depth` の検証付き設定）・`PrefetchDataLoader<D>`（`Sampler` の添字を worker スレッドへ分配して並列に `Dataset::batch` を実行し、結果を呼び出し順に並べ直す）・`PrefetchBatches<D>`（そのイテレータ）。
- **rayon ではなく `std::thread`／`std::sync::mpsc` を使う**（§1.1。イシュータイトルは「rayon」と書いているが、本 PR の契約〈`Cargo.toml`／`Cargo.lock` 変更なし・依存追加は承認事項〉と両立しないため意図的に差し替えた。rayon への切り替え自体は §8 の承認事項として記録する）。
- **（#2506 で facade 公開済み。§4・§8 参照。以下は #2183 時点の記録）** **facade 公開（`docs/compat-api-scope.md` §5 経路 2 相当）・`Sequential::fit` への結線はいずれも未承認のまま保留**。#2183・親 #2131 のいずれにも承認コメントは見当たらない（着手時点確認）。よって `crates/facade/src/**` は変更せず、`PrefetchConfig`／`PrefetchDataLoader`／`PrefetchBatches` は一切公開しない（兄弟イシュー #2182〈`docs/tensor-core-data-sampler-hooks-decision.md`〉と同型の保留パターン。ただし本イシューは新規の保留 doctest 足場を追加していない——理由は §5 参照）。既存の facade 公開面は不変。
- **本 PR のマージで #2183 は COMPLETED とする**（前例と同じ「保留記録を残した PR のマージで issue をクローズし、承認が得られたら新規 issue か reopen で経路 2 を実施する」方針）。

## 1. 依存・並行処理方式の設計判断

### 1.1 rayon ではなく `std::thread` を使う理由

- `crates/tensor-core/Cargo.toml` の `[dependencies]` は `half` のみ。rayon を追加するには `rayon.workspace = true` を足す必要があり、これは `.claude/rules/deps-policy.md` の許容依存区分（第 1〜8 区分）への新規追加になる。依存の追加・更新はユーザー承認必須（`CLAUDE.md` Conventions）であり、本 PR の契約（`Cargo.toml`／`Cargo.lock` 変更なし）とも矛盾する。
- `backend-cpu` の GEMM 等の計算カーネルは rayon のグローバル pool を使う（`.claude/rules/deps-policy.md` の CPU 並列区分）。loader の worker を同じ pool に載せると、重ねたい相手（学習ステップのカーネル）と worker を取り合う。専用の OS スレッドならこの競合を避けられる。
- 以上から、worker pool は `std::thread::Builder::spawn` で作り、`std::sync::mpsc` の有界な in-flight 制御（§3）で先読み量を抑える構成にした。
- 「rayon を tensor-core に結線する」案は **承認事項**として §8 に記録し、実施しない。

### 1.2 既存の型は変えず、新しい型を追加する

- `DataLoader`／`DataLoaderConfig`（facade から再エクスポート済み）にフィールド・メソッドを足すと facade 公開面の拡張になり承認事項に触れる（`docs/tensor-core-data-sampler-hooks-decision.md` §1.1 と同じ理由）。`SamplerDataLoader`／`HookedDataLoader` も同様に変更していない。
- #2182 と同じく `tensor-core::data` に新しい型（`PrefetchConfig`／`PrefetchDataLoader<D>`／`PrefetchBatches<D>`）を追加した。名前は `DATA_HOOKS_TYPE_NAMES`（10 個）と重ならない。
- 型境界は `D: Dataset + Send + Sync + 'static`、`D::Batch: Send + 'static`。`TensorDataset<T>`（`Element: Send + Sync`）とタプル実装はこれを満たす。
- transform／collate フック（`HookedDataLoader` 相当）の並列化は対象外（§6）。worker 上の transform がグローバル RNG を使う augmentation だと、消費順がスケジューリングに依存し決定性契約（§2）を破るため。

## 2. 決定性の契約（R2）

- `PrefetchDataLoader::iter` は呼び出しスレッド上で `Sampler::start_epoch` を 1 回呼ぶ。添字生成（`Sampler::next_batch`）もその後は呼び出しスレッド（consumer）上でのみ `k` の昇順に 1 回ずつ行うが、**`prefetch_depth` が定める投入窓ぶんだけ有界に**行う（§10「レビュー是正」参照。初版は epoch 全体を `Vec<Vec<usize>>` として eager に一括確定していたが、`prefetch_depth` が先読み量を抑える契約に反して大きな epoch でメモリを使い切り得たため是正した）。組み込みの 3 `Sampler` 実装（`SequentialSampler`／`RandomSampler`／`WeightedRandomSampler`）は RNG 消費を `start_epoch` 内の 1 回に限り、`next_batch` は確定済み順列（`IndexBatcher`）から純粋に切り出すだけのため、生成タイミングを分散させても RNG 消費順（＝添字順）は不変で決定性契約は保たれる。`next_batch` 側で RNG を消費するカスタム `Sampler` は本契約の対象外（transform／collate フックと同じ制約）。worker はグローバル RNG に一切触れない。
  - `start_epoch` が失敗した場合は、既存の `SamplerBatches` と同じ `pending_error` 方式にする（最初の `next()` で `Err` を 1 回だけ返し、以降は `None`。スレッドは起動しない）。
- worker は `dataset.batch(&indices[k])` だけを実行し、`(k, result)` を結果チャネルへ送る。consumer は `BTreeMap<usize, Result<..>>` の reorder buffer で `k` の順に並べ直して yield する。
- 保証する内容:
  - 任意の `(num_workers, prefetch_depth)` の組み合わせで、出力列（`Err` の位置と内容を含む）が同じ `Sampler` を使う `SamplerDataLoader` と bit 完全一致する。
  - `RandomSampler` 経由なら `DataLoader{shuffle=true}` とも bit 完全一致する（#2182 で確立済みの等価性）。
  - epoch 終了後のグローバル RNG 状態も逐次版と一致する。
  - 前提は `Dataset::batch` が添字だけで決まる純関数であること（`TensorDataset`・タプル実装は該当）。
- `Dataset::batch` が返す `Err`（例: `IndexOutOfRange`）は `Sampler` の枯渇とは独立の事象であり、`SamplerBatches::next` と同じく epoch を打ち切らない（既存契約。逐次経路・並列経路とも同じ挙動——`crates/tensor-core/src/data.rs::tests::prefetch_error_position_matches_sequential` で固定）。

## 3. ライフサイクル・先読み量の制御

- consumer は `delivered + prefetch_depth` を超えない範囲でタスク `(k, indices)` を投入する（`advance_after_delivery`）。タスクキューは `Arc<Mutex<Receiver>>` を worker 間で共有し、結果チャネルは通常の `mpsc::channel`（in-flight 数が `prefetch_depth` 以下に抑えられるため有界化は不要）。メモリ上限はおおよそ `prefetch_depth` バッチ分になる。
- `num_workers == 0` のときはスレッドを起動せず、呼び出しスレッド上で逐次に処理する（PyTorch と同じ意味）。
- `Drop for PrefetchBatches`: タスク送信側（`task_tx`）を先に drop してタスクキューを閉じ、結果チャネルの受信側（`result_rx`）も drop してから、各 worker の `JoinHandle::join` を待つ（結果は捨てる。worker の panic が drop 時の panic に波及しない）。
  - worker 側は「タスクキューが閉じられた（`task_rx.lock().recv()` が `Err`）」または「結果送信が失敗した（consumer が drop 済み）」のいずれかで自然にループを抜けるため、consumer が epoch を最後まで消費せず `break` してもデッドロックしない（`crates/tensor-core/src/data.rs::tests::prefetch_partial_consumption_drop_does_not_hang` で固定）。
- worker ループは `dataset.batch(&idx)` の呼び出しを `std::panic::catch_unwind` で包む（レビュー指摘・イシュー #2183 コメントで是正）。捕捉しないと panic した worker のスレッドだけが終了し、その worker が持つ `result_tx` クローンだけが drop される。`task_tx`（consumer 側）はまだ生きているため他の worker は `task_rx.recv()` でブロックし続け、`result_tx` も全クローンが尽きない限り `Err` にならない——`num_workers >= 2` では該当位置の結果が二度と届かず `next()` が恒久的にハングする（`num_workers == 1` のときのみ「唯一の送信側が尽きる」ため後述の disconnect 経路が機能していた）。`catch_unwind` で panic をその場で `DataError::WorkerFailed { batch_index: k }`（`k` は panic した位置そのもの）へ変換して結果チャネルへ送ることで、スレッドは終了せず次のタスクへ進み、consumer は通常の reorder buffer 経由で当該位置のエラーを受け取る（他の位置の結果と同じ経路・同じ順序保証）。
  - 上記に加えて、スレッド生成失敗やチャネル自体の異常等で worker が結果を送れないまま終了し得る場合に備え、「全 worker の送信側が尽きて結果チャネルが disconnect した状態で、まだ届いていない位置がある」ケースも `DataError::WorkerFailed { batch_index }` を 1 回だけ yield するフォールバックとして残す（以降は `None` を返す）。`batch_index` はこの場合「検出時点で consumer が待っていた位置」を指す。
- 構築時の検証: `prefetch_depth >= 1`（ゼロは先読みなしを表せないため非対応）。`num_workers` に上限 `PREFETCH_MAX_WORKERS = 64` を設ける（スレッドの大量生成による資源枯渇防止。OWASP A04）。スレッド生成の失敗（`io::Error`）は `DataError::WorkerSpawn` として型付きで返す（起動済みの worker は `task_tx` を drop してから `join` して後始末する）。

`DataError` への追加 variant（`#[non_exhaustive]` のため非破壊。#2182 が先例）: `InvalidPrefetchConfig { reason: &'static str }`・`WorkerSpawn { reason: String }`・`WorkerFailed { batch_index: usize }`。`Display` も対応。

## 4. facade・`Sequential::fit` の扱い

- facade への公開は承認事項なので行わない。`crates/facade/src/data.rs` は変更しない（6 型の集合は `data_module_reexports_exactly_expected_surface` が固定している）。
- `crates/facade/tests/api_surface.rs` に否定ガード（`facade_does_not_reexport_or_declare_prefetch`）とインベントリ（`workspace_declares_prefetch_names_only_in_tensor_core_data`）を追加した。`PrefetchConfig`／`PrefetchDataLoader`／`PrefetchBatches` の 3 名について、`DATA_HOOKS_TYPE_NAMES` と同型の走査（`pub use` の葉・独自 `trait`／`struct`／`enum`／`type` 宣言）で facade 非公開・`tensor-core/src/data.rs` 単一定義を固定する。
  - **#2182 との差分（意図的）**: #2182 は `DataHooksHoldDoctestGuard`（正のプローブ doctest。facade の全 `pub mod` を glob import した状態で新規メソッドを未修飾呼び出しし、コンパイルが通れば「まだ公開されていない」ことを確認する形）を追加したが、本イシューは追加していない。理由: プローブ doctest は「将来 facade が同名のメソッド・関数を実装して迂回する」ことを検出する多層防御だが、`PrefetchConfig::new`／`num_workers`／`prefetch_depth`、`PrefetchDataLoader::new`／`dataset`／`config`／`num_batches`／`iter` はいずれも既存の facade 公開型（`DataLoaderConfig`／`DataLoader` 等）が持たない新規メソッド名ではなく、他の多数の型が同名メソッドを持ちうる汎用的な名前（`new`／`config`／`iter` 等）であるため、glob import プローブで「facade のどの型がこの名前を実装してもコンパイルが通らなくなる」形の固定は作れない（`DATA_HOOKS_FN_NAMES` が `with_transform` 等の非汎用名だったのとは異なる）。ソース走査ガード（型名 3 個の `pub use`／独自宣言の不在固定）のみで多層防御としては十分と判断した。
- `Sequential::fit`（`crates/facade/src/compat/training.rs::run_fit`）への結線もしない。`FitConfig` に `num_workers` を足すことは facade 公開面の拡張であり、承認事項として記録するだけにする（§8）。

### #2506 実装記録（facade 公開。親 #2500・ルート #2499）

- `PrefetchConfig`／`PrefetchDataLoader`／`PrefetchBatches` の 3 名を `crates/facade/src/data.rs` に `pub use fandhe_ai_tensor_core::data::{…};` 1 行で素の再エクスポートした（別名・facade 独自の型／関数なし。純再エクスポートの契約は不変）。`PREFETCH_MAX_WORKERS`／`PREFETCH_MAX_DEPTH` は承認範囲外のため公開しない。`Sequential::fit` への結線（#2603）・rayon 化は含まない。
- ガード反転: 否定ガード `facade_does_not_reexport_or_declare_prefetch` を承認形のみを許す正ガード `facade_reexports_prefetch_items_only_in_approved_shape` へ反転し、自己テスト `…_detects_each_category` を追加した。インベントリ `workspace_declares_prefetch_names_only_in_tensor_core_data` は再エクスポートが宣言でないため維持。`data_module_reexports_exactly_expected_surface` の期待集合は 17 名から 20 名へ更新した。
- `data_types_are_reachable_via_facade_only` を拡張（`num_workers` が 0／2 の双方）し、`crates/facade/src/data.rs` に利用例 doctest を追加した。統合テスト `data_loader_prefetch.rs`・`#[ignore]` ベンチ `data_loader_prefetch_bench.rs` は `fandhe_ai::data` 経由の import へ切り替えた。
- CUDA／Metal 実機 parity は対象外（ホスト側で完結し `Op`／`BackendOps`／VJP を経由しないため。#2505 と同じ扱い）。承認日は Issue #2506 記載のとおり。

## 5. 保留 doctest 足場を追加しない理由

上記§4 の説明のとおり、`PrefetchHoldDoctestGuard` のような正のプローブ doctest は本イシューの新規公開名（汎用的なメソッド名）に対しては検出力を持たないため追加していない。ソース走査ガード（`facade_does_not_reexport_or_declare_prefetch`）とインベントリ（`workspace_declares_prefetch_names_only_in_tensor_core_data`）の 2 層で「facade 未公開」状態を機械固定する。

## 6. 対象外（out-of-scope。`out-of-scope-tracking.md` に従い記録）

- transform／collate フック（`HookedDataLoader` 相当）の並列化と、worker ごとの `Generator`（#2156）による RNG 分離。
- persistent workers（epoch をまたぐ worker スレッドの再利用）・`pin_memory`。
- GPU DMA prefetch・デバイス常駐データセット（GPU 上でのバッチ切り出し）。
- iterable-style dataset。
- `Sequential::fit` への結線（§4・§8。facade 公開は #2506 で実施済み）。

## 7. セキュリティ考慮（OWASP Top 10）

- **A03／A04（入力検証と資源枯渇）**: worker が処理する添字は必ず `Dataset::batch` の境界検査（`IndexOutOfRange`）を通る。`prefetch_depth >= 1` と `num_workers <= PREFETCH_MAX_WORKERS` を構築時に検証し、有界な in-flight 窓によりメモリ使用はおおよそ `prefetch_depth × batch` に収まる。チャネル操作・`join`・スレッド生成に `unwrap`／`expect` を使わず、型付きの `DataError` として返す。worker の panic は `WorkerFailed` として位置付きで伝え、`Drop` 時にも panic を波及させない（`join` の結果を捨てる）。
- **A06（脆弱・古いコンポーネント）**: 依存の追加・更新はない（`Cargo.toml`／`Cargo.lock` は変わらない）。rayon の結線は承認事項として保留する（§8）。
- **A08（データ整合性）**: reorder buffer により、学習データの順序が seed で決まる順序から黙ってずれることはない。これを bit 一致テスト（`crates/tensor-core/src/data.rs::tests`）とベンチの checksum の hard assert（`docs/perf/logs/data-loader-prefetch-2183/README.md`）で固定する。
- **並行処理の安全性**: `unsafe` は使わない。`Send`／`Sync` の境界でデータ競合をコンパイル時に防ぐ（`D: Send + Sync + 'static`、`D::Batch: Send + 'static`）。worker はグローバル RNG に触れない（§2）。

## 8. 承認事項（本 PR では実施しない）

- `rayon.workspace = true` を `fandhe-ai-tensor-core` の依存へ追加し、worker pool を rayon へ移すこと（manifest の変更。イシュータイトルが挙げる方式だが依存管理規約上の承認事項）。
- facade（`fandhe_ai::data`）への `PrefetchConfig`／`PrefetchDataLoader`／`PrefetchBatches` の再エクスポート（`docs/compat-api-scope.md` §5 経路 2）。→ **#2506 で実施済み**（§4）。
- `Sequential::fit` への結線（`FitConfig` に `num_workers`／`prefetch_depth` を追加。facade 公開面の拡張）。

## 9. 実装記録

- `crates/tensor-core/src/data.rs`: `DataError` へ 3 variant（`InvalidPrefetchConfig`／`WorkerSpawn`／`WorkerFailed`）追加・`PREFETCH_MAX_WORKERS`・`PrefetchConfig`・`PrefetchDataLoader<D>`・`PrefetchBatches<D>`（worker ループ・reorder buffer・`Drop`）を新設。単体テスト 12 件追加（`(workers, depth)` 全組み合わせでの `SamplerDataLoader`／`DataLoader{shuffle=true}` との bit 完全一致・epoch 後 RNG 状態一致・エラー位置一致・途中 drop の非デッドロック確認・`num_workers=0` の逐次経路・`start_epoch` 失敗の 1 回限り yield・タプルデータセットでの添字整合・`size_hint` 契約・構築時検証・`Send` 境界）。
- `crates/facade/tests/api_surface.rs`: `facade_does_not_reexport_or_declare_prefetch`・`workspace_declares_prefetch_names_only_in_tensor_core_data` を追加。
- `crates/facade/tests/data_loader_prefetch.rs`（新規）: `fandhe_ai_tensor_core::data` を直接 import する統合テスト（bit 完全一致確認・ミニバッチ学習ループでの loss 減少・最終パラメータ bit 完全一致）。
- `crates/facade/tests/data_loader_prefetch_bench.rs`（新規・`#[ignore]`）: 性能 A/B（W1: 取得ボトルネック構成、W2: fit 相当の大型データセット学習）。5 run 中央値・checksum hard assert・比率 record_only。実測は `docs/perf/logs/data-loader-prefetch-2183/README.md`。
- facade 新規公開面: なし（既存 6 型のまま不変。#2506 で 3 名を公開。§4 参照）。新規 `Op`／`BackendOps`／VJP なし。

## 10. レビュー是正（PR #2315・codex-review P1 2 件）

- **`PrefetchBatches::next` の逐次経路（`num_workers == 0`）で `.expect()` を使っていた**（`.claude/rules/coding-rust.md` の panic 禁止規約違反）。是正: `dataset: Arc<D>`・`sampler: &'s mut dyn Sampler` を `PrefetchBatches` の非 `Option` フィールドにし（構築契約により両方とも構築時に必ず存在する値へ変更）、`.expect()` を要する分岐自体を削除した。
- **`PrefetchDataLoader::iter` が `Sampler::next_batch` を空の番兵が出るまで呼んで epoch 全体の添字列を `Vec<Vec<usize>>` として一括 eager 確定していた**ため、`prefetch_depth` が先読み量を抑えるはずの契約に反して大きな epoch でメモリを使い切り得た。是正: 添字生成を `PrefetchBatches` 内部（`spawn`・`advance_after_delivery`）へ移し、`sampler: &'s mut dyn Sampler` を `PrefetchBatches<'s, D>` の生存期間だけ借用する設計に変更。`spawn` は初回投入窓（`prefetch_depth` 個、または `Sampler` が先に尽きるまで）だけ添字を生成してから worker を起動し、以降は `advance_after_delivery`（1 バッチ配送完了ごとに呼ばれる）が次の 1 個だけを生成してタスク投入する。保持する添字は常に高々 in-flight 窓ぶん（`prefetch_depth` 程度）に収まり、`num_workers == 0` の逐次経路も同様に `next()` 呼び出しのたびに 1 個だけ生成する（事前一括確定をしない）。生成される `k` の順序自体は eager 版と変わらないため §2 の決定性契約は不変（`crates/tensor-core/src/data.rs::tests::prefetch_matches_sampler_data_loader_sequential`・`prefetch_random_sampler_matches_shuffle_true_data_loader_and_rng_state` 等の bit 完全一致テストが是正後も green であることを確認済み）。
- 上記の是正に伴い `PrefetchBatches<D>` は `PrefetchBatches<'s, D>`（`sampler` への可変借用を保持するためのライフタイム）へ変更。公開型だが facade へは再エクスポートされていない内部 API のため破壊的変更の扱いは不要（§4・§8）。`PrefetchDataLoader::iter` は返す `PrefetchBatches<'_, D>` が `self.sampler` を可変借用するため、同一ローダーで前の epoch のイテレータが生存している間は次の `iter()` を呼べない（`&mut self` 契約により通常の Rust の借用検査で強制される）。
