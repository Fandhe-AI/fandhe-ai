# DataLoader マルチワーカー prefetch の設計判断記録（#2183）

イシュー #2183「DataLoader マルチワーカー prefetch（rayon）の実装と性能検証」。親 #2131（Phase 5「PyTorch／TF 置き換えの API 網羅」5-A 基盤）。

本ドキュメントは `tensor-core::data` 側実装（PyTorch `torch.utils.data.DataLoader(num_workers=...)` 相当）と、facade 公開面拡張・`Sequential::fit` 結線（いずれも承認事項）の切り分け、および決定性契約（seed のみで決まる出力順序）を記録する。`crates/facade/src/data.rs`（純再エクスポート。無変更）・`crates/facade/src/compat/training.rs`（`FitConfig` 無変更）・`CLAUDE.md`・`docs/spec/`（正本 submodule）・`Cargo.toml`／`Cargo.lock`（全クレート）・tolerance／baseline・ガードレール閾値・CI／hooks は変更しない。

## 0. 結論・段階

- **内部クレート側（`tensor-core::data`）は実装済み**（本 PR）。`PrefetchConfig`（`num_workers`／`prefetch_depth` の検証付き設定）・`PrefetchDataLoader<D>`（`Sampler` の添字を worker スレッドへ分配して並列に `Dataset::batch` を実行し、結果を呼び出し順に並べ直す）・`PrefetchBatches<D>`（そのイテレータ）。
- **rayon ではなく `std::thread`／`std::sync::mpsc` を使う**（§1.1。イシュータイトルは「rayon」と書いているが、本 PR の契約〈`Cargo.toml`／`Cargo.lock` 変更なし・依存追加は承認事項〉と両立しないため意図的に差し替えた。rayon への切り替え自体は §8 の承認事項として記録する）。
- **（#2506 で facade 公開済み。§4・§8 参照。以下は #2183 時点の記録）** **facade 公開（`docs/compat-api-scope.md` §5 経路 2 相当）・`Sequential::fit` への結線はいずれも未承認のまま保留**（fit 結線の推奨案は #2604 で §4 に記録・承認待ち）。#2183・親 #2131 のいずれにも承認コメントは見当たらない（着手時点確認）。よって `crates/facade/src/**` は変更せず、`PrefetchConfig`／`PrefetchDataLoader`／`PrefetchBatches` は一切公開しない（兄弟イシュー #2182〈`docs/tensor-core-data-sampler-hooks-decision.md`〉と同型の保留パターン。ただし本イシューは新規の保留 doctest 足場を追加していない——理由は §5 参照）。既存の facade 公開面は不変。
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
- `Sequential::fit`（`crates/facade/src/compat/training.rs::run_fit`）への結線もしない。`FitConfig` に `num_workers` を足すことは facade 公開面の拡張であり、承認事項として記録するだけにする（§8）。（#2604 で公開形の推奨案を下記小節へ記録。`FitConfig` へのフィールド追加は親 #2603 の契約により採らない）

### #2506 実装記録（facade 公開。親 #2500・ルート #2499）

- `PrefetchConfig`／`PrefetchDataLoader`／`PrefetchBatches` の 3 名を `crates/facade/src/data.rs` に `pub use fandhe_ai_tensor_core::data::{…};` 1 行で素の再エクスポートした（別名・facade 独自の型／関数なし。純再エクスポートの契約は不変）。`PREFETCH_MAX_WORKERS`／`PREFETCH_MAX_DEPTH` は承認範囲外のため公開しない。`Sequential::fit` への結線（#2603）・rayon 化は含まない。
- ガード反転: 否定ガード `facade_does_not_reexport_or_declare_prefetch` を承認形のみを許す正ガード `facade_reexports_prefetch_items_only_in_approved_shape` へ反転し、自己テスト `…_detects_each_category` を追加した。インベントリ `workspace_declares_prefetch_names_only_in_tensor_core_data` は再エクスポートが宣言でないため維持。`data_module_reexports_exactly_expected_surface` の期待集合は 17 名から 20 名へ更新した。
- `data_types_are_reachable_via_facade_only` を拡張（`num_workers` が 0／2 の双方）し、`crates/facade/src/data.rs` に利用例 doctest を追加した。統合テスト `data_loader_prefetch.rs`・`#[ignore]` ベンチ `data_loader_prefetch_bench.rs` は `fandhe_ai::data` 経由の import へ切り替えた。
- CUDA／Metal 実機 parity は対象外（ホスト側で完結し `Op`／`BackendOps`／VJP を経由しないため。#2505 と同じ扱い）。承認日は Issue #2506 記載のとおり。

### #2604 fit 結線の公開形（推奨案・承認待ち。親 #2603・ルート #2499）

**本節は推奨案の記録であり、承認の取得を意味しない。確定形ではない。** `Sequential::fit` 系への結線は未実施のまま（`crates/` は無変更）。実装は承認後に #2605 で行い、承認コメントが付くまで着手しない。

#### 着手時判定

- 調査基準: `origin/main` = `f5cc299b`（2026-10-05）。#2604・#2603 ともコメント 0 件（承認は未取得）。
- `FitConfig` は `#[derive(Debug, Clone, Copy, PartialEq, Eq)]`・全フィールド非公開（`crates/facade/src/compat/training.rs:283-284`。v0.10.0 タグでも同 derive）。親 #2603 の契約は「`FitConfig` へのフィールド追加は破壊的変更として扱い行わない」。
- fit の公開入口は `fit`（:1255）・`fit_with_callbacks`（:1385）・`fit_with_metrics`（:1443）・`fit_with_train_step`（:1545）の 4 本で、すべて非公開の `fit_with_callbacks_named`（:1590）→ `run_fit`（:1837）へ委譲する。`run_fit` は `TensorDataset` のタプルから `DataLoader` を構築して反復する（:1852-1854）。validation は `run_evaluate_with_metrics` が別途 `DataLoader` を構築する（:2397-2399）。`fit_with_weights` は保留中（`docs/compat-fit-sample-weighting-decision.md` §11〜§13）。
- `PrefetchConfig`／`PrefetchDataLoader`／`PrefetchBatches` は #2506 で `fandhe_ai::data` から公開済み。`(TensorDataset<f32>, TensorDataset<f32>)` を `PrefetchDataLoader` へ渡す統合テストが既にある（`crates/facade/tests/data_loader_prefetch.rs`）。
- 決定性契約（§2）: 任意の `(num_workers, prefetch_depth)` で同一 `Sampler` の `SamplerDataLoader` と bit 一致する。
- **性能実測**: W1（`Dataset::batch` が重い合成データ）は prefetch で改善、W2（fit 相当: `TensorDataset` ペア + MLP 学習）は prefetch で遅くなる（record_only・共有機。`docs/perf/logs/data-loader-prefetch-2183/README.md`）。
- fit 結線を対象とする保留ガードは存在しない（`crates/facade/tests/api_surface.rs` にあるのは 3 型の再エクスポートの正ガードと定義元インベントリのみ）。`fit_with_prefetch`／`fit_prefetch`／`fit_loader`／`fit_batches` は `crates/`・`docs/` に出現しない（本記録を除く）。
- 先例との食い違い: `docs/compat-fit-sample-weighting-decision.md` §11.1・§11.5 は「`FitConfig` への非公開フィールド追加は `Copy + Eq` を保つので非破壊」と判定している。一方、本件の契約はフィールド追加を行わないとする。本記録はどちらが正しいかを裁定せず、決定事項 (c) としてユーザーへ提示する。

#### 候補比較

| 案 | 形 | 評価の要点 |
|---|---|---|
| 0: 結線しない | 公開面の追加なし。利用者は `fandhe_ai::data::PrefetchDataLoader` を自前の学習ループで回す（#2506 で可能） | 完全に非破壊で W2 実測と整合。ただし親 #2603 の「結線する」目的は満たさない |
| A: `FitConfig` ビルダー追加 | `FitConfig::prefetch(self, PrefetchConfig) -> Self`（非公開 `Option<PrefetchConfig>`） | 既存 4 入口すべてに効き派生入口と合成できる唯一の案。ただし契約がフィールド追加を禁じるため本記録では不採用（上記の食い違いを (c) で確認） |
| B: 別メソッド | `Sequential::fit_with_prefetch(&mut self, x, y, config: FitConfig, prefetch: PrefetchConfig, validation, callbacks, metrics) -> Result<History, AutodiffError>` | 追加のみで非破壊。新しい型なし（公開済みの `PrefetchConfig` を再利用）。`fit_with_train_step`（公開済み）・`fit_with_weights`（保留）と同じ「派生入口を足す」パターン。弱点は派生入口どうしを合成できないこと、W2 実測では利得が見込めないこと |
| C: ローダーを受け取る入口 | `fit_...(&mut PrefetchDataLoader<D>, epochs, ..)` | 利得が実測された領域（`Dataset::batch` が重い自前データセット）に届く唯一の形。ただし `FitConfig` と `Sampler` の二重指定、`accumulate_steps` の渡し方、引数検査の作り直し、`run_fit` のバッチ供給元の抽象化が要り公開面が大きい。`predict_batches` 側の方針（`docs/facade-predict-batches-phase-metrics-decision.md` §8.4・承認待ち）と一体で決めるべきで本件の範囲外 |
| D: 独自型 | `FitOptions` 等の新設定型と新入口 | 中身は `FitConfig` + `PrefetchConfig` の組でしかなく B に対する利点がない。設計を先取りして固定する |
| E: グローバル設定 | `set_fit_prefetch(..)` | 既存シグネチャは不変だが、グローバル状態が fit の挙動を変え、テスト並列実行・再現性の説明が難しい。不採用 |

Issue が例示する 3 類型との対応: 「`Var` 委譲」は非該当（結線対象は `Sequential` の学習ループで `Var` を経由しない）、「モジュール再エクスポート」は非該当（再エクスポートは #2506 で完了済み。fit 結線は項目の公開ではなく振る舞いの追加）、「独自型」は案 D。

名前衝突の観点:

- facade 内: `fit_with_prefetch` は `crates/facade/src/` に無い。`impl Sequential` の inherent メソッド 1 件の追加で済む。
- 既存の保留群: `FitConfig`／`Sequential` へローカル trait を実装して UFCS で呼ぶ保留プローブ（`use_ema`・`validation_split`・`class_weight`・`sample_weight`・`fit_with_weights` 等）の名前に `fit_with_prefetch` は含まれない見込み。断定は #2605 の実機検証に委ねる。
- 下流利用者: 下流が自前 trait の同名メソッドを `Sequential` に実装している場合は inherent 優先で解決が変わりうる。公開項目の追加として通常 minor の範囲だが、互換性は断定しない。

#### 推奨案（承認待ち）

**案 B（`Sequential::fit_with_prefetch`）を推奨案とする。** 根拠:

1. 親 #2603 が求める「非破壊な別経路」を、新しい型を増やさず満たす最小の形である。
2. 既存の派生入口と同じ追加パターンで、`predict_batches` 記録の「別名メソッドで足す」方針とも向きがそろう。
3. 意味論を既存契約から導出できる。`config.shuffle` に応じて `SequentialSampler`／`RandomSampler` を選び `drop_last` を渡せば、§2 の bit 一致契約により、任意の `(num_workers, prefetch_depth)` で `fit_with_metrics` と `History`・最終パラメータ・epoch 後のグローバル RNG 状態が一致する（#2605 のテストで固定する契約として提案）。
4. 変更が facade に閉じ、新規 `Op`／`BackendOps`／VJP／`unsafe`／依存が不要。

**推奨の限界**（推奨を実測以上に強く見せない）:

- B は性能改善を根拠にした推奨ではない。`fit(x, y, ..)` 系はテンソルから `TensorDataset` を作るしかなく、W2 実測ではこの構成は prefetch で遅くなっている。B が利得を生む条件は現時点で実測されていない。
- 利得を目的にするなら、案 0（今回は見送り）+ 案 C（別ツリーで設計）の組合せが実測と整合する。どちらを取るかはユーザー判断（(a)）。
- B は派生入口どうしを合成できない。合成性を優先するなら案 A だが契約が禁じている（(c)）。

0.10.0 非破壊チェック: 既存 4 入口・`FitConfig`・`History`・`FitTarget` のシグネチャと意味論は不変／追加は `impl Sequential` の inherent メソッド 1 件／エラー型の variant 追加なし／`FitConfig` にフィールドを足さない／tolerance・baseline・依存・閾値・`docs/spec` 不変。

#### ユーザーに決めてほしい事項

- (a) そもそも今 fit へ結線するか（W2 実測を踏まえ案 0 も選択肢）。
- (b) 公開形: 0／A／B／C／D／E のどれか。推奨は B。
- (c) `FitConfig` へのフィールド追加を禁じる契約を維持するか。fit 重み付け記録 §11.5 との食い違いをどう扱うか（維持なら A は不可のまま）。
- (d) メソッド名。推奨は `fit_with_prefetch`。
- (e) 引数構成。推奨は `fit_with_metrics` と同じ並び + `prefetch: PrefetchConfig`（値渡し）。新しい設定型は作らない。
- (f) 適用範囲。推奨は学習ローダーのみ（validation／`evaluate` 内部のローダーは従来どおり）。
- (g) 他の派生入口（`fit_with_train_step`・保留中の `fit_with_weights`）と併用できないことを許容するか。
- (h) エラー型。推奨は既存の `AutodiffError::InvalidArgument` へのマッピングのみ（`DataError::WorkerSpawn`／`WorkerFailed` を含む。新 variant なし）。
- (i) 意味論契約。推奨は「任意の `(num_workers, prefetch_depth)` で `fit_with_metrics` と bit 一致（`History`・最終パラメータ・RNG 状態）」。
- (j) 保留ガード。現在 fit 結線の保留ガードは無い。推奨は #2605 で正ガードを新設する形（承認前は `crates/` を変えない）。
- (k) #2604 の閉じ方と #2605 の着手条件（承認コメントが付いてから着手）。

#### 承認後の実装スケッチ（#2605。本記録では実施しない）

- `crates/facade/src/compat/training.rs`: `fit_with_callbacks_named`／`run_fit` に `prefetch: Option<PrefetchConfig>` を通す（既存 4 入口は `None` で委譲し演算列を変えない）。`Some` のとき `DataLoader` の代わりに `PrefetchDataLoader::new((x_ds, y_ds), sampler, cfg)` を構築する。
- 確認事項（断定しない）: `PrefetchDataLoader` の `Send + Sync + 'static` 境界が `run_fit` の境界で追加なしに満たせるか／`iter(&mut self)` を epoch ごとに呼ぶ借用の形／L-BFGS・AMP・`accumulate_steps` はローダー層と直交し追加の拒否条件が要らない見込み／worker の panic・スレッド生成失敗時に train／eval モードと `compiled` の復元経路を通ること。資源上限（`PREFETCH_MAX_WORKERS`／`PREFETCH_MAX_DEPTH` = 64、`prefetch_depth >= 1`）は `PrefetchConfig::new` の検証をそのまま引き継ぐ。
- `crates/facade/tests/api_surface.rs`: `impl Sequential` 内の `fit_with_prefetch` 宣言がちょうど 1 件であることを固定する正ガードと検出テスト、facade のみを import した到達性テスト。
- テスト: `fit_with_prefetch` と `fit_with_metrics` の bit 一致（`num_workers` 0／2、`shuffle`・`drop_last` の有無）、エラー経路。
- docs: 本記録への実装記録、`docs/compat-api-scope.md` §5 の適用記録、`docs/README.md`。
- 新規演算なし・ホスト側のみで完結するため、CUDA／Metal の新規 `#[ignore]` テストは不要見込み（#2605 で再確認）。`fit_with_prefetch` 経由の性能再計測は任意の申し送り。

#### 本節の位置づけ

承認は未取得。本節の追記は `crates/`・`Cargo.*`・tolerance・`docs/spec` を変更していない。

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
- `Sequential::fit` への結線。当初案「`FitConfig` に `num_workers`／`prefetch_depth` を追加」は親 #2603 の契約（フィールド追加は破壊的変更として扱い行わない）により不採用。公開形の推奨案は §4 の「#2604 fit 結線の公開形」小節に記録（承認待ち。実装は承認後に #2605）。

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
