# DataLoader の Sampler・collate・transform フックの設計判断記録（#2182）

イシュー #2182「DataLoader の collate・Sampler・transform フック」。親 #2131（Phase 5「PyTorch／TF 置き換えの API 網羅」5-A 基盤）。

本ドキュメントは `tensor-core::data` 側実装（PyTorch `torch.utils.data.Sampler`／`BatchSampler`／`collate_fn`／`Dataset` の `transform` 相当）と、facade 公開面拡張（承認事項）の切り分けを記録する。`crates/facade/src/data.rs`（純再エクスポート。無変更）・`crates/facade/src/lib.rs`（保留 doctest 足場 `DataHooksHoldDoctestGuard` の追加を除く）・`tests/api_surface.rs`（否定ガード 4 件の追加を除く）・`CLAUDE.md`・`docs/spec/`（正本 submodule）・`Cargo.toml`／`Cargo.lock`・tolerance／baseline・ガードレール閾値・CI／hooks は変更しない。

## 0. 結論・段階

- **内部クレート側（`tensor-core::data`）は実装済み**（本 PR）。`Sampler` trait（`SequentialSampler`／`RandomSampler`／`WeightedRandomSampler` の 3 実装）・`SamplerDataLoader<D>`・`HookedDataLoader<T>`（`TransformFn`／`CollateFn`・既定 `default_collate`）。
- **（#2505 で公開済み。§8 参照。以下は #2182 時点の記録）facade 公開（承認事項・`docs/compat-api-scope.md` §5 経路 2）は未承認のまま保留**。イシュー #2182 本文には facade 公開の言及があるが、承認前の実施を禁じる契約であり、#2182・親 #2131 のいずれにも承認コメントは見当たらない（着手時点確認）。よって `crates/facade/src/**` は保留 doctest 足場（`DataHooksHoldDoctestGuard`）を追加するのみで、`Sampler` 系 6 型・`TransformFn`／`CollateFn`／`default_collate` は一切公開しない（兄弟イシュー #2156〈`docs/rng-distributions-generator-decision.md`〉と同型の保留パターン）。既存の facade 公開面（`fandhe_ai::data::{Batches, DataError, DataLoader, DataLoaderConfig, Dataset, TensorDataset}` の 6 型）は不変。
- **本 PR のマージで #2182 は COMPLETED とする**（前例と同じ「保留記録を残した PR のマージで issue をクローズし、承認が得られたら新規 issue か reopen で経路 2 を実施する」方針）。facade 公開面（`fandhe_ai::data::{Sampler, SequentialSampler, RandomSampler, WeightedRandomSampler, SamplerDataLoader, SamplerBatches, HookedDataLoader, HookedBatches, TransformFn, CollateFn, default_collate}`）の実装着手は、`docs/compat-api-scope.md` §5 経路 2 の承認取得後に別 issue／reopen・別 PR で行う。

## 1. API 設計（`crates/tensor-core/src/data.rs`）

### 1.1 既存 `DataLoader`／`DataLoaderConfig` を拡張しない理由

- `DataLoaderConfig` は `#[derive(Clone, Copy, ...)]`・`pub` フィールドを持つ非 `#[non_exhaustive]` 型。フィールド追加は利用者の構造体リテラル構築を壊す（0.9.0 公開 API の破壊）。`Box<dyn Fn>` フィールドの追加は `Copy` も壊す。
- `DataLoader<D>` は `fandhe_ai::data::DataLoader` として facade から再エクスポート済みのため、inherent メソッド追加（`with_transform` 等）は型名を名指ししなくても facade 利用者から呼べてしまい、承認前に facade 公開面を拡張したことになる。
- `Send`／`Sync` 境界のない `Box<dyn Fn>` フィールド追加は auto trait を黙って失う（これも API 破壊）。
- 結論として `tensor-core::data` に新しい型を追加し、`DataLoader` への統合はしない（最終形。暫定の迂回ではない）。

### 1.2 タプルデータセット対応: 2 型構成（案 a）を採用

`docs/dataset-dataloader-design.md` §1 は「f32 特徴量と i32 ラベルを同一のシャッフル順で取り出せること」を必須要件としている。`Sampler` を `TensorDataset<T>` 専用にすると分類タスクで使えなくなるため、次の 2 型に分ける。

- **`SamplerDataLoader<D: Dataset>`**: `Sampler` が返す添字列をそのまま `Dataset::batch(&indices)` に渡す。タプルデータセット（`(TensorDataset<f32>, TensorDataset<i32>)` 等）でも同じ添字が全成分へ適用される。
- **`HookedDataLoader<T: Element>`**: 内部に `SamplerDataLoader<TensorDataset<T>>` を持ち（合成）、`transform: Option<TransformFn<T>>`・`collate: Option<CollateFn<T>>` を加える。issue のコールバックシグネチャが単一の `Tensor` 型のため、フック付きの経路は `TensorDataset<T>` 限定。

トレードオフ: タプルデータセットでのサンプル単位 transform／collate は対象外（§6「対象外」参照）。

### 1.3 `Sampler` trait

```rust
pub trait Sampler: Send {
    fn start_epoch(&mut self) -> Result<(), DataError>;
    fn next_batch(&mut self) -> Vec<usize>;
    fn num_batches(&self) -> Option<usize> { None }
}
```

- `start_epoch` は epoch 開始時に 1 回呼ぶ。抽選が必要な実装（`RandomSampler`・`WeightedRandomSampler`）はここで 1 回の `with_global_rng` クロージャ内にまとめて抽選する（既存の「シャッフル契約」と同じ「複数値をまとめて引く操作の原子性」）。
- `next_batch` の空 `Vec` は epoch 終了の番兵（空バッチは yield しない）。
- `Send` 境界は将来のマルチワーカー化に備えたもの（§6 参照）。
- 3 実装は共通の private `IndexBatcher`（既存 `Batches::next` と同じ切り出し式）を経由し、`batch_size == 0` は `DataError::ZeroBatchSize`。
  - `SequentialSampler::new(len, batch_size, drop_last)`: RNG を一切消費しない。`checked_numel_for::<usize>(&[len])` を検査してから `0..len` を確定する。
  - `RandomSampler::new(len, batch_size, drop_last)`: 既存 private `shuffled_indices` を 1 回の `with_global_rng` 内で呼ぶ。同一 `manual_seed` の下で `DataLoader{shuffle=true}` と添字順が**bit 完全一致**する。
  - `WeightedRandomSampler::new(weights: Vec<f32>, num_samples, replacement, batch_size, drop_last)`: 構築時に `crate::rng::validate_multinomial`（`fn` → `pub(crate)` へ変更。RNG 消費なし）で検証する。`start_epoch` では公開 API `crate::rng::multinomial`（1 回の `with_global_rng` を使う原子的な抽選）を呼ぶ。同一シードの下で `rng::multinomial(&weights, num_samples, replacement)` と抽選列が**bit 完全一致**する。`num_samples == 0` は RNG を消費しない。

### 1.4 callback 型と既定 collate

本番経路で `unwrap`／`expect` を使わない規約（`.claude/rules/coding-rust.md`）のため、内部には fallible な型を保存する。issue の infallible シグネチャは `Ok` で包む builder（`with_transform`／`with_collate`）で満たし、fallible な `with_try_transform`／`with_try_collate` を併設する。

```rust
pub type TransformFn<T> = Box<dyn Fn(Tensor<T>) -> Result<Tensor<T>, DataError> + Send + Sync>;
pub type CollateFn<T>   = Box<dyn Fn(&[Tensor<T>]) -> Result<Tensor<T>, DataError> + Send + Sync>;
pub fn default_collate<T: Element>(samples: &[Tensor<T>]) -> Result<Tensor<T>, DataError>;
```

- サンプルの取り出しには `TensorDataset::get` を使わない。`get` は `reshape` を使うため非 contiguous な dataset（`transpose` した view 等）で `NonContiguousReshape` になりうる。代わりに private `sample_owned(idx)` を新設し、`narrow(0, idx, 1)` → `host_slice()` → `Tensor::new` の順で組み立てる（`gather_rows` と同じく境界検査を先に行う）。
- **fast path**: transform も collate も未設定のときは `Dataset::batch(&indices)`（`gather_rows`）へ直行する（コピー 1 回）。fast path と slow path（恒等 transform を明示した場合）の出力は bit 一致する（`hooked_default_collate_is_bit_identical_to_dataset_batch` で固定）。
- **処理順（AC-4）**: `HookedBatches::next` は「`sampler.next_batch()` → 各添字を `sample_owned` → `transform` → `Vec<Tensor<T>>` へ集める → collate（未設定なら `default_collate`）」の順で処理する。

### 1.5 `DataError` への variant 追加

`#[non_exhaustive]` のため追加は非破壊。先例は #2156（`RngError` に variant を追加）と同型。

- `Rng(crate::rng::RngError)` と `impl From<RngError> for DataError`。
- `SampleShapeMismatch { position: usize, expected: Vec<usize>, found: Vec<usize> }`
- `EmptyBatch`

## 2. 対象ファイルと変更概要

| パス | 変更内容 |
|------|----------|
| `crates/tensor-core/src/rng.rs` | `validate_multinomial` を `fn` → `pub(crate) fn`（可視性のみ）。呼び出し文脈のコメント追記 |
| `crates/tensor-core/src/data.rs` | §1 の実装（`DataError` 3 variant＋`From<RngError>`・`Sampler` trait・`IndexBatcher`・3 sampler・`SamplerDataLoader`／`SamplerBatches`・`TensorDataset::sample_owned`・`TransformFn`／`CollateFn`／`default_collate`・`HookedDataLoader`／`HookedBatches`）。単体テスト 14 件追加 |
| `crates/facade/src/lib.rs` | `DataHooksHoldDoctestGuard`（`#[cfg(doctest)]` 保留足場。正のプローブ 1 ブロック方式。`RngDistributionsHoldDoctestGuard` と同型） |
| `crates/facade/tests/api_surface.rs` | `data_hooks_hold_doctest_globs_all_pub_modules`・`data_hooks_hold_doctest_probe_body_matches_fixed_contract`・`facade_does_not_reexport_or_declare_data_hooks`・`workspace_declares_data_hooks_names_only_in_tensor_core_data` の 4 テスト |
| `crates/facade/tests/data_sampler_hooks.rs`（新規） | facade 未公開の内部クレートを直接 import する統合テスト 3 件（`data_loader.rs` と同型のパターン） |
| `docs/tensor-core-data-sampler-hooks-decision.md`（本ファイル） | 設計判断記録 |
| `docs/dataset-dataloader-design.md` | §11「追補（#2182）」を新設 |
| `docs/compat-feature-gap.md` | §2.14「データ」の #1615 追補直後に #2182 追補を追加 |
| `docs/compat-api-scope.md` | l.234「Dataset／DataLoader」行へ #2182 の追補を追加 |
| `docs/README.md` | 索引に本 doc の行を追加 |

facade `src/data.rs`（6 型の純再エクスポート）は無変更。

## 3. PyTorch との差分（意図的な差異）

- PyTorch の `Sampler`（添字を 1 件ずつ返す）と `BatchSampler`（それをバッチへ束ねる）を、本実装では 1 つの「バッチ sampler」trait（`Sampler::next_batch` がバッチ添字列を直接返す）へ統合している。Rust では trait object を介した 2 段合成よりも単純な形になるため。
- `WeightedRandomSampler` の内部添字は `i32`（`crate::rng::multinomial` の出力 dtype。本リポの index／targets 型契約に合わせる。#2156 と同じ意図的差異）で、`usize` へ変換してから使う。
- transform は Dataset 側ではなくローダー側（`HookedDataLoader`）に置く。PyTorch は `Dataset.__getitem__` 内で transform を適用する設計だが、本リポの `Dataset` trait は「添字列 → バッチ」を中心に置く設計（`data.rs` モジュール doc）のため、サンプル単位の変換はローダー側の責務とする方が既存設計と整合する。

## 4. CUDA／Metal（バックエンド別カーネル）

該当なし。`Sampler`／`HookedDataLoader` はホスト側だけで完結するデータ供給ユーティリティで `Op`／`BackendOps`／VJP を一切経由しない（`data.rs` モジュール冒頭の既存注記と同じ理由）。バックエンド別カーネルを持たないため REQ-2 複合判定の対象自体が無く、`#[ignore]` テスト・`docs/perf/logs` への申し送りは不要（`docs/dataset-dataloader-design.md` §7 の既存判断を踏襲）。

## 5. facade 保留と承認依頼用の事前設計

> **#2505 で実施済み**（ルート #2499 の 2026-10-04 ユーザー承認〈Phase 1〜3 の facade 公開を設計判断記録の推奨形で実装してよい〉に基づく）。以下は当時の事前設計の記録であり、実装結果は §8「#2505 実装記録」を参照。

承認後に `crates/facade/src/data.rs` へ追加する想定の `pub use` 行（純再エクスポート方式。`DataLoader` への統合はしない設計を維持）:

```rust
pub use fandhe_ai_tensor_core::data::{
    Sampler, SequentialSampler, RandomSampler, WeightedRandomSampler,
    SamplerDataLoader, SamplerBatches, HookedDataLoader, HookedBatches,
    TransformFn, CollateFn, default_collate,
};
```

承認取得後に外す保留ガード:

- `crates/facade/src/lib.rs::DataHooksHoldDoctestGuard`（本 doctest 全体を削除）。
- `crates/facade/tests/api_surface.rs` の否定ガード 4 件（`data_hooks_hold_doctest_globs_all_pub_modules`・`data_hooks_hold_doctest_probe_body_matches_fixed_contract`・`facade_does_not_reexport_or_declare_data_hooks`・`workspace_declares_data_hooks_names_only_in_tensor_core_data`）を、`data_module_reexports_exactly_expected_surface`（既存。期待集合を 6→17 名へ更新）と同型の正のガードへ置き換える。
- `crates/facade/tests/data_sampler_hooks.rs` の import を `fandhe_ai_tensor_core::data::*` から `fandhe_ai::data::*` へ切り替える。

## 6. 対象外（out-of-scope。`out-of-scope-tracking.md` に従い記録）

- タプルデータセットでのサンプル単位 transform／collate（§1.2 のトレードオフ。成分ごとの型を持つフックは型設計が別途必要）。
- マルチワーカー prefetch は #2183 で実装済み（`docs/tensor-core-data-prefetch-decision.md`）。persistent workers は #2183 でも対象外のまま。**イシュー本文のスコープ外欄が追跡先として挙げる「#2181」は番号の誤記**（#2181 は実際には AMP の `DeviceParamStore` 結線〈PR #2309 でマージ済み・クローズ済み〉であり無関係）。正しい追跡先は #2183 だった。
- `DistributedSampler`（分散学習向け sampler）・iterable-style dataset・`pin_memory`。
- facade 公開（§0・§5 参照。ユーザー承認待ち）。

## 7. セキュリティ考慮（OWASP Top 10）

- **A02 暗号化の失敗**: `RandomSampler`／`WeightedRandomSampler` は xorshift64* を使う（暗号学的に安全な PRNG ではない）。秘匿用途に使わないことをモジュール doc・型 doc に明記（`rng.rs`・`data.rs` の既存注記と同じ制約）。
- **A03／A04**: sampler が返す添字（ユーザー実装の `Sampler` を含む）は `Dataset::batch`／`sample_owned` の境界検査を必ず経由し、範囲外は `DataError::IndexOutOfRange` で fail-closed に拒否する（境界検査を省略しない。REQ-8 の趣旨）。添字列・サンプル `Vec`・collate 出力の各確保は `checked_numel_for` で事前検査する。本番経路で `unwrap`／`expect` を使わない（`i32 → usize` の変換も `try_from` で行う）。`WeightedRandomSampler` の重みの非有限値・負値・総和 0 は構築時に拒否し、この時点では RNG を消費しない。
- **A06**: 依存の追加・更新はない（`Cargo.toml`／`Cargo.lock` 不変）。
- **A08**: タプルデータセットの長さ不一致は `SamplerDataLoader::new` の `validate()` で拒否し（既存 `DataLoader::new` と同じ契約）、成分間で添字がずれた学習データが黙って供給されることを防ぐ。facade 公開の迂回経路（`DataLoader` へのメソッド追加・別名 `pub use`）は保留ガードの doctest と `api_surface` の多層検査で CI 上で検出する。
- **ユーザー callback**: transform／collate はユーザーのコードをそのまま実行する（信頼境界はライブラリ利用者側にある）。ライブラリ自身は panic 経路を持たない。`Send + Sync` 境界により将来の並列化時のデータ競合をコンパイル時に防ぐ。
- `unsafe` は新たに使わない。

## 8. 実装記録

- `crates/tensor-core/src/rng.rs`: `validate_multinomial` を `pub(crate)` 化。
- `crates/tensor-core/src/data.rs`: `Sampler` trait・`IndexBatcher`（private）・`SequentialSampler`／`RandomSampler`／`WeightedRandomSampler`・`SamplerDataLoader`／`SamplerBatches`・`TensorDataset::sample_owned`（private）・`TransformFn`／`CollateFn`／`default_collate`・`HookedDataLoader`／`HookedBatches`・`DataError` 3 variant 追加。単体テスト 14 件（RNG 消費順序の bit 一致・`num_batches` 整合・タプル添字共有・境界外添字の fail-closed 拒否・`start_epoch` エラーの単発 yield・fast/slow path の bit 一致・パイプライン順序のイベントログ検証・collate エラー伝播・`Send` 確認）。
- `crates/facade/src/lib.rs`: `DataHooksHoldDoctestGuard`（正のプローブ doctest）。
- `crates/facade/tests/api_surface.rs`: 否定ガード 4 件を追加（既存 `data_module_reexports_exactly_expected_surface` 等の 3 テストは不変）。
- `crates/facade/tests/data_sampler_hooks.rs`（新規）: `HookedDataLoader` と既存 `DataLoader` の bit 完全一致・`SamplerDataLoader`（タプル）の分類スモーク・`WeightedRandomSampler` + transform + collate の組み合わせ学習ループ（loss 減少確認）の 3 件。
- facade 新規公開面: なし（既存 6 型のみ。保留）。

### #2505 実装記録（facade 公開。親 #2500・ルート #2499）

- 公開した 11 名: `Sampler`・`SequentialSampler`・`RandomSampler`・`WeightedRandomSampler`・`SamplerDataLoader`・`SamplerBatches`・`HookedDataLoader`・`HookedBatches`・`TransformFn`・`CollateFn`・`default_collate`。`crates/facade/src/data.rs` に純再エクスポート（別名なし）。`DataLoader`／`DataLoaderConfig` は不変で、統合もしない。
- §5 のコードブロックは複数行だが、既存 `data_module_reexports_exactly_expected_surface` のパーサが 1 行完結の `pub use …::{…};` を要求するため、単一行 4 本に分けて記述した（パーサは拡張していない）。
- 削除: `DataHooksHoldDoctestGuard`（lib.rs）・doctest ドリフト検査 2 件・`DATA_HOOKS_HOLD_PROBE_BODY`。
- 反転: `facade_does_not_reexport_or_declare_data_hooks` → `facade_reexports_data_hooks_items_only_in_approved_shape`（＋自己テスト）。承認形（`fandhe_ai_tensor_core::data::` 接頭辞・別名なし）が `src/data.rs` にちょうど 1 回ずつ存在することを固定し、facade 独自の型宣言と `with_transform` 等の `fn` 宣言（`DataLoader` への統合）は引き続き違反とする。
- 維持: `workspace_declares_data_hooks_names_only_in_tensor_core_data`（定義元インベントリ。再エクスポートは宣言ではないため反転後も真）。
- 更新: `data_module_reexports_exactly_expected_surface` の期待集合を 6 → 17 名へ。`data_types_are_reachable_via_facade_only` を Sampler／フック系に拡張。`facade_pub_use_leaves_are_not_modules` の小文字葉 allowlist に `default_collate` を追加。`data_sampler_hooks.rs` の import を `fandhe_ai::data` 経由へ切替。
- 利用例 doctest を `crates/facade/src/data.rs` に追加。
- GPU parity: 本機能はホスト側で完結し `Op`／`BackendOps`／VJP を経由しない（§4）ため、実機 parity テスト・`docs/perf/logs` の申し送りは不要。
