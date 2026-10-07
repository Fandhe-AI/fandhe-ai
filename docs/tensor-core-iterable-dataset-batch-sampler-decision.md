# IterableDataset・BatchSampler の設計判断記録

イシュー #2662（親 #2660）。`torch.utils.data.IterableDataset`（と `DataLoader(iterable_ds, batch_size=k, drop_last=d)`）・
`torch.utils.data.BatchSampler` 相当を内部クレート `tensor-core::data` に追加した記録。

## 0. 結論・段階

- **内部実装済み**（`crates/tensor-core/src/data/iterable.rs`・`crates/tensor-core/src/data/batch_sampler.rs`）。
- **facade 非公開**（保留）。#2662 に承認コメントは無く、承認依頼先は #2677、公開自体は承認後の #2679。
  `IterableBatchSamplerHoldDoctestGuard`（`crates/facade/src/lib.rs`）と `crates/facade/tests/api_surface.rs` の否定ガードで機械固定した。
- 既存の公開型・trait へは何も足していない（追加のみ。0.10.0 公開 API 非破壊）。`DataError` は `#[non_exhaustive]` のため
  variant（`IterableStream`）追加は非破壊。

## 1. API 設計

公開パスは `fandhe_ai_tensor_core::data::{IterableDataset, IterableDataLoader, IterableBatches, StackSamples, BatchSampler}`。

- `IterableDataset`: `type Sample` と `fn iter_samples(&self) -> Box<dyn Iterator<Item = Result<Sample, DataError>> + '_>`。
  map-style の `Dataset`（`len()`／`batch(&[usize])`）を継承しない別系統の trait。要素を `Result` とするのは供給元の失敗を
  panic ではなく型付きエラー（`DataError::IterableStream { reason }`。実装者が生成する）で伝えるため。長さヒントは持たない
  （長さ不定が定義。`IterableBatches::size_hint` は `(0, None)`、終了後は `(0, Some(0))`）。メソッド名を `iter`／`len` にしないのは、
  `api_surface` のインベントリが `fn` 名で数えるため汎用名では検査不能になるから。
- `StackSamples`: サンプル列 `Vec<Self>` を 1 バッチへ積む trait。`Tensor<T>` は既存 `default_collate` へ委譲（再実装しない）、
  2／3 要素タプルは成分ごとに再帰（`ConcatBatch` と同型）。タプル対応は `docs/dataset-dataloader-design.md` §1 が「f32 特徴量と
  i32 ラベルを同順で取り出す」ことを必須要件としているため。`pub` なのは公開 `impl Iterator for IterableBatches` の `where` 節に
  現れるため（`pub(crate)` だと `private_bounds` が `-D warnings` で失敗する）。
- `IterableDataLoader<D>::new(dataset, batch_size, drop_last)`／`iter()`: `batch_size == 0` は `ZeroBatchSize`。`iter()` は呼ぶたびに
  `iter_samples()` で新しい epoch を始める。`IterableBatches::next` は `batch_size` 件溜まるごとに yield、終了時の端数は
  `drop_last` なら捨てる。ストリーム途中の `Err`・`stack_samples` の失敗では**溜めていた部分バッチを捨てて `Err` を 1 回だけ
  yield し以降は `None`**（fail-closed）。バッファは `Vec::new()` から逐次 push し `with_capacity(batch_size)` を使わない
  （`batch_size` は利用者入力で `usize::MAX` もありうるため。capacity overflow・過大確保を避ける）。
- `BatchSampler::new(order: Vec<usize>, batch_size, drop_last)`: 既存 `Sampler` trait の新しい具象型。利用者が与えた明示的な添字列を
  束ねる。`start_epoch` は `order` を複製して既存 `IndexBatcher::set_order` へ渡す（切り出し式を重複実装せず、変更経路を 1 本に保つ）。
  グローバル RNG 非消費・毎 epoch 同順。`num_batches` は PyTorch の `len(BatchSampler)` と同じ式。添字の範囲検査は `Dataset::batch`
  （`IndexOutOfRange`）が行う（`BatchSampler` はデータセット長を知らない。カスタム `Sampler` と同じ既存契約）。

### 不採用とした案

| 案 | 理由 |
|----|------|
| `IterableDataset: Dataset` とする／`Dataset` へ `iter` 系メソッドを足す | map-style の `len()`／`batch(&[usize])` を満たせない。facade 公開済み trait への追加は保留の迂回 |
| 関連型 `type Iter`（ライフタイムなし）や GAT `type Iter<'a>` | 前者は `&self` を借用できず所有型イテレータを強制する。後者は実装者にクロージャ合成の型名記述を強いる |
| `IterableDataLoader` を `Tensor<T>` サンプル限定にする | 特徴量＋ラベルを同順で供給できず学習に使えない |
| ユーザー指定 collate／transform の builder | iterable では実装者側のイテレータ合成で足りる。既存インベントリ名（`with_collate` 等）と衝突する |
| 「添字 1 件ずつの sampler」trait を新設して `BatchSampler` を載せる／遅延イテレータ・クロージャを受ける | #2182 が sampler とバッチ化を 1 trait へ統合済み。epoch ごとに変わる順や遅延・無限列は既存 `Sampler` の独自実装で表現できる |
| 既存 `DataLoader`／`SamplerDataLoader` へ `with_batch_sampler` 等を追加 | 再エクスポート済み型への inherent メソッド追加は facade 公開面の拡張 |
| `TensorDataset` 等へ `IterableDataset` を impl | 既存公開型へ何も足さない方針（#2661 から踏襲） |

## 2. 対象ファイル

`crates/tensor-core/src/data/iterable.rs`・`crates/tensor-core/src/data/batch_sampler.rs`（新規）・`crates/tensor-core/src/data.rs`
（`mod`／`pub use`／`DataError::IterableStream`）・`crates/tensor-core/tests/fixtures/iterable-batch-sampler-pytorch-reference/`・
`crates/facade/tests/data_iterable_batch_sampler.rs`・`crates/facade/src/lib.rs`（保留ガード）・`crates/facade/tests/api_surface.rs`
（否定ガード）。`crates/facade/src/data.rs`（再エクスポートは 20 名のまま）・全 `Cargo.toml`／`Cargo.lock` は不変。

## 3. PyTorch との差分と fixture の検証範囲

意図的差異:

- `shuffle=True`／`sampler=`／`batch_sampler=` と iterable の併用: torch は実行時 `ValueError`、本実装は型の上で書けない
  （統合テストの `STRUCTURALLY_IMPOSSIBLE` と対応）。
- `len(DataLoader(iterable))`: torch は `__len__` 未定義で `TypeError`、本実装は長さ API を持たない。
- `batch_size=None`（サンプルをそのまま返す）は無い（`iter_samples` を直接回せばよい）。
- `BatchSampler` は `Iterable[int]` ではなく `Vec<usize>` を受け取る。負の `batch_size`・非 bool の `drop_last` は型で排除。
- マルチワーカー時のサンプル重複／`worker_init_fn` によるシャーディングは対象外（`PrefetchDataLoader` へ結線しない）。
- ストリーム途中の失敗は例外ではなく `Err` 要素 1 回で epoch を打ち切る（失敗前に完成したバッチ数は torch と一致）。

fixture の検証範囲: 添字バッチ列・バッチ数（`len(BatchSampler)`）は完全一致、f32 の行は bit 一致を確認したうえで統一複合判定
（`assert_parity`。tolerance 不変）にも通す。乱数 sampler 経由のケースは `randperm` の数列が一致しないため含めない。
`error_cases` は torch の成否と一対一で照合する。

## 4. CUDA／Metal

`Op`／`BackendOps`／VJP を経由しないホスト側ユーティリティのため、専用カーネルと REQ-2 parity は該当しない
（#2182・#2661 記録 §4 と同じ）。`Tape::var` アップロードの round-trip のみ `#[ignore]` で分離し
（`iterable_batch_upload_round_trips_on_{cuda,metal}_tape`）、実機未実測として `docs/perf/logs/iterable-batch-sampler-2662/README.md` へ申し送る。

## 5. facade 公開形の推奨案（1 つ。ルート #2499 の 2026-10-07 コメントで承認・#2679 で公開。§9 参照）

**`fandhe_ai::data` への純再エクスポート**（`Var` 委譲・`Sequential::add_*` は該当しない）。本節は推奨案の記録であり、
承認を得たことの記録ではない。承認後に次を行う。

- `crates/facade/src/data.rs` へ 1 行完結の `pub use fandhe_ai_tensor_core::data::{BatchSampler, IterableBatches, IterableDataLoader, IterableDataset, StackSamples};` を追加する（既存パーサが 1 行完結を要求する）。
- `data_module_reexports_exactly_expected_surface` の期待集合を更新する（#2661 分と合わせた件数は公開時点で確定）。
- `IterableBatchSamplerHoldDoctestGuard` と対応する否定ガード（`facade_does_not_reexport_or_declare_iterable_batch_sampler` 系）を削除または正ガードへ反転する。
- `crates/facade/tests/data_iterable_batch_sampler.rs` の import を `fandhe_ai::data` へ切り替える。

## 6. 対象外

- facade 公開（承認依頼 #2677・公開 #2679）。
- iterable の長さヒント（任意 `__len__` 相当）・`batch_size=None` 相当・`ChainDataset`・ユーザー指定 collate／transform フック。
- iterable のマルチワーカー供給（ワーカー間シャーディング）・`PrefetchDataLoader`／`Sequential::fit` への結線。
- 遅延／無限の添字列を受け取る `BatchSampler`（既存 `Sampler` の独自実装で表現可能）。
- `DistributedSampler`（親 #2660 が Phase 5 対象として除外済み）。
- CUDA（GB10）・Metal 実機での `Tape::var` round-trip 実測。

## 7. セキュリティ考慮（OWASP Top 10）

- A02: 乱数を使わない（グローバル RNG 非消費を単体テストで固定）。
- A03／A04: `batch_size == 0` は構築時に拒否。iterable のバッファは `batch_size` で事前確保しない。積み上げ時の出力要素数は
  `default_collate` 内の `checked_numel_for` で確保前に検査。`BatchSampler` の添字は `Dataset::batch` の境界検査を必ず通る。
  ストリーム途中の `Err`・shape 不一致では部分バッチを出さずに打ち切る（fail-closed）。無限ストリームでも保持するのは 1 バッチ分のみ
  （停止しないストリームの待ちは利用者側の責務）。本番経路で `unwrap`／`expect`／panic なし。
- A05: 既定で facade 非公開。再エクスポート・独自宣言・既存公開型への同名メソッド追加による迂回を doctest プローブとソース走査で検出する。
- A06: 依存の追加・更新なし。torch は fixture 生成用の使い捨て venv のみ（CI・ビルドは Python に依存しない）。
- A08: タプルサンプルは 1 サンプル内で特徴量とラベルが対になり成分間の順ずれが構造上起きない。fixture は生成スクリプトと sha256 を同梱。
- A09: `IterableStream { reason }` は利用者が与える文字列をそのまま保持し、ライブラリ側で環境変数やパスを埋め込まない。`unsafe` なし。

## 8. 実装記録

- 単体テスト（`data::iterable` 12 件・`data::batch_sampler` 9 件）、doctest 1 件（利用例）、統合テスト
  （`data_iterable_batch_sampler`）7 件＋`#[ignore]` 実機 round-trip（CUDA 1・macOS の Metal 1）、`api_surface` 否定ガード 5 件、
  保留 doctest 1 件。
- fixture sha256 は `crates/tensor-core/tests/fixtures/iterable-batch-sampler-pytorch-reference/README.md` を参照
  （実 PyTorch 2.14.0+cpu 実行値）。
- `api_surface` のインベントリ期待値: `iter_samples` 2 件（trait 宣言 1 + src 内テスト用 `FnStream` の impl 1）、
  `stack_samples` 4 件（trait 宣言 1 + `Tensor<T>`／2 要素／3 要素タプルの impl 各 1）。
- ガードの実効性は、`crates/facade/src/data.rs` へ `BatchSampler` の再エクスポートを一時的に足して doctest（E0659）と
  `facade_does_not_reexport_or_declare_iterable_batch_sampler` が落ちることで確認済み（確認後に復元）。

## 9. #2679 実装記録（facade 公開）


- 状態: **§5 の推奨形を #2679 で公開した。** 承認根拠はルート #2499 の 2026-10-07 ユーザー承認コメント（issuecomment-6033824965。「Phase 4（#2625）」節で `docs/compat-api-scope.md` §5.1 の行 26 を各決定記録の推奨形で承認）。本書中の「未承認」「承認依頼は #2677」の記述は、#2679 時点で当該コメントの承認に更新された（#2677 の「承認の記録」コメントの割り振りでは公開は #2679）。承認は推奨形に限り、記録に形が書かれていない点は実装せず承認依頼へ戻す条件つき。
- 公開した識別子: `fandhe_ai::data` へ `pub use fandhe_ai_tensor_core::data::{BatchSampler, IterableBatches, IterableDataLoader};` と
  `pub use fandhe_ai_tensor_core::data::{IterableDataset, StackSamples};`。
- ガード: `IterableBatchSamplerHoldDoctestGuard` は削除し、ソース走査は内部クレートの glob 再エクスポート・型の独自宣言・`iter_samples`／`stack_samples`／`with_batch_sampler` の
  `fn` 宣言の禁止へ縮小した（`tensor-core-dataset-compose-decision.md` §9 と同じ構成）。宣言場所インベントリは維持。`data_iterable_batch_sampler.rs` は `fandhe_ai::data` 経由へ切り替えた。
- 依存・tolerance・baseline・ガードレール閾値・`docs/spec` は変更していない。`fandhe-ai =0.10.0` の既存公開 API・`pub use` 行・署名は変更せず、追加のみ。
