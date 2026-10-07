# Dataset 合成ユーティリティ（Subset・ConcatDataset・random_split）の設計判断記録

イシュー #2661（親 #2660）。`torch.utils.data.{Subset, ConcatDataset, random_split}` 相当を内部クレート
`tensor-core::data` に追加した記録。

## 0. 結論・段階

- **内部実装済み**（`crates/tensor-core/src/data/compose.rs`）。
- **facade 非公開**（保留）。#2661 に承認コメントは無く、承認依頼先は #2677、公開自体は承認後の #2679。
  `DatasetComposeHoldDoctestGuard`（`crates/facade/src/lib.rs`）と `crates/facade/tests/api_surface.rs` の否定ガードで機械固定した。
- 既存の公開型・trait へは何も足していない（追加のみ。0.10.0 公開 API 非破壊）。`DataError` は `#[non_exhaustive]` のため
  variant 追加は非破壊。

## 1. API 設計

公開パスは `fandhe_ai_tensor_core::data::{Subset, ConcatDataset, ConcatBatch, random_split, random_split_fractions}`。

- `Subset<D: Dataset>`: `Arc<D>`＋添字列。`random_split` が 1 つの元から複数の `Subset` を返すため共有が必須。構築時に
  `validate()` と全添字の境界検査（`IndexOutOfRange`）を行う。重複・空は許容。入れ子は合成で成立する。
- `ConcatDataset<D: Dataset>`: `Vec<D>`＋`cumulative_sizes`（PyTorch と同じ累積和。`checked_add`）。位置決定は
  `partition_point(|&c| c <= idx)`（`bisect_right` 同値。長さ 0 の成分は飛ばされる）。空 `Vec` は `EmptyConcat`。
- `ConcatBatch` trait: `Dataset::batch` が不透明な `Self::Batch` を返すため、成分をまたぐ添字列を束ねる手段として
  先頭軸連結 `concat_batches` を `Tensor<T>`・2／3 要素タプルへ実装。`pub` なのは公開 `impl Dataset for ConcatDataset<D>` の
  `where` 節に現れるため（`pub(crate)` だと `private_bounds` が `-D warnings` で失敗する）。
- 連結は **同一成分が連続する run ごと**に取り出して元の順で連結する（成分ごとにまとめると行順が入れ替わるため不採用）。
- `random_split(dataset, &[usize])`（個数指定）／`random_split_fractions(dataset, &[f64])`（割合指定）。割合は
  `floor(n*frac)` に余りを先頭から round-robin で配分し、合計は CPython 3.12+ と同じ Neumaier 補償和で求め
  `isclose(sum, 1)`（rel_tol=1e-9）かつ `sum <= 1` を要求する（torch 2.14.0 の `dataset.py` を読んで確定）。

### 不採用とした案

| 案 | 理由 |
|----|------|
| 所有型 `Subset` | `random_split` が複数の `Subset` を返すため共有が要る |
| `&D`／`Arc<D>`／`Box<dyn Dataset>` への blanket impl | facade から名指しなしで到達できる公開面拡張になる（#2182 記録 §1.1 と同じ判断） |
| 成分ごとにまとめる連結 | 行順が入れ替わり PyTorch と不一致 |
| `Dataset` へのメソッド追加／既存ローダーへの inherent メソッド | 同上（保留の迂回になる） |

## 2. 対象ファイル

`crates/tensor-core/src/data/compose.rs`（新規）・`crates/tensor-core/src/data.rs`（`mod`／`pub use`／`DataError` variant 4 件）・
`crates/tensor-core/tests/fixtures/dataset-compose-pytorch-reference/`・`crates/facade/tests/data_dataset_compose.rs`・
`crates/facade/src/lib.rs`（保留ガード）・`crates/facade/tests/api_surface.rs`（否定ガード）。`crates/facade/src/data.rs`
（再エクスポートは 20 名のまま）・全 `Cargo.toml`／`Cargo.lock` は不変。

## 3. PyTorch との差分と fixture の検証範囲

意図的差異:

- `randperm` の数列は一致しない（RNG は xorshift64*。#2156 の RNG 決定と同じ）。同一 `manual_seed` の下で分割した添字列の
  連結は `RandomSampler`／`DataLoader{shuffle=true}` の順列と bit 完全一致する。
- `Subset` は範囲外添字を構築時に拒否する（torch は参照時に遅延失敗。fail-closed）。
- 整数長版と割合版を型で分ける（torch は合計が 1 に近い整数列を割合として解釈する）。
- 長さ 0 の分割は警告なしで許容する。負添字はない（`usize`）。
- `ConcatDataset` は同型の `Vec<D>` 限定（異種成分は全域 `Subset` で型を揃える）。

fixture の検証範囲: 整数表（長さ列・`cumulative_sizes`・添字）は完全一致、f32 の行は bit 一致を確認したうえで統一複合判定
（`assert_parity`。tolerance 不変）にも通す。乱数列そのものは比較対象外。`error_cases` は torch の成否と一対一で照合し、
意図的差異は `INTENDED_DIFFS` に記録する。

## 4. CUDA／Metal

`Op`／`BackendOps`／VJP を経由しないホスト側ユーティリティのため、専用カーネルと REQ-2 parity は該当しない
（#2182 記録 §4 と同じ）。`Tape::var` アップロードの round-trip のみ `#[ignore]` で分離し
（`composed_batch_upload_round_trips_on_{cuda,metal}_tape`）、実機未実測として `docs/perf/logs/dataset-compose-2661/README.md` へ申し送る。

## 5. facade 公開形の推奨案（1 つ。ルート #2499 の 2026-10-07 コメントで承認・#2679 で公開。§9 参照）

**`fandhe_ai::data` への純再エクスポート**（`Var` 委譲・`Sequential::add_*` は該当しない）。承認後に次を行う。

- `crates/facade/src/data.rs` へ 1 行完結の `pub use fandhe_ai_tensor_core::data::{ConcatBatch, ConcatDataset, Subset, random_split, random_split_fractions};` を追加する（既存パーサが 1 行完結を要求する）。
- `data_module_reexports_exactly_expected_surface` の期待集合を 20 → 25 名へ更新し、小文字葉 allowlist へ `random_split`・`random_split_fractions` を追加する。
- `DatasetComposeHoldDoctestGuard` と対応する否定ガード（`facade_does_not_reexport_or_declare_dataset_compose` 系 5 件）を削除または正ガードへ反転する。
- `crates/facade/tests/data_dataset_compose.rs` の import を `fandhe_ai::data` へ切り替える。

## 6. 対象外

- `Generator` 指定版 `random_split`（PyTorch の `generator=`）。
- 異種成分の連結（`Box<dyn Dataset>` 対応は公開面拡張＝承認事項）。
- `IterableDataset`／`BatchSampler`（#2662）・`DistributedSampler`。

## 7. セキュリティ考慮（OWASP Top 10）

- A02: 分割順は非暗号学的な xorshift64*。秘匿用途に使わない。
- A03／A04: 添字は構築時と `batch` 時の二重で境界検査。確保前に `checked_numel_for`、累積和は `checked_add`／`checked_sub`、
  割合は有限性・範囲を先に検査。`ConcatBatch` は trailing shape を検査してからコピー。本番経路で `unwrap`／`expect` なし。
- A05: 既定で facade 非公開。既存公開型へのメソッド追加や blanket impl による迂回公開をガードで検出する。
- A06: 依存の追加・更新なし。torch は fixture 生成用の使い捨て venv のみ（CI・ビルドは Python に依存しない）。
- A08: 検査失敗時と `n == 0` で RNG を消費しない。タプルの長さ不一致は `validate()` 経由で拒否される。fixture は生成スクリプトと
  sha256 を同梱。`unsafe` なし。

## 8. 実装記録

- 単体テスト（`data::compose`）9 件、統合テスト（`data_dataset_compose`）8 件＋`#[ignore]` 実機 round-trip（CUDA 1・macOS の Metal 1）、
  `api_surface` 否定ガード 5 件、doctest 1 件。
- fixture sha256 は `crates/tensor-core/tests/fixtures/dataset-compose-pytorch-reference/README.md` を参照
  （実 PyTorch 2.14.0+cpu 実行値）。
- ガードの実効性は、`crates/facade/src/data.rs` へ `Subset` の再エクスポートを一時的に足して doctest（E0659）と
  `facade_does_not_reexport_or_declare_dataset_compose` が落ちることで確認済み（確認後に復元）。

## 9. #2679 実装記録（facade 公開）


- 状態: **§5 の推奨形を #2679 で公開した。** 承認根拠はルート #2499 の 2026-10-07 ユーザー承認コメント（issuecomment-6033824965。「Phase 4（#2625）」節で `docs/compat-api-scope.md` §5.1 の行 25 を各決定記録の推奨形で承認）。本書中の「未承認」「承認依頼は #2677」の記述は、#2679 時点で当該コメントの承認に更新された（#2677 の「承認の記録」コメントの割り振りでは公開は #2679）。承認は推奨形に限り、記録に形が書かれていない点は実装せず承認依頼へ戻す条件つき。
- 公開した識別子: `fandhe_ai::data`（`crates/facade/src/data.rs`）へ `pub use fandhe_ai_tensor_core::data::{ConcatBatch, ConcatDataset, Subset};` と
  `pub use fandhe_ai_tensor_core::data::{random_split, random_split_fractions};`。
- ガード: `DatasetComposeHoldDoctestGuard` は型名・自由関数の衝突プローブが承認形と衝突するため削除し、ソース走査は「内部クレートの glob 再エクスポート・型の独自宣言・
  `random_split`／`random_split_fractions`／`subset`／`concat_batches` の `fn` 宣言（facade への inherent メソッド追加経路）の禁止」へ縮小した。承認形の過不足は
  `facade_exposes_phase4_training_data_only_in_approved_shape`、到達性は `phase4_data_types_are_reachable_via_facade_only`。宣言場所インベントリは維持。
  `data_dataset_compose.rs` は `fandhe_ai::data` 経由へ切り替えた。
- 依存・tolerance・baseline・ガードレール閾値・`docs/spec` は変更していない。`fandhe-ai =0.10.0` の既存公開 API・`pub use` 行・署名は変更せず、追加のみ。
