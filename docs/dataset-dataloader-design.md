# Dataset／DataLoader（イシュー #1615・親 #1602）

PyTorch `torch.utils.data.Dataset`／`DataLoader` 相当の薄いバッチ供給
ユーティリティを追加する設計・実装記録。

## 1. 背景・目的

`docs/compat-feature-gap.md` §2.14「データ」行は `Dataset`／`DataLoader`
を「なし」（難度 M。イテレータ・シャッフル・バッチ化の薄い層）と記録し
ていた。既存の学習系テスト（`optim_train_loop.rs`・
`compat_sequential_train.rs`）はフルバッチで `tape.var(&x_data)` を毎
step 登録するのみで、ミニバッチ学習を書く定型手段がなかった。

`#1724`〜`#1726` で `manual_seed`／`randn`／`rand`／`randint`／
`arange` 等の「ホスト側完結の生成系」レイヤー
（`tensor-core::rng`／`creation`）が整備済み。本イシューは同じ層に
**データ供給（batch／shuffle）** を追加し、`manual_seed` 下で epoch
ごとのシャッフル順が再現できる契約を確立する。

本リポの index／targets 契約は `Tensor<i32>`（`Var::cross_entropy_loss`
／`nll_loss`／`gather`／`index_select`）であり、**f32 特徴量＋ i32
ラベルを同一のシャッフル順で取り出せること**を必須要件とする。

## 2. 配置

- 本体: `crates/tensor-core/src/data.rs`。`rng.rs`／`creation.rs` と同じ
  層構造上の理由——必要なのは `Tensor<T>`・`rng::with_global_rng`・
  `checked_numel_for`（`pub(crate)`）のみで、`BackendOps`／`Op`／`Var`
  を一切経由しない。`lib.rs` に `pub mod data;` を追加（トップレベル
  `pub use` は追加しない。モジュール単位で到達させる）。
- `autodiff`: 変更なし（facade は `fandhe-ai-tensor-core` へ直接依存
  済み。`Tensor`／`Device` と同じ直接再エクスポート経路を使う）。
- facade: `crates/facade/src/data.rs`（純再エクスポートのみ）＋
  `lib.rs` に `pub mod data;`。`optim.rs` と同型（facade 独自の型・
  関数を持ち込まない）。`tests/api_surface.rs` に
  `data_module_is_pure_reexport`／
  `data_module_reexports_exactly_expected_surface` を追加して機械固定
  する。

## 3. 公開 API

```rust
pub trait Dataset {
    type Batch;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn validate(&self) -> Result<(), DataError> { Ok(()) }
    fn batch(&self, indices: &[usize]) -> Result<Self::Batch, DataError>;
}

pub struct TensorDataset<T: Element> { /* tensor: Tensor<T> */ }
impl<T: Element> TensorDataset<T> {
    pub fn new(tensor: Tensor<T>) -> Result<Self, DataError>;
    pub fn tensor(&self) -> &Tensor<T>;
    pub fn get(&self, index: usize) -> Result<Tensor<T>, DataError>;
}
impl<T: Element> Dataset for TensorDataset<T> { type Batch = Tensor<T>; ... }

// 異種 dtype／複数列はタプルで表す（同一の添字列を全成分に適用）。
impl<A: Dataset, B: Dataset> Dataset for (A, B) { ... }
impl<A: Dataset, B: Dataset, C: Dataset> Dataset for (A, B, C) { ... }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataLoaderConfig { pub batch_size: usize, pub shuffle: bool, pub drop_last: bool }

pub struct DataLoader<D: Dataset> { /* dataset: D, config: DataLoaderConfig */ }
impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, config: DataLoaderConfig) -> Result<Self, DataError>;
    pub fn len(&self) -> usize;
    pub fn iter(&self) -> Batches<'_, D>;
    // ...
}

pub struct Batches<'a, D: Dataset> { /* ... */ }
impl<D: Dataset> ExactSizeIterator for Batches<'_, D> {}
impl<'a, D: Dataset> IntoIterator for &'a DataLoader<D> { ... }

#[non_exhaustive]
pub enum DataError {
    ZeroBatchSize,
    ScalarTensor,
    LengthMismatch { expected: usize, found: usize, component: usize },
    IndexOutOfRange { index: usize, len: usize },
    Shape(ShapeError),
}
```

### 3.1 タプル方式を採用した理由（異種 dtype）

PyTorch `Dataset.__getitem__`（単一サンプル）ではなく「添字列 →
バッチ」を trait の中心に置く。Rust では collate（サンプルの束ね方）
を型で表現する必要があるため、異種 dtype／複数列は
`(TensorDataset<f32>, TensorDataset<i32>)` のようなタプルで表す。
同一の添字列を全成分に適用するため、シャッフル順が成分間で必ず一致
する（`tuple_batches_share_the_same_permutation` テストで固定）。

`TensorDataset::get` は PyTorch `__getitem__` 相当（1 サンプルを
`shape[1..]` で返す）も別途提供する。

## 4. シャッフル契約（`manual_seed` との整合）

順列は Fisher–Yates（Durstenfeld）で生成し、各抽選は `rng::randint` と
同じ **rejection sampling**（`next_u64` の整数演算のみ）で剰余バイアス
を排除する。整数演算のみのためプラットフォーム横断で bit 同一。

順列生成は 1 回の `with_global_rng` クロージャ内で完結させ（複数値を
まとめて引く操作の原子性。`rng.rs` doc 参照）、`DataLoader::iter` 呼び
出し時に eager に `Vec<usize>` を確定する（`Batches` はロックを持たず
添字を切り出すだけ）。`shuffle=false` はグローバル RNG を**一切消費
しない**（順序は恒等）。空データセット（`len()==0`）はシャッフルの
有無に関わらず抽選を消費しない。

公開 `randperm` は追加しない（内部 private ヘルパー `shuffled_indices`
のみ。`rng-global-contract-design.md` §8 が「対象外」と記録済みの範囲
を維持）。

## 5. バッチ組み立て（`gather_rows`）

`TensorDataset<T>::batch(indices)` の出力 shape は
`[indices.len(), tensor.shape()[1..]]`。行ごとに `narrow(0, idx, 1)` →
`host_slice()`（contiguous なら借用・非 contiguous なら 1 回だけ実体
化）で読み出し `extend_from_slice` する。出力要素数は
`checked_numel_for::<T>` で事前検査してから `Vec::with_capacity` する
（A03/A04 対策）。

`shuffle=false` の連番バッチは `tensor.narrow(0, start, len).contiguous()`
と **bit 完全一致**する（純粋コピーで算術を伴わないため）。

## 6. エッジケース

| 条件 | 挙動 |
|---|---|
| `batch_size == 0` | `DataLoader::new` が `DataError::ZeroBatchSize` |
| rank 0（スカラー）テンソル | `TensorDataset::new` が `DataError::ScalarTensor` |
| タプル成分間の長さ不一致 | `DataLoader::new` の `validate()` が `DataError::LengthMismatch` |
| `indices` が `len()` 以上 | `Dataset::batch` が `DataError::IndexOutOfRange` |
| `drop_last=false` | 最終バッチは端数長（floor+1 バッチ） |
| `drop_last=true` かつ `len < batch_size` | バッチ 0 件 |
| `len() == 0` | バッチ 0 件（shuffle の有無に依らず） |

## 7. 受入基準テンプレートの読み替え（#1602 ツリー共通の注記）

`Dataset`／`DataLoader` はホスト側のデータ供給ユーティリティであり
**Op／VJP／バックエンド別カーネルを持たない**（`torch.utils.data` にも
勾配は無い。`creation.rs`／`rng.rs` と同じ理由）。

- `Op`／`BackendOps`／`Var` メソッド／VJP: **追加しない**。
- parity: 「ホスト側でバッチ化 → CPU `Tape::var` アップロードの bit
  完全一致」（`crates/facade/tests/data_loader.rs::
  batch_upload_to_cpu_tape_is_bit_identical`）＋ CUDA／Metal
  `#[ignore]` round-trip（`batch_upload_round_trips_on_cuda_tape`／
  `batch_upload_round_trips_on_metal_tape`。本エージェント実行環境に
  実機がないため未実測のまま Mac／GB10 セッションへ申し送り）。
  バックエンド別カーネルが存在しないため REQ-2 複合判定の対象自体が
  無い。
- tolerance／baseline: 不変。

## 8. セキュリティ考慮（OWASP Top 10）

- **A02 暗号化の失敗**: xorshift64* は暗号学的に安全な PRNG ではない
  （`rng.rs` と同じ制約）。DataLoader の shuffle は学習再現性のための
  決定的順列であり、セキュリティ用途（トークン・鍵・秘匿順序）に使わ
  ないことをモジュール doc に明記する。
- **A03 インジェクション**: 外部入力（ファイル・文字列）のパースを
  一切行わない。添字はライブラリ内部で生成し、`batch()` で `len()` に
  対する境界検査を必ず行う（`Tensor::narrow` の `NarrowOutOfBounds` に
  も二重に守られる）。
- **A04／整合性**: 出力要素数は `checked_numel_for` で計算し、
  オーバーフローや確保不能を panic ではなく `DataError`／`ShapeError`
  として返す（本番経路で `unwrap`／`expect` を使わない）。
- **A06 脆弱な依存**: 外部依存の追加・更新なし。
- **A08 データ整合性**: タプルデータセットの長さ不一致は
  `DataLoader::new` の `validate()` で fail-closed に拒否し、成分間で
  シャッフル順がずれた学習データが黙って供給されない。

`unsafe` なし。

## 9. 対象外（out-of-scope。`out-of-scope-tracking.md` に従い記録）

- `Sampler`／`BatchSampler` の公開抽象・重み付き／分散 sampler・
  `num_workers` 並列プリフェッチ・`pin_memory`・`collate_fn` クロー
  ジャ・iterable-style dataset・`ConcatDataset`／`Subset`／
  `random_split`。
- 公開 `randperm`（内部ヘルパーのみ。rng doc §8 の対象外を維持）。
- デバイス常駐データセット（GPU 上でのバッチ切り出し）・
  `DeviceParamStore` との結線。
- `nn::Dropout`（#1603）等ほかの確率的演算。
- CUDA／Metal 実機での round-trip 実測（`#[ignore]` テストは用意し
  未実測明記）。
- `Tensor` 以外（`Var`）を保持するデータセット。

## 10. 実装記録

- `crates/tensor-core/src/data.rs`: `Dataset` trait・`TensorDataset<T>`
  ・タプル impl（2／3 要素）・`DataLoaderConfig`／`DataLoader`／
  `Batches`・`DataError`・private `shuffled_indices`／
  `uniform_below`。単体テスト 12 件（決定的シャッフル・順列性・
  `shuffle=false` の RNG 非消費・bit 完全一致・`ExactSizeIterator`
  整合等）。
- `crates/facade/src/data.rs`: 純再エクスポート（`Batches`／
  `DataError`／`DataLoader`／`DataLoaderConfig`／`Dataset`／
  `TensorDataset` の 6 型）。
- `crates/facade/tests/api_surface.rs`: `data_module_reexports_
  exactly_expected_surface`／`data_module_is_pure_reexport`／
  `data_types_are_reachable_via_facade_only` を追加。
- `crates/facade/tests/data_loader.rs`: `fandhe_ai` のみに依存する
  ミニバッチ学習ループ統合テスト（loss 減少確認）・分類バッチの型
  整合スモーク（`fandhe_ai_autodiff::Reduction` を直接 import する
  既存ギャップは `nll_kl_div_backend_parity.rs` と同様に許容）・bit
  完全一致確認・CUDA／Metal `#[ignore]` round-trip（未実測のまま
  申し送り）。
- facade 新規公開面: `fandhe_ai::data::{Batches, DataError, DataLoader,
  DataLoaderConfig, Dataset, TensorDataset}` の 6 型のみ。新規 `Op`／
  `BackendOps`／VJP なし。

## 11. 追補（#2182）: Sampler・collate・transform フック

上記 §9 の対象外欄に挙げていた「`Sampler`／`BatchSampler` の公開抽象・
重み付き sampler・`collate_fn` クロージャ」を #2182 で実装した。詳細な
設計判断・facade 保留の経緯は `docs/tensor-core-data-sampler-hooks-
decision.md` を正とする。要点のみ記す:

- `tensor-core::data` に `Sampler` trait（`SequentialSampler`／
  `RandomSampler`／`WeightedRandomSampler` の 3 実装）・
  `SamplerDataLoader<D>`（`Sampler` の添字をそのまま `Dataset::batch`
  へ渡す）・`HookedDataLoader<T>`（`TensorDataset<T>` 限定でサンプル
  単位 `TransformFn`／collate 単位 `CollateFn` を追加）を新設した。
  既存の `DataLoader`／`DataLoaderConfig`（本ドキュメント §3）は
  **不変**のまま維持する（フィールド追加・inherent メソッド追加とも
  行わない）。
- facade（`fandhe_ai::data`）への再エクスポートは #2182 時点では未承認のまま
  保留だったが、#2505 で 11 名を純再エクスポートとして公開済み
  （`docs/tensor-core-data-sampler-hooks-decision.md` §5・§8）。§10 の
  6 型再エクスポートは不変。
- 上記 §9 の残る対象外（`num_workers` 並列プリフェッチのみ #2183 で
  実装。§12。`pin_memory`・iterable-style dataset・`ConcatDataset`／
  `Subset`／`random_split`・タプルデータセットでのサンプル単位フック）
  は引き続き対象外。「マルチワーカー prefetch」の追跡先イシュー番号は
  issue #2182 本文の誤記（実際には無関係の #2181。`docs/tensor-core-
  data-sampler-hooks-decision.md` §6 参照）であり、正しい追跡先は
  #2183 だった。

## 12. 追補（#2183）: マルチワーカー prefetch

上記 §9・§11 の対象外欄に挙げていた「`num_workers` 並列プリフェッチ」
を #2183 で実装した。設計判断・性能実測は `docs/tensor-core-data-
prefetch-decision.md`・`docs/perf/logs/data-loader-prefetch-2183/
README.md` を正とする。要点のみ記す:

- `tensor-core::data` に `PrefetchConfig`（`num_workers`／
  `prefetch_depth` の検証付き設定）・`PrefetchDataLoader<D>`（`Sampler`
  の添字を worker スレッドへ分配して並列に `Dataset::batch` を実行し、
  結果を呼び出し順に並べ直す）・`PrefetchBatches<D>`（そのイテレータ）
  を新設した。`rayon` ではなく `std::thread`／`std::sync::mpsc` を
  使う（依存追加は承認事項のため未実施）。
- 出力列は同一の `Sampler` を使う `SamplerDataLoader` と、任意の
  `(num_workers, prefetch_depth)` の組み合わせで bit 完全一致する
  （決定性契約。前提は `Dataset::batch` が添字だけで決まる純関数で
  あること）。
- facade（`fandhe_ai::data`）への再エクスポートは #2506 で公開済み。

## 13. 追補（#2661）: Subset・ConcatDataset・random_split

上記 §9 の対象外欄に挙げていた `ConcatDataset`／`Subset`／`random_split` を #2661 で
`tensor-core::data`（`data/compose.rs`）へ実装した。設計判断は
`docs/tensor-core-dataset-compose-decision.md` を正とする。要点のみ記す:

- `Subset<D>`（`Arc` 共有の添字部分集合）・`ConcatDataset<D>`（累積和・同型成分限定。
  先頭軸連結用の `ConcatBatch` trait を伴う）・`random_split`／`random_split_fractions` を追加した。
  既存の公開型・trait へは何も足していない。
- 同一 `manual_seed` の下で分割した添字列の連結は `RandomSampler` の順列と bit 完全一致する。
- **facade 非公開**（承認依頼 #2677・公開は承認後の #2679。保留ガードで機械固定）。
  `Sequential::fit`（`compat::training::run_fit`）への結線は引き続き保留
  （`docs/tensor-core-data-prefetch-decision.md` §8）。
- 上記 §9 の残る対象外（`pin_memory`・iterable-style dataset・
  `ConcatDataset`／`Subset`／`random_split`・タプルデータセットでの
  サンプル単位フック・transform／collate フックの並列化・persistent
  workers・GPU DMA prefetch）は引き続き対象外。
