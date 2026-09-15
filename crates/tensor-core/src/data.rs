//! map-style データセット・バッチ供給ユーティリティ（PyTorch
//! `torch.utils.data.Dataset`／`DataLoader` 相当。イシュー #1615。
//! 親 #1602）。
//!
//! # 位置づけ
//!
//! `rng.rs`／`creation.rs` と同じ「ホスト側だけで完結する生成系」レイヤー
//! に属する。必要なのは [`crate::tensor::Tensor`]・[`crate::rng::with_global_rng`]・
//! `checked_numel_for`（`pub(crate)`）のみであり、`BackendOps`／`Op`／
//! `Var` を一切経由しない。バッチ化されたテンソルは既存の `Tape::var`
//! アップロード経路でデバイスへ反映する想定である（本モジュール自体は
//! デバイスに触れない）。
//!
//! # 受入基準テンプレートの非適用（#1602 ツリー共通の注記。`creation.rs`
//! ／`rng.rs` と同じ理由）
//!
//! `Dataset`／`DataLoader` はホスト側のデータ供給ユーティリティであり
//! **Op／VJP／バックエンド別カーネルを持たない**（`torch.utils.data` にも
//! 勾配は無い）。「parity」は「ホスト側でバッチ化 → CPU `Tape::var`
//! アップロードの bit 完全一致」＋ CUDA／Metal `#[ignore]` round-trip
//! （`crates/facade/tests/data_loader.rs`）で満たす——バックエンド別
//! カーネルが存在しないため REQ-2 複合判定の対象自体が無い
//! （`docs/dataset-dataloader-design.md`）。
//!
//! # シャッフル契約（`manual_seed` との整合）
//!
//! 順列は Fisher–Yates（Durstenfeld）で生成し、各抽選は [`crate::rng`]
//! の `randint` と同じ **rejection sampling**（`next_u64` の整数演算の
//! み）で剰余バイアスを排除する。整数演算のみのためプラットフォーム
//! 横断で bit 同一（ゴールデン値テストが可能）。順列生成は 1 回の
//! [`crate::rng::with_global_rng`] クロージャ内で完結させ（複数値を
//! まとめて引く操作の原子性。`rng.rs` doc 参照）、[`DataLoader::iter`]
//! 呼び出し時に eager に `Vec<usize>` を確定する（[`Batches`] はロックを
//! 持たず添字を切り出すだけ）。`shuffle=false` はグローバル RNG を
//! **一切消費しない**。空データセット（`len()==0`）はシャッフルの
//! 有無に関わらず抽選を消費しない。
//!
//! # セキュリティ考慮（OWASP Top 10）
//!
//! - **A02 暗号化の失敗**: xorshift64* は暗号学的に安全な PRNG では
//!   ない（`rng.rs` と同じ制約）。シャッフル順序を秘匿性が必要な用途
//!   （トークン・鍵生成等）に使わないこと。
//! - **A03／A04**: `batch()` はすべて `len()` に対する境界検査
//!   （[`DataError::IndexOutOfRange`]）を経てからアクセスする。出力
//!   要素数は `checked_numel_for` で確保前に検査し、本番経路で
//!   `unwrap()`／`expect()` を使わない。
//! - **A08**: タプルデータセットの長さ不一致は [`DataLoader::new`] の
//!   `validate()` で fail-closed に拒否し、成分間でシャッフル順がずれた
//!   学習データが黙って供給されない（[`DataError::LengthMismatch`]）。

use crate::element::Element;
use crate::error::ShapeError;
use crate::rng::{Xorshift64Star, with_global_rng};
use crate::tensor::{Tensor, checked_numel_for};

/// [`Dataset`]／[`DataLoader`] 専用のエラー型。shape 起因の不整合
/// （要素数積のオーバーフロー等）は `Tensor::new`／`narrow` 等と同じ
/// [`ShapeError`] へ委譲する（[`From<ShapeError>`] 実装。`rng::RngError`／
/// `creation::CreationError` と同型の設計）。
///
/// `#[non_exhaustive]`: 公開 API 非破壊（`.claude/rules/security.md`）
/// のため後続の検査項目追加に備える。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataError {
    /// [`DataLoaderConfig::batch_size`] がゼロ（1 バッチも構成できない）。
    ZeroBatchSize,
    /// [`TensorDataset::new`] に rank 0（スカラー）のテンソルが渡された
    /// （スカラーにはサンプル軸がなくバッチ化できない）。
    ScalarTensor,
    /// タプルデータセットの成分間で `len()` が一致しない
    /// （`component` は先頭〈index 0〉を基準とした不一致成分の位置。
    /// [`DataLoader::new`] の構築時検査で検出する）。
    LengthMismatch {
        expected: usize,
        found: usize,
        component: usize,
    },
    /// `batch(indices)` に渡された添字がデータセットの `len()` を
    /// 超える。
    IndexOutOfRange { index: usize, len: usize },
    /// shape 起因の不整合（[`ShapeError`] への委譲）。
    Shape(ShapeError),
}

impl From<ShapeError> for DataError {
    fn from(err: ShapeError) -> Self {
        DataError::Shape(err)
    }
}

impl std::fmt::Display for DataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataError::ZeroBatchSize => write!(f, "DataLoaderConfig の batch_size がゼロ"),
            DataError::ScalarTensor => {
                write!(
                    f,
                    "TensorDataset にはサンプル軸を持つ rank >= 1 のテンソルが必要"
                )
            }
            DataError::LengthMismatch {
                expected,
                found,
                component,
            } => write!(
                f,
                "タプルデータセットの成分 {component} の長さ {found} が先頭成分の長さ {expected} と一致しない"
            ),
            DataError::IndexOutOfRange { index, len } => {
                write!(f, "添字 {index} がデータセットの長さ {len} を外れている")
            }
            DataError::Shape(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for DataError {}

/// map-style データセット。`len()` 個のサンプルを持ち、[`Self::batch`]
/// で添字集合をまとめて 1 バッチへ組み立てる。
///
/// PyTorch `Dataset.__getitem__`（単一サンプル）ではなく「添字列 →
/// バッチ」を trait の中心に置く（Rust では collate〈サンプルの束ね方〉
/// を型で表現する必要があるため。異種 dtype／複数列はタプル（[`Dataset`]
/// の 2/3 要素タプル実装）で表す——同一の添字列を全成分に適用するため
/// シャッフル順が成分間で必ず一致する）。
pub trait Dataset {
    /// [`Self::batch`] が返すバッチの型（[`TensorDataset<T>`] なら
    /// `Tensor<T>`、タプルなら成分ごとの `Batch` のタプル）。
    type Batch;

    /// サンプル総数。
    fn len(&self) -> usize;

    /// `len() == 0` か。
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 構築時整合性検査（既定 `Ok(())`。タプル実装が成分間の長さ一致を
    /// 検査する）。[`DataLoader::new`] が構築時に一度だけ呼ぶ。
    fn validate(&self) -> Result<(), DataError> {
        Ok(())
    }

    /// `indices` の各サンプルを先頭軸へ積んだバッチを返す。`indices` の
    /// いずれかが `len()` 以上のとき [`DataError::IndexOutOfRange`]。
    fn batch(&self, indices: &[usize]) -> Result<Self::Batch, DataError>;
}

/// `tensor`（先頭軸＝サンプル軸）1 本からなるデータセット（PyTorch
/// `torch.utils.data.TensorDataset` の単一テンソル版）。
pub struct TensorDataset<T: Element> {
    tensor: Tensor<T>,
}

impl<T: Element> std::fmt::Debug for TensorDataset<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `Tensor<T>` は `Debug` 非実装（`tensor.rs` 参照）のため、
        // `facade::Tape` と同様に `finish_non_exhaustive` で内部を隠す。
        f.debug_struct("TensorDataset").finish_non_exhaustive()
    }
}

impl<T: Element> TensorDataset<T> {
    /// `tensor` の先頭軸をサンプル軸とみなして構築する。`tensor` が
    /// rank 0（スカラー。サンプル軸を持てない）のとき
    /// [`DataError::ScalarTensor`] を返す。
    pub fn new(tensor: Tensor<T>) -> Result<Self, DataError> {
        if tensor.rank() == 0 {
            return Err(DataError::ScalarTensor);
        }
        Ok(Self { tensor })
    }

    /// 保持するテンソルへの参照。
    pub fn tensor(&self) -> &Tensor<T> {
        &self.tensor
    }

    /// `index` 番目のサンプル 1 件を、先頭軸を潰した shape（`shape[1..]`）
    /// で返す（PyTorch `__getitem__` 相当）。
    pub fn get(&self, index: usize) -> Result<Tensor<T>, DataError> {
        let len = self.len();
        if index >= len {
            return Err(DataError::IndexOutOfRange { index, len });
        }
        // `narrow(0, index, 1)` は shape[0] を 1 に潰すのみで残りの軸の
        // strides は不変のため、`self.tensor` が contiguous であれば
        // 結果も contiguous（`Tensor::is_contiguous` はサイズ 1 の軸の
        // stride を判定に用いないため）。非 contiguous な入力（転置
        // view 等）は `reshape` が `ShapeError::NonContiguousReshape`
        // を返しうるが、これは `Tensor::reshape` 自体の既定契約であり
        // 本メソッド固有の緩和は行わない。
        let row = self.tensor.narrow(0, index, 1)?;
        let rest = &self.tensor.shape()[1..];
        Ok(row.reshape(rest)?)
    }
}

impl<T: Element> Dataset for TensorDataset<T> {
    type Batch = Tensor<T>;

    fn len(&self) -> usize {
        // `TensorDataset::new` が rank 0 を拒否済みのため、`shape()` は
        // 常に少なくとも 1 要素を持つ（REQ-8「境界検査を省略しない」の
        // 趣旨に沿い、検査済みの前提のみで添字アクセスする）。
        self.tensor.shape()[0]
    }

    fn batch(&self, indices: &[usize]) -> Result<Self::Batch, DataError> {
        gather_rows(&self.tensor, indices)
    }
}

/// `tensor` の先頭軸から `indices` の行を集めて 1 本のテンソルへ組み
/// 立てる（[`TensorDataset::batch`] の実体）。出力 shape は
/// `[indices.len(), tensor.shape()[1..]]`。
///
/// `indices` はすべて `tensor.shape()[0]` 未満であることを事前検査
/// （境界検査を省略しない。`.claude/rules/coding-rust.md`）してから
/// 実体化する。出力要素数は `checked_numel_for::<T>` で確保前に検査
/// する（A03/A04 対策）。`shuffle=false`（順序が `0..len` の連番）の
/// バッチは `narrow(0, start, len).contiguous()` と bit 完全一致する
/// （純粋コピーで算術を伴わないため）。
fn gather_rows<T: Element>(tensor: &Tensor<T>, indices: &[usize]) -> Result<Tensor<T>, DataError> {
    let len = tensor.shape()[0];
    for &idx in indices {
        if idx >= len {
            return Err(DataError::IndexOutOfRange { index: idx, len });
        }
    }
    let rest = &tensor.shape()[1..];
    let mut out_shape = Vec::with_capacity(1 + rest.len());
    out_shape.push(indices.len());
    out_shape.extend_from_slice(rest);
    let numel = checked_numel_for::<T>(&out_shape)?;
    let mut data = Vec::with_capacity(numel);
    for &idx in indices {
        // `idx < len` は上のループで検査済みのため `narrow` は必ず
        // 成功する（`ShapeError::NarrowOutOfBounds` には到達しない）。
        let row = tensor.narrow(0, idx, 1)?;
        // `row` は非 contiguous になりうる（`tensor` 自体が転置 view 等
        // の場合）ため `host_slice()`（contiguous なら借用・非
        // contiguous なら 1 回だけ実体化）で読み出す（`tensor.rs`
        // `host_slice` doc 参照）。
        data.extend_from_slice(&row.host_slice());
    }
    Ok(Tensor::new(data, &out_shape)?)
}

/// 2 要素タプルによる異種 dtype／複数列データセット（例:
/// `(TensorDataset<f32>, TensorDataset<i32>)` で特徴量とラベルを
/// 同一のシャッフル順で取り出す）。`len()` は先頭成分（`.0`）の長さ。
impl<A: Dataset, B: Dataset> Dataset for (A, B) {
    type Batch = (A::Batch, B::Batch);

    fn len(&self) -> usize {
        self.0.len()
    }

    fn validate(&self) -> Result<(), DataError> {
        self.0.validate()?;
        self.1.validate()?;
        let expected = self.0.len();
        let found = self.1.len();
        if found != expected {
            return Err(DataError::LengthMismatch {
                expected,
                found,
                component: 1,
            });
        }
        Ok(())
    }

    fn batch(&self, indices: &[usize]) -> Result<Self::Batch, DataError> {
        Ok((self.0.batch(indices)?, self.1.batch(indices)?))
    }
}

/// 3 要素タプル版（2 要素タプルと同じ規約）。
impl<A: Dataset, B: Dataset, C: Dataset> Dataset for (A, B, C) {
    type Batch = (A::Batch, B::Batch, C::Batch);

    fn len(&self) -> usize {
        self.0.len()
    }

    fn validate(&self) -> Result<(), DataError> {
        self.0.validate()?;
        self.1.validate()?;
        self.2.validate()?;
        let expected = self.0.len();
        for (offset, found) in [self.1.len(), self.2.len()].into_iter().enumerate() {
            if found != expected {
                return Err(DataError::LengthMismatch {
                    expected,
                    found,
                    component: offset + 1,
                });
            }
        }
        Ok(())
    }

    fn batch(&self, indices: &[usize]) -> Result<Self::Batch, DataError> {
        Ok((
            self.0.batch(indices)?,
            self.1.batch(indices)?,
            self.2.batch(indices)?,
        ))
    }
}

/// [`DataLoader`] の構成（PyTorch `DataLoader(batch_size=, shuffle=,
/// drop_last=)` 相当のサブセット）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataLoaderConfig {
    pub batch_size: usize,
    pub shuffle: bool,
    pub drop_last: bool,
}

impl DataLoaderConfig {
    /// `shuffle=false`・`drop_last=false` の既定構成。
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            shuffle: false,
            drop_last: false,
        }
    }

    /// builder: `shuffle` を設定する。
    pub fn shuffle(mut self, on: bool) -> Self {
        self.shuffle = on;
        self
    }

    /// builder: `drop_last` を設定する。
    pub fn drop_last(mut self, on: bool) -> Self {
        self.drop_last = on;
        self
    }
}

/// `drop_last` の有無に応じたバッチ数（floor／ceil）。`batch_size == 0`
/// は [`DataLoader::new`] が事前に拒否するため呼び出し側では発生しない
/// が、防御的に 0 を返す。
fn batch_count(len: usize, batch_size: usize, drop_last: bool) -> usize {
    if batch_size == 0 {
        return 0;
    }
    if drop_last {
        len / batch_size
    } else {
        len.div_ceil(batch_size)
    }
}

/// `dataset` を [`DataLoaderConfig`] に従ってミニバッチへ分割する
/// （PyTorch `torch.utils.data.DataLoader` 相当）。
pub struct DataLoader<D: Dataset> {
    dataset: D,
    config: DataLoaderConfig,
}

impl<D: Dataset> std::fmt::Debug for DataLoader<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `D`（`Dataset` 実装）は一般には `Debug` を要求しないため、
        // `TensorDataset` と同様に `finish_non_exhaustive` で内部を隠す。
        f.debug_struct("DataLoader")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<D: Dataset> DataLoader<D> {
    /// `config.batch_size == 0` は [`DataError::ZeroBatchSize`]。
    /// `dataset.validate()` が失敗した場合はその結果をそのまま返す
    /// （タプルデータセットの長さ不一致を構築時に fail-closed に拒否
    /// する。モジュール冒頭 A08 注記）。
    pub fn new(dataset: D, config: DataLoaderConfig) -> Result<Self, DataError> {
        if config.batch_size == 0 {
            return Err(DataError::ZeroBatchSize);
        }
        dataset.validate()?;
        Ok(Self { dataset, config })
    }

    /// 保持するデータセットへの参照。
    pub fn dataset(&self) -> &D {
        &self.dataset
    }

    /// 構成のコピー（`DataLoaderConfig: Copy`）。
    pub fn config(&self) -> DataLoaderConfig {
        self.config
    }

    /// 1 epoch あたりのバッチ数。
    pub fn len(&self) -> usize {
        batch_count(
            self.dataset.len(),
            self.config.batch_size,
            self.config.drop_last,
        )
    }

    /// `len() == 0` か。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 1 epoch 分のイテレータを返す。`shuffle=true` のときは呼び出し
    /// ごとに順列を引き直す（PyTorch の epoch ごと再シャッフルと同義。
    /// モジュール冒頭「シャッフル契約」参照）。
    pub fn iter(&self) -> Batches<'_, D> {
        let n = self.dataset.len();
        let order = if self.config.shuffle {
            with_global_rng(|rng| shuffled_indices(n, rng))
        } else {
            (0..n).collect()
        };
        Batches {
            dataset: &self.dataset,
            order,
            batch_size: self.config.batch_size,
            drop_last: self.config.drop_last,
            cursor: 0,
        }
    }

    /// 内部の `dataset` を取り出す。
    pub fn into_dataset(self) -> D {
        self.dataset
    }
}

impl<'a, D: Dataset> IntoIterator for &'a DataLoader<D> {
    type Item = Result<D::Batch, DataError>;
    type IntoIter = Batches<'a, D>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// [`DataLoader::iter`] が返す 1 epoch 分のバッチイテレータ。
///
/// `order`（`shuffle` の有無に応じて事前に確定した添字順列）を先頭から
/// `batch_size` 個ずつ切り出し [`Dataset::batch`] へ渡す。`drop_last`
/// が真の場合、末尾の端数（`batch_size` 未満）は yield しない。
pub struct Batches<'a, D: Dataset> {
    dataset: &'a D,
    order: Vec<usize>,
    batch_size: usize,
    drop_last: bool,
    cursor: usize,
}

impl<D: Dataset> Batches<'_, D> {
    /// 残バッチ数（`ExactSizeIterator::len` と同一の計算式）。
    fn remaining_batches(&self) -> usize {
        if self.cursor >= self.order.len() {
            return 0;
        }
        batch_count(
            self.order.len() - self.cursor,
            self.batch_size,
            self.drop_last,
        )
    }
}

impl<D: Dataset> Iterator for Batches<'_, D> {
    type Item = Result<D::Batch, DataError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.order.len() {
            return None;
        }
        let remaining = self.order.len() - self.cursor;
        if self.drop_last && remaining < self.batch_size {
            return None;
        }
        let take = remaining.min(self.batch_size);
        let indices = &self.order[self.cursor..self.cursor + take];
        let result = self.dataset.batch(indices);
        self.cursor += take;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.remaining_batches();
        (n, Some(n))
    }
}

impl<D: Dataset> ExactSizeIterator for Batches<'_, D> {
    fn len(&self) -> usize {
        self.remaining_batches()
    }
}

/// Fisher–Yates（Durstenfeld）順列生成。`i` を `n-1` から `1` まで降順に
/// 走査し、各回 `[0, i]` の一様な `j` と swap する（標準的な in-place
/// シャッフルアルゴリズム。モジュール冒頭「シャッフル契約」参照）。
fn shuffled_indices(n: usize, rng: &mut Xorshift64Star) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        // `bound = i + 1`（`i >= 1` のため常に `>= 2`）。
        let j = uniform_below(rng, i as u64 + 1) as usize;
        order.swap(i, j);
    }
    order
}

/// `[0, bound)` の一様分布に従う `u64` を rejection sampling で返す
/// （`rng::randint` と同一方式・同一定数式。剰余バイアスを排除する）。
fn uniform_below(rng: &mut Xorshift64Star, bound: u64) -> u64 {
    debug_assert!(bound > 0, "uniform_below: bound は正である必要がある");
    let zone = bound.wrapping_mul(u64::MAX / bound);
    loop {
        let x = rng.next_u64();
        if x < zone {
            return x % bound;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{global_rng_test_lock, manual_seed};

    fn tensor_1d(values: &[f32]) -> Tensor<f32> {
        Tensor::new(values.to_vec(), &[values.len()])
            .unwrap_or_else(|e| panic!("test fixture: 1d tensor 構築に失敗: {e}"))
    }

    fn tensor_2d(rows: usize, cols: usize) -> Tensor<f32> {
        let data: Vec<f32> = (0..rows * cols).map(|v| v as f32).collect();
        Tensor::new(data, &[rows, cols])
            .unwrap_or_else(|e| panic!("test fixture: 2d tensor 構築に失敗: {e}"))
    }

    #[test]
    fn tensor_dataset_rejects_scalar() {
        let scalar = Tensor::scalar(1.0f32);
        let err = TensorDataset::new(scalar).unwrap_err();
        assert_eq!(err, DataError::ScalarTensor);
    }

    #[test]
    fn data_loader_rejects_zero_batch_size() {
        let ds = TensorDataset::new(tensor_2d(4, 2)).unwrap();
        let err = DataLoader::new(ds, DataLoaderConfig::new(0)).unwrap_err();
        assert_eq!(err, DataError::ZeroBatchSize);
    }

    #[test]
    fn tuple_dataset_length_mismatch_is_rejected_at_construction() {
        let a = TensorDataset::new(tensor_2d(4, 2)).unwrap();
        let b = TensorDataset::new(tensor_1d(&[0.0, 1.0, 2.0])).unwrap();
        let err = DataLoader::new((a, b), DataLoaderConfig::new(2)).unwrap_err();
        assert_eq!(
            err,
            DataError::LengthMismatch {
                expected: 4,
                found: 3,
                component: 1,
            }
        );
    }

    #[test]
    fn sequential_batches_are_bit_identical_to_narrow_slices() {
        let tensor = tensor_2d(10, 3);
        let ds = TensorDataset::new(tensor.clone()).unwrap();
        let loader = DataLoader::new(ds, DataLoaderConfig::new(4)).unwrap();

        let batches: Vec<Tensor<f32>> = loader
            .iter()
            .map(|b| b.unwrap_or_else(|e| panic!("test fixture: batch が失敗した: {e}")))
            .collect();
        assert_eq!(batches.len(), 3); // 4, 4, 2

        let mut start = 0usize;
        for batch in &batches {
            let len = batch.shape()[0];
            let expected = tensor
                .narrow(0, start, len)
                .unwrap_or_else(|e| panic!("test fixture: narrow が失敗した: {e}"))
                .contiguous();
            assert_eq!(batch.host_slice().as_ref(), expected.host_slice().as_ref());
            start += len;
        }
    }

    #[test]
    fn rank1_dataset_batches_and_items() {
        let labels = tensor_1d(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let ds = TensorDataset::new(labels).unwrap();
        assert_eq!(ds.len(), 5);

        let item = ds.get(2).unwrap();
        assert_eq!(item.shape(), &[] as &[usize]);
        assert_eq!(item.get(&[]), Some(2.0));

        let batch = ds.batch(&[0, 2, 4]).unwrap();
        assert_eq!(batch.shape(), &[3]);
        assert_eq!(batch.host_slice().as_ref(), &[0.0, 2.0, 4.0]);
    }

    #[test]
    fn batches_size_hint_matches_data_loader_len() {
        for n in [0usize, 3, 10] {
            for drop_last in [false, true] {
                let base = TensorDataset::new(tensor_2d(n.max(1), 1)).unwrap();
                // `n == 0` は `tensor_2d` が rank 0 軸を作れないため
                // narrow(0, 0, 0) で空データセットを模す。
                let ds = if n == 0 {
                    TensorDataset::new(base.tensor().narrow(0, 0, 0).unwrap_or_else(|e| {
                        panic!("test fixture: 空データセットの narrow が失敗した: {e}")
                    }))
                    .unwrap()
                } else {
                    base
                };
                let loader =
                    DataLoader::new(ds, DataLoaderConfig::new(4).drop_last(drop_last)).unwrap();
                let expected_len = loader.len();
                let mut batches = loader.iter();
                assert_eq!(batches.len(), expected_len, "n={n} drop_last={drop_last}");
                let mut yielded = 0usize;
                while batches.next().is_some() {
                    yielded += 1;
                }
                assert_eq!(yielded, expected_len, "n={n} drop_last={drop_last}");
            }
        }
    }

    #[test]
    fn drop_last_batch_counts() {
        let ds = TensorDataset::new(tensor_2d(10, 1)).unwrap();
        let loader = DataLoader::new(ds, DataLoaderConfig::new(4)).unwrap();
        assert_eq!(loader.len(), 3); // 4, 4, 2

        let ds = TensorDataset::new(tensor_2d(10, 1)).unwrap();
        let loader = DataLoader::new(ds, DataLoaderConfig::new(4).drop_last(true)).unwrap();
        assert_eq!(loader.len(), 2); // 4, 4 のみ

        let ds = TensorDataset::new(tensor_2d(3, 1)).unwrap();
        let loader = DataLoader::new(ds, DataLoaderConfig::new(4).drop_last(true)).unwrap();
        assert_eq!(loader.len(), 0);
        assert!(loader.is_empty());
    }

    #[test]
    fn shuffle_is_deterministic_under_manual_seed() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let build = || {
            DataLoader::new(
                TensorDataset::new(tensor_2d(8, 1)).unwrap(),
                DataLoaderConfig::new(3).shuffle(true),
            )
            .unwrap()
        };

        manual_seed(123);
        let loader_a = build();
        let epoch1_a: Vec<Vec<f32>> = loader_a
            .iter()
            .map(|b| b.unwrap().host_slice().to_vec())
            .collect();
        let epoch2_a: Vec<Vec<f32>> = loader_a
            .iter()
            .map(|b| b.unwrap().host_slice().to_vec())
            .collect();

        manual_seed(123);
        let loader_b = build();
        let epoch1_b: Vec<Vec<f32>> = loader_b
            .iter()
            .map(|b| b.unwrap().host_slice().to_vec())
            .collect();

        assert_eq!(
            epoch1_a, epoch1_b,
            "同一 seed からの 1 epoch 目は一致するはず"
        );
        assert_ne!(epoch1_a, epoch2_a, "epoch ごとに再シャッフルされるはず");
    }

    #[test]
    fn shuffle_yields_a_permutation() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        manual_seed(99);

        let ds = TensorDataset::new(tensor_2d(20, 1)).unwrap();
        let loader = DataLoader::new(ds, DataLoaderConfig::new(6).shuffle(true)).unwrap();

        for _ in 0..3 {
            let mut seen: Vec<usize> = loader
                .iter()
                .flat_map(|b| {
                    b.unwrap()
                        .host_slice()
                        .iter()
                        .map(|&v| v as usize)
                        .collect::<Vec<_>>()
                })
                .collect();
            seen.sort_unstable();
            assert_eq!(seen, (0..20).collect::<Vec<_>>());
        }
    }

    #[test]
    fn shuffle_false_does_not_consume_global_rng() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        manual_seed(55);
        let ds = TensorDataset::new(tensor_2d(10, 1)).unwrap();
        let loader = DataLoader::new(ds, DataLoaderConfig::new(3)).unwrap();
        for b in loader.iter() {
            b.unwrap();
        }
        let after_no_shuffle: Vec<f32> = crate::rng::rand(&[4]).unwrap().host_slice().to_vec();

        manual_seed(55);
        let direct: Vec<f32> = crate::rng::rand(&[4]).unwrap().host_slice().to_vec();

        assert_eq!(after_no_shuffle, direct);
    }

    #[test]
    fn shuffle_golden_order() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        manual_seed(7);
        let order = with_global_rng(|rng| shuffled_indices(8, rng));
        // 整数演算のみの決定的順列（`rng::rand` のゴールデン値テストと
        // 同方式で許容される）。アルゴリズム変更時のみ更新する
        // （下記値は本実装に対して実測したものを固定した golden value。
        // `manual_seed(7)` から `shuffled_indices(8, ..)` を実行した実測値
        // をそのまま固定しており、順列であることの検査のみに留まっていた
        // 従来のアサーションを golden value 自体の固定へ強化する）。
        assert_eq!(order, vec![0, 4, 1, 2, 6, 7, 5, 3]);
    }

    #[test]
    fn tuple_batches_share_the_same_permutation() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        manual_seed(321);

        let features = TensorDataset::new(tensor_2d(12, 1)).unwrap();
        let labels_data: Vec<f32> = (0..12).map(|v| (v * 100) as f32).collect();
        let labels = TensorDataset::new(tensor_1d(&labels_data)).unwrap();

        let loader =
            DataLoader::new((features, labels), DataLoaderConfig::new(4).shuffle(true)).unwrap();

        for (x, y) in loader.iter().map(|b| b.unwrap()) {
            let xs = x.host_slice();
            let ys = y.host_slice();
            for (xv, yv) in xs.iter().zip(ys.iter()) {
                assert_eq!(*yv, xv * 100.0, "x={xv} y={yv} はずれた添字を指している");
            }
        }
    }
}
