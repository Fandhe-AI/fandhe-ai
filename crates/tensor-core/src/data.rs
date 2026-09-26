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
//! 有無に関わらず抽選を消費しない。`Dataset::len()` が返す `n` に対し
//! `checked_numel_for::<usize>(&[n])` の事前検査に失敗する場合（`n`
//! が `Vec<usize>` の allocation 上限を超える）は、この eager 確定・
//! RNG 消費とも一切行わず（`shuffle=true` でも RNG は消費しない）、
//! 検査失敗を [`DataError::Shape`] として最初の [`Batches::next`]
//! 呼び出しへ持ち越す（PR #1867 codex-review P1 是正・下記
//! `iter`／`Batches` doc 参照）。
//!
//! # セキュリティ考慮（OWASP Top 10）
//!
//! - **A02 暗号化の失敗**: xorshift64* は暗号学的に安全な PRNG では
//!   ない（`rng.rs` と同じ制約）。シャッフル順序を秘匿性が必要な用途
//!   （トークン・鍵生成等）に使わないこと。
//! - **A03／A04**: `batch()` はすべて `len()` に対する境界検査
//!   （[`DataError::IndexOutOfRange`]）を経てからアクセスする。出力
//!   要素数は `checked_numel_for` で確保前に検査し、本番経路で
//!   `unwrap()`／`expect()` を使わない。[`DataLoader::iter`] が構築
//!   する添字順列 `Vec<usize>`（非 shuffle 時の連番・`shuffled_
//!   indices` の順列）も同様に `checked_numel_for::<usize>` で確保前
//!   に検査し、capacity overflow パニックを起こさず型付きエラーへ
//!   落とす（PR #1867 codex-review P1 是正）。
//! - **A08**: タプルデータセットの長さ不一致は [`DataLoader::new`] の
//!   `validate()` で fail-closed に拒否し、成分間でシャッフル順がずれた
//!   学習データが黙って供給されない（[`DataError::LengthMismatch`]）。
//!
//! # Sampler／フック（イシュー #2182・親 #2131）
//!
//! [`DataLoader`]／[`DataLoaderConfig`] は既存の公開 API のまま
//! **不変**とする（`#[non_exhaustive]` ではない `DataLoaderConfig` へ
//! フィールドを足すと利用者の構造体リテラル構築を壊し、`DataLoader`
//! への inherent メソッド追加は facade 公開保留の迂回になるため。
//! `docs/tensor-core-data-sampler-hooks-decision.md` §2.1）。代わりに
//! [`Sampler`] trait（添字の選び方を差し替える拡張点）と、それを使う
//! 2 つの新しいローダー型を追加する:
//!
//! - [`SamplerDataLoader<D>`][SamplerDataLoader]: [`Sampler`] が返す
//!   添字列をそのまま [`Dataset::batch`] へ渡す。タプルデータセットでも
//!   成分間で同じ添字が使われる。
//! - [`HookedDataLoader<T>`][HookedDataLoader]:
//!   `SamplerDataLoader<TensorDataset<T>>` を内部に持ち、サンプル単位の
//!   [`TransformFn`] と collate 単位の [`CollateFn`] を追加できる
//!   （`TensorDataset<T>` 限定。issue の callback シグネチャが単一の
//!   `Tensor` 型のため）。
//!
//! **RNG 消費順序の契約**: [`Sampler::start_epoch`] の抽選は epoch
//! 開始時にすべて確定させ（上記「シャッフル契約」と同じ「1 回の
//! `with_global_rng` クロージャで完結」方式）、ユーザーの
//! transform／collate クロージャはその後にバッチごとに走る。
//! したがって transform が乱数を使う augmentation であっても、
//! `manual_seed` 下の再現性を sampler の抽選順のみで説明できる。
//!
//! [`RandomSampler`] は同一シードの下で `DataLoader{shuffle=true}` と、
//! [`WeightedRandomSampler`] は [`crate::rng::multinomial`] と、それぞれ
//! 添字列・抽選列が bit 完全一致する（各型の doc 参照）。
//!
//! facade（`fandhe_ai::data`）への再エクスポートは未承認のため保留中
//! （`crates/facade/src/lib.rs::DataHooksHoldDoctestGuard`・
//! `docs/tensor-core-data-sampler-hooks-decision.md` §5）。

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
    /// [`Sampler`] の抽選（[`RandomSampler`]・[`WeightedRandomSampler`]）
    /// が [`crate::rng`] へ委譲した処理の失敗（イシュー #2182）。
    Rng(crate::rng::RngError),
    /// [`HookedDataLoader`] の collate（既定の [`default_collate`]
    /// またはユーザー指定）に渡されたサンプル列で、`position` 番目の
    /// サンプルの shape が先頭サンプル（`expected`）と一致しない
    /// （イシュー #2182）。
    SampleShapeMismatch {
        position: usize,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    /// [`HookedDataLoader`] の collate に渡されたサンプル列が空
    /// （[`Sampler::next_batch`] が空でない添字列を返したのにサンプル
    /// 収集後の列が空になることは無いが、[`default_collate`] を直接
    /// 呼び出す利用者向けに防御的に定義する。イシュー #2182）。
    EmptyBatch,
}

impl From<ShapeError> for DataError {
    fn from(err: ShapeError) -> Self {
        DataError::Shape(err)
    }
}

impl From<crate::rng::RngError> for DataError {
    fn from(err: crate::rng::RngError) -> Self {
        DataError::Rng(err)
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
            DataError::Rng(err) => write!(f, "{err}"),
            DataError::SampleShapeMismatch {
                position,
                expected,
                found,
            } => write!(
                f,
                "collate 対象のサンプル {position} の shape {found:?} が先頭サンプルの shape {expected:?} と一致しない"
            ),
            DataError::EmptyBatch => write!(f, "collate 対象のサンプル列が空"),
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

    /// [`Self::get`] と異なり `reshape` を使わず `host_slice()` から
    /// 直接組み立てる（イシュー #2182。[`HookedDataLoader`] が使う）。
    /// `reshape` は非 contiguous な `tensor`（転置 view 等）で
    /// `ShapeError::NonContiguousReshape` を返しうるが、`host_slice()`
    /// は contiguous なら借用・非 contiguous なら 1 回だけ実体化して
    /// 読み出すため（`tensor.rs` `host_slice` doc 参照）、この経路は
    /// `tensor` の contiguity に依存しない。境界検査は先に行う（REQ-8）。
    pub(crate) fn sample_owned(&self, index: usize) -> Result<Tensor<T>, DataError> {
        let len = self.len();
        if index >= len {
            return Err(DataError::IndexOutOfRange { index, len });
        }
        let row = self.tensor.narrow(0, index, 1)?;
        let rest = &self.tensor.shape()[1..];
        Ok(Tensor::new(row.host_slice().into_owned(), rest)?)
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
    ///
    /// `n = self.dataset.len()` から確定する添字順列 `Vec<usize>` は
    /// `checked_numel_for::<usize>(&[n])` で確保前にバイトサイズまで
    /// 検査してから構築する。`Dataset::len()` は実装者が任意の値を
    /// 返せるため（例: 要素数ゼロの `Tensor::new(vec![], &[usize::MAX,
    /// 0])` から `len() == usize::MAX` のデータセットを作れる）、検査
    /// なしに `(0..n).collect()`／`shuffled_indices` の `Vec::with_
    /// capacity` を呼ぶと `numel * size_of::<usize>() > isize::MAX` で
    /// capacity overflow パニックしうる（本番経路 panic 禁止規約
    /// `.claude/rules/coding-rust.md`。PR #1867 codex-review P1 是正）。
    ///
    /// この `iter()` 自体は（設計どおり）`Result` を返さない infallible
    /// な API のまま維持する。検査に失敗した場合は `order` を空のまま
    /// にし、`shuffle=true` でも `with_global_rng` を呼ばず（RNG を一切
    /// 消費しない）、[`DataError::Shape`]（`ShapeError::
    /// ElementCountOverflow`）を [`Batches`] へ持ち越して**最初の
    /// `next()` 呼び出しで 1 回だけ** `Err` として yield し、以降は
    /// `None` を返す（`ExactSizeIterator::len()` もこの 1 件に整合させ
    /// る。下記 `Batches::next` 参照）。
    pub fn iter(&self) -> Batches<'_, D> {
        let n = self.dataset.len();
        match checked_numel_for::<usize>(&[n]) {
            Ok(_) => {
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
                    pending_error: None,
                }
            }
            Err(err) => Batches {
                dataset: &self.dataset,
                order: Vec::new(),
                batch_size: self.config.batch_size,
                drop_last: self.config.drop_last,
                cursor: 0,
                pending_error: Some(DataError::from(err)),
            },
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

/// バッチ添字を供給する拡張点（PyTorch `Sampler` + `BatchSampler` を
/// 合わせた「バッチ sampler」相当。イシュー #2182・親 #2131）。
/// [`SamplerDataLoader`] がこの trait を経由して添字列を得る。
///
/// # 契約
///
/// - [`Self::start_epoch`] は epoch 開始時に 1 回だけ呼ぶ。抽選が
///   必要な実装（[`RandomSampler`]・[`WeightedRandomSampler`]）は
///   ここで 1 回の [`crate::rng::with_global_rng`] クロージャ内に
///   まとめて抽選し、`manual_seed` 下の再現性を保つ
///   （モジュール冒頭「シャッフル契約」と同じ「複数値をまとめて引く
///   操作の原子性」を踏襲する）。
/// - [`Self::next_batch`] は次バッチの添字列を返す。空の `Vec` は
///   epoch 終了を表す番兵であり、空バッチとして yield しない。
/// - [`Self::num_batches`] が `Some` を返す実装では、その値が
///   [`Self::next_batch`] が実際に空以外を返す回数と一致する
///   （`ExactSizeIterator` 相当の `size_hint` に用いる）。
/// - `Send`: 将来のマルチワーカー化（#2181 の番号は issue 本文の誤記。
///   `docs/tensor-core-data-sampler-hooks-decision.md` §6 参照）に
///   備え、ローダー型を `Send` に保つための境界。
pub trait Sampler: Send {
    /// epoch 開始時に 1 回呼ぶ。抽選を伴う実装はここで確定させる。
    fn start_epoch(&mut self) -> Result<(), DataError>;

    /// 次バッチの添字列。空の `Vec` は epoch 終了の番兵。
    fn next_batch(&mut self) -> Vec<usize>;

    /// 1 epoch あたりのバッチ数（既知なら `Some`。既定は `None`）。
    fn num_batches(&self) -> Option<usize> {
        None
    }
}

/// [`Sampler`] の 3 実装が共有する「確定済み添字順列から
/// `batch_size` 個ずつ切り出す」処理（イシュー #2182）。切り出し式は
/// 既存 [`Batches::next`] の式と揃える（同一の `drop_last` 契約）。
struct IndexBatcher {
    order: Vec<usize>,
    cursor: usize,
    batch_size: usize,
    drop_last: bool,
}

impl IndexBatcher {
    fn new(batch_size: usize, drop_last: bool) -> Self {
        Self {
            order: Vec::new(),
            cursor: 0,
            batch_size,
            drop_last,
        }
    }

    /// 新しい epoch の順列を設定し、カーソルを先頭へ戻す。
    fn set_order(&mut self, order: Vec<usize>) {
        self.order = order;
        self.cursor = 0;
    }

    /// [`Batches::next`] と同じ切り出し式で次バッチの添字列を返す。
    /// 枯渇時・`drop_last` により端数を捨てる場合は空 `Vec`（[`Sampler`]
    /// の「空 Vec は epoch 終了の番兵」契約に対応）。
    fn next_batch(&mut self) -> Vec<usize> {
        if self.cursor >= self.order.len() {
            return Vec::new();
        }
        let remaining = self.order.len() - self.cursor;
        if self.drop_last && remaining < self.batch_size {
            return Vec::new();
        }
        let take = remaining.min(self.batch_size);
        let batch = self.order[self.cursor..self.cursor + take].to_vec();
        self.cursor += take;
        batch
    }

    fn num_batches(&self, len: usize) -> usize {
        batch_count(len, self.batch_size, self.drop_last)
    }
}

/// 連番順（シャッフルなし）でバッチ添字を供給する（PyTorch
/// `SequentialSampler` 相当。イシュー #2182）。グローバル RNG を
/// **一切消費しない**。
pub struct SequentialSampler {
    len: usize,
    batcher: IndexBatcher,
}

impl std::fmt::Debug for SequentialSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SequentialSampler").finish_non_exhaustive()
    }
}

impl SequentialSampler {
    /// `len` はデータセットのサンプル総数。`batch_size == 0` は
    /// [`DataError::ZeroBatchSize`]。
    pub fn new(len: usize, batch_size: usize, drop_last: bool) -> Result<Self, DataError> {
        if batch_size == 0 {
            return Err(DataError::ZeroBatchSize);
        }
        Ok(Self {
            len,
            batcher: IndexBatcher::new(batch_size, drop_last),
        })
    }
}

impl Sampler for SequentialSampler {
    fn start_epoch(&mut self) -> Result<(), DataError> {
        // `Dataset::len()` は実装者が任意の値を返せるため、
        // `DataLoader::iter` と同じ事前検査を経てから `0..len` を
        // 確定する（capacity overflow パニック防止。PR #1867 是正の
        // 契約を Sampler 側へも及ぼす）。
        checked_numel_for::<usize>(&[self.len])?;
        self.batcher.set_order((0..self.len).collect());
        Ok(())
    }

    fn next_batch(&mut self) -> Vec<usize> {
        self.batcher.next_batch()
    }

    fn num_batches(&self) -> Option<usize> {
        Some(self.batcher.num_batches(self.len))
    }
}

/// 順列でバッチ添字を供給する（PyTorch `RandomSampler` 相当。イシュー
/// #2182）。同一の `manual_seed` の下で `DataLoader{shuffle=true}` と
/// **添字順が bit 完全一致**する（[`start_epoch`](Sampler::start_epoch)
/// が `shuffled_indices` を 1 回の [`crate::rng::with_global_rng`]
/// クロージャ内で呼ぶため）。
pub struct RandomSampler {
    len: usize,
    batcher: IndexBatcher,
}

impl std::fmt::Debug for RandomSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RandomSampler").finish_non_exhaustive()
    }
}

impl RandomSampler {
    /// `batch_size == 0` は [`DataError::ZeroBatchSize`]。
    pub fn new(len: usize, batch_size: usize, drop_last: bool) -> Result<Self, DataError> {
        if batch_size == 0 {
            return Err(DataError::ZeroBatchSize);
        }
        Ok(Self {
            len,
            batcher: IndexBatcher::new(batch_size, drop_last),
        })
    }
}

impl Sampler for RandomSampler {
    fn start_epoch(&mut self) -> Result<(), DataError> {
        // `DataLoader::iter` の `shuffle=true` 経路と同一の検査・
        // 抽選手順（`checked_numel_for` → 1 回の `with_global_rng`）。
        // `len == 0` は `shuffled_indices` が空順列を返すのみで RNG は
        // 消費する（`DataLoader` 既存契約と揃える。事前検査失敗時のみ
        // RNG を消費しない）。
        checked_numel_for::<usize>(&[self.len])?;
        let order = with_global_rng(|rng| shuffled_indices(self.len, rng));
        self.batcher.set_order(order);
        Ok(())
    }

    fn next_batch(&mut self) -> Vec<usize> {
        self.batcher.next_batch()
    }

    fn num_batches(&self) -> Option<usize> {
        Some(self.batcher.num_batches(self.len))
    }
}

/// 重み付き復元・非復元抽出でバッチ添字を供給する（PyTorch
/// `WeightedRandomSampler` 相当。イシュー #2182）。構築時
/// （[`Self::new`]）に `crate::rng::validate_multinomial` で検証する
/// （RNG を消費しない）。[`start_epoch`](Sampler::start_epoch) は公開 API
/// [`crate::rng::multinomial`]（1 回の `with_global_rng` を使う原子的な
/// 抽選）を呼び、同一シードの下で `rng::multinomial(&weights,
/// num_samples, replacement)` と抽選列が bit 完全一致する。
pub struct WeightedRandomSampler {
    weights: Tensor<f32>,
    num_samples: usize,
    replacement: bool,
    batcher: IndexBatcher,
}

impl std::fmt::Debug for WeightedRandomSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightedRandomSampler")
            .finish_non_exhaustive()
    }
}

impl WeightedRandomSampler {
    /// `weights`（rank 1: 各要素が対応する添字の重み）・抽選する
    /// サンプル総数 `num_samples`・復元抽出の有無・バッチ化条件を
    /// 受け取る。構築時に有限性・非負・総和が正・非復元抽出時の
    /// 重み不足を検証する（`crate::rng::validate_multinomial` を
    /// 再利用。RNG は一切消費しない）。`batch_size == 0` は
    /// [`DataError::ZeroBatchSize`]。
    pub fn new(
        weights: Vec<f32>,
        num_samples: usize,
        replacement: bool,
        batch_size: usize,
        drop_last: bool,
    ) -> Result<Self, DataError> {
        if batch_size == 0 {
            return Err(DataError::ZeroBatchSize);
        }
        let n = weights.len();
        let weights = Tensor::new(weights, &[n])?;
        // 構築時検証のみを行い `(rows, n, out_shape)` は使わない
        // （`start_epoch` で `crate::rng::multinomial` を呼ぶ際に
        // 改めて検証される。二重検証だが RNG 消費前の fail-closed
        // ガードとして両呼び出しサイトで独立に成立させる）。
        crate::rng::validate_multinomial(&weights, num_samples, replacement)?;
        Ok(Self {
            weights,
            num_samples,
            replacement,
            batcher: IndexBatcher::new(batch_size, drop_last),
        })
    }
}

impl Sampler for WeightedRandomSampler {
    fn start_epoch(&mut self) -> Result<(), DataError> {
        if self.num_samples == 0 {
            // `rng::multinomial` は `num_samples == 0` のとき RNG を
            // 消費せず空テンソルを返すが、ここでは呼び出し自体を
            // 省略して契約を明示する（`RandomSampler` の `len == 0`
            // とは異なり、こちらは検証済みのため `?` は到達しない）。
            self.batcher.set_order(Vec::new());
            return Ok(());
        }
        let drawn = crate::rng::multinomial(&self.weights, self.num_samples, self.replacement)?;
        let order = drawn
            .host_slice()
            .iter()
            .map(|&v| {
                usize::try_from(v).map_err(|_| {
                    DataError::Rng(crate::rng::RngError::InvalidArgument {
                        reason: "multinomial の抽選結果が負値（i32 → usize 変換不可）",
                    })
                })
            })
            .collect::<Result<Vec<usize>, DataError>>()?;
        self.batcher.set_order(order);
        Ok(())
    }

    fn next_batch(&mut self) -> Vec<usize> {
        self.batcher.next_batch()
    }

    fn num_batches(&self) -> Option<usize> {
        Some(self.batcher.num_batches(self.num_samples))
    }
}

/// [`Sampler`] が返す添字列をそのまま [`Dataset::batch`] へ渡す
/// ローダー（イシュー #2182）。タプルデータセット（`(TensorDataset<f32>,
/// TensorDataset<i32>)` 等）でも同じ添字が全成分に適用されるため、
/// 分類タスクのように特徴量とラベルを同一のシャッフル順で取り出す
/// 要件を満たす（[`HookedDataLoader`] はサンプル単位 transform／collate
/// のため `TensorDataset<T>` 限定になるが、本型はタプルにも使える）。
pub struct SamplerDataLoader<D: Dataset> {
    dataset: D,
    sampler: Box<dyn Sampler>,
}

impl<D: Dataset> std::fmt::Debug for SamplerDataLoader<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SamplerDataLoader").finish_non_exhaustive()
    }
}

impl<D: Dataset> SamplerDataLoader<D> {
    /// `dataset.validate()` を実行し、タプル成分の長さ不一致を
    /// fail-closed に拒否する（[`DataLoader::new`] と同じ契約）。
    pub fn new(dataset: D, sampler: impl Sampler + 'static) -> Result<Self, DataError> {
        dataset.validate()?;
        Ok(Self {
            dataset,
            sampler: Box::new(sampler),
        })
    }

    /// 保持するデータセットへの参照。
    pub fn dataset(&self) -> &D {
        &self.dataset
    }

    /// 内部の `dataset` を取り出す。
    pub fn into_dataset(self) -> D {
        self.dataset
    }

    /// 1 epoch あたりのバッチ数（[`Sampler::num_batches`] が `Some` を
    /// 返す場合のみ）。
    pub fn num_batches(&self) -> Option<usize> {
        self.sampler.num_batches()
    }

    /// 1 epoch 分のイテレータを返す。[`Sampler::start_epoch`] を呼んで
    /// から yield を始める（失敗した場合は最初の `next()` で 1 回だけ
    /// `Err` を返し、以降は `None`。[`Batches`] の `pending_error` と
    /// 同じ方式）。
    pub fn iter(&mut self) -> SamplerBatches<'_, D> {
        let pending_error = self.sampler.start_epoch().err();
        SamplerBatches {
            dataset: &self.dataset,
            sampler: self.sampler.as_mut(),
            pending_error,
            done: false,
        }
    }
}

/// [`SamplerDataLoader::iter`] が返す 1 epoch 分のバッチイテレータ
/// （イシュー #2182）。`Sampler::next_batch` が空 `Vec` を返した後は
/// `done` を立て、sampler を再び呼ばない。
pub struct SamplerBatches<'a, D: Dataset> {
    dataset: &'a D,
    sampler: &'a mut dyn Sampler,
    pending_error: Option<DataError>,
    done: bool,
}

impl<D: Dataset> Iterator for SamplerBatches<'_, D> {
    type Item = Result<D::Batch, DataError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(err) = self.pending_error.take() {
            self.done = true;
            return Some(Err(err));
        }
        if self.done {
            return None;
        }
        let indices = self.sampler.next_batch();
        if indices.is_empty() {
            self.done = true;
            return None;
        }
        Some(self.dataset.batch(&indices))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.pending_error.is_some() {
            return (1, Some(1));
        }
        match self.sampler.num_batches() {
            Some(n) => (n, Some(n)),
            None => (0, None),
        }
    }
}

/// サンプル単位の変換フック（PyTorch `Dataset` の `transform` 相当。
/// イシュー #2182）。[`HookedDataLoader::with_try_transform`] が保持する
/// 内部表現（fallible）。本番経路で `unwrap`／`expect` を使わない規約
/// （`.claude/rules/coding-rust.md`）のため、[`HookedDataLoader::
/// with_transform`]（issue の字句どおりの infallible シグネチャ）は
/// `Ok` で包んでこの型へ格納する。
pub type TransformFn<T> = Box<dyn Fn(Tensor<T>) -> Result<Tensor<T>, DataError> + Send + Sync>;

/// サンプル列からバッチを組み立てる collate 関数（PyTorch
/// `collate_fn` 相当。イシュー #2182）。[`TransformFn`] と同じ理由で
/// fallible な内部表現を持つ。
pub type CollateFn<T> = Box<dyn Fn(&[Tensor<T>]) -> Result<Tensor<T>, DataError> + Send + Sync>;

/// 既定の collate（[`HookedDataLoader`] が `collate` 未設定時に使う。
/// イシュー #2182）。`samples` を先頭軸へ積んだ `[k, sample_shape..]`
/// のテンソルを返す。
///
/// 空スライスは [`DataError::EmptyBatch`]。全サンプルの shape が先頭
/// サンプルと一致しなければ [`DataError::SampleShapeMismatch`]。出力
/// 要素数は `checked_numel_for` で確保前に検査する（A03/A04 対策。
/// モジュール冒頭 OWASP 節）。
///
/// transform も collate も未設定の fast path（[`HookedDataLoader::
/// next`] 相当）では本関数を経由せず [`Dataset::batch`]（`gather_rows`）
/// へ直行するため、その場合の出力は `gather_rows` と bit 完全一致
/// する。本関数（slow path・恒等 transform 経由）を明示的に呼んだ
/// 場合も、純粋なコピーで算術を含まないため同じ入力に対し bit 完全
/// 一致する。
pub fn default_collate<T: Element>(samples: &[Tensor<T>]) -> Result<Tensor<T>, DataError> {
    let Some(first) = samples.first() else {
        return Err(DataError::EmptyBatch);
    };
    let sample_shape = first.shape().to_vec();
    for (position, sample) in samples.iter().enumerate().skip(1) {
        if sample.shape() != sample_shape.as_slice() {
            return Err(DataError::SampleShapeMismatch {
                position,
                expected: sample_shape,
                found: sample.shape().to_vec(),
            });
        }
    }
    let mut out_shape = Vec::with_capacity(1 + sample_shape.len());
    out_shape.push(samples.len());
    out_shape.extend_from_slice(&sample_shape);
    let numel = checked_numel_for::<T>(&out_shape)?;
    let mut data = Vec::with_capacity(numel);
    for sample in samples {
        data.extend_from_slice(&sample.host_slice());
    }
    Ok(Tensor::new(data, &out_shape)?)
}

/// [`Sampler`] で選んだサンプルへ transform／collate フックをかけて
/// バッチ化するローダー（PyTorch `DataLoader(sampler=, collate_fn=)`
/// 相当。イシュー #2182）。内部に
/// `SamplerDataLoader<TensorDataset<T>>` を持つ（合成）。issue の
/// callback シグネチャ（`Tensor` 単体）に合わせ `TensorDataset<T>`
/// 限定とする（タプルデータセットでのサンプル単位フックは対象外。
/// `docs/tensor-core-data-sampler-hooks-decision.md` §2.2・§6）。
///
/// # 処理順（AC-4）
///
/// [`HookedBatches::next`] は「サンプリング → サンプルごとに transform
/// → collate」の順で処理する。transform も collate も未設定なら
/// [`Dataset::batch`] への fast path を通る（コピー 1 回。[`Self::
/// with_transform`]／[`Self::with_collate`] を一度でも呼ぶと slow
/// path に切り替わる）。
pub struct HookedDataLoader<T: Element> {
    inner: SamplerDataLoader<TensorDataset<T>>,
    transform: Option<TransformFn<T>>,
    collate: Option<CollateFn<T>>,
}

impl<T: Element> std::fmt::Debug for HookedDataLoader<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookedDataLoader").finish_non_exhaustive()
    }
}

impl<T: Element> HookedDataLoader<T> {
    /// [`SamplerDataLoader::new`] と同じ契約（タプルではないため
    /// `validate()` は常に `Ok(())`）。
    pub fn new(
        dataset: TensorDataset<T>,
        sampler: impl Sampler + 'static,
    ) -> Result<Self, DataError> {
        Ok(Self {
            inner: SamplerDataLoader::new(dataset, sampler)?,
            transform: None,
            collate: None,
        })
    }

    /// builder: infallible な transform を設定する（issue の字句どおり
    /// のシグネチャ）。内部では `Ok` で包んで保持する。
    pub fn with_transform(
        mut self,
        f: impl Fn(Tensor<T>) -> Tensor<T> + Send + Sync + 'static,
    ) -> Self {
        self.transform = Some(Box::new(move |t| Ok(f(t))));
        self
    }

    /// builder: fallible な transform を設定する。
    pub fn with_try_transform(
        mut self,
        f: impl Fn(Tensor<T>) -> Result<Tensor<T>, DataError> + Send + Sync + 'static,
    ) -> Self {
        self.transform = Some(Box::new(f));
        self
    }

    /// builder: infallible な collate を設定する（issue の字句どおりの
    /// シグネチャ）。
    pub fn with_collate(
        mut self,
        f: impl Fn(&[Tensor<T>]) -> Tensor<T> + Send + Sync + 'static,
    ) -> Self {
        self.collate = Some(Box::new(move |samples| Ok(f(samples))));
        self
    }

    /// builder: fallible な collate を設定する。
    pub fn with_try_collate(
        mut self,
        f: impl Fn(&[Tensor<T>]) -> Result<Tensor<T>, DataError> + Send + Sync + 'static,
    ) -> Self {
        self.collate = Some(Box::new(f));
        self
    }

    /// 保持するデータセットへの参照。
    pub fn dataset(&self) -> &TensorDataset<T> {
        self.inner.dataset()
    }

    /// 内部の `dataset` を取り出す。
    pub fn into_dataset(self) -> TensorDataset<T> {
        self.inner.into_dataset()
    }

    /// 1 epoch あたりのバッチ数（[`Sampler::num_batches`] が `Some` を
    /// 返す場合のみ）。
    pub fn num_batches(&self) -> Option<usize> {
        self.inner.num_batches()
    }

    /// 1 epoch 分のイテレータを返す。
    pub fn iter(&mut self) -> HookedBatches<'_, T> {
        let pending_error = self.inner.sampler.start_epoch().err();
        HookedBatches {
            dataset: &self.inner.dataset,
            sampler: self.inner.sampler.as_mut(),
            transform: self.transform.as_ref(),
            collate: self.collate.as_ref(),
            pending_error,
            done: false,
        }
    }
}

/// [`HookedDataLoader::iter`] が返す 1 epoch 分のバッチイテレータ
/// （イシュー #2182）。`dataset`／`sampler` を [`SamplerDataLoader`] の
/// private フィールドから直接分割借用する（`&mut HookedDataLoader` から
/// `&TensorDataset<T>` と `&mut dyn Sampler` を同時に得るため。
/// `Dataset::batch` へ委譲すると sampler 側の可変借用と衝突するため
/// 経由しない）。
pub struct HookedBatches<'a, T: Element> {
    dataset: &'a TensorDataset<T>,
    sampler: &'a mut dyn Sampler,
    transform: Option<&'a TransformFn<T>>,
    collate: Option<&'a CollateFn<T>>,
    pending_error: Option<DataError>,
    done: bool,
}

impl<T: Element> Iterator for HookedBatches<'_, T> {
    type Item = Result<Tensor<T>, DataError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(err) = self.pending_error.take() {
            self.done = true;
            return Some(Err(err));
        }
        if self.done {
            return None;
        }
        let indices = self.sampler.next_batch();
        if indices.is_empty() {
            self.done = true;
            return None;
        }
        // fast path: transform も collate も未設定なら `Dataset::batch`
        // （`gather_rows`）へ直行し、追加のコピーを発生させない
        // （`HookedDataLoader` doc「処理順」節）。
        if self.transform.is_none() && self.collate.is_none() {
            return Some(self.dataset.batch(&indices));
        }
        let mut samples: Vec<Tensor<T>> = Vec::with_capacity(indices.len());
        for &idx in &indices {
            let sample = match self.dataset.sample_owned(idx) {
                Ok(s) => s,
                Err(e) => return Some(Err(e)),
            };
            let sample = match self.transform {
                Some(f) => match f(sample) {
                    Ok(s) => s,
                    Err(e) => return Some(Err(e)),
                },
                None => sample,
            };
            samples.push(sample);
        }
        let result = match self.collate {
            Some(f) => f(&samples),
            None => default_collate(&samples),
        };
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.pending_error.is_some() {
            return (1, Some(1));
        }
        match self.sampler.num_batches() {
            Some(n) => (n, Some(n)),
            None => (0, None),
        }
    }
}

/// [`DataLoader::iter`] が返す 1 epoch 分のバッチイテレータ。
///
/// `order`（`shuffle` の有無に応じて事前に確定した添字順列）を先頭から
/// `batch_size` 個ずつ切り出し [`Dataset::batch`] へ渡す。`drop_last`
/// が真の場合、末尾の端数（`batch_size` 未満）は yield しない。
///
/// `pending_error`: [`DataLoader::iter`] が `order` 構築前の
/// `checked_numel_for` 検査で `Err` を得た場合に保持する（`order` は
/// 空のまま）。`next()` はこれを最初の呼び出しで 1 回だけ `Err` として
/// yield し（[`DataError::Shape`]）、以降は通常の枯渇イテレータとして
/// `None` を返す（`order` が空のため `cursor >= order.len()` が常に
/// 真になる）。
pub struct Batches<'a, D: Dataset> {
    dataset: &'a D,
    order: Vec<usize>,
    batch_size: usize,
    drop_last: bool,
    cursor: usize,
    pending_error: Option<DataError>,
}

impl<D: Dataset> Batches<'_, D> {
    /// 残バッチ数（`ExactSizeIterator::len` と同一の計算式）。
    /// `pending_error` が残っている間はそれ自体が未 yield の 1 件
    /// として数える（実際に `next()` が返す件数と一致させる）。
    fn remaining_batches(&self) -> usize {
        if self.pending_error.is_some() {
            return 1;
        }
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
        if let Some(err) = self.pending_error.take() {
            return Some(Err(err));
        }
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

    /// `Dataset::len()` が `Vec<usize>` の allocation 上限を超える
    /// 極端な shape（`Tensor::new(vec![], &[usize::MAX, 0])`。要素数は
    /// ゼロのため構築自体は成功する）に対し、`iter()`（非 shuffle）が
    /// capacity overflow で panic せず、[`DataError::Shape`]
    /// （[`ShapeError::ElementCountOverflow`]）を最初の `next()` で
    /// 1 回だけ yield することを確認する（PR #1867 codex-review P1
    /// 是正の再現例）。
    #[test]
    fn iter_reports_overflow_instead_of_panicking_without_shuffle() {
        let huge = Tensor::<f32>::new(vec![], &[usize::MAX, 0]).unwrap();
        let ds = TensorDataset::new(huge).unwrap();
        let loader = DataLoader::new(ds, DataLoaderConfig::new(1)).unwrap();

        let mut it = loader.iter();
        assert_eq!(it.len(), 1, "pending_error 分の 1 件のみ");
        // `Tensor<f32>` は `PartialEq` 非実装のため `Result` ごとの
        // `assert_eq!` は使えず、`Err` variant のみを直接照合する。
        match it.next() {
            Some(Err(DataError::Shape(ShapeError::ElementCountOverflow))) => {}
            other => panic!("expected Some(Err(ElementCountOverflow)), got {other:?}"),
        }
        assert_eq!(it.len(), 0);
        assert!(it.next().is_none(), "枯渇後は None を返し続ける");
        assert!(it.next().is_none());
    }

    /// 上記の `shuffle=true` 版。加えてオーバーフロー検査に失敗した
    /// 場合はグローバル RNG を一切消費しないことを確認する
    /// （`shuffle=false` と同じ「eager 確定前に検査する」契約。
    /// `DataLoader::iter` doc 参照）。
    #[test]
    fn iter_reports_overflow_instead_of_panicking_with_shuffle_and_consumes_no_rng() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let huge = Tensor::<f32>::new(vec![], &[usize::MAX, 0]).unwrap();
        let ds = TensorDataset::new(huge).unwrap();
        let loader = DataLoader::new(ds, DataLoaderConfig::new(1).shuffle(true)).unwrap();

        manual_seed(42);
        let mut it = loader.iter();
        assert_eq!(it.len(), 1);
        match it.next() {
            Some(Err(DataError::Shape(ShapeError::ElementCountOverflow))) => {}
            other => panic!("expected Some(Err(ElementCountOverflow)), got {other:?}"),
        }
        assert!(it.next().is_none());
        let after_overflow_iter: Vec<f32> = crate::rng::rand(&[4]).unwrap().host_slice().to_vec();

        manual_seed(42);
        let direct: Vec<f32> = crate::rng::rand(&[4]).unwrap().host_slice().to_vec();

        assert_eq!(
            after_overflow_iter, direct,
            "overflow 検査失敗時は shuffle=true でも RNG を消費しない"
        );
    }

    // =====================================================================
    // Sampler／フック（イシュー #2182）
    // =====================================================================

    #[test]
    fn sequential_sampler_matches_data_loader_without_shuffle() {
        for drop_last in [false, true] {
            let ds = TensorDataset::new(tensor_2d(10, 2)).unwrap();
            let loader =
                DataLoader::new(ds, DataLoaderConfig::new(4).drop_last(drop_last)).unwrap();
            let expected: Vec<Vec<f32>> = loader
                .iter()
                .map(|b| b.unwrap().host_slice().to_vec())
                .collect();

            let ds2 = TensorDataset::new(tensor_2d(10, 2)).unwrap();
            let sampler = SequentialSampler::new(10, 4, drop_last).unwrap();
            let mut sampler_loader = SamplerDataLoader::new(ds2, sampler).unwrap();
            let actual: Vec<Vec<f32>> = sampler_loader
                .iter()
                .map(|b| b.unwrap().host_slice().to_vec())
                .collect();
            assert_eq!(actual, expected, "drop_last={drop_last}");
        }

        // RNG を消費しないことの確認。
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        manual_seed(11);
        let ds = TensorDataset::new(tensor_2d(10, 1)).unwrap();
        let mut loader =
            SamplerDataLoader::new(ds, SequentialSampler::new(10, 3, false).unwrap()).unwrap();
        for b in loader.iter() {
            b.unwrap();
        }
        let after: Vec<f32> = crate::rng::rand(&[4]).unwrap().host_slice().to_vec();
        manual_seed(11);
        let direct: Vec<f32> = crate::rng::rand(&[4]).unwrap().host_slice().to_vec();
        assert_eq!(after, direct);
    }

    #[test]
    fn random_sampler_matches_data_loader_shuffle_under_manual_seed() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        for drop_last in [false, true] {
            manual_seed(202);
            let ds = TensorDataset::new(tensor_2d(9, 1)).unwrap();
            let loader = DataLoader::new(
                ds,
                DataLoaderConfig::new(4).shuffle(true).drop_last(drop_last),
            )
            .unwrap();
            let expected: Vec<Vec<f32>> = loader
                .iter()
                .map(|b| b.unwrap().host_slice().to_vec())
                .collect();

            manual_seed(202);
            let ds2 = TensorDataset::new(tensor_2d(9, 1)).unwrap();
            let mut sampler_loader =
                SamplerDataLoader::new(ds2, RandomSampler::new(9, 4, drop_last).unwrap()).unwrap();
            let actual: Vec<Vec<f32>> = sampler_loader
                .iter()
                .map(|b| b.unwrap().host_slice().to_vec())
                .collect();
            assert_eq!(actual, expected, "drop_last={drop_last}");
        }
    }

    #[test]
    fn random_sampler_empty_dataset_consumes_no_rng() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        manual_seed(303);
        let ds = TensorDataset::new(tensor_2d(1, 1).narrow(0, 0, 0).unwrap()).unwrap();
        let mut loader =
            SamplerDataLoader::new(ds, RandomSampler::new(0, 4, false).unwrap()).unwrap();
        assert!(loader.iter().next().is_none());
        let after: Vec<f32> = crate::rng::rand(&[4]).unwrap().host_slice().to_vec();
        manual_seed(303);
        let direct: Vec<f32> = crate::rng::rand(&[4]).unwrap().host_slice().to_vec();
        assert_eq!(
            after, direct,
            "len == 0 の RandomSampler は epoch 開始時に RNG を消費しない\
             （shuffled_indices(0, ..) が rng を呼ばないため）"
        );
    }

    #[test]
    fn weighted_sampler_matches_rng_multinomial() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        for replacement in [true, false] {
            let weights = vec![1.0f32, 2.0, 3.0, 4.0];
            manual_seed(404);
            let w_tensor = Tensor::new(weights.clone(), &[4]).unwrap();
            let expected = crate::rng::multinomial(&w_tensor, 4, replacement)
                .unwrap()
                .host_slice()
                .to_vec();

            manual_seed(404);
            let ds = TensorDataset::new(tensor_2d(4, 1)).unwrap();
            let sampler = WeightedRandomSampler::new(weights, 4, replacement, 4, false).unwrap();
            let mut loader = SamplerDataLoader::new(ds, sampler).unwrap();
            let mut actual_indices: Vec<i32> = Vec::new();
            for b in loader.iter() {
                let batch = b.unwrap();
                actual_indices.extend(batch.host_slice().iter().map(|&v| v as i32));
            }
            // バッチは元データ列の値そのもの（tensor_2d の行値は
            // `row_index as f32`）のため、抽選添字列と一致する。
            assert_eq!(actual_indices, expected, "replacement={replacement}");
        }
    }

    #[test]
    fn weighted_sampler_rejects_invalid_weights_without_consuming_rng() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let cases: Vec<Vec<f32>> = vec![vec![-1.0, 2.0], vec![f32::NAN, 1.0], vec![0.0, 0.0]];
        for weights in cases {
            manual_seed(505);
            let err = WeightedRandomSampler::new(weights, 2, true, 2, false).unwrap_err();
            assert!(matches!(err, DataError::Rng(_)));
            let after: Vec<f32> = crate::rng::rand(&[2]).unwrap().host_slice().to_vec();
            manual_seed(505);
            let direct: Vec<f32> = crate::rng::rand(&[2]).unwrap().host_slice().to_vec();
            assert_eq!(after, direct, "検証失敗時は RNG を消費しない");
        }

        // 非復元抽出で重み不足。
        let err = WeightedRandomSampler::new(vec![1.0, 0.0], 2, false, 2, false).unwrap_err();
        assert!(matches!(err, DataError::Rng(_)));

        // batch_size == 0。
        let err = WeightedRandomSampler::new(vec![1.0], 1, true, 0, false).unwrap_err();
        assert_eq!(err, DataError::ZeroBatchSize);
    }

    #[test]
    fn sampler_num_batches_matches_yield_count() {
        for drop_last in [false, true] {
            let ds = TensorDataset::new(tensor_2d(10, 1)).unwrap();
            let mut loader =
                SamplerDataLoader::new(ds, SequentialSampler::new(10, 4, drop_last).unwrap())
                    .unwrap();
            let expected = loader.num_batches();
            let actual = loader.iter().count();
            assert_eq!(Some(actual), expected, "drop_last={drop_last}");
        }
    }

    #[test]
    fn sampler_data_loader_tuple_dataset_shares_indices() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        manual_seed(606);

        let features = TensorDataset::new(tensor_2d(12, 1)).unwrap();
        let labels_data: Vec<f32> = (0..12).map(|v| (v * 100) as f32).collect();
        let labels = TensorDataset::new(tensor_1d(&labels_data)).unwrap();

        let sampler = RandomSampler::new(12, 4, false).unwrap();
        let mut loader = SamplerDataLoader::new((features, labels), sampler).unwrap();
        for (x, y) in loader.iter().map(|b| b.unwrap()) {
            let xs = x.host_slice();
            let ys = y.host_slice();
            for (xv, yv) in xs.iter().zip(ys.iter()) {
                assert_eq!(*yv, xv * 100.0, "x={xv} y={yv} はずれた添字を指している");
            }
        }
    }

    /// 範囲外の添字を返すカスタム [`Sampler`]。
    struct OutOfRangeSampler {
        yielded: bool,
    }

    impl Sampler for OutOfRangeSampler {
        fn start_epoch(&mut self) -> Result<(), DataError> {
            self.yielded = false;
            Ok(())
        }

        fn next_batch(&mut self) -> Vec<usize> {
            if self.yielded {
                Vec::new()
            } else {
                self.yielded = true;
                vec![0, 9999]
            }
        }
    }

    #[test]
    fn custom_sampler_out_of_range_is_index_out_of_range() {
        let ds = TensorDataset::new(tensor_2d(4, 1)).unwrap();
        let mut loader = SamplerDataLoader::new(ds, OutOfRangeSampler { yielded: false }).unwrap();
        let err = loader.iter().next().unwrap().unwrap_err();
        assert_eq!(
            err,
            DataError::IndexOutOfRange {
                index: 9999,
                len: 4
            }
        );
    }

    /// `start_epoch` が常に失敗するカスタム [`Sampler`]。
    struct FailingStartEpochSampler;

    impl Sampler for FailingStartEpochSampler {
        fn start_epoch(&mut self) -> Result<(), DataError> {
            Err(DataError::ZeroBatchSize)
        }

        fn next_batch(&mut self) -> Vec<usize> {
            panic!("start_epoch が失敗したら next_batch は呼ばれないはず");
        }
    }

    #[test]
    fn sampler_start_epoch_error_is_yielded_once() {
        let ds = TensorDataset::new(tensor_2d(4, 1)).unwrap();
        let mut loader = SamplerDataLoader::new(ds, FailingStartEpochSampler).unwrap();
        let mut it = loader.iter();
        match it.next() {
            Some(Err(DataError::ZeroBatchSize)) => {}
            other => panic!("expected Some(Err(ZeroBatchSize)), got {other:?}"),
        }
        assert!(it.next().is_none());
        assert!(it.next().is_none());
    }

    #[test]
    fn hooked_default_collate_is_bit_identical_to_dataset_batch() {
        // fast path（フックなし）。
        let ds = TensorDataset::new(tensor_2d(10, 3)).unwrap();
        let reference = ds.batch(&[0, 3, 7]).unwrap();
        let ds2 = TensorDataset::new(tensor_2d(10, 3)).unwrap();
        let mut loader =
            HookedDataLoader::new(ds2, SequentialSampler::new(10, 3, false).unwrap()).unwrap();
        let first = loader.iter().next().unwrap().unwrap();
        assert_eq!(
            first.host_slice().as_ref(),
            ds.batch(&[0, 1, 2]).unwrap().host_slice().as_ref()
        );
        let _ = reference;

        // slow path（恒等 transform を明示指定）が fast path と一致する。
        let ds3 = TensorDataset::new(tensor_2d(10, 3)).unwrap();
        let mut loader_slow =
            HookedDataLoader::new(ds3, SequentialSampler::new(10, 3, false).unwrap())
                .unwrap()
                .with_transform(|t| t);
        let first_slow = loader_slow.iter().next().unwrap().unwrap();
        assert_eq!(
            first.host_slice().as_ref(),
            first_slow.host_slice().as_ref()
        );

        // 非 contiguous（transpose した view）の dataset でも一致する。
        let base = tensor_2d(3, 4);
        let transposed = base.transpose(0, 1).unwrap();
        let ds_t = TensorDataset::new(transposed.contiguous()).unwrap();
        let ds_t2 = TensorDataset::new(transposed.contiguous()).unwrap();
        let expected_t = ds_t.batch(&[0, 1]).unwrap();
        let mut loader_t =
            HookedDataLoader::new(ds_t2, SequentialSampler::new(4, 2, false).unwrap()).unwrap();
        let actual_t = loader_t.iter().next().unwrap().unwrap();
        assert_eq!(
            expected_t.host_slice().as_ref(),
            actual_t.host_slice().as_ref()
        );

        // rank1 データセット（サンプル shape が `[]`）。
        let labels = tensor_1d(&[10.0, 20.0, 30.0, 40.0]);
        let ds_r1 = TensorDataset::new(labels.clone()).unwrap();
        let ds_r1_slow = TensorDataset::new(labels).unwrap();
        let expected_r1 = ds_r1.batch(&[0, 1, 2, 3]).unwrap();
        let mut loader_r1 =
            HookedDataLoader::new(ds_r1_slow, SequentialSampler::new(4, 4, false).unwrap())
                .unwrap()
                .with_transform(|t| t);
        let actual_r1 = loader_r1.iter().next().unwrap().unwrap();
        assert_eq!(
            expected_r1.host_slice().as_ref(),
            actual_r1.host_slice().as_ref()
        );
    }

    #[test]
    fn hooked_pipeline_order_is_sample_transform_collate() {
        use std::sync::{Arc, Mutex};

        #[derive(Debug, Clone, PartialEq)]
        enum Event {
            Transform(usize),
            Collate(usize),
        }

        let log: Arc<Mutex<Vec<Event>>> = Arc::new(Mutex::new(Vec::new()));

        let ds = TensorDataset::new(tensor_1d(&[1.0, 2.0, 3.0, 4.0])).unwrap();
        let log_t = Arc::clone(&log);
        let log_c = Arc::clone(&log);
        let mut loader = HookedDataLoader::new(ds, SequentialSampler::new(4, 4, false).unwrap())
            .unwrap()
            .with_transform(move |t| {
                let v = t.get(&[]).unwrap_or(0.0);
                log_t.lock().unwrap().push(Event::Transform(v as usize));
                Tensor::new(vec![v * 10.0], &[]).unwrap()
            })
            .with_collate(move |samples| {
                log_c.lock().unwrap().push(Event::Collate(samples.len()));
                default_collate(samples).unwrap()
            });

        let batch = loader.iter().next().unwrap().unwrap();
        assert_eq!(batch.host_slice().as_ref(), &[10.0, 20.0, 30.0, 40.0]);

        let events = log.lock().unwrap().clone();
        // transform は各サンプルについて先に走り、collate は最後に 1 回。
        let transform_count = events
            .iter()
            .filter(|e| matches!(e, Event::Transform(_)))
            .count();
        assert_eq!(transform_count, 4);
        assert_eq!(events.last(), Some(&Event::Collate(4)));
        // collate は変換後の値（10 倍済み）を受け取ったことを確認済み
        // （上の assert_eq! で `[10, 20, 30, 40]` を検証）。
    }

    #[test]
    fn custom_collate_and_try_variants_propagate_errors() {
        let ds = TensorDataset::new(tensor_1d(&[1.0, 2.0])).unwrap();
        let mut loader = HookedDataLoader::new(ds, SequentialSampler::new(2, 2, false).unwrap())
            .unwrap()
            .with_try_transform(|t| {
                let v = t.get(&[]).unwrap_or(0.0);
                if v > 1.5 {
                    Err(DataError::ZeroBatchSize)
                } else {
                    Ok(t)
                }
            });
        let err = loader.iter().next().unwrap().unwrap_err();
        assert_eq!(err, DataError::ZeroBatchSize);

        let ds2 = TensorDataset::new(tensor_1d(&[1.0, 2.0])).unwrap();
        let mut loader2 = HookedDataLoader::new(ds2, SequentialSampler::new(2, 2, false).unwrap())
            .unwrap()
            .with_try_collate(|_samples| Err(DataError::EmptyBatch));
        let err2 = loader2.iter().next().unwrap().unwrap_err();
        assert_eq!(err2, DataError::EmptyBatch);
    }

    #[test]
    fn default_collate_rejects_shape_mismatch_and_empty() {
        let empty: Vec<Tensor<f32>> = Vec::new();
        assert_eq!(default_collate(&empty).unwrap_err(), DataError::EmptyBatch);

        let a = Tensor::new(vec![1.0f32, 2.0], &[2]).unwrap();
        let b = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        let err = default_collate(&[a, b]).unwrap_err();
        assert!(matches!(
            err,
            DataError::SampleShapeMismatch { position: 1, .. }
        ));
    }

    #[test]
    fn loaders_are_send_and_existing_data_loader_auto_traits_unchanged() {
        fn assert_send<T: Send>() {}
        assert_send::<SamplerDataLoader<TensorDataset<f32>>>();
        assert_send::<HookedDataLoader<f32>>();
        // `DataLoader<TensorDataset<f32>>` は `D: Dataset` のみに束縛
        // されるため、既存の auto trait（`Send`／`Sync`。`Dataset` 実装が
        // 両方を満たす場合）は本イシューの変更で退行しない。
        assert_send::<DataLoader<TensorDataset<f32>>>();
    }
}
