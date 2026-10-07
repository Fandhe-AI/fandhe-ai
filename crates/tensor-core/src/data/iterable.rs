//! 長さ不定のストリームを供給する [`IterableDataset`] とそのバッチ化
//! [`IterableDataLoader`]（PyTorch `torch.utils.data.IterableDataset` と
//! `DataLoader(iterable_ds, batch_size=k, drop_last=d)` 相当。
//! イシュー #2662・親 #2660）。
//!
//! # 位置づけ
//!
//! `data.rs` の子モジュールで、map-style の [`super::Dataset`]
//! （`len()` と `batch(&[usize])`）では表現できない「添字アクセス不可・
//! 長さ未知」の供給元向けの別系統の trait を足す。`BackendOps`／`Op`／
//! `Var` を経由しないホスト側ユーティリティで、CUDA／Metal 専用カーネルと
//! REQ-2 の parity は該当しない。呼び出し元は利用者コード。バッチ化は
//! 既存の [`default_collate`]（[`StackSamples`] の `Tensor<T>` 実装が委譲）
//! を再利用する。
//!
//! **facade 非公開**: 公開は承認待ち（#2677・公開は #2679）。既存の公開型・
//! trait へメソッドも blanket impl も足していない。
//!
//! # 契約
//!
//! - グローバル RNG を一切消費しない。シャッフル・`Sampler` との併用は
//!   型の上で存在しない（PyTorch は実行時 `ValueError`）。
//! - バッファは `batch_size` で事前確保しない（`batch_size` は利用者入力で
//!   `usize::MAX` もありうるため。capacity overflow・過大確保を避ける）。
//!   無限ストリームでも保持するのは高々 1 バッチ分である。ストリームが
//!   停止しない場合の待ちは利用者側の責務。
//! - ストリーム途中の `Err`・collate の失敗では、溜めていた部分バッチを
//!   捨てて `Err` を 1 回だけ yield し、以降は `None`（fail-closed）。
//!
//! # セキュリティ考慮（OWASP Top 10）
//!
//! - **A03／A04**: `batch_size == 0` は構築時に [`DataError::ZeroBatchSize`]。
//!   積み上げ時の出力要素数は [`default_collate`] 内の `checked_numel_for`
//!   で確保前に検査される。本番経路で `unwrap`／`expect`／panic を使わない。
//! - **A08**: タプルサンプルは 1 サンプル内で特徴量とラベルが対になるため
//!   成分間の順ずれが構造上起きない。
//!
//! # 利用例
//!
//! ```
//! use fandhe_ai_tensor_core::Tensor;
//! use fandhe_ai_tensor_core::data::{DataError, IterableDataLoader, IterableDataset};
//!
//! struct Counter(usize);
//!
//! impl IterableDataset for Counter {
//!     type Sample = Tensor<f32>;
//!     fn iter_samples(&self) -> Box<dyn Iterator<Item = Result<Tensor<f32>, DataError>> + '_> {
//!         Box::new((0..self.0).map(|i| Ok(Tensor::new(vec![i as f32], &[1])?)))
//!     }
//! }
//!
//! let loader = IterableDataLoader::new(Counter(5), 2, false).unwrap();
//! let sizes: Vec<usize> = loader.iter().map(|b| b.unwrap().shape()[0]).collect();
//! assert_eq!(sizes, vec![2, 2, 1]);
//! ```

use super::{DataError, default_collate};
use crate::element::Element;
use crate::tensor::Tensor;

/// 1 epoch 分のサンプル列を先頭から返す iterable-style データセット。
///
/// [`super::Dataset`] を継承・実装しない（`len()`／添字アクセスを持たない）。
/// 要素を `Result` とするのは、供給元の失敗を panic ではなく型付きエラー
/// （[`DataError::IterableStream`] 等）で伝えるため。
pub trait IterableDataset {
    /// 1 サンプルの型（`Tensor<T>`、または特徴量とラベルのタプル等）。
    type Sample;

    /// サンプル列を先頭から返す（PyTorch `__iter__` 相当）。呼ぶたびに
    /// 新しいストリームを返す（[`IterableDataLoader::iter`] が epoch ごとに
    /// 1 回呼ぶ）。
    fn iter_samples(&self) -> Box<dyn Iterator<Item = Result<Self::Sample, DataError>> + '_>;
}

/// サンプル列（`Vec<Self>`）を 1 バッチへ積み上げる型（[`IterableBatches`]
/// が使う）。
///
/// `pub` である理由: 公開 `impl Iterator for IterableBatches` の `where`
/// 節に現れるため、`pub(crate)` にすると `private_bounds` lint
/// （`-D warnings`）に抵触する。実装は `Tensor<T>` と 2／3 要素タプルのみ。
pub trait StackSamples: Sized {
    /// 積み上げ後のバッチ型。
    type Batch;

    /// `samples` を先頭軸へ積む。空は [`DataError::EmptyBatch`]。
    fn stack_samples(samples: Vec<Self>) -> Result<Self::Batch, DataError>;
}

impl<T: Element> StackSamples for Tensor<T> {
    type Batch = Tensor<T>;

    fn stack_samples(samples: Vec<Self>) -> Result<Self::Batch, DataError> {
        default_collate(&samples)
    }
}

impl<A: StackSamples, B: StackSamples> StackSamples for (A, B) {
    type Batch = (A::Batch, B::Batch);

    fn stack_samples(samples: Vec<Self>) -> Result<Self::Batch, DataError> {
        let (a, b): (Vec<A>, Vec<B>) = samples.into_iter().unzip();
        Ok((A::stack_samples(a)?, B::stack_samples(b)?))
    }
}

impl<A: StackSamples, B: StackSamples, C: StackSamples> StackSamples for (A, B, C) {
    type Batch = (A::Batch, B::Batch, C::Batch);

    fn stack_samples(samples: Vec<Self>) -> Result<Self::Batch, DataError> {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();
        for (x, y, z) in samples {
            a.push(x);
            b.push(y);
            c.push(z);
        }
        Ok((
            A::stack_samples(a)?,
            B::stack_samples(b)?,
            C::stack_samples(c)?,
        ))
    }
}

/// [`IterableDataset`] を `batch_size` ごとに束ねて供給するローダー
/// （PyTorch `DataLoader(iterable_ds, batch_size=k, drop_last=d)` 相当）。
pub struct IterableDataLoader<D: IterableDataset> {
    dataset: D,
    batch_size: usize,
    drop_last: bool,
}

impl<D: IterableDataset> IterableDataLoader<D> {
    /// `batch_size == 0` は [`DataError::ZeroBatchSize`]。
    pub fn new(dataset: D, batch_size: usize, drop_last: bool) -> Result<Self, DataError> {
        if batch_size == 0 {
            return Err(DataError::ZeroBatchSize);
        }
        Ok(Self {
            dataset,
            batch_size,
            drop_last,
        })
    }

    /// 保持しているデータセットへの参照。
    pub fn dataset(&self) -> &D {
        &self.dataset
    }

    /// データセットを取り出す。
    pub fn into_dataset(self) -> D {
        self.dataset
    }

    /// 1 バッチあたりのサンプル数。
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// 端数バッチを捨てるか。
    pub fn drop_last(&self) -> bool {
        self.drop_last
    }

    /// 1 epoch 分のイテレータを返す。呼ぶたびに
    /// [`IterableDataset::iter_samples`] で新しいストリームを開始する。
    pub fn iter(&self) -> IterableBatches<'_, D> {
        IterableBatches {
            stream: self.dataset.iter_samples(),
            batch_size: self.batch_size,
            drop_last: self.drop_last,
            buffer: Vec::new(),
            done: false,
        }
    }
}

/// [`IterableDataLoader::iter`] が返す 1 epoch 分のバッチイテレータ。
///
/// 長さ不定のため `size_hint` は `(0, None)`（終了後は `(0, Some(0))`）。
pub struct IterableBatches<'a, D: IterableDataset> {
    stream: Box<dyn Iterator<Item = Result<D::Sample, DataError>> + 'a>,
    batch_size: usize,
    drop_last: bool,
    // `Vec::with_capacity(batch_size)` を使わない（モジュール doc 参照）。
    buffer: Vec<D::Sample>,
    done: bool,
}

impl<D> IterableBatches<'_, D>
where
    D: IterableDataset,
    D::Sample: StackSamples,
{
    fn flush(&mut self) -> Result<<D::Sample as StackSamples>::Batch, DataError> {
        let samples = std::mem::take(&mut self.buffer);
        let out = <D::Sample as StackSamples>::stack_samples(samples);
        if out.is_err() {
            self.done = true;
        }
        out
    }
}

impl<D> Iterator for IterableBatches<'_, D>
where
    D: IterableDataset,
    D::Sample: StackSamples,
{
    type Item = Result<<D::Sample as StackSamples>::Batch, DataError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.done {
                return None;
            }
            match self.stream.next() {
                Some(Ok(sample)) => {
                    self.buffer.push(sample);
                    if self.buffer.len() >= self.batch_size {
                        return Some(self.flush());
                    }
                }
                Some(Err(err)) => {
                    self.done = true;
                    self.buffer.clear();
                    return Some(Err(err));
                }
                None => {
                    self.done = true;
                    if self.buffer.is_empty() || self.drop_last {
                        self.buffer.clear();
                        return None;
                    }
                    return Some(self.flush());
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done { (0, Some(0)) } else { (0, None) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DataLoader, DataLoaderConfig, TensorDataset};
    use crate::rng::{global_rng_test_lock, manual_seed, with_global_rng};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    type Make<S> = Box<dyn Fn() -> Box<dyn Iterator<Item = Result<S, DataError>>>>;

    /// テスト用の唯一の `IterableDataset` 実装（インベントリ上の
    /// `fn iter_samples` 宣言数を trait 宣言 1 + 本 impl 1 に保つため
    /// 集約する。挙動の違いは `make` クロージャで切り替える）。
    struct FnStream<S> {
        make: Make<S>,
        opened: Arc<AtomicUsize>,
    }

    impl<S: 'static> IterableDataset for FnStream<S> {
        type Sample = S;
        fn iter_samples(&self) -> Box<dyn Iterator<Item = Result<S, DataError>> + '_> {
            self.opened.fetch_add(1, Ordering::SeqCst);
            (self.make)()
        }
    }

    fn row(v: f32) -> Tensor<f32> {
        Tensor::new(vec![v, v + 0.5], &[2]).unwrap()
    }

    fn rows(n: usize) -> FnStream<Tensor<f32>> {
        FnStream {
            make: Box::new(move || Box::new((0..n).map(|i| Ok(row(i as f32))))),
            opened: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn shapes(loader: &IterableDataLoader<FnStream<Tensor<f32>>>) -> Vec<Vec<usize>> {
        loader.iter().map(|b| b.unwrap().shape().to_vec()).collect()
    }

    #[test]
    fn remainder_and_drop_last() {
        let l = IterableDataLoader::new(rows(5), 2, false).unwrap();
        assert_eq!(shapes(&l), vec![vec![2, 2], vec![2, 2], vec![1, 2]]);
        let l = IterableDataLoader::new(rows(5), 2, true).unwrap();
        assert_eq!(shapes(&l), vec![vec![2, 2], vec![2, 2]]);
        let l = IterableDataLoader::new(rows(4), 2, false).unwrap();
        assert_eq!(shapes(&l), vec![vec![2, 2], vec![2, 2]]);
        let l = IterableDataLoader::new(rows(3), 5, false).unwrap();
        assert_eq!(shapes(&l), vec![vec![3, 2]]);
        let l = IterableDataLoader::new(rows(3), 5, true).unwrap();
        assert!(shapes(&l).is_empty());
    }

    #[test]
    fn empty_stream_yields_nothing() {
        let l = IterableDataLoader::new(rows(0), 3, false).unwrap();
        assert!(l.iter().next().is_none());
    }

    #[test]
    fn zero_batch_size_is_rejected() {
        assert!(matches!(
            IterableDataLoader::new(rows(1), 0, false),
            Err(DataError::ZeroBatchSize)
        ));
    }

    #[test]
    fn huge_batch_size_does_not_preallocate() {
        let l = IterableDataLoader::new(rows(3), usize::MAX, false).unwrap();
        assert_eq!(shapes(&l), vec![vec![3, 2]]);
    }

    #[test]
    fn mid_stream_error_is_yielded_once_without_partial_batch() {
        let ds = FnStream::<Tensor<f32>> {
            make: Box::new(|| {
                let items: Vec<Result<Tensor<f32>, DataError>> = vec![
                    Ok(row(0.0)),
                    Ok(row(1.0)),
                    Ok(row(2.0)),
                    Err(DataError::IterableStream {
                        reason: "boom".into(),
                    }),
                    Ok(row(4.0)),
                ];
                Box::new(items.into_iter())
            }),
            opened: Arc::new(AtomicUsize::new(0)),
        };
        let l = IterableDataLoader::new(ds, 2, false).unwrap();
        let mut it = l.iter();
        assert_eq!(it.next().unwrap().unwrap().shape(), &[2, 2]);
        assert!(matches!(
            it.next(),
            Some(Err(DataError::IterableStream { .. }))
        ));
        assert!(it.next().is_none());
        assert!(it.next().is_none());
    }

    #[test]
    fn shape_mismatch_is_reported_then_stops() {
        let ds = FnStream::<Tensor<f32>> {
            make: Box::new(|| {
                let items = vec![
                    Ok(Tensor::new(vec![0.0], &[1]).unwrap()),
                    Ok(Tensor::new(vec![0.0, 1.0], &[2]).unwrap()),
                    Ok(Tensor::new(vec![0.0], &[1]).unwrap()),
                ];
                Box::new(items.into_iter())
            }),
            opened: Arc::new(AtomicUsize::new(0)),
        };
        let l = IterableDataLoader::new(ds, 2, false).unwrap();
        let mut it = l.iter();
        assert!(matches!(
            it.next(),
            Some(Err(DataError::SampleShapeMismatch { position: 1, .. }))
        ));
        assert!(it.next().is_none());
    }

    #[test]
    fn tuple_samples_keep_feature_and_label_order() {
        let ds = FnStream::<(Tensor<f32>, Tensor<i32>)> {
            make: Box::new(|| {
                Box::new((0..5).map(|i| Ok((row(i as f32), Tensor::new(vec![i], &[1]).unwrap()))))
            }),
            opened: Arc::new(AtomicUsize::new(0)),
        };
        let l = IterableDataLoader::new(ds, 2, false).unwrap();
        let batches: Vec<_> = l.iter().map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].1.host_slice().to_vec(), vec![0, 1]);
        assert_eq!(batches[2].1.host_slice().to_vec(), vec![4]);
        assert_eq!(batches[1].0.host_slice().to_vec(), vec![2.0, 2.5, 3.0, 3.5]);
    }

    #[test]
    fn epochs_reopen_stream_and_do_not_repoll_after_end() {
        let polls = Arc::new(AtomicUsize::new(0));
        let p = polls.clone();
        let ds = FnStream::<Tensor<f32>> {
            make: Box::new(move || {
                let p = p.clone();
                let mut i = 0usize;
                Box::new(std::iter::from_fn(move || {
                    p.fetch_add(1, Ordering::SeqCst);
                    i += 1;
                    (i <= 2).then(|| Ok(row(i as f32)))
                }))
            }),
            opened: Arc::new(AtomicUsize::new(0)),
        };
        let l = IterableDataLoader::new(ds, 2, false).unwrap();
        let mut it = l.iter();
        assert!(it.next().is_some());
        assert!(it.next().is_none());
        let after_end = polls.load(Ordering::SeqCst);
        assert!(it.next().is_none());
        assert_eq!(polls.load(Ordering::SeqCst), after_end);
        assert_eq!(it.size_hint(), (0, Some(0)));
        let _ = l.iter().count();
        assert_eq!(l.dataset().opened.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn size_hint_is_unbounded_while_streaming() {
        let l = IterableDataLoader::new(rows(3), 2, false).unwrap();
        assert_eq!(l.iter().size_hint(), (0, None));
    }

    #[test]
    fn accessors_round_trip() {
        let l = IterableDataLoader::new(rows(3), 2, true).unwrap();
        assert_eq!(l.batch_size(), 2);
        assert!(l.drop_last());
        let _ds = l.into_dataset();
    }

    #[test]
    fn does_not_consume_global_rng() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(11);
        let expected = with_global_rng(|r| r.next_u64());
        manual_seed(11);
        let l = IterableDataLoader::new(rows(5), 2, false).unwrap();
        let _ = l.iter().count();
        assert_eq!(with_global_rng(|r| r.next_u64()), expected);
    }

    #[test]
    fn matches_unshuffled_data_loader_bit_for_bit() {
        let data: Vec<f32> = (0..14).map(|v| v as f32 * 0.37).collect();
        let t = Tensor::<f32>::new(data.clone(), &[7, 2]).unwrap();
        let map_ds = TensorDataset::new(t).unwrap();
        let mut cfg = DataLoaderConfig::new(3);
        cfg.shuffle = false;
        let map_loader = DataLoader::new(map_ds, cfg).unwrap();
        let expected: Vec<Vec<f32>> = map_loader
            .iter()
            .map(|b| b.unwrap().host_slice().to_vec())
            .collect();

        let ds =
            FnStream::<Tensor<f32>> {
                make: Box::new(move || {
                    let data = data.clone();
                    Box::new((0..7).map(move |i| {
                        Ok(Tensor::new(data[i * 2..i * 2 + 2].to_vec(), &[2]).unwrap())
                    }))
                }),
                opened: Arc::new(AtomicUsize::new(0)),
            };
        let l = IterableDataLoader::new(ds, 3, false).unwrap();
        let got: Vec<Vec<f32>> = l.iter().map(|b| b.unwrap().host_slice().to_vec()).collect();
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(&expected) {
            assert_eq!(
                g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                e.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}
