//! Dataset の合成ユーティリティ（`Subset`・`ConcatDataset`・`random_split`。
//! PyTorch `torch.utils.data.{Subset, ConcatDataset, random_split}` 相当。
//! イシュー #2661・親 #2660）。
//!
//! # 位置づけ
//!
//! `data.rs`（`Dataset`／`DataLoader` 群）の子モジュールで、既存の
//! [`Dataset`] trait の上に「部分集合・連結・乱数分割」を重ねるだけの
//! ホスト側ユーティリティである。`BackendOps`／`Op`／`Var` を経由せず、
//! CUDA／Metal 専用カーネルと REQ-2 の parity は該当しない（親モジュール
//! doc・`docs/tensor-core-dataset-compose-decision.md` §4）。呼び出し元は
//! 利用者コードと各ローダー（`DataLoader`／`SamplerDataLoader`／
//! `PrefetchDataLoader`）で、合成後の型もそのまま [`Dataset`] として渡せる。
//!
//! **facade 非公開**: 公開は承認待ち（#2677）。既存の公開型・trait へは
//! メソッドも blanket impl も足していない（facade から名指しなしで到達
//! できる公開面拡張になるため。`docs/tensor-core-data-sampler-hooks-decision.md`
//! §1.1 と同じ判断）。
//!
//! # RNG 契約（`random_split`）
//!
//! 順列は親モジュールの `shuffled_indices` を **1 回の**
//! [`with_global_rng`] クロージャ内で 1 回だけ呼んで得る。したがって
//! 同一 `manual_seed` の下で、分割した添字列を順に連結したものは
//! `RandomSampler`／`DataLoader{shuffle=true}` の順列と bit 完全一致する。
//! 検査（長さ合計・割合・`checked_numel_for`）に失敗した場合と `n == 0`
//! の場合は RNG を一切消費しない。PyTorch の `randperm` とは数列が一致しない
//! （RNG 方式の意図的差異。`docs/tensor-core-dataset-compose-decision.md` §3）。
//!
//! # セキュリティ考慮（OWASP Top 10）
//!
//! - **A02**: 分割順は xorshift64*（非暗号学的）。秘匿用途に使わないこと。
//! - **A03／A04**: 添字は構築時と `batch` 時の二重で境界検査し
//!   （[`DataError::IndexOutOfRange`]）、確保前に `checked_numel_for`、
//!   累積和は `checked_add`／`checked_sub`、割合は有限性・範囲を先に検査する。
//!   本番経路で `unwrap`／`expect`／panic を使わない。
//! - **A08**: タプルの長さ不一致は `validate()` 経由で拒否される。

use std::sync::Arc;

use super::{DataError, Dataset, shuffled_indices};
use crate::element::Element;
use crate::error::ShapeError;
use crate::rng::with_global_rng;
use crate::tensor::{Tensor, checked_numel_for};

/// [`Dataset::Batch`] を先頭軸で連結できる型（[`ConcatDataset`] が成分を
/// またぐ添字列を 1 バッチへ束ねるために使う）。
///
/// `pub` である理由: 公開 `impl Dataset for ConcatDataset<D>` の `where`
/// 節に現れるため、`pub(crate)` にすると `private_bounds` lint（`-D warnings`）
/// に抵触する。実装は `Tensor<T>` と 2／3 要素タプルのみ。
pub trait ConcatBatch: Sized {
    /// `parts` を先頭軸で順に連結する。`parts` が空なら
    /// [`DataError::EmptyBatch`]。
    fn concat_batches(parts: Vec<Self>) -> Result<Self, DataError>;
}

impl<T: Element> ConcatBatch for Tensor<T> {
    fn concat_batches(mut parts: Vec<Self>) -> Result<Self, DataError> {
        let Some(first) = parts.first() else {
            return Err(DataError::EmptyBatch);
        };
        if first.rank() == 0 {
            return Err(DataError::ScalarTensor);
        }
        let rest: Vec<usize> = first.shape()[1..].to_vec();
        let mut rows = 0usize;
        for (part, p) in parts.iter().enumerate() {
            if p.rank() == 0 || p.shape()[1..] != rest[..] {
                return Err(DataError::ConcatShapeMismatch {
                    part,
                    expected: first.shape().to_vec(),
                    found: p.shape().to_vec(),
                });
            }
            rows = rows
                .checked_add(p.shape()[0])
                .ok_or(ShapeError::ElementCountOverflow)?;
        }
        if parts.len() == 1 {
            return Ok(parts.remove(0));
        }
        let mut out_shape = Vec::with_capacity(1 + rest.len());
        out_shape.push(rows);
        out_shape.extend_from_slice(&rest);
        // 確保前に要素数・バイト数を検査する（A03/A04）。
        let numel = checked_numel_for::<T>(&out_shape)?;
        let mut data: Vec<T> = Vec::with_capacity(numel);
        for p in &parts {
            data.extend_from_slice(&p.host_slice());
        }
        Ok(Tensor::new(data, &out_shape)?)
    }
}

impl<A: ConcatBatch, B: ConcatBatch> ConcatBatch for (A, B) {
    fn concat_batches(parts: Vec<Self>) -> Result<Self, DataError> {
        let (a, b): (Vec<A>, Vec<B>) = parts.into_iter().unzip();
        Ok((A::concat_batches(a)?, B::concat_batches(b)?))
    }
}

impl<A: ConcatBatch, B: ConcatBatch, C: ConcatBatch> ConcatBatch for (A, B, C) {
    fn concat_batches(parts: Vec<Self>) -> Result<Self, DataError> {
        let mut a = Vec::with_capacity(parts.len());
        let mut b = Vec::with_capacity(parts.len());
        let mut c = Vec::with_capacity(parts.len());
        for (x, y, z) in parts {
            a.push(x);
            b.push(y);
            c.push(z);
        }
        Ok((
            A::concat_batches(a)?,
            B::concat_batches(b)?,
            C::concat_batches(c)?,
        ))
    }
}

/// 元データセットの添字部分集合（PyTorch `Subset` 相当）。
///
/// 元データセットは `Arc` で共有する（[`random_split`] が 1 つの元から
/// 複数の `Subset` を返すため）。重複添字・空の添字列は許容する。
/// PyTorch は参照時に遅延失敗するが、本実装は構築時に範囲外添字を
/// 拒否する（fail-closed。意図的差異）。
pub struct Subset<D: Dataset> {
    dataset: Arc<D>,
    indices: Vec<usize>,
}

impl<D: Dataset> std::fmt::Debug for Subset<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `D` は `Debug` とは限らないため内部を隠す。
        f.debug_struct("Subset")
            .field("len", &self.indices.len())
            .finish_non_exhaustive()
    }
}

impl<D: Dataset> Clone for Subset<D> {
    fn clone(&self) -> Self {
        Self {
            dataset: Arc::clone(&self.dataset),
            indices: self.indices.clone(),
        }
    }
}

impl<D: Dataset> Subset<D> {
    /// `dataset` の `indices` 位置だけを見せる部分集合を構築する。
    pub fn new(dataset: D, indices: Vec<usize>) -> Result<Self, DataError> {
        Self::from_shared(Arc::new(dataset), indices)
    }

    /// 共有済みの元データセットから構築する。`dataset.validate()` を実行し、
    /// 全添字が `dataset.len()` 未満であることを検査する。
    pub fn from_shared(dataset: Arc<D>, indices: Vec<usize>) -> Result<Self, DataError> {
        dataset.validate()?;
        let len = dataset.len();
        for &index in &indices {
            if index >= len {
                return Err(DataError::IndexOutOfRange { index, len });
            }
        }
        Ok(Self { dataset, indices })
    }

    /// 元データセットへの参照。
    pub fn dataset(&self) -> &D {
        &self.dataset
    }

    /// 元データセット内の添字列。
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }
}

impl<D: Dataset> Dataset for Subset<D> {
    type Batch = D::Batch;

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn validate(&self) -> Result<(), DataError> {
        self.dataset.validate()
    }

    fn batch(&self, indices: &[usize]) -> Result<Self::Batch, DataError> {
        let len = self.indices.len();
        for &i in indices {
            if i >= len {
                return Err(DataError::IndexOutOfRange { index: i, len });
            }
        }
        checked_numel_for::<usize>(&[indices.len()])?;
        let mapped: Vec<usize> = indices.iter().map(|&i| self.indices[i]).collect();
        self.dataset.batch(&mapped)
    }
}

/// 同型の複数データセットの先頭軸連結（PyTorch `ConcatDataset` 相当）。
///
/// `cumulative_sizes` は PyTorch と同じ累積和。位置決定は `bisect_right`
/// 同値の `partition_point(|&c| c <= idx)` で、長さ 0 の成分は自然に
/// 飛ばされる。異種成分の連結は対象外（全域 [`Subset`] で型を揃える）。
pub struct ConcatDataset<D: Dataset> {
    datasets: Vec<D>,
    cumulative_sizes: Vec<usize>,
}

impl<D: Dataset> std::fmt::Debug for ConcatDataset<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConcatDataset")
            .field("cumulative_sizes", &self.cumulative_sizes)
            .finish_non_exhaustive()
    }
}

impl<D: Dataset> ConcatDataset<D> {
    /// 空の `datasets` は [`DataError::EmptyConcat`]。各成分の
    /// `validate()` を実行し、累積和の overflow は
    /// [`ShapeError::ElementCountOverflow`]。
    pub fn new(datasets: Vec<D>) -> Result<Self, DataError> {
        if datasets.is_empty() {
            return Err(DataError::EmptyConcat);
        }
        let mut cumulative_sizes = Vec::with_capacity(datasets.len());
        let mut total = 0usize;
        for d in &datasets {
            d.validate()?;
            total = total
                .checked_add(d.len())
                .ok_or(ShapeError::ElementCountOverflow)?;
            cumulative_sizes.push(total);
        }
        Ok(Self {
            datasets,
            cumulative_sizes,
        })
    }

    /// 成分データセット列。
    pub fn datasets(&self) -> &[D] {
        &self.datasets
    }

    /// 累積長（PyTorch `cumulative_sizes`）。
    pub fn cumulative_sizes(&self) -> &[usize] {
        &self.cumulative_sizes
    }

    /// 成分を取り出して消費する。
    pub fn into_datasets(self) -> Vec<D> {
        self.datasets
    }
}

impl<D: Dataset> Dataset for ConcatDataset<D>
where
    D::Batch: ConcatBatch,
{
    type Batch = D::Batch;

    fn len(&self) -> usize {
        self.cumulative_sizes.last().copied().unwrap_or(0)
    }

    fn validate(&self) -> Result<(), DataError> {
        for d in &self.datasets {
            d.validate()?;
        }
        Ok(())
    }

    fn batch(&self, indices: &[usize]) -> Result<Self::Batch, DataError> {
        let len = self.len();
        for &index in indices {
            if index >= len {
                return Err(DataError::IndexOutOfRange { index, len });
            }
        }
        checked_numel_for::<usize>(&[indices.len()])?;
        let Some(first) = self.datasets.first() else {
            return Err(DataError::EmptyConcat);
        };
        if indices.is_empty() {
            return first.batch(&[]);
        }
        // 同一成分が連続する run ごとに取り出し、元の順序で連結する
        // （成分ごとにまとめると行順が入れ替わるため採らない）。
        let mut parts: Vec<D::Batch> = Vec::new();
        let mut run_component = usize::MAX;
        let mut local: Vec<usize> = Vec::new();
        for &idx in indices {
            let k = self.cumulative_sizes.partition_point(|&c| c <= idx);
            let base = if k == 0 {
                0
            } else {
                self.cumulative_sizes[k - 1]
            };
            if k != run_component && !local.is_empty() {
                parts.push(self.datasets[run_component].batch(&local)?);
                local.clear();
            }
            run_component = k;
            local.push(idx - base);
        }
        if !local.is_empty() {
            parts.push(self.datasets[run_component].batch(&local)?);
        }
        D::Batch::concat_batches(parts)
    }
}

/// 長さ列 `lengths`（合計が `dataset.len()` と一致）で乱数分割する
/// （PyTorch `random_split` の整数長版）。RNG 契約はモジュール doc 参照。
pub fn random_split<D: Dataset>(
    dataset: D,
    lengths: &[usize],
) -> Result<Vec<Subset<D>>, DataError> {
    dataset.validate()?;
    split_by_lengths(dataset, lengths)
}

/// 割合列 `fractions`（合計が 1）で乱数分割する（PyTorch `random_split` の
/// 割合版）。各長さは `floor(n * frac)` に余りを先頭から round-robin で
/// 1 ずつ配分する。合計は CPython 3.12+ の `sum` と同じ Neumaier 補償和で
/// 求め、`isclose(sum, 1)`（rel_tol=1e-9）かつ `sum <= 1` を要求する。
pub fn random_split_fractions<D: Dataset>(
    dataset: D,
    fractions: &[f64],
) -> Result<Vec<Subset<D>>, DataError> {
    dataset.validate()?;
    let n = dataset.len();
    let lengths = resolve_fractions(n, fractions)?;
    split_by_lengths(dataset, &lengths)
}

fn resolve_fractions(n: usize, fractions: &[f64]) -> Result<Vec<usize>, DataError> {
    let k = fractions.len();
    for (index, &f) in fractions.iter().enumerate() {
        if !f.is_finite() {
            return Err(DataError::InvalidSplitFraction {
                index,
                reason: "割合が非有限",
            });
        }
        if !(0.0..=1.0).contains(&f) {
            return Err(DataError::InvalidSplitFraction {
                index,
                reason: "割合が 0〜1 の範囲外",
            });
        }
    }
    let sum = compensated_sum(fractions);
    let close = (sum - 1.0).abs() <= 1e-9 * sum.abs().max(1.0);
    if k == 0 || !close || sum > 1.0 {
        return Err(DataError::InvalidSplitFraction {
            index: k,
            reason: "割合の合計が 1 でない（または 1 を超える）",
        });
    }
    let mut lengths = Vec::with_capacity(k);
    let mut total = 0usize;
    for (index, &f) in fractions.iter().enumerate() {
        let v = (n as f64 * f).floor();
        if !(0.0..=n as f64).contains(&v) {
            return Err(DataError::InvalidSplitFraction {
                index,
                reason: "割合から長さを導出できない",
            });
        }
        let l = v as usize;
        total = total
            .checked_add(l)
            .ok_or(ShapeError::ElementCountOverflow)?;
        lengths.push(l);
    }
    let remainder = n.checked_sub(total).ok_or(DataError::SplitLengthMismatch {
        expected: n,
        found: total,
    })?;
    // round-robin で 1 ずつ配分するのと同値（`i % k` 番目へ 1 ずつ）。
    let (base, extra) = (remainder / k, remainder % k);
    for (i, l) in lengths.iter_mut().enumerate() {
        *l += base + usize::from(i < extra);
    }
    Ok(lengths)
}

/// CPython 3.12+ の浮動小数 `sum` と同じ Neumaier 補償和（有限値のみ）。
fn compensated_sum(xs: &[f64]) -> f64 {
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for &x in xs {
        let t = s + x;
        if s.abs() >= x.abs() {
            c += (s - t) + x;
        } else {
            c += (x - t) + s;
        }
        s = t;
    }
    s + c
}

fn split_by_lengths<D: Dataset>(
    dataset: D,
    lengths: &[usize],
) -> Result<Vec<Subset<D>>, DataError> {
    let n = dataset.len();
    let mut total = 0usize;
    for &l in lengths {
        total = total
            .checked_add(l)
            .ok_or(ShapeError::ElementCountOverflow)?;
    }
    if total != n {
        return Err(DataError::SplitLengthMismatch {
            expected: n,
            found: total,
        });
    }
    checked_numel_for::<usize>(&[n])?;
    // ここまで RNG 非消費。`n == 0` も消費しない。
    let perm: Vec<usize> = if n > 0 {
        with_global_rng(|rng| shuffled_indices(n, rng))
    } else {
        Vec::new()
    };
    let shared = Arc::new(dataset);
    let mut out = Vec::with_capacity(lengths.len());
    let mut offset = 0usize;
    for &l in lengths {
        let end = offset + l;
        out.push(Subset::from_shared(
            Arc::clone(&shared),
            perm[offset..end].to_vec(),
        )?);
        offset = end;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DataLoader, DataLoaderConfig, RandomSampler, Sampler, TensorDataset};
    use crate::rng::{global_rng_test_lock, manual_seed};

    fn ds(n: usize, d: usize) -> TensorDataset<f32> {
        let data: Vec<f32> = (0..n * d).map(|v| v as f32).collect();
        TensorDataset::new(Tensor::new(data, &[n, d]).unwrap()).unwrap()
    }

    fn first_col(t: &Tensor<f32>) -> Vec<f32> {
        let d = t.shape()[1];
        t.host_slice().chunks(d).map(|r| r[0]).collect()
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn subset_and_concat_are_send_sync() {
        assert_send_sync::<Subset<TensorDataset<f32>>>();
        assert_send_sync::<ConcatDataset<TensorDataset<f32>>>();
    }

    #[test]
    fn subset_rejects_out_of_range_and_maps_nested() {
        assert!(matches!(
            Subset::new(ds(4, 2), vec![0, 4]),
            Err(DataError::IndexOutOfRange { index: 4, len: 4 })
        ));
        let s = Subset::new(ds(6, 2), vec![5, 1, 1, 3]).unwrap();
        let t = s.batch(&[0, 1, 2, 3]).unwrap();
        assert_eq!(first_col(&t), vec![10.0, 2.0, 2.0, 6.0]);
        let nested = Subset::new(s, vec![3, 0]).unwrap();
        assert_eq!(first_col(&nested.batch(&[0, 1]).unwrap()), vec![6.0, 10.0]);
        assert!(nested.batch(&[2]).is_err());
        let empty = Subset::new(ds(3, 2), vec![]).unwrap();
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.batch(&[]).unwrap().shape(), &[0, 2]);
    }

    #[test]
    fn concat_preserves_row_order_and_skips_empty() {
        let a = ds(3, 2); // 0,2,4
        let b = ds(0, 2);
        let c = TensorDataset::new(
            Tensor::new((0..8).map(|v| 100.0 + v as f32).collect(), &[4, 2]).unwrap(),
        )
        .unwrap(); // 100,102,104,106
        let cat = ConcatDataset::new(vec![a, b, c]).unwrap();
        assert_eq!(cat.cumulative_sizes(), &[3, 3, 7]);
        assert_eq!(cat.len(), 7);
        let t = cat.batch(&[6, 0, 3, 2, 4, 3]).unwrap();
        assert_eq!(first_col(&t), vec![106.0, 0.0, 100.0, 4.0, 102.0, 100.0]);
        assert!(matches!(
            cat.batch(&[7]),
            Err(DataError::IndexOutOfRange { index: 7, len: 7 })
        ));
        assert_eq!(cat.batch(&[]).unwrap().shape(), &[0, 2]);
    }

    #[test]
    fn concat_rejects_empty_and_trailing_shape_mismatch() {
        assert_eq!(
            ConcatDataset::<TensorDataset<f32>>::new(vec![]).unwrap_err(),
            DataError::EmptyConcat
        );
        let cat = ConcatDataset::new(vec![ds(2, 2), ds(2, 3)]).unwrap();
        assert!(matches!(
            cat.batch(&[0, 3]),
            Err(DataError::ConcatShapeMismatch { part: 1, .. })
        ));
    }

    #[test]
    fn concat_tuple_propagates_length_mismatch() {
        let ok = (ds(2, 2), ds(2, 1));
        let bad = (ds(2, 2), ds(3, 1));
        assert!(matches!(
            ConcatDataset::new(vec![ok, bad]),
            Err(DataError::LengthMismatch { .. })
        ));
        let cat = ConcatDataset::new(vec![(ds(2, 2), ds(2, 1)), (ds(1, 2), ds(1, 1))]).unwrap();
        let (x, y) = cat.batch(&[2, 0]).unwrap();
        assert_eq!(x.shape(), &[2, 2]);
        assert_eq!(y.shape(), &[2, 1]);
    }

    #[test]
    fn random_split_matches_random_sampler_permutation() {
        let _g = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        manual_seed(2661);
        let parts = random_split(ds(10, 1), &[3, 3, 4]).unwrap();
        let split: Vec<usize> = parts.iter().flat_map(|s| s.indices().to_vec()).collect();
        manual_seed(2661);
        let mut sampler = RandomSampler::new(10, 10, false).unwrap();
        sampler.start_epoch().unwrap();
        assert_eq!(split, sampler.next_batch());
        let mut sorted = split.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
        manual_seed(2661);
        let again = random_split(ds(10, 1), &[3, 3, 4]).unwrap();
        assert_eq!(parts[0].indices(), again[0].indices());
        manual_seed(1);
        let other = random_split(ds(10, 1), &[3, 3, 4]).unwrap();
        assert_ne!(parts[0].indices(), other[0].indices());
    }

    #[test]
    fn random_split_failures_and_empty_do_not_consume_rng() {
        let _g = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        manual_seed(7);
        let expected = with_global_rng(|r| r.next_u64());
        manual_seed(7);
        assert!(matches!(
            random_split(ds(4, 1), &[1, 1]),
            Err(DataError::SplitLengthMismatch {
                expected: 4,
                found: 2
            })
        ));
        assert!(random_split_fractions(ds(4, 1), &[0.5, 0.6]).is_err());
        assert!(random_split_fractions(ds(4, 1), &[f64::NAN]).is_err());
        let empty = random_split(ds_empty(), &[0, 0]).unwrap();
        assert_eq!(empty.len(), 2);
        assert_eq!(with_global_rng(|r| r.next_u64()), expected);
    }

    fn ds_empty() -> TensorDataset<f32> {
        TensorDataset::new(Tensor::new(Vec::<f32>::new(), &[0, 1]).unwrap()).unwrap()
    }

    #[test]
    fn fractions_resolve_like_pytorch() {
        assert_eq!(resolve_fractions(10, &[0.3, 0.3, 0.4]).unwrap(), [3, 3, 4]);
        assert_eq!(resolve_fractions(7, &[0.5, 0.5]).unwrap(), [4, 3]);
        assert_eq!(
            resolve_fractions(3, &[0.1; 10])
                .unwrap()
                .iter()
                .sum::<usize>(),
            3
        );
        assert_eq!(resolve_fractions(0, &[0.5, 0.5]).unwrap(), [0, 0]);
        assert!(resolve_fractions(5, &[]).is_err());
        assert!(resolve_fractions(5, &[-0.1, 1.1]).is_err());
        assert!(resolve_fractions(5, &[0.5, 0.5000001]).is_err());
    }

    #[test]
    fn composed_datasets_work_with_data_loader() {
        let _g = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        manual_seed(3);
        let parts = random_split(ds(9, 2), &[6, 3]).unwrap();
        let mut it = parts.into_iter();
        let train = it.next().unwrap();
        let loader = DataLoader::new(train, DataLoaderConfig::new(4)).unwrap();
        let rows: usize = loader.iter().map(|b| b.unwrap().shape()[0]).sum();
        assert_eq!(rows, 6);
    }
}
