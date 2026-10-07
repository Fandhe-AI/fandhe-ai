//! 明示的な添字列を `batch_size`／`drop_last` で束ねる [`BatchSampler`]
//! （PyTorch `torch.utils.data.BatchSampler` 相当。イシュー #2662・親 #2660）。
//!
//! # 位置づけ
//!
//! `data.rs` の子モジュールで、既存の [`Sampler`] trait の具象型を 1 つ足す
//! だけのホスト側ユーティリティ。`BackendOps`／`Op`／`Var` を経由せず、
//! CUDA／Metal 専用カーネルと REQ-2 の parity は該当しない。呼び出し元は
//! 利用者コードと各ローダー（`SamplerDataLoader`／`HookedDataLoader`／
//! `PrefetchDataLoader`）で、切り出し処理は親モジュールの private な
//! `IndexBatcher` を再利用する（切り出し式を重複実装しない）。
//!
//! **facade 非公開**: 公開は承認待ち（#2677・公開は #2679）。
//!
//! # 契約
//!
//! - グローバル RNG を **一切消費しない**。毎 epoch 同じ順で供給する。
//! - 添字の範囲検査は行わない（データセット長を知らないため）。範囲外は
//!   `Dataset::batch` が [`DataError::IndexOutOfRange`] で拒否する
//!   （カスタム [`Sampler`] と同じ既存契約。境界検査を省かない）。
//! - 重複添字・空の順列は許容する。
//! - `num_batches` は PyTorch `len(BatchSampler)` と同じ式。
//!
//! # セキュリティ考慮（OWASP Top 10）
//!
//! - **A03／A04**: `batch_size == 0` は構築時に [`DataError::ZeroBatchSize`]。
//!   本番経路で `unwrap`／`expect`／panic を使わない。
//! - **A02**: 乱数を使わない。

use super::{DataError, IndexBatcher, Sampler};

/// 利用者が与えた添字列を `batch_size` ごとに束ねて供給する [`Sampler`]。
pub struct BatchSampler {
    order: Vec<usize>,
    batcher: IndexBatcher,
}

impl std::fmt::Debug for BatchSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchSampler").finish_non_exhaustive()
    }
}

impl BatchSampler {
    /// `order`（供給する添字の並び）を `batch_size` ごとに束ねる。
    /// `drop_last` が真なら端数バッチを捨てる。`batch_size == 0` は
    /// [`DataError::ZeroBatchSize`]。
    pub fn new(order: Vec<usize>, batch_size: usize, drop_last: bool) -> Result<Self, DataError> {
        if batch_size == 0 {
            return Err(DataError::ZeroBatchSize);
        }
        Ok(Self {
            order,
            batcher: IndexBatcher::new(batch_size, drop_last),
        })
    }
}

impl Sampler for BatchSampler {
    fn start_epoch(&mut self) -> Result<(), DataError> {
        self.batcher.set_order(self.order.clone());
        Ok(())
    }

    fn next_batch(&mut self) -> Vec<usize> {
        self.batcher.next_batch()
    }

    fn num_batches(&self) -> Option<usize> {
        Some(self.batcher.num_batches(self.order.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{SamplerDataLoader, SequentialSampler, TensorDataset};
    use crate::rng::{global_rng_test_lock, manual_seed, with_global_rng};
    use crate::tensor::Tensor;

    fn drain(s: &mut impl Sampler) -> Vec<Vec<usize>> {
        s.start_epoch().unwrap();
        let mut out = Vec::new();
        loop {
            let b = s.next_batch();
            if b.is_empty() {
                break;
            }
            out.push(b);
        }
        out
    }

    #[test]
    fn batches_and_count_match_for_both_drop_last() {
        for n in 0..9usize {
            for bs in 1..=10usize {
                for drop_last in [false, true] {
                    let mut s = BatchSampler::new((0..n).collect(), bs, drop_last).unwrap();
                    let batches = drain(&mut s);
                    assert_eq!(Some(batches.len()), s.num_batches());
                    let flat: Vec<usize> = batches.concat();
                    let kept = if drop_last { n / bs * bs } else { n };
                    assert_eq!(flat, (0..kept).collect::<Vec<_>>());
                }
            }
        }
    }

    #[test]
    fn zero_batch_size_is_rejected() {
        assert_eq!(
            BatchSampler::new(vec![0], 0, false).unwrap_err(),
            DataError::ZeroBatchSize
        );
    }

    #[test]
    fn next_batch_before_start_epoch_is_empty() {
        let mut s = BatchSampler::new(vec![0, 1], 1, false).unwrap();
        assert!(s.next_batch().is_empty());
    }

    #[test]
    fn explicit_order_with_duplicates_repeats_every_epoch() {
        let mut s = BatchSampler::new(vec![3, 3, 0, 2], 3, false).unwrap();
        let first = drain(&mut s);
        assert_eq!(first, vec![vec![3, 3, 0], vec![2]]);
        assert_eq!(drain(&mut s), first);
    }

    #[test]
    fn does_not_consume_global_rng() {
        let _guard = global_rng_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        manual_seed(7);
        let expected = with_global_rng(|r| r.next_u64());
        manual_seed(7);
        let mut s = BatchSampler::new((0..5).collect(), 2, false).unwrap();
        let _ = drain(&mut s);
        assert_eq!(with_global_rng(|r| r.next_u64()), expected);
    }

    #[test]
    fn is_send_and_debug_hides_internals() {
        fn assert_send<T: Send>() {}
        assert_send::<BatchSampler>();
        let s = BatchSampler::new(vec![0], 1, false).unwrap();
        assert_eq!(format!("{s:?}"), "BatchSampler { .. }");
    }

    #[test]
    fn matches_sequential_sampler_batches() {
        let mut a = BatchSampler::new((0..7).collect(), 3, true).unwrap();
        let mut b = SequentialSampler::new(7, 3, true).unwrap();
        assert_eq!(drain(&mut a), drain(&mut b));
    }

    #[test]
    fn feeds_sampler_data_loader_in_given_order() {
        let t = Tensor::<f32>::new(vec![0.0, 1.0, 2.0, 3.0], &[4, 1]).unwrap();
        let ds = TensorDataset::new(t).unwrap();
        let sampler = BatchSampler::new(vec![3, 1, 0], 2, false).unwrap();
        let mut loader = SamplerDataLoader::new(ds, sampler).unwrap();
        let got: Vec<Vec<f32>> = loader
            .iter()
            .map(|b| b.unwrap().host_slice().to_vec())
            .collect();
        assert_eq!(got, vec![vec![3.0, 1.0], vec![0.0]]);
    }

    #[test]
    fn out_of_range_index_is_rejected_by_dataset() {
        let t = Tensor::<f32>::new(vec![0.0, 1.0], &[2, 1]).unwrap();
        let ds = TensorDataset::new(t).unwrap();
        let sampler = BatchSampler::new(vec![0, 5], 2, false).unwrap();
        let mut loader = SamplerDataLoader::new(ds, sampler).unwrap();
        let first = loader.iter().next().unwrap();
        assert!(matches!(
            first,
            Err(DataError::IndexOutOfRange { index: 5, len: 2 })
        ));
    }
}
