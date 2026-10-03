//! Dataset／DataLoader 公開面（イシュー #1615・親 #1602。`docs/
//! dataset-dataloader-design.md`）。
//!
//! `fandhe_ai_tensor_core::data`（`Dataset`・`TensorDataset`・
//! `DataLoader`・`DataLoaderConfig`・`Batches`・`DataError`）と、
//! イシュー #2182 で追加し #2505（ルート #2499 のユーザー承認〈依頼文記載の承認日 2026-10-04〉、
//! `docs/tensor-core-data-sampler-hooks-decision.md` §5）で公開した
//! Sampler／フック系 11 名（`Sampler`・`SequentialSampler`・
//! `RandomSampler`・`WeightedRandomSampler`・`SamplerDataLoader`・
//! `SamplerBatches`・`HookedDataLoader`・`HookedBatches`・`TransformFn`・
//! `CollateFn`・`default_collate`）をそのまま再エクスポートする
//! **純再エクスポートモジュール**（`crate::optim` と同型。facade 独自の
//! 型・関数は持ち込まず、既存 `DataLoader` への統合もしない）。
//!
//! Dataset／DataLoader はホスト側だけで完結するバッチ供給ユーティリティ
//! であり `Op`／`BackendOps`／VJP を経由しないため、`autodiff`
//! （`fandhe_ai_autodiff`）を経由せず `fandhe_ai_tensor_core` から直接
//! 再エクスポートする（`crate::{Device, Tensor}` 等の既存トップレベル
//! 再エクスポートと同じ経路。`fandhe_ai::rng`／`creation` 相当のモジュール
//! を今後追加する場合も同型になる想定）。
//!
//! # 利用例
//!
//! ```
//! use fandhe_ai::data::{DataLoader, DataLoaderConfig, Dataset, TensorDataset};
//! use fandhe_ai::Tensor;
//!
//! let features = Tensor::new(vec![0.0f32, 1.0, 2.0, 3.0], &[4, 1]).unwrap();
//! let labels = Tensor::new(vec![0i32, 1, 0, 1], &[4]).unwrap();
//! let dataset = (
//!     TensorDataset::new(features).unwrap(),
//!     TensorDataset::new(labels).unwrap(),
//! );
//! let loader = DataLoader::new(dataset, DataLoaderConfig::new(2)).unwrap();
//! assert_eq!(loader.len(), 2);
//! for batch in &loader {
//!     let (x, y) = batch.unwrap();
//!     assert_eq!(x.shape()[0], y.shape()[0]);
//! }
//! ```
//!
//! # Sampler／フックの利用例
//!
//! ```
//! use fandhe_ai::Tensor;
//! use fandhe_ai::data::{HookedDataLoader, SequentialSampler, TensorDataset};
//!
//! let features = Tensor::new(vec![0.0f32, 1.0, 2.0, 3.0], &[4, 1]).unwrap();
//! let dataset = TensorDataset::new(features).unwrap();
//! let sampler = SequentialSampler::new(4, 2, false).unwrap();
//! let mut loader = HookedDataLoader::new(dataset, sampler)
//!     .unwrap()
//!     .with_transform(|t| {
//!         let scaled: Vec<f32> = t.host_slice().iter().map(|&v| v * 10.0).collect();
//!         Tensor::new(scaled, t.shape()).unwrap()
//!     });
//! let mut seen = Vec::new();
//! for batch in loader.iter() {
//!     let b = batch.unwrap();
//!     assert_eq!(b.shape(), &[2, 1]);
//!     seen.extend(b.host_slice().iter().copied());
//! }
//! assert_eq!(seen, vec![0.0, 10.0, 20.0, 30.0]);
//! ```

pub use fandhe_ai_tensor_core::data::{Batches, DataError, DataLoader};
pub use fandhe_ai_tensor_core::data::{CollateFn, TransformFn, default_collate};
pub use fandhe_ai_tensor_core::data::{DataLoaderConfig, Dataset, TensorDataset};
pub use fandhe_ai_tensor_core::data::{HookedBatches, HookedDataLoader};
pub use fandhe_ai_tensor_core::data::{RandomSampler, Sampler, SequentialSampler};
pub use fandhe_ai_tensor_core::data::{SamplerBatches, SamplerDataLoader, WeightedRandomSampler};
