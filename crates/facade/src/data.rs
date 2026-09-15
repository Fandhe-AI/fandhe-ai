//! Dataset／DataLoader 公開面（イシュー #1615・親 #1602。`docs/
//! dataset-dataloader-design.md`）。
//!
//! `fandhe_ai_tensor_core::data`（[`Dataset`]・[`TensorDataset`]・
//! [`DataLoader`]・[`DataLoaderConfig`]・[`Batches`]・[`DataError`]）を
//! そのまま再エクスポートする**純再エクスポートモジュール**
//! （`crate::optim` と同型。facade 独自の型・関数は持ち込まない）。
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

pub use fandhe_ai_tensor_core::data::{Batches, DataError, DataLoader};
pub use fandhe_ai_tensor_core::data::{DataLoaderConfig, Dataset, TensorDataset};
