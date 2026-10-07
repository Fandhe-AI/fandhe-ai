//! バッチ推論（`Sequential::predict_batches`）と推論フェーズ計測の公開面
//! （イシュー #2192・#2582・親 #2131・#2581）。
//!
//! `compat::Sequential::predict_batches` が `DataLoader` を全バッチ反復して
//! 推論する際の内訳（データロード・tape 構築・forward・デバイス転送の累計
//! 時間・呼び出し回数）を、本モジュールの [`get_phase_metrics`]・
//! [`reset_phase_metrics`]・[`PhaseMetrics`] で読む。公開形は
//! `docs/facade-predict-batches-phase-metrics-decision.md` §8.4 の確定形
//! （(a)〜(f)）に従い、実体は非公開の `batch` モジュールに置いて本ファイルの
//! `pub use` でフラットに公開する。
//!
//! **集計単位はスレッド単位**: 累計は thread-local で、「呼び出しスレッドが
//! `predict_batches` で計測した累計」を意味する（他スレッドの計測は含まない。
//! ホットパスにロックが要らずテストが分離できるため）。呼び出し前後の差分は
//! [`PhaseMetrics::since`] で取る。[`InferencePhase::DeviceTransfer`] は CPU
//! 固定経路では常に `calls == 0`（予約）。
//!
//! # 使用例
//!
//! ```
//! use fandhe_ai::Tensor;
//! use fandhe_ai::compat::Sequential;
//! use fandhe_ai::data::{DataLoader, DataLoaderConfig, TensorDataset};
//! use fandhe_ai::inference::{InferencePhase, get_phase_metrics, reset_phase_metrics};
//!
//! let model = Sequential::new().add_linear(2, 1, 7).unwrap();
//! let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
//! let loader = DataLoader::new(
//!     TensorDataset::new(x).unwrap(),
//!     DataLoaderConfig::new(2),
//! )
//! .unwrap();
//!
//! reset_phase_metrics();
//! let before = get_phase_metrics();
//! let outputs = model.predict_batches(&loader).unwrap();
//! let delta = get_phase_metrics().since(&before);
//!
//! assert_eq!(outputs.len(), 2);
//! assert_eq!(delta.batches(), 2);
//! assert_eq!(delta.samples(), 3);
//! assert_eq!(delta.phase(InferencePhase::Forward).calls(), 2);
//! assert_eq!(delta.phase(InferencePhase::DeviceTransfer).calls(), 0);
//! assert_eq!(delta.total().calls(), 1);
//! ```
pub(crate) mod batch;

pub use batch::{
    InferencePhase, PhaseMetrics, PhaseStat, PredictBatchInput, get_phase_metrics,
    reset_phase_metrics,
};
