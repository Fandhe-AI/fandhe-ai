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
//!
//! # 自己回帰生成 `generate`（イシュー #2575・親 #2499）
//!
//! `fandhe_ai_autodiff::generate` の自己回帰ループを [`AutoregressiveModel`]・
//! [`GenerateConfig`]・[`SamplingStrategy`]・[`generate`] の 4 名で純再エクスポート
//! する（`nn::rnn`／`nn::kv_cache` と同型。facade 独自の newtype・`Tape::generate`・
//! `Sequential::generate` は作らない）。公開形の正は `docs/facade-generate-decision.md`
//! §13.2・§17（リポジトリ所有者本人の承認コメント issuecomment-6033824965）。
//! エラー型は既存の [`crate::AutodiffError`] を流用する。
//!
//! - 入出力は token id 列（`Tensor<i32>`）のみ（トークナイザは対象外）。EOS による
//!   早期停止はなく、常に `max_length` まで生成する。`top_k > vocab` は拒否する。
//! - [`GenerateConfig`] の検証（`validate`）は非公開のまま [`generate`] の入口で必ず
//!   呼ばれる（pub フィールドを書き換えて矛盾させた設定は `Err`）。
//! - 乱数は独立した `Generator`（xorshift64*）のみを使い、グローバル RNG は消費しない。
//!   暗号学的に安全ではないため、生成 token をセキュリティ用途に使わないこと。
//!
//! **既知の制限（未決事項）**: `forward_step` へ渡される `caches: &mut [KvCache]` へ
//! facade のみで書き込む手段はない（`KvCache` に外部からテンソルを入れるセッターは無く、
//! `MultiheadAttention::forward_with_cache` は `docs/kv-cache-design.md` §11.4 のとおり
//! 非公開）。KV キャッシュを使うモデルは `RefCell<nn::kv_cache::StatefulAttention>` を
//! 内部に持ち、[`crate::tape`] で作った `Tape` の `stateful_attention_forward` を
//! 呼ぶ形になり（渡された `caches` は使わず `num_kv_layers()` は 0 を返す）、
//! prefill 前のキャッシュ reset は利用者の責務となる。
//!
//! ```
//! use fandhe_ai::Tensor;
//! use fandhe_ai::nn::kv_cache::KvCache;
//! use fandhe_ai::inference::{AutoregressiveModel, GenerateConfig, SamplingStrategy, generate};
//! use fandhe_ai::AutodiffError;
//!
//! /// 直前トークンだけで logits が決まる表引きモデル（状態なし）。
//! struct Table;
//!
//! impl AutoregressiveModel for Table {
//!     fn num_kv_layers(&self) -> usize {
//!         0
//!     }
//!     fn forward_step(
//!         &self,
//!         new_ids: &Tensor<i32>,
//!         _caches: &mut [KvCache],
//!     ) -> Result<Tensor<f32>, AutodiffError> {
//!         let (b, l) = (new_ids.shape()[0], new_ids.shape()[1]);
//!         let ids = new_ids.as_slice().unwrap();
//!         let mut out = vec![0.0f32; b * l * 4];
//!         for (i, &id) in ids.iter().enumerate() {
//!             // 次トークンは (id + 1) % 4 が最大。
//!             out[i * 4 + ((id as usize + 1) % 4)] = 1.0;
//!         }
//!         Ok(Tensor::new(out, &[b, l, 4]).unwrap())
//!     }
//! }
//!
//! let prompt = Tensor::<i32>::new(vec![0], &[1]).unwrap();
//! let cfg = GenerateConfig::new(5, SamplingStrategy::Greedy);
//! let out = generate(&Table, &prompt, &cfg).unwrap();
//! assert_eq!(out.shape(), &[5]);
//! assert_eq!(out.as_slice().unwrap(), &[0, 1, 2, 3, 0]);
//! ```
pub(crate) mod batch;

pub use batch::{
    InferencePhase, PhaseMetrics, PhaseStat, PredictBatchInput, get_phase_metrics,
    reset_phase_metrics,
};

// 自己回帰生成（イシュー #2575）。autodiff の形をそのまま再エクスポートする（承認形。
// `api_surface.rs::GENERATE_APPROVED_REEXPORT` が完全一致を検査するため、分割・別名・
// 別ファイルへの移動をしない）。
pub use fandhe_ai_autodiff::generate::{
    AutoregressiveModel, GenerateConfig, SamplingStrategy, generate,
};
