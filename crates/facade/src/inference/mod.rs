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
//!
//! # speculative decoding（greedy）と連続バッチング第 1 段階（イシュー #2934・親 #2932）
//!
//! `fandhe_ai_autodiff::generate` の [`generate_speculative`]・[`SpeculativeConfig`]、
//! `generate::scheduler` の [`BatchScheduler`]・[`RequestId`]・[`SchedulerLimits`] の 5 名を
//! 純再エクスポートする（facade 独自の newtype・`Tape`／`Sequential` のメソッド・別名は作らない。
//! `fandhe-ai =0.10.0` に対しては追加のみ）。公開形の正は
//! `docs/facade-speculative-decoding-batching-design.md` §17（リポジトリ所有者本人の承認コメント
//! issuecomment-6067263650 の項 1。実装記録は §18）。
//!
//! - [`generate_speculative`]: **Greedy・B = 1 に限る**（Greedy 以外は `Err`。サンプリング版は
//!   公開しない）。入力は `[T]`／`[1, T]`。受理統計は返さない。先読み長 `k` は残り長 `R` に応じて
//!   丸められる（`R >= 2` は `min(k, R - 1)`、`R == 1` は 1 step decode）。`num_kv_layers() == 0` の
//!   モデルは `Err` で拒否する（fail-closed）。
//! - [`BatchScheduler`]: [`SchedulerLimits`] の 3 上限（同時実行数・待ち行列長・`max_length`）は
//!   必須で 0 は拒否する。要求単位の失敗は他の要求から切り離して `take_failed` に積む。完了条件は
//!   `max_length` 到達のみ（EOS 停止なし）。呼び出しスレッド上で同期実行し、スレッドも I/O も持たない。
//! - **対象モデルの契約**: 対象は、生成状態を渡された `caches` だけに持つモデルと、状態を持たない
//!   モデル。「内部状態保持型」（生成状態を、渡された `caches` の外に持つ型。`RefCell<StatefulAttention>`
//!   等）は `num_kv_layers()` の値によらず**対象外**で、出力を保証しない。型でも実行時でも検出できない
//!   ため、利用者が保証する（設計記録 §17.5）。
//! - 乱数は独立した `Generator`（xorshift64*）のみで非暗号。生成 token をセキュリティ用途に使わないこと。
//!
//! **既知の制限（保留論点 3）**: facade だけの利用者は `forward_with_cache` を使えず、[`crate::nn::kv_cache::KvCache`]
//! にもセッターが無い。そのため [`generate_speculative`] に渡せる「`caches` を正しく進めるモデル」を
//! facade の公開面だけで組む経路はなく、下の doctest は fail-closed 契約のみを示す（成功経路は
//! `crates/facade/tests/speculative_batching_facade.rs` の結合テストが担う）。この経路は本イシューでは足さない。
//!
//! ```
//! use fandhe_ai::inference::{
//!     AutoregressiveModel, BatchScheduler, GenerateConfig, SamplingStrategy, SchedulerLimits,
//!     generate,
//! };
//! use fandhe_ai::nn::kv_cache::KvCache;
//! use fandhe_ai::{AutodiffError, Tensor};
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
//!             out[i * 4 + ((id as usize + 1) % 4)] = 1.0;
//!         }
//!         Ok(Tensor::new(out, &[b, l, 4]).unwrap())
//!     }
//! }
//!
//! let limits = SchedulerLimits::new(2, 4, 8).unwrap();
//! let mut scheduler = BatchScheduler::new(limits);
//! let cfg = GenerateConfig::new(5, SamplingStrategy::Greedy);
//! let p0 = Tensor::<i32>::new(vec![0], &[1]).unwrap();
//! let p1 = Tensor::<i32>::new(vec![2], &[1]).unwrap();
//! let id0 = scheduler.submit(&p0, &cfg).unwrap();
//! let id1 = scheduler.submit(&p1, &cfg).unwrap();
//! assert_ne!(id0, id1);
//!
//! let mut finished = Vec::new();
//! for _ in 0..16 {
//!     scheduler.step(&Table).unwrap();
//!     finished.extend(scheduler.take_finished());
//!     if scheduler.queued_len() + scheduler.active_len() == 0 {
//!         break;
//!     }
//! }
//! assert!(scheduler.take_failed().is_empty());
//! assert_eq!(finished.len(), 2);
//! // 各要求の出力は単独の `generate` と token 列が一致する。
//! for (id, out) in &finished {
//!     let prompt = if *id == id0 { &p0 } else { &p1 };
//!     let alone = generate(&Table, prompt, &cfg).unwrap();
//!     assert_eq!(out.as_slice().unwrap(), alone.as_slice().unwrap());
//! }
//! ```
//!
//! ```
//! use fandhe_ai::inference::{
//!     AutoregressiveModel, GenerateConfig, SamplingStrategy, SpeculativeConfig,
//!     generate_speculative,
//! };
//! use fandhe_ai::nn::kv_cache::KvCache;
//! use fandhe_ai::{AutodiffError, Tensor};
//!
//! /// 状態なしのモデル（`num_kv_layers() == 0`）。
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
//!         Ok(Tensor::new(vec![0.0f32; b * l * 4], &[b, l, 4]).unwrap())
//!     }
//! }
//!
//! let prompt = Tensor::<i32>::new(vec![0], &[1]).unwrap();
//! let spec = SpeculativeConfig::new(2);
//!
//! // KV を使わない（`num_kv_layers() == 0` の）モデルは fail-closed で拒否される。
//! let greedy = GenerateConfig::new(5, SamplingStrategy::Greedy);
//! let r = generate_speculative(&Table, &Table, &prompt, &greedy, &spec);
//! assert!(matches!(r, Err(AutodiffError::InvalidArgument(_))));
//!
//! // サンプリング版は公開しない（Greedy 以外は `Err`）。
//! let topk = GenerateConfig::new(5, SamplingStrategy::TopK(2));
//! let r = generate_speculative(&Table, &Table, &prompt, &topk, &spec);
//! assert!(matches!(r, Err(AutodiffError::InvalidArgument(_))));
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

// 連続バッチングのスケジューラ（第 1 段階。イシュー #2934・親 #2932）。autodiff の形を
// そのまま再エクスポートする（承認形。承認根拠: リポジトリ所有者本人の承認コメント
// issuecomment-6067263650 の項 1、公開形の正は `docs/facade-speculative-decoding-batching-design.md`
// §17.2。`api_surface.rs::SCHEDULER_APPROVED_REEXPORT` が完全一致を検査するため、分割・
// 別名・モジュール自体の再エクスポート・別ファイルへの移動をしない）。
pub use fandhe_ai_autodiff::generate::scheduler::{BatchScheduler, RequestId, SchedulerLimits};

// greedy 版 speculative decoding（B = 1。イシュー #2934・親 #2932）。承認根拠・検査は
// 上の連続バッチング文と同じ（`api_surface.rs::SPECULATIVE_APPROVED_REEXPORT`）。
pub use fandhe_ai_autodiff::generate::speculative::{SpeculativeConfig, generate_speculative};
