//! DataLoader 反復推論のフェーズ計測（イシュー #2192・親 #2131）。
//!
//! `compat::sequential::Sequential::run_loader_inference`（`compat/
//! sequential.rs`。本モジュールと同じく `pub(crate)` 限定）が本モジュール
//! の型・thread-local アキュムレータを使って、`Sequential::predict` を
//! バッチ単位で反復する際の内訳（データロード・tape 構築・forward・
//! デバイス転送の各フェーズの累計時間・呼び出し回数）を記録する。
//!
//! facade 公開は保留のため（`crate::PredictBatchesHoldDoctestGuard`
//! doc・`docs/facade-predict-batches-phase-metrics-decision.md` §5）、
//! 本モジュールの全項目は `pub(crate)` に留める。承認後は該当項目を
//! `pub` に昇格し `crate::inference` を `pub mod` にする想定。
//!
//! **`#[cfg(test)]` 分離について**: `Sequential::predict`（既存公開 API。
//! `NoopPhaseRecorder` 経由）はこのモジュールの一部を常時経由するが、
//! `Sequential::run_loader_inference`（facade 未公開・`compat/
//! sequential.rs` で `#[cfg(test)]` 限定）専用の型・関数
//! （`PhaseStat`・`InferencePhaseStats`・`TimingPhaseRecorder`・
//! thread-local アキュムレータ・[`LoaderInferenceInput`]）は、通常
//! ビルドでは呼び出し元が存在せず `dead_code` lint に抵触するため、
//! 同じく `#[cfg(test)]` を付けて隔離する（`compat/training.rs` の
//! `CustomStepHook`／`fit_custom_step_for_test`〈イシュー #2184〉と同じ
//! 方式。承認後、`run_loader_inference` を `pub fn` へ昇格する際に
//! 本ファイルの `#[cfg(test)]` も併せて外す）。

/// 推論の各フェーズの識別子（R2）。`Sequential::run_loader_inference`
/// の内部実装（`compat/sequential.rs`）が計測点ごとに指定する。
/// `DataLoad`／`DeviceTransfer` は `run_loader_inference`（`#[cfg(test)]`
/// 限定）専用のため同じく `#[cfg(test)]` を付ける——`TapeBuild`／
/// `Forward` は [`crate::compat::Sequential::predict`]（常時公開）が
/// `NoopPhaseRecorder` 経由で参照するため無条件で存在する。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InferencePhase {
    /// `DataLoader::iter`／`Batches::next` の呼び出し（バッチの切り出し）。
    #[cfg(test)]
    DataLoad,
    /// tape 経路（`predict_via_tape`）の `crate::tape()`。tape 不要経路
    /// では計上されない（`calls == 0`）。
    TapeBuild,
    /// 層の forward 実行（tape 不要経路は `forward_host` 連鎖、tape 経路
    /// は `tape.var(..)`・`forward`・`to_tensor()`）。
    Forward,
    /// デバイス常駐経路（`predict_resident` 等）のホスト⇔デバイス転送。
    /// CPU 固定の `predict` 経路では発生しないため常に `calls == 0`
    /// （`docs/facade-predict-batches-phase-metrics-decision.md` §2.1）。
    #[cfg(test)]
    DeviceTransfer,
}

/// 推論フェーズの計測点を差し込むためのトレイト（`predict_recorded`・
/// `run_loader_inference` の内部実装が経由する）。単相化されるため
/// [`NoopPhaseRecorder`] 使用時（既定の `Sequential::predict`）はオーバー
/// ヘッドを追加しない。
pub(crate) trait PhaseRecorder {
    /// `phase` の経過時間を計測しつつ `f` を実行し、その戻り値を返す。
    fn record<T>(&mut self, phase: InferencePhase, f: impl FnOnce() -> T) -> T;
}

/// 既定の `Sequential::predict`（単体呼び出し）が使う no-op recorder。
/// hot path への計測オーバーヘッドを避けるため、単体 `predict` は常時
/// 計測しない設計（`docs/facade-predict-batches-phase-metrics-decision.md`
/// §6 スコープ外）。
pub(crate) struct NoopPhaseRecorder;

impl PhaseRecorder for NoopPhaseRecorder {
    #[inline]
    fn record<T>(&mut self, _phase: InferencePhase, f: impl FnOnce() -> T) -> T {
        f()
    }
}

/// [`recorded`] の公開面（`#[cfg(test)]` 限定。`compat/sequential.rs` の
/// `run_loader_inference` からのみ参照する）。クリップィ
/// `items_after_test_module`（`recorded` 内部の `#[cfg(test)] mod tests`
/// より後ろに項目を置けない）を避けるため、`use` は `mod recorded`
/// 宣言より前に置く（Rust の項目宣言は順序非依存のため到達可能性に
/// 影響しない）。
#[cfg(test)]
pub(crate) use recorded::{
    LoaderInferenceInput, TimingPhaseRecorder, clear_inference_phase_stats,
    inference_phase_stats_snapshot, merge_inference_phase_stats,
};

/// `run_loader_inference`（`#[cfg(test)]` 限定）専用の型・関数
/// （PhaseStat 集計・`TimingPhaseRecorder`・thread-local アキュムレータ・
/// [`LoaderInferenceInput`]）。モジュール doc の「`#[cfg(test)]` 分離
/// について」参照。
#[cfg(test)]
mod recorded {
    use std::cell::Cell;
    use std::time::{Duration, Instant};

    use super::{InferencePhase, PhaseRecorder};
    use crate::Tensor;

    /// 1 つの推論フェーズの累計時間・呼び出し回数（R4: 呼び出し前後の
    /// スナップショット差分比較のため `Copy` にし、`Duration::
    /// saturating_add` で overflow による panic を避ける）。
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub(crate) struct PhaseStat {
        total: Duration,
        calls: u64,
    }

    impl PhaseStat {
        /// 1 回分の経過時間を累積する（飽和加算。`Duration::MAX` 付近
        /// でも panic しない）。
        fn add(&mut self, elapsed: Duration) {
            self.total = self.total.saturating_add(elapsed);
            self.calls = self.calls.saturating_add(1);
        }

        /// 累計時間をマイクロ秒で返す（`u128`。`Duration::as_micros` と
        /// 同じ精度で、呼び出し側〈bench-harness 等〉が浮動小数へ変換
        /// しやすい形）。
        pub(crate) fn total_micros(&self) -> u128 {
            self.total.as_micros()
        }

        /// 累計呼び出し回数。
        pub(crate) fn calls(&self) -> u64 {
            self.calls
        }

        /// 飽和減算（`before` より小さくならない。R4 の before/after
        /// 差分用）。
        fn saturating_sub(&self, before: &Self) -> Self {
            Self {
                total: self.total.saturating_sub(before.total),
                calls: self.calls.saturating_sub(before.calls),
            }
        }

        /// 2 つの `PhaseStat` を飽和加算で合算する
        /// （`InferencePhaseStats::merge` が各フィールドに適用する）。
        fn merged(&self, other: &Self) -> Self {
            Self {
                total: self.total.saturating_add(other.total),
                calls: self.calls.saturating_add(other.calls),
            }
        }
    }

    /// `run_loader_inference` 1 回分（または thread-local への蓄積後）の
    /// フェーズ別集計（R2・R4）。`batches`／`samples` は反復したバッチ数・
    /// 総サンプル数（rank 0 バッチは 1 件として飽和加算）。
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub(crate) struct InferencePhaseStats {
        data_load: PhaseStat,
        tape_build: PhaseStat,
        forward: PhaseStat,
        device_transfer: PhaseStat,
        /// `run_loader_inference` 呼び出し全体（成功・失敗を問わない）の
        /// 経過時間。個々のフェーズの合計とは独立に計測する（フェーズ間
        /// の計測されない隙間時間も含むため）。
        total: PhaseStat,
        batches: u64,
        samples: u64,
    }

    impl InferencePhaseStats {
        fn phase_mut(&mut self, phase: InferencePhase) -> &mut PhaseStat {
            match phase {
                InferencePhase::DataLoad => &mut self.data_load,
                InferencePhase::TapeBuild => &mut self.tape_build,
                InferencePhase::Forward => &mut self.forward,
                InferencePhase::DeviceTransfer => &mut self.device_transfer,
            }
        }

        /// 指定フェーズの集計値を返す。
        pub(crate) fn phase(&self, phase: InferencePhase) -> PhaseStat {
            match phase {
                InferencePhase::DataLoad => self.data_load,
                InferencePhase::TapeBuild => self.tape_build,
                InferencePhase::Forward => self.forward,
                InferencePhase::DeviceTransfer => self.device_transfer,
            }
        }

        /// `total`（`run_loader_inference` 呼び出し全体）の集計値。
        pub(crate) fn total(&self) -> PhaseStat {
            self.total
        }

        /// 反復したバッチ数の累計。
        pub(crate) fn batches(&self) -> u64 {
            self.batches
        }

        /// 反復した総サンプル数の累計（飽和加算）。
        pub(crate) fn samples(&self) -> u64 {
            self.samples
        }

        /// `other` を自身へ飽和加算で合算する（thread-local への蓄積用。
        /// `run_loader_inference` は途中で失敗しても、それまでに計測
        /// した分を失わないようこのメソッドで thread-local へ merge
        /// してから返す）。
        fn merge(&mut self, other: &Self) {
            self.data_load = self.data_load.merged(&other.data_load);
            self.tape_build = self.tape_build.merged(&other.tape_build);
            self.forward = self.forward.merged(&other.forward);
            self.device_transfer = self.device_transfer.merged(&other.device_transfer);
            self.total = self.total.merged(&other.total);
            self.batches = self.batches.saturating_add(other.batches);
            self.samples = self.samples.saturating_add(other.samples);
        }

        /// `self`（after）と `before` の飽和差分を返す（R4: before/after
        /// スナップショット比較）。
        pub(crate) fn since(&self, before: &Self) -> Self {
            Self {
                data_load: self.data_load.saturating_sub(&before.data_load),
                tape_build: self.tape_build.saturating_sub(&before.tape_build),
                forward: self.forward.saturating_sub(&before.forward),
                device_transfer: self.device_transfer.saturating_sub(&before.device_transfer),
                total: self.total.saturating_sub(&before.total),
                batches: self.batches.saturating_sub(before.batches),
                samples: self.samples.saturating_sub(before.samples),
            }
        }

        fn record_batch(&mut self, samples: u64) {
            self.batches = self.batches.saturating_add(1);
            self.samples = self.samples.saturating_add(samples);
        }
    }

    /// `run_loader_inference` が使う実計測 recorder。内部に累積した
    /// [`InferencePhaseStats`] を [`Self::into_stats`] で取り出す。
    #[derive(Debug, Clone, Copy, Default)]
    pub(crate) struct TimingPhaseRecorder {
        stats: InferencePhaseStats,
    }

    impl TimingPhaseRecorder {
        pub(crate) fn into_stats(self) -> InferencePhaseStats {
            self.stats
        }

        /// バッチ 1 件分の反復を記録する（`DataLoad`／`Forward` 等の
        /// `record` とは独立に、バッチ数・サンプル数を積む）。
        pub(crate) fn record_batch(&mut self, samples: u64) {
            self.stats.record_batch(samples);
        }

        /// `total`（呼び出し全体）の経過時間を記録する。
        pub(crate) fn record_total(&mut self, elapsed: Duration) {
            self.stats.total.add(elapsed);
        }
    }

    impl PhaseRecorder for TimingPhaseRecorder {
        fn record<T>(&mut self, phase: InferencePhase, f: impl FnOnce() -> T) -> T {
            let start = Instant::now();
            let out = f();
            self.stats.phase_mut(phase).add(start.elapsed());
            out
        }
    }

    thread_local! {
        /// 呼び出しスレッドごとの累計 phase 計測値（R2）。プロセス全体
        /// ではなくスレッド単位にする理由: ロック不要でテスト間の干渉を
        /// 避け、「このスレッドが `run_loader_inference` で計測した
        /// 累計」という単純な意味論にするため（`docs/facade-predict-
        /// batches-phase-metrics-decision.md` §2.2）。承認時にプロセス
        /// 全体集計へ変えるかは同 §5 の確認事項。
        static INFERENCE_PHASE_STATS: Cell<InferencePhaseStats> =
            Cell::new(InferencePhaseStats::default());
    }

    /// 呼び出しスレッドの累計 phase 計測値のスナップショットを返す。
    pub(crate) fn inference_phase_stats_snapshot() -> InferencePhaseStats {
        INFERENCE_PHASE_STATS.with(|cell| cell.get())
    }

    /// 呼び出しスレッドの累計 phase 計測値をゼロへ戻す。
    pub(crate) fn clear_inference_phase_stats() {
        INFERENCE_PHASE_STATS.with(|cell| cell.set(InferencePhaseStats::default()));
    }

    /// `run_loader_inference` の呼び出し 1 回分の計測値を、呼び出し
    /// スレッドの累計へ飽和加算で反映する（途中失敗した試行の計測値も
    /// 失わない）。
    pub(crate) fn merge_inference_phase_stats(delta: &InferencePhaseStats) {
        INFERENCE_PHASE_STATS.with(|cell| {
            let mut current = cell.get();
            current.merge(delta);
            cell.set(current);
        });
    }

    /// `DataLoader<D>::iter()` が生成するバッチ型から、推論対象の
    /// `Tensor<f32>` を取り出すためのトレイト（R1）。ラベル付き
    /// データセット（`(Tensor<f32>, B)`／`(Tensor<f32>, B, C)`）では
    /// ラベルを無視する（Keras `model.predict(dataset)` が特徴量のみを
    /// 使うのと同じ扱い。`docs/facade-predict-batches-phase-metrics-
    /// decision.md` §3）。
    pub(crate) trait LoaderInferenceInput {
        fn inference_input(&self) -> &Tensor<f32>;
    }

    impl LoaderInferenceInput for Tensor<f32> {
        fn inference_input(&self) -> &Tensor<f32> {
            self
        }
    }

    impl<B> LoaderInferenceInput for (Tensor<f32>, B) {
        fn inference_input(&self) -> &Tensor<f32> {
            &self.0
        }
    }

    impl<B, C> LoaderInferenceInput for (Tensor<f32>, B, C) {
        fn inference_input(&self) -> &Tensor<f32> {
            &self.0
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn phase_stat_add_accumulates_total_and_calls() {
            let mut stat = PhaseStat::default();
            stat.add(Duration::from_micros(10));
            stat.add(Duration::from_micros(5));
            assert_eq!(stat.total_micros(), 15);
            assert_eq!(stat.calls(), 2);
        }

        #[test]
        fn phase_stat_add_saturates_instead_of_panicking() {
            let mut stat = PhaseStat {
                total: Duration::MAX,
                calls: u64::MAX,
            };
            stat.add(Duration::from_secs(1));
            assert_eq!(stat.total, Duration::MAX);
            assert_eq!(stat.calls, u64::MAX);
        }

        #[test]
        fn phase_stat_saturating_sub_never_underflows() {
            let before = PhaseStat {
                total: Duration::from_micros(100),
                calls: 10,
            };
            let after = PhaseStat {
                total: Duration::from_micros(50),
                calls: 3,
            };
            let delta = after.saturating_sub(&before);
            assert_eq!(delta.total, Duration::ZERO);
            assert_eq!(delta.calls, 0);
        }

        #[test]
        fn inference_phase_stats_since_returns_only_the_later_delta() {
            let mut a = InferencePhaseStats::default();
            a.phase_mut(InferencePhase::Forward)
                .add(Duration::from_micros(10));
            a.record_batch(4);

            let mut b = a;
            b.phase_mut(InferencePhase::Forward)
                .add(Duration::from_micros(7));
            b.record_batch(4);

            let delta = b.since(&a);
            assert_eq!(delta.phase(InferencePhase::Forward).calls(), 1);
            assert_eq!(delta.phase(InferencePhase::Forward).total_micros(), 7);
            assert_eq!(delta.batches(), 1);
            assert_eq!(delta.samples(), 4);
        }

        #[test]
        fn merge_accumulates_across_multiple_deltas() {
            let mut acc = InferencePhaseStats::default();
            let mut delta1 = InferencePhaseStats::default();
            delta1
                .phase_mut(InferencePhase::DataLoad)
                .add(Duration::from_micros(3));
            delta1.record_batch(2);
            let mut delta2 = InferencePhaseStats::default();
            delta2
                .phase_mut(InferencePhase::DataLoad)
                .add(Duration::from_micros(4));
            delta2.record_batch(3);

            acc.merge(&delta1);
            acc.merge(&delta2);

            assert_eq!(acc.phase(InferencePhase::DataLoad).calls(), 2);
            assert_eq!(acc.phase(InferencePhase::DataLoad).total_micros(), 7);
            assert_eq!(acc.batches(), 2);
            assert_eq!(acc.samples(), 5);
        }

        #[test]
        fn thread_local_snapshot_is_isolated_per_thread() {
            clear_inference_phase_stats();
            let mut delta = InferencePhaseStats::default();
            delta
                .phase_mut(InferencePhase::Forward)
                .add(Duration::from_micros(1));
            delta.record_batch(1);
            merge_inference_phase_stats(&delta);

            let main_snapshot = inference_phase_stats_snapshot();
            assert_eq!(main_snapshot.batches(), 1);

            let other_thread_snapshot = std::thread::spawn(inference_phase_stats_snapshot)
                .join()
                .unwrap();
            // 別スレッドの thread-local は常にゼロから始まる
            // （このスレッドの蓄積が漏れ伝わらない）。
            assert_eq!(other_thread_snapshot.batches(), 0);

            clear_inference_phase_stats();
            assert_eq!(inference_phase_stats_snapshot().batches(), 0);
        }

        #[test]
        fn loader_inference_input_ignores_labels() {
            let x = Tensor::<f32>::new(vec![1.0, 2.0], &[2]).unwrap();
            let y = 42i32;
            let pair = (x.clone(), y);
            assert_eq!(pair.inference_input().shape(), x.shape());
            let triple = (x.clone(), y, "ignored");
            assert_eq!(triple.inference_input().shape(), x.shape());
            assert_eq!(x.inference_input().shape(), x.shape());
        }
    }
}
