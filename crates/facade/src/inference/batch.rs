//! DataLoader 反復推論（`Sequential::predict_batches`）のフェーズ計測
//! （イシュー #2192・#2582・親 #2131・#2581）。
//!
//! `compat::sequential::Sequential::predict_batches`（`compat/sequential.rs`）
//! が本モジュールの型・thread-local アキュムレータを使って、
//! `Sequential::predict` をバッチ単位で反復する際の内訳（データロード・
//! tape 構築・forward・デバイス転送の各フェーズの累計時間・呼び出し回数）を
//! 記録する。
//!
//! 本ファイルは `pub(crate)` モジュールで、公開項目（`PhaseMetrics`・
//! `PhaseStat`・`InferencePhase`・`PredictBatchInput`・`get_phase_metrics`・
//! `reset_phase_metrics`）は `crate::inference`（`pub mod`）が `pub use` で
//! フラットに公開する（`docs/facade-predict-batches-phase-metrics-decision.md`
//! §8.4 (e)）。`PhaseRecorder`・`NoopPhaseRecorder`・`TimingPhaseRecorder`・
//! `merge_inference_phase_stats` は crate 内専用に留める。

/// 推論の各フェーズの識別子（`PhaseMetrics::phase` の引数）。
///
/// `#[non_exhaustive]`（variant の追加は非破壊）。`Sequential::predict_batches`
/// の内部実装が計測点ごとに指定する。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InferencePhase {
    /// `DataLoader::iter`／`Batches::next` の呼び出し（バッチの切り出し）。
    DataLoad,
    /// tape 経路（`predict_via_tape`）の `crate::tape()`。tape 不要経路
    /// では計上されない（`calls == 0`）。
    TapeBuild,
    /// 層の forward 実行（tape 不要経路は `forward_host` 連鎖、tape 経路
    /// は `tape.var(..)`・`forward`・`to_tensor()`）。
    Forward,
    /// デバイス常駐経路のホスト⇔デバイス転送用に予約された識別子。CPU 固定
    /// の `predict_batches` 経路では転送が発生しないため**常に `calls == 0`**
    /// （`docs/facade-predict-batches-phase-metrics-decision.md` §2.1・§8.4 (c)）。
    DeviceTransfer,
}

/// 推論フェーズの計測点を差し込むためのトレイト（`predict_recorded`・
/// `predict_batches` の内部実装が経由する。crate 内専用）。単相化される
/// ため [`NoopPhaseRecorder`] 使用時（既定の `Sequential::predict`）は
/// オーバーヘッドを追加しない。
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

// 公開 5 名（`crate::inference` が `InferencePhase` と合わせて `pub use` する）。
// `pub(crate) use` は `pub use` で再公開できない（E0364）ため、公開分と crate
// 内専用分を分けて再エクスポートする。`use` は `mod recorded` 宣言より前に置く
// （内部の `#[cfg(test)] mod tests` より後ろに項目を置くと clippy
// `items_after_test_module` に抵触するため。順序は到達可能性に影響しない）。
pub use recorded::{
    PhaseMetrics, PhaseStat, PredictBatchInput, get_phase_metrics, reset_phase_metrics,
};
pub(crate) use recorded::{TimingPhaseRecorder, merge_inference_phase_stats};

/// 計測値の型・thread-local アキュムレータ・入力トレイトの実体。
mod recorded {
    use std::cell::Cell;
    use std::time::{Duration, Instant};

    use super::{InferencePhase, PhaseRecorder};
    use crate::Tensor;

    /// 1 つの推論フェーズの累計時間・呼び出し回数。フィールドは非公開で、
    /// accessor（[`Self::total_micros`]・[`Self::calls`]・[`Self::total`]）
    /// で読む。加算・減算は飽和演算で panic しない（`Duration::MAX`／
    /// `u64::MAX` 付近でも安全）。
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct PhaseStat {
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
        /// 同じ精度で、呼び出し側が浮動小数へ変換しやすい形）。
        pub fn total_micros(&self) -> u128 {
            self.total.as_micros()
        }

        /// 累計呼び出し回数。
        pub fn calls(&self) -> u64 {
            self.calls
        }

        /// 累計時間を `Duration` で返す（`total_micros` と同じ値の
        /// ナノ秒精度版）。
        pub fn total(&self) -> Duration {
            self.total
        }

        /// 飽和減算（`before` より小さくならない。before/after 差分用）。
        fn saturating_sub(&self, before: &Self) -> Self {
            Self {
                total: self.total.saturating_sub(before.total),
                calls: self.calls.saturating_sub(before.calls),
            }
        }

        /// 2 つの `PhaseStat` を飽和加算で合算する
        /// （`PhaseMetrics::merge` が各フィールドに適用する）。
        fn merged(&self, other: &Self) -> Self {
            Self {
                total: self.total.saturating_add(other.total),
                calls: self.calls.saturating_add(other.calls),
            }
        }
    }

    /// `Sequential::predict_batches` が計測した、フェーズ別の集計
    /// （呼び出し 1 回分、または thread-local への蓄積後）。
    ///
    /// フィールドは非公開で accessor（[`Self::phase`]・[`Self::total`]・
    /// [`Self::batches`]・[`Self::samples`]・[`Self::since`]）で読む。
    /// `#[non_exhaustive]`（フィールド追加は非破壊）。`batches`／`samples`
    /// は反復したバッチ数・総サンプル数（入力バッチの先頭軸から数える。
    /// rank 0 バッチは 1 件として飽和加算）。
    ///
    /// # Examples
    ///
    /// ラベル付きデータセット（`(features, labels)`）でもラベルは無視され、特徴量だけが
    /// 推論される。呼び出し前後のスナップショット差分は [`Self::since`] で取る
    /// （時間値は環境依存のため、ここでは回数のみ確認する）。
    ///
    /// ```
    /// use fandhe_ai::Tensor;
    /// use fandhe_ai::compat::Sequential;
    /// use fandhe_ai::data::{DataLoader, DataLoaderConfig, TensorDataset};
    /// use fandhe_ai::inference::{InferencePhase, PhaseMetrics, get_phase_metrics};
    ///
    /// let model = Sequential::new().add_linear(2, 1, 7).unwrap();
    /// let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let y = Tensor::<i32>::new(vec![0, 1], &[2]).unwrap();
    /// let loader = DataLoader::new(
    ///     (TensorDataset::new(x).unwrap(), TensorDataset::new(y).unwrap()),
    ///     DataLoaderConfig::new(1),
    /// )
    /// .unwrap();
    ///
    /// let before: PhaseMetrics = get_phase_metrics();
    /// let outputs = model.predict_batches(&loader).unwrap();
    /// let delta = get_phase_metrics().since(&before);
    ///
    /// assert_eq!(outputs.len(), 2);
    /// assert_eq!(delta.batches(), 2);
    /// assert_eq!(delta.samples(), 2);
    /// assert!(delta.phase(InferencePhase::DataLoad).calls() >= 2);
    /// let _micros: u128 = delta.phase(InferencePhase::Forward).total_micros();
    /// assert_eq!(delta.phase(InferencePhase::DeviceTransfer).calls(), 0);
    /// ```
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    #[non_exhaustive]
    pub struct PhaseMetrics {
        data_load: PhaseStat,
        tape_build: PhaseStat,
        forward: PhaseStat,
        device_transfer: PhaseStat,
        /// `predict_batches` 呼び出し全体（成功・失敗を問わない）の
        /// 経過時間。個々のフェーズの合計とは独立に計測する（フェーズ間
        /// の計測されない隙間時間も含むため）。
        total: PhaseStat,
        batches: u64,
        samples: u64,
    }

    impl PhaseMetrics {
        fn phase_mut(&mut self, phase: InferencePhase) -> &mut PhaseStat {
            match phase {
                InferencePhase::DataLoad => &mut self.data_load,
                InferencePhase::TapeBuild => &mut self.tape_build,
                InferencePhase::Forward => &mut self.forward,
                InferencePhase::DeviceTransfer => &mut self.device_transfer,
            }
        }

        /// 指定フェーズの集計値を返す。
        pub fn phase(&self, phase: InferencePhase) -> PhaseStat {
            match phase {
                InferencePhase::DataLoad => self.data_load,
                InferencePhase::TapeBuild => self.tape_build,
                InferencePhase::Forward => self.forward,
                InferencePhase::DeviceTransfer => self.device_transfer,
            }
        }

        /// `predict_batches` 呼び出し全体の集計値（`calls` は呼び出し回数）。
        pub fn total(&self) -> PhaseStat {
            self.total
        }

        /// 反復したバッチ数の累計。
        pub fn batches(&self) -> u64 {
            self.batches
        }

        /// 反復した総サンプル数の累計（飽和加算）。
        pub fn samples(&self) -> u64 {
            self.samples
        }

        /// `other` を自身へ飽和加算で合算する（thread-local への蓄積用。
        /// `predict_batches` は途中で失敗しても、それまでに計測した分を
        /// 失わないようこのメソッドで thread-local へ merge してから返す）。
        fn merge(&mut self, other: &Self) {
            self.data_load = self.data_load.merged(&other.data_load);
            self.tape_build = self.tape_build.merged(&other.tape_build);
            self.forward = self.forward.merged(&other.forward);
            self.device_transfer = self.device_transfer.merged(&other.device_transfer);
            self.total = self.total.merged(&other.total);
            self.batches = self.batches.saturating_add(other.batches);
            self.samples = self.samples.saturating_add(other.samples);
        }

        /// `self`（after）と `before` の飽和差分を返す（before/after
        /// スナップショット比較。`before` が大きい項目は 0 になる）。
        pub fn since(&self, before: &Self) -> Self {
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

    /// `predict_batches` が使う実計測 recorder（crate 内専用）。内部に累積した
    /// [`PhaseMetrics`] を [`Self::into_metrics`] で取り出す。
    #[derive(Debug, Clone, Copy, Default)]
    pub(crate) struct TimingPhaseRecorder {
        metrics: PhaseMetrics,
    }

    impl TimingPhaseRecorder {
        pub(crate) fn into_metrics(self) -> PhaseMetrics {
            self.metrics
        }

        /// バッチ 1 件分の反復を記録する（`DataLoad`／`Forward` 等の
        /// `record` とは独立に、バッチ数・サンプル数を積む）。
        pub(crate) fn record_batch(&mut self, samples: u64) {
            self.metrics.record_batch(samples);
        }

        /// `total`（呼び出し全体）の経過時間を記録する。
        pub(crate) fn record_total(&mut self, elapsed: Duration) {
            self.metrics.total.add(elapsed);
        }
    }

    impl PhaseRecorder for TimingPhaseRecorder {
        fn record<T>(&mut self, phase: InferencePhase, f: impl FnOnce() -> T) -> T {
            let start = Instant::now();
            let out = f();
            self.metrics.phase_mut(phase).add(start.elapsed());
            out
        }
    }

    thread_local! {
        /// 呼び出しスレッドごとの累計 phase 計測値。プロセス全体ではなく
        /// スレッド単位にする理由: ホットパスにロックが不要でテスト間の
        /// 干渉を避け、「このスレッドが `predict_batches` で計測した
        /// 累計」という単純な意味論にするため（`docs/facade-predict-
        /// batches-phase-metrics-decision.md` §2.3・§8.4 (f)）。
        static INFERENCE_PHASE_METRICS: Cell<PhaseMetrics> =
            Cell::new(PhaseMetrics::default());
    }

    /// 呼び出しスレッドが `Sequential::predict_batches` で計測した累計
    /// phase 計測値のスナップショットを返す（スレッド単位。他スレッドの
    /// 計測は含まない）。呼び出し前後の差分は [`PhaseMetrics::since`] で
    /// 取る。
    pub fn get_phase_metrics() -> PhaseMetrics {
        INFERENCE_PHASE_METRICS.with(|cell| cell.get())
    }

    /// 呼び出しスレッドの累計 phase 計測値をゼロへ戻す（他スレッドの
    /// 累計には影響しない）。
    pub fn reset_phase_metrics() {
        INFERENCE_PHASE_METRICS.with(|cell| cell.set(PhaseMetrics::default()));
    }

    /// `predict_batches` の呼び出し 1 回分の計測値を、呼び出し
    /// スレッドの累計へ飽和加算で反映する（途中失敗した試行の計測値も
    /// 失わない。crate 内専用）。
    pub(crate) fn merge_inference_phase_stats(delta: &PhaseMetrics) {
        INFERENCE_PHASE_METRICS.with(|cell| {
            let mut current = cell.get();
            current.merge(delta);
            cell.set(current);
        });
    }

    /// 封印用の private supertrait（`PredictBatchInput` の外部実装を
    /// 禁止する。後から unseal するのは非破壊だが逆は破壊的なため。
    /// 決定記録 §8.4 (a)）。
    mod sealed {
        pub trait Sealed {}
    }

    /// `DataLoader<D>` が生成するバッチ型から、推論対象の `Tensor<f32>` を
    /// 取り出すための封印済みトレイト。`Sequential::predict_batches` の
    /// `D::Batch` 境界に使う。ラベル付きデータセット
    /// （`(Tensor<f32>, B)`／`(Tensor<f32>, B, C)`）ではラベルを無視する
    /// （Keras `model.predict(dataset)` が特徴量のみを使うのと同じ扱い。
    /// 決定記録 §3）。
    ///
    /// 実装は `Tensor<f32>`・`(Tensor<f32>, B)`・`(Tensor<f32>, B, C)` の
    /// 3 形に固定され、利用者が実装することも、メソッドを呼ぶ必要もない。
    pub trait PredictBatchInput: sealed::Sealed {
        /// 推論入力の `Tensor<f32>` への参照（ラベルは含まない）。
        fn inference_input(&self) -> &Tensor<f32>;
    }

    impl sealed::Sealed for Tensor<f32> {}
    impl<B> sealed::Sealed for (Tensor<f32>, B) {}
    impl<B, C> sealed::Sealed for (Tensor<f32>, B, C) {}

    impl PredictBatchInput for Tensor<f32> {
        fn inference_input(&self) -> &Tensor<f32> {
            self
        }
    }

    impl<B> PredictBatchInput for (Tensor<f32>, B) {
        fn inference_input(&self) -> &Tensor<f32> {
            &self.0
        }
    }

    impl<B, C> PredictBatchInput for (Tensor<f32>, B, C) {
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
        fn phase_stat_total_matches_total_micros() {
            let mut stat = PhaseStat::default();
            stat.add(Duration::from_micros(1500));
            assert_eq!(stat.total(), Duration::from_micros(1500));
            assert_eq!(stat.total().as_micros(), stat.total_micros());
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
        fn phase_metrics_since_returns_only_the_later_delta() {
            let mut a = PhaseMetrics::default();
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
            let mut acc = PhaseMetrics::default();
            let mut delta1 = PhaseMetrics::default();
            delta1
                .phase_mut(InferencePhase::DataLoad)
                .add(Duration::from_micros(3));
            delta1.record_batch(2);
            let mut delta2 = PhaseMetrics::default();
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
            reset_phase_metrics();
            let mut delta = PhaseMetrics::default();
            delta
                .phase_mut(InferencePhase::Forward)
                .add(Duration::from_micros(1));
            delta.record_batch(1);
            merge_inference_phase_stats(&delta);

            let main_snapshot = get_phase_metrics();
            assert_eq!(main_snapshot.batches(), 1);

            let other_thread_snapshot = std::thread::spawn(get_phase_metrics).join().unwrap();
            // 別スレッドの thread-local は常にゼロから始まる
            // （このスレッドの蓄積が漏れ伝わらない）。
            assert_eq!(other_thread_snapshot.batches(), 0);

            reset_phase_metrics();
            assert_eq!(get_phase_metrics().batches(), 0);
        }

        #[test]
        fn predict_batch_input_ignores_labels() {
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
