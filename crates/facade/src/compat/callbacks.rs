//! callbacks（`EarlyStopping`／`ModelCheckpoint`／LR scheduler 連携。
//! イシュー #1763・親 #1618。`docs/compat-callbacks-design.md`）。
//!
//! [`Sequential::fit_with_callbacks`]（`training.rs`）へ `&mut [Callback]`
//! として渡す、epoch 境界のホスト側制御ロジックのみを提供する——
//! **新規 `Op`／`BackendOps`／`Var` メソッド／VJP／カーネルは一切
//! 追加しない**（`docs/compat-fit-evaluate-design.md` と同じ REQ-9
//! 「互換 API 層は自作コアの上の薄いラッパーに徹する」方針の継承。
//! テンソル演算ではなくホスト側の状態機械のみで構成される）。
//!
//! # 監視値（[`Monitor`]）と改善判定
//!
//! [`Monitor::Loss`] は `History::loss[epoch]`（学習損失）、
//! [`Monitor::ValLoss`]（既定）は `History::val_loss[epoch]`（検証損失。
//! `validation` 引数が `None` の `fit_with_callbacks` 呼び出しでは
//! 使えず `AutodiffError::InvalidArgument` になる）を指す。
//!
//! 改善判定は [`MonitorMode::Min`]／[`MonitorMode::Max`] と
//! （[`EarlyStopping`] のみ）`min_delta` に応じて行う——
//! `fandhe_ai_autodiff::nn::optim::reduce_lr_on_plateau::ReduceLrOnPlateau::
//! is_better` と同型の比較式（`value < best - min_delta`／
//! `value > best + min_delta`）を使う。`f32` の順序比較演算子は NaN に
//! 対して必ず `false` を返す言語仕様のため、**NaN はこの比較式を
//! 素通りするだけで「決して改善として扱われない」契約が自然に成立
//! する**（明示的な `is_nan()` 分岐を追加していない）。`best` は
//! 内部で `Option<f32>`（未観測は `None`）として保持し、比較のたびに
//! `best.unwrap_or(mode.initial_best())`（`Min` なら `+INF`・`Max` なら
//! `-INF`）を実効の比較対象にする——`mode()` ビルダーが `new()` の
//! 後・観測前に呼ばれても常に正しい初期値になる（先に固定値へ書き込む
//! 方式だと呼び出し順序に依存してしまう）。この結果、初回観測が NaN の
//! 場合でも `NaN < +INF` は `false` となり NaN が `best` になることは
//! ない（NaN が続く限り非改善カウンタが増加し続け、いずれ `patience`
//! で停止する）。
//!
//! # epoch 番号の数え方（callback ごとに異なる。誤認防止のため明記）
//!
//! - [`EarlyStopping`] は [`Sequential::fit_with_callbacks`] 呼び出し
//!   ごとに内部状態をリセットする（Keras `on_train_begin` と同じ。
//!   [`EarlyStopping::stopped_epoch`]／[`EarlyStopping::best_epoch`]
//!   節参照）ため、ここでの「epoch」は**その fit 呼び出し内の
//!   ローカル epoch 番号**（`History::loss`／`val_loss` の添字と同じ
//!   0 始まり）である。
//! - [`ModelCheckpoint`]・[`LrSchedule`] は複数回の `fit_with_callbacks`
//!   呼び出しをまたいで状態を継続する（前者は「これまでの最良」を、
//!   後者は内部の `next_epoch` カウンタを維持する。[`Sequential::
//!   fit_with_callbacks`] doc の「fit 呼び出しをまたぐ状態の扱い」
//!   節を参照）ため、ここでの「epoch」は**その callback インスタンス
//!   が観測した epoch 末呼び出しの通算回数**を指す（`fit(2 epochs)`
//!   を 2 回呼べば 4 回分進む）。[`ModelCheckpoint::best_epoch`] は
//!   この通算番号を返す。
//! - [`CsvLogger`]・[`JsonLogger`]・[`LambdaCallback`]（イシュー #2571）は
//!   fit をまたぐ状態を持たず、epoch 番号は常に fit ローカル（0 始まり）。
//!
//! 両者の定義は独立であり、`History` の添字（常に fit 呼び出し
//! ローカル）とは別物であることに注意する。
//!
//! # LR 同期のタイミング（[`LrSchedule`]）
//!
//! optimizer への学習率書き込みは **epoch 開始時のみ**行う（`kind` が
//! `PerEpoch` なら `scheduler.lr_at(next_epoch)`、`Plateau` なら
//! `sched.current_lr()`）。epoch 末の [`LrSchedule::advance`] は
//! （`Plateau` の場合）`ReduceLrOnPlateau::step` を呼び内部 `lr` を
//! 進めるだけで、optimizer への書き戻しはその時点では行わない——
//! 次回の epoch 開始同期（`current_lr()` で読み出す）が必ず直後に
//! 続くため、epoch 末での即時書き戻しは冗長（2 回の fit 呼び出しを
//! またぐ場合でも、`sched` 自身の内部状態が呼び出しをまたいで
//! 継続するため次の fit 呼び出しの epoch 開始同期で正しい値が反映
//! される）。
//!
//! # ロガーと Lambda（イシュー #2571）
//!
//! [`CsvLogger`]／[`JsonLogger`] は epoch 末に学習経過をファイルへ書き
//! （形式・append・原子性は各型の doc）、[`LambdaCallback`] は epoch 末に
//! ユーザー定義クロージャを呼ぶ。いずれもホスト側の処理のみで学習の演算列
//! には影響しない。ファイル I/O の下請け（整形・上限付き読み込み・原子的
//! 書き出し）は private な `logger_io` モジュールが担う（決定記録
//! `docs/compat-callbacks-loggers-decision.md`）。
//!
//! # 対象外・切り出し候補
//!
//! ユーザー定義 callback（trait object による拡張点）・metrics
//! （accuracy 等）・`DataLoader` を直接受ける `fit`
//! 入口・追加 `Loss` variant・デバイス常駐学習（`DeviceParamStore`）／
//! GPU `Tape`／AMP／gradient clipping との結線・`TerminateOnNaN` 相当
//! （現状は [`fandhe_ai_autodiff::nn::optim::ReduceLrOnPlateau::step`]
//! の非有限拒否で fail-closed に停止する）・OneCycle 等の**バッチ
//! 単位**スケジューリング（epoch 単位のみ対応）は対象外
//! （`docs/compat-callbacks-design.md` §8 参照）。
//!
//! # `ModelCheckpoint` のファイル保存（イシュー #2073）
//!
//! [`ModelCheckpoint::to_file`] でパスを指定すると、in-memory
//! スナップショット（[`ModelCheckpoint::observe`]）を更新した epoch
//! ごとに safetensors ファイルへも書き出す（safetensors save 自体は
//! #2019 で facade 公開済み〈[`crate::interop::safetensors`]〉。
//! [`crate::interop::safetensors::save_safetensors_f32`] への薄い
//! 結線のみを追加する。REQ-9「互換 API 層は自作コアの上の薄い
//! ラッパーに徹する」）。親ディレクトリが存在しなければ
//! `std::fs::create_dir_all` で作成し、書き出し自体は
//! `save_safetensors_f32` の一時ファイル + `rename`（POSIX atomic）に
//! 委譲するため、途中クラッシュでも正規パスには完全なファイルか
//! 元のファイルのいずれかのみが存在する。**永続化に成功した場合に
//! 限り** in-memory スナップショット（`best`／`best_epoch`／`state`）
//! を前進させる——保存に失敗した場合は改善前の値へロールバックし、
//! その epoch の更新はコミットしない（[`ModelCheckpoint::observe`]
//! 節参照）。ロールバックしない設計だと、`save_best_only(true)` 下で
//! 一時的な保存失敗（権限不足・ディスク障害等）が解消した後も、
//! 以後の監視値が失敗した best を上回らない限り再保存が試行され
//! なくなってしまう。[`Sequential::fit_with_callbacks`] 側は保存
//! 失敗を `AutodiffError::InvalidArgument` として fit 全体を打ち切る
//! （`training.rs` の `'epochs_block` 契約に従い、打ち切り後も
//! [`EarlyStopping::restore_best_weights`] の復元・モード復元・
//! `compiled` 書き戻しは通常どおり実行される）。次回以降の
//! `fit_with_callbacks` 呼び出し（`ModelCheckpoint` は fit 呼び出しを
//! またいで状態を継続する）で同一または改善した監視値が観測されれば、
//! ロールバック済みの `best` を基準に改善判定が再び成立し、再保存が
//! 試行される。
//!
//! `to_file` を指定しない場合の挙動は変更しない（既定 `None` で
//! ファイル I/O は一切発生しない）。`ModelCheckpoint::restore_best_weights`
//! 相当のビルダー・safetensors metadata（best 値・epoch）の埋め込みは
//! 本イシューのスコープ外のまま切り出し候補として残す
//! （`docs/compat-callbacks-design.md` §8）。復元フローは
//! [`EarlyStopping::restore_best_weights`] との併用、または
//! [`crate::interop::safetensors::load_safetensors_f32`] →
//! [`Sequential::load_state_dict`] を呼び出し側が組み合わせて行う。

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::interop::safetensors::SaveError;
use crate::optim::{LrScheduler, ReduceLrOnPlateau};
use crate::{AutodiffError, Tensor};

use super::logger_io;
use super::metrics::Metrics;
use super::sequential::Sequential;
use super::training::History;

/// callback が監視する指標（`History` のどの列を見るか）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Monitor {
    /// 学習損失（[`History::loss`]）。
    Loss,
    /// 検証損失（[`History::val_loss`]）。`validation` が `None` の
    /// `fit_with_callbacks` 呼び出しでは使えない
    /// （`AutodiffError::InvalidArgument`）。
    ValLoss,
    /// 検証 metrics（[`History::val_metrics`]。イシュー #2072）。
    /// `validation` が `None`、または内包する [`Metrics`] が
    /// `Sequential::fit_with_metrics` に渡した `metrics` スライスに
    /// 含まれない場合は `AutodiffError::InvalidArgument`
    /// （`training.rs` 事前検査節）。[`Metrics::ConfusionMatrix`]
    /// （非スカラー）を内包する場合も同様に拒否する——値が定義
    /// できない監視対象を黙ってスキップしない。`MonitorMode` の既定
    /// は `Min`（loss 系向け）のままのため、accuracy 等（大きいほど
    /// 良い指標）を監視する場合は呼び出し側が
    /// `.mode(MonitorMode::Max)` を明示する必要がある（自動推定は
    /// しない）。
    ValMetric(Metrics),
}

impl Monitor {
    /// `history` からこの `Monitor` が指す fit ローカル epoch
    /// `epoch_local`（0 始まり。`History::loss`／`val_loss`／
    /// `val_metrics` の添字と同じ）の値を取り出す。呼び出し時点では
    /// [`Sequential::fit_with_callbacks`]／`fit_with_metrics` の事前
    /// 検査（`ValLoss`／`ValMetric` かつ `validation.is_none()` を
    /// 早期 `Err` する。`ValMetric` は要求 `metrics` との整合も検査
    /// 済み）を通過済みのため必ず非空だが、境界外アクセスを避ける
    /// ため `Option` で返す（呼び出し元は `epoch_local` が push 済みの
    /// 添字である契約を守る）。
    fn value_at(self, history: &History, epoch_local: usize) -> Option<f32> {
        match self {
            Monitor::Loss => history.loss.get(epoch_local).copied(),
            Monitor::ValLoss => history.val_loss.get(epoch_local).copied(),
            Monitor::ValMetric(m) => history.val_metrics.get(epoch_local).and_then(|r| match m {
                Metrics::Accuracy => r.accuracy,
                Metrics::Precision => r.precision,
                Metrics::Recall => r.recall,
                Metrics::F1 => r.f1,
                Metrics::ConfusionMatrix => None,
            }),
        }
    }

    /// `validation` 引数が必須かどうか（[`Monitor::ValLoss`]／
    /// [`Monitor::ValMetric`]）。
    fn requires_validation(self) -> bool {
        matches!(self, Monitor::ValLoss | Monitor::ValMetric(_))
    }
}

/// 監視指標の改善方向。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonitorMode {
    /// 指標が小さいほど良い（既定。loss 系向け）。
    Min,
    /// 指標が大きいほど良い（例: accuracy 系。metrics 自体は本イシュー
    /// では対象外のため使い道は限定的だが、`EarlyStopping`／
    /// `ModelCheckpoint` の対称性のため用意する）。
    Max,
}

impl MonitorMode {
    /// 未観測時の実効 `best`（`Min` は `+INF`・`Max` は `-INF`。
    /// `fandhe_ai_autodiff::nn::optim::ReduceLrOnPlateau::new` と同型）。
    fn initial_best(self) -> f32 {
        match self {
            MonitorMode::Min => f32::INFINITY,
            MonitorMode::Max => f32::NEG_INFINITY,
        }
    }

    /// `value` が `best` より `min_delta` 以上改善しているか。
    /// `value` が NaN の場合は比較演算子の言語仕様により常に `false`
    /// （モジュール冒頭 doc「監視値と改善判定」節参照）。
    fn is_improvement(self, value: f32, best: f32, min_delta: f32) -> bool {
        match self {
            MonitorMode::Min => value < best - min_delta,
            MonitorMode::Max => value > best + min_delta,
        }
    }
}

/// 監視指標が `patience` epoch 連続で改善しなければ学習を打ち切る
/// callback（Keras `EarlyStopping` 相当）。
///
/// 状態（`best`／`best_epoch`／`wait`／`stopped_epoch`／
/// `best_state`）は [`Sequential::fit_with_callbacks`] 呼び出しの
/// たびにリセットする（モジュール冒頭 doc「epoch 番号の数え方」節）。
#[derive(Debug)]
pub struct EarlyStopping {
    monitor: Monitor,
    mode: MonitorMode,
    patience: usize,
    min_delta: f32,
    restore_best_weights: bool,
    best: Option<f32>,
    best_epoch: Option<usize>,
    wait: usize,
    stopped_epoch: Option<usize>,
    best_state: Option<HashMap<String, Tensor<f32>>>,
}

impl EarlyStopping {
    /// `patience`（改善なしを許容する連続 epoch 数）のみを指定して
    /// 構築する。既定値: `monitor` = [`Monitor::ValLoss`]・`mode` =
    /// [`MonitorMode::Min`]・`min_delta` = `0.0`・
    /// `restore_best_weights` = `false`（Keras `EarlyStopping` の既定と
    /// 同じ）。
    pub fn new(patience: usize) -> Self {
        EarlyStopping {
            monitor: Monitor::ValLoss,
            mode: MonitorMode::Min,
            patience,
            min_delta: 0.0,
            restore_best_weights: false,
            best: None,
            best_epoch: None,
            wait: 0,
            stopped_epoch: None,
            best_state: None,
        }
    }

    /// 監視する指標を設定して返す（ビルダー）。
    pub fn monitor(mut self, m: Monitor) -> Self {
        self.monitor = m;
        self
    }

    /// 改善方向を設定して返す（ビルダー）。
    pub fn mode(mut self, m: MonitorMode) -> Self {
        self.mode = m;
        self
    }

    /// 改善判定のしきい値を設定して返す（ビルダー）。
    ///
    /// # Errors
    ///
    /// `d` が非有限または負値の場合は
    /// `AutodiffError::InvalidArgument`（fail-closed）。
    pub fn min_delta(mut self, d: f32) -> Result<Self, AutodiffError> {
        if !d.is_finite() || d < 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "EarlyStopping::min_delta: must be finite and >= 0.0, got {d}"
            )));
        }
        self.min_delta = d;
        Ok(self)
    }

    /// 早期停止時（および完走時、`best` が存在すれば）に best
    /// スナップショットへ復元するかどうかを設定して返す（ビルダー。
    /// Keras 3 の `EarlyStopping(restore_best_weights=True)` と同じ
    /// 意味論: 早期停止・完走いずれの終了経路でも、`best` が観測
    /// 済みであれば最後にそのスナップショットへ復元する）。
    pub fn restore_best_weights(mut self, on: bool) -> Self {
        self.restore_best_weights = on;
        self
    }

    /// 学習を打ち切った fit ローカル epoch 番号（0 始まり）。停止せず
    /// 完走した場合は `None`。
    pub fn stopped_epoch(&self) -> Option<usize> {
        self.stopped_epoch
    }

    /// 現時点の best 指標値（未観測または今回の fit で改善が一度も
    /// 記録されていない場合は `None`）。
    pub fn best_value(&self) -> Option<f32> {
        self.best
    }

    /// best を観測した fit ローカル epoch 番号（0 始まり）。
    pub fn best_epoch(&self) -> Option<usize> {
        self.best_epoch
    }

    fn requires_validation(&self) -> bool {
        self.monitor.requires_validation()
    }

    /// `self.monitor` が指す fit ローカル epoch `epoch_local` の値を
    /// `history` から取り出す（[`Monitor::value_at`] への委譲。
    /// `training.rs` から `self.monitor` を直接読ませない薄いラッパー）。
    pub(super) fn monitor_value_at(&self, history: &History, epoch_local: usize) -> Option<f32> {
        self.monitor.value_at(history, epoch_local)
    }

    /// [`Sequential::fit_with_callbacks`] 呼び出し開始時に内部状態を
    /// リセットする（モジュール冒頭 doc「epoch 番号の数え方」節）。
    pub(super) fn reset_for_fit(&mut self) {
        self.best = None;
        self.best_epoch = None;
        self.wait = 0;
        self.stopped_epoch = None;
        self.best_state = None;
    }

    /// epoch 末に 1 回呼ぶ。`value` は `self.monitor` が指す指標値、
    /// `epoch_local` はこの fit 呼び出し内の 0 始まり epoch 番号。
    /// `snapshot` は `restore_best_weights` かつ改善時にのみ評価する
    /// 遅延クロージャ（[`Sequential::state_dict`] の clone コストを
    /// 改善 epoch のみへ限定する）。
    ///
    /// 戻り値は「この epoch の終わりに学習を打ち切るか」（`true` の
    /// 場合、呼び出し元は当該 epoch の他の callback も処理し終えて
    /// から fit ループを抜ける契約——本メソッド自体は参照せず
    /// [`Sequential::fit_with_callbacks`] doc「callback 処理順序」節を
    /// 正とする）。
    pub(super) fn observe(
        &mut self,
        value: f32,
        epoch_local: usize,
        snapshot: impl FnOnce() -> HashMap<String, Tensor<f32>>,
    ) -> bool {
        let best_so_far = self.best.unwrap_or(self.mode.initial_best());
        if self.mode.is_improvement(value, best_so_far, self.min_delta) {
            self.best = Some(value);
            self.best_epoch = Some(epoch_local);
            self.wait = 0;
            if self.restore_best_weights {
                self.best_state = Some(snapshot());
            }
            false
        } else {
            self.wait += 1;
            if self.wait >= self.patience {
                self.stopped_epoch = Some(epoch_local);
                true
            } else {
                false
            }
        }
    }

    /// fit 終了時に呼ぶ: `restore_best_weights` かつ best スナップ
    /// ショットが存在すれば取り出す（`self.best_state` は消費される。
    /// 以後 `best_value`／`best_epoch` は引き続き読めるが、
    /// スナップショット自体は 2 度適用しない）。
    pub(super) fn take_restore_state(&mut self) -> Option<HashMap<String, Tensor<f32>>> {
        if self.restore_best_weights {
            self.best_state.take()
        } else {
            None
        }
    }
}

/// 監視指標が改善したとき（既定）またはすべての epoch 末に、モデルの
/// `state_dict()` スナップショットを保持する callback（Keras
/// `ModelCheckpoint` 相当。in-memory スナップショット（`state`）に加え、
/// [`Self::to_file`] でパスを指定すればファイル保存も行う
/// （モジュール冒頭 doc「`ModelCheckpoint` のファイル保存」節）。
///
/// [`EarlyStopping`] と異なり、状態（`best`／`best_epoch`／`state`）は
/// 複数回の [`Sequential::fit_with_callbacks`] 呼び出しをまたいで
/// 継続する（モジュール冒頭 doc「epoch 番号の数え方」節）。
#[derive(Debug)]
pub struct ModelCheckpoint {
    monitor: Monitor,
    mode: MonitorMode,
    save_best_only: bool,
    best: Option<f32>,
    best_epoch: Option<usize>,
    epoch_count: usize,
    state: Option<HashMap<String, Tensor<f32>>>,
    file_path: Option<PathBuf>,
}

impl ModelCheckpoint {
    /// 既定値: `monitor` = [`Monitor::ValLoss`]・`mode` =
    /// [`MonitorMode::Min`]・`save_best_only` = `true`・ファイル保存
    /// なし（`to_file` 未指定）。
    pub fn new() -> Self {
        ModelCheckpoint {
            monitor: Monitor::ValLoss,
            mode: MonitorMode::Min,
            save_best_only: true,
            best: None,
            best_epoch: None,
            epoch_count: 0,
            state: None,
            file_path: None,
        }
    }

    /// 監視する指標を設定して返す（ビルダー）。
    pub fn monitor(mut self, m: Monitor) -> Self {
        self.monitor = m;
        self
    }

    /// 改善方向を設定して返す（ビルダー）。
    pub fn mode(mut self, m: MonitorMode) -> Self {
        self.mode = m;
        self
    }

    /// `true`（既定）なら改善した epoch のみスナップショットを更新する。
    /// `false` ならすべての epoch 末でスナップショットを上書きする
    /// （結果として最終 epoch のものが残る）。
    pub fn save_best_only(mut self, on: bool) -> Self {
        self.save_best_only = on;
        self
    }

    /// スナップショット更新時（内部の `observe` メソッドが in-memory
    /// `state` を書き換えた時）に safetensors ファイルへも書き出す
    /// （ビルダー・FS には一切触れない infallible 操作。実際の I/O は
    /// `observe` 呼び出し時のみ発生する。モジュール冒頭 doc
    /// 「`ModelCheckpoint` のファイル保存」節）。
    ///
    /// 未指定（既定）の場合は従来どおり in-memory のみで動作する。
    pub fn to_file(mut self, path: impl AsRef<Path>) -> Self {
        self.file_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// 現時点で保持しているスナップショットへの参照。
    pub fn best_state_dict(&self) -> Option<&HashMap<String, Tensor<f32>>> {
        self.state.as_ref()
    }

    /// 現時点で保持しているスナップショットを取り出す
    /// （[`Sequential::load_state_dict`] へそのまま渡せる。取り出した
    /// 後 [`Self::best_state_dict`] は `None` を返す）。
    pub fn take_best_state_dict(&mut self) -> Option<HashMap<String, Tensor<f32>>> {
        self.state.take()
    }

    /// 現時点の best 指標値。
    pub fn best_value(&self) -> Option<f32> {
        self.best
    }

    /// best を観測した通算 epoch 番号（モジュール冒頭 doc「epoch
    /// 番号の数え方」節: 複数回の fit をまたいだ通算回数）。
    pub fn best_epoch(&self) -> Option<usize> {
        self.best_epoch
    }

    fn requires_validation(&self) -> bool {
        self.monitor.requires_validation()
    }

    /// [`EarlyStopping::monitor_value_at`] と同型（`Monitor::value_at`
    /// への委譲）。
    pub(super) fn monitor_value_at(&self, history: &History, epoch_local: usize) -> Option<f32> {
        self.monitor.value_at(history, epoch_local)
    }

    /// epoch 末に 1 回呼ぶ。`value` は `self.monitor` が指す指標値、
    /// `model` はスナップショット取得元。
    ///
    /// `Self::to_file` でパスを指定していれば、in-memory `state` を
    /// 更新する場合に限り safetensors ファイルへも書き出す
    /// （`state` を更新しない呼び出しでは I/O を発生させない）。
    /// **`to_file` 指定時は永続化に成功した場合に限り `best`／
    /// `best_epoch`／`state` を前進させる**——保存が失敗した場合は
    /// これらの更新をコミットせず、改善前の値のまま据え置く
    /// （モジュール冒頭 doc「`ModelCheckpoint` のファイル保存」節）。
    /// これにより、一時的な保存失敗が解消した後の次回 `observe`
    /// 呼び出しでも、ロールバックされた `best` を基準に改善判定が
    /// 成立し、再保存が試行される（保存失敗した値をそのまま `best`
    /// として確定してしまうと、以後それを上回る値しか再保存の
    /// トリガーにならず、恒久的に再試行されなくなる）。`to_file`
    /// 未指定（`file_path` が `None`）の場合は永続化自体が発生しない
    /// ため常に成功扱いとなり、従来どおり改善時に即座にコミットする。
    /// 呼び出し元（`training.rs::run_fit`）は `Err` を
    /// `AutodiffError` へ写像して fit 全体を打ち切る。
    pub(super) fn observe(&mut self, value: f32, model: &Sequential) -> Result<(), SaveError> {
        let best_so_far = self.best.unwrap_or(self.mode.initial_best());
        let improved = self.mode.is_improvement(value, best_so_far, 0.0);
        let mut persist_result = Ok(());
        if improved || !self.save_best_only {
            let state = model.state_dict();
            if let Some(path) = &self.file_path {
                persist_result = persist(path, &state);
            }
            // 永続化に成功した場合（`file_path` 未設定で persist_result が
            // 常に `Ok` のままの場合を含む）のみ in-memory 側をコミットする。
            // 失敗時は `best`／`best_epoch`／`state` を書き換えないまま
            // 据え置き、次回 observe での再試行を可能にする（doc 節参照）。
            if persist_result.is_ok() {
                if improved {
                    self.best = Some(value);
                    self.best_epoch = Some(self.epoch_count);
                }
                self.state = Some(state);
            }
        }
        self.epoch_count += 1;
        persist_result
    }
}

/// [`ModelCheckpoint::observe`] のファイル保存本体（private）。
///
/// 親ディレクトリが存在しなければ作成してから
/// [`crate::interop::safetensors::save_safetensors_f32`] へ委譲する
/// （一時ファイル + `rename` による atomic 上書きは委譲先の契約
/// そのまま。`path.parent()` が空文字列（裸のファイル名。カレント
/// ディレクトリ相対）の場合は `create_dir_all` をスキップする）。
fn persist(path: &Path, state: &HashMap<String, Tensor<f32>>) -> Result<(), SaveError> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(SaveError::Io)?;
    }
    crate::interop::safetensors::save_safetensors_f32(path, state)
}

impl Default for ModelCheckpoint {
    fn default() -> Self {
        Self::new()
    }
}

/// LR scheduler を optimizer へ結線する callback（epoch 単位のみ。
/// モジュール冒頭 doc「対象外・切り出し候補」節）。
///
/// 2 通りの構築方法がある: [`LrSchedule::per_epoch`]（`StepLr` 等の
/// stateless [`LrScheduler`] 実装をそのまま epoch 番号で駆動する）・
/// [`LrSchedule::plateau`]／[`LrSchedule::plateau_with_monitor`]
/// （[`ReduceLrOnPlateau`] を監視指標で駆動する）。
pub struct LrSchedule {
    kind: LrScheduleKind,
    next_epoch: usize,
}

enum LrScheduleKind {
    PerEpoch(Box<dyn LrScheduler>),
    Plateau {
        sched: ReduceLrOnPlateau,
        monitor: Monitor,
    },
}

impl LrSchedule {
    /// `s`（`StepLr`／`CosineAnnealingLr` 等の stateless
    /// [`LrScheduler`]）を epoch 番号（`self.epoch()`。0 始まり・
    /// [`Sequential::fit_with_callbacks`] 呼び出しをまたいで継続）で
    /// 駆動する。
    pub fn per_epoch(s: impl LrScheduler + 'static) -> Self {
        LrSchedule {
            kind: LrScheduleKind::PerEpoch(Box::new(s)),
            next_epoch: 0,
        }
    }

    /// `s`（[`ReduceLrOnPlateau`]）を [`Monitor::ValLoss`]（既定）で
    /// 駆動する。
    pub fn plateau(s: ReduceLrOnPlateau) -> Self {
        Self::plateau_with_monitor(s, Monitor::ValLoss)
    }

    /// `s` を明示的な `monitor` で駆動する（[`Self::plateau`] は
    /// `monitor = Monitor::ValLoss` の糖衣）。[`Self::per_epoch`] は
    /// 監視値を使わないため対応する `_with_monitor` 版は設けない。
    pub fn plateau_with_monitor(s: ReduceLrOnPlateau, monitor: Monitor) -> Self {
        LrSchedule {
            kind: LrScheduleKind::Plateau { sched: s, monitor },
            next_epoch: 0,
        }
    }

    /// 次に適用する epoch 番号（[`Sequential::fit_with_callbacks`] を
    /// またいで継続する通算値。モジュール冒頭 doc「epoch 番号の
    /// 数え方」節）。
    pub fn epoch(&self) -> usize {
        self.next_epoch
    }

    fn monitor(&self) -> Option<Monitor> {
        match &self.kind {
            LrScheduleKind::PerEpoch(_) => None,
            LrScheduleKind::Plateau { monitor, .. } => Some(*monitor),
        }
    }

    fn requires_validation(&self) -> bool {
        matches!(self.monitor(), Some(m) if m.requires_validation())
    }

    /// `self.monitor()` が `Some` の場合（`Plateau`）に限り、対応する
    /// 指標値を `history` から取り出す（`PerEpoch` は常に `None`。
    /// [`EarlyStopping::monitor_value_at`] と同型）。
    pub(super) fn monitor_value_at(&self, history: &History, epoch_local: usize) -> Option<f32> {
        self.monitor()
            .and_then(|m| m.value_at(history, epoch_local))
    }

    /// epoch 開始時に optimizer へ書き込むべき学習率（モジュール
    /// 冒頭 doc「LR 同期のタイミング」節）。
    pub(super) fn lr_for_epoch_begin(&self) -> f32 {
        match &self.kind {
            LrScheduleKind::PerEpoch(s) => s.lr_at(self.next_epoch),
            LrScheduleKind::Plateau { sched, .. } => sched.current_lr(),
        }
    }

    /// epoch 末に 1 回呼ぶ。`monitor_value` は `self.monitor()` が
    /// `Some` の場合（`Plateau`）にのみ使う観測値（`PerEpoch` の場合
    /// `None` を渡してよい）。`Plateau` は内部で `sched.step(value)`
    /// を呼び学習率を進める（optimizer への即時書き戻しは行わない。
    /// モジュール冒頭 doc「LR 同期のタイミング」節）。
    ///
    /// # Errors
    ///
    /// `Plateau` かつ `monitor_value` が `None`（呼び出し元の契約
    /// 違反。通常到達しない防御的検査）、または `sched.step` が
    /// 非有限値を検出した場合（`ReduceLrOnPlateau::step` doc 参照）に
    /// `AutodiffError::InvalidArgument`。
    pub(super) fn advance(&mut self, monitor_value: Option<f32>) -> Result<(), AutodiffError> {
        match &mut self.kind {
            LrScheduleKind::PerEpoch(_) => {}
            LrScheduleKind::Plateau { sched, .. } => {
                let value = monitor_value.ok_or_else(|| {
                    AutodiffError::InvalidArgument(
                        "LrSchedule::advance: Plateau には monitor_value が必要\
                         （呼び出し元の契約違反）"
                            .to_string(),
                    )
                })?;
                sched.step(value)?;
            }
        }
        self.next_epoch += 1;
        Ok(())
    }
}

impl std::fmt::Debug for LrSchedule {
    // `Box<dyn LrScheduler>`／`ReduceLrOnPlateau` はいずれも `Debug` を
    // 実装していない（`ReduceLrOnPlateau` は `nn::optim::
    // reduce_lr_on_plateau` モジュール doc 参照。`dyn LrScheduler` は
    // `lr_at` のみの trait でユーザー実装型を含みうるため `Debug`
    // 境界を要求しない設計）ため、variant 名と観測可能な要約値のみを
    // 出す非網羅的な `Debug` を手書きする（`derive` 不可）。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            LrScheduleKind::PerEpoch(_) => f
                .debug_struct("LrSchedule::PerEpoch")
                .field("next_epoch", &self.next_epoch)
                .finish(),
            LrScheduleKind::Plateau { sched, monitor } => f
                .debug_struct("LrSchedule::Plateau")
                .field("current_lr", &sched.current_lr())
                .field("monitor", monitor)
                .field("next_epoch", &self.next_epoch)
                .finish(),
        }
    }
}

/// 各 epoch の学習経過を CSV 1 行として書くロガー（Keras `CSVLogger`
/// 相当。イシュー #2571・`docs/compat-callbacks-loggers-decision.md`）。
///
/// [`Callback::CsvLogger`] として [`Sequential::fit_with_callbacks`]／
/// [`Sequential::fit_with_metrics`] へ渡す。学習の演算列・[`History`] には
/// 一切影響しない（ホスト側のファイル書き込みのみ）。
///
/// # ファイル形式
///
/// - 区切りは `,`、改行は `\n`。ヘッダ行 `epoch,loss,lr` に、`validation`
///   があれば `val_loss`、`fit_with_metrics` で要求した scalar metrics が
///   あれば `val_accuracy`／`val_precision`／`val_recall`／`val_f1` を
///   この固定順で追加する（`metrics` 引数の並び・重複には依存しない。
///   [`Metrics::ConfusionMatrix`] は非スカラーのため列にしない）。列集合は
///   fit 呼び出しの開始時に確定する。
/// - `epoch` は fit ローカルの 0 始まり（[`History`] の添字と同じ）。`lr` は
///   その epoch の開始時に記録された学習率。
/// - 数値は `f32` の `Display`（最短往復表現）で書くため `parse::<f32>()`
///   で bit 一致に読み戻せる。非有限値は `NaN`／`inf`／`-inf`（Keras の
///   `nan` 表記とは異なる）。
///
/// # truncate／append と原子性
///
/// 既定（`append(false)`）では fit 開始時にファイルを truncate してヘッダ
/// を書く。`append(true)` では追記し、ファイルが存在しないか空のときだけ
/// ヘッダを書く。既存ファイルが空でない場合は先頭行を上限付きで読み、
/// 今回の列集合のヘッダと完全一致することを検証する（不一致・読み込み
/// 失敗は fit を開始せず `Err`。`compiled` は書き戻される）。既存ファイルの
/// 末尾が改行で終わっていない場合（前回 fit の中断）は、改行を 1 つ補って
/// から追記する。追記方式の
/// ため一時ファイル＋`rename` の原子性は使えず、epoch ごとに 1 行を書いて
/// 次へ進む（途中でクラッシュしても完了した epoch の行は残る）。親
/// ディレクトリが無ければ作成する。I/O 失敗は
/// [`AutodiffError::InvalidArgument`] に写像して fit を打ち切る。
///
/// # Examples
///
/// ```
/// use fandhe_ai::compat::{Callback, CsvLogger, FitConfig, Loss, Optimizer, Sequential};
/// use fandhe_ai::optim::SgdConfig;
/// use fandhe_ai::Tensor;
///
/// let dir = std::env::temp_dir().join(format!(
///     "fandhe-csv-doc-{}-{}",
///     std::process::id(),
///     std::time::SystemTime::now()
///         .duration_since(std::time::UNIX_EPOCH)
///         .map(|d| d.as_nanos())
///         .unwrap_or(0)
/// ));
/// let path = dir.join("log.csv");
///
/// let mut model = Sequential::new().add_linear(2, 1, 7).unwrap();
/// model
///     .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
///     .unwrap();
/// let x = Tensor::new(vec![0.0f32, 1.0, 1.0, 0.0], &[2, 2]).unwrap();
/// let y = Tensor::new(vec![1.0f32, 1.0], &[2, 1]).unwrap();
/// let mut cbs = [Callback::CsvLogger(CsvLogger::new(&path))];
/// model
///     .fit_with_callbacks(&x, &y, FitConfig::new(2, 2), None, &mut cbs)
///     .unwrap();
///
/// let text = std::fs::read_to_string(&path).unwrap();
/// let lines: Vec<&str> = text.lines().collect();
/// assert_eq!(lines[0], "epoch,loss,lr");
/// assert_eq!(lines.len(), 3);
/// let _ = std::fs::remove_dir_all(&dir);
/// ```
pub struct CsvLogger {
    path: PathBuf,
    append: bool,
    /// fit 実行中のみ `Some`（`begin_fit`〜`end_fit`）。
    state: Option<CsvFitState>,
}

/// [`LambdaCallback`] が保持するクロージャ型（`Send` 境界なし。型 doc 参照）。
type EpochEndFn = Box<dyn FnMut(usize, &History) -> Result<(), AutodiffError>>;

/// [`CsvLogger`] の fit 内状態（列集合と開いたファイル）。
struct CsvFitState {
    cols: Vec<&'static str>,
    file: std::fs::File,
}

impl CsvLogger {
    /// `path` へ書き出すロガーを作る（既定は `append(false)`。I/O は
    /// fit 開始時まで発生しない）。
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            append: false,
            state: None,
        }
    }

    /// `true` なら既存ファイルへ追記する（既定 `false`。Keras
    /// `append=False` と同じ）。
    pub fn append(mut self, on: bool) -> Self {
        self.append = on;
        self
    }

    fn err(&self, method: &str, detail: impl std::fmt::Display) -> AutodiffError {
        AutodiffError::InvalidArgument(format!(
            "Sequential::{method}: CsvLogger（{}）: {detail}",
            self.path.display()
        ))
    }

    /// fit 開始時の準備（`training.rs` の `reset_for_fit` と同じ位置から
    /// 呼ばれる）。列集合の確定・ファイルの truncate／ヘッダ検証。
    pub(super) fn begin_fit(
        &mut self,
        method: &str,
        has_validation: bool,
        metrics: &[Metrics],
    ) -> Result<(), AutodiffError> {
        self.state = None;
        let cols = logger_io::columns(has_validation, metrics);
        let header = logger_io::csv_header(&cols);
        let file = self
            .open_for_fit(&header)
            .map_err(|d| self.err(method, d))?;
        self.state = Some(CsvFitState { cols, file });
        Ok(())
    }

    fn open_for_fit(&self, header: &str) -> Result<std::fs::File, String> {
        use std::io::{Read, Seek, SeekFrom, Write};
        logger_io::ensure_parent_dir(&self.path)?;
        let existing_nonempty = self.append && logger_io::existing_nonempty(&self.path)?;
        if existing_nonempty {
            let first = logger_io::read_first_line_limited(&self.path)?.unwrap_or_default();
            if first != header {
                return Err(format!(
                    "既存ファイルのヘッダ（{first:?}）が今回の列集合（{header:?}）と一致しない"
                ));
            }
            // 前回 fit が行の途中で中断し末尾が改行で終わっていないと、追記行が
            // 壊れた行へ連結されるため、改行を補ってから追記する。
            let mut f = std::fs::OpenOptions::new()
                .read(true)
                .append(true)
                .open(&self.path)
                .map_err(|e| e.to_string())?;
            f.seek(SeekFrom::End(-1)).map_err(|e| e.to_string())?;
            let mut last = [0u8; 1];
            f.read_exact(&mut last).map_err(|e| e.to_string())?;
            if last[0] != b'\n' {
                f.write_all(b"\n")
                    .and_then(|_| f.flush())
                    .map_err(|e| e.to_string())?;
            }
            return Ok(f);
        }
        let mut f = if self.append {
            std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(&self.path)
        } else {
            std::fs::File::create(&self.path)
        }
        .map_err(|e| e.to_string())?;
        f.write_all(format!("{header}\n").as_bytes())
            .and_then(|_| f.flush())
            .map_err(|e| e.to_string())?;
        Ok(f)
    }

    /// epoch 末に 1 行を書く（`training.rs` の epoch 末 callbacks ループ
    /// から呼ばれる）。
    pub(super) fn write_epoch(
        &mut self,
        method: &str,
        history: &History,
        epoch_local: usize,
    ) -> Result<(), AutodiffError> {
        use std::io::Write;
        let Some(st) = self.state.as_mut() else {
            return Err(self.err(method, "fit 開始前に epoch 末が呼ばれた（内部契約違反）"));
        };
        let line = logger_io::collect_values(&st.cols, history, epoch_local)
            .map(|v| format!("{}\n", logger_io::csv_row(epoch_local, &v)));
        let res = line.and_then(|l| {
            st.file
                .write_all(l.as_bytes())
                .and_then(|_| st.file.flush())
                .map_err(|e| e.to_string())
        });
        res.map_err(|d| self.err(method, d))
    }

    /// fit 終了時のハンドル解放（成功・失敗いずれの経路でも呼ばれる）。
    pub(super) fn end_fit(&mut self) {
        self.state = None;
    }
}

impl std::fmt::Debug for CsvLogger {
    // 開いたファイルハンドルは出さず、設定値だけを出す（出力は安定 API
    // ではない）。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsvLogger")
            .field("path", &self.path)
            .field("append", &self.append)
            .finish()
    }
}

/// 各 epoch の学習経過を JSON 配列として書くロガー（Keras の JSON 出力
/// 系 callback 相当。イシュー #2571・`docs/compat-callbacks-loggers-
/// decision.md`）。
///
/// [`Callback::JsonLogger`] として fit へ渡す。キー・列集合・`epoch`／`lr`
/// の意味は [`CsvLogger`] と同じ（`{"epoch":0,"loss":1.5,"lr":0.1}` の
/// ような epoch オブジェクトの配列）。
///
/// # 非有限値と number 文法
///
/// 有限値は `f32` の `Display`（JSON number に合致）。非有限値は RFC 8259
/// 準拠のクォート付き文字列 `"NaN"`／`"Infinity"`／`"-Infinity"` で書く
/// ため、標準の JSON パーサで読み込める。したがってフィールドの JSON 型は
/// 値により number／string に変わる。
///
/// # truncate／append と原子性
///
/// 配列全体を in-memory に持ち、epoch ごとに一時ファイル＋`rename` で書き
/// 直す（途中クラッシュ時も正規パスには完全な JSON だけが残る）。fit
/// 開始時にも一度書き出し、書き込み可否を早期に検出する。`append(true)` で
/// 既存の空でないファイルがあれば、上限付きで読み込み「オブジェクトの
/// 配列」であること・各要素が今回の列キーを含むことを検証して初期値と
/// する（不正は fit を開始せず `Err`。既存要素は原文のまま保持し再整形
/// しない）。`append(false)`、またはファイルが無い・空なら空配列から
/// 始める。I/O 失敗は [`AutodiffError::InvalidArgument`] に写像する。
///
/// # Examples
///
/// ```
/// use fandhe_ai::compat::{Callback, FitConfig, JsonLogger, Loss, Optimizer, Sequential};
/// use fandhe_ai::optim::SgdConfig;
/// use fandhe_ai::Tensor;
///
/// let dir = std::env::temp_dir().join(format!(
///     "fandhe-json-doc-{}-{}",
///     std::process::id(),
///     std::time::SystemTime::now()
///         .duration_since(std::time::UNIX_EPOCH)
///         .map(|d| d.as_nanos())
///         .unwrap_or(0)
/// ));
/// let path = dir.join("log.json");
///
/// let mut model = Sequential::new().add_linear(2, 1, 7).unwrap();
/// model
///     .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
///     .unwrap();
/// let x = Tensor::new(vec![0.0f32, 1.0, 1.0, 0.0], &[2, 2]).unwrap();
/// let y = Tensor::new(vec![1.0f32, 1.0], &[2, 1]).unwrap();
/// let mut cbs = [Callback::JsonLogger(JsonLogger::new(&path))];
/// model
///     .fit_with_callbacks(&x, &y, FitConfig::new(2, 2), None, &mut cbs)
///     .unwrap();
///
/// let text = std::fs::read_to_string(&path).unwrap();
/// assert!(text.starts_with("[\n"));
/// assert_eq!(text.matches("\"epoch\"").count(), 2);
/// let _ = std::fs::remove_dir_all(&dir);
/// ```
pub struct JsonLogger {
    path: PathBuf,
    append: bool,
    /// fit 実行中のみ `Some`（`begin_fit`〜`end_fit`）。
    state: Option<JsonFitState>,
}

/// [`JsonLogger`] の fit 内状態（列集合と配列要素の原文）。
struct JsonFitState {
    cols: Vec<&'static str>,
    elems: Vec<String>,
}

impl JsonLogger {
    /// `path` へ書き出すロガーを作る（既定は `append(false)`。I/O は
    /// fit 開始時まで発生しない）。
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            append: false,
            state: None,
        }
    }

    /// `true` なら既存の JSON 配列へ追記する（既定 `false`）。
    pub fn append(mut self, on: bool) -> Self {
        self.append = on;
        self
    }

    fn err(&self, method: &str, detail: impl std::fmt::Display) -> AutodiffError {
        AutodiffError::InvalidArgument(format!(
            "Sequential::{method}: JsonLogger（{}）: {detail}",
            self.path.display()
        ))
    }

    /// fit 開始時の準備（[`CsvLogger`] と同じ位置・契約）。
    pub(super) fn begin_fit(
        &mut self,
        method: &str,
        has_validation: bool,
        metrics: &[Metrics],
    ) -> Result<(), AutodiffError> {
        self.state = None;
        let cols = logger_io::columns(has_validation, metrics);
        let elems = self
            .load_initial(&cols)
            .and_then(|elems| {
                logger_io::write_atomic(&self.path, logger_io::json_array(&elems).as_bytes())
                    .map(|_| elems)
            })
            .map_err(|d| self.err(method, d))?;
        self.state = Some(JsonFitState { cols, elems });
        Ok(())
    }

    fn load_initial(&self, cols: &[&'static str]) -> Result<Vec<String>, String> {
        logger_io::ensure_parent_dir(&self.path)?;
        let nonempty = self.append && logger_io::existing_nonempty(&self.path)?;
        if !nonempty {
            return Ok(Vec::new());
        }
        let text = logger_io::read_limited(&self.path)?;
        // 空扱いは長さ 0 のファイルだけ（上の `existing_nonempty` で除外済み）。
        // 空白だけの非空ファイルは JSON ログとして不正であり、検証を迂回して
        // 上書きしないよう `parse_log_array` のエラーへ倒す（fail-closed）。
        let parsed = logger_io::parse_log_array(&text)
            .map_err(|e| format!("既存ファイルが JSON ログとして不正: {e}"))?;
        for (i, el) in parsed.iter().enumerate() {
            for key in std::iter::once("epoch").chain(cols.iter().copied()) {
                if !el.keys.iter().any(|k| k == key) {
                    return Err(format!("既存の要素 {i} にキー {key:?} が無い"));
                }
            }
        }
        Ok(parsed.into_iter().map(|e| e.raw).collect())
    }

    /// epoch 末に 1 要素を追加して全体を原子的に書き直す。
    pub(super) fn write_epoch(
        &mut self,
        method: &str,
        history: &History,
        epoch_local: usize,
    ) -> Result<(), AutodiffError> {
        let Some(st) = self.state.as_mut() else {
            return Err(self.err(method, "fit 開始前に epoch 末が呼ばれた（内部契約違反）"));
        };
        let res = logger_io::collect_values(&st.cols, history, epoch_local).and_then(|v| {
            st.elems
                .push(logger_io::json_row(&st.cols, epoch_local, &v));
            logger_io::write_atomic(&self.path, logger_io::json_array(&st.elems).as_bytes())
        });
        res.map_err(|d| self.err(method, d))
    }

    /// fit 終了時の in-memory 配列の解放。
    pub(super) fn end_fit(&mut self) {
        self.state = None;
    }
}

impl std::fmt::Debug for JsonLogger {
    // in-memory 配列は肥大しうるため出さず、設定値と行数だけを出す。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonLogger")
            .field("path", &self.path)
            .field("append", &self.append)
            .field("rows", &self.state.as_ref().map(|s| s.elems.len()))
            .finish()
    }
}

/// epoch 末にユーザー定義クロージャを呼ぶ callback（Keras
/// `LambdaCallback(on_epoch_end=…)` 相当。イシュー #2571）。
///
/// クロージャへ渡すのは fit ローカル epoch 番号（0 始まり）と、その epoch
/// までの読み取り専用 [`History`] だけである（`&mut Sequential` は渡さない。
/// `compiled` の一時取り外し・`bind` の借用と衝突するため。
/// `docs/compat-callbacks-design.md` §4.1）。`Err` を返すと fit はその
/// エラーで打ち切られ、`restore_best_weights`・モード復元・`compiled`
/// 書き戻しは通常どおり実行される。学習の停止要求は引き続き
/// [`EarlyStopping`] の責務で、本 callback は持たない。
///
/// クロージャは `Send` を要求しない（[`LrSchedule`] が `Send` 境界の無い
/// `Box<dyn LrScheduler>` を持つため [`Callback`] は元々 `!Send` で、
/// auto trait を後退させない。後から `Send` を足す変更は破壊的になる）。
///
/// # Examples
///
/// ```
/// use fandhe_ai::compat::{Callback, FitConfig, LambdaCallback, Loss, Optimizer, Sequential};
/// use fandhe_ai::optim::SgdConfig;
/// use fandhe_ai::Tensor;
/// use std::{cell::RefCell, rc::Rc};
///
/// let seen = Rc::new(RefCell::new(Vec::new()));
/// let sink = Rc::clone(&seen);
/// let mut cbs = [Callback::Lambda(LambdaCallback::on_epoch_end(move |epoch, h| {
///     sink.borrow_mut().push((epoch, h.loss.len()));
///     Ok(())
/// }))];
///
/// let mut model = Sequential::new().add_linear(2, 1, 7).unwrap();
/// model
///     .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::Mse)
///     .unwrap();
/// let x = Tensor::new(vec![0.0f32, 1.0, 1.0, 0.0], &[2, 2]).unwrap();
/// let y = Tensor::new(vec![1.0f32, 1.0], &[2, 1]).unwrap();
/// model
///     .fit_with_callbacks(&x, &y, FitConfig::new(3, 2), None, &mut cbs)
///     .unwrap();
/// assert_eq!(*seen.borrow(), vec![(0, 1), (1, 2), (2, 3)]);
/// ```
pub struct LambdaCallback {
    // `Send` 境界は意図的に付けない（型 doc 参照）。
    f: EpochEndFn,
    /// 手書き `Debug` 専用の呼び出し回数（公開アクセサは設けない）。
    calls: usize,
}

impl LambdaCallback {
    /// epoch 末に呼ぶクロージャ `f(epoch_local, &history)` を指定する。
    pub fn on_epoch_end(
        f: impl FnMut(usize, &History) -> Result<(), AutodiffError> + 'static,
    ) -> Self {
        Self {
            f: Box::new(f),
            calls: 0,
        }
    }

    /// epoch 末の呼び出し（`training.rs` から。`Err` はそのまま返す）。
    pub(super) fn invoke(
        &mut self,
        epoch_local: usize,
        history: &History,
    ) -> Result<(), AutodiffError> {
        self.calls += 1;
        (self.f)(epoch_local, history)
    }
}

impl std::fmt::Debug for LambdaCallback {
    // boxed closure は `Debug` を持たないため、型名と呼び出し回数だけを
    // 出す（`LrSchedule` の手書き `Debug` と同型）。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LambdaCallback")
            .field("calls", &self.calls)
            .finish()
    }
}
/// [`Sequential::fit_with_callbacks`] へ渡す callback（閉じた集合。
/// trait object によるユーザー拡張は対象外——`&mut Sequential` を
/// callback へ渡すと `fit` 内部の借用構造〈`compiled` 取り外し・
/// `bind`〉と衝突するため。モジュール冒頭 doc「対象外・切り出し
/// 候補」節）。
#[non_exhaustive]
#[derive(Debug)]
pub enum Callback {
    EarlyStopping(EarlyStopping),
    ModelCheckpoint(ModelCheckpoint),
    LrSchedule(LrSchedule),
    /// CSV ロガー（[`CsvLogger`]。イシュー #2571）。
    CsvLogger(CsvLogger),
    /// JSON ロガー（[`JsonLogger`]。イシュー #2571）。
    JsonLogger(JsonLogger),
    /// epoch 末クロージャ（[`LambdaCallback`]。イシュー #2571）。
    Lambda(LambdaCallback),
}

impl Callback {
    pub(super) fn requires_validation(&self) -> bool {
        match self {
            Callback::EarlyStopping(es) => es.requires_validation(),
            Callback::ModelCheckpoint(mc) => mc.requires_validation(),
            Callback::LrSchedule(ls) => ls.requires_validation(),
            Callback::CsvLogger(_) | Callback::JsonLogger(_) | Callback::Lambda(_) => false,
        }
    }

    /// fit 終了時の後始末（ロガーのハンドル・in-memory 配列の解放）。
    /// `training.rs` が成功・失敗・打ち切りのいずれの経路でも呼ぶ。
    pub(super) fn end_fit(&mut self) {
        match self {
            Callback::CsvLogger(l) => l.end_fit(),
            Callback::JsonLogger(l) => l.end_fit(),
            _ => {}
        }
    }

    /// このコールバックが監視する [`Monitor`]（`LrSchedule::PerEpoch`
    /// のみ監視指標を持たず `None`）。`training.rs` の事前検査
    /// （`ValMetric(m)` の `m` が要求 `metrics` に含まれるか・非スカラー
    /// でないかの検査。イシュー #2072）専用の読み取り専用アクセサ
    /// （同一モジュール内のため各 variant の非公開 `monitor` フィールド
    /// へ直接アクセスする）。
    pub(super) fn monitor(&self) -> Option<Monitor> {
        match self {
            Callback::EarlyStopping(es) => Some(es.monitor),
            Callback::ModelCheckpoint(mc) => Some(mc.monitor),
            Callback::LrSchedule(ls) => ls.monitor(),
            Callback::CsvLogger(_) | Callback::JsonLogger(_) | Callback::Lambda(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hist(loss: Vec<f32>, val_loss: Vec<f32>) -> History {
        History {
            loss,
            val_loss,
            lr: Vec::new(),
            val_metrics: Vec::new(),
        }
    }

    // =====================================================================
    // Monitor::value_at
    // =====================================================================

    #[test]
    fn monitor_value_at_reads_correct_column() {
        let h = hist(vec![1.0, 2.0], vec![10.0, 20.0]);
        assert_eq!(Monitor::Loss.value_at(&h, 1), Some(2.0));
        assert_eq!(Monitor::ValLoss.value_at(&h, 1), Some(20.0));
        assert_eq!(Monitor::Loss.value_at(&h, 5), None);
    }

    #[test]
    fn monitor_requires_validation_only_for_val_loss() {
        assert!(!Monitor::Loss.requires_validation());
        assert!(Monitor::ValLoss.requires_validation());
    }

    // =====================================================================
    // MonitorMode::is_improvement（NaN 非改善契約を含む）
    // =====================================================================

    #[test]
    fn min_mode_improvement_respects_min_delta() {
        let mode = MonitorMode::Min;
        assert!(mode.is_improvement(0.89, 1.0, 0.1));
        assert!(!mode.is_improvement(0.91, 1.0, 0.1));
        assert!(!mode.is_improvement(1.0, 1.0, 0.0));
    }

    #[test]
    fn max_mode_improvement_respects_min_delta() {
        let mode = MonitorMode::Max;
        assert!(mode.is_improvement(1.11, 1.0, 0.1));
        assert!(!mode.is_improvement(1.09, 1.0, 0.1));
    }

    #[test]
    fn nan_value_is_never_an_improvement() {
        for mode in [MonitorMode::Min, MonitorMode::Max] {
            assert!(!mode.is_improvement(f32::NAN, mode.initial_best(), 0.0));
            assert!(!mode.is_improvement(f32::NAN, 1.0, 0.0));
        }
    }

    #[test]
    fn nan_best_never_blocks_a_finite_improvement() {
        // best 自体が NaN になることはない契約だが、防御的に:
        // 比較式は NaN を含むいかなる組み合わせでも false を返す。
        assert!(!MonitorMode::Min.is_improvement(1.0, f32::NAN, 0.0));
    }

    // =====================================================================
    // EarlyStopping
    // =====================================================================

    #[test]
    fn early_stopping_first_observation_always_improves() {
        let mut es = EarlyStopping::new(2).monitor(Monitor::Loss);
        let stop = es.observe(5.0, 0, HashMap::new);
        assert!(!stop);
        assert_eq!(es.best_value(), Some(5.0));
        assert_eq!(es.best_epoch(), Some(0));
    }

    #[test]
    fn early_stopping_nan_first_observation_never_becomes_best() {
        // patience=2: 1 回目の非改善（wait=1）ではまだ停止しないことを
        // 確認したいので patience=1 ではなく 2 を使う（patience=1 だと
        // 1 回目の非改善で wait(=1) >= patience(=1) が成立し即停止する
        // ため、「NaN 初回観測は改善扱いされない」ことと「即座には
        // 停止しない」ことの両方を 1 テストで区別できない）。
        let mut es = EarlyStopping::new(2).monitor(Monitor::Loss);
        let stop = es.observe(f32::NAN, 0, HashMap::new);
        assert!(
            !stop,
            "NaN 初回観測は改善扱いされないが、patience=2 の 1 回目では停止しない"
        );
        assert_eq!(es.best_value(), None);
        assert_eq!(es.best_epoch(), None);
        // 2 回目の非改善（NaN のまま）で patience=2 に到達し停止する。
        let stop2 = es.observe(f32::NAN, 1, HashMap::new);
        assert!(stop2);
        assert_eq!(es.stopped_epoch(), Some(1));
    }

    #[test]
    fn early_stopping_patience_zero_does_not_stop_on_improving_epoch() {
        let mut es = EarlyStopping::new(0).monitor(Monitor::Loss);
        assert!(!es.observe(1.0, 0, HashMap::new));
        // 非改善の最初の epoch で即停止（patience=0）。
        assert!(es.observe(1.0, 1, HashMap::new));
        assert_eq!(es.stopped_epoch(), Some(1));
    }

    #[test]
    fn early_stopping_reset_for_fit_clears_state() {
        let mut es = EarlyStopping::new(0).monitor(Monitor::Loss);
        assert!(!es.observe(1.0, 0, HashMap::new));
        assert!(es.observe(2.0, 1, HashMap::new));
        assert_eq!(es.stopped_epoch(), Some(1));
        es.reset_for_fit();
        assert_eq!(es.stopped_epoch(), None);
        assert_eq!(es.best_value(), None);
        assert_eq!(es.best_epoch(), None);
    }

    #[test]
    fn early_stopping_restore_best_weights_off_never_snapshots() {
        let mut es = EarlyStopping::new(1).monitor(Monitor::Loss);
        let mut calls = 0;
        es.observe(1.0, 0, || {
            calls += 1;
            HashMap::new()
        });
        assert_eq!(
            calls, 0,
            "restore_best_weights=false ではスナップショットしない"
        );
        assert!(es.take_restore_state().is_none());
    }

    #[test]
    fn early_stopping_restore_best_weights_on_snapshots_only_on_improvement() {
        let mut es = EarlyStopping::new(2)
            .monitor(Monitor::Loss)
            .restore_best_weights(true);
        let mut calls = 0;
        {
            let mut snap = || {
                calls += 1;
                let mut m = HashMap::new();
                m.insert(
                    "k".to_string(),
                    Tensor::<f32>::new(vec![1.0], &[1]).unwrap(),
                );
                m
            };
            es.observe(2.0, 0, &mut snap); // 改善 → snapshot
            es.observe(3.0, 1, &mut snap); // 非改善 → snapshot なし
        }
        assert_eq!(calls, 1);
        let restored = es.take_restore_state();
        assert!(restored.is_some());
        // 取り出し後は 2 度目の take は None。
        assert!(es.take_restore_state().is_none());
    }

    // =====================================================================
    // ModelCheckpoint
    // =====================================================================

    #[test]
    fn model_checkpoint_epoch_counter_persists_across_observe_calls() {
        let mut mc = ModelCheckpoint::new().monitor(Monitor::Loss);
        let model = Sequential::new();
        mc.observe(2.0, &model)
            .expect("in-memory のみでは失敗しない");
        mc.observe(1.0, &model)
            .expect("in-memory のみでは失敗しない"); // 改善
        mc.observe(1.5, &model)
            .expect("in-memory のみでは失敗しない"); // 非改善
        assert_eq!(mc.best_value(), Some(1.0));
        assert_eq!(mc.best_epoch(), Some(1));
    }

    #[test]
    fn model_checkpoint_save_best_only_false_keeps_last_epoch_snapshot() {
        let mut mc = ModelCheckpoint::new()
            .monitor(Monitor::Loss)
            .save_best_only(false);
        let model = Sequential::new();
        mc.observe(1.0, &model)
            .expect("in-memory のみでは失敗しない");
        mc.observe(2.0, &model)
            .expect("in-memory のみでは失敗しない"); // 非改善でもスナップショットは更新される
        assert!(mc.best_state_dict().is_some());
        assert_eq!(mc.best_value(), Some(1.0), "best 値自体は改善時のみ更新");
    }

    #[test]
    fn model_checkpoint_mode_max_after_new_uses_correct_initial_best() {
        // `mode()` を `new()` の後・観測前に呼んだ場合でも、`best` は
        // `Option` のため `Max` の実効初期値（-INF）が使われる（`f32`
        // で固定初期値を先に書き込む実装だと `Min` 用の `+INF` が
        // 残ってしまい最初の観測が絶対に改善にならないバグを生む）。
        let mut mc = ModelCheckpoint::new().mode(MonitorMode::Max);
        let model = Sequential::new();
        mc.observe(-1.0, &model)
            .expect("in-memory のみでは失敗しない");
        assert_eq!(mc.best_value(), Some(-1.0));
    }

    // =====================================================================
    // LrSchedule
    // =====================================================================

    struct ConstAt {
        value: f32,
    }
    impl LrScheduler for ConstAt {
        fn lr_at(&self, _step: usize) -> f32 {
            self.value
        }
    }

    #[test]
    fn lr_schedule_per_epoch_advance_increments_epoch_without_monitor() {
        let mut ls = LrSchedule::per_epoch(ConstAt { value: 0.1 });
        assert_eq!(ls.epoch(), 0);
        assert_eq!(ls.lr_for_epoch_begin(), 0.1);
        ls.advance(None).unwrap();
        assert_eq!(ls.epoch(), 1);
        assert_eq!(ls.monitor(), None);
        assert!(!ls.requires_validation());
    }

    #[test]
    fn lr_schedule_plateau_requires_validation_by_default() {
        let sched = ReduceLrOnPlateau::new(
            0.1,
            fandhe_ai_autodiff::nn::optim::ReduceLrOnPlateauConfig::default(),
        )
        .unwrap();
        let ls = LrSchedule::plateau(sched);
        assert_eq!(ls.monitor(), Some(Monitor::ValLoss));
        assert!(ls.requires_validation());
    }

    #[test]
    fn lr_schedule_plateau_advance_without_monitor_value_is_rejected() {
        let sched = ReduceLrOnPlateau::new(
            0.1,
            fandhe_ai_autodiff::nn::optim::ReduceLrOnPlateauConfig::default(),
        )
        .unwrap();
        let mut ls = LrSchedule::plateau_with_monitor(sched, Monitor::Loss);
        let err = ls.advance(None).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }
}
