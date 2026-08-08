//! プロセス起動コスト計測ハーネス（TASK-13.1a・イシュー #170）。
//!
//! v1（旧 `rust-ai-library-v1`）は CubeCL/Burn の JIT・autotune を前提に
//! コールド（`target/autotune/` 削除後の初回起動）・ウォームの起動コストを
//! PyTorch 比で実測していた（PoC-5。ウォーム時 約 1.9〜2.7 倍・コールド時
//! 約 21〜24 倍。`docs/spec/03-poc/poc-5-performance/README.md`）。v2 自作
//! カーネル（CUDA は NVRTC 実行時コンパイル・autotune 探索なし）では
//! 「コールド」に対応する既存の永続キャッシュが存在しないため、本モジュールは
//! v2 向けにコールド／ウォームを再定義したうえで、再現可能な計測を行う
//! ハーネスのみを提供する（実測の実施・v1 差分の記録は兄弟イシュー #171・
//! TASK-13.1b のスコープ。本イシューでは行わない）。
//!
//! ## コールド／ウォームの v2 定義
//!
//! CUDA の起動コストに影響する永続状態は、NVRTC コンパイル結果を保持する
//! CUDA ドライバの JIT キャッシュ（既定 `CUDA_CACHE_PATH=~/.nv/ComputeCache`）
//! である（`crates/backend-cuda/src/nvrtc.rs` はコンパイル結果をディスクへ
//! 永続化しないため、NVRTC コンパイル自体は毎プロセス発生する。差が出るのは
//! ドライバ側 JIT キャッシュと OS ページキャッシュ）。
//!
//! | 状態 | 定義 | 実現方法 |
//! |------|------|---------|
//! | コールド | ドライバ JIT キャッシュなしの初回起動 | 試行ごとに新規の空ディレクトリを作成し子プロセスの `CUDA_CACHE_PATH` に設定する |
//! | ウォーム | キャッシュ存在下での再起動 | priming 実行を 1 回行った後、同一の `CUDA_CACHE_PATH` を再利用して計測する |
//!
//! ユーザーの実キャッシュ（`~/.nv/ComputeCache`）には一切触れない
//! （hermetic・再現可能・並列実行される他イシューの作業を汚さない）。
//! 環境変数の設定は [`std::process::Command::env`] で子プロセスのみに適用し、
//! 本プロセス（ハーネス）・共有環境は変更しない（`.claude/rules/security.md` A08）。
//!
//! CPU バックエンドは JIT を持たないため、コールド／ウォームの実測値は
//! 理論上ほぼ同一になるはずであり、その事実自体が v1 との重要な差分データ点
//! となる（#171 の実測で確認する）。
//!
//! ## 計測プロトコル（`protocol` モジュールとの回数使い分け）
//!
//! [`protocol::MIN_ITERATIONS`]（20 回以上）はプロセス内カーネル単発計測用の
//! 下限であり、プロセスそのものを毎回 spawn するプロセスレベル計測に同じ下限を
//! 課すと 1 フェーズあたり数十プロセスの起動が必要になり過大である。本モジュールは
//! `.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値を採用し」に従い、
//! 既定 [`DEFAULT_STARTUP_TRIALS`]（5 回）の中央値＋Q1/Q3 を採用する独立した
//! [`StartupConfig`] を用いる（`protocol::MeasurementConfig` の下限とは無関係）。
//!
//! ## 計測 2 系統
//!
//! - **内部計測**: probe 子プロセス（[`crate`] の `bin/startup_probe`）が
//!   `main()` 冒頭の [`std::time::Instant`] から (1) バックエンド初期化完了・
//!   (2) 初回カーネル完了（`BackendOps::gemm` 呼び出しが同期込みで返った時点。
//!   `sync` モジュールの契約と同様「ホスト転送を伴わない完了待ち」の後）までを
//!   自己計測し、`ProbeReport` として標準出力へ 1 行 JSON で出力する。
//! - **外部計測**: 本モジュール（親ハーネス）が `Command::spawn` 前後の
//!   [`std::time::Instant`] で計測する wall time（プロセス生成・動的リンク込み。
//!   v1 の `time` コマンド計測に対応）。
//!
//! ## セキュリティ（OWASP Top 10。`.claude/rules/security.md`）
//!
//! - probe への引数は許可リスト方式（[`StartupBackend::parse`]）で検証し、
//!   `Command` の引数配列で子プロセスを起動する（シェル経由の文字列展開を行わない。A03）。
//! - probe 標準出力の JSON は読み取りサイズ上限（[`PROBE_STDOUT_LIMIT_BYTES`]）付きで
//!   読み取ったうえで `serde_json` により型付きパースし、[`StartupReport::validate`] /
//!   [`ProbeReport`] のスキーマ検証を経ずに生値へアクセスできる経路は設けない（A03・A08）。
//! - 一時キャッシュディレクトリは一意名で [`std::fs::create_dir`]（既存なら失敗）し、
//!   予測可能パスの事前作成・シンボリックリンク差し替えによる書き込み先誘導を排除する（A01）。

use crate::stats::{self, BenchError as StatsError};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// [`StartupReport`] の JSON スキーマバージョン（`report::SCHEMA_VERSION` と同じ手法。
/// 未知バージョンは [`StartupReport::validate`] が fail-closed で拒否する）。
pub const STARTUP_SCHEMA_VERSION: &str = "1";

/// probe 子プロセスが出力する [`ProbeReport`] の JSON スキーマバージョン。
/// `StartupReport` とは独立に管理する（probe バイナリのみが生成・消費する内部契約のため）。
pub const PROBE_SCHEMA_VERSION: &str = "1";

/// 既定の試行回数（1 フェーズあたり。モジュール冒頭ドキュメント参照）。
pub const DEFAULT_STARTUP_TRIALS: usize = 5;

/// probe 子プロセス標準出力の読み取り上限（バイト）。
/// 外部プロセス出力を無制限に読み込むと、異常終了・想定外出力時のメモリ膨張に
/// さらされるため上限を設ける（`.claude/rules/security.md` A03: 外部入力の検証）。
const PROBE_STDOUT_LIMIT_BYTES: usize = 1 << 20; // 1 MiB

/// コールド／ウォームの試行間で使う一意なディレクトリ名を採番するカウンタ。
/// プロセス内で複数フェーズ・複数試行を実行しても衝突しないよう、
/// タイムスタンプ（ナノ秒）・プロセス ID に加えて用いる（`scratch_dir` 参照）。
static SCRATCH_COUNTER: AtomicU64 = AtomicU64::new(0);

/// 起動コスト計測ハーネス固有のエラー。
///
/// 本番経路で `unwrap`/`expect` を使わない方針（`.claude/rules/coding-rust.md`）に基づき、
/// probe 起動失敗・出力検証失敗をすべて型付きエラーとして呼び出し元へ返す。
#[derive(Debug)]
pub enum StartupError {
    /// 引数検証（[`StartupBackend::parse`]）の失敗。
    InvalidArgument(String),
    /// ファイル I/O（probe spawn・一時ディレクトリ操作）の失敗。
    Io(String),
    /// probe が非ゼロ終了コードで終了した（stderr を含む）。
    ProbeExitFailure { status: String, stderr: String },
    /// probe 標準出力が [`PROBE_STDOUT_LIMIT_BYTES`] を超過した。
    ProbeOutputTooLarge(usize),
    /// probe 標準出力の JSON デコードに失敗した。
    ProbeJsonInvalid(String),
    /// スキーマバージョン不一致・値域逸脱などプロトコル違反（fail-closed 拒否）。
    ProtocolViolation(String),
    /// 分位点計算（[`stats::median_q1_q3`]）の失敗。
    Stats(StatsError),
}

impl fmt::Display for StartupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StartupError::InvalidArgument(msg) => write!(f, "引数不正: {msg}"),
            StartupError::Io(msg) => write!(f, "I/O エラー: {msg}"),
            StartupError::ProbeExitFailure { status, stderr } => {
                write!(f, "probe が異常終了（{status}）: {stderr}")
            }
            StartupError::ProbeOutputTooLarge(len) => {
                write!(f, "probe 標準出力が上限を超過（{len} バイト）")
            }
            StartupError::ProbeJsonInvalid(msg) => {
                write!(f, "probe 標準出力の JSON 解析失敗: {msg}")
            }
            StartupError::ProtocolViolation(msg) => write!(f, "計測プロトコル違反: {msg}"),
            StartupError::Stats(e) => write!(f, "統計計算エラー: {e}"),
        }
    }
}

impl std::error::Error for StartupError {}

impl From<StatsError> for StartupError {
    fn from(e: StatsError) -> Self {
        StartupError::Stats(e)
    }
}

/// 計測対象バックエンド識別子。probe バイナリの第 1 引数・
/// [`StartupReport::backend`] の両方で用いる自由文字列表現を持つ。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartupBackend {
    Cpu,
    Cuda,
    /// `cfg(target_os = "macos")` 限定（`.claude/rules/deps-policy.md`）。
    /// 非 macOS でも列挙自体は許容し、probe 側が実行時に
    /// 「このプラットフォームでは Metal 未対応」を型付きエラーで返す。
    Metal,
}

impl StartupBackend {
    /// probe 引数・レポート `backend` フィールドで用いる文字列表現。
    pub fn as_str(&self) -> &'static str {
        match self {
            StartupBackend::Cpu => "cpu",
            StartupBackend::Cuda => "cuda",
            StartupBackend::Metal => "metal",
        }
    }

    /// 許可リスト方式の引数検証（`.claude/rules/security.md` A03）。
    /// `cpu`／`cuda`／`metal` 以外はすべて [`StartupError::InvalidArgument`] とする。
    pub fn parse(s: &str) -> Result<Self, StartupError> {
        match s {
            "cpu" => Ok(StartupBackend::Cpu),
            "cuda" => Ok(StartupBackend::Cuda),
            "metal" => Ok(StartupBackend::Metal),
            other => Err(StartupError::InvalidArgument(format!(
                "未知のバックエンド指定 {other:?}（cpu / cuda / metal のいずれかを指定する）"
            ))),
        }
    }
}

/// コールド／ウォームの計測フェーズ（モジュール冒頭「v2 定義」参照）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartupPhase {
    Cold,
    Warm,
}

impl StartupPhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            StartupPhase::Cold => "cold",
            StartupPhase::Warm => "warm",
        }
    }
}

/// probe 子プロセスが標準出力へ JSON で返す内部計測結果（1 試行分）。
///
/// `startup_probe` バイナリと本モジュールの driver（[`run_phase`]）が共有する
/// プロセス間契約。`schema_version` 不一致は [`run_probe_once`] が fail-closed で拒否する
/// （`report::BenchReport` と同じ「検証を経ない値へアクセスできる経路を設けない」方針）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeReport {
    pub schema_version: String,
    pub backend: String,
    /// `main()` 開始からバックエンド初期化完了までの秒数。
    ///
    /// CPU／CUDA の `*BackendOps::new()` はハンドル構築のみで driver 初期化を
    /// 行わない契約のため（`crates/backend-cuda/src/ops.rs` doc コメント参照）、
    /// CUDA についてはこの区間で明示的に `CudaDevice::new` を呼び、実際の
    /// driver 初期化コストをここに含める（`startup_probe.rs` 実装参照）。
    /// CPU はハンドル構築コストのみを計測する（初期化不要な参照点）。
    pub device_init_secs: f64,
    /// `main()` 開始から初回カーネル（GEMM）完了（同期込み）までの秒数。
    /// `device_init_secs` 以上になる（デバイス初期化 → NVRTC コンパイル・カーネル起動・
    /// 完了待ちの順で進むため）。
    pub first_kernel_secs: f64,
}

impl ProbeReport {
    /// [`PROBE_SCHEMA_VERSION`] 一致・値の有限性・単調性を fail-closed で検証する。
    pub fn validate(&self, expected_backend: StartupBackend) -> Result<(), StartupError> {
        if self.schema_version != PROBE_SCHEMA_VERSION {
            return Err(StartupError::ProtocolViolation(format!(
                "probe の未知の schema_version: 期待値 {PROBE_SCHEMA_VERSION:?}, 実際 {:?}",
                self.schema_version
            )));
        }
        if self.backend != expected_backend.as_str() {
            return Err(StartupError::ProtocolViolation(format!(
                "probe backend 不一致: 期待値 {:?}, 実際 {:?}",
                expected_backend.as_str(),
                self.backend
            )));
        }
        if !self.device_init_secs.is_finite() || self.device_init_secs < 0.0 {
            return Err(StartupError::ProtocolViolation(format!(
                "device_init_secs が非有限値または負値: {}",
                self.device_init_secs
            )));
        }
        if !self.first_kernel_secs.is_finite() || self.first_kernel_secs < self.device_init_secs {
            return Err(StartupError::ProtocolViolation(format!(
                "first_kernel_secs が非有限値、または device_init_secs を下回る: first={}, device_init={}",
                self.first_kernel_secs, self.device_init_secs
            )));
        }
        Ok(())
    }

    /// 標準出力へ 1 行 JSON として書き出す文字列を生成する（probe バイナリから使用）。
    pub fn to_json(&self) -> Result<String, StartupError> {
        serde_json::to_string(self)
            .map_err(|e| StartupError::ProtocolViolation(format!("JSON エンコード失敗: {e}")))
    }
}

/// 分位点（中央値・Q1・Q3）の JSON 表現。[`stats::Quartiles`] は `Serialize` を
/// 持たないため（値の意味づけを計測コアの関心事に閉じる設計。`stats.rs` 冒頭コメント）、
/// 出力専用の変換先としてここに定義する。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuartileSecs {
    pub median: f64,
    pub q1: f64,
    pub q3: f64,
}

impl From<stats::Quartiles> for QuartileSecs {
    fn from(q: stats::Quartiles) -> Self {
        Self {
            median: q.median,
            q1: q.q1,
            q3: q.q3,
        }
    }
}

/// 1 試行分の計測結果（外部計測 + 内部計測の統合）。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StartupTrial {
    /// 外部計測: `Command::spawn` 前後の wall time（プロセス生成・動的リンク込み）。
    pub wall_secs: f64,
    /// 内部計測（[`ProbeReport::device_init_secs`] 由来）。
    pub device_init_secs: f64,
    /// 内部計測（[`ProbeReport::first_kernel_secs`] 由来）。
    pub first_kernel_secs: f64,
}

/// 起動コスト計測の構造化出力（1 バックエンド × 1 フェーズ分）。
///
/// `report::BenchReport` と同じ「検証済み DTO」方針を踏襲する:
/// [`Self::to_json`] は書き出し前に、[`Self::from_json`] は読み込み直後に
/// 必ず [`Self::validate`] を通す（`.claude/rules/security.md` A08）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StartupReport {
    pub schema_version: String,
    pub backend: String,
    pub phase: String,
    pub trials: usize,
    pub wall_secs: QuartileSecs,
    pub device_init_secs: QuartileSecs,
    pub first_kernel_secs: QuartileSecs,
    /// 全試行の生データ（#171 での詳細分析・再現性検証に使う）。
    pub samples: Vec<StartupTrial>,
}

impl StartupReport {
    /// `trials` 分の [`StartupTrial`] から分位点を集計して構築する。
    fn from_trials(
        backend: StartupBackend,
        phase: StartupPhase,
        samples: Vec<StartupTrial>,
    ) -> Result<Self, StartupError> {
        let wall: Vec<f64> = samples.iter().map(|t| t.wall_secs).collect();
        let device_init: Vec<f64> = samples.iter().map(|t| t.device_init_secs).collect();
        let first_kernel: Vec<f64> = samples.iter().map(|t| t.first_kernel_secs).collect();

        let report = Self {
            schema_version: STARTUP_SCHEMA_VERSION.to_string(),
            backend: backend.as_str().to_string(),
            phase: phase.as_str().to_string(),
            trials: samples.len(),
            wall_secs: stats::median_q1_q3(&wall)?.into(),
            device_init_secs: stats::median_q1_q3(&device_init)?.into(),
            first_kernel_secs: stats::median_q1_q3(&first_kernel)?.into(),
            samples,
        };
        report.validate()?;
        Ok(report)
    }

    /// スキーマ・試行数整合・値の有限性を fail-closed で検証する
    /// （`report::BenchReport::validate` と同型の方針。詳細は同モジュール参照）。
    pub fn validate(&self) -> Result<(), StartupError> {
        if self.schema_version != STARTUP_SCHEMA_VERSION {
            return Err(StartupError::ProtocolViolation(format!(
                "未知の schema_version: 期待値 {STARTUP_SCHEMA_VERSION:?}, 実際 {:?}",
                self.schema_version
            )));
        }
        if self.phase != "cold" && self.phase != "warm" {
            return Err(StartupError::ProtocolViolation(format!(
                "未知の phase: {:?}（cold / warm のいずれかのはず）",
                self.phase
            )));
        }
        if self.trials == 0 {
            return Err(StartupError::ProtocolViolation(
                "trials は 1 以上が必須".to_string(),
            ));
        }
        if self.samples.len() != self.trials {
            return Err(StartupError::ProtocolViolation(format!(
                "samples の要素数（{}）が trials（{}）と不一致",
                self.samples.len(),
                self.trials
            )));
        }
        for (label, q) in [
            ("wall_secs", &self.wall_secs),
            ("device_init_secs", &self.device_init_secs),
            ("first_kernel_secs", &self.first_kernel_secs),
        ] {
            if !(q.median.is_finite() && q.q1.is_finite() && q.q3.is_finite()) {
                return Err(StartupError::ProtocolViolation(format!(
                    "{label} に非有限値が含まれている"
                )));
            }
            if !(q.q1 <= q.median && q.median <= q.q3) {
                return Err(StartupError::ProtocolViolation(format!(
                    "{label}: q1 <= median <= q3 を満たさない（q1={}, median={}, q3={}）",
                    q.q1, q.median, q.q3
                )));
            }
        }
        for t in &self.samples {
            if !(t.wall_secs.is_finite() && t.wall_secs >= 0.0) {
                return Err(StartupError::ProtocolViolation(
                    "samples に非有限値または負の wall_secs が含まれている".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// JSON へシリアライズする（書き出し前に [`Self::validate`] を実行する）。
    pub fn to_json(&self) -> Result<String, StartupError> {
        self.validate()?;
        serde_json::to_string_pretty(self)
            .map_err(|e| StartupError::ProtocolViolation(format!("JSON エンコード失敗: {e}")))
    }

    /// JSON からデシリアライズし、[`Self::validate`] を通してから返す。
    pub fn from_json(json: &str) -> Result<Self, StartupError> {
        let report: Self = serde_json::from_str(json)
            .map_err(|e| StartupError::ProtocolViolation(format!("JSON デコード失敗: {e}")))?;
        report.validate()?;
        Ok(report)
    }
}

/// 起動コスト計測の実行設定。
#[derive(Debug, Clone)]
pub struct StartupConfig {
    pub backend: StartupBackend,
    /// 1 フェーズあたりの試行回数（[`DEFAULT_STARTUP_TRIALS`] 未満でも動作するが、
    /// 分位点の意味を持たせるため呼び出し側は既定値の使用を推奨する）。
    pub trials: usize,
    /// `startup_probe` バイナリへの絶対・相対パス。
    pub probe_path: PathBuf,
}

impl StartupConfig {
    /// `trials` が 1 以上であることを検証して構築する（0 回計測は分位点が定義できないため拒否）。
    pub fn new(
        backend: StartupBackend,
        trials: usize,
        probe_path: impl Into<PathBuf>,
    ) -> Result<Self, StartupError> {
        if trials == 0 {
            return Err(StartupError::InvalidArgument(
                "trials は 1 以上を指定する".to_string(),
            ));
        }
        Ok(Self {
            backend,
            trials,
            probe_path: probe_path.into(),
        })
    }
}

/// 一意な一時ディレクトリを `create_dir`（既存なら失敗）で新規作成する。
///
/// 予測可能パスの事前作成・シンボリックリンク差し替えによる書き込み先誘導を排除するため
/// （`.claude/rules/security.md` A01）、`SystemTime` のナノ秒・プロセス ID・
/// プロセス内カウンタを組み合わせた一意名を用いる。ベース（`std::env::temp_dir()` 配下の
/// 固定サブディレクトリ）自体は共有されるが、`create_dir_all` は既存ディレクトリを
/// エラーにしない一方、葉ディレクトリは `create_dir`（既存ならエラー）で作成するため、
/// 攻撃者が同名パスを事前に用意していた場合は本関数がエラーで検知する。
fn create_scratch_dir(label: &str) -> Result<PathBuf, StartupError> {
    let base = std::env::temp_dir().join("rust-ai-library-startup-bench");
    std::fs::create_dir_all(&base)
        .map_err(|e| StartupError::Io(format!("スクラッチ基点ディレクトリ作成失敗: {e}")))?;

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let counter = SCRATCH_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let leaf = base.join(format!("{label}-{pid}-{nanos}-{counter}"));

    std::fs::create_dir(&leaf)
        .map_err(|e| StartupError::Io(format!("スクラッチディレクトリ作成失敗 {leaf:?}: {e}")))?;
    Ok(leaf)
}

/// probe を 1 回起動し、外部計測（wall time）と内部計測（[`ProbeReport`]）を得る。
///
/// `cache_dir`: `Some` の場合は子プロセスの `CUDA_CACHE_PATH` にそのディレクトリを設定する
/// （コールド／ウォームいずれも本関数を経由する。呼び出し側がディレクトリの生成・再利用を
/// 制御する）。`None` の場合は `CUDA_CACHE_PATH` を明示的に子プロセス環境から除去する
/// （CPU 計測時や、親プロセスの環境変数を意図せず継承させないための明示化）。
///
/// 環境変数の設定は `Command::env`/`env_remove` により**子プロセスのみ**に適用され、
/// 本ハーネス（親プロセス）・共有環境は変更しない（`.claude/rules/security.md` A08）。
fn run_probe_once(
    config: &StartupConfig,
    cache_dir: Option<&Path>,
) -> Result<StartupTrial, StartupError> {
    let mut cmd = Command::new(&config.probe_path);
    cmd.arg(config.backend.as_str());
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    match cache_dir {
        Some(dir) => {
            cmd.env("CUDA_CACHE_PATH", dir);
        }
        None => {
            cmd.env_remove("CUDA_CACHE_PATH");
        }
    }

    let start = Instant::now();
    let output = cmd
        .output()
        .map_err(|e| StartupError::Io(format!("probe 起動失敗（{:?}）: {e}", config.probe_path)))?;
    let wall_secs = start.elapsed().as_secs_f64();

    if !output.status.success() {
        return Err(StartupError::ProbeExitFailure {
            status: output.status.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    if output.stdout.len() > PROBE_STDOUT_LIMIT_BYTES {
        return Err(StartupError::ProbeOutputTooLarge(output.stdout.len()));
    }

    let stdout_text = String::from_utf8_lossy(&output.stdout);
    let probe_report: ProbeReport = serde_json::from_str(stdout_text.trim())
        .map_err(|e| StartupError::ProbeJsonInvalid(e.to_string()))?;
    probe_report.validate(config.backend)?;

    Ok(StartupTrial {
        wall_secs,
        device_init_secs: probe_report.device_init_secs,
        first_kernel_secs: probe_report.first_kernel_secs,
    })
}

/// コールドフェーズを計測する: 試行ごとに新規の空 `CUDA_CACHE_PATH` を割り当てる
/// （モジュール冒頭「v2 定義」参照）。各試行のディレクトリは計測後に削除する
/// （削除失敗は計測結果の正しさに影響しないため `StartupError` に昇格させず無視する。
/// OS 側の一時領域クリーンアップに委ねる fail-safe）。
fn run_cold(config: &StartupConfig) -> Result<StartupReport, StartupError> {
    let mut samples = Vec::with_capacity(config.trials);
    for i in 0..config.trials {
        let dir = create_scratch_dir(&format!("cold-{i}"))?;
        let trial = run_probe_once(config, Some(&dir));
        let _ = std::fs::remove_dir_all(&dir);
        samples.push(trial?);
    }
    StartupReport::from_trials(config.backend, StartupPhase::Cold, samples)
}

/// ウォームフェーズを計測する: 単一の `CUDA_CACHE_PATH` を用意し、
/// priming 実行を 1 回行った後、同一ディレクトリを再利用して `config.trials` 回計測する
/// （モジュール冒頭「v2 定義」参照）。priming 自体の失敗はウォーム計測の前提が崩れるため
/// 通常のエラーとして呼び出し元へ伝播する。
fn run_warm(config: &StartupConfig) -> Result<StartupReport, StartupError> {
    let dir = create_scratch_dir("warm")?;
    let result = (|| {
        // priming: キャッシュを温めるための 1 回（計測結果には含めない）。
        run_probe_once(config, Some(&dir))?;

        let mut samples = Vec::with_capacity(config.trials);
        for _ in 0..config.trials {
            samples.push(run_probe_once(config, Some(&dir))?);
        }
        StartupReport::from_trials(config.backend, StartupPhase::Warm, samples)
    })();
    let _ = std::fs::remove_dir_all(&dir);
    result
}

/// 指定フェーズ（コールド／ウォーム）の起動コストを計測する（本モジュールの公開入口）。
///
/// `startup_bench` CLI（`src/bin/startup_bench.rs`）・統合テスト
/// （`tests/startup_harness.rs`）から呼ばれる。
pub fn run_phase(
    config: &StartupConfig,
    phase: StartupPhase,
) -> Result<StartupReport, StartupError> {
    match phase {
        StartupPhase::Cold => run_cold(config),
        StartupPhase::Warm => run_warm(config),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(wall: f64, device_init: f64, first_kernel: f64) -> StartupTrial {
        StartupTrial {
            wall_secs: wall,
            device_init_secs: device_init,
            first_kernel_secs: first_kernel,
        }
    }

    #[test]
    fn startup_backend_parse_accepts_allowed_values() {
        assert_eq!(StartupBackend::parse("cpu").unwrap(), StartupBackend::Cpu);
        assert_eq!(StartupBackend::parse("cuda").unwrap(), StartupBackend::Cuda);
        assert_eq!(
            StartupBackend::parse("metal").unwrap(),
            StartupBackend::Metal
        );
    }

    #[test]
    fn startup_backend_parse_rejects_unknown_and_injection_like_values() {
        for bad in ["gpu", "cpu; rm -rf /", "", "CPU", "cuda\n"] {
            assert!(
                StartupBackend::parse(bad).is_err(),
                "許可リスト外の値 {bad:?} は拒否されるはず"
            );
        }
    }

    #[test]
    fn startup_config_rejects_zero_trials() {
        let err = StartupConfig::new(StartupBackend::Cpu, 0, "probe").unwrap_err();
        assert!(matches!(err, StartupError::InvalidArgument(_)));
    }

    #[test]
    fn startup_report_from_trials_computes_quartiles() {
        let samples = vec![
            sample(0.10, 0.01, 0.05),
            sample(0.12, 0.02, 0.06),
            sample(0.11, 0.015, 0.055),
        ];
        let report =
            StartupReport::from_trials(StartupBackend::Cpu, StartupPhase::Warm, samples).unwrap();
        assert_eq!(report.trials, 3);
        assert_eq!(report.phase, "warm");
        assert_eq!(report.backend, "cpu");
        assert!(report.wall_secs.q1 <= report.wall_secs.median);
        assert!(report.wall_secs.median <= report.wall_secs.q3);
    }

    #[test]
    fn startup_report_json_roundtrip_preserves_content() {
        let samples = vec![sample(0.10, 0.01, 0.05), sample(0.12, 0.02, 0.06)];
        let report =
            StartupReport::from_trials(StartupBackend::Cuda, StartupPhase::Cold, samples).unwrap();
        let json = report.to_json().unwrap();
        let round_tripped = StartupReport::from_json(&json).unwrap();
        assert_eq!(report, round_tripped);
    }

    #[test]
    fn startup_report_from_json_rejects_unknown_schema_version() {
        let bad_json = r#"{"schema_version":"999","backend":"cpu","phase":"warm","trials":1,
            "wall_secs":{"median":0.1,"q1":0.1,"q3":0.1},
            "device_init_secs":{"median":0.01,"q1":0.01,"q3":0.01},
            "first_kernel_secs":{"median":0.05,"q1":0.05,"q3":0.05},
            "samples":[{"wall_secs":0.1,"device_init_secs":0.01,"first_kernel_secs":0.05}]}"#;
        let err = StartupReport::from_json(bad_json).unwrap_err();
        assert!(matches!(err, StartupError::ProtocolViolation(_)));
    }

    #[test]
    fn startup_report_from_json_rejects_trials_sample_count_mismatch() {
        let bad_json = format!(
            r#"{{"schema_version":"{STARTUP_SCHEMA_VERSION}","backend":"cpu","phase":"warm","trials":2,
            "wall_secs":{{"median":0.1,"q1":0.1,"q3":0.1}},
            "device_init_secs":{{"median":0.01,"q1":0.01,"q3":0.01}},
            "first_kernel_secs":{{"median":0.05,"q1":0.05,"q3":0.05}},
            "samples":[{{"wall_secs":0.1,"device_init_secs":0.01,"first_kernel_secs":0.05}}]}}"#
        );
        let err = StartupReport::from_json(&bad_json).unwrap_err();
        assert!(matches!(err, StartupError::ProtocolViolation(_)));
    }

    #[test]
    fn probe_report_validate_rejects_backend_mismatch() {
        let report = ProbeReport {
            schema_version: PROBE_SCHEMA_VERSION.to_string(),
            backend: "cuda".to_string(),
            device_init_secs: 0.01,
            first_kernel_secs: 0.02,
        };
        let err = report.validate(StartupBackend::Cpu).unwrap_err();
        assert!(matches!(err, StartupError::ProtocolViolation(_)));
    }

    #[test]
    fn probe_report_validate_rejects_first_kernel_before_device_init() {
        let report = ProbeReport {
            schema_version: PROBE_SCHEMA_VERSION.to_string(),
            backend: "cpu".to_string(),
            device_init_secs: 0.05,
            first_kernel_secs: 0.01,
        };
        let err = report.validate(StartupBackend::Cpu).unwrap_err();
        assert!(matches!(err, StartupError::ProtocolViolation(_)));
    }

    #[test]
    fn create_scratch_dir_returns_unique_paths_across_calls() {
        let a = create_scratch_dir("test").unwrap();
        let b = create_scratch_dir("test").unwrap();
        assert_ne!(a, b, "同一ラベルでも呼び出しごとに一意なパスになるはず");
        let _ = std::fs::remove_dir_all(&a);
        let _ = std::fs::remove_dir_all(&b);
    }
}
