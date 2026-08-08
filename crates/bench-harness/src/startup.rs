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
//!   (2) 初回カーネル完了（`BackendOps::gemm` 相当の呼び出しが完了待ち込みで
//!   返った時点）までを自己計測し、`ProbeReport` として標準出力へ 1 行 JSON で
//!   出力する。CUDA 経路は `CudaGemm::run_tiled_f32` がホスト側スライスを
//!   受け取り内部で `clone_htod`／`clone_dtoh` を行う契約であるため、(2) の
//!   区間には「NVRTC コンパイル＋カーネル起動＋完了待ち」に加えホスト⇔デバイス間
//!   転送コストが含まれる（`sync` モジュールの「ホスト転送を伴わない完了待ち」
//!   契約とは異なる。転送を除いたカーネル単体計測には `backend-cuda` 側に
//!   デバイス常駐バッファを扱う新規 API が要る。PR #360 codex-review 指摘・
//!   Medium「Kernel metric includes host transfers」への対応として、実装を
//!   変えず本ドキュメントの記述を実測経路に合わせて訂正した）。
//! - **外部計測**: 本モジュール（親ハーネス）が `Command::spawn` 前後の
//!   [`std::time::Instant`] で計測する wall time（プロセス生成・動的リンク込み。
//!   v1 の `time` コマンド計測に対応）。
//!
//! ## セキュリティ（OWASP Top 10。`.claude/rules/security.md`）
//!
//! - probe への引数は許可リスト方式（[`StartupBackend::parse`]）で検証し、
//!   `Command` の引数配列で子プロセスを起動する（シェル経由の文字列展開を行わない。A03）。
//! - probe 標準出力・標準エラー出力は `Command::spawn` 後に上限付きストリーミング読み取り
//!   （[`PROBE_STDOUT_LIMIT_BYTES`]／[`PROBE_STDERR_LIMIT_BYTES`]）を行い、超過を検知した
//!   時点で子プロセスを kill・reap してから拒否する（`Command::output` による全量バッファ
//!   後の事後チェックでは、上限チェックが効く前に異常肥大化した出力でメモリ枯渇しうるため
//!   採用しない。PR #360 codex-review 指摘 P1）。上限内に収まった標準出力のみ `serde_json`
//!   により型付きパースし、[`StartupReport::validate`] / [`ProbeReport::validate`] の
//!   スキーマ検証を経ずに生値へアクセスできる経路は設けない（A03・A08）。
//! - 一時キャッシュディレクトリは一意名で [`std::fs::create_dir`]（既存なら失敗）し、
//!   予測可能パスの事前作成・シンボリックリンク差し替えによる書き込み先誘導を排除する（A01）。
//! - probe 1 回の実行は [`PROBE_TIMEOUT`] を上限とし、`mpsc::Receiver::recv_timeout` で
//!   監視する。GPU ドライバ・カーネル停止等により probe が生存したまま無出力になっても
//!   `run_probe_once`（延いては `startup_bench`）が無期限にハングしないよう、上限超過時は
//!   子プロセスを kill・reap して型付きエラー（[`StartupError::ProbeTimeout`]）を返す
//!   （PR #360 codex-review 指摘 P1）。

use crate::stats::{self, BenchError as StatsError};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// [`StartupReport`] の JSON スキーマバージョン（`report::SCHEMA_VERSION` と同じ手法。
/// 未知バージョンは [`StartupReport::validate`] が fail-closed で拒否する）。
pub const STARTUP_SCHEMA_VERSION: &str = "1";

/// probe 子プロセスが出力する [`ProbeReport`] の JSON スキーマバージョン。
/// `StartupReport` とは独立に管理する（probe バイナリのみが生成・消費する内部契約のため）。
pub const PROBE_SCHEMA_VERSION: &str = "1";

/// 既定の試行回数（1 フェーズあたり。モジュール冒頭ドキュメント参照）。
pub const DEFAULT_STARTUP_TRIALS: usize = 5;

/// 1 フェーズあたりの試行回数の許容上限。
///
/// コールド計測は試行ごとにプロセス spawn ＋一時ディレクトリ作成を伴うため
/// （[`run_cold`]）、CLI から極端に大きい値（例: `u64::MAX`）を渡されると
/// [`run_cold`]／[`run_warm`] の `Vec::with_capacity(config.trials)` が
/// 過大なメモリ確保を試み、通常の [`StartupError`] では回収できない abort や
/// メモリ枯渇を招く（PR #360 codex-review P1 指摘）。実運用の試行回数
/// （[`DEFAULT_STARTUP_TRIALS`] 5 回・手動デバッグでの数十〜数百回）を
/// 十分に超える桁で上限を設け、[`StartupConfig::new`] で fail-closed に拒否する。
pub const MAX_STARTUP_TRIALS: usize = 10_000;

/// probe 子プロセス標準出力の許容上限（バイト）。
///
/// [`run_probe_once`] は `Command::spawn` 後に [`read_capped`] で標準出力を
/// この上限＋1 チャンク分までしか読み取らず、超過を検知した時点で子プロセスを
/// kill・reap してから [`StartupError::ProbeOutputTooLarge`] を返す（全量バッファ後に
/// 上限チェックする `Command::output` 方式は、異常肥大化した出力に対しチェックが効く前に
/// メモリ枯渇しうるため採用しない。PR #360 codex-review 指摘 P1）。probe は同一ビルドの
/// 第一者バイナリのため通常は超過しないが、異常終了・想定外出力（無限ループ等の実装バグや
/// `probe_path` 差し替え）時の異常データを打ち切る fail-closed な検証として上限を設ける
/// （`.claude/rules/security.md` A03）。
const PROBE_STDOUT_LIMIT_BYTES: usize = 1 << 20; // 1 MiB

/// probe 子プロセス標準エラー出力の許容上限（バイト）。[`PROBE_STDOUT_LIMIT_BYTES`] と
/// 同じ理由・同じ [`read_capped`] 経路で適用する（旧実装は stderr に上限がなく、
/// 異常終了時の `ProbeExitFailure` 生成経路がメモリ膨張に無防備だった。PR #360 指摘 P1）。
const PROBE_STDERR_LIMIT_BYTES: usize = 1 << 20; // 1 MiB

/// probe 子プロセス 1 回の起動あたりに許容する最大経過時間。
///
/// [`run_probe_once`] は stdout／stderr 読み取りスレッドからの通知を [`mpsc::Receiver::recv_timeout`]
/// で待ち受け、`start`（`Command::spawn` 直後）からの経過がこの上限を超えた時点で
/// [`StartupError::ProbeTimeout`] を返す。無条件の `rx.recv()`（タイムアウトなし）は、
/// GPU ドライバ・カーネル側が停止し probe プロセスが生存したまま無出力（stdout／stderr
/// いずれのパイプも EOF に至らない）になった場合、読み取りスレッドの `read()` 自体が
/// 戻ってこないため `run_probe_once`（延いては `startup_bench` 全体）を無期限にハングさせる
/// （PR #360 codex-review 指摘 P1）。上限超過を検知した時点で [`std::process::Child::kill`]・
/// `wait` により子プロセスを回収し、パイプの書き込み端を閉じることで読み取りスレッドの
/// `read()` に EOF を届けて終了させる（`ProbeOutputTooLarge` 経路と同じ回収パターン）。
const PROBE_TIMEOUT: Duration = Duration::from_secs(60);

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
    /// probe の 1 回の実行が [`PROBE_TIMEOUT`] を超過した（GPU ドライバ・カーネル停止等による
    /// ハングを検知し、子プロセスを kill・reap した後に返す）。
    ProbeTimeout(Duration),
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
            StartupError::ProbeTimeout(timeout) => {
                write!(f, "probe の実行が上限時間を超過（{timeout:?}）")
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
    ///
    /// CUDA 経路は `device_init_secs` 計測で取得済みの `CudaDevice` を
    /// `CudaGemm::new` に明示的に渡すため（`startup_probe.rs::run_cuda`）、
    /// `first_kernel_secs - device_init_secs` の区間に device の二重初期化は
    /// 含まれない（#170 レビュー指摘への対応）。ただし `CudaGemm::run_tiled_f32`
    /// はホスト側スライスを受け取り内部で `clone_htod`／`clone_dtoh` を行う契約
    /// のため、同区間には「NVRTC コンパイル＋カーネル起動＋完了待ち」に加えて
    /// ホスト⇔デバイス間転送コストも含まれる（PR #360 codex-review 指摘・Medium
    /// 「Kernel metric includes host transfers」。転送を含まないカーネル単体計測が
    /// 要る場合は `backend-cuda` にデバイス常駐バッファを扱う API を追加する必要が
    /// あり、本タスクのスコープ外）。Metal 経路は `MetalBackendOps::gemm` 内部で
    /// `MetalContext` が都度再構築される現行実装（本 OS でビルド確認不能のため
    /// 計測経路は未変更）のため、同区間に `MetalContext` 再構築コストも混入しうる
    /// （`startup_probe.rs::run_metal` のコメント参照）。
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

    /// スキーマ・試行数整合・値の有限性・分位点の再計算一致を fail-closed で検証する
    /// （`report::BenchReport::validate` と同型の方針。詳細は同モジュール参照）。
    ///
    /// `backend` は許可リスト（[`StartupBackend::parse`]）で検証し、改ざんされた自由文字列を
    /// 「検証済み DTO」として後続の性能比較に渡さない（#170 レビュー指摘への対応。
    /// `.claude/rules/security.md` A08）。分位点（`wall_secs`／`device_init_secs`／
    /// `first_kernel_secs`）は非負性に加え、`samples` から [`stats::median_q1_q3`] で
    /// 再計算した値と完全一致することを要求する（同関数は決定的な
    /// ソート＋インデックス選択のみで構成され丸め誤差が入らないため、`==` 比較で安全。
    /// `stats.rs` の実装参照）。これにより「正常な samples と恣意的な集計値を併記した JSON」を
    /// 弾く。各 sample についても `wall_secs`（子プロセス全体の外部計測）が
    /// `first_kernel_secs`（内部計測）以上という契約（モジュール冒頭「計測 2 系統」参照）を検証する。
    pub fn validate(&self) -> Result<(), StartupError> {
        if self.schema_version != STARTUP_SCHEMA_VERSION {
            return Err(StartupError::ProtocolViolation(format!(
                "未知の schema_version: 期待値 {STARTUP_SCHEMA_VERSION:?}, 実際 {:?}",
                self.schema_version
            )));
        }
        StartupBackend::parse(&self.backend)
            .map_err(|e| StartupError::ProtocolViolation(format!("backend が許可リスト外: {e}")))?;
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

        for (i, t) in self.samples.iter().enumerate() {
            if !(t.wall_secs.is_finite() && t.wall_secs >= 0.0) {
                return Err(StartupError::ProtocolViolation(format!(
                    "samples[{i}] に非有限値または負の wall_secs が含まれている: {}",
                    t.wall_secs
                )));
            }
            if !(t.device_init_secs.is_finite() && t.device_init_secs >= 0.0) {
                return Err(StartupError::ProtocolViolation(format!(
                    "samples[{i}] に非有限値または負の device_init_secs が含まれている: {}",
                    t.device_init_secs
                )));
            }
            if !(t.first_kernel_secs.is_finite() && t.first_kernel_secs >= t.device_init_secs) {
                return Err(StartupError::ProtocolViolation(format!(
                    "samples[{i}] の first_kernel_secs が非有限値、または device_init_secs を \
                     下回る: first={}, device_init={}",
                    t.first_kernel_secs, t.device_init_secs
                )));
            }
            if !(t.wall_secs.is_finite() && t.wall_secs >= t.first_kernel_secs) {
                return Err(StartupError::ProtocolViolation(format!(
                    "samples[{i}] の wall_secs が非有限値、または first_kernel_secs を \
                     下回る（子プロセス全体の外部計測は内部計測以上のはず）: \
                     wall={}, first_kernel={}",
                    t.wall_secs, t.first_kernel_secs
                )));
            }
        }

        let wall: Vec<f64> = self.samples.iter().map(|t| t.wall_secs).collect();
        let device_init: Vec<f64> = self.samples.iter().map(|t| t.device_init_secs).collect();
        let first_kernel: Vec<f64> = self.samples.iter().map(|t| t.first_kernel_secs).collect();

        for (label, q, recomputed_from) in [
            ("wall_secs", &self.wall_secs, &wall),
            ("device_init_secs", &self.device_init_secs, &device_init),
            ("first_kernel_secs", &self.first_kernel_secs, &first_kernel),
        ] {
            if !(q.median.is_finite() && q.q1.is_finite() && q.q3.is_finite()) {
                return Err(StartupError::ProtocolViolation(format!(
                    "{label} に非有限値が含まれている"
                )));
            }
            if !(q.q1 >= 0.0 && q.median >= 0.0 && q.q3 >= 0.0) {
                return Err(StartupError::ProtocolViolation(format!(
                    "{label} に負の分位点が含まれている（q1={}, median={}, q3={}）",
                    q.q1, q.median, q.q3
                )));
            }
            if !(q.q1 <= q.median && q.median <= q.q3) {
                return Err(StartupError::ProtocolViolation(format!(
                    "{label}: q1 <= median <= q3 を満たさない（q1={}, median={}, q3={}）",
                    q.q1, q.median, q.q3
                )));
            }
            let recomputed: QuartileSecs = stats::median_q1_q3(recomputed_from)?.into();
            if recomputed != *q {
                return Err(StartupError::ProtocolViolation(format!(
                    "{label}: samples から再計算した分位点と格納値が不一致（改ざんの疑い）: \
                     recomputed={{median={}, q1={}, q3={}}}, stored={{median={}, q1={}, q3={}}}",
                    recomputed.median, recomputed.q1, recomputed.q3, q.median, q.q1, q.q3
                )));
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
    /// `trials` が `1..=MAX_STARTUP_TRIALS` の範囲内であることを検証して構築する
    /// （0 回計測は分位点が定義できないため拒否・上限超過は過大な `Vec::with_capacity`
    /// によるリソース枯渇を防ぐため拒否。上限根拠は [`MAX_STARTUP_TRIALS`] 参照）。
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
        if trials > MAX_STARTUP_TRIALS {
            return Err(StartupError::InvalidArgument(format!(
                "trials は {MAX_STARTUP_TRIALS} 以下を指定する（指定値: {trials}）"
            )));
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

/// [`run_probe_once`] の子プロセス読み取りスレッドがどちらのストリームを
/// 担当しているかをチャネル越しに親スレッドへ伝えるための識別子。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeStream {
    Stdout,
    Stderr,
}

/// 読み取りストリームを `limit` バイトを超えるまでチャンク単位で読み取る。
///
/// [`run_probe_once`] が probe 子プロセスの stdout／stderr パイプに用いる。
/// `Command::output` のように全量をバッファしてから上限検査する方式だと、
/// 異常肥大化した出力に対してチェックが効く前にメモリ枯渇しうる（PR #360
/// codex-review 指摘 P1）。本関数は読み取りループ中に `limit` 超過を検知した
/// 時点（最大でも 1 チャンク分の超過）で打ち切り、`(読み取り済みバイト列, 超過フラグ)`
/// を返す。呼び出し側は超過フラグが立った場合、破棄前提のバッファとして扱い、
/// 子プロセスを kill・reap する。
fn read_capped<R: Read>(mut reader: R, limit: usize) -> std::io::Result<(Vec<u8>, bool)> {
    let mut buf = Vec::with_capacity(limit.min(64 * 1024));
    let mut chunk = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut chunk)?;
        if n == 0 {
            return Ok((buf, false));
        }
        buf.extend_from_slice(&chunk[..n]);
        if buf.len() > limit {
            return Ok((buf, true));
        }
    }
}

/// probe を 1 回起動し、外部計測（wall time）と内部計測（[`ProbeReport`]）を得る。
///
/// `cache_dir`: `Some` の場合は子プロセスの `CUDA_CACHE_PATH` にそのディレクトリを設定する
/// （コールド／ウォームいずれも本関数を経由する。呼び出し側がディレクトリの生成・再利用を
/// 制御する）。`None` の場合は `CUDA_CACHE_PATH` を明示的に子プロセス環境から除去する
/// （CPU 計測時や、親プロセスの環境変数を意図せず継承させないための明示化）。
///
/// `CUDA_CACHE_DISABLE` は `cache_dir` の値によらず常に子プロセス環境から除去する。
/// この変数は driver の JIT キャッシュ機構そのものを無効化するため、親プロセス側の
/// 環境で `1` が設定されたまま継承させると、`cache_dir` へ独自の `CUDA_CACHE_PATH` を
/// 与えてもウォームフェーズの priming（1 回目の実行でキャッシュを充填する）が
/// 効かず、コールド／ウォームの計測結果が無言で同一値に収束してしまう
/// （PR #360 codex-review 指摘・Medium「Warm phase ignores cache disable」）。
///
/// 環境変数の設定は `Command::env`/`env_remove` により**子プロセスのみ**に適用され、
/// 本ハーネス（親プロセス）・共有環境は変更しない（`.claude/rules/security.md` A08）。
///
/// stdout／stderr は `spawn` 後に別スレッドで [`read_capped`] を用い上限付きに読み取る
/// （2 ストリームを同一スレッドで順に `read_to_end` すると、子プロセスが埋めた側の
/// パイプバッファが詰まりデッドロックしうるため並行読み取りが必須）。
///
/// 2 スレッドの結果は `mpsc` チャネルで親スレッドへ通知する（`JoinHandle::join` を
/// stdout → stderr の順に単純に呼ぶ構成は使わない）。stdout 側が上限超過を検知して
/// 早期終了しても、子プロセスはまだ生きたまま stdout パイプへ書き込み続けようとして
/// ブロックしうる。子プロセスが stdout 書き込みでブロックしている間は stderr 側の
/// EOF も届かないため、「stdout join 完了 → stderr join 開始」の順序だと stderr の
/// join が永久に返らずデッドロックする（PR #360 codex-review 指摘 P1）。
/// 本実装はどちらのスレッドが先に結果を送っても即座に受信できるよう両者からの
/// メッセージを同一チャネルで待ち受け、いずれかが上限超過を報告した時点で
/// （もう一方の完了を待たずに）直ちに子プロセスを kill する。これにより、
/// もう一方のスレッドが待っていた EOF は子プロセス終了に伴うパイプクローズで
/// 発生し、ブロックが解消される。
fn run_probe_once(
    config: &StartupConfig,
    cache_dir: Option<&Path>,
) -> Result<StartupTrial, StartupError> {
    run_probe_once_with_timeout(config, cache_dir, PROBE_TIMEOUT)
}

/// [`run_probe_once`] の本体。`timeout` を外部から注入できるようにし、テストが
/// [`PROBE_TIMEOUT`]（既定 60 秒）を待たずにタイムアウト経路を再現できるようにする
/// （`tests` モジュール `run_probe_once_rejects_probe_that_hangs_without_output` 参照）。
fn run_probe_once_with_timeout(
    config: &StartupConfig,
    cache_dir: Option<&Path>,
    timeout: Duration,
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
    // 親プロセスの `CUDA_CACHE_DISABLE=1` を継承すると `CUDA_CACHE_PATH` の設定が
    // 無意味化し、ウォームフェーズの priming が機能しなくなる（上記ドキュメント参照）。
    cmd.env_remove("CUDA_CACHE_DISABLE");

    let start = Instant::now();
    let mut child = cmd
        .spawn()
        .map_err(|e| StartupError::Io(format!("probe 起動失敗（{:?}）: {e}", config.probe_path)))?;

    // `Stdio::piped()` を指定しているため取得は必ず成功するが、型上は Option なので
    // 万一失敗した場合は型付きエラーとして扱う（本番経路で unwrap/expect を使わない方針。
    // `.claude/rules/coding-rust.md`）。
    let stdout_pipe = child
        .stdout
        .take()
        .ok_or_else(|| StartupError::Io("probe の stdout パイプ取得に失敗".to_string()))?;
    let stderr_pipe = child
        .stderr
        .take()
        .ok_or_else(|| StartupError::Io("probe の stderr パイプ取得に失敗".to_string()))?;

    // 各読み取りスレッドの完了通知を単一チャネルへ集約する。送信側は `(ストリーム種別,
    // read_capped の結果)` を積むだけで、上限超過時の子プロセス kill 判断は
    // 受信側（親スレッド）が一元的に行う（関数ドキュメント参照）。
    let (tx, rx) = mpsc::channel::<(ProbeStream, std::io::Result<(Vec<u8>, bool)>)>();
    let tx_stdout = tx.clone();
    let stdout_handle = std::thread::spawn(move || {
        let result = read_capped(stdout_pipe, PROBE_STDOUT_LIMIT_BYTES);
        let _ = tx_stdout.send((ProbeStream::Stdout, result));
    });
    let stderr_handle = std::thread::spawn(move || {
        let result = read_capped(stderr_pipe, PROBE_STDERR_LIMIT_BYTES);
        let _ = tx.send((ProbeStream::Stderr, result));
    });

    // `start`（spawn 直後）からの絶対期限。`rx.recv()`（タイムアウトなし）ではなく
    // `recv_timeout` を用いることで、GPU ドライバ・カーネル停止等により probe が
    // 生存したまま無出力になるケース（読み取りスレッドの `read()` 自体が戻らない）
    // でも [`PROBE_TIMEOUT`] を超えた時点で確実に制御を取り戻す（PR #360 codex-review
    // 指摘 P1。関数ドキュメント参照）。
    let deadline = start + timeout;
    let mut stdout_result: Option<std::io::Result<(Vec<u8>, bool)>> = None;
    let mut stderr_result: Option<std::io::Result<(Vec<u8>, bool)>> = None;
    let mut killed_for_overflow = false;
    let mut timed_out = false;
    while stdout_result.is_none() || stderr_result.is_none() {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            timed_out = true;
            break;
        }
        match rx.recv_timeout(remaining) {
            Ok((stream, result)) => {
                if matches!(&result, Ok((_, true))) && !killed_for_overflow {
                    // 上限超過を検知した時点で、もう一方のスレッドの完了を待たずに
                    // 子プロセスを kill する（もう一方が子プロセスの EOF 待ちで
                    // ブロックし続けるのを防ぐ）。
                    let _ = child.kill();
                    killed_for_overflow = true;
                }
                match stream {
                    ProbeStream::Stdout => stdout_result = Some(result),
                    ProbeStream::Stderr => stderr_result = Some(result),
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                timed_out = true;
                break;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // 両送信側が（panic 等で）送信せずに drop された。以降は各スレッドの
                // join エラーとして検出させるためループを抜ける。
                break;
            }
        }
    }

    if timed_out {
        // 上限時間超過: 直接の子プロセスを kill・reap する（`wait` を挟むことでゾンビ化を
        // 避ける）。ここで `stdout_handle`/`stderr_handle` を `join` してはならない
        // （回帰テスト `run_probe_once_returns_probe_timeout_when_process_hangs_without_output`
        // が実測で確認した通り、直接の子プロセスが自身の子（孫プロセス）へパイプの
        // 書き込み端 fd を継承させていた場合、直接の子を kill してもパイプは
        // クローズされず、孫プロセスが生き続ける限り読み取りスレッドの `read()` は
        // EOF を受け取れず永久に返らない。Rust 標準ライブラリに `JoinHandle` の
        // タイムアウト付き join は存在しないため、`join` を呼ぶと `ProbeTimeout` の
        // 意味＝「上限時間で確実に制御を返す」が破綻する）。読み取りスレッドは
        // join せず切り離し、バックグラウンドで動作したまま関数を抜ける
        // （最終的にプロセス終了時に回収される。孫プロセスの残存は probe 自体の
        // 実装バグ・GPU ドライバ側の問題であり、本ハーネス側では検知だけを保証する）。
        let _ = child.kill();
        let _ = child.wait();
        drop(stdout_handle);
        drop(stderr_handle);
        return Err(StartupError::ProbeTimeout(start.elapsed()));
    }

    stdout_handle
        .join()
        .map_err(|_| StartupError::Io("probe stdout 読み取りスレッドが panic した".to_string()))?;
    stderr_handle
        .join()
        .map_err(|_| StartupError::Io("probe stderr 読み取りスレッドが panic した".to_string()))?;

    let (stdout_buf, stdout_exceeded) = stdout_result
        .ok_or_else(|| {
            StartupError::Io("probe stdout 読み取りスレッドが結果を送信せず終了した".to_string())
        })?
        .map_err(|e| StartupError::Io(format!("probe stdout 読み取り失敗: {e}")))?;
    let (stderr_buf, stderr_exceeded) = stderr_result
        .ok_or_else(|| {
            StartupError::Io("probe stderr 読み取りスレッドが結果を送信せず終了した".to_string())
        })?
        .map_err(|e| StartupError::Io(format!("probe stderr 読み取り失敗: {e}")))?;

    if stdout_exceeded || stderr_exceeded {
        // 上限超過時はループ内で既に kill 済みだが、まだ確定していない経路
        // （例: 送信直後の panic で killed_for_overflow が立たなかった場合）に
        // 備えて冪等な保険として再度試み、ゾンビプロセス化を避けるため wait は
        // 必ず試みる（kill/wait 自体の失敗はすでに異常系のため握りつぶす）。
        let _ = child.kill();
        let _ = child.wait();
        let overflowed_len = if stdout_exceeded {
            stdout_buf.len()
        } else {
            stderr_buf.len()
        };
        return Err(StartupError::ProbeOutputTooLarge(overflowed_len));
    }

    let status = child
        .wait()
        .map_err(|e| StartupError::Io(format!("probe 待機失敗（{:?}）: {e}", config.probe_path)))?;
    let wall_secs = start.elapsed().as_secs_f64();

    if !status.success() {
        return Err(StartupError::ProbeExitFailure {
            status: status.to_string(),
            stderr: String::from_utf8_lossy(&stderr_buf).into_owned(),
        });
    }

    let stdout_text = String::from_utf8_lossy(&stdout_buf);
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

    /// PR #360 codex-review 指摘 P1 の回帰テスト: `read_capped` は上限以下の入力を
    /// 全量そのまま読み取り、超過フラグを立てない。
    #[test]
    fn read_capped_returns_full_data_when_within_limit() {
        let data = vec![b'a'; 100];
        let (buf, exceeded) = read_capped(data.as_slice(), 1000).unwrap();
        assert_eq!(buf, data);
        assert!(!exceeded);
    }

    /// PR #360 codex-review 指摘 P1 の回帰テスト: 上限を超える入力に対しては、
    /// 入力全量（無制限）を読み切る前に打ち切り、超過フラグを立てる
    /// （`run_probe_once` はこのフラグを見て子プロセスを kill・reap する）。
    /// 読み取り量が「上限 + 1 チャンク（64 KiB）」程度に収まることも検証し、
    /// 全量バッファする旧実装（`Command::output`）との違いを固定する。
    #[test]
    fn read_capped_stops_shortly_after_limit_is_exceeded() {
        let limit = 10;
        let data = vec![b'x'; 10 * 1024 * 1024]; // 10 MiB。旧実装ならこの量が丸ごとバッファされていた。
        let (buf, exceeded) = read_capped(data.as_slice(), limit).unwrap();
        assert!(exceeded);
        assert!(buf.len() > limit);
        assert!(
            buf.len() <= limit + 64 * 1024,
            "read_capped は上限 + 1 チャンク程度で打ち切るべき（実際: {} バイト）",
            buf.len()
        );
    }

    /// probe が上限を超える標準出力を生成した場合、[`run_probe_once`] が
    /// `StartupError::ProbeOutputTooLarge` を返し、子プロセスを kill・reap して
    /// 全量読み切る前に処理を終えることを実機（`/bin/yes` 相当）で検証する。
    /// `probe_path` 差し替え・実装バグによる無限出力を想定した回帰テスト
    /// （PR #360 codex-review 指摘 P1）。
    #[test]
    fn run_probe_once_rejects_probe_that_emits_unbounded_stdout() {
        // `yes` は POSIX 環境に共通して存在し、"y\n" を無限に出力し続けるため、
        // 上限超過検知後に確実に kill しないとテストがハングする実測用の題材として使う。
        let probe_path = PathBuf::from("/usr/bin/yes");
        if !probe_path.exists() {
            eprintln!("skip: /usr/bin/yes が存在しない環境のためスキップ");
            return;
        }
        let config = StartupConfig::new(StartupBackend::Cpu, 1, probe_path).unwrap();
        let result = run_probe_once(&config, None);
        assert!(
            matches!(result, Err(StartupError::ProbeOutputTooLarge(_))),
            "unbounded stdout は ProbeOutputTooLarge で拒否されるべき: {result:?}"
        );
    }

    /// probe が生存したまま無出力（stdout／stderr いずれも EOF に至らない）になった場合、
    /// [`run_probe_once`] が無期限にハングせず [`StartupError::ProbeTimeout`] を返し、
    /// 子プロセスを kill・reap して速やかに制御を返すことを検証する（PR #360 codex-review
    /// 指摘 P1 の回帰テスト）。GPU ドライバ・カーネル停止による probe ハングを、
    /// 引数を無視して長時間 `sleep` するシェルスクリプトで模擬する。
    #[test]
    fn run_probe_once_returns_probe_timeout_when_process_hangs_without_output() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let script_path = std::env::temp_dir().join(format!(
            "rust-ai-library-startup-timeout-test-{}-{nanos}.sh",
            std::process::id()
        ));
        // 引数（バックエンド名）を一切参照せず、無出力のまま長時間生存し続ける probe を
        // 模擬する（`run_probe_once` は probe に対し `config.backend.as_str()` を
        // 単一引数として渡すが、本スクリプトはそれを無視する）。
        std::fs::write(&script_path, "#!/bin/sh\nsleep 300\n")
            .expect("テスト用スクリプトの書き込みに失敗");
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&script_path)
                .expect("テスト用スクリプトの metadata 取得に失敗")
                .permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&script_path, perms)
                .expect("テスト用スクリプトへの実行権限付与に失敗");
        }

        let config = StartupConfig::new(StartupBackend::Cpu, 1, script_path.clone()).unwrap();
        let started = Instant::now();
        let result = run_probe_once_with_timeout(&config, None, Duration::from_millis(200));
        let elapsed = started.elapsed();

        let _ = std::fs::remove_file(&script_path);

        assert!(
            matches!(result, Err(StartupError::ProbeTimeout(_))),
            "無出力のままハングする probe は ProbeTimeout で拒否されるべき: {result:?}"
        );
        assert!(
            elapsed < Duration::from_secs(10),
            "タイムアウト検知後は速やかに制御が返るべき（実際: {elapsed:?}）"
        );
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
    fn startup_config_rejects_trials_above_max() {
        // PR #360 codex-review P1: 過大な trials（例: u64::MAX 相当）を
        // Vec::with_capacity に到達させる前に拒否する回帰テスト。
        let err =
            StartupConfig::new(StartupBackend::Cpu, MAX_STARTUP_TRIALS + 1, "probe").unwrap_err();
        assert!(matches!(err, StartupError::InvalidArgument(_)));
        assert!(StartupConfig::new(StartupBackend::Cpu, usize::MAX, "probe").is_err());
    }

    #[test]
    fn startup_config_accepts_trials_at_max() {
        assert!(StartupConfig::new(StartupBackend::Cpu, MAX_STARTUP_TRIALS, "probe").is_ok());
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
    fn startup_report_from_json_rejects_corrupted_sample_device_init_secs() {
        // #170 レビュー指摘の再現ケース: samples[0].device_init_secs が
        // first_kernel_secs を上回る負値でも、quartile（分位点）側の検証だけでは
        // すり抜けていた（validate() が per-sample の device_init_secs /
        // first_kernel_secs を検証していなかったため）。
        let bad_json = r#"{"schema_version":"1","backend":"cpu","phase":"warm","trials":1,
            "wall_secs":{"median":0.1,"q1":0.1,"q3":0.1},
            "device_init_secs":{"median":0.01,"q1":0.01,"q3":0.01},
            "first_kernel_secs":{"median":0.05,"q1":0.05,"q3":0.05},
            "samples":[{"wall_secs":0.1,"device_init_secs":-999.0,"first_kernel_secs":0.05}]}"#;
        let err = StartupReport::from_json(bad_json).unwrap_err();
        assert!(matches!(err, StartupError::ProtocolViolation(_)));
    }

    #[test]
    fn startup_report_from_json_rejects_unknown_backend() {
        // #170 Codex レビュー指摘の再現ケース: backend が許可リスト（cpu/cuda/metal）外の
        // 自由文字列でも、旧実装は許可リスト検証をしていなかったため受理してしまっていた。
        let bad_json = format!(
            r#"{{"schema_version":"{STARTUP_SCHEMA_VERSION}","backend":"bogus","phase":"warm","trials":1,
            "wall_secs":{{"median":0.1,"q1":0.1,"q3":0.1}},
            "device_init_secs":{{"median":0.01,"q1":0.01,"q3":0.01}},
            "first_kernel_secs":{{"median":0.05,"q1":0.05,"q3":0.05}},
            "samples":[{{"wall_secs":0.1,"device_init_secs":0.01,"first_kernel_secs":0.05}}]}}"#
        );
        let err = StartupReport::from_json(&bad_json).unwrap_err();
        assert!(matches!(err, StartupError::ProtocolViolation(_)));
    }

    #[test]
    fn startup_report_from_json_rejects_negative_quartile() {
        // #170 Codex レビュー指摘の再現ケース: 分位点が負値でも
        // 「q1 <= median <= q3」の順序関係だけでは検出できなかった。
        let bad_json = format!(
            r#"{{"schema_version":"{STARTUP_SCHEMA_VERSION}","backend":"cpu","phase":"warm","trials":1,
            "wall_secs":{{"median":-0.1,"q1":-0.2,"q3":-0.05}},
            "device_init_secs":{{"median":0.01,"q1":0.01,"q3":0.01}},
            "first_kernel_secs":{{"median":0.05,"q1":0.05,"q3":0.05}},
            "samples":[{{"wall_secs":0.1,"device_init_secs":0.01,"first_kernel_secs":0.05}}]}}"#
        );
        let err = StartupReport::from_json(&bad_json).unwrap_err();
        assert!(matches!(err, StartupError::ProtocolViolation(_)));
    }

    #[test]
    fn startup_report_from_json_rejects_tampered_aggregate_mismatch() {
        // #170 Codex レビュー指摘の再現ケース: 正常な samples に対して恣意的な
        // 集計値（分位点）を併記した JSON は、samples からの再計算比較なしには検出できなかった。
        let bad_json = format!(
            r#"{{"schema_version":"{STARTUP_SCHEMA_VERSION}","backend":"cpu","phase":"warm","trials":1,
            "wall_secs":{{"median":0.001,"q1":0.001,"q3":0.001}},
            "device_init_secs":{{"median":0.01,"q1":0.01,"q3":0.01}},
            "first_kernel_secs":{{"median":0.05,"q1":0.05,"q3":0.05}},
            "samples":[{{"wall_secs":0.1,"device_init_secs":0.01,"first_kernel_secs":0.05}}]}}"#
        );
        let err = StartupReport::from_json(&bad_json).unwrap_err();
        assert!(matches!(err, StartupError::ProtocolViolation(_)));
    }

    #[test]
    fn startup_report_from_json_rejects_wall_secs_below_first_kernel_secs() {
        // #170 Codex レビュー指摘の再現ケース: 子プロセス全体の外部計測（wall_secs）が
        // 内部計測（first_kernel_secs）を下回る契約違反は、device_init_secs との
        // 比較だけでは検出できなかった。
        let bad_json = format!(
            r#"{{"schema_version":"{STARTUP_SCHEMA_VERSION}","backend":"cpu","phase":"warm","trials":1,
            "wall_secs":{{"median":0.02,"q1":0.02,"q3":0.02}},
            "device_init_secs":{{"median":0.01,"q1":0.01,"q3":0.01}},
            "first_kernel_secs":{{"median":0.05,"q1":0.05,"q3":0.05}},
            "samples":[{{"wall_secs":0.02,"device_init_secs":0.01,"first_kernel_secs":0.05}}]}}"#
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
