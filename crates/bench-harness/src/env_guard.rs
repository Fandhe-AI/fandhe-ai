//! ベンチ実行前の環境ガード（イシュー #1264・親 #1263。トラッキング #1242・
//! Phase 親 #1246）。
//!
//! ## 背景・目的
//!
//! #1186／#1187（Metal 転置ルーティング A/B）は、同一マシンで並走する他
//! セッションの負荷（`uptime` 実測 load average 3.4〜8.6・実行中の再上昇を
//! `docs/perf/logs/metal-gemm-transpose-route-ab-1187/uptime_during_run4.txt`
//! で確認）により `ab::run_stability`（[`crate::ab::STABILITY_SPREAD_GATE`]）の
//! フェーズ 1 安定性ゲートが 4 試行とも不成立・`verdict=undetermined` のまま
//! 終わった。環境状態の確認は手動記録（`docs/perf/logs/
//! metal-gemm-transpose-route-ab-1187/env_info.txt`）に依存していた。
//!
//! 本モジュールは、その環境確認を機械化する **API 層**（設定型・取得・
//! 判定・結果型）のみを提供する。ガード不成立時のバックオフ再試行・
//! `env_info` への自動記録・`examples/gemm_transpose_route_ab_bench.rs` への
//! 結線は兄弟イシュー #1265 のスコープであり、本モジュールでは行わない。
//!
//! ## 設計方針
//!
//! - **取得（I/O）と判定（純粋関数）を分離する**（[`EnvSample::collect`] と
//!   [`EnvGuardConfig::evaluate`]）。`crates/backend-cpu/src/thread_limit.rs`
//!   （`detect_big_cores` を cfg 分離し、判定ロジック〈`big_cores_from_sysctl`〉
//!   を純粋関数化する設計）と同型。parse 関数群はプラットフォーム cfg に
//!   依存しないため、Linux CI 上で macOS 側の固定文字列 fixture を使った
//!   ユニットテストが書ける。
//! - **既定閾値を埋め込まない**。[`EnvGuardConfig::new`] は呼び出し側が明示した
//!   `max_load_avg_1min` を検証するのみで、組み込みの既定値・`Default` 実装は
//!   持たない（ガード閾値はテスト許容誤差・ガードレール閾値と同様の性質を持つ
//!   ため `.claude/rules/security.md`「自己修復ループ固有のガードレール」に
//!   倣いユーザー承認事項として扱う。#1265 向けの提案値は
//!   `docs/perf/metal-bench-noise-protocol.md` に「提案・未承認」として記す）。
//! - **取得不能は「未判定」（[`GuardVerdict::Undetermined`]）とし、fail-closed の
//!   ブロック要因にしない**。イシュー #1264 本文の要件（GPU プロセス検出が
//!   「取得不能時は未判定として記録し fail-closed にしない」）を、load average・
//!   GPU 検出の両方へ一貫して適用する。ブロック（[`EnvGuardReport::is_blocking`]
//!   が `true`）になるのは明確な `Fail` のみ。
//! - **GPU プロセス検出は「他プロセスの存在」では判定しない**。本機
//!   （M4 Max）で `ioreg -r -c IOAccelerator -l` を実測したところ、
//!   `WindowServer`・`runningboardd`・`Safari` 等の常駐プロセスだけで
//!   `AGXDeviceUserClient` が約 90 件存在し、「自プロセス以外が 1 つでも
//!   あれば Fail」は常時 Fail になってしまう。そのため、呼び出し側が指定する
//!   **プロセス名の部分一致 watchlist**（空なら記録のみで `Pass`）と、任意の
//!   GPU 使用率上限（`Device Utilization %`）で判定する。
//!
//! ## セキュリティ（`.claude/rules/security.md` A03 対応）
//!
//! 外部コマンドは `run_fixed_command`（非公開ヘルパ）経由でのみ起動する。絶対パス・固定
//! 引数・`env_clear()`・標準入力 `Stdio::null()` で直接 exec し、シェル
//! （`sh -c` 等）を経由しない。取得したプロセス名・watchlist 文字列は非信頼
//! データとして保存・部分一致比較にのみ用い、コマンド引数・パス・フォーマット
//! 文字列へ再展開しない。`sudo`・`powermetrics` 等の特権コマンドは使わない。
//! ホスト名・ユーザー名・機体識別子は取得・記録しない
//! （`docs/real-hardware-verification-env.md` 方針）。

use crate::stats::BenchError;
use serde::Serialize;
#[cfg(any(target_os = "macos", target_os = "linux", test))]
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

/// 個別項目・全体の判定結果。
///
/// `Undetermined` は「取得できなかった」ことを表し、[`EnvGuardReport::is_blocking`]
/// の判定ではブロック要因にしない（モジュール doc「設計方針」参照）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardVerdict {
    Pass,
    Fail,
    Undetermined,
}

/// load average（1/5/15 分）の組。
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct LoadAvg {
    pub one: f64,
    pub five: f64,
    pub fifteen: f64,
}

impl LoadAvg {
    /// 全フィールドが有限かつ非負であるかを検査する。
    ///
    /// [`EnvSample`]・[`LoadAvg`] は公開フィールドを持つため、公開評価 API
    /// （[`EnvGuardConfig::evaluate`]）の呼び出し側が `parse_proc_loadavg` 等の
    /// 検証を経ない任意の値（`NaN`・負値等）を直接構築して渡せてしまう。
    /// `evaluate_load_avg` はこの検査を通過した値のみを判定対象とし、
    /// 通過しない値は `None`（未判定）と同じ扱いにする（review #1456 指摘対応）。
    fn is_valid(&self) -> bool {
        self.one.is_finite()
            && self.five.is_finite()
            && self.fifteen.is_finite()
            && self.one >= 0.0
            && self.five >= 0.0
            && self.fifteen >= 0.0
    }
}

/// GPU（Metal）を使用中と観測されたプロセス。
///
/// `name` は `ioreg` 出力由来の非信頼データであり、記録・部分一致比較にのみ
/// 使う（コマンド・パスへ再展開しない）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GpuProcess {
    pub pid: u32,
    pub name: String,
}

/// GPU プロセス取得の結果。取得不能な場合は理由付きで `Unavailable` を返す
/// （fail-closed にしない設計。モジュール doc 参照）。
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum GpuSample {
    Unavailable {
        reason: String,
    },
    Available {
        processes: Vec<GpuProcess>,
        device_utilization_percent: Option<u8>,
    },
}

/// [`EnvSample::collect`] が返す生の実測値。全フィールドが `Option`（または
/// 取得不能を明示する [`GpuSample::Unavailable`]）であり、`collect` 自体は
/// エラーを返さない（panic 経路なし・`unwrap`／`expect` 不使用）。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EnvSample {
    /// 取得時刻（UNIX epoch 秒）。`SystemTime` が `UNIX_EPOCH` より前を返す
    /// ことは通常ないが、念のため `Option` にして panic 経路を作らない。
    pub collected_at_unix_secs: Option<u64>,
    pub load_avg: Option<LoadAvg>,
    pub uptime_secs: Option<u64>,
    /// `/usr/bin/uptime` の生出力（記録用。判定には使わない）。
    pub raw_uptime_line: Option<String>,
    pub gpu: GpuSample,
}

impl EnvSample {
    /// 現在の環境を実測する（I/O）。プラットフォームごとに cfg 分離した
    /// 取得関数（`load_avg_now`・`uptime_secs_now`・`gpu_sample_now`。いずれも非公開ヘルパ）を
    /// 呼び出すのみで、判定ロジックは持たない。
    pub fn collect() -> Self {
        let collected_at_unix_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .map(|d| d.as_secs());
        let raw_uptime_line = read_uptime_line();
        let load_avg = load_avg_now().or_else(|| {
            raw_uptime_line
                .as_deref()
                .and_then(parse_uptime_line_load_avg)
        });
        EnvSample {
            collected_at_unix_secs,
            load_avg,
            uptime_secs: uptime_secs_now(),
            raw_uptime_line,
            gpu: gpu_sample_now(),
        }
    }
}

/// load average（1 分）の判定結果。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LoadAvgCheck {
    pub observed: Option<LoadAvg>,
    pub max_1min: f64,
    pub verdict: GuardVerdict,
}

/// GPU プロセス検出の判定結果。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GpuProcessCheck {
    /// 観測された全プロセス（watchlist 非一致含む）。`Unavailable` 時は空。
    pub processes: Vec<GpuProcess>,
    /// watchlist に部分一致したプロセス（Fail の根拠）。
    pub flagged: Vec<GpuProcess>,
    pub device_utilization_percent: Option<u8>,
    pub verdict: GuardVerdict,
    /// 未判定・記録のみ等の補足説明。
    pub note: Option<String>,
}

/// uptime の記録（判定なし。記録のみ）。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UptimeRecord {
    pub uptime_secs: Option<u64>,
    pub raw_line: Option<String>,
    pub collected_at_unix_secs: Option<u64>,
}

/// [`EnvGuardConfig::evaluate`] の戻り値。項目ごとの実測値・合否と、
/// 全体判定 [`EnvGuardReport::overall`] を持つ。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EnvGuardReport {
    pub load_avg: LoadAvgCheck,
    pub gpu: GpuProcessCheck,
    pub uptime: UptimeRecord,
    pub overall: GuardVerdict,
}

impl EnvGuardReport {
    /// `overall == GuardVerdict::Fail` のときのみ `true`。呼び出し側
    /// （#1265 のバックオフ再試行）は本メソッドの結果でのみ計測を中断する
    /// 想定（`Undetermined` では中断しない。モジュール doc「設計方針」参照）。
    pub fn is_blocking(&self) -> bool {
        self.overall == GuardVerdict::Fail
    }
}

/// 呼び出し側が明示するガード条件。既定値・`Default` 実装は持たない
/// （モジュール doc「設計方針」参照）。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EnvGuardConfig {
    max_load_avg_1min: f64,
    gpu_process_watchlist: Vec<String>,
    max_gpu_device_utilization_percent: Option<u8>,
}

impl EnvGuardConfig {
    /// `max_load_avg_1min` の検証付きコンストラクタ。有限かつ正でなければ
    /// `BenchError::ProtocolViolation` を返す（ガード閾値の誤設定を
    /// 早期に fail-closed で弾く）。
    pub fn new(max_load_avg_1min: f64) -> Result<Self, BenchError> {
        if !max_load_avg_1min.is_finite() || max_load_avg_1min <= 0.0 {
            return Err(BenchError::ProtocolViolation(format!(
                "max_load_avg_1min は有限かつ正である必要がある（実際: {max_load_avg_1min}）"
            )));
        }
        Ok(EnvGuardConfig {
            max_load_avg_1min,
            gpu_process_watchlist: Vec::new(),
            max_gpu_device_utilization_percent: None,
        })
    }

    /// GPU プロセス watchlist（名前部分一致）を設定する。空のままなら
    /// GPU プロセス項目は記録のみで `Pass` になる。
    #[must_use]
    pub fn with_gpu_process_watchlist(mut self, names: Vec<String>) -> Self {
        self.gpu_process_watchlist = names;
        self
    }

    /// GPU 使用率（`Device Utilization %`）の上限を設定する。`percent` は
    /// 0〜100 の範囲でなければ `BenchError::ProtocolViolation` を返す。
    pub fn with_max_gpu_device_utilization_percent(
        mut self,
        percent: u8,
    ) -> Result<Self, BenchError> {
        if percent > 100 {
            return Err(BenchError::ProtocolViolation(format!(
                "max_gpu_device_utilization_percent は 0〜100 である必要がある（実際: {percent}）"
            )));
        }
        self.max_gpu_device_utilization_percent = Some(percent);
        Ok(self)
    }

    /// 設定済みの load average（1 分）上限を返す。
    pub fn max_load_avg_1min(&self) -> f64 {
        self.max_load_avg_1min
    }

    /// 設定済みの GPU プロセス watchlist（名前部分一致）を返す。
    pub fn gpu_process_watchlist(&self) -> &[String] {
        &self.gpu_process_watchlist
    }

    /// 設定済みの GPU 使用率上限（未設定なら `None`）を返す。
    pub fn max_gpu_device_utilization_percent(&self) -> Option<u8> {
        self.max_gpu_device_utilization_percent
    }

    /// 実測値 `sample` に対しガード条件を適用する純粋関数（I/O を行わない）。
    /// 判定規則はモジュール doc「設計方針」を参照。
    pub fn evaluate(&self, sample: &EnvSample) -> EnvGuardReport {
        let load_avg = self.evaluate_load_avg(sample);
        let gpu = self.evaluate_gpu(sample);
        let uptime = UptimeRecord {
            uptime_secs: sample.uptime_secs,
            raw_line: sample.raw_uptime_line.clone(),
            collected_at_unix_secs: sample.collected_at_unix_secs,
        };
        let overall = combine_verdicts(load_avg.verdict, gpu.verdict);
        EnvGuardReport {
            load_avg,
            gpu,
            uptime,
            overall,
        }
    }

    /// 現在の環境を実測してから評価する（[`EnvSample::collect`] +
    /// [`Self::evaluate`] の合成）。
    pub fn check(&self) -> EnvGuardReport {
        self.evaluate(&EnvSample::collect())
    }

    fn evaluate_load_avg(&self, sample: &EnvSample) -> LoadAvgCheck {
        // 公開評価 API 経由で妥当性検証を経ていない値（NaN・負値等）が
        // 渡される可能性があるため、判定前に `LoadAvg::is_valid` で検証する
        // （review #1456 指摘対応。`observed` フィールド自体は記録用に
        // 生値をそのまま残す）。
        let validated = sample.load_avg.filter(LoadAvg::is_valid);
        let verdict = match validated {
            Some(observed) if observed.one > self.max_load_avg_1min => GuardVerdict::Fail,
            Some(_) => GuardVerdict::Pass,
            None => GuardVerdict::Undetermined,
        };
        LoadAvgCheck {
            observed: sample.load_avg,
            max_1min: self.max_load_avg_1min,
            verdict,
        }
    }

    fn evaluate_gpu(&self, sample: &EnvSample) -> GpuProcessCheck {
        match &sample.gpu {
            GpuSample::Unavailable { reason } => GpuProcessCheck {
                processes: Vec::new(),
                flagged: Vec::new(),
                device_utilization_percent: None,
                verdict: GuardVerdict::Undetermined,
                note: Some(reason.clone()),
            },
            GpuSample::Available {
                processes,
                device_utilization_percent,
            } => {
                let flagged: Vec<GpuProcess> = processes
                    .iter()
                    .filter(|p| {
                        self.gpu_process_watchlist
                            .iter()
                            .any(|watch| !watch.is_empty() && p.name.contains(watch.as_str()))
                    })
                    .cloned()
                    .collect();
                // 公開評価 API（`GpuSample::Available` は直接構築可能）経由で
                // `EnvSample::collect` の 0〜100 変換（100 超は `None`）を経て
                // いない値（例: `Some(255)`）が渡されうるため、判定前に範囲
                // 検証する（review #1456 指摘対応。`LoadAvg::is_valid` と同型。
                // `device_utilization_percent` フィールド自体は記録用に生値
                // をそのまま残す）。
                let validated_utilization =
                    device_utilization_percent.filter(|percent| *percent <= 100);
                let utilization_exceeded = match (
                    validated_utilization,
                    self.max_gpu_device_utilization_percent,
                ) {
                    (Some(observed), Some(max)) => observed > max,
                    _ => false,
                };
                // 使用率上限を設定したのに検証済み `Device Utilization %` を
                // 取得できなかった場合は、判定不能を `Pass` に落とさず
                // `Undetermined` にする（review #1456 指摘対応。モジュール
                // doc「設計方針」の「取得不能は未判定」を使用率上限にも
                // 一貫して適用する。範囲外値〈例 255〉も未取得と同じ扱い）。
                let utilization_undetermined = self.max_gpu_device_utilization_percent.is_some()
                    && validated_utilization.is_none();
                let verdict = if !flagged.is_empty() || utilization_exceeded {
                    GuardVerdict::Fail
                } else if utilization_undetermined {
                    GuardVerdict::Undetermined
                } else {
                    GuardVerdict::Pass
                };
                // note は「未判定」の補足説明専用のため、watchlist 一致等で
                // 既に `Fail` が確定している場合は付与しない（Cursor Bugbot
                // 指摘対応。verdict が `Fail` のまま note が「未判定」を主張
                // する矛盾を防ぐ）。
                let note = if self.gpu_process_watchlist.is_empty()
                    && self.max_gpu_device_utilization_percent.is_none()
                {
                    Some("watchlist・使用率上限とも未設定のため記録のみ".to_string())
                } else if verdict == GuardVerdict::Undetermined && utilization_undetermined {
                    Some(
                        "使用率上限を設定したが Device Utilization % を取得できなかったため未判定\
                         （範囲外値〈0〜100 外〉も含む）"
                            .to_string(),
                    )
                } else {
                    None
                };
                GpuProcessCheck {
                    processes: processes.clone(),
                    flagged,
                    device_utilization_percent: *device_utilization_percent,
                    verdict,
                    note,
                }
            }
        }
    }
}

/// 項目別判定から全体判定を導く。優先順位は `Fail` > `Undetermined` > `Pass`
/// （明確な悪化のみをブロック要因にし、未判定は記録に留める設計）。
fn combine_verdicts(a: GuardVerdict, b: GuardVerdict) -> GuardVerdict {
    use GuardVerdict::{Fail, Pass, Undetermined};
    match (a, b) {
        (Fail, _) | (_, Fail) => Fail,
        (Undetermined, _) | (_, Undetermined) => Undetermined,
        (Pass, Pass) => Pass,
    }
}

// ---------------------------------------------------------------------
// parse 関数群（cfg 非依存。ユニットテスト対象）
// ---------------------------------------------------------------------

/// Linux `/proc/loadavg` 形式（`"0.52 0.58 0.59 1/1234 5678\n"`）を parse する。
/// 先頭 3 フィールドが有限の非負数でなければ `None`。
fn parse_proc_loadavg(text: &str) -> Option<LoadAvg> {
    let mut fields = text.split_whitespace();
    let one: f64 = fields.next()?.parse().ok()?;
    let five: f64 = fields.next()?.parse().ok()?;
    let fifteen: f64 = fields.next()?.parse().ok()?;
    if !(one.is_finite() && five.is_finite() && fifteen.is_finite())
        || one < 0.0
        || five < 0.0
        || fifteen < 0.0
    {
        return None;
    }
    Some(LoadAvg { one, five, fifteen })
}

/// macOS `sysctl -n vm.loadavg` 形式（`"{ 2.72 4.31 4.66 }\n"`）を parse する。
#[cfg(any(target_os = "macos", test))]
fn parse_sysctl_vm_loadavg(text: &str) -> Option<LoadAvg> {
    let trimmed = text.trim();
    let inner = trimmed.strip_prefix('{')?.strip_suffix('}')?;
    parse_proc_loadavg(inner)
}

/// `uptime` コマンド出力中の load average 部分を parse する。macOS
/// （`load averages: 2.72 4.31 4.66`）・Linux（`load average: 0.52, 0.58, 0.59`）
/// 双方の形式に対応する（カンマ区切り・空白区切りいずれも許容）。
fn parse_uptime_line_load_avg(line: &str) -> Option<LoadAvg> {
    let marker = if let Some(idx) = line.find("load averages:") {
        idx + "load averages:".len()
    } else {
        line.find("load average:")? + "load average:".len()
    };
    let tail = &line[marker..];
    let normalized = tail.replace(',', " ");
    parse_proc_loadavg(&normalized)
}

/// Linux `/proc/uptime` 形式（`"12345.67 98765.43\n"`）の先頭フィールドを
/// 秒数として parse する。本番到達は Linux の [`uptime_secs_now`] からのみ
/// のため、他プラットフォームの非テストビルドでは未使用警告が出る
/// （`cfg` でテスト・Linux 限定にする）。
#[cfg(any(target_os = "linux", test))]
fn parse_proc_uptime(text: &str) -> Option<u64> {
    let first = text.split_whitespace().next()?;
    let secs: f64 = first.parse().ok()?;
    if !secs.is_finite() || secs < 0.0 {
        return None;
    }
    Some(secs as u64)
}

/// macOS `sysctl -n kern.boottime` 形式
/// （`"{ sec = 1787111274, usec = 868910 } Wed Sep  3 ...\n"`）から
/// `sec = <N>` を抽出する。
#[cfg(any(target_os = "macos", test))]
fn parse_kern_boottime_sec(text: &str) -> Option<u64> {
    let idx = text.find("sec =")?;
    let after = &text[idx + "sec =".len()..];
    let digits: String = after
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

/// `ioreg -r -c IOAccelerator -l` 出力から `"IOUserClientCreator" = "pid N, name"`
/// 行を抽出し、`self_pid` を除外したうえで pid 単位に重複除去する。
///
/// 出力は非信頼データであり、抽出した `name` はコマンド・パスへ再展開せず
/// 記録・部分一致比較にのみ用いる（モジュール doc「セキュリティ」参照）。
#[cfg(any(target_os = "macos", test))]
fn parse_ioreg_user_clients(text: &str, self_pid: u32) -> Vec<GpuProcess> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();
    for line in text.lines() {
        let Some(value_start) = line.find("\"IOUserClientCreator\"") else {
            continue;
        };
        let rest = &line[value_start..];
        // 値は `= "pid 1234, WindowServer"` の形式。`=` の後ろの
        // 引用符で囲まれた部分だけを取り出す。
        let Some(eq_idx) = rest.find('=') else {
            continue;
        };
        let after_eq = &rest[eq_idx + 1..];
        let Some(first_quote) = after_eq.find('"') else {
            continue;
        };
        let after_first_quote = &after_eq[first_quote + 1..];
        let Some(second_quote) = after_first_quote.find('"') else {
            continue;
        };
        let inner = &after_first_quote[..second_quote];
        // `inner` は "pid <N>, <name>" 形式。
        let Some(inner) = inner.strip_prefix("pid ") else {
            continue;
        };
        let Some((pid_str, name)) = inner.split_once(',') else {
            continue;
        };
        let Ok(pid) = pid_str.trim().parse::<u32>() else {
            continue;
        };
        if pid == self_pid {
            continue;
        }
        if !seen.insert(pid) {
            continue;
        }
        result.push(GpuProcess {
            pid,
            name: name.trim().to_string(),
        });
    }
    result
}

/// `ioreg -r -c IOAccelerator -l` 出力中の `"Device Utilization %"=N` を
/// parse する（0〜100 に収まらない値・parse 不能は `None`）。
#[cfg(any(target_os = "macos", test))]
fn parse_ioreg_device_utilization(text: &str) -> Option<u8> {
    let idx = text.find("\"Device Utilization %\"")?;
    let rest = &text[idx + "\"Device Utilization %\"".len()..];
    let eq_idx = rest.find('=')?;
    let after_eq = rest[eq_idx + 1..].trim_start();
    let digits: String = after_eq
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        return None;
    }
    let value: u32 = digits.parse().ok()?;
    if value > 100 {
        return None;
    }
    u8::try_from(value).ok()
}

// ---------------------------------------------------------------------
// 取得層（I/O。cfg 分離）
// ---------------------------------------------------------------------

/// 固定バイナリを絶対パス・固定引数・`env_clear()`・標準入力 `Stdio::null()`
/// で直接起動し、成功時のみ stdout（UTF-8 lossy 変換）を返す
/// （`.claude/rules/security.md` A03。シェル経由・引数連結を行わない）。
#[cfg(any(target_os = "macos", target_os = "linux", test))]
fn run_fixed_command(path: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(path)
        .env_clear()
        .args(args)
        .stdin(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn read_uptime_line() -> Option<String> {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        run_fixed_command("/usr/bin/uptime", &[]).map(|s| s.trim().to_string())
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
fn load_avg_now() -> Option<LoadAvg> {
    let text = std::fs::read_to_string("/proc/loadavg").ok()?;
    parse_proc_loadavg(&text)
}

#[cfg(target_os = "macos")]
fn load_avg_now() -> Option<LoadAvg> {
    let text = run_fixed_command("/usr/sbin/sysctl", &["-n", "vm.loadavg"])?;
    parse_sysctl_vm_loadavg(&text)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn load_avg_now() -> Option<LoadAvg> {
    None
}

#[cfg(target_os = "linux")]
fn uptime_secs_now() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/uptime").ok()?;
    parse_proc_uptime(&text)
}

#[cfg(target_os = "macos")]
fn uptime_secs_now() -> Option<u64> {
    let text = run_fixed_command("/usr/sbin/sysctl", &["-n", "kern.boottime"])?;
    let boot_sec = parse_kern_boottime_sec(&text)?;
    let now_sec = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs();
    now_sec.checked_sub(boot_sec)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn uptime_secs_now() -> Option<u64> {
    None
}

/// GPU プロセス検出（macOS 限定。`ioreg -r -c IOAccelerator -l` を
/// サブツリー限定〈`-r -c IOAccelerator`〉で起動し出力を有界に保つ。
/// 本機実測で約 78 KB。`-l` 全体ダンプは使わない）。
#[cfg(target_os = "macos")]
fn gpu_sample_now() -> GpuSample {
    let Some(text) = run_fixed_command("/usr/sbin/ioreg", &["-r", "-c", "IOAccelerator", "-l"])
    else {
        return GpuSample::Unavailable {
            reason: "ioreg コマンドの実行に失敗した".to_string(),
        };
    };
    // `-c IOAccelerator` は対象デバイスが見つからない場合でも exit success・
    // 空 stdout を返しうる（本機実測）。これを検出データなしの `Available`
    // として扱うと watchlist 設定時に GPU 使用中でも `Pass` になりうるため、
    // 理由付き `Unavailable` として明示的に未判定にする（review #1456 指摘対応）。
    if text.trim().is_empty() {
        return GpuSample::Unavailable {
            reason: "ioreg の出力が空だった（IOAccelerator デバイスが見つからない）".to_string(),
        };
    }
    let self_pid = std::process::id();
    let processes = parse_ioreg_user_clients(&text, self_pid);
    let device_utilization_percent = parse_ioreg_device_utilization(&text);
    GpuSample::Available {
        processes,
        device_utilization_percent,
    }
}

/// Linux での GPU プロセス検出は本イシューのスコープ外（`nvidia-smi` 連携は
/// 後続候補。イシュー #1264 計画「スコープ外」節）。常に理由付きで
/// `Unavailable` を返す（fail-closed にしない設計と整合）。
#[cfg(target_os = "linux")]
fn gpu_sample_now() -> GpuSample {
    GpuSample::Unavailable {
        reason: "Linux では GPU プロセス検出が未実装（cfg 分離。イシュー #1264 スコープ外）"
            .to_string(),
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn gpu_sample_now() -> GpuSample {
    GpuSample::Unavailable {
        reason: "未対応プラットフォームのため GPU プロセス検出ができない".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- parse_proc_loadavg ---------------------------------------

    #[test]
    fn parse_proc_loadavg_valid() {
        let got = parse_proc_loadavg("0.52 0.58 0.59 1/1234 5678\n").unwrap();
        assert_eq!(got.one, 0.52);
        assert_eq!(got.five, 0.58);
        assert_eq!(got.fifteen, 0.59);
    }

    #[test]
    fn parse_proc_loadavg_empty() {
        assert!(parse_proc_loadavg("").is_none());
    }

    #[test]
    fn parse_proc_loadavg_missing_field() {
        assert!(parse_proc_loadavg("0.52 0.58").is_none());
    }

    #[test]
    fn parse_proc_loadavg_non_numeric() {
        assert!(parse_proc_loadavg("abc def ghi").is_none());
    }

    #[test]
    fn parse_proc_loadavg_negative_rejected() {
        assert!(parse_proc_loadavg("-0.1 0.5 0.5").is_none());
    }

    // --- parse_sysctl_vm_loadavg -----------------------------------

    #[test]
    fn parse_sysctl_vm_loadavg_valid() {
        let got = parse_sysctl_vm_loadavg("{ 2.72 4.31 4.66 }\n").unwrap();
        assert_eq!(got.one, 2.72);
        assert_eq!(got.five, 4.31);
        assert_eq!(got.fifteen, 4.66);
    }

    #[test]
    fn parse_sysctl_vm_loadavg_missing_braces() {
        assert!(parse_sysctl_vm_loadavg("2.72 4.31 4.66").is_none());
    }

    // --- parse_uptime_line_load_avg ---------------------------------

    #[test]
    fn parse_uptime_line_load_avg_macos_form() {
        let line = "12:34  up 5 days, 21:10, 3 users, load averages: 2.72 4.31 4.66";
        let got = parse_uptime_line_load_avg(line).unwrap();
        assert_eq!(got.one, 2.72);
        assert_eq!(got.five, 4.31);
        assert_eq!(got.fifteen, 4.66);
    }

    #[test]
    fn parse_uptime_line_load_avg_linux_form() {
        let line = " 12:34:56 up 10 days,  2:03,  1 user,  load average: 0.52, 0.58, 0.59";
        let got = parse_uptime_line_load_avg(line).unwrap();
        assert_eq!(got.one, 0.52);
        assert_eq!(got.five, 0.58);
        assert_eq!(got.fifteen, 0.59);
    }

    #[test]
    fn parse_uptime_line_load_avg_no_marker() {
        assert!(parse_uptime_line_load_avg("no load average here").is_none());
    }

    // --- parse_proc_uptime / parse_kern_boottime_sec ----------------

    #[test]
    fn parse_proc_uptime_valid() {
        assert_eq!(parse_proc_uptime("12345.67 98765.43\n"), Some(12345));
    }

    #[test]
    fn parse_proc_uptime_invalid() {
        assert!(parse_proc_uptime("").is_none());
        assert!(parse_proc_uptime("abc").is_none());
    }

    #[test]
    fn parse_kern_boottime_sec_valid() {
        let text = "{ sec = 1787111274, usec = 868910 } Wed Sep  3 12:34:56 2026\n";
        assert_eq!(parse_kern_boottime_sec(text), Some(1787111274));
    }

    #[test]
    fn parse_kern_boottime_sec_missing() {
        assert!(parse_kern_boottime_sec("no sec field here").is_none());
    }

    // --- parse_ioreg_user_clients ------------------------------------

    #[test]
    fn parse_ioreg_user_clients_dedup_and_self_exclusion() {
        let text = concat!(
            "    | |   \"IOUserClientCreator\" = \"pid 111, WindowServer\"\n",
            "    | |   \"IOUserClientCreator\" = \"pid 111, WindowServer\"\n",
            "    | |   \"IOUserClientCreator\" = \"pid 222, VTDecoderXPCServ\"\n",
            "    | |   \"IOUserClientCreator\" = \"pid 999, self_process\"\n",
        );
        let got = parse_ioreg_user_clients(text, 999);
        assert_eq!(got.len(), 2);
        assert!(got.iter().any(|p| p.pid == 111 && p.name == "WindowServer"));
        assert!(
            got.iter()
                .any(|p| p.pid == 222 && p.name == "VTDecoderXPCServ")
        );
        assert!(got.iter().all(|p| p.pid != 999));
    }

    #[test]
    fn parse_ioreg_user_clients_empty_text() {
        assert!(parse_ioreg_user_clients("", 1).is_empty());
    }

    #[test]
    fn parse_ioreg_user_clients_malformed_line_ignored() {
        let text = "    | |   \"IOUserClientCreator\" = \"not-a-pid-format\"\n";
        assert!(parse_ioreg_user_clients(text, 1).is_empty());
    }

    // --- parse_ioreg_device_utilization ------------------------------

    #[test]
    fn parse_ioreg_device_utilization_present() {
        let text = "  | |   \"Device Utilization %\"=10\n";
        assert_eq!(parse_ioreg_device_utilization(text), Some(10));
    }

    #[test]
    fn parse_ioreg_device_utilization_absent() {
        assert!(parse_ioreg_device_utilization("no such field").is_none());
    }

    #[test]
    fn parse_ioreg_device_utilization_out_of_range_is_none() {
        // ioreg の実出力は仕様上 0〜100 のみだが、doc コメントが約束する
        // 「0〜100 に収まらない値は None」を境界値（101・255）で自己検証する
        // （u8 の型範囲チェックのみに留まる実装への回帰を防ぐ。Review 指摘対応）。
        let text = "  | |   \"Device Utilization %\"=101\n";
        assert!(parse_ioreg_device_utilization(text).is_none());
        let text = "  | |   \"Device Utilization %\"=255\n";
        assert!(parse_ioreg_device_utilization(text).is_none());
    }

    // --- EnvGuardConfig::new / with_* --------------------------------

    #[test]
    fn config_new_rejects_nan() {
        assert!(EnvGuardConfig::new(f64::NAN).is_err());
    }

    #[test]
    fn config_new_rejects_infinite() {
        assert!(EnvGuardConfig::new(f64::INFINITY).is_err());
    }

    #[test]
    fn config_new_rejects_zero_and_negative() {
        assert!(EnvGuardConfig::new(0.0).is_err());
        assert!(EnvGuardConfig::new(-1.0).is_err());
    }

    #[test]
    fn config_new_accepts_positive_finite() {
        assert!(EnvGuardConfig::new(4.0).is_ok());
    }

    #[test]
    fn config_utilization_percent_rejects_over_100() {
        let cfg = EnvGuardConfig::new(4.0).unwrap();
        assert!(cfg.with_max_gpu_device_utilization_percent(101).is_err());
    }

    #[test]
    fn config_utilization_percent_accepts_boundary() {
        let cfg = EnvGuardConfig::new(4.0).unwrap();
        assert!(cfg.with_max_gpu_device_utilization_percent(100).is_ok());
    }

    // --- evaluate: load average ---------------------------------------

    fn sample_with(load_avg: Option<LoadAvg>, gpu: GpuSample) -> EnvSample {
        EnvSample {
            collected_at_unix_secs: Some(1_700_000_000),
            load_avg,
            uptime_secs: Some(100),
            raw_uptime_line: Some("fixture".to_string()),
            gpu,
        }
    }

    #[test]
    fn evaluate_load_avg_fail_when_exceeded() {
        let cfg = EnvGuardConfig::new(4.0).unwrap();
        let sample = sample_with(
            Some(LoadAvg {
                one: 5.0,
                five: 4.0,
                fifteen: 3.0,
            }),
            GpuSample::Unavailable {
                reason: "test".to_string(),
            },
        );
        let report = cfg.evaluate(&sample);
        assert_eq!(report.load_avg.verdict, GuardVerdict::Fail);
    }

    #[test]
    fn evaluate_load_avg_pass_within_limit() {
        let cfg = EnvGuardConfig::new(4.0).unwrap();
        let sample = sample_with(
            Some(LoadAvg {
                one: 2.0,
                five: 2.0,
                fifteen: 2.0,
            }),
            GpuSample::Unavailable {
                reason: "test".to_string(),
            },
        );
        let report = cfg.evaluate(&sample);
        assert_eq!(report.load_avg.verdict, GuardVerdict::Pass);
    }

    #[test]
    fn evaluate_load_avg_undetermined_when_missing() {
        let cfg = EnvGuardConfig::new(4.0).unwrap();
        let sample = sample_with(
            None,
            GpuSample::Unavailable {
                reason: "test".to_string(),
            },
        );
        let report = cfg.evaluate(&sample);
        assert_eq!(report.load_avg.verdict, GuardVerdict::Undetermined);
    }

    // --- evaluate: gpu ---------------------------------------------

    #[test]
    fn evaluate_gpu_unavailable_is_undetermined() {
        let cfg = EnvGuardConfig::new(4.0).unwrap();
        let sample = sample_with(
            Some(LoadAvg {
                one: 1.0,
                five: 1.0,
                fifteen: 1.0,
            }),
            GpuSample::Unavailable {
                reason: "no ioreg".to_string(),
            },
        );
        let report = cfg.evaluate(&sample);
        assert_eq!(report.gpu.verdict, GuardVerdict::Undetermined);
        assert_eq!(report.overall, GuardVerdict::Undetermined);
    }

    #[test]
    fn evaluate_gpu_empty_watchlist_is_pass_and_recorded_only() {
        let cfg = EnvGuardConfig::new(4.0).unwrap();
        let sample = sample_with(
            Some(LoadAvg {
                one: 1.0,
                five: 1.0,
                fifteen: 1.0,
            }),
            GpuSample::Available {
                processes: vec![GpuProcess {
                    pid: 42,
                    name: "SomeUnrelatedProc".to_string(),
                }],
                device_utilization_percent: Some(5),
            },
        );
        let report = cfg.evaluate(&sample);
        assert_eq!(report.gpu.verdict, GuardVerdict::Pass);
        assert!(report.gpu.flagged.is_empty());
        assert!(report.gpu.note.is_some());
    }

    #[test]
    fn evaluate_gpu_watchlist_partial_match_fails() {
        let cfg = EnvGuardConfig::new(4.0)
            .unwrap()
            .with_gpu_process_watchlist(vec!["python".to_string()]);
        let sample = sample_with(
            Some(LoadAvg {
                one: 1.0,
                five: 1.0,
                fifteen: 1.0,
            }),
            GpuSample::Available {
                processes: vec![GpuProcess {
                    pid: 42,
                    name: "python3.12".to_string(),
                }],
                device_utilization_percent: None,
            },
        );
        let report = cfg.evaluate(&sample);
        assert_eq!(report.gpu.verdict, GuardVerdict::Fail);
        assert_eq!(report.gpu.flagged.len(), 1);
    }

    #[test]
    fn evaluate_gpu_utilization_exceeded_fails() {
        let cfg = EnvGuardConfig::new(4.0)
            .unwrap()
            .with_max_gpu_device_utilization_percent(50)
            .unwrap();
        let sample = sample_with(
            Some(LoadAvg {
                one: 1.0,
                five: 1.0,
                fifteen: 1.0,
            }),
            GpuSample::Available {
                processes: Vec::new(),
                device_utilization_percent: Some(80),
            },
        );
        let report = cfg.evaluate(&sample);
        assert_eq!(report.gpu.verdict, GuardVerdict::Fail);
    }

    #[test]
    fn evaluate_gpu_utilization_within_limit_passes() {
        let cfg = EnvGuardConfig::new(4.0)
            .unwrap()
            .with_max_gpu_device_utilization_percent(50)
            .unwrap();
        let sample = sample_with(
            Some(LoadAvg {
                one: 1.0,
                five: 1.0,
                fifteen: 1.0,
            }),
            GpuSample::Available {
                processes: Vec::new(),
                device_utilization_percent: Some(10),
            },
        );
        let report = cfg.evaluate(&sample);
        assert_eq!(report.gpu.verdict, GuardVerdict::Pass);
    }

    #[test]
    fn evaluate_gpu_utilization_unavailable_with_limit_is_undetermined() {
        // review #1456 指摘対応: 使用率上限を設定したが `Device Utilization %`
        // を取得できない（`None`）場合、watchlist 不一致だけを根拠に `Pass`
        // へ落としてはならない（取得不能は未判定というモジュール設計方針の
        // 一貫適用）。
        let cfg = EnvGuardConfig::new(4.0)
            .unwrap()
            .with_max_gpu_device_utilization_percent(50)
            .unwrap();
        let sample = sample_with(
            Some(LoadAvg {
                one: 1.0,
                five: 1.0,
                fifteen: 1.0,
            }),
            GpuSample::Available {
                processes: Vec::new(),
                device_utilization_percent: None,
            },
        );
        let report = cfg.evaluate(&sample);
        assert_eq!(report.gpu.verdict, GuardVerdict::Undetermined);
        assert_eq!(report.overall, GuardVerdict::Undetermined);
    }

    #[test]
    fn evaluate_gpu_flagged_overrides_utilization_undetermined() {
        // 使用率が未判定でも watchlist 一致（明確な悪化）は `Fail` を優先する。
        let cfg = EnvGuardConfig::new(4.0)
            .unwrap()
            .with_gpu_process_watchlist(vec!["python".to_string()])
            .with_max_gpu_device_utilization_percent(50)
            .unwrap();
        let sample = sample_with(
            Some(LoadAvg {
                one: 1.0,
                five: 1.0,
                fifteen: 1.0,
            }),
            GpuSample::Available {
                processes: vec![GpuProcess {
                    pid: 123,
                    name: "python3.11".to_string(),
                }],
                device_utilization_percent: None,
            },
        );
        let report = cfg.evaluate(&sample);
        assert_eq!(report.gpu.verdict, GuardVerdict::Fail);
    }

    #[test]
    fn evaluate_gpu_flagged_overrides_utilization_undetermined_note_absent() {
        // review #1456（Cursor Bugbot）指摘対応: watchlist 一致で verdict が
        // 既に `Fail` に確定している場合、note が「未判定」を主張する矛盾を
        // 起こしてはならない。
        let cfg = EnvGuardConfig::new(4.0)
            .unwrap()
            .with_gpu_process_watchlist(vec!["python".to_string()])
            .with_max_gpu_device_utilization_percent(50)
            .unwrap();
        let sample = sample_with(
            Some(LoadAvg {
                one: 1.0,
                five: 1.0,
                fifteen: 1.0,
            }),
            GpuSample::Available {
                processes: vec![GpuProcess {
                    pid: 123,
                    name: "python3.11".to_string(),
                }],
                device_utilization_percent: None,
            },
        );
        let report = cfg.evaluate(&sample);
        assert_eq!(report.gpu.verdict, GuardVerdict::Fail);
        assert!(
            report.gpu.note.is_none(),
            "Fail 確定時に note が「未判定」を主張してはならない: {:?}",
            report.gpu.note
        );
    }

    #[test]
    fn evaluate_gpu_utilization_out_of_range_is_undetermined_not_fail() {
        // review #1456（codex-review P2）指摘対応: 公開 `GpuSample::Available`
        // を直接構築すると `collect()` の 0〜100 変換（100 超は `None`）を経
        // ずに範囲外値（例 255）を渡せてしまう。範囲外値は未取得と同じ
        // `Undetermined` として扱い、不正な観測値を根拠に `Fail` にしない。
        let cfg = EnvGuardConfig::new(4.0)
            .unwrap()
            .with_max_gpu_device_utilization_percent(50)
            .unwrap();
        let sample = sample_with(
            Some(LoadAvg {
                one: 1.0,
                five: 1.0,
                fifteen: 1.0,
            }),
            GpuSample::Available {
                processes: Vec::new(),
                device_utilization_percent: Some(255),
            },
        );
        let report = cfg.evaluate(&sample);
        assert_eq!(report.gpu.verdict, GuardVerdict::Undetermined);
        assert_eq!(report.overall, GuardVerdict::Undetermined);
        assert!(report.gpu.note.is_some());
    }

    #[test]
    fn evaluate_gpu_utilization_out_of_range_with_flagged_is_fail_without_undetermined_note() {
        // 範囲外の使用率観測値があっても、watchlist 一致（明確な悪化）は
        // `Fail` を優先し、note は「未判定」を主張しない（上記 2 指摘の
        // 複合ケース）。
        let cfg = EnvGuardConfig::new(4.0)
            .unwrap()
            .with_gpu_process_watchlist(vec!["python".to_string()])
            .with_max_gpu_device_utilization_percent(50)
            .unwrap();
        let sample = sample_with(
            Some(LoadAvg {
                one: 1.0,
                five: 1.0,
                fifteen: 1.0,
            }),
            GpuSample::Available {
                processes: vec![GpuProcess {
                    pid: 123,
                    name: "python3.11".to_string(),
                }],
                device_utilization_percent: Some(255),
            },
        );
        let report = cfg.evaluate(&sample);
        assert_eq!(report.gpu.verdict, GuardVerdict::Fail);
        assert!(report.gpu.note.is_none());
    }

    #[test]
    fn evaluate_load_avg_invalid_observed_is_undetermined() {
        // review #1456 指摘対応: 公開評価 API 経由で妥当性検証を経ていない
        // 値（NaN・負値）が観測値として渡された場合、`Pass`／`Fail` へ倒さず
        // `Undetermined` にする。
        let cfg = EnvGuardConfig::new(4.0).unwrap();
        let sample = sample_with(
            Some(LoadAvg {
                one: f64::NAN,
                five: 1.0,
                fifteen: 1.0,
            }),
            GpuSample::Unavailable {
                reason: "test".to_string(),
            },
        );
        let report = cfg.evaluate(&sample);
        assert_eq!(report.load_avg.verdict, GuardVerdict::Undetermined);
        // 生の観測値は記録用にそのまま保持する。
        assert!(report.load_avg.observed.unwrap().one.is_nan());
    }

    #[test]
    fn evaluate_load_avg_negative_observed_is_undetermined() {
        let cfg = EnvGuardConfig::new(4.0).unwrap();
        let sample = sample_with(
            Some(LoadAvg {
                one: -1.0,
                five: 1.0,
                fifteen: 1.0,
            }),
            GpuSample::Unavailable {
                reason: "test".to_string(),
            },
        );
        let report = cfg.evaluate(&sample);
        assert_eq!(report.load_avg.verdict, GuardVerdict::Undetermined);
    }

    // --- overall / is_blocking ---------------------------------------

    #[test]
    fn overall_priority_fail_over_undetermined() {
        assert_eq!(
            combine_verdicts(GuardVerdict::Fail, GuardVerdict::Undetermined),
            GuardVerdict::Fail
        );
        assert_eq!(
            combine_verdicts(GuardVerdict::Undetermined, GuardVerdict::Fail),
            GuardVerdict::Fail
        );
    }

    #[test]
    fn overall_priority_undetermined_over_pass() {
        assert_eq!(
            combine_verdicts(GuardVerdict::Undetermined, GuardVerdict::Pass),
            GuardVerdict::Undetermined
        );
    }

    #[test]
    fn overall_pass_when_both_pass() {
        assert_eq!(
            combine_verdicts(GuardVerdict::Pass, GuardVerdict::Pass),
            GuardVerdict::Pass
        );
    }

    #[test]
    fn is_blocking_true_only_for_fail() {
        let cfg = EnvGuardConfig::new(4.0).unwrap();
        let fail_sample = sample_with(
            Some(LoadAvg {
                one: 10.0,
                five: 1.0,
                fifteen: 1.0,
            }),
            GpuSample::Unavailable {
                reason: "test".to_string(),
            },
        );
        assert!(cfg.evaluate(&fail_sample).is_blocking());

        let undetermined_sample = sample_with(
            None,
            GpuSample::Unavailable {
                reason: "test".to_string(),
            },
        );
        assert!(!cfg.evaluate(&undetermined_sample).is_blocking());

        let pass_sample = sample_with(
            Some(LoadAvg {
                one: 1.0,
                five: 1.0,
                fifteen: 1.0,
            }),
            GpuSample::Available {
                processes: Vec::new(),
                device_utilization_percent: None,
            },
        );
        assert!(!cfg.evaluate(&pass_sample).is_blocking());
    }

    // --- serde ---------------------------------------------------------

    #[test]
    fn report_serializes_to_json() {
        let cfg = EnvGuardConfig::new(4.0).unwrap();
        let sample = sample_with(
            Some(LoadAvg {
                one: 1.0,
                five: 1.0,
                fifteen: 1.0,
            }),
            GpuSample::Available {
                processes: Vec::new(),
                device_utilization_percent: None,
            },
        );
        let report = cfg.evaluate(&sample);
        let json = serde_json::to_string(&report).expect("有限値のみのため serialize は成功する");
        assert!(json.contains("overall"));
    }

    // --- collect: 全プラットフォームで panic しない -----------------

    #[test]
    fn collect_never_panics() {
        let sample = EnvSample::collect();
        // Linux（GitHub ホステッド runner）では `/proc/loadavg` が読める
        // ことを追加確認する（実測手段が機能していることの回帰検知）。
        #[cfg(target_os = "linux")]
        {
            assert!(sample.load_avg.is_some());
        }
        // どのプラットフォームでも `collect` 自体が値を返せばよい
        // （unreachable な panic が発生しないことの確認が主目的）。
        let _ = sample.uptime_secs;
    }

    #[test]
    #[ignore = "実機（macOS）セッション限定。GPU プロセス検出の実取得を確認する"]
    fn collect_macos_gpu_available() {
        let sample = EnvSample::collect();
        assert!(sample.load_avg.is_some());
        match sample.gpu {
            GpuSample::Available { processes, .. } => {
                assert!(!processes.is_empty(), "常駐プロセスが 1 件も無いのは想定外");
            }
            GpuSample::Unavailable { reason } => {
                panic!("macOS では Available を期待する（reason: {reason}）");
            }
        }
    }
}
