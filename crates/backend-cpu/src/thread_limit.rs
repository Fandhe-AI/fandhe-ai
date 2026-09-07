//! `gemm_blis` 並列 GEMM の既定並列度を「物理大コア数」へ限定する判定
//! モジュール（イシュー #1363・親 #1362・祖 #1361・ルート #1269）。
//!
//! ## 背景・目的
//!
//! `docs/perf/cpu-gemm-candle-gate-remeasurement.md` §8.2 の
//! `RAYON_NUM_THREADS` スイープで、Apple M4 Max（P12+E4）・DGX Spark
//! GB10（X925×10+A725×10）とも「大コア数付近でスループットが落ち込み
//! 全コアで部分回復する」という非単調性が観測された。`gemm_blis_parallel`
//! （[`crate::gemm_blis`]）の静的等分割行パネル（`c.par_chunks_mut(panel_rows
//! * n)`。`panel_rows = m.div_ceil(rayon::current_num_threads())`）が
//! 異種コア（big.LITTLE 系）構成で little コア律速になる仮説を検証する
//! 前提として、本モジュールは `RAYON_NUM_THREADS` 未指定時の既定並列度を
//! 大コア数へ限定するプラットフォーム判定を実装する。**性能上の採否
//! 判断（勝敗）は本モジュールのスコープ外**であり、実測は #1364
//! （両実機・framework-compare 前後比較・5 回計測中央値）へ引き継ぐ
//! （`docs/perf/cpu-gemm-default-thread-limit.md` 参照）。
//!
//! ## 判定方式
//!
//! - **macOS**: `hw.perflevel0.logicalcpu`（P コア論理数）を
//!   `hw.logicalcpu`（全論理コア数）と比較し、前者が後者より小さい場合の
//!   み P コア数を大コア数とする（両方取得できない・両者が等しい
//!   〈非対称構成でない〉場合は `None` = 判定不能）。取得手段は
//!   `std::process::Command`（絶対パス `/usr/sbin/sysctl`・固定引数・
//!   `env_clear()`）を用いる。`cache_params::sysctl_ffi`（`#[cfg(test)]`
//!   限定・本番非到達）の `unsafe extern "C" sysctlbyname` FFI を本番
//!   到達化するのは実質的な unsafe 面の新設に当たり、自動運転（ユーザー
//!   承認を得られない）では安全側の `Command` 経路を採る
//!   （`.claude/rules/security.md`「unsafe」節・PR #766「常に不活性な
//!   sysctl 経路」撤去の教訓）
//! - **Linux**: `/sys/devices/system/cpu/cpu<N>/cpu_capacity`
//!   （sched capacity-aware scheduling が公開する非対称コア容量）を
//!   全 CPU について走査し、値が非一様（big.LITTLE 系）なら最大値と
//!   一致するコア数を大コア数とする。1 つでも欠損・parse 不能なら
//!   `None`、全 CPU が一様（同種コア。x86_64 CI 環境等）なら `None`
//!   （no-op）とする
//! - **上記 2 プラットフォーム以外**: 常に `None`
//!
//! いずれのプラットフォームでも判定失敗（sandbox でのプロセス spawn
//! 不可・sysfs 不在・同種コア構成等）は fail-safe に「現行既定
//! （`rayon::current_num_threads()`）をそのまま使う」へフォールバックする
//! （panic 経路なし・`unwrap`／`expect` 不使用。R3・R4 に対応）。
//!
//! ## `RAYON_NUM_THREADS` との関係
//!
//! `RAYON_NUM_THREADS` が有効な正の整数として設定されている場合
//! （rayon 自身がグローバルプールのスレッド数として読む既存の環境
//! 変数。本モジュールは新しい環境変数を追加しない）、その値を
//! rayon が採用した既定として尊重し、大コア数による上限は適用しない
//! （明示指定を上書きしない。§3.3 の設計判断）。
//!
//! ## 呼び出し元
//!
//! [`effective_num_threads`] は [`crate::gemm_blis`] の並列 GEMM 入口
//! （本番: `gemm_blis_parallel_with_transpose`・`gemm_blis_bias_act_parallel`。
//! `#[cfg(test)]` の A/B 計測ハーネス: `gemm_blis_parallel_with_blocks`・
//! `gemm_blis_parallel_row_panel_with_blocks`・`gemm_blis_parallel_2d_with_blocks`
//! 等）から `rayon::current_num_threads()` の直後に呼ばれ、返り値が
//! 行パネル数（`panel_rows = m.div_ceil(effective)`）の算出に使われる。
//! `BIG_CORE_LIMIT_ENABLED` の単一 const ゲートで無効化でき、#1364 の
//! 実機実測が後退と判断した場合は 1 行差し戻すだけで済む
//! （`docs/perf/cuda-gemm-auto-f16-mma-switch.md` の
//! `MMA_PRIORITY_PRODUCTION_ENABLED` 前例と同型のロールバック機構）。

// Linux 判定（[`big_cores_from_sysfs`]）とテスト fixture のみが `Path` を
// 使う。macOS 単体の非テストビルドでは未使用になるため
// `cfg(any(target_os = "linux", test))` を付ける（`use` 単体の dead_code
// 検出は個別の `#[cfg]` では防げないため import 自体をゲートする）。
#[cfg(any(target_os = "linux", test))]
use std::path::Path;
use std::sync::OnceLock;

/// 大コア数限定を有効化する単一ゲート。#1364（両実機 framework-compare
/// 前後比較）が性能後退と判断した場合、ここを `false` へ 1 行差し戻す
/// だけで [`effective_num_threads`] は常に `current` をそのまま返す
/// （本番結線は事前承認済み。実測根拠は PR 本文・
/// `docs/perf/cpu-gemm-default-thread-limit.md` に記録する）。
pub(crate) const BIG_CORE_LIMIT_ENABLED: bool = true;

/// 診断・#1364 の env_info 記録用に判定結果を可視化する構造体
/// （[`thread_limit_report`] の戻り値）。`facade` 等の公開 API 面へは
/// 昇格しない（`.claude/rules/delegation-impl.md`「禁止事項」・本イシュー
/// 計画「スコープ外」節）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadLimitReport {
    /// 呼び出し時点の `rayon::current_num_threads()`。
    pub current: usize,
    /// プラットフォーム判定で検出した大コア数（判定不能なら `None`）。
    pub detected_big_cores: Option<usize>,
    /// `RAYON_NUM_THREADS` が有効な正整数として設定されているか。
    pub env_override: bool,
    /// `effective_num_threads`（クレート内部の実効スレッド数算出関数）が
    /// 実際に返す値。
    pub effective: usize,
}

/// [`resolve`] が一度だけ評価した判定結果（`detected`・`env_override`）を
/// キャッシュする。初回の並列 GEMM 呼び出しで 1 回だけプラットフォーム
/// I/O（macOS: `sysctl` 子プロセス spawn・Linux: sysfs 読み取り）を行い、
/// 以降の呼び出しはキャッシュ値を再利用する（ロード時初期化・`ctor` は
/// 使わず、`OnceLock` による遅延初期化のみで完結させる）。
static DECISION: OnceLock<(Option<usize>, bool)> = OnceLock::new();

/// [`crate::gemm_blis`] の並列 GEMM 入口から呼ばれる実効スレッド数の
/// 算出関数。`current`（`rayon::current_num_threads()`）を受け取り、
/// [`BIG_CORE_LIMIT_ENABLED`]・`RAYON_NUM_THREADS` 明示設定・プラット
/// フォーム判定結果に応じて `1 <= 戻り値 <= current` を満たす値を返す
/// （契約は [`resolve`] のドキュメント参照）。
///
/// `current <= 1`（シングルスレッドプール。例: `bench-harness` の
/// 起動プローブが単スレッドで動く場合）ではプラットフォーム I/O を
/// 省略し `1` を即座に返す（判定コストが起動プローブの計測値に乗る
/// ことを避ける。設計 §3.4）。
pub(crate) fn effective_num_threads(current: usize) -> usize {
    if current <= 1 {
        return 1;
    }
    let (detected, env_override) = *DECISION.get_or_init(|| {
        (
            detect_big_cores(),
            parse_env_num_threads(std::env::var("RAYON_NUM_THREADS").ok().as_deref()).is_some(),
        )
    });
    resolve(current, detected, env_override, BIG_CORE_LIMIT_ENABLED)
}

/// #1364 の env_info 記録・診断用に判定結果全体を可視化する（`examples/
/// gemm_bench.rs`・`docs/perf/cpu-gemm-default-thread-limit.md` から
/// 利用する想定。本番 GEMM 経路からは呼ばれない）。`effective_num_threads`
/// と同じ `OnceLock` キャッシュを共有するため、初回呼び出し順序に
/// 関わらず両者は同一の判定結果を参照する。
pub fn thread_limit_report() -> ThreadLimitReport {
    let current = rayon::current_num_threads().max(1);
    let (detected, env_override) = *DECISION.get_or_init(|| {
        (
            detect_big_cores(),
            parse_env_num_threads(std::env::var("RAYON_NUM_THREADS").ok().as_deref()).is_some(),
        )
    });
    ThreadLimitReport {
        current,
        detected_big_cores: detected,
        env_override,
        effective: resolve(current, detected, env_override, BIG_CORE_LIMIT_ENABLED),
    }
}

/// 判定結果を実効スレッド数へ写像する純関数（全プラットフォームで
/// 単体テスト可能。プラットフォーム I/O を持たない）。
///
/// 契約:
/// - `enabled == false` → `current`（機能全体を無効化）
/// - `env_override == true` → `current`（`RAYON_NUM_THREADS` 明示時は
///   rayon の採用値をそのまま尊重し上限をかけない。R2）
/// - `detected` が `None`／`Some(0)`／`current` 以上 → `current`
///   （判定不能・不正値・上限にならない値はすべて現行既定へ
///   フォールバック。R3）
/// - それ以外（`0 < detected < current`）→ `detected`
///
/// 常に `1 <= 戻り値 <= current` を満たす（`current == 0` は呼び出し元
/// [`effective_num_threads`] が `current <= 1` で先に `1` を返すため
/// 到達しないが、本関数単体では `current` をそのまま返し矛盾しない）。
fn resolve(current: usize, detected: Option<usize>, env_override: bool, enabled: bool) -> usize {
    if !enabled || env_override {
        return current;
    }
    match detected {
        Some(big) if big > 0 && big < current => big,
        _ => current,
    }
}

/// `RAYON_NUM_THREADS` の値を rayon 自身の解釈に合わせて parse する
/// （正の `usize` として parse できる場合のみ「明示設定あり」とみなす。
/// 空文字列・`0`・負数・非数値は rayon 側でも無視されるため、本関数でも
/// 同様に「未設定扱い」として `None` を返す）。
fn parse_env_num_threads(raw: Option<&str>) -> Option<usize> {
    let raw = raw?.trim();
    let value: usize = raw.parse().ok()?;
    if value == 0 { None } else { Some(value) }
}

/// `sysctl -n <name>` の stdout（末尾改行あり）から大コア数を parse する
/// 純関数。空・非数値・0・上限（4096。実在しない CPU 数のガード値）
/// 超過はすべて `None`（判定不能）として扱う。
fn parse_sysctl_stdout(bytes: &[u8]) -> Option<usize> {
    let text = std::str::from_utf8(bytes).ok()?;
    let value: usize = text.trim().parse().ok()?;
    if value == 0 || value > 4096 {
        None
    } else {
        Some(value)
    }
}

/// macOS の P/E コア非対称構成判定: `perflevel0`（P コア論理数）が
/// `total`（全論理コア数）より真に小さい場合のみ P コア数を大コア数と
/// する。等しい場合（同種コア構成。`perflevel0` 自体が存在しない旧
/// アーキテクチャ等）は非対称でないため `None` を返す。
fn big_cores_from_sysctl(perflevel0: Option<usize>, total: Option<usize>) -> Option<usize> {
    match (perflevel0, total) {
        (Some(p), Some(t)) if p > 0 && p < t => Some(p),
        _ => None,
    }
}

/// Linux sysfs の `cpu_capacity` 走査結果から大コア数を判定する純関数。
/// `entries` は `(コア番号, capacity 値)` のペア列（走査順は問わない）。
///
/// - 1 件も無い → `None`
/// - 最大値と最小値が等しい（同種コア。x86_64 等） → `None`（no-op）
/// - 非一様 → 最大値と一致するエントリ数を大コア数として返す
///
/// 本番経路では Linux の [`detect_big_cores`] からのみ呼ばれるため
/// `cfg(any(target_os = "linux", test))` で条件付きコンパイルする
/// （macOS 単体ビルドでは未使用になり `dead_code` を誤検出するため。
/// `cfg(test)` を残すのは `mod tests` から直接ユニットテストするため）。
#[cfg(any(target_os = "linux", test))]
fn big_cores_from_capacities(entries: &[(usize, usize)]) -> Option<usize> {
    if entries.is_empty() {
        return None;
    }
    let max = entries.iter().map(|(_, cap)| *cap).max()?;
    let min = entries.iter().map(|(_, cap)| *cap).min()?;
    if max == min {
        return None;
    }
    let big_count = entries.iter().filter(|(_, cap)| *cap == max).count();
    if big_count == 0 {
        None
    } else {
        Some(big_count)
    }
}

/// `/sys/devices/system/cpu` 配下の `cpu<数字>` ディレクトリのみを走査し
/// `cpu_capacity` を読む（`cpufreq`／`cpuidle`／`cpuN` 以外の名前は
/// デコイとして無視する）。1 つでも `cpu_capacity` の読み取り・parse に
/// 失敗したエントリがあれば全体を `None` とする（部分的な判定は
/// 誤った大コア数を導きうるため、fail-safe に判定不能へ倒す。R3）。
///
/// [`big_cores_from_capacities`] と同じ理由で
/// `cfg(any(target_os = "linux", test))` を付ける。
#[cfg(any(target_os = "linux", test))]
fn big_cores_from_sysfs(root: &Path) -> Option<usize> {
    let read_dir = std::fs::read_dir(root).ok()?;
    let mut entries: Vec<(usize, usize)> = Vec::new();
    for entry in read_dir {
        let entry = entry.ok()?;
        let name = entry.file_name();
        let name = name.to_str()?;
        let Some(num_str) = name.strip_prefix("cpu") else {
            continue;
        };
        // `cpufreq`／`cpuidle`／`cpu-map` 等は `strip_prefix("cpu")` 後も
        // 数字にならないため、ここで自然に除外される（デコイ排除）。
        let Ok(cpu_num) = num_str.parse::<usize>() else {
            continue;
        };
        let capacity_path = entry.path().join("cpu_capacity");
        let raw = std::fs::read_to_string(&capacity_path).ok()?;
        let capacity: usize = raw.trim().parse().ok()?;
        entries.push((cpu_num, capacity));
    }
    big_cores_from_capacities(&entries)
}

/// macOS: `/usr/sbin/sysctl -n <name>` を絶対パス・固定引数・
/// `env_clear()`（PATH 等の環境依存を排除し `A03 インジェクション`
/// 観点でユーザー入力の連結を一切行わない。`.claude/rules/security.md`）
/// で起動し、stdout を [`parse_sysctl_stdout`] で parse する。
#[cfg(target_os = "macos")]
fn read_sysctl(name: &str) -> Option<usize> {
    use std::process::Command;
    let output = Command::new("/usr/sbin/sysctl")
        .env_clear()
        .args(["-n", name])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_sysctl_stdout(&output.stdout)
}

#[cfg(target_os = "macos")]
fn detect_big_cores() -> Option<usize> {
    let perflevel0 = read_sysctl("hw.perflevel0.logicalcpu");
    let total = read_sysctl("hw.logicalcpu");
    big_cores_from_sysctl(perflevel0, total)
}

#[cfg(target_os = "linux")]
fn detect_big_cores() -> Option<usize> {
    big_cores_from_sysfs(Path::new("/sys/devices/system/cpu"))
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn detect_big_cores() -> Option<usize> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- resolve ---------------------------------------------------

    #[test]
    fn resolve_disabled_returns_current() {
        assert_eq!(resolve(16, Some(12), false, false), 16);
    }

    #[test]
    fn resolve_env_override_returns_current() {
        assert_eq!(resolve(16, Some(12), true, true), 16);
    }

    #[test]
    fn resolve_none_returns_current() {
        assert_eq!(resolve(16, None, false, true), 16);
    }

    #[test]
    fn resolve_zero_returns_current() {
        assert_eq!(resolve(16, Some(0), false, true), 16);
    }

    #[test]
    fn resolve_detected_ge_current_returns_current() {
        assert_eq!(resolve(16, Some(16), false, true), 16);
        assert_eq!(resolve(16, Some(20), false, true), 16);
    }

    #[test]
    fn resolve_detected_lt_current_returns_detected() {
        assert_eq!(resolve(16, Some(12), false, true), 12);
    }

    #[test]
    fn resolve_current_one_returns_current() {
        // effective_num_threads は current<=1 を先に弾くが、resolve 単体
        // としても current=1 で矛盾しないことを確認する。
        assert_eq!(resolve(1, Some(12), false, true), 1);
    }

    #[test]
    fn resolve_current_zero_returns_current() {
        assert_eq!(resolve(0, Some(12), false, true), 0);
    }

    // --- parse_env_num_threads --------------------------------------

    #[test]
    fn parse_env_num_threads_valid() {
        assert_eq!(parse_env_num_threads(Some("16")), Some(16));
        assert_eq!(parse_env_num_threads(Some(" 4 ")), Some(4));
    }

    #[test]
    fn parse_env_num_threads_invalid() {
        assert_eq!(parse_env_num_threads(None), None);
        assert_eq!(parse_env_num_threads(Some("")), None);
        assert_eq!(parse_env_num_threads(Some("abc")), None);
        assert_eq!(parse_env_num_threads(Some("0")), None);
        assert_eq!(parse_env_num_threads(Some("-1")), None);
    }

    // --- parse_sysctl_stdout ------------------------------------------

    #[test]
    fn parse_sysctl_stdout_valid() {
        assert_eq!(parse_sysctl_stdout(b"12\n"), Some(12));
        assert_eq!(parse_sysctl_stdout(b"  16  "), Some(16));
    }

    #[test]
    fn parse_sysctl_stdout_invalid() {
        assert_eq!(parse_sysctl_stdout(b""), None);
        assert_eq!(parse_sysctl_stdout(b"abc"), None);
        assert_eq!(parse_sysctl_stdout(b"0\n"), None);
        assert_eq!(parse_sysctl_stdout(b"-1\n"), None);
        assert_eq!(parse_sysctl_stdout(b"999999\n"), None);
        assert_eq!(parse_sysctl_stdout(&[0xff, 0xfe]), None); // 不正 UTF-8
    }

    // --- big_cores_from_sysctl -----------------------------------------

    #[test]
    fn big_cores_from_sysctl_asymmetric() {
        assert_eq!(big_cores_from_sysctl(Some(12), Some(16)), Some(12));
    }

    #[test]
    fn big_cores_from_sysctl_uniform_returns_none() {
        assert_eq!(big_cores_from_sysctl(Some(16), Some(16)), None);
    }

    #[test]
    fn big_cores_from_sysctl_missing_returns_none() {
        assert_eq!(big_cores_from_sysctl(None, Some(16)), None);
        assert_eq!(big_cores_from_sysctl(Some(12), None), None);
        assert_eq!(big_cores_from_sysctl(None, None), None);
    }

    #[test]
    fn big_cores_from_sysctl_perflevel0_ge_total_returns_none() {
        // 理論上不整合な値（P コア数が全体以上）は非対称構成でないとみなす。
        assert_eq!(big_cores_from_sysctl(Some(16), Some(16)), None);
        assert_eq!(big_cores_from_sysctl(Some(20), Some(16)), None);
    }

    // --- big_cores_from_capacities --------------------------------------

    #[test]
    fn big_cores_from_capacities_asymmetric() {
        let mut entries = Vec::new();
        for i in 0..12 {
            entries.push((i, 1024));
        }
        for i in 12..16 {
            entries.push((i, 512));
        }
        assert_eq!(big_cores_from_capacities(&entries), Some(12));
    }

    #[test]
    fn big_cores_from_capacities_uniform_returns_none() {
        let entries: Vec<(usize, usize)> = (0..16).map(|i| (i, 1024)).collect();
        assert_eq!(big_cores_from_capacities(&entries), None);
    }

    #[test]
    fn big_cores_from_capacities_empty_returns_none() {
        assert_eq!(big_cores_from_capacities(&[]), None);
    }

    // --- big_cores_from_sysfs (fixture ベース) --------------------------

    /// テストごとに一意な一時ディレクトリを作り、`cpu<N>/cpu_capacity`
    /// fixture を書き込む（依存追加なしで `std::env::temp_dir()` 配下に
    /// 作成・テスト終了時に `Drop` で削除する。#1363 実装計画 §5 手順 2）。
    struct SysfsFixture {
        root: std::path::PathBuf,
    }

    impl SysfsFixture {
        fn new(name: &str) -> Self {
            let root = std::env::temp_dir().join(format!(
                "fandhe-ai-thread-limit-test-{name}-{}-{}",
                std::process::id(),
                // テスト並列実行時の衝突を避けるための簡易 nonce。
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0),
            ));
            std::fs::create_dir_all(&root).expect("create fixture root");
            Self { root }
        }

        fn write_cpu_capacity(&self, cpu: usize, capacity: &str) {
            let dir = self.root.join(format!("cpu{cpu}"));
            std::fs::create_dir_all(&dir).expect("create cpu dir");
            std::fs::write(dir.join("cpu_capacity"), capacity).expect("write cpu_capacity");
        }

        fn write_decoy_dir(&self, name: &str) {
            std::fs::create_dir_all(self.root.join(name)).expect("create decoy dir");
        }
    }

    impl Drop for SysfsFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn big_cores_from_sysfs_asymmetric() {
        let fx = SysfsFixture::new("asymmetric");
        for i in 0..12 {
            fx.write_cpu_capacity(i, "1024\n");
        }
        for i in 12..16 {
            fx.write_cpu_capacity(i, "512\n");
        }
        assert_eq!(big_cores_from_sysfs(&fx.root), Some(12));
    }

    #[test]
    fn big_cores_from_sysfs_uniform_returns_none() {
        let fx = SysfsFixture::new("uniform");
        for i in 0..16 {
            fx.write_cpu_capacity(i, "1024\n");
        }
        assert_eq!(big_cores_from_sysfs(&fx.root), None);
    }

    #[test]
    fn big_cores_from_sysfs_missing_file_returns_none() {
        let fx = SysfsFixture::new("missing-file");
        fx.write_cpu_capacity(0, "1024\n");
        // cpu1 は cpu_capacity を持たない（欠損）。
        std::fs::create_dir_all(fx.root.join("cpu1")).expect("create cpu1 dir");
        assert_eq!(big_cores_from_sysfs(&fx.root), None);
    }

    #[test]
    fn big_cores_from_sysfs_invalid_value_returns_none() {
        let fx = SysfsFixture::new("invalid-value");
        fx.write_cpu_capacity(0, "1024\n");
        fx.write_cpu_capacity(1, "not-a-number\n");
        assert_eq!(big_cores_from_sysfs(&fx.root), None);
    }

    #[test]
    fn big_cores_from_sysfs_ignores_decoy_dirs() {
        let fx = SysfsFixture::new("decoy");
        for i in 0..12 {
            fx.write_cpu_capacity(i, "1024\n");
        }
        for i in 12..16 {
            fx.write_cpu_capacity(i, "512\n");
        }
        // `cpufreq`／`cpuidle`／`cpu-map` 等は `cpu` prefix を持つが
        // 数字が続かない、または非 CPU ディレクトリのため無視される。
        fx.write_decoy_dir("cpufreq");
        fx.write_decoy_dir("cpuidle");
        fx.write_decoy_dir("cpu-map");
        assert_eq!(big_cores_from_sysfs(&fx.root), Some(12));
    }

    #[test]
    fn big_cores_from_sysfs_empty_dir_returns_none() {
        let fx = SysfsFixture::new("empty");
        assert_eq!(big_cores_from_sysfs(&fx.root), None);
    }

    #[test]
    fn big_cores_from_sysfs_nonexistent_root_returns_none() {
        let root = std::env::temp_dir().join("fandhe-ai-thread-limit-test-nonexistent-root-xyz");
        assert_eq!(big_cores_from_sysfs(&root), None);
    }

    // --- effective_num_threads (統合。プラットフォーム非依存の契約) -----

    #[test]
    fn effective_num_threads_single_thread_pool_returns_one() {
        assert_eq!(effective_num_threads(0), 1);
        assert_eq!(effective_num_threads(1), 1);
    }

    #[test]
    fn effective_num_threads_bounds_hold_for_current_platform() {
        // 実プラットフォーム判定（キャッシュ経由）を用いた契約テスト:
        // 戻り値は常に 1..=current の範囲に収まる（判定成否・
        // RAYON_NUM_THREADS の実行時設定に関わらず成立する）。
        for current in [1usize, 2, 4, 8, 16, 32] {
            let effective = effective_num_threads(current);
            assert!(effective >= 1);
            assert!(effective <= current.max(1));
        }
    }
}
