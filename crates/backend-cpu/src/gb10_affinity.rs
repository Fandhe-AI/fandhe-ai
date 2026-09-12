//! GB10（DGX Spark GB10。Cortex-X925 ×10 + Cortex-A725 ×10）の小形状
//! GEMM 向け大コア OS affinity 自機判定モジュール（イシュー #1576・
//! 親 #1571・依存 #1574）。
//!
//! ## 背景・[`crate::thread_limit`] との違い
//!
//! `docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/rayon-sweep.jsonl`／
//! `rayon-sweep-pinned.jsonl`（実機実測）により、GB10 の小形状 GEMM
//! （train reuse・size=64）は次の非自明な特性を持つことが確定している。
//!
//! | 条件 | train reuse（size=64・3 起動中央値） |
//! |---|---|
//! | 無 pin T20（全コア） | 1.029 s |
//! | 無 pin T10（スレッド数のみ大コア数に一致） | 2.359 s（**T20 より遅い**） |
//! | 大コア pin T10（`taskset -c 5-9,15-19`） | 0.857 s（**T20 より速い**） |
//!
//! [`crate::thread_limit`]（#1363・`BIG_CORE_LIMIT_ENABLED`）はスレッド
//! **数**を大コア数へ制限するだけで OS レベルの affinity は設定せず、
//! GB10 実機実測（#1364）で REJECT 確定済み（`cpu_capacity` sysfs 誤検出
//! に加え、上表が示す通りスレッド数を絞るだけではむしろ悪化する）。
//! 効果があるのは実際に OS レベルで大コアへ thread affinity を設定する
//! ことであり、本モジュールはそのための**別系統**の機構を実装する
//! （[`crate::thread_limit`] の判定・ゲート・キャッシュとは完全に独立
//! しており、両モジュールが同時に活性化することはない。本モジュールは
//! 常に既定 OFF・[`GB10_AFFINITY_ENABLED`] 単一 const ゲートで無効化
//! できる。詳細設計は `docs/backend-cpu-gb10-affinity-design.md`）。
//!
//! ## 大コア判定（検出）
//!
//! `cpu_capacity`（#1364 で誤検出確定）は使わず、GB10 実機実測
//! （`docs/perf/logs/cpu-gemm-rayon-sweep-1305/env_info_raw-dgx.txt`）で
//! 相互に完全一致することを確認済みの 2 つの独立 sysfs 指標を
//! **クロスバリデーション**する:
//!
//! 1. `cpufreq/cpuinfo_max_freq`（`/sys/devices/system/cpu/cpu<N>/cpufreq/
//!    cpuinfo_max_freq`）: 2 群に分かれる（3900MHz 群＝X925 / 2808MHz
//!    群＝A725）
//! 2. `regs/identification/midr_el1`（ARM MIDR_EL1 の partnum ビット
//!    `[15:4]`）: `cpu0`＝`0xd87`（A725）・`cpu5`＝`0xd85`（X925）
//!
//! 両指標がそれぞれ厳密に 2 群へ分かれ、かつ両者の CPU-id 分割が
//! （群の順序に依らず）完全一致する場合のみ、周波数の高い方の群を
//! 大コア群として確定する（partnum の大小や ARM コアコード表の
//! ハードコードは行わない＝将来の SoC 世代でも破綻しない）。
//! いずれかのファイルが 1 つでも欠損・parse 不能、2 群にきれいに
//! 分かれない、または両指標の分割が食い違う場合は `None`（判定不能・
//! no-op）とする（[`crate::thread_limit`] と同じ fail-safe 方針）。
//!
//! `midr_el1` は x86_64 には存在しないため、この判定は構造的に ARM 系の
//! 非対称構成にのみ発火する（Intel P/E コア機・CI の x86_64 ランナーは
//! 自然に `None` へフォールバックする）。
//!
//! ### cgroup／cpuset 制約への fail-closed 対応
//!
//! `/proc/self/status` の `Cpus_allowed_list:` 行を読み、検出した大コア
//! CPU-id 集合が allowed 集合の**部分集合**であることを確認する
//! （`sched_setaffinity` が禁止 CPU に対し `EINVAL` を返すのを待たず、
//! 事前に安全側へ倒す）。allowed 一覧自体が読めない・parse できない
//! 場合も安全側（判定不能）へ倒す。
//!
//! ## `RAYON_NUM_THREADS` との関係
//!
//! [`crate::thread_limit`] と同じ設計判断として、`RAYON_NUM_THREADS` が
//! 有効な正整数として明示設定されている場合は本機構自体を適用しない
//! （ユーザーの明示指定を上書きしない）。
//!
//! ## Affinity 設定（unsafe FFI）
//!
//! Linux では `sched_setaffinity(2)` システムコールが必要で Rust std に
//! 対応 API が無い。許容依存 9 区分（`.claude/rules/deps-policy.md`）に
//! `libc`／`core_affinity` は含まれないため、新規クレートを追加せず
//! [`crate::gemm_blis::cache_params`] の macOS `sysctlbyname` FFI・
//! [`crate::thread_limit`] doc 記載の `sysctlbyname` FFI と同型の、
//! glibc に常にリンクされる C ABI 関数への直接 `extern "C"` 宣言で
//! 実装する（[`affinity_ffi`] モジュール参照）。`unsafe` はこの syscall
//! 呼び出し 1 箇所のみで、戻り値のエラーは panic させず無視する
//! （affinity 未設定のまま続行しても正しさに影響しない。
//! `.claude/rules/coding-rust.md`「本番経路で `unwrap`/`expect` を
//! 使わない」）。
//!
//! ## 専用スレッドプール（グローバルプールを汚染しない）
//!
//! `rayon::ThreadPoolBuilder::build_global()` はプロセス全体の既定
//! プールを差し替え、`backend-cpu` 以外の rayon 利用（[`crate::mse`]
//! 等）にも影響するため、GEMM 専用の `OnceLock<Option<ThreadPool>>`
//! （[`affinity_pool`]）を新設し影響範囲を GEMM の並列公開入口 2 関数
//! （[`crate::gemm_blis::gemm_blis_parallel_with_transpose`]・
//! [`crate::gemm_blis::gemm_blis_bias_act_parallel`]）に限定する。
//! 専用プール（大コア数スレッド）とグローバルプール（全コア数
//! スレッド）が同時に活性化しうる（例: train の backward 中に非 GEMM
//! rayon 処理が並走する場合）ためオーバーサブスクリプションの余地は
//! 残るが、GEMM 専用プールは `.install()` される短命スコープに限られる。
//!
//! ## 適用範囲（小形状限定）
//!
//! 大形状（N=1024/2048/4096 の正方 GEMM）を専用プール（大コア数のみ）
//! へ回すと、既に本番採用済みの 2D 動的分配（[`crate::gemm_blis::
//! TWO_D_DYNAMIC_PRODUCTION_ENABLED`]。全コア前提・#1313 で ADOPT）から
//! 性能を奪う。よって全 GEMM 呼び出しを無条件に専用プールへ通さず、
//! `m * n * k`（総仕事量）が [`GB10_AFFINITY_MAX_WORK`] 以下の形状に
//! 限定する（[`should_route_to_affinity_pool`]）。
//!
//! ## 正しさ（bit 完全一致）契約
//!
//! `gemm_blis` の並列分割はスレッド数に依らず出力が bit 完全一致で
//! あることが既存の全 A/B（#1364・#1312・#1367・#1318・#1481 等）で
//! 確認済みの不変条件。本モジュールはスレッドの**実行位置**のみを
//! 変え、GEMM の数学的分割は `rayon::current_num_threads()`
//! （専用プール内では専用プールのスレッド数を返す）を経由して既存
//! ロジックがそのまま処理するため、新規の数値一致リスクは生じない。

use std::sync::OnceLock;

#[cfg(any(target_os = "linux", test))]
use std::path::Path;

/// 本機構を有効化する単一ゲート。既定 `false`
/// （GB10 実機 A/B 実測は本イシューの実行環境に GB10 実機への到達手段が
/// なく未実施のまま記入欄を残す。`docs/perf/cpu-gemm-gb10-affinity-ab.md`
/// 参照。ADOPT が確定した場合のみ `true` へ切替える単一行ロールバック
/// 機構は [`crate::thread_limit::BIG_CORE_LIMIT_ENABLED`] と同型）。
pub(crate) const GB10_AFFINITY_ENABLED: bool = false;

/// 専用 affinity プールへルーティングする仕事量（`m * n * k`）の上限。
///
/// 診断対象形状（`bench-fandhe --task train/infer --size 64` の MLP:
/// `BATCH=64・D_IN=784・D_HIDDEN=256・D_OUT=10`）の layer1 GEMM は
/// `m*n*k = 64*256*784 ≈ 12.8M`。対して非対象の正方 GEMM 最小形状
/// N=512 は `512^3 ≈ 134M`。この間に十分な余裕を持つ値として
/// `32 * 1024 * 1024`（約 33.5M）を初期値とする（実測に基づく
/// チューニングは #1576 §4 ガードセル A/B で確認する。判定規則自体は
/// 事後緩和しない）。
pub(crate) const GB10_AFFINITY_MAX_WORK: usize = 32 * 1024 * 1024;

/// `m * n * k` が [`GB10_AFFINITY_MAX_WORK`] 以下の小形状のみ専用
/// affinity プールへルーティング対象とする（大形状の 2D 動的分配
/// 〈全コア前提〉から性能を奪わないための境界。モジュール doc
/// 「適用範囲」節参照）。オーバーフロー時は `saturating_mul` で
/// `usize::MAX` に飽和させ、常に範囲外（ルーティング対象外）と
/// 判定する（`.claude/rules/security.md` OWASP A03 観点。GEMM 本体
/// 側の `DimProductOverflow` 検証とは独立にここでも安全側へ倒す）。
pub(crate) fn should_route_to_affinity_pool(m: usize, n: usize, k: usize) -> bool {
    let work = m.saturating_mul(n).saturating_mul(k);
    work > 0 && work <= GB10_AFFINITY_MAX_WORK
}

/// [`crate::gemm_blis::gemm_blis_parallel_with_transpose`]・
/// [`crate::gemm_blis::gemm_blis_bias_act_parallel`] の GEMM 本体
/// （2D 動的分配／行パネル分割いずれか）を包むルーティングヘルパ。
///
/// [`should_route_to_affinity_pool`] が対象と判定し、かつ
/// [`affinity_pool`] が実際にプールを構築できた場合のみ `f` をその
/// 専用プール上で `install` する。ゲート OFF（既定）・
/// `RAYON_NUM_THREADS` 明示設定・プラットフォーム判定失敗・
/// 大形状のいずれでも `f()` を直接呼び出し、現行の global pool 実行と
/// 完全に同一の挙動になる（bit 完全一致契約はこの分岐が既存 GEMM
/// ロジックへ一切手を加えないことにより成立する）。
pub(crate) fn with_gb10_affinity_if_applicable<F, R>(m: usize, n: usize, k: usize, f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    if should_route_to_affinity_pool(m, n, k)
        && let Some(pool) = affinity_pool()
    {
        return pool.install(f);
    }
    f()
}

/// GEMM 専用の大コア pin 済み `rayon::ThreadPool`（モジュール doc
/// 「専用スレッドプール」節参照）。初回呼び出しで 1 回だけ構築し
/// 以降はキャッシュを再利用する（プラットフォーム I/O・スレッド生成の
/// コストを毎呼び出しに乗せない。[`crate::thread_limit::DECISION`] と
/// 同じ `OnceLock` パターン）。構築失敗（ゲート OFF・環境変数明示・
/// 判定不能・`ThreadPoolBuilder::build` 自体の失敗）はすべて `None` を
/// キャッシュし、以降の呼び出しは常に `f()` 直呼びへフォールバックする。
fn affinity_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(build_affinity_pool).as_ref()
}

fn build_affinity_pool() -> Option<rayon::ThreadPool> {
    if !GB10_AFFINITY_ENABLED || env_override_active() {
        return None;
    }
    let big_core_ids = detect_big_core_ids()?;
    if big_core_ids.is_empty() {
        return None;
    }
    // `start_handler` は各 worker スレッド自身の上で 1 回だけ実行される
    // （rayon の契約）ため、`worker_idx` 番目の要素を当該スレッドへ
    // pin すれば重複なく大コア群全体を専用プールへ割り当てられる。
    let ids = big_core_ids;
    rayon::ThreadPoolBuilder::new()
        .num_threads(ids.len())
        .start_handler(move |worker_idx| {
            if let Some(&cpu_id) = ids.get(worker_idx) {
                let _ = pin_current_thread_to_cpu(cpu_id);
            }
        })
        .build()
        .ok()
}

/// `RAYON_NUM_THREADS` が有効な正整数として設定されているかを判定する
/// （[`crate::thread_limit::parse_env_num_threads`] と同じ parse 規則を
/// 共有し、rayon 自身の解釈・[`crate::thread_limit`] の判定と齟齬を
/// 生まない）。
fn env_override_active() -> bool {
    crate::thread_limit::parse_env_num_threads(std::env::var("RAYON_NUM_THREADS").ok().as_deref())
        .is_some()
}

/// 診断・#1576 の env_info 記録用に判定結果全体を可視化する構造体
/// （[`gb10_affinity_report`] の戻り値）。`facade` 等の公開 API 面へは
/// 昇格しない（`.claude/rules/delegation-impl.md`「禁止事項」・本イシュー
/// 計画「スコープ外」節）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gb10AffinityReport {
    /// [`GB10_AFFINITY_ENABLED`] の値。
    pub enabled: bool,
    /// `RAYON_NUM_THREADS` が有効な正整数として設定されているか。
    pub env_override: bool,
    /// プラットフォーム判定で検出した大コア CPU-id 集合
    /// （判定不能なら `None`）。
    pub detected_big_core_ids: Option<Vec<usize>>,
    /// 専用 affinity プールが実際に活性化しているか
    /// （`enabled && !env_override && detected_big_core_ids.is_some()`
    /// と同値）。
    pub pool_active: bool,
}

/// #1576 の env_info 記録・診断用に判定結果全体を可視化する
/// （本番 GEMM 経路からは呼ばれない。[`crate::thread_limit::
/// thread_limit_report`] と同型）。
pub fn gb10_affinity_report() -> Gb10AffinityReport {
    let enabled = GB10_AFFINITY_ENABLED;
    let env_override = env_override_active();
    let detected_big_core_ids = detect_big_core_ids();
    let pool_active = enabled && !env_override && detected_big_core_ids.is_some();
    Gb10AffinityReport {
        enabled,
        env_override,
        detected_big_core_ids,
        pool_active,
    }
}

// ---------------------------------------------------------------------
// 検出（プラットフォーム判定）
// ---------------------------------------------------------------------

/// 2 つの独立指標（周波数・MIDR partnum）から `(cpu_id, value)` の列を
/// 受け取り、値ごとにちょうど 2 群へ分かれる場合のみ
/// `(高い値の群, 低い値の群)`（各群は CPU-id の昇順ソート済み `Vec`）を
/// 返す純関数。1 群のみ（同種コア構成）・3 群以上（想定外の分布）・
/// 空入力はすべて `None`（判定不能）。
///
/// [`crate::thread_limit::big_cores_from_capacities`] と異なり、本関数
/// は値の**大小**ではなく「ちょうど 2 群」という構造のみを見る（大コア
/// 判定自体は周波数群の方を用い、本関数は周波数・MIDR 両方の分割検査に
/// 共有する）。
#[cfg(any(target_os = "linux", test))]
fn partition_into_two_groups(entries: &[(usize, u64)]) -> Option<(Vec<usize>, Vec<usize>)> {
    if entries.is_empty() {
        return None;
    }
    let mut distinct: Vec<u64> = entries.iter().map(|(_, v)| *v).collect();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() != 2 {
        return None;
    }
    let high_value = distinct[1];
    let low_value = distinct[0];
    let mut high: Vec<usize> = entries
        .iter()
        .filter(|(_, v)| *v == high_value)
        .map(|(id, _)| *id)
        .collect();
    let mut low: Vec<usize> = entries
        .iter()
        .filter(|(_, v)| *v == low_value)
        .map(|(id, _)| *id)
        .collect();
    high.sort_unstable();
    low.sort_unstable();
    Some((high, low))
}

/// `regs/identification/midr_el1` の 16 進文字列（`0x` 接頭辞の有無を
/// 問わない）から ARM MIDR_EL1 の partnum フィールド（ビット
/// `[15:4]`。12 bit 幅）を抽出する純関数。parse 失敗は `None`。
#[cfg(any(target_os = "linux", test))]
fn parse_midr_partnum(raw: &str) -> Option<u64> {
    let trimmed = raw.trim();
    let hex = trimmed.strip_prefix("0x").unwrap_or(trimmed);
    let value = u64::from_str_radix(hex, 16).ok()?;
    Some((value >> 4) & 0xFFF)
}

/// `/sys/devices/system/cpu` 配下（`root` 引数はテスト用 fixture 差し替え
/// のため）を走査し、`cpufreq/cpuinfo_max_freq`・
/// `regs/identification/midr_el1` の両方が揃っている CPU のみを対象に
/// `(cpu_id, freq)`・`(cpu_id, midr_partnum)` の列を組み立てる純関数。
/// 1 つでも欠損・parse 不能なコードパスに当たれば `None`
/// （[`crate::thread_limit::big_cores_from_sysfs`] と同じ fail-safe
/// 方針: 部分的な判定は誤った大コア群を導きうるため全体を判定不能に
/// 倒す）。`cpufreq`／`cpuidle`／`cpu-map` 等のデコイディレクトリは
/// `strip_prefix("cpu")` 後に数字へ parse できないため自然に除外される。
#[cfg(any(target_os = "linux", test))]
fn collect_freq_and_midr(root: &Path) -> Option<(Vec<(usize, u64)>, Vec<(usize, u64)>)> {
    let read_dir = std::fs::read_dir(root).ok()?;
    let mut freqs: Vec<(usize, u64)> = Vec::new();
    let mut midrs: Vec<(usize, u64)> = Vec::new();
    for entry in read_dir {
        let entry = entry.ok()?;
        let name = entry.file_name();
        let name = name.to_str()?;
        let Some(num_str) = name.strip_prefix("cpu") else {
            continue;
        };
        let Ok(cpu_id) = num_str.parse::<usize>() else {
            continue;
        };
        let freq_raw =
            std::fs::read_to_string(entry.path().join("cpufreq/cpuinfo_max_freq")).ok()?;
        let freq: u64 = freq_raw.trim().parse().ok()?;
        let midr_raw =
            std::fs::read_to_string(entry.path().join("regs/identification/midr_el1")).ok()?;
        let partnum = parse_midr_partnum(&midr_raw)?;
        freqs.push((cpu_id, freq));
        midrs.push((cpu_id, partnum));
    }
    if freqs.is_empty() {
        return None;
    }
    Some((freqs, midrs))
}

/// 2 つの CPU-id 集合（走査順は問わない。呼び出し元で昇順ソート済み）が
/// 同一の要素集合を表すかを判定する。
#[cfg(any(target_os = "linux", test))]
fn same_id_set(a: &[usize], b: &[usize]) -> bool {
    a == b
}

/// [`collect_freq_and_midr`] の結果を周波数・MIDR partnum それぞれで
/// [`partition_into_two_groups`] にかけ、両者の分割が（群の順序に依らず）
/// 完全一致する場合のみ、周波数の高い方の群（大コア群。X925 は A725 より
/// 高クロック）を `Some` で返す純関数。`root` はテスト用 fixture
/// 差し替えのため引数化する（モジュール doc「大コア判定」節の
/// クロスバリデーション仕様）。
#[cfg(any(target_os = "linux", test))]
fn big_core_ids_from_sysfs(root: &Path) -> Option<Vec<usize>> {
    let (freqs, midrs) = collect_freq_and_midr(root)?;
    let (freq_high, freq_low) = partition_into_two_groups(&freqs)?;
    let (midr_a, midr_b) = partition_into_two_groups(&midrs)?;
    let matches = (same_id_set(&freq_high, &midr_a) && same_id_set(&freq_low, &midr_b))
        || (same_id_set(&freq_high, &midr_b) && same_id_set(&freq_low, &midr_a));
    if matches { Some(freq_high) } else { None }
}

/// `/proc/self/status` の `Cpus_allowed_list:` 行（例: `0-19` や
/// `5-9,15-19`）を parse し、許可された CPU-id の集合を返す純関数。
/// 行が無い・値が parse できない場合は `None`（cgroup／cpuset 制約を
/// 検証できない以上、安全側〈大コア判定を採用しない〉へ倒すため
/// 呼び出し元は `None` を「制約検査失敗」として扱う）。
#[cfg(any(target_os = "linux", test))]
fn parse_cpus_allowed_list(text: &str) -> Option<Vec<usize>> {
    let line = text
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))?;
    let mut ids = Vec::new();
    for part in line.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            let start: usize = start.trim().parse().ok()?;
            let end: usize = end.trim().parse().ok()?;
            if start > end {
                return None;
            }
            ids.extend(start..=end);
        } else {
            ids.push(part.parse().ok()?);
        }
    }
    if ids.is_empty() { None } else { Some(ids) }
}

/// 検出した大コア CPU-id 集合が `/proc/self/status` の
/// `Cpus_allowed_list:` の部分集合であることを確認する。`status_path`
/// はテスト用 fixture 差し替えのため引数化する。読み取り・parse に
/// 失敗した場合は制約を検証できないため `false`（安全側。
/// [`big_core_ids_within_allowed`] ドキュメント参照）。
#[cfg(any(target_os = "linux", test))]
fn big_core_ids_within_allowed(big_core_ids: &[usize], status_path: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(status_path) else {
        return false;
    };
    let Some(allowed) = parse_cpus_allowed_list(&text) else {
        return false;
    };
    big_core_ids.iter().all(|id| allowed.contains(id))
}

#[cfg(target_os = "linux")]
fn detect_big_core_ids() -> Option<Vec<usize>> {
    let big_core_ids = big_core_ids_from_sysfs(Path::new("/sys/devices/system/cpu"))?;
    if big_core_ids_within_allowed(&big_core_ids, Path::new("/proc/self/status")) {
        Some(big_core_ids)
    } else {
        None
    }
}

#[cfg(not(target_os = "linux"))]
fn detect_big_core_ids() -> Option<Vec<usize>> {
    None
}

// ---------------------------------------------------------------------
// Affinity 設定（unsafe FFI 境界）
// ---------------------------------------------------------------------

/// [`build_affinity_pool`]（cfg 非分岐の共通関数）の `start_handler` から
/// 呼ばれるプラットフォーム振り分け薄いラッパ。Linux では
/// [`affinity_ffi::pin_current_thread_to_cpu`] へ委譲し、それ以外の
/// プラットフォームでは常に `false`（no-op）を返す（[`detect_big_core_ids`]
/// が非 Linux で常に `None` を返すため、この分岐が実際に呼ばれる経路は
/// 到達しないが、`start_handler` クロージャ自体は cfg 分岐せずコンパイル
/// されるため、シグネチャを全プラットフォームで揃える必要がある）。
#[cfg(target_os = "linux")]
fn pin_current_thread_to_cpu(cpu_id: usize) -> bool {
    affinity_ffi::pin_current_thread_to_cpu(cpu_id)
}

#[cfg(not(target_os = "linux"))]
fn pin_current_thread_to_cpu(_cpu_id: usize) -> bool {
    false
}

/// `sched_setaffinity(2)`（Linux glibc が提供する標準 syscall ラッパー）
/// による、呼び出しスレッド自身の CPU affinity 設定
/// （`cfg(target_os = "linux")` 限定）。
///
/// 追加クレート依存を使わない理由:
/// `libc`／`core_affinity` は許容 9 区分（`.claude/rules/deps-policy.md`）
/// 外であり、Rust std は glibc に動的リンクしているため
/// `crates/backend-cpu/src/gemm_blis/cache_params.rs::sysctl_ffi`
/// （macOS `sysctlbyname`）と同方針で `extern "C"` の自前宣言で足りる。
#[cfg(target_os = "linux")]
mod affinity_ffi {
    use std::os::raw::c_int;

    /// glibc 既定の `cpu_set_t`（`CPU_SETSIZE == 1024` bit）を
    /// `[u64; 16]`（1024 / 64 = 16）として手動再現する。ビット `i` が
    /// セットされていれば CPU `i` を許可する（`CPU_SET(3)` と同じ
    /// レイアウト）。
    pub(super) type CpuSet = [u64; 16];

    /// `cpu_set_t` が表現可能な最大 CPU 番号（`CPU_SETSIZE`）。
    pub(super) const CPU_SETSIZE: usize = CpuSet::len_bits();

    trait CpuSetLenBits {
        fn len_bits() -> usize;
    }
    impl CpuSetLenBits for CpuSet {
        fn len_bits() -> usize {
            16 * 64
        }
    }

    // SAFETY: この `extern "C"` 宣言は Linux glibc が公開する標準 API
    // `sched_setaffinity`（`<sched.h>`、`man 2 sched_setaffinity`）の
    // シグネチャと一致させている: 戻り値は `c_int`（0 は成功、非 0 は
    // エラー。errno 相当）、`pid`（0 は「呼び出しスレッド自身」を指す。
    // rayon の `start_handler` は各 worker スレッド上で実行されるため
    // `pid=0` で当該スレッドのみに適用される）・`cpusetsize`（`mask` の
    // バイト長）・`mask`（読み取り専用の CPU 集合ビットマスクへの
    // ポインタ）で、C ABI 上の型幅・呼び出し規約（`extern "C"`）は
    // glibc のヘッダ定義と 1:1 対応する。シンボルは全 Linux 実行環境で
    // 常にリンクされる glibc が提供するため動的ロード不要で解決可能
    // （`cfg(target_os = "linux")` 限定でのみコンパイルされ、他 OS では
    // 宣言自体が存在しない）。個々の呼び出し引数の安全性（ポインタ
    // 有効性・長さ整合）は呼び出し側 [`pin_current_thread_to_cpu`] の
    // SAFETY コメントを参照。`unsafe` はこの 1 箇所に限定する
    // （`.claude/rules/coding-rust.md`「`unsafe` は FFI 境界等の必要
    // 最小限に留め、理由をコメントで明記」）。
    unsafe extern "C" {
        fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const CpuSet) -> c_int;
    }

    /// 呼び出しスレッド自身（`pid=0`）の CPU affinity を `cpu_id` 1 個
    /// のみへ設定する。`cpu_id` が [`CPU_SETSIZE`] 以上、または syscall
    /// 自体が失敗（戻り値 != 0。例: 対象 CPU が現在の cpuset で許可
    /// されていない）した場合は `false` を返すのみで panic しない
    /// （呼び出し元 [`super::build_affinity_pool`] は戻り値を無視し
    /// affinity 未設定のまま続行する。`.claude/rules/coding-rust.md`
    /// 「本番経路で `unwrap()` / `expect()` を使わない」）。
    pub(super) fn pin_current_thread_to_cpu(cpu_id: usize) -> bool {
        if cpu_id >= CPU_SETSIZE {
            return false;
        }
        let mut mask: CpuSet = [0u64; 16];
        mask[cpu_id / 64] |= 1u64 << (cpu_id % 64);
        // SAFETY: `mask` はこの呼び出しの生存期間中有効なスタック上の
        // 固定長配列（`[u64; 16]` = 128 バイト）。`cpusetsize` に
        // `size_of::<CpuSet>()`（128 バイト）を渡し `mask` の実サイズと
        // 一致させるため、`sched_setaffinity` がバッファ長を超えて
        // 読み取ることはない。`pid=0` は「呼び出しスレッド自身」を
        // 指す既定契約（`man 2 sched_setaffinity`）で、他スレッド・
        // 他プロセスの状態を変更しない。戻り値は呼び出し直後に検査し
        // エラーは無視して continue する（fail-safe。上記ドキュメント
        // コメント参照）。`unsafe` はこの 1 箇所に限定する。
        let ret = unsafe {
            sched_setaffinity(0, std::mem::size_of::<CpuSet>(), std::ptr::from_ref(&mask))
        };
        ret == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- should_route_to_affinity_pool ---------------------------------

    #[test]
    fn should_route_small_shape_true() {
        // MLP layer1 相当（診断対象形状）。
        assert!(should_route_to_affinity_pool(64, 256, 784));
    }

    #[test]
    fn should_route_large_square_false() {
        // 非対象の正方 GEMM 最小形状（N=512）。
        assert!(!should_route_to_affinity_pool(512, 512, 512));
    }

    #[test]
    fn should_route_zero_work_false() {
        assert!(!should_route_to_affinity_pool(0, 10, 10));
        assert!(!should_route_to_affinity_pool(10, 0, 10));
        assert!(!should_route_to_affinity_pool(10, 10, 0));
    }

    #[test]
    fn should_route_boundary_exact_threshold_true() {
        // m*n*k がちょうど閾値と一致する境界値は「以下」なので対象。
        let (m, n, k) = (1usize, 1usize, GB10_AFFINITY_MAX_WORK);
        assert!(should_route_to_affinity_pool(m, n, k));
        assert!(!should_route_to_affinity_pool(m, n, k + 1));
    }

    #[test]
    fn should_route_overflow_saturates_to_false() {
        assert!(!should_route_to_affinity_pool(usize::MAX, usize::MAX, 2));
    }

    // --- with_gb10_affinity_if_applicable（ゲート既定 OFF の契約）------

    #[test]
    fn with_affinity_gate_disabled_calls_closure_directly() {
        // GB10_AFFINITY_ENABLED は既定 false のため、対象形状であっても
        // affinity_pool() は常に None を返し f() が直接呼ばれる。
        assert!(!GB10_AFFINITY_ENABLED);
        let result = with_gb10_affinity_if_applicable(64, 256, 784, || 42);
        assert_eq!(result, 42);
    }

    #[test]
    fn with_affinity_large_shape_calls_closure_directly() {
        let result = with_gb10_affinity_if_applicable(4096, 4096, 4096, || 7);
        assert_eq!(result, 7);
    }

    // --- partition_into_two_groups --------------------------------------

    #[test]
    fn partition_two_groups_ok() {
        let entries: Vec<(usize, u64)> = (0..10)
            .map(|i| (i, 3900))
            .chain((10..20).map(|i| (i, 2808)))
            .collect();
        let (high, low) = partition_into_two_groups(&entries).expect("two groups");
        assert_eq!(high, (0..10).collect::<Vec<_>>());
        assert_eq!(low, (10..20).collect::<Vec<_>>());
    }

    #[test]
    fn partition_uniform_returns_none() {
        let entries: Vec<(usize, u64)> = (0..20).map(|i| (i, 3900)).collect();
        assert_eq!(partition_into_two_groups(&entries), None);
    }

    #[test]
    fn partition_three_groups_returns_none() {
        let entries: Vec<(usize, u64)> = vec![(0, 1), (1, 2), (2, 3)];
        assert_eq!(partition_into_two_groups(&entries), None);
    }

    #[test]
    fn partition_empty_returns_none() {
        assert_eq!(partition_into_two_groups(&[]), None);
    }

    // --- parse_midr_partnum ----------------------------------------------

    #[test]
    fn parse_midr_partnum_x925_big() {
        // GB10 実機実測相当の MIDR_EL1 生値（implementer=0x41〈ARM〉・
        // variant=2・archtype=f・partnum=0xd85〈X925〉・revision=1）。
        // partnum フィールドはビット [15:4] に位置するため、下位 1 桁
        // （revision）を挟んだ `...d851` から `>> 4 & 0xfff` で 0xd85 を
        // 取り出せることを確認する。
        assert_eq!(parse_midr_partnum("0x412fd851"), Some(0xd85));
    }

    #[test]
    fn parse_midr_partnum_a725_little() {
        // GB10 実機実測相当の MIDR_EL1 生値（partnum=0xd87〈A725〉）。
        assert_eq!(parse_midr_partnum("0x412fd870"), Some(0xd87));
    }

    #[test]
    fn parse_midr_partnum_no_prefix() {
        assert_eq!(parse_midr_partnum("412fd851"), Some(0xd85));
    }

    #[test]
    fn parse_midr_partnum_invalid() {
        assert_eq!(parse_midr_partnum("not-hex"), None);
        assert_eq!(parse_midr_partnum(""), None);
    }

    // --- fixture ベースの sysfs 走査 --------------------------------------

    /// テストごとに一意な一時ディレクトリを作り、GB10 実機の sysfs
    /// レイアウト（`cpu<N>/cpufreq/cpuinfo_max_freq`・
    /// `cpu<N>/regs/identification/midr_el1`）を fixture として書き込む
    /// （`crate::thread_limit::tests::SysfsFixture` と同型。依存追加なし）。
    struct SysfsFixture {
        root: std::path::PathBuf,
    }

    impl SysfsFixture {
        fn new(name: &str) -> Self {
            let root = std::env::temp_dir().join(format!(
                "fandhe-ai-gb10-affinity-test-{name}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0),
            ));
            std::fs::create_dir_all(&root).expect("create fixture root");
            Self { root }
        }

        fn write_cpu(&self, cpu: usize, freq: &str, midr: &str) {
            let dir = self.root.join(format!("cpu{cpu}"));
            std::fs::create_dir_all(dir.join("cpufreq")).expect("create cpufreq dir");
            std::fs::write(dir.join("cpufreq/cpuinfo_max_freq"), freq)
                .expect("write cpuinfo_max_freq");
            std::fs::create_dir_all(dir.join("regs/identification"))
                .expect("create regs/identification dir");
            std::fs::write(dir.join("regs/identification/midr_el1"), midr).expect("write midr_el1");
        }
    }

    impl Drop for SysfsFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    fn write_gb10_like(fx: &SysfsFixture) {
        // X925（big）: cpu0-9・3900MHz・partnum 0xd85。
        for i in 0..10 {
            fx.write_cpu(i, "3900000\n", "0x412fd851\n");
        }
        // A725（little）: cpu10-19・2808MHz・partnum 0xd87。
        for i in 10..20 {
            fx.write_cpu(i, "2808000\n", "0x412fd870\n");
        }
    }

    #[test]
    fn big_core_ids_from_sysfs_gb10_like_cross_validates() {
        let fx = SysfsFixture::new("gb10-like");
        write_gb10_like(&fx);
        let big = big_core_ids_from_sysfs(&fx.root).expect("cross-validated big core group");
        assert_eq!(big, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn big_core_ids_from_sysfs_uniform_returns_none() {
        let fx = SysfsFixture::new("uniform");
        for i in 0..20 {
            fx.write_cpu(i, "3900000\n", "0x412fd851\n");
        }
        assert_eq!(big_core_ids_from_sysfs(&fx.root), None);
    }

    #[test]
    fn big_core_ids_from_sysfs_missing_midr_returns_none() {
        let fx = SysfsFixture::new("missing-midr");
        // cpu0 は cpufreq のみで regs/identification を持たない（欠損）。
        let dir = fx.root.join("cpu0/cpufreq");
        std::fs::create_dir_all(&dir).expect("create cpufreq dir");
        std::fs::write(dir.join("cpuinfo_max_freq"), "3900000\n").expect("write freq");
        assert_eq!(big_core_ids_from_sysfs(&fx.root), None);
    }

    #[test]
    fn big_core_ids_from_sysfs_mismatched_partition_returns_none() {
        // 周波数は 10/10 で 2 群に分かれるが、MIDR partnum の分割点が
        // ずれている（cpu9 だけ big 側の周波数なのに little の partnum）
        // ケース。クロスバリデーションが不一致を検出し None を返す。
        let fx = SysfsFixture::new("mismatched");
        for i in 0..9 {
            fx.write_cpu(i, "3900000\n", "0x412fd851\n");
        }
        fx.write_cpu(9, "3900000\n", "0x412fd870\n"); // 周波数は big・MIDR は little
        for i in 10..20 {
            fx.write_cpu(i, "2808000\n", "0x412fd870\n");
        }
        assert_eq!(big_core_ids_from_sysfs(&fx.root), None);
    }

    #[test]
    fn big_core_ids_from_sysfs_empty_dir_returns_none() {
        let fx = SysfsFixture::new("empty");
        assert_eq!(big_core_ids_from_sysfs(&fx.root), None);
    }

    #[test]
    fn big_core_ids_from_sysfs_nonexistent_root_returns_none() {
        let root = std::env::temp_dir().join("fandhe-ai-gb10-affinity-test-nonexistent-xyz");
        assert_eq!(big_core_ids_from_sysfs(&root), None);
    }

    // --- parse_cpus_allowed_list ------------------------------------------

    #[test]
    fn parse_cpus_allowed_list_range() {
        let text = "Other: 1\nCpus_allowed_list:\t0-19\nMore: 2\n";
        assert_eq!(
            parse_cpus_allowed_list(text),
            Some((0..=19).collect::<Vec<_>>())
        );
    }

    #[test]
    fn parse_cpus_allowed_list_mixed_ranges() {
        let text = "Cpus_allowed_list:\t5-9,15-19\n";
        let mut expected: Vec<usize> = (5..=9).collect();
        expected.extend(15..=19);
        assert_eq!(parse_cpus_allowed_list(text), Some(expected));
    }

    #[test]
    fn parse_cpus_allowed_list_single_values() {
        let text = "Cpus_allowed_list:\t0,2,4\n";
        assert_eq!(parse_cpus_allowed_list(text), Some(vec![0, 2, 4]));
    }

    #[test]
    fn parse_cpus_allowed_list_missing_line_returns_none() {
        let text = "Other: 1\n";
        assert_eq!(parse_cpus_allowed_list(text), None);
    }

    #[test]
    fn parse_cpus_allowed_list_invalid_returns_none() {
        assert_eq!(parse_cpus_allowed_list("Cpus_allowed_list:\tabc\n"), None);
        assert_eq!(parse_cpus_allowed_list("Cpus_allowed_list:\t9-5\n"), None);
    }

    // --- big_core_ids_within_allowed ---------------------------------------

    #[test]
    fn within_allowed_subset_true() {
        let fx = SysfsFixture::new("allowed-subset");
        let status_path = fx.root.join("status");
        std::fs::write(&status_path, "Cpus_allowed_list:\t0-19\n").expect("write status");
        assert!(big_core_ids_within_allowed(
            &(0..10).collect::<Vec<_>>(),
            &status_path
        ));
    }

    #[test]
    fn within_allowed_not_subset_false() {
        let fx = SysfsFixture::new("allowed-not-subset");
        let status_path = fx.root.join("status");
        // 大コア群 (0..10) の一部 (cpu9) が許可集合から外れているケース。
        std::fs::write(&status_path, "Cpus_allowed_list:\t0-8,10-19\n").expect("write status");
        assert!(!big_core_ids_within_allowed(
            &(0..10).collect::<Vec<_>>(),
            &status_path
        ));
    }

    #[test]
    fn within_allowed_missing_file_false() {
        let missing = std::env::temp_dir().join("fandhe-ai-gb10-affinity-test-no-status-xyz");
        assert!(!big_core_ids_within_allowed(&[0, 1], &missing));
    }

    #[test]
    fn within_allowed_unparseable_false() {
        let fx = SysfsFixture::new("allowed-unparseable");
        let status_path = fx.root.join("status");
        std::fs::write(&status_path, "no such line here\n").expect("write status");
        assert!(!big_core_ids_within_allowed(&[0, 1], &status_path));
    }

    // --- gb10_affinity_report（統合。プラットフォーム非依存の契約）-------

    #[test]
    fn report_reflects_disabled_gate() {
        let report = gb10_affinity_report();
        assert_eq!(report.enabled, GB10_AFFINITY_ENABLED);
        assert!(!report.pool_active); // ゲート既定 OFF のため常に不活性。
    }
}
