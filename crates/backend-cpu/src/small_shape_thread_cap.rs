//! 小形状 GEMM の仕事量ベース rayon 並列度上限（イシュー #1575・親ツリー
//! `docs/perf/lowlayer-diagnosis-2026-09-12.md` §3・§7 A-1a）。
//!
//! ## 背景・目的
//!
//! 低レイヤー診断（#1574）で、Apple M4 Max の batch 64 学習・推論
//! （小形状 GEMM 主体。例: 学習の 64×256×784・64×10×256 等）は、プロセス
//! 全体のスレッド数をグローバルプールの既定値（16）から `RAYON_NUM_THREADS=4`
//! に絞ると train 約 1.47 倍・infer 約 1.30 倍速いことが観測された
//! （小形状では並列化の恩恵よりワーカー起床・idle spin のコストが上回る
//! 仮説）。
//!
//! 本モジュールは [`crate::thread_limit`]（コア種別 `cpu_capacity` に基づく
//! 大コア数上限。#1363/#1364 実測により既定 `false`）とは**独立の別機構**
//! として、**M×N×K の仕事量**のみに基づき、小形状 GEMM を専用の小さい
//! rayon スレッドプールで実行する。コア種別判定を経由しないため、
//! GB10（DGX Spark）の `cpu_capacity` 誤検出（#1364）のような後退経路を
//! 構造的に持たない。
//!
//! [`crate::gemm_blis::GEMM_THREADING_THRESHOLD`]／`should_serialize`
//! （#811/#1027）とも別機構である: あちらは「並列 GEMM を捨てて T=1 で
//! 直列実行する」の二値判定（`#[cfg(test)]` 限定・本番未結線）であり、
//! 本モジュールは「グローバルプールより小さい専用プールで**引き続き
//! 並列**実行する」ことで、job 数を絞るだけでは解消しない
//! グローバルプールの余剰ワーカー起床コスト（[`crate::thread_limit`] の
//! `effective_num_threads` が [`crate::gemm_blis::partition::job_grid`] へ
//! 渡す `num_threads` を絞るだけでは、`jobs.par_iter_mut()` が依然として
//! グローバルプール〈16 スレッド〉上で走るため、余剰 12 ワーカーの
//! wake-up／idle spin コストは残る）を狙って解消する。
//!
//! ## 自機判定（M4 Max 限定）
//!
//! P/E 非対称構成（`hw.perflevel0.logicalcpu` が `hw.logicalcpu` より
//! 真に小さい。[`crate::thread_limit::detect_big_cores`] と同じ sysctl
//! 経由の判定だが、本モジュールは判定結果の**値**〈大コア数〉ではなく
//! 「非対称構成であるか」のみを見る）に加え、`machdep.cpu.brand_string`
//! が allowlist（現在 `"Apple M4 Max"` のみ）に完全一致する場合にのみ
//! 有効化する。allowlist の拡張は他の Apple Silicon 実機での個別実測が
//! 前提であり、本 PR のスコープ外とする（イシュー #1575 実装計画
//! §9「スコープ外」）。
//!
//! macOS 以外・brand 判定不能・非対称構成でない場合は常に無効
//! （`eligible() == false`）。
//!
//! ## `RAYON_NUM_THREADS` との関係
//!
//! [`crate::thread_limit`] と同じ契約: `RAYON_NUM_THREADS` が有効な正の
//! 整数として設定されている場合は明示指定を尊重し、専用プールへは
//! 切り替えない（rayon の採用値をそのまま使う）。
//!
//! ## 呼び出し元
//!
//! [`run_capped`] は [`crate::gemm_blis::gemm_blis_parallel_with_transpose`]・
//! [`crate::gemm_blis::gemm_blis_bias_act_parallel`]（本番公開入口。2D 動的
//! 分配〈`dispatch_two_d_dynamic`〉呼び出しのみを包む）から呼ばれる。
//!
//! ## bit 完全一致契約
//!
//! 本機構は**並列度のみ**を変える（専用プールで実行するか・グローバル
//! プールで実行するかの違い）。GEMM カーネル本体（マイクロカーネル・
//! packing・FMA 契約・累積順序）は変更しないため、出力は常にグローバル
//! プール実行と bit 完全一致する（`small_shape_thread_cap::tests::
//! dedicated_pool_execution_matches_global_pool_bit_exact` で検証）。

use std::sync::OnceLock;

/// 本機構全体を有効化する単一ゲート（#1313 `TWO_D_DYNAMIC_PRODUCTION_ENABLED`
/// と同型のロールバック機構）。
///
/// Phase 1（framework-compare 同一バイナリ on/off。事前登録規則は
/// イシュー #1575 コメント参照）の実測結果に基づき確定する。ADOPT な
/// ら `true`・REJECT なら `false` へ 1 行差し戻すだけで済む。
///
/// **M4 Max 実機実測により `false` へ確定済み（REJECT）**: 判定対象
/// 4 セル（train/infer cpu fresh/reuse。各 5 run 中央値・
/// `SMALL_SHAPE_CAP_ENABLED=true` でビルドしたバイナリを
/// `RAYON_NUM_THREADS` の有無で run 単位 interleave 計測）のうち
/// `train:fresh` が `ratio(after/before)=1.0307`（>1.00）で事前登録規則
/// 「1 セルでも >1.00 なら REJECT」に抵触した（`train:reuse=0.9243`・
/// `infer:fresh=0.7706`・`infer:reuse=0.8535` は基準充足）。checksum は
/// 全 10 セル（判定対象 4 ＋参考 6）で完全一致（bit 完全一致契約は
/// 維持）。共有負荷下（record_only。計測中 load average 22〜23）での
/// 単発計測であり `train:fresh` の後退幅（+3%）はノイズ帯の可能性も
/// あるが、事前登録規則の事後緩和は行わない。実測記録は
/// `docs/perf/cpu-gemm-small-shape-thread-cap.md`「Phase 1」節・
/// `docs/perf/logs/cpu-gemm-small-shape-thread-cap-1575/` を参照。
pub(crate) const SMALL_SHAPE_CAP_ENABLED: bool = false;

/// 専用プールのスレッド数（Phase 0 スイープで確定。M4 Max 実機・
/// `examples/small_shape_cap_sweep.rs` の学習 5 形状 5 回中央値実測
/// （`docs/perf/logs/cpu-gemm-small-shape-thread-cap-1575/`）で、
/// `dedicated:{2,4,6,8}` のうち学習 5 形状すべてで off 比 `ratio<=1.00`
/// を満たしたのは `dedicated:6` のみ（`dedicated:2`／`4` は
/// `train_64x256x784_nn` で `ratio>1.00`・`dedicated:8` は同形状で
/// `ratio=1.0272` により不成立）。事前登録規則の選択規則（成立候補中
/// 幾何平均 ratio 最小）は唯一の成立候補のため自動的に確定した
/// （幾何平均 ratio ≈0.62。`docs/perf/cpu-gemm-small-shape-thread-cap.md`
/// §Phase 0 参照）。
pub(crate) const SMALL_SHAPE_CAP_THREADS: usize = 6;

/// cap 対象とする仕事量（`m * n.max(N_CLAMP) * k`）の上限（未満なら cap
/// 対象）。Phase 0 実測（[`SMALL_SHAPE_CAP_THREADS`] ドキュメント参照）で、
/// `dedicated:6` は学習 5 形状（最大仕事量 64×256×784=12,845,056）・
/// 交差確認用正方 128（2,097,152）・256（16,777,216）で `ratio<=1.00`
/// だったが、正方 512（134,217,728）で `ratio=1.0980` と後退へ転じた。
/// 事前登録規則の tie-break（学習最大仕事量を含み・後退開始仕事量を
/// 含まない 2 のべき乗のうち最小＝最も保守的なもの）に従い、
/// 12,845,056 を含み 134,217,728 を含まない最小の 2 のべき乗
/// `1 << 24`（16,777,216）を採用する（`1 << 25` も条件を満たすが
/// tie-break により不採用。正方 256〈16,777,216〉はこの境界値と
/// ちょうど等しいため cap 対象外＝グローバルプールのまま。ratio は
/// 元々改善方向だったため保守的に倒しても後退はしない）。
pub(crate) const SMALL_SHAPE_CAP_MAX_WORK: usize = 1 << 24;

/// 仕事量下限クランプ（[`crate::gemm_blis::NR_CLAMP`] と同じ理由。細長
/// 形状〈例 m=512, n=1, k=512〉の実効仕事量を過小評価しないための下限。
/// `GEMM_THREADING_THRESHOLD` の下限とは独立の定数として持つ）。
const N_CLAMP: usize = 8;

/// M4 Max 実機（`machdep.cpu.brand_string`）の allowlist。他の Apple
/// Silicon 世代への拡張は個別実機実測が前提（本イシューのスコープ外）。
///
/// 呼び出し元（[`eligible_from_probe`]）は macOS 専用 [`detect_eligible`]
/// からのみ実行時に到達するため、非 macOS ビルドでは `#[cfg(test)]` の
/// 単体テスト経由でしか参照されない。実際の到達範囲に合わせて
/// `cfg(any(target_os = "macos", test))` で明示し、非 macOS 非テスト
/// ビルド（Linux CI 等）での `dead_code` 誤検出を避ける。
#[cfg(any(target_os = "macos", test))]
const ELIGIBLE_BRANDS: &[&str] = &["Apple M4 Max"];

/// `m * n.max(N_CLAMP) * k < SMALL_SHAPE_CAP_MAX_WORK` を判定する純関数。
/// 飽和乗算を用い、オーバーフロー時は cap しない（安全側＝並列側）。
pub(crate) fn should_cap(m: usize, n: usize, k: usize) -> bool {
    let work = m.saturating_mul(n.max(N_CLAMP)).saturating_mul(k);
    work < SMALL_SHAPE_CAP_MAX_WORK
}

/// P/E 非対称構成かつ brand が allowlist に完全一致するかを判定する
/// 純関数（プラットフォーム I/O を持たない。実際の sysctl 読み取りは
/// [`detect_eligible`] が行う）。[`ELIGIBLE_BRANDS`] と同じ理由で
/// `cfg(any(target_os = "macos", test))` を付ける（非 macOS 非テスト
/// ビルドでは到達不能なため）。
#[cfg(any(target_os = "macos", test))]
pub(crate) fn eligible_from_probe(
    perflevel0: Option<usize>,
    total: Option<usize>,
    brand: Option<&str>,
) -> bool {
    let asymmetric = matches!((perflevel0, total), (Some(p), Some(t)) if p > 0 && p < t);
    let brand_ok = brand.is_some_and(|b| ELIGIBLE_BRANDS.contains(&b));
    asymmetric && brand_ok
}

/// macOS: `/usr/sbin/sysctl -n machdep.cpu.brand_string` を絶対パス・
/// 固定引数・`env_clear()`（[`crate::thread_limit::read_sysctl`] と同じ
/// 理由。A03 インジェクション対策）で読み取る。stdout の先頭・末尾
/// 空白を除去した文字列をそのまま返す（brand 文字列は allowlist
/// 完全一致でのみ使い、シェル・パスへ連結しない）。
#[cfg(target_os = "macos")]
fn read_sysctl_string(name: &str) -> Option<String> {
    use std::process::Command;
    let output = Command::new("/usr/sbin/sysctl")
        .env_clear()
        .args(["-n", name])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = std::str::from_utf8(&output.stdout).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

#[cfg(target_os = "macos")]
fn detect_eligible() -> bool {
    // 数値 sysctl は [`crate::thread_limit::read_sysctl`]（`pub(crate)`）
    // を再利用する（同じ判定ロジックの重複を避ける）。
    let perflevel0 = crate::thread_limit::read_sysctl("hw.perflevel0.logicalcpu");
    let total = crate::thread_limit::read_sysctl("hw.logicalcpu");
    let brand = read_sysctl_string("machdep.cpu.brand_string");
    eligible_from_probe(perflevel0, total, brand.as_deref())
}

#[cfg(not(target_os = "macos"))]
fn detect_eligible() -> bool {
    false
}

/// 自機判定結果（初回呼び出しで sysctl I/O を 1 回だけ行いキャッシュ
/// する。[`crate::thread_limit::DECISION`] と同じ `OnceLock` 遅延初期化
/// 方針）。
static ELIGIBLE: OnceLock<bool> = OnceLock::new();

/// 専用スレッドプール（初回 cap 発火時にのみ生成する。生成失敗
/// （`unwrap`/`expect` 不使用）・非該当プラットフォームでは `None`
/// を保持し、以降 cap は常に無効として扱う）。
static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();

fn pool() -> Option<&'static rayon::ThreadPool> {
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(SMALL_SHAPE_CAP_THREADS)
            .thread_name(|i| format!("fandhe-ai-small-shape-cap-{i}"))
            .build()
            .ok()
    })
    .as_ref()
}

/// [`crate::thread_limit::parse_env_num_threads`] を再利用し
/// `RAYON_NUM_THREADS` の明示設定有無を判定する。
fn env_override() -> bool {
    crate::thread_limit::parse_env_num_threads(std::env::var("RAYON_NUM_THREADS").ok().as_deref())
        .is_some()
}

/// GEMM 本体（`dispatch_two_d_dynamic` 呼び出し）を、条件を満たす場合
/// のみ専用の小さい rayon プールで実行する。条件を満たさない場合
/// （機構無効・非該当プラットフォーム・`RAYON_NUM_THREADS` 明示設定・
/// 現行プールが既に cap 以下・仕事量が閾値以上・プール生成失敗）は
/// `f()` をそのまま呼ぶ（現行プール内で実行。呼び出し元の契約を変えない）。
///
/// `rayon::current_num_threads() <= SMALL_SHAPE_CAP_THREADS` の判定は
/// 呼び出しごとに評価する（ネストしたプール内から呼ばれる場合など
/// 文脈依存のため、[`OnceLock`] でキャッシュしない）。
///
/// 呼び出し元がグローバルプールのワーカー内から呼んだ場合、
/// `pool.install(f)` はそのワーカースレッドをブロックする（専用プールは
/// グローバルプールを待たない独立プールのためデッドロックしない）。
pub(crate) fn run_capped<R: Send>(m: usize, n: usize, k: usize, f: impl FnOnce() -> R + Send) -> R {
    if !SMALL_SHAPE_CAP_ENABLED
        || !should_cap(m, n, k)
        || env_override()
        || rayon::current_num_threads() <= SMALL_SHAPE_CAP_THREADS
        || !*ELIGIBLE.get_or_init(detect_eligible)
    {
        return f();
    }
    match pool() {
        Some(p) => p.install(f),
        None => f(),
    }
}

/// #1575 の env_info 記録・診断用に判定結果全体を可視化する構造体
/// （[`small_shape_cap_report`] の戻り値）。`facade` 等の公開 API 面へは
/// 昇格しない（`.claude/rules/delegation-impl.md`「禁止事項」・本イシュー
/// 実装計画「スコープ外」節）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmallShapeCapReport {
    /// 呼び出し時点の `rayon::current_num_threads()`。
    pub current: usize,
    /// 自機判定（P/E 非対称かつ brand allowlist 一致）の結果。
    pub eligible: bool,
    /// `RAYON_NUM_THREADS` が有効な正整数として設定されているか。
    pub env_override: bool,
    /// 専用プールのスレッド数（`SMALL_SHAPE_CAP_THREADS`）。
    pub cap_threads: usize,
    /// cap 対象の仕事量上限（`SMALL_SHAPE_CAP_MAX_WORK`）。
    pub max_work: usize,
    /// 専用プールが**既に**生成済みで有効か（`POOL` の現在の状態を
    /// 参照するのみ。診断のために新規生成はしない。codex-review・
    /// Cursor Bugbot 指摘: 機構無効時や非対象環境でも
    /// `small_shape_cap_report` を呼ぶだけで 6 スレッドの専用プールが
    /// 生成・永続化されてしまうと、環境情報の採取自体が計測対象の
    /// スレッド構成を変えてしまう。専用プールは実際に cap が発火した
    /// 呼び出し（[`run_capped`]）でのみ生成する設計を維持するため、
    /// ここでは `POOL.get()` で既存状態のみを読む）。
    pub pool_active: bool,
}

/// [`SmallShapeCapReport`] を構築する（本番 GEMM 経路からは呼ばれず、
/// 診断・ベンチ・env_info 記録専用）。
///
/// `pool_active` は [`POOL`] を初期化しない（`pool()` を呼ばない）。
/// 呼び出しても専用プールを新規生成しないため、機構無効時・
/// 非対象プラットフォームで診断のためだけにスレッドが立つことはない。
pub fn small_shape_cap_report() -> SmallShapeCapReport {
    SmallShapeCapReport {
        current: rayon::current_num_threads().max(1),
        eligible: *ELIGIBLE.get_or_init(detect_eligible),
        env_override: env_override(),
        cap_threads: SMALL_SHAPE_CAP_THREADS,
        max_work: SMALL_SHAPE_CAP_MAX_WORK,
        pool_active: POOL.get().is_some_and(|p| p.is_some()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_cap_below_threshold() {
        // 学習形状 64x256x784 = 12,845,056 < 1<<24 (16,777,216)。
        assert!(should_cap(64, 256, 784));
    }

    #[test]
    fn should_cap_above_threshold() {
        // 512^3 = 134,217,728 >= 1<<24。
        assert!(!should_cap(512, 512, 512));
    }

    #[test]
    fn should_cap_n_clamp_applies() {
        // n=1 は N_CLAMP=8 にクランプされる: m*8*k で判定。
        // 4096*8*4096 = 134,217,728 >= 1<<24 → cap しない。
        assert!(!should_cap(4096, 1, 4096));
        // 64*8*64 = 32,768 < 1<<24 → cap する。
        assert!(should_cap(64, 1, 64));
    }

    #[test]
    fn should_cap_overflow_saturates_to_no_cap() {
        assert!(!should_cap(usize::MAX, usize::MAX, usize::MAX));
    }

    #[test]
    fn should_cap_zero_dims() {
        assert!(should_cap(0, 0, 0));
    }

    #[test]
    fn eligible_from_probe_asymmetric_and_matching_brand() {
        assert!(eligible_from_probe(
            Some(12),
            Some(16),
            Some("Apple M4 Max")
        ));
    }

    #[test]
    fn eligible_from_probe_symmetric_returns_false() {
        assert!(!eligible_from_probe(
            Some(16),
            Some(16),
            Some("Apple M4 Max")
        ));
    }

    #[test]
    fn eligible_from_probe_unknown_brand_returns_false() {
        assert!(!eligible_from_probe(
            Some(12),
            Some(16),
            Some("Apple M3 Max")
        ));
    }

    #[test]
    fn eligible_from_probe_missing_data_returns_false() {
        assert!(!eligible_from_probe(None, Some(16), Some("Apple M4 Max")));
        assert!(!eligible_from_probe(Some(12), None, Some("Apple M4 Max")));
        assert!(!eligible_from_probe(Some(12), Some(16), None));
    }

    #[test]
    fn eligible_from_probe_perflevel0_ge_total_returns_false() {
        assert!(!eligible_from_probe(
            Some(20),
            Some(16),
            Some("Apple M4 Max")
        ));
    }

    #[test]
    fn run_capped_disabled_mechanism_calls_f_directly() {
        // SMALL_SHAPE_CAP_ENABLED == false の間は常に f() が現行プール
        // （テストランナーのグローバルプール）で呼ばれる契約を確認する。
        let result = run_capped(64, 256, 784, rayon::current_num_threads);
        assert_eq!(result, rayon::current_num_threads());
    }

    #[test]
    fn run_capped_large_shape_calls_f_directly() {
        // 仕事量が閾値以上なら（機構が有効でも）cap しない。
        let result = run_capped(4096, 4096, 4096, || 42);
        assert_eq!(result, 42);
    }

    /// [`run_capped`] の bit 完全一致契約（モジュール冒頭ドキュメント
    /// コメント「bit 完全一致契約」参照）を、[`SMALL_SHAPE_CAP_ENABLED`]
    /// の値に依存しない形で直接検証する: 専用の小さいプール
    /// （`pool()` と同じスレッド数）内で [`crate::gemm_blis::
    /// gemm_blis_parallel`] を実行した結果が、現行プール（テスト
    /// ランナーのグローバルプール）で実行した結果と bit 完全一致する
    /// ことを確認する。本機構は並列度のみを変え GEMM カーネル本体
    /// （マイクロカーネル・packing・FMA 契約・累積順序）を変更しない
    /// ため、`install` の有無に関わらず出力は一意に定まる契約
    /// （`crate::gemm_blis` 側の `matches_naive_bit_exact_across_thread_pools`
    /// 系テストが検証する不変条件と同種）。
    #[test]
    fn dedicated_pool_execution_matches_global_pool_bit_exact() {
        use bench_harness::rng::Xorshift64Star;

        // 学習形状の代表例（64x256x784）。
        let (m, n, k) = (64usize, 256usize, 784usize);
        let a = Xorshift64Star::new(1).fill_vec(m * k);
        let b = Xorshift64Star::new(2).fill_vec(k * n);

        let mut c_global = vec![0.0f32; m * n];
        crate::gemm_blis::gemm_blis_parallel(&a, &b, &mut c_global, m, n, k)
            .expect("global pool gemm_blis_parallel");

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(SMALL_SHAPE_CAP_THREADS)
            .build()
            .expect("build dedicated test pool");
        let mut c_dedicated = vec![0.0f32; m * n];
        pool.install(|| {
            crate::gemm_blis::gemm_blis_parallel(&a, &b, &mut c_dedicated, m, n, k)
                .expect("dedicated pool gemm_blis_parallel")
        });

        assert_eq!(
            c_global, c_dedicated,
            "専用プール実行とグローバルプール実行が bit 完全一致しない"
        );
    }

    #[test]
    fn small_shape_cap_report_bounds() {
        let report = small_shape_cap_report();
        assert!(report.current >= 1);
        assert_eq!(report.cap_threads, SMALL_SHAPE_CAP_THREADS);
        assert_eq!(report.max_work, SMALL_SHAPE_CAP_MAX_WORK);
        // 非 macOS では構造的に false（M4 Max allowlist に一致しない）。
        #[cfg(not(target_os = "macos"))]
        assert!(!report.eligible);
    }
}
