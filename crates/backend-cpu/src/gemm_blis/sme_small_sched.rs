//! SME 経路の小形状（N=256 級）専用スケジューリング機構（イシュー #4162。#4165 で本番結線）。
//!
//! 役割: SME 経路（[`super::sme_kernel_for_shape`] が `Some`）のうち m・n・k がすべて
//! [`SME_SMALL_SCHED_MAX_DIM`] 以下の形状に限り、(1) 専用 rayon プールのスレッド数
//! [`SME_SMALL_SCHED_THREADS`] と (2) 2D 動的分配の `jobs_per_worker`
//! [`SME_SMALL_SCHED_JPW`] を差し替える。#4162 の A/B が ADOPT(t4-jpw1)
//! （T=4・jpw=1）となり、#4165 で **既定 ON**（2026-11-13 所有者承認）に結線した。
//! 帯外、またはゲートを OFF（無効値）に戻した場合、[`select`] は渡された `jobs_per_worker` と
//! `None` を返し、従来と同一の挙動になる。ロールバックは 3 定数を無効値（false・0・0）へ戻し、
//! 本ファイルのテストを反転する。腕の定義は `docs/perf/logs/sme-n256-sched-ab-4162/arm-*.patch`。
//! SME 非搭載機（GB10 等）は [`super::sme_kernel_for_shape`] が `None` で本機構へ到達しない。
//!
//! 背景: Mac mini（Apple M4・4P+6E）の gemm 256 は SME ON でも PyTorch 比 0.888 で、
//! 小形状では T=4 が最大 GFLOPS で T=10 は 0.90 倍に落ちる実測がある
//! （`docs/perf/cpu-sme-thread-scaling-macmini.md` §2・§3、#3175）。
//!
//! 呼び出し元: aarch64 版 `dispatch_two_d_dynamic` の SME 分岐だけ。NEON・x86・scalar 経路は
//! 通らない。`small_shape_thread_cap`（#1575）とは別機構: あちらは brand allowlist が
//! `"Apple M4 Max"` 限定で Mac mini に効かず、256³ は仕事量しきい値 `1 << 24` 未満にならない。
//!
//! bit 一致の根拠: スレッド数と `jobs_per_worker` は job の大きさと並列度を決めるだけで、
//! 各 C 要素は 1 job の中で同じ FMA 連鎖により計算される（`small_m.rs` 冒頭・2D 設計
//! `docs/cpu-gemm-2d-dynamic-partition-design.md` §3 条件 6 と同じ）。FMA 契約・tolerance は不変。
//!
//! 新しい `unsafe`・`asm!` は持たない。専用プールの生成失敗は既定経路へ戻る（panic しない）。

#[cfg(any(target_arch = "aarch64", test))]
use std::sync::OnceLock;

#[cfg(any(target_arch = "aarch64", test))]
use super::TWO_D_JOBS_PER_WORKER;

/// SME 適格しきい値（`super::SME_MIN_M/N/K` と同値。aarch64 限定定数のため非 aarch64 の
/// lib テストでも使えるよう複製し、aarch64 では一致をテストで固定する）。
#[cfg(any(target_arch = "aarch64", test))]
const BAND_MIN: (usize, usize, usize) = (256, 256, 64);

/// 本機構の単一ゲート（既定 ON。#4162 ADOPT(t4-jpw1)・#4165 で結線。ロールバックは `false` に戻す）。
#[cfg(any(target_arch = "aarch64", test))]
pub(super) const SME_SMALL_SCHED_ENABLED: bool = true;

/// 専用プールのスレッド数（ADOPT(t4-jpw1) の 4）。0 は「今のプールのまま」。
#[cfg(any(target_arch = "aarch64", test))]
pub(super) const SME_SMALL_SCHED_THREADS: usize = 4;

/// 差し替える `jobs_per_worker`（ADOPT(t4-jpw1) の 1）。0 は「渡された値のまま」。
#[cfg(any(target_arch = "aarch64", test))]
pub(super) const SME_SMALL_SCHED_JPW: usize = 1;

/// 対象帯の上限（m・n・k のすべてがこれ以下）。SME 適格（m≥256・n≥256・k≥64）と合わせて
/// 実質 m=n=256・k∈[64,256] になる。
#[cfg(any(target_arch = "aarch64", test))]
pub(super) const SME_SMALL_SCHED_MAX_DIM: usize = 256;

/// [`select`] の結果。
#[cfg(any(target_arch = "aarch64", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SchedPlan {
    /// `Some(t)` なら専用プール（t スレッド）で実行する候補。
    pub(super) pool_threads: Option<usize>,
    /// 2D 動的分配へ渡す `jobs_per_worker`。
    pub(super) jpw: usize,
}

/// 対象帯の判定（純関数）。SME 適格かつ全次元が上限以下。
#[cfg(any(target_arch = "aarch64", test))]
pub(super) fn in_band(m_total: usize, n: usize, k: usize) -> bool {
    m_total >= BAND_MIN.0
        && n >= BAND_MIN.1
        && k >= BAND_MIN.2
        && m_total <= SME_SMALL_SCHED_MAX_DIM
        && n <= SME_SMALL_SCHED_MAX_DIM
        && k <= SME_SMALL_SCHED_MAX_DIM
}

/// プランを決める純関数（ゲート状態に依らずテストできるよう引数化）。
/// `incoming_jpw` が既定（[`TWO_D_JOBS_PER_WORKER`]）以外なら `small_m` の job 統合や
/// #4161 の推論 jpw1 が先に決めた値なので、それを優先する。
#[cfg(any(target_arch = "aarch64", test))]
pub(super) fn select(
    enabled: bool,
    threads: usize,
    jpw: usize,
    m_total: usize,
    n: usize,
    k: usize,
    incoming_jpw: usize,
) -> SchedPlan {
    let off = SchedPlan {
        pool_threads: None,
        jpw: incoming_jpw,
    };
    if !enabled || !in_band(m_total, n, k) {
        return off;
    }
    SchedPlan {
        pool_threads: (threads > 0).then_some(threads),
        jpw: if jpw > 0 && incoming_jpw == TWO_D_JOBS_PER_WORKER {
            jpw
        } else {
            incoming_jpw
        },
    }
}

/// 本番ゲート定数でプランを決める。
#[cfg(any(target_arch = "aarch64", test))]
pub(super) fn plan_for(m_total: usize, n: usize, k: usize, incoming_jpw: usize) -> SchedPlan {
    let plan = select(
        SME_SMALL_SCHED_ENABLED,
        SME_SMALL_SCHED_THREADS,
        SME_SMALL_SCHED_JPW,
        m_total,
        n,
        k,
        incoming_jpw,
    );
    #[cfg(test)]
    if plan
        != (SchedPlan {
            pool_threads: None,
            jpw: incoming_jpw,
        })
    {
        FIRES.with(|c| c.set(c.get() + 1));
    }
    plan
}

#[cfg(test)]
thread_local! {
    /// テスト専用の発火カウンタ（P-FIRE 用。呼び出しスレッドごと。本番ビルドには入らない）。
    static FIRES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// テスト専用: 呼び出しスレッドの発火数。
#[cfg(test)]
pub(super) fn fires() -> usize {
    FIRES.with(|c| c.get())
}

#[cfg(any(target_arch = "aarch64", test))]
static POOLS: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();

/// 専用プール（初回発火時に 1 度だけ生成。腕ごとに 1 スレッド数のため単一スロットで足りる）。
/// 生成失敗は `None`（`unwrap`／`expect` 不使用）。
#[cfg(any(target_arch = "aarch64", test))]
fn pool(threads: usize) -> Option<&'static rayon::ThreadPool> {
    POOLS
        .get_or_init(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| format!("fandhe-ai-sme-small-sched-{i}"))
                .build()
                .ok()
        })
        .as_ref()
}

/// `plan` に従って `f` を実行する。専用プールを使わない条件（`pool_threads` が `None`・
/// `RAYON_NUM_THREADS` 明示・現行プールが既に t 以下・生成失敗）では `f()` をそのまま呼ぶ。
#[cfg(any(target_arch = "aarch64", test))]
pub(super) fn run_in_plan<R: Send>(plan: SchedPlan, f: impl FnOnce() -> R + Send) -> R {
    let Some(t) = plan.pool_threads else {
        return f();
    };
    if crate::thread_limit::parse_env_num_threads(
        std::env::var("RAYON_NUM_THREADS").ok().as_deref(),
    )
    .is_some()
        || rayon::current_num_threads() <= t
    {
        return f();
    }
    match pool(t) {
        Some(p) => p.install(f),
        None => f(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sme_small_sched_enabled_with_t4_jpw1() {
        // 実行時 assert（const assert だと値を反転したツリーで lib テストがコンパイル不能になる。
        // `small_m::gate_is_off_by_default` と同じ理由）。ロールバック時は本テストを反転する。
        assert!(std::hint::black_box(SME_SMALL_SCHED_ENABLED));
        assert_eq!(std::hint::black_box(SME_SMALL_SCHED_THREADS), 4);
        assert_eq!(std::hint::black_box(SME_SMALL_SCHED_JPW), 1);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn band_min_matches_sme_thresholds() {
        use super::super::{SME_MIN_K, SME_MIN_M, SME_MIN_N};
        assert_eq!(BAND_MIN, (SME_MIN_M, SME_MIN_N, SME_MIN_K));
    }

    #[test]
    fn band_boundaries() {
        assert!(in_band(256, 256, 64));
        assert!(in_band(256, 256, 256));
        assert!(!in_band(256, 256, 63));
        assert!(!in_band(256, 256, 257));
        assert!(!in_band(257, 256, 256));
        assert!(!in_band(256, 257, 256));
        assert!(!in_band(255, 256, 256));
        assert!(!in_band(784, 256, 64));
        assert!(!in_band(512, 512, 512));
    }

    #[test]
    fn select_passthrough_when_off_or_out_of_band() {
        let off = SchedPlan {
            pool_threads: None,
            jpw: 2,
        };
        assert_eq!(select(false, 4, 1, 256, 256, 256, 2), off);
        assert_eq!(select(true, 4, 1, 512, 512, 512, 2), off);
        assert_eq!(select(true, 4, 1, 784, 256, 64, 2), off);
        assert_eq!(select(true, 0, 0, 256, 256, 256, 2), off);
    }

    #[test]
    fn select_applies_in_band() {
        assert_eq!(
            select(true, 4, 1, 256, 256, 256, 2),
            SchedPlan {
                pool_threads: Some(4),
                jpw: 1
            }
        );
        assert_eq!(
            select(true, 0, 4, 256, 256, 128, 2),
            SchedPlan {
                pool_threads: None,
                jpw: 4
            }
        );
        assert_eq!(
            select(true, 6, 0, 256, 256, 256, 2),
            SchedPlan {
                pool_threads: Some(6),
                jpw: 2
            }
        );
    }

    #[test]
    fn select_keeps_incoming_non_default_jpw() {
        assert_eq!(
            select(true, 4, 4, 256, 256, 256, 1),
            SchedPlan {
                pool_threads: Some(4),
                jpw: 1
            }
        );
    }

    #[test]
    fn run_in_plan_passthrough_without_pool() {
        let plan = SchedPlan {
            pool_threads: None,
            jpw: 2,
        };
        assert_eq!(run_in_plan(plan, || 7), 7);
        // 現行プール以上のスレッド数指定は素通し（current <= t）。
        let big = SchedPlan {
            pool_threads: Some(usize::MAX),
            jpw: 2,
        };
        assert_eq!(
            run_in_plan(big, rayon::current_num_threads),
            rayon::current_num_threads()
        );
    }

    #[test]
    fn sme_small_sched_enabled_plan_fires_in_band_only() {
        // 本番定数の plan は純関数で決まり、SME の有無に依らない（SME 非搭載機は呼び出し側の
        // `sme_kernel_for_shape` が `None` で、そもそも本機構へ到達しない）。
        let on = SchedPlan {
            pool_threads: Some(4),
            jpw: 1,
        };
        let before = fires();
        assert_eq!(plan_for(256, 256, 256, 2), on);
        assert_eq!(plan_for(256, 256, 64, 2), on);
        assert_eq!(fires(), before + 2);
        // 帯外は素通り（incoming_jpw を保つ・専用プールなし）で発火しない。
        let pass = SchedPlan {
            pool_threads: None,
            jpw: 2,
        };
        assert_eq!(plan_for(512, 512, 512, 2), pass);
        assert_eq!(plan_for(784, 256, 64, 2), pass);
        assert_eq!(fires(), before + 2);
    }

    /// 専用プール（T=4）と jpw ∈ {1,2,4} の実行が、既定プール・既定 jpw と bit 完全一致する。
    /// Linux CI では NEON／scalar カーネルで検証する（SME 実機は `#[ignore]` 側）。
    #[test]
    fn dedicated_pool_and_jpw_bit_exact_vs_default() {
        use super::super::{
            GemmTranspose, default_blocks, gemm_blis_parallel_two_d_dynamic_with_params,
        };
        let (m, n, k) = (256usize, 256usize, 128usize);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 31 % 97) as f32 - 48.0) / 17.0)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i * 17 % 89) as f32 - 44.0) / 13.0)
            .collect();
        let run = |threads: usize, jpw: usize| -> Vec<u32> {
            let mut c = vec![0.0f32; m * n];
            super::super::install_in_pool_for_test(threads, || {
                gemm_blis_parallel_two_d_dynamic_with_params(
                    &a,
                    &b,
                    &mut c,
                    m,
                    n,
                    k,
                    default_blocks(),
                    jpw,
                    GemmTranspose::Nn,
                )
                .expect("gemm");
            });
            c.iter().map(|v| v.to_bits()).collect()
        };
        let base = run(3, 2);
        for &(t, j) in &[(4usize, 1usize), (4, 2), (4, 4), (2, 1), (5, 4)] {
            assert_eq!(run(t, j), base, "t={t} jpw={j}");
        }
    }

    /// P-FIRE 用（`#[ignore]`・SME 実機）。本番入口 `gemm_blis_parallel_with_transpose` を
    /// 256³・512³・(784,256,64) で呼び、発火数を `FIRE256=<n>` 形式で標準出力へ出す。
    /// 候補の腕では 256³ が 1 以上、base／aa では 0、512³ と (784,256,64) は全腕で 0。
    #[test]
    #[ignore = "SME 実機（Mac mini）専用の P-FIRE プローブ"]
    fn report_sme_small_sched_probe() {
        use super::super::{GemmTranspose, gemm_blis_parallel_with_transpose};
        let fire = |m: usize, n: usize, k: usize| -> usize {
            let a = vec![0.5f32; m * k];
            let b = vec![0.25f32; k * n];
            let mut c = vec![0.0f32; m * n];
            let before = fires();
            gemm_blis_parallel_with_transpose(&a, &b, &mut c, m, n, k, GemmTranspose::Nn)
                .expect("gemm");
            fires() - before
        };
        println!("FIRE256={}", fire(256, 256, 256));
        println!("FIRE512={}", fire(512, 512, 512));
        println!("FIRE784x256x64={}", fire(784, 256, 64));
    }

    /// P-BIT 用（`#[ignore]`・SME 実機）。プランを強制した場合と既定の場合で 256³ の出力の
    /// `to_bits` が全一致することを確かめる。SME 非搭載ホストでは skip を出して pass。
    #[test]
    #[ignore = "SME 実機（Mac mini）専用の P-BIT"]
    fn sme_small_sched_bit_exact_vs_default_when_sme_available() {
        #[cfg(target_arch = "aarch64")]
        {
            use super::super::{GemmTranspose, gemm_blis_parallel_with_transpose, microkernel};
            if microkernel::SmeKernel::try_new().is_none() {
                println!("SKIP: SME 非搭載");
                return;
            }
            let (m, n, k) = (256usize, 256usize, 256usize);
            let a: Vec<f32> = (0..m * k)
                .map(|i| ((i * 31 % 97) as f32 - 48.0) / 17.0)
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|i| ((i * 17 % 89) as f32 - 44.0) / 13.0)
                .collect();
            let run = || {
                let mut c = vec![0.0f32; m * n];
                gemm_blis_parallel_with_transpose(&a, &b, &mut c, m, n, k, GemmTranspose::Nn)
                    .expect("gemm");
                c.iter().map(|v| v.to_bits()).collect::<Vec<u32>>()
            };
            // 参照: 単一連続 K ループ・FMA 契約（`f32::mul_add`）の素朴実装。#3231 が本番入口を
            // この参照と bit 一致として固定している。腕ツリーでは run() が差し替え後のプール・
            // jpw で走るので、差し替えで bit が動かないことを確かめる。
            let mut reference = vec![0u32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for p in 0..k {
                        acc = a[i * k + p].mul_add(b[p * n + j], acc);
                    }
                    reference[i * n + j] = acc.to_bits();
                }
            }
            assert_eq!(run(), reference, "プール／分配の違いで出力 bit が変わった");
        }
        #[cfg(not(target_arch = "aarch64"))]
        println!("SKIP: 非 aarch64");
    }
}
