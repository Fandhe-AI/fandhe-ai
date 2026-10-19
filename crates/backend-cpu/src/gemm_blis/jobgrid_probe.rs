//! イシュー #3177 診断専用: m=64 級 GEMM の (mc, nc) 2D 動的分配
//! （`gemm_blis_two_d_dynamic_region`）が払う固定費の「回数」を実行ログで
//! 数える `#[cfg(test)]` 限定プローブ。
//!
//! ## 役割と責務境界
//!
//! - `mod.rs` の 4 か所（`gemm_blis_two_d_dynamic_region`・`split_c_into_jobs`・
//!   `run_two_d_job`・`gemm_blis_ic_loop`）に置いた文単位の `#[cfg(test)]`
//!   フックから呼ばれ、job 数・使用 worker 数・A／B pack 語数・job 内確保
//!   回数・C staging 語数を atomic で集計する。本番ビルドにはコンパイル
//!   されず、本番の定数・分岐・数値結果は一切変わらない（診断のみ）。
//! - 計数専用。**計時には使わない**（計時と A/B は `examples/
//!   cpu_jobgrid_bench.rs` が cfg(test) なしのバイナリで行う）。
//! - 他テストの並列 GEMM 呼び出しを数えないよう、計測対象の呼び出しへ渡す
//!   B バッファのアドレスをキーに arm し、キー不一致の呼び出しは無視する。
//!   probe 同士は [`ProbeGuard`] が保持する静的 `Mutex` で直列化する。
//!
//! 確保回数は rayon 内部の確保を含まず、`vec!` を発行する各 site を
//! 解析的に 1 回ずつ数えた値（job 内 3 回、`split_c_into_jobs` 側は
//! `2 + col_bands + jobs`）である。

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use super::partition::JobGrid;

/// arm 中の B バッファ先頭アドレス（0 は未 arm）。
static KEY: AtomicUsize = AtomicUsize::new(0);
static PROBE_LOCK: Mutex<()> = Mutex::new(());

static REGIONS: AtomicU64 = AtomicU64::new(0);
static MR: AtomicU64 = AtomicU64::new(0);
static NR: AtomicU64 = AtomicU64::new(0);
static EFF_THREADS: AtomicU64 = AtomicU64::new(0);
static RAYON_THREADS: AtomicU64 = AtomicU64::new(0);
static ROW_BANDS: AtomicU64 = AtomicU64::new(0);
static COL_BANDS: AtomicU64 = AtomicU64::new(0);
static MC_JOB: AtomicU64 = AtomicU64::new(0);
static NC_JOB: AtomicU64 = AtomicU64::new(0);
static JOBS: AtomicU64 = AtomicU64::new(0);
static WORKER_MASK: AtomicU64 = AtomicU64::new(0);
static A_PACK_TILES: AtomicU64 = AtomicU64::new(0);
static A_PACK_WORDS: AtomicU64 = AtomicU64::new(0);
static B_PACK_TILES: AtomicU64 = AtomicU64::new(0);
static B_PACK_WORDS: AtomicU64 = AtomicU64::new(0);
static ALLOC_JOB: AtomicU64 = AtomicU64::new(0);
static ALLOC_SPLIT: AtomicU64 = AtomicU64::new(0);
static C_STAGE_WORDS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// 呼び出しスレッドが arm 中の region 内にいる（split 確保を数える）。
    static IN_REGION: Cell<bool> = const { Cell::new(false) };
    /// worker スレッドが arm 中の job を処理中（pack 語数を数える）。
    static IN_JOB: Cell<bool> = const { Cell::new(false) };
}

fn armed(b_ptr: usize) -> bool {
    let key = KEY.load(Ordering::Acquire);
    key != 0 && key == b_ptr
}

/// `gemm_blis_two_d_dynamic_region` が `job_grid` 確定直後に呼ぶ。
pub(super) fn on_region(
    b_ptr: usize,
    mr: usize,
    nr: usize,
    eff_threads: usize,
    rayon_threads: usize,
    grid: &JobGrid,
) {
    if !armed(b_ptr) {
        return;
    }
    REGIONS.fetch_add(1, Ordering::Relaxed);
    MR.store(mr as u64, Ordering::Relaxed);
    NR.store(nr as u64, Ordering::Relaxed);
    EFF_THREADS.store(eff_threads as u64, Ordering::Relaxed);
    RAYON_THREADS.store(rayon_threads as u64, Ordering::Relaxed);
    ROW_BANDS.store(grid.row_bands as u64, Ordering::Relaxed);
    COL_BANDS.store(grid.col_bands as u64, Ordering::Relaxed);
    MC_JOB.store(grid.mc_job as u64, Ordering::Relaxed);
    NC_JOB.store(grid.nc_job as u64, Ordering::Relaxed);
    IN_REGION.with(|f| f.set(true));
}

/// region の `split_c_into_jobs` 呼び出しが終わったら呼び、split 計数を止める。
pub(super) fn on_region_split_done() {
    IN_REGION.with(|f| f.set(false));
}

/// `split_c_into_jobs` が行帯・列帯数を確定した直後に呼ぶ。
/// 確保 site: `per_col_band` 外側 1 + 内側 col_bands + `jobs` 1 + job ごとの `c_rows`。
pub(super) fn on_split(col_bands: usize, jobs: usize) {
    if IN_REGION.with(Cell::get) {
        ALLOC_SPLIT.fetch_add((2 + col_bands + jobs) as u64, Ordering::Relaxed);
    }
}

/// `run_two_d_job` 入口（`mc_len`／`nc_len_job` 確定後）で呼ぶ。
/// 返すガードが drop されるまで、このスレッドの pack を job 内として数える。
pub(super) struct JobScope(bool);

impl Drop for JobScope {
    fn drop(&mut self) {
        if self.0 {
            IN_JOB.with(|f| f.set(false));
        }
    }
}

pub(super) fn on_job(b_ptr: usize, mc_len: usize, nc_len_job: usize) -> JobScope {
    if !armed(b_ptr) {
        return JobScope(false);
    }
    JOBS.fetch_add(1, Ordering::Relaxed);
    let bit = match rayon::current_thread_index() {
        Some(i) => i.min(62),
        None => 63,
    };
    WORKER_MASK.fetch_or(1u64 << bit, Ordering::Relaxed);
    // c_local・PanelBuffers の b_panel／a_panel の 3 回。
    ALLOC_JOB.fetch_add(3, Ordering::Relaxed);
    // copy-in と copy-out の 2 回ぶん。
    C_STAGE_WORDS.fetch_add(2 * (mc_len * nc_len_job) as u64, Ordering::Relaxed);
    IN_JOB.with(|f| f.set(true));
    JobScope(true)
}

/// `run_two_d_job` の B pack tile ごとに呼ぶ。
pub(super) fn on_b_pack(words: usize) {
    if IN_JOB.with(Cell::get) {
        B_PACK_TILES.fetch_add(1, Ordering::Relaxed);
        B_PACK_WORDS.fetch_add(words as u64, Ordering::Relaxed);
    }
}

/// `gemm_blis_ic_loop` の A pack tile ごとに呼ぶ（パディングを含む語数）。
pub(super) fn on_a_pack(words: usize) {
    if IN_JOB.with(Cell::get) {
        A_PACK_TILES.fetch_add(1, Ordering::Relaxed);
        A_PACK_WORDS.fetch_add(words as u64, Ordering::Relaxed);
    }
}

/// 1 回の GEMM 呼び出しぶんの計数結果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ProbeReport {
    pub regions: u64,
    pub mr: usize,
    pub nr: usize,
    pub eff_threads: usize,
    pub rayon_threads: usize,
    pub row_bands: usize,
    pub col_bands: usize,
    pub mc_job: usize,
    pub nc_job: usize,
    pub jobs: u64,
    pub workers_used: u32,
    pub a_pack_tiles: u64,
    pub a_pack_words: u64,
    pub b_pack_tiles: u64,
    pub b_pack_words: u64,
    pub alloc_job: u64,
    pub alloc_split: u64,
    pub c_stage_words: u64,
}

/// arm の RAII ガード。生存中は他の probe を排他し、drop で disarm する。
pub(super) struct ProbeGuard {
    _lock: MutexGuard<'static, ()>,
}

impl ProbeGuard {
    /// `b` のアドレスをキーに arm し、全カウンタを 0 に戻す。
    pub(super) fn arm(b: &[f32]) -> Self {
        let lock = PROBE_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        for c in [
            &REGIONS,
            &MR,
            &NR,
            &EFF_THREADS,
            &RAYON_THREADS,
            &ROW_BANDS,
            &COL_BANDS,
            &MC_JOB,
            &NC_JOB,
            &JOBS,
            &WORKER_MASK,
            &A_PACK_TILES,
            &A_PACK_WORDS,
            &B_PACK_TILES,
            &B_PACK_WORDS,
            &ALLOC_JOB,
            &ALLOC_SPLIT,
            &C_STAGE_WORDS,
        ] {
            c.store(0, Ordering::Relaxed);
        }
        KEY.store(b.as_ptr() as usize, Ordering::Release);
        ProbeGuard { _lock: lock }
    }

    pub(super) fn report(&self) -> ProbeReport {
        let l = |c: &AtomicU64| c.load(Ordering::Relaxed);
        ProbeReport {
            regions: l(&REGIONS),
            mr: l(&MR) as usize,
            nr: l(&NR) as usize,
            eff_threads: l(&EFF_THREADS) as usize,
            rayon_threads: l(&RAYON_THREADS) as usize,
            row_bands: l(&ROW_BANDS) as usize,
            col_bands: l(&COL_BANDS) as usize,
            mc_job: l(&MC_JOB) as usize,
            nc_job: l(&NC_JOB) as usize,
            jobs: l(&JOBS),
            workers_used: l(&WORKER_MASK).count_ones(),
            a_pack_tiles: l(&A_PACK_TILES),
            a_pack_words: l(&A_PACK_WORDS),
            b_pack_tiles: l(&B_PACK_TILES),
            b_pack_words: l(&B_PACK_WORDS),
            alloc_job: l(&ALLOC_JOB),
            alloc_split: l(&ALLOC_SPLIT),
            c_stage_words: l(&C_STAGE_WORDS),
        }
    }
}

impl Drop for ProbeGuard {
    fn drop(&mut self) {
        KEY.store(0, Ordering::Release);
    }
}

/// A を 1 回だけ pack した場合の語数（`ceil(m/mr)·mr·k`）。
pub(super) fn a_ideal_words(m: usize, k: usize, mr: usize) -> u64 {
    (m.div_ceil(mr) * mr * k) as u64
}

/// B を 1 回だけ pack した場合の語数（`ceil(n/nr)·nr·k`）。
pub(super) fn b_ideal_words(n: usize, k: usize, nr: usize) -> u64 {
    (n.div_ceil(nr) * nr * k) as u64
}

#[cfg(test)]
mod tests {
    use super::super::{
        GemmTranspose, default_blocks, gemm_blis, gemm_blis_parallel_with_transpose, partition,
    };
    use super::*;

    /// 6 形状 `(ラベル, 転置, m, n, k)`（MLP 784→256→10・batch 64）。
    const SHAPES: [(&str, GemmTranspose, usize, usize, usize); 6] = [
        ("l1_fwd", GemmTranspose::Nn, 64, 256, 784),
        ("l1_dw", GemmTranspose::Tn, 784, 256, 64),
        ("l1_dx", GemmTranspose::Nt, 64, 784, 256),
        ("l2_fwd", GemmTranspose::Nn, 64, 10, 256),
        ("l2_dw", GemmTranspose::Tn, 256, 10, 64),
        ("l2_dx", GemmTranspose::Nt, 64, 256, 10),
    ];

    fn trans_label(t: GemmTranspose) -> &'static str {
        match t {
            GemmTranspose::Nn => "nn",
            GemmTranspose::Nt => "nt",
            GemmTranspose::Tn => "tn",
        }
    }

    fn fill(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                (((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) % 997) as f32 / 997.0
                    - 0.5
            })
            .collect()
    }

    /// 論理 `[m,k]` の A を `Tn` なら `[k,m]` 格納、B を `Nt` なら `[n,k]` 格納で用意する。
    fn run_probed(
        pool_threads: Option<usize>,
        t: GemmTranspose,
        m: usize,
        n: usize,
        k: usize,
    ) -> (ProbeReport, Vec<f32>, Vec<f32>, Vec<f32>) {
        let a = fill(m * k, 1);
        let b = fill(k * n, 2);
        let mut c = vec![0.0f32; m * n];
        let guard = ProbeGuard::arm(&b);
        let call = |c: &mut [f32]| {
            gemm_blis_parallel_with_transpose(&a, &b, c, m, n, k, t).expect("gemm");
        };
        match pool_threads {
            Some(th) => rayon::ThreadPoolBuilder::new()
                .num_threads(th)
                .build()
                .expect("pool")
                .install(|| call(&mut c)),
            None => call(&mut c),
        }
        let report = guard.report();
        drop(guard);
        (report, a, b, c)
    }

    /// 再計算（Issue #3177 の 22／44 job ほか）を CI で固定する純関数テスト。
    #[test]
    fn job_grid_recompute_matches_issue() {
        let blocks = default_blocks();
        // (m, n, T, jpw) -> (row_bands, col_bands)（NEON 相当 MR=8／NR=12）。
        let cases = [
            (64usize, 256usize, 10usize, 2usize, 2usize, 11usize),
            (64, 256, 20, 2, 4, 11),
            (64, 256, 10, 1, 2, 5),
            (784, 256, 10, 2, 10, 2),
            (784, 256, 20, 2, 10, 4),
            (64, 784, 10, 2, 1, 22),
            (64, 784, 20, 2, 2, 22),
            (64, 10, 10, 2, 8, 1),
            (256, 10, 10, 2, 32, 1),
            (64, 256, 20, 2, 4, 11),
        ];
        for (m, n, t, jpw, rb, cb) in cases {
            let g = partition::job_grid(m, n, 8, 12, &blocks, t, jpw).expect("grid");
            assert_eq!(
                (g.row_bands, g.col_bands),
                (rb, cb),
                "m={m} n={n} T={t} jpw={jpw}"
            );
        }
    }

    /// 小形状を本番入口で実行し、計数が解析式と一致し結果が直列と bit 一致する。
    #[test]
    fn probe_counts_match_analytic_model() {
        let blocks = default_blocks();
        for t in [GemmTranspose::Nn, GemmTranspose::Nt, GemmTranspose::Tn] {
            let (m, n, k) = (40usize, 50usize, 70usize);
            let (r, a, b, c) = run_probed(Some(4), t, m, n, k);
            assert_eq!(r.regions, 1, "{t:?}");
            let grid = partition::job_grid(m, n, r.mr, r.nr, &blocks, r.eff_threads, 2).unwrap();
            assert_eq!((r.row_bands, r.col_bands), (grid.row_bands, grid.col_bands));
            let jobs = (r.row_bands * r.col_bands) as u64;
            assert_eq!(r.jobs, jobs, "{t:?}");
            assert_eq!(r.alloc_job, 3 * jobs);
            assert_eq!(r.alloc_split, (2 + r.col_bands) as u64 + jobs);
            assert_eq!(r.c_stage_words, 2 * (m * n) as u64);
            // k=70 <= KC かつ job 幅 <= NC のため pc／jc ループは 1 周: job ごとに
            // A は ceil(mc/mr)·mr·k 語、B は ceil(nc/nr)·nr·k 語 pack される。
            let rows = partition::bands(m, r.mc_job);
            let cols = partition::bands(n, r.nc_job);
            let (mut a_exp, mut b_exp) = (0u64, 0u64);
            for rb in &rows {
                for cb in &cols {
                    a_exp += a_ideal_words(rb.end - rb.start, k, r.mr);
                    b_exp += b_ideal_words(cb.end - cb.start, k, r.nr);
                }
            }
            assert_eq!(r.a_pack_words, a_exp, "{t:?}");
            assert_eq!(r.b_pack_words, b_exp, "{t:?}");
            // 直列参照との bit 一致（Nn のみ直列 gemm_blis で比較可能）。
            if t == GemmTranspose::Nn {
                let mut c_ref = vec![0.0f32; m * n];
                gemm_blis(&a, &b, &mut c_ref, m, n, k).unwrap();
                assert!(
                    c.iter()
                        .zip(&c_ref)
                        .all(|(x, y)| x.to_bits() == y.to_bits())
                );
            }
        }
    }

    /// 未 arm・キー不一致のとき何も数えない。
    #[test]
    fn probe_is_inert_when_not_armed_or_key_mismatch() {
        let (m, n, k) = (40usize, 50usize, 70usize);
        let a = fill(m * k, 1);
        let b = fill(k * n, 2);
        let other = fill(k * n, 3);
        let mut c = vec![0.0f32; m * n];
        let guard = ProbeGuard::arm(&other);
        gemm_blis_parallel_with_transpose(&a, &b, &mut c, m, n, k, GemmTranspose::Nn).unwrap();
        let r = guard.report();
        assert_eq!(r.regions, 0);
        assert_eq!(r.jobs, 0);
        assert_eq!(
            r.a_pack_words + r.b_pack_words + r.alloc_job + r.alloc_split,
            0
        );
    }

    /// 6 形状 × 計数腕を 1 行ずつ出力する診断（#3177 の実測用）。
    #[test]
    #[ignore = "診断専用（#3177）: cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored jobgrid_probe_mlp_shapes --exact --nocapture --test-threads=1"]
    fn jobgrid_probe_mlp_shapes() {
        for arm in ["native", "T1", "T10", "T20"] {
            let pool = match arm {
                "T1" => Some(1),
                "T10" => Some(10),
                "T20" => Some(20),
                _ => None,
            };
            for (label, t, m, n, k) in SHAPES {
                let (r, ..) = run_probed(pool, t, m, n, k);
                let a_ideal = a_ideal_words(m, k, r.mr);
                let b_ideal = b_ideal_words(n, k, r.nr);
                println!(
                    "probe=jobgrid shape={label} trans={} m={m} n={n} k={k} arm={arm} \
rayon_threads={} eff_threads={} mr={} nr={} row_bands={} col_bands={} mc_job={} nc_job={} \
regions={} jobs={} workers_used={} a_pack_tiles={} a_pack_words={} a_ideal_words={a_ideal} \
a_repack_x={:.3} b_pack_tiles={} b_pack_words={} b_ideal_words={b_ideal} b_repack_x={:.3} \
alloc_job={} alloc_split={} c_stage_words={}",
                    trans_label(t),
                    r.rayon_threads,
                    r.eff_threads,
                    r.mr,
                    r.nr,
                    r.row_bands,
                    r.col_bands,
                    r.mc_job,
                    r.nc_job,
                    r.regions,
                    r.jobs,
                    r.workers_used,
                    r.a_pack_tiles,
                    r.a_pack_words,
                    r.a_pack_words as f64 / a_ideal as f64,
                    r.b_pack_tiles,
                    r.b_pack_words,
                    r.b_pack_words as f64 / b_ideal as f64,
                    r.alloc_job,
                    r.alloc_split,
                    r.c_stage_words,
                );
            }
        }
    }
}
