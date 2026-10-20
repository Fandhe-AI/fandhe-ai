//! イシュー #3187（`docs/perf/cpu-gemm-reuse-fresh-inversion.md`）: GB10 の
//! CPU GEMM で reuse が fresh より遅い逆転の帰属用フェーズ分解診断。
//!
//! 背景: `docs/perf/cpu-gemm-candle-gate-remeasurement.md` §16.2 は「host_copy・
//! checksum の比率が高い」とするだけで reuse と fresh の差の原因を特定して
//! いない。本ファイルは bench-fandhe の `run_gemm`（fresh）・`run_gemm_reuse`
//! （reuse）と同じ呼び出し列を facade の外側から写し、区間別に計時する。
//! 因果を切り分けるため「fresh のまま Tape を保持する腕（結果蓄積の再現）」
//! 「backend の `gemm` を直接呼び結果を捨てる／保持する腕」「host arena
//! （#2104）ON の腕」を加える。
//!
//! - `Var::matmul` は eager（`materialize_fallible` → `matmul_forward` →
//!   `push_eager`）で、`matmul` 区間にカーネル時間が入る。`host_copy` 区間は
//!   CPU の既定 readout（`host_view` の借用）で、legacy 腕のみ
//!   `to_tensor().contiguous().as_slice().to_vec()`
//! - 入力は bench-common と同じ `SEED_A = 0xA11CE`・`SEED_B = 0xB0B`
//!   （framework-compare は独立 workspace で path 依存できないため値を複製）
//! - テスト 1（CI 実行）は全腕の出力が公開経路と bit 一致することを hard
//!   assert する。テスト 2（`#[ignore]`・実機用）は record-only で、判定規則は
//!   `docs/perf/logs/cpu-gemm-reuse-fresh-3187/RULE.txt`（実測前に固定）に従い
//!   `orchestrate.sh`／`aggregate.py` が独立 5 プロセスの出力を集計する
//!
//! 本番コード・tolerance・baseline は変更しない。

use std::time::Instant;

use bench_harness::median_q1_q3;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai::{Tensor, Var};
use fandhe_ai_autodiff::Tape;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::BackendOps;
use fandhe_ai_tensor_core::alloc::{
    HostArenaStats, clear_thread_arena, override_enabled_for_scope, reset_stats, stats,
};

/// bench-common `SEED_A`／`SEED_B` と同値（出典: framework-compare/bench-common/src/lib.rs）。
const SEED_A: u64 = 0xA11CE;
const SEED_B: u64 = 0xB0B;
const WARMUP_ITERS: usize = 20;
const MEASURE_ITERS: usize = 20;
const ROUNDS: usize = 4;
const SIZES: [usize; 4] = [256, 512, 1024, 2048];
/// 標準出力の 1 行 1 区間 JSON の接頭辞（`orchestrate.sh` が抽出）。
const PREFIX: &str = "DIAG_JSON ";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Fresh,
    Reuse,
    FreshHold,
    BackendDrop,
    BackendHold,
}

#[derive(Clone, Copy)]
struct Arm {
    name: &'static str,
    kind: Kind,
    /// host arena（#2104）を現スレッドで ON にして実行するか。
    arena: bool,
    /// readout を legacy（`to_tensor().contiguous().as_slice().to_vec()`）にするか。
    legacy: bool,
}

const fn arm(name: &'static str, kind: Kind, arena: bool, legacy: bool) -> Arm {
    Arm {
        name,
        kind,
        arena,
        legacy,
    }
}

const ARMS: [Arm; 9] = [
    arm("fresh", Kind::Fresh, false, false),
    arm("reuse", Kind::Reuse, false, false),
    arm("fresh_hold", Kind::FreshHold, false, false),
    arm("fresh_arena", Kind::Fresh, true, false),
    arm("reuse_arena", Kind::Reuse, true, false),
    arm("backend_drop", Kind::BackendDrop, false, false),
    arm("backend_hold", Kind::BackendHold, false, false),
    arm("fresh_legacy", Kind::Fresh, false, true),
    arm("reuse_legacy", Kind::Reuse, false, true),
];

fn inputs(n: usize) -> (Tensor<f32>, Tensor<f32>) {
    let a = Xorshift64Star::new(SEED_A).fill_vec(n * n);
    let b = Xorshift64Star::new(SEED_B).fill_vec(n * n);
    (
        Tensor::new(a, &[n, n]).expect("形状とデータ長は一致させている"),
        Tensor::new(b, &[n, n]).expect("形状とデータ長は一致させている"),
    )
}

fn new_tape() -> Tape {
    // `resolve_ops(Device::Cpu)` と同じ構成（cpu_predict_resident_fixedcost_diag.rs と同方針）。
    Tape::new_with_ops(Box::new(CpuBackendOps::new()))
}

fn checksum(v: &[f32]) -> f64 {
    v.iter().map(|&x| f64::from(x)).sum()
}

/// /proc/self/stat の minflt。`GEMM_REUSE_MINFLT=1` の別 run でのみ有効
/// （/proc 読み出しが計時を歪めるため、タイミング系列では OFF）。
fn minflt_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("GEMM_REUSE_MINFLT").is_ok_and(|v| v == "1"))
}

#[cfg(target_os = "linux")]
fn minflt() -> Option<u64> {
    if !minflt_enabled() {
        return None;
    }
    let s = std::fs::read_to_string("/proc/self/stat").ok()?;
    // comm は空白・括弧を含みうるため最後の ')' 以降で分割する。
    let rest = s.rsplit_once(')')?.1;
    rest.split_whitespace().nth(7)?.parse().ok()
}
#[cfg(not(target_os = "linux"))]
fn minflt() -> Option<u64> {
    None
}

#[derive(Default)]
struct Cell {
    secs: Vec<f64>,
    flts: Vec<u64>,
    flt_missing: bool,
}

/// (arm, n, phase) ごとのサンプル蓄積器。
#[derive(Default)]
struct Rec {
    cells: Vec<((&'static str, usize, &'static str), Cell)>,
    /// 記録するか（warmup 中は false）。
    on: bool,
    arm: &'static str,
    n: usize,
    /// (arm, n) → 全計測反復で共通だった checksum の bit。
    checksums: Vec<((&'static str, usize), u64)>,
    arena_stats: Vec<((&'static str, usize), HostArenaStats)>,
}

impl Rec {
    fn cell(&mut self, phase: &'static str) -> &mut Cell {
        let key = (self.arm, self.n, phase);
        let i = match self.cells.iter().position(|(k, _)| *k == key) {
            Some(i) => i,
            None => {
                self.cells.push((key, Cell::default()));
                self.cells.len() - 1
            }
        };
        &mut self.cells[i].1
    }

    fn push(&mut self, phase: &'static str, dt: f64, f0: Option<u64>) {
        if !self.on {
            return;
        }
        let f1 = minflt();
        let c = self.cell(phase);
        c.secs.push(dt);
        match (f0, f1) {
            (Some(a), Some(b)) => c.flts.push(b.saturating_sub(a)),
            _ => c.flt_missing = true,
        }
    }

    fn timed<T>(&mut self, phase: &'static str, f: impl FnOnce() -> T) -> T {
        let f0 = minflt();
        let t = Instant::now();
        let out = f();
        let dt = t.elapsed().as_secs_f64();
        self.push(phase, dt, f0);
        out
    }

    /// 計測反復の checksum を記録する。同一 (arm, n) の全反復で bit 一致を要求し、
    /// 不一致なら panic する（決定的な CPU 経路では常に一致する想定）。
    fn record_checksum(&mut self, s: f64) {
        if !self.on {
            return;
        }
        let bits = s.to_bits();
        let key = (self.arm, self.n);
        match self.checksums.iter().find(|(k, _)| *k == key) {
            None => self.checksums.push((key, bits)),
            Some((_, prev)) => assert_eq!(
                *prev, bits,
                "arm {} n={}: 反復間で checksum が不一致",
                key.0, key.1
            ),
        }
    }
}

/// fresh_hold 腕が保持する Tape・backend_hold 腕が保持する結果（ブロック終了まで解放しない）。
#[derive(Default)]
struct Held {
    tapes: Vec<Tape>,
    outs: Vec<Tensor<f32>>,
}

/// readout（`host_copy` 区間）と checksum 区間を計時して checksum を返す。
/// `capture` が Some のときだけ出力を複製する（テスト 1 専用）。
fn read_sum(rec: &mut Rec, c: &Var, legacy: bool, capture: Option<&mut Vec<f32>>) -> f64 {
    if legacy {
        let out = rec.timed("host_copy", || {
            c.to_tensor()
                .contiguous()
                .as_slice()
                .expect("contiguous 直後は as_slice が Some")
                .to_vec()
        });
        let s = rec.timed("checksum", || checksum(&out));
        if let Some(cap) = capture {
            *cap = out;
        }
        s
    } else {
        let view = rec.timed("host_copy", || c.host_view());
        let s = rec.timed("checksum", || checksum(&view));
        if let Some(cap) = capture {
            *cap = view.to_vec();
        }
        s
    }
}

/// 1 反復の入力一式（backend 腕は ops・a・b、reuse 腕は使い回す葉 Var）。
struct Ctx<'a, 't> {
    ops: &'a CpuBackendOps,
    a: &'a Tensor<f32>,
    b: &'a Tensor<f32>,
    leaves: Option<(&'a Var<'t>, &'a Var<'t>)>,
}

/// 1 反復。fresh は bench `run_gemm`、reuse は `run_gemm_reuse` と同じ呼び出し列。
/// `tape_drop` は iter_total の窓の外（bench で `elapsed` 取得後に Tape が drop
/// されるのと同じ位置）で別計時する。
fn one_iter(ctx: Ctx, arm: Arm, rec: &mut Rec, held: &mut Held, capture: Option<&mut Vec<f32>>) {
    let Ctx { ops, a, b, leaves } = ctx;
    let t_all = Instant::now();
    let f_all = minflt();
    match arm.kind {
        Kind::Fresh | Kind::FreshHold => {
            let tape = rec.timed("tape_build", new_tape);
            let (va, vb) = rec.timed("var_upload", || (tape.var(a), tape.var(b)));
            let c = rec.timed("matmul", || va.matmul(&vb).expect("matmul"));
            let s = read_sum(rec, &c, arm.legacy, capture);
            rec.push("iter_total", t_all.elapsed().as_secs_f64(), f_all);
            rec.record_checksum(s);
            if arm.kind == Kind::FreshHold {
                held.tapes.push(tape);
            } else {
                rec.timed("tape_drop", move || drop(tape));
            }
        }
        Kind::Reuse => {
            let (va, vb) = leaves.expect("reuse は葉 Var を持つ");
            let c = rec.timed("matmul", || va.matmul(vb).expect("matmul"));
            let s = read_sum(rec, &c, arm.legacy, capture);
            rec.push("iter_total", t_all.elapsed().as_secs_f64(), f_all);
            rec.record_checksum(s);
            // 結果ノードは tape に積まれたまま（結果蓄積）。
            std::hint::black_box(&c);
        }
        Kind::BackendDrop | Kind::BackendHold => {
            let out = rec.timed("kernel_call", || ops.gemm(a, b).expect("gemm"));
            rec.push("iter_total", t_all.elapsed().as_secs_f64(), f_all);
            let slice = out.as_slice().expect("gemm 出力は contiguous");
            rec.record_checksum(checksum(slice));
            if let Some(cap) = capture {
                *cap = slice.to_vec();
            }
            if arm.kind == Kind::BackendHold {
                held.outs.push(out);
            }
        }
    }
}

/// 1 ブロック（同一 arm・同一 n の warmup + 計測）を実行する。reuse 系はブロック
/// ごとに新しい Tape と葉 Var（init は計測外）、保持系はブロック終了で一括解放する。
fn drive(
    rec: &mut Rec,
    arm: Arm,
    n: usize,
    (a, b): (&Tensor<f32>, &Tensor<f32>),
    (warmup, iters): (usize, usize),
    mut captured: Option<&mut Vec<Vec<f32>>>,
) {
    rec.arm = arm.name;
    rec.n = n;
    let ops = CpuBackendOps::new();
    let reuse_tape = (arm.kind == Kind::Reuse).then(new_tape);
    let leaves = reuse_tape.as_ref().map(|t| (t.var(a), t.var(b)));
    let mut held = Held::default();
    for i in 0..warmup + iters {
        rec.on = i >= warmup;
        let mut out = Vec::new();
        let want = rec.on && captured.is_some();
        let lv = leaves.as_ref().map(|(x, y)| (x, y));
        one_iter(
            Ctx {
                ops: &ops,
                a,
                b,
                leaves: lv,
            },
            arm,
            rec,
            &mut held,
            want.then_some(&mut out),
        );
        if want && let Some(c) = captured.as_deref_mut() {
            c.push(out);
        }
    }
    rec.on = false;
}

/// ブロック実行。arena 腕は scope guard で ON にし、ブロック前に free list と
/// 統計をリセットしてブロック後（保持物の解放後）の統計を記録する。
fn run_block(rec: &mut Rec, arm: Arm, n: usize, a: &Tensor<f32>, b: &Tensor<f32>) {
    let _guard = arm.arena.then(|| override_enabled_for_scope(true));
    if arm.arena {
        clear_thread_arena();
        reset_stats();
    }
    drive(rec, arm, n, (a, b), (WARMUP_ITERS, MEASURE_ITERS), None);
    if arm.arena {
        rec.arena_stats.push(((arm.name, n), stats()));
        clear_thread_arena();
    }
}

/// 公開経路（`tape.var → matmul → host_view`）の出力。
fn public_output(a: &Tensor<f32>, b: &Tensor<f32>) -> Vec<f32> {
    let tape = new_tape();
    let c = tape.var(a).matmul(&tape.var(b)).expect("matmul");
    c.host_view().to_vec()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn cpu_gemm_reuse_fresh_arms_match_public_path_bit_exact() {
    for n in [64usize, 128] {
        let (a, b) = inputs(n);
        let expect = bits(&public_output(&a, &b));
        for arm in ARMS {
            let mut rec = Rec::default();
            let mut outs = Vec::new();
            let _guard = arm.arena.then(|| override_enabled_for_scope(true));
            if arm.arena {
                clear_thread_arena();
                reset_stats();
            }
            // arena 腕は 2 反復目以降で再利用が起きるため複数反復を回す。
            drive(&mut rec, arm, n, (&a, &b), (0, 3), Some(&mut outs));
            assert_eq!(outs.len(), 3);
            for out in &outs {
                assert_eq!(
                    bits(out),
                    expect,
                    "arm {} n={n}: 出力が公開経路と bit 不一致",
                    arm.name
                );
            }
            if arm.arena && arm.kind == Kind::Fresh {
                // fresh+arena は Tape drop で出力が回収され次反復で再利用される
                // （override が呼び出しスレッドで効くことの確認）。
                let st = stats();
                assert!(
                    st.hit > 0,
                    "fresh_arena で arena の再利用が発生しない（stats={st:?}）"
                );
            }
            if arm.arena {
                clear_thread_arena();
            }
        }
    }
}

/// 実機用 record-only 診断（`orchestrate.sh` が独立プロセスで 5 回起動）。
#[test]
#[ignore = "実機計測用の record-only 診断（RULE.txt・orchestrate.sh 参照）"]
fn cpu_gemm_reuse_fresh_phases() {
    let mut rec = Rec::default();
    for n in SIZES {
        let (a, b) = inputs(n);
        for round in 0..ROUNDS {
            // 順序効果を打ち消すためラウンドごとに arm 順を反転する。
            let order: Vec<Arm> = if round % 2 == 0 {
                ARMS.to_vec()
            } else {
                ARMS.iter().rev().copied().collect()
            };
            for arm in order {
                run_block(&mut rec, arm, n, &a, &b);
            }
        }
    }

    let threads = std::env::var("RAYON_NUM_THREADS").ok();
    for ((arm, n, phase), cell) in &rec.cells {
        let q = median_q1_q3(&cell.secs).expect("サンプルは非空・非 NaN");
        let mut sorted = cell.secs.clone();
        sorted.sort_by(f64::total_cmp);
        let minflt_mean = if !cell.flt_missing && !cell.flts.is_empty() {
            Some(cell.flts.iter().sum::<u64>() as f64 / cell.flts.len() as f64)
        } else {
            None
        };
        let is_total = *phase == "iter_total";
        let checksum_bits = is_total
            .then(|| {
                rec.checksums
                    .iter()
                    .find(|(k, _)| *k == (*arm, *n))
                    .map(|(_, b)| format!("{b:016x}"))
            })
            .flatten();
        // arena 統計はブロック(ラウンド)ごとに 1 件。合算して出す。
        let arena = if is_total {
            let all: Vec<&HostArenaStats> = rec
                .arena_stats
                .iter()
                .filter(|(k, _)| *k == (*arm, *n))
                .map(|(_, s)| s)
                .collect();
            (!all.is_empty()).then(|| {
                serde_json::json!({
                    "hit": all.iter().map(|s| s.hit).sum::<u64>(),
                    "miss": all.iter().map(|s| s.miss).sum::<u64>(),
                    "recycled": all.iter().map(|s| s.recycled).sum::<u64>(),
                    "rejected": all.iter().map(|s| s.rejected).sum::<u64>(),
                })
            })
        } else {
            None
        };
        let line = serde_json::json!({
            "issue": 3187,
            "arm": arm,
            "n": n,
            "phase": phase,
            "median_s": q.median,
            "q1_s": q.q1,
            "q3_s": q.q3,
            "min_s": sorted.first(),
            "max_s": sorted.last(),
            "n_samples": cell.secs.len(),
            "checksum_bits": checksum_bits,
            "arena_stats": arena,
            "minflt_delta_per_iter": minflt_mean,
            "minflt_enabled": minflt_enabled(),
            "rayon_num_threads_env": threads,
            "target_arch": std::env::consts::ARCH,
            "target_os": std::env::consts::OS,
        });
        println!("{PREFIX}{line}");
    }
}
