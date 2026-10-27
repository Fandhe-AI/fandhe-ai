//! イシュー #3646（`docs/perf/cpu-gemm-reuse-fresh-inversion.md` §9）: GB10 の CPU GEMM で
//! 常駐入力経路の reuse が fresh より遅い原因の帰属用フェーズ分解診断。
//!
//! 背景: #3272 で bench-fandhe の `gemm --mode reuse` を常駐入力経路
//! （`Tape::upload_resident` → `matmul_resident`）へ切り替えたところ、GB10 の CPU gemm
//! だけ reuse が同 round の fresh より遅くなった。#3187 の診断（`cpu_gemm_reuse_fresh_diag.rs`。
//! ホスト入力・Tape 経路）の構造を写し、常駐経路の窓にだけ入る 3 処理（`download` の
//! 追加確保＋単一スレッド memcpy・`gemm_device` 出力の窓内解放・入力の寿命）を腕で分離する。
//!
//! - `resident_public` は facade 公開経路（bench `measure_gemm_reuse_resident` の窓の写し）。
//!   `resident_split` は同じ処理列を内部 API（`MemoryOps::download` 等）で区間に分けて計時する
//! - `gemm_device_hold` / `gemm_device_drop_in` / `gemm_device_download` は
//!   `gemm_device` 単体・窓内 drop・download の寄与を引き算で出すための基準腕
//! - 入力は bench-common と同じ `SEED_A = 0xA11CE`・`SEED_B = 0xB0B`
//!   （framework-compare は独立 workspace で path 依存できないため値を複製）
//! - テスト 1（CI 実行）は全腕の出力が公開経路と bit 一致することを hard assert する。
//!   テスト 2（`#[ignore]`・実機用）は record-only で、判定規則は
//!   `docs/perf/logs/cpu-gemm-resident-reuse-3646/RULE.txt`（実測前に固定）に従い
//!   `orchestrate.sh`／`aggregate.py` が独立 5 プロセスの出力を集計する
//!
//! 本番コード・tolerance・baseline は変更しない。新しい `unsafe` は使わない。

use std::time::Instant;

use bench_harness::median_q1_q3;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, Tape, Tensor, tape_for};
use fandhe_ai_autodiff::Tape as InnerTape;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::{BackendOps, DeviceBuffer, MemoryOps};

/// bench-common `SEED_A`／`SEED_B` と同値（出典: framework-compare/bench-common/src/lib.rs）。
const SEED_A: u64 = 0xA11CE;
const SEED_B: u64 = 0xB0B;
const ROUNDS: usize = 4;
const SIZES: [usize; 5] = [256, 512, 1024, 2048, 4096];
/// 標準出力の 1 行 1 区間 JSON の接頭辞（`orchestrate.sh` が抽出）。
const PREFIX: &str = "DIAG_JSON ";

/// N ごとの (warmup, 計測) 反復数。N=4096 は 1 反復が長いので RULE.txt で別に固定している。
fn iters_for(n: usize) -> (usize, usize) {
    if n >= 4096 { (5, 10) } else { (20, 20) }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Fresh,
    ResidentPublic,
    ResidentSplit,
    GemmHold,
    GemmDropIn,
    GemmDownload,
    ResidentReupload,
    BackendDrop,
}

#[derive(Clone, Copy)]
struct Arm {
    name: &'static str,
    kind: Kind,
}

const ARMS: [Arm; 8] = [
    Arm {
        name: "fresh",
        kind: Kind::Fresh,
    },
    Arm {
        name: "resident_public",
        kind: Kind::ResidentPublic,
    },
    Arm {
        name: "resident_split",
        kind: Kind::ResidentSplit,
    },
    Arm {
        name: "gemm_device_hold",
        kind: Kind::GemmHold,
    },
    Arm {
        name: "gemm_device_drop_in",
        kind: Kind::GemmDropIn,
    },
    Arm {
        name: "gemm_device_download",
        kind: Kind::GemmDownload,
    },
    Arm {
        name: "resident_reupload",
        kind: Kind::ResidentReupload,
    },
    Arm {
        name: "backend_drop",
        kind: Kind::BackendDrop,
    },
];

fn inputs(n: usize) -> (Tensor<f32>, Tensor<f32>) {
    let a = Xorshift64Star::new(SEED_A).fill_vec(n * n);
    let b = Xorshift64Star::new(SEED_B).fill_vec(n * n);
    (
        Tensor::new(a, &[n, n]).expect("形状とデータ長は一致させている"),
        Tensor::new(b, &[n, n]).expect("形状とデータ長は一致させている"),
    )
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
}

impl Rec {
    fn push(&mut self, phase: &'static str, dt: f64, f0: Option<u64>) {
        if !self.on {
            return;
        }
        let f1 = minflt();
        let key = (self.arm, self.n, phase);
        let i = match self.cells.iter().position(|(k, _)| *k == key) {
            Some(i) => i,
            None => {
                self.cells.push((key, Cell::default()));
                self.cells.len() - 1
            }
        };
        let c = &mut self.cells[i].1;
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

    /// 同一 (arm, n) の全計測反復で bit 一致を要求し、不一致なら panic する。
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

/// 常駐腕の共有状態。facade 公開経路（`ResidentTensor`）と内部 API（`DeviceBuffer`）を
/// 同じ入力で持つ。upload はすべてブロック開始時（計測窓の外）。
struct Resident {
    tape: Tape,
    store: fandhe_ai::DeviceParamStore,
    ra: fandhe_ai::ResidentTensor,
    rb: fandhe_ai::ResidentTensor,
    ops: CpuBackendOps,
    da: DeviceBuffer<f32>,
    db: DeviceBuffer<f32>,
}

impl Resident {
    fn new(a: &Tensor<f32>, b: &Tensor<f32>) -> Self {
        let tape = tape_for(Device::Cpu).expect("CPU tape");
        // bench `gemm_binding_store` と同じ束縛用 store（`matmul_resident` は世代を見ない）。
        let store = Sequential::new()
            .add_linear(1, 1, 1)
            .expect("linear")
            .init_device_param_store(&tape)
            .expect("store");
        let ra = tape.upload_resident(&store, a).expect("upload a");
        let rb = tape.upload_resident(&store, b).expect("upload b");
        let ops = CpuBackendOps::new();
        let da = ops.upload(a).expect("upload a");
        let db = ops.upload(b).expect("upload b");
        Self {
            tape,
            store,
            ra,
            rb,
            ops,
            da,
            db,
        }
    }

    fn reupload(&mut self, a: &Tensor<f32>, b: &Tensor<f32>) {
        self.ra = self.tape.upload_resident(&self.store, a).expect("upload a");
        self.rb = self.tape.upload_resident(&self.store, b).expect("upload b");
    }
}

fn new_inner_tape() -> InnerTape {
    // `resolve_ops(Device::Cpu)` と同じ構成（cpu_gemm_reuse_fresh_diag.rs と同方針）。
    InnerTape::new_with_ops(Box::new(CpuBackendOps::new()))
}

/// 1 反復。窓は `iter_total`。解放位置は腕ごとに明示する。
fn one_iter(
    rec: &mut Rec,
    arm: Arm,
    res: &mut Resident,
    (a, b): (&Tensor<f32>, &Tensor<f32>),
    capture: Option<&mut Vec<f32>>,
) {
    let mut cap = capture;
    let mut take = |s: &[f32]| {
        if let Some(c) = cap.as_deref_mut() {
            *c = s.to_vec();
        }
    };
    match arm.kind {
        Kind::Fresh => {
            let t_all = Instant::now();
            let f_all = minflt();
            let tape = rec.timed("tape_build", new_inner_tape);
            let (va, vb) = rec.timed("var_upload", || (tape.var(a), tape.var(b)));
            let c = rec.timed("matmul", || va.matmul(&vb).expect("matmul"));
            let view = rec.timed("readout", || c.host_view());
            let s = rec.timed("checksum", || checksum(&view));
            take(&view);
            rec.push("iter_total", t_all.elapsed().as_secs_f64(), f_all);
            rec.record_checksum(s);
            drop(view);
            // Tape の drop は窓の外（bench の `elapsed` 取得後と同じ位置）。
            rec.timed("tape_drop", move || drop(tape));
        }
        Kind::ResidentPublic | Kind::ResidentReupload => {
            if arm.kind == Kind::ResidentReupload {
                res.reupload(a, b);
            }
            let t_all = Instant::now();
            let f_all = minflt();
            let c = rec.timed("matmul", || {
                res.tape
                    .matmul_resident(&res.ra, &res.rb)
                    .expect("matmul_resident")
            });
            let out = rec.timed("readout", || c.contiguous());
            let sl = out.as_slice().expect("contiguous");
            let s = rec.timed("checksum", || checksum(sl));
            take(sl);
            rec.push("iter_total", t_all.elapsed().as_secs_f64(), f_all);
            rec.record_checksum(s);
        }
        Kind::ResidentSplit => {
            let t_all = Instant::now();
            let f_all = minflt();
            let out = rec.timed("gemm_device", || {
                res.ops.gemm_device(&res.da, &res.db).expect("gemm_device")
            });
            let host = rec.timed("download", || res.ops.download(&out).expect("download"));
            rec.timed("out_drop", move || drop(out));
            let c = rec.timed("readout", || host.contiguous());
            let sl = c.as_slice().expect("contiguous");
            let s = rec.timed("checksum", || checksum(sl));
            take(sl);
            rec.push("iter_total", t_all.elapsed().as_secs_f64(), f_all);
            rec.record_checksum(s);
        }
        Kind::GemmHold | Kind::GemmDropIn | Kind::GemmDownload => {
            // 反復間処理を 3 腕で揃える: 窓の外では出力（と download 結果）の drop だけを行い、
            // 追加の GEMM・download は反復ループに置かない（確保・解放の回数と入力キャッシュ状態を
            // 腕間で変えないため。出力の検証は `drive` がループ後の独立ブロックで行う）。
            let t_all = Instant::now();
            let f_all = minflt();
            let out = res.ops.gemm_device(&res.da, &res.db).expect("gemm_device");
            match arm.kind {
                Kind::GemmDownload => {
                    let h = res.ops.download(&out).expect("download");
                    rec.push("iter_total", t_all.elapsed().as_secs_f64(), f_all);
                    drop((out, h));
                }
                Kind::GemmDropIn => {
                    drop(out);
                    rec.push("iter_total", t_all.elapsed().as_secs_f64(), f_all);
                }
                _ => {
                    rec.push("iter_total", t_all.elapsed().as_secs_f64(), f_all);
                    drop(out);
                }
            }
        }
        Kind::BackendDrop => {
            let t_all = Instant::now();
            let f_all = minflt();
            let out = rec.timed("kernel_call", || res.ops.gemm(a, b).expect("gemm"));
            rec.push("iter_total", t_all.elapsed().as_secs_f64(), f_all);
            let sl = out.as_slice().expect("gemm 出力は contiguous");
            rec.record_checksum(checksum(sl));
            take(sl);
        }
    }
}

/// 1 ブロック（同一 arm・同一 n の warmup + 計測）。入力の upload は窓の外。
fn drive(
    rec: &mut Rec,
    arm: Arm,
    n: usize,
    ab: (&Tensor<f32>, &Tensor<f32>),
    (warmup, iters): (usize, usize),
    mut captured: Option<&mut Vec<Vec<f32>>>,
) {
    rec.arm = arm.name;
    rec.n = n;
    let mut res = Resident::new(ab.0, ab.1);
    for i in 0..warmup + iters {
        rec.on = i >= warmup;
        let mut out = Vec::new();
        let want = rec.on && captured.is_some();
        one_iter(rec, arm, &mut res, ab, want.then_some(&mut out));
        if want
            && !out.is_empty()
            && let Some(c) = captured.as_deref_mut()
        {
            c.push(out);
        }
    }
    rec.on = false;
    if matches!(
        arm.kind,
        Kind::GemmHold | Kind::GemmDropIn | Kind::GemmDownload
    ) {
        // 計測ループの外の独立ブロック: 出力の checksum と bit 一致用の出力をブロックごとに 1 回だけ確定する。
        let chk = res.ops.gemm_device(&res.da, &res.db).expect("gemm_device");
        let h = res.ops.download(&chk).expect("download");
        let sl = h.as_slice().expect("contiguous");
        rec.on = true;
        rec.record_checksum(checksum(sl));
        rec.on = false;
        if let Some(c) = captured {
            c.push(sl.to_vec());
        }
    }
}

/// 公開経路（`upload_resident → matmul_resident`）の出力。
fn public_output(a: &Tensor<f32>, b: &Tensor<f32>) -> Vec<f32> {
    let res = Resident::new(a, b);
    let out = res
        .tape
        .matmul_resident(&res.ra, &res.rb)
        .expect("matmul_resident");
    out.contiguous().as_slice().expect("contiguous").to_vec()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn cpu_gemm_resident_arms_match_public_path_bit_exact() {
    for n in [64usize, 128] {
        let (a, b) = inputs(n);
        let expect = bits(&public_output(&a, &b));
        for arm in ARMS {
            let mut rec = Rec::default();
            let mut outs = Vec::new();
            drive(&mut rec, arm, n, (&a, &b), (0, 3), Some(&mut outs));
            // gemm_device 系 3 腕は出力検証をループ外で 1 回だけ行う（反復間処理を揃えるため）。
            let want_len = if matches!(
                arm.kind,
                Kind::GemmHold | Kind::GemmDropIn | Kind::GemmDownload
            ) {
                1
            } else {
                3
            };
            assert_eq!(outs.len(), want_len);
            for out in &outs {
                assert_eq!(
                    bits(out),
                    expect,
                    "arm {} n={n}: 出力が公開経路と bit 不一致",
                    arm.name
                );
            }
        }
    }
}

/// 実機用 record-only 診断（`orchestrate.sh` が独立プロセスで 5 回起動）。
#[test]
#[ignore = "実機計測用の record-only 診断（RULE.txt・orchestrate.sh 参照）"]
fn cpu_gemm_resident_reuse_phases() {
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
                drive(&mut rec, arm, n, (&a, &b), iters_for(n), None);
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
        let checksum_bits = (*phase == "iter_total")
            .then(|| {
                rec.checksums
                    .iter()
                    .find(|(k, _)| *k == (*arm, *n))
                    .map(|(_, b)| format!("{b:016x}"))
            })
            .flatten();
        let line = serde_json::json!({
            "issue": 3646,
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
            "minflt_delta_per_iter": minflt_mean,
            "minflt_enabled": minflt_enabled(),
            "rayon_num_threads_env": threads,
            "target_arch": std::env::consts::ARCH,
            "target_os": std::env::consts::OS,
        });
        println!("{PREFIX}{line}");
    }
}
