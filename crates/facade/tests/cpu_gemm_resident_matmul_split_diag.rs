//! イシュー #3693（`docs/perf/cpu-gemm-reuse-fresh-inversion.md` §15）: `matmul_resident` の末尾を
//! `download` から `download_owned` に変えたあと Mac mini の `resident_public` N=256 に残った後退（T2）が、
//! matmul 区間の内側のどこから来るかを切り分けるための細分化診断。
//!
//! 役割: `cpu_gemm_resident_reuse_diag.rs`（#3646・#3679 の計測器。変更しない）は matmul 区間を
//! これ以上分けられない。本ファイルは after の処理列（`gemm_device` → `download_owned`）と
//! before の処理列（`gemm_device` → `download` → 出力の drop）を内部 API（`MemoryOps`）で写した
//! 診断腕を足し、同一プロセス内で区間ごとの差を取れるようにする。
//!
//! - `resident_public`: 公開経路。木（`resident_input.rs` の 1 行の差）に従って経路が変わる唯一の診断腕
//! - `mirror_owned` / `mirror_copy`: 内部 API による処理列の写し。**木に依らない**
//!   （木の間で写しの ratio を見ると「木の差が経路の外にも出ているか」の対照になる）
//! - `alloc_probe_owned` / `alloc_probe_copy`: 計算を除いた確保・書き込み・受け渡しの補助（記録のみ）
//!
//! 判定規則は `docs/perf/logs/cpu-gemm-matmul-split-3693/RULE.txt`（実測前に単独コミットで固定）。
//! `orchestrate.sh`／`split_judge.py` が独立プロセスの出力を集計する。写しは公開経路の dyn ディスパッチと
//! 入力検証を通らないため、RULE の健全性 B（写しが T2 を再現するか）で確かめる。
//! テスト側から `CpuBufferHandle` は見えないので、`take_data`・prepack 破棄・抜け殻の drop は
//! `download_owned` の 1 区間に合算される（分けるには本番側に計時点が要り、本 Issue の制約と合わない）。
//!
//! 入力は bench-common と同じ `SEED_A = 0xA11CE`・`SEED_B = 0xB0B`（独立 workspace のため値を複製）。
//! 本番コード・tolerance・baseline は変更しない。新しい `unsafe` は使わない。

use std::time::Instant;

use bench_harness::median_q1_q3;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, Tape, Tensor, tape_for};
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::{BackendOps, DeviceBuffer, MemoryOps};

/// bench-common `SEED_A`／`SEED_B` と同値（出典: framework-compare/bench-common/src/lib.rs）。
const SEED_A: u64 = 0xA11CE;
const SEED_B: u64 = 0xB0B;
const ROUNDS: usize = 4;
/// N=256 が判定対象、N=512 は記録のみ（RULE.txt）。N=256 を先に処理する（#3679 と同じ）。
const SIZES: [usize; 2] = [256, 512];
/// 標準出力の 1 行 1 区間 JSON の接頭辞（`orchestrate.sh` が抽出）。
const PREFIX: &str = "DIAG_JSON ";

/// N ごとの (warmup, 計測) 反復数。縮めない（RULE.txt）。
fn iters_for(n: usize) -> (usize, usize) {
    if n >= 4096 { (5, 10) } else { (20, 20) }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    ResidentPublic,
    MirrorOwned,
    MirrorCopy,
    AllocOwned,
    AllocCopy,
}

#[derive(Clone, Copy)]
struct Arm {
    name: &'static str,
    kind: Kind,
}

const ARMS: [Arm; 5] = [
    Arm {
        name: "resident_public",
        kind: Kind::ResidentPublic,
    },
    Arm {
        name: "mirror_owned",
        kind: Kind::MirrorOwned,
    },
    Arm {
        name: "mirror_copy",
        kind: Kind::MirrorCopy,
    },
    Arm {
        name: "alloc_probe_owned",
        kind: Kind::AllocOwned,
    },
    Arm {
        name: "alloc_probe_copy",
        kind: Kind::AllocCopy,
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

/// alloc_probe の決定的な書き込み値（全要素に書いて初回ページ参照を起こす。値は腕間で同一）。
fn probe_value(i: usize) -> f32 {
    (i % 1021) as f32 * 0.5
}

#[derive(Default)]
struct Cell {
    secs: Vec<f64>,
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
    fn push(&mut self, phase: &'static str, dt: f64) {
        if !self.on {
            return;
        }
        let key = (self.arm, self.n, phase);
        let i = match self.cells.iter().position(|(k, _)| *k == key) {
            Some(i) => i,
            None => {
                self.cells.push((key, Cell::default()));
                self.cells.len() - 1
            }
        };
        self.cells[i].1.secs.push(dt);
    }

    fn timed<T>(&mut self, phase: &'static str, f: impl FnOnce() -> T) -> T {
        let t = Instant::now();
        let out = f();
        let dt = t.elapsed().as_secs_f64();
        self.push(phase, dt);
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

/// 常駐腕の共有状態。facade 公開経路（`ResidentTensor`）と内部 API（`DeviceBuffer`）を同じ入力で持つ。
/// upload はすべてブロック開始時（計測窓の外）。
struct Resident {
    tape: Tape,
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
            ra,
            rb,
            ops,
            da,
            db,
        }
    }
}

/// 1 反復。窓は `iter_total`。窓の外（`result_drop`）では、結果を保持する変数の drop を
/// 明示的に計時する（既存診断では暗黙だった位置と同じ）。
fn one_iter(rec: &mut Rec, arm: Arm, res: &mut Resident, n: usize, capture: Option<&mut Vec<f32>>) {
    let mut cap = capture;
    let mut take = |s: &[f32]| {
        if let Some(c) = cap.as_deref_mut() {
            *c = s.to_vec();
        }
    };
    match arm.kind {
        Kind::ResidentPublic => {
            let t_all = Instant::now();
            let c = rec.timed("matmul", || {
                res.tape
                    .matmul_resident(&res.ra, &res.rb)
                    .expect("matmul_resident")
            });
            let out = rec.timed("readout", || c.contiguous());
            let sl = out.as_slice().expect("contiguous");
            let s = rec.timed("checksum", || checksum(sl));
            take(sl);
            rec.push("iter_total", t_all.elapsed().as_secs_f64());
            rec.record_checksum(s);
            rec.timed("result_drop", move || drop((c, out)));
        }
        Kind::MirrorOwned => {
            let t_all = Instant::now();
            let out = rec.timed("gemm_device", || {
                res.ops.gemm_device(&res.da, &res.db).expect("gemm_device")
            });
            let host = rec.timed("download_owned", || {
                res.ops.download_owned(out).expect("download_owned")
            });
            let c = rec.timed("readout", || host.contiguous());
            let sl = c.as_slice().expect("contiguous");
            let s = rec.timed("checksum", || checksum(sl));
            take(sl);
            rec.push("iter_total", t_all.elapsed().as_secs_f64());
            rec.record_checksum(s);
            rec.timed("result_drop", move || drop((host, c)));
        }
        Kind::MirrorCopy => {
            let t_all = Instant::now();
            let out = rec.timed("gemm_device", || {
                res.ops.gemm_device(&res.da, &res.db).expect("gemm_device")
            });
            let host = rec.timed("download", || res.ops.download(&out).expect("download"));
            rec.timed("out_drop", move || drop(out));
            let c = rec.timed("readout", || host.contiguous());
            let sl = c.as_slice().expect("contiguous");
            let s = rec.timed("checksum", || checksum(sl));
            take(sl);
            rec.push("iter_total", t_all.elapsed().as_secs_f64());
            rec.record_checksum(s);
            rec.timed("result_drop", move || drop((host, c)));
        }
        Kind::AllocOwned => {
            let t_all = Instant::now();
            let mut v = rec.timed("alloc", || vec![0.0f32; n * n]);
            rec.timed("touch", || {
                for (i, x) in v.iter_mut().enumerate() {
                    *x = probe_value(i);
                }
            });
            let t = rec.timed("wrap", || Tensor::new(v, &[n, n]).expect("形状一致"));
            rec.push("iter_total", t_all.elapsed().as_secs_f64());
            let sl = t.as_slice().expect("contiguous");
            rec.record_checksum(checksum(sl));
            take(sl);
            rec.timed("result_drop", move || drop(t));
        }
        Kind::AllocCopy => {
            let t_all = Instant::now();
            let mut v = rec.timed("alloc", || vec![0.0f32; n * n]);
            rec.timed("touch", || {
                for (i, x) in v.iter_mut().enumerate() {
                    *x = probe_value(i);
                }
            });
            let copy = rec.timed("copy", || v.to_vec());
            rec.timed("src_drop", move || drop(v));
            rec.push("iter_total", t_all.elapsed().as_secs_f64());
            rec.record_checksum(checksum(&copy));
            take(&copy);
            rec.timed("result_drop", move || drop(copy));
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
        one_iter(rec, arm, &mut res, n, want.then_some(&mut out));
        if want
            && !out.is_empty()
            && let Some(c) = captured.as_deref_mut()
        {
            c.push(out);
        }
    }
    rec.on = false;
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

/// 公開経路と 3 つの計算腕が bit 一致し、alloc_probe 2 腕は互いに checksum が一致する（CI 実行）。
#[test]
fn cpu_gemm_resident_matmul_split_arms_match_public_path_bit_exact() {
    for n in [64usize, 128] {
        let (a, b) = inputs(n);
        let expect = bits(&public_output(&a, &b));
        let mut probe_sums: Vec<u64> = Vec::new();
        for arm in ARMS {
            let mut rec = Rec::default();
            let mut outs = Vec::new();
            drive(&mut rec, arm, n, (&a, &b), (0, 3), Some(&mut outs));
            assert_eq!(outs.len(), 3);
            match arm.kind {
                Kind::AllocOwned | Kind::AllocCopy => {
                    let s = rec
                        .checksums
                        .iter()
                        .find(|(k, _)| *k == (arm.name, n))
                        .map(|(_, b)| *b)
                        .expect("checksum が記録される");
                    probe_sums.push(s);
                }
                _ => {
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
        assert_eq!(probe_sums.len(), 2);
        assert_eq!(
            probe_sums[0], probe_sums[1],
            "alloc_probe の checksum 不一致"
        );
    }
}

/// 実機用 record-only 診断（`orchestrate.sh` が独立プロセスで 5 回起動）。
#[test]
#[ignore = "実機計測用の record-only 診断（RULE.txt・orchestrate.sh 参照）"]
fn cpu_gemm_resident_matmul_split_phases() {
    let mut rec = Rec::default();
    for n in SIZES {
        let (a, b) = inputs(n);
        for round in 0..ROUNDS {
            // 順序効果を打ち消すためラウンドごとに診断腕の順を反転する。
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
        let checksum_bits = (*phase == "iter_total")
            .then(|| {
                rec.checksums
                    .iter()
                    .find(|(k, _)| *k == (*arm, *n))
                    .map(|(_, b)| format!("{b:016x}"))
            })
            .flatten();
        let line = serde_json::json!({
            "issue": 3693,
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
            "rayon_num_threads_env": threads,
            "target_arch": std::env::consts::ARCH,
            "target_os": std::env::consts::OS,
        });
        println!("{PREFIX}{line}");
    }
}
