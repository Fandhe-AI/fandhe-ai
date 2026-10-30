//! イシュー #3701（`docs/perf/cpu-gemm-reuse-fresh-inversion.md` §16）: #3693 で T2 の主因と判定された
//! S1（`gemm_device` 区間）の内側と S2（`download_owned` 区間）の内側を、計測専用パッチの計時点で分けて測る診断。
//!
//! 役割: #3693 の `cpu_gemm_resident_matmul_split_diag.rs`（変更しない）の診断腕・外側の区間・入力・反復数を
//! 複製し、外側の区間の計時が終わった直後に [`probe_take`] で内側の区間を取り出して記録する。
//! **本ファイルの [`probe_take`] は main では常に `None` を返すスタブ**で、本番コード（`crates/backend-cpu`）には
//! 計時点がない。実機計測では `docs/perf/logs/cpu-gemm-gemm-device-split-3701/probe-3701.patch` を当てた木で
//! 本関数の本体が `fandhe_ai_backend_cpu::split_probe` の取り出し口へ置き換わる
//! （パッチがテスト側で変えるのはこの関数の本体だけ）。
//!
//! - `resident_public`: 公開経路。木（`resident_input.rs` の 1 行の差）に従って経路が変わる唯一の診断腕
//! - `mirror_owned` / `mirror_copy`: 内部 API による処理列の写し（木に依らない）
//! - `alloc_probe_owned` / `alloc_probe_copy`: 計算を除いた確保・書き込み・受け渡しの補助（記録のみ）
//!
//! CI で走る `#[test]`（N=64・128 の bit 一致）はプローブを要求しない。実機用の `#[ignore]` テストは
//! プローブ無しなら panic する（fail-closed。パッチ無しのバイナリはデータを出さない）。
//! 判定規則は `docs/perf/logs/cpu-gemm-gemm-device-split-3701/RULE.txt`（実測前に単独コミットで固定）。
//! 入力は bench-common と同じ `SEED_A = 0xA11CE`・`SEED_B = 0xB0B`。tolerance・baseline は変更しない。新しい `unsafe` は使わない。

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

/// 計時点の区間名（`probe-3701.patch` の `split_probe` と同じ並び。RULE.txt が正）。
const GD_PHASES: [&str; 4] = ["gd_checks", "gd_alloc_zeroed", "gd_compute", "gd_wrap"];
const DO_PHASES: [&str; 5] = [
    "do_checks",
    "do_prepack_clear",
    "do_take",
    "do_tensor_new",
    "do_shell_drop",
];
const DC_PHASES: [&str; 3] = ["dc_checks", "dc_to_vec", "dc_tensor_new"];

/// 計測専用パッチが取り出す内側の区間。固定長配列だけで持ち、取り出しと保持で確保しない
/// （計測窓 `iter_total` 内に Vec の確保・解放を入れず、経路ごとに窓の中身を変えないため）。
/// `gd` は `gemm_device` の内側、`dl` は `download_owned`／`download` の内側（コピー版は先頭 3 要素だけ有効）。
#[derive(Clone, Copy)]
struct ProbeSplit {
    gd: [f64; 4],
    /// `owned`／`copy`／`owned_pool_fallback`。
    dl_kind: &'static str,
    dl: [f64; 5],
}

/// 計時点の取り出し口（main ではスタブで常に `None`）。
/// `probe-3701.patch` が本体を `fandhe_ai_backend_cpu::split_probe::take_*` 呼び出しへ置き換える。
fn probe_take() -> Option<ProbeSplit> {
    None
}

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
    /// true なら `probe_take()` が `None` の反復で panic する（実機診断。パッチ無しバイナリを止める）。
    require_probe: bool,
    /// 窓内で取り出した内側の区間。`iter_total` の記録後に `flush_probe` が cells へ移す（窓内で Vec を伸ばさない）。
    pending_probe: Option<ProbeSplit>,
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

    /// 外側の区間の計時が終わった直後に内側の区間を取り出して記録する。
    /// `want_kind` が Some なら経路種別の一致を要求する（`owned_pool_fallback` など別経路の計測を止める）。
    fn push_probe(&mut self, want_kind: Option<&'static str>) {
        let p = probe_take();
        if self.require_probe && p.is_none() {
            panic!(
                "計時点がない（probe-3701.patch を当てていないバイナリ）。RULE.txt に従いデータを出さない"
            );
        }
        let Some(p) = p else { return };
        if let Some(k) = want_kind {
            assert_eq!(p.dl_kind, k, "arm {}: 経路種別が想定と異なる", self.arm);
        } else {
            assert_ne!(
                p.dl_kind, "owned_pool_fallback",
                "arm {}: プール退避経路は測らない",
                self.arm
            );
        }
        // 確保を伴う記録は窓の外（`flush_probe`）で行う。ここは固定長のコピーだけ。
        self.pending_probe = Some(p);
    }

    /// `iter_total` の記録後に呼び、保留した内側の区間を cells へ記録する。
    fn flush_probe(&mut self) {
        let Some(p) = self.pending_probe.take() else {
            return;
        };
        for (ph, dt) in GD_PHASES.iter().copied().zip(p.gd) {
            self.push(ph, dt);
        }
        if p.dl_kind == "owned" {
            for (ph, dt) in DO_PHASES.iter().copied().zip(p.dl) {
                self.push(ph, dt);
            }
        } else {
            for (ph, dt) in DC_PHASES.iter().copied().zip(p.dl) {
                self.push(ph, dt);
            }
        }
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
            rec.push_probe(None);
            let out = rec.timed("readout", || c.contiguous());
            let sl = out.as_slice().expect("contiguous");
            let s = rec.timed("checksum", || checksum(sl));
            take(sl);
            rec.push("iter_total", t_all.elapsed().as_secs_f64());
            rec.flush_probe();
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
            rec.push_probe(Some("owned"));
            let c = rec.timed("readout", || host.contiguous());
            let sl = c.as_slice().expect("contiguous");
            let s = rec.timed("checksum", || checksum(sl));
            take(sl);
            rec.push("iter_total", t_all.elapsed().as_secs_f64());
            rec.flush_probe();
            rec.record_checksum(s);
            rec.timed("result_drop", move || drop((host, c)));
        }
        Kind::MirrorCopy => {
            let t_all = Instant::now();
            let out = rec.timed("gemm_device", || {
                res.ops.gemm_device(&res.da, &res.db).expect("gemm_device")
            });
            let host = rec.timed("download", || res.ops.download(&out).expect("download"));
            rec.push_probe(Some("copy"));
            rec.timed("out_drop", move || drop(out));
            let c = rec.timed("readout", || host.contiguous());
            let sl = c.as_slice().expect("contiguous");
            let s = rec.timed("checksum", || checksum(sl));
            take(sl);
            rec.push("iter_total", t_all.elapsed().as_secs_f64());
            rec.flush_probe();
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
fn cpu_gemm_s1_inner_split_arms_match_public_path_bit_exact() {
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
#[ignore = "実機計測用の record-only 診断（RULE.txt・orchestrate.sh 参照。probe-3701.patch 必須）"]
fn cpu_gemm_s1_inner_split_phases() {
    let mut rec = Rec {
        require_probe: true,
        ..Rec::default()
    };
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
            "issue": 3701,
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
