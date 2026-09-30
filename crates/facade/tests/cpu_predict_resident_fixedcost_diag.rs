//! イシュー #2105（`docs/perf/cpu-predict-resident-fixedcost.md`）: CPU
//! `Sequential::predict_resident` 固定費の切り分け用フェーズ分解診断。
//!
//! `docs/perf/infer-reuse-phase-breakdown.md` §10.6.2〜§10.6.4 は、
//! DGX Spark GB10 の CPU で reuse（`predict_resident`）が fresh
//! （`predict`）より遅い逆転（+18.9 µs）を報告している。既存ハーネス
//! （bench-fandhe）では `predict_resident` が単一区間で内訳を取れない。
//! 本ファイルは `predict_resident` の呼び出し経路
//! （`tape_for` → `snapshot_resident_params` → `build_device_chain_steps`
//! → `predict_device_chain`）を facade の外側から忠実に写して分解する。
//!
//! - fresh 側の分解は `infer_predict_phase_diag.rs`（#1218）、chain の
//!   bit 一致は `predict_device_chain_cpu_bit_exact.rs`（#1688）が担う。
//!   本ファイルはそれらの間、reuse 側の固定費帰属を担う
//! - `facade::Tape` の内部は `pub(crate)` のため、`tape_for(Device::Cpu)`
//!   と等価な `fandhe_ai_autodiff::Tape::new_with_ops(CpuBackendOps)` を
//!   直接構築して写す（`resolve_ops(Device::Cpu)` と同じ構成）
//! - テスト 1（CI 実行）は、分解した写しが公開 API と bit 一致することを
//!   hard assert する。一致しないと区間への帰属が信用できないため
//! - テスト 2（`#[ignore]`・実機用）は record-only。値は hard gate に
//!   しない。判定規則は `docs/perf/logs/cpu-predict-resident-fixedcost-2105/
//!   RULE.txt`（実測前に固定）に従い、`orchestrate.sh`／`aggregate.py` が
//!   独立 5 プロセスの出力を集計する
//!
//! 本番コード・tolerance・baseline は変更しない。

use std::time::Instant;

use bench_harness::median_q1_q3;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::nn::Linear;
use fandhe_ai_autodiff::optim::DeviceParamStore;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::{Activation, BackendOps, DeviceBuffer, DeviceBufferView};

const BATCH: usize = 64;
const IN_FEATURES: usize = 784;
const HIDDEN: usize = 256;
const OUT_FEATURES: usize = 10;
const WARMUP: usize = 20;
const ROUNDS: usize = 4;
const ITERS_PER_ROUND: usize = 20;
/// 標準出力の 1 行 1 フェーズ JSON の接頭辞（`orchestrate.sh` が抽出）。
const PREFIX: &str = "DIAG_JSON ";

const W1S: [usize; 2] = [IN_FEATURES, HIDDEN];
const B1S: [usize; 1] = [HIDDEN];
const W2S: [usize; 2] = [HIDDEN, OUT_FEATURES];
const B2S: [usize; 1] = [OUT_FEATURES];

fn make_input() -> Tensor<f32> {
    let data = Xorshift64Star::new(0x9000).fill_vec(BATCH * IN_FEATURES);
    Tensor::new(data, &[BATCH, IN_FEATURES]).expect("形状とデータ長は一致させている")
}

fn make_model() -> Sequential {
    Sequential::new()
        .add_linear(IN_FEATURES, HIDDEN, 42)
        .expect("Linear 構築")
        .add_relu()
        .add_linear(HIDDEN, OUT_FEATURES, 43)
        .expect("Linear 構築")
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    host_copy(t).into_iter().map(f32::to_bits).collect()
}

fn host_copy(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous 直後は as_slice が Some")
        .to_vec()
}

fn checksum(v: &[f32]) -> f64 {
    v.iter().map(|x| f64::from(*x)).sum()
}

/// 常駐 weight/bias を `upload` した backend レベルの写し用バッファ群。
struct DeviceParams {
    w1: DeviceBuffer<f32>,
    b1: DeviceBuffer<f32>,
    w2: DeviceBuffer<f32>,
    b2: DeviceBuffer<f32>,
}

fn upload_params(ops: &CpuBackendOps, model: &Sequential) -> DeviceParams {
    let mem = ops.memory_ops().expect("CPU は MemoryOps を実装");
    let p = model.trainable_parameters();
    let up = |i: usize| mem.upload(p[i]).expect("upload");
    DeviceParams {
        w1: up(0),
        b1: up(1),
        w2: up(2),
        b2: up(3),
    }
}

fn l1_forward(ops: &CpuBackendOps, dp: &DeviceParams, a: &DeviceBuffer<f32>) -> DeviceBuffer<f32> {
    ops.linear_forward_device(
        a,
        DeviceBufferView::new(&dp.w1, 0, &W1S).expect("view"),
        Some(DeviceBufferView::new(&dp.b1, 0, &B1S).expect("view")),
        Activation::Relu,
    )
    .expect("L1")
}

fn l2_forward(ops: &CpuBackendOps, dp: &DeviceParams, h: &DeviceBuffer<f32>) -> DeviceBuffer<f32> {
    ops.linear_forward_device(
        h,
        DeviceBufferView::new(&dp.w2, 0, &W2S).expect("view"),
        Some(DeviceBufferView::new(&dp.b2, 0, &B2S).expect("view")),
        Activation::None,
    )
    .expect("L2")
}

/// backend レベルの手動写し（`predict_device_chain` の中身と同じ呼び出し列）。
fn backend_chain(ops: &CpuBackendOps, dp: &DeviceParams, input: &Tensor<f32>) -> Tensor<f32> {
    let mem = ops.memory_ops().expect("CPU は MemoryOps を実装");
    let a = mem.upload(input).expect("upload");
    let h = l1_forward(ops, dp, &a);
    let y = l2_forward(ops, dp, &h);
    mem.download(&y).expect("download")
}

/// autodiff レベルの写し: 自前 Tape → snapshot → 手組み steps → chain。
fn autodiff_chain(store: &DeviceParamStore, input: &Tensor<f32>) -> Tensor<f32> {
    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let leaves = store.snapshot_resident_params(&tape).expect("snapshot");
    let steps = vec![
        (&leaves[0], Some(&leaves[1]), Activation::Relu),
        (&leaves[2], Some(&leaves[3]), Activation::None),
    ];
    store
        .predict_device_chain(&tape, input, &steps)
        .expect("chain")
}

#[test]
fn cpu_predict_resident_decomposition_matches_public_api_bit_exact() {
    let model = make_model();
    let input = make_input();
    let init_tape = fandhe_ai::tape();
    let store = model.init_device_param_store(&init_tape).expect("store");
    drop(init_tape);

    let public = model.predict_resident(&store, &input).expect("public");
    let autodiff = autodiff_chain(&store, &input);
    let ops = CpuBackendOps::new();
    let dp = upload_params(&ops, &model);
    let backend = backend_chain(&ops, &dp, &input);

    assert_eq!(
        bits(&public),
        bits(&autodiff),
        "公開 predict_resident と autodiff レベルの写しが bit 不一致"
    );
    assert_eq!(
        bits(&public),
        bits(&backend),
        "公開 predict_resident と backend レベルの写しが bit 不一致"
    );
}

/// minflt 採取は `/proc` の読み出しが区間ごとに数 µs〜十数 µs かかり、
/// 計測対象（数十〜数百 µs）を歪めるため、既定 OFF。`FIXEDCOST_MINFLT=1`
/// の別 run でのみ有効化する（H3 の補助情報。タイミング系列とは分ける）。
fn minflt_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FIXEDCOST_MINFLT").is_ok_and(|v| v == "1"))
}

#[cfg(target_os = "linux")]
fn minflt() -> Option<u64> {
    if !minflt_enabled() {
        return None;
    }
    let s = std::fs::read_to_string("/proc/self/stat").ok()?;
    // comm は空白・括弧を含みうるため最後の ')' 以降で分割する。
    // 以降のフィールド: state ppid pgrp session tty tpgid flags minflt ...
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
    checksum: Option<u64>,
}

/// (arm, phase) ごとのサンプル蓄積器。
#[derive(Default)]
struct Rec {
    cells: Vec<((&'static str, &'static str), Cell)>,
    /// 記録するか（warmup 中は false）。
    on: bool,
}

impl Rec {
    fn cell(&mut self, arm: &'static str, phase: &'static str) -> &mut Cell {
        let i = match self.cells.iter().position(|(k, _)| *k == (arm, phase)) {
            Some(i) => i,
            None => {
                self.cells.push(((arm, phase), Cell::default()));
                self.cells.len() - 1
            }
        };
        &mut self.cells[i].1
    }

    fn push(&mut self, arm: &'static str, phase: &'static str, dt: f64, f0: Option<u64>) {
        if !self.on {
            return;
        }
        let f1 = minflt();
        let c = self.cell(arm, phase);
        c.secs.push(dt);
        match (f0, f1) {
            (Some(a), Some(b)) => c.flts.push(b.saturating_sub(a)),
            _ => c.flt_missing = true,
        }
    }

    /// 区間を計測して結果を返す。
    fn timed<T>(&mut self, arm: &'static str, phase: &'static str, f: impl FnOnce() -> T) -> T {
        let f0 = minflt();
        let t = Instant::now();
        let out = f();
        let dt = t.elapsed().as_secs_f64();
        self.push(arm, phase, dt, f0);
        out
    }
}

struct Ctx {
    model: Sequential,
    store: DeviceParamStore,
    input: Tensor<f32>,
    l1: Linear,
    l2: Linear,
    h_fixture: Tensor<f32>,
    ops: CpuBackendOps,
    dp: DeviceParams,
}

const ARMS: [&str; 4] = ["fresh", "reuse", "reuse_decomposed", "ablation"];

fn run_public(c: &Ctx, rec: &mut Rec, arm: &'static str) {
    let f0 = minflt();
    let t = Instant::now();
    let out = if arm == "fresh" {
        rec.timed(arm, "predict", || {
            c.model.predict(&c.input).expect("predict")
        })
    } else {
        rec.timed(arm, "predict", || {
            c.model
                .predict_resident(&c.store, &c.input)
                .expect("resident")
        })
    };
    let v = rec.timed(arm, "host_copy", || host_copy(&out));
    let s = rec.timed(arm, "checksum", || checksum(&v));
    rec.push(arm, "iter_total", t.elapsed().as_secs_f64(), f0);
    if rec.on {
        rec.cell(arm, "iter_total").checksum = Some(s.to_bits());
    }
}

fn run_decomposed(c: &Ctx, rec: &mut Rec) {
    let arm = "reuse_decomposed";
    let ops = &c.ops;
    let mem = ops.memory_ops().expect("MemoryOps");
    let f_all = minflt();
    let t_all = Instant::now();

    let f_b = minflt();
    let t_b = Instant::now();
    let tape = rec.timed(arm, "tape_new", || {
        Tape::new_with_ops(Box::new(CpuBackendOps::new()))
    });
    let leaves = rec.timed(arm, "snapshot", || {
        c.store.snapshot_resident_params(&tape).expect("snapshot")
    });
    let steps = vec![
        (&leaves[0], Some(&leaves[1]), Activation::Relu),
        (&leaves[2], Some(&leaves[3]), Activation::None),
    ];
    std::hint::black_box(&steps);
    rec.push(arm, "tape_build", t_b.elapsed().as_secs_f64(), f_b);

    let f_f = minflt();
    let t_f = Instant::now();
    let a = rec.timed(arm, "upload", || mem.upload(&c.input).expect("upload"));
    let h = rec.timed(arm, "l1_linear_forward_device", || {
        l1_forward(ops, &c.dp, &a)
    });
    let y = rec.timed(arm, "l2_linear_forward_device", || {
        l2_forward(ops, &c.dp, &h)
    });
    rec.push(arm, "forward_resident", t_f.elapsed().as_secs_f64(), f_f);

    let out = rec.timed(arm, "readout", || mem.download(&y).expect("download"));
    let v = rec.timed(arm, "host_copy", || host_copy(&out));
    let s = rec.timed(arm, "checksum", || checksum(&v));
    rec.push(arm, "iter_total", t_all.elapsed().as_secs_f64(), f_all);
    if rec.on {
        rec.cell(arm, "iter_total").checksum = Some(s.to_bits());
    }

    // 上の分解は `linear_forward_device` を直接呼ぶため、公開経路
    // （`predict_device_chain`）が持つ `check_not_poisoned`／`check_device`・
    // tape_id 検証・`checked_resident_buffer`・形状検証・
    // `linear_forward_device_tracked`・事後 poison 再検査は区間に含まれない。
    // 省略分を別区間 `chain_public` として同じ tape・steps で実測する
    // （iter_total の外側。`aggregate.py` が
    // `chain_public - forward_resident - readout` を検証・tracked 差として
    // H4 の残差から分離して帰属する。RULE.txt 参照）。
    rec.timed(arm, "chain_public", || {
        std::hint::black_box(
            c.store
                .predict_device_chain(&tape, &c.input, &steps)
                .expect("chain"),
        )
    });
}

/// 帰属用の比較のみ（fresh の層別内訳・H1: 非融合 L1・H2: 入力コピー単独）。
fn run_ablation(c: &Ctx, rec: &mut Rec) {
    let ops = &c.ops;
    rec.timed("fresh_layers", "l1_linear_relu_fused", || {
        std::hint::black_box(
            c.l1.forward_host_with_activation(ops, &c.input, Activation::Relu)
                .expect("L1 fused"),
        )
    });
    rec.timed("fresh_layers", "l2_linear_gemm_add", || {
        let y = ops.gemm(&c.h_fixture, c.l2.weight()).expect("gemm");
        std::hint::black_box(ops.add(&y, c.l2.bias().expect("bias")).expect("add"))
    });
    rec.timed("ablation", "l1_host_unfused_gemm_add_relu", || {
        let y = ops.gemm(&c.input, c.l1.weight()).expect("gemm");
        let y = ops.add(&y, c.l1.bias().expect("bias")).expect("add");
        std::hint::black_box(ops.relu(&y).expect("relu"))
    });
    rec.timed("ablation", "input_to_vec_copy", || {
        std::hint::black_box(host_copy(&c.input))
    });
}

fn run_arm(c: &Ctx, rec: &mut Rec, arm: &'static str) {
    match arm {
        "reuse_decomposed" => run_decomposed(c, rec),
        "ablation" => run_ablation(c, rec),
        _ => run_public(c, rec, arm),
    }
}

/// 実機用 record-only 診断（`orchestrate.sh` が独立プロセスで 5 回起動）。
#[test]
#[ignore = "実機計測用の record-only 診断（RULE.txt・orchestrate.sh 参照）"]
fn cpu_predict_resident_fixedcost_phases() {
    let model = make_model();
    let init_tape = fandhe_ai::tape();
    let store = model.init_device_param_store(&init_tape).expect("store");
    drop(init_tape);
    let ops = CpuBackendOps::new();
    let dp = upload_params(&ops, &model);
    let input = make_input();
    let l1 = Linear::new(IN_FEATURES, HIDDEN, true, 42).expect("L1");
    let l2 = Linear::new(HIDDEN, OUT_FEATURES, true, 43).expect("L2");
    let h_fixture = l1
        .forward_host_with_activation(&ops, &input, Activation::Relu)
        .expect("h");
    let c = Ctx {
        model,
        store,
        input,
        l1,
        l2,
        h_fixture,
        ops,
        dp,
    };

    let mut rec = Rec::default();
    for _ in 0..WARMUP {
        for arm in ARMS {
            run_arm(&c, &mut rec, arm);
        }
    }
    rec.on = true;
    for round in 0..ROUNDS {
        // 順序効果（H5）を打ち消すためラウンドごとに arm 順を反転する。
        let order: Vec<&'static str> = if round % 2 == 0 {
            ARMS.to_vec()
        } else {
            ARMS.iter().rev().copied().collect()
        };
        for _ in 0..ITERS_PER_ROUND {
            for arm in &order {
                run_arm(&c, &mut rec, arm);
            }
        }
    }

    let threads = std::env::var("RAYON_NUM_THREADS").ok();
    for ((arm, phase), cell) in &rec.cells {
        let q = median_q1_q3(&cell.secs).expect("サンプルは非空・非 NaN");
        let mut sorted = cell.secs.clone();
        sorted.sort_by(f64::total_cmp);
        let minflt_mean = if !cell.flt_missing && !cell.flts.is_empty() {
            Some(cell.flts.iter().sum::<u64>() as f64 / cell.flts.len() as f64)
        } else {
            None
        };
        let line = serde_json::json!({
            "issue": 2105,
            "arm": arm,
            "phase": phase,
            "median_s": q.median,
            "q1_s": q.q1,
            "q3_s": q.q3,
            "min_s": sorted.first(),
            "max_s": sorted.last(),
            "n": cell.secs.len(),
            "checksum_bits": cell.checksum.map(|b| format!("{b:016x}")),
            "minflt_delta_per_iter": minflt_mean,
            "minflt_enabled": minflt_enabled(),
            "rayon_num_threads_env": threads,
            "target_arch": std::env::consts::ARCH,
            "target_os": std::env::consts::OS,
        });
        println!("{PREFIX}{line}");
    }
}
