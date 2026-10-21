//! イシュー #3188（再オープン分）: CPU reuse 学習の `device_update` のうち、
//! 2106 の「H4 prologue + 残差」（head の GB10 で約 142.8 µs、T の 84%）の内訳診断。
//!
//! 2106（`cpu_reuse_device_update_diag.rs`）は T と standalone の `sgd_kernel`・`stage` を
//! 測り、残差 H4 = T - (K + S) を未分解のまま残した。本ファイルは facade の公開 API
//! だけで、H4 を次の 2 項へ分ける（本番へ計装は入れない。2105/2106 と同じ方針）。
//!
//! - H4a（prologue）: `DeviceParamStore::step` の検査・小 Vec 構築・`resident_filled_slots`・
//!   bias 勾配の `grads.get`／`clone`。
//! - H4b（in-situ カーネルの超過分）: 学習ループ内の `sgd_step_device` が standalone より
//!   遅い分（staging は backward の rayon ワーカーが書いた dirty line、params は forward の
//!   ワーカーが読んだ状態になるため）。
//!
//! # prologue probe の根拠
//!
//! `step` は `sgd_used == true` かつ velocity 未確保のとき、`momentum != 0` の
//! `SgdConfig` を受けると「momentum configuration changed mid-training」の
//! `InvalidArgument` を返す（`crates/autodiff/src/optim/device_store.rs` の momentum 一貫性
//! 検査）。この検査は検査群・`vars` 構築・`resident_filled_slots`・bias 勾配の
//! `grads.get`／shape 検査／`clone` の全部を通った後で、`pending.take()`・`upload_into`・
//! カーネルより前にある。よって「momentum 0.9 の step が Err を返すまで」の時間 P は
//! prologue の実測になる（エラー文字列の `format!` を含むため僅かに過大）。`pending` は
//! 消費されないので、直後に本物の step（momentum 0）を呼べる。その時間を R とする。
//!
//! # 構成
//!
//! - テスト 1（CI 実行）: probe が期待どおり Err になり params を変えないこと、probe 付き
//!   系列と direct 系列が bit 一致することを hard assert する。崩れたら帰属は信用できない。
//! - テスト 2（`#[ignore]`・実機用）: record-only。標準出力へ 1 行 1 区間の `DIAG_JSON`
//!   を出す。判定規則は `docs/perf/logs/cpu-train-fixedcost-gb10-3188/reopen/RULE.txt`
//!   （実測前に固定）で、`orchestrate_h4.sh`／`aggregate_h4.py` が集計する。
//!
//! 補助情報（判定に使わない）として Linux では `/proc/thread-self/stat` から minflt 差と
//! 実行コア（processor 欄）の移動回数を記録する（std のみ・unsafe/libc なし）。
//!
//! 本番コード・tolerance・baseline は変更しない。

use std::time::Instant;

use bench_harness::median_q1_q3;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{DeviceParamStore, SgdConfig, Tape, Tensor};
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::{BackendError, BackendOps, DeviceBuffer, MemoryOps, SgdStepConfig};

const BATCH: usize = 64;
const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;
const LR: f32 = 0.01;
const WARMUP: usize = 20;
const ROUNDS: usize = 4;
const ITERS_PER_ROUND: usize = 20;
/// 標準出力の 1 行 1 区間 JSON の接頭辞（`orchestrate_h4.sh` が抽出）。
const PREFIX: &str = "DIAG_JSON ";
/// `zip_xcore` 腕で勾配 Vec を書くスレッド数（2106 の `XTHREADS` と同じ）。
const XTHREADS: usize = 4;
/// cold 腕で LLC を追い出すために触るバッファ（256 MiB）。
const EVICT_BYTES: usize = 256 << 20;

fn make_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, 42)
        .expect("Linear 構築")
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, 43)
        .expect("Linear 構築")
}

fn make_data() -> (Tensor<f32>, Tensor<f32>) {
    let x = Xorshift64Star::new(0x2106).fill_vec(BATCH * D_IN);
    let y = Xorshift64Star::new(0x2107).fill_vec(BATCH * D_OUT);
    (
        Tensor::new(x, &[BATCH, D_IN]).expect("形状一致"),
        Tensor::new(y, &[BATCH, D_OUT]).expect("形状一致"),
    )
}

fn new_store(model: &Sequential) -> DeviceParamStore {
    let tape = fandhe_ai::tape();
    let store = model.init_device_param_store(&tape).expect("store");
    // bench-fandhe と同じく init 直後に 1 回 sync してから tape を捨てる。
    let _ = tape.sync_device_param_store_to_host(&store).expect("sync");
    store
}

fn flatten(ts: &[Tensor<f32>]) -> Vec<f32> {
    let mut v = Vec::new();
    for t in ts {
        v.extend_from_slice(t.contiguous().as_slice().expect("contiguous"));
    }
    v
}

fn bits_hash(v: &[f32]) -> u64 {
    // FNV-1a（決定的・依存なし）。
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn store_params(store: &DeviceParamStore) -> Vec<f32> {
    let tape = fandhe_ai::tape();
    flatten(&tape.sync_device_param_store_to_host(store).expect("sync"))
}

/// forward → backward までを行い（未計時）、勾配を返す。
fn fwd_bwd(
    tape: &Tape,
    model: &Sequential,
    store: &mut DeviceParamStore,
    x: &Tensor<f32>,
    y: &Tensor<f32>,
) -> fandhe_ai::Gradients {
    let xv = tape.var(x);
    let yv = tape.var(y);
    let pred = model.forward_resident(tape, &xv, store).expect("fwd");
    let loss = pred.mse_loss(&yv).expect("mse");
    tape.backward_device_param_store(&loss, store).expect("bwd")
}

fn sgd_cfg() -> SgdStepConfig {
    SgdStepConfig {
        lr: LR,
        momentum: 0.0,
        dampening: 0.0,
        weight_decay: 0.0,
        nesterov: false,
        is_first_step: false,
    }
}

fn upload_flat(mem: &dyn MemoryOps, v: &[f32]) -> DeviceBuffer<f32> {
    let t = Tensor::new(v.to_vec(), &[v.len()]).expect("形状一致");
    mem.upload(&t).expect("upload")
}

fn download_flat(mem: &dyn MemoryOps, b: &DeviceBuffer<f32>) -> Vec<f32> {
    let mut out = Vec::new();
    mem.with_host_view(b, &mut |s| out = s.to_vec())
        .expect("host view");
    out
}

fn zip_kernel(p: &mut [f32], g: &[f32]) {
    for (p, g) in p.iter_mut().zip(g) {
        *p -= LR * *g;
    }
}

/// probe（momentum 0.9 の step）。期待どおり momentum 一貫性検査で Err になったかを返す。
fn probe_step(tape: &Tape, store: &mut DeviceParamStore, grads: &fandhe_ai::Gradients) -> bool {
    let cfg = SgdConfig::new(LR).with_momentum(0.9);
    match tape.step_device_param_store(store, grads, &cfg) {
        Err(BackendError::InvalidArgument(m)) => m.contains("momentum configuration changed"),
        _ => false,
    }
}

// ---- テスト 1（CI 実行） ----

#[test]
fn cpu_reuse_device_update_h4_probe_contract_holds_bit_exact() {
    let model = make_model();
    let (x, y) = make_data();
    let mut direct = new_store(&model);
    let mut probed = new_store(&model);
    // 1 step 目は sgd_used を立てるための本物の step（probe は 2 step 目以降で有効）。
    for i in 0..4 {
        let tape = fandhe_ai::tape();
        let g = fwd_bwd(&tape, &model, &mut direct, &x, &y);
        tape.step_device_param_store(&mut direct, &g, &SgdConfig::new(LR))
            .expect("direct step");

        let tape = fandhe_ai::tape();
        let g = fwd_bwd(&tape, &model, &mut probed, &x, &y);
        if i > 0 {
            // CPU では weight slot が resident 経由（Some）・bias slot が host 経由（None）。
            let resident = tape.resident_grads_to_host(&probed, &g).expect("resident");
            let some: Vec<bool> = resident.iter().map(Option::is_some).collect();
            assert_eq!(
                some,
                vec![true, false, true, false],
                "CPU の resident 勾配 slot 構成が前提と異なる。分解を見直すこと"
            );
            let before = bits_hash(&store_params(&probed));
            assert!(
                probe_step(&tape, &mut probed, &g),
                "probe が momentum 一貫性検査の Err にならない（prologue の帰属が成立しない）"
            );
            assert_eq!(
                before,
                bits_hash(&store_params(&probed)),
                "probe が params を変えた"
            );
        }
        tape.step_device_param_store(&mut probed, &g, &SgdConfig::new(LR))
            .expect("probed 本物の step が probe 後に失敗");
    }
    assert_eq!(
        bits_hash(&store_params(&direct)),
        bits_hash(&store_params(&probed)),
        "probe 付き系列と direct 系列が bit 不一致（区間を比較できない）"
    );
}

// ---- テスト 2（実機用・record-only） ----

#[derive(Default)]
struct Cell {
    secs: Vec<f64>,
    checksum: Option<u64>,
    minflt: u64,
    core_moves: u64,
    proc_ok: bool,
}

#[derive(Default)]
struct Rec {
    cells: Vec<((&'static str, &'static str), Cell)>,
    on: bool,
}

/// `/proc/thread-self/stat` の (minflt, processor)。Linux 以外・解析失敗は None。
fn thread_stat() -> Option<(u64, u64)> {
    let s = std::fs::read_to_string("/proc/thread-self/stat").ok()?;
    // comm は括弧内に空白を含み得るため最後の ')' 以降を数える。state は field 3 = idx 0。
    let rest: Vec<&str> = s[s.rfind(')')? + 1..].split_whitespace().collect();
    let minflt = rest.get(7)?.parse().ok()?; // field 10
    let cpu = rest.get(36)?.parse().ok()?; // field 39
    Some((minflt, cpu))
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

    fn timed<T>(&mut self, arm: &'static str, phase: &'static str, f: impl FnOnce() -> T) -> T {
        let s0 = if self.on { thread_stat() } else { None };
        let t = Instant::now();
        let out = f();
        let dt = t.elapsed().as_secs_f64();
        if self.on {
            let s1 = thread_stat();
            let c = self.cell(arm, phase);
            c.secs.push(dt);
            if let (Some((f0, c0)), Some((f1, c1))) = (s0, s1) {
                c.proc_ok = true;
                c.minflt += f1.saturating_sub(f0);
                c.core_moves += u64::from(c0 != c1);
            }
        }
        out
    }

    fn set_checksum(&mut self, arm: &'static str, phase: &'static str, v: u64) {
        self.cell(arm, phase).checksum = Some(v);
    }
}

/// standalone 区間用の固定 fixture（total_numel・layout は本物の store と同一）。
struct Standalone {
    ops: CpuBackendOps,
    total: usize,
    g: Vec<f32>,
    p_hot: DeviceBuffer<f32>,
    p_cold: DeviceBuffer<f32>,
    g_hot: DeviceBuffer<f32>,
    g_cold: DeviceBuffer<f32>,
    stage_dst: DeviceBuffer<f32>,
    b1: Tensor<f32>,
    b2: Tensor<f32>,
    stage_off1: usize,
    stage_off2: usize,
    zip_p_hot: Vec<f32>,
    zip_g_hot: Vec<f32>,
    zip_p_x: Vec<f32>,
    zip_g_x: Vec<f32>,
    evict: Vec<f32>,
}

impl Standalone {
    fn new(p0: &[f32], g: &[f32]) -> Self {
        let ops = CpuBackendOps::new();
        let mem = ops.memory_ops().expect("MemoryOps");
        let total = p0.len();
        let (w1, w2) = (D_IN * D_HIDDEN, D_HIDDEN * D_OUT);
        let stage_off1 = w1;
        let stage_off2 = w1 + D_HIDDEN + w2;
        Self {
            total,
            g: g.to_vec(),
            p_hot: upload_flat(mem, p0),
            p_cold: upload_flat(mem, p0),
            g_hot: upload_flat(mem, g),
            g_cold: upload_flat(mem, g),
            stage_dst: upload_flat(mem, &vec![0.0; total]),
            b1: Tensor::new(g[stage_off1..stage_off1 + D_HIDDEN].to_vec(), &[D_HIDDEN])
                .expect("形状一致"),
            b2: Tensor::new(g[stage_off2..stage_off2 + D_OUT].to_vec(), &[D_OUT])
                .expect("形状一致"),
            stage_off1,
            stage_off2,
            zip_p_hot: p0.to_vec(),
            zip_g_hot: g.to_vec(),
            zip_p_x: p0.to_vec(),
            zip_g_x: g.to_vec(),
            evict: vec![0.0; EVICT_BYTES / 4],
            ops,
        }
    }

    fn run_all(&mut self, rec: &mut Rec) {
        let total = self.total;
        let mem = self.ops.memory_ops().expect("MemoryOps");
        // stage: bias 2 本を total_numel 長バッファのオフセットへ書き込む。
        let (o1, o2) = (self.stage_off1, self.stage_off2);
        let (b1, b2) = (&self.b1, &self.b2);
        let dst = &mut self.stage_dst;
        rec.timed("standalone", "stage", || {
            mem.upload_into(b1, dst, o1).expect("stage b1");
            mem.upload_into(b2, dst, o2).expect("stage b2");
        });
        // K（hot）: 勾配は直前に同スレッドで書いた状態。
        let gt = Tensor::new(self.g.clone(), &[total]).expect("形状一致");
        mem.upload_into(&gt, &mut self.g_hot, 0).expect("g write");
        let ops = &self.ops;
        let (ph, gh) = (&mut self.p_hot, &self.g_hot);
        rec.timed("standalone", "sgd_kernel", || {
            ops.sgd_step_device(ph, gh, None, &sgd_cfg()).expect("sgd");
        });
        // K（cold）: 勾配を書いた後、256 MiB を触って LLC を追い出してから計時。
        mem.upload_into(&gt, &mut self.g_cold, 0).expect("g write");
        for v in self.evict.iter_mut().step_by(16) {
            *v += 1.0;
        }
        std::hint::black_box(&self.evict);
        let (pc, gc) = (&mut self.p_cold, &self.g_cold);
        rec.timed("standalone", "sgd_kernel_cold", || {
            ops.sgd_step_device(pc, gc, None, &sgd_cfg()).expect("sgd");
        });
        // zip（hot）: main が書いた勾配 Vec に対する zip 形更新。
        self.zip_g_hot.copy_from_slice(&self.g);
        let (zp, zg) = (&mut self.zip_p_hot, &self.zip_g_hot);
        rec.timed("standalone", "zip_hot", || zip_kernel(zp, zg));
        // zip（xcore）: 別スレッド N 本が勾配 Vec をチャンク書き込みした直後の main の zip。
        let chunk = total.div_ceil(XTHREADS);
        let src = &self.g;
        std::thread::scope(|s| {
            for (dstc, srcc) in self.zip_g_x.chunks_mut(chunk).zip(src.chunks(chunk)) {
                s.spawn(move || dstc.copy_from_slice(srcc));
            }
        });
        let (xp, xg) = (&mut self.zip_p_x, &self.zip_g_x);
        rec.timed("standalone", "zip_xcore", || zip_kernel(xp, xg));
    }

    /// 最終状態の bit ハッシュ（全腕が同一回数だけ同一更新を受けた後の健全性検査）。
    fn final_hashes(&self) -> [(&'static str, u64); 4] {
        let mem = self.ops.memory_ops().expect("MemoryOps");
        [
            ("sgd_kernel", bits_hash(&download_flat(mem, &self.p_hot))),
            (
                "sgd_kernel_cold",
                bits_hash(&download_flat(mem, &self.p_cold)),
            ),
            ("zip_hot", bits_hash(&self.zip_p_hot)),
            ("zip_xcore", bits_hash(&self.zip_p_x)),
        ]
    }
}

/// in-situ 腕: 本物の学習ループ内で計時する。2 つの store（同一初期値・同一データ）を
/// 反復ごとに進める。probe は params を変えないため最終 bit は一致する。
struct InSitu {
    model: Sequential,
    direct: DeviceParamStore,
    probed: DeviceParamStore,
    x: Tensor<f32>,
    y: Tensor<f32>,
}

impl InSitu {
    fn step_direct(&mut self, rec: &mut Rec) {
        let tape = fandhe_ai::tape();
        let grads = fwd_bwd(&tape, &self.model, &mut self.direct, &self.x, &self.y);
        let cfg = SgdConfig::new(LR);
        let store = &mut self.direct;
        rec.timed("insitu_direct", "device_update", || {
            tape.step_device_param_store(store, &grads, &cfg)
                .expect("step");
        });
    }

    fn step_probed(&mut self, rec: &mut Rec, first: bool) {
        let tape = fandhe_ai::tape();
        let grads = fwd_bwd(&tape, &self.model, &mut self.probed, &self.x, &self.y);
        let store = &mut self.probed;
        if !first {
            let ok = rec.timed("insitu_probe", "prologue_probe", || {
                probe_step(&tape, store, &grads)
            });
            assert!(ok, "probe が momentum 一貫性検査の Err にならない");
        }
        let cfg = SgdConfig::new(LR);
        rec.timed("insitu_probe", "after_probe", || {
            tape.step_device_param_store(store, &grads, &cfg)
                .expect("step");
        });
    }
}

/// 実機用 record-only 診断（`orchestrate_h4.sh` が独立プロセスで起動）。
#[test]
#[ignore = "実機計測用の record-only 診断（reopen/RULE.txt・orchestrate_h4.sh 参照）"]
fn cpu_reuse_device_update_h4_phases() {
    let model = make_model();
    let (x, y) = make_data();
    let mut insitu = InSitu {
        direct: new_store(&model),
        probed: new_store(&model),
        model,
        x,
        y,
    };
    let (p0, g0) = {
        let tape = fandhe_ai::tape();
        let mut s = new_store(&insitu.model);
        let grads = fwd_bwd(&tape, &insitu.model, &mut s, &insitu.x, &insitu.y);
        (
            store_params(&s),
            flatten(&tape.param_grads_to_host(&s, &grads).expect("grads")),
        )
    };
    let mut standalone = Standalone::new(&p0, &g0);

    let mut rec = Rec::default();
    let total_iters = WARMUP + ROUNDS * ITERS_PER_ROUND;
    for it in 0..total_iters {
        if it == WARMUP {
            rec.on = true;
        }
        // ラウンドごとに in-situ 2 腕の順序を反転して順序効果を打ち消す。
        let round = it.saturating_sub(WARMUP) / ITERS_PER_ROUND;
        let rev = it >= WARMUP && round % 2 == 1;
        // 1 反復目は sgd_used を立てる本物の step のみ（probe を省く）。
        if rev {
            insitu.step_probed(&mut rec, it == 0);
            insitu.step_direct(&mut rec);
        } else {
            insitu.step_direct(&mut rec);
            insitu.step_probed(&mut rec, it == 0);
        }
        standalone.run_all(&mut rec);
    }

    let hd = bits_hash(&store_params(&insitu.direct));
    let hp = bits_hash(&store_params(&insitu.probed));
    assert_eq!(
        hd, hp,
        "insitu_direct と insitu_probe の最終 params が不一致"
    );
    rec.set_checksum("insitu_direct", "device_update", hd);
    rec.set_checksum("insitu_probe", "after_probe", hp);
    let finals = standalone.final_hashes();
    for (name, h) in finals {
        assert_eq!(
            h, finals[0].1,
            "standalone の {name} の最終 params が不一致"
        );
        rec.set_checksum("standalone", name, h);
    }

    let threads = std::env::var("RAYON_NUM_THREADS").ok();
    for ((arm, phase), cell) in &rec.cells {
        if cell.secs.is_empty() {
            continue;
        }
        let q = median_q1_q3(&cell.secs).expect("サンプルは非空・非 NaN");
        let mut sorted = cell.secs.clone();
        sorted.sort_by(f64::total_cmp);
        let line = serde_json::json!({
            "issue": 3188,
            "arm": arm,
            "phase": phase,
            "median_s": q.median,
            "q1_s": q.q1,
            "q3_s": q.q3,
            "min_s": sorted.first(),
            "max_s": sorted.last(),
            "n": cell.secs.len(),
            "checksum_bits": cell.checksum.map(|b| format!("{b:016x}")),
            "minflt_total": cell.proc_ok.then_some(cell.minflt),
            "core_moves": cell.proc_ok.then_some(cell.core_moves),
            "rayon_num_threads_env": threads,
            "target_arch": std::env::consts::ARCH,
            "target_os": std::env::consts::OS,
        });
        println!("{PREFIX}{line}");
    }
}
