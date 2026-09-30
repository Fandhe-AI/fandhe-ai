//! イシュー #2106（`docs/perf/cpu-reuse-device-update.md`）: CPU reuse 学習の
//! `device_update` フェーズ内訳切り分け用の診断。
//!
//! `docs/perf/train-step-phase-breakdown.md` §17.6.2 は、DGX Spark GB10 の
//! CPU reuse 学習で `device_update` が 277.5 µs（step_total の 25.7%）を
//! 占めると報告している（M4 Max は 122.3 µs〈§17.2〉。同じ算術をホスト Vec
//! 上で行う fresh 側 `host_sgd` の約 4 倍）。本ファイルはその内訳を、本番
//! コードへ計装を入れずに facade の公開 API だけで分解する。
//!
//! # 計測対象の経路（イシュー本文の `optim/sgd.rs` ではない点に注意）
//!
//! bench-fandhe `measure_train_reuse_phases` の `PHASE_DEVICE_UPDATE` 区間は
//! `Tape::step_device_param_store`（facade の薄い委譲）
//! → `DeviceParamStore::step`（`crates/autodiff/src/optim/device_store.rs`）
//! → `CpuBackendOps::sgd_step_device`（`crates/backend-cpu/src/ops.rs`）
//! であり、ホスト `Sgd::step` は通らない。`step` の内容は prologue（検査・
//! `Vec` 構築）／alloc（`flat_grad` の `with_capacity`）／stage（CPU は bias
//! 勾配が host 経由のため `upload_into`）／`sgd_step_device`（添字ループ 1 回。
//! compute と apply は融合）。
//!
//! # sub-phase とイシュー呼称の対応
//!
//! - alloc → `alloc`（と補助の `stage`）
//! - sgd_compute → `sgd_kernel`（融合）／`sgd_compute_split`（非融合の写し）
//! - apply_params → この経路では融合のため独立区間なし。`apply_params_split`
//!   を書き戻しトラフィックの参考値として持つ
//!
//! # 構成
//!
//! - テスト 1（CI 実行）: 写し（融合・非融合・zip 形）が本物の
//!   `step_device_param_store` と bit 一致することを hard assert する。
//!   一致しないと区間への帰属が信用できない。あわせて CPU で bias slot が
//!   host 経由になる前提と、pretouch が状態を変えないことを固定する。
//! - テスト 2（`#[ignore]`・実機用）: record-only。標準出力へ 1 行 1 区間の
//!   `DIAG_JSON` を出す。判定規則は
//!   `docs/perf/logs/cpu-reuse-device-update-2106/RULE.txt`（実測前に固定）で、
//!   `orchestrate.sh`／`aggregate.py` が独立 5 プロセスの出力を集計する。
//!
//! 本番コード・tolerance・baseline は変更しない。

use std::time::Instant;

use bench_harness::median_q1_q3;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{DeviceParamStore, SgdConfig, Tape, Tensor};
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::{BackendOps, DeviceBuffer, MemoryOps, SgdStepConfig};

const BATCH: usize = 64;
const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;
const LR: f32 = 0.01;
const WARMUP: usize = 20;
const ROUNDS: usize = 4;
const ITERS_PER_ROUND: usize = 20;
/// 標準出力の 1 行 1 区間 JSON の接頭辞（`orchestrate.sh` が抽出）。
const PREFIX: &str = "DIAG_JSON ";
/// `xthread` 補助腕で書き込み元 Tensor を生成するスレッド数（H2 の補助。判定には使わない）。
const XTHREADS: usize = 4;

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
    // FNV-1a（決定的・依存なし）。`f32::to_bits` 列を畳み込む。
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

/// forward → backward までを行い（未計時）、勾配と tape を返す。
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

/// 写し (a): 融合カーネル（本番 `sgd_step_device` を単独で呼ぶ）。
fn mirror_fused(ops: &CpuBackendOps, p: &[f32], g: &[f32]) -> Vec<f32> {
    let mem = ops.memory_ops().expect("MemoryOps");
    let mut pb = upload_flat(mem, p);
    let gb = upload_flat(mem, g);
    ops.sgd_step_device(&mut pb, &gb, None, &sgd_cfg())
        .expect("sgd");
    download_flat(mem, &pb)
}

/// 写し (b): 非融合。ホストで `p - lr*g` を計算してから params へ書き戻す。
fn mirror_split(ops: &CpuBackendOps, p: &[f32], g: &[f32]) -> Vec<f32> {
    let mem = ops.memory_ops().expect("MemoryOps");
    let mut pb = upload_flat(mem, p);
    let scratch: Vec<f32> = split_compute(p, g);
    let t = Tensor::new(scratch, &[p.len()]).expect("形状一致");
    mem.upload_into(&t, &mut pb, 0).expect("upload_into");
    download_flat(mem, &pb)
}

/// 写し (c): 境界検査なしの zip 形（H1 プローブ）。
fn mirror_zip(p: &[f32], g: &[f32]) -> Vec<f32> {
    let mut out = p.to_vec();
    zip_kernel(&mut out, g);
    out
}

/// 非融合の compute 区間。本番と同じ式・同じ演算順（mul → sub。FMA なし）。
fn split_compute(p: &[f32], g: &[f32]) -> Vec<f32> {
    p.iter().zip(g).map(|(p, g)| *p - LR * *g).collect()
}

fn zip_kernel(p: &mut [f32], g: &[f32]) {
    for (p, g) in p.iter_mut().zip(g) {
        *p -= LR * *g;
    }
}

/// 更新前 params・勾配を公開 API で再構成し、本物の step 結果と写しの
/// bit 一致を確認する。`pretouch` が真なら step 前に読み出しを挟む。
fn one_step_checked(
    ops: &CpuBackendOps,
    model: &Sequential,
    store: &mut DeviceParamStore,
    x: &Tensor<f32>,
    y: &Tensor<f32>,
    pretouch: bool,
) {
    let tape = fandhe_ai::tape();
    let grads = fwd_bwd(&tape, model, store, x, y);

    // CPU では weight slot が resident 経由（Some）・bias slot が host 経由
    // （None）。stage 区間が存在するというこの診断の前提を固定する。
    let resident = tape
        .resident_grads_to_host(store, &grads)
        .expect("resident");
    let some: Vec<bool> = resident.iter().map(Option::is_some).collect();
    assert_eq!(
        some,
        vec![true, false, true, false],
        "CPU の resident 勾配 slot 構成が前提（weight=Some・bias=None）と異なる。分解を見直すこと"
    );

    let p0 = store_params(store);
    let g = flatten(&tape.param_grads_to_host(store, &grads).expect("grads"));
    assert_eq!(p0.len(), g.len());
    let want_a = mirror_fused(ops, &p0, &g);
    let want_b = mirror_split(ops, &p0, &g);
    let want_c = mirror_zip(&p0, &g);

    if pretouch {
        let _ = tape.param_grads_to_host(store, &grads).expect("touch g");
        let _ = tape
            .sync_device_param_store_to_host(store)
            .expect("touch p");
    }
    tape.step_device_param_store(store, &grads, &SgdConfig::new(LR))
        .expect("step");
    let got = store_params(store);

    let h = |v: &[f32]| bits_hash(v);
    assert_eq!(
        h(&got),
        h(&want_a),
        "本物の step と融合カーネルの写しが bit 不一致"
    );
    assert_eq!(
        h(&got),
        h(&want_b),
        "本物の step と非融合の写しが bit 不一致"
    );
    assert_eq!(
        h(&got),
        h(&want_c),
        "本物の step と zip 形の写しが bit 不一致"
    );
    assert_eq!(
        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        want_a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "ハッシュ一致でも要素単位で不一致"
    );
}

#[test]
fn cpu_reuse_device_update_decomposition_matches_public_api_bit_exact() {
    let ops = CpuBackendOps::new();
    let model = make_model();
    let (x, y) = make_data();

    let mut direct = new_store(&model);
    let mut touched = new_store(&model);
    // warmup 1 + 検証 2 step。直接 step と pretouch 付き step の系列が
    // 同一状態になること（pretouch が状態を変えないこと）も確認する。
    for _ in 0..3 {
        one_step_checked(&ops, &model, &mut direct, &x, &y, false);
        one_step_checked(&ops, &model, &mut touched, &x, &y, true);
    }
    assert_eq!(
        bits_hash(&store_params(&direct)),
        bits_hash(&store_params(&touched)),
        "pretouch 有無で更新結果が異なる（区間の腕を比較できない）"
    );
}

// ---- テスト 2（実機用・record-only） ----

#[derive(Default)]
struct Cell {
    secs: Vec<f64>,
    checksum: Option<u64>,
}

#[derive(Default)]
struct Rec {
    cells: Vec<((&'static str, &'static str), Cell)>,
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

    fn timed<T>(&mut self, arm: &'static str, phase: &'static str, f: impl FnOnce() -> T) -> T {
        let t = Instant::now();
        let out = f();
        let dt = t.elapsed().as_secs_f64();
        if self.on {
            self.cell(arm, phase).secs.push(dt);
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
    // 融合腕 / zip 腕 / 非融合腕 / xthread 腕はそれぞれ独立の params を持つ
    // （同一入力から同一回数だけ更新し、最終 bit 一致で健全性を確認する）。
    p_fused: DeviceBuffer<f32>,
    p_zip: DeviceBuffer<f32>,
    p_split: DeviceBuffer<f32>,
    p_xthread: DeviceBuffer<f32>,
    /// `sgd_kernel_fixed` 腕（1 要素。`sgd_step_device` のループ外固定費の計測用）。
    p_fixed: DeviceBuffer<f32>,
    g_fixed: DeviceBuffer<f32>,
    /// 非融合腕のホスト鏡像（compute の入力）。
    host_p: Vec<f32>,
    g_buf: DeviceBuffer<f32>,
    g_buf_x: DeviceBuffer<f32>,
    /// stage 区間の宛先（total_numel 長）と bias 2 本。
    stage_dst: DeviceBuffer<f32>,
    b1: Tensor<f32>,
    b2: Tensor<f32>,
    stage_off1: usize,
    stage_off2: usize,
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
            p_fused: upload_flat(mem, p0),
            p_zip: upload_flat(mem, p0),
            p_split: upload_flat(mem, p0),
            p_xthread: upload_flat(mem, p0),
            p_fixed: upload_flat(mem, &[1.0]),
            g_fixed: upload_flat(mem, &[1.0]),
            host_p: p0.to_vec(),
            g_buf: upload_flat(mem, g),
            g_buf_x: upload_flat(mem, g),
            stage_dst: upload_flat(mem, &vec![0.0; total]),
            b1: Tensor::new(g[stage_off1..stage_off1 + D_HIDDEN].to_vec(), &[D_HIDDEN])
                .expect("形状一致"),
            b2: Tensor::new(g[stage_off2..stage_off2 + D_OUT].to_vec(), &[D_OUT])
                .expect("形状一致"),
            stage_off1,
            stage_off2,
            ops,
        }
    }

    fn mem(&self) -> &dyn MemoryOps {
        self.ops.memory_ops().expect("MemoryOps")
    }

    fn run_all(&mut self, rec: &mut Rec) {
        let total = self.total;
        // alloc（H3）: `flat_grad` 相当の未使用 capacity 確保と小 Vec 群に加え、
        // `DeviceParamStore::step` が bias 勾配を host 経由で stage する際の
        // `grad.clone()`（2 本）も含める（含めないと H4 残差へ混入し H3/H4 の帰属を誤らせる）。
        let (cb1, cb2) = (&self.b1, &self.b2);
        rec.timed("standalone", "alloc", || {
            // `&Tensor` への `.clone()` が参照コピーに解決されないよう `Tensor::clone` を明示する。
            let c1 = Tensor::clone(cb1);
            let c2 = Tensor::clone(cb2);
            std::hint::black_box((&c1, &c2));
            let flat: Vec<f32> = Vec::with_capacity(total);
            let vars: Vec<u64> = Vec::with_capacity(4);
            let filled: Vec<bool> = vec![false; 4];
            std::hint::black_box((&flat, &vars, &filled));
        });
        // stage: bias 2 本を total_numel 長バッファのオフセットへ書き込む。
        let (o1, o2) = (self.stage_off1, self.stage_off2);
        let (b1, b2) = (&self.b1, &self.b2);
        let mem = self.ops.memory_ops().expect("MemoryOps");
        let dst = &mut self.stage_dst;
        rec.timed("standalone", "stage", || {
            mem.upload_into(b1, dst, o1).expect("stage b1");
            mem.upload_into(b2, dst, o2).expect("stage b2");
        });
        // 融合カーネル（勾配は直前に同スレッドで書いた hot 状態）。
        let mem = self.ops.memory_ops().expect("MemoryOps");
        let gt = Tensor::new(self.g.clone(), &[total]).expect("形状一致");
        mem.upload_into(&gt, &mut self.g_buf, 0).expect("g write");
        let ops = &self.ops;
        let (pf, gb) = (&mut self.p_fused, &self.g_buf);
        rec.timed("standalone", "sgd_kernel", || {
            ops.sgd_step_device(pf, gb, None, &sgd_cfg()).expect("sgd");
        });
        // 固定費（H1 の分離用）: 同じ `sgd_step_device` を 1 要素で呼び、device・shape・
        // handle 検査と設定分岐などループ外のコストだけを計時する。
        let (pfx, gfx) = (&mut self.p_fixed, &self.g_fixed);
        rec.timed("standalone", "sgd_kernel_fixed", || {
            ops.sgd_step_device(pfx, gfx, None, &sgd_cfg())
                .expect("sgd");
        });
        // 非融合: compute → apply。
        let scratch = rec.timed("standalone", "sgd_compute_split", || {
            split_compute(&self.host_p, &self.g)
        });
        self.host_p.copy_from_slice(&scratch);
        let t = Tensor::new(scratch, &[total]).expect("形状一致");
        let mem = self.ops.memory_ops().expect("MemoryOps");
        let ps = &mut self.p_split;
        rec.timed("standalone", "apply_params_split", || {
            mem.upload_into(&t, ps, 0).expect("apply");
        });
        // zip 形（H1 プローブ）と xthread 補助腕は別メソッドで計時する。
        self.run_zip(rec);
        self.run_xthread(rec);
    }

    /// zip 形の本体（H1 プローブ）。`p_zip` の取り出し・書き戻しは区間外に置き、
    /// 境界検査なしの `iter_mut().zip()` ループだけを計時する。
    fn run_zip(&mut self, rec: &mut Rec) {
        let mem = self.ops.memory_ops().expect("MemoryOps");
        let mut host = download_flat(mem, &self.p_zip);
        let g = &self.g;
        rec.timed("standalone", "sgd_kernel_zip", || zip_kernel(&mut host, g));
        let t = Tensor::new(host, &[self.total]).expect("形状一致");
        mem.upload_into(&t, &mut self.p_zip, 0).expect("zip 反映");
    }

    /// 勾配の元データを別スレッドで生成（Tensor 化）してから main が `upload_into` した直後の
    /// 融合カーネル（H2 の補助・判定に使わない）。
    /// 注意: `DeviceBuffer` は `Send` でないため対象バッファへの書き込み自体は main
    /// スレッドで行う。ワーカーが書くのは書き込み元のホスト Tensor だけであり、
    /// 「別スレッドが勾配バッファを書いた直後の cache 状態」は再現していない
    /// （その再現は insitu_direct/insitu_pretouch の差で見る）。
    fn run_xthread(&mut self, rec: &mut Rec) {
        let mem = self.ops.memory_ops().expect("MemoryOps");
        // 書き込み元 Tensor は各スレッドが別チャンクで生成する（区間外）。
        let chunk = self.total.div_ceil(XTHREADS);
        let g = &self.g;
        let parts: Vec<Tensor<f32>> = std::thread::scope(|s| {
            let hs: Vec<_> = g
                .chunks(chunk)
                .map(|c| s.spawn(move || Tensor::new(c.to_vec(), &[c.len()]).expect("形状一致")))
                .collect();
            hs.into_iter().map(|h| h.join().expect("join")).collect()
        });
        for (i, t) in parts.iter().enumerate() {
            mem.upload_into(t, &mut self.g_buf_x, i * chunk)
                .expect("xthread write");
        }
        let ops = &self.ops;
        let (px, gb) = (&mut self.p_xthread, &self.g_buf_x);
        rec.timed("standalone", "sgd_kernel_xthread", || {
            ops.sgd_step_device(px, gb, None, &sgd_cfg()).expect("sgd");
        });
    }

    /// 最終状態の bit ハッシュ（全腕が同一回数だけ同一更新を受けた後の健全性検査）。
    fn final_hashes(&self) -> [(&'static str, u64); 4] {
        let mem = self.mem();
        [
            ("sgd_kernel", bits_hash(&download_flat(mem, &self.p_fused))),
            (
                "sgd_kernel_zip",
                bits_hash(&download_flat(mem, &self.p_zip)),
            ),
            (
                "apply_params_split",
                bits_hash(&download_flat(mem, &self.p_split)),
            ),
            (
                "sgd_kernel_xthread",
                bits_hash(&download_flat(mem, &self.p_xthread)),
            ),
        ]
    }
}

/// in-situ 腕: 本物の学習ループ内で `step_device_param_store` を計時する。
/// 2 腕は別 store（同一初期値・同一データ）を持ち、反復ごとに交互に 1 step
/// 進める。pretouch は読み出しのみで状態を変えないため最終 bit は一致する。
struct InSitu {
    model: Sequential,
    direct: DeviceParamStore,
    touched: DeviceParamStore,
    x: Tensor<f32>,
    y: Tensor<f32>,
}

impl InSitu {
    fn step(&mut self, arm: &'static str, rec: &mut Rec) {
        let (store, pretouch) = if arm == "insitu_direct" {
            (&mut self.direct, false)
        } else {
            (&mut self.touched, true)
        };
        let tape = fandhe_ai::tape();
        let grads = fwd_bwd(&tape, &self.model, store, &self.x, &self.y);
        if pretouch {
            // 未計時: staging と params を main スレッドで読み、cache 状態を揃える。
            let _ = tape.param_grads_to_host(store, &grads).expect("touch g");
            let _ = tape
                .sync_device_param_store_to_host(store)
                .expect("touch p");
        }
        let cfg = SgdConfig::new(LR);
        rec.timed(arm, "device_update", || {
            tape.step_device_param_store(store, &grads, &cfg)
                .expect("step");
        });
    }
}

/// 実機用 record-only 診断（`orchestrate.sh` が独立プロセスで 5 回起動）。
#[test]
#[ignore = "実機計測用の record-only 診断（RULE.txt・orchestrate.sh 参照）"]
fn cpu_reuse_device_update_phases() {
    let model = make_model();
    let (x, y) = make_data();
    let mut insitu = InSitu {
        direct: new_store(&model),
        touched: new_store(&model),
        model,
        x,
        y,
    };

    // standalone の fixture は、本物の store の初期 params と 1 step 目の勾配。
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
    let insitu_arms = ["insitu_direct", "insitu_pretouch"];
    let total_iters = WARMUP + ROUNDS * ITERS_PER_ROUND;
    for it in 0..total_iters {
        if it == WARMUP {
            rec.on = true;
        }
        // ラウンドごとに腕の順序を反転して順序効果を打ち消す。
        let round = it.saturating_sub(WARMUP) / ITERS_PER_ROUND;
        let rev = it >= WARMUP && round % 2 == 1;
        let order: Vec<&'static str> = if rev {
            insitu_arms.iter().rev().copied().collect()
        } else {
            insitu_arms.to_vec()
        };
        for arm in order {
            insitu.step(arm, &mut rec);
        }
        standalone.run_all(&mut rec);
    }

    // checksum（hard・fail-closed）。in-situ の 2 腕と standalone の 4 腕は、
    // それぞれ内部で bit 一致していなければ系列を無効とする。
    let hd = bits_hash(&store_params(&insitu.direct));
    let ht = bits_hash(&store_params(&insitu.touched));
    assert_eq!(
        hd, ht,
        "insitu_direct と insitu_pretouch の最終 params が不一致"
    );
    rec.set_checksum("insitu_direct", "device_update", hd);
    rec.set_checksum("insitu_pretouch", "device_update", ht);
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
            "issue": 2106,
            "arm": arm,
            "phase": phase,
            "median_s": q.median,
            "q1_s": q.q1,
            "q3_s": q.q3,
            "min_s": sorted.first(),
            "max_s": sorted.last(),
            "n": cell.secs.len(),
            "checksum_bits": cell.checksum.map(|b| format!("{b:016x}")),
            "rayon_num_threads_env": threads,
            "target_arch": std::env::consts::ARCH,
            "target_os": std::env::consts::OS,
        });
        println!("{PREFIX}{line}");
    }
}
