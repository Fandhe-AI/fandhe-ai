//! イシュー #2116（`docs/perf/cuda-train-forward-resident-diagnosis.md`）:
//! CUDA train reuse の `forward_resident` 区間と、fresh の `param_readout`
//! 区間の CPU 側オーバーヘッドを、facade の公開 API と autodiff の内部 API
//! だけで分解する診断（上位層）。
//!
//! `docs/perf/train-step-phase-breakdown.md` §17.6（#1980・GB10）は、cuda reuse
//! の `forward_resident` が 155.9 µs（step_total の 49.1%）で最大の区間になり、
//! cuda fresh の `param_readout` が同一ホスト cpu fresh の約 2.2 倍あると報告して
//! いる。本ファイルは本番コードへ計装を入れずにその内訳を取る。層ごとの
//! H2D／確保／起動／同期 D2H の下位内訳は
//! `crates/backend-cuda/src/train_forward_resident_diag_tests.rs`（下位層）が担当する。
//!
//! # 分解の 2 層とサブフェーズ名の対応
//!
//! | イシュー上の名称 | 実コード上の区間 |
//! |---|---|
//! | param_apply | `register_resident_params`（resident leaf の tape 登録。H2D なし） |
//! | linear_forward_device（学習経路） | `linear_forward_with_activation` → `gemm_resident_rhs_act`（CUDA は既定合成: `gemm_resident_rhs` + ホスト往復 `relu`） |
//! | alloc | 下位層の `mem_new`／`alloc_c`。本ファイルの `paired/residual` は tape ノード push 等の残差 |
//!
//! `public` arm は bench-fandhe `measure_train_reuse_phases` と同じ呼び出し列
//! （`Sequential::forward_resident` + `mse_loss`）。`decomposed` arm は同じ 1 step を
//! `register`／`l1_linear_relu`／`l2_linear`／`mse_loss` に分けて計時する。ガードと
//! `forward_from_flat_leaves` の走査コストは反復単位の残差（`public` − Σ`decomposed`）
//! として `paired/residual` に出す（同一プロセス・同一反復内で両 arm を実行し、
//! ラウンドごとに順序を反転する）。
//!
//! # fresh の `param_readout`（R0〜R3）
//!
//! `Tensor` はホスト storage のみを持つため、この区間に D2H・デバイス同期は
//! 構造上ない（grad は backward 内で実体化済み）。診断対象は host alloc・page fault・
//! 非 contiguous の gather・memcpy に限る。
//! R0 = bench ブロックの逐語写し、R1 = tensor ごとの `contiguous()`／`to_vec()` 分解、
//! R2 = 事前タッチ済み再利用 `Vec` への `copy_from_slice`、R3 = CPU backend の対照。
//!
//! # 構成
//!
//! - テスト A（CI 実行・CPU）: 写しが本物と bit 一致すること（`public`／`decomposed` の
//!   pred・loss・最終 params、R0／R1／R2 のバイト列）を hard assert する。
//! - テスト B（`#[ignore]`・CUDA 実機）: 同じ bit 一致を CUDA で hard assert する。
//! - テスト C（`#[ignore]`・CUDA 実機）: record-only。`DIAG_JSON` を 1 行 1 セルで出す。
//!   判定規則は `docs/perf/logs/cuda-train-forward-resident-2116/RULE.txt`
//!   （実測前に固定）で、`orchestrate.sh`／`aggregate.py` が独立 5 プロセスを集計する。
//!
//! 実行は必ず `--test-threads=1`。本番コード・tolerance・baseline は変更しない。

use std::time::Instant;

use bench_harness::median_q1_q3;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, DeviceParamStore, SgdConfig, Tensor};
use fandhe_ai_autodiff::Tape as InnerTape;
use fandhe_ai_tensor_core::Activation;

const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;
const LR: f32 = 0.01;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;
const SEED_X: u64 = 0xDA7A_0001;
const SEED_Y: u64 = 0xDA7A_0002;
/// bench 既定の batch。fresh の readout 診断はこの batch だけで行う。
const BENCH_BATCH: usize = 64;
const BATCHES: [usize; 4] = [16, 64, 256, 1024];
const WARMUP: usize = 20;
const ROUNDS: usize = 4;
const ITERS_PER_ROUND: usize = 20;
const PREFIX: &str = "DIAG_JSON ";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Cpu,
    Cuda,
}

impl Kind {
    /// 出力レコードへ記録する backend 識別子。aggregate.py が `cuda` 以外の混入を fail-closed で拒否する。
    fn label(self) -> &'static str {
        match self {
            Kind::Cpu => "cpu",
            Kind::Cuda => "cuda",
        }
    }

    fn device(self) -> Device {
        match self {
            Kind::Cpu => Device::Cpu,
            Kind::Cuda => Device::Cuda(0),
        }
    }

    /// facade 経由の tape（`Sequential` の公開 API に渡す）。
    fn facade_tape(self) -> fandhe_ai::Tape {
        fandhe_ai::tape_for(self.device()).expect("tape_for")
    }

    /// facade の resolve_ops と同じ構成の autodiff tape（`decomposed` arm 用）。
    fn inner_tape(self) -> InnerTape {
        match self {
            Kind::Cpu => {
                InnerTape::new_with_ops(Box::new(fandhe_ai_backend_cpu::CpuBackendOps::new()))
            }
            Kind::Cuda => {
                InnerTape::new_with_ops(Box::new(fandhe_ai_backend_cuda::CudaBackendOps::new(0)))
            }
        }
    }
}

fn make_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .expect("Linear 構築")
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .expect("Linear 構築")
}

fn make_data(batch: usize) -> (Tensor<f32>, Tensor<f32>) {
    let x = Xorshift64Star::new(SEED_X).fill_vec(batch * D_IN);
    let y = Xorshift64Star::new(SEED_Y).fill_vec(batch * D_OUT);
    (
        Tensor::new(x, &[batch, D_IN]).expect("形状一致"),
        Tensor::new(y, &[batch, D_OUT]).expect("形状一致"),
    )
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

fn flatten(ts: &[Tensor<f32>]) -> Vec<f32> {
    let mut v = Vec::new();
    for t in ts {
        v.extend_from_slice(t.contiguous().as_slice().expect("contiguous"));
    }
    v
}

fn tensor_bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn new_store(kind: Kind, model: &Sequential) -> DeviceParamStore {
    let tape = kind.facade_tape();
    let store = model.init_device_param_store(&tape).expect("store");
    // bench-fandhe と同じく init 直後に 1 回 sync してから tape を捨てる。
    let _ = tape.sync_device_param_store_to_host(&store).expect("sync");
    store
}

fn store_params(kind: Kind, store: &DeviceParamStore) -> Vec<f32> {
    let tape = kind.facade_tape();
    flatten(&tape.sync_device_param_store_to_host(store).expect("sync"))
}

// ---- 時間記録 ----

#[derive(Default)]
struct Cell {
    secs: Vec<f64>,
}

#[derive(Default)]
struct Rec {
    cells: Vec<((&'static str, &'static str, usize), Cell)>,
    checksums: Vec<((&'static str, usize), u64)>,
    on: bool,
}

impl Rec {
    fn push(&mut self, arm: &'static str, phase: &'static str, batch: usize, secs: f64) {
        if !self.on {
            return;
        }
        let key = (arm, phase, batch);
        let i = match self.cells.iter().position(|(k, _)| *k == key) {
            Some(i) => i,
            None => {
                self.cells.push((key, Cell::default()));
                self.cells.len() - 1
            }
        };
        self.cells[i].1.secs.push(secs);
    }

    fn timed<T>(
        &mut self,
        arm: &'static str,
        phase: &'static str,
        batch: usize,
        f: impl FnOnce() -> T,
    ) -> (T, f64) {
        let t = Instant::now();
        let out = f();
        let dt = t.elapsed().as_secs_f64();
        self.push(arm, phase, batch, dt);
        (out, dt)
    }
}

// ---- forward_resident の 2 arm ----

/// 1 step 分の結果（pred は forward の出力。loss は最終 step の値の bit）。
struct StepOut {
    pred_bits: Vec<u32>,
    loss_bits: u32,
    forward_secs: f64,
    parts_secs: f64,
}

/// `public` arm: bench-fandhe と同じ呼び出し列で 1 step（forward → backward → update）。
fn public_step(
    kind: Kind,
    model: &Sequential,
    store: &mut DeviceParamStore,
    x: &Tensor<f32>,
    y: &Tensor<f32>,
    rec: &mut Rec,
    batch: usize,
) -> StepOut {
    let tape = kind.facade_tape();
    let xv = tape.var(x);
    let yv = tape.var(y);
    let (pred_loss, forward_secs) = rec.timed("public", "forward_resident", batch, || {
        let pred = model.forward_resident(&tape, &xv, store).expect("fwd");
        let loss = pred.mse_loss(&yv).expect("mse");
        (pred, loss)
    });
    let (pred, loss) = pred_loss;
    let loss_bits = loss.to_tensor().get(&[]).expect("scalar").to_bits();
    let pred_bits = tensor_bits(&pred.to_tensor());
    let grads = tape.backward_device_param_store(&loss, store).expect("bwd");
    tape.step_device_param_store(store, &grads, &SgdConfig::new(LR))
        .expect("step");
    StepOut {
        pred_bits,
        loss_bits,
        forward_secs,
        parts_secs: 0.0,
    }
}

/// `decomposed` arm: 同じ 1 step を register／l1／l2／mse に分けて計時する。
fn decomposed_step(
    kind: Kind,
    store: &mut DeviceParamStore,
    x: &Tensor<f32>,
    y: &Tensor<f32>,
    rec: &mut Rec,
    batch: usize,
) -> StepOut {
    let tape = kind.inner_tape();
    let xv = tape.var(x);
    let yv = tape.var(y);
    let (leaves, t_reg) = rec.timed("decomposed", "register", batch, || {
        store.register_resident_params(&tape).expect("register")
    });
    let (h, t_l1) = rec.timed("decomposed", "l1_linear_relu", batch, || {
        store
            .linear_forward_with_activation(
                &tape,
                &xv,
                &leaves[0],
                Some(&leaves[1]),
                Activation::Relu,
            )
            .expect("l1")
    });
    let (pred, t_l2) = rec.timed("decomposed", "l2_linear", batch, || {
        store
            .linear_forward_with_activation(
                &tape,
                &h,
                &leaves[2],
                Some(&leaves[3]),
                Activation::None,
            )
            .expect("l2")
    });
    let (loss, t_mse) = rec.timed("decomposed", "mse_loss", batch, || {
        pred.mse_loss(&yv).expect("mse")
    });
    let loss_bits = loss.to_tensor().get(&[]).expect("scalar").to_bits();
    let pred_bits = tensor_bits(&pred.to_tensor());
    let grads = store.backward(&tape, &loss).expect("bwd");
    store
        .step(&tape, &grads, &SgdConfig::new(LR))
        .expect("step");
    StepOut {
        pred_bits,
        loss_bits,
        forward_secs: 0.0,
        parts_secs: t_reg + t_l1 + t_l2 + t_mse,
    }
}

/// 1 batch 分の系列（warmup + 計測）。両 arm を同一反復内で実行し、ラウンドごとに
/// 順序を反転する。pred・loss の bit と最終 params の一致を hard assert する。
fn run_forward_series(
    kind: Kind,
    batch: usize,
    warmup: usize,
    rounds: usize,
    iters: usize,
    rec: &mut Rec,
) {
    let model = make_model();
    let (x, y) = make_data(batch);
    let mut s_pub = new_store(kind, &model);
    let mut s_dec = new_store(kind, &model);
    let total = warmup + rounds * iters;
    let mut last = (0u32, 0u32);
    for it in 0..total {
        rec.on = it >= warmup;
        let round = it.saturating_sub(warmup) / iters.max(1);
        let dec_first = rec.on && round % 2 == 1;
        let (p, d);
        if dec_first {
            d = decomposed_step(kind, &mut s_dec, &x, &y, rec, batch);
            p = public_step(kind, &model, &mut s_pub, &x, &y, rec, batch);
        } else {
            p = public_step(kind, &model, &mut s_pub, &x, &y, rec, batch);
            d = decomposed_step(kind, &mut s_dec, &x, &y, rec, batch);
        }
        assert_eq!(
            p.pred_bits, d.pred_bits,
            "public と decomposed の pred が bit 不一致 (batch={batch}, it={it})"
        );
        assert_eq!(
            p.loss_bits, d.loss_bits,
            "public と decomposed の loss が bit 不一致 (batch={batch}, it={it})"
        );
        // 反復単位の残差（差を取ってから中央値。§17.3・§17.6.3 が未実施と記録した集計）。
        rec.push("paired", "residual", batch, p.forward_secs - d.parts_secs);
        last = (p.loss_bits, d.loss_bits);
    }
    let hp = bits_hash(&store_params(kind, &s_pub));
    let hd = bits_hash(&store_params(kind, &s_dec));
    assert_eq!(
        hp, hd,
        "public と decomposed の最終 params が bit 不一致 (batch={batch})"
    );
    rec.checksums
        .push((("public", batch), hp ^ u64::from(last.0)));
    rec.checksums
        .push((("decomposed", batch), hd ^ u64::from(last.1)));
}

// ---- fresh の param_readout ----

/// readout の腕（R0〜R3）。
#[derive(Clone, Copy, PartialEq, Eq)]
enum Readout {
    Mirror,
    Split,
    Pretouched,
}

/// 事前タッチ済みの再利用バッファ（R2）。
#[derive(Default)]
struct Reuse {
    params: Vec<Vec<f32>>,
    grads: Vec<Vec<f32>>,
}

/// fresh の 1 step。readout 区間だけを `arm`/`param_readout` として計時し、
/// bench と同じ host_sgd・apply_parameters で params を進める。戻り値は最終 loss の bit。
fn fresh_step(
    (kind, readout, arm): (Kind, Readout, &'static str),
    model: &mut Sequential,
    (x, y): (&Tensor<f32>, &Tensor<f32>),
    reuse: &mut Reuse,
    rec: &mut Rec,
    batch: usize,
) -> u32 {
    let tape = kind.facade_tape();
    let bound = model.bind(&tape);
    let xv = tape.var(x);
    let yv = tape.var(y);
    let pred = bound.forward(&tape, &xv).expect("fwd");
    let loss = pred.mse_loss(&yv).expect("mse");
    let loss_bits = loss.to_tensor().get(&[]).expect("scalar").to_bits();
    let grads = tape.backward(&loss).expect("bwd");
    let grad_refs = bound.trainable_grads(&grads).expect("grads");
    let param_refs = model.trainable_parameters();

    let mut host_params: Vec<Vec<f32>> = Vec::new();
    let mut host_grads: Vec<Vec<f32>> = Vec::new();
    let mut shapes = Vec::new();
    match readout {
        Readout::Mirror => {
            let t0 = Instant::now();
            let mut hp = Vec::with_capacity(param_refs.len());
            let mut hg = Vec::with_capacity(param_refs.len());
            for (param, grad) in param_refs.iter().zip(grad_refs.iter()) {
                hp.push(param.contiguous().as_slice().expect("param").to_vec());
                hg.push(grad.contiguous().as_slice().expect("grad").to_vec());
                shapes.push(param.shape().to_vec());
            }
            rec.push(arm, "param_readout", batch, t0.elapsed().as_secs_f64());
            host_params = hp;
            host_grads = hg;
        }
        Readout::Split => {
            let (mut p_c, mut p_v, mut g_c, mut g_v) = (0.0, 0.0, 0.0, 0.0);
            let mut noncontig_grads = 0usize;
            let t0 = Instant::now();
            for (param, grad) in param_refs.iter().zip(grad_refs.iter()) {
                let t = Instant::now();
                let pc = param.contiguous();
                p_c += t.elapsed().as_secs_f64();
                let t = Instant::now();
                host_params.push(pc.as_slice().expect("param").to_vec());
                p_v += t.elapsed().as_secs_f64();
                if !grad.is_contiguous() {
                    noncontig_grads += 1;
                }
                let t = Instant::now();
                let gc = grad.contiguous();
                g_c += t.elapsed().as_secs_f64();
                let t = Instant::now();
                host_grads.push(gc.as_slice().expect("grad").to_vec());
                g_v += t.elapsed().as_secs_f64();
                shapes.push(param.shape().to_vec());
            }
            rec.push(arm, "param_readout", batch, t0.elapsed().as_secs_f64());
            rec.push(arm, "param_contiguous", batch, p_c);
            rec.push(arm, "param_to_vec", batch, p_v);
            rec.push(arm, "grad_contiguous", batch, g_c);
            rec.push(arm, "grad_to_vec", batch, g_v);
            rec.push(arm, "noncontig_grad_count", batch, noncontig_grads as f64);
        }
        Readout::Pretouched => {
            if reuse.params.is_empty() {
                for (param, grad) in param_refs.iter().zip(grad_refs.iter()) {
                    // 確保 + 全ページのタッチ（0 以外を書いて zero page 共有を避ける）。
                    reuse
                        .params
                        .push(vec![1.0; param.contiguous().as_slice().expect("p").len()]);
                    reuse
                        .grads
                        .push(vec![1.0; grad.contiguous().as_slice().expect("g").len()]);
                }
            }
            let t0 = Instant::now();
            for (i, (param, grad)) in param_refs.iter().zip(grad_refs.iter()).enumerate() {
                reuse.params[i].copy_from_slice(param.contiguous().as_slice().expect("param"));
                reuse.grads[i].copy_from_slice(grad.contiguous().as_slice().expect("grad"));
                shapes.push(param.shape().to_vec());
            }
            rec.push(arm, "param_readout", batch, t0.elapsed().as_secs_f64());
            // 再利用バッファは複製せず、そのまま更新処理へ渡す（複製の確保費用が
            // R0 との比較に混入しない計測区間外の差にならないよう、複製自体を行わない）。
        }
    }
    drop(param_refs);
    drop(grad_refs);
    drop(bound);
    let (upd_params, upd_grads): (&[Vec<f32>], &[Vec<f32>]) = if readout == Readout::Pretouched {
        (&reuse.params, &reuse.grads)
    } else {
        (&host_params, &host_grads)
    };
    let mut next = Vec::with_capacity(upd_params.len());
    for ((p, g), shape) in upd_params.iter().zip(upd_grads.iter()).zip(shapes.iter()) {
        let upd: Vec<f32> = p.iter().zip(g.iter()).map(|(p, g)| p - LR * g).collect();
        next.push(Tensor::from_slice(&upd, shape).expect("形状一致"));
    }
    model.apply_parameters(next).expect("apply");
    loss_bits
}

fn model_hash(model: &Sequential) -> u64 {
    let mut all = Vec::new();
    for t in model.trainable_parameters() {
        all.extend_from_slice(t.contiguous().as_slice().expect("contiguous"));
    }
    bits_hash(&all)
}

/// R0〜R3 の系列。R0／R1／R2 は同一 backend で同一更新を受けるので最終 params が
/// bit 一致する（hard assert）。R3 は CPU 対照のため別 checksum で記録する。
fn run_readout_series(kind: Kind, warmup: usize, rounds: usize, iters: usize, rec: &mut Rec) {
    let (x, y) = make_data(BENCH_BATCH);
    let arms: [(&'static str, Readout, Kind); 4] = [
        ("readout_r0_mirror", Readout::Mirror, kind),
        ("readout_r1_split", Readout::Split, kind),
        ("readout_r2_pretouched", Readout::Pretouched, kind),
        ("readout_r3_cpu_control", Readout::Mirror, Kind::Cpu),
    ];
    let mut models: Vec<Sequential> = arms.iter().map(|_| make_model()).collect();
    let mut reuse = Reuse::default();
    let mut last = [0u32; 4];
    let total = warmup + rounds * iters;
    for it in 0..total {
        rec.on = it >= warmup;
        let round = it.saturating_sub(warmup) / iters.max(1);
        let mut order: Vec<usize> = (0..arms.len()).collect();
        if rec.on && round % 2 == 1 {
            order.reverse();
        }
        for i in order {
            let (name, ro, k) = arms[i];
            last[i] = fresh_step(
                (k, ro, name),
                &mut models[i],
                (&x, &y),
                &mut reuse,
                rec,
                BENCH_BATCH,
            );
        }
    }
    let hs: Vec<u64> = models.iter().map(model_hash).collect();
    assert_eq!(hs[0], hs[1], "R0 と R1 の最終 params が bit 不一致");
    assert_eq!(hs[0], hs[2], "R0 と R2 の最終 params が bit 不一致");
    assert_eq!(last[0], last[1], "R0 と R1 の最終 loss が bit 不一致");
    assert_eq!(last[0], last[2], "R0 と R2 の最終 loss が bit 不一致");
    for (i, (name, _, _)) in arms.iter().enumerate() {
        rec.checksums
            .push(((*name, BENCH_BATCH), hs[i] ^ u64::from(last[i])));
    }
}

// ---- テスト ----

#[test]
fn cuda_train_forward_resident_decomposition_bit_identical_cpu() {
    // CI 実行。CPU は `gemm_resident_rhs_act` を融合 override しているため CUDA 固有の
    // 既定合成の bit 一致は検証できない（下位層と `#[ignore]` テスト B が担当）。
    let mut rec = Rec::default();
    run_forward_series(Kind::Cpu, 16, 1, 1, 2, &mut rec);
    run_readout_series(Kind::Cpu, 1, 1, 2, &mut rec);
}

#[test]
#[ignore = "CUDA 実機必須"]
fn cuda_train_forward_resident_decomposition_bit_identical_cuda() {
    let mut rec = Rec::default();
    run_forward_series(Kind::Cuda, 16, 1, 1, 2, &mut rec);
    run_readout_series(Kind::Cuda, 1, 1, 2, &mut rec);
}

#[test]
#[ignore = "CUDA 実機計測用の record-only 診断（RULE.txt・orchestrate.sh 参照）"]
fn cuda_train_forward_resident_phases() {
    // 既定は CUDA。`TRAIN_FWD_DIAG_KIND=cpu` は出力形式とパイプラインの動作確認用
    // （判定外。RULE.txt の対象は CUDA 実測のみ）。
    let kind = match std::env::var("TRAIN_FWD_DIAG_KIND").as_deref() {
        Ok("cpu") => Kind::Cpu,
        _ => Kind::Cuda,
    };
    let mut rec = Rec::default();
    for batch in BATCHES {
        run_forward_series(kind, batch, WARMUP, ROUNDS, ITERS_PER_ROUND, &mut rec);
    }
    run_readout_series(kind, WARMUP, ROUNDS, ITERS_PER_ROUND, &mut rec);

    for ((arm, phase, batch), cell) in &rec.cells {
        if cell.secs.is_empty() {
            continue;
        }
        let q = median_q1_q3(&cell.secs).expect("サンプルは非空・非 NaN");
        let mut sorted = cell.secs.clone();
        sorted.sort_by(f64::total_cmp);
        let line = serde_json::json!({
            "issue": 2116,
            "layer": "facade",
            "backend": kind.label(),
            "record": "cell",
            "arm": arm,
            "phase": phase,
            "batch": batch,
            "median_s": q.median,
            "q1_s": q.q1,
            "q3_s": q.q3,
            "min_s": sorted.first(),
            "max_s": sorted.last(),
            "n": cell.secs.len(),
        });
        println!("{PREFIX}{line}");
    }
    for ((arm, batch), bits) in &rec.checksums {
        let line = serde_json::json!({
            "issue": 2116,
            "layer": "facade",
            "backend": kind.label(),
            "record": "checksum",
            "arm": arm,
            "batch": batch,
            "bits": format!("{bits:016x}"),
        });
        println!("{PREFIX}{line}");
    }
}
