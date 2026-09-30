//! CUDA train reuse の `forward_resident` 区間（`scripts/bench/framework-compare`
//! の `bench-fandhe --task train --mode reuse --phases`）の内側を、層の種類別・
//! batch 別に分解する診断テスト（イシュー #2116。下位層）。
//!
//! # 背景
//!
//! `docs/perf/train-step-phase-breakdown.md` §17.6（#1980・DGX Spark GB10）で
//! cuda reuse の `forward_resident` が 155.9 µs（step_total の 49.1%）と最大の
//! 区間になった。学習 forward は `Sequential::forward_resident` →
//! `DeviceParamStore::linear_forward_with_activation` →
//! `BackendOps::gemm_resident_rhs_act` を通る。CUDA は `gemm_resident_rhs_act` を
//! override していないため trait 既定の 2 段合成（`gemm_resident_rhs` + ホスト
//! 往復の `relu`）になり、層ごと・活性化ごと・損失ごとに H2D／確保／起動／
//! 同期 D2H のホスト往復が発生する（`docs/perf/train-linear-epilogue-fusion.md`
//! §4）。本ファイルはその往復を区間に分けて実測する。上位 API の固定費
//! （register・tape ノード push・ガード）は上位層
//! `crates/facade/tests/cuda_train_forward_resident_diag.rs` が担当する。
//!
//! # 配置理由（`gemm_reuse_phase_diag_tests.rs` と同じ判断）
//!
//! `context_cache::{cached_device, cached_gemm}`・`CudaGemm::
//! launch_tiled_bias_act_f32_resident`・`CudaBufferHandle`（いずれも非公開）へ
//! 到達する必要があるため、integration test ではなく `lib.rs` の兄弟
//! モジュールとして配置する。本番コードは変更しない。
//!
//! # arm 構成
//!
//! - `prod`: 本番経路そのもの（`CudaBackendOps::gemm_resident_rhs` /
//!   `relu` / `mse_loss` を 1 回呼ぶ）。区間和の fidelity 評価の基準。
//! - `nosync`: `gemm_resident_rhs` を本番と同じスケジュール（同期は最後の
//!   D2H のみ）で区間に分けて計時する。区間和と `prod` の一致は本 arm だけで
//!   評価する。
//! - `syncsplit`: `launch_issue` の直後に明示 `synchronize` を挿入して
//!   `kernel_wait` を別区間に分ける。本番にない同期を足すため区間和が `prod`
//!   を上回りうる。fidelity 評価には使わない（H4 の値だけに使う）。
//! - `whatif`（record-only）: L1 相当を `act_relu = true` の融合起動 1 回で
//!   行う参照値。「GEMM 往復 + relu 往復」との比で削減規模を見積もる。
//!
//! # 判定と出力
//!
//! 判定規則は `docs/perf/logs/cuda-train-forward-resident-2116/RULE.txt`
//! （実測前に固定）が正。`#[ignore]` テストは 1 行 1 セルの `DIAG_JSON` を出力
//! し、`orchestrate.sh`／`aggregate.py` が独立 5 プロセスの出力を集計する。
//! 分解した L1／L2 出力が本番 `gemm_resident_rhs` の出力と bit 一致すること
//! （同一カーネル・同一入力）だけを hard assert する。時間の大小へは assert
//! しない。実行は必ず `--test-threads=1`（同一 GPU 上のスレッド競合回避）。
//!
//! # 出力ハッシュ
//!
//! `checksum_bits` は本番出力の FNV-1a（`f32::to_bits` 列）で、5 run 間の
//! 一致を RULE.txt の checksum 規則で要求する。

use std::time::Instant;

use bench_harness::{median_q1_q3, rng::Xorshift64Star};
use fandhe_ai_tensor_core::buffer::{DeviceBufferView, MemoryOps};
use fandhe_ai_tensor_core::{BackendOps, MseReduction, Tensor};

use crate::context_cache::{cached_device, cached_gemm};
use crate::memory::{CudaBufferHandle, CudaMemory};
use crate::ops::CudaBackendOps;

const WARMUP: usize = 20;
const ROUNDS: usize = 4;
const ITERS_PER_ROUND: usize = 20;
/// 上位層と同じ MLP 形状（784→256→10）。
const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;
/// 計測する batch（64 が bench 既定）。
const BATCHES: [usize; 4] = [16, 64, 256, 1024];
/// 標準出力の 1 行 1 セル JSON の接頭辞（`orchestrate.sh` が抽出）。
const PREFIX: &str = "DIAG_JSON ";

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

fn tensor(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    Tensor::new(Xorshift64Star::new(seed).fill_vec(n), shape).expect("形状一致")
}

/// (arm, layer, batch, phase)。
type CellKey = (String, String, usize, String);

/// (arm, layer, batch, phase) ごとの時間標本。
#[derive(Default)]
struct Rec {
    cells: Vec<(CellKey, Vec<f64>)>,
    checksums: Vec<((String, String, usize), u64)>,
}

impl Rec {
    fn push(&mut self, arm: &str, layer: &str, batch: usize, phase: &str, secs: f64) {
        let key = (arm.to_string(), layer.to_string(), batch, phase.to_string());
        match self.cells.iter().position(|(k, _)| *k == key) {
            Some(i) => self.cells[i].1.push(secs),
            None => self.cells.push((key, vec![secs])),
        }
    }
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    let out = f();
    (out, t.elapsed().as_secs_f64())
}

/// 常駐 w・bias を持つ 1 層分の fixture。
struct Layer {
    name: &'static str,
    k: usize,
    n: usize,
    w: fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
    b: fandhe_ai_tensor_core::buffer::DeviceBuffer<f32>,
}

impl Layer {
    fn new(ops: &CudaBackendOps, name: &'static str, k: usize, n: usize, seed: u64) -> Self {
        let mem = ops.memory_ops().expect("CUDA MemoryOps");
        let w = mem
            .upload(&tensor(seed, &[k, n]))
            .expect("w の常駐アップロード");
        let b = mem
            .upload(&tensor(seed ^ 0xB1A5, &[n]))
            .expect("bias の常駐アップロード");
        Self { name, k, n, w, b }
    }
}

/// 本番 `gemm_resident_rhs` を 1 回呼ぶ（`act_relu = false`）。
fn run_prod(ops: &CudaBackendOps, l: &Layer, a: &Tensor<f32>) -> Tensor<f32> {
    let wshape = [l.k, l.n];
    let wv = DeviceBufferView::new(&l.w, 0, &wshape).expect("w view");
    let bshape = [l.n];
    let bv = DeviceBufferView::new(&l.b, 0, &bshape).expect("bias view");
    ops.gemm_resident_rhs(a, wv, Some(bv))
        .expect("gemm_resident_rhs")
}

/// 分解 arm の 1 反復で得た区間。
#[derive(Default)]
struct Parts {
    device_handle: f64,
    mem_new: f64,
    h2d_upload: f64,
    alloc_c: f64,
    gemm_lookup: f64,
    launch_issue: f64,
    kernel_wait: f64,
    d2h_download: f64,
}

/// `gemm_resident_rhs` の本体を区間に分けて再実行する。`sync_after_launch` が
/// 真なら起動直後に明示同期して `kernel_wait` を分離する（syncsplit）。
/// `act_relu` は what-if（融合 relu）用で、本番は `false`。
fn run_decomposed(
    l: &Layer,
    a: &Tensor<f32>,
    sync_after_launch: bool,
    act_relu: bool,
) -> (Tensor<f32>, Parts) {
    let m = a.shape()[0];
    let mut p = Parts::default();
    let (device, dt) = timed(|| cached_device(0).expect("CUDA device"));
    p.device_handle = dt;
    let (mem, dt) = timed(|| CudaMemory::new(&device));
    p.mem_new = dt;
    let (a_buf, dt) = timed(|| mem.upload(a).expect("a の H2D"));
    p.h2d_upload = dt;
    let (mut c_buf, dt) = timed(|| mem.alloc_zeroed(&[m, l.n]).expect("c の確保"));
    p.alloc_c = dt;
    let (gemm, dt) = timed(|| cached_gemm(&device).expect("CudaGemm"));
    p.gemm_lookup = dt;

    let (_, dt) = timed(|| {
        let a_h = a_buf
            .downcast_handle::<CudaBufferHandle>()
            .expect("a handle");
        let a_arg = a_h.storage.as_ref().expect("a storage").as_arg();
        let w_h = l.w.downcast_handle::<CudaBufferHandle>().expect("w handle");
        let w_view = w_h.storage.as_ref().expect("w storage").view(0..l.k * l.n);
        let b_h = l.b.downcast_handle::<CudaBufferHandle>().expect("b handle");
        let b_view = b_h.storage.as_ref().expect("b storage").view(0..l.n);
        let c_h = c_buf
            .downcast_handle_mut::<CudaBufferHandle>()
            .expect("c handle");
        let mut c_arg = c_h.storage.as_mut().expect("c storage").as_arg_mut();
        gemm.launch_tiled_bias_act_f32_resident(
            &a_arg,
            &w_view,
            Some(&b_view),
            act_relu,
            &mut c_arg,
            m as u32,
            l.n as u32,
            l.k as u32,
        )
        .expect("launch_tiled_bias_act_f32_resident");
    });
    p.launch_issue = dt;

    if sync_after_launch {
        let (_, dt) = timed(|| device.stream().synchronize().expect("stream synchronize"));
        p.kernel_wait = dt;
    }
    let (out, dt) = timed(|| mem.download(&c_buf).expect("c の D2H"));
    p.d2h_download = dt;
    (out, p)
}

fn record_parts(rec: &mut Rec, arm: &str, l: &Layer, batch: usize, p: &Parts) {
    let sync = arm == "syncsplit";
    let mut total = 0.0;
    for (name, v) in [
        ("device_handle", p.device_handle),
        ("mem_new", p.mem_new),
        ("h2d_upload", p.h2d_upload),
        ("alloc_c", p.alloc_c),
        ("gemm_lookup", p.gemm_lookup),
        ("launch_issue", p.launch_issue),
        ("d2h_download", p.d2h_download),
    ] {
        rec.push(arm, l.name, batch, name, v);
        total += v;
    }
    if sync {
        rec.push(arm, l.name, batch, "kernel_wait", p.kernel_wait);
        total += p.kernel_wait;
    }
    rec.push(arm, l.name, batch, "total", total);
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn relu_host(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous")
        .iter()
        .map(|v| if *v > 0.0 { *v } else { 0.0 }.to_bits())
        .collect()
}

/// 1 batch 分の計測。層の種類（l1／l2／relu／mse）と what-if を回す。
fn run_batch(ops: &CudaBackendOps, batch: usize, rec: &mut Rec) {
    let l1 = Layer::new(ops, "l1", D_IN, D_HIDDEN, 0x1111_1111);
    let l2 = Layer::new(ops, "l2", D_HIDDEN, D_OUT, 0x2222_2222);
    let x = tensor(0xDA7A_0001, &[batch, D_IN]);
    let target = tensor(0xDA7A_0002, &[batch, D_OUT]);

    // 入力に依存する中間値と bit 一致の基準（未計時）。
    let y1_ref = run_prod(ops, &l1, &x);
    let h_ref = ops.relu(&y1_ref).expect("relu");
    let y2_ref = run_prod(ops, &l2, &h_ref);
    let loss_ref = ops
        .mse_loss(&y2_ref, &target, MseReduction::Mean)
        .expect("mse");

    // hard assert: 分解した出力が本番出力と bit 一致（同一カーネル・同一入力）。
    for (l, a, want) in [(&l1, &x, &y1_ref), (&l2, &h_ref, &y2_ref)] {
        for sync in [false, true] {
            let (got, _) = run_decomposed(l, a, sync, false);
            assert_eq!(
                bits(&got),
                bits(want),
                "分解した {} の出力が本番 gemm_resident_rhs と bit 不一致 (batch={batch}, sync={sync})",
                l.name
            );
        }
    }
    // what-if（record-only）: 融合 relu が「本番 GEMM + ホスト relu」と bit 一致するか。
    let (fused, _) = run_decomposed(&l1, &x, false, true);
    let whatif_bits_equal = bits(&fused) == relu_host(&y1_ref);

    let total_iters = WARMUP + ROUNDS * ITERS_PER_ROUND;
    for it in 0..total_iters {
        let on = it >= WARMUP;
        let round = it.saturating_sub(WARMUP) / ITERS_PER_ROUND;
        // ラウンドごとに arm 順を反転して順序効果を打ち消す。
        let mut arms = ["prod", "nosync", "syncsplit"];
        if on && round % 2 == 1 {
            arms.reverse();
        }
        for arm in arms {
            for (l, a) in [(&l1, &x), (&l2, &h_ref)] {
                match arm {
                    "prod" => {
                        let (_, dt) = timed(|| run_prod(ops, l, a));
                        if on {
                            rec.push("prod", l.name, batch, "total", dt);
                        }
                    }
                    _ => {
                        let (_, p) = run_decomposed(l, a, arm == "syncsplit", false);
                        if on {
                            record_parts(rec, arm, l, batch, &p);
                        }
                    }
                }
            }
        }
        let (_, dt) = timed(|| ops.relu(&y1_ref).expect("relu"));
        let (_, dt_mse) = timed(|| {
            ops.mse_loss(&y2_ref, &target, MseReduction::Mean)
                .expect("mse")
        });
        let (_, p) = run_decomposed(&l1, &x, false, true);
        if on {
            rec.push("prod", "relu", batch, "total", dt);
            rec.push("prod", "mse", batch, "total", dt_mse);
            let total = p.device_handle
                + p.mem_new
                + p.h2d_upload
                + p.alloc_c
                + p.gemm_lookup
                + p.launch_issue
                + p.d2h_download;
            rec.push("whatif", "l1", batch, "total", total);
        }
    }

    for (layer, out) in [
        ("l1", bits_hash(y1_ref.contiguous().as_slice().expect("c"))),
        ("l2", bits_hash(y2_ref.contiguous().as_slice().expect("c"))),
        ("relu", bits_hash(h_ref.contiguous().as_slice().expect("c"))),
        (
            "mse",
            bits_hash(loss_ref.contiguous().as_slice().expect("c")),
        ),
    ] {
        rec.checksums
            .push((("prod".into(), layer.into(), batch), out));
    }
    rec.checksums.push((
        ("whatif_bits_equal".into(), "l1".into(), batch),
        u64::from(whatif_bits_equal),
    ));
}

/// 実機（CUDA）依存の診断。batch × 層の種類 × arm を計測し `DIAG_JSON` を出す。
/// `--test-threads=1` 必須（ファイル冒頭コメント参照）。
#[test]
#[ignore = "CUDA 実機必須の record-only 診断（RULE.txt・orchestrate.sh 参照）"]
fn cuda_train_forward_resident_backend_phases() {
    let ops = CudaBackendOps::new(0);
    let mut rec = Rec::default();
    for batch in BATCHES {
        run_batch(&ops, batch, &mut rec);
    }
    for ((arm, layer, batch, phase), secs) in &rec.cells {
        let q = median_q1_q3(secs).expect("サンプルは非空・非 NaN");
        let mut sorted = secs.clone();
        sorted.sort_by(f64::total_cmp);
        let chk = rec
            .checksums
            .iter()
            .find(|((a, ly, b), _)| a == arm && ly == layer && b == batch)
            .filter(|_| phase == "total")
            .map(|(_, h)| format!("\"{h:016x}\""))
            .unwrap_or_else(|| "null".to_string());
        // 文字列値は本ファイル内の固定リテラルのみ（エスケープ不要）。数値は有限。
        println!(
            "{PREFIX}{{\"issue\":2116,\"layer\":\"backend\",\"arm\":\"{arm}\",\"kind\":\"{layer}\",\"batch\":{batch},\"phase\":\"{phase}\",\"median_s\":{:e},\"q1_s\":{:e},\"q3_s\":{:e},\"min_s\":{:e},\"max_s\":{:e},\"n\":{},\"checksum_bits\":{chk}}}",
            q.median,
            q.q1,
            q.q3,
            sorted[0],
            sorted[sorted.len() - 1],
            secs.len(),
        );
    }
    for ((arm, layer, batch), v) in &rec.checksums {
        if arm == "whatif_bits_equal" {
            println!(
                "{PREFIX}{{\"issue\":2116,\"layer\":\"backend\",\"arm\":\"whatif\",\"kind\":\"{layer}\",\"batch\":{batch},\"phase\":\"bits_equal\",\"value\":{}}}",
                *v == 1
            );
        }
    }
}
