//! イシュー #3205（`docs/perf/infer-chain-fixed-cost-reduction-3205.md` §2・§7）: 推論チェーンの
//! 「層の発行の超過分」（約 10.5 µs）を、確保・解放・残り（driver 呼び出し境界＋lookup＋起動発行）へ
//! 分ける record-only 診断。`#3189` の `infer_chain_phase_diag_3189.rs` と同じ公開 trait
//! （`BackendOps`／`MemoryOps`）だけを使い、`src/` は変更せず `unsafe` も書かない。
//!
//! # 区間
//!
//! | 区間 | 中身 |
//! |---|---|
//! | `alloc_{in,h,out}` | `MemoryOps::alloc_zeroed`（cuMemAllocAsync＋memset の発行）を GPU idle で単独計時（形状 `[64,784]`／`[64,256]`／`[64,10]`） |
//! | `drop_{in,h,out}` | 上記バッファの drop（cuMemFreeAsync＋event 記録の発行）の単独計時 |
//! | `l1_issue`／`l2_issue` | 本番チェーンと同じ列での `linear_forward_device` の戻りまで＋直前層バッファの drop（3189 と同定義） |
//! | `l1_other`／`l2_other` | **導出値**: `issue − alloc − drop` の中央値差。driver 呼び出し境界・`cached_gemm` lookup・起動発行・世代 `Vec` の合算（公開 API から個別には切れない） |
//!
//! 単独計時は GPU idle（各反復の末尾で小バッファを download して同期）で行うため、チェーン内
//! （カーネル実行と重なる）の確保・解放費用とは一致しない。fidelity（`alloc＋drop` の単独値が `issue` に
//! 占める割合）は集計側で併記し、単独値を削減見込みの上限の目安としてのみ使う。
//!
//! 実機計測は `docs/perf/logs/infer-chain-fixed-cost-3205/README.md` の手順で
//! `--ignored --exact --nocapture --test-threads=1` を 5 run 実行する。`BACKEND_JSON` を 1 行 1 区間で出す
//! （時間の大小は assert しない）。環境変数 `INFER_CHAIN_DIAG_KIND` は未設定または `cuda` のみ受け付ける
//! （CPU 混入の拒否。A03）。

use std::time::Instant;

use bench_harness::median_q1_q3;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_tensor_core::buffer::{DeviceBuffer, DeviceBufferView};
use fandhe_ai_tensor_core::{Activation, BackendOps, Tensor};

const BATCH: usize = 64;
const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;
const WARMUP: usize = 20;
const ROUNDS: usize = 4;
const ITERS_PER_ROUND: usize = 20;
const PREFIX: &str = "BACKEND_JSON ";

fn tensor(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    Tensor::new(Xorshift64Star::new(seed).fill_vec(n), shape)
        .expect("形状とデータ長は一致させている")
}

/// 常駐 w・bias（本番では `DeviceParamStore` が保持する resident leaf に相当）。
struct Params {
    w1: DeviceBuffer<f32>,
    b1: DeviceBuffer<f32>,
    w2: DeviceBuffer<f32>,
    b2: DeviceBuffer<f32>,
}

const W1S: [usize; 2] = [D_IN, D_HIDDEN];
const B1S: [usize; 1] = [D_HIDDEN];
const W2S: [usize; 2] = [D_HIDDEN, D_OUT];
const B2S: [usize; 1] = [D_OUT];

fn upload_params(ops: &CudaBackendOps) -> Params {
    let mem = ops.memory_ops().expect("CUDA MemoryOps");
    let up = |seed: u64, shape: &[usize]| mem.upload(&tensor(seed, shape)).expect("常駐 upload");
    Params {
        w1: up(0x1111_1111, &W1S),
        b1: up(0x1111_1111 ^ 0xB1A5, &B1S),
        w2: up(0x2222_2222, &W2S),
        b2: up(0x2222_2222 ^ 0xB1A5, &B2S),
    }
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    let out = f();
    (out, t.elapsed().as_secs_f64())
}

/// 本番チェーン（H2D 以降）の l1_issue／l2_issue を 3189 と同じ定義で計り、同期して返す。
fn chain_issue(ops: &CudaBackendOps, p: &Params, input: &Tensor<f32>) -> (f64, f64) {
    let mem = ops.memory_ops().expect("CUDA MemoryOps");
    let a = mem.upload(input).expect("入力 H2D");
    let (h, l1) = timed(move || {
        let h = ops
            .linear_forward_device(
                &a,
                DeviceBufferView::new(&p.w1, 0, &W1S).expect("w1 view"),
                Some(DeviceBufferView::new(&p.b1, 0, &B1S).expect("b1 view")),
                Activation::Relu,
            )
            .expect("L1");
        drop(a);
        h
    });
    let (y, l2) = timed(move || {
        let y = ops
            .linear_forward_device(
                &h,
                DeviceBufferView::new(&p.w2, 0, &W2S).expect("w2 view"),
                Some(DeviceBufferView::new(&p.b2, 0, &B2S).expect("b2 view")),
                Activation::None,
            )
            .expect("L2");
        drop(h);
        y
    });
    let _ = mem.download(&y).expect("D2H（同期）");
    (l1, l2)
}

/// 形状 `shape` の alloc_zeroed と drop を GPU idle で単独計時する（alloc, drop）。
fn alloc_drop(ops: &CudaBackendOps, sync_probe: &DeviceBuffer<f32>, shape: &[usize]) -> (f64, f64) {
    let mem = ops.memory_ops().expect("CUDA MemoryOps");
    let (buf, alloc) = timed(|| mem.alloc_zeroed(shape).expect("alloc_zeroed"));
    let ((), free) = timed(move || drop(buf));
    // 次反復へ未完了の memset／free を持ち越さない（計時窓の外）。
    let _ = mem.download(sync_probe).expect("同期用 download");
    (alloc, free)
}

fn emit(phase: &str, samples: &[f64]) -> f64 {
    let q = median_q1_q3(samples).expect("標本は非 NaN");
    println!(
        "{PREFIX}{{\"backend\":\"cuda\",\"phase\":\"{phase}\",\"n\":{},\"median_s\":{:.9},\"q1_s\":{:.9},\"q3_s\":{:.9}}}",
        samples.len(),
        q.median,
        q.q1,
        q.q3
    );
    q.median
}

#[test]
#[ignore = "CUDA 実機必須の record-only 診断（docs/perf/logs/infer-chain-fixed-cost-3205/README.md 参照）"]
fn infer_chain_issue_breakdown_diag_3205() {
    match std::env::var("INFER_CHAIN_DIAG_KIND").as_deref() {
        Err(_) | Ok("cuda") => {}
        Ok(other) => panic!("INFER_CHAIN_DIAG_KIND={other} は不可（CUDA 実測のみ）"),
    }
    let ops = CudaBackendOps::new(0);
    let params = upload_params(&ops);
    let input = tensor(0x9000, &[BATCH, D_IN]);
    let probe = ops
        .memory_ops()
        .expect("CUDA MemoryOps")
        .upload(&tensor(0x7000, &[1, 1]))
        .expect("同期用 probe");

    for _ in 0..WARMUP {
        let _ = chain_issue(&ops, &params, &input);
        for s in [[BATCH, D_IN], [BATCH, D_HIDDEN], [BATCH, D_OUT]] {
            let _ = alloc_drop(&ops, &probe, &s);
        }
    }

    let n = ROUNDS * ITERS_PER_ROUND;
    let (mut l1s, mut l2s) = (Vec::new(), Vec::new());
    let mut ad: [(Vec<f64>, Vec<f64>); 3] = Default::default();
    for _ in 0..n {
        let (l1, l2) = chain_issue(&ops, &params, &input);
        l1s.push(l1);
        l2s.push(l2);
        for (i, s) in [[BATCH, D_IN], [BATCH, D_HIDDEN], [BATCH, D_OUT]]
            .iter()
            .enumerate()
        {
            let (a, d) = alloc_drop(&ops, &probe, s);
            ad[i].0.push(a);
            ad[i].1.push(d);
        }
    }
    let l1 = emit("l1_issue", &l1s);
    let l2 = emit("l2_issue", &l2s);
    let mut alloc = [0.0; 3];
    let mut free = [0.0; 3];
    for (i, name) in ["in", "h", "out"].iter().enumerate() {
        alloc[i] = emit(&format!("alloc_{name}"), &ad[i].0);
        free[i] = emit(&format!("drop_{name}"), &ad[i].1);
    }
    // 導出値（中央値差。負は 0 に丸めず符号付きで記録して集計側が fidelity を判定する）。
    let l1_other = l1 - alloc[1] - free[0];
    let l2_other = l2 - alloc[2] - free[1];
    for (name, v) in [
        ("l1_other_derived", l1_other),
        ("l2_other_derived", l2_other),
    ] {
        println!(
            "{PREFIX}{{\"backend\":\"cuda\",\"phase\":\"{name}\",\"n\":1,\"median_s\":{v:.9}}}"
        );
    }
}
