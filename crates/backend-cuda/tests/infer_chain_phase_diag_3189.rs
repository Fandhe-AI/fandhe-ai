//! イシュー #3189（`docs/perf/infer-reuse-phase-breakdown.md` §12）: GB10 の CUDA infer reuse
//! （`Sequential::predict_resident` → `DeviceParamStore::predict_device_chain`）の下位層分解診断。
//!
//! 本ファイルは backend-cuda の公開 trait（`BackendOps`／`MemoryOps`）だけで、本番チェーンと同じ
//! 呼び出し列（`upload` → L1 `linear_forward_device`〈784→256＋bias＋ReLU 融合〉→ L2
//! `linear_forward_device`〈256→10＋bias〉→ `download`）を区間に分けて計時する。`src/` は変更せず
//! `unsafe` も書かない（2116 の下位層テスト `train_forward_resident_diag_tests.rs` は非公開 API へ
//! 届くため `lib.rs` に mod 行を要したが、本ファイルは公開 API だけで足りる）。
//!
//! # 区間
//!
//! | 区間 | 中身 |
//! |---|---|
//! | `h2d_upload` | 入力の `MemoryOps::upload`（H2D＋確保） |
//! | `l1_issue`／`l2_issue` | `linear_forward_device` の戻りまで（確保＋カーネル起動の発行。完了は待たない） |
//! | `download` | `MemoryOps::download`（先行カーネルの完了待ち＋D2H。同期点） |
//! | `download_again` | 同じバッファの 2 回目の `download`（カーネル完了済みなので純 D2H） |
//! | `kernel_wait` | `download − download_again`（導出値。公開 API からは stream を直接同期できないための代替） |
//!
//! `download_again` は 1 反復あたり 1 回余分に D2H するため、`prod` 区間の合計（fidelity の評価）には
//! 含めない。`prod` 区間 = h2d_upload＋l1_issue＋l2_issue＋download の合計が、本番の
//! `predict_device_chain` のうち backend 呼び出し部分に相当する（上位の tape 生成・snapshot は
//! `crates/facade/tests/cuda_infer_reuse_phase_diag_3189.rs` が担当）。
//!
//! 実機計測は `docs/perf/logs/cuda-train-infer-gb10-3189/RULE.txt` 規則 4 に従い
//! `--ignored --exact --nocapture --test-threads=1` で実行する。`BACKEND_JSON` を 1 行 1 区間で出す
//! record-only 診断（時間の大小は assert しない）。`checksum` は 5 run で一致することを集計側が
//! fail-closed で検査する（bit 同一の証拠ではない）。
//!
//! 環境変数 `INFER_CHAIN_DIAG_KIND` は未設定または `cuda` のみ受け付ける（CPU 混入の拒否。A03）。

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

fn host(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous 直後は as_slice が Some")
        .to_vec()
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

#[derive(Default, Clone, Copy)]
struct Parts {
    h2d_upload: f64,
    l1_issue: f64,
    l2_issue: f64,
    download: f64,
    download_again: f64,
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    let out = f();
    (out, t.elapsed().as_secs_f64())
}

/// 本番チェーンと同じ呼び出し列を 1 回回し、区間と出力を返す。
fn chain(ops: &CudaBackendOps, p: &Params, input: &Tensor<f32>) -> (Tensor<f32>, Parts) {
    let mem = ops.memory_ops().expect("CUDA MemoryOps");
    let mut parts = Parts::default();
    let (a, dt) = timed(|| mem.upload(input).expect("入力 H2D"));
    parts.h2d_upload = dt;
    let (h, dt) = timed(|| {
        ops.linear_forward_device(
            &a,
            DeviceBufferView::new(&p.w1, 0, &W1S).expect("w1 view"),
            Some(DeviceBufferView::new(&p.b1, 0, &B1S).expect("b1 view")),
            Activation::Relu,
        )
        .expect("L1")
    });
    parts.l1_issue = dt;
    let (y, dt) = timed(|| {
        ops.linear_forward_device(
            &h,
            DeviceBufferView::new(&p.w2, 0, &W2S).expect("w2 view"),
            Some(DeviceBufferView::new(&p.b2, 0, &B2S).expect("b2 view")),
            Activation::None,
        )
        .expect("L2")
    });
    parts.l2_issue = dt;
    let (out, dt) = timed(|| mem.download(&y).expect("D2H"));
    parts.download = dt;
    let (again, dt) = timed(|| mem.download(&y).expect("D2H（2 回目）"));
    parts.download_again = dt;
    // 2 回の download が同一内容であること（計測窓の外の健全性確認）。
    assert_eq!(
        host(&out).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        host(&again).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "同一バッファの 2 回の download が不一致"
    );
    (out, parts)
}

fn emit(phase: &str, samples: &[f64]) {
    let q = median_q1_q3(samples).expect("標本は非 NaN");
    println!(
        "{PREFIX}{{\"backend\":\"cuda\",\"phase\":\"{phase}\",\"n\":{},\"median_s\":{:.9},\"q1_s\":{:.9},\"q3_s\":{:.9}}}",
        samples.len(),
        q.median,
        q.q1,
        q.q3
    );
}

#[test]
#[ignore = "CUDA 実機必須の record-only 診断（RULE.txt・orchestrate_gb10.sh 参照）"]
fn infer_chain_phase_diag_3189() {
    match std::env::var("INFER_CHAIN_DIAG_KIND").as_deref() {
        Err(_) | Ok("cuda") => {}
        Ok(other) => panic!("INFER_CHAIN_DIAG_KIND={other} は不可（CUDA 実測のみ）"),
    }
    let ops = CudaBackendOps::new(0);
    let params = upload_params(&ops);
    let input = tensor(0x9000, &[BATCH, D_IN]);

    // 参照出力（warmup 前の 1 回）。以降の全反復が bit 一致することを hard assert する。
    let (reference, _) = chain(&ops, &params, &input);
    let ref_bits: Vec<u32> = host(&reference).iter().map(|v| v.to_bits()).collect();
    for _ in 0..WARMUP {
        let _ = chain(&ops, &params, &input);
    }

    let mut cols: [Vec<f64>; 7] = Default::default();
    for _ in 0..ROUNDS * ITERS_PER_ROUND {
        let (out, p) = chain(&ops, &params, &input);
        let bits: Vec<u32> = host(&out).iter().map(|v| v.to_bits()).collect();
        assert_eq!(bits, ref_bits, "反復間で出力が bit 不一致");
        let kernel_wait = (p.download - p.download_again).max(0.0);
        let total = p.h2d_upload + p.l1_issue + p.l2_issue + p.download;
        for (i, v) in [
            p.h2d_upload,
            p.l1_issue,
            p.l2_issue,
            p.download,
            p.download_again,
            kernel_wait,
            total,
        ]
        .into_iter()
        .enumerate()
        {
            cols[i].push(v);
        }
    }
    for (name, col) in [
        "h2d_upload",
        "l1_issue",
        "l2_issue",
        "download",
        "download_again",
        "kernel_wait_derived",
        "total",
    ]
    .iter()
    .zip(cols.iter())
    {
        emit(name, col);
    }
    println!(
        "{PREFIX}{{\"backend\":\"cuda\",\"phase\":\"checksum\",\"checksum\":{}}}",
        bits_hash(&host(&reference))
    );
}
