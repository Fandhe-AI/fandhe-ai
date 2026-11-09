//! MLP 推論の 2 形状（l1_fwd 64x256x784・l2_fwd 64x10x256）の GEMM「純カーネル時間」を
//! cudaEvent で計測する診断 example（イシュー #4147。GB10 実機専用）。
//!
//! # 役割と呼び出し文脈
//!
//! - `docs/perf/logs/diag-0140-mlp-gb10-4147/run_round.sh` が 1 round につき 1 プロセスで起動し、
//!   標準出力の `MLPEVENT4147` 行を同ディレクトリの `to_values_4147.py` が厳密にパースして
//!   `G-CUDA-INF` の `fandhe_pure_kernel`（clock=`cuda_event`）にする。規則の正は同ディレクトリの
//!   `RULE.txt`（計測前に固定）。相手側は `opp_kernel_cuda_mlp_diag_4147.py`（cuBLAS の同形状 event 計測）。
//! - 境界は `gemm_candidate_event_ab_3215.rs` / `opp_kernel_cuda_diag.py::mm_event`（reps=1）と同じ:
//!   event 対で 1 launch を挟み、H2D・出力確保・event 生成は区間外。warmup 20・計測 20 の中央値。
//!
//! # 意図的な逸脱
//!
//! 本番の infer チェーンは bias・活性化の epilogue を融合しうるが、純 GEMM の入口だけを測る
//! （`CudaGemm::launch_tiled_f32`。選択ロジックは本番と同じ `select_tiled_f32_kernel(0, m, n, k)`）。
//! GEMM のみで ReLU・bias は含めない境界は RULE 項 4 に明記する。
//!
//! # 不変事項
//!
//! 本番ディスパッチ・NVRTC カーネル文字列・REQ-8 の手動境界検査・FMA 契約・tolerance は一切変更しない。
//! 新規 `unsafe` は 0 件（event は cudarc の safe API）。
//!
//! # 実行方法
//!
//! `cargo run --release -p fandhe-ai-backend-cuda --features internal-diagnostics
//! --example cuda_mlp_event_bench_4147`

use bench_harness::rng::Xorshift64Star;
use cudarc::driver::sys::CUevent_flags;
use fandhe_ai_backend_cuda::{CudaDevice, CudaGemm};

/// (shape_id, m, n, k)。`opp_kernel_torch_diag.py::MLP_SHAPES` の推論 2 形状と一致させる。
const SHAPES: [(&str, u32, u32, u32); 2] = [("l1_fwd", 64, 256, 784), ("l2_fwd", 64, 10, 256)];
const WARMUP: usize = 20;
const MEASURED: usize = 20;
/// 入力シード（`gemm_candidate_event_ab_3215` と同値）。
const SEED: u64 = 0xC0FFEE;

/// 中央値（偶数個は中央 2 要素の平均）。空・非有限値は呼び出し側の不変条件違反。
fn median(samples: &[f64]) -> f64 {
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("samples must be NaN-free"));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// `i` 回目の計測で形状の実行順を巡回させる（位置バイアスを均す）。
fn rotated_order(count: usize, i: usize) -> Vec<usize> {
    (0..count).map(|j| (j + i) % count).collect()
}

fn main() {
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(e) => {
            println!("ERROR4147 CudaDevice::new failed: {e}");
            std::process::exit(2);
        }
    };
    let (maj, min) = device.compute_capability();
    println!("DEVICE4147 compute_capability={maj}.{min}");
    let gemm = CudaGemm::new(&device).expect("CudaGemm::new");
    let stream = device.stream().clone();
    let ctx = stream.context().clone();
    // `new_event(None)` は DISABLE_TIMING になるため DEFAULT を明示する。event は区間外で事前生成する。
    let e0 = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .expect("event");
    let e1 = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .expect("event");

    let mut rng = Xorshift64Star::new(SEED);
    let mut bufs = Vec::new();
    for &(_, m, n, k) in &SHAPES {
        let a: Vec<f32> = rng.fill_vec((m * k) as usize);
        let b: Vec<f32> = rng.fill_vec((k * n) as usize);
        let (a_dev, b_dev) = gemm.upload_f32(&a, &b).expect("upload");
        let out = gemm.alloc_output_f32(m, n).expect("alloc");
        bufs.push((a_dev, b_dev, out));
    }

    for (idx, &(_, m, n, k)) in SHAPES.iter().enumerate() {
        for _ in 0..WARMUP {
            let (a, b, o) = &mut bufs[idx];
            gemm.launch_tiled_f32(a, b, o, m, n, k).expect("warmup");
        }
    }
    stream.synchronize().expect("sync");

    let mut ev: Vec<Vec<f64>> = vec![Vec::new(); SHAPES.len()];
    for i in 0..MEASURED {
        for idx in rotated_order(SHAPES.len(), i) {
            let (_, m, n, k) = SHAPES[idx];
            stream.synchronize().expect("drain");
            e0.record(&stream).expect("rec");
            {
                let (a, b, o) = &mut bufs[idx];
                gemm.launch_tiled_f32(a, b, o, m, n, k).expect("launch");
            }
            e1.record(&stream).expect("rec");
            stream.synchronize().expect("sync");
            ev[idx].push(f64::from(e0.elapsed_ms(&e1).expect("elapsed")));
        }
    }
    for (idx, &(name, m, n, k)) in SHAPES.iter().enumerate() {
        println!(
            "MLPEVENT4147 shape={name} m={m} n={n} k={k} kern_event_ms={:.9} samples={}",
            median(&ev[idx]),
            ev[idx].len()
        );
    }
    println!("done4147");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_odd_and_even() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }

    #[test]
    fn rotation_covers_every_shape() {
        for i in 0..4 {
            let mut o = rotated_order(2, i);
            o.sort_unstable();
            assert_eq!(o, vec![0, 1]);
        }
    }
}
