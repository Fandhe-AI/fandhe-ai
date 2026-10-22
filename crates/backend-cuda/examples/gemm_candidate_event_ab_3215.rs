//! FP32 GEMM カーネル候補 C1（TMA 2 段 128x64）の「純カーネル時間」を cudaEvent で
//! A/B 計測する診断 example（イシュー #3215。GB10 実機専用）。
//!
//! # 役割と呼び出し文脈
//!
//! - `docs/perf/logs/cuda-tma-stage2-ab-gb10-3215/orchestrate_gb10.sh` が N 固定ではなく
//!   本 example を 1 プロセス 1 run として 5 回起動し、出力の `ARM3215`／`BITIDENT3215`
//!   行を同ディレクトリの `aggregate.py` が厳密にパースして判定する（規則の正は同
//!   ディレクトリの `RULE.txt`。計測前に固定済み）。
//! - 既存の `gemm_tiled_pipeline_bench` は「起動＋同期」の wall 時計で cudaEvent ではない。
//!   fandhe 側の N>=1024 の event 計時は #3185 の残課題だったため、cuBLAS の `mm_event`
//!   （reps=1。`opp_kernel_cuda_diag.py`）と同じ境界を取る本 example で埋める。
//!
//! # 腕（N ごと。計画 §2.1）
//!
//! - `cur`: 本番の `select_tiled_f32_kernel` が実際に選ぶカーネル
//!   （`CudaGemm::launch_tiled_f32`。N<1024 は 64x64 pipeline、N>=1024 かつ K>=1024 は
//!   128x64 pipeline）。計画書は `launch_tiled_f32_pooled(Select)` を挙げていたが、同関数は
//!   `cfg(test)` 限定で example から到達できない。選択ロジックは同一
//!   （`select_tiled_f32_kernel(0, m, n, k)`）で、出力確保は計測区間外のため差は生じない
//!   （意図的な逸脱として RULE.txt に記録）。
//! - `pipe128`: 同タイル・同 3 stage の cp.async 128x64（帰属用の参照腕）。
//! - `tma128_none`／`tma128_strided`: 候補 C1 の 2 腕（#3214）。
//!
//! TMA のテンソルマップは `prepare_tiled_pipeline_tma_maps` で計測区間の外に 1 回だけ
//! encode し、計測は `launch_tiled_pipeline_tma_f32_prepared` のみを挟む。
//!
//! # 区間
//!
//! - 主指標 `kern_event_ms`: event 対で 1 launch を挟む（H2D・出力確保・event 生成は区間外）。
//! - 副指標 `kern_b2b_ms`（記録のみ）: 連続 10 launch を挟み 1 回あたりに割る。
//! - 各腕で warmup 20・計測 20 を取り中央値を出す。計測ごとに腕の実行順を巡回させる。
//!
//! # 不変事項
//!
//! 本番ディスパッチ・NVRTC カーネル文字列・REQ-8 の手動境界検査・FMA 契約・tolerance は
//! 一切変更しない。新規 `unsafe` は 0 件（event は cudarc の safe API）。bit 一致は
//! `f32::to_bits` の FNV-1a 64bit で示し、checksum では代用しない。
//!
//! # 実行方法
//!
//! `cargo run --release -p fandhe-ai-backend-cuda --features internal-diagnostics
//! --example gemm_candidate_event_ab_3215`

use bench_harness::rng::Xorshift64Star;
use cudarc::driver::sys::CUevent_flags;
use fandhe_ai_backend_cuda::{
    CudaDevice, CudaError, CudaGemm, TiledPipelineFunction, TmaRowMap, TmaSwizzleA,
    TmaTiledPipelineFunction,
};

/// 対象 N（正方）。判定対象（結線範囲）は N>=1024 だが 256・512 も記録する。
const SIZES: [usize; 5] = [256, 512, 1024, 2048, 4096];
const WARMUP: usize = 20;
const MEASURED: usize = 20;
/// `kern_b2b` で連続投入する launch 数。
const B2B_LAUNCHES: usize = 10;
/// 128x64 pipeline の既定 stage 数（`TP128_DEFAULT_STAGES` と同値。内部定数は非公開）。
const STAGES_128X64: u32 = 3;
/// 入力シード（`gemm_tiled_pipeline_bench` と同値）。
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

/// 出力全要素の `to_bits` を FNV-1a 64bit でハッシュする（bit 同一性の証拠。
/// checksum は bit 同一の証拠にならないため使わない）。
fn fnv1a_bits(values: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in values {
        for byte in v.to_bits().to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// `i` 回目の計測で腕を巡回させた実行順を返す（位置バイアスを均す）。
fn rotated_order(arm_count: usize, i: usize) -> Vec<usize> {
    (0..arm_count).map(|j| (j + i) % arm_count).collect()
}

struct Arm {
    name: &'static str,
    kind: ArmKind,
}

enum ArmKind {
    Cur,
    Pipe128(TiledPipelineFunction),
    Tma(TmaTiledPipelineFunction),
}

fn main() {
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(e) => {
            println!("ERROR3215 CudaDevice::new failed: {e}");
            std::process::exit(2);
        }
    };
    let (cc_major, cc_minor) = device.compute_capability();
    println!("DEVICE3215 compute_capability={cc_major}.{cc_minor}");
    let gemm = CudaGemm::new(&device).expect("CudaGemm::new");

    let compile = |name: &'static str, rows: TmaRowMap| -> Arm {
        let f =
            CudaGemm::compile_tiled_pipeline_tma_128x64_variant(&device, TmaSwizzleA::None, rows)
                .unwrap_or_else(|e| panic!("compile {name}: {e}"));
        Arm {
            name,
            kind: ArmKind::Tma(f),
        }
    };
    let arms: Vec<Arm> = vec![
        Arm {
            name: "cur",
            kind: ArmKind::Cur,
        },
        Arm {
            name: "pipe128",
            kind: ArmKind::Pipe128(
                CudaGemm::compile_tiled_pipeline_128x64_variant(&device, STAGES_128X64)
                    .expect("compile pipe128"),
            ),
        },
        compile("tma128_none", TmaRowMap::Contiguous),
        compile("tma128_strided", TmaRowMap::Strided),
    ];

    let stream = device.stream().clone();
    let ctx = stream.context().clone();
    // `new_event(None)` は DISABLE_TIMING になるため DEFAULT を明示する。
    // event は計測区間の外で事前生成し、生成コストを区間へ混ぜない。
    let e0 = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .expect("event");
    let e1 = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .expect("event");

    for &size in &SIZES {
        let n = size as u32;
        let mut rng = Xorshift64Star::new(SEED);
        let a: Vec<f32> = rng.fill_vec(size * size);
        let b: Vec<f32> = rng.fill_vec(size * size);
        let (a_dev, b_dev) = gemm.upload_f32(&a, &b).expect("upload");
        let mut outs: Vec<_> = arms
            .iter()
            .map(|_| gemm.alloc_output_f32(n, n).expect("alloc"))
            .collect();
        // TMA 腕のテンソルマップは計測区間の外で 1 回だけ encode する。
        let maps: Vec<_> = arms
            .iter()
            .map(|arm| match &arm.kind {
                ArmKind::Tma(f) => Some(
                    gemm.prepare_tiled_pipeline_tma_maps(f, &a_dev, &b_dev, (n, n, n))
                        .expect("prepare maps"),
                ),
                _ => None,
            })
            .collect();

        let launch = |idx: usize, outs: &mut Vec<_>| -> Result<(), CudaError> {
            match (&arms[idx].kind, &maps[idx]) {
                (ArmKind::Cur, _) => gemm.launch_tiled_f32(&a_dev, &b_dev, &mut outs[idx], n, n, n),
                (ArmKind::Pipe128(f), _) => {
                    gemm.launch_tiled_pipeline_f32(f, &a_dev, &b_dev, &mut outs[idx], n, n, n)
                }
                (ArmKind::Tma(f), Some(m)) => {
                    gemm.launch_tiled_pipeline_tma_f32_prepared(f, m, &mut outs[idx])
                }
                (ArmKind::Tma(_), None) => unreachable!("maps prepared for every TMA arm"),
            }
        };

        // warmup（腕ごと WARMUP 回）。
        for idx in 0..arms.len() {
            for _ in 0..WARMUP {
                launch(idx, &mut outs).expect("warmup launch");
            }
        }
        stream.synchronize().expect("sync");

        // bit 一致: 計測窓の外で各腕の出力全要素をハッシュする。
        let mut hashes = Vec::new();
        for idx in 0..arms.len() {
            launch(idx, &mut outs).expect("verify launch");
            stream.synchronize().expect("sync");
            let host = gemm.download_f32(&outs[idx]).expect("download");
            hashes.push(fnv1a_bits(&host));
        }
        for (idx, arm) in arms.iter().enumerate() {
            println!(
                "BITIDENT3215 n={size} arm={} fnv=0x{:016x} elems={}",
                arm.name,
                hashes[idx],
                size * size
            );
        }

        // 主指標: 1 launch を event 対で挟む。
        let mut ev: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        for i in 0..MEASURED {
            for idx in rotated_order(arms.len(), i) {
                stream.synchronize().expect("drain");
                e0.record(&stream).expect("rec");
                launch(idx, &mut outs).expect("launch");
                e1.record(&stream).expect("rec");
                stream.synchronize().expect("sync");
                ev[idx].push(f64::from(e0.elapsed_ms(&e1).expect("elapsed")));
            }
        }
        // 副指標: 連続 B2B_LAUNCHES launch を挟み 1 回あたりに割る。
        let mut b2b: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        for i in 0..MEASURED {
            for idx in rotated_order(arms.len(), i) {
                stream.synchronize().expect("drain");
                e0.record(&stream).expect("rec");
                for _ in 0..B2B_LAUNCHES {
                    launch(idx, &mut outs).expect("launch");
                }
                e1.record(&stream).expect("rec");
                stream.synchronize().expect("sync");
                b2b[idx]
                    .push(f64::from(e0.elapsed_ms(&e1).expect("elapsed")) / B2B_LAUNCHES as f64);
            }
        }

        let cur_tile = gemm
            .tiled_pipeline_tile_for(n, n)
            .map_or_else(|| "Classic".to_string(), |t| format!("{t:?}"));
        for (idx, arm) in arms.iter().enumerate() {
            let tile = if arm.name == "cur" {
                cur_tile.as_str()
            } else {
                "Bm128Bn64"
            };
            let kern = median(&ev[idx]);
            let tflops = 2.0 * (size as f64).powi(3) / (kern / 1e3) / 1e12;
            println!(
                "ARM3215 n={size} arm={} kern_event_ms={:.6} kern_b2b_ms={:.6} tflops={:.4} \
                 tile={tile} samples={}",
                arm.name,
                kern,
                median(&b2b[idx]),
                tflops,
                ev[idx].len()
            );
        }
    }
    println!("done3215");
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
    fn fnv_distinguishes_signed_zero_and_is_deterministic() {
        assert_ne!(fnv1a_bits(&[0.0]), fnv1a_bits(&[-0.0]));
        assert_eq!(fnv1a_bits(&[1.0, 2.0]), fnv1a_bits(&[1.0, 2.0]));
        assert_ne!(fnv1a_bits(&[1.0, 2.0]), fnv1a_bits(&[2.0, 1.0]));
    }

    #[test]
    fn rotation_covers_every_arm_each_iteration() {
        for i in 0..8 {
            let mut o = rotated_order(4, i);
            assert_eq!(o.len(), 4);
            o.sort_unstable();
            assert_eq!(o, vec![0, 1, 2, 3]);
        }
        assert_eq!(rotated_order(4, 1), vec![1, 2, 3, 0]);
    }
}
