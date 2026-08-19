//! TF32 opt-staged（cp.async 多段パイプライン）カーネルの段数
//! （`stages`）スイープ計測バイナリ（イシュー #742）。
//!
//! `kernels_wmma_opt.rs::WMMA_TF32_STAGED_STAGES`（既定 3）は GB10 実機の
//! occupancy 実測（ncu 16.6%）を踏まえて採否が未確定のまま固定されている。
//! 既定ブロックタイル（block_m=block_n=64・k_tile=16）の static
//! `__shared__` 宣言では 48KiB 上限（
//! [`crate::kernels_mma::MMA_STATIC_SMEM_LIMIT_BYTES`]）により stages<=3
//! しか焼き込めないため、本 example は**計測専用の動的共有メモリ変種**
//! （`backend_cuda::diagnostics::render_wmma_tf32_staged_dyn`。
//! `WMMA_TF32_STAGED_DYNAMIC_SMEM=1` の `#if` 分岐。opt-in 属性
//! `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` で 48KiB を超える
//! 割り当てを行う）を使い、stages 2..=10 を GB10 実測 optin 予算
//! （`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`。
//! `device.rs::shared_memory_per_block_optin`）の範囲で計測する。
//!
//! **本番経路（static・stages=3・`gemm.rs` の 3 段フォールバック選択）は
//! 一切変更しない**。動的 SMEM 変種は `internal-diagnostics` feature
//! （既定 off）配下でのみコンパイルされ、比較基準行として本番経路
//! （[`CudaGemm::launch_wmma_tf32`]。staged 優先選択）も同条件で計測する。
//!
//! ## 実行手順
//!
//! ```sh
//! cargo run -p backend-cuda --example gemm_wmma_tf32_staged_stages_bench \
//!     --release --features internal-diagnostics
//! ```
//!
//! CUDA 非搭載・NVRTC 非搭載・cc<8.0（cp.async 前提）・opt-in 予算
//! 未取得環境では、理由を表示して正常終了する
//! （`gemm_mma_swizzle_bench.rs` と同じ環境適応分岐）。段数ごとの
//! コンパイル失敗・optin 予算超過・parity 不一致は理由付きで
//! SKIP／FAIL 表示し、残りの段数の計測は継続する（fail-closed だが
//! スイープ全体は止めない設計。実装計画 §6「リスクと安全側の倒し方」）。
//!
//! 実測値・SMEM/occupancy 試算・採否判断は
//! `docs/perf/cuda-gemm-wmma-tf32-staged-stages-sweep.md` へ記録する。

use backend_cuda::{CudaDevice, CudaError, CudaGemm, diagnostics};
use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};

/// 決定的シード（`gemm_mma_bench.rs`・`gemm_mma_swizzle_bench.rs` と同一値。
/// 過去 PoC・他ベンチと同じ入力分布に揃える）。
const SEED: u64 = 0xC0FFEE;

/// スイープ対象の段数レンジ（イシュー #742 タイトル「stages 2..10」）。
const STAGES_RANGE: std::ops::RangeInclusive<u32> = 2..=10;

/// スイープ対象の形状（イシュー #742 実装計画 §3「stages ∈ 2..=10 ×
/// size ∈ {2048, 4096}」）。
const BENCH_SIZES: [usize; 2] = [2048, 4096];

/// 正しさ検査用の小形状。M を BLOCK_M（64）の非倍数にして、エピローグ
/// guarded store（REQ-8）の境界分岐を実際に踏ませる
/// （実装計画 §3「非整列端を含む 1 形状」）。N/K は cp.async 16 バイト
/// 整列制約（4 の倍数）を満たす必要があるため崩さない。
const CORRECTNESS_M: u32 = 513;
const CORRECTNESS_N: u32 = 512;
const CORRECTNESS_K: u32 = 512;

/// 統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」
/// （`.claude/rules/coding-rust.md`「バックエンド構成」。緩和しない）。
fn within_tolerance(actual: f32, expected: f32) -> bool {
    let abs_diff = (actual - expected).abs();
    abs_diff < 1e-5 || abs_diff < 1e-3 * expected.abs()
}

/// CPU 参照実装（`f32::mul_add`。FMA 契約を GPU 側と揃える方針
/// `.claude/rules/coding-rust.md`「バックエンド構成」）。M×N×K が
/// 小さい正しさ検査専用形状のみに使うため素朴な 3 重ループで十分。
fn cpu_reference_f32(a: &[f32], b: &[f32], m: u32, n: u32, k: u32) -> Vec<f32> {
    let (m, n, k) = (m as usize, n as usize, k as usize);
    let mut c = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc = a[i * k + p].mul_add(b[p * n + j], acc);
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn tflops(size: usize, median_secs: f64) -> f64 {
    let flops = 2.0 * (size as f64).powi(3);
    flops / median_secs / 1e12
}

/// 動的 SMEM 変種の GPU 実行のみを計測する（H2D/D2H・出力確保は計測区間
/// 外。`gemm_mma_swizzle_bench.rs::measure_mma_f16` と同じ計測方針）。
fn measure_dyn_staged(
    compiled: &backend_cuda::diagnostics::CompiledWmmaTf32StagedDynKernel,
    device: &CudaDevice,
    gemm: &CudaGemm,
    size: usize,
    config: &MeasurementConfig,
) -> Result<f64, CudaError> {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f32> = rng.fill_vec(size * size);
    let b: Vec<f32> = rng.fill_vec(size * size);

    let (a_dev, b_dev) = gemm.upload_f32(&a, &b)?;
    let mut c_dev = gemm.alloc_output_f32(size as u32, size as u32)?;

    let stream = device.stream();
    let measurement = bench_run(config, || {
        compiled
            .launch_tf32_staged_dyn(
                stream,
                &a_dev,
                &b_dev,
                &mut c_dev,
                size as u32,
                size as u32,
                size as u32,
            )
            .expect("dyn staged GEMM launch must succeed on CUDA-equipped runner");
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    Ok(tflops(size, measurement.median_secs))
}

/// 本番経路（static・3 段フォールバック選択。多くの場合 staged
/// stages=3）の GPU 実行のみを計測する（比較基準行）。
fn measure_production_wmma_tf32(gemm: &CudaGemm, size: usize, config: &MeasurementConfig) -> f64 {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f32> = rng.fill_vec(size * size);
    let b: Vec<f32> = rng.fill_vec(size * size);

    let (a_dev, b_dev) = gemm
        .upload_f32(&a, &b)
        .expect("f32 upload must succeed on CUDA-equipped runner");
    let mut c_dev = gemm
        .alloc_output_f32(size as u32, size as u32)
        .expect("f32 output allocation must succeed on CUDA-equipped runner");

    let measurement = bench_run(config, || {
        gemm.launch_wmma_tf32(
            &a_dev,
            &b_dev,
            &mut c_dev,
            size as u32,
            size as u32,
            size as u32,
        )
        .expect("production wmma tf32 GEMM must succeed on CUDA-equipped runner");
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    tflops(size, measurement.median_secs)
}

fn main() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            println!(
                "backend-cuda gemm_wmma_tf32_staged_stages_bench: CUDA driver unavailable \
                 ({detail}); skipping."
            );
            return;
        }
        Err(other) => {
            println!(
                "backend-cuda gemm_wmma_tf32_staged_stages_bench: CudaDevice::new failed \
                 ({other}); skipping."
            );
            return;
        }
    };

    let num_sms = device.multiprocessor_count();
    let smem_per_sm = device.shared_memory_per_multiprocessor();
    let optin_budget = device.shared_memory_per_block_optin();
    println!(
        "num_sms={num_sms:?} smem_per_multiprocessor={smem_per_sm:?} \
         smem_per_block_optin={optin_budget:?}"
    );

    let optin_budget = match optin_budget {
        Some(b) => b,
        None => {
            println!(
                "backend-cuda gemm_wmma_tf32_staged_stages_bench: \
                 CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN unavailable; skipping."
            );
            return;
        }
    };

    let gemm = match CudaGemm::new(&device) {
        Ok(g) => g,
        Err(e) => {
            println!(
                "backend-cuda gemm_wmma_tf32_staged_stages_bench: CudaGemm::new failed \
                 ({e}); nothing to measure. See docs/perf/cuda-gemm-wmma-tf32-staged-stages-sweep.md."
            );
            return;
        }
    };

    // 正しさ検査用の CPU 参照値（統一複合判定。緩和しない）。
    let mut rng = Xorshift64Star::new(SEED);
    let a_ref: Vec<f32> = rng.fill_vec((CORRECTNESS_M * CORRECTNESS_K) as usize);
    let b_ref: Vec<f32> = rng.fill_vec((CORRECTNESS_K * CORRECTNESS_N) as usize);
    let expected = cpu_reference_f32(&a_ref, &b_ref, CORRECTNESS_M, CORRECTNESS_N, CORRECTNESS_K);

    println!(
        "stages,smem_bytes,blocks_per_sm_limit,warps_per_sm_limit,{}",
        BENCH_SIZES
            .iter()
            .map(|s| format!("dyn_tflops_{s},static_tflops_{s},ratio_{s}"))
            .collect::<Vec<_>>()
            .join(",")
    );

    for stages in STAGES_RANGE {
        let cfg = diagnostics::WmmaTf32StagedKernelConfig {
            stages,
            ..diagnostics::WmmaTf32StagedKernelConfig::default_tf32_staged()
        };

        let smem_bytes = match diagnostics::wmma_tf32_staged_dyn_smem_bytes(&cfg) {
            Ok(v) => v,
            Err(e) => {
                println!("stages={stages}: SKIP (smem byte calculation failed: {e})");
                continue;
            }
        };

        // occupancy 上限概算（受入条件 4: 段数ごとの SMEM 所要と occupancy
        // 上限の算出・出力）。`floor(SM あたり SMEM / ブロックあたり SMEM)`
        // が SM あたり同時常駐可能ブロック数の上限（レジスタ圧・スレッド数
        // 上限による追加制約は考慮しない粗い上限。ncu 実測との突合は
        // docs/perf/cuda-gemm-wmma-tf32-staged-stages-sweep.md 側で行う）。
        let blocks_per_sm_limit = smem_per_sm.map(|s| u64::from(s) / smem_bytes.max(1));
        // WMMA_TF32_STAGED_THREADS は 128 スレッド = 4 warp 固定
        // （kernels_wmma_opt.rs::WMMA_TF32_STAGED_THREADS）。
        const WARPS_PER_BLOCK: u64 = 4;
        let warps_per_sm_limit = blocks_per_sm_limit.map(|b| b * WARPS_PER_BLOCK);

        let rendered = match diagnostics::render_wmma_tf32_staged_dyn(&cfg, optin_budget) {
            Ok(r) => r,
            Err(e) => {
                println!(
                    "stages={stages}: SKIP (validate_wmma_tf32_staged_dyn_config rejected: {e})"
                );
                continue;
            }
        };
        let compiled = match rendered.compile(&device) {
            Ok(c) => c,
            Err(e) => {
                println!("stages={stages}: SKIP (NVRTC compile / opt-in attribute failed: {e})");
                continue;
            }
        };

        // 正しさ検査（先行）: SKIP/FAIL は理由付きで表示し、この段数の
        // 計測をスキップして残りの段数の計測を継続する（実装計画 §6）。
        let (a_dev, b_dev) = match gemm.upload_f32(&a_ref, &b_ref) {
            Ok(bufs) => bufs,
            Err(e) => {
                println!("stages={stages}: SKIP (correctness upload failed: {e})");
                continue;
            }
        };
        let mut c_dev = match gemm.alloc_output_f32(CORRECTNESS_M, CORRECTNESS_N) {
            Ok(buf) => buf,
            Err(e) => {
                println!("stages={stages}: SKIP (correctness output alloc failed: {e})");
                continue;
            }
        };
        if let Err(e) = compiled.launch_tf32_staged_dyn(
            device.stream(),
            &a_dev,
            &b_dev,
            &mut c_dev,
            CORRECTNESS_M,
            CORRECTNESS_N,
            CORRECTNESS_K,
        ) {
            println!("stages={stages}: FAIL (correctness launch failed: {e})");
            continue;
        }
        let actual = match gemm.download_f32(&c_dev) {
            Ok(v) => v,
            Err(e) => {
                println!("stages={stages}: SKIP (correctness download failed: {e})");
                continue;
            }
        };
        let mismatch = actual
            .iter()
            .zip(expected.iter())
            .filter(|(a, e)| !within_tolerance(**a, **e))
            .count();
        if mismatch > 0 {
            println!(
                "stages={stages}: FAIL (parity mismatch vs CPU f32::mul_add reference: \
                 {mismatch}/{} elements outside rel 1e-3 / abs 1e-5 tolerance; not measuring)",
                actual.len()
            );
            continue;
        }

        let mut row = format!(
            "{stages},{smem_bytes},{},{}",
            blocks_per_sm_limit.map_or("n/a".to_string(), |v| v.to_string()),
            warps_per_sm_limit.map_or("n/a".to_string(), |v| v.to_string()),
        );
        for &size in &BENCH_SIZES {
            let config = MeasurementConfig::default();
            let dyn_result = measure_dyn_staged(&compiled, &device, &gemm, size, &config);
            let static_tflops = measure_production_wmma_tf32(&gemm, size, &config);
            match dyn_result {
                Ok(dyn_tflops) => {
                    let ratio = if static_tflops != 0.0 {
                        dyn_tflops / static_tflops
                    } else {
                        f64::NAN
                    };
                    row.push_str(&format!(",{dyn_tflops:.4},{static_tflops:.4},{ratio:.4}"));
                }
                Err(e) => {
                    println!(
                        "stages={stages} size={size}: SKIP measurement (dyn launch failed: {e})"
                    );
                    row.push_str(",n/a,n/a,n/a");
                }
            }
        }
        println!("{row}");
    }
}
