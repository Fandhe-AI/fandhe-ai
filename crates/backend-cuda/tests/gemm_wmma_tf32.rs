//! WMMA(TF32) GEMM の CPU-CUDA 数値一致回帰テスト（TASK-11.1c・#62）。
//!
//! `tests/cpu_cuda_parity.rs`（naive f32、#54）と同じ方針で、判定式・閾値は
//! `fandhe_ai_backend_cpu::assert_parity`（統一複合判定「相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満」の唯一の実体）に一本化し、ここでローカル複製しない
//! （`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差を
//! 単独で緩和しない」）。CPU 参照実装は `fandhe_ai_backend_cpu::matmul_reference_fma`
//! （FMA 契約の唯一の参照点）を用いる。
//!
//! **TF32 経路特有の留意点**（`docs/cuda-tensor-core-design.md` 6 節）: TF32 は
//! f32 の仮数部 23bit を 10bit に丸めて Tensor Core へ投入するため、tiled f32
//! （フル精度）との比較よりも誤差が大きくなりうる。統一複合判定は TF32 前提の
//! 複合指標として改定済みであり、本テストはその閾値内への収束を実測で確認する
//! （閾値自体は変更しない。変更が必要と判明した場合は #186 へ引き渡す）。
//!
//! **実機依存の分離**: 形状網羅・K=4096 ストレスケースは `#[ignore]`（DGX Spark
//! GB10 等の実機必須）。環境適応スモークテストのみ通常 CI（self-hosted・CUDA
//! toolkit 非搭載）で実行され、CUDA/NVRTC 非搭載環境では早期 return で green に
//! なる（`tests/cpu_cuda_parity.rs` の分岐パターンを踏襲）。

use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaGemm};

/// 決定的シードで A・B（f32）を生成し、CPU 参照実装と WMMA(TF32) カーネルの
/// 出力を [`fandhe_ai_backend_cpu::assert_parity`] で照合する。
fn assert_wmma_tf32_parity(gemm: &CudaGemm, context: &str, seed: u64, m: u32, n: u32, k: u32) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    let mut c_ref = vec![0.0f32; (m as usize) * (n as usize)];
    fandhe_ai_backend_cpu::matmul_reference_fma(
        &a, &b, &mut c_ref, m as usize, n as usize, k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed test input");

    let c_gpu = gemm
        .run_wmma_tf32(&a, &b, m, n, k)
        .expect("CudaGemm::run_wmma_tf32 must succeed on a compute capability >= 8.0 test runner");

    fandhe_ai_backend_cpu::assert_parity(context, &c_gpu, &c_ref);
}

/// 環境適応型のスモークテスト（`#[ignore]` なし。通常 CI で実行）。
///
/// `tests/cpu_cuda_parity.rs::naive_f32_parity_smoke_env_adaptive` と同じ
/// 分岐パターンで CUDA driver／NVRTC 非搭載環境のエラーを早期 return し
/// green とする。CUDA+toolkit+Tensor Core（compute capability 8.0 以降）搭載
/// 環境でのみ小形状（32×32×32。WMMA ブロックタイル 1 個ぶん）で
/// `assert_parity` による複合判定を実施する。
///
/// レビュー指摘 #62 反映: `CudaGemm::new` は WMMA(TF32) カーネルのコンパイル
/// 失敗を `Err` として早期 return せず `wmma_tf32_error` に退避する
/// （`gemm.rs::CudaGemm::new` 参照）ため、`new` はここでは
/// `NvrtcUnavailable` 以外で失敗しない。WMMA(TF32) 固有の失敗
/// （`<mma.h>` 未解決・compute capability 8.0 未満）は `run_wmma_tf32` の
/// `CudaError::WmmaUnavailable` として表面化するため、`new` 成功後に
/// `run_wmma_tf32` 側で早期 return を判定する。
#[test]
fn wmma_tf32_parity_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            assert!(!detail.is_empty(), "detail message must not be empty");
            return;
        }
        Err(CudaError::Driver(_)) => return,
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    };

    let gemm = match CudaGemm::new(&device) {
        Ok(gemm) => gemm,
        Err(CudaError::NvrtcUnavailable { detail }) => {
            // libcuda はあるが libnvrtc が dlopen できない環境（CUDA driver
            // のみで toolkit 非搭載）。naive/tiled 4 カーネルのコンパイルも
            // NVRTC を要求するため、この分岐で `new` 自体が失敗する。
            assert!(!detail.is_empty());
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaGemm::new: {other}"),
    };

    wmma_tf32_parity_or_skip(&gemm, "smoke 32x32x32", 1, 32, 32, 32);
}

/// `assert_wmma_tf32_parity` のスモークテスト専用版。WMMA(TF32) カーネルが
/// `CudaGemm::new` 時点でコンパイル・ロードに失敗していた場合
/// （`CudaError::WmmaUnavailable`。`<mma.h>` 未解決・compute capability 8.0
/// 未満の環境）は early return 相当として何もせず戻る（レビュー指摘 #62）。
/// naive/tiled は `new` の時点で既に使用可能であることが確定しているため
/// 巻き添えにしない。
fn wmma_tf32_parity_or_skip(gemm: &CudaGemm, context: &str, seed: u64, m: u32, n: u32, k: u32) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    let c_gpu = match gemm.run_wmma_tf32(&a, &b, m, n, k) {
        Ok(c_gpu) => c_gpu,
        Err(CudaError::WmmaUnavailable { detail }) => {
            assert!(!detail.is_empty());
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from run_wmma_tf32: {other}"),
    };

    let mut c_ref = vec![0.0f32; (m as usize) * (n as usize)];
    fandhe_ai_backend_cpu::matmul_reference_fma(
        &a, &b, &mut c_ref, m as usize, n as usize, k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed test input");
    fandhe_ai_backend_cpu::assert_parity(context, &c_gpu, &c_ref);
}

/// 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須の形状網羅
/// テスト。受け入れ条件の本体。
///
/// 形状ケース: ブロックタイル倍数（32×32×32・64×64×64）・非倍数
/// （WMMA_TF32_BLOCK_M/N=32、WMMA_TF32_K_TILE=8 の端境界を踏む
/// 17×23×19・33×31×65）・非正方（64×96×128）・極小（1×1×1、K タイル未満）。
/// `tests/cpu_cuda_parity.rs::naive_f32_matches_reference_across_shapes` と
/// 同じ形状セットを踏襲しつつ、WMMA 固有のタイル境界（32・8）を追加する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
fn wmma_tf32_matches_reference_across_shapes() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");

    let cases: &[(u32, u32, u32)] = &[
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (512, 512, 512),
        (64, 96, 128),
        (1, 1, 1),
        (17, 23, 19),
        (33, 31, 65),
    ];
    for (idx, &(m, n, k)) in cases.iter().enumerate() {
        let context = format!("shape m={m} n={n} k={k}");
        assert_wmma_tf32_parity(&gemm, &context, 2000 + idx as u64, m, n, k);
    }
}

/// K 大のストレスケース群（PoC-v2-5 準拠。`tests/cpu_cuda_parity.rs` の
/// naive f32 版と同じ形状・シード方針を WMMA(TF32) 経路にも適用する）。
///
/// TF32 経路は仮数部の丸め（23bit → 10bit）を経由するため、フル精度
/// tiled f32 版より誤差が大きくなりうる（`docs/cuda-tensor-core-design.md`
/// 6 節）。統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）が
/// この丸め込みを織り込み済みとして機能するかを、この K=4096 ストレス
/// ケースで確認する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
fn wmma_tf32_k4096_stress_poc_v2_5() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");

    assert_wmma_tf32_parity(&gemm, "PoC-v2-5 stress 256x256x4096", 8888, 256, 256, 4096);
    assert_wmma_tf32_parity(
        &gemm,
        "PoC-v2-5 stress 512x512x4096",
        0xFACADE,
        512,
        512,
        4096,
    );
}

/// `k == 0`（`num_k_tiles == 0` 経路。`kernels.rs` の
/// `(k > 0) ? (k - 1) / WMMA_TF32_K_TILE + 1 : 0` 参照）で C が全 0 になる
/// ことを確認する。`tests/gemm_tiled.rs::tiled_f32_zero_k_returns_all_zero`
/// の WMMA(TF32) 版（レビュー指摘 #62: tiled 版には存在するのに WMMA(TF32)
/// 版には対応するテストが無いという指摘への対応）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
fn wmma_tf32_zero_k_returns_all_zero() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");

    let (m, n, k) = (4u32, 4u32, 0u32);
    let c = gemm
        .run_wmma_tf32(&[], &[], m, n, k)
        .expect("k==0 must be a valid no-accumulation shape, not a launch error");
    assert_eq!(c.len(), (m as usize) * (n as usize));
    assert!(c.iter().all(|&v| v == 0.0), "k==0 output must be all zero");
}

/// m==0／n==0（`backend-cpu::gemm_naive` と同じ no-op 形状）で
/// `run_wmma_tf32` を呼んでも CUDA 起動自体が発生せず（`gemm.rs` の
/// `run_wmma_f32_kernel` 早期 return）、空の結果を返すことを実機で確認する。
/// `tests/gemm_tiled.rs::tiled_f32_zero_dim_shape_returns_empty_without_launch`
/// の WMMA(TF32) 版（レビュー指摘 #62）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
fn wmma_tf32_zero_dim_shape_returns_empty_without_launch() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");

    let c = gemm
        .run_wmma_tf32(&[], &[1.0, 2.0, 3.0, 4.0], 0, 4, 1)
        .expect("m==0 must be treated as a no-op, not a driver launch error");
    assert!(c.is_empty());

    let c = gemm
        .run_wmma_tf32(&[1.0, 2.0], &[], 2, 0, 1)
        .expect("n==0 must be treated as a no-op, not a driver launch error");
    assert!(c.is_empty());
}
