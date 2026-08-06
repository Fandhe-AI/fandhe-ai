//! WMMA(TF32) GEMM の CPU-CUDA 数値一致回帰テスト（TASK-11.1c・#62）。
//!
//! `tests/cpu_cuda_parity.rs`（naive f32、#54）と同じ方針で、判定式・閾値は
//! `backend_cpu::assert_parity`（統一複合判定「相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満」の唯一の実体）に一本化し、ここでローカル複製しない
//! （`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差を
//! 単独で緩和しない」）。CPU 参照実装は `backend_cpu::matmul_reference_fma`
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

use backend_cuda::{CudaDevice, CudaError, CudaGemm};

/// 決定的シードで A・B（f32）を生成し、CPU 参照実装と WMMA(TF32) カーネルの
/// 出力を [`backend_cpu::assert_parity`] で照合する。
fn assert_wmma_tf32_parity(gemm: &CudaGemm, context: &str, seed: u64, m: u32, n: u32, k: u32) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    let mut c_ref = vec![0.0f32; (m as usize) * (n as usize)];
    backend_cpu::matmul_reference_fma(&a, &b, &mut c_ref, m as usize, n as usize, k as usize)
        .expect("matmul_reference_fma shape validation must pass for well-formed test input");

    let c_gpu = gemm
        .run_wmma_tf32(&a, &b, m, n, k)
        .expect("CudaGemm::run_wmma_tf32 must succeed on a compute capability >= 8.0 test runner");

    backend_cpu::assert_parity(context, &c_gpu, &c_ref);
}

/// 環境適応型のスモークテスト（`#[ignore]` なし。通常 CI で実行）。
///
/// `tests/cpu_cuda_parity.rs::naive_f32_parity_smoke_env_adaptive` と同じ
/// 分岐パターンで CUDA driver／NVRTC 非搭載環境のエラーを早期 return し
/// green とする。CUDA+toolkit+Tensor Core（compute capability 8.0 以降）搭載
/// 環境でのみ小形状（32×32×32。WMMA ブロックタイル 1 個ぶん）で
/// `assert_parity` による複合判定を実施する。
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
            // のみで toolkit 非搭載）。WMMA(TF32) カーネルは `<mma.h>` を
            // 要求するため toolkit 必須（設計メモ 3.2 節）であり、この分岐は
            // naive/tiled 版と同じ理由で early return する。
            assert!(!detail.is_empty());
            return;
        }
        Err(CudaError::Compile(detail)) => {
            // NVRTC は搭載されているが `<mma.h>` の include パス解決に失敗、
            // または対象 GPU が compute capability 8.0 未満で TF32 fragment
            // が NVRTC に受理されない環境。WMMA(TF32) カーネル固有の
            // 失敗経路であり、naive/tiled のコンパイルには影響しない
            // （`CudaGemm::new` は naive/tiled/WMMA(TF32) の 5 カーネルを
            // 一括コンパイルするため、いずれか 1 つの失敗で `Err` になる。
            // `gemm.rs::CudaGemm::new` 参照）。
            let _ = detail;
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaGemm::new: {other}"),
    };

    assert_wmma_tf32_parity(&gemm, "smoke 32x32x32", 1, 32, 32, 32);
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
