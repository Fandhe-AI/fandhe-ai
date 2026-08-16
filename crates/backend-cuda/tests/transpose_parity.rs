//! smem パディング + スウィズルによる転置カーネル（`CudaTranspose`）の
//! 環境適応型テスト＋実機必須テスト（イシュー #601・親 #582 の G-11）。
//!
//! `tests/gemm_naive.rs`/`tests/gemm_tiled.rs` と同じ構成方針（CUDA
//! 搭載・非搭載どちらの環境でも green になる入口テスト＋受け入れ条件その
//! ものである数値一致検証を `#[ignore]` で分離）を踏襲する
//! （`.claude/rules/coding-rust.md` の実機依存テスト分離規約）。
//!
//! 転置は演算を伴わない純置換（積和のような丸め誤差が入らない）ため、
//! CPU 参照実装との照合は `backend_cpu::parity::assert_parity`（複合誤差
//! 判定）ではなく **bit 完全一致**（`assert_eq!`）で行う（実装計画 6.2 節
//! 「転置は演算なしの純置換のため tolerance 不要。緩和なし」）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p backend-cuda --release --test transpose_parity -- --ignored --nocapture
//! ```

use backend_cuda::{CudaDevice, CudaError, CudaGemm, CudaTranspose};
use bench_harness::rng::Xorshift64Star;
use half::f16;

/// CPU 参照転置（f32）。`dst[col*m+row] = src[row*n+col]`。
fn cpu_transpose_f32(src: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            dst[col * m + row] = src[row * n + col];
        }
    }
    dst
}

/// CPU 参照転置（f16）。[`cpu_transpose_f32`] の f16 版。
fn cpu_transpose_f16(src: &[f16], m: usize, n: usize) -> Vec<f16> {
    let mut dst = vec![f16::ZERO; m * n];
    for row in 0..m {
        for col in 0..n {
            dst[col * m + row] = src[row * n + col];
        }
    }
    dst
}

/// `CudaTranspose::new` は CUDA 非搭載環境で panic せず型付きエラーを
/// 返す（`tests/gemm_naive.rs::new_does_not_panic_and_returns_typed_result`
/// と同じ分岐パターン）。
#[test]
fn new_compiles_transpose_kernels_or_returns_typed_error_without_panicking() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            assert!(!detail.is_empty(), "detail message must not be empty");
            return;
        }
        Err(CudaError::Driver(_)) => {
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    };

    match CudaTranspose::new(&device) {
        Ok(_transpose) => {
            // CUDA 搭載環境: naive/smem 6 カーネル + 融合転置カーネルの
            // コンパイルが成功した。
        }
        Err(CudaError::NvrtcUnavailable { detail }) => {
            assert!(!detail.is_empty());
        }
        Err(other) => panic!("unexpected CudaError variant from CudaTranspose::new: {other}"),
    }
}

/// 形状網羅（実装計画 6.2 節）: タイル倍数（64×64）・非倍数境界（33×65）・
/// 非正方（17×97）・極小（1×1・1×5・5×1）。
const SHAPES: &[(usize, usize)] = &[(64, 64), (33, 65), (17, 97), (1, 1), (1, 5), (5, 1)];

/// naive／smem（パディングのみ・パディング+スウィズル）転置（f32）が
/// いずれも CPU 参照転置と bit 完全一致することを、形状網羅で検証する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn transpose_f32_all_variants_bit_exact_match_cpu_reference() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let transpose = CudaTranspose::new(&device).expect("transpose kernel compilation must succeed");

    for &(m, n) in SHAPES {
        let mut rng = Xorshift64Star::new(0xABCD ^ (m as u64) ^ ((n as u64) << 16));
        let src: Vec<f32> = rng.fill_vec(m * n);
        let expected = cpu_transpose_f32(&src, m, n);

        let naive = transpose
            .run_naive_f32(&src, m as u32, n as u32)
            .expect("naive transpose must succeed on CUDA-equipped test runner");
        assert_eq!(naive, expected, "naive f32 mismatch at m={m} n={n}");

        let smem_pad = transpose
            .run_smem_f32(&src, m as u32, n as u32, false)
            .expect("smem(pad) transpose must succeed on CUDA-equipped test runner");
        assert_eq!(smem_pad, expected, "smem(pad) f32 mismatch at m={m} n={n}");

        let smem_swizzle = transpose
            .run_smem_f32(&src, m as u32, n as u32, true)
            .expect("smem(pad+swizzle) transpose must succeed on CUDA-equipped test runner");
        assert_eq!(
            smem_swizzle, expected,
            "smem(pad+swizzle) f32 mismatch at m={m} n={n}"
        );
    }
}

/// [`transpose_f32_all_variants_bit_exact_match_cpu_reference`] の f16 版。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn transpose_f16_all_variants_bit_exact_match_cpu_reference() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let transpose = CudaTranspose::new(&device).expect("transpose kernel compilation must succeed");

    for &(m, n) in SHAPES {
        let mut rng = Xorshift64Star::new(0x1234 ^ (m as u64) ^ ((n as u64) << 16));
        let src: Vec<f16> = rng.fill_vec_f16(m * n);
        let expected = cpu_transpose_f16(&src, m, n);

        let naive = transpose
            .run_naive_f16(&src, m as u32, n as u32)
            .expect("naive transpose must succeed on CUDA-equipped test runner");
        assert_eq!(naive, expected, "naive f16 mismatch at m={m} n={n}");

        let smem_pad = transpose
            .run_smem_f16(&src, m as u32, n as u32, false)
            .expect("smem(pad) transpose must succeed on CUDA-equipped test runner");
        assert_eq!(smem_pad, expected, "smem(pad) f16 mismatch at m={m} n={n}");

        let smem_swizzle = transpose
            .run_smem_f16(&src, m as u32, n as u32, true)
            .expect("smem(pad+swizzle) transpose must succeed on CUDA-equipped test runner");
        assert_eq!(
            smem_swizzle, expected,
            "smem(pad+swizzle) f16 mismatch at m={m} n={n}"
        );
    }
}

/// GEMM epilogue 融合転置（opt-in）が (a) `run_tiled_f32` 出力のホスト側
/// 転置と bit 完全一致し、(b) CPU 参照実装（`matmul_reference_fma` を
/// 転置したもの）とも一致することを検証する（実装計画 6.2 節「融合変種が
/// (a)...(b)...」）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn tiled_transposed_f32_matches_host_transposed_tiled_and_cpu_reference() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled kernel compilation must succeed");
    let transpose = CudaTranspose::new(&device).expect("transpose kernel compilation must succeed");

    // (4, 0, 4): k==0 の早期 return 分岐（`run_tiled_transposed_f32` が
    // `upload_f32` を回避し 0 バイトデバイス確保を driver に要求しない
    // ことの実機回帰。advisor 指摘）を経路として実際に踏む。
    for &(m, k, n) in &[
        (64usize, 64usize, 64usize),
        (33, 47, 65),
        (1usize, 1, 1),
        (4, 0, 4),
    ] {
        let mut rng_a = Xorshift64Star::new(0x5EED1 ^ (m as u64));
        let a: Vec<f32> = rng_a.fill_vec(m * k);
        let mut rng_b = Xorshift64Star::new(0x5EED2 ^ (n as u64));
        let b: Vec<f32> = rng_b.fill_vec(k * n);

        // (a) run_tiled_f32 の出力をホスト側で転置した結果と bit 完全一致。
        let c = gemm
            .run_tiled_f32(&a, &b, m as u32, n as u32, k as u32)
            .expect("run_tiled_f32 must succeed on CUDA-equipped test runner");
        let c_host_transposed = cpu_transpose_f32(&c, m, n);

        let c_fused = transpose
            .run_tiled_transposed_f32(&a, &b, m as u32, n as u32, k as u32)
            .expect("run_tiled_transposed_f32 must succeed on CUDA-equipped test runner");
        assert_eq!(
            c_fused, c_host_transposed,
            "fused epilogue transpose must bit-exact match host-transposed tiled output \
             (m={m}, n={n}, k={k})"
        );

        // (b) CPU 参照実装（matmul_reference_fma）を転置したものとも一致
        // （tiled カーネルのアキュムレーション自体が CPU FMA 契約と揃って
        // いることの回帰。既存 `tests/parity_nonregression.rs`/
        // `tests/cpu_cuda_parity.rs` と同じ判定式は使わず、転置の
        // bit 完全一致という本テスト固有の主張に閉じる）。
        let mut c_cpu = vec![0.0f32; m * n];
        backend_cpu::matmul_reference_fma(&a, &b, &mut c_cpu, m, n, k)
            .expect("matmul_reference_fma must succeed for valid shapes");
        let c_cpu_transposed = cpu_transpose_f32(&c_cpu, m, n);
        assert_eq!(
            c_fused, c_cpu_transposed,
            "fused epilogue transpose must bit-exact match CPU reference transpose \
             (m={m}, n={n}, k={k})"
        );
    }
}

/// `launch_tiled_transposed_f32`（safe な公開 API）の `k == 0` 分岐が、
/// 呼び出し元から渡された非ゼロ初期化バッファを明示的にゼロクリアする
/// ことを検証する（codex-review 指摘 P1・PR #690 レビュー回帰）。
///
/// `run_tiled_transposed_f32`（上記テスト）は自前でゼロ初期化済みの
/// バッファ（`alloc_output_f32`）を割り当てるため k==0 の欠陥を踏まない。
/// `launch_tiled_transposed_f32` は呼び出し元が確保・再利用したバッファを
/// そのまま受け取る契約のため、ここでは非ゼロ値で埋めたバッファを渡し、
/// カーネル起動を省略する k==0 経路でも `(A @ B)^T = 0` の数学的契約
/// （全 0）が保たれることを実機で確認する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn launch_tiled_transposed_f32_zero_clears_stale_buffer_when_k_is_zero() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let transpose = CudaTranspose::new(&device).expect("transpose kernel compilation must succeed");

    let (m, k, n) = (4u32, 0u32, 5u32);
    let a: Vec<f32> = Vec::new();
    let b: Vec<f32> = Vec::new();
    let a_dev = transpose
        .upload_f32(&a)
        .expect("upload_f32 must succeed for empty (k=0) slice");
    let b_dev = transpose
        .upload_f32(&b)
        .expect("upload_f32 must succeed for empty (k=0) slice");

    // c_t_dev（n x m）を非ゼロ値（stale なバッファ再利用を模す）で埋める。
    let stale = vec![1.0f32; (n as usize) * (m as usize)];
    let mut c_t_dev = transpose
        .upload_f32(&stale)
        .expect("upload_f32 must succeed for stale fill buffer");

    transpose
        .launch_tiled_transposed_f32(&a_dev, &b_dev, &mut c_t_dev, m, n, k)
        .expect("launch_tiled_transposed_f32 must succeed for k=0");

    let c_t = transpose
        .download_f32(&c_t_dev)
        .expect("download_f32 must succeed");
    assert_eq!(
        c_t,
        vec![0.0f32; (n as usize) * (m as usize)],
        "k==0 must zero-clear a caller-supplied non-zero c_t_dev buffer, not leave stale values"
    );
}

// 既存 `tests/parity_nonregression.rs` の非後退確認（§1.2 (2)）は
// 本ファイルではなく `tests/parity_nonregression.rs` 自身が担う
// （本イシューは既存カーネル・tolerance 定数を変更しないため、当該
// テストファイル自体には手を入れていない）。
