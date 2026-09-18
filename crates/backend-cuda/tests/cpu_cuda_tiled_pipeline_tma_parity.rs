//! TMA（cp.async.bulk.tensor）ロード経路 Stage 1（64×64 pipeline・
//! shared::cta。イシュー #1975）の CPU-CUDA 数値一致・既存 cp.async
//! pipeline 版との出力 bit 同一回帰テスト。
//!
//! `tests/cpu_cuda_tiled_pipeline_persistent_parity.rs` と同じ方針で、
//! 判定式・閾値は `fandhe_ai_backend_cpu::assert_parity`（統一複合判定
//! 「相対誤差 1e-3 未満または絶対誤差 1e-5 未満」の唯一の実体）に一本化
//! し、ここでローカル複製しない（`.claude/rules/coding-rust.md`）。
//!
//! **本番経路との関係**: `CudaGemm::run_tiled_pipeline_tma_f32` は本番
//! 既定経路（`run_tiled_f32`）を置き換えない選択可能な opt-in 変種で
//! あり、本ファイルの全テストは明示的にこの API を呼ぶ
//! （`kernels_tiled_pipeline.rs`「TMA」節冒頭コメント参照）。
//!
//! **AC（受け入れ基準本体）**: [`tiled_pipeline_tma_none_matches_pipeline_
//! bit_exact`] が、既存 cp.async pipeline 版（`CudaGemm::
//! run_tiled_pipeline_f32`）と TMA 版（`None` swizzle 腕。`CudaGemm::
//! run_tiled_pipeline_tma_f32`）の出力が**全形状で bit 同一**であること
//! を `assert_eq!` で検証する。`B64` swizzle 腕は仮説段階
//! （`kernels_tiled_pipeline.rs`「swizzle」節）のため、同一検証を行うが
//! 不一致を CI 失敗として扱わない別テスト
//! （[`tiled_pipeline_tma_b64_matches_pipeline_or_records_hypothesis_gap`]）
//! に分離する（実機での正式判定は #1976）。
//!
//! **実機依存の分離**: 環境適応スモークのみ通常 CI で実行、
//! CUDA/NVRTC 非搭載環境・compute capability 9.0 未満（TMA 前提）の
//! 環境では早期 return で green（`tests/cpu_cuda_tiled_pipeline_persistent_
//! parity.rs` と同じ分岐パターン）。

use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaGemm, TmaSwizzleA};

/// `Vec<f32>` の bit 完全一致を `to_bits()` 比較で判定する（codex-review
/// P2 是正・PR #2027）。`f32` の `PartialEq`（`assert_eq!`／`!=` が使う
/// 比較演算子）は `+0.0 == -0.0` を真とし、bit パターンの差を見逃す
/// ため、本ファイルが主張する「出力 bit 同一」契約の検証には
/// `to_bits()`（符号ビット込みの厳密な bit パターン比較。NaN の bit
/// パターンも区別する）を使う。
fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// タイル倍数形状・端数形状（cp.async 版と同じ整列条件〈`n % 4 == 0 &&
/// k % 4 == 0`〉を満たしつつブロックタイル・K タイル非倍数）を含む
/// （`tests/cpu_cuda_tiled_pipeline_persistent_parity.rs::
/// persistent_bit_exact_shapes` と同型。TMA の部分 OOB box 検証を兼ねる
/// ため、端数形状〈`(60, 68, 36)` 等〉を必ず含める。設計 doc §3.3）。
fn tma_bit_exact_shapes() -> Vec<(u32, u32, u32)> {
    vec![
        (64, 64, 64),
        (128, 64, 64),
        (1024, 1024, 1024),
        (256, 256, 4096),
        // 端数タイル（部分 OOB box の検出器。設計 doc §4.3・§8）。
        (60, 68, 36),
        (544, 256, 2048),
        (4100, 1028, 64),
    ]
}

/// [`tma_bit_exact_shapes`] の全形状で `run_tiled_pipeline_f32`
/// （既存 cp.async pipeline 版）と `run_tiled_pipeline_tma_f32`
/// （TMA 版・`None` swizzle 腕）の出力が bit 同一であること、および CPU
/// 参照実装との複合判定を検証する（AC 本体）。
#[test]
#[ignore = "CUDA 実機（compute capability 9.0 以降、TMA 対応）必須"]
fn tiled_pipeline_tma_none_matches_pipeline_bit_exact() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    assert!(
        gemm.tiled_pipeline_available(),
        "cp.async tiled pipeline kernel must be available on this ignored test runner \
         (reason: {:?})",
        gemm.tiled_pipeline_unavailable_reason()
    );

    let func = CudaGemm::compile_tiled_pipeline_tma_variant(&device, TmaSwizzleA::None)
        .expect("compile_tiled_pipeline_tma_variant(None) must succeed on a TMA-capable runner");

    for (seed, (m, n, k)) in tma_bit_exact_shapes().into_iter().enumerate() {
        let mut rng = bench_harness::rng::Xorshift64Star::new(seed as u64 + 1);
        let a = rng.fill_vec((m as usize) * (k as usize));
        let b = rng.fill_vec((k as usize) * (n as usize));

        let c_pipeline = gemm
            .run_tiled_pipeline_f32(&a, &b, m, n, k)
            .unwrap_or_else(|e| {
                panic!("run_tiled_pipeline_f32 must succeed for m={m},n={n},k={k}: {e}")
            });
        let c_tma = gemm
            .run_tiled_pipeline_tma_f32(&func, &a, &b, m, n, k)
            .unwrap_or_else(|e| {
                panic!("run_tiled_pipeline_tma_f32(None) must succeed for m={m},n={n},k={k}: {e}")
            });

        assert!(
            bits_eq(&c_tma, &c_pipeline),
            "TMA 版（None swizzle）と cp.async pipeline 版の出力が bit 同一ではありません \
             (m={m}, n={n}, k={k})"
        );

        let mut c_ref = vec![0.0f32; (m as usize) * (n as usize)];
        fandhe_ai_backend_cpu::matmul_reference_fma(
            &a, &b, &mut c_ref, m as usize, n as usize, k as usize,
        )
        .expect("matmul_reference_fma shape validation must pass for well-formed test input");
        fandhe_ai_backend_cpu::assert_parity(
            &format!("TMA tiled pipeline (None) m={m} n={n} k={k}"),
            &c_tma,
            &c_ref,
        );

        // `k == 0` は `tiled_pipeline_tma_k_zero_produces_all_zero_output`
        // が別途検証する（`run_tiled_pipeline_tma_f32` はホスト側で早期
        // return する契約。`tma_bit_exact_shapes` は k>0 の形状のみ）。
    }
}

/// [`tma_bit_exact_shapes`] の全形状で、一括 API
/// `CudaGemm::launch_tiled_pipeline_tma_f32`（毎起動でテンソルマップを
/// 内部 encode する）と、イシュー #1976 で追加した分離 API
/// `CudaGemm::prepare_tiled_pipeline_tma_maps` +
/// `CudaGemm::launch_tiled_pipeline_tma_f32_prepared`（事前 encode 済み
/// テンソルマップを使い回す）の出力が bit 同一であることを検証する
/// （`launch_tiled_pipeline_tma_f32` は分離 API への委譲として実装され
/// ているため挙動は完全に同一のはずだが、委譲の正しさを実機で直接
/// 検証する。`m == 0 || n == 0`／`k == 0` の早期 return 経路〈`NoOp`／
/// `ZeroK`〉も [`tma_bit_exact_shapes`] の端数形状群と
/// [`tiled_pipeline_tma_k_zero_produces_all_zero_output`] がそれぞれ
/// 別途カバーするためここでは通常経路〈`Ready`〉のみ確認する）。
#[test]
#[ignore = "CUDA 実機（compute capability 9.0 以降、TMA 対応）必須"]
fn tiled_pipeline_tma_prepared_matches_one_shot_launch_bit_exact() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    let func = CudaGemm::compile_tiled_pipeline_tma_variant(&device, TmaSwizzleA::None)
        .expect("compile_tiled_pipeline_tma_variant(None) must succeed on a TMA-capable runner");

    for (seed, (m, n, k)) in tma_bit_exact_shapes().into_iter().enumerate() {
        let mut rng = bench_harness::rng::Xorshift64Star::new(seed as u64 + 1001);
        let a = rng.fill_vec((m as usize) * (k as usize));
        let b = rng.fill_vec((k as usize) * (n as usize));

        // 一括 API（毎起動で encode）。
        let c_one_shot = gemm
            .run_tiled_pipeline_tma_f32(&func, &a, &b, m, n, k)
            .unwrap_or_else(|e| {
                panic!("run_tiled_pipeline_tma_f32(None) must succeed for m={m},n={n},k={k}: {e}")
            });

        // 分離 API（事前 encode 1 回 + 起動）。
        let (a_dev, b_dev) = gemm
            .upload_f32(&a, &b)
            .unwrap_or_else(|e| panic!("upload_f32 must succeed for m={m},n={n},k={k}: {e}"));
        let mut c_dev = gemm
            .alloc_output_f32(m, n)
            .unwrap_or_else(|e| panic!("alloc_output_f32 must succeed for m={m},n={n},k={k}: {e}"));
        let maps = gemm
            .prepare_tiled_pipeline_tma_maps(&func, &a_dev, &b_dev, (m, n, k))
            .unwrap_or_else(|e| {
                panic!("prepare_tiled_pipeline_tma_maps must succeed for m={m},n={n},k={k}: {e}")
            });
        gemm.launch_tiled_pipeline_tma_f32_prepared(&func, &maps, &mut c_dev)
            .unwrap_or_else(|e| {
                panic!(
                    "launch_tiled_pipeline_tma_f32_prepared must succeed for m={m},n={n},k={k}: \
                     {e}"
                )
            });
        let c_prepared = gemm
            .download_f32(&c_dev)
            .unwrap_or_else(|e| panic!("download_f32 must succeed for m={m},n={n},k={k}: {e}"));

        assert!(
            bits_eq(&c_prepared, &c_one_shot),
            "一括 API と分離 API（prepare + launch_prepared）の出力が bit 同一ではありません \
             (m={m}, n={n}, k={k})"
        );
    }
}

/// `k == 0` の出力が全ゼロであることを検証する（設計計画 Step 3 (e)。
/// `run_tiled_pipeline_tma_f32`／`launch_tiled_pipeline_tma_f32` は
/// `m == 0 || n == 0` の直後で `k == 0` を早期 return し、`c_dev` を
/// 明示的にゼロ化する契約〈テンソルマップ構築がゼロ次元を扱えないため。
/// `gemm.rs::tma_tiled_pipeline::CudaGemm::launch_tiled_pipeline_tma_f32`
/// ドキュメンテーションコメント参照〉。
#[test]
#[ignore = "CUDA 実機（compute capability 9.0 以降、TMA 対応）必須"]
fn tiled_pipeline_tma_k_zero_produces_all_zero_output() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    let func = CudaGemm::compile_tiled_pipeline_tma_variant(&device, TmaSwizzleA::None)
        .expect("compile_tiled_pipeline_tma_variant(None) must succeed on a TMA-capable runner");

    let (m, n, k) = (64u32, 64u32, 0u32);
    let a: Vec<f32> = Vec::new();
    let b: Vec<f32> = Vec::new();
    let c = gemm
        .run_tiled_pipeline_tma_f32(&func, &a, &b, m, n, k)
        .expect("run_tiled_pipeline_tma_f32(None) must succeed for k=0");
    assert!(c.iter().all(|&v| v == 0.0), "k=0 の出力は全ゼロのはずです");
}

/// 同一入力に対する TMA 版（`None` swizzle 腕）の連続 2 回起動が同一
/// 出力を返す（決定性契約。`tiled_pipeline_persistent_repeated_launch_
/// is_deterministic` と同型）ことを検証する。
#[test]
#[ignore = "CUDA 実機（compute capability 9.0 以降、TMA 対応）必須"]
fn tiled_pipeline_tma_none_repeated_launch_is_deterministic() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    let func = CudaGemm::compile_tiled_pipeline_tma_variant(&device, TmaSwizzleA::None)
        .expect("compile_tiled_pipeline_tma_variant(None) must succeed on a TMA-capable runner");

    let mut rng = bench_harness::rng::Xorshift64Star::new(7);
    let (m, n, k) = (512u32, 512u32, 512u32);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    let first = gemm
        .run_tiled_pipeline_tma_f32(&func, &a, &b, m, n, k)
        .expect("first run_tiled_pipeline_tma_f32(None) call must succeed");
    let second = gemm
        .run_tiled_pipeline_tma_f32(&func, &a, &b, m, n, k)
        .expect("second run_tiled_pipeline_tma_f32(None) call must succeed");

    assert!(
        bits_eq(&first, &second),
        "TMA 版（None swizzle）の連続 2 回起動（同一ハンドル・同一入力）は同一出力になるはず"
    );
}

/// 事前転置したホスト入力を NN として渡す 1 形状（設計 doc §3.5。TMA
/// カーネルは NN 専用であり、転置は上流〈#1214 の GPU 側 smem 転置
/// カーネル方式〉で処理される想定のため、本テストは「事前転置済み
/// 入力を NN として渡した場合に cp.async pipeline 版と bit 同一になる」
/// という API レベルの契約のみを検証する）。
#[test]
#[ignore = "CUDA 実機（compute capability 9.0 以降、TMA 対応）必須"]
fn tiled_pipeline_tma_none_matches_pipeline_with_pretransposed_host_input() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    let func = CudaGemm::compile_tiled_pipeline_tma_variant(&device, TmaSwizzleA::None)
        .expect("compile_tiled_pipeline_tma_variant(None) must succeed on a TMA-capable runner");

    let (m, n, k) = (128u32, 96u32, 64u32);
    let mut rng = bench_harness::rng::Xorshift64Star::new(31);
    // A を [k, m]（転置形状）として生成し、ホスト側で [m, k] へ転置して
    // から NN 入力として渡す（`crate::pool` 等が持つ GPU 側転置カーネル
    // の代わりに、ホスト側転置で「NN として渡す」契約のみを検証する）。
    let a_t = rng.fill_vec((k as usize) * (m as usize));
    let mut a = vec![0.0f32; (m as usize) * (k as usize)];
    for row in 0..m as usize {
        for col in 0..k as usize {
            a[row * k as usize + col] = a_t[col * m as usize + row];
        }
    }
    let b = rng.fill_vec((k as usize) * (n as usize));

    let c_pipeline = gemm
        .run_tiled_pipeline_f32(&a, &b, m, n, k)
        .expect("run_tiled_pipeline_f32 must succeed for the pretransposed-input shape");
    let c_tma = gemm
        .run_tiled_pipeline_tma_f32(&func, &a, &b, m, n, k)
        .expect("run_tiled_pipeline_tma_f32(None) must succeed for the pretransposed-input shape");

    assert!(
        bits_eq(&c_tma, &c_pipeline),
        "事前転置済みホスト入力を NN として渡した場合の TMA 版・cp.async pipeline 版の \
         出力が bit 同一ではありません"
    );
}

/// `B64`（仮説段階）swizzle 腕の出力を cp.async pipeline 版・CPU 参照
/// 実装と突き合わせる。物理配置の仮説（`kernels_tiled_pipeline.rs`
/// 「swizzle」節）が実機で成立するかどうかは #1976 の意味論プローブが
/// 確定するため、本テストは不一致を記録するのみで CI 失敗にはしない
/// （`assert_eq!`／`panic!` を使わず `eprintln!` で結果を出力する）。
#[test]
#[ignore = "CUDA 実機（compute capability 9.0 以降、TMA 対応）必須"]
fn tiled_pipeline_tma_b64_matches_pipeline_or_records_hypothesis_gap() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("tiled pipeline kernel compilation must succeed");
    let func = CudaGemm::compile_tiled_pipeline_tma_variant(&device, TmaSwizzleA::B64)
        .expect("compile_tiled_pipeline_tma_variant(B64) must succeed on a TMA-capable runner");

    let mut mismatched_shapes: Vec<(u32, u32, u32)> = Vec::new();
    for (seed, (m, n, k)) in tma_bit_exact_shapes().into_iter().enumerate() {
        let mut rng = bench_harness::rng::Xorshift64Star::new(seed as u64 + 101);
        let a = rng.fill_vec((m as usize) * (k as usize));
        let b = rng.fill_vec((k as usize) * (n as usize));

        let c_pipeline = gemm
            .run_tiled_pipeline_f32(&a, &b, m, n, k)
            .unwrap_or_else(|e| {
                panic!("run_tiled_pipeline_f32 must succeed for m={m},n={n},k={k}: {e}")
            });
        let c_tma = gemm
            .run_tiled_pipeline_tma_f32(&func, &a, &b, m, n, k)
            .unwrap_or_else(|e| {
                panic!("run_tiled_pipeline_tma_f32(B64) must succeed for m={m},n={n},k={k}: {e}")
            });

        if !bits_eq(&c_tma, &c_pipeline) {
            mismatched_shapes.push((m, n, k));
        }
    }

    if mismatched_shapes.is_empty() {
        println!(
            "B64 swizzle 仮説（kernels_tiled_pipeline.rs::tma_swizzled_chunk_a 相当）は \
             全 {} 形状で cp.async pipeline 版と bit 同一でした",
            tma_bit_exact_shapes().len()
        );
    } else {
        eprintln!(
            "B64 swizzle 仮説は {}/{} 形状で cp.async pipeline 版と不一致でした（実機での \
             smem 物理配置確定はイシュー #1976 の意味論プローブへ引き継ぐ）: {mismatched_shapes:?}",
            mismatched_shapes.len(),
            tma_bit_exact_shapes().len()
        );
    }
}

/// 環境適応型のスモークテスト（`#[ignore]` なし。通常 CI で実行）。
/// CUDA/NVRTC 非搭載環境・compute capability 9.0 未満（TMA 前提）の
/// 環境では早期 return で green
/// （`tiled_pipeline_persistent_parity_smoke_env_adaptive` と同じ分岐
/// パターン）。
#[test]
fn tiled_pipeline_tma_parity_smoke_env_adaptive() {
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
            assert!(!detail.is_empty());
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaGemm::new: {other}"),
    };

    if !gemm.tiled_pipeline_available() {
        // cp.async は sm_80 (Ampere) 以降限定。TMA 版は cp.async pipeline
        // 版との比較を行うため、比較対象自体が使えない環境では早期
        // return する。
        return;
    }

    let func = match CudaGemm::compile_tiled_pipeline_tma_variant(&device, TmaSwizzleA::None) {
        Ok(func) => func,
        Err(CudaError::TiledPipelineUnavailable { detail }) => {
            // compute capability 9.0 未満（TMA 前提。`kernels_tiled_
            // pipeline.rs`「TMA」節参照）。
            assert!(!detail.is_empty());
            return;
        }
        Err(CudaError::NvrtcUnavailable { detail }) => {
            assert!(!detail.is_empty());
            return;
        }
        Err(other) => {
            panic!("unexpected CudaError variant from compile_tiled_pipeline_tma_variant: {other}")
        }
    };

    let mut rng = bench_harness::rng::Xorshift64Star::new(1);
    let (m, n, k) = (128u32, 64u32, 64u32);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    let c_pipeline = gemm
        .run_tiled_pipeline_f32(&a, &b, m, n, k)
        .expect("CudaGemm::run_tiled_pipeline_f32 must succeed on a TMA-capable test runner");
    let c_tma = gemm
        .run_tiled_pipeline_tma_f32(&func, &a, &b, m, n, k)
        .expect(
            "CudaGemm::run_tiled_pipeline_tma_f32(None) must succeed on a TMA-capable test \
             runner",
        );
    assert!(
        bits_eq(&c_tma, &c_pipeline),
        "smoke 128x64x64: TMA 版（None swizzle）と cp.async pipeline 版の出力が bit 同一では \
         ありません"
    );
}
