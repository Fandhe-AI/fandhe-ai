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

mod common;

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

/// 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須の厳密ゼロ
/// fail 判定テスト（イシュー #1106 案 A で対象形状を縮小）。
///
/// GB10 実機実測（2026-09-02 の診断ダンプ `wmma_tf32_parity_diagnostic_dump_issue_1106`。
/// 修正確定に伴い削除済み）により、`assert_parity`（厳密ゼロ fail 判定）
/// が実際に成立するのは 1×1×1（sub-K-tile。K 方向蓄積が発生せず TF32
/// 丸め誤差が蓄積しない）のみと判明した。他の 7 形状（32×32×32・
/// 64×64×64・128×128×128・512×512×512・64×96×128・17×23×19・33×31×65）は
/// `ParityBaseline::BASELINES`（`ParityPath::WmmaTf32`）へ実測値付きで
/// 移し、[`wmma_tf32_routed_path_baselines_do_not_regress`]（本ファイル
/// 下部。公開 API `run_wmma_tf32` 経由の非後退監視）が検査する。
/// tolerance 定数は変更していない（ユーザー承認 2026-09-02。詳細は
/// `docs/perf/cuda-parity-baseline.md` §10.5）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
fn wmma_tf32_matches_reference_across_shapes() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");

    // seed は 2005（元の 8 ケース配列での 1×1×1 の位置 idx=5 に対応する
    // 2000+5）を直接指定する。実測は seed=2005 のみで fail_count=0/1 を
    // 確認済み（他の未測定 seed への一般化はしない。`ParityBaseline`
    // ドキュメンテーションコメントと同じ「推定値の捏造をしない」方針）。
    assert_wmma_tf32_parity(&gemm, "shape m=1 n=1 k=1", 2005, 1, 1, 1);
}

/// [`wmma_tf32_matches_reference_across_shapes`] ドキュメンテーション
/// コメントで縮小した 7 形状・旧 `wmma_tf32_k4096_stress_poc_v2_5`
/// （両ケースとも非ゼロ fail のため全ケースがここへ移管された。削除済み）
/// の非後退監視。`ParityPath::WmmaTf32` 行を公開 API `run_wmma_tf32`
/// （3 段選択そのまま）経由で検査する——`common::parity_baseline::
/// parity_baselines_do_not_regress`（`parity_nonregression.rs`）は
/// `WmmaTf32`／`WmmaTf32Opt` 行を明示的に skip する（private 強制経路を
/// 使う lib 側検査〈`wmma_tf32_basic_kernel_parity_does_not_regress`〉と
/// 二重検査しないため）ため、公開 API 経由のルーティング検証はここでしか
/// 行われない。1 行の fail で残りの行の検査を打ち切らない（`parity_nonregression.rs::
/// parity_baselines_do_not_regress` と同じ集約方式）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
fn wmma_tf32_routed_path_baselines_do_not_regress() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");

    let mut failures: Vec<String> = Vec::new();
    for baseline in common::parity_baseline::BASELINES
        .iter()
        .filter(|b| b.path == common::parity_baseline::ParityPath::WmmaTf32)
    {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut rng = bench_harness::rng::Xorshift64Star::new(baseline.seed);
            let a = rng.fill_vec((baseline.m as usize) * (baseline.k as usize));
            let b = rng.fill_vec((baseline.k as usize) * (baseline.n as usize));

            let mut c_ref = vec![0.0f32; (baseline.m as usize) * (baseline.n as usize)];
            fandhe_ai_backend_cpu::matmul_reference_fma(
                &a,
                &b,
                &mut c_ref,
                baseline.m as usize,
                baseline.n as usize,
                baseline.k as usize,
            )
            .expect(
                "matmul_reference_fma shape validation must pass for well-formed baseline input",
            );

            let c_gpu = gemm
                .run_wmma_tf32(&a, &b, baseline.m, baseline.n, baseline.k)
                .expect(
                    "CudaGemm::run_wmma_tf32 must succeed on a compute capability >= 8.0 test \
                     runner",
                );

            let report = fandhe_ai_backend_cpu::compare(&c_gpu, &c_ref)
                .expect("shape must match baseline fixture");
            common::parity_baseline::assert_no_parity_regression(
                baseline.context,
                &report,
                baseline,
            );
        }));
        if let Err(err) = result {
            let msg = err
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panic (詳細不明)".to_string());
            failures.push(format!("{}: {msg}", baseline.context));
        }
    }

    assert!(
        failures.is_empty(),
        "parity 非後退契約 FAIL（{} 件）:\n{}",
        failures.len(),
        failures.join("\n")
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
