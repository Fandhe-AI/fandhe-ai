//! shape 特化カーネル（C-6・#516・PR #643／C-7・#519・PR #674）の数値一致
//! 回帰テスト（イシュー #531）。
//!
//! `gemm_auto::run_specialized_mma_f16`／`SpecializedMmaKernelHandle`
//! （crate root 再エクスポート。テスト・ベンチ専用のため `internal-diagnostics`
//! feature〈既定 off〉でゲートされている。本テストは `Cargo.toml` の
//! `[[test]]` セクションで `required-features = ["internal-diagnostics"]`
//! を指定し、`cargo test --all-features`〈CI の test ジョブ・`make test`
//! が使うコマンド〉でのみビルド・実行される。PR #685 codex-review P1
//! 指摘の是正）を通じて `CompiledDims::{DYNAMIC_ALL,
//! STATIC_NK, STATIC_MNK}` の 3 プリセットを実際に NVRTC コンパイル・
//! 起動し、以下を検証する:
//!
//! 1. **既定カーネルとの bit 一致**（全形状 × 全プリセット）: 特化は
//!    `#define` によるコンパイル時定数の焼き込みのみで演算命令列・
//!    アキュムレート順序を変えない（`kernels_mma.rs` 冒頭コメント・
//!    PR #643 の契約）ため、`CudaMmaGemm::run_f16`（既定・全 Dynamic）
//!    の出力と bit 一致する。`gemm_mma.rs::tests::
//!    mma_f16_stage_count_does_not_change_bit_exact_output` と同じ論法
//!    （同一バックエンド内の実装詳細比較のため tolerance の対象外。
//!    `.claude/rules/coding-rust.md` の「バックエンド間数値一致テストの
//!    許容誤差を単独で緩和しない」契約に抵触しない）。
//! 2. **CPU 参照との判定**（4096³・(512,64,4096) を除く形状。spec REQ-2
//!    「2026-09-02 追記」の形状別判定方式・イシュー #1161）:
//!    `cpu_cuda_mma_parity.rs::assert_mma_f16_parity` と同一手順
//!    （f16→f32→`fandhe_ai_backend_cpu::matmul_reference_fma`→f16 丸め→f32）で
//!    値を算出したうえで、判定方式を形状ごとに分岐する。GB10 実機 sweep
//!    （イシュー #1159・PR #1181・`docs/perf/logs/
//!    specialized-mma-f16-sweep-1159/`・2 回実行で全値完全一致）の結果、
//!    CPU 参照検査対象 8 形状 × 3 プリセットのうち `(256, 512, 1024)` の
//!    3 プリセットのみ恒常非ゼロ fail（`fail_count=30/131072`）を示した。
//!    厳密ゼロ fail 判定が成立する 7 形状（`(64,128,32)`・`(128,256,128)`・
//!    `(40,24,72)`・`(65,136,40)`・`(63,120,24)`・`(200,264,104)`・
//!    `(1,136,40)`）は引き続き `fandhe_ai_backend_cpu::assert_parity`（厳密
//!    ゼロ fail 判定）を維持し、`(256,512,1024)` のみ
//!    `common::parity_baseline::assert_no_parity_regression`（実測 baseline
//!    非後退方式・`ParityPath::SpecializedMmaF16`）へ切り替える
//!    （`WmmaTf32Opt`〈#1106・PR #1124〉・`MmaTf32VsWmmaStaged`〈#1122・
//!    PR #1133〉と同型の形状別分岐パターン）。tolerance 定数
//!    （`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`）・カーネル実装は
//!    一切変更しない。
//! 3. **fail-closed 検査**（負系）: `STATIC_MNK` でコンパイルしたカーネル
//!    を不一致形状で起動すると `CudaError::InvalidKernelConfig` になる
//!    こと（`validate_launch_shape` の実効性）。
//! 4. **`STATIC_NK` 動的次元再利用**: 同一コンパイル済みカーネル
//!    （`SpecializedMmaKernelHandle`）を N/K 固定・M 可変で複数回起動し、
//!    いずれも parity を満たすこと。
//! 5. **プロセス内 LRU カーネルモジュールキャッシュ**（イシュー #511・
//!    C-4）: `SpecializedMmaKernelHandle::compile`（内部で
//!    `RenderedMmaKernel::compile` を呼ぶ）を同一形状・同一 `CompiledDims`
//!    で 2 回実行すると、2 回目はプロセス内 LRU をヒットし
//!    （`fandhe_ai_backend_cuda::diagnostics::module_cache_hit_count` の増加で観測）、
//!    かつキャッシュ経由でロードしたカーネルの実行結果が非キャッシュ時
//!    （既定カーネル）と bit 一致すること（キャッシュがソース・数値経路
//!    に影響しないことの回帰）。
//!
//! 環境適応スモークのみ通常 CI で実行し（CUDA/NVRTC 非搭載・cc<8.0 の
//! いずれの環境でも早期 return で green。`cpu_cuda_mma_parity.rs` と
//! 同じ分岐パターン）、形状網羅・負系・再利用検証は `#[ignore]` で分離
//! する（`.claude/rules/coding-rust.md`「実機依存テストは `#[ignore]`
//! で分離」）。

mod common;

use common::parity_baseline::{BASELINES, ParityPath, assert_no_parity_regression};
use fandhe_ai_backend_cuda::{
    CompiledDims, CudaDevice, CudaError, CudaMmaGemm, SpecializedMmaKernelHandle,
    run_specialized_mma_f16,
};
use fandhe_ai_tensor_core::dispatch::GemmShape;
use half::f16;

/// 決定的シードで A・B（f16）を生成する（`cpu_cuda_mma_parity.rs` と
/// 同じ生成方法。呼び出し元が参照値・特化カーネル出力の双方に同一入力
/// を使うためのヘルパー）。
fn gen_ab(seed: u64, m: u32, n: u32, k: u32) -> (Vec<f16>, Vec<f16>) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
    let b: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));
    (a, b)
}

/// f16→f32→`fandhe_ai_backend_cpu::matmul_reference_fma`→f16 丸め→f32 で CPU 参照値
/// を得る（`cpu_cuda_mma_parity.rs::assert_mma_f16_parity` と同一手順。
/// FMA 契約は `fandhe_ai_backend_cpu::matmul_reference_fma`〈`f32::mul_add`〉を
/// 再利用し複製しない）。
fn cpu_reference_f32(a: &[f16], b: &[f16], m: u32, n: u32, k: u32) -> Vec<f32> {
    let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
    let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();
    let mut c_ref_f32 = vec![0.0f32; (m as usize) * (n as usize)];
    fandhe_ai_backend_cpu::matmul_reference_fma(
        &a_f32,
        &b_f32,
        &mut c_ref_f32,
        m as usize,
        n as usize,
        k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed test input");
    c_ref_f32
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect()
}

/// 環境適応型のスモークテスト（`#[ignore]` なし。通常 CI で実行）。
///
/// CUDA 非搭載・NVRTC 非搭載・cc<8.0 のいずれの環境でも早期 return し
/// green とする（`cpu_cuda_mma_parity.rs::mma_f16_parity_smoke_env_adaptive`
/// と同じ分岐パターン）。`CudaMmaGemm::new` を先に経由することで
/// NVRTC・cc ゲートの判定を単一の真実源に委ね、本テスト自身は複製しない。
/// CUDA+toolkit+cc>=8.0 環境でのみ `STATIC_NK` 特化カーネルが既定経路と
/// bit 一致することを最小形状（16x8x16）で確認する。
#[test]
fn specialized_mma_f16_matches_default_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            assert!(!detail.is_empty(), "detail message must not be empty");
            return;
        }
        Err(CudaError::Driver(_)) => return,
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    };

    let default_gemm = match CudaMmaGemm::new(&device) {
        Ok(gemm) => gemm,
        Err(CudaError::NvrtcUnavailable { detail }) => {
            assert!(!detail.is_empty());
            return;
        }
        Err(CudaError::TensorCoreUnsupported { detail }) => {
            assert!(!detail.is_empty());
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaMmaGemm::new: {other}"),
    };

    let (m, n, k) = (16u32, 8u32, 16u32);
    let (a, b) = gen_ab(1, m, n, k);

    let default_c = default_gemm
        .run_f16(&a, &b, m, n, k)
        .expect("CudaMmaGemm::run_f16 must succeed on CUDA-equipped test runner");
    let specialized_c = run_specialized_mma_f16(&device, CompiledDims::STATIC_NK, &a, &b, m, n, k)
        .expect("run_specialized_mma_f16 must succeed on CUDA-equipped test runner");

    assert_eq!(
        default_c, specialized_c,
        "STATIC_NK 特化カーネルの出力が既定カーネルと bit 一致しません"
    );
}

/// 環境適応型のスモークテスト（`#[ignore]` なし。通常 CI で実行）:
/// `SpecializedMmaKernelHandle::compile` を `CudaMmaGemm::new` を経由せず
/// **直接**呼び出す負系回帰（PR #685 codex-review P2 指摘への対応）。
///
/// 上の `specialized_mma_f16_matches_default_smoke_env_adaptive` は
/// `CudaMmaGemm::new`（`gemm_mma.rs`）を先に経由するため、
/// `SpecializedMmaKernelHandle::compile` 自身が cc ゲート
/// （`check_min_compute_capability`）を NVRTC コンパイルより前に呼ぶ
/// ことを検出できない（`CudaMmaGemm::new` 側のゲートが先に働き
/// `TensorCoreUnsupported` を返すため、`compile` 側のゲートが仮に
/// 欠落・後退しても本テストの `CudaMmaGemm::new` 呼び出し部分では
/// 気付けない）。本テストは `compile` を直接呼ぶことでこの検出不能性
/// を解消する。
///
/// CUDA 非搭載・NVRTC 非搭載・cc<8.0 のいずれの環境でも早期 return し
/// green とする（本ファイル冒頭の環境適応スモークと同じ分岐パターン）。
/// cc<8.0 環境（`CudaMmaGemm::new` と同じ `MIN_COMPUTE_CAPABILITY_MAJOR`
/// ゲートを共有する `gemm_mma.rs::check_min_compute_capability`）では
/// `CudaError::TensorCoreUnsupported` が返ることを検証し（本テストの
/// 主目的）、CUDA+NVRTC+cc>=8.0 環境では `compile` 自体が成功すること
/// （過剰拒否していないことの健全性確認）を確認する。
#[test]
fn specialized_mma_kernel_handle_compile_direct_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            assert!(!detail.is_empty(), "detail message must not be empty");
            return;
        }
        Err(CudaError::Driver(_)) => return,
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    };

    // `CudaMmaGemm::new` を経由せず `SpecializedMmaKernelHandle::compile`
    // を直接呼ぶ（本テストの主張の核）。`compiled` プリセットは
    // `DYNAMIC_ALL`（既定カーネルと同一次元構成）を使う。
    match SpecializedMmaKernelHandle::compile(
        &device,
        GemmShape::new(16, 8, 16),
        CompiledDims::DYNAMIC_ALL,
    ) {
        Ok(_handle) => {
            // CUDA+NVRTC+cc>=8.0 環境: `compile` が過剰拒否していないことの
            // 健全性確認（成功のみを主張し、以降の launch_f16 検証は
            // `specialized_mma_f16_matches_default_smoke_env_adaptive` 等が
            // 別途カバーする）。
        }
        Err(CudaError::NvrtcUnavailable { detail }) => {
            assert!(!detail.is_empty());
        }
        Err(CudaError::TensorCoreUnsupported { detail }) => {
            // cc < 8.0 の実機（`mma.sync`/`ldmatrix`/`cp.async` 非対応）。
            // `SpecializedMmaKernelHandle::compile` 自身が
            // `check_min_compute_capability` を NVRTC コンパイル前に
            // 呼んで拒否した、という本テストが検証したい負系そのもの。
            assert!(!detail.is_empty());
            assert!(
                detail.contains("compute capability"),
                "TensorCoreUnsupported detail must mention compute capability: {detail}"
            );
        }
        Err(other) => {
            panic!("unexpected CudaError variant from SpecializedMmaKernelHandle::compile: {other}")
        }
    }
}

/// 実機（compute capability 8.0 以上・NVRTC 搭載）必須の形状網羅テスト。
/// 受け入れ条件の本体（#531）。
///
/// M=N=K=4096 を含み、非正方・非（タイル）倍数形状を含む 10 形状 ×
/// `CompiledDims::{DYNAMIC_ALL, STATIC_NK, STATIC_MNK}` の 3 プリセット
/// を検査する（イシュー #531 実装計画 §3.3 形状マトリクス）。
/// `MMA_BM=64`/`MMA_BN=128`/`MMA_BK=32`（#494 時点）に対する非倍数・
/// 端タイル形状を含み、shape 特化でコンパイル時に境界が自明になっても
/// カーネル側の手動境界チェック（REQ-8）が実際に踏まれることの回帰
/// 対象とする。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須"]
fn specialized_mma_f16_matches_default_and_reference_across_shapes() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let default_gemm = CudaMmaGemm::new(&device).expect("mma kernel compilation must succeed");

    // (m, n, k, CPU 参照も検査するか)。4096³・(512,64,4096) は CPU
    // 参照実装が現実的でないため bit 一致検査のみとする（本ファイル
    // 冒頭コメント「1」参照）。
    let cases: &[(u32, u32, u32, bool)] = &[
        (4096, 4096, 4096, false), // 受入基準必須。大形状・K 周回ストレス
        (64, 128, 32, true),       // 1 ブロックタイルちょうど（コントロール）
        (128, 256, 128, true),     // 複数タイル倍数（コントロール）
        (256, 512, 1024, true),    // 非正方・タイル倍数
        (40, 24, 72, true),        // サブタイル・非倍数（#187 形状の踏襲）
        (65, 136, 40, true),       // 全次元タイル+端数（M=BM+1・N=BN+8・K=BK+8）
        (63, 120, 24, true),       // 全次元タイル未満の非倍数
        (200, 264, 104, true),     // 複数タイル+全次元端数
        (1, 136, 40, true),        // m=1 極小行（guarded store 境界）
        (512, 64, 4096, false),    // 非正方・K 大
    ];

    for (idx, &(m, n, k, check_cpu_reference)) in cases.iter().enumerate() {
        let seed = 4000 + idx as u64;
        let (a, b) = gen_ab(seed, m, n, k);

        let default_c = default_gemm.run_f16(&a, &b, m, n, k).unwrap_or_else(|err| {
            panic!("shape (m={m}, n={n}, k={k}): default run_f16 failed: {err}")
        });

        // `CompiledDims` の `Debug` はプリセット名（`DYNAMIC_ALL` 等）を
        // 出力しない（`CompiledDims { m: false, … }` 形式）ため、baseline
        // fixture の `context` 文字列検索キー（下記 §「(256,512,1024) の
        // baseline 検索」参照）用にラベルをローカルで対応付ける。
        for (compiled, label) in [
            (CompiledDims::DYNAMIC_ALL, "DYNAMIC_ALL"),
            (CompiledDims::STATIC_NK, "STATIC_NK"),
            (CompiledDims::STATIC_MNK, "STATIC_MNK"),
        ] {
            let specialized_c = run_specialized_mma_f16(&device, compiled, &a, &b, m, n, k)
                .unwrap_or_else(|err| {
                    panic!(
                        "shape (m={m}, n={n}, k={k}) compiled={compiled:?}: \
                         run_specialized_mma_f16 failed: {err}"
                    )
                });

            assert_eq!(
                default_c, specialized_c,
                "shape (m={m}, n={n}, k={k}) compiled={compiled:?}: 特化カーネルの出力が \
                 既定カーネルと bit 一致しません（特化はコンパイル時定数の焼き込みのみで \
                 演算命令列・アキュムレート順序を変えない契約に違反していないか確認する）"
            );

            if check_cpu_reference {
                let c_ref_rounded = cpu_reference_f32(&a, &b, m, n, k);
                let c_specialized_f32: Vec<f32> =
                    specialized_c.iter().map(|x| x.to_f32()).collect();

                // (256, 512, 1024) の baseline 検索: #1159 sweep で他 7
                // 形状はゼロ fail 成立を実機確認済みだが、本形状のみ
                // 恒常非ゼロ fail（TF32/f16 丸め由来の推定。ファイル冒頭
                // コメント「2」参照）のため非後退方式（spec REQ-2
                // 「2026-09-02 追記」項目 1・2）で検査する。
                if (m, n, k) == (256, 512, 1024) {
                    let baseline = BASELINES
                        .iter()
                        .find(|b| {
                            b.path == ParityPath::SpecializedMmaF16
                                && (b.m, b.n, b.k) == (m, n, k)
                                && b.context.contains(label)
                        })
                        .unwrap_or_else(|| {
                            panic!(
                                "SpecializedMmaF16 の 256x512x1024 {label} baseline 行が \
                                 存在しません"
                            )
                        });
                    assert_eq!(
                        baseline.seed, seed,
                        "baseline の seed がテストループの seed と一致しません"
                    );
                    let report = fandhe_ai_backend_cpu::compare(&c_specialized_f32, &c_ref_rounded)
                        .expect("shape must match baseline fixture");
                    assert_no_parity_regression(baseline.context, &report, baseline);
                } else {
                    fandhe_ai_backend_cpu::assert_parity(
                        &format!(
                            "specialized mma_f16 compiled={compiled:?} shape m={m} n={n} k={k}"
                        ),
                        &c_specialized_f32,
                        &c_ref_rounded,
                    );
                }
            }
        }
    }
}

/// fail-closed 負系（実機必須）: `STATIC_MNK` でコンパイルしたカーネル
/// を不一致形状で起動すると `CudaError::InvalidKernelConfig` になる
/// こと（`MmaKernelConfig::validate_launch_shape` の実効性。イシュー
/// #531 実装計画 §3.2 点 3）を実機で確認する。
///
/// `kernels_mma.rs::tests::mma_kernel_config_validate_launch_shape_rejects_k_mismatch`
/// 等が同じ契約を実機非依存の単体テストとして既に検査しているが、本
/// テストは `SpecializedMmaKernelHandle::compile`→`launch_f16` という
/// 実際の公開経路（NVRTC コンパイル・実起動）を通じて同じ拒否が機能
/// することを end-to-end で確認する。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須"]
fn specialized_mma_f16_static_mnk_rejects_mismatched_launch_shape() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");

    let (compiled_m, compiled_n, compiled_k) = (65u32, 136u32, 40u32);
    let handle = SpecializedMmaKernelHandle::compile(
        &device,
        GemmShape::new(compiled_m, compiled_n, compiled_k),
        CompiledDims::STATIC_MNK,
    )
    .expect("STATIC_MNK specialization must compile on ignored test runner");

    // コンパイルした Static(65, 136, 40) と一致する起動は成功する
    // （fail-closed 判定が過剰拒否していないことの健全性確認）。
    let (a_match, b_match) = gen_ab(4242, compiled_m, compiled_n, compiled_k);
    handle
        .launch_f16(&a_match, &b_match, compiled_m, compiled_n, compiled_k)
        .expect("launch with matching (m, n, k) must succeed");

    // M のみ不一致（64 != 65）の起動は InvalidKernelConfig で拒否される。
    let mismatched_m = compiled_m - 1;
    let (a_mismatch, b_mismatch) = gen_ab(4243, mismatched_m, compiled_n, compiled_k);
    let result = handle.launch_f16(
        &a_mismatch,
        &b_mismatch,
        mismatched_m,
        compiled_n,
        compiled_k,
    );
    assert!(
        matches!(result, Err(CudaError::InvalidKernelConfig { .. })),
        "STATIC_MNK でコンパイル済みのカーネルへ不一致形状 (m={mismatched_m}) を渡した起動は \
         InvalidKernelConfig で拒否されるべきです: {result:?}"
    );
}

/// `STATIC_NK`（N/K 静的化・M 動的）動的次元再利用（実機必須）: 同一
/// コンパイル済みカーネルを N/K 固定・M 可変で複数回起動しても parity
/// を満たすこと（イシュー #531 実装計画 §3.2 点 3）を確認する。
///
/// M=65（コンパイル時の代表形状。`SpecializedMmaKernelHandle::compile`
/// ドキュメンテーションコメント参照: dim_m=Dynamic のため代表値は
/// 起動結果に影響しない）でコンパイルした後、M=65・M=1 の双方で起動し、
/// いずれも CPU 参照実装と一致することを確認する。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須"]
fn specialized_mma_f16_static_nk_reuses_across_dynamic_m() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");

    let (n, k) = (136u32, 40u32);
    let handle = SpecializedMmaKernelHandle::compile(
        &device,
        GemmShape::new(65, n, k),
        CompiledDims::STATIC_NK,
    )
    .expect("STATIC_NK specialization must compile on ignored test runner");

    for (idx, &m) in [65u32, 1u32].iter().enumerate() {
        let seed = 4300 + idx as u64;
        let (a, b) = gen_ab(seed, m, n, k);

        let specialized_c = handle.launch_f16(&a, &b, m, n, k).unwrap_or_else(|err| {
            panic!("m={m}: SpecializedMmaKernelHandle::launch_f16 failed: {err}")
        });

        let c_ref_rounded = cpu_reference_f32(&a, &b, m, n, k);
        let c_specialized_f32: Vec<f32> = specialized_c.iter().map(|x| x.to_f32()).collect();
        fandhe_ai_backend_cpu::assert_parity(
            &format!("STATIC_NK dynamic M reuse m={m} n={n} k={k}"),
            &c_specialized_f32,
            &c_ref_rounded,
        );
    }
}

/// `k==0` no-op 形状（実機必須）: `SpecializedMmaKernelHandle::launch_f16`
/// が `gemm_mma.rs::CudaMmaGemm::run_f16`・`run_specialized_mma_f16` と
/// 同一契約で `k==0` を早期 return し、C を全 0 として返すことを確認する
/// （PR #685 Bugbot 再指摘〈Medium〉への回帰テスト）。
///
/// `n=7`（8 の倍数ではない）を選び、修正前は no-op 判定を欠いたまま
/// `validate_mma_alignment(n=7, k=0)` に到達して誤って `InvalidShape` を
/// 返していたことを再現する（`gemm_mma.rs::tests::
/// validate_mma_alignment_rejects_misaligned_n_independent_of_noop_shape`
/// が単体で検査する非整列拒否と、`run_f16`／本テストが検査する「no-op
/// 早期 return がその非整列拒否より前に効く」契約は別物である点に注意）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須"]
fn specialized_mma_f16_handles_k_zero_noop_with_misaligned_n() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");

    let (m, n, k) = (8u32, 7u32, 0u32);
    let handle = SpecializedMmaKernelHandle::compile(
        &device,
        GemmShape::new(m, n, k),
        CompiledDims::DYNAMIC_ALL,
    )
    .expect("DYNAMIC_ALL specialization must compile on ignored test runner");

    // k=0 のため A・B は空スライスでよい（`gemm::validate_gemm_dims` は
    // m*k=0・k*n=0 の長さ一致を要求するのみ）。
    let a: Vec<f16> = Vec::new();
    let b: Vec<f16> = Vec::new();

    let c = handle.launch_f16(&a, &b, m, n, k).expect(
        "k==0 no-op shape (m=8, n=7, k=0) must succeed despite n not being a multiple of 8",
    );

    assert_eq!(
        c,
        vec![f16::ZERO; (m as usize) * (n as usize)],
        "k==0 no-op shape must return an all-zero C"
    );
}

/// プロセス内 LRU カーネルモジュールキャッシュの再利用検証（実機必須。
/// イシュー #511・C-4）。
///
/// 同一形状（`STATIC_NK`・`(m,n,k)=(64,128,32)`。1 ブロックタイル
/// ちょうどで NVRTC コンパイルコストを最小化する）に対して
/// `SpecializedMmaKernelHandle::compile` を 2 回呼び、2 回目は
/// `crate::module_cache::KernelModuleCache`（プロセス内 LRU）をヒットする
/// ことを `fandhe_ai_backend_cuda::diagnostics::module_cache_hit_count` の増加で
/// 確認する。
///
/// プロセス内 LRU はプロセスワイドの `static`（`OnceLock`）であり、他の
/// テスト関数（同一バイナリ内で並行実行されうる）も同じキャッシュへ
/// アクセスしうる。本テストは「ヒット件数が呼び出し前より増加した」こと
/// のみを主張し、絶対値や他テストの介在を仮定しないことで並行実行安全性
/// を保つ（`cargo test` の既定並行実行と両立する設計）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須"]
fn specialized_mma_kernel_handle_compile_reuses_process_local_module_cache() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let shape = GemmShape::new(64, 128, 32);

    // 1 回目: ミスしうる（他テストが未実行なら NVRTC コンパイルが走る）。
    let _first = SpecializedMmaKernelHandle::compile(&device, shape, CompiledDims::STATIC_NK)
        .expect("1st compile must succeed on ignored test runner");

    let hits_before = fandhe_ai_backend_cuda::diagnostics::module_cache_hit_count()
        .expect("module cache must be initialized after at least one compile() call");

    // 2 回目: 同一形状・同一 CompiledDims のため、プロセス内 LRU をヒット
    // する（`RenderedMmaKernel::compile` の 1 段目）はずである。
    let second = SpecializedMmaKernelHandle::compile(&device, shape, CompiledDims::STATIC_NK)
        .expect("2nd compile must succeed on ignored test runner");

    let hits_after = fandhe_ai_backend_cuda::diagnostics::module_cache_hit_count()
        .expect("module cache must remain initialized");

    assert!(
        hits_after > hits_before,
        "2nd compile() with the same shape/CompiledDims must hit the process-local LRU \
         (hits_before={hits_before}, hits_after={hits_after})"
    );

    // キャッシュ経由でロードしたカーネルの実行結果が既定カーネルと
    // bit 一致すること（キャッシュが数値経路に影響しないことの回帰）。
    let default_gemm = CudaMmaGemm::new(&device).expect("mma kernel compilation must succeed");
    let (a, b) = gen_ab(9001, 64, 128, 32);
    let default_c = default_gemm
        .run_f16(&a, &b, 64, 128, 32)
        .expect("default run_f16 must succeed on ignored test runner");
    let cached_c = second
        .launch_f16(&a, &b, 64, 128, 32)
        .expect("launch via cache-backed handle must succeed");

    assert_eq!(
        default_c, cached_c,
        "cache-backed specialized kernel output must bit-match the default kernel"
    );
}
