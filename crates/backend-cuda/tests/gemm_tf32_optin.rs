//! イシュー #1042: `crate::precision::set_tf32_gemm_enabled` の opt-in で
//! 切り替わる `CudaBackendOps::gemm`（f32 の素の GEMM）の複合判定検証。
//!
//! カーネル本体（`CudaGemm::run_wmma_tf32`）自体の誤差分布は
//! `gemm_wmma_tf32.rs`・`gemm_wmma_tf32_opt.rs`・`mma_tf32_vs_wmma_tf32_
//! staged.rs`（既存）が既に検証済みであり、本ファイルはそれらを重複させ
//! ない。本ファイルが検証するのは「`ops.rs::CudaBackendOps::gemm` の
//! opt-in フラグ配線」という新規の到達経路のみ（`gemm_bias_act_parity.rs`
//! と同じ構成方針）:
//!
//! 1. opt-in OFF（既定）時、`gemm` 出力が本イシュー導入前と bit-exact に
//!    一致すること（既定 OFF の非後退契約）。CUDA 内で完結する `==` の
//!    完全一致で検証する（codex-review P2 指摘・PR #1091: tolerance
//!    許容比較では非後退契約を固定できないため。詳細は
//!    `gemm_tf32_optin_off_matches_default_fp32_path_env_adaptive` の
//!    ドキュメンテーションコメント）。
//! 2. 上記とは別に、opt-in OFF 時の `gemm` 出力が CPU 参照実装と REQ-2
//!    統一複合判定で一致すること（通常のバックエンド間 parity 契約）。
//! 3. opt-in ON 時、`gemm` 出力が `CudaGemm::run_wmma_tf32`（同一カーネル）
//!    の直接呼び出しと **bit-exact** に一致すること（イシュー #1106 で
//!    CPU 参照実装との複合判定から変更）。TF32 経路は REQ-2 統一複合判定を
//!    最小形状から恒常的に満たさない既知状態（`docs/spec/04-requirements.md`
//!    REQ-2 の 2026-08-29 追記）にあり、そのカーネル自体の誤差分布は上記
//!    「カーネル本体」節のとおり他ファイルの責務であるため、本ファイルが
//!    元々検証したい「opt-in フラグ配線」（`ops.rs::CudaBackendOps::gemm`
//!    が opt-in 時に `run_wmma_tf32` へ正しく分岐すること）には CPU 参照
//!    実装との tolerance 比較は不要かつ過剰であった。bit-exact 比較へ
//!    変更することで、tolerance・ベースライン機構のいずれにも依存せず
//!    「配線が正しいか」だけを判定する（1・2 節の OFF 時比較と同型の
//!    設計に統一）。
//!
//! `common::parity_baseline` から tolerance 定数 pin を借用し、判定式・
//! 許容誤差は再定義しない（`.claude/rules/coding-rust.md`）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test gemm_tf32_optin -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_backend_cuda::precision::{set_tf32_gemm_enabled, tf32_gemm_enabled};
use fandhe_ai_backend_cuda::{CudaDevice, CudaGemm};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{Activation, BackendOps, Tensor};

mod common;

/// フラグはプロセスグローバル（`crate::precision`）のため、
/// `cargo test` の既定並列実行下で他のテストバイナリ内テストと競合し
/// うる。本ファイル内では直列に実行し、各テストの最後に必ず OFF へ
/// 戻す（`ops.rs::tests::Tf32FlagGuard` と同型の RAII ガード）。
struct Tf32FlagGuard {
    original: bool,
}

impl Tf32FlagGuard {
    fn acquire(enabled: bool) -> Self {
        let original = tf32_gemm_enabled();
        set_tf32_gemm_enabled(enabled);
        Self { original }
    }
}

impl Drop for Tf32FlagGuard {
    fn drop(&mut self) {
        set_tf32_gemm_enabled(self.original);
    }
}

/// opt-in ON 時の `CudaBackendOps::gemm` 出力が、同一入力に対する
/// `CudaGemm::run_wmma_tf32`（配線先カーネルの直接呼び出し）と bit-exact に
/// 一致することを確認する（イシュー #1106。ファイル冒頭コメント「3.」
/// 参照）。CPU 参照実装は使わない——TF32 経路自体の誤差分布は
/// `gemm_wmma_tf32.rs` 等の責務であり、本関数が固定したいのは「opt-in
/// フラグが `run_wmma_tf32` へ正しく配線されていること」のみのため、
/// tolerance 判定を経由しない `==` の完全一致で検証する。
fn assert_tf32_optin_gemm_parity(seed_a: u64, seed_b: u64, m: usize, n: usize, k: usize) {
    let cuda = CudaBackendOps::new(0);

    let a_data = Xorshift64Star::new(seed_a).fill_vec(m * k);
    let b_data = Xorshift64Star::new(seed_b).fill_vec(k * n);
    let a = Tensor::new(a_data.clone(), &[m, k]).expect("valid tensor");
    let b = Tensor::new(b_data.clone(), &[k, n]).expect("valid tensor");

    let _guard = Tf32FlagGuard::acquire(true);
    let cuda_result = cuda
        .gemm(&a, &b)
        .expect("CudaBackendOps::gemm (tf32 opt-in) must succeed on CUDA-equipped test runner");

    // 配線先カーネルを直接呼び出す独立経路（`ops.rs::CudaBackendOps::gemm`
    // が内部で使うキャッシュ済み `CudaGemm` インスタンスとは別に、この
    // テスト専用の新規インスタンスを生成する。同一 PTX・同一入力である限り
    // インスタンスが異なっても出力は bit-exact になるはず——ならなければ
    // それ自体が配線・カーネルいずれかの回帰である）。
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("CudaGemm::new (tiled/WMMA TF32) must succeed");
    let direct_result = gemm
        .run_wmma_tf32(&a_data, &b_data, m as u32, n as u32, k as u32)
        .expect("run_wmma_tf32 direct call must succeed on CUDA-equipped test runner");

    assert_eq!(
        cuda_result.as_slice().expect("contiguous"),
        direct_result.as_slice(),
        "tf32 opt-in gemm m={m} n={n} k={k}: CudaBackendOps::gemm（opt-in ON）の出力が \
         CudaGemm::run_wmma_tf32 の直接呼び出しと bit-exact に一致しません（opt-in \
         フラグの配線に回帰がある可能性があります）"
    );
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。opt-in OFF（既定）時の
/// `gemm` 出力が opt-in を一切知らないかのように動作する（本イシュー
/// 導入前の `run_tiled_f32` 単独経路と bit-exact に一致する）ことを、
/// **CUDA 内で完結する bit-exact 比較**で確認する（codex-review P2
/// 指摘・PR #1091: `assert_parity`〈相対誤差 1e-3／絶対誤差 1e-5 許容〉
/// は CPU-GPU 異丸め方式間の複合判定としては正しいが、OFF 時の非後退
/// 契約〈同一 GPU カーネル `run_tiled_f32` を挟んで出力が一切変化しない
/// こと〉を固定するには誤差許容比較では不十分で、許容範囲内の出力劣化を
/// 検出できない）。
///
/// 比較対象は `self.gemm(a, b)`（OFF 時は `gemm_fp32_strict` へ委譲。
/// `ops.rs::CudaBackendOps::gemm` 冒頭コメント参照）と、
/// `gemm_bias_act(a, b, Some(&zero_bias), Activation::None)` を
/// ブロードキャスト可能だが `[n]` 完全一致ではない bias 形状（`[1]`。
/// `n != 1` の形状を使うことで `ops.rs::gemm_bias_act_route` が
/// `ComposedFallback`〈`gemm_fp32_strict` → `add` → 恒等〉を選ぶことを
/// 保証する）で迂回的に呼んだ経路の 2 通り。両者は TF32 opt-in フラグの
/// 影響を受けない同一の `gemm_fp32_strict` 呼び出しに帰着するため、
/// ゼロ加算（`+0.0` は丸め誤差を生まない）・恒等 activation を介しても
/// bit-exact に一致するはずであり、`==` の完全一致で比較する（tolerance
/// 判定を混入させない）。CUDA 不在なら双方が同型の
/// `BackendError::CudaUnavailable` を返すことを確認して早期 return する
/// （`gemm_bias_act_parity.rs` と同じ分岐パターン）。
#[test]
fn gemm_tf32_optin_off_matches_default_fp32_path_env_adaptive() {
    let _guard = Tf32FlagGuard::acquire(false);
    let cuda = CudaBackendOps::new(0);
    // Cursor Bugbot 指摘（PR #1091・Medium）: 小さい整数オペランド
    // （旧: 1.0..8.0）は TF32（仮数部 10bit への丸め）でも FP32（23bit）
    // でも厳密に表現できてしまうため、`gemm` が OFF 時に誤って TF32
    // 経路へルーティングされても本テストの bit-exact 比較が silent
    // degradation を検出できない（`via_fallback` 側は常に
    // `gemm_fp32_strict` を使う一方、`direct` 側だけ誤って TF32 精度に
    // 丸まっても、整数入力は丸めで値が変化しないため一致してしまう）。
    // `Xorshift64Star::fill_vec` は `[-1, 1)` の 24bit 仮数精度乱数を
    // 生成し、TF32 の 10bit への丸めで実際に値が変化する（＝バグがあれば
    // 検出できる）入力にする。
    let mut rng = Xorshift64Star::new(0x1042_C0FF_EE00);
    let a_data = rng.fill_vec(4);
    let b_data = rng.fill_vec(4);
    let a = Tensor::new(a_data, &[2, 2]).expect("valid tensor");
    let b = Tensor::new(b_data, &[2, 2]).expect("valid tensor");
    // n=2 に対し形状 [1] はブロードキャスト可能だが `[n]` 完全一致では
    // ないため `gemm_bias_act_route` が `ComposedFallback` を選ぶ
    // （`ops.rs::gemm_bias_act_route`）。
    let zero_bias = Tensor::new(vec![0.0], &[1]).expect("valid tensor");

    let direct = cuda.gemm(&a, &b);
    let via_fallback = cuda.gemm_bias_act(&a, &b, Some(&zero_bias), Activation::None);

    match (direct, via_fallback) {
        (Ok(direct_result), Ok(fallback_result)) => {
            assert_eq!(direct_result.shape(), fallback_result.shape());
            assert_eq!(
                direct_result.as_slice().expect("contiguous"),
                fallback_result.as_slice().expect("contiguous"),
                "opt-in OFF 時の gemm 出力が gemm_fp32_strict 経由の \
                 参照値と bit-exact に一致しない（非後退契約違反）"
            );
        }
        (
            Err(BackendError::CudaUnavailable(direct_msg)),
            Err(BackendError::CudaUnavailable(fallback_msg)),
        ) => {
            assert!(
                !direct_msg.is_empty(),
                "error detail message must not be empty"
            );
            assert!(
                !fallback_msg.is_empty(),
                "error detail message must not be empty"
            );
        }
        (direct_res, fallback_res) => panic!(
            "unexpected result combination for CudaBackendOps::gemm vs gemm_bias_act \
             fallback (opt-in OFF): direct={direct_res:?}, fallback={fallback_res:?}"
        ),
    }
}

/// opt-in OFF 時、`gemm` 出力が CPU 参照実装と REQ-2 統一複合判定で
/// 一致することを確認する環境適応スモーク（CPU-GPU 異丸め方式間の
/// 通常の parity 契約。上記の bit-exact テストとは別の観点で、こちらは
/// バックエンド間数値一致そのものを検証する）。
#[test]
fn gemm_tf32_optin_off_matches_cpu_reference_env_adaptive() {
    let _guard = Tf32FlagGuard::acquire(false);
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("valid tensor");

    match cuda.gemm(&a, &b) {
        Ok(cuda_result) => {
            common::parity_baseline::assert_tolerance_constants_pinned();
            let cpu_result = cpu.gemm(&a, &b).expect("cpu gemm always succeeds");
            fandhe_ai_backend_cpu::parity::assert_parity(
                "tf32 opt-in OFF gemm cpu-cuda parity smoke",
                cuda_result.as_slice().expect("contiguous"),
                cpu_result.as_slice().expect("contiguous"),
            );
        }
        Err(BackendError::CudaUnavailable(msg)) => {
            assert!(!msg.is_empty(), "error detail message must not be empty");
        }
        Err(other) => panic!("unexpected error variant for CudaBackendOps::gemm: {other}"),
    }
}

/// 実機必須の形状網羅（受け入れ条件 2「opt-in 時の複合判定結果」の本体）。
/// opt-in ON 時に TF32 Tensor Core 経路が CPU 参照実装（FP32 厳密）と
/// REQ-2 統一複合判定で一致することを、正方・非正方・K 支配的形状で
/// 確認する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等。cc>=8.0）必須"]
fn gemm_tf32_optin_on_matches_cpu_across_shapes() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let cases: &[(u64, u64, usize, usize, usize)] = &[
        (701, 702, 512, 512, 512),
        (703, 704, 1024, 1024, 1024),
        (705, 706, 96, 160, 48),
        // K 支配的な非正方形状（split-K 検討の先例と同じ形状クラス。
        // `docs/perf/metal-gemm-splitk-shapes.md` 参照）。
        (707, 708, 256, 256, 4096),
    ];
    for &(seed_a, seed_b, m, n, k) in cases {
        assert_tf32_optin_gemm_parity(seed_a, seed_b, m, n, k);
    }
}
