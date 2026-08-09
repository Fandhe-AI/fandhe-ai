//! CPU-Metal ペアの数値一致回帰テスト: f16 `gemm_simdgroup_f16`
//! （TASK-8.3b・#156）。
//!
//! `tests/cpu_metal_parity.rs`（f32・TASK-2.2c）と同じ判定基盤
//! （`backend_cpu::assert_parity`。REQ-2 統一複合判定「相対誤差 1e-3 未満
//! または 絶対誤差 1e-5 未満」の唯一の実体）を使うが、参照実装との比較
//! 方法は CUDA 側 f16 WMMA パリティテスト
//! （`crates/backend-cuda/tests/cpu_cuda_wmma_parity.rs`）の方式を移植する
//! （実装計画 §4.2 の指示・advisor 助言: 「Metal パリティテストが参照値
//! 構築で CUDA 版と異なると reviewer に弾かれる」）。
//!
//! # 参照実装との比較方法（`cpu_cuda_wmma_parity.rs` と同一の 3 段階）
//!
//! 1. f16 入力を f32 化し `backend_cpu::matmul_reference_fma`（FMA 契約の
//!    唯一の参照点）で参照値を計算する。
//! 2. 参照値を f16 経由で丸める（カーネルの `simdgroup_store` による half
//!    エピローグ store と同じ量子化をホスト側でも再現し、丸め方式の差では
//!    なく計算経路の差のみを判定対象にする。CUDA 版の `__float2half` と
//!    同じ役割）。
//! 3. GPU 出力（f16）・丸め済み参照値（f16→f32）の双方を f32 化して
//!    `assert_parity` へ渡す。
//!
//! # f16 適用の位置づけ（`cpu_cuda_wmma_parity.rs` 冒頭コメントと同じ整理）
//!
//! 本ファイルは #156 の受け入れ条件「実測記録が残されている」に対応する
//! 実測手段の一部であり、`gemm_simdgroup_f16` 経路
//! （[`backend_metal::MetalGemm::dispatch_f16_unverified`]）にのみ複合判定を
//! 適用する明示的な例外として扱う（他の f16 経路一般を対象化するものでは
//! ない）。
//!
//! # 累算精度契約（`shaders/gemm.metal::gemm_simdgroup_f16` と同一の注意）
//!
//! `gemm_simdgroup_f16` は A・B に `simdgroup_half8x8`、アキュムレータに
//! `simdgroup_float8x8`（f32 累算）を使う（カーネル冒頭コメント参照）。
//! イシュー #380 の Apple Silicon 実機 spike で「A/B=half・アキュムレータ=
//! float の混在オーバーロードは実在するが `simdgroup_store` は float→half
//! 直接変換に非対応」と判明し、f32 累算 + スレッド単位 half 変換の
//! 2 段エピローグへ変更済み（実装計画時点の half 統一からの変更）。CUDA 側
//! WMMA f16（`f32.f16.f16.f32`。f32 累算）と精度契約が整合する。実機実測
//! （M4 Max・macOS 26.6）で本ファイル 6 件全件が複合判定 green であることを
//! 確認済み（`docs/backend-metal-real-device-testing.md`）。
//!
//! # 実行環境
//!
//! `tests/cpu_metal_parity.rs` と同じく `#![cfg(target_os = "macos")]` で
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する。CI（self-hosted・
//! Linux）ではコンパイル対象外になる。全ケース `#[ignore]` とし、
//! `cargo test -p backend-metal --release -- --ignored --nocapture` で
//! 実行する（K=4096 ストレスケースは debug では遅いため release 推奨）。
//! CUDA 側のような「環境適応型スモーク」（`#[ignore]` なしで通常 CI 実行）
//! は設けない: Metal は self-hosted Linux runner 上に実機がなく
//! `#![cfg(target_os = "macos")]` によりそもそも Linux でコンパイルされない
//! ため、CUDA の「ドライバ非搭載を実行時検出して early return」という
//! 環境適応パターンが成立しない（`tests/cpu_metal_parity.rs` と同じ制約）。

#![cfg(target_os = "macos")]

use backend_cpu::parity::assert_parity;
use backend_metal::{MetalContext, MetalGemm};
use bench_harness::rng::Xorshift64Star;
use half::f16;

/// 決定的シードで A・B（f16）を生成し、f16→f32→参照 matmul→f16 丸め→f32 の
/// 経路で得た参照値と `gemm_simdgroup_f16` の出力（f16→f32）を
/// `assert_parity` で照合する（ファイル冒頭コメント「参照実装との比較
/// 方法」参照。`cpu_cuda_wmma_parity.rs::assert_wmma_f16_parity` と同型）。
fn assert_metal_f16_parity(
    ctx: &MetalContext,
    gemm: &MetalGemm,
    context: &str,
    seed: u64,
    m: usize,
    n: usize,
    k: usize,
) {
    let mut rng = Xorshift64Star::new(seed);
    let a_f16: Vec<f16> = rng.fill_vec_f16(m * k);
    let b_f16: Vec<f16> = rng.fill_vec_f16(k * n);

    let a_f32: Vec<f32> = a_f16.iter().map(|x| x.to_f32()).collect();
    let b_f32: Vec<f32> = b_f16.iter().map(|x| x.to_f32()).collect();
    let mut c_ref_f32 = vec![0.0f32; m * n];
    backend_cpu::parity::matmul_reference_fma(&a_f32, &b_f32, &mut c_ref_f32, m, n, k)
        .expect("matmul_reference_fma shape validation must pass for well-formed test input");
    // `gemm_simdgroup_f16` の `simdgroup_store`（half エピローグ store）と
    // 同じ量子化を参照側にも適用し、計算経路の差のみを判定対象にする。
    let c_ref_rounded: Vec<f32> = c_ref_f32
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect();

    let c_gpu_f16 = gemm
        .dispatch_f16_unverified(ctx, &a_f16, &b_f16, m, n, k)
        .expect("MetalGemm::dispatch_f16_unverified must succeed on Metal-equipped test runner");
    let c_gpu_f32: Vec<f32> = c_gpu_f16.iter().map(|x| x.to_f32()).collect();

    assert_parity(context, &c_gpu_f32, &c_ref_rounded);
}

/// 8 の倍数ぴったりの最小形状（`gemm_simdgroup_f16` の 8x8 タイル 1 個分。
/// パディング境界を跨がない基準ケース）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn f16_parity_baseline_8x8x8() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した（f16 含む）");
    assert_metal_f16_parity(&ctx, &gemm, "f16 baseline 8x8x8", 101, 8, 8, 8);
}

/// PoC-v2-5 基準形状（M=N=K=512。f32 版 `cpu_metal_parity.rs` と同じ規模で
/// 比較できるようにする）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn f16_parity_baseline_shape_512() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した（f16 含む）");
    assert_metal_f16_parity(&ctx, &gemm, "f16 baseline 512x512x512", 111, 512, 512, 512);
}

/// 8 の倍数でない非倍数エッジ形状（REQ-8 手動境界検査・`pad8` パディング
/// 経路の回帰。`cpu_metal_parity.rs::boundary_shapes_non_multiple_of_threadgroup`
/// とは意図的に異なる形状を選び直接の重複を避ける）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn f16_parity_boundary_shapes_non_multiple_of_eight() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した（f16 含む）");
    assert_metal_f16_parity(&ctx, &gemm, "f16 boundary 17x19x23", 121, 17, 19, 23);
    assert_metal_f16_parity(&ctx, &gemm, "f16 boundary 130x70x90", 122, 130, 70, 90);
}

/// K 大のストレスケース（PoC-v2-5 準拠の積和蓄積検証。f32 累算の桁落ち
/// 耐性を確認する中核ケース）。実装計画時点（half 統一アキュムレータ）の
/// 実機実測では 60155/65536 要素が複合判定を外れていたが、イシュー #380 の
/// f32 累算化（ファイル冒頭コメント参照）後は実機（M4 Max・macOS 26.6）で
/// green を確認済み。今後複合判定を外れた場合も緩和せず FAIL 事実を
/// `docs/perf/metal-f16-vs-mps-f16.md` へ記録し #386（REQ-8 Metal f16 下限
/// 確定）へ引き継ぐ。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn f16_k4096_stress() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した（f16 含む）");
    assert_metal_f16_parity(
        &ctx,
        &gemm,
        "f16 K4096 stress 256x256x4096",
        131,
        256,
        256,
        4096,
    );
}

/// 決定性テスト: 同一シードで 2 回 dispatch した結果が bit 完全一致する
/// こと（`cpu_metal_parity.rs::dispatch_is_bit_deterministic_across_runs`
/// の f16 版）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn f16_dispatch_is_bit_deterministic_across_runs() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した（f16 含む）");

    let m = 64;
    let n = 64;
    let k = 128;
    let mut rng = Xorshift64Star::new(141);
    let a: Vec<f16> = rng.fill_vec_f16(m * k);
    let b: Vec<f16> = rng.fill_vec_f16(k * n);

    let first = gemm
        .dispatch_f16_unverified(&ctx, &a, &b, m, n, k)
        .expect("1 回目の dispatch_f16_unverified に失敗した");
    let second = gemm
        .dispatch_f16_unverified(&ctx, &a, &b, m, n, k)
        .expect("2 回目の dispatch_f16_unverified に失敗した");

    assert_eq!(
        first, second,
        "同一入力の 2 回 dispatch_f16_unverified が bit 完全一致しない"
    );
}

/// `MetalGemm::dispatch_f16_prepared_unverified` の入力検証回帰（PR #346 codex-review
/// P1-1 指摘）。呼び出し元が公開コンストラクタ（`MetalHalfBuffer::
/// new_with_data`/`new_zeroed`）で確保した任意長のバッファを渡せるため、
/// `a_buf`/`b_buf`/`c_buf` の実長が実効次元（m_eff*k_eff 等）と一致しない
/// 場合・実効次元が 8 の倍数でない場合にエンコード前でエラーを返し、
/// Metal カーネルへ到達しないことを確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn f16_dispatch_prepared_rejects_undersized_and_misaligned_inputs() {
    use backend_metal::error::MetalError;
    use backend_metal::half_buffer::MetalHalfBuffer;

    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した（f16 含む）");

    // m_eff=n_eff=k_eff=8（8x8x8 タイル 1 個分）の正しいバッファ長は
    // 8*8=64 要素。ここでは意図的に短い（32 要素）バッファを渡す。
    let short = vec![f16::from_f32(0.0); 32];
    let correct = vec![f16::from_f32(0.0); 64];
    let a_buf = MetalHalfBuffer::new_with_data(&ctx, &short).expect("a_buf 確保に失敗した");
    let b_buf = MetalHalfBuffer::new_with_data(&ctx, &correct).expect("b_buf 確保に失敗した");
    let c_buf = MetalHalfBuffer::new_zeroed(&ctx, 64).expect("c_buf 確保に失敗した");

    let err = gemm
        .dispatch_f16_prepared_unverified(&ctx, &a_buf, &b_buf, &c_buf, 8, 8, 8)
        .expect_err(
            "短い a_buf に対して dispatch_f16_prepared_unverified はエラーを返す必要がある",
        );
    assert!(
        matches!(
            err,
            MetalError::ALenMismatch {
                expected: 64,
                actual: 32,
            }
        ),
        "unexpected error: {err:?}"
    );

    // 実効次元が 8 の倍数でない場合（バッファ長は正しくても）拒否する。
    let a_buf9 = MetalHalfBuffer::new_with_data(&ctx, &[f16::from_f32(0.0); 72])
        .expect("a_buf9 確保に失敗した");
    let b_buf9 = MetalHalfBuffer::new_with_data(&ctx, &[f16::from_f32(0.0); 72])
        .expect("b_buf9 確保に失敗した");
    let c_buf9 = MetalHalfBuffer::new_zeroed(&ctx, 81).expect("c_buf9 確保に失敗した");
    let err = gemm
        .dispatch_f16_prepared_unverified(&ctx, &a_buf9, &b_buf9, &c_buf9, 9, 9, 8)
        .expect_err(
            "非 8 の倍数の実効次元に対して dispatch_f16_prepared_unverified はエラーを返す必要がある",
        );
    assert!(
        matches!(err, MetalError::NotEightAligned { .. }),
        "unexpected error: {err:?}"
    );
}
