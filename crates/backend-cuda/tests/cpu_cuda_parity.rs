//! CPU-CUDA ペアの数値一致回帰テスト（TASK-2.2b・#54）。
//!
//! REQ-2 統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」を、
//! `backend_cpu::parity`（TASK-2.2a・#53）の共通ユーティリティ
//! （[`backend_cpu::assert_parity`]）を通して naive GEMM（f32）の
//! CPU 参照実装（[`backend_cpu::matmul_reference_fma`]。FMA 契約の
//! 唯一の参照点）と CUDA naive カーネル（[`CudaGemm::run_naive_f32`]）の
//! 間で検証する。判定式・閾値定数はここで再定義せず `parity.rs` の実体を
//! 唯一の参照とする（`.claude/rules/coding-rust.md`
//! 「バックエンド間数値一致テストの許容誤差を単独で緩和しない」）。
//!
//! **`gemm_naive.rs` からの移管**: `tests/gemm_naive.rs`（#33・PR #240）に
//! 存在した数値一致テストはローカル複製の複合判定式
//! （`rel >= TOL && diff >= TOL` という否定形）を使っており、これは
//! NaN vs 有限値・Inf vs 有限値のケースで両辺 false となり誤って合格
//! 判定してしまう盲点を持っていた（`parity.rs` の `compare` 関数
//! ドキュメントコメント参照。Cursor Bugbot 指摘・PR #239 で `parity.rs`
//! 側は修正済みだが `gemm_naive.rs` 側の複製は未修正のまま残っていた）。
//! 本ファイルはその複製を廃し `backend_cpu::assert_parity` に一本化する
//! （受け入れ条件: 判定ロジックの重複実装をしない。`parity.rs` の
//! ドキュメントコメントに明記された想定）。
//!
//! **スコープ**（イシュー #54 実装計画の「安全側の判断」節）:
//! - 対象カーネルは naive GEMM f32 のみ。tiled GEMM の CPU-CUDA ペアは
//!   #34（tiled 実装）マージ後に本スイートへ追加する
//! - f16 は複合判定（1e-3/1e-5）が f32 前提であり、適用は実質的な
//!   許容誤差変更（ユーザー承認必須）にあたるため対象外（`gemm_naive.rs`
//!   の既存 f16 shape テストを変更せず維持する）。
//!   **例外**: WMMA f16 経路（`tests/cpu_cuda_wmma_parity.rs`・#61）のみ、
//!   #61 の受け入れ条件「f16 GEMM が複合判定で参照実装と一致する」が
//!   明示的に複合判定の適用を要求しているため対象とする。本ファイルの
//!   対象外方針（naive f16 等、他の f16 経路）はそれ以外に適用される。
//!   判定基準の一般化（複合判定を f16 全般へ拡大すべきか）の検討は
//!   #186（Tensor Core 経路の数値一致閾値の実測再評価）へ委ねる
//! - elementwise・reduction は CUDA 側カーネルが未実装のため対象外
//!   （PoC-v2-5 の判定カバレッジ〈GEMM・elementwise・reduction〉との差異）
//! - CPU-Metal ペアは #55 のスコープ
//!
//! **実機依存の分離**（`.claude/rules/coding-rust.md`）: 形状網羅・
//! K=4096 ストレスケースは `#[ignore]` で分離する。環境適応テスト
//! （`naive_f32_parity_smoke_env_adaptive`）のみ通常 CI（self-hosted・
//! CUDA toolkit 非搭載）で実行され、CUDA 非搭載環境では早期 return で
//! green になる（`tests/device_init.rs`・`tests/gemm_naive.rs` の
//! 分岐パターンを踏襲）。

use backend_cuda::{CudaDevice, CudaError, CudaGemm};

/// 決定的シードで A・B（f32）を生成し、CPU 参照実装
/// （[`backend_cpu::matmul_reference_fma`]。FMA 契約の唯一の参照点）と
/// CUDA naive カーネルの出力を [`backend_cpu::assert_parity`] で照合する。
///
/// `context` は失敗時の診断メッセージ（`assert_parity` が付与する
/// 誤差分布統計）の先頭に付く識別子で、呼び出し元の形状ケースを
/// 特定しやすくするために渡す。
fn assert_naive_f32_parity(gemm: &CudaGemm, context: &str, seed: u64, m: u32, n: u32, k: u32) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    let mut c_ref = vec![0.0f32; (m as usize) * (n as usize)];
    backend_cpu::matmul_reference_fma(&a, &b, &mut c_ref, m as usize, n as usize, k as usize)
        .expect("matmul_reference_fma shape validation must pass for well-formed test input");

    let c_gpu = gemm
        .run_naive_f32(&a, &b, m, n, k)
        .expect("CudaGemm::run_naive_f32 must succeed on CUDA-equipped test runner");

    backend_cpu::assert_parity(context, &c_gpu, &c_ref);
}

/// 環境適応型のスモークテスト（`#[ignore]` なし。通常 CI で実行）。
///
/// `tests/gemm_naive.rs::new_does_not_panic_and_returns_typed_result` と
/// 同じ分岐パターンで `CudaDevice::new`／`CudaGemm::new` の非搭載環境
/// エラーを早期 return し green とする。CUDA+toolkit 搭載環境でのみ
/// 小形状（64×64×64）で `assert_parity` による複合判定を実施し、
/// コンパイル成立に加えて実行時の判定経路そのものも self-hosted CI 上で
/// 継続的に確認できるようにする。
#[test]
fn naive_f32_parity_smoke_env_adaptive() {
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
            // のみで toolkit 非搭載）。本環境の開発コンテナ・self-hosted CI
            // がこれに該当し、以降の実行判定はスキップして green とする。
            assert!(!detail.is_empty());
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaGemm::new: {other}"),
    };

    assert_naive_f32_parity(&gemm, "smoke 64x64x64", 1, 64, 64, 64);
}

/// 実機（DGX Spark GB10 等）必須の形状網羅テスト。受け入れ条件の本体。
///
/// CI self-hosted runner は CUDA toolkit 非搭載のため通常実行ではスキップ
/// される（`cargo test -- --ignored` での実機実行を前提とする。実行導線の
/// 整備は #36 のスコープ）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn naive_f32_matches_reference_across_shapes() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("naive kernel compilation must succeed");

    // 形状ケース: 正方（128^3・512^3）・非正方（64x96x128）・
    // NAIVE_BLOCK_DIM（16x16）の端ブロック境界を踏む形状（1x1x1・
    // 17x23x19・33x31x65。REQ-8 手動境界検査の回帰対象）。
    let cases: &[(u32, u32, u32)] = &[
        (128, 128, 128),
        (512, 512, 512),
        (64, 96, 128),
        (1, 1, 1),
        (17, 23, 19),
        (33, 31, 65),
    ];
    for (idx, &(m, n, k)) in cases.iter().enumerate() {
        let context = format!("shape m={m} n={n} k={k}");
        assert_naive_f32_parity(&gemm, &context, 1000 + idx as u64, m, n, k);
    }
}

/// K 大のストレスケース群（PoC-v2-5 準拠）。
///
/// PoC-v2-5（`docs/spec/03-poc/poc-v2-5-backend-numeric-parity/README.md`）は
/// M=N=512, K=4096 の積和蓄積ストレスケースで、CPU 参照実装が
/// `f32::mul_add`（FMA 契約）を使わない場合に複合判定が僅かに未達となる
/// ことを実測した（262,144 セル中 7 セル）。CPU 側を `mul_add` に
/// 揃えれば bit 完全一致することも同 PoC で確認済みであり、
/// `matmul_reference_fma` はその remediation を実装した関数そのもの
/// （このケースが FMA 契約統一の妥当性を確認する回帰の中核）。
///
/// `tests/gemm_naive.rs`（#33）にあった M=N=256, K=4096 ケースも
/// 形状多様性維持のため併存させる（実行は実機のみのためコスト許容）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn naive_f32_k4096_stress_poc_v2_5() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("naive kernel compilation must succeed");

    assert_naive_f32_parity(&gemm, "PoC-v2-5 stress 256x256x4096", 9999, 256, 256, 4096);
    assert_naive_f32_parity(
        &gemm,
        "PoC-v2-5 stress 512x512x4096",
        0xC0FFEE,
        512,
        512,
        4096,
    );
}
