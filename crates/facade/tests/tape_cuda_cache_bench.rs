//! `tape_for(Device::Cuda(_))` プロセス内キャッシュの実機ベンチ（イシュー
//! #929・受け入れ条件 1）。
//!
//! `crates/backend-cuda/src/context_cache.rs` の導入により、
//! `CudaBackendOps`（[`fandhe_ai::tape_for`] が結線する `BackendOps`
//! 実装）の各演算メソッドは `CudaDevice`／`CudaGemm` 等をプロセス内
//! キャッシュから取得するようになった。本ベンチは「同一プロセス内で
//! 2 回目以降の `tape_for(Device::Cuda(0))` + 最小 GEMM 1 回」が、
//! プロセス最初の呼び出し（cold: `CudaContext` 生成 + NVRTC カーネル
//! コンパイルを含む）より高速であることを記録する。
//!
//! `crates/backend-cuda/src/jit_cache_bench_tests.rs`（Phase C-12・
//! イシュー #534）と同じ「実機必須・5 回計測中央値・タイミング値は
//! record only（hard assert しない）」方針を踏襲する（GPU クロック
//! 挙動・他プロセス競合等の環境揺らぎを hard assert に持ち込むと
//! flaky 化するため。`.claude/rules/coding-rust.md`「ベンチは 5 回計測
//! の中央値」）。ただし本ベンチが計測する対象はカーネル起動そのもの
//! ではなく「`tape_for` からの一連の結線 + 初回演算呼び出し」という
//! より粗い区間であるため、`backend-cuda` の `jit_cache_bench_tests.rs`
//! （NVRTC コンパイル・モジュールロードを直接計測）とは計測レイヤが
//! 異なる点に注意する。
//!
//! # なぜ facade（本クレート）に置くか
//!
//! 受け入れ条件 1 が検証対象とするのは `tape_for(Device::Cuda(_))` の
//! 呼び出しコストそのもの（composition root の結線 + `CudaBackendOps`
//! 経由のキャッシュ効果）であり、`backend-cuda` 単体の API
//! （`CudaDevice::new`／`CudaGemm::new` の直接呼び出し）ではない。
//! よって `tests/tape_construction.rs` と同じ facade 側 integration
//! test として置く。
//!
//! # 「cold」をプロセス内で 1 回しか観測できないことについて
//!
//! プロセス内キャッシュは `static + OnceLock` によるプロセスワイド
//! シングルトンであるため、同一テストバイナリ内で「cold」状態を
//! 複数回再現することはできない（2 回目以降は必ずキャッシュヒットに
//! なる）。そのため cold は 1 回だけ計測し、warm はその後の
//! `WARM_TRIALS` 回（5 回計測中央値方針）を計測する非対称な設計とする
//! （`measure_cold_warm_trial`〈`jit_cache_bench_tests.rs`〉が trial ごとに
//! 独立した一時ディレクトリで「毎回コールドスタート」を作れるのとは
//! 前提が異なる: 本ベンチが計測するプロセス内キャッシュはプロセス単位の
//! 状態であり、trial 単位でリセットする手段を持たない）。

use std::time::Instant;

use bench_harness::median_q1_q3;
use fandhe_ai::Device;
use fandhe_ai::Tensor;
use fandhe_ai_backend_cuda::CudaDevice;

/// 5 回計測中央値方針（`.claude/rules/coding-rust.md`）。
const WARM_TRIALS: usize = 5;

fn sample_tensor() -> Tensor<f32> {
    // 小さめの正方行列。ベンチの主目的は GEMM 自体のスループットではなく
    // 「`tape_for` + 初回演算呼び出し」の結線オーバーヘッドの cold/warm 差
    // であるため、形状は最小限（カーネル起動が成立する最小サイズ）に留める。
    Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("sample tensor: shape 一致")
}

/// 1 回分の「`tape_for(Device::Cuda(0))` → `matmul` → `backward`」を計測する。
///
/// `Var::matmul`（`fandhe_ai_autodiff::Var::matmul`）を使う理由: `gemm`
/// （[`fandhe_ai_tensor_core::BackendOps::gemm`]）を実際に起動する経路の
/// うち facade 公開 API から到達できる最小の呼び出しであり、
/// `context_cache::cached_gemm`（`ops::CudaBackendOps::gemm`）を通す。
///
/// `product`（非スカラー）をそのまま `backward` の loss に渡す理由
/// （Cursor Bugbot 指摘。イシュー #929 PR #946）: `product.sum(None)` は
/// `BackendOps::sum` を直接呼ぶが、`CudaBackendOps::sum`
/// （`backend-cuda/src/ops.rs`）は汎用 reduction カーネル未実装のため
/// 常に `BackendError::Unsupported` を返す（#599 スコープ外・別イシュー）。
/// このため本ベンチは cold/warm いずれの試行も `sum(None)` の時点で必ず
/// panic し、計測自体が実行できていなかった。`Tape::backward` は
/// 非スカラー loss に対し「全要素 1 のシードで逆伝播する」（`sum(loss)
/// .backward()` と数学的に等価な暗黙の総和射影。`backward.rs`
/// 「非スカラー loss のセマンティクス」参照）契約を持つため、`sum` を
/// 経由せず `product` を直接 loss として渡せば計測対象（`tape_for` の
/// 結線 + `matmul` 経由の `cached_gemm` + `backward`）を変えずに
/// `CudaBackendOps::sum` の未実装を回避できる。
fn measure_tape_for_cuda_matmul() -> f64 {
    let t = Instant::now();
    let tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("CUDA driver 搭載環境（本ベンチは #[ignore] 実機専用）では成功するはず");
    let a = tape.var(&sample_tensor());
    let b = tape.var(&sample_tensor());
    let product = a
        .matmul(&b)
        .expect("2x2 正方行列同士の matmul は shape 一致で成功する");
    let _grads = tape.backward(&product).expect("backward は成功する");
    t.elapsed().as_secs_f64()
}

/// 受け入れ基準 1: 2 回目以降の `tape_for(Device::Cuda(_))` が初期化
/// コスト（`CudaContext` 生成・NVRTC カーネルコンパイル）を支払わない
/// ことを cold（プロセス最初の呼び出し）対 warm（5 回計測中央値）の
/// 方向性（`warm_median < cold`）で記録する。
///
/// 絶対閾値でのアサーションは行わない（本ファイル冒頭コメント「record
/// only」方針）。CI（CUDA 非搭載ホステッドランナー）では実行されない
/// （`#[ignore]`）。
#[test]
#[ignore = "CUDA 実機必須。イシュー #929 受け入れ条件 1（cold/warm 初期化コスト比較）"]
fn tape_for_cuda_second_call_avoids_reinitialization_cost() {
    // `CudaDevice::device_count()`（static。ドライバへの device count
    // 問い合わせのみで `CudaContext` を構築しない軽量プローブ）を使う
    // 理由（Cursor Bugbot 指摘。イシュー #929 PR #946）: 旧実装は
    // `CudaDeviceProvider::new().is_available()`（`DeviceProvider`
    // トレイト実装。`device.rs` 参照）を呼んでいたが、これは
    // `enumerate()` → `probe()` → `context_cache::cached_device(ordinal)`
    // を経由するため、このガード呼び出し自体が cold サンプル計測より前に
    // プロセス全体のデバイスキャッシュを温めてしまい、本ベンチが計測
    // すると主張する `CudaContext::new`（受け入れ条件 1 のコスト本体）が
    // cold パスから漏れ落ちる欠陥があった。`device_count()` はデバイス
    // キャッシュに触れない（コンテキストを構築しない）ため、ガードが
    // 計測対象へ副作用を与えない。「選択可能デバイスが 1 台以上ある」の
    // 確認は `count > 0` で行う（`is_available()` トレイト実装の
    // `enumerate` 経由チェックより弱いが、`probe` 成功まで確認しない分
    // だけキャッシュを汚染しない）。
    let device_count = CudaDevice::device_count().unwrap_or(0);
    assert!(
        device_count > 0,
        "本ベンチは CUDA driver 搭載・選択可能デバイスが 1 台以上ある実機でのみ実行する \
         （#[ignore] で通常 CI からは除外される）"
    );

    // cold: プロセス内でこの呼び出しが最初の `tape_for(Device::Cuda(0))`
    // + 演算呼び出しであることが本テストの前提（本ファイル冒頭コメント
    // 「cold をプロセス内で 1 回しか観測できないことについて」参照）。
    // 同一テストバイナリ内の他テストが先に CUDA 経路を触っていると cold
    // 側の値が汚染されるため、本テストは他の CUDA 実機テストと同時実行
    // しないことを前提とする（`cargo test -- --ignored --test-threads=1`
    // 等。`jit_cache_bench_tests.rs` の実機ベンチ群と同じ運用上の注意）。
    let cold_secs = measure_tape_for_cuda_matmul();

    let mut warm_samples = Vec::with_capacity(WARM_TRIALS);
    for _ in 0..WARM_TRIALS {
        warm_samples.push(measure_tape_for_cuda_matmul());
    }
    let warm_q = median_q1_q3(&warm_samples)
        .expect("WARM_TRIALS 個の non-NaN warm サンプルは quartiles を持つはず");

    println!(
        "[tape_cuda_cache_bench] cold_s={cold_secs:.6} \
         warm_median_s={:.6} (q1={:.6}, q3={:.6}) \
         speedup_x={:.2} warm_faster={} \
         — record only, non-gating（本ファイル冒頭コメント参照）",
        warm_q.median,
        warm_q.q1,
        warm_q.q3,
        cold_secs / warm_q.median.max(f64::EPSILON),
        warm_q.median < cold_secs,
    );
}
