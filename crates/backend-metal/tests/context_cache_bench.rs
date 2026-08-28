//! `crate::context_cache`（プロセス内コンテキスト／カーネルスイート
//! キャッシュ。イシュー #930）の実機ベンチ・回帰テスト。
//!
//! `MetalBackendOps` の各演算メソッドは #930 で `MetalContext`／
//! `MetalGemm` 等の都度構築（診断 #927 が特定した約 5 ms・N 非依存の
//! 固定オーバーヘッドの主因）をやめ、`context_cache` 経由のプロセス内
//! キャッシュから取得するようになった。本ファイルはこの効果を
//! 「同一プロセス内で 2 回目以降の `MetalBackendOps::gemm` 呼び出しが
//! 初回（cold）より速い」ことを cold/warm 計測で記録する（CUDA 側
//! `crates/facade/tests/tape_cuda_cache_bench.rs`〈イシュー #929〉と同じ
//! 「実機必須・5 回計測中央値・タイミング値は record only（hard assert
//! しない）」方針。`.claude/rules/coding-rust.md`「ベンチは 5 回計測の
//! 中央値」）。
//!
//! facade 側（`crates/facade/tests/`）ではなく本クレート直下に置く理由:
//! 並列実行中のイシュー #929 が `crates/facade/tests/` を編集しており、
//! 同一ファイル・ディレクトリの並行編集衝突を避けるため（実装計画 §3.6・
//! `.claude/rules/delegation-impl.md`「複数 Agent に同一ファイルを並行
//! 編集させない」）。検証対象自体も `backend-metal` クレート内部の
//! `context_cache` であり、`MetalBackendOps` を直接使えば facade を
//! 経由する必要がない。
//!
//! # 「cold」をプロセス内で 1 回しか観測できないことについて
//!
//! `context_cache` は `static + OnceLock` によるプロセスワイド
//! シングルトンであるため、同一テストバイナリ内で「cold」状態を複数回
//! 再現することはできない（2 回目以降は必ずキャッシュヒットになる）。
//! そのため cold は 1 回だけ計測し、warm はその後の `WARM_TRIALS` 回
//! （5 回計測中央値方針）を計測する非対称な設計とする（CUDA 側
//! `tape_cuda_cache_bench.rs` と同じ理由）。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する
//! （`backend_ops_real_device.rs` と同方針）。
//!
//! Linux CI での型検査（実機なしでもコンパイル可能性を担保）:
//!
//! ```sh
//! cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin
//! ```
//!
//! 実行コマンド（Apple Silicon 実機。`--release` 推奨）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --test context_cache_bench -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `--test-threads=1` が必須の理由: 本ファイルの
//! `gemm_second_call_avoids_reconstruction_cost` は「プロセス内で最初の
//! `MetalBackendOps::gemm` 呼び出し」を cold として計測する前提であり
//! （「cold をプロセス内で 1 回しか観測できないことについて」節参照）、
//! 同じテストバイナリ内の `gemm_repeated_calls_are_deterministic_via_cache`
//! と並列実行されると、どちらが先に Metal 経路へ触れるかが不定になり
//! cold 側の計測値が汚染される（`context_cache` のプロセスワイド
//! シングルトン性質上、後から実行された方は必ずキャッシュヒットになる
//! ため）。`--test-threads=1` で直列化するか、
//! `-- --ignored gemm_second_call_avoids_reconstruction_cost` のように
//! 対象テストを単独指定して実行すること。

#![cfg(target_os = "macos")]

use std::time::Instant;

use bench_harness::median_q1_q3;
use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

/// 5 回計測中央値方針（`.claude/rules/coding-rust.md`）。
const WARM_TRIALS: usize = 5;

fn sample_tensor() -> Tensor<f32> {
    // 小さめの正方行列。本ベンチの主目的は GEMM 自体のスループットでは
    // なく「演算メソッド呼び出し」の cold/warm 差であるため、形状は
    // カーネル起動が成立する最小限に留める（CUDA 側ベンチと同じ判断）。
    Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("sample tensor: shape 一致")
}

/// 1 回分の `MetalBackendOps::gemm` 呼び出しを計測する。
fn measure_gemm_call(ops: &MetalBackendOps) -> f64 {
    let a = sample_tensor();
    let b = sample_tensor();
    let t = Instant::now();
    let _out = ops
        .gemm(&a, &b)
        .expect("Metal 実機（本ベンチは #[ignore] 実機専用）では成功するはず");
    t.elapsed().as_secs_f64()
}

/// 受け入れ条件 1: 2 回目以降の `MetalBackendOps::gemm` 呼び出しが
/// `MetalContext`／`MetalGemm` の構築コスト（デバイス取得・MSL コンパイル）
/// を支払わないことを cold（プロセス最初の呼び出し）対 warm（5 回計測
/// 中央値）の方向性（`warm_median < cold`）で記録する。
///
/// 絶対閾値でのアサーションは行わない（本ファイル冒頭コメント「record
/// only」方針）。CI（Linux・ホステッドランナー）では実行されない
/// （`#[ignore]`）。
#[test]
#[ignore = "Metal 実機必須。イシュー #930 受け入れ条件 1（cold/warm 固定オーバーヘッド比較）"]
fn gemm_second_call_avoids_reconstruction_cost() {
    let ops = MetalBackendOps::new();

    // cold: プロセス内でこの呼び出しが最初の `MetalBackendOps::gemm`
    // 呼び出しであることが本テストの前提（本ファイル冒頭コメント「cold を
    // プロセス内で 1 回しか観測できないことについて」参照）。同一テスト
    // バイナリ内の他テストが先に Metal 経路を触っていると cold 側の値が
    // 汚染されるため、本テストは他の Metal 実機テストと同時実行しない
    // ことを前提とする（`cargo test -- --ignored --test-threads=1` 等）。
    let cold_secs = measure_gemm_call(&ops);

    let mut warm_samples = Vec::with_capacity(WARM_TRIALS);
    for _ in 0..WARM_TRIALS {
        warm_samples.push(measure_gemm_call(&ops));
    }
    let warm_q = median_q1_q3(&warm_samples)
        .expect("WARM_TRIALS 個の non-NaN warm サンプルは quartiles を持つはず");

    println!(
        "[context_cache_bench] cold_s={cold_secs:.6} \
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

/// キャッシュヒット経路の数値同一性回帰: 連続 2 回の `MetalBackendOps::gemm`
/// が同一結果を返す（`context_cache` 経由でも既存カーネル・許容誤差・
/// 境界検査には一切触れていないことの直接検証。REQ-2 複合判定ではなく
/// 「同一プロセス内で厳密に同じ入力を 2 回投げたら厳密に同じ出力になる」
/// という決定的性質を確認する。CUDA 側 parity テストとは目的が異なる
/// ため許容誤差は使わず `assert_eq!` で厳密一致を要求する）。
#[test]
#[ignore = "Metal 実機依存。CI では実行しない"]
fn gemm_repeated_calls_are_deterministic_via_cache() {
    let ops = MetalBackendOps::new();
    let a = sample_tensor();
    let b = sample_tensor();

    let first = ops.gemm(&a, &b).expect("1 回目の gemm は成功するはず");
    let second = ops
        .gemm(&a, &b)
        .expect("2 回目の gemm（キャッシュヒット経路）も成功するはず");

    assert_eq!(
        first.as_slice(),
        second.as_slice(),
        "context_cache 経由でも同一入力に対する gemm 出力は厳密に一致するはず"
    );
}
