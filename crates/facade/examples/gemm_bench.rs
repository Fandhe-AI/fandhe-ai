//! `Var::matmul`（GEMM）の最小計測例（イシュー #875）。
//!
//! サイト原稿（`site/examples/gemm-bench.md`）に転記するコード例の一次
//! ソース（`getting_started.rs`〈#874〉と同じ理由で二重実装を避ける。
//! `.claude/rules/code-comment-style.md`）。本 example の実行成功
//! （`cargo run -p facade --example gemm_bench`）が原稿の受け入れ条件
//! （コード例がコンパイル・動作確認済みであること）を担保する。
//!
//! **計測規約**: ウォームアップ 1 回の後、5 回計測し中央値を採用する
//! （`.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値を
//! 採用」）。CI（GitHub ホステッド・実機 GPU 非搭載）でも数秒以内に
//! 終わるよう N=256 の正方 GEMM に抑える。**`--release` を付けて実行
//! すること**（`cargo run --release -p facade --example gemm_bench`）:
//! debug ビルドの計測値は最適化なしの実行速度を測るだけで GEMM 演算の
//! 性能デモとしては意味を持たないため。
//!
//! **本格計測との違い**: 本 example は `std::time::Instant` による
//! 簡易計測のデモであり、性能下限（REQ-8）判定・回帰検出を目的とした
//! 本格計測は `criterion`（`dev-dependencies` 限定。`.claude/rules/
//! deps-policy.md`）を使う `bench-harness` クレートの領分である
//! （`docs/performance-targets.md`）。

use facade::Tensor;
use std::time::Instant;

const N: usize = 256;

fn build_matmul_inputs() -> Result<(Tensor<f32>, Tensor<f32>), Box<dyn std::error::Error>> {
    // 決定的な固定値（外部入力・乱数への依存を避け、計測対象を GEMM 演算
    // 自体に絞る。`.claude/rules/security.md` A03「外部入力を検証」と
    // 同じ理由でここでは外部入力そのものを持ち込まない）。
    let a = Tensor::new(vec![0.01_f32; N * N], &[N, N])?;
    let b = Tensor::new(vec![0.02_f32; N * N], &[N, N])?;
    Ok((a, b))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (a_data, b_data) = build_matmul_inputs()?;

    // ウォームアップ 1 回（アロケータ・キャッシュの初回コストを計測対象
    // から除く）。
    {
        let tape = facade::tape();
        let a = tape.var(&a_data);
        let b = tape.var(&b_data);
        let _ = a.matmul(&b)?;
    }

    let mut durations = Vec::with_capacity(5);
    for _ in 0..5 {
        // 計測ごとに新しい Tape を構築する（前回計測の計算グラフを
        // 引き継いで累積させないため。同一 Tape を使い回すとステップ数
        // に応じてグラフ管理コストが増え、計測対象の GEMM 単体コストと
        // 混ざってしまう）。
        let tape = facade::tape();
        let a = tape.var(&a_data);
        let b = tape.var(&b_data);

        let start = Instant::now();
        let _ = a.matmul(&b)?;
        durations.push(start.elapsed());
    }

    durations.sort();
    let median = durations[durations.len() / 2];

    let flops = 2.0 * (N as f64).powi(3);
    let gflops = flops / median.as_secs_f64() / 1e9;

    println!("N={N} median={median:?} GFLOP/s={gflops:.3}");
    println!(
        "本格計測（性能下限判定・回帰検出）は criterion ベースの bench-harness の領分です \
         (docs/performance-targets.md)。"
    );

    Ok(())
}
