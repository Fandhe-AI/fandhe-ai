# GEMM ベンチ

`Var::matmul`（GEMM）の最小計測例です。ウォームアップ 1 回の後、5 回
計測し中央値を採用します（[性能の考え方](/guides/performance/)の
計測規約）。CI（GitHub ホステッド・実機 GPU 非搭載）でも数秒以内に
終わるよう N=256 の正方 GEMM に抑えています。

```rust
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
```

**必ず `--release` を付けて実行してください**（`debug` ビルドは最適化
なしの実行速度を測るだけで GEMM 演算の性能デモとしては意味を持ちません）。

```
cargo run --release -p facade --example gemm_bench
```

このコードブロックは `crates/facade/examples/gemm_bench.rs` の実行
コード部分（冒頭のモジュールドキュメンテーションコメントを除く `use`
以降）と同一です。以下は上記コマンドでの実行確認済みの出力例です
（`release` プロファイル・実行環境依存の実測値であり、計測値そのもの
は環境ごとに変わります）。

```
N=256 median=383.917µs GFLOP/s=87.400
本格計測（性能下限判定・回帰検出）は criterion ベースの bench-harness の領分です (docs/performance-targets.md)。
```

**この数値は REQ-8 の性能下限判定の実測値ではありません。** 計測値
そのもの（`median`・`GFLOP/s`）は実行環境の CPU 性能・ビルド設定に
依存するため、原稿としては「5 回計測の中央値を取る手順」が再現できる
ことを主眼にしています。性能下限（REQ-8）判定を目的とした正式な本格
計測は
[`docs/performance-targets.md`](https://github.com/Fandhe-AI/rust-ai-library/blob/main/docs/performance-targets.md)
と `bench-harness` クレート（criterion ベース）の領分です。
