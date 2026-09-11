# Metal split-K GEMM: 経路別誤差の f64 真値突合（診断専用・イシュー #1549）

## 目的

`docs/backend-metal-splitk-two-pass.md`・`docs/backend-metal-splitk-parity-judgment-decision.md`
は split-K 経路と CPU f32 参照実装（`matmul_reference_fma`）を
`assert_no_split_k_parity_regression`（実測ベースライン非後退方式）で
比較しているが、この比較は「split-K がどちらの方向にどれだけ
`f64` 厳密解からずれているか」までは示さない。本記録は split-K 到達
11 形状で、以下 3 経路の出力を `f64` 真値と突き合わせた誤差
（`max_abs`・`mean_abs`・`max_rel`。いずれも `f64` で算出）を実測し、
事実のみを記録する。

- `split_k`: split-K 2 パス経路（`MetalGemm::new_with_split_k_auto(&ctx, true)`
  → `dispatch_auto_with_route`。`GemmRoute::SplitK` への到達を assert）
- `classic`: classic 経路（`MetalGemm::new_with_split_k_auto(&ctx, false)`
  → `dispatch_auto_with_route`。`GemmRoute::Classic` への到達を assert）
- `cpu_f32`: CPU 参照実装（`fandhe_ai_backend_cpu::parity::matmul_reference_fma`）

**本記録は判定基準を持たない**。`assert_no_split_k_parity_regression` の
baseline・tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・
既存テストはいずれも変更していない。良し悪しの判断・baseline 変更の提案は
含まない。

## 方式

- **対象 11 形状**: `crates/backend-metal/tests/gemm_splitk_parity.rs::TARGET_SHAPES`
  と同一（`docs/backend-metal-splitk-decision.md` §3 の対象 9 形状 +
  K 端数を含む境界ケース 2 点）
- **入力生成**: `bench_harness::rng::Xorshift64Star::new(seed).fill_vec(len)`。
  シード `a: m*7+k+1`／`b: n*11+k+2`（`gemm_splitk_parity.rs` と完全に
  同一の生成方式。既存の記録済み baseline と同じ入力に基づく誤差として
  紐付けられるようにするため）
- **転置パターン**: NN（非転置）のみ。`gemm_splitk_parity.rs` が検証する
  NT/TN/TT は対象外（目的が経路間の数値特性の参考比較であり、全転置
  パターンを網羅する必要はないと判断したため）
- **f64 真値の定義**: f32 入力を `f64` へ昇格し、i 外側・k 中間・j 内側の
  逐次加算（`c[i][j] += a[i][p] as f64 * b[p][j] as f64`）で求める厳密解
  （`crates/backend-cuda/tests/specialized_mma_f16_triage.rs::exact_reference_f64`
  の f32 版。`f32::mul_add` によるまとめ丸めを経由しない）
- **誤差式**（`compare()` の複合判定とは異なる診断専用の指標。分母は真値）:
  - `abs = |actual as f64 - truth|`
  - `rel = abs / max(|truth|, 1e-12)`
  - `max_abs`／`mean_abs`／`max_rel` は全要素に対する集計値
- **assert する項目**（誤差の閾値 assert はしない）:
  - split-K 経路が実際に `GemmRoute::SplitK` へ到達したこと（フォールバック
    による自明合格の排除）・classic 経路が `GemmRoute::Classic` へ到達したこと
  - 出力長が `m * n` と一致すること
  - 各経路の出力に `NaN` が含まれないこと

実装: `crates/backend-metal/tests/gemm_splitk_f64_truth.rs`
（`#[ignore]`・`internal-diagnostics` feature 限定）。

実機実行:
```sh
cargo test -p fandhe-ai-backend-metal --release --features internal-diagnostics \
  --test gemm_splitk_f64_truth -- --ignored --nocapture --test-threads=1
```

## 事前登録した報告項目（判定基準なし）

1. 形状 × 経路ごとの `max_abs`／`mean_abs`／`max_rel`（f64 真値比）
2. 「split-K が classic より `f64` 真値から遠い形状数」（`max_abs` 基準・
   `mean_abs` 基準それぞれで集計）
3. run 間の決定性（同一バイナリを 2 回実行し数値が一致するか）

## 実測結果（M4 Max・run1）

`docs/perf/logs/metal-gemm-splitk-f64-truth-1549/run1.log` から転記
（`env_info.txt` に実機・環境情報）。

| m | n | k | route | max_abs | mean_abs | max_rel |
|---|---|---|-------|---------|----------|---------|
| 32 | 32 | 2048 | split_k | 8.8279008708e-6 | 1.7636333815e-6 | 1.2555909295e-4 |
| 32 | 32 | 2048 | classic | 8.1630067996e-5 | 8.8185123303e-6 | 2.3078407246e-4 |
| 32 | 32 | 2048 | cpu_f32 | 8.1630067996e-5 | 8.8185123303e-6 | 2.3078407246e-4 |
| 32 | 32 | 4096 | split_k | 1.6941246059e-5 | 3.5593857532e-6 | 8.3513665195e-5 |
| 32 | 32 | 4096 | classic | 1.9942149009e-4 | 1.6399941070e-5 | 1.8023278501e-4 |
| 32 | 32 | 4096 | cpu_f32 | 1.9942149009e-4 | 1.6399941070e-5 | 1.8023278501e-4 |
| 32 | 32 | 8192 | split_k | 3.2211228486e-5 | 6.7130147743e-6 | 1.6212616405e-4 |
| 32 | 32 | 8192 | classic | 3.4245093747e-4 | 3.4137445036e-5 | 9.9169591591e-4 |
| 32 | 32 | 8192 | cpu_f32 | 3.4245093747e-4 | 3.4137445036e-5 | 9.9169591591e-4 |
| 64 | 64 | 2048 | split_k | 8.4214700138e-6 | 1.7418342511e-6 | 1.5718227561e-3 |
| 64 | 64 | 2048 | classic | 9.3996977384e-5 | 8.5732995160e-6 | 3.9098382407e-3 |
| 64 | 64 | 2048 | cpu_f32 | 9.3996977384e-5 | 8.5732995160e-6 | 3.9098382407e-3 |
| 64 | 64 | 4096 | split_k | 1.5760041137e-5 | 3.5232744735e-6 | 9.7952118142e-5 |
| 64 | 64 | 4096 | classic | 2.0033651511e-4 | 1.6974821647e-5 | 8.4868498624e-4 |
| 64 | 64 | 4096 | cpu_f32 | 2.0033651511e-4 | 1.6974821647e-5 | 8.4868498624e-4 |
| 64 | 64 | 8192 | split_k | 3.7671708242e-5 | 6.9035408296e-6 | 1.2266085684e-3 |
| 64 | 64 | 8192 | classic | 3.0952430320e-4 | 3.4516822997e-5 | 3.4517055705e-3 |
| 64 | 64 | 8192 | cpu_f32 | 3.0952430320e-4 | 3.4516822997e-5 | 3.4517055705e-3 |
| 128 | 128 | 2048 | split_k | 2.2176054273e-5 | 3.4114887911e-6 | 2.4540148835e-3 |
| 128 | 128 | 2048 | classic | 1.2039732289e-4 | 8.4983176606e-6 | 1.4612180883e-3 |
| 128 | 128 | 2048 | cpu_f32 | 1.2039732289e-4 | 8.4983176606e-6 | 1.4612180883e-3 |
| 128 | 128 | 4096 | split_k | 2.7179620005e-5 | 4.8681089732e-6 | 1.4949627184e-3 |
| 128 | 128 | 4096 | classic | 1.5987782444e-4 | 1.7205961615e-5 | 7.0472944421e-3 |
| 128 | 128 | 4096 | cpu_f32 | 1.5987782444e-4 | 1.7205961615e-5 | 7.0472944421e-3 |
| 128 | 128 | 8192 | split_k | 3.6175522553e-5 | 6.8370959737e-6 | 1.2258185091e-3 |
| 128 | 128 | 8192 | classic | 4.0485548175e-4 | 3.4482836523e-5 | 1.5771180105e-3 |
| 128 | 128 | 8192 | cpu_f32 | 4.0485548175e-4 | 3.4482836523e-5 | 1.5771180105e-3 |
| 64 | 64 | 2056 | split_k | 8.4880601179e-6 | 1.7511014706e-6 | 4.3040887174e-4 |
| 64 | 64 | 2056 | classic | 1.2723484838e-4 | 8.8683549017e-6 | 9.9239843466e-4 |
| 64 | 64 | 2056 | cpu_f32 | 1.2723484838e-4 | 8.8683549017e-6 | 9.9239843466e-4 |
| 128 | 128 | 2064 | split_k | 2.0713349343e-5 | 3.4655187989e-6 | 1.3032191297e-1 |
| 128 | 128 | 2064 | classic | 1.1940125553e-4 | 8.6633565686e-6 | 1.9462941386e-1 |
| 128 | 128 | 2064 | cpu_f32 | 1.1940125553e-4 | 8.6633565686e-6 | 1.9462941386e-1 |

集計:

```
split_farther_than_classic_by_max_abs=0/11
split_farther_than_classic_by_mean_abs=0/11
```

## run1 と run2 の一致

同一バイナリを 2 回実行し（`docs/perf/logs/metal-gemm-splitk-f64-truth-1549/run1.log`・
`run2.log`）、数値行・集計行を `diff` で突合した結果、**完全に一致した**
（決定的）。

## 所見（事実のみ）

- 対象 11 形状すべてで、`classic` と `cpu_f32` の誤差指標（`max_abs`・
  `mean_abs`・`max_rel`）は表示桁（10 桁）で一致した（`gemm_splitk_
  parity.rs` の doc コメントが記す「classic 経路は全対象形状で CPU
  参照実装と bit 完全一致する」という既知の事実〈本テストでは bit 単位の
  突合自体は行っていない〉と整合する）
- 対象 11 形状すべてで、`split_k` の `max_abs`／`mean_abs` は `classic`／
  `cpu_f32` より小さかった（`split_farther_than_classic_by_max_abs=0/11`・
  `split_farther_than_classic_by_mean_abs=0/11`）。すなわち本実測の範囲
  （対象 11 形状・NN・単一乱数系列）では、split-K 経路の出力は classic
  経路・CPU f32 参照実装よりも `f64` 厳密解に近かった
- `max_rel` は 11 形状中 10 形状で `split_k` が `classic`／`cpu_f32` を
  下回った。唯一の例外は `(128,128,2048)` で、`split_k` の `max_rel`
  （2.4540148835e-3）が `classic`／`cpu_f32` の `max_rel`
  （1.4612180883e-3）を上回った（`max_abs`／`mean_abs` は同形状でも
  `split_k` が下回っている。表参照）
- `(128,128,2064)` は 3 経路とも `max_rel` が他形状より 2〜3 桁大きい
  （`split_k`: 0.130・`classic`/`cpu_f32`: 0.195）。`max_rel` の分母が
  真値の絶対値であるため、真値が 0 に近い要素が存在すると相対誤差が
  跳ね上がりうる（`compare()` の絶対誤差救済閾値と同種の性質）。この
  形状・`(128,128,2048)` の `max_rel` 逆転を含め、原因の切り分けは
  本記録のスコープ外とする

## ログ・env_info

- `docs/perf/logs/metal-gemm-splitk-f64-truth-1549/run1.log`
- `docs/perf/logs/metal-gemm-splitk-f64-truth-1549/run2.log`
- `docs/perf/logs/metal-gemm-splitk-f64-truth-1549/env_info.txt`
