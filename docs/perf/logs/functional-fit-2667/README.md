# Functional モデルの fit・保存（#2667）CUDA／Metal 実機未実測の申し送り

`docs/facade-functional-api-decision.md` §18「#2667 実装記録」参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、次の 2 テストは `#[ignore]` のまま未実測である。

| テスト | 場所 | 対象 |
|---|---|---|
| `cuda_parameter_gradients_match_cpu_reference`・`metal_parameter_gradients_match_cpu_reference`（`cfg(target_os = "macos")`） | `crates/facade/src/compat/functional/fit_parity_tests.rs`（`--lib`。実装が `#[cfg(test)]` 限定の `pub(crate)` のため結合テストでは実行できない） | fixture の `grad_*` 2 ケース（2 入力・fan-out・結合 4 種・多出力）の `bind → forward → backward` のパラメータ勾配と損失値を、CPU tape と `tape_for(device)` 上で同じ初期重みのまま比較する |

`fit` 自体は `Sequential::fit` と同じく `crate::tape()`（CPU 経路）で走るため実機テストの対象外
（`docs/compat-api-scope.md` §1.2）。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --lib -- --ignored --nocapture cuda_parameter_gradients_match_cpu_reference

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --lib -- --ignored --nocapture metal_parameter_gradients_match_cpu_reference
```

CPU 版は属性なしテスト（`parameter_gradients_match_pytorch`・`fit_trajectories_match_pytorch`〈PyTorch 2.14.0 実行値との
突合〉・`fit_tests.rs` の `Sequential` との bit 一致回帰）で検証済み（green）。

## 期待結果

- 本実装は新規 GPU カーネルを持たない（新規 `Op`・`BackendOps` メソッド・VJP なし）。実機で走るのは **既存の `Var` 演算**
  （`linear`・活性化・`concat`・`add`・`mul`・`div`・`mse_loss` と各 VJP）であり、確認対象は新規カーネルの parity ではなく
  既存カーネルの新しい呼び出し形（同一 tape 上の複数 `bind`・結合の出力を別のブロックへ入れる連鎖・fan-out による勾配合流）である。
- いずれも CPU tape と実機 tape のパラメータ勾配・損失が REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を
  満たす見込み。形状は小さく、Metal の split-K 経路は発動しない。

この前提が崩れる場合は本 README の「期待結果」を更新し、REQ-2 判定を外れた事実を PR へ記録すること
（tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 CUDA | 未実測 | |
| 未実測 | Apple Silicon（M4 Max） | 上記 Metal | 未実測 | |
