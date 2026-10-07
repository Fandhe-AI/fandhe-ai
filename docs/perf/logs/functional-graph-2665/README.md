# Functional API（多入力グラフの構築と forward。#2665）CUDA／Metal 実機未実測の申し送り

`docs/facade-functional-api-decision.md` 末尾「#2665 実装記録」参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、`crates/facade/src/compat/functional/tests.rs` の CUDA（`Device::Cuda(0)`）・Metal
（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_functional_graph_matches_cpu_reference`・`metal_functional_graph_matches_cpu_reference`）は
`#[ignore]` のまま未実測である。実装が `#[cfg(test)]` 限定の `pub(crate)` のため、結合テスト（`tests/`）ではなく
クレート内ユニットテスト（`--lib`）として実行する。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --lib -- --ignored --nocapture cuda_functional_graph_matches_cpu_reference

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --lib -- --ignored --nocapture metal_functional_graph_matches_cpu_reference
```

CPU 版は同ファイルの属性なしテスト（`matches_pytorch_reference_outputs_and_input_grads`。PyTorch 2.14.0 実行値との
統一複合判定）で検証済み（green）。

## 期待結果

- 本実装は新規 GPU カーネルを持たない（新規 `Op`・`BackendOps` メソッドなし）。実機で走るのは **既存の `Var` 演算**
  （`gemm`・bias 加算・活性化・各 VJP）であり、確認対象は新規カーネルの parity ではなく既存カーネルの新しい呼び出し形
  （fan-out する上流ノードの勾配合流）である。
- fixture の `fan_out` ケース（入力 1・共有ブロック 1・分岐ブロック 2・出力順は挿入順の逆）を CPU tape と実機 tape で
  同じ重みのまま forward／backward し、出力と入力勾配が REQ-2 統一複合判定（相対誤差 1e-3 未満 または
  絶対誤差 1e-5 未満）を満たす見込み。
- 形状は小さく（バッチ 3・幅 ≤ 6）、Metal の split-K 経路が発動する形状は含まない。

この前提が崩れる場合は本 README の「期待結果」を更新し、REQ-2 判定を外れた事実を PR へ記録すること
（tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 CUDA | 未実測 | |
| 未実測 | Apple Silicon（M4 Max） | 上記 Metal | 未実測 | |
