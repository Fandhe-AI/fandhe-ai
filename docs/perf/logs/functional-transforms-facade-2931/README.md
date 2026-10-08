# facade `Tape::vjp`／`hvp`／`vmap`（#2931）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-functional-transforms-design.md` §23・§24 参照。本実装エージェント実行環境は
CUDA／Metal 実機に到達できないため、`crates/facade/tests/functional_transforms_facade.rs` のうち
CUDA（`Device::Cuda(0)`）・Metal（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_functional_transforms_match_cpu_reference`・`metal_functional_transforms_match_cpu_reference`）は
`#[ignore]` のまま未実測である。

内部 API 層（`fandhe_ai_autodiff::functional_ops`）の実機 parity は #2881 の
`docs/perf/logs/functional-transforms-2881/README.md` が担当し、本件は facade 層（`tape_for(device)` で作った
テープと子テープを `Tape::vjp`／`hvp`／`vmap` へ渡す形）を担当する。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test functional_transforms_facade -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test functional_transforms_facade -- --ignored --nocapture metal
```

CPU 版は同テストファイルの属性なしテスト（`facade_cpu_matches_internal_functional_ops` ほか）で検証済み
（内部実装の結果と shape・値が REQ-2 統一複合判定で一致）。

## 期待結果

- 新規カーネルは存在しない。facade の 3 メソッドは `functional-transforms-2881` と同じ既存カーネル
  （matmul・tanh・sigmoid・mul・sum・narrow／reshape・contiguous・concat 等）の合成を呼ぶだけで、確認対象は
  「実機テープ（子テープも同一デバイス）でも CPU と同じ shape・REQ-2 統一複合判定内の値になること」である。
- 形状は小さく、Metal split-K が発動する形状は使っていない。`hvp` と 1 階 VJP の bit 同一、`vmap` とバッチなし
  実行の bit 一致は契約にしない（設計記録 §23.3）。
- 前提が崩れる場合は本 README の「期待結果」を更新し、REQ-2 判定を外れた事実を PR へ記録すること
  （tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 CUDA | 未実測 | |
| 未実測 | Apple Silicon（M4 Max） | 上記 Metal | 未実測 | |
