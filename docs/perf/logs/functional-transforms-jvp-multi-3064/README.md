# 複数入力 jvp（`Tape::jvp_multi`。#3064）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-functional-transforms-design.md` §30 参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、`crates/facade/tests/functional_transforms_jvp_multi_backend_parity.rs` の 2 テスト
（`cuda_jvp_multi_matches_cpu_reference`・`metal_jvp_multi_matches_cpu_reference`。後者は `cfg(target_os = "macos")` 限定）
は `#[ignore]` のまま**未実測**である。実測は #3171（GB10）・#3172（Mac mini）が担当する。

同テストは本物の facade `Tape::jvp_multi` を `tape_for(device)` の実機 tape で呼び、同デバイスの `jacobian` 由来の
`Σ J_i·v_i` 参照および `Device::Cpu` の結果と比べる。`functional_ops_jvp_backend_parity.rs`（#2942 のミラー）は置き換えない。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test functional_transforms_jvp_multi_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test functional_transforms_jvp_multi_backend_parity -- --ignored --nocapture metal
```

注記: `scripts/test-ignored.sh metal-facade` はテストバイナリを名前で絞っており本テストを含まない。上記の直接コマンドを使う。
CPU 版は `crates/facade/tests/functional_transforms_jvp_facade.rs` の属性なしテスト（`multi_input_*`）で検証済み（green）。

## 期待結果

- 新規カーネルは存在しない。`jvp_multi` は既存 Op（add・matmul・tanh・mul・sum 等）の合成であり、確認対象は
  「既存カーネルの新しい呼び出し形（複数の追跡あり入力を持つ親 `backward_create_graph` と子テープ上の合計 1 回の
  backward）が CPU と同じ結果になること」である。
- 判定は REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
  `fandhe_ai_backend_cpu::parity::assert_parity`）。形状は小さく、Metal split-K が発動する形状は使っていない。
- 判定を外れた場合は事実を記録して申し送る（tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。
  `Unsupported` 等の型付きエラーが出た場合も推測で直さず事実を記録する。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測（#3171） | DGX Spark GB10 | 上記 cuda コマンド | 未実測 | |
| 未実測（#3172） | Apple Silicon（Mac mini） | 上記 metal コマンド | 未実測 | |
