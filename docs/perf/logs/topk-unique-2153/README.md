# topk_unique_ops（#2153）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-topk-unique-ops-decision.md` §7・§8「実装記録」参照。
本実装エージェント実行環境は CUDA／Metal 実機に到達できないため、
`fandhe_ai_autodiff::topk_unique_ops`（`topk_with_options`・
`unique_with_options`・`unique_consecutive`）の
`crates/facade/tests/topk_unique_ops_backend_parity.rs` のうち CUDA
（`Device::Cuda(0)`）・Metal（`Device::Metal`。`cfg(target_os =
"macos")` 限定）を対象とする 4 テスト（`unique_with_options`・
`topk_with_options（sorted=false）` × CUDA／Metal）は `#[ignore]` の
まま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test topk_unique_ops_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test topk_unique_ops_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト
（`cpu_topk_sorted_false_negative_dim_forward_bit_matches_naive_reference`・
`cpu_topk_sorted_false_gradient_bit_matches_naive_reference`・
`cpu_unique_with_options_dim_none_bit_matches_naive_reference`・
`cpu_unique_with_options_dim_specified_bit_matches_naive_reference`・
`cpu_unique_consecutive_bit_matches_naive_reference`）で既に検証済み
（green）。

## 期待結果

- `topk_with_options`（`sorted=false`）・`unique_with_options`・
  `unique_consecutive` はいずれも選択演算（丸めなし）のため、
  CUDA／Metal でも CPU と **bit 完全一致**するはず（REQ-2 複合判定は
  用いない）。
- `unique_ext`（`BackendOps`）は CUDA／Metal で override していない
  ため既定 `Unsupported` を返し、実機テストは常にホスト参照実装
  （`fandhe_ai_autodiff::eval::unique_ext`）へフォールバックする経路
  になる想定——CUDA／Metal 側のホスト↔デバイス転送を経由しても
  ホスト計算自体は同一アルゴリズムのため bit 完全一致が崩れる理由が
  ない。もし fail する場合は転送経路（`f32`／`i32` の host readback）
  の丸め・順序に問題がある可能性があり、REQ-2 判定を緩めるのではなく
  原因調査を優先すること（`.claude/rules/coding-rust.md` テスト・
  ベンチ節）。
- `topk_with_options`（`sorted=false`）は既存 `topk`（`BackendOps::
  topk`）を経由するため、CUDA／Metal に `topk` の実カーネルが実装
  済みであればそちらを、`Unsupported` ならホストフォールバックを経由
  する。いずれの経路でも `resort_topk_by_index`（ホスト側の index
  昇順並べ替え）は同一実装のため bit 完全一致するはず。

## 実行後の記録方法

実機セッションで上記コマンドを実行し、結果（pass/fail・実測ログ）を
本ファイルへ追記する。fail した場合は診断結果を
`docs/autodiff-topk-unique-ops-decision.md` §8 へ反映し、必要なら
イシューを起票してユーザーへ報告する（ガードレール閾値・tolerance の
独自緩和は行わない）。
