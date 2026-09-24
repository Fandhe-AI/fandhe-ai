# kv-cache（#2084）CUDA／Metal 実機未実測の申し送り

`docs/kv-cache-design.md`「9. 実装記録（#2084）」参照。本実装エージェント
実行環境は CUDA／Metal 実機に到達できないため、KV キャッシュ付き
attention（`MultiheadAttentionVars::forward_with_cache`）の
`crates/facade/tests/kv_cache_backend_parity.rs` のうち CUDA
（`tape_for(Device::Cuda(0))`）・Metal（`tape_for(Device::Metal)`。
`cfg(target_os = "macos")` 限定）を対象とする 2 テストは `#[ignore]` の
まま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test kv_cache_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test kv_cache_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は上記テストファイルの属性なしテスト
（`cpu_prefill_then_decode_matches_naive_reference`・
`cpu_prefill_then_decode_matches_full_recompute_on_cpu_backend`）で既に
検証済み（green。`docs/kv-cache-design.md` §9「parity 結果（CPU・観測値）」
参照）。

## 期待結果

`forward_with_cache` は新規 `Op`／`BackendOps` メソッド／カーネルを追加
せず、既存の `project`／`split_heads`／`sdpa_compose`（matmul／softmax／
masked_fill／`Var::cat`）の合成のみで構成される
（`docs/kv-cache-design.md` §3.1 contract 確認表）。したがって CUDA／
Metal 上でも、既存の `MultiheadAttentionVars::forward` の parity 契約
（REQ-2 統一複合判定。`mha_backend_parity.rs` の実機テストと同型）が
そのまま成立し、decode 列の最終 `cache.k()`／`cache.v()` が CPU と
REQ-2 一致するはず。新しい tolerance／baseline の導入はない。この前提が
崩れる場合（`Var::cat` のホストフォールバック経路・GPU 経路の中間形状
依存の丸め等で不一致が生じる場合）は本 README の「期待結果」を更新し、
不一致の実測を型付き findings として PR へ記録すること（tolerance の
単独緩和は行わない。`.claude/rules/coding-rust.md`）。
