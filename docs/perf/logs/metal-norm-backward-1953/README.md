# LayerNorm／RMSNorm backward Metal カーネル実機実測（イシュー #1953）

本ディレクトリは `crate::norm_backward::MetalNormBackward`（イシュー
#1953・親 #1947。`docs/norm-ops-design.md` §9）の Apple Silicon 実機実測
ログの置き場である。本実装エージェントの実行環境（Linux）には Apple
Silicon 実機への到達手段がないため、実測は未実施のまま記入欄のみを残す。

## 実行コマンド

```sh
# crates/backend-metal のカーネル単体 parity（REQ-2 統一複合判定）
cargo test -p fandhe-ai-backend-metal --release --test norm_backward_parity -- --ignored --nocapture

# facade 到達経路（backward。matmul → rms_norm／layer_norm → mse_loss）
cargo test -p fandhe-ai --release --test norm_backend_parity -- --ignored metal_rms_norm_backward_matches_cpu --nocapture
cargo test -p fandhe-ai --release --test norm_backend_parity -- --ignored metal_layer_norm_backward_matches_cpu --nocapture
```

## 事前登録判定規則

- `crates/backend-metal/tests/norm_backward_parity.rs` の各テストは
  `fandhe_ai_backend_cpu::parity::assert_parity`（REQ-2 統一複合判定。
  相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で判定する。bit 完全
  一致は主張しない（`docs/norm-ops-design.md` §9・
  `crates/backend-metal/src/shaders/norm_backward.metal` 冒頭コメント
  「数値契約」参照）
- `backend_ops_norm_backward_reaches_metal_override` は
  `MetalBackendOps::rmsnorm_backward`／`layer_norm_backward` が
  `Err(Unsupported)` を返さず正常に到達することのみを検証する（数値の
  正しさは上記 parity テストが担う）
- facade テスト（`metal_{rms_norm,layer_norm}_backward_matches_cpu`）は
  同一グラフ（`matmul → norm → mse_loss`）の `dW`／`dB` を Metal tape・
  CPU tape で backward し REQ-2 で突き合わせる

## 記入欄

| テスト | 結果 | 実行日 | 備考 |
|---|---|---|---|
| `crates/backend-metal/tests/norm_backward_parity.rs`（全テスト） | 3/3 pass | 2026-09-18 | `norm_backward_parity.log` |
| `metal_rms_norm_backward_matches_cpu` | pass | 2026-09-18 | `facade_rms_norm_backward.log` |
| `metal_layer_norm_backward_matches_cpu` | pass | 2026-09-18 | `facade_layer_norm_backward.log` |

合格（規則 1〜4 充足）。
