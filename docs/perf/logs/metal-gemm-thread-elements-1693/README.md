# thread_elements() 方式 BlockMMA 候補（イシュー #1693）実測ログ置き場

本ディレクトリは Mac セッション（#1694）が実機実測ログを置くための
記入欄。本 PR 時点ではログは未生成（本エージェント実行環境に Apple
Silicon 実機がないため）。

## 保存すべきファイル一覧（予定）

- `probe_run.log`: `crate::gemm::tests::te_layout_probe_matches_model`
  の実行ログ（R0 前提ゲート）。
- `parity_run.log`: `tests/gemm_te_parity.rs` 全ケース
  （`--ignored --nocapture --test-threads=1`）の実行ログ。
- `all_staged_candidates_run.log`:
  `crate::gemm::tests::all_staged_candidates_match_te_cpu_reference_
  512_nn`（クレート内テスト）の実行ログ。
- `bit_match_run.log`: `te_bit_match_with_production_dispatch_auto`
  の実行ログ。
- `env_info.txt`: 実機情報（GPU コア数・OS バージョン等。内部ホスト名は
  含めない）。

## 実行コマンド

```sh
# クレート内テスト（probe・all_staged_candidates）
cargo test -p fandhe-ai-backend-metal --release --lib gemm::tests:: \
  -- --ignored --nocapture --test-threads=1

# 外部テスト（parity・bit 一致・非 staged 拒否）
cargo test -p fandhe-ai-backend-metal --release --test gemm_te_parity \
  -- --ignored --nocapture --test-threads=1
```
