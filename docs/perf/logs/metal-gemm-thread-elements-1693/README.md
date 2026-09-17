# thread_elements() 方式 BlockMMA 候補（イシュー #1693）実測ログ置き場

## 実測結果（2026-09-18・Apple M4 Max）

すべてのゲート（R0〜R3）が通過。実測ログは下記「保存すべきファイル一覧」に記録済み。詳細は `docs/perf/metal-gemm-thread-elements-candidate.md` §4 を参照。

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
