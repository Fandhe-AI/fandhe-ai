# イシュー #1736 実測記録先（BatchNorm1d／2d の Metal train／infer 融合カーネル）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree・x86_64）には Apple Silicon
実機への到達手段（`docs/real-hardware-verification-env.local.md`・
`METAL_NODE` 環境変数・`~/.ssh/config` のいずれも確認できず）がないため、
**実測値は一切含まれていない**。本ディレクトリは実行コマンド・保存
すべきログ一覧・事前登録判定規則のみを提供し、実測は Apple Silicon
実機を持つセッションへ申し送る（`docs/perf/logs/metal-conv1d-1769/
README.md` と同型の運用）。

## 目的

イシュー #1736「BatchNorm1d／2d（train／eval・running stats）を
Metal で実装する」の実機正しさ検証記録先。`docs/batch-norm-ops-
design.md` §9 が設計・数値契約（soft-f64 方式。issue 題名の Neumaier
＋scale/ssq は不採用——理由は同 §9.1）を確定済み。本ディレクトリは
Apple Silicon 実機上での REQ-2 統一複合判定・run-to-run 決定性・
既存 `#[ignore]` 群の非後退確認の受け皿。性能最適化・実測は本イシュー
のスコープ外（対象外事項は同 §9.7）。

## 実行コマンド

```bash
# 1) crates/backend-metal の BatchNorm train／infer REQ-2 突合
#    （形状網羅・極端値・NaN 伝播・決定性・非 contiguous・エラー写像・
#    空軸早期 return）
cargo test -p fandhe-ai-backend-metal --release --test batch_norm_parity -- --ignored --nocapture

# 2) crates/facade の BatchNorm train／infer／backward の Metal vs CPU 突合
#    （facade 公開面のみを import する統合テスト）
cargo test -p fandhe-ai --release --test batch_norm_backend_parity -- --ignored --nocapture

# 3) 既存 #[ignore] 群の非後退確認（本イシューの変更が他カーネルへ
#    影響していないことの確認。特に layer_norm・rmsnorm・softmax 等の
#    row_kernel 共有系）
make test-ignored-metal
```

## 事前登録判定規則

- **正しさ**: `crates/backend-metal/tests/batch_norm_parity.rs`・
  `crates/facade/tests/batch_norm_backend_parity.rs` の `#[ignore]`
  テストが全て pass（`fandhe_ai_backend_cpu::parity::assert_parity`
  によるREQ-2 統一複合判定。`batch_mean` の bit 一致は `--nocapture`
  出力で報告のみに留め assert しない——`docs/batch-norm-ops-design.md`
  §9.3「判定契約」参照）
- **決定性**: `batch_norm_train_is_deterministic_across_runs` が
  run-to-run で bit 単位一致
- **非後退**: 本イシュー変更前から存在する `#[ignore]` テスト群
  （`layer_norm_parity.rs`・`rmsnorm_parity.rs`・`softmax_parity.rs`
  等）が引き続き全て pass すること（`shaders/batch_norm.metal` は
  独立ファイルのため通常は無関係だが、`context_cache.rs`・`lib.rs`
  変更の副作用がないことを機械的に確認する）

tolerance／baseline 定数の変更はなし（本イシューでは提案・実施しない）。

## 保存すべきログ

実測を実施するセッションは、下記をこのディレクトリへ保存すること
（内部ホスト名は含めない）:

- `batch_norm_parity.log`（上記コマンド 1 の全出力）
- `batch_norm_backend_parity.log`（上記コマンド 2 の全出力）
- `test_ignored_metal_before.log`／`test_ignored_metal_after.log`
  （既存 `#[ignore]` 群の非後退確認。本変更前後の比較が可能な場合）
- `env_info.txt`（`sw_vers`・`sysctl -n machdep.cpu.brand_string`・
  `system_profiler SPDisplaysDataType | grep Chipset` 等。内部ホスト名
  は含めない）
