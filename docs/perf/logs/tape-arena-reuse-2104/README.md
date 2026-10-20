# tape・ホスト arena 再利用の A/B ログ（イシュー #2104）

状態: **GB10 は実測済み・判定 undetermined（2026-10-20・イシュー #3190。label `2104-gb10-3190`）**。M4 Max・Mac mini は未実測（#3184）。
GB10: train fresh 0.8648・train reuse 1.0172・infer fresh 0.9995・infer reuse 1.0035（checksum 全て完全一致・5/5 round が load1 < 8.0）。reuse の 2 セルが ratio > 1.00 だが 5/5 round 一貫ではなく、REJECT にも ADOPT にも当たらない。生ログは `*2104-gb10-3190*`、結論は `docs/perf/optin-ab-gb10.md` B1。

- 設計: `docs/tape-arena-reuse-design.md`
- 判定規則（実測前に固定）: `RULE.txt`

## 実行手順（各機体で 1 回ずつ。専有ゲートを通す）

1. before 用に main を checkout した worktree、after 用に同一コミットで `crates/tensor-core/src/alloc.rs` の `HOST_ARENA_DEFAULT_ENABLED` を `true` にした worktree を用意する（after は計測専用でコミットしない）。
2. `AB_BEFORE_FACADE_PATH=<before>/crates/facade AB_AFTER_FACADE_PATH=<after>/crates/facade bash scripts/bench/framework-compare/run_ab_tape_arena_cpu.sh <label>`（label は機体を含める。例 `2104-m4max`・`2104-gb10`）。
3. 出力（`compare-{train,infer}-2104-cpu-<label>.md`・JSONL・負荷ゲートログ・tree・sha）と `env_info.txt` を本ディレクトリへ収録する。ホスト名・絶対パスは `<home>` 等へマスクする。
4. `RULE.txt` に従って機械的に判定する。

## 収録予定物

生ログ（JSONL）・`aggregate.md`・`env_info.txt`・負荷ゲートログ。
