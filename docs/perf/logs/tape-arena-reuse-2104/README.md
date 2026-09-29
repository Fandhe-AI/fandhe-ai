# tape・ホスト arena 再利用の A/B ログ（イシュー #2104）

状態: **未実測**（M4 Max・GB10 とも申し送り。x86_64 ホストで実装と CI テストのみ完了）。

- 設計: `docs/tape-arena-reuse-design.md`
- 判定規則（実測前に固定）: `RULE.txt`

## 実行手順（各機体で 1 回ずつ。専有ゲートを通す）

1. before 用に main を checkout した worktree、after 用に同一コミットで `crates/tensor-core/src/alloc.rs` の `HOST_ARENA_DEFAULT_ENABLED` を `true` にした worktree を用意する（after は計測専用でコミットしない）。
2. `AB_BEFORE_FACADE_PATH=<before>/crates/facade AB_AFTER_FACADE_PATH=<after>/crates/facade bash scripts/bench/framework-compare/run_ab_tape_arena_cpu.sh <label>`（label は機体を含める。例 `2104-m4max`・`2104-gb10`）。
3. 出力（`compare-{train,infer}-2104-cpu-<label>.md`・JSONL・負荷ゲートログ・tree・sha）と `env_info.txt` を本ディレクトリへ収録する。ホスト名・絶対パスは `<home>` 等へマスクする。
4. `RULE.txt` に従って機械的に判定する。

## 収録予定物

生ログ（JSONL）・`aggregate.md`・`env_info.txt`・負荷ゲートログ。
