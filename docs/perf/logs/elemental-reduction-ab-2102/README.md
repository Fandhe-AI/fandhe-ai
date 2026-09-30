# 縮約しきい値化の両機体 A/B（イシュー #2102）

## 状態: 未実測

x86_64 ホスト（判定対象の M4 Max・GB10 のいずれでもない）で A/B 基盤・事前登録規則・CI テストまで完了。
**実測値はなく、既定は OFF のまま**（`REDUCTION_SEQUENTIAL_FALLBACK_ENABLED = false`）。
`m4max/`・`gb10/` に推定値を書かない。判定は `RULE.txt`（実測前に固定済み）に従う。

## 手順（機体ごと）

1. Stage 0: `bash run_reduction_sweep.sh <machine-label> <絶対パスの出力先>`（5 プロセス独立・env_info 採取）。
   生ログは 2101 の `elemental-reduction-threshold-2101/{m4max,gb10}/` へ収録する。
2. 集計: `python3 aggregate_sweep.py --m4max <dir> --gb10 <dir> --out aggregate.md`
   （checksum 一致・違反点・T・利得判定。`--self-test` あり）。T <= 640 なら REJECT で停止。
3. Stage 1: main の worktree（before）と、同一コミットで `reduction.rs` の 2 定数だけを変えた
   計測専用 worktree（after。コミットしない）を作り、次を実行する。
   ```
   AB_BEFORE_FACADE_PATH=<abs>/crates/facade AB_AFTER_FACADE_PATH=<abs>/crates/facade \
   AB_THRESHOLD_T=<T> bash scripts/bench/framework-compare/run_ab_reduction_threshold_cpu.sh 2102-<machine>
   ```
   gemm N=4096 を含むため 1 round あたりの所要時間が #2104 より延びる（CPU 4096 は 1 回 0.1〜数秒）。
4. `RULE.txt` の総合判定（ADOPT / REJECT / undetermined）。ADOPT なら結線 PR（2 定数の切替）を別途作る。

## 収録予定物（機体別ディレクトリ）

run ログ・`aggregate.md`・`env_info.txt`・`load_gate.log`・compare 表・JSONL・tree・sha。
ホスト名・絶対パス・ユーザー名は `<home>` 等にマスクしてから置く。

## 併せて行う確認（#2101 からの申し送り）

Metal・CUDA の reduce 系 parity（`cargo test -p fandhe-ai-backend-metal|cuda -- --ignored`）を同セッションで実行する。
