# metal-gemm-candidate-ab-2111

candle／MLX steel 解析差分由来の Metal GEMM 候補（イシュー #2110）の kernel_gpu 5 run A/B 用スキャフォールド。
候補の定義・根拠・除外表は [`docs/perf/metal-gemm-steel-candidates.md`](../../metal-gemm-steel-candidates.md)。

## 分担

| イシュー | 担当 |
|---|---|
| #2110 | 機構（`UNROLL_LOAD_ENABLED`・candle 相当タイル arm）・bit 一致／parity の自己テスト・計測ハーネス・本スキャフォールド（`RULE.txt` は実測前に固定済み） |
| #2111 | M4 Max 実機での gate・5 run 計測・`aggregate.py` の判定・結線判断（本番既定の変更はユーザー承認事項） |

`RULE.txt` は実測前に固定した規則で、実測結果を見た後に緩和・変更しない。

## 実行手順（Apple Silicon 実機・#2111）

1. `env_info.txt` を記入する（ホスト名・ユーザー名・絶対パスは書かない）。
2. `./orchestrate.sh gate` — bit 一致 3 本と CPU 参照 parity 1 本。FAIL なら A/B は実施しない（RULE.txt 1.）。
3. `./orchestrate.sh 1` … `./orchestrate.sh 5` — プロセス独立の 5 run。各 run の前に load1 < 8.0 を最大 30 分待つ。
4. `python3 aggregate.py` — arm 別に ADOPT_CANDIDATE／REJECT／UNDETERMINED／NOT_ADOPTABLE／INCOMPLETE を出力する。
5. 出力を `aggregate.md` として保存し、`docs/perf/metal-gemm-steel-candidates.md` の実測記入欄を埋める。

Linux 等では `./orchestrate.sh gate --dry-run`／`./orchestrate.sh 1 --dry-run` で分岐だけ検証できる。
`python3 aggregate.py --self-test` は判定ロジックの固定 fixture 検証（実測不要）。

`aggregate.py` は fail-closed: `load_gate.log` の run 1〜5 各記録・各 `kernel_gpu_run{i}.log` の対象テスト成功（`test result: ok`・`0 failed`）・`env_info.txt` の `run{i} completed` 記録のいずれかが欠ける／失敗の場合、全 arm を INCOMPLETE にし採用判定を出さない（RULE.txt 2.・7.）。

## 保存するファイル

`gate_run.log`・`kernel_gpu_run{1..5}.log`・`load_gate.log`・`run{1..5}_monitor.log`・`run{1..5}_procs.txt`・
`uptime_before_run{1..5}.txt`・`pmset_therm_{before,after}_run{1..5}.txt`・`env_info.txt`・`aggregate.md`。

コミット前に、ログ中のホスト名・ユーザー名・絶対パスを `<home>` 等へマスクする
（`.claude/rules/security.md`。並走プロセスは `run*_procs.txt` に件数のみ記録される）。
