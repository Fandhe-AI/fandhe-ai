# GB10 小形状 GEMM 大コア affinity（イシュー #1576）実測ログ置き場

`docs/perf/cpu-gemm-gb10-affinity-ab.md` の事前登録判定規則に基づく実測ログをここへ保存する。

本エージェント実行環境に DGX Spark GB10 実機への到達手段が無いため、本 PR 時点ではログは未生成（記入欄のみ）。GB10 実機実測を実施するセッションが以下を保存すること。

- `before.jsonl`／`after.jsonl`（`bench-fandhe --phases` 出力。各セル 5 run）
- `on-arm.patch`（`GB10_AFFINITY_ENABLED = false → true` の 1 行差分。#1301/#1481 と同型）
- `env_info.txt`（内部ホスト名は含めない）
- `compare.md`（`docs/perf/cpu-gemm-gb10-affinity-ab.md` の判定規則に基づく判定結果）
