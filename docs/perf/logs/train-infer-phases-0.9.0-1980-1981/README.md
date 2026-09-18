# train／infer `--phases` 5 run 中央値（registry `fandhe-ai =0.9.0`。イシュー #1980／#1981・両実機分）

正式記録は `docs/perf/train-step-phase-breakdown.md` §17
（train。Mac 分 §17.2〜§17.5・GB10 分 §17.6）・
`docs/perf/infer-reuse-phase-breakdown.md` §10（infer。Mac 分
§10.2〜§10.5・GB10 分 §10.6）を参照。本ディレクトリはその実測ログ一式。

## 目的

`bench-fandhe`（crates.io 公開版 `fandhe-ai =0.9.0` にピンした
registry ビルド。`crates/facade` への path patch なし・cargo を
起動しない事前ビルド済みバイナリ）の `--task train|infer --phases`
を Apple M4 Max（cpu／metal）・DGX Spark GB10（cpu／cuda）それぞれで
× fresh／reuse の 8 セル・5 プロセス独立起動し、フェーズ分解の中央値表を
作る。**採否判定を伴わない記録**（フェーズ表・施策案の列挙のみ。
tolerance・判定規則・本番定数は変更しない）。

## 構成

- `orchestrate_m4max.sh` — Mac 分の 5 プロセス独立起動スクリプト
  （1 run = 8 セルを 1 JSONL へ出力。cpu／metal）。事前ビルド済み
  `scripts/bench/framework-compare/target/release/bench-fandhe` を使い
  cargo を起動しない。各 run 開始前に load1 < 8.0 を待つ
- `orchestrate_gb10.sh` — GB10 分の同スクリプト（cpu／cuda）。Linux
  `/proc/loadavg` と `nvidia-smi` を読み、各 run 開始前に「load1 < 1.0
  かつ GPU utilization 0%」を 30 秒間隔・最大 30 分待つ（専有機のため
  Mac 分より厳しいゲート）。読めない場合は `gate=unavailable` と記録して
  run は実行する（系列は参考扱い）。既存 `run*.jsonl` があれば差し替え
  禁止として停止する
- `aggregate.py` — 集計スクリプト（python3 標準ライブラリのみ）。5 run
  間の `median_s`（各 JSONL 行の top-level キー）の中央値・最小・最大と
  合計フェーズ（train は `step_total`・infer は `iter_total`）に対する
  比を出力。`--devices`（期待するデバイス集合。既定 `cpu,metal`）・
  `--machine`（見出しの実機名。既定 `Apple M4 Max`）を GB10 分で追加。
  既定引数の出力は `m4max/aggregate.md` と byte 同一・
  `--devices cpu,cuda --machine "DGX Spark GB10"` の出力は
  `gb10/aggregate.md` と byte 同一（いずれも再生成で確認済み）。
  `m4max/RULE.txt` の `stats.median_s` という表記は同じ値を指す表記ゆれ
  で、規則自体は不変
- `m4max/` — Apple M4 Max の実測一式（2026-09-17）
  - `run{1..5}.jsonl` — 各 run の生の計測結果（8 セル ×
    `train_phases`／`infer_phases` の全フェーズ）
  - `run{1..5}.err` — 各 run の標準エラー出力（5 本とも空。失敗なし）
  - `load_gate.log` — 負荷ゲート記録（5/5 通過）
  - `RULE.txt` — 事前登録規則（固定日時 2026-09-17T16:31:36Z）
  - `aggregate.md` — 8 セル × フェーズ表の集計結果
  - `env_info.txt` — 実行環境・時刻の記録（内部ホスト名は含めない）
- `gb10/` — DGX Spark GB10 の実測一式（2026-09-18）
  - `run{1..5}.jsonl` — 各 run の生の計測結果（cpu／cuda 8 セル）
  - `run{1..5}.err` — 5 本とも 0 バイト（失敗なし）
  - `load_gate.log` — 負荷ゲート記録（5/5 `gate=pass`。load1
    0.16〜0.61・gpu_util 0・`compute_apps=2`〈常駐 2 プロセス。停止せず
    存在のみ記録〉）
  - `RULE.txt` — 事前登録規則（固定日時 2026-09-18T01:33:53Z）
  - `aggregate.md` — 8 セル × フェーズ表の集計結果
  - `env_info.txt` — 実行環境・時刻・転送元コミットの記録（hostname は
    masked）

## 条件

- registry `fandhe-ai =0.9.0`（`scripts/bench/framework-compare/`
  の承認済みピン。deps-policy.md 第 9 区分）
- size=64（`train`／`infer` 既定形状）
- Mac 分: 5 round・round 単位で load1 < 8.0 ゲート（`m4max/RULE.txt`）。
  共有負荷下（load1 6.08〜7.03）
- GB10 分: 5 round・round 単位で「load1 < 1.0 かつ GPU utilization
  0%」ゲート（`gb10/RULE.txt`）。5/5 通過につき専有ゲート付き系列として
  正式。バイナリはノード上で 2026-09-18 に再ビルド（転送元コミット
  `536c56a8`・path patch なし・計測中に cargo 非起動）。CPU は registry
  既定（GB10 affinity 機構は既定 OFF・スレッド数無指定）
- 両実機が #1980／#1981 の受け入れ条件「両実機 5 run 中央値のフェーズ表」
  であり、本ディレクトリで両方が揃った

## 結果要約

- train（Mac）: backward が cpu 59.0〜60.6%・metal 49.4〜53.5% で最大の
  残差
- train（GB10）: backward が cpu 41.9〜44.7%・cuda fresh 40.1% で最大の
  残差。cuda reuse のみ forward_resident（49.1%）と backward（46.9%）が
  拮抗。cpu reuse は `device_update` が 25.7%（277.5 µs）。step_total は
  cpu fresh 1130.5／cpu reuse 1080.6／cuda fresh 518.0／cuda reuse
  317.5 µs
- infer（Mac）: cpu は `predict`／`predict_resident` がほぼ 100%、metal
  は fresh が `forward`＋`to_tensor` 分解可能・reuse は
  `predict_resident` 単独区間
- infer（GB10）: cuda fresh は `forward` 87.2%＋`to_tensor` 12.0%・
  cuda reuse 100.0 µs は cpu reuse 196.0 µs より速い（Mac の metal
  reuse が cpu reuse より遅い関係とは逆）。cpu reuse は cpu fresh
  177.1 µs より遅い
- 詳細は `docs/perf/train-step-phase-breakdown.md` §17・
  `docs/perf/infer-reuse-phase-breakdown.md` §10 を参照

## 限界

- backward 内部の GEMM／非 GEMM 内訳は診断計装パッチ
  （`docs/perf/logs/lowlayer-diagnosis-2026-09-12/diag-instrumentation.patch`）
  が必要で、registry 版（cargo を起動しない事前ビルド済みバイナリ）
  では取得不能。本計測では両実機とも未取得
- GB10 分は常駐 2 プロセスを停止していないため完全な専有ではない
  （gpu_util は各 run 開始時 0%）
