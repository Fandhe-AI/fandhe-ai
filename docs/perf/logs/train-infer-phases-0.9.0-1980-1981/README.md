# train／infer `--phases` 5 run 中央値（registry `fandhe-ai =0.9.0`。イシュー #1980／#1981・Mac 分）

正式記録は `docs/perf/train-step-phase-breakdown.md` §17
（train）・`docs/perf/infer-reuse-phase-breakdown.md` §10（infer）を
参照。本ディレクトリはその実測ログ一式。

## 目的

`bench-fandhe`（crates.io 公開版 `fandhe-ai =0.9.0` にピンした
registry ビルド。`crates/facade` への path patch なし・cargo を
起動しない事前ビルド済みバイナリ）の `--task train|infer --phases`
を cpu／metal × fresh／reuse の 8 セルで 5 プロセス独立起動し、
フェーズ分解の中央値表を作る。**採否判定を伴わない記録**（フェーズ表・
施策案の列挙のみ。tolerance・判定規則・本番定数は変更しない）。

## 構成

- `orchestrate_m4max.sh` — 5 プロセス独立起動スクリプト（1 run = 8 セル
  を 1 JSONL へ出力）。事前ビルド済み
  `scripts/bench/framework-compare/target/release/bench-fandhe` を使い
  cargo を起動しない
- `aggregate.py` — 集計スクリプト（python3 標準ライブラリのみ）。5 run
  間の `median_s`（各 JSONL 行の top-level キー）の中央値・最小・最大と
  合計フェーズ（train は `step_total`・infer は `iter_total`）に対する
  比を出力。計測後に JSONL キー参照のみ是正済み（判定規則は不変。
  docstring・`RULE.txt` 側の `stats.median_s` という表記は旧いまま
  未更新で残っている）
- `m4max/` — Apple M4 Max の実測一式
  - `run{1..5}.jsonl` — 各 run の生の計測結果（8 セル ×
    `train_phases`／`infer_phases` の全フェーズ）
  - `run{1..5}.err` — 各 run の標準エラー出力（5 本とも空。失敗なし）
  - `load_gate.log` — 負荷ゲート記録（5/5 通過）
  - `RULE.txt` — 事前登録規則（固定日時 2026-09-17T16:31:36Z）
  - `aggregate.md` — 8 セル × フェーズ表の集計結果
  - `env_info.txt` — 実行環境・時刻の記録（内部ホスト名は含めない）

## 条件

- registry `fandhe-ai =0.9.0`（`scripts/bench/framework-compare/`
  の承認済みピン。deps-policy.md 第 9 区分）
- size=64（`train`／`infer` 既定形状）
- 5 round・round 単位で load1 < 8.0 ゲート（RULE.txt）
- GB10（DGX Spark GB10）側は別セッションで未実測。両実機が
  #1980／#1981 の受け入れ条件のため、両 issue はこの Mac 分のみでは
  close しない

## 結果要約

- train：backward が cpu 59.0〜60.6%・metal 49.4〜53.5% で最大の残差
- infer：cpu は `predict`／`predict_resident` がほぼ 100%、metal は
  fresh が `forward`＋`to_tensor` 分解可能・reuse は `predict_resident`
  単独区間
- 詳細は `docs/perf/train-step-phase-breakdown.md` §17・
  `docs/perf/infer-reuse-phase-breakdown.md` §10 を参照

## 限界

- backward 内部の GEMM／非 GEMM 内訳は診断計装パッチ
  （`docs/perf/logs/lowlayer-diagnosis-2026-09-12/diag-instrumentation.patch`）
  が必要で、registry 版（cargo を起動しない事前ビルド済みバイナリ）
  では取得不能。本計測では未取得
