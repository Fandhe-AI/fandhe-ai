# framework-compare 両機体再計測キット（イシュー #2120・Phase 3 採否反映）

設計・前提のずれ・22 セル記入欄・申し送りは `docs/perf/framework-compare-phase3-remeasure.md` を正とする。
本ディレクトリは計測キット（実測前に固定する判定規則を含む）であり、**両機体の実測ログは未収録**（実機セッションで収録する）。

## ファイル

| パス | 内容 |
|---|---|
| `RULE.txt` | 事前登録判定規則（実測前に固定。腕 A/B/C・セル範囲・ゲート・前提 P1〜P5・非後退 ⇔ B→C 比 1.00 以下・checksum 完全一致・期待値） |
| `orchestrate.sh` | 本体。3 腕ビルド（`Cargo.lock` sha256 突合・invocation 限定 `--config patch`）→ 専有ゲート付き 5 run。GB10 はゲート不通過で停止（前提条件）・M4 Max は記録して続行（record_only）。完了記録 `env_info.txt` は最後に書く。`SMOKE=1` で疎通確認 |
| `orchestrate_gb10.sh`／`orchestrate_m4max.sh` | 機体名を渡すだけのラッパー |
| `switches.sh` | Phase 3 スイッチの既定値をツリーから機械抽出（`switches-B.txt`／`switches-C.txt` の元。C は欠落で fail-closed・B は `--allow-missing`） |
| `aggregate.py` | 集計（python3 標準ライブラリのみ・`--self-test` は RULE.txt 条項ごとの表駆動）。前提 P1〜P5 を `check_prerequisites` の単一ゲートで評価し、不成立なら何も出力しない。成立時は `aggregate.md`・腕別の派生 JSONL・集計 JSON（`<prefix>-aggregate.json`。B→C 比・ゲート・派生 JSONL の sha256）を出力 |
| `scoreboard/gen_2120.py`／`body_2120.html` | スコアボード生成器（`gen_1988.py` 派生）と本文。既定の正式モードは集計 JSON 必須で、前比・カード・`--tsv` の B→C 比を集計 JSON の `bc_med` から転記する（再計算しない）。`--legacy-1988` は gen_1988 互換（非後退確認用） |
| （追記） | `gen_2120.py` の `--gb-py`／`--gb-extra` は同一セルの重複行を拒否する。例外は RULE.txt「Python FW 行の文書化済み上書き」の許可キー（`pytorch gemm cpu 4096 fresh`）だけで、規定の `--gb-py`（#1988 `results-dgx-py-precision-class.jsonl`）の 5 行目（2026-09-12）→ 29 行目（#1988 採用 run・2026-09-18）が記録済み sha256 と一致する場合に限り後着行を採用し（gen_1988 の後勝ちと同じ意味）、採用・破棄をスコアボードの GB10 計測条件注記と標準出力（`py-override`）に残す。行に計測日が無いため後着行は出典の実物と byte 一致する行本文の sha256 で識別する。`--legacy-1988` の既定の比較元（`--m4-prev`／`--gb-prev` 省略時）は CSS と同じく `docs/perf/logs` 基点で解決し存在を検査する |
| （追記） | `orchestrate.sh` は `HEAD_TREE`・`PRE_TREE`・`LOGD`（`LOGD` は作成後）と自身の配置ディレクトリを、`framework-compare` へ `cd` する前に `pwd -P` で絶対パスへ正規化する（存在確認を兼ねる。相対パス指定でも switches.sh の引数・`--config patch`・ログ出力先・`build.log` のマスクが cwd に依存しない） |

実測後は `m4max/`・`gb10/` 配下へ `run{1..5}/`・`gate.log`・`build.log`・`env_info.txt`・`switches-{B,C}.txt`・`aggregate.md`・
`<prefix>-{A,B,C}-full.jsonl`・`<prefix>-aggregate.json` を収録する（RULE.txt 保存物）。

## 再現手順・所要時間

`docs/perf/framework-compare-phase3-remeasure.md` §5 を参照。

## スコアボード生成器の非後退確認

`gen_2120.py --legacy-1988 --body ../../framework-compare-precision-class-remeasure-1988/scoreboard/body_1988.html --prev-label 0.8.0` を
#1988 README の元入力で実行した HTML と標準出力は、`gen_1988.py` の出力と byte 同一（`gen_2120.py --self-test` が比較する）。
正式モード（集計 JSON 使用）の出力は gen_1988 と同一ではない（前比の定義が RULE.txt 判定 3 の B→C 比に替わるため）。

## 変更していないもの

tolerance 4 定数・判定式・`ParityBaseline`／`BASELINES`・`Cargo.toml`／`Cargo.lock`・ガードレール閾値・`docs/spec`・
承認ピン・`SME_PRODUCTION_ENABLED`／`GB10_AFFINITY_ENABLED`・既存 REJECT／undetermined 実験。
