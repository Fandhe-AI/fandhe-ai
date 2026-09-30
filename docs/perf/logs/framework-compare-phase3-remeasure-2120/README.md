# framework-compare 両機体再計測キット（イシュー #2120・Phase 3 採否反映）

設計・前提のずれ・22 セル記入欄・申し送りは `docs/perf/framework-compare-phase3-remeasure.md` を正とする。
本ディレクトリは計測キット（実測前に固定する判定規則を含む）であり、**両機体の実測ログは未収録**（実機セッションで収録する）。

## ファイル

| パス | 内容 |
|---|---|
| `RULE.txt` | 事前登録判定規則（実測前に固定。腕 A/B/C・セル範囲・ゲート・非後退 ⇔ B→C 比 1.00 以下・checksum 完全一致・期待値） |
| `orchestrate.sh` | 本体。3 腕ビルド（`Cargo.lock` sha256 突合・invocation 限定 `--config patch`）→ 専有ゲート付き 5 run。`SMOKE=1` で疎通確認 |
| `orchestrate_gb10.sh`／`orchestrate_m4max.sh` | 機体名を渡すだけのラッパー |
| `switches.sh` | Phase 3 スイッチの既定値をツリーから機械抽出（`switches-B.txt`／`switches-C.txt` の元。C は欠落で fail-closed・B は `--allow-missing`） |
| `aggregate.py` | 集計（python3 標準ライブラリのみ・`--self-test`）。`aggregate.md` と腕別の派生 JSONL を出力。5 run × 3 腕が欠ければ停止 |
| `scoreboard/gen_2120.py`／`body_2120.html` | スコアボード生成器（`gen_1988.py` 派生。`--prev-label`・`--main-label`・`--tsv` 追加）と本文 |

実測後は `m4max/`・`gb10/` 配下へ `run{1..5}/`・`gate.log`・`build.log`・`env_info.txt`・`switches-{B,C}.txt`・`aggregate.md` を収録する。

## 再現手順・所要時間

`docs/perf/framework-compare-phase3-remeasure.md` §5 を参照。

## スコアボード生成器の非後退確認

`gen_2120.py --body ../../framework-compare-precision-class-remeasure-1988/scoreboard/body_1988.html --prev-label 0.8.0` を #1988 README の
元入力で実行した HTML と検証出力は、`gen_1988.py` の出力と byte 同一（`cmp` で確認）。

## 変更していないもの

tolerance 4 定数・判定式・`ParityBaseline`／`BASELINES`・`Cargo.toml`／`Cargo.lock`・ガードレール閾値・`docs/spec`・
承認ピン・`SME_PRODUCTION_ENABLED`／`GB10_AFFINITY_ENABLED`・既存 REJECT／undetermined 実験。
