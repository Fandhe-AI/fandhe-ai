# 学習チェーンと Graph capture の ordinal 単位排他 GB10 検証（イシュー #3596）

**状態: 初回（RULE.txt）は判定 REGRESSION（終了コード 3。G1 不合格で chain セル未計測）。修正（#3601）後の再実測（RULE-r2）も判定 REGRESSION（終了コード 3。前提ゲート・hard 条件は合格し、既定 OFF の chain セルが ratio 1.002 > 1.00）。RULE-r3 の再実測は判定 UNDETERMINED（終了コード 4。前提ゲート・hard 条件・G1・`#[ignore]` は合格したが、既定 OFF の chain セルが ratio 1.007 > 1.00 で 5/5 の符号一貫がない）。完了条件は未達のため #3596 は閉じない**（2026-10-26。出典: `judge-3596.md`・`judge-3596-r2.md`・`judge-3596-r3.md`・`gb10/ignored-compare-3596.md`・`gb10-r2/ignored-compare-3596-r2.md`・`gb10-r3/ignored-compare-3596-r3.md`）

排他の実装は PR #3597 で main に入っている。完了条件の GB10 実機検証（下記 3 項目）は、本ディレクトリで事前登録して実測した。
設計は `docs/backend-cuda-graph-step-capture-design.md` §8.11、既定 ON 反転の取り下げ記録は §8.12。
本検証が完了しても #3594（既定 ON 反転）の前提は満たされない（§8.12.2 の未解決事項が残る）。`TRAIN_GRAPH_DEFAULT_ENABLED` は `false` のまま。

## 収録物

| ファイル | 内容 |
|---|---|
| `judge-3596.md`・`judge-3596.exit` | 初回の判定ツールの出力と終了コード（REGRESSION・3） |
| `judge-3596-r2.md`・`judge-3596-r2.exit` | r2 の判定ツールの出力と終了コード（REGRESSION・3） |
| `RULE.txt` | 計測前に固定した事前登録規則（腕・前提ゲート・hard 条件・判定・限界） |
| `RULE-r2.txt` | 原因修正（学習 graph の起動で MSE ワークスペースのロックを取る。設計記録 §8.11.6）後の再実測の事前登録規則（計測前に登録。閾値は RULE.txt と同一）。ランナーは `AB_RULE_NAME=RULE-r2.txt` で読む |
| `judge-3596-r3.md`・`judge-3596-r3.exit` | r3 の判定ツールの出力と終了コード（UNDETERMINED・4） |
| `RULE-r3.txt` | r2 の REGRESSION 後、既定 OFF の ordinal で排他スコープを素通りにする修正（設計記録 §8.11.8）の再実測の事前登録規則（計測前に登録。閾値・腕・ゲートは RULE-r2 と同一）。ランナーは `AB_RULE_NAME=RULE-r3.txt` で読む。マージ SHA `46a2a43d9`（2026-10-26T20:36+09:00）で、計測より前に main へ登録済み。r3 の GB10 実測結果は下記と設計記録 §8.11.9 |
| `gb10/` | 初回のマスク済みの生ログ・JSONL・`env_info`・前提ゲートのログ・`#[ignore]` 全件の結果と分類・`judge-3596.md`／`.exit` |
| `gb10-r3/` | r3 のマスク済みの生ログ・JSONL・`env_info`・前提ゲートのログ・`ignored/`・`ignored-compare-3596-r3.md`・`rev-stamps-and-runner-out.txt` |
| `gb10-r2/` | r2 のマスク済みの生ログ・JSONL・`env_info`・前提ゲートのログ・`diag/`（原因切り分けのログとテスト専用パッチ）・`ignored/`・`ignored-compare-3596-r2.md` |

ランナー: `scripts/bench/framework-compare/run_ab_train_chain_exclusion_cuda.sh`、判定: `judge_train_chain_exclusion_ab.py`（テスト: `_test.py`、CI で実行）。

## 完了条件と状態

| 条件 | 初回（RULE.txt） | r2（RULE-r2） | r3（RULE-r3） |
|---|---|---|---|
| G1: `unfenced_chain_and_capture_threads_do_not_poison_and_stay_bit_exact`（30 回・bit 一致）が GB10 で pass | **不合格**（8 回中 7 回失敗） | **合格**（前提ゲート g1 と `#[ignore]` コマンド 3 で pass） | **合格**（前提ゲート g1 と `#[ignore]` コマンド 3 で pass） |
| backend-cuda・facade の `#[ignore]` 全件を #3597 より前の main と突き合わせ、新しい fail がない | 新規 fail 1 件（G1） | **排他が原因の新規 fail 0 件**（`gb10-r2/ignored-compare-3596-r2.md`） | **排他が原因の新規 fail 0 件**（`gb10-r3/ignored-compare-3596-r3.md`） |
| chain セル（1 スレッド）で排他あり・なしの非後退（既定 OFF・opt-in ON） | 未計測 | **既定 OFF が後退判定**（ratio 1.002、5/5 round で > 1.00）。opt-in ON は 1.000 | **UNDETERMINED**（既定 OFF が ratio 1.007 > 1.00 だが 5/5 の符号一貫なし〈ratio_r 0.995〜1.009〉。opt-in ON は 1.000）。**未達** |
| `TRAIN_GRAPH_DEFAULT_ENABLED` が `false` のまま | 確認済み | 確認済み（`crates/backend-cuda/src/graph.rs` の const assert） | 確認済み（`graph.rs` に差分なし） |

## r3 再実測（RULE-r3）の結果

ラベル `3596-r3-gb10`・出力先 `gb10-r3/`。RULE.txt・RULE-r2.txt の記録（REGRESSION）も残す。

| 項目 | 結果 |
|---|---|
| 前提ゲート g1〜g4 | **合格**（g1 `unfenced…` 1 passed・g2 3260 診断 9 passed・g3 3 種・g4 facade と 2 プロセス bit 一致 BITS 90 行・差分空） |
| 専有ゲート・hard 条件 | 5 round とも通過（load1 0.07〜0.08）。checksum（0.080541）・PARAM_DIGEST は全 run・全腕で一致 |
| chain セル ratio（excl1 / excl0、5 round 中央値） | off: **1.007**（ratio_r 0.995, 1.006, 1.006, 1.007, 1.009。排他なし 0.1344 ms・あり 0.1353 ms）／ on: 1.000（1.000〜1.001。0.1126 ms） |
| 判定 | **UNDETERMINED（終了コード 4）**: off の ratio 1.007 > 1.00 だが round 1 の ratio_r が 0.995 で 5/5 の符号一貫がない（RULE-r3 [判定]）。NON-REGRESSION の条件（両モード ratio <= 1.00）は満たさない |
| `#[ignore]` 全件の突き合わせ | 排他が原因の新規 fail は 0 件。after にだけ出た 1 件（`gemm_device_matches_host_gemm_bitwise`）は単独 2/2 pass の一過性（`gb10-r3/ignored-compare-3596-r3.md`） |

- 仮説（ratio 0.99〜1.00）は外れた（事前表明は推定で合格ラインではない）。off の差は r2（1.002）より大きく、約 0.9 µs。round 1 のみ逆符号で、ratio_r が 0.995〜1.009 で符号が一貫しないため、規則上は後退とも非後退とも確定できない。原因の推定は記録しない（追加の切り分けは未実施）。
- 規則・閾値・判定ツール・コード（`crates/`）は変えていない。完了条件（`judge-3596-r3.exit` が 0）が未達のため、本記録では #3596 を閉じない（PR は `Refs`）。

## r2 再実測（RULE-r2）の結果

ラベル `3596-r2-gb10`・出力先 `gb10-r2/`。RULE.txt の記録（REGRESSION）も残す。

| 項目 | 結果 |
|---|---|
| 前提ゲート g1〜g4 | **合格**（g1 `unfenced…` 1 passed・g2 3260 診断 9 passed・g3 3 種・g4 facade と 2 プロセス bit 一致 BITS 90 行・差分空） |
| 専有ゲート・hard 条件 | 5 round とも通過。checksum（0.080541）・PARAM_DIGEST は全 run・全腕で一致 |
| chain セル ratio（excl1 / excl0、5 round 中央値） | off: **1.002**（ratio_r 1.001, 1.002, 1.004, 1.002, 1.004。排他なし 0.1346 ms・あり 0.1349 ms）／ on: 1.000（0.999〜1.001。0.1126 ms） |
| 判定 | **REGRESSION**: off で ratio 1.002 > 1.00 かつ 5/5 round で ratio_r > 1.00（RULE-r2 の閾値は変更していない） |
| `#[ignore]` 全件の突き合わせ | 排他が原因の新規 fail は 0 件。G1 を含む `train_step_graph_diag_tests_3260` は after で 9 件全件 pass。after にだけ出た 2 件は一過性・既存の不安定（分類は `gb10-r2/ignored-compare-3596-r2.md`） |
| 原因切り分け（判定外） | after は 5/5 pass。before（`f7fe6b4df`＋テスト専用パッチ）と修正前 main（`c1204755e`＋同）は落ちた（#3597 以前から存在した欠陥）。パラメータは一致（`params_equal=true`）。**ただしずれた loss は参照列のどの index とも bit 一致しなかった（`equals-reference-idx []`）ため、設計記録 §8.11.6 の「別 index の値を読む」という推定とは食い違う**。修正の効果（after 5/5 pass）は実機で確認できたが、機構は未確定 |

off の差 0.2%（約 0.3 µs）は規則の閾値 1.00 を 5/5 round で超えたため REGRESSION と判定した。2 腕は別ビルドで、#3597・#3601 以外の変更も含みうる（RULE-r2 [限界]）。規則・閾値・コードは変えていない。

## 設計の要点

- 排他に実行時トグルは無いため 2 ツリー比較: excl0 = `f7fe6b4df`（#3597 の直前）、excl1 = `crates/` が RULE の基準コミットと同一の after ツリー（初回: `d3f38f71b`、r2: `63f85eff2`。r2 の実測は `616be4ff7`）。
- 腕は {excl0, excl1} × {graph off, on}。判定は ratio <= 1.00（両モード）と、前提ゲート・checksum・PARAM_DIGEST の一致。

## 実施条件

- r3: 同一機体（GB10・aarch64・driver 580.173.02・rustc 1.97.1）。before = `f7fe6b4df`、after = `46a2a43d991f7099eaa4dfb97e07878e106b5c61`（RULE-r3 のマージ SHA は `46a2a43d9`。`crates/` は基準 `2794cfba6` と同一で、ランナーの事前検査 `precheck=assert ok` を通過）。基準コミット `2794cfba6`（squash 前の履歴）は bundle に含めた。計測専用パッチの sha256 は RULE-r3 の値（`92d0a8d4…3a819`）と一致。`#[ignore]` の突き合わせは計測と並走させず、計測前に実施した。

- 初回（RULE.txt）: 機体は GB10（aarch64・driver 580.173.02・rustc 1.97.1）。before = `f7fe6b4dfaabbdbc51a900709fe6648c36120c58`、after = `087e7f6d463b42b24cb65769986d78948d61d53f`（`crates/` は `d3f38f71b` と同一）。ホスト名・ホームディレクトリはマスク済み（`<home>`）。
- r2: 同一機体・同一 driver・同一 rustc。before = `f7fe6b4df`、after = `616be4ff7658657933319a7e8cb8cec7a95d8422`、診断用の修正前 main = `c1204755e938a7908265c630968d864ec3443569`。計測専用パッチの sha256 は RULE-r2 の値（`92d0a8d4…3a819`）と一致。テスト専用パッチ（診断テストと G1 だけを足す。本体コードは含まない）の sha256: `diag-test-only-f7fe6b4df.patch` = `e38de09c8be5c8f080584559fde7f436240d02f7469c1be793fcae7a5f967109`、`diag-test-only-c1204755e.patch` = `e5916951f01733d4741fe821e7af839c615c91685871c0b91fbd1d6dd98b708b`（`gb10-r2/diag/`）。
- r2 の 1 回目の起動は、after ツリーに RULE-r2 の基準コミット `63f85eff2`（squash 前の履歴）が無くランナーの事前検査で停止した（計測前。`gb10-r2/rev-stamps-and-runner-out.txt` の `## attempt1` 節）。基準コミットを bundle で追加し、同じラベルで再起動した。停止時点で計測は行われておらず、規則・入力は変えていない。
- 転送は `git bundle`（追跡ファイルのみ）。バイナリ本体はコミットしない。

## 所有者判断事項

- **（r3 の結果）** off の ratio 1.007 は UNDETERMINED（符号の一貫なし）。追加の切り分け（round 数を増やす別名 RULE の事前登録、同一バイナリ内の比較など）を行うか、現状の差を許容して判定基準を見直すか。閾値 1.00 は規則どおり変えていない。
- r2 の off モード ratio 1.002（REGRESSION）は、RULE-r3 の再実測（上記）で ratio 1.007・UNDETERMINED となった。r2 の扱いは先頭の r3 の項目に統合する。
- §8.11.6 の機構の推定が実機の `DIAG3596` と食い違った点（ずれた loss が参照列のどの index とも一致しない）の追加調査。修正自体は after 5/5 pass。
- #3594（既定 ON 反転）の前提は、§8.12.2 の未解決事項が残るため引き続き満たされない。

## GB10 での実行手順（再実施する場合）

1. 実機ロック `~/work/.fandhe-hw-lock` を取る（ノードは `spark-939e` 推奨）。
2. after（本ランナーを含むマージコミット）と before（`f7fe6b4df`）を `git bundle` で転送して clone し、各ツリーに `.rev-stamp`（1 行目 HEAD・2 行目 `dirty=0`）を作り `.git/info/exclude` に足す。RULE の `crates_identical_to` の基準コミットが after の履歴に無い場合はその commit も bundle に含める。両ツリーに `bench-resident-chain.patch` を `git apply` する。
3. 計測の前に RULE の「`#[ignore]` 全件の突き合わせ」を before → after の順に実行する。
4. `AB_RULE_NAME=<RULE ファイル名> AB_BEFORE_FACADE_PATH=… AB_AFTER_FACADE_PATH=… bash run_ab_train_chain_exclusion_cuda.sh <ラベル>` を切り離し起動する（前提ゲート → 5 round）。
5. 回収後にホスト名・`$HOME`・ユーザー名をマスクし、`judge_train_chain_exclusion_ab.py --rule … --dir …/<出力先> --label <ラベル>` の出力と終了コードを保存する。
6. 設計 doc §8.11 に実測節を追記する。結果が後退・判定不能でも規則・閾値は変えず、所有者判断として報告する。
