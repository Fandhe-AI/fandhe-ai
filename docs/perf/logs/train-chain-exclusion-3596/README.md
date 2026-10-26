# 学習チェーンと Graph capture の ordinal 単位排他 GB10 検証（イシュー #3596）

**状態: GB10 で実施済み・判定 REGRESSION（終了コード 3）。G1 が不合格のため chain セルの計測は前提ゲートで停止し、未計測**（2026-10-26。出典: `judge-3596.md`・`gb10/ignored-compare-3596.md`）

排他の実装は PR #3597 で main に入っている。完了条件の GB10 実機検証（下記 3 項目）が未実測のため、本ディレクトリで事前登録して実測する。
設計は `docs/backend-cuda-graph-step-capture-design.md` §8.11、既定 ON 反転の取り下げ記録は §8.12。
本検証が完了しても #3594（既定 ON 反転）の前提は満たされない（§8.12.2 の未解決事項が残る）。`TRAIN_GRAPH_DEFAULT_ENABLED` は `false` のまま。

## 収録物

| ファイル | 内容 |
|---|---|
| `judge-3596.md`・`judge-3596.exit` | 判定ツールの出力と終了コード（REGRESSION・3） |
| `RULE.txt` | 計測前に固定した事前登録規則（腕・前提ゲート・hard 条件・判定・限界） |
| `gb10/` | マスク済みの生ログ・JSONL・`env_info`・前提ゲートのログ・`#[ignore]` 全件の結果と分類・`judge-3596.md`／`.exit` |

ランナー: `scripts/bench/framework-compare/run_ab_train_chain_exclusion_cuda.sh`、判定: `judge_train_chain_exclusion_ab.py`（テスト: `_test.py`、CI で実行）。

## 完了条件と状態

| 条件 | 状態 |
|---|---|
| G1: `unfenced_chain_and_capture_threads_do_not_poison_and_stay_bit_exact`（30 回・bit 一致）が GB10 で pass | **不合格**。非 capture スレッドの loss 列が単独実行と食い違う（8 回中 7 回失敗。`gb10/g2-3260-3596-gb10.log`・`gb10/ignored/ignored-after-3.log`・`gb10/ignored/g1-rerun-after-*.log`） |
| backend-cuda・facade の `#[ignore]` 全件を #3597 より前の main と突き合わせ、新しい fail がない | **新規 fail 1 件（G1）**。ほか補足実行の `small_shape_matrix_unit_has_no_floor_tflops_record` は単独実行 3/3 pass のフレーク。詳細 `gb10/ignored-compare-3596.md` |
| chain セル（1 スレッド）で排他あり・なしの非後退（既定 OFF・opt-in ON） | **未計測**（前提ゲート g2 の失敗でランナーが停止。`gb10/skipped-train-chain-exclusion-3596-gb10.log`）。g3・g4（2 プロセス bit 一致 BITS 90 行・差分空）は合格 |
| `TRAIN_GRAPH_DEFAULT_ENABLED` が `false` のまま | 確認済み（`crates/backend-cuda/src/graph.rs` の const assert） |

## 設計の要点

- 排他に実行時トグルは無いため 2 ツリー比較: excl0 = `f7fe6b4df`（#3597 の直前）、excl1 = `crates/` が `d3f38f71b` と同一の after ツリー。
- 腕は {excl0, excl1} × {graph off, on}。判定は ratio <= 1.00（両モード）と、前提ゲート・checksum・PARAM_DIGEST の一致。

## 実施条件

- 機体: GB10（aarch64・driver 580.173.02・rustc 1.97.1）。before = `f7fe6b4dfaabbdbc51a900709fe6648c36120c58`、after = `087e7f6d463b42b24cb65769986d78948d61d53f`（`crates/` は `d3f38f71b` と同一）。ホスト名・ホームディレクトリはマスク済み（`<home>`）。
- 転送は `git bundle`（追跡ファイルのみ）。計測専用パッチの sha256 は RULE の値と一致。バイナリ本体はコミットしない。
- ランナーは RULE どおり前提ゲート不合格のため停止した（再実行・ラベル差し替え・閾値変更はしていない）。

## 所有者判断事項

- G1 の失敗（排他だけでは並行時の結果取り違えが残る）の原因調査と修正方針。`crates/` は本 Issue の範囲外のため未変更。
- 修正後に本規則とは別名の RULE で再登録して再実測するか。#3594（既定 ON 反転）の前提は引き続き満たされない。

## GB10 での実行手順（再実施する場合）

1. 実機ロック `~/work/.fandhe-hw-lock` を取る（ノードは `spark-939e` 推奨）。
2. after（本ランナーを含むマージコミット）と before（`f7fe6b4df`）を `git bundle` で転送して clone し、各ツリーに `.rev-stamp`（1 行目 HEAD・2 行目 `dirty=0`）を作り `.git/info/exclude` に足す。両ツリーに `bench-resident-chain.patch` を `git apply` する。
3. 計測の前に RULE の「`#[ignore]` 全件の突き合わせ」を before → after の順に実行する。
4. `AB_BEFORE_FACADE_PATH=… AB_AFTER_FACADE_PATH=… bash run_ab_train_chain_exclusion_cuda.sh 3596-gb10` を切り離し起動する（前提ゲート → 5 round）。
5. 回収後にホスト名・`$HOME`・ユーザー名をマスクし、`judge_train_chain_exclusion_ab.py --rule … --dir …/gb10 --label 3596-gb10` の出力と終了コードを保存する。
6. 設計 doc §8.11 に実測節を追記する。結果が後退・判定不能でも規則・閾値は変えず、所有者判断として報告する。
