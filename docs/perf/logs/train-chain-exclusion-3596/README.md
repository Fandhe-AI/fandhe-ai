# 学習チェーンと Graph capture の ordinal 単位排他 GB10 検証（イシュー #3596）

**状態: 未実測**（事前登録規則・ランナー・判定ツールのみ。GB10 実機での実測は未実施）

排他の実装は PR #3597 で main に入っている。完了条件の GB10 実機検証（下記 3 項目）が未実測のため、本ディレクトリで事前登録して実測する。
設計は `docs/backend-cuda-graph-step-capture-design.md` §8.11、既定 ON 反転の取り下げ記録は §8.12。
本検証が完了しても #3594（既定 ON 反転）の前提は満たされない（§8.12.2 の未解決事項が残る）。`TRAIN_GRAPH_DEFAULT_ENABLED` は `false` のまま。

## 収録物

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 計測前に固定した事前登録規則（腕・前提ゲート・hard 条件・判定・限界） |
| `gb10/`（実測後に追加） | マスク済みの生ログ・JSONL・`env_info`・前提ゲートのログ・`#[ignore]` 全件の結果と分類・`judge-3596.md`／`.exit` |

ランナー: `scripts/bench/framework-compare/run_ab_train_chain_exclusion_cuda.sh`、判定: `judge_train_chain_exclusion_ab.py`（テスト: `_test.py`、CI で実行）。

## 完了条件と状態

| 条件 | 状態 |
|---|---|
| G1: `unfenced_chain_and_capture_threads_do_not_poison_and_stay_bit_exact`（30 回・bit 一致）が GB10 で pass | 未実測 |
| backend-cuda・facade の `#[ignore]` 全件を #3597 より前の main と突き合わせ、新しい fail がない | 未実測 |
| chain セル（1 スレッド）で排他あり・なしの非後退（既定 OFF・opt-in ON の両方） | 未実測 |
| `TRAIN_GRAPH_DEFAULT_ENABLED` が `false` のまま | 確認済み（`crates/backend-cuda/src/graph.rs` の const assert） |

## 設計の要点

- 排他に実行時トグルは無いため 2 ツリー比較: excl0 = `f7fe6b4df`（#3597 の直前）、excl1 = `crates/` が `d3f38f71b` と同一の after ツリー。
- 腕は {excl0, excl1} × {graph off, on}。判定は ratio <= 1.00（両モード）と、前提ゲート・checksum・PARAM_DIGEST の一致。

## GB10 での実行手順（実機が使えるセッションで実施）

1. 実機ロック `~/work/.fandhe-hw-lock` を取る（ノードは `spark-939e` 推奨）。
2. after（本ランナーを含むマージコミット）と before（`f7fe6b4df`）を `git bundle` で転送して clone し、各ツリーに `.rev-stamp`（1 行目 HEAD・2 行目 `dirty=0`）を作り `.git/info/exclude` に足す。両ツリーに `bench-resident-chain.patch` を `git apply` する。
3. 計測の前に RULE の「`#[ignore]` 全件の突き合わせ」を before → after の順に実行する。
4. `AB_BEFORE_FACADE_PATH=… AB_AFTER_FACADE_PATH=… bash run_ab_train_chain_exclusion_cuda.sh 3596-gb10` を切り離し起動する（前提ゲート → 5 round）。
5. 回収後にホスト名・`$HOME`・ユーザー名をマスクし、`judge_train_chain_exclusion_ab.py --rule … --dir …/gb10 --label 3596-gb10` の出力と終了コードを保存する。
6. 設計 doc §8.11 に実測節を追記する。結果が後退・判定不能でも規則・閾値は変えず、所有者判断として報告する。
