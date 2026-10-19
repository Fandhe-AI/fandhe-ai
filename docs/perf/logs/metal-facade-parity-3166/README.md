# Transformer・Functional・可変長系列 7 群の Metal 実機 parity（#3166）

親 #3160（phase:8）の Metal 側。対象は `#[ignore]` 付き Metal テスト 8 本（7 群）。機体は Mac mini（Apple M4）
（`docs/real-hardware-verification-env.md` §1・§7）。M4 Max 開発機では測らない。

- 事前登録: [`RULE.txt`](./RULE.txt)（実機結果を見た後に入力・判定・コマンドを変えない）
- 実行スクリプト: [`run_metal_parity.sh`](./run_metal_parity.sh)（Mac mini 上で手動実行。CI からは呼ばない）
- 判定契約（REQ-2 統一複合判定・tolerance 定数・parity baseline）は不変

## 状態

段 B 完了（2026-10-19・Mac mini〈Apple M4〉・実行コミット 6570171a〈フル SHA は [`rev-stamp.txt`](./rev-stamp.txt)。dirty=0〉）。
**8 本中 pass 8・fail 0・undetermined 0**。事前登録は 2026-10-19T04:11Z に再登録した（旧宣言の対象コミットは main の perf 変更で有効条件を満たさなくなったため）。
環境は [`env_info.txt`](./env_info.txt)、ビルドは [`build.log`](./build.log)（判定外）、集計は [`summary.tsv`](./summary.tsv)。
各ログは `../<群>/metal/<テスト関数名>.log`（群ごとの `metal/rev-stamp.txt` は中央のものの写し）。
GPU カーネル経路か Unsupported によるホストフォールバックかはログから断定できず、確かめていない。
測定はデバッグビルド（`--release` なし）で、所要秒は参考値（時間計測ではない）。

## 結果表

| # | 群（ログディレクトリ） | テスト | 結果 |
|---|---|---|---|
| 1 | [mha-config-sequential-2530](../mha-config-sequential-2530/README.md) | `metal_multihead_attention_config_matches_cpu` | pass（1/1・0.05s） |
| 2 | [transformer-decoder-sequential-2532](../transformer-decoder-sequential-2532/README.md) | `metal_transformer_decoder_layer_matches_cpu` | pass（1/1・0.05s） |
| 3 | [transformer-sequential-2533](../transformer-sequential-2533/README.md) | `metal_transformer_matches_cpu` | pass（1/1・0.06s） |
| 4 | [functional-graph-2665](../functional-graph-2665/README.md) | `compat::functional::tests::metal_functional_graph_matches_cpu_reference`（`--lib`） | pass（1/1・0.05s） |
| 5 | [functional-fit-2667](../functional-fit-2667/README.md) | `compat::functional::fit_parity_tests::metal_parameter_gradients_match_cpu_reference`（`--lib`） | pass（1/1・0.10s） |
| 6 | [merge-ops-2666](../merge-ops-2666/README.md) | `metal_merge_ops_match_cpu_reference` | pass（1/1・0.05s） |
| 7 | [merge-ops-2666](../merge-ops-2666/README.md) | `compat::functional::merge_tests::metal_merge_graph_matches_cpu_reference`（`--lib`） | pass（1/1・0.05s） |
| 8 | [packed-sequence-2647](../packed-sequence-2647/README.md) | `metal_packed_sequence_matches_cpu_reference` | pass（1/1・0.26s） |

## 手順

1. RULE.txt の条件（`crates/` 等が対象コミットと同一）を `git diff --quiet` で確かめ、worktree が clean であることを確認する。
2. `docs/real-hardware-verification-env.md` §3 の rsync で Mac mini の専用ディレクトリ `~/work/rust-ai-library-run-3166/` へ転送し、
   2 行形式の `.rev-stamp`（SHA + `dirty=0`）の全桁一致を確認する（転送元の `.rev-stamp` は転送後に削除し、コミットしない）。
3. ノード側で切り離して `zsh -lc` 経由で `bash docs/perf/logs/metal-facade-parity-3166/run_metal_parity.sh <tree> <未存在の out>` を実行する
   （`CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai`）。
4. 出力先を回収し、ホスト名・ユーザー名・`/Users/` が無いことを再確認してから、`rev-stamp.txt`・`env_info.txt`・`build.log`・`summary.tsv`
   を本ディレクトリへ、各ログを `../<群>/metal/<テスト関数名>.log` へ置く。
5. 各群 README の Metal 行と上表、`docs/README.md` の記述を実測結果（pass／fail と件数）で更新する。fail は数値を引用し「要対応」とする。
