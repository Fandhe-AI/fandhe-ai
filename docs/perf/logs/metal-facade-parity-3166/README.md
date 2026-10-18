# Transformer・Functional・可変長系列 7 群の Metal 実機 parity（#3166）

親 #3160（phase:8）の Metal 側。対象は `#[ignore]` 付き Metal テスト 8 本（7 群）。機体は Mac mini（Apple M4）
（`docs/real-hardware-verification-env.md` §1・§7）。M4 Max 開発機では測らない。

- 事前登録: [`RULE.txt`](./RULE.txt)（実機結果を見た後に入力・判定・コマンドを変えない）
- 実行スクリプト: [`run_metal_parity.sh`](./run_metal_parity.sh)（Mac mini 上で手動実行。CI からは呼ばない）
- 判定契約（REQ-2 統一複合判定・tolerance 定数・parity baseline）は不変

## 状態

段 A（事前登録・実行スクリプト・各 README の Metal 行の機体名の修正）のみ完了。**段 B（Mac mini での実行・ログ回収・記入）は未実施**。
実装エージェントの環境（Linux）から Mac mini に到達できなかった（`docs/real-hardware-verification-env.local.md` が無く、
`METAL_NODE` も未設定）ため、Mac 側へ申し送る。下表の結果は全て未実測で、推測の記入はしない。

## 結果表

| # | 群（ログディレクトリ） | テスト | 結果 |
|---|---|---|---|
| 1 | [mha-config-sequential-2530](../mha-config-sequential-2530/README.md) | `metal_multihead_attention_config_matches_cpu` | 未実測（Mac 側へ申し送り） |
| 2 | [transformer-decoder-sequential-2532](../transformer-decoder-sequential-2532/README.md) | `metal_transformer_decoder_layer_matches_cpu` | 未実測（同上） |
| 3 | [transformer-sequential-2533](../transformer-sequential-2533/README.md) | `metal_transformer_matches_cpu` | 未実測（同上） |
| 4 | [functional-graph-2665](../functional-graph-2665/README.md) | `compat::functional::tests::metal_functional_graph_matches_cpu_reference`（`--lib`） | 未実測（同上） |
| 5 | [functional-fit-2667](../functional-fit-2667/README.md) | `compat::functional::fit_parity_tests::metal_parameter_gradients_match_cpu_reference`（`--lib`） | 未実測（同上） |
| 6 | [merge-ops-2666](../merge-ops-2666/README.md) | `metal_merge_ops_match_cpu_reference` | 未実測（同上） |
| 7 | [merge-ops-2666](../merge-ops-2666/README.md) | `compat::functional::merge_tests::metal_merge_graph_matches_cpu_reference`（`--lib`） | 未実測（同上） |
| 8 | [packed-sequence-2647](../packed-sequence-2647/README.md) | `metal_packed_sequence_matches_cpu_reference` | 未実測（同上） |

## 段 B の手順（Mac 側）

1. 段 A のコミットを含む HEAD で、RULE.txt の条件（`crates/` 等が対象コミットと同一）を `git diff --quiet` で確かめ、
   `git rev-parse HEAD > .rev-stamp` を作る（コミットしない）。
2. `docs/real-hardware-verification-env.md` §3・§7.1 の rsync で Mac mini へ転送し、`.rev-stamp` の全桁一致を確認する。
3. `zsh -lc "CARGO_TARGET_DIR=\$HOME/work/target-fandhe-ai bash <tree>/docs/perf/logs/metal-facade-parity-3166/run_metal_parity.sh <tree> <未存在の out>"` を実行する。
4. 出力先を回収し、ホスト名・`$HOME`・ユーザー名が無いことを再確認してから、`.rev-stamp`・`env_info.txt` を本ディレクトリへ、
   各ログを `../<群>/metal/<テスト関数名>.log` へ置く。
5. 各群 README の Metal 行と上表、`docs/README.md` の記述を実測結果（pass／fail と件数）で更新する。fail は数値を引用し「要対応」とする。
