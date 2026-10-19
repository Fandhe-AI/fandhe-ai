# Transformer・Functional・可変長系列 7 群の CUDA 実機 parity（#3165）

親 #3160（phase:8）の CUDA 側。対象は `#[ignore]` 付き CUDA テスト 8 本（7 群）。機体は DGX Spark GB10
（`docs/real-hardware-verification-env.md` §1）。Metal 側は [`metal-facade-parity-3166`](../metal-facade-parity-3166/README.md)。

- 事前登録: [`RULE.txt`](./RULE.txt)（実機結果を見た後に入力・判定・コマンドを変えない。実測より前のコミットで固定）
- 実行スクリプト: [`run_cuda_parity.sh`](./run_cuda_parity.sh)（GB10 上で手動実行。CI からは呼ばない）
- 判定契約（REQ-2 統一複合判定・tolerance 定数・parity baseline）は不変

## 結果表

| # | 群（ログディレクトリ） | テスト | 結果 |
|---|---|---|---|
| 1 | [mha-config-sequential-2530](../mha-config-sequential-2530/README.md) | `cuda_multihead_attention_config_matches_cpu` | 未実測 |
| 2 | [transformer-decoder-sequential-2532](../transformer-decoder-sequential-2532/README.md) | `cuda_transformer_decoder_layer_matches_cpu` | 未実測 |
| 3 | [transformer-sequential-2533](../transformer-sequential-2533/README.md) | `cuda_transformer_matches_cpu` | 未実測 |
| 4 | [functional-graph-2665](../functional-graph-2665/README.md) | `compat::functional::tests::cuda_functional_graph_matches_cpu_reference`（`--lib`） | 未実測 |
| 5 | [functional-fit-2667](../functional-fit-2667/README.md) | `compat::functional::fit_parity_tests::cuda_parameter_gradients_match_cpu_reference`（`--lib`） | 未実測 |
| 6 | [merge-ops-2666](../merge-ops-2666/README.md) | `cuda_merge_ops_match_cpu_reference` | 未実測 |
| 7 | [merge-ops-2666](../merge-ops-2666/README.md) | `compat::functional::merge_tests::cuda_merge_graph_matches_cpu_reference`（`--lib`） | 未実測 |
| 8 | [packed-sequence-2647](../packed-sequence-2647/README.md) | `cuda_packed_sequence_matches_cpu_reference` | 未実測 |

## 手順

1. RULE.txt の条件（`crates/` 等が対象コミットと同一）を `git diff --quiet` で確かめ、worktree が clean であることを確認する。
2. `docs/real-hardware-verification-env.md` §3 の rsync で GB10 の `~/work/rust-ai-library-run-3165/` へ転送し、
   ノード側に 2 行形式の `.rev-stamp`（SHA + `dirty=0`）を書いて全桁一致を確認する。
3. ノード上で `bash docs/perf/logs/cuda-facade-parity-3165/run_cuda_parity.sh <tree> <未存在の out>` を実行する。
4. 出力先を回収し、ホスト名・ユーザー名・`/home/` が無いことを再確認してから、`rev-stamp.txt`・`env_info.txt`・`build.log`・`summary.tsv`
   を本ディレクトリへ、各ログを `../<群>/cuda/<テスト関数名>.log` へ置く。
5. 各群 README の CUDA 行と上表、`docs/README.md` の記述を実測結果（pass／fail と件数）で更新する。fail は数値を引用し「要対応」とする。
