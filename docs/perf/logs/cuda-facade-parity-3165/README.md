# Transformer・Functional・可変長系列 7 群の CUDA 実機 parity（#3165）

親 #3160（phase:8）の CUDA 側。対象は `#[ignore]` 付き CUDA テスト 8 本（7 群）。機体は DGX Spark GB10
（`docs/real-hardware-verification-env.md`）。PR #3469 の測り直し（回収後のログ手編集と、実行スクリプトとコミット済みスクリプトの不一致により close）。

- 事前登録: [`RULE.txt`](./RULE.txt)（実機結果を見た後に入力・判定・コマンドを変えない）
- 実行スクリプト: [`run_cuda_parity.sh`](./run_cuda_parity.sh)（GB10 上で手動実行。CI からは呼ばない）
- 判定契約（REQ-2 統一複合判定・tolerance 定数・parity baseline）は不変

## 状態

未実測（コミット A 時点）。事前登録とスクリプトのみ。`bash run_cuda_parity.sh --self-test` は pass
（マスクが crate 名・パッケージ名を変えないこと、ユーザー名・ホスト名・HOME を置換することを確認）。

## 結果表

| # | 群（ログディレクトリ） | テスト | 結果 |
|---|---|---|---|
| 1 | [mha-config-sequential-2530](../mha-config-sequential-2530/README.md) | `cuda_multihead_attention_config_matches_cpu` | |
| 2 | [transformer-decoder-sequential-2532](../transformer-decoder-sequential-2532/README.md) | `cuda_transformer_decoder_layer_matches_cpu` | |
| 3 | [transformer-sequential-2533](../transformer-sequential-2533/README.md) | `cuda_transformer_matches_cpu` | |
| 4 | [functional-graph-2665](../functional-graph-2665/README.md) | `compat::functional::tests::cuda_functional_graph_matches_cpu_reference`（`--lib`） | |
| 5 | [functional-fit-2667](../functional-fit-2667/README.md) | `compat::functional::fit_parity_tests::cuda_parameter_gradients_match_cpu_reference`（`--lib`） | |
| 6 | [merge-ops-2666](../merge-ops-2666/README.md) | `cuda_merge_ops_match_cpu_reference` | |
| 7 | [merge-ops-2666](../merge-ops-2666/README.md) | `compat::functional::merge_tests::cuda_merge_graph_matches_cpu_reference`（`--lib`） | |
| 8 | [packed-sequence-2647](../packed-sequence-2647/README.md) | `cuda_packed_sequence_matches_cpu_reference` | |

## 手順

1. スクリプトと RULE.txt をコミットし、`git status --porcelain` が空で、`git diff --quiet <対象コミット> HEAD -- crates/ Cargo.toml Cargo.lock rust-toolchain.toml` が成立することを確かめる。
2. `git rev-parse HEAD > .rev-stamp`（コミットしない）を作り、rsync で GB10 の専用ディレクトリへ転送して、`.rev-stamp` とスクリプトの SHA-256 の一致を確認する。
3. GB10 上で `run_cuda_parity.sh <tree> <未存在の out>` を実行する。
4. 出力先を回収し、`SHA256SUMS` で無改変を確かめる。**ログは手で編集しない。** 漏れがあればスクリプトを直して測り直す。
5. 回収物をそのまま本ディレクトリと各群の `cuda/` へ置き、各群 README の CUDA 行と `docs/README.md` を更新する。
