# Transformer・Functional・可変長系列 7 群の CUDA 実機 parity（#3165）

親 #3160（phase:8）の CUDA 側。対象は `#[ignore]` 付き CUDA テスト 8 本（7 群）。機体は DGX Spark GB10
（`docs/real-hardware-verification-env.md`）。PR #3469 の測り直し（回収後のログ手編集と、実行スクリプトとコミット済みスクリプトの不一致により close）。

- 事前登録: [`RULE.txt`](./RULE.txt)（実機結果を見た後に入力・判定・コマンドを変えない）
- 実行スクリプト: [`run_cuda_parity.sh`](./run_cuda_parity.sh)（GB10 上で手動実行。CI からは呼ばない）
- 判定契約（REQ-2 統一複合判定・tolerance 定数・parity baseline）は不変

## 状態

**実測済み: 8 本中 8 本 pass、fail 0、undetermined 0**（2026-10-19・DGX Spark GB10・REQ-2 統一複合判定。tolerance・baseline・判定規則は不変）。

- 実行コミット（`rev-stamp.txt`・各群 `cuda/rev-stamp.txt`）: `a115595c6716b3614869fa157aaa7a3ad8378c76`。事前登録（RULE.txt）と
  スクリプトを含むコミット A そのもので、送信時に作業ツリーは clean、`crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は対象コミット `d79f4895d` と同一。
- スクリプト対応: `env_info.txt` の `script_sha256`（`e6bb0c3a…f354a`）は、コミット A の `run_cuda_parity.sh` の SHA-256 と一致する。
  ノードでも `--self-test` は pass した（GNU sed。開発機 Mac の BSD sed・bash 3.2 でも pass）。
- 回収物は手で編集していない。`SHA256SUMS` はこのディレクトリで `sha256sum -c SHA256SUMS` が通る（`logs/` 配下を含む）。
  ホスト名・ユーザー名・`/home/` の残存は 0 件、`fandhe-ai`・`fandhe_ai*`・`target-fandhe-ai` は変化していない。
  このマスク対象のユーザー名は crate 名と同じ文字列だったため、境界付き置換が実際に効いたことになる。
- RULE.txt の宣言日時（`2026-10-19T08:50Z`）は書き込み時の概算の誤記で、`env_info.txt` の実行時刻（ノード時計 `08:40Z`）より後になっている。
  RULE.txt・スクリプトはコミット A として実機へ送る前に固定した（実行コミット＝コミット A）。時刻の食い違いは事前登録の順序を変える性質のものではないが、事実として記す。
- 備考: 各テストは 0.4〜0.7 秒で終わった。GPU 経路かホストフォールバックかは確かめていない（pass は CUDA デバイス指定の実行結果が CPU 参照と判定内で一致したことのみを示す）。
  GB10 は常駐プロセス 3 件あり、実行前後の GPU 使用率は 0%。

## 結果表

| # | 群（ログディレクトリ） | テスト | 結果 |
|---|---|---|---|
| 1 | [mha-config-sequential-2530](../mha-config-sequential-2530/README.md) | `cuda_multihead_attention_config_matches_cpu` | pass（1/1） |
| 2 | [transformer-decoder-sequential-2532](../transformer-decoder-sequential-2532/README.md) | `cuda_transformer_decoder_layer_matches_cpu` | pass（1/1） |
| 3 | [transformer-sequential-2533](../transformer-sequential-2533/README.md) | `cuda_transformer_matches_cpu` | pass（1/1） |
| 4 | [functional-graph-2665](../functional-graph-2665/README.md) | `compat::functional::tests::cuda_functional_graph_matches_cpu_reference`（`--lib`） | pass（1/1） |
| 5 | [functional-fit-2667](../functional-fit-2667/README.md) | `compat::functional::fit_parity_tests::cuda_parameter_gradients_match_cpu_reference`（`--lib`） | pass（1/1） |
| 6 | [merge-ops-2666](../merge-ops-2666/README.md) | `cuda_merge_ops_match_cpu_reference` | pass（1/1） |
| 7 | [merge-ops-2666](../merge-ops-2666/README.md) | `compat::functional::merge_tests::cuda_merge_graph_matches_cpu_reference`（`--lib`） | pass（1/1） |
| 8 | [packed-sequence-2647](../packed-sequence-2647/README.md) | `cuda_packed_sequence_matches_cpu_reference` | pass（1/1） |

## 手順

1. スクリプトと RULE.txt をコミットし、`git status --porcelain` が空で、`git diff --quiet <対象コミット> HEAD -- crates/ Cargo.toml Cargo.lock rust-toolchain.toml` が成立することを確かめる。
2. `git rev-parse HEAD > .rev-stamp`（コミットしない）を作り、rsync で GB10 の専用ディレクトリへ転送して、`.rev-stamp` とスクリプトの SHA-256 の一致を確認する。
3. GB10 上で `run_cuda_parity.sh <tree> <未存在の out>` を実行する。
4. 出力先を回収し、`SHA256SUMS` で無改変を確かめる。**ログは手で編集しない。** 漏れがあればスクリプトを直して測り直す。
5. 回収物をそのまま本ディレクトリと各群の `cuda/` へ置き、各群 README の CUDA 行と `docs/README.md` を更新する。
