# f64 自動微分・create_graph・jvp／jacfwd の CUDA 実機 parity（#3168）

対象は `#[ignore]` 付き CUDA テスト 7 本（5 群）。機体は DGX Spark GB10（`docs/real-hardware-verification-env.md`）。
#3165（`../cuda-facade-parity-3165/`）と同じ手順・同型のスクリプトで測った。

- 事前登録: [`RULE.txt`](./RULE.txt)（実機結果を見た後に入力・判定・コマンドを変えない）
- 実行スクリプト: [`run_cuda_parity.sh`](./run_cuda_parity.sh)（GB10 上で手動実行。CI からは呼ばない。`--self-test` 付き）
- tolerance 定数・parity baseline・判定規則・テストコードは不変

## 状態

**実測済み: 7 本中 7 本 pass、fail 0、undetermined 0**（2026-10-19・DGX Spark GB10）。

- 実行コミット（`rev-stamp.txt`・各群 `cuda/rev-stamp.txt`）: `9e2f7ddcc1e1e07d46480f52bba0327ee877f5d1`。事前登録（RULE.txt）と
  スクリプトを含むコミットそのもので、送信時に作業ツリーは clean、`crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は
  対象コミット `d8e38b479` と同一。
- スクリプト対応: `env_info.txt` の `script_sha256`（`09027a7d…55f5`）は、上記コミットの `run_cuda_parity.sh` の SHA-256 と一致する。
  ノードでも `--self-test` は pass した（開発機 Mac の BSD sed・bash 3.2 でも pass）。
- 回収物は手で編集していない。`SHA256SUMS` はこのディレクトリで `shasum -a 256 -c SHA256SUMS`（Linux は `sha256sum -c`）が通る。
  ホスト名・`/home/`・ユーザー名の残存は 0 件。
- 備考: 各テストは 0.2〜1.4 秒で終わった。GPU 経路かホストフォールバックかは確かめていない
  （pass は CUDA デバイス指定の実行結果が参照と判定内で一致したことのみを示す）。GB10 は常駐プロセス 3 件あり、実行前後の GPU 使用率は 0〜1%。
- 5 番は索引が「実測は #3168・#3169」と申し送っていた #3087 の拡張 Op の分で、同じテストバイナリなので本イシューの表（6 本）に加えて測った。
  4 番は型付きエラー（`DeviceMismatch`）の検査で parity ではなく、exit code と出力件数で判定した。
- 既に 2026-10-09 に pass 済みの `functional_ops_jvp_backend_parity`（double-VJP ミラー）は再測していない。Metal は #3169。

## 結果表

| # | 群（ログディレクトリ） | テスト | 判定契約 | 結果 |
|---|---|---|---|---|
| 1 | [f64-autograd-facade-2599](../f64-autograd-facade-2599/README.md) | `f64_autograd_cuda_matches_cpu` | REQ-2 | pass（1/1） |
| 2 | [var-f64-autograd-backend-parity-3168](../var-f64-autograd-backend-parity-3168/README.md) | `cuda_f64_autograd_forward_and_grad_match_host_reference` | bit 完全一致 | pass（1/1） |
| 3 | [create-graph-facade-2545](../create-graph-facade-2545/README.md) | `create_graph_hessian_cuda_matches_cpu` | REQ-2 | pass（1/1） |
| 4 | [create-graph-facade-2545](../create-graph-facade-2545/README.md) | `create_graph_rejects_device_mismatch` | 型付きエラー | pass（1/1） |
| 5 | [create-graph-extended-ops-3087](../create-graph-extended-ops-3087/README.md) | `create_graph_extended_ops_cuda_matches_cpu` | REQ-2 | pass（1/1） |
| 6 | [functional-transforms-jvp-2942](../functional-transforms-jvp-2942/README.md) | `cuda_jvp_jacfwd_basic_ops_match_cpu_reference` | REQ-2 | pass（1/1） |
| 7 | [functional-transforms-jvp-2942](../functional-transforms-jvp-2942/README.md) | `cuda_jvp_jacfwd_extended_ops_match_cpu_reference` | REQ-2 | pass（1/1） |

## 手順

1. スクリプトと RULE.txt をコミットし、`git status --porcelain` が空で、`git diff --quiet <対象コミット> HEAD -- crates/ Cargo.toml Cargo.lock rust-toolchain.toml` が成立することを確かめる。
2. `git rev-parse HEAD > .rev-stamp`（コミットしない）を作り、rsync で GB10 の専用ディレクトリ `~/work/rust-ai-library-run-3168` へ転送して、`.rev-stamp` とスクリプトの SHA-256 の一致を確認する。
3. GB10 上で `run_cuda_parity.sh <tree> <未存在の out>` を実行する。
4. 出力先を回収し、`SHA256SUMS` で無改変を確かめる。**ログは手で編集しない。** 漏れがあればスクリプトを直して測り直す。
5. 回収物をそのまま本ディレクトリと各群の `cuda/` へ置き、各群 README の CUDA 行と `docs/README.md` を更新する。
