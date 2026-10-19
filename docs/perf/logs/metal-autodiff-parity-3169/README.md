# f64 自動微分・create_graph・jvp／jacfwd の Metal 実機 parity（#3169）

親 #3160（phase:8）の Metal 側。CUDA 側は #3168（`../cuda-autodiff-parity-3168/`）。対象は `#[ignore]` 付き Metal テスト 6 本（5 群）。
機体は Mac mini（Apple M4）（`docs/real-hardware-verification-env.md` §1・§7）。M4 Max 開発機では測らない。

- 事前登録: [`RULE.txt`](./RULE.txt)（実機結果を見た後に入力・判定・コマンドを変えない）
- 実行スクリプト: [`run_metal_parity.sh`](./run_metal_parity.sh)（Mac mini 上で手動実行。CI からは呼ばない。`--self-test` でマスクの不変条件を検査できる）
- 判定契約は各テストのコードが定める（REQ-2 統一複合判定、var-f64 のみ bit 完全一致）。tolerance 定数・parity baseline は不変

## 状態

実測完了（2026-10-19・Mac mini〈Apple M4〉・実行コミット bf894108a〈フル SHA は [`rev-stamp.txt`](./rev-stamp.txt)。dirty=0〉）。
**6 本中 pass 6・fail 0・undetermined 0**。
環境は [`env_info.txt`](./env_info.txt)（`script_sha256` は提出した `run_metal_parity.sh` の SHA-256 と一致）、ビルドは [`build.log`](./build.log)（判定外）、
集計は [`summary.tsv`](./summary.tsv)、無改変検証は `shasum -a 256 -c SHA256SUMS`（本ディレクトリで実行）。
各ログは本ディレクトリの `logs/<群>/` と、群ごとの `../<群>/metal/`（写し。`metal/rev-stamp.txt` は中央のものの写し）にある。
測定はデバッグビルド（`--release` なし）で、所要秒は参考値（時間計測ではない）。

Metal は MSL が `double` 非対応で `typed_ops_f64()` が常に `None` のため、f64 グラフは設計上ホスト経路になる。
1・2 は新規カーネルの parity ではなく「Metal を指定してもホスト経路が参照と一致する」ことの確認である。
3〜6（f32 グラフ）も、GPU カーネル経路か `Unsupported` によるホストフォールバックかはログから断定できず、確かめていない。

## 結果表

| # | 群（ログディレクトリ） | テスト | 判定契約 | 結果 |
|---|---|---|---|---|
| 1 | [f64-autograd-facade-2599](../f64-autograd-facade-2599/README.md) | `f64_autograd_metal_matches_cpu` | REQ-2 | pass（1/1・0.03s） |
| 2 | [var-f64-autograd-backend-parity-3168](../var-f64-autograd-backend-parity-3168/README.md) | `metal_f64_autograd_forward_and_grad_match_host_reference` | bit 完全一致 | pass（1/1・0.00s） |
| 3 | [create-graph-facade-2545](../create-graph-facade-2545/README.md) | `create_graph_hessian_metal_matches_cpu` | REQ-2 | pass（1/1・0.14s） |
| 4 | [create-graph-extended-ops-3087](../create-graph-extended-ops-3087/README.md) | `create_graph_extended_ops_metal_matches_cpu` | REQ-2 | pass（1/1・0.29s） |
| 5 | [functional-transforms-jvp-2942](../functional-transforms-jvp-2942/README.md) | `metal_jvp_jacfwd_basic_ops_match_cpu_reference` | REQ-2（同デバイス jacobian 参照含む） | pass（1/1・0.10s） |
| 6 | [functional-transforms-jvp-2942](../functional-transforms-jvp-2942/README.md) | `metal_jvp_jacfwd_extended_ops_match_cpu_reference` | REQ-2 | pass（1/1・0.21s） |

対象外: `create_graph_rejects_device_mismatch`（CUDA 専用の型付きエラー検査で Metal 版なし）、
`metal_double_vjp_jvp_jacfwd_match_cpu_reference`（double-VJP ミラー。2026-10-09 に M4 Max で pass 記録済みで再測しない）。

## 手順

1. RULE.txt の条件（`crates/` 等が対象コミットと同一）を `git diff --quiet` で確かめ、worktree が clean であることを確認する。
2. `docs/real-hardware-verification-env.md` §3 の rsync で Mac mini の専用ディレクトリ `~/work/rust-ai-library-run-3169/` へ転送し、
   2 行形式の `.rev-stamp`（SHA + `dirty=0`）の全桁一致を確認する（転送元の `.rev-stamp` は転送後に削除し、コミットしない）。
3. ノード側で切り離して `zsh -lc` 経由で `bash docs/perf/logs/metal-autodiff-parity-3169/run_metal_parity.sh <tree> <未存在の out>` を実行する
   （`CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai`）。
4. 出力先を回収し、ホスト名・ユーザー名・`/Users/` が無いことを再確認してから、本ディレクトリと各群の `metal/` へ置く。
5. 各群 README の Metal 行と上表、`docs/README.md` の記述を実測結果で更新する。fail は数値を引用し「要対応」とする。
