# compat::Sequential 層 9 群の Metal 実機 parity 再実測（イシュー #3164・2 回目）

1 回目（PR #3468・コミット `b640875f`。各群の `docs/perf/logs/compat-sequential-*/metal/`）は 10 本中 9 本 pass・1 本 fail
（`metal_dropout_embedding_bag_bit_exact`。`MetalBackendOps::max` が `Unsupported`）だった。#3475（PR #3479）で `max` の
GPU カーネルが結線されたため、イシュー追記（2026-10-19）の条件どおり **10 本すべてを main HEAD 相当で測り直した**。
1 回目の記録は書き換えていない。コード・tolerance 定数・parity baseline・判定式・テスト入力は不変。

## 結果

**10 本中 pass 10 / fail 0 / undetermined 0**（各 1 回・rc=0・`running 1 test`・`1 passed; 0 failed`）。事前登録規則は `RULE.txt`。

| # | 群 | テスト | 判定 | 結果 |
|---|---|---|---|---|
| 1 | activation-layers-2529 | `metal_activation_layers_parity` | bit 一致＋REQ-2 | pass |
| 2 | activation-scalar-layers-2679 | `metal_activation_scalar_layers_parity` | REQ-2 | pass |
| 3 | adaptive-max-global-pool-2527 | `metal_adaptive_max_global_pool_bit_exact` | bit 一致 | pass |
| 4 | conv-transpose2d-2523 | `metal_conv_transpose2d_forward_backward_matches_cpu` | REQ-2 | pass |
| 5 | conv3d-2524 | `metal_conv3d_forward_backward_matches_cpu` | REQ-2 | pass |
| 6 | dropout-embedding-bag-2528 | `metal_dropout_embedding_bag_bit_exact` | bit 一致 | pass（1 回目は fail。#3475 の結線後） |
| 7 | group-instance-norm-2525 | `metal_group_instance_norm_matches_cpu` | REQ-2 | pass |
| 8 | pixel-shuffle-2526 | `metal_pixel_shuffle_layers_bit_exact` | bit 一致 | pass |
| 9 | spatial-2522 | `metal_spatial_layers_nearest_bit_exact` | bit 一致 | pass |
| 10 | spatial-2522 | `metal_spatial_layers_bilinear_matches_cpu` | REQ-2 | pass |

REQ-2 は統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。集計は `summary.tsv`、ログは `logs/<群>/<テスト関数名>.log`、改竄検知は `SHA256SUMS`。

## 有効条件・環境

- 対象コミット: `b728b9cb85b718daea90884cdd66b5e06f357bea`（`RULE.txt`・`run_metal_parity.sh` を追加した事前登録コミット。`rev-stamp.txt`）。
- main との一致: `git diff` が `crates`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` で main `e1a3c5e25e232d6ed7376f786f0af7bb35997c33` と差分なし
  （#3479 の `max` 結線を含む）。送信前の作業ツリーは clean。
- 展開ツリーの `crates/` 指紋（`find crates -type f | sort | xargs shasum | shasum`）: ローカルとノードで一致（`931d909e52d880c618ee9db114de6e73755fdf10`）。
- 環境（`env_info.txt`）: Mac mini（Apple M4）・macOS 27.0.1・rustc 1.99.0・debug プロファイル・並列度 1。ホスト名は記録しない。
  他の cargo／rustc は並走なし（待機 0 回）。実行日 2026-10-19（UTC）。
- 判定外のビルド（`build.log`）を先に済ませ、各テストの前に `--exact --list` で選択件数が 1 であることを検証した（全件 1）。

## 逸脱・注記

- 展開先は既定ではなく専用の `~/work/rust-ai-library-run-3164`（並列セッションとの衝突回避。`RULE.txt` に事前宣言）。
- `RULE.txt` の宣言日時に訂正がある（同ファイル末尾。規則本文は不変）。
- GPU カーネル経路かホストフォールバック経路かはログから断定できず未確認。
- `SHA256SUMS` は回収後にローカルで生成した（実行スクリプトは生成しない）。ログは手で編集していない。
- 対象外: 各決定記録・`site/` の「Metal 実機未実測」表記の更新、他イシューの測り直し、1 回目の成果物の修正。
