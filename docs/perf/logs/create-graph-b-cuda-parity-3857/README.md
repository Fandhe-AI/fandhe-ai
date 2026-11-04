# 二階微分（create_graph）の B 区分と拒否範囲の CUDA 実機 parity（#3857）

対象は `crates/facade/tests/create_graph_gpu_parity.rs` の `#[ignore]` 付き CUDA テスト 15 本（B 区分 #3797〜#3808 の 12 族、
拒否範囲 #3809・#3810 の 2 族、追跡表の項目 5 に含まれる #3811 の 1 族）と、経路の自己検査 1 本（scaffold。測定対象外）。
機体は `<cuda-node>`（DGX Spark GB10。`docs/real-hardware-verification-env.md`）。雛形は `../create-graph-gpu-parity-template/RULE.txt`
（#3776）、測り方は `docs/autodiff-higher-order-grad-decision.md` §48。

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-04T12:50Z。実機実行より前にコミット 48d0263ac で push した。実行後に変えていない）
- 実行スクリプト: [`run_cuda_parity.sh`](./run_cuda_parity.sh)（#3856 版 `../create-graph-c-cuda-parity-3856/run_cuda_parity.sh` の複製で、
  対象一覧だけを変えた。GB10 上で手動実行。CI からは呼ばない。`--self-test` 付き）
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline（空のまま）・判定規則・テストコードは不変

## 状態

**実測済み: 測定対象 15 本中 15 本 pass、fail 0、undetermined 0**（2026-11-04・`<cuda-node>`）。自己検査（scaffold）も pass。

- イシュー本文の対象（#3797〜#3810）のうち族があるもの 14 本はすべて pass。#3801（CtcLoss）は #3767 で対象外（恒久 fail-closed。
  decision §21.2・§60）に確定しており、登録表に族が無いため測定対象外（未実測にも pass にも数えない）。
- #3811（`embedding_padding`）はイシュー本文の列挙に無いが、追跡表の項目 5 が #3811 を含み、その実機 parity 欄が #3857 を GB10 の担当と
  しているため、RULE.txt で事前宣言して同じテストバイナリで測った。
- 実行コミット（`rev-stamp.txt`）: `48d0263ac3f7fb69f336d49c1203936fa8b787ae`・`dirty=0`。事前登録と実行スクリプトを含むコミットそのもので、
  `crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は対象コミット `b82b1e5d1` と同一（送信前に `git diff --quiet` で確認）。
  転送は rsync だけで、ノード側で `.rev-stamp` とスクリプトの SHA-256 を読んで送信側と照合した。
- スクリプト対応: `env_info.txt` の `script_sha256`（`4c1905ca…9edb`）は、上記コミットの `run_cuda_parity.sh` の SHA-256 と一致する。
- 実機ロック: `env_info.txt` の実行前・実行後とも `hw_lock: held owner=#3857 2026-11-04T12:51:06Z`。終了後にロックが外れていることを確かめた。
  ほかのサービスのプロセスは止めていない（常駐 compute プロセスは実行前後とも 3 件。GPU 使用率は前後とも 0%）。並走する cargo／rustc は無し。
- 回収物は手で編集していない。`SHA256SUMS` はこのディレクトリで `shasum -a 256 -c SHA256SUMS`（Linux は `sha256sum -c`）が通る。
  ホスト名・`/home/`・ユーザー名の残存は 0 件（`<cuda-node>`／`<user>`／`<home>` へマスク済み）。
- デバイス経路: 各テストは `fandhe_ai::tape_for(Device::Cuda(0))` を `expect` で受ける（`create_graph_gpu_parity.rs` の `tapes_for`）。
  CUDA が使えなければ panic するため、CPU への無言のフォールバックで pass することは無い。route H のケースの pass は経路の parity であり、
  カーネル同士の parity ではない（GPU linalg ゲートは既定 OFF のまま測ったので、linalg 族はホスト実行の経路を測っている）。
- 各テストは 0.34〜1.05 秒で終わった（`test result` 行の `finished in`）。

## 結果表

判定契約は全ケース `Judge::Strict`（Q1 同デバイス bit 一致＋GPU・CPU の REQ-2 複合判定／Q2 cg.grad・Q3 H v〈ゼロでない〉・
Q4 孫テープ再生の REQ-2 統一複合判定／Q5 拒否は `Err(Backward)` かつ子テープが空）。

| # | 実装 issue | テスト（`--exact`） | ケース（route） | 結果 |
|---|---|---|---|---|
| 0 | #3776（自己検査） | `create_graph_scaffold_cuda_matches_cpu` | scaffold_cubic（K） | pass |
| 1 | #3797 | `create_graph_conv3d_family_cuda_matches_cpu` | conv3d・conv_transpose3d（H） | pass |
| 2 | #3798 | `create_graph_loss_regression_cuda_matches_cpu` | kl_div（K）、poisson_nll・gaussian_nll（H） | pass |
| 3 | #3799 | `create_graph_loss_binary_margin_cuda_matches_cpu` | bce（K）、bce_pos_weight・soft_margin・hinge_embedding・margin_ranking・sigmoid_focal（H） | pass |
| 4 | #3800 | `create_graph_loss_class_cuda_matches_cpu` | cewo・multi_margin・multilabel_margin・multilabel_soft_margin・cosine_embedding・triplet_margin（H） | pass |
| — | #3801 | （族なし） | CtcLoss は対象外（#3767） | 測定対象外 |
| 5 | #3802 | `create_graph_scatter_reduce_cuda_matches_cpu` | sum・mean・amax・amin・prod（H） | pass |
| 6 | #3803 | `create_graph_cumulative_b_cuda_matches_cpu` | cumprod（K）、logcumsumexp（H） | pass |
| 7 | #3804 | `create_graph_stft_family_cuda_matches_cpu` | stft・istft（H） | pass |
| 8 | #3805 | `create_graph_linalg_cholesky_pinv_lstsq_cuda_matches_cpu` | cholesky・pinv_tall・pinv_wide・lstsq（H） | pass |
| 9 | #3806 | `create_graph_linalg_norm_rank_cuda_matches_cpu` | matrix_norm_fro・one・inf、slogdet_sign、matrix_rank（H） | pass |
| 10 | #3807 | `create_graph_linalg_qr_cuda_matches_cpu` | qr_tall・qr_square（H） | pass |
| 11 | #3808 | `create_graph_linalg_svd_cuda_matches_cpu` | svd_tall・svd_wide（H） | pass |
| 12 | #3808 | `create_graph_linalg_eigh_cuda_matches_cpu` | eigh_sym3（H） | pass |
| 13 | #3809 | `create_graph_norm_prescale_cuda_matches_cpu` | layer_norm_wide（K） | pass |
| 14 | #3810 | `create_graph_det_family_rowscale_cuda_matches_cpu` | diag_large（H） | pass |
| 15 | #3811（項目 5 分） | `create_graph_embedding_padding_cuda_matches_cpu` | dup_ids（K） | pass |

各行の根拠は `summary.tsv`（exit code・`running 1 test` の件数・`test result` 行）と `logs/<群>/<テスト名>.log`（末尾に `rc=0`）。
対象の実装 issue #3797〜#3811 は既に CLOSED。Metal（Mac mini）は #3859 が測る。

## 回収物

| ファイル | 内容 |
|---|---|
| `rev-stamp.txt` | 展開ツリーの `.rev-stamp`（2 行形式） |
| `env_info.txt` | 実行前後の環境（GPU・ドライバ・nvcc・rustc・負荷・常駐件数・ロック保持） |
| `build.log` | 事前ビルド（`--no-run`）のログ |
| `summary.tsv` | 16 行の判定表 |
| `logs/<群>/` | 各テストの全出力（`--nocapture`）と終了コード |
| `SHA256SUMS` | 上記の SHA-256 |
