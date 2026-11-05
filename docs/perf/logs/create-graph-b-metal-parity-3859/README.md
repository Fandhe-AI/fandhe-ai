# 二階微分（create_graph）の B 区分と拒否範囲の族の Metal 実機 parity（#3859）

対象は `crates/facade/tests/create_graph_gpu_parity.rs` の `#[ignore]` 付き Metal テスト 15 本（#3797〜#3810 の族 14 本と、
項目 5 の #3811 の 1 本）と、経路の自己検査 1 本（scaffold。測定対象外）。機体は `<metal-node>`（Mac mini・Apple M4・メモリ 16GB。
`docs/real-hardware-verification-env.md` §7）。GB10 版は #3857。雛形は `../create-graph-gpu-parity-template/RULE.txt`（#3776・#3777）、
測り方は `docs/autodiff-higher-order-grad-decision.md` §48。

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-04T14:03Z。実機実行より前にコミット bbfcaee6b で push した。実行後に変えていない）
- 実行スクリプト: [`run_metal_parity.sh`](./run_metal_parity.sh)（Mac mini 上で手動実行。CI からは呼ばない。`--self-test` 付き）。
  `../create-graph-c-metal-parity-3858/run_metal_parity.sh`（#3858）の複製で、ISSUE・CASES・N_CASES だけを変えた
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline（空のまま）・判定規則・テストコードは不変

## 状態

**実測済み: 測定対象 15 本中 15 本 pass、fail 0、undetermined 0**（2026-11-04・`<metal-node>`）。自己検査（scaffold）も pass。
イシュー本文の対象（#3797〜#3810 のうち族がある 14 本）はすべて pass。#3801（CtcLoss）は #3767 で対象外に確定し族が無いため測定対象外。

- 実行コミット（`rev-stamp.txt`）: `bbfcaee6bdd1d7cc0034aab14f8832e7ea8d57e1`・`dirty=0`。事前登録と実行スクリプトを含むコミットそのもので、
  `crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は対象コミット `b82b1e5d1` と同一（送信前に `git diff --quiet` で確認）。
  転送は rsync だけで、ノード側で `.rev-stamp` を読んで送信側の HEAD と照合した。
- スクリプト対応: `env_info.txt` の `script_sha256`（`82724d80…6ac3`）は、上記コミットの `run_metal_parity.sh` の SHA-256 と一致する。
- 実機ロック: `env_info.txt` の実行前・実行後とも `hw_lock: held owner=#3859 2026-11-04T14:04:01Z`。終了後にロックが外れていることを確かめた。
  ほかの利用者のプロセスは止めていない。並走する cargo／rustc は事前ビルド前・各テスト前とも無し。メモリの空き率は実行前後とも 72%。
- 回収物は手で編集していない。`SHA256SUMS` はこのディレクトリで `shasum -a 256 -c SHA256SUMS` が通る（36 ファイルすべて OK）。
  ホスト名・`/Users/`・ユーザー名の残存は 0 件（`<metal-node>`／`<user>`／`<home>` へマスク済み。送信側でも grep で確認）。
- デバイス経路: 各テストは `fandhe_ai::tape_for(Device::Metal)` を `expect` で受ける（`create_graph_gpu_parity.rs` の `tapes_for`）。
  Metal が使えなければ `Err` で panic するため、CPU への無言のフォールバックで pass することは無い。route H のケースの pass は
  経路の parity であり、カーネル同士の parity ではない（ホスト関数で計算する Op を含む。linalg 族は GPU linalg ゲート既定 OFF のまま）。
- S3（split-K 非発火）: conv3d と linalg の 5 族は `Listed`（列挙された k は最大 48）、残りは `NoMatMul`。`Listed` の全形状で
  `should_split_k` が `None` であることは各テストの中（`check_split_k_not_fired`）で機械検査され、どのログにも S3 違反は出ていない。
  `Rule`・`Judge::Baseline` の族は無く、baseline の Metal 行は要らなかった。
- 各テストは 0.16〜0.62 秒で終わった（`test result` 行の `finished in`）。

## 結果表

判定契約は全ケース `Judge::Strict`（Q1 同デバイス bit 一致＋GPU・CPU の REQ-2 複合判定／Q2 cg.grad・Q3 H v〈ゼロでない〉・
Q4 孫テープ再生の REQ-2 統一複合判定／Q5 拒否は `Err(Backward)` かつ子テープが空）。

| # | 実装 issue | テスト（`--exact`） | S3 | 結果 |
|---|---|---|---|---|
| 0 | #3777（自己検査） | `create_graph_scaffold_metal_matches_cpu` | NoMatMul | pass |
| 1 | #3797 | `create_graph_conv3d_family_metal_matches_cpu` | Listed | pass |
| 2 | #3798 | `create_graph_loss_regression_metal_matches_cpu` | NoMatMul | pass |
| 3 | #3799 | `create_graph_loss_binary_margin_metal_matches_cpu` | NoMatMul | pass |
| 4 | #3800 | `create_graph_loss_class_metal_matches_cpu` | NoMatMul | pass |
| - | #3801 | （族なし。CtcLoss は対象外） | - | 測定対象外 |
| 5 | #3802 | `create_graph_scatter_reduce_metal_matches_cpu` | NoMatMul | pass |
| 6 | #3803 | `create_graph_cumulative_b_metal_matches_cpu` | NoMatMul | pass |
| 7 | #3804 | `create_graph_stft_family_metal_matches_cpu` | NoMatMul | pass |
| 8 | #3805 | `create_graph_linalg_cholesky_pinv_lstsq_metal_matches_cpu` | Listed | pass |
| 9 | #3806 | `create_graph_linalg_norm_rank_metal_matches_cpu` | NoMatMul | pass |
| 10 | #3807 | `create_graph_linalg_qr_metal_matches_cpu` | Listed | pass |
| 11 | #3808 | `create_graph_linalg_svd_metal_matches_cpu` | Listed | pass |
| 12 | #3808 | `create_graph_linalg_eigh_metal_matches_cpu` | Listed | pass |
| 13 | #3809 | `create_graph_norm_prescale_metal_matches_cpu` | NoMatMul | pass |
| 14 | #3810 | `create_graph_det_family_rowscale_metal_matches_cpu` | NoMatMul | pass |
| 15 | #3811（項目 5） | `create_graph_embedding_padding_metal_matches_cpu` | NoMatMul | pass |

ケースごとの route と S3 の形状は `RULE.txt` の「経路区分」「Metal の S3 宣言」欄。各行の根拠は `summary.tsv`（exit code・
`running 1 test` の件数・`test result` 行）と `logs/<群>/<テスト名>.log`（末尾に `rc=0`）。
対象の実装 issue #3797〜#3811 は既に CLOSED。CUDA（GB10）は #3857（PR #4021・マージ済み）で 15 本 pass。

## 回収物

| ファイル | 内容 |
|---|---|
| `rev-stamp.txt` | 展開ツリーの `.rev-stamp`（2 行形式） |
| `env_info.txt` | 実行前後の環境（macOS・CPU・メモリ量と空き率・rustc・負荷・ロック保持） |
| `build.log` | 事前ビルド（`--no-run`）のログ |
| `summary.tsv` | 16 行の判定表 |
| `logs/<群>/` | 各テストの全出力（`--nocapture`）と終了コード |
| `SHA256SUMS` | 上記の SHA-256 |

## #4044 追記（共有 target の成果物の再利用の棚卸し）

- 分類: **A（再コンパイルを確認）**
- 根拠: build.log に 9 メンバー全部の `Compiling` がある。
- 判定基準・全 16 計測の表・限界は [`docs/real-hardware-shared-target-reuse-decision.md`](../../../real-hardware-shared-target-reuse-decision.md) §4。本追記は結果を変えない（既存の本文・回収ログは無改変）。
