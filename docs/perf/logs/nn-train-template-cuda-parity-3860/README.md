# 雛形の nn 族（学習ステップ）の CUDA 実機 parity（#3860 の再計測）

`crates/facade/tests/train_gpu_parity.rs` の nn 族（`train_parity_nn_cuda_matches_cpu`）を、`<cuda-node>`（DGX Spark GB10。
`docs/real-hardware-verification-env.md`）で CPU tape と突き合わせた。nn 族の 18 ケースは #4026（PR #4034）が登録した。
前回の #3860（`../nn-layers-cuda-parity-3860/`・PR #4022）では、雛形の nn 族は登録 0 件で undetermined だった。その記録は書き換えていない。
雛形は `../train-gpu-parity-template/RULE.txt`（#3778）。

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-04T19:02Z。実機実行より前にコミット `bf118b365` で push した。実行後に変えていない）
- 実行スクリプト: [`run_cuda_parity.sh`](./run_cuda_parity.sh)（#3864 版 `../onnx-train-cuda-parity-3864/run_cuda_parity.sh` の複製。
  差分はヘッダに書いた 4 点で、記録行の grep を行頭固定にしない、exit 0 でも 18 ケースの出現と verdict を確かめる、
  `tests.tsv`／`summary.tsv` の分担、対象 1 本。GB10 上で手動実行。CI からは呼ばない。`--self-test` 付き）
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline・判定規則・テストコードは変えていない

## 状態

**実測済み: 1 本 pass、fail 0、undetermined 0**（2026-11-04・`<cuda-node>`）。18 ケース 120 量がすべて REQ-2 で pass した
（`fail_count` は全量 0）。

- 実行コミット（`rev-stamp.txt`）: `bf118b365d8fc28b52d579ed31ac3daa001dac73`・`dirty=0`。事前登録と実行スクリプトを入れた最初のコミットで、
  `crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は対象コミット `2f59b9638` と同じ（送信前に `git diff --quiet` で確かめた）。
  転送は rsync だけで、ノード側で `.rev-stamp` とスクリプトの SHA-256 を読み、送信側と照合した。
- スクリプトの対応: `env_info.txt` の `script_sha256`（`1be2808c…72c71`）は、上記コミットの `run_cuda_parity.sh` の SHA-256 と一致する。
- 実機ロック: `env_info.txt` の実行前・実行後とも `hw_lock: held owner=#3860 2026-11-04T19:05:37Z`。終了後にロックが外れていることを確かめた。
  ほかのサービスのプロセスは止めていない（常駐 compute プロセスは実行前後とも 3 件。GPU 使用率は前後とも 0%）。並走する cargo／rustc は無かった。
- 回収物は手で編集していない。`SHA256SUMS` はこのディレクトリで `shasum -a 256 -c SHA256SUMS`（Linux は `sha256sum -c`）が通る。
  ホスト名・`/home/`・ユーザー名の残存は 0 件（`<cuda-node>`／`<user>`／`<home>` へマスク済み）。
- デバイス経路: 腕 A は `fandhe_ai::tape_for(Device::Cuda(0))`。`Err` なら run 関数の `Err`（undetermined）になり、CPU への無言のフォールバックは無い。
  route H のケースの pass は経路の parity であり、カーネル同士の parity ではない。
- 集計: `--nocapture` では最初の `TRAIN_PARITY` 記録が `test train_parity_nn_cuda_matches_cpu ... ` と同じ行に出る（行頭は 119 行、合計 120 行）。
  スクリプトは行頭に限らず拾い、`summary.tsv` に 1 行 1 量で書いた。
- テストは 1.65 秒で終わった（`test result` 行の `finished in`）。

## 結果表（ケースごと。`summary.tsv` の 120 行を集計）

judge は全量 Req2（REQ-2 統一複合判定。相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）、arm は `Cuda(0)`、verdict は全量 pass。
max_abs_diff・max_rel_err はケース内の量の最大で、括弧はその量（0 のときは `-`）。量ごとの値は `summary.tsv` にある。

| ケース | route | 量の数 | 要素数 | fail_count | max_abs_diff | max_rel_err | 追跡表 |
|---|---|---|---|---|---|---|---|
| nn_upsample_scale_factor_train_step_s1 | K | 7 | 1040 | 0 | 1.490116e-08（loss） | 7.832865e-08（loss） | 7 |
| nn_upsample_scale_factor_train_step_s2 | K | 7 | 953 | 0 | 2.980232e-08（loss_after_step） | 9.005881e-08（loss_after_step） | 7 |
| nn_shrink_train_step_s1 | H | 7 | 392 | 0 | 0（-） | 0（-） | 9 |
| nn_shrink_train_step_s2 | H | 7 | 386 | 0 | 2.980232e-08（loss） | 8.733040e-08（loss_after_step） | 9 |
| nn_batch_norm3d_train_step_s1 | K | 9 | 758 | 0 | 5.960464e-08（loss_after_step） | 8.740704e-08（loss_after_step） | 11 |
| nn_batch_norm3d_train_step_s2 | K | 9 | 725 | 0 | 0（-） | 0（-） | 11 |
| nn_channel_shuffle_train_step_s1 | H | 9 | 667 | 0 | 1.490116e-08（loss） | 8.420731e-08（loss） | 14 |
| nn_channel_shuffle_train_step_s2 | H | 9 | 658 | 0 | 0（-） | 0（-） | 14 |
| nn_adaptive_pool3d_train_step_s1 | H | 7 | 248 | 0 | 5.960464e-08（loss_after_step） | 1.659605e-07（loss_after_step） | 15 |
| nn_adaptive_pool3d_train_step_s2 | H | 7 | 251 | 0 | 2.980232e-08（loss_after_step） | 1.331295e-07（loss_after_step） | 15 |
| nn_lp_pool_train_step_s1 | H | 7 | 149 | 0 | 0（-） | 0（-） | 15 |
| nn_lp_pool_train_step_s2 | H | 7 | 125 | 0 | 0（-） | 0（-） | 15 |
| nn_fractional_max_pool_train_step_s1 | H | 7 | 149 | 0 | 2.980232e-08（loss_after_step） | 1.069236e-07（loss_after_step） | 15 |
| nn_fractional_max_pool_train_step_s2 | H | 7 | 179 | 0 | 5.960464e-08（loss） | 8.960326e-08（loss） | 15 |
| nn_bilinear_train_step_s1 | H | 4 | 67 | 0 | 0（-） | 0（-） | 10 |
| nn_bilinear_train_step_s2 | H | 4 | 131 | 0 | 2.384186e-07（loss） | 6.949009e-08（loss_after_step） | 10 |
| nn_similarity_distance_train_step_s1 | H | 3 | 32 | 0 | 0（-） | 0（-） | 12・13 |
| nn_similarity_distance_train_step_s2 | H | 3 | 93 | 0 | 9.536743e-07（loss） | 8.545184e-08（loss） | 12・13 |

量は、Sequential 骨格の 14 ケースが `forward`・`loss`・`grad_p<i>`（スロット順）・`loss_after_step`、
Var グラフの 4 ケース（bilinear・similarity）が `loss`・`grad_p<i>`・`loss_after_step`。形状は `RULE.txt` の `shapes` にある。

| # | テスト（`--exact`） | 位置づけ | 結果 |
|---|---|---|---|
| 1 | `train_parity_nn_cuda_matches_cpu` | 測定対象（#4026 の nn 族 18 ケース） | pass |

対象の実装 issue #3814・#3817〜#3827 は既に CLOSED。Metal（Mac mini）の雛形 nn 族は #3861 が測る。

## 回収物

| ファイル | 内容 |
|---|---|
| `rev-stamp.txt` | 展開ツリーの `.rev-stamp`（2 行形式） |
| `env_info.txt` | 実行前後の環境（GPU・ドライバ・nvcc・rustc・負荷・常駐件数・ロック保持） |
| `build.log` | 事前ビルド（`--no-run`）のログ |
| `tests.tsv` | テスト単位の判定表（1 行） |
| `summary.tsv` | `TRAIN_PARITY` 記録の集計（1 行 1 量・120 行） |
| `logs/nn-train-4026/` | テストの全出力（`--nocapture`）と終了コード |
| `SHA256SUMS` | 上記の SHA-256 |
