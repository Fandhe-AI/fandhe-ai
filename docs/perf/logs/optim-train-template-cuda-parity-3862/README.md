# 雛形の optim 族（param groups 付き GPU 常駐更新）の CUDA 実機 parity（#3862 の再計測）

`crates/facade/tests/train_gpu_parity.rs` の optim 族（`train_parity_optim_cuda_matches_cpu`）を、`<cuda-node>`（DGX Spark GB10。
`docs/real-hardware-verification-env.md`）で測った。腕 A は param groups 付きのデバイス常駐更新（`init_device_param_store_with_param_groups`
ほか）、腕 B はホスト参照（`ParamGroupStep::step_with_groups` ＋ `apply_parameters`）。optim 族の 6 ケースは #4026（PR #4034）が登録した。
前回の #3862（`../param-groups-resident-cuda-parity-3862/`・PR #4023）では、雛形の optim 族は登録 0 件で undetermined だった。
その記録は書き換えていない。雛形は `../train-gpu-parity-template/RULE.txt`（#3778）。

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-04T19:08Z。実機実行より前にコミット `f81eedd9c` で push した。実行後に変えていない）
- 実行スクリプト: [`run_cuda_parity.sh`](./run_cuda_parity.sh)（#3860 再計測版 `../nn-train-template-cuda-parity-3860/run_cuda_parity.sh` の複製で、
  ISSUE・対象・族名・期待ケースだけを変えた。GB10 上で手動実行。CI からは呼ばない。`--self-test` 付き）
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline・判定規則・テストコードは変えていない

## 状態

**実測済み: 1 本 pass、fail 0、undetermined 0**（2026-11-04・`<cuda-node>`）。6 ケース 24 量がすべて REQ-2 で pass した
（`fail_count` は全量 0）。Adam・AdamW の 12 量は `max_abs_diff`・`max_rel_err` とも 0、SGD の 12 量は max_abs_diff 2.980232e-8 以下。

- 実行コミット（`rev-stamp.txt`）: `f81eedd9cd4478ff3f57e65f3d82c9a8e0c3f8a1`・`dirty=0`。事前登録と実行スクリプトを入れた最初のコミットで、
  `crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は対象コミット `2f59b9638` と同じ（送信前に `git diff --quiet` で確かめた）。
  転送は rsync だけで、ノード側で `.rev-stamp` とスクリプトの SHA-256 を読み、送信側と照合した。
- スクリプトの対応: `env_info.txt` の `script_sha256`（`5080bae2…399e5`）は、上記コミットの `run_cuda_parity.sh` の SHA-256 と一致する。
- 実機ロック: `env_info.txt` の実行前・実行後とも `hw_lock: held owner=#3862 2026-11-04T19:09:11Z`。終了後にロックが外れていることを確かめた。
  ほかのサービスのプロセスは止めていない（常駐 compute プロセスは実行前後とも 3 件。GPU 使用率は前後とも 0%）。並走する cargo／rustc は無かった。
- 回収物は手で編集していない。`SHA256SUMS` はこのディレクトリで `shasum -a 256 -c SHA256SUMS`（Linux は `sha256sum -c`）が通る。
  ホスト名・`/home/`・ユーザー名の残存は 0 件（`<cuda-node>`／`<user>`／`<home>` へマスク済み）。
- デバイス経路: 腕 A は `fandhe_ai::tape_for(Device::Cuda(0))`。`Err` なら run 関数の `Err`（undetermined）になり、CPU への無言のフォールバックは無い。
- 集計: `TRAIN_PARITY` 記録 24 行（最初の 1 行は `test train_parity_optim_cuda_matches_cpu ... ` と同じ行）を `summary.tsv` に 1 行 1 量で書いた。
- テストは 1.23 秒で終わった（`test result` 行の `finished in`）。

## 結果表（`summary.tsv` の写し）

judge は全量 Req2（REQ-2 統一複合判定。相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）、arm は `Cuda(0)`、route は K。
量は 20 step 後のパラメータ（`param_w1`・`param_b1`・`param_w2`・`param_b2`）。形状は `RULE.txt` の `shapes` にある。

| ケース | 量 | total | fail_count | max_abs_diff | mean_abs_diff | max_rel_err | verdict |
|---|---|---|---|---|---|---|---|
| optim_param_groups_resident_sgd_s1 | param_w1 | 128 | 0 | 2.980232e-8 | 9.094947e-10 | 2.311265e-7 | pass |
| optim_param_groups_resident_sgd_s1 | param_b1 | 16 | 0 | 2.980232e-8 | 3.841706e-9 | 2.157306e-7 | pass |
| optim_param_groups_resident_sgd_s1 | param_w2 | 64 | 0 | 2.980232e-8 | 3.412424e-9 | 7.364657e-6 | pass |
| optim_param_groups_resident_sgd_s1 | param_b2 | 4 | 0 | 1.490116e-8 | 1.024455e-8 | 2.020266e-7 | pass |
| optim_param_groups_resident_sgd_s2 | param_w1 | 221 | 0 | 1.490116e-8 | 2.187396e-10 | 4.173096e-7 | pass |
| optim_param_groups_resident_sgd_s2 | param_b1 | 17 | 0 | 7.450581e-9 | 6.026205e-10 | 1.653750e-7 | pass |
| optim_param_groups_resident_sgd_s2 | param_w2 | 119 | 0 | 1.490116e-8 | 1.582368e-9 | 1.377204e-6 | pass |
| optim_param_groups_resident_sgd_s2 | param_b2 | 7 | 0 | 2.980232e-8 | 8.248857e-9 | 4.456076e-7 | pass |
| optim_param_groups_resident_adam_s1 | param_w1 | 128 | 0 | 0.000000e0 | 0.000000e0 | 0.000000e0 | pass |
| optim_param_groups_resident_adam_s1 | param_b1 | 16 | 0 | 0.000000e0 | 0.000000e0 | 0.000000e0 | pass |
| optim_param_groups_resident_adam_s1 | param_w2 | 64 | 0 | 0.000000e0 | 0.000000e0 | 0.000000e0 | pass |
| optim_param_groups_resident_adam_s1 | param_b2 | 4 | 0 | 0.000000e0 | 0.000000e0 | 0.000000e0 | pass |
| optim_param_groups_resident_adam_s2 | param_w1 | 221 | 0 | 0.000000e0 | 0.000000e0 | 0.000000e0 | pass |
| optim_param_groups_resident_adam_s2 | param_b1 | 17 | 0 | 0.000000e0 | 0.000000e0 | 0.000000e0 | pass |
| optim_param_groups_resident_adam_s2 | param_w2 | 119 | 0 | 0.000000e0 | 0.000000e0 | 0.000000e0 | pass |
| optim_param_groups_resident_adam_s2 | param_b2 | 7 | 0 | 0.000000e0 | 0.000000e0 | 0.000000e0 | pass |
| optim_param_groups_resident_adamw_s1 | param_w1 | 128 | 0 | 0.000000e0 | 0.000000e0 | 0.000000e0 | pass |
| optim_param_groups_resident_adamw_s1 | param_b1 | 16 | 0 | 0.000000e0 | 0.000000e0 | 0.000000e0 | pass |
| optim_param_groups_resident_adamw_s1 | param_w2 | 64 | 0 | 0.000000e0 | 0.000000e0 | 0.000000e0 | pass |
| optim_param_groups_resident_adamw_s1 | param_b2 | 4 | 0 | 0.000000e0 | 0.000000e0 | 0.000000e0 | pass |
| optim_param_groups_resident_adamw_s2 | param_w1 | 221 | 0 | 0.000000e0 | 0.000000e0 | 0.000000e0 | pass |
| optim_param_groups_resident_adamw_s2 | param_b1 | 17 | 0 | 0.000000e0 | 0.000000e0 | 0.000000e0 | pass |
| optim_param_groups_resident_adamw_s2 | param_w2 | 119 | 0 | 0.000000e0 | 0.000000e0 | 0.000000e0 | pass |
| optim_param_groups_resident_adamw_s2 | param_b2 | 7 | 0 | 0.000000e0 | 0.000000e0 | 0.000000e0 | pass |

| # | テスト（`--exact`） | 位置づけ | 結果 |
|---|---|---|---|
| 1 | `train_parity_optim_cuda_matches_cpu` | 測定対象（#4026 の optim 族 6 ケース） | pass |

対象の実装 issue #3837・#3838 は既に CLOSED。Metal（Mac mini）の雛形 optim 族は #3863 が測る。

## 回収物

| ファイル | 内容 |
|---|---|
| `rev-stamp.txt` | 展開ツリーの `.rev-stamp`（2 行形式） |
| `env_info.txt` | 実行前後の環境（GPU・ドライバ・nvcc・rustc・負荷・常駐件数・ロック保持） |
| `build.log` | 事前ビルド（`--no-run`）のログ |
| `tests.tsv` | テスト単位の判定表（1 行） |
| `summary.tsv` | `TRAIN_PARITY` 記録の集計（1 行 1 量・24 行） |
| `logs/optim-train-4026/` | テストの全出力（`--nocapture`）と終了コード |
| `SHA256SUMS` | 上記の SHA-256 |
