# ONNX モデルの学習の GPU tape 実機 parity（CUDA。#3864）

対象は #3842 の公開経路（`OnnxModel::from_sequential` → `bind_autograd` → `forward` → `MseLoss(Mean)` → `backward`）で GPU 上に作った
`Tape` の学習結果。`crates/facade/tests/train_gpu_parity.rs` の onnx 族（`train_parity_onnx_cuda_matches_cpu`）を、`<cuda-node>`
（DGX Spark GB10。`docs/real-hardware-verification-env.md`）で CPU tape と突き合わせた。#2078 の申し送り（`../onnx-autograd-2078/README.md`）の測定先。
雛形は `../train-gpu-parity-template/RULE.txt`（#3778）。

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-04T13:03Z。実機実行より前にコミット a05e3caeb で push した。実行後に変えていない）
- 実行スクリプト: [`run_cuda_parity.sh`](./run_cuda_parity.sh)（#3862 版 `../param-groups-resident-cuda-parity-3862/run_cuda_parity.sh` の複製で、
  対象一覧だけを変えた。GB10 上で手動実行。CI からは呼ばない。`--self-test` 付き）
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline・判定規則・テストコードは不変

## 状態

**実測済み: 測定対象（onnx 族）1 本 pass、fail 0、undetermined 0**（2026-11-04・`<cuda-node>`）。2 ケース 12 量がすべて REQ-2 pass。
経路の自己検査（scaffold 族。3 ケース 16 量）も pass。

- 実行コミット（`rev-stamp.txt`）: `a05e3caeb2d303cbceb15102dfea3df1b9fb68b1`・`dirty=0`。事前登録と実行スクリプトを含むコミットそのもので、
  `crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は対象コミット `b82b1e5d1` と同一（送信前に `git diff --quiet` で確認）。
  転送は rsync だけで、ノード側で `.rev-stamp` とスクリプトの SHA-256 を読んで送信側と照合した。
- スクリプト対応: `env_info.txt` の `script_sha256`（`a8e249fd…fea70`）は、上記コミットの `run_cuda_parity.sh` の SHA-256 と一致する。
- 実機ロック: `env_info.txt` の実行前・実行後とも `hw_lock: held owner=#3864 2026-11-04T13:04:02Z`。終了後にロックが外れていることを確かめた
  （ビルドが warm で全体が数秒で終わったため、起動 3 秒後の確認時点で既に解放済みだった）。ほかのサービスのプロセスは止めていない
  （常駐 compute プロセスは実行前後とも 3 件。GPU 使用率は前後とも 0%）。並走する cargo／rustc は無し。
- 回収物は手で編集していない。`SHA256SUMS` はこのディレクトリで `shasum -a 256 -c SHA256SUMS`（Linux は `sha256sum -c`）が通る。
  ホスト名・`/home/`・ユーザー名の残存は 0 件（`<cuda-node>`／`<user>`／`<home>` へマスク済み）。
- デバイス経路: 腕 A は `fandhe_ai::tape_for(Device::Cuda(0))`。`Err` なら run 関数の `Err`（undetermined）になり、CPU への無言のフォールバックは無い。
  route は H（`bind_autograd` の 11 op の計算はホスト実行。対象コミットの CASES のコメント）で、pass は経路の parity でありカーネル同士の parity ではない。
- 集計上の注意: `--nocapture` では各ログの最初の `TRAIN_PARITY` 記録が `test <名前> ... ` と同じ行に出る。下表は行頭に限らず
  `TRAIN_PARITY` 以降を数えた（onnx 族 12 行・scaffold 族 16 行。verdict は 28 行すべて pass）。スクリプトの判定は exit 0 と
  `test result` 行で決まり、`TRAIN_PARITY` の行頭 grep は非 0 終了時の fail／undetermined の区別にだけ使う（今回は非 0 終了なし）。
- テストは scaffold 1.17 秒、onnx 0.24 秒で終わった（`test result` 行の `finished in`）。

## 結果表（onnx 族。`logs/onnx-train-3842/train_parity_onnx_cuda_matches_cpu.log` の `TRAIN_PARITY` 行の写し）

judge は全量 Req2（REQ-2 統一複合判定。相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）、arm は `Cuda(0)`、route は H。

| ケース | 量 | total | fail_count | max_abs_diff | max_rel_err | verdict |
|---|---|---|---|---|---|---|
| onnx_from_sequential_forward_backward_s1 | forward | 16 | 0 | 0.000000e0 | 0.000000e0 | pass |
| 同上 | loss | 1 | 0 | 5.960464e-8 | 1.138059e-7 | pass |
| 同上 | grad_0_bias | 16 | 0 | 0.000000e0 | 0.000000e0 | pass |
| 同上 | grad_0_weight | 128 | 0 | 0.000000e0 | 0.000000e0 | pass |
| 同上 | grad_2_bias | 4 | 0 | 0.000000e0 | 0.000000e0 | pass |
| 同上 | grad_2_weight | 64 | 0 | 0.000000e0 | 0.000000e0 | pass |
| onnx_from_sequential_forward_backward_s2 | forward | 35 | 0 | 0.000000e0 | 0.000000e0 | pass |
| 同上 | loss | 1 | 0 | 2.980232e-8 | 9.010528e-8 | pass |
| 同上 | grad_0_bias | 17 | 0 | 0.000000e0 | 0.000000e0 | pass |
| 同上 | grad_0_weight | 221 | 0 | 0.000000e0 | 0.000000e0 | pass |
| 同上 | grad_2_bias | 7 | 0 | 0.000000e0 | 0.000000e0 | pass |
| 同上 | grad_2_weight | 119 | 0 | 0.000000e0 | 0.000000e0 | pass |

形状は S1: batch=4 d_in=8 hidden=16 d_out=4、S2: batch=5 d_in=13 hidden=17 d_out=7（seed=0xC0FFEE）。

| # | テスト（`--exact`） | 位置づけ | 結果 |
|---|---|---|---|
| 0 | `train_parity_scaffold_cuda_matches_cpu` | 経路の自己検査（測定対象外。3 ケース 16 量・route K） | pass |
| 1 | `train_parity_onnx_cuda_matches_cpu` | 測定対象（#3842） | pass |

対象の実装 issue #3842 は既に CLOSED。Metal（Mac mini）は #3865 が測る。

## 回収物

| ファイル | 内容 |
|---|---|
| `rev-stamp.txt` | 展開ツリーの `.rev-stamp`（2 行形式） |
| `env_info.txt` | 実行前後の環境（GPU・ドライバ・nvcc・rustc・負荷・常駐件数・ロック保持） |
| `build.log` | 事前ビルド（`--no-run`）のログ |
| `summary.tsv` | 2 行の判定表 |
| `logs/<群>/` | 各テストの全出力（`--nocapture`）と終了コード |
| `SHA256SUMS` | 上記の SHA-256 |

## #4044 追記（共有 target の成果物の再利用の棚卸し）

- 分類: **C（再利用の可能性を排除できない）**
- 根拠: build.log に `Compiling` が 0 件（`Finished … in 0.06s`）。テストバイナリ `train_gpu_parity-7f8cdf980f4362cb` は 3862 のビルドにも既に存在し、作った回が記録に無いため、そのときのテストのコードを確かめられない。ライブラリのコードは前任（3862）と同一（差分 0 件）。取り直すかどうかは所有者の判断。
- 判定基準・全 16 計測の表・限界は [`docs/real-hardware-shared-target-reuse-decision.md`](../../../real-hardware-shared-target-reuse-decision.md) §4。本追記は結果を変えない（既存の本文・回収ログは無改変）。
