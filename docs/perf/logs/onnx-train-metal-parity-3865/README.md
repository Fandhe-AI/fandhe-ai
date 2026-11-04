# ONNX モデルの学習の GPU tape 実機 parity（Metal・#3865）

対象は ONNX モデルの学習（#3842 の公開経路 `OnnxModel::from_sequential` → `bind_autograd` → forward → `MseLoss(Mean)` → backward）で、
`crates/facade/tests/train_gpu_parity.rs` の onnx 族を Metal の GPU tape で CPU tape と突き合わせた。経路の自己検査として scaffold 族も 1 本実行した
（測定対象外）。機体は `<metal-node>`（Mac mini・Apple M4・メモリ 16GB。`docs/real-hardware-verification-env.md` §7）。GB10 版は #3864。
雛形は `../train-gpu-parity-template/RULE.txt`（#3778）。

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-04T14:19Z。実機実行より前にコミット 55fa6e3e3 で push した。実行後に変えていない）
- 実行スクリプト: [`run_metal_parity.sh`](./run_metal_parity.sh)（Mac mini 上で手動実行。CI からは呼ばない。`--self-test` 付き）。
  `../param-groups-resident-metal-parity-3863/run_metal_parity.sh`（#3863）の複製で、対象一覧だけを変えた
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline・判定規則・テストコードは不変

## 状態

**実測済み: 測定対象 1 本（onnx 族・2 ケース 12 量）が pass、fail 0、undetermined 0**（2026-11-04・`<metal-node>`）。自己検査（scaffold 族・3 ケース 16 量）も pass。

- 実行コミット（`rev-stamp.txt`）: `55fa6e3e3255aba4df28491bb567911d7b3e1226`・`dirty=0`。事前登録と実行スクリプトを含むコミットそのもので、
  `crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は対象コミット `b82b1e5d1` と同一（送信前に `git diff --quiet` で確認）。
  転送は rsync だけで、ノード側で `.rev-stamp` を読んで送信側の HEAD と照合した。
- スクリプト対応: `env_info.txt` の `script_sha256`（`c48a7e7e…f966`）は、上記コミットの `run_metal_parity.sh` の SHA-256 と一致する。
- 実機ロック: `env_info.txt` の実行前・実行後とも `hw_lock: held owner=#3865 2026-11-04T14:19:19Z`。終了後にロックが外れていることを確かめた。
  ほかの利用者のプロセスは止めていない。並走する cargo／rustc は事前ビルド前・各テスト前とも無し。メモリの空き率は実行前後とも 71%。
- 回収物は手で編集していない。`SHA256SUMS` はこのディレクトリで `shasum -a 256 -c SHA256SUMS` が通る（8 ファイルすべて OK）。
  ホスト名・`/Users/`・ユーザー名の残存は 0 件（`<metal-node>`／`<user>`／`<home>` へマスク済み。送信側でも grep で確認）。
- デバイス経路: 腕 A は `fandhe_ai::tape_for(Device::Metal)`。Err なら run 関数の Err で undetermined になり、CPU への無言のフォールバックは無い。
  onnx 族は route H（`bind_autograd` の 11 op の計算はホストで実行される）で、pass は経路の parity でありカーネル同士の parity ではない。
- 事前ビルド（`--no-run`）は 10.19 秒（`build.log`）。各テストは 0.20〜0.28 秒で終わった。

## 結果表（`TRAIN_PARITY` 行の集計。判定はすべて Req2）

| # | 族 | ケース | 量 | fail_count | 最大絶対誤差 | 結果 |
|---|---|---|---|---|---|---|
| 0 | scaffold（自己検査） | s1・s2・sgd_resident_vs_host | 16 量 | 0 | 4.470348e-8 | pass |
| 1 | onnx（#3842） | `onnx_from_sequential_forward_backward_s1` | forward・loss・grad_0_bias・grad_0_weight・grad_2_bias・grad_2_weight | 0 | 0 | pass |
| 1 | onnx（#3842） | `onnx_from_sequential_forward_backward_s2` | 同上 | 0 | 2.980232e-8（loss。ほかは 0） | pass |

onnx 族の 12 量はすべて REQ-2 統一複合判定で一致した（loss 以外の 11 量は CPU tape と誤差 0）。各量の値は
`logs/onnx-train-3842/train_parity_onnx_metal_matches_cpu.log` の `TRAIN_PARITY` 行（total・fail_count・max_abs_diff・mean_abs_diff・max_rel_err・verdict）。
対象の実装 issue #3842 は既に CLOSED。CUDA（GB10）は #3864 が測る。申し送り元 `../onnx-autograd-2078/README.md` に結果を追記した。

## 回収物

| ファイル | 内容 |
|---|---|
| `rev-stamp.txt` | 展開ツリーの `.rev-stamp`（2 行形式） |
| `env_info.txt` | 実行前後の環境（macOS・CPU・メモリ量と空き率・rustc・負荷・ロック保持） |
| `build.log` | 事前ビルド（`--no-run`）のログ |
| `summary.tsv` | 2 行の判定表 |
| `logs/<群>/` | 各テストの全出力（`--nocapture`）と終了コード |
| `SHA256SUMS` | 上記の SHA-256 |
