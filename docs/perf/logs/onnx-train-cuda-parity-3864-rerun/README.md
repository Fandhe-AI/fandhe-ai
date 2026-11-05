# ONNX モデルの学習の GPU tape 実機 parity の測り直し（CUDA。#4046）

元の回 [`../onnx-train-cuda-parity-3864/`](../onnx-train-cuda-parity-3864/README.md)（#3864）は、#4044 の棚卸しで **C**（テストバイナリの由来を確かめられない）になった
（[`docs/real-hardware-shared-target-reuse-decision.md`](../../../real-hardware-shared-target-reuse-decision.md) §4）。
同じ判定規則で、共有 target の対策（全メンバーの `cargo clean -p` と成果物の更新時刻の検査。`docs/real-hardware-verification-env.md` §4.11）を入れて `<cuda-node>`（DGX Spark GB10）で測り直した。
対象は `crates/facade/tests/train_gpu_parity.rs` の `train_parity_onnx_cuda_matches_cpu` の 1 本（`--release` なし、dev プロファイル）。

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-05T06:22:03Z。実機実行より前にコミット `284dcc417` で push した。実行後に変えていない）
- 実行スクリプト: [`run_cuda_parity.sh`](./run_cuda_parity.sh)（雛形 `../hw-parity-run-template/run_parity_template.sh` の複製。差分は固定値の 9 行だけ。手動実行。CI からは呼ばない）
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline・判定規則・判定の helper・テストコードは変えていない

## 状態

**実測済み: pass**（2026-11-05・`<cuda-node>`）。exit 0、`running 1 test`、`test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out; finished in 0.28s`。
onnx 族は 2 ケース 12 量すべて REQ-2 pass で、成果物の確認も成立した。

route は H（`bind_autograd` の 11 op の計算はホスト実行）で、pass は経路の parity であり、カーネル同士の parity ではない。

| 確認項目 | 結果 | 根拠 |
|---|---|---|
| 実行コミット | `284dcc41795d47a2eac9ee896cfb8d2fbc862f94`・`dirty=0`（RULE とスクリプトだけのコミット） | `rev-stamp.txt` |
| スクリプト | `script_sha256` `d89c9f0c…c86a` がコミットのファイルと一致 | `env_info.txt` |
| ソースの指紋 | テストの前後とも `faf953ec…4c8` で、RULE の `source_fingerprint` と一致。回収後にノードで読み直しても同じ | `env_info.txt` |
| 対象ソース | `crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は `target_commit` `ff71e7773` と同一（送信前に確認） | RULE の条件 |
| 実機ロック | owner `#4046 run 2026-11-05T06:22:21Z` を前後とも保持。終了後に解放を確認 | `env_info.txt` |
| 並走 | 他の cargo／rustc なし | `env_info.txt` |
| 実行したバイナリ | `Running` 行は `Executable` 行と同じ `train_gpu_parity-7f8cdf980f4362cb`。`--list` と実行で再コンパイルなし（`Finished … in 0.04s`） | `build.log`・`list_1.log`・`train_parity_onnx_cuda_1.log` |
| 回収物 | `SHA256SUMS` が 9 ファイルとも通る。手で編集していない。ホスト名・`/home/`・ユーザー名の残存 0 件 | `SHA256SUMS` |

## 成果物の確認（`build_artifacts.txt`・`clean.log`・`build.log`）

- clean: 13 メンバーすべてに `cargo clean -p` を実行した（8 メンバーで削除、Linux でこのビルドの対象にならない 5 メンバーは 0 ファイル）
- ビルド: 8 メンバーすべてが `Compiling`（`Finished … in 11.68s`）
- ロック取得（ノード上の owner ファイルの mtime）: 2026-11-05T06:22:22Z
- 17 ファイル（テストバイナリ、8 メンバーの `.rlib`・`.rmeta`。facade 自身の `libfandhe_ai-8edd81afa5c808a3` を含む）がすべて `after_lock`。`NOT_AFTER_LOCK`・`MISSING` は 0 件
- 最も早いものは tensor-core の 06:22:30Z で、ロック取得の 8 秒後。テストバイナリは 06:22:40Z

テストバイナリの名前は元の回と同じ `train_gpu_parity-7f8cdf980f4362cb` で、名前では区別できない（決定記録 §2）。今回の回で作られたことの根拠は、clean の記録と更新時刻である。

## 結果表（onnx 族。`train_parity_onnx_cuda_1.log` の `TRAIN_PARITY` 行の写し）

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

集計（元の回の README と同じく、行頭に限らず `TRAIN_PARITY` 以降を数えた）: onnx 族 12 行、verdict はすべて pass、`fail_count` はすべて 0。
**scaffold 族は 0 行**（`train_parity_scaffold_cuda_matches_cpu` はこの回の対象外で、実行していない。元の回の 16 行はこの回の値ではない）。scaffold は元の回でも完了判定に数えない。

12 行は元の回のログの 12 行とバイト単位で同じだった。

## 元の回との違い

- テストのコード: `target_commit` は元の回の `b82b1e5d1` より新しく、`train_gpu_parity.rs` に nn・optim 族が足されている（#4034）。onnx 族の定義に差分は無い。`12 filtered out`（元の回は 9）はこのため
- スクリプト: 元の回は #3862 版の複製、今回は #4044 の雛形の複製。雛形の `env_info` は常駐 compute プロセスの件数を持たないので、§6.1 の形で送信側で確かめた（下）。雛形の `summary.tsv` は非 0 終了をすべて fail と書く（今回は exit 0 で該当しない。RULE `record:`）
- ロックの owner: `#4046 run <UTC>`。雛形の `LOCK_OWNER_RE` が 3 語を要求するため語 `run` を入れた（RULE `hw_lock:`）
- `REQUIRED_LIBS`: `fandhe_ai` に加えて、このビルドの対象の残り 7 メンバーの lib 名を入れた

## 占有状況（送信側で §6.1 の形で確認）

| 時点（UTC） | 常駐 compute プロセス | GPU 使用率 |
|---|---|---|
| 実行前 06:22:21Z | 3 件 | 0 % |
| 実行後 06:23:06Z | 3 件 | 0 % |

ほかのサービスのプロセスは止めていない。parity 測定で時間計測ではないため専有条件は課していない。

## 限界

- 時刻検査は「成果物がロック取得より後に書かれた」ことを確かめる。内容が現在のソースと対応することは、ソースの指紋と clean の組で担保する
- crates.io の依存の成果物は clean していない（`Cargo.lock` で固定された版の再利用で、意図どおり）

## 回収物

| ファイル | 内容 |
|---|---|
| `rev-stamp.txt` | 展開ツリーの `.rev-stamp`（2 行形式） |
| `env_info.txt` | 実行前後の環境・スクリプトの SHA-256・指紋・ロック保持・`build_artifacts_ok` |
| `clean.log` | clean したメンバーの一覧と削除件数 |
| `build.log` | 事前ビルド（`--no-run`）のログ |
| `list_1.log` | `--ignored --exact --list` の出力（1 件） |
| `build_artifacts.txt` | 成果物の更新時刻とロック取得時刻の照合 |
| `train_parity_onnx_cuda_1.log`・`exit_code_1` | テストの全出力（`--nocapture`）と終了コード |
| `summary.tsv` | 判定（1 行） |
| `SHA256SUMS` | 上記の SHA-256（このディレクトリで `shasum -a 256 -c SHA256SUMS`） |
