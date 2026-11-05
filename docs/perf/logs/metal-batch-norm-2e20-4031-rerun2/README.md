# Metal の BatchNorm 2e20 級入力の実機 parity の 2 回目の測り直し（#4031）

#4031 の判定に使う回である。対象は `crates/backend-metal/tests/batch_norm_parity.rs` の
`batch_norm_train_2e20_scale_has_no_intermediate_overflow_and_matches_cpu_reference` の 1 本で、機体は `<metal-node>`（Mac mini・Apple M4・メモリ 16GB）。
期待値の決め方は `docs/batch-norm-ops-design.md` §9.8。

先の 2 回は、どちらも所有者の判断で無効になった。

| 回 | 無効の理由 |
|---|---|
| [`../metal-batch-norm-2e20-4031/`](../metal-batch-norm-2e20-4031/) | 実行中に展開ツリーが別セッションの rsync で上書きされた |
| [`../metal-batch-norm-2e20-4031-rerun/`](../metal-batch-norm-2e20-4031-rerun/) | 検証対象の `libfandhe_ai_backend_metal` と `libfandhe_ai_tensor_core` が、無効回のビルド成果物の再利用だった |

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-05T03:31:55Z。実機実行より前に、この 2 ファイルだけのコミット `82326b253` で push した。実行後に変えていない）
- 実行スクリプト: [`run_metal_parity.sh`](./run_metal_parity.sh)（rerun の複製。判定は同じで、成果物の削除と確認を足した。差分はヘッダの R4・R5）
- tolerance 定数・parity baseline・閾値・判定の helper・テストコードは不変

## 状態

**実測済み（pass）。** 1 本実行、exit 0、`running 1 test`、`test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 9 filtered out; finished in 0.02s`
（`batch_norm_parity_2e20_4031.log`・`summary.tsv`）。成果物の確認も成立した（下の節）。

| 量 | 判定（RULE.txt の judgment。テストコードが正） | 結果 |
|---|---|---|
| `out` | 全要素が有限、かつ REQ-2 統一複合判定（`assert_parity`）。§9.8 の中間 overflow の主証拠 | pass |
| `mean` | REQ-2 統一複合判定（`assert_parity`） | pass |
| `var` | §9.8 のクラス一致。ch0（2e20 級）は `+inf` 同士、有限の ch1（1e18 級）・ch2（通常値）は REQ-2 統一複合判定 | pass |

「affine なし」「affine あり」の 2 通りで確かめる。テストは pass のとき値を出力しないので、**誤差の値はログに無い**。

## 成果物の確認

実機ロックの取得は 03:32:07Z（ノードの owner ファイルの mtime。`build_artifacts.txt`）。

- **削除（`clean.log`）**: `cargo metadata --no-deps` の 13 メンバーすべてに `cargo clean --release -p <name>` を実行した。ファイルを消したのは 8 メンバーで、`fandhe-ai-tensor-core` は 35 ファイル、`fandhe-ai-backend-metal` は 615 ファイル、`fandhe-ai-backend-cpu` は 134 ファイル、`bench-harness` は 35 ファイル。残る 5 メンバーは release の成果物が無く 0 ファイル。crates.io の依存は消していない。
- **ビルド（`build.log`）**: `fandhe-ai-tensor-core`・`fandhe-ai-backend-cpu`・`fandhe-ai-backend-cuda`・`fandhe-ai-backend-metal`・`fandhe-ai-autodiff`・`bench-harness` の 6 メンバーをこの回のツリーからコンパイルした。
- **更新時刻（`build_artifacts.txt`）**: 一覧の 13 ファイルすべてが `after_lock` で、`MISSING` は 0 行。主なものは次のとおり。

| 成果物 | 書かれた時刻（UTC） |
|---|---|
| テストバイナリ `batch_norm_parity-dc6fa32b852194cd` | 03:32:29 |
| `libfandhe_ai_backend_metal-1cf957da39e15641.rlib`（検証対象の Metal カーネルを含む） | 03:32:20 |
| `libfandhe_ai_tensor_core-086aa333dc653a6d.rlib` | 03:32:15 |
| `libfandhe_ai_backend_cpu-04ab852d0686ddb4.rlib` | 03:32:19 |
| `libbench_harness-51b54d4b07b63390.rlib` | 03:32:20 |

一覧で最も古いのは `libfandhe_ai_tensor_core` の rmeta の 03:32:13 で、ロックの取得より 6 秒後である。
rlib とテストバイナリのファイル名のハッシュは rerun の回と同じだが、上のとおり削除してからこの回で書き直したものである。
実行ログの `Running` 行のバイナリは、`build.log` の `Executable` 行および `list.log` の `Running` 行と同じ。`--list` とテストの実行で再コンパイルは起きていない。

## 実行の対応

- 実行コミット（`rev-stamp.txt`）: `82326b2537f89b4e5c5bf1129b69c0c3044442f0`・`dirty=0`。`crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は対象コミット `86797e1b7` と同一。
- ツリーの照合: `env_info.txt` の実行前・実行後とも、読み直した `rev_stamp_now`・`script_sha256_now`・`source_fingerprint_now` が開始時の値・期待値と一致した。回収後に送信側からノードのツリーを読み直しても同じで、
  ノードのファイル（`crates/` など・このディレクトリ・`.rev-stamp`）の ctime は転送の 03:32:07〜10 の範囲に収まる。上書きは起きていない。
- ソースの指紋 `faf953ec…c84c8` は、`RULE.txt` の手順で `git archive` から再現できる。
- スクリプト対応: `script_sha256`（`5b87675e…9529`）は、上記コミットの `run_metal_parity.sh` の SHA-256 と一致する。
- 実機ロック: 転送の前に送信側で取得した。`owner=#4031 rerun2 2026-11-05T03:32:06Z`。取得の直前に、ロック・展開先・出力先が無く、他の cargo／rustc も無いことを確かめた。`env_info.txt` の実行前・実行後とも保持しており、終了後に外れている。
- 選択の確認: `--release --ignored --exact --list` で対象がちょうど 1 件（`list.log`）。
- 回収物は手で編集していない。`shasum -a 256 -c SHA256SUMS` が通る（9 ファイルすべて OK）。ホスト名・`/Users/`・ユーザー名の残存は 0 件。メモリの空き率は実行前 71%・実行後 72%。

## 範囲外

§9.8 のとおり、infer 経路・facade／`BackendOps` 経路・CUDA は測っていない。

## 回収物

| ファイル | 内容 |
|---|---|
| `rev-stamp.txt` | 展開ツリーの `.rev-stamp`（2 行形式） |
| `env_info.txt` | 実行前後の環境（macOS・CPU・メモリ・rustc・負荷・ロック保持・ツリーの照合値） |
| `clean.log` | workspace メンバーの一覧と `cargo clean --release -p` の出力 |
| `build.log` | 事前ビルド（`--release --no-run`）のログ |
| `list.log` | `--release --ignored --exact --list` の出力 |
| `build_artifacts.txt` | ロック取得の時刻と、テストバイナリ・rlib・rmeta の更新時刻の一覧 |
| `batch_norm_parity_2e20_4031.log` | テストの全出力（`--nocapture`。末尾に `rc=`） |
| `exit_code` | テストの終了コード |
| `summary.tsv` | 判定（1 行） |
| `SHA256SUMS` | 上記の SHA-256 |
