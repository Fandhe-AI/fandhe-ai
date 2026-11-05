# Metal の BatchNorm 2e20 級入力の実機 parity の測り直し（#4031）

前回 [`../metal-batch-norm-2e20-4031/`](../metal-batch-norm-2e20-4031/) は実行中に展開ツリーが別セッションの rsync で上書きされ、所有者の判断で無効になった。
その測り直しである。対象は同じ `crates/backend-metal/tests/batch_norm_parity.rs` の
`batch_norm_train_2e20_scale_has_no_intermediate_overflow_and_matches_cpu_reference` の 1 本で、機体も同じ `<metal-node>`（Mac mini・Apple M4・メモリ 16GB）。
期待値の決め方は `docs/batch-norm-ops-design.md` §9.8。

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-05T03:06:43Z。実機実行より前に、この 2 ファイルだけのコミット `3493be202` で push した。実行後に変えていない）
- 実行スクリプト: [`run_metal_parity.sh`](./run_metal_parity.sh)（前回の複製。判定は同じで、ロックの引き継ぎとツリーの照合を足した。差分はヘッダの R1〜R3）
- tolerance 定数・parity baseline・閾値・判定の helper・テストコードは不変

## 状態

**要確認（所有者の判断待ち）: テストは pass で、RULE.txt の条件もすべて満たしたが、テストバイナリの一部が無効になった回のビルド成果物だった**（下の「リンクされた成果物」）。
そのため PR は `Refs #4031` のままにした。

1 本実行、exit 0、`running 1 test`、`test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 9 filtered out; finished in 0.02s`
（`batch_norm_parity_2e20_4031.log`・`summary.tsv`）。判定は pass。

| 量 | 判定（RULE.txt の judgment。テストコードが正） | 結果 |
|---|---|---|
| `out` | 全要素が有限、かつ REQ-2 統一複合判定（`assert_parity`）。§9.8 の中間 overflow の主証拠 | pass |
| `mean` | REQ-2 統一複合判定（`assert_parity`） | pass |
| `var` | §9.8 のクラス一致。ch0（2e20 級）は `+inf` 同士、有限の ch1（1e18 級）・ch2（通常値）は REQ-2 統一複合判定 | pass |

「affine なし」「affine あり」の 2 通りで確かめる。テストは pass のとき値を出力しないので、**誤差の値はログに無い**。

## リンクされた成果物（要確認）

`RULE.txt` のとおり `CARGO_TARGET_DIR` は共有の `<home>/work/target-fandhe-ai` を使った。実行後に deps の時刻を読み取りで確かめた結果（UTC）:

| 成果物 | 書かれた時刻 | どの回か |
|---|---|---|
| テストバイナリ `batch_norm_parity-dc6fa32b852194cd` | 03:07:00 | この回 |
| `libfandhe_ai_backend_cpu`・`libbench_harness` | 03:06:58〜59 | この回 |
| `libfandhe_ai_backend_metal-1cf957da39e15641`（検証対象の Metal カーネルを含む） | 02:54:28 | **無効になった回**（fingerprint の `invoked.timestamp` は 02:54:19） |
| `libfandhe_ai_tensor_core-086aa333dc653a6d` | 02:54:21 | **無効になった回** |

`build.log` の `Compiling fandhe-ai-backend-metal` は統合テストのターゲットの分で、lib は再ビルドされていない（上の時刻）。
無効になった回が上書きされた時刻は 02:54:17〜19 で、この 2 つはその直後にビルドされた。
つまり、このテストバイナリは事前登録コミット `3493be202` のツリーだけからビルドされたものではない。

- 確かめられること: `crates/` などの内容は `86797e1b7`・`a0ae7bdbc`・`cc3917d93`・`3493be202` で同一（`git diff` で再現できる）。
- 確かめられないこと: 再利用した rlib が、このソースからビルドされたものかどうかは、その回のツリーの照合が壊れているので記録からは示せない。
- 再利用された理由は、ソースのファイルの mtime（worktree を作った時刻）が成果物より古く、cargo が fresh と判断したためと考えられる。確かめてはいない。

## 実行の対応

- 実行コミット（`rev-stamp.txt`）: `3493be20254b2dabe2b44533b0aa323e7ae9dd97`・`dirty=0`。`crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は対象コミット `86797e1b7` と同一。
- ツリーの照合: `env_info.txt` の実行前・実行後とも、読み直した `rev_stamp_now`・`script_sha256_now`・`source_fingerprint_now` が開始時の値・期待値と一致した。回収後に送信側からノードのツリーを読み直しても同じで、
  ノードのファイル（`crates/` など・このディレクトリ・`.rev-stamp`）の ctime は転送の 03:06:50〜52 の範囲に収まる。上書きは起きていない。
- ソースの指紋 `faf953ec…c84c8` は、`RULE.txt` の手順で `git archive` から再現できる。
- スクリプト対応: `script_sha256`（`a454d1a9…2f54`）は、上記コミットの `run_metal_parity.sh` の SHA-256 と一致する。
- 実機ロック: 転送の前に送信側で取得した。`owner=#4031 rerun 2026-11-05T03:06:49Z`。取得の直前に、ロック・展開先・出力先が無く、他の cargo／rustc も無いことを確かめた。`env_info.txt` の実行前・実行後とも保持しており、終了後に外れている。
- 選択の確認: `--release --ignored --exact --list` で対象がちょうど 1 件（`list.log`。この回は `--list` で再コンパイルは起きていない）。
- 回収物は手で編集していない。`shasum -a 256 -c SHA256SUMS` が通る（7 ファイルすべて OK）。ホスト名・`/Users/`・ユーザー名の残存は 0 件。メモリの空き率は実行前後とも 71%。

## 範囲外

§9.8 のとおり、infer 経路・facade／`BackendOps` 経路・CUDA は測っていない。

## 回収物

| ファイル | 内容 |
|---|---|
| `rev-stamp.txt` | 展開ツリーの `.rev-stamp`（2 行形式） |
| `env_info.txt` | 実行前後の環境（macOS・CPU・メモリ・rustc・負荷・ロック保持・ツリーの照合値） |
| `build.log` | 事前ビルド（`--release --no-run`）のログ |
| `list.log` | `--release --ignored --exact --list` の出力 |
| `batch_norm_parity_2e20_4031.log` | テストの全出力（`--nocapture`。末尾に `rc=`） |
| `exit_code` | テストの終了コード |
| `summary.tsv` | 判定（1 行） |
| `SHA256SUMS` | 上記の SHA-256 |
