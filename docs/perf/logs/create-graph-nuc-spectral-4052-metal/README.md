# 二階微分（create_graph）の linalg_norm_rank 族の Metal 実機 parity（#4052）

対象は `crates/facade/tests/create_graph_gpu_parity.rs` の `create_graph_linalg_norm_rank_metal_matches_cpu` の 1 本（dev プロファイル）。PR #4054 で、この族に `matrix_norm` の `Nuc`・`Spectral` の値のケースが加わった。機体は `<metal-node>`（Mac mini・Apple M4・メモリ 16GB。`docs/real-hardware-verification-env.md` §7）。測り方は `docs/autodiff-higher-order-grad-decision.md` §48。GB10 側は別の計測で、本 README の範囲外。

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-05T10:26:48Z。実機実行より前にコミット `df9e060a0` で push した。実行後に変えていない）
- 実行スクリプト: [`run_metal_parity.sh`](./run_metal_parity.sh)（`../hw-parity-run-template/run_parity_template.sh` の複製で、差分は固定値の 8 行だけ）
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline・判定規則・テストコードは不変

## 状態

**テスト 1 本 pass。fail 0、undetermined 0（2026-11-05・`<metal-node>`）。ただし `linalg_matrix_norm_nuc`・`linalg_matrix_norm_spectral` がログ上で実行された直接の跡は無い。**

このテストは pass のときケース名を出力しない（`run_family`・`check_case`）。実行ログに `linalg_matrix_norm` は 0 件である。`RULE.txt` の `record:` が宣言時にこれを記録しており、宣言どおり、この pass を Nuc・Spectral の値の経路の pass の根拠にはしない。**Nuc・Spectral の pass と扱うかどうかは所有者の判断待ち**で、#4052 は Refs に留める。

## 結果表

判定はテストが行う（全ケース `Judge::Strict`。Q1〜Q4、`reject` があるケースは Q5）。下の「結果」はテスト全体の判定で、ケース別の値はログに無い（fail のときだけ panic メッセージにケース名・Q 番号・量が出る）。

| 族のケース（`CASES`） | route | S3 | Q5 | ログの直接の跡 | テスト全体の結果 |
|---|---|---|---|---|---|
| `linalg_matrix_norm_fro` | H | NoMatMul | `reject_matrix_norm_nuc_degenerate` | なし | pass |
| `linalg_matrix_norm_one` | H | NoMatMul | - | なし | pass |
| `linalg_matrix_norm_inf` | H | NoMatMul | - | なし | pass |
| `linalg_matrix_norm_nuc` | H | Rule | - | なし | pass（ケース別の根拠にしない） |
| `linalg_matrix_norm_spectral` | H | Rule | - | なし | pass（ケース別の根拠にしない） |
| `linalg_slogdet_sign` | H | NoMatMul | - | なし | pass |
| `linalg_matrix_rank` | H | NoMatMul | - | なし | pass |

7 ケースとも直接の跡は無い。Nuc・Spectral にだけ注記があるのは、既存の 5 ケースが #3859 までの族単位の pass の扱いのままで、新しい 2 ケースだけ今回の計測指示がログの直接の跡を求めたためである。

根拠は `summary.tsv`（exit code 0・`running` 1・`test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 76 filtered out; finished in 0.19s`）と `create_graph_linalg_norm_rank_metal_1.log`（末尾 `rc=0`）。route H の pass は経路の parity で、カーネル同士の parity ではない（linalg は GPU linalg ゲート既定 OFF のまま）。

- **S3**: `Nuc`・`Spectral` の `Rule` は実行時に機械検査されない（`check_split_k_not_fired` が `should_split_k` で調べるのは `Listed` だけ）。S3 違反の出力は無いが、それは非発火の実測ではない。規則の文はテストの宣言のままで、本計測で確かめ直していない。

## 間接の根拠（直接の跡の代わりにはしない）

1. `env_info.txt` の `source_fingerprint_now` は実行前後とも `32191e98…7792` で、`target_commit`（`0545513ec`）の `git archive` から計算した値と一致する。そのコミットの `CASES` には `family: "linalg_norm_rank"` の 2 ケースが各 1 件あり、`run_family` は族の全ケースを filter して `collect` まで判定する（途中で打ち切らない）。テストが pass したのは `failures` が空だったから、というのはコードを読んで分かることで、実行の観測ではない。
2. `case_symbols.txt`（回収の後に送信側から 1 回読んだ `nm -a` の該当行。判定外）に `run_linalg_nr_nuc`・`run_linalg_nr_spectral` がある。テストバイナリに関数が含まれることを示すだけで、実行されたことは示さない。script の `SHA256SUMS` の外のファイルである。

実行時間は根拠にしない。`finished in 0.19s` は、5 ケースだった #3859 の同じテストの 0.25s より短い（`../create-graph-b-metal-parity-3859/logs/linalg-norm-rank-3806/`）。別の回の比較で、ケース数の増減を読み取れる値ではない。

## 回収物の確認

| 項目 | 結果 |
|---|---|
| rev_stamp | `df9e060a0ac0e0fda149bd44a21db7960798864e`（事前登録コミット）・`dirty=0`。`crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は `0545513ec` と同一（送信前に `git diff --quiet` で確認） |
| script_sha256 | `909b7e28…8b7c`。実行前後で一致し、コミットのファイルの SHA-256 とも一致（起動前に送信側で照合） |
| ソースの指紋 | 期待値・実行前・実行後とも `32191e981ef0467ccf93c858b7501cd4433edc0d9c247a85035aa11667e09792` |
| `build_artifacts.txt` | 19 行すべて `after_lock`（テストバイナリ 1 と、9 メンバーの `.rlib`・`.rmeta`。`fandhe_ai` を含む）。MISSING・NOT_AFTER_LOCK なし。ロックの mtime 2026-11-05T10:27:30Z、最も古い成果物 10:27:44Z |
| `clean.log`・`build.log` | 13 メンバーすべてを `cargo clean -p`。`--no-run` で 9 メンバーを `Compiling`（10.11s） |
| `--list`（`list_1.log`） | `: test` の行がちょうど 1 件 |
| `SHA256SUMS` | `shasum -a 256 -c SHA256SUMS` で 9 ファイルすべて OK |
| マスク | スクリプトは `leak=0`。回収後に送信側でユーザー名・`/Users/`・SSH 別名・実ホスト名を grep し、`RULE.txt`・スクリプトの規則の文（`/Users/`）のほかに該当なし |
| 実機ロック | owner は `#4052 run 2026-11-05T10:27:29Z`。実行前後とも held、終了後は解放を確認。`lock_lost=0`。並走する cargo／rustc なし |
| 環境（送信側で読んだ値） | メモリの空き率は実行前後とも 72%。負荷（1 分平均）は実行前 1.44、実行直後 7.97（直前のビルドを含む）。ほかのサービスのプロセスは止めていない |
| 雛形の検査 | `bash -n`。`--self-test` は開発機の `/bin/bash` 3.2、ノードの `/bin/bash` 3.2・`zsh -lc` の `bash` 5.3 で pass。実行はノードの `bash` 5.3 |

## 回収物

| ファイル | 内容 |
|---|---|
| `create_graph_linalg_norm_rank_metal_1.log`・`exit_code_1` | 実行ログ全文と終了コード |
| `summary.tsv` | 判定表 |
| `rev-stamp.txt`・`env_info.txt` | 展開ツリーの `.rev-stamp`、実行前後の環境と照合値 |
| `clean.log`・`build.log`・`build_artifacts.txt`・`list_1.log` | clean・事前ビルド・成果物の時刻検査・選択の検証 |
| `SHA256SUMS` | 上記の SHA-256 |
| `case_symbols.txt` | `nm -a` の該当 4 行（`SHA256SUMS` の外） |
