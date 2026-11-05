# f64 の create_graph・jvp 系の Metal 実機 parity（#4053）

対象は次の 2 本（dev プロファイル。`--release` なし）。機体は `<metal-node>`（Mac mini・Apple M4・メモリ 16GB。`docs/real-hardware-verification-env.md` §7）。CUDA 側は別の計測で、本 README の範囲外。

- `crates/facade/tests/f64_create_graph_facade.rs` の `f64_create_graph_metal_matches_cpu`（#3786）
- `crates/facade/tests/f64_functional_transforms_facade.rs` の `f64_jvp_family_metal_matches_cpu`（#3788）

記録の位置づけ:

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-05T11:36:15Z。実機実行より前にコミット `5c3e1472e` で push した。実行後に変えていない）
- 実行スクリプト: [`run_metal_parity_create_graph.sh`](./run_metal_parity_create_graph.sh)・[`run_metal_parity_jvp.sh`](./run_metal_parity_jvp.sh)（どちらも `../hw-parity-run-template/run_parity_template.sh` の複製で、差分は固定値の 8 行だけ）
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline・判定規則・判定の helper（`assert_parity_f64`）・テストコードは不変

## 状態

**2 本とも pass。fail 0、undetermined 0（2026-11-05・`<metal-node>`）。各本とも 1 回の実行で、再実行していない。**

| テスト | exit code | running | test result | 判定 |
|---|---|---|---|---|
| `f64_create_graph_metal_matches_cpu` | 0 | 1 | `ok. 1 passed; 0 failed; 0 ignored; 0 measured; 8 filtered out; finished in 0.02s` | pass |
| `f64_jvp_family_metal_matches_cpu` | 0 | 1 | `ok. 1 passed; 0 failed; 0 ignored; 0 measured; 7 filtered out; finished in 0.02s` | pass |

根拠は各ディレクトリの `summary.tsv`・`exit_code_1`・実行ログ（末尾 `rc=0`）。

読み方の注意が 2 つある（どちらも `RULE.txt` に宣言済み）:

- **経路の parity であり、カーネル同士の parity ではない。** Metal の f64 は GPU カーネルを持たず、`typed_ops_f64()` が `None` でホスト参照実装を通る（`docs/autodiff-functional-transforms-design.md`）。この pass は Metal デバイスの tape 経由の経路と CPU の一致を示すもので、GPU カーネルの検証ではない。
- **結果はテスト単位の pass で、量ごとの pass と数値は主張しない。** `gpu_matches_cpu`・`assert_parity_f64` は pass のとき量の名前も数値も出力しないため、ログに量ごとの跡は無い。比べた量はコードから読んだもので、実行の観測ではない（create_graph は `二階 a`・`二階 b` の 2 量、jvp 系は `jvp`・`jacfwd`・`jvp_multi`・`jvp_create_graph.output`・`jvp_create_graph.d_output_d_a` の 5 量）。`fail 要素数` は、`assert_parity_f64` が `fail_count > 0` で panic するので pass なら 0 と読める、という意味で、ログに出た件数ではない。

実行時間は根拠にしない。

## 回収物の確認

| 項目 | `create-graph/` | `jvp/` |
|---|---|---|
| rev_stamp | `5c3e1472e22ead338f910fe3c89bf3ddde2db875`（事前登録コミット）・`dirty=0` | 同じ |
| script_sha256（実行前後で一致。起動前に送信側のコミットのファイルとも照合） | `b328f401…b620` | `3092a34b…676e` |
| ソースの指紋（期待値・実行前・実行後） | `32191e981ef0467ccf93c858b7501cd4433edc0d9c247a85035aa11667e09792` | 同じ |
| `build_artifacts.txt` | 19 行すべて `after_lock` | 19 行すべて `after_lock` |
| ロックの mtime／最も古い成果物 | 2026-11-05T11:36:55Z／11:37:02Z | 2026-11-05T11:37:13Z／11:37:15Z |
| `clean.log`・`build.log` | 13 メンバーを `cargo clean -p`。`--no-run` で 9 メンバーを `Compiling`（9.95s） | 同じ（9.41s） |
| `list_1.log` | `: test` の行がちょうど 1 件 | 同じ |
| `SHA256SUMS` | `shasum -a 256 -c` で 9 ファイルすべて OK | 同じ |
| スクリプトの `done:` 行 | `any_fail=0 any_undetermined=0 leak=0 lock_lost=0` | 同じ |
| 実機ロックの owner（実行前後とも held） | `#4053 run 2026-11-05T11:36:54Z` | `#4053 run 2026-11-05T11:37:13Z` |

`build_artifacts.txt` の 19 行は、テストバイナリ 1 と 9 メンバー（`fandhe_ai` を含む）の `.rlib`・`.rmeta` である。`MISSING`・`NOT_AFTER_LOCK` は無い。

共通の確認:

- `crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は `target_commit`（`67a02e60c`）と同一（送信前に `git diff --quiet` で確認）。指紋はそのコミットの `git archive` から計算した
- 環境（`env_info.txt`）: macOS 27.0.1・Apple M4・rustc 1.99.0。並走する cargo／rustc なし
- 環境（送信側で読んだ値）: メモリの空き率は実行前後とも 72%。負荷（1 分平均）は実行前 2.09、実行直後 10.89（直前のビルドを含む）。ほかのサービスのプロセスは止めていない
- マスク: スクリプトは `leak=0`。回収後に送信側でユーザー名・`/Users/`・SSH 別名を grep し、`RULE.txt`・スクリプトの規則の文（`/Users/`）のほかに該当なし
- 雛形の検査（コミット前）: 2 本とも `bash -n`。`--self-test` は開発機の `/bin/bash` 3.2、ノードの `/bin/bash` 3.2・`zsh -lc` の `bash` 5.3 で pass。実行はノードの `bash` 5.3

## ロックと起動の順序

雛形は終了時の `trap` でロックを外し、出力先が既にあると止まる。固定値以外は変えないので、`RULE.txt` に次を宣言して測った:

- ロックは 1 本ごとに取った。1 本目は送信側が転送の前に取り、2 本目は 1 本目が exit 0 で終わった後に起動役が取り直した（owner が 2 つあるのはこのため）。**取り直しの間はロックが外れている。** `mkdir` は成功した
- 出力先は `out-4053-metal/` の下の `create-graph/`・`jvp/` に分けた。本ディレクトリの同名のサブディレクトリが回収物である
- 起動役は `RULE.txt` の `drive-begin`〜`drive-end` から取り出した一時ファイル（SHA-256 `32800d871cd3031eec63b7927ad59ead84ee907c78d8dbffaddd9746c04233f3`）。ノード側のものと起動前に照合した。`SHA256SUMS` の外にある

起動役のログの判定行:

```
driver: create-graph rc=0 owner=#4053 run 2026-11-05T11:36:54Z
driver: jvp rc=0 owner=#4053 run 2026-11-05T11:37:13Z
driver: finished
```

## 回収物

`create-graph/`・`jvp/` のそれぞれに次がある。

| ファイル | 内容 |
|---|---|
| `f64_create_graph_metal_1.log`（`jvp/` は `f64_jvp_family_metal_1.log`）・`exit_code_1` | 実行ログ全文と終了コード |
| `summary.tsv` | 判定表 |
| `rev-stamp.txt`・`env_info.txt` | 展開ツリーの `.rev-stamp`、実行前後の環境と照合値 |
| `clean.log`・`build.log`・`build_artifacts.txt`・`list_1.log` | clean・事前ビルド・成果物の時刻検査・選択の検証 |
| `SHA256SUMS` | 上記の SHA-256 |
