# f64 の create_graph と jvp 系の CUDA 実機 parity（#4053）

対象は次の 2 本（`--release` なし、dev プロファイル）。機体は `<cuda-node>`（DGX Spark GB10）。Mac mini 側は別の回が扱う。

- `crates/facade/tests/f64_create_graph_facade.rs` の `f64_create_graph_cuda_matches_cpu`（#3786）
- `crates/facade/tests/f64_functional_transforms_facade.rs` の `f64_jvp_family_cuda_matches_cpu`（#3788）

記録の根拠:

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-05T11:34:33Z。実機実行より前にコミット `c9b5e1635` で push した。実行後に変えていない）
- 実行スクリプト: [`run_cuda_parity_create_graph.sh`](./run_cuda_parity_create_graph.sh)・[`run_cuda_parity_jvp.sh`](./run_cuda_parity_jvp.sh)（雛形 `../hw-parity-run-template/run_parity_template.sh` の複製。差分は固定値の 9 行だけ。手動実行。CI からは呼ばない）
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline・判定規則・判定の helper・テストコードは変えていない

## 状態

**実測済み: 2 本とも pass**（2026-11-05・`<cuda-node>`）。fail 0、undetermined 0。

## 結果表

| テスト | 比較した量 | exit | `running` | `test result` | 判定 |
|---|---|---|---|---|---|
| `f64_create_graph_cuda_matches_cpu` | 二階 a・二階 b（2 つ） | 0 | 1 test | `ok. 1 passed; 0 failed; 0 ignored; 0 measured; 7 filtered out; finished in 0.42s` | pass |
| `f64_jvp_family_cuda_matches_cpu` | `jvp`・`jacfwd`・`jvp_multi`・`jvp_create_graph.output`・`jvp_create_graph.d_output_d_a`（5 つ） | 0 | 1 test | `ok. 1 passed; 0 failed; 0 ignored; 0 measured; 6 filtered out; finished in 0.31s` | pass |

判定は既存テストの REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。`assert_parity_f64`）で、CPU の結果と比べる。

**結果はテスト単位の pass である。** `assert_parity_f64` は pass のとき何も出力しないので、量ごとの `max_abs_diff` などの数値はログに無い。`fail_count` が 0 でなければ panic する（`crates/backend-cpu/src/parity.rs:290`）ので、テストの pass は上の量の `fail_count` が 0 であることを意味する。

## 実行の確認（各本の `env_info.txt`・`rev-stamp.txt`・`build.log`・`list_1.log`）

| 確認項目 | create_graph | jvp |
|---|---|---|
| 実行コミット | `c9b5e16356a0ab174dacdc1b717f5e6a021cc30d`・`dirty=0` | 同じ |
| `script_sha256` | `4b1f0145…9f73`（コミットのファイルと一致） | `995b2e60…26ca`（コミットのファイルと一致） |
| ソースの指紋（テスト前後） | `32191e98…9792` で一致 | 同じ |
| 対象ソース | `crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は `target_commit` `67a02e60c` と同一（送信前に確認） | 同じ |
| 実機ロック | owner `#4053 run 2026-11-05T11:35:19Z` を前後とも保持 | owner `#4053 run 2026-11-05T11:35:56Z` を前後とも保持 |
| 並走 | 他の cargo／rustc なし | 同じ |
| `--list` | 1 件 | 1 件 |
| 実行したバイナリ | `f64_create_graph_facade-86d513369da7eea7`。`Running` 行は `Executable` 行と同じ | `f64_functional_transforms_facade-b3867639f64e1d74`。同じ |
| `SHA256SUMS` | 9 ファイルとも通る | 9 ファイルとも通る |

雛形は終了時にロックを外すので、RULE の `hw_lock:` に事前に書いたとおり、2 本目の前にロックを取り直した。1 本目の終了を確かめた 11:35:42Z から取り直しの 11:35:57Z まではロックが無い。取り直しの時点で他の cargo／rustc は無く、転送は 1 回だけで、2 本目のソースの指紋も一致している。終了後にロックの解放を確かめた。

## 成果物の確認（各本の `build_artifacts.txt`・`clean.log`・`build.log`）

| 項目 | create_graph | jvp |
|---|---|---|
| clean | 13 メンバーすべてに `cargo clean -p`（8 メンバーで削除、このビルドの対象にならない 5 メンバーは 0 ファイル） | 同じ |
| ビルド | 8 メンバーすべてが `Compiling`（`Finished … in 11.22s`） | 8 メンバーすべてが `Compiling`（`Finished … in 11.36s`） |
| ロック取得（ノード上の owner ファイルの mtime） | 11:35:20Z | 11:35:57Z |
| 最も早い成果物 | tensor-core 11:35:28Z | tensor-core 11:36:00Z |
| テストバイナリ | 11:35:38Z | 11:36:10Z |
| 17 ファイル（テストバイナリ、8 メンバーの `.rlib`・`.rmeta`。facade 自身の `libfandhe_ai-a2d6c9c718a5d0ea` を含む） | すべて `after_lock` | すべて `after_lock` |
| `NOT_AFTER_LOCK`・`MISSING` | 0 件 | 0 件 |

## 占有状況（送信側で §6.1 の形で確認）

| 時点（UTC） | 常駐 compute プロセス | GPU 使用率 |
|---|---|---|
| 1 本目の実行前 11:35:25Z | 3 件 | 0 % |
| 1 本目の実行後 11:35:42Z | 3 件 | 0 % |
| 2 本目の実行前 11:35:57Z | 3 件 | 0 % |
| 2 本目の実行後 11:36:13Z | 3 件 | 0 % |

ほかのサービスのプロセスは止めていない。parity 測定で時間計測ではないため専有条件は課していない。

## 限界

- 量ごとの数値は出ない（上記）
- 実行時に GPU のカーネルを通ったことは、ログから直接には確かめていない。経路は設計上のもので、`add`・`mul`・`matmul`・`sum` は CUDA の `TypedOps<f64>`、`mean` はその `sum` とホストでの 1 回の除算である（`crates/autodiff/src/f64_autograd.rs:53`・`:55`）。`tape_for(Device::Cuda(0))` が失敗すればテストは panic するので、pass は CUDA の tape を作れたことまでを示す
- 時刻検査は「成果物がロック取得より後に書かれた」ことを確かめる。内容が現在のソースと対応することは、ソースの指紋と clean の組で担保する
- crates.io の依存の成果物は clean していない（`Cargo.lock` で固定された版の再利用で、意図どおり）

## 回収物

`create_graph/`・`jvp/` の各ディレクトリに次を置いた。手で編集していない（`shasum -a 256 -c SHA256SUMS` で確かめられる）。

| ファイル | 内容 |
|---|---|
| `rev-stamp.txt` | 展開ツリーの `.rev-stamp`（2 行形式） |
| `env_info.txt` | 実行前後の環境・スクリプトの SHA-256・指紋・ロック保持・`build_artifacts_ok` |
| `clean.log` | clean したメンバーの一覧と削除件数 |
| `build.log` | 事前ビルド（`--no-run`）のログ |
| `list_1.log` | `--ignored --exact --list` の出力（1 件） |
| `build_artifacts.txt` | 成果物の更新時刻とロック取得時刻の照合 |
| `f64_create_graph_cuda_1.log`／`f64_jvp_family_cuda_1.log`・`exit_code_1` | テストの全出力（`--nocapture`）と終了コード |
| `summary.tsv` | 判定（1 行） |
| `SHA256SUMS` | 上記の SHA-256 |
