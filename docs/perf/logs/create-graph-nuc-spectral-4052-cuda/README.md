# create_graph の MatrixNorm Nuc・Spectral の値のケースの CUDA 実機 parity（#4052）

対象は `crates/facade/tests/create_graph_gpu_parity.rs` の `create_graph_linalg_norm_rank_cuda_matches_cpu` の 1 本（`--release` なし、dev プロファイル）。機体は `<cuda-node>`（DGX Spark GB10）。
PR #4054 で `linalg_norm_rank` 族に足した `linalg_matrix_norm_nuc`・`linalg_matrix_norm_spectral` を、既存ケースと同じテストで測った。Mac mini 側は別の回が扱う。

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-05T10:27:55Z。実機実行より前にコミット `8966befde` で push した。実行後に変えていない）
- 実行スクリプト: [`run_cuda_parity.sh`](./run_cuda_parity.sh)（雛形 `../hw-parity-run-template/run_parity_template.sh` の複製。差分は固定値の 9 行だけ。手動実行。CI からは呼ばない）
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline・判定規則・判定の helper・テストコードは変えていない

## 状態

**実測済み: 族単位で pass**（2026-11-05・`<cuda-node>`）。exit 0、`running 1 test`、`test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 51 filtered out; finished in 1.25s`。fail 0、undetermined 0。成果物の確認と代替証拠（下記）も成立した。

**ログにケース名は出ていない。** このテストは pass 時に何も出力しないので、`linalg_matrix_norm_nuc`・`linalg_matrix_norm_spectral` が実行された直接の跡はログに無い。ケースの実行の根拠は、計測前に所有者の判断で RULE の `case_evidence:` に定めた代替証拠である。ケース別の pass と数値は主張しない。

route は全ケース H で、pass は経路の parity であり、カーネル同士の parity ではない（TF32 opt-in・GPU linalg ゲートは既定 OFF のまま）。

## 結果表

判定契約は全ケース `Judge::Strict`（Q1 同デバイス bit 一致＋GPU・CPU の REQ-2 複合判定／Q2 cg.grad・Q3 H v〈ゼロでない〉・Q4 孫テープ再生の REQ-2 統一複合判定／Q5 拒否は `Err(Backward)` かつ子テープが空）。
結果の欄はテスト 1 本の結果で、ケースごとに観測した値ではない。失敗があれば panic にケース名と Q 番号が出るが、今回は出ていない。

| ケース | route | Q5 | テストの結果 | ケース別の出力 |
|---|---|---|---|---|
| `linalg_matrix_norm_fro` | H | `reject_matrix_norm_nuc_degenerate` | pass（族単位） | なし |
| `linalg_matrix_norm_one` | H | なし | pass（族単位） | なし |
| `linalg_matrix_norm_inf` | H | なし | pass（族単位） | なし |
| `linalg_matrix_norm_nuc` | H | なし | pass（族単位） | なし |
| `linalg_matrix_norm_spectral` | H | なし | pass（族単位） | なし |
| `linalg_slogdet_sign` | H | なし | pass（族単位） | なし |
| `linalg_matrix_rank` | H | なし | pass（族単位） | なし |

`max_abs_diff` などの数値は、pass 時にテストが出力しないため記録に無い。

## ケースの実行の代替証拠（RULE `case_evidence:`）

| 項目 | 内容 | 結果 | 根拠 |
|---|---|---|---|
| e1 | 実行したテストバイナリに両ケース名が入っている（`grep -a -o`） | `linalg_matrix_norm_nuc` 1 件、`linalg_matrix_norm_spectral` 1 件、対照の `linalg_matrix_norm_absent_4052` 0 件。バイナリの更新時刻 1793874524 は `build_artifacts.txt` と同じ | `case_evidence.txt` |
| e2 | ソースの指紋が `target_commit` と一致する | テストの前後とも `32191e98…9792` | `env_info.txt` |
| e3 | `run_family` が族の全ケースを打ち切らずに回す | `filter_map` で全ケースの失敗を集めて最後に assert する | `crates/facade/tests/create_graph_gpu_parity.rs:2831`・`:2839`、ケースの登録は同 `:892`・`:903` |

e1〜e3 は実行の直接の観測ではない。e1 はケース名がバイナリにあることを、e3 はコードの読みとしてそれが実行されることを示す。

## 実行の確認

| 確認項目 | 結果 | 根拠 |
|---|---|---|
| 実行コミット | `8966befdee52b0d0b850ea02ee46e9229654644e`・`dirty=0`（RULE とスクリプトだけのコミット） | `rev-stamp.txt` |
| スクリプト | `script_sha256` `1ee65e01…b96f` が転送前のコミットのファイルと一致 | `env_info.txt` |
| 対象ソース | `crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は `target_commit` `0545513ec` と同一（送信前に確認） | RULE の条件 |
| 実機ロック | owner `#4052 run 2026-11-05T10:28:25Z` を前後とも保持。終了後に解放を確認 | `env_info.txt`・`case_evidence.txt` |
| 並走 | 他の cargo／rustc なし | `env_info.txt` |
| 実行したバイナリ | `Running` 行は `Executable` 行と同じ `create_graph_gpu_parity-7cd85656ab24ea57`。`--list` と実行で再コンパイルなし（`Finished … in 0.04s`） | `build.log`・`list_1.log`・`create_graph_linalg_norm_rank_cuda_1.log` |
| 回収物 | `SHA256SUMS` が 9 ファイルとも通る。手で編集していない | `SHA256SUMS` |

## 成果物の確認（`build_artifacts.txt`・`clean.log`・`build.log`）

- clean: 13 メンバーすべてに `cargo clean -p` を実行した（8 メンバーで削除、Linux でこのビルドの対象にならない 5 メンバーは 0 ファイル）
- ビルド: 8 メンバーすべてが `Compiling`（`Finished … in 11.94s`）
- ロック取得（ノード上の owner ファイルの mtime）: 2026-11-05T10:28:26Z
- 17 ファイル（テストバイナリ、8 メンバーの `.rlib`・`.rmeta`。facade 自身の `libfandhe_ai-a2d6c9c718a5d0ea` を含む）がすべて `after_lock`。`NOT_AFTER_LOCK`・`MISSING` は 0 件
- 最も早いものは tensor-core の 10:28:33Z で、ロック取得の 7 秒後。テストバイナリは 10:28:44Z

## 占有状況（送信側で §6.1 の形で確認）

| 時点（UTC） | 常駐 compute プロセス | GPU 使用率 |
|---|---|---|
| 実行前 10:28:26Z | 3 件 | 0 % |
| 実行後 10:28:47Z | 3 件 | 0 % |

ほかのサービスのプロセスは止めていない。parity 測定で時間計測ではないため専有条件は課していない。

## 限界

- ケース別の実行は直接には観測していない（上記）。直接の跡が要るなら、テストにケース別の出力を足す別の変更と再登録が要る
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
| `create_graph_linalg_norm_rank_cuda_1.log`・`exit_code_1` | テストの全出力（`--nocapture`）と終了コード |
| `summary.tsv` | 判定（1 行） |
| `SHA256SUMS` | 上記の SHA-256（このディレクトリで `shasum -a 256 -c SHA256SUMS`） |
| `case_evidence.txt` | 代替証拠 e1。実行後に送信側がノード上で取ったもので、`SHA256SUMS` の対象外 |
