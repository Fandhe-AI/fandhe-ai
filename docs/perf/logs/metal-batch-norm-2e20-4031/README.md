# Metal の BatchNorm 2e20 級入力の実機 parity（#4031）

対象は `crates/backend-metal/tests/batch_norm_parity.rs` の
`batch_norm_train_2e20_scale_has_no_intermediate_overflow_and_matches_cpu_reference`（PR #4039 で追加した `#[ignore]` テスト）の 1 本。
Metal の BatchNorm train カーネル（soft-f64）に 2e20 級の有限入力を与え、CPU 参照 `run_batch_norm_train_f32` と突き合わせた。
機体は `<metal-node>`（Mac mini・Apple M4・メモリ 16GB。`docs/real-hardware-verification-env.md` §7）。
期待値の決め方は `docs/batch-norm-ops-design.md` §9.8、手順は `../metal-batch-norm-1736/README.md` の「#4031 追記」節。

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-05T02:53:49Z。実機実行より前にコミット `a0ae7bdbc` で push した。実行後に変えていない）
- 実行スクリプト: [`run_metal_parity.sh`](./run_metal_parity.sh)（Mac mini 上で手動実行。CI からは呼ばない。`--self-test` 付き）。
  `../optim-train-template-metal-parity-3863/run_metal_parity.sh` の複製で、対象をこの 1 本に絞り `--release` を付けた（差分はヘッダに記載）
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline・閾値・判定の helper・テストコードは不変

## 状態

**実測済み: pass**（2026-11-05・`<metal-node>`）。1 本実行、exit 0、`running 1 test`、
`test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 9 filtered out; finished in 0.08s`（`batch_norm_parity_2e20_4031.log`・`summary.tsv`）。

| 量 | 判定（RULE.txt の judgment。テストコードが正） | 結果 |
|---|---|---|
| `out` | 全要素が有限、かつ REQ-2 統一複合判定（`assert_parity`）。§9.8 の中間 overflow の主証拠 | pass |
| `mean` | REQ-2 統一複合判定（`assert_parity`） | pass |
| `var` | §9.8 のクラス一致。ch0（2e20 級）は `+inf` 同士、有限の ch1（1e18 級）・ch2（通常値）は REQ-2 統一複合判定 | pass |

上の 3 量を「affine なし」「affine あり」の 2 通りで確かめる。テスト内の前提検査（ch0 の f64 分散が有限かつ `f32::MAX` 超、CPU の `var[0]` が `+inf`、
`var[1]`・`var[2]` が有限）も通った。テストは assert で判定し、pass のときは値を出力しないので、**誤差の値はログに無い**。
量ごとの pass は、テスト全体が exit 0 で終わったことから分かる（量ごとの記録行は無い）。

- 実行コミット（`rev-stamp.txt`）: `a0ae7bdbc48ebf2a1be86dbab2afe69a10c23e80`・`dirty=0`。事前登録と実行スクリプトを入れた最初のコミットそのもので、
  `crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は対象コミット `86797e1b7`（origin/main）と同一（送信前に `git diff --quiet` で確認）。
  テストを追加した `0c79dfc2f` とも同一。転送は rsync だけで、ノード側で `.rev-stamp` を読んで送信側の HEAD と照合した。
- スクリプト対応: `env_info.txt` の `script_sha256`（`a1becc34…bdaa`）は、上記コミットの `run_metal_parity.sh` の SHA-256 と一致する（送信側・ノード側の両方で確認）。
- 実機ロック: `env_info.txt` の実行前・実行後とも `hw_lock: held owner=#4031 2026-11-05T02:54:17Z`。終了後にロックが外れていることを確かめた。
  ほかの利用者のプロセスは止めていない。並走する cargo／rustc は事前ビルド前・テスト前とも無し。メモリの空き率は実行前後とも 72%。
- 選択の確認: `--release --ignored --exact --list` で対象がちょうど 1 件（`list.log`）。
- 回収物は手で編集していない。`SHA256SUMS` はこのディレクトリで `shasum -a 256 -c SHA256SUMS` が通る（7 ファイルすべて OK）。
  ホスト名・`/Users/`・ユーザー名の残存は 0 件（`<metal-node>`／`<user>`／`<home>` へマスク済み。送信側でも grep で確認）。

## 記録しておく事実（ビルドの順序）

RULE.txt は `--no-run` でビルドを終える手順だが、`--no-run`（19.44 秒。`build.log`）の後の `--list` の時点で、3 クレート
（`fandhe-ai-backend-cpu`・`bench-harness`・`fandhe-ai-backend-metal`）が再度コンパイルされた（4.24 秒。`list.log`）。原因は調べていない。
再コンパイルはテストの実行より前に終わっており、テストの実行では `Finished ... in 0.03s` でコンパイルは走っていない
（`batch_norm_parity_2e20_4031.log`）。テストの実行前の検査で、並走する cargo／rustc は無かった。テストバイナリ名は `--no-run` と `--list`・実行で同じ
（`batch_norm_parity-dc6fa32b852194cd`）。RULE.txt の判定（pass・fail・undetermined）には関わらない。

## 範囲外

§9.8 のとおり、infer 経路（固定の f32 統計では 2e20 級の分散を持てない）・facade／`BackendOps` 経路・CUDA は測っていない。

## 回収物

| ファイル | 内容 |
|---|---|
| `rev-stamp.txt` | 展開ツリーの `.rev-stamp`（2 行形式） |
| `env_info.txt` | 実行前後の環境（macOS・CPU・メモリ量と空き率・rustc・負荷・ロック保持） |
| `build.log` | 事前ビルド（`--release --no-run`）のログ |
| `list.log` | `--release --ignored --exact --list` の出力 |
| `batch_norm_parity_2e20_4031.log` | テストの全出力（`--nocapture`。末尾に `rc=`） |
| `exit_code` | テストの終了コード |
| `summary.tsv` | 判定（1 行） |
| `SHA256SUMS` | 上記の SHA-256 |
