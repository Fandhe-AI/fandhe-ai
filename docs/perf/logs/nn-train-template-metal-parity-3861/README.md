# 学習ステップの parity 雛形 nn 族の Metal 実機 parity（#3861 の 2 回目）

対象は `crates/facade/tests/train_gpu_parity.rs` の nn 族（#4026・PR #4034 で登録した 18 ケース。Phase 6 の NN 層を骨格に入れた
1 学習ステップの forward・loss・勾配・1 step 後の loss）。`train_parity_nn_metal_matches_cpu` を 1 本だけ実行し、Metal の腕を
CPU の腕と突き合わせた。機体は `<metal-node>`（Mac mini・Apple M4・メモリ 16GB。`docs/real-hardware-verification-env.md` §7）。

1 回目（[`../nn-layers-metal-parity-3861/`](../nn-layers-metal-parity-3861/)。各実装 issue の `#[ignore]` Metal テスト 19 本）では、
雛形の nn 族が登録 0 件で undetermined だった。この記録は #4026 のマージ後に雛形テストを測り直したもので、1 回目のディレクトリは書き換えていない。
GB10 版は #3860。雛形は `../train-gpu-parity-template/`（#3778）。

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-04T19:05Z。実機実行より前にコミット `c71ef53a2` で push した。実行後に変えていない）
- 実行スクリプト: [`run_metal_parity.sh`](./run_metal_parity.sh)（Mac mini 上で手動実行。CI からは呼ばない。`--self-test` 付き）。
  1 回目のスクリプトの複製で、対象を雛形の 1 本に絞り、`TRAIN_PARITY` 記録の集計（`summary.tsv`）と記録行数の検査を足した（差分はヘッダに記載）
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline・判定規則・テストコードは不変

## 状態

**実測済み: `TRAIN_PARITY` 記録 120 行中 120 行 pass、fail 0、undetermined 0**（2026-11-04・`<metal-node>`）。
テストは exit 0・`running 1 test`・`test result: ok. 1 passed; 0 failed`（`test_result.tsv`）。記録行数は RULE.txt の
`expected_records: 120` と一致した。1 回目の 19 本 pass と合わせ、#3861 の完了条件を満たす。

- 実行コミット（`rev-stamp.txt`）: `c71ef53a2b3e3194df4a7b3de9461f821a662ae0`・`dirty=0`。事前登録と実行スクリプトを入れた最初のコミットそのもので、
  `crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は対象コミット `2f59b9638`（origin/main）と同一（送信前に `git diff --quiet` で確認）。
  転送は rsync だけで、ノード側で `.rev-stamp` を読んで送信側の HEAD と照合した。
- スクリプト対応: `env_info.txt` の `script_sha256`（`371600c2…660c`）は、上記コミットの `run_metal_parity.sh` の SHA-256 と一致する
  （送信側・ノード側の両方で確認）。
- 実機ロック: `env_info.txt` の実行前・実行後とも `hw_lock: held owner=#3861 2026-11-04T19:05:38Z`。終了後にロックが外れていることを確かめた。
  ほかの利用者のプロセスは止めていない。並走する cargo／rustc は事前ビルド前・テスト前とも無し。メモリの空き率は実行前 72%・実行後 73%。
- 回収物は手で編集していない。`SHA256SUMS` はこのディレクトリで `shasum -a 256 -c SHA256SUMS` が通る（7 ファイルすべて OK）。
  ホスト名・`/Users/`・ユーザー名の残存は 0 件（`<metal-node>`／`<user>`／`<home>` へマスク済み。送信側でも grep で確認）。
- 事前ビルド（`--no-run`）は 10.40 秒（`build.log`）。テストは 0.51 秒で終わった。
- デバイス経路: `harness::run_family(CASES, "nn", Device::Metal)` が各 run 関数へ `Device::Metal` を渡す。route H の pass は経路の parity であり、
  カーネル同士の parity ではない（RULE.txt の judgment）。

## 結果表（ケース別）

各行は `summary.tsv`（1 記録 1 行。列は case・family・quantity・arm・route・judge・total・fail_count・max_abs_diff・mean_abs_diff・max_rel_err・verdict・reason）
をケースごとにまとめたもの。判定は全行 Req2（REQ-2 統一複合判定。相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。

| ケース | route | judge | 記録 | pass | 要素数 | fail_count | 最大 max_abs_diff（量） | 最大 max_rel_err（量） |
|---|---|---|---|---|---|---|---|---|
| `nn_upsample_scale_factor_train_step_s1` | K | Req2 | 7 | 7 | 1040 | 0 | 8.940697e-08（`forward`） | 1.236759e-05（`grad_p2`） |
| `nn_upsample_scale_factor_train_step_s2` | K | Req2 | 7 | 7 | 953 | 0 | 4.470348e-08（`forward`） | 3.116376e-05（`grad_p2`） |
| `nn_shrink_train_step_s1` | H | Req2 | 7 | 7 | 392 | 0 | 0 | 0 |
| `nn_shrink_train_step_s2` | H | Req2 | 7 | 7 | 386 | 0 | 2.980232e-08（`loss`） | 8.733040e-08（`loss_after_step`） |
| `nn_batch_norm3d_train_step_s1` | K | Req2 | 9 | 9 | 758 | 0 | 1.192093e-07（`grad_p4`） | 6.003562e-01（`grad_p1`） |
| `nn_batch_norm3d_train_step_s2` | K | Req2 | 9 | 9 | 725 | 0 | 1.192093e-07（`forward`） | 1.200014e+00（`grad_p1`） |
| `nn_channel_shuffle_train_step_s1` | H | Req2 | 9 | 9 | 667 | 0 | 2.980232e-08（`forward`） | 4.614050e-06（`grad_p4`） |
| `nn_channel_shuffle_train_step_s2` | H | Req2 | 9 | 9 | 658 | 0 | 2.980232e-08（`grad_p5`） | 2.084191e-04（`grad_p4`） |
| `nn_adaptive_pool3d_train_step_s1` | H | Req2 | 7 | 7 | 248 | 0 | 2.980232e-08（`loss`） | 8.298027e-08（`loss_after_step`） |
| `nn_adaptive_pool3d_train_step_s2` | H | Req2 | 7 | 7 | 251 | 0 | 2.980232e-08（`loss_after_step`） | 1.331295e-07（`loss_after_step`） |
| `nn_lp_pool_train_step_s1` | H | Req2 | 7 | 7 | 149 | 0 | 1.490116e-08（`loss_after_step`） | 6.735516e-08（`loss_after_step`） |
| `nn_lp_pool_train_step_s2` | H | Req2 | 7 | 7 | 125 | 0 | 1.192093e-07（`loss`） | 7.951777e-08（`loss`） |
| `nn_fractional_max_pool_train_step_s1` | H | Req2 | 7 | 7 | 149 | 0 | 2.980232e-08（`loss_after_step`） | 1.069236e-07（`loss_after_step`） |
| `nn_fractional_max_pool_train_step_s2` | H | Req2 | 7 | 7 | 179 | 0 | 5.960464e-08（`loss`） | 8.960326e-08（`loss`） |
| `nn_bilinear_train_step_s1` | H | Req2 | 4 | 4 | 67 | 0 | 0 | 0 |
| `nn_bilinear_train_step_s2` | H | Req2 | 4 | 4 | 131 | 0 | 4.768372e-07（`loss`） | 1.389802e-07（`loss_after_step`） |
| `nn_similarity_distance_train_step_s1` | H | Req2 | 3 | 3 | 32 | 0 | 0 | 0 |
| `nn_similarity_distance_train_step_s2` | H | Req2 | 3 | 3 | 93 | 0 | 9.536743e-07（`loss`） | 8.545184e-08（`loss`） |
| 計 | | | 120 | 120 | 7003 | 0 | | |

- 相対誤差が 1e-3 を超えた記録は 2 行だけで、どちらも `nn_batch_norm3d_train_step` の `grad_p1`（要素数 3）。
  s1 は max_abs_diff 1.410262e-08・max_rel_err 6.003562e-01、s2 は max_abs_diff 1.049244e-08・max_rel_err 1.200014e+00。
  いずれも絶対誤差 1e-5 未満の側で統一複合判定を満たし、fail_count は 0（判定は変えていない）。
- BatchNorm3d の統計は 1 回目 README の「BatchNorm3d の統計の契約」のとおり Metal の soft-f64 経路で蓄積する。入力は小さな有限値で、
  2e20 級の overflow 確認は #4031 の範囲でありこの測定ではしていない。

## 回収物

| ファイル | 内容 |
|---|---|
| `rev-stamp.txt` | 展開ツリーの `.rev-stamp`（2 行形式） |
| `env_info.txt` | 実行前後の環境（macOS・CPU・メモリ量と空き率・rustc・負荷・ロック保持） |
| `build.log` | 事前ビルド（`--no-run`）のログ |
| `test_result.tsv` | テスト単位の判定表（1 行） |
| `summary.tsv` | `TRAIN_PARITY` 記録の集計（120 行。スクリプトがマスク後のログから作る） |
| `logs/nn-template-3861/` | テストの全出力（`--nocapture`。末尾に `rc=` と `train_parity_records:`）と終了コード |
| `SHA256SUMS` | 上記の SHA-256 |

## #4044 追記（共有 target の成果物の再利用の棚卸し）

- 分類: **A（再コンパイルを確認）**
- 根拠: build.log に 9 メンバー全部の `Compiling` がある。
- 判定基準・全 16 計測の表・限界は [`docs/real-hardware-shared-target-reuse-decision.md`](../../../real-hardware-shared-target-reuse-decision.md) §4。本追記は結果を変えない（既存の本文・回収ログは無改変）。
