# 学習ステップの parity 雛形 optim 族の Metal 実機 parity（#3863 の 2 回目）

対象は `crates/facade/tests/train_gpu_parity.rs` の optim 族（#4026・PR #4034 で登録した 6 ケース。param groups 付きの
デバイス常駐更新〈SGD〈momentum・nesterov〉・Adam・AdamW〉を 20 step 回した後のパラメータ）。`train_parity_optim_metal_matches_cpu` を
1 本だけ実行し、Metal の常駐更新の腕をホスト参照（`ParamGroupStep::step_with_groups` + `apply_parameters`）と突き合わせた。
機体は `<metal-node>`（Mac mini・Apple M4・メモリ 16GB。`docs/real-hardware-verification-env.md` §7）。

1 回目（[`../param-groups-resident-metal-parity-3863/`](../param-groups-resident-metal-parity-3863/)。`docs/autodiff-param-groups-decision.md`
§19.4 の Metal 側 `#[ignore]` 3 本）では、雛形の optim 族が登録 0 件で undetermined だった。この記録は #4026 のマージ後に雛形テストを
測り直したもので、1 回目のディレクトリは書き換えていない。GB10 版は #3862。雛形は `../train-gpu-parity-template/`（#3778）。

- 事前登録: [`RULE.txt`](./RULE.txt)（実機実行より前にコミット `31656bccc` で push した。実行後に変えていない。宣言時刻の誤記は下の「正誤」）
- 実行スクリプト: [`run_metal_parity.sh`](./run_metal_parity.sh)（Mac mini 上で手動実行。CI からは呼ばない。`--self-test` 付き）。
  `../nn-layers-metal-parity-3861/run_metal_parity.sh` の複製に、#3861 の 2 回目（`../nn-train-template-metal-parity-3861/`）と同じ改変
  （対象を雛形の 1 本に絞る・`TRAIN_PARITY` 記録の集計と行数検査）を入れた。#3861 の 2 回目との違いは `ISSUE`・`CASES`・`EXPECTED_RECORDS` だけ
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline・判定規則・テストコードは不変

## 状態

**実測済み: `TRAIN_PARITY` 記録 24 行中 24 行 pass、fail 0、undetermined 0**（2026-11-04・`<metal-node>`）。
テストは exit 0・`running 1 test`・`test result: ok. 1 passed; 0 failed`（`test_result.tsv`）。記録行数は RULE.txt の
`expected_records: 24` と一致した。1 回目の 3 本 pass と合わせ、#3863 の完了条件を満たす。

- 実行コミット（`rev-stamp.txt`）: `31656bccc2484d601b7b1bfa0c34b949b2e7b79e`・`dirty=0`。事前登録と実行スクリプトを入れた最初のコミットそのもので、
  `crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は対象コミット `2f59b9638`（origin/main）と同一（送信前に `git diff --quiet` で確認）。
  転送は rsync だけで、ノード側で `.rev-stamp` を読んで送信側の HEAD と照合した。
- スクリプト対応: `env_info.txt` の `script_sha256`（`c0553ec3…f195`）は、上記コミットの `run_metal_parity.sh` の SHA-256 と一致する
  （送信側・ノード側の両方で確認）。
- 実機ロック: `env_info.txt` の実行前・実行後とも `hw_lock: held owner=#3863 2026-11-04T19:09:39Z`。終了後にロックが外れていることを確かめた。
  ほかの利用者のプロセスは止めていない。並走する cargo／rustc は事前ビルド前・テスト前とも無し。メモリの空き率は実行前後とも 72%。
- 回収物は手で編集していない。`SHA256SUMS` はこのディレクトリで `shasum -a 256 -c SHA256SUMS` が通る（7 ファイルすべて OK）。
  ホスト名・`/Users/`・ユーザー名の残存は 0 件（`<metal-node>`／`<user>`／`<home>` へマスク済み。送信側でも grep で確認）。
- 事前ビルド（`--no-run`）は 10.09 秒（`build.log`）。テストは 0.53 秒で終わった。

## 正誤（RULE.txt の宣言時刻）

RULE.txt の `declared_at_utc: 2026-11-04T19:10Z` は誤記である。RULE.txt を入れたコミット `31656bccc` のコミット時刻は 2026-11-04T19:09:16Z、
実行開始（実機ロックの owner）は 2026-11-04T19:09:39Z で、記載の 19:10Z はどちらより後の分になっている（宣言を書いた時点で分を先回りして記入した）。
事前登録の実体（RULE.txt と実行スクリプトを実行より前にコミットして push したこと）は、コミット時刻と `rev-stamp.txt` で確かめられる。
RULE.txt は計測後に変えない規則のため、ここで正誤として記録し、RULE.txt 本体は書き換えていない。

## 結果表（ケース別）

各行は `summary.tsv`（1 記録 1 行。列は case・family・quantity・arm・route・judge・total・fail_count・max_abs_diff・mean_abs_diff・max_rel_err・verdict・reason）
をケースごとにまとめたもの。判定は全行 Req2（REQ-2 統一複合判定。相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
`docs/autodiff-param-groups-decision.md` §19.5 (3)）。量は 20 step 後の `param_w1`・`param_b1`・`param_w2`・`param_b2`。

| ケース | route | judge | 記録 | pass | 要素数 | fail_count | 最大 max_abs_diff（量） | 最大 max_rel_err（量） |
|---|---|---|---|---|---|---|---|---|
| `optim_param_groups_resident_sgd_s1` | K | Req2 | 4 | 4 | 212 | 0 | 2.980232e-08（`param_w1`） | 7.364657e-06（`param_w2`） |
| `optim_param_groups_resident_sgd_s2` | K | Req2 | 4 | 4 | 364 | 0 | 2.980232e-08（`param_b2`） | 1.377204e-06（`param_w2`） |
| `optim_param_groups_resident_adam_s1` | K | Req2 | 4 | 4 | 212 | 0 | 0 | 0 |
| `optim_param_groups_resident_adam_s2` | K | Req2 | 4 | 4 | 364 | 0 | 0 | 0 |
| `optim_param_groups_resident_adamw_s1` | K | Req2 | 4 | 4 | 212 | 0 | 0 | 0 |
| `optim_param_groups_resident_adamw_s2` | K | Req2 | 4 | 4 | 364 | 0 | 0 | 0 |
| 計 | | | 24 | 24 | 1728 | 0 | | |

- 相対誤差が 1e-3 を超えた記録は無い。Adam・AdamW の 4 ケースは全量で max_abs_diff が 0（判定は Req2 のまま。bit 一致を主張するものではない）。

## 回収物

| ファイル | 内容 |
|---|---|
| `rev-stamp.txt` | 展開ツリーの `.rev-stamp`（2 行形式） |
| `env_info.txt` | 実行前後の環境（macOS・CPU・メモリ量と空き率・rustc・負荷・ロック保持） |
| `build.log` | 事前ビルド（`--no-run`）のログ |
| `test_result.tsv` | テスト単位の判定表（1 行） |
| `summary.tsv` | `TRAIN_PARITY` 記録の集計（24 行。スクリプトがマスク後のログから作る） |
| `logs/optim-template-3863/` | テストの全出力（`--nocapture`。末尾に `rc=` と `train_parity_records:`）と終了コード |
| `SHA256SUMS` | 上記の SHA-256 |

## #4044 追記（共有 target の成果物の再利用の棚卸し）

- 分類: **A（再コンパイルを確認）**
- 根拠: build.log に 9 メンバー全部の `Compiling` がある。
- 判定基準・全 16 計測の表・限界は [`docs/real-hardware-shared-target-reuse-decision.md`](../../../real-hardware-shared-target-reuse-decision.md) §4。本追記は結果を変えない（既存の本文・回収ログは無改変）。
