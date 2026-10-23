# #3195 MSE backward 逐次しきい値の再測定（T = 4096）

状態: **判定済み: ADOPT**（2026-10-23。規則 `RULE.txt` §6 を機械的に適用）。
`MSE_BACKWARD_PARALLEL_MIN_ELEMS` を 0 から 4096 へ変更した。#1578（T = 1<<18・REJECT）の記録は不変。

| 項目 | 内容 |
|---|---|
| 規則 | `RULE.txt`（計測前に固定。本計測では編集していない）。Stage 0 集計器 `aggregate_stage0.py` は実機ログより前のコミットで固定 |
| ハーネス | `scripts/bench/framework-compare/run_ab_mse_backward_threshold_cpu.sh`（未変更） |
| 候補 | T = 4096（単一） |
| before 腕 | main HEAD `fd1ca01a9` の `crates/facade`（定数 0。clean） |
| after 腕 | 同一 sha + `mse.rs` 定数 1 行のみ `= 4096;`（未コミット。`rev-stamp.txt` の diff 参照） |
| runner | 作業ブランチ先端 `c8bb8448d`（集計器コミットを含む）。ハーネス・Stage 0 をここから実行 |
| 判定セル | train 64 × {fresh, reuse}（cpu）。`--phases` は記録のみ（判定に使わない） |
| 機体 | Mac mini（`macmini/`）・GB10（`gb10/`）。Stage 1 は各 5 round・各 round で起動順反転 |
| 腕の同定 | `.rev-stamp` は未追跡ファイル検査に掛かるため置かず、`rev-stamp.txt`（3 sha）と `tree-*.txt`・`sha-*.txt` で代替 |
| 実行手順 | `session-scripts/`（`setup.sh`・`stage0.sh`・`stage1.sh`）。マスク済み |
| 過去記録 | #1578 の REJECT は不変。本件はそのデータを読み直していない |

## 事前確認
- 3 者 bit 一致テスト `mse::tests::mse_loss_backward_threshold_bit_exact` を含む `mse::tests` 全件が両機体で pass（`*/bit-exact.log`。9 passed・1 ignored）。FAIL 条件に該当しない。
- preflight: 両機体とも他セッションの cargo／rustc／bench プロセス 0 件（`*/preflight-procs.txt`）。`RAYON_NUM_THREADS`・`CARGO_TARGET_DIR` は unset（`*/env_info.txt`）。
- 前提差異: RULE §1 条件 2（GB10 device_update 279 -> 169 µs）は規則側の記述を継承し、本作業では `cpu-reuse-device-update.md` との再突合は行っていない（判定には使わない前提条件）。

## Stage 0（`stage0-aggregate.md`。5 プロセス中央値）
| 機体 | n | seq 中央値 ns | par 中央値 ns | r(n)=seq/par |
|---|---|---|---|---|
| Mac mini | 640 | 42 | 23833 | 0.0018 |
| Mac mini | 2560 | 125 | 30208 | 0.0041 |
| Mac mini | 4096 | 167 | 30791 | 0.0055 |
| GB10 | 640 | 48 | 33360 | 0.0014 |
| GB10 | 2560 | 128 | 42992 | 0.0030 |
| GB10 | 4096 | 192 | 45488 | 0.0042 |

- 前提ゲート (a) 全行 seq/par checksum 一致: 成立。
- 前提ゲート (b) 両機体 n in {640, 2560} で 5/5 プロセス r < 1.00: 成立（個別 r の最大は GB10 n=2560 の 0.013）。

## Stage 1（train 64・cpu。`compare-train-*.md`）
| 機体 | mode | before 中央値 | after 中央値 | after/before | checksum | round ごとの ratio | 判定 |
|---|---|---|---|---|---|---|---|
| Mac mini | fresh | 721.8 µs | 692.6 µs | 0.9595 | 完全一致 | 0.9904, 0.9667, 0.9549, 0.9559, 0.9962 | 非後退 |
| Mac mini | reuse | 698.5 µs | 670.5 µs | 0.9599 | 完全一致 | 0.9598, 0.9650, 0.9474, 0.9669, 0.9608 | 非後退 |
| GB10 | fresh | 885.4 µs | 850.2 µs | 0.9602 | 完全一致 | 0.9482, 0.9875, 0.9570, 0.9103, 0.9774 | 非後退 |
| GB10 | reuse | 862.7 µs | 827.3 µs | 0.9589 | 完全一致 | 0.9367, 0.9624, 0.9644, 1.0493, 0.9556 | 非後退 |

- 負荷ゲート（`load_gate-*.log`）: 両機体とも 5/5 round が gate=pass・gate_ok=1・series=official（Mac mini は load1 < 3.0、GB10 は load1 < 1.0）。
- GB10 reuse の round 4 のみ ratio > 1.00（1.0493）だが、5/5 一貫した後退ではなく中央値は 0.9589。RULE §6 の REJECT（全 round で > 1.00）には該当しない。
- 同一ラベルの再実行・run の差し替え・追加起動・セル除外は行っていない（各機体 1 系列のみ）。

## 総合判定（RULE.txt §6）
- ADOPT 条件: Stage 0 前提ゲート成立・両機体の全判定セル（train fresh・reuse）で 5 round 中央値 ratio <= 1.00・checksum 完全一致・両機体とも正式系列 → **すべて成立**。
- 解釈上の注記（RULE §1 注記どおり）: train セル（n = 640）では after 腕と #1578 の T = 1<<18 は同一の逐次分岐を通る。#1578 で GB10 reuse が後退した（1.0167）のに対し今回は 0.9589 だった差は、RULE §1 条件 2〜4（device_update 固定費の低下・負荷ゲート）による。系列間の差を本作業で因果確定していない。

## 結線
- `crates/backend-cpu/src/mse.rs` の `MSE_BACKWARD_PARALLEL_MIN_ELEMS` を `4096` に変更。tolerance・baseline・依存・公開 API・docs/spec は不変。
- テスト: 定数固定テスト `mse_backward_default_threshold_value_is_pinned` を追加し、`mse_loss_backward_threshold_bit_exact` の sizes に 4095・4096・4097 を追加。

## マスク（RULE §9）
ホスト名・ユーザー名・絶対パスは `<home>` に置換済み。ELF バイナリ（`bench-fandhe-3195-*`）は収録していない。
