# m が小さい GEMM の分割方式 A/B（イシュー #3201）

設計は [`docs/cpu-gemm-small-m-job-merge-prepack-decision.md`](../../../cpu-gemm-small-m-job-merge-prepack-decision.md)（§13 に結果）、
雛形は [`../cpu-small-m-gemm-partition-3198/`](../cpu-small-m-gemm-partition-3198/README.md)、結論は
[`docs/perf/cpu-gemm-small-m-partition-ab.md`](../../cpu-gemm-small-m-partition-ab.md)。

## 状態

**計測完了**。Mac mini 3 腕・GB10 6 腕とも正式系列（5/5 round が負荷ゲート pass）。判定は全腕 REJECT（`aggregate.md`）。
既定は変えていない（反転は #3227）。

## 収録物

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（#3198 の雛形の逐語コピー。末尾の `fixed-at` と計測コミットだけを計測前に記入） |
| `RULE-addendum.txt` | 判定を変えない運用上の明確化（lib テストの扱い・発火確認の入口・参考比） |
| `orchestrate.sh` | N 腕オーケストレーション（P-TREE・ビルド・機構発火の前提確認・bit 一致・5 round ローテーション計測）。`--dry-run` あり |
| `aggregate.py` | fail-closed 集計（標準ライブラリのみ。`python3 -I aggregate.py --self-test`）。出力は `aggregate.md`（手で直さない） |
| `aggregate.md` | `python3 -I aggregate.py` の出力 |
| `ab_patch_merge.diff` | G1 `SMALL_M_JOB_MERGE_ENABLED` を true にする 1 行 patch |
| `ab_patch_prepack.diff` | G2 `RESIDENT_W_PREPACK_ENABLED` を true にする 1 行 patch |
| `ab_patch_aff.diff` | `GB10_AFFINITY_ENABLED` を true にする 1 行 patch（GB10 の組み合わせ腕用） |
| `{macmini,gb10}/` | 実測ログ（マスク済み）。`.rev-stamp`・`env_info.txt`・`uname.txt`・`load_gate.log`・`tree_verify.txt`・`patch_sha256.txt`・`bench_sha256.txt`・`tree-<arm>.txt`・`build-*.log`・`prereq/<arm>.log`・`bitmatch.log`・`ab/run{N}-{arm}.log`・`fc/results-*.jsonl`・`uptime.log`・`DONE` |

腕: Mac mini は off／merge（merge.diff）／mergeprepack（merge+prepack.diff）。GB10 は各腕に aff.diff を足した `*-aff` を加えた 6 腕。

## 計測を可能にするために入れたテスト変更（PR #3523）

- `gate_is_off_by_default` 3 件を `const { assert! }` から実行時 assert へ変更。const のままだとゲート反転ツリーで
  lib テストがコンパイル不能（E0080）になり、report テストを実行できなかった。main（OFF）での固定は不変。
- `resident_w_prepack_fire_report`（`#[ignore]`）を追加。本番のゲート定数を読む入口で G2 の構築・引き当て回数を表示する。

## 手順

1. 計測コミットの HEAD とツリー指紋（`git archive` の sha256）を `.rev-stamp`（1 行目 HEAD・2 行目指紋）へ書き、両機体の専用ディレクトリへ rsync する。
2. 各機体で `BASE=<同期ツリー> WORK=<同期ツリー外> AB_TORCH_PYTHON=<torch 入り python> bash orchestrate.sh <macmini|gb10>` を切り離して起動する（失敗したら再実行・差し替えはしない）。
3. ログを回収し、ホーム絶対パス・ホスト名・ユーザー名がないことを確認して `macmini/`・`gb10/` へ置く。`orchestrate.sh` の `mask()` を通らない `build-*.log` はホームの絶対パスを `<home>` へ置換した。
4. `python3 -I aggregate.py > aggregate.md`。

## 逸脱

- 初回起動（コミット `2b7f11263`）は事前確認の検査で `probe=` 行の先頭が `test … ... ` 接頭辞付きになり行数検査が不一致となって、計測前に停止した（計測は 0 件）。
  `orchestrate.sh` を修正（`2ba22c432`）して新しいコミットからやり直した。`.rev-stamp` はこの 2 回目のコミット。
  RULE.txt・RULE-addendum.txt・判定式・閾値は不変。GB10 の初回起動はビルド後に手動で停止した（計測前）。
- `compare_gemm_ab.py` による train／infer の照合は実施していない（`aggregate.py` が同じ 5 回中央値・checksum 契約で判定）。
