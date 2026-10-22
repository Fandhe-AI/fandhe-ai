# m が小さい GEMM の分割方式 A/B（イシュー #3201）

設計は [`docs/cpu-gemm-small-m-job-merge-prepack-decision.md`](../../../cpu-gemm-small-m-job-merge-prepack-decision.md)、
雛形は [`../cpu-small-m-gemm-partition-3198/`](../cpu-small-m-gemm-partition-3198/README.md)。

## 状態

**実機計測は未実施（未計測）**。本 dir は計測の前提となる基盤だけを収録する。判定（ADOPT／REJECT）は
1 腕も出していない。既定は変えていない（反転は #3227）。数値を埋めた箇所はない。

## 収録物

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（#3198 の雛形の逐語コピー。`fixed-at` と計測コミットは計測前に記入するため未記入のまま） |
| `RULE-addendum.txt` | 判定を変えない運用上の明確化（lib テストの扱い・発火確認の入口・参考比） |
| `ab_patch_merge.diff` | G1 `SMALL_M_JOB_MERGE_ENABLED` を true にする 1 行 patch |
| `ab_patch_prepack.diff` | G2 `RESIDENT_W_PREPACK_ENABLED` を true にする 1 行 patch |
| `ab_patch_aff.diff` | `GB10_AFFINITY_ENABLED` を true にする 1 行 patch（GB10 の組み合わせ腕用） |

腕: Mac mini は off／merge（merge.diff）／merge+prepack（merge+prepack.diff）。GB10 は各腕に aff.diff を足した 6 腕。

## 計測を可能にするために入れたテスト変更

- `gate_is_off_by_default` 3 件を `const { assert! }` から実行時 assert へ変更。const のままだとゲート反転ツリーで
  lib テストがコンパイル不能（E0080）になり、report テストを実行できなかった。main（OFF）での固定は不変。
- `resident_w_prepack_fire_report`（`#[ignore]`）を追加。本番のゲート定数を読む入口で G2 の構築・引き当て回数を表示する。
  OFF（main）で `cache_len == 0`、ON（定数反転）で `builds == 1`・`hits >= 1` を手元の macOS で確認済み。

## 残作業（未実施）

- 両機体のオーケストレーション（`orchestrate.sh`）・集計（`aggregate.py`）は未作成。前例は
  [`../cpu-jobgrid-3177/orchestrate.sh`](../cpu-jobgrid-3177/orchestrate.sh)。
- 実機（Mac mini・GB10）での 5 round 計測、checksum 照合、bit 一致ログ、`.rev-stamp` の収録。
