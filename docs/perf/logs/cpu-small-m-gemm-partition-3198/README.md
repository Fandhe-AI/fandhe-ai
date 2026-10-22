# m が小さい GEMM の job 統合・事前 pack の A/B 雛形（イシュー #3198）

設計は [`docs/cpu-gemm-small-m-job-merge-prepack-decision.md`](../../../cpu-gemm-small-m-job-merge-prepack-decision.md)。
本 dir は #3201 が A/B を実行するための雛形だけを置く（計測ログはまだない）。

## 収録物

| ファイル | 内容 |
|---|---|
| `RULE.template.txt` | 事前登録規則の雛形。腕・セル・反復・専有ゲート・4 値判定まで固定済み |

## 使い方（#3201）

1. 計測前に自分のログ dir へ `RULE.template.txt` を `RULE.txt` としてコピーし、`fixed-at` と計測コミットだけを書き足す。判定条件は緩めない。
2. after 腕は計測専用の checkout で const を 1 行反転して作る（patch を収録。main へはコミットしない）。
   - G1: `crates/backend-cpu/src/gemm_blis/small_m.rs` の `SMALL_M_JOB_MERGE_ENABLED`
   - G2: `RESIDENT_W_PREPACK_ENABLED`（#3200 で追加）
   - GB10 の組み合わせ腕は `GB10_AFFINITY_ENABLED`（`crates/backend-cpu/src/gb10_affinity.rs`）も反転する。
3. 機構が発火したことは、#3199・#3200 が用意する `#[ignore]` の report テストで確認する。未用意の間は prepack・A 共有の腕は undetermined とする。
4. 前例の手順（rsync・オーケストレーション・マスク）は [`../cpu-jobgrid-3177/README.md`](../cpu-jobgrid-3177/README.md) に従う。
