# Mac mini（4P＋6E）の CPU スレッド方針 A/B（イシュー #3178）

## 状態

**計測基盤と事前登録規則まで完了。実測は未実施（判定 undetermined）**。本 PR を作った実行環境から Mac mini へ到達できなかったため、
値は推定で埋めていない。実測後にこの doc の「結果」節を埋める。main のゲート定数（`BIG_CORE_LIMIT_ENABLED = false`・`SMALL_SHAPE_CAP_ENABLED = false`）は変更していない。

## 目的

0.11.0 の Mac mini 系列 B では GEMM CPU の reuse 比（相手最速 ÷ fandhe reuse）が N=256/512/1024/2048 で 1.04/1.04/0.89/0.69 だった。
M4 Max の N=2048 は 0.80 で、Mac mini で落ちる原因は E コア比率の高さと推測されているが切り分けていない。
PyTorch は macOS／aarch64 で CPU スレッド数を P コア数にする（`docs/analysis/pytorch-2.14-cpu-path.md` §6.1）。この方針の再現を試す診断として、
既定 OFF の 2 ゲートを個別に反転して測る。結果は #3198（m が小さい GEMM の job 統合・pack 共有の設計）と #3227（ADOPT 候補の既定 ON 化）の前提データになる。

## 過去の REJECT との条件の違い

#1364（`docs/perf/cpu-gemm-default-thread-limit.md`）と #1575（`docs/perf/cpu-gemm-small-shape-thread-cap.md`）の判定は覆さない。本計測は条件を変えた再測定である。

| 項目 | 過去の REJECT | 本計測 |
|---|---|---|
| 機体 | #1364: GB10（`cpu_capacity` の誤検出を含む）と M4 Max（P12＋E4）の両機判定 / #1575: M4 Max | Mac mini M4（P4＋E6、論理 10） |
| 負荷 | #1575: load 22〜23 の共有環境・単発（record_only） | 専有機・load1 < 3.0・5 round |
| 分配方式 | #1364: 旧静的行パネル | 現行 `TwoDDynamic` |

## 方法（規則の正は `logs/cpu-thread-policy-macmini-3178/RULE.txt`）

- 腕: before = main、after = 計測専用 checkout にゲート 1 本だけを反転する patch を当てたもの。big-core 腕は 1 行、small-cap 腕は 2 行（`SMALL_SHAPE_CAP_ENABLED` と `ELIGIBLE_BRANDS` への Mac mini の brand 追加）。
- セル 12 個（gemm N=256/512/1024/2048、train／infer size=64、各 fresh/reuse）× 5 round・起動順を round ごとに反転・同じ run の中で PyTorch も測る。
- 機構発火の前提をベンチ前に `#[ignore]` の結合テスト（`thread_limit_report`・`small_shape_cap_report`）で fail-closed に確認する。
- 実行: `scripts/bench/framework-compare/run_ab_thread_policy_macmini_cpu.sh`（`compare_gemm_ab.py --sizes threadpolicy`）。集計: `logs/cpu-thread-policy-macmini-3178/aggregate.py`。

## 解釈上の注意

- big-core 腕は「4 スレッドで動かす」ことと同じではない。`effective_num_threads` は `job_grid` に渡す並列度を 4 に絞るだけで、job はグローバルプール（10 スレッド）上で走る。affinity もプール縮小もない。PyTorch の `set_num_threads(4)` とは意味が異なる。
- small-cap 腕の発火条件は `m * max(n, 8) * k < 2^24`。GEMM N≥256 の正方 GEMM はすべて発火しないガードセルで、処置セルは train／infer の 4 セルだけ。判定は 12 セル全部に同じ規則を適用する。

## 結果

未実測。実測後に次を記入する（2 ゲート × 12 セル）。

| ゲート | 12 セルの中央値比（after/before） | 5/5 round 一貫性 | checksum | PyTorch 比（before／after） | 総合判定 |
|---|---|---|---|---|---|
| big-core | 未実測 | 未実測 | 未実測 | 未実測 | undetermined |
| small-cap | 未実測 | 未実測 | 未実測 | 未実測 | undetermined |

## 申し送り・スコープ外

- #3198／#3227: 実測前は ADOPT／REJECT の根拠にならない。ADOPT 候補が出ても BIG_CORE は GB10 で REJECT のままなので M4 限定の分岐が前提になる。
- 範囲外: 本番の `ELIGIBLE_BRANDS` 追加と既定の反転（#3227）、`SMALL_SHAPE_CAP_THREADS = 4` の変種、`RAYON_NUM_THREADS=4` の参考腕、GB10 での再計測、affinity 機構、SME（#3175）。
