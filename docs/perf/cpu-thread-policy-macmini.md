# Mac mini（4P＋6E）の CPU スレッド方針 A/B（イシュー #3178）

## 状態

**実測済み（2026-10-21。Mac mini M4・専有ゲート 5/5 round 成立）。2 ゲートとも総合判定は REJECT**（規則 6 を機械適用。ADOPT 候補なし）。
計測した before の commit は `3637b4af283f72bde163fb5b799603b856247ef7`（`logs/cpu-thread-policy-macmini-3178/macmini/rev-stamp.txt`）。
main のゲート定数（`BIG_CORE_LIMIT_ENABLED = false`・`SMALL_SHAPE_CAP_ENABLED = false`）は変更していない（計測は計測専用 checkout。本 PR は docs のみ）。
計測時の main は `SME_PRODUCTION_ENABLED = false`（before／after 共通）。過去の REJECT（#1364・#1575）は覆さない。

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

実測ログ（マスク済み）は `logs/cpu-thread-policy-macmini-3178/macmini/{big-core,small-cap}/`、集計は `aggregate-big-core.md`・`aggregate-small-cap.md`（`aggregate.py` の出力そのまま）。

### 実行条件と前提の確認（規則 3・5）

| 項目 | big-core | small-cap |
|---|---|---|
| 機体 | Mac mini M4（brand `Apple M4`・論理 10・P4＋E6・16 GiB・macOS 27.0.1・rustc 1.99.0）。`RAYON_NUM_THREADS` 未設定 | 同左 |
| 専有ゲート（`load_gate-*`） | 5/5 round `gate_ok=1`・`series=official`（load1 2.41〜2.70。閾値 3.0） | 5/5 round `gate_ok=1`・`series=official`（load1 2.33〜2.87） |
| 機構発火の前提（`precheck-*`） | 成立。before は `effective=10`、after は `detected_big_cores=Some(4)`・`env_override=false`・`effective=4` | 成立。before は `eligible=false`・`pool_active=false`、after は `eligible=true`・`env_override=false`・`cap_threads=6`（`current=10`）・`pool_active=true` |
| 腕の差分（`gate_diff.txt`） | `thread_limit.rs` の 1 行（false→true） | `small_shape_thread_cap.rs` の 2 行（`SMALL_SHAPE_CAP_ENABLED` と `ELIGIBLE_BRANDS` へ `"Apple M4"`）。brand は実測値と一致していたため patch の修正なし |
| checksum | 12 セル全 round 一致 | 同左 |
| 判定不能（`skipped-*`）・ラッパー終了コード | 空・`run_ab-big-core.rc` = 0 | 空・`run_ab-small-cap.rc` = 0 |
| `compare_gemm_ab.py` の終了コード | gemm=3（REJECT）・train=0・infer=0 | gemm=3・train=3・infer=0 |

load1 は閾値 3.0 を下回ったが 2.3〜2.9 で、専有機とはいえ余裕は大きくない（計測前の確認時の load1 は 1.35、ゲート待ちは round 1 で最長 270 秒。この機体は他イシューの計測にも使われており、load の内訳は未特定）。
各 round の `uptime-*` を収録した。

### 12 セルの中央値比（after/before。5 round。1.00 以下が望ましい）

| セル | big-core | big-core 1.00 以下の round | small-cap | small-cap 1.00 以下の round |
|---|---|---|---|---|
| gemm 256 fresh | 1.0046 | 4/5 | 0.9386 | 3/5 |
| gemm 256 reuse | 1.0432 | 2/5 | 0.9659 | 4/5 |
| gemm 512 fresh | 0.9852 | 3/5 | 1.0098 | 1/5 |
| gemm 512 reuse | 1.0199 | 1/5 | 0.9794 | 5/5 |
| gemm 1024 fresh | 1.1174 | 0/5 | 0.9869 | 5/5 |
| gemm 1024 reuse | 1.1435 | 0/5 | 0.9923 | 5/5 |
| gemm 2048 fresh | 0.9906 | 4/5 | 0.9944 | 3/5 |
| gemm 2048 reuse | 1.0254 | 0/5 | 1.0125 | 2/5 |
| train 64 fresh | 0.9888 | 4/5 | 1.0432 | 0/5 |
| train 64 reuse | 0.9564 | 4/5 | 0.9832 | 3/5 |
| infer 64 fresh | 0.9744 | 4/5 | 0.8962 | 5/5 |
| infer 64 reuse | 0.9742 | 5/5 | 0.8488 | 5/5 |

### 総合判定（規則 6）

| ゲート | 中央値比 > 1.00 のセル | 5/5 round 一貫で改善したセル | checksum | PyTorch 比（before／after。gemm N=2048 reuse） | 総合判定 |
|---|---|---|---|---|---|
| big-core | 6 セル（gemm 256 fresh・256 reuse・512 reuse・1024 fresh・1024 reuse・2048 reuse） | infer 64 reuse のみ | 全一致 | 0.502／0.495 | **REJECT** |
| small-cap | 3 セル（gemm 512 fresh・2048 reuse・train 64 fresh） | gemm 512 reuse・1024 fresh・1024 reuse・infer 64 fresh・infer 64 reuse（5 セル） | 全一致 | 0.504／0.493 | **REJECT** |

### PyTorch 比（規則 7。記録のみ・判定に使わない）

同じ round 内の（PyTorch `median_s` ÷ fandhe `median_s`）の 5 round 中央値。1 未満は PyTorch の所要時間が短い（fandhe が遅い）ことを表す。
全 12 セル × before／after の値は `aggregate-*.md` の列に記録した。要点は次のとおり。

- gemm は両ゲートとも before 0.50〜0.75・after 0.49〜0.75 の範囲で、ゲートの on/off で順位は変わらない。N=2048 は 0.49〜0.54 で、E コア比率の対策になりうる big-core 腕でも 0.495（reuse）・0.539（fresh）のまま。
- big-core の gemm N=1024 は after で PyTorch 比が悪化した（fresh 0.635→0.568、reuse 0.606→0.532）。
- small-cap の infer 64 は after で改善した（fresh 0.154→0.178、reuse 0.157→0.183）。それでも PyTorch 比は 0.2 未満で、差は埋まらない。train 64 は 0.19 前後でほぼ不変。

### 読み取れること（断定しない）

- **big-core**: 並列度を P コア数（4）に絞る腕は、gemm N=1024 を 5/5 round で約 12〜14% 遅くした。N=2048 では fresh が 0.9906 と中立、reuse が 1.0254 で 5 round 中 0 round が 1.00 以下だった。train／infer は中立〜小幅改善（0.956〜0.989）にとどまる。本腕は job_grid の並列度だけを 4 にし、プールは 10 スレッドのまま affinity もないため、PyTorch の `set_num_threads(4)` とは別物である。
- **small-cap**: 処置セルは train／infer の 4 セルで、infer は 5/5 round で 10〜15% 改善、train fresh は 0/5 round で約 4% 悪化、train reuse は中立だった。発火しないはずのガードセル（gemm 8 セル）にも ±6% 程度の揺れがあり（gemm 256 fresh 0.9386 等）、これは本計測のノイズ幅の目安になる。小形状 GEMM の job 統合・pack 共有（#3198）の動機として、infer の改善余地が実在することは示唆される。
- **E コア仮説**（Mac mini で gemm N=2048 reuse 比が落ちる原因は E コア比率）: 本結果の範囲では**支持されない**。並列度を 4 に絞っても N=2048 の PyTorch 比は 0.49〜0.54 のままで、reuse はむしろ悪化側（1.0254）だった。ただし本腕は affinity もプール縮小もないため、「P コアだけで走らせた場合」の検証ではなく、仮説の否定でもない。

## 申し送り・スコープ外

- #3198（m が小さい GEMM の job 統合・pack 共有の設計）: small-cap の infer 64 の 10〜15% 改善は、専用プールで小形状を分離する価値の根拠になりうる。train 64 fresh の悪化（約 4%）と合わせて、設計時の対照データにする。
- #3227（ADOPT 候補の既定 ON 化・本番の `ELIGIBLE_BRANDS` 追加）: 本計測は 2 ゲートとも REJECT のため、既定反転の根拠は得られなかった。反転・追加は行っていない。
- 範囲外: 本番の `ELIGIBLE_BRANDS` 追加と既定の反転（#3227）、`SMALL_SHAPE_CAP_THREADS = 4` の変種、`RAYON_NUM_THREADS=4` の参考腕、GB10 での再計測、affinity 機構（P コアへの固定）、SME（#3175）。
- 運用メモ: `run_ab_thread_policy_macmini_cpu.sh` は出力（`results/raw/`・`compare-*.md`・`target-ab-*`）が gitignore 対象外で、同じ before ツリーで 2 本目を走らせると未追跡検査で止まる。本計測はゲートごとに before／after の clone 対を分けて回避した（`logs/cpu-thread-policy-macmini-3178/README.md`）。恒久対応（出力先をツリー外へ移す等）は別途判断する。
