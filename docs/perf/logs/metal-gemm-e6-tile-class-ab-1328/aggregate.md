# イシュー #1328 集計（E6 タイルクラス分割 2 クラス経路 vs 現行経路）

実行環境: `env_info.txt`。base commit: `b41bba6`（#1327・PR #1388 マージ後）。

## 抽出コマンド

```bash
# 候補 0/4/5/8（AC 形状）
grep -E "^N=[0-9]+ cand[0-9]+ mode=|split_over_legacy_kernel_gpu=|tile_class_edge_dispatch_increment=" kernel_gpu_run*.log

# 本番 select_for_device 選択構成（追補）
grep -E "^\s*\[(legacy|split)\] N=|commit_wait\.kernel_gpu:" kernel_gpu_production_select_run*.log
```

## A. 候補 0/4/5/8（AC 形状。`tile::CANDIDATES[i]`）× N=1024/2048/4096

5 プロセス起動（run1〜run5）・各 20 warmup + 20 測定・`kernel_gpu`（GPU タイムスタンプ）中央値。
`tile_class_edge_dispatch_increment` は全 (N, cand) で 5 run とも `0`、
`tile_class_interior_dispatch_increment` は全 (N, cand) で 5 run とも `40`
（= `WARMUP_TRIALS + MEASURED_TRIALS`）——AC 形状では `TileClassMode::Split`
が「Interior クラス（direct-load 強制）のみを grid 全体へ適用した 1
dispatch」へ確実に縮退することを裏付ける（`docs/perf/
metal-gemm-tile-class-split.md` の `TileClassMode` ドキュメンテーション
コメント参照。Edge＝staged 強制・Interior＝direct-load 強制）。

| N | cand | legacy median (ms, 5 run 中央値) | split median (ms, 5 run 中央値) | run 別比 (split/legacy) | 中央値比 |
|---|---|---|---|---|---|
| 1024 | 0 | 5.6569 | 3.6782 | 0.6806 / 0.6502 / 0.6501 / 0.6499 / 0.6511 | **0.6502** |
| 1024 | 4 | 2.4562 | 2.0830 | 0.8539 / 0.8650 / 0.8530 / 0.8465 / 0.8472 | **0.8530** |
| 1024 | 5 | 0.3552 | 1.1016 | 3.1010 / 2.7412 / 2.6662 / 2.6827 / 2.7380 | **2.7380** |
| 1024 | 8 | 1.9392 | 1.1774 | 0.6181 / 0.5025 / 0.5923 / 0.5722 / 0.6254 | **0.5923** |
| 2048 | 0 | 13.6808 | 7.0745 | 0.5704 / 0.5193 / 0.5169 / 0.5213 / 0.5155 | **0.5193** |
| 2048 | 4 | 34.8062 | 16.5568 | 0.4762 / 0.4761 / 0.4756 / 0.4751 / 0.4761 | **0.4761** |
| 2048 | 5 | 1.8344 | 4.8899 | 2.6653 / 2.6646 / 2.6693 / 2.6396 / 2.6163 | **2.6646** |
| 2048 | 8 | 15.7692 | 7.1534 | 0.4560 / 0.4481 / 0.4502 / 0.4536 / 0.4620 | **0.4536** |
| 4096 | 0 | 106.4910 | 57.2960 | 0.5384 / 0.5390 / 0.5375 / 0.5341 / 0.5542 | **0.5384** |
| 4096 | 4 | 306.0033 | 132.5602 | 0.4231 / 0.4275 / 0.4339 / 0.4365 / 0.4681 | **0.4339** |
| 4096 | 5 | 18.6987 | 43.5757 | 2.1945 / 2.1929 / 2.3304 / 2.4310 / 2.4784 | **2.3304** |
| 4096 | 8 | 127.0828 | 59.1988 | 0.4610 / 0.4658 / 0.4602 / 0.4770 / 0.4981 | **0.4658** |

全ての `TILE_CLASS_SPLIT_FALLBACK_COUNT` 増分は 0（fail-closed assert 内蔵。
`gemm::tests::tile_class_split_bit_match_*` 4 テストも全 pass。
`smoke_tile_class_bit_match.log` 参照）。head/base の trial 0 出力は全
(N, cand) で bit 完全一致（テスト内 `assert_eq!` で確認）。

**候補依存で符号が割れる**: cand0/4/8（いずれも `bk=16`）は Split が
一貫して legacy より速い（0.45〜0.85 倍）。cand5（`bk=32`。本 CANDIDATES
中で唯一の `bk=32`）は Split が一貫して legacy より遅い（2.3〜2.7 倍）。
run 間の符号はどの (N, cand) の組でも一貫している。

## B. 本番 `dispatch_auto` 選択構成（`tile::select_for_device`）× N=512/1024/2048/4096（追補）

候補 0/4/5/8 はいずれも本番非選択構成（M4 Max では N=512→`CANDIDATES[5]`・
1024→`[6]`・2048→`[1]`・4096→`[2]`）であるため、上記 A の結果だけでは
本番 `dispatch_auto` への外挿ができない（計画「§3.4 採否・本番結線の
判断規則」）。`gemm_reuse_phase_diag_tests::run_size_with`（`select_for_
device` で構成解決）で base（Legacy）/head（Split）を本番選択構成のまま
直接比較した。

| N | 選択構成 (bm,bn,bk,wm,wn) | legacy median (ms, 5 run) | split median (ms, 5 run) | run 別比 (split/legacy) | 中央値比 |
|---|---|---|---|---|---|
| 512  | (64,32,32,2,2)＝`CANDIDATES[5]` | 0.2015 | 0.1081 | 2.492 / 2.495 / 0.536 / 0.535 / 0.534 | 0.5365（符号不一致。後述） |
| 1024 | (64,32,8,4,1)＝`CANDIDATES[6]`  | 0.2245 | 0.5667 | 2.517 / 0.552 / 2.495 / 2.529 / 2.524 | **2.5243**（4/5 run が後退方向。1 run 外れ値） |
| 2048 | (64,32,16,2,2)＝`CANDIDATES[1]` | 1.6030 | 4.9021 | 4.330 / 3.054 / 3.061 / 2.334 / 3.058 | **3.0581**（5/5 run とも後退方向） |
| 4096 | (32,64,16,2,2)＝`CANDIDATES[2]` | 13.7657 | 32.1276 | 2.334 / 2.242 / 2.117 / 2.357 / 2.344 | **2.3339**（5/5 run とも後退方向） |

N=512 は選択構成が `bk=32`（A 節の cand5 と同族）で、5 run 中 2 run が
「legacy と同程度」・3 run が「legacy の半分」という二峰性を示し
run 間の符号が一貫しない（`docs/perf/
cuda-large-buffer-percall-alloc-transfer-threshold.md` 等で既知の環境
揺らぎ系と同種の可能性があるが未確定）。N=1024 は 5 run 中 4 run が
2.5 倍前後の一貫した後退を示す一方 1 run のみ 0.552 倍の外れ値がある。
N=2048／4096 は 5 run すべてが 2.1〜4.3 倍の一貫した後退を示す。

**総括**: 本番が実際に選択する構成（`bk=16` 系の `CANDIDATES[1]/[2]/[6]`）
では `TileClassMode::Split` は N=1024/2048/4096 のいずれでも一貫して
legacy より遅い（run 間の符号もおおむね一貫）。A 節で観測した cand0/4/8
の改善は、これらが本番非選択の別構成であることに起因し、本番選択構成
には外挿できない。
