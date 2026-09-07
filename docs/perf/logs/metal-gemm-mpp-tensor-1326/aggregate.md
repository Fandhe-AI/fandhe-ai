# イシュー #1326 実測集計（M4 Max・2026-09-07）

## 抽出コマンド

```
grep '^N=' docs/perf/logs/metal-gemm-mpp-tensor-1326/kernel_gpu_run*.log
```

## head_over_base_kernel_gpu（MPP `matmul2d`（head）/ 本番選択構成（base）。純カーネル時間）

| N | run1 | run2 | run3 | run4 | run5 | 5 run 中央値 | 符号一貫性 |
|---|---|---|---|---|---|---|---|
| 1024 | 1.0223 | 0.9660 | 1.0311 | 1.0178 | 1.0347 | **1.0223** | 4/5 が head>base（1 run のみ head<base。共有負荷下のノイズ） |
| 2048 | 1.3217 | 1.1465 | 1.1128 | 1.2632 | 1.2861 | **1.2632** | 5/5 head>base |
| 4096 | 2.1312 | 1.8300 | 1.9015 | 1.9255 | 1.8540 | **1.9015** | 5/5 head>base |

## kernel_gpu_median_ms（参考。各 run 内の 20 反復中央値）

| N | mode | run1 | run2 | run3 | run4 | run5 |
|---|---|---|---|---|---|---|
| 1024 | base (production select) | 0.2246 | 1.0306 | 0.2249 | 0.2250 | 0.2246 |
| 1024 | head (MPP matmul2d) | 0.2296 | 0.9955 | 0.2319 | 0.2290 | 0.2324 |
| 2048 | base | 1.6004 | 1.9554 | 2.9469 | 1.6074 | 1.6076 |
| 2048 | head | 2.1153 | 2.2419 | 3.2794 | 2.0304 | 2.0675 |
| 4096 | base | 13.6056 | 13.6046 | 13.5967 | 13.5852 | 13.6067 |
| 4096 | head | 28.9961 | 24.8969 | 25.8540 | 26.1582 | 25.2264 |

run2/run3 の N=1024/2048 base 側に外れ値（他セッション負荷。`uptime_before_run*.txt` 参照）が見えるが、`head_over_base` の比率自体は 5 run とも同じ符号方向（N=2048/4096 は head 側が一貫して遅い）を保っている。

## 本番選択構成（`tile::select_for_device`。参考）

- N=1024 → `TileConfig { bm: 64, bn: 32, bk: 8, wm: 4, wn: 1, staged: true }`
- N=2048 → `TileConfig { bm: 64, bn: 32, bk: 16, wm: 2, wn: 2, staged: true }`
- N=4096 → `TileConfig { bm: 32, bn: 64, bk: 16, wm: 2, wn: 2, staged: true }`

## 正確性（REQ-2 複合判定）

- `mpp_matches_cpu_reference`（N=8/64/100・CPU 参照実装 `matmul_reference_fma` との比較）: 3/3 pass、いずれも `mean_abs_diff=0`（bit 完全一致）。
- `mpp_kernel_gpu_ab_vs_production_select`（本番選択構成との比較）: 5 run すべて trial 0 の複合判定 pass（`report.passes()` の `assert!` が全 run 通過）。

## 実行環境

- `env_info.txt` 参照（内部ホスト名は含めない）。
- `uptime_before_run*.txt`／`pmset_therm_before.txt`／`pmset_therm_after.txt` 参照。他セッションと共有の実機のため一部 run（特に run2/run3 の N=1024/2048）に負荷起因の外れ値が見られるが、比率の符号自体は変わらない。
