## Apple M4 Max

### threads=16 size=1024

| candidate | jobs_per_worker | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |
|---|---|---|---|---|
| RowPanel | 0 | 763.584 | 1.0000 | — |
| TwoDDynamic | 2 | 941.294 | 1.2327 | 5/5 |
| TwoDDynamic | 4 | 936.233 | 1.2261 | 5/5 |

### threads=16 size=2048

| candidate | jobs_per_worker | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |
|---|---|---|---|---|
| RowPanel | 0 | 848.848 | 1.0000 | — |
| TwoDDynamic | 2 | 1110.968 | 1.3088 | 5/5 |
| TwoDDynamic | 4 | 1113.666 | 1.3120 | 5/5 |

### threads=16 size=4096

| candidate | jobs_per_worker | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |
|---|---|---|---|---|
| RowPanel | 0 | 754.372 | 1.0000 | — |
| TwoDDynamic | 2 | 774.048 | 1.0261 | 4/5 |
| TwoDDynamic | 4 | 823.574 | 1.0917 | 5/5 |

