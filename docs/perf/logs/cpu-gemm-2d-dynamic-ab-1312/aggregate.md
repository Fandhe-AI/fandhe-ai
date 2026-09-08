## DGX Spark GB10

### threads=8 size=1024

| candidate | jobs_per_worker | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |
|---|---|---|---|---|
| RowPanel | 0 | 722.691 | 1.0000 | — |
| TwoDDynamic | 2 | 711.040 | 0.9839 | 1/5 |
| TwoDDynamic | 4 | 675.314 | 0.9344 | 0/5 |

### threads=8 size=2048

| candidate | jobs_per_worker | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |
|---|---|---|---|---|
| RowPanel | 0 | 871.027 | 1.0000 | — |
| TwoDDynamic | 2 | 728.004 | 0.8358 | 0/5 |
| TwoDDynamic | 4 | 758.153 | 0.8704 | 0/5 |

### threads=8 size=4096

| candidate | jobs_per_worker | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |
|---|---|---|---|---|
| RowPanel | 0 | 926.764 | 1.0000 | — |
| TwoDDynamic | 2 | 709.736 | 0.7658 | 0/5 |
| TwoDDynamic | 4 | 846.348 | 0.9132 | 0/5 |

### threads=10 size=1024

| candidate | jobs_per_worker | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |
|---|---|---|---|---|
| RowPanel | 0 | 346.559 | 1.0000 | — |
| TwoDDynamic | 2 | 610.148 | 1.7606 | 4/5 |
| TwoDDynamic | 4 | 583.863 | 1.6847 | 3/5 |

### threads=10 size=2048

| candidate | jobs_per_worker | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |
|---|---|---|---|---|
| RowPanel | 0 | 1025.662 | 1.0000 | — |
| TwoDDynamic | 2 | 1022.253 | 0.9967 | 3/5 |
| TwoDDynamic | 4 | 1002.834 | 0.9777 | 0/5 |

### threads=10 size=4096

| candidate | jobs_per_worker | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |
|---|---|---|---|---|
| RowPanel | 0 | 1108.308 | 1.0000 | — |
| TwoDDynamic | 2 | 1029.329 | 0.9287 | 0/5 |
| TwoDDynamic | 4 | 1030.627 | 0.9299 | 0/5 |

### threads=20 size=1024

| candidate | jobs_per_worker | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |
|---|---|---|---|---|
| RowPanel | 0 | 543.901 | 1.0000 | — |
| TwoDDynamic | 2 | 713.834 | 1.3124 | 5/5 |
| TwoDDynamic | 4 | 693.188 | 1.2745 | 5/5 |

### threads=20 size=2048

| candidate | jobs_per_worker | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |
|---|---|---|---|---|
| RowPanel | 0 | 712.477 | 1.0000 | — |
| TwoDDynamic | 2 | 1285.474 | 1.8042 | 5/5 |
| TwoDDynamic | 4 | 1172.275 | 1.6454 | 5/5 |

### threads=20 size=4096

| candidate | jobs_per_worker | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |
|---|---|---|---|---|
| RowPanel | 0 | 1112.779 | 1.0000 | — |
| TwoDDynamic | 2 | 1214.659 | 1.0916 | 5/5 |
| TwoDDynamic | 4 | 1300.746 | 1.1689 | 5/5 |

### 非単調性クロスチェック（size=1024。AC-2）

| candidate | jobs_per_worker | T=10 median | T=8 median | T10/T8 比 |
|---|---|---|---|---|
| RowPanel | 0 | 346.559 | 722.691 | 0.4795 |
| TwoDDynamic | 2 | 610.148 | 711.040 | 0.8581 |
| TwoDDynamic | 4 | 583.863 | 675.314 | 0.8646 |

## Apple M4 Max

### threads=16 size=1024

| candidate | jobs_per_worker | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |
|---|---|---|---|---|
| RowPanel | 0 | 393.673 | 1.0000 | — |
| TwoDDynamic | 2 | 467.768 | 1.1882 | 5/5 |
| TwoDDynamic | 4 | 448.046 | 1.1381 | 4/5 |

### threads=16 size=2048

| candidate | jobs_per_worker | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |
|---|---|---|---|---|
| RowPanel | 0 | 430.647 | 1.0000 | — |
| TwoDDynamic | 2 | 510.529 | 1.1855 | 5/5 |
| TwoDDynamic | 4 | 500.705 | 1.1627 | 5/5 |

### threads=16 size=4096

| candidate | jobs_per_worker | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |
|---|---|---|---|---|
| RowPanel | 0 | 408.255 | 1.0000 | — |
| TwoDDynamic | 2 | 396.482 | 0.9712 | 3/5 |
| TwoDDynamic | 4 | 422.861 | 1.0358 | 5/5 |

