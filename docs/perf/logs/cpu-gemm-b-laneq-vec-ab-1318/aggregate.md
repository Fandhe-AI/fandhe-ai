## Apple M4 Max

### size=1024

| variant | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |
|---|---|---|---|
| RowPanel | 799.250 | 1.0000 | — |
| SharedB | 495.611 | 0.6201 | 0/5 |
| SharedBPcOuter | 478.104 | 0.5982 | 0/5 |
| IcDynamic | 663.409 | 0.8300 | 0/5 |
| RowPanelBLaneqVec | 782.563 | 0.9791 | 0/5 |

### size=2048

| variant | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |
|---|---|---|---|
| RowPanel | 914.455 | 1.0000 | — |
| SharedB | 661.491 | 0.7234 | 0/5 |
| SharedBPcOuter | 662.712 | 0.7247 | 0/5 |
| IcDynamic | 890.241 | 0.9735 | 0/5 |
| RowPanelBLaneqVec | 910.700 | 0.9959 | 2/5 |

### size=4096

| variant | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |
|---|---|---|---|
| RowPanel | 1034.145 | 1.0000 | — |
| SharedB | 756.054 | 0.7311 | 0/5 |
| SharedBPcOuter | 756.536 | 0.7316 | 0/5 |
| IcDynamic | 1058.217 | 1.0233 | 4/5 |
| RowPanelBLaneqVec | 1047.375 | 1.0128 | 4/5 |

## DGX Spark GB10

### size=1024

| variant | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |
|---|---|---|---|
| RowPanel | 535.388 | 1.0000 | — |
| SharedB | 229.149 | 0.4280 | 0/5 |
| SharedBPcOuter | 252.295 | 0.4712 | 0/5 |
| IcDynamic | 316.669 | 0.5915 | 0/5 |
| RowPanelBLaneqVec | 515.485 | 0.9628 | 0/5 |

### size=2048

| variant | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |
|---|---|---|---|
| RowPanel | 701.226 | 1.0000 | — |
| SharedB | 375.099 | 0.5349 | 0/5 |
| SharedBPcOuter | 387.736 | 0.5529 | 0/5 |
| IcDynamic | 594.161 | 0.8473 | 0/5 |
| RowPanelBLaneqVec | 697.408 | 0.9946 | 2/5 |

### size=4096

| variant | 5 run median GFLOP/s | 対 RowPanel 比 | RowPanel に勝った run 数 |
|---|---|---|---|
| RowPanel | 1098.415 | 1.0000 | — |
| SharedB | 458.572 | 0.4175 | 0/5 |
| SharedBPcOuter | 463.828 | 0.4223 | 0/5 |
| IcDynamic | 1093.053 | 0.9951 | 0/5 |
| RowPanelBLaneqVec | 1256.580 | 1.1440 | 5/5 |

