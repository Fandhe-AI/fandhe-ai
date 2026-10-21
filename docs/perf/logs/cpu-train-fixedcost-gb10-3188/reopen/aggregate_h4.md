## 系列 multi

| arm | phase | median(us) | min-max(us) | q1/q3 median(us) | run1..5 median(us) |
|---|---|---|---|---|---|
| insitu_direct | device_update | 163.2 | 158.9-164.4 | 148.4/170.1 | 163.2 / 161.8 / 164.4 / 163.6 / 158.9 |
| insitu_probe | prologue_probe | 2.3 | 1.9-2.6 | 2.0/2.6 | 1.9 / 2.1 / 2.6 / 2.4 / 2.3 |
| insitu_probe | after_probe | 157.4 | 153.7-158.1 | 139.7/165.3 | 157.0 / 158.0 / 157.4 / 158.1 / 153.7 |
| standalone | stage | 1.0 | 0.8-1.0 | 0.8/1.1 | 0.9 / 0.8 / 1.0 / 1.0 / 1.0 |
| standalone | sgd_kernel | 34.2 | 33.6-34.8 | 32.2/37.4 | 34.2 / 33.6 / 34.8 / 34.2 / 33.6 |
| standalone | sgd_kernel_cold | 49.9 | 48.7-50.8 | 46.0/56.5 | 48.7 / 49.0 / 50.8 / 49.9 / 49.9 |
| standalone | zip_hot | 33.8 | 33.3-35.7 | 31.6/60.3 | 33.8 / 33.3 / 35.7 / 34.6 / 33.3 |
| standalone | zip_xcore | 41.7 | 40.6-42.8 | 37.9/52.7 | 40.6 / 40.8 / 42.8 / 42.8 / 41.7 |

checksum: run 間・腕間一致 OK

補助（判定に使わない）: 計時区間中の minflt 合計と main スレッドのコア移動回数（5 run 合計）

| セル | minflt | core_moves |
|---|---|---|
| insitu_direct/device_update | 0 | 0 |
| insitu_probe/prologue_probe | 0 | 0 |
| insitu_probe/after_probe | 0 | 0 |
| standalone/stage | 0 | 0 |
| standalone/sgd_kernel | 0 | 0 |
| standalone/sgd_kernel_cold | 0 | 0 |
| standalone/zip_hot | 0 | 0 |
| standalone/zip_xcore | 0 | 0 |

### 判定（multi）
規則 1 加算性: |P+R-T| = 3.5 us（許容 24.5 us）-> 成立
規則 2 H4' = T-(K+S) = 128.0 us（T の 78.5%）
規則 3 H4a(prologue)=P = 2.3 us（H4' の 1.8%）-> 未確定
        H4b(in-situ カーネル超過)=R-(K+S) = 122.2 us（H4' の 95.5%）-> 支持
規則 4 候補 cold(追い出し後) = 49.9 us（x=R-S=156.4 us との差 -68.1%）
規則 4 候補 xcore(他コア書き込み後) = 42.1 us（x=R-S=156.4 us との差 -73.1%）
        -> 未確定（±25% 以内の候補なし）

## 系列 single

| arm | phase | median(us) | min-max(us) | q1/q3 median(us) | run1..5 median(us) |
|---|---|---|---|---|---|
| insitu_direct | device_update | 26.2 | 21.5-27.1 | 24.3/27.0 | 22.1 / 26.2 / 27.1 / 26.6 / 21.5 |
| insitu_probe | prologue_probe | 2.1 | 1.6-2.6 | 1.8/2.8 | 1.6 / 2.6 / 2.1 / 2.3 / 1.6 |
| insitu_probe | after_probe | 25.4 | 20.8-26.1 | 23.7/26.7 | 23.8 / 25.7 / 26.1 / 25.4 / 20.8 |
| standalone | stage | 0.9 | 0.8-1.0 | 0.8/1.1 | 0.9 / 1.0 / 0.9 / 0.9 / 0.8 |
| standalone | sgd_kernel | 33.0 | 30.1-33.7 | 32.7/33.5 | 32.1 / 33.0 / 33.3 / 33.7 / 30.1 |
| standalone | sgd_kernel_cold | 49.3 | 43.7-51.8 | 48.9/49.7 | 48.8 / 49.3 / 49.5 / 51.8 / 43.7 |
| standalone | zip_hot | 33.2 | 31.2-34.2 | 32.9/33.9 | 32.8 / 33.2 / 33.6 / 34.2 / 31.2 |
| standalone | zip_xcore | 40.7 | 37.2-44.8 | 38.6/42.7 | 38.7 / 42.7 / 44.8 / 40.7 / 37.2 |

checksum: run 間・腕間一致 OK

補助（判定に使わない）: 計時区間中の minflt 合計と main スレッドのコア移動回数（5 run 合計）

| セル | minflt | core_moves |
|---|---|---|
| insitu_direct/device_update | 0 | 0 |
| insitu_probe/prologue_probe | 0 | 0 |
| insitu_probe/after_probe | 0 | 0 |
| standalone/stage | 0 | 0 |
| standalone/sgd_kernel | 0 | 0 |
| standalone/sgd_kernel_cold | 0 | 0 |
| standalone/zip_hot | 0 | 0 |
| standalone/zip_xcore | 0 | 0 |

### 判定（single）
規則 1 加算性: |P+R-T| = 1.3 us（許容 3.9 us）-> 成立
規則 2 H4' = T-(K+S) = -7.8 us（T の -29.8%）
規則 3 H4' が非正のため判定不能 -> 未確定
規則 4 H4b が支持されないため状態の説明は付けない

## 規則 5 書き手コアの判定
(T_single-K_single) = -6.8 us、(T_multi-K_multi) = 129.0 us、比 -0.05
-> マルチコア由来の状態を支持
