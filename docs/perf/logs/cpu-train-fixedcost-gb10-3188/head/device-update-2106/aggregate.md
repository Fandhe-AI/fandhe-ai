| arm | phase | median(us) | min-max(us) | q1/q3 median(us) |
|---|---|---|---|---|
| insitu_direct | device_update | 169.2 | 165.5-172.6 | 154.5/176.0 |
| insitu_pretouch | device_update | 172.2 | 168.1-174.9 | 140.0/178.0 |
| standalone | alloc | 0.9 | 0.9-1.0 | 0.8/1.4 |
| standalone | stage | 0.7 | 0.7-0.7 | 0.6/0.8 |
| standalone | sgd_kernel | 27.7 | 23.9-31.7 | 22.7/36.5 |
| standalone | sgd_compute_split | 32.5 | 32.0-33.3 | 25.3/36.4 |
| standalone | apply_params_split | 17.0 | 16.1-19.9 | 14.7/20.4 |
| standalone | sgd_kernel_zip | 22.6 | 22.0-22.9 | 17.8/31.1 |
| standalone | sgd_kernel_xthread | 20.8 | 19.5-31.1 | 16.6/37.1 |
| standalone | sgd_kernel_fixed | 0.1 | 0.1-0.1 | 0.1/0.2 |

checksum: run 間・腕間一致 OK
T = insitu_direct/device_update = 169.2 us（bench §17.6.2 の 277.5 us〈GB10〉は参考値）

| 項 | 値(us) | T 比 | 判定 |
|---|---|---|---|
| H2 cache 状態(direct-pretouch) | -3.0 | -0.02 | - |
| H1 ループ形((kernel-fixed)-zip) | 5.0 | 0.03 | - |
| H3 ホスト確保(alloc+stage) | 1.6 | 0.01 | - |
| H4 prologue+残差(pretouch-(kernel+alloc+stage)) | 142.8 | 0.84 | 支持 |
| 参考: kernel 全体(sgd_kernel) | 27.7 | 0.16 | 参考（判定対象外） |

帰属: H4 prologue+残差(pretouch-(kernel+alloc+stage))
補助: apply_params_split / (compute+apply)_split = 0.34
補助: sgd_kernel_xthread = 20.8 us（H2 の補助・別スレッド生成 Tensor 経由でありクロスコア書き込みは再現しない。判定に使わない）
