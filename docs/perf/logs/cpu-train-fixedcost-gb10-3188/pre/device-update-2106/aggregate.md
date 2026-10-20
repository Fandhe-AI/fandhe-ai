| arm | phase | median(us) | min-max(us) | q1/q3 median(us) |
|---|---|---|---|---|
| insitu_direct | device_update | 279.1 | 270.9-281.0 | 268.2/287.8 |
| insitu_pretouch | device_update | 323.9 | 317.1-325.1 | 313.6/343.8 |
| standalone | alloc | 0.9 | 0.8-1.0 | 0.8/1.2 |
| standalone | stage | 0.7 | 0.6-0.7 | 0.6/0.8 |
| standalone | sgd_kernel | 107.1 | 105.6-121.5 | 105.3/200.0 |
| standalone | sgd_compute_split | 32.4 | 31.3-32.7 | 26.9/34.7 |
| standalone | apply_params_split | 17.9 | 15.9-19.2 | 14.7/20.2 |
| standalone | sgd_kernel_zip | 22.3 | 22.0-22.5 | 17.6/23.8 |
| standalone | sgd_kernel_xthread | 108.9 | 106.6-188.2 | 105.4/218.1 |
| standalone | sgd_kernel_fixed | 0.1 | 0.1-0.1 | 0.1/0.2 |

checksum: run 間・腕間一致 OK
T = insitu_direct/device_update = 279.1 us（bench §17.6.2 の 277.5 us〈GB10〉は参考値）

| 項 | 値(us) | T 比 | 判定 |
|---|---|---|---|
| H2 cache 状態(direct-pretouch) | -44.8 | -0.16 | - |
| H1 ループ形((kernel-fixed)-zip) | 84.7 | 0.30 | - |
| H3 ホスト確保(alloc+stage) | 1.6 | 0.01 | - |
| H4 prologue+残差(pretouch-(kernel+alloc+stage)) | 215.2 | 0.77 | 支持 |
| 参考: kernel 全体(sgd_kernel) | 107.1 | 0.38 | 参考（判定対象外） |

帰属: H4 prologue+残差(pretouch-(kernel+alloc+stage))
補助: apply_params_split / (compute+apply)_split = 0.36
補助: sgd_kernel_xthread = 108.9 us（H2 の補助・別スレッド生成 Tensor 経由でありクロスコア書き込みは再現しない。判定に使わない）
