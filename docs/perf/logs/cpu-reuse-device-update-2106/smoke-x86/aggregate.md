| arm | phase | median(us) | min-max(us) | q1/q3 median(us) |
|---|---|---|---|---|
| insitu_direct | device_update | 242.3 | 236.8-519.5 | 149.6/530.5 |
| insitu_pretouch | device_update | 249.3 | 242.1-532.6 | 148.8/531.0 |
| standalone | alloc | 0.4 | 0.4-0.5 | 0.4/0.5 |
| standalone | stage | 0.6 | 0.5-0.6 | 0.4/0.7 |
| standalone | sgd_kernel | 171.4 | 168.0-212.4 | 111.8/216.3 |
| standalone | sgd_compute_split | 73.9 | 61.6-90.1 | 59.6/87.2 |
| standalone | apply_params_split | 39.3 | 33.4-51.9 | 35.2/48.7 |
| standalone | sgd_kernel_zip | 29.4 | 28.4-30.1 | 27.5/32.6 |
| standalone | sgd_kernel_xthread | 173.2 | 162.7-212.4 | 113.5/214.7 |
| standalone | sgd_kernel_fixed | 0.2 | 0.1-0.2 | 0.1/0.2 |

checksum: run 間・腕間一致 OK
T = insitu_direct/device_update = 242.3 us（bench §17.6.2 の 277.5 us〈GB10〉は参考値）

| 項 | 値(us) | T 比 | 判定 |
|---|---|---|---|
| H2 cache 状態(direct-pretouch) | -6.9 | -0.03 | - |
| H1 ループ形((kernel-fixed)-zip) | 141.9 | 0.59 | 支持 |
| H3 ホスト確保(alloc+stage) | 1.0 | 0.00 | - |
| kernel 全体(sgd_kernel) | 171.4 | 0.71 | 支持 |
| H4 prologue+残差(pretouch-(kernel+alloc+stage)) | 76.8 | 0.32 | - |

帰属: H1 ループ形((kernel-fixed)-zip), kernel 全体(sgd_kernel)
補助: apply_params_split / (compute+apply)_split = 0.35
補助: sgd_kernel_xthread = 173.2 us（H2 の補助・別スレッド生成 Tensor 経由でありクロスコア書き込みは再現しない。判定に使わない）
