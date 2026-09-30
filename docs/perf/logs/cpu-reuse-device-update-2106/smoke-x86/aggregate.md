| arm | phase | median(us) | min-max(us) | q1/q3 median(us) |
|---|---|---|---|---|
| insitu_direct | device_update | 261.2 | 248.6-275.9 | 174.1/536.7 |
| insitu_pretouch | device_update | 275.9 | 256.9-307.0 | 245.6/541.8 |
| standalone | alloc | 0.4 | 0.4-0.5 | 0.4/0.5 |
| standalone | stage | 0.6 | 0.6-0.7 | 0.5/0.7 |
| standalone | sgd_kernel | 191.6 | 176.8-211.1 | 151.5/213.9 |
| standalone | sgd_compute_split | 67.9 | 57.6-70.8 | 57.2/75.4 |
| standalone | apply_params_split | 35.0 | 31.1-36.1 | 25.9/40.0 |
| standalone | sgd_kernel_zip | 28.9 | 28.3-29.0 | 27.3/31.0 |
| standalone | sgd_kernel_xthread | 187.7 | 180.1-208.5 | 151.5/212.7 |

checksum: run 間・腕間一致 OK
T = insitu_direct/device_update = 261.2 us（bench §17.6.2 の 277.5 us〈GB10〉は参考値）

| 項 | 値(us) | T 比 | 判定 |
|---|---|---|---|
| H2 cache 状態(direct-pretouch) | -14.7 | -0.06 | - |
| H1 ループ形(kernel-zip) | 162.6 | 0.62 | 支持 |
| H3 ホスト確保(alloc+stage) | 1.0 | 0.00 | - |
| kernel 全体(sgd_kernel) | 191.6 | 0.73 | 支持 |
| H4 prologue+残差(pretouch-(kernel+alloc+stage)) | 83.4 | 0.32 | - |

帰属: H1 ループ形(kernel-zip), kernel 全体(sgd_kernel)
補助: apply_params_split / (compute+apply)_split = 0.34
補助: sgd_kernel_xthread = 187.7 us（H2 の補助。判定に使わない）
