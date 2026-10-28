# 集計（#3646）: gb10-minflt（系列種別 minflt）

**判定対象外の系列: 数値記録のみ（RULE.txt）。帰属判定は行わない。**

checksum: 全 n・全腕・全 run で iter_total の checksum_bits が一致（hard 検査通過）。

## 判定表（1 run 中央値。単位 us）

### N=256

- fresh.iter_total = 232.5, resident_public.iter_total = 209.2, ratio = 0.900
- resident_split: 区間和 184.8 / iter_total 250.0（残差 65.2。記録のみ）

### N=512

- fresh.iter_total = 833.8, resident_public.iter_total = 862.7, ratio = 1.035
- resident_split: 区間和 736.0 / iter_total 820.9（残差 84.9。記録のみ）

### N=1024

- fresh.iter_total = 3341.4, resident_public.iter_total = 3468.0, ratio = 1.038
- resident_split: 区間和 3007.4 / iter_total 3232.8（残差 225.4。記録のみ）

### N=2048

- fresh.iter_total = 16183.4, resident_public.iter_total = 16848.8, ratio = 1.041
- resident_split: 区間和 16721.7 / iter_total 16915.0（残差 193.3。記録のみ）

### N=4096

- fresh.iter_total = 124755.7, resident_public.iter_total = 153095.3, ratio = 1.227
- resident_split: 区間和 152707.9 / iter_total 153143.1（残差 435.1。記録のみ）

## minflt / iter（補助・判定外）

| N | arm | phase | minflt/iter |
|---|---|---|---|
| 256 | fresh | tape_build | 0.0 |
| 256 | fresh | var_upload | 0.0 |
| 256 | fresh | matmul | 1.3 |
| 256 | fresh | readout | 0.0 |
| 256 | fresh | checksum | 0.0 |
| 256 | fresh | iter_total | 1.3 |
| 256 | fresh | tape_drop | 0.0 |
| 256 | resident_public | matmul | 0.8 |
| 256 | resident_public | readout | 0.0 |
| 256 | resident_public | checksum | 0.0 |
| 256 | resident_public | iter_total | 0.8 |
| 256 | resident_reupload | matmul | 0.8 |
| 256 | resident_reupload | readout | 0.0 |
| 256 | resident_reupload | checksum | 0.0 |
| 256 | resident_reupload | iter_total | 0.8 |
| 256 | resident_split | gemm_device | 0.5 |
| 256 | resident_split | download | 2.3 |
| 256 | resident_split | out_drop | 0.0 |
| 256 | resident_split | readout | 0.0 |
| 256 | resident_split | checksum | 0.0 |
| 256 | resident_split | iter_total | 2.8 |
| 256 | gemm_device_hold | iter_total | 0.0 |
| 256 | gemm_device_drop_in | iter_total | 0.0 |
| 256 | gemm_device_download | iter_total | 0.0 |
| 256 | backend_drop | kernel_call | 0.0 |
| 256 | backend_drop | iter_total | 0.0 |
| 512 | fresh | tape_build | 0.0 |
| 512 | fresh | var_upload | 0.0 |
| 512 | fresh | matmul | 0.0 |
| 512 | fresh | readout | 0.0 |
| 512 | fresh | checksum | 0.0 |
| 512 | fresh | iter_total | 0.0 |
| 512 | fresh | tape_drop | 0.0 |
| 512 | resident_public | matmul | 0.0 |
| 512 | resident_public | readout | 0.0 |
| 512 | resident_public | checksum | 0.0 |
| 512 | resident_public | iter_total | 0.0 |
| 512 | resident_reupload | matmul | 0.0 |
| 512 | resident_reupload | readout | 0.0 |
| 512 | resident_reupload | checksum | 0.0 |
| 512 | resident_reupload | iter_total | 0.0 |
| 512 | resident_split | gemm_device | 0.0 |
| 512 | resident_split | download | 0.0 |
| 512 | resident_split | out_drop | 0.0 |
| 512 | resident_split | readout | 0.0 |
| 512 | resident_split | checksum | 0.0 |
| 512 | resident_split | iter_total | 0.0 |
| 512 | gemm_device_hold | iter_total | 0.0 |
| 512 | gemm_device_drop_in | iter_total | 0.0 |
| 512 | gemm_device_download | iter_total | 0.0 |
| 512 | backend_drop | kernel_call | 0.0 |
| 512 | backend_drop | iter_total | 0.0 |
| 1024 | fresh | tape_build | 0.0 |
| 1024 | fresh | var_upload | 0.0 |
| 1024 | fresh | matmul | 0.0 |
| 1024 | fresh | readout | 0.0 |
| 1024 | fresh | checksum | 0.0 |
| 1024 | fresh | iter_total | 0.0 |
| 1024 | fresh | tape_drop | 0.0 |
| 1024 | resident_public | matmul | 0.0 |
| 1024 | resident_public | readout | 0.0 |
| 1024 | resident_public | checksum | 0.0 |
| 1024 | resident_public | iter_total | 0.0 |
| 1024 | resident_reupload | matmul | 0.0 |
| 1024 | resident_reupload | readout | 0.0 |
| 1024 | resident_reupload | checksum | 0.0 |
| 1024 | resident_reupload | iter_total | 0.0 |
| 1024 | resident_split | gemm_device | 0.0 |
| 1024 | resident_split | download | 0.0 |
| 1024 | resident_split | out_drop | 0.0 |
| 1024 | resident_split | readout | 0.0 |
| 1024 | resident_split | checksum | 0.0 |
| 1024 | resident_split | iter_total | 0.0 |
| 1024 | gemm_device_hold | iter_total | 0.0 |
| 1024 | gemm_device_drop_in | iter_total | 0.0 |
| 1024 | gemm_device_download | iter_total | 0.0 |
| 1024 | backend_drop | kernel_call | 0.0 |
| 1024 | backend_drop | iter_total | 0.0 |
| 2048 | fresh | tape_build | 0.0 |
| 2048 | fresh | var_upload | 0.0 |
| 2048 | fresh | matmul | 0.0 |
| 2048 | fresh | readout | 0.0 |
| 2048 | fresh | checksum | 0.0 |
| 2048 | fresh | iter_total | 0.0 |
| 2048 | fresh | tape_drop | 0.0 |
| 2048 | resident_public | matmul | 0.0 |
| 2048 | resident_public | readout | 0.0 |
| 2048 | resident_public | checksum | 0.0 |
| 2048 | resident_public | iter_total | 0.0 |
| 2048 | resident_reupload | matmul | 0.0 |
| 2048 | resident_reupload | readout | 0.0 |
| 2048 | resident_reupload | checksum | 0.0 |
| 2048 | resident_reupload | iter_total | 0.0 |
| 2048 | resident_split | gemm_device | 0.0 |
| 2048 | resident_split | download | 0.0 |
| 2048 | resident_split | out_drop | 0.0 |
| 2048 | resident_split | readout | 0.0 |
| 2048 | resident_split | checksum | 0.0 |
| 2048 | resident_split | iter_total | 0.0 |
| 2048 | gemm_device_hold | iter_total | 0.0 |
| 2048 | gemm_device_drop_in | iter_total | 0.0 |
| 2048 | gemm_device_download | iter_total | 0.0 |
| 2048 | backend_drop | kernel_call | 0.0 |
| 2048 | backend_drop | iter_total | 0.0 |
| 4096 | fresh | tape_build | 0.0 |
| 4096 | fresh | var_upload | 0.0 |
| 4096 | fresh | matmul | 38070.8 |
| 4096 | fresh | readout | 0.0 |
| 4096 | fresh | checksum | 0.0 |
| 4096 | fresh | iter_total | 38070.8 |
| 4096 | fresh | tape_drop | 0.0 |
| 4096 | resident_public | matmul | 53761.1 |
| 4096 | resident_public | readout | 0.0 |
| 4096 | resident_public | checksum | 0.0 |
| 4096 | resident_public | iter_total | 53761.1 |
| 4096 | resident_reupload | matmul | 54724.6 |
| 4096 | resident_reupload | readout | 0.0 |
| 4096 | resident_reupload | checksum | 0.0 |
| 4096 | resident_reupload | iter_total | 54724.6 |
| 4096 | resident_split | gemm_device | 37948.2 |
| 4096 | resident_split | download | 16385.0 |
| 4096 | resident_split | out_drop | 0.0 |
| 4096 | resident_split | readout | 0.0 |
| 4096 | resident_split | checksum | 0.0 |
| 4096 | resident_split | iter_total | 54333.2 |
| 4096 | gemm_device_hold | iter_total | 38001.4 |
| 4096 | gemm_device_drop_in | iter_total | 37619.7 |
| 4096 | gemm_device_download | iter_total | 54505.5 |
| 4096 | backend_drop | kernel_call | 37535.0 |
| 4096 | backend_drop | iter_total | 37535.0 |

## 区間表（1 run 中央値 [min–max]、us）

### N=256

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.0 [0.0–0.0] |
| fresh | var_upload | 0.2 [0.2–0.2] |
| fresh | matmul | 142.0 [142.0–142.0] |
| fresh | readout | 0.1 [0.1–0.1] |
| fresh | checksum | 34.0 [34.0–34.0] |
| fresh | iter_total | 232.5 [232.5–232.5] |
| fresh | tape_drop | 0.6 [0.6–0.6] |
| resident_public | matmul | 138.7 [138.7–138.7] |
| resident_public | readout | 0.1 [0.1–0.1] |
| resident_public | checksum | 33.6 [33.6–33.6] |
| resident_public | iter_total | 209.2 [209.2–209.2] |
| resident_reupload | matmul | 172.3 [172.3–172.3] |
| resident_reupload | readout | 0.2 [0.2–0.2] |
| resident_reupload | checksum | 33.6 [33.6–33.6] |
| resident_reupload | iter_total | 246.9 [246.9–246.9] |
| resident_split | gemm_device | 144.4 [144.4–144.4] |
| resident_split | download | 6.4 [6.4–6.4] |
| resident_split | out_drop | 0.3 [0.3–0.3] |
| resident_split | readout | 0.2 [0.2–0.2] |
| resident_split | checksum | 33.6 [33.6–33.6] |
| resident_split | iter_total | 250.0 [250.0–250.0] |
| gemm_device_hold | iter_total | 132.4 [132.4–132.4] |
| gemm_device_drop_in | iter_total | 131.3 [131.3–131.3] |
| gemm_device_download | iter_total | 137.2 [137.2–137.2] |
| backend_drop | kernel_call | 141.9 [141.9–141.9] |
| backend_drop | iter_total | 159.2 [159.2–159.2] |

### N=512

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.1 [0.1–0.1] |
| fresh | var_upload | 0.4 [0.4–0.4] |
| fresh | matmul | 614.7 [614.7–614.7] |
| fresh | readout | 0.2 [0.2–0.2] |
| fresh | checksum | 134.9 [134.9–134.9] |
| fresh | iter_total | 833.8 [833.8–833.8] |
| fresh | tape_drop | 0.8 [0.8–0.8] |
| resident_public | matmul | 648.9 [648.9–648.9] |
| resident_public | readout | 0.2 [0.2–0.2] |
| resident_public | checksum | 136.3 [136.3–136.3] |
| resident_public | iter_total | 862.7 [862.7–862.7] |
| resident_reupload | matmul | 699.4 [699.4–699.4] |
| resident_reupload | readout | 0.3 [0.3–0.3] |
| resident_reupload | checksum | 139.3 [139.3–139.3] |
| resident_reupload | iter_total | 945.1 [945.1–945.1] |
| resident_split | gemm_device | 573.3 [573.3–573.3] |
| resident_split | download | 27.4 [27.4–27.4] |
| resident_split | out_drop | 0.3 [0.3–0.3] |
| resident_split | readout | 0.2 [0.2–0.2] |
| resident_split | checksum | 134.8 [134.8–134.8] |
| resident_split | iter_total | 820.9 [820.9–820.9] |
| gemm_device_hold | iter_total | 513.9 [513.9–513.9] |
| gemm_device_drop_in | iter_total | 504.8 [504.8–504.8] |
| gemm_device_download | iter_total | 660.5 [660.5–660.5] |
| backend_drop | kernel_call | 576.1 [576.1–576.1] |
| backend_drop | iter_total | 605.6 [605.6–605.6] |

### N=1024

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.1 [0.1–0.1] |
| fresh | var_upload | 0.6 [0.6–0.6] |
| fresh | matmul | 2536.9 [2536.9–2536.9] |
| fresh | readout | 0.3 [0.3–0.3] |
| fresh | checksum | 748.0 [748.0–748.0] |
| fresh | iter_total | 3341.4 [3341.4–3341.4] |
| fresh | tape_drop | 1.3 [1.3–1.3] |
| resident_public | matmul | 2686.9 [2686.9–2686.9] |
| resident_public | readout | 0.5 [0.5–0.5] |
| resident_public | checksum | 554.0 [554.0–554.0] |
| resident_public | iter_total | 3468.0 [3468.0–3468.0] |
| resident_reupload | matmul | 2609.2 [2609.2–2609.2] |
| resident_reupload | readout | 0.6 [0.6–0.6] |
| resident_reupload | checksum | 755.6 [755.6–755.6] |
| resident_reupload | iter_total | 3464.1 [3464.1–3464.1] |
| resident_split | gemm_device | 2322.5 [2322.5–2322.5] |
| resident_split | download | 129.2 [129.2–129.2] |
| resident_split | out_drop | 0.5 [0.5–0.5] |
| resident_split | readout | 0.5 [0.5–0.5] |
| resident_split | checksum | 554.7 [554.7–554.7] |
| resident_split | iter_total | 3232.8 [3232.8–3232.8] |
| gemm_device_hold | iter_total | 2094.9 [2094.9–2094.9] |
| gemm_device_drop_in | iter_total | 2043.6 [2043.6–2043.6] |
| gemm_device_download | iter_total | 2258.8 [2258.8–2258.8] |
| backend_drop | kernel_call | 2317.9 [2317.9–2317.9] |
| backend_drop | iter_total | 2386.0 [2386.0–2386.0] |

### N=2048

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.3 [0.3–0.3] |
| fresh | var_upload | 1.4 [1.4–1.4] |
| fresh | matmul | 12964.8 [12964.8–12964.8] |
| fresh | readout | 1.0 [1.0–1.0] |
| fresh | checksum | 3041.5 [3041.5–3041.5] |
| fresh | iter_total | 16183.4 [16183.4–16183.4] |
| fresh | tape_drop | 3.2 [3.2–3.2] |
| resident_public | matmul | 13741.8 [13741.8–13741.8] |
| resident_public | readout | 0.7 [0.7–0.7] |
| resident_public | checksum | 3009.2 [3009.2–3009.2] |
| resident_public | iter_total | 16848.8 [16848.8–16848.8] |
| resident_reupload | matmul | 13815.1 [13815.1–13815.1] |
| resident_reupload | readout | 0.7 [0.7–0.7] |
| resident_reupload | checksum | 3008.5 [3008.5–3008.5] |
| resident_reupload | iter_total | 16990.7 [16990.7–16990.7] |
| resident_split | gemm_device | 13112.1 [13112.1–13112.1] |
| resident_split | download | 598.6 [598.6–598.6] |
| resident_split | out_drop | 0.9 [0.9–0.9] |
| resident_split | readout | 0.7 [0.7–0.7] |
| resident_split | checksum | 3009.3 [3009.3–3009.3] |
| resident_split | iter_total | 16915.0 [16915.0–16915.0] |
| gemm_device_hold | iter_total | 12874.9 [12874.9–12874.9] |
| gemm_device_drop_in | iter_total | 12835.3 [12835.3–12835.3] |
| gemm_device_download | iter_total | 13460.6 [13460.6–13460.6] |
| backend_drop | kernel_call | 12940.7 [12940.7–12940.7] |
| backend_drop | iter_total | 13047.8 [13047.8–13047.8] |

### N=4096

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.3 [0.3–0.3] |
| fresh | var_upload | 2.2 [2.2–2.2] |
| fresh | matmul | 112849.9 [112849.9–112849.9] |
| fresh | readout | 0.9 [0.9–0.9] |
| fresh | checksum | 12124.0 [12124.0–12124.0] |
| fresh | iter_total | 124755.7 [124755.7–124755.7] |
| fresh | tape_drop | 8451.9 [8451.9–8451.9] |
| resident_public | matmul | 144099.1 [144099.1–144099.1] |
| resident_public | readout | 0.8 [0.8–0.8] |
| resident_public | checksum | 8793.9 [8793.9–8793.9] |
| resident_public | iter_total | 153095.3 [153095.3–153095.3] |
| resident_reupload | matmul | 143547.3 [143547.3–143547.3] |
| resident_reupload | readout | 0.8 [0.8–0.8] |
| resident_reupload | checksum | 8784.0 [8784.0–8784.0] |
| resident_reupload | iter_total | 152386.2 [152386.2–152386.2] |
| resident_split | gemm_device | 113320.0 [113320.0–113320.0] |
| resident_split | download | 21269.6 [21269.6–21269.6] |
| resident_split | out_drop | 9316.3 [9316.3–9316.3] |
| resident_split | readout | 0.8 [0.8–0.8] |
| resident_split | checksum | 8801.4 [8801.4–8801.4] |
| resident_split | iter_total | 153143.1 [153143.1–153143.1] |
| gemm_device_hold | iter_total | 113181.1 [113181.1–113181.1] |
| gemm_device_drop_in | iter_total | 123021.6 [123021.6–123021.6] |
| gemm_device_download | iter_total | 133623.9 [133623.9–133623.9] |
| backend_drop | kernel_call | 115311.0 [115311.0–115311.0] |
| backend_drop | iter_total | 115445.4 [115445.4–115445.4] |

## run 別 iter_total（fresh / resident_public、us）

| N | arm | run1 |
|---|---|---|
| 256 | fresh | 232.5 |
| 256 | resident_public | 209.2 |
| 512 | fresh | 833.8 |
| 512 | resident_public | 862.7 |
| 1024 | fresh | 3341.4 |
| 1024 | resident_public | 3468.0 |
| 2048 | fresh | 16183.4 |
| 2048 | resident_public | 16848.8 |
| 4096 | fresh | 124755.7 |
| 4096 | resident_public | 153095.3 |

