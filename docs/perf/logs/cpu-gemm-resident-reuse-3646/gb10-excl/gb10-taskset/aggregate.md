# 集計（#3646）: gb10-taskset（系列種別 taskset）

**判定対象外の系列: 数値記録のみ（RULE.txt）。帰属判定は行わない。**

checksum: 全 n・全腕・全 run で iter_total の checksum_bits が一致（hard 検査通過）。

## 判定表（5 run 中央値。単位 us）

### N=256

- fresh.iter_total = 138.0, resident_public.iter_total = 133.6, ratio = 0.968
- resident_split: 区間和 130.4 / iter_total 130.0（残差 -0.4。記録のみ）

### N=512

- fresh.iter_total = 1008.4, resident_public.iter_total = 1047.6, ratio = 1.039
- resident_split: 区間和 1057.5 / iter_total 1057.8（残差 0.3。記録のみ）

### N=1024

- fresh.iter_total = 3373.4, resident_public.iter_total = 3627.1, ratio = 1.075
- resident_split: 区間和 3632.6 / iter_total 3638.3（残差 5.7。記録のみ）

### N=2048

- fresh.iter_total = 18793.0, resident_public.iter_total = 19306.0, ratio = 1.027
- resident_split: 区間和 19244.8 / iter_total 19258.1（残差 13.3。記録のみ）

### N=4096

- fresh.iter_total = 151652.2, resident_public.iter_total = 180140.6, ratio = 1.188
- resident_split: 区間和 180085.7 / iter_total 179641.0（残差 -444.7。記録のみ）

## 区間表（5 run 中央値 [min–max]、us）

### N=256

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.0 [0.0–0.0] |
| fresh | var_upload | 0.2 [0.2–0.2] |
| fresh | matmul | 102.5 [99.7–105.1] |
| fresh | readout | 0.1 [0.1–0.1] |
| fresh | checksum | 33.9 [33.8–33.9] |
| fresh | iter_total | 138.0 [135.0–140.8] |
| fresh | tape_drop | 0.4 [0.4–0.4] |
| resident_public | matmul | 99.4 [94.0–102.6] |
| resident_public | readout | 0.1 [0.1–0.1] |
| resident_public | checksum | 33.6 [33.6–33.6] |
| resident_public | iter_total | 133.6 [128.0–137.3] |
| resident_reupload | matmul | 116.3 [112.8–123.9] |
| resident_reupload | readout | 0.1 [0.1–0.1] |
| resident_reupload | checksum | 33.6 [33.6–33.6] |
| resident_reupload | iter_total | 150.3 [146.9–157.8] |
| resident_split | gemm_device | 90.7 [88.8–94.7] |
| resident_split | download | 5.8 [5.0–6.3] |
| resident_split | out_drop | 0.2 [0.2–0.2] |
| resident_split | readout | 0.1 [0.1–0.1] |
| resident_split | checksum | 33.6 [33.6–33.6] |
| resident_split | iter_total | 130.0 [129.5–135.0] |
| gemm_device_hold | iter_total | 87.6 [85.7–87.8] |
| gemm_device_drop_in | iter_total | 86.3 [85.4–87.1] |
| gemm_device_download | iter_total | 101.2 [99.5–102.1] |
| backend_drop | kernel_call | 99.4 [97.4–118.1] |
| backend_drop | iter_total | 99.4 [97.5–118.3] |

### N=512

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.1 [0.1–0.1] |
| fresh | var_upload | 0.5 [0.5–0.5] |
| fresh | matmul | 872.6 [835.3–910.4] |
| fresh | readout | 0.2 [0.2–0.2] |
| fresh | checksum | 134.8 [134.8–134.8] |
| fresh | iter_total | 1008.4 [976.7–1046.4] |
| fresh | tape_drop | 0.8 [0.8–1.0] |
| resident_public | matmul | 911.2 [750.2–937.8] |
| resident_public | readout | 0.3 [0.3–0.3] |
| resident_public | checksum | 134.6 [134.6–134.7] |
| resident_public | iter_total | 1047.6 [885.4–1073.9] |
| resident_reupload | matmul | 923.9 [786.9–992.5] |
| resident_reupload | readout | 0.3 [0.3–0.3] |
| resident_reupload | checksum | 134.5 [134.5–134.6] |
| resident_reupload | iter_total | 1060.1 [923.2–1128.3] |
| resident_split | gemm_device | 896.7 [811.2–924.7] |
| resident_split | download | 25.7 [25.2–26.2] |
| resident_split | out_drop | 0.2 [0.2–0.3] |
| resident_split | readout | 0.3 [0.3–0.3] |
| resident_split | checksum | 134.6 [134.5–134.7] |
| resident_split | iter_total | 1057.8 [970.3–1093.2] |
| gemm_device_hold | iter_total | 409.3 [337.4–611.1] |
| gemm_device_drop_in | iter_total | 417.1 [337.8–533.4] |
| gemm_device_download | iter_total | 672.7 [450.9–794.9] |
| backend_drop | kernel_call | 843.5 [662.0–892.7] |
| backend_drop | iter_total | 843.7 [662.9–893.2] |

### N=1024

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.1 [0.0–0.1] |
| fresh | var_upload | 0.8 [0.5–0.9] |
| fresh | matmul | 2832.0 [2613.3–2855.7] |
| fresh | readout | 0.2 [0.2–0.3] |
| fresh | checksum | 543.4 [539.6–544.3] |
| fresh | iter_total | 3373.4 [3155.8–3400.5] |
| fresh | tape_drop | 1.2 [0.8–1.5] |
| resident_public | matmul | 3084.9 [3022.6–3155.0] |
| resident_public | readout | 0.4 [0.3–0.4] |
| resident_public | checksum | 542.7 [542.4–546.0] |
| resident_public | iter_total | 3627.1 [3564.6–3698.4] |
| resident_reupload | matmul | 3094.4 [3060.7–3149.0] |
| resident_reupload | readout | 0.4 [0.3–0.5] |
| resident_reupload | checksum | 542.8 [541.2–547.8] |
| resident_reupload | iter_total | 3636.2 [3613.5–3692.6] |
| resident_split | gemm_device | 2973.9 [2923.7–3000.2] |
| resident_split | download | 114.9 [105.2–122.0] |
| resident_split | out_drop | 0.3 [0.3–0.3] |
| resident_split | readout | 0.4 [0.3–0.5] |
| resident_split | checksum | 543.0 [542.5–545.6] |
| resident_split | iter_total | 3638.3 [3597.2–3655.9] |
| gemm_device_hold | iter_total | 2587.2 [2567.5–2670.8] |
| gemm_device_drop_in | iter_total | 2599.8 [2562.9–2641.3] |
| gemm_device_download | iter_total | 2799.1 [2634.8–2855.8] |
| backend_drop | kernel_call | 2691.4 [2613.3–2933.9] |
| backend_drop | iter_total | 2692.8 [2613.7–2934.3] |

### N=2048

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.2 [0.2–0.2] |
| fresh | var_upload | 1.3 [1.2–1.3] |
| fresh | matmul | 16619.0 [16517.8–16723.3] |
| fresh | readout | 0.5 [0.4–0.5] |
| fresh | checksum | 2198.6 [2192.8–2209.9] |
| fresh | iter_total | 18793.0 [18726.7–18946.9] |
| fresh | tape_drop | 2.0 [2.0–2.1] |
| resident_public | matmul | 17118.1 [17066.5–17235.1] |
| resident_public | readout | 0.7 [0.7–0.8] |
| resident_public | checksum | 2169.4 [2165.6–2185.4] |
| resident_public | iter_total | 19306.0 [19250.4–19420.8] |
| resident_reupload | matmul | 17212.7 [17121.4–17297.8] |
| resident_reupload | readout | 0.7 [0.6–0.7] |
| resident_reupload | checksum | 2171.4 [2163.8–2184.6] |
| resident_reupload | iter_total | 19379.5 [19306.7–19482.6] |
| resident_split | gemm_device | 16509.9 [16418.9–16543.8] |
| resident_split | download | 564.7 [552.9–640.6] |
| resident_split | out_drop | 0.5 [0.5–0.6] |
| resident_split | readout | 0.7 [0.6–0.8] |
| resident_split | checksum | 2169.1 [2162.5–2184.1] |
| resident_split | iter_total | 19258.1 [19202.1–19323.0] |
| gemm_device_hold | iter_total | 16394.9 [16363.6–16632.7] |
| gemm_device_drop_in | iter_total | 16519.7 [16384.5–16569.6] |
| gemm_device_download | iter_total | 17103.7 [17074.9–17177.2] |
| backend_drop | kernel_call | 16674.8 [16592.9–16703.3] |
| backend_drop | iter_total | 16677.2 [16594.7–16704.6] |

### N=4096

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.2 [0.2–0.2] |
| fresh | var_upload | 1.5 [1.4–1.6] |
| fresh | matmul | 142966.9 [142112.1–143371.6] |
| fresh | readout | 0.5 [0.5–0.5] |
| fresh | checksum | 8766.4 [8695.1–8773.5] |
| fresh | iter_total | 151652.2 [150899.8–152136.1] |
| fresh | tape_drop | 10232.8 [10178.8–10341.2] |
| resident_public | matmul | 171530.5 [170014.8–172637.0] |
| resident_public | readout | 1.1 [1.0–1.2] |
| resident_public | checksum | 8726.3 [8607.7–8738.6] |
| resident_public | iter_total | 180140.6 [178647.7–181370.9] |
| resident_reupload | matmul | 171653.6 [170710.7–172523.5] |
| resident_reupload | readout | 1.0 [1.0–1.2] |
| resident_reupload | checksum | 8725.6 [8608.7–8738.2] |
| resident_reupload | iter_total | 180408.2 [179373.7–181145.1] |
| resident_split | gemm_device | 142418.0 [141565.9–145475.3] |
| resident_split | download | 19700.9 [19520.2–19853.3] |
| resident_split | out_drop | 9241.5 [9214.9–9264.9] |
| resident_split | readout | 1.1 [0.9–1.1] |
| resident_split | checksum | 8724.2 [8603.2–8734.6] |
| resident_split | iter_total | 179641.0 [179080.7–183000.1] |
| gemm_device_hold | iter_total | 142770.1 [141839.2–143883.2] |
| gemm_device_drop_in | iter_total | 152822.6 [151839.3–153638.4] |
| gemm_device_download | iter_total | 162042.2 [161914.6–163333.1] |
| backend_drop | kernel_call | 142363.0 [141945.2–149344.1] |
| backend_drop | iter_total | 142365.0 [141947.1–149346.1] |

## run 別 iter_total（fresh / resident_public、us）

| N | arm | run1 | run2 | run3 | run4 | run5 |
|---|---|---|---|---|---|---|
| 256 | fresh | 138.6 | 138.0 | 140.8 | 135.0 | 135.6 |
| 256 | resident_public | 133.6 | 134.2 | 137.3 | 128.0 | 133.5 |
| 512 | fresh | 1008.4 | 976.7 | 1046.4 | 984.2 | 1023.0 |
| 512 | resident_public | 1041.8 | 885.4 | 1073.9 | 1058.0 | 1047.6 |
| 1024 | fresh | 3395.1 | 3155.8 | 3400.5 | 3373.4 | 3236.2 |
| 1024 | resident_public | 3698.4 | 3564.6 | 3665.7 | 3622.0 | 3627.1 |
| 2048 | fresh | 18831.7 | 18758.7 | 18793.0 | 18726.7 | 18946.9 |
| 2048 | resident_public | 19306.0 | 19300.8 | 19310.8 | 19250.4 | 19420.8 |
| 4096 | fresh | 151652.2 | 150899.8 | 152136.1 | 152080.8 | 150918.5 |
| 4096 | resident_public | 180589.2 | 179811.7 | 181370.9 | 180140.6 | 178647.7 |

