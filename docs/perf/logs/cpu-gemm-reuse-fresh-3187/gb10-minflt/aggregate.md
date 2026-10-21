# 集計（#3187）: gb10-minflt

**GEMM_REUSE_MINFLT=1 の補助系列: 数値記録のみ。再現判定・帰属・因果・arena 判定は行わない（RULE.txt: 判定対象外）。**

checksum: 全 n・全腕・全 run で iter_total の checksum_bits が一致（hard 検査通過）。

## 記録表（判定なし。1 run 中央値。単位 us）

### N=256

- fresh.iter_total = 239.2, reuse.iter_total = 297.9, ratio = 1.245
- 残差 (iter_total - sum(区間)): fresh = 54.4, reuse = 36.2, reuse - fresh = -18.2（総時間差 58.7）

### N=512

- fresh.iter_total = 916.5, reuse.iter_total = 1364.2, ratio = 1.488
- 残差 (iter_total - sum(区間)): fresh = 73.5, reuse = 24.0, reuse - fresh = -49.5（総時間差 447.7）

### N=1024

- fresh.iter_total = 3222.3, reuse.iter_total = 4453.1, ratio = 1.382
- 残差 (iter_total - sum(区間)): fresh = 192.2, reuse = 116.5, reuse - fresh = -75.7（総時間差 1230.8）

### N=2048

- fresh.iter_total = 16156.2, reuse.iter_total = 18360.5, ratio = 1.136
- 残差 (iter_total - sum(区間)): fresh = 178.6, reuse = 137.3, reuse - fresh = -41.3（総時間差 2204.4）

## ページ確保の機構（補助・判定外）: minflt / iter

pagesize = 4096 B。基準ページ数 = N^2*4/PAGESIZE（f32 の N x N 出力 1 枚分）。

| N | arm | phase | minflt/iter (run 中央値) | N^2*4/PAGESIZE | 比 |
|---|---|---|---|---|---|
| 256 | fresh | tape_build | 0.0 | 64.0 | 0.00 |
| 256 | fresh | var_upload | 0.0 | 64.0 | 0.00 |
| 256 | fresh | matmul | 0.1 | 64.0 | 0.00 |
| 256 | fresh | host_copy | 0.0 | 64.0 | 0.00 |
| 256 | fresh | checksum | 0.0 | 64.0 | 0.00 |
| 256 | fresh | iter_total | 0.1 | 64.0 | 0.00 |
| 256 | fresh | tape_drop | 0.0 | 64.0 | 0.00 |
| 256 | fresh_hold | tape_build | 0.0 | 64.0 | 0.00 |
| 256 | fresh_hold | var_upload | 0.0 | 64.0 | 0.00 |
| 256 | fresh_hold | matmul | 41.5 | 64.0 | 0.65 |
| 256 | fresh_hold | host_copy | 0.0 | 64.0 | 0.00 |
| 256 | fresh_hold | checksum | 0.0 | 64.0 | 0.00 |
| 256 | fresh_hold | iter_total | 41.5 | 64.0 | 0.65 |
| 256 | fresh_arena | tape_build | 0.0 | 64.0 | 0.00 |
| 256 | fresh_arena | var_upload | 0.0 | 64.0 | 0.00 |
| 256 | fresh_arena | matmul | 0.0 | 64.0 | 0.00 |
| 256 | fresh_arena | host_copy | 0.0 | 64.0 | 0.00 |
| 256 | fresh_arena | checksum | 0.0 | 64.0 | 0.00 |
| 256 | fresh_arena | iter_total | 0.0 | 64.0 | 0.00 |
| 256 | fresh_arena | tape_drop | 0.0 | 64.0 | 0.00 |
| 256 | fresh_legacy | tape_build | 0.0 | 64.0 | 0.00 |
| 256 | fresh_legacy | var_upload | 0.0 | 64.0 | 0.00 |
| 256 | fresh_legacy | matmul | 0.0 | 64.0 | 0.00 |
| 256 | fresh_legacy | host_copy | 0.0 | 64.0 | 0.00 |
| 256 | fresh_legacy | checksum | 0.0 | 64.0 | 0.00 |
| 256 | fresh_legacy | iter_total | 0.0 | 64.0 | 0.00 |
| 256 | fresh_legacy | tape_drop | 0.0 | 64.0 | 0.00 |
| 256 | reuse | matmul | 87.6 | 64.0 | 1.37 |
| 256 | reuse | host_copy | 0.0 | 64.0 | 0.00 |
| 256 | reuse | checksum | 0.0 | 64.0 | 0.00 |
| 256 | reuse | iter_total | 87.6 | 64.0 | 1.37 |
| 256 | reuse_arena | matmul | 38.4 | 64.0 | 0.60 |
| 256 | reuse_arena | host_copy | 0.0 | 64.0 | 0.00 |
| 256 | reuse_arena | checksum | 0.0 | 64.0 | 0.00 |
| 256 | reuse_arena | iter_total | 38.4 | 64.0 | 0.60 |
| 256 | reuse_legacy | matmul | 0.0 | 64.0 | 0.00 |
| 256 | reuse_legacy | host_copy | 41.6 | 64.0 | 0.65 |
| 256 | reuse_legacy | checksum | 0.0 | 64.0 | 0.00 |
| 256 | reuse_legacy | iter_total | 41.6 | 64.0 | 0.65 |
| 256 | backend_drop | kernel_call | 0.0 | 64.0 | 0.00 |
| 256 | backend_drop | iter_total | 0.0 | 64.0 | 0.00 |
| 256 | backend_hold | kernel_call | 38.4 | 64.0 | 0.60 |
| 256 | backend_hold | iter_total | 38.4 | 64.0 | 0.60 |
| 512 | fresh | tape_build | 0.0 | 256.0 | 0.00 |
| 512 | fresh | var_upload | 0.0 | 256.0 | 0.00 |
| 512 | fresh | matmul | 0.0 | 256.0 | 0.00 |
| 512 | fresh | host_copy | 0.0 | 256.0 | 0.00 |
| 512 | fresh | checksum | 0.0 | 256.0 | 0.00 |
| 512 | fresh | iter_total | 0.0 | 256.0 | 0.00 |
| 512 | fresh | tape_drop | 0.0 | 256.0 | 0.00 |
| 512 | fresh_hold | tape_build | 0.0 | 256.0 | 0.00 |
| 512 | fresh_hold | var_upload | 0.0 | 256.0 | 0.00 |
| 512 | fresh_hold | matmul | 415.6 | 256.0 | 1.62 |
| 512 | fresh_hold | host_copy | 0.0 | 256.0 | 0.00 |
| 512 | fresh_hold | checksum | 0.0 | 256.0 | 0.00 |
| 512 | fresh_hold | iter_total | 415.6 | 256.0 | 1.62 |
| 512 | fresh_arena | tape_build | 0.0 | 256.0 | 0.00 |
| 512 | fresh_arena | var_upload | 0.0 | 256.0 | 0.00 |
| 512 | fresh_arena | matmul | 0.0 | 256.0 | 0.00 |
| 512 | fresh_arena | host_copy | 0.0 | 256.0 | 0.00 |
| 512 | fresh_arena | checksum | 0.0 | 256.0 | 0.00 |
| 512 | fresh_arena | iter_total | 0.0 | 256.0 | 0.00 |
| 512 | fresh_arena | tape_drop | 0.0 | 256.0 | 0.00 |
| 512 | fresh_legacy | tape_build | 0.0 | 256.0 | 0.00 |
| 512 | fresh_legacy | var_upload | 0.0 | 256.0 | 0.00 |
| 512 | fresh_legacy | matmul | 0.0 | 256.0 | 0.00 |
| 512 | fresh_legacy | host_copy | 0.0 | 256.0 | 0.00 |
| 512 | fresh_legacy | checksum | 0.0 | 256.0 | 0.00 |
| 512 | fresh_legacy | iter_total | 0.0 | 256.0 | 0.00 |
| 512 | fresh_legacy | tape_drop | 0.0 | 256.0 | 0.00 |
| 512 | reuse | matmul | 441.6 | 256.0 | 1.73 |
| 512 | reuse | host_copy | 0.0 | 256.0 | 0.00 |
| 512 | reuse | checksum | 0.0 | 256.0 | 0.00 |
| 512 | reuse | iter_total | 441.6 | 256.0 | 1.73 |
| 512 | reuse_arena | matmul | 256.0 | 256.0 | 1.00 |
| 512 | reuse_arena | host_copy | 0.0 | 256.0 | 0.00 |
| 512 | reuse_arena | checksum | 0.0 | 256.0 | 0.00 |
| 512 | reuse_arena | iter_total | 256.0 | 256.0 | 1.00 |
| 512 | reuse_legacy | matmul | 0.0 | 256.0 | 0.00 |
| 512 | reuse_legacy | host_copy | 256.0 | 256.0 | 1.00 |
| 512 | reuse_legacy | checksum | 0.0 | 256.0 | 0.00 |
| 512 | reuse_legacy | iter_total | 256.0 | 256.0 | 1.00 |
| 512 | backend_drop | kernel_call | 0.0 | 256.0 | 0.00 |
| 512 | backend_drop | iter_total | 0.0 | 256.0 | 0.00 |
| 512 | backend_hold | kernel_call | 256.0 | 256.0 | 1.00 |
| 512 | backend_hold | iter_total | 256.0 | 256.0 | 1.00 |
| 1024 | fresh | tape_build | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh | var_upload | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh | matmul | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh | host_copy | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh | checksum | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh | iter_total | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh | tape_drop | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh_hold | tape_build | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh_hold | var_upload | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh_hold | matmul | 1917.9 | 1024.0 | 1.87 |
| 1024 | fresh_hold | host_copy | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh_hold | checksum | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh_hold | iter_total | 1917.9 | 1024.0 | 1.87 |
| 1024 | fresh_arena | tape_build | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh_arena | var_upload | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh_arena | matmul | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh_arena | host_copy | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh_arena | checksum | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh_arena | iter_total | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh_arena | tape_drop | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh_legacy | tape_build | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh_legacy | var_upload | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh_legacy | matmul | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh_legacy | host_copy | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh_legacy | checksum | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh_legacy | iter_total | 0.0 | 1024.0 | 0.00 |
| 1024 | fresh_legacy | tape_drop | 0.0 | 1024.0 | 0.00 |
| 1024 | reuse | matmul | 1855.9 | 1024.0 | 1.81 |
| 1024 | reuse | host_copy | 0.0 | 1024.0 | 0.00 |
| 1024 | reuse | checksum | 0.0 | 1024.0 | 0.00 |
| 1024 | reuse | iter_total | 1855.9 | 1024.0 | 1.81 |
| 1024 | reuse_arena | matmul | 1906.3 | 1024.0 | 1.86 |
| 1024 | reuse_arena | host_copy | 0.0 | 1024.0 | 0.00 |
| 1024 | reuse_arena | checksum | 0.0 | 1024.0 | 0.00 |
| 1024 | reuse_arena | iter_total | 1906.3 | 1024.0 | 1.86 |
| 1024 | reuse_legacy | matmul | 51.2 | 1024.0 | 0.05 |
| 1024 | reuse_legacy | host_copy | 819.2 | 1024.0 | 0.80 |
| 1024 | reuse_legacy | checksum | 0.0 | 1024.0 | 0.00 |
| 1024 | reuse_legacy | iter_total | 870.5 | 1024.0 | 0.85 |
| 1024 | backend_drop | kernel_call | 0.0 | 1024.0 | 0.00 |
| 1024 | backend_drop | iter_total | 0.0 | 1024.0 | 0.00 |
| 1024 | backend_hold | kernel_call | 1969.7 | 1024.0 | 1.92 |
| 1024 | backend_hold | iter_total | 1969.7 | 1024.0 | 1.92 |
| 2048 | fresh | tape_build | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh | var_upload | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh | matmul | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh | host_copy | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh | checksum | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh | iter_total | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh | tape_drop | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh_hold | tape_build | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh_hold | var_upload | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh_hold | matmul | 6251.6 | 4096.0 | 1.53 |
| 2048 | fresh_hold | host_copy | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh_hold | checksum | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh_hold | iter_total | 6251.6 | 4096.0 | 1.53 |
| 2048 | fresh_arena | tape_build | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh_arena | var_upload | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh_arena | matmul | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh_arena | host_copy | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh_arena | checksum | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh_arena | iter_total | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh_arena | tape_drop | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh_legacy | tape_build | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh_legacy | var_upload | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh_legacy | matmul | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh_legacy | host_copy | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh_legacy | checksum | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh_legacy | iter_total | 0.0 | 4096.0 | 0.00 |
| 2048 | fresh_legacy | tape_drop | 0.0 | 4096.0 | 0.00 |
| 2048 | reuse | matmul | 7528.5 | 4096.0 | 1.84 |
| 2048 | reuse | host_copy | 0.0 | 4096.0 | 0.00 |
| 2048 | reuse | checksum | 0.0 | 4096.0 | 0.00 |
| 2048 | reuse | iter_total | 7528.5 | 4096.0 | 1.84 |
| 2048 | reuse_arena | matmul | 5146.0 | 4096.0 | 1.26 |
| 2048 | reuse_arena | host_copy | 0.0 | 4096.0 | 0.00 |
| 2048 | reuse_arena | checksum | 0.0 | 4096.0 | 0.00 |
| 2048 | reuse_arena | iter_total | 5146.0 | 4096.0 | 1.26 |
| 2048 | reuse_legacy | matmul | 921.8 | 4096.0 | 0.23 |
| 2048 | reuse_legacy | host_copy | 2969.9 | 4096.0 | 0.73 |
| 2048 | reuse_legacy | checksum | 0.0 | 4096.0 | 0.00 |
| 2048 | reuse_legacy | iter_total | 3891.7 | 4096.0 | 0.95 |
| 2048 | backend_drop | kernel_call | 0.0 | 4096.0 | 0.00 |
| 2048 | backend_drop | iter_total | 0.0 | 4096.0 | 0.00 |
| 2048 | backend_hold | kernel_call | 5149.3 | 4096.0 | 1.26 |
| 2048 | backend_hold | iter_total | 5149.3 | 4096.0 | 1.26 |

## 区間表（1 run 中央値 [min–max]、us）

### N=256

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.2 [0.2–0.2] |
| fresh | var_upload | 0.2 [0.2–0.2] |
| fresh | matmul | 150.2 [150.2–150.2] |
| fresh | host_copy | 0.1 [0.1–0.1] |
| fresh | checksum | 34.0 [34.0–34.0] |
| fresh | iter_total | 239.2 [239.2–239.2] |
| fresh | tape_drop | 0.7 [0.7–0.7] |
| fresh_hold | tape_build | 0.2 [0.2–0.2] |
| fresh_hold | var_upload | 0.3 [0.3–0.3] |
| fresh_hold | matmul | 221.7 [221.7–221.7] |
| fresh_hold | host_copy | 0.1 [0.1–0.1] |
| fresh_hold | checksum | 33.8 [33.8–33.8] |
| fresh_hold | iter_total | 311.2 [311.2–311.2] |
| fresh_arena | tape_build | 0.2 [0.2–0.2] |
| fresh_arena | var_upload | 0.4 [0.4–0.4] |
| fresh_arena | matmul | 148.4 [148.4–148.4] |
| fresh_arena | host_copy | 0.1 [0.1–0.1] |
| fresh_arena | checksum | 34.0 [34.0–34.0] |
| fresh_arena | iter_total | 238.3 [238.3–238.3] |
| fresh_arena | tape_drop | 0.7 [0.7–0.7] |
| fresh_legacy | tape_build | 0.2 [0.2–0.2] |
| fresh_legacy | var_upload | 0.3 [0.3–0.3] |
| fresh_legacy | matmul | 139.2 [139.2–139.2] |
| fresh_legacy | host_copy | 6.2 [6.2–6.2] |
| fresh_legacy | checksum | 33.6 [33.6–33.6] |
| fresh_legacy | iter_total | 235.3 [235.3–235.3] |
| fresh_legacy | tape_drop | 0.8 [0.8–0.8] |
| reuse | matmul | 227.7 [227.7–227.7] |
| reuse | host_copy | 0.1 [0.1–0.1] |
| reuse | checksum | 33.8 [33.8–33.8] |
| reuse | iter_total | 297.9 [297.9–297.9] |
| reuse_arena | matmul | 198.4 [198.4–198.4] |
| reuse_arena | host_copy | 0.1 [0.1–0.1] |
| reuse_arena | checksum | 33.8 [33.8–33.8] |
| reuse_arena | iter_total | 267.6 [267.6–267.6] |
| reuse_legacy | matmul | 147.6 [147.6–147.6] |
| reuse_legacy | host_copy | 79.2 [79.2–79.2] |
| reuse_legacy | checksum | 33.6 [33.6–33.6] |
| reuse_legacy | iter_total | 300.5 [300.5–300.5] |
| backend_drop | kernel_call | 136.1 [136.1–136.1] |
| backend_drop | iter_total | 153.3 [153.3–153.3] |
| backend_hold | kernel_call | 183.2 [183.2–183.2] |
| backend_hold | iter_total | 200.3 [200.3–200.3] |

### N=512

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.3 [0.3–0.3] |
| fresh | var_upload | 0.6 [0.6–0.6] |
| fresh | matmul | 652.6 [652.6–652.6] |
| fresh | host_copy | 0.2 [0.2–0.2] |
| fresh | checksum | 189.3 [189.3–189.3] |
| fresh | iter_total | 916.5 [916.5–916.5] |
| fresh | tape_drop | 1.3 [1.3–1.3] |
| fresh_hold | tape_build | 0.3 [0.3–0.3] |
| fresh_hold | var_upload | 0.7 [0.7–0.7] |
| fresh_hold | matmul | 1141.7 [1141.7–1141.7] |
| fresh_hold | host_copy | 0.3 [0.3–0.3] |
| fresh_hold | checksum | 136.9 [136.9–136.9] |
| fresh_hold | iter_total | 1402.1 [1402.1–1402.1] |
| fresh_arena | tape_build | 0.3 [0.3–0.3] |
| fresh_arena | var_upload | 0.8 [0.8–0.8] |
| fresh_arena | matmul | 661.3 [661.3–661.3] |
| fresh_arena | host_copy | 0.2 [0.2–0.2] |
| fresh_arena | checksum | 138.0 [138.0–138.0] |
| fresh_arena | iter_total | 882.6 [882.6–882.6] |
| fresh_arena | tape_drop | 1.3 [1.3–1.3] |
| fresh_legacy | tape_build | 0.3 [0.3–0.3] |
| fresh_legacy | var_upload | 0.5 [0.5–0.5] |
| fresh_legacy | matmul | 650.8 [650.8–650.8] |
| fresh_legacy | host_copy | 29.7 [29.7–29.7] |
| fresh_legacy | checksum | 186.9 [186.9–186.9] |
| fresh_legacy | iter_total | 946.4 [946.4–946.4] |
| fresh_legacy | tape_drop | 1.2 [1.2–1.2] |
| reuse | matmul | 1152.1 [1152.1–1152.1] |
| reuse | host_copy | 0.3 [0.3–0.3] |
| reuse | checksum | 187.8 [187.8–187.8] |
| reuse | iter_total | 1364.2 [1364.2–1364.2] |
| reuse_arena | matmul | 1076.3 [1076.3–1076.3] |
| reuse_arena | host_copy | 0.3 [0.3–0.3] |
| reuse_arena | checksum | 187.5 [187.5–187.5] |
| reuse_arena | iter_total | 1305.9 [1305.9–1305.9] |
| reuse_legacy | matmul | 710.7 [710.7–710.7] |
| reuse_legacy | host_copy | 382.1 [382.1–382.1] |
| reuse_legacy | checksum | 186.9 [186.9–186.9] |
| reuse_legacy | iter_total | 1373.9 [1373.9–1373.9] |
| backend_drop | kernel_call | 661.0 [661.0–661.0] |
| backend_drop | iter_total | 703.5 [703.5–703.5] |
| backend_hold | kernel_call | 1045.2 [1045.2–1045.2] |
| backend_hold | iter_total | 1070.9 [1070.9–1070.9] |

### N=1024

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.3 [0.3–0.3] |
| fresh | var_upload | 0.7 [0.7–0.7] |
| fresh | matmul | 2483.2 [2483.2–2483.2] |
| fresh | host_copy | 0.3 [0.3–0.3] |
| fresh | checksum | 545.6 [545.6–545.6] |
| fresh | iter_total | 3222.3 [3222.3–3222.3] |
| fresh | tape_drop | 1.6 [1.6–1.6] |
| fresh_hold | tape_build | 0.3 [0.3–0.3] |
| fresh_hold | var_upload | 1.0 [1.0–1.0] |
| fresh_hold | matmul | 3849.2 [3849.2–3849.2] |
| fresh_hold | host_copy | 0.4 [0.4–0.4] |
| fresh_hold | checksum | 547.1 [547.1–547.1] |
| fresh_hold | iter_total | 4515.5 [4515.5–4515.5] |
| fresh_arena | tape_build | 0.3 [0.3–0.3] |
| fresh_arena | var_upload | 1.1 [1.1–1.1] |
| fresh_arena | matmul | 2328.2 [2328.2–2328.2] |
| fresh_arena | host_copy | 0.4 [0.4–0.4] |
| fresh_arena | checksum | 548.1 [548.1–548.1] |
| fresh_arena | iter_total | 3155.5 [3155.5–3155.5] |
| fresh_arena | tape_drop | 1.8 [1.8–1.8] |
| fresh_legacy | tape_build | 0.3 [0.3–0.3] |
| fresh_legacy | var_upload | 0.6 [0.6–0.6] |
| fresh_legacy | matmul | 2718.7 [2718.7–2718.7] |
| fresh_legacy | host_copy | 93.1 [93.1–93.1] |
| fresh_legacy | checksum | 548.2 [548.2–548.2] |
| fresh_legacy | iter_total | 3557.0 [3557.0–3557.0] |
| fresh_legacy | tape_drop | 1.2 [1.2–1.2] |
| reuse | matmul | 3793.1 [3793.1–3793.1] |
| reuse | host_copy | 0.4 [0.4–0.4] |
| reuse | checksum | 543.0 [543.0–543.0] |
| reuse | iter_total | 4453.1 [4453.1–4453.1] |
| reuse_arena | matmul | 3821.3 [3821.3–3821.3] |
| reuse_arena | host_copy | 0.6 [0.6–0.6] |
| reuse_arena | checksum | 757.0 [757.0–757.0] |
| reuse_arena | iter_total | 4549.4 [4549.4–4549.4] |
| reuse_legacy | matmul | 2893.3 [2893.3–2893.3] |
| reuse_legacy | host_copy | 1450.2 [1450.2–1450.2] |
| reuse_legacy | checksum | 544.1 [544.1–544.1] |
| reuse_legacy | iter_total | 4961.1 [4961.1–4961.1] |
| backend_drop | kernel_call | 2340.9 [2340.9–2340.9] |
| backend_drop | iter_total | 2385.9 [2385.9–2385.9] |
| backend_hold | kernel_call | 3845.4 [3845.4–3845.4] |
| backend_hold | iter_total | 3886.0 [3886.0–3886.0] |

### N=2048

| arm | phase | median [min–max] |
|---|---|---|
| fresh | tape_build | 0.3 [0.3–0.3] |
| fresh | var_upload | 1.4 [1.4–1.4] |
| fresh | matmul | 12964.3 [12964.3–12964.3] |
| fresh | host_copy | 0.9 [0.9–0.9] |
| fresh | checksum | 3010.7 [3010.7–3010.7] |
| fresh | iter_total | 16156.2 [16156.2–16156.2] |
| fresh | tape_drop | 3.5 [3.5–3.5] |
| fresh_hold | tape_build | 0.3 [0.3–0.3] |
| fresh_hold | var_upload | 1.8 [1.8–1.8] |
| fresh_hold | matmul | 15255.7 [15255.7–15255.7] |
| fresh_hold | host_copy | 0.9 [0.9–0.9] |
| fresh_hold | checksum | 3013.1 [3013.1–3013.1] |
| fresh_hold | iter_total | 18404.3 [18404.3–18404.3] |
| fresh_arena | tape_build | 0.3 [0.3–0.3] |
| fresh_arena | var_upload | 1.9 [1.9–1.9] |
| fresh_arena | matmul | 12989.0 [12989.0–12989.0] |
| fresh_arena | host_copy | 0.9 [0.9–0.9] |
| fresh_arena | checksum | 3012.5 [3012.5–3012.5] |
| fresh_arena | iter_total | 16175.7 [16175.7–16175.7] |
| fresh_arena | tape_drop | 3.9 [3.9–3.9] |
| fresh_legacy | tape_build | 0.4 [0.4–0.4] |
| fresh_legacy | var_upload | 1.3 [1.3–1.3] |
| fresh_legacy | matmul | 13216.1 [13216.1–13216.1] |
| fresh_legacy | host_copy | 640.0 [640.0–640.0] |
| fresh_legacy | checksum | 3007.9 [3007.9–3007.9] |
| fresh_legacy | iter_total | 17059.0 [17059.0–17059.0] |
| fresh_legacy | tape_drop | 3.3 [3.3–3.3] |
| reuse | matmul | 15186.7 [15186.7–15186.7] |
| reuse | host_copy | 0.9 [0.9–0.9] |
| reuse | checksum | 3035.6 [3035.6–3035.6] |
| reuse | iter_total | 18360.5 [18360.5–18360.5] |
| reuse_arena | matmul | 14940.9 [14940.9–14940.9] |
| reuse_arena | host_copy | 0.9 [0.9–0.9] |
| reuse_arena | checksum | 3017.0 [3017.0–3017.0] |
| reuse_arena | iter_total | 18096.2 [18096.2–18096.2] |
| reuse_legacy | matmul | 13292.1 [13292.1–13292.1] |
| reuse_legacy | host_copy | 5628.6 [5628.6–5628.6] |
| reuse_legacy | checksum | 3011.3 [3011.3–3011.3] |
| reuse_legacy | iter_total | 22842.0 [22842.0–22842.0] |
| backend_drop | kernel_call | 12952.2 [12952.2–12952.2] |
| backend_drop | iter_total | 13062.4 [13062.4–13062.4] |
| backend_hold | kernel_call | 14953.5 [14953.5–14953.5] |
| backend_hold | iter_total | 15063.8 [15063.8–15063.8] |

## run 別 iter_total（fresh / reuse、us）

| N | arm | run1 |
|---|---|---|
| 256 | fresh | 239.2 |
| 256 | reuse | 297.9 |
| 512 | fresh | 916.5 |
| 512 | reuse | 1364.2 |
| 1024 | fresh | 3222.3 |
| 1024 | reuse | 4453.1 |
| 2048 | fresh | 16156.2 |
| 2048 | reuse | 18360.5 |

