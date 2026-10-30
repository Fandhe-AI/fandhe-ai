# #3701 帰属判定（split_inner_judge.py の出力。手で書き換えない）

RULE.txt sha256: `08b0782b77009959b90a15cdecdfc89449623d2d9e77643adbaeaf7fc6b4bee9`

## 結論

| 機体 | N | 位置づけ | 結論 |
|---|---|---|---|
| macmini | 256 | 判定対象 | **主因の区間を特定した: S1a** |
| macmini | 512 | 記録のみ | **判定不能（健全性 A: T2 が出ない）** |
| gb10 | 256 | 記録のみ | **判定不能（健全性 A: T2 が出ない）** |
| gb10 | 512 | 記録のみ | **判定不能（健全性 A: T2 が出ない）** |

- macmini N=256: 段 1（#3693 と同じ S1〜S4）の主因 S1。S1 主因の再現: する
- macmini N=512: resident_public iter_total の ratio（after/before）が k>=12 かつ全系列の中央値>1.00 を満たさない（k=1）。計時点が T2 を消した場合も本結論になる
- gb10 N=256: resident_public iter_total の ratio（after/before）が k>=12 かつ全系列の中央値>1.00 を満たさない（k=7）。計時点が T2 を消した場合も本結論になる
- gb10 N=512: resident_public iter_total の ratio（after/before）が k>=12 かつ全系列の中央値>1.00 を満たさない（k=6）。計時点が T2 を消した場合も本結論になる

## 系列の扱い

- macmini s1（先頭腕 before）: 判定に使用
- macmini s2（先頭腕 after）: 判定に使用
- macmini s3（先頭腕 before）: 判定に使用
- gb10 s1（先頭腕 before）: 判定に使用
- gb10 s2（先頭腕 after）: 判定に使用
- gb10 s3（先頭腕 before）: 判定に使用

## macmini N=256（判定対象）

- 健全性 A（resident_public iter_total の木の間 ratio）: k=14 中央値=1.0654 成立=True
- 健全性 B（after の木の owned/copy ratio）: k=15 中央値=1.0764 成立=True

### after の木の帰属（owned - copy の中央値。秒。段 1 は S1〜S4、段 2 は内側の区間）

| 項目 | 値 (s) | D_iter 比 |
|---|---|---|
| D_iter | 1.708e-05 | 1.00 |
| S1 | 2.171e-05 | 1.27 |
| S2 | -3.959e-06 | -0.23 |
| S3 | 0.000e+00 | 0.00 |
| S4 | 0.000e+00 | 0.00 |
| S1k | 0.000e+00 | 0.00 |
| S1a | 1.600e-05 | 0.94 |
| S1b | 4.375e-06 | 0.26 |
| S1c | 0.000e+00 | 0.00 |
| S2k | 4.200e-08 | 0.00 |
| S2m | -3.959e-06 | -0.23 |
| S2w | -4.100e-08 | -0.00 |
| S2d | -8.300e-08 | -0.00 |
| S1_resid | 5.420e-07 | 0.03 |
| S2_resid | 8.200e-08 | 0.00 |
| residual | -5.420e-07 | -0.03 |
| result_drop | 2.910e-07 | 0.02 |

（S1k/S1a/S1b/S1c は S1、S2k/S2m/S2w/S2d は S2 の内側。S1_resid・S2_resid は内側の区間の和で説明できない分。記録のみ）

### before の木の同じ表（記録のみ）

| 項目 | 値 (s) | D_iter 比 |
|---|---|---|
| D_iter | 1.550e-05 | 1.00 |
| S1 | 1.933e-05 | 1.25 |
| S2 | -3.959e-06 | -0.26 |
| S3 | -1.000e-09 | -0.00 |
| S4 | 0.000e+00 | 0.00 |
| S1k | 0.000e+00 | 0.00 |
| S1a | 1.575e-05 | 1.02 |
| S1b | 4.125e-06 | 0.27 |
| S1c | 0.000e+00 | 0.00 |
| S2k | 4.200e-08 | 0.00 |
| S2m | -4.000e-06 | -0.26 |
| S2w | -4.100e-08 | -0.00 |
| S2d | -8.300e-08 | -0.01 |
| S1_resid | 5.010e-07 | 0.03 |
| S2_resid | 8.300e-08 | 0.01 |
| residual | -1.270e-07 | -0.01 |
| result_drop | 2.500e-07 | 0.02 |

（S1k/S1a/S1b/S1c は S1、S2k/S2m/S2w/S2d は S2 の内側。S1_resid・S2_resid は内側の区間の和で説明できない分。記録のみ）

### resident_public の木の間 ratio（after/before の 15 個の中央値。記録のみ）

| 区間 | 中央値 ratio |
|---|---|
| iter_total | 1.0654 |
| matmul | 1.0756 |
| readout | 1.0000 |
| checksum | 1.0000 |
| result_drop | 3.3280 |
| gd_checks | 1.0000 |
| gd_alloc_zeroed | 6.2186 |
| gd_compute | 1.0130 |
| gd_wrap | 1.3360 |

### 写し 2 診断腕の木の間 ratio（記録のみ）

- mirror_owned: k=7 中央値=0.9968 木の差が経路の外にも出ている=False
- mirror_copy: k=4 中央値=0.9934 木の差が経路の外にも出ている=False

### alloc_probe 2 診断腕の区間差（after の木・owned - copy の 15 個の中央値。記録のみ）

| 区間 | 差 (s) |
|---|---|
| iter_total | -3.208e-06 |
| alloc | 1.292e-06 |
| touch | 0.000e+00 |
| result_drop | 1.000e-09 |
| wrap（owned） - copy + src_drop（copy） | -4.416e-06 |

## macmini N=512（記録のみ）

- 健全性 A（resident_public iter_total の木の間 ratio）: k=1 中央値=0.9816 成立=False

### before の木の同じ表（記録のみ）

| 項目 | 値 (s) | D_iter 比 |
|---|---|---|
| D_iter | -1.454e-05 | 1.00 |
| S1 | 2.042e-06 | -0.14 |
| S2 | -1.379e-05 | 0.95 |
| S3 | 0.000e+00 | -0.00 |
| S4 | -8.400e-08 | 0.01 |
| S1k | 0.000e+00 | -0.00 |
| S1a | -8.400e-08 | 0.01 |
| S1b | 5.430e-07 | -0.04 |
| S1c | 0.000e+00 | -0.00 |
| S2k | 4.200e-08 | -0.00 |
| S2m | -1.379e-05 | 0.95 |
| S2w | -4.100e-08 | 0.00 |
| S2d | -8.300e-08 | 0.01 |
| S1_resid | 1.708e-06 | -0.12 |
| S2_resid | 1.230e-07 | -0.01 |
| residual | -4.570e-07 | 0.03 |
| result_drop | 4.200e-08 | -0.00 |

（S1k/S1a/S1b/S1c は S1、S2k/S2m/S2w/S2d は S2 の内側。S1_resid・S2_resid は内側の区間の和で説明できない分。記録のみ）

### resident_public の木の間 ratio（after/before の 15 個の中央値。記録のみ）

| 区間 | 中央値 ratio |
|---|---|
| iter_total | 0.9816 |
| matmul | 0.9739 |
| readout | 1.0000 |
| checksum | 1.0003 |
| result_drop | 1.3360 |
| gd_checks | 1.0000 |
| gd_alloc_zeroed | 0.9971 |
| gd_compute | 0.9926 |
| gd_wrap | 1.4881 |

### 写し 2 診断腕の木の間 ratio（記録のみ）

- mirror_owned: k=2 中央値=0.9975 木の差が経路の外にも出ている=False
- mirror_copy: k=3 中央値=0.9945 木の差が経路の外にも出ている=False

### alloc_probe 2 診断腕の区間差（after の木・owned - copy の 15 個の中央値。記録のみ）

| 区間 | 差 (s) |
|---|---|
| iter_total | -1.667e-05 |
| alloc | 1.167e-06 |
| touch | -4.200e-08 |
| result_drop | 4.100e-08 |
| wrap（owned） - copy + src_drop（copy） | -1.763e-05 |

## gb10 N=256（記録のみ）

- 健全性 A（resident_public iter_total の木の間 ratio）: k=7 中央値=0.9934 成立=False

### before の木の同じ表（記録のみ）

| 項目 | 値 (s) | D_iter 比 |
|---|---|---|
| D_iter | -7.729e-06 | 1.00 |
| S1 | -1.344e-06 | 0.17 |
| S2 | -6.352e-06 | 0.82 |
| S3 | -1.600e-08 | 0.00 |
| S4 | 1.600e-07 | -0.02 |
| S1k | -1.600e-08 | 0.00 |
| S1a | 6.304e-06 | -0.82 |
| S1b | -8.576e-06 | 1.11 |
| S1c | 0.000e+00 | -0.00 |
| S2k | 2.240e-07 | -0.03 |
| S2m | -6.384e-06 | 0.83 |
| S2w | -2.400e-07 | 0.03 |
| S2d | -1.440e-07 | 0.02 |
| S1_resid | -4.160e-07 | 0.05 |
| S2_resid | 1.280e-07 | -0.02 |
| residual | -2.623e-06 | 0.34 |
| result_drop | 9.600e-08 | -0.01 |

（S1k/S1a/S1b/S1c は S1、S2k/S2m/S2w/S2d は S2 の内側。S1_resid・S2_resid は内側の区間の和で説明できない分。記録のみ）

### resident_public の木の間 ratio（after/before の 15 個の中央値。記録のみ）

| 区間 | 中央値 ratio |
|---|---|
| iter_total | 0.9934 |
| matmul | 1.0218 |
| readout | 0.7692 |
| checksum | 1.0052 |
| result_drop | 1.3889 |
| gd_checks | 1.0000 |
| gd_alloc_zeroed | 2.4833 |
| gd_compute | 1.0274 |
| gd_wrap | 1.0000 |

### 写し 2 診断腕の木の間 ratio（記録のみ）

- mirror_owned: k=9 中央値=1.0910 木の差が経路の外にも出ている=False
- mirror_copy: k=4 中央値=0.9502 木の差が経路の外にも出ている=False

### alloc_probe 2 診断腕の区間差（after の木・owned - copy の 15 個の中央値。記録のみ）

| 区間 | 差 (s) |
|---|---|
| iter_total | -2.400e-06 |
| alloc | 1.120e-07 |
| touch | 1.500e-08 |
| result_drop | 8.000e-08 |
| wrap（owned） - copy + src_drop（copy） | -2.400e-06 |

## gb10 N=512（記録のみ）

- 健全性 A（resident_public iter_total の木の間 ratio）: k=6 中央値=0.9730 成立=False

### before の木の同じ表（記録のみ）

| 項目 | 値 (s) | D_iter 比 |
|---|---|---|
| D_iter | -4.032e-06 | 1.00 |
| S1 | 2.214e-05 | -5.49 |
| S2 | -2.955e-05 | 7.33 |
| S3 | -6.400e-08 | 0.02 |
| S4 | 2.048e-06 | -0.51 |
| S1k | 3.200e-08 | -0.01 |
| S1a | 3.855e-06 | -0.96 |
| S1b | 1.538e-05 | -3.81 |
| S1c | 1.600e-08 | -0.00 |
| S2k | 5.760e-07 | -0.14 |
| S2m | -2.981e-05 | 7.39 |
| S2w | -4.320e-07 | 0.11 |
| S2d | -1.440e-07 | 0.04 |
| S1_resid | 7.680e-07 | -0.19 |
| S2_resid | 2.400e-07 | -0.06 |
| residual | 6.883e-06 | -1.71 |
| result_drop | 1.600e-08 | -0.00 |

（S1k/S1a/S1b/S1c は S1、S2k/S2m/S2w/S2d は S2 の内側。S1_resid・S2_resid は内側の区間の和で説明できない分。記録のみ）

### resident_public の木の間 ratio（after/before の 15 個の中央値。記録のみ）

| 区間 | 中央値 ratio |
|---|---|
| iter_total | 0.9730 |
| matmul | 0.9653 |
| readout | 1.0625 |
| checksum | 1.0001 |
| result_drop | 0.9615 |
| gd_checks | 1.0000 |
| gd_alloc_zeroed | 1.3466 |
| gd_compute | 0.9835 |
| gd_wrap | 1.0476 |

### 写し 2 診断腕の木の間 ratio（記録のみ）

- mirror_owned: k=7 中央値=0.9912 木の差が経路の外にも出ている=False
- mirror_copy: k=6 中央値=0.9915 木の差が経路の外にも出ている=False

### alloc_probe 2 診断腕の区間差（after の木・owned - copy の 15 個の中央値。記録のみ）

| 区間 | 差 (s) |
|---|---|
| iter_total | -1.094e-05 |
| alloc | -1.440e-07 |
| touch | 1.600e-08 |
| result_drop | 6.400e-08 |
| wrap（owned） - copy + src_drop（copy） | -1.075e-05 |
