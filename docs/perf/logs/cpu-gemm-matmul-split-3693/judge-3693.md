# #3693 帰属判定（split_judge.py の出力。手で書き換えない）

RULE.txt sha256: `2fe453eb45382a29bab55eac4b4195712fb0b289f9a8920e4e8c749b63ba56d8`

## 結論

| 機体 | N | 位置づけ | 結論 |
|---|---|---|---|
| macmini | 256 | 判定対象 | **主因の区間を特定した: S1** |
| macmini | 512 | 記録のみ | **判定不能（健全性 A: T2 が出ない）** |
| gb10 | 256 | 記録のみ | **判定不能（健全性 A: T2 が出ない）** |
| gb10 | 512 | 記録のみ | **判定不能（健全性 A: T2 が出ない）** |

- macmini N=512: resident_public iter_total の ratio（after/before）が k>=12 かつ全系列の中央値>1.00 を満たさない（k=1）
- gb10 N=256: resident_public iter_total の ratio（after/before）が k>=12 かつ全系列の中央値>1.00 を満たさない（k=7）
- gb10 N=512: resident_public iter_total の ratio（after/before）が k>=12 かつ全系列の中央値>1.00 を満たさない（k=6）

## 系列の扱い

- macmini s1（先頭腕 before）: 判定に使用
- macmini s2（先頭腕 after）: 判定に使用
- macmini s3（先頭腕 before）: 判定に使用
- gb10 s1（先頭腕 before）: 判定に使用
- gb10 s2（先頭腕 after）: 判定に使用
- gb10 s3（先頭腕 before）: 判定に使用

## macmini N=256（判定対象）

- 健全性 A（resident_public iter_total の木の間 ratio）: k=15 中央値=1.0623 成立=True
- 健全性 B（after の木の owned/copy ratio）: k=14 中央値=1.0710 成立=True

### after の木の帰属（owned - copy の中央値。秒）

| 項目 | 値 (s) | D_iter 比 |
|---|---|---|
| D_iter | 1.629e-05 | 1.00 |
| S1 | 2.100e-05 | 1.29 |
| S2 | -4.043e-06 | -0.25 |
| S3 | 0.000e+00 | 0.00 |
| S4 | 0.000e+00 | 0.00 |
| residual | 1.000e-09 | 0.00 |
| result_drop | 2.510e-07 | 0.02 |

### before の木の同じ表（記録のみ）

| 項目 | 値 (s) | D_iter 比 |
|---|---|---|
| D_iter | 1.213e-05 | 1.00 |
| S1 | 1.642e-05 | 1.35 |
| S2 | -4.042e-06 | -0.33 |
| S3 | 0.000e+00 | 0.00 |
| S4 | 1.000e-09 | 0.00 |
| residual | 1.250e-07 | 0.01 |
| result_drop | 2.920e-07 | 0.02 |

### resident_public の木の間 ratio（after/before の 15 個の中央値。記録のみ）

| 区間 | 中央値 ratio |
|---|---|
| iter_total | 1.0623 |
| matmul | 1.0751 |
| readout | 1.0000 |
| checksum | 1.0000 |
| result_drop | 3.3280 |

### 写し 2 診断腕の木の間 ratio（記録のみ）

- mirror_owned: k=10 中央値=1.0150 木の差が経路の外にも出ている=False
- mirror_copy: k=5 中央値=0.9910 木の差が経路の外にも出ている=False

### alloc_probe 2 診断腕の区間差（after の木・owned - copy の 15 個の中央値。記録のみ）

| 区間 | 差 (s) |
|---|---|
| iter_total | -3.083e-06 |
| alloc | 1.333e-06 |
| touch | 1.000e-09 |
| result_drop | 0.000e+00 |
| wrap（owned） - copy + src_drop（copy） | -4.375e-06 |

## macmini N=512（記録のみ）

- 健全性 A（resident_public iter_total の木の間 ratio）: k=1 中央値=0.9853 成立=False

### before の木の同じ表（記録のみ）

| 項目 | 値 (s) | D_iter 比 |
|---|---|---|
| D_iter | -1.579e-05 | 1.00 |
| S1 | -4.170e-07 | 0.03 |
| S2 | -1.392e-05 | 0.88 |
| S3 | 0.000e+00 | -0.00 |
| S4 | 4.100e-08 | -0.00 |
| residual | -5.810e-07 | 0.04 |
| result_drop | 4.200e-08 | -0.00 |

### resident_public の木の間 ratio（after/before の 15 個の中央値。記録のみ）

| 区間 | 中央値 ratio |
|---|---|
| iter_total | 0.9853 |
| matmul | 0.9792 |
| readout | 1.0000 |
| checksum | 1.0003 |
| result_drop | 1.3360 |

### 写し 2 診断腕の木の間 ratio（記録のみ）

- mirror_owned: k=8 中央値=1.0007 木の差が経路の外にも出ている=False
- mirror_copy: k=6 中央値=0.9960 木の差が経路の外にも出ている=False

### alloc_probe 2 診断腕の区間差（after の木・owned - copy の 15 個の中央値。記録のみ）

| 区間 | 差 (s) |
|---|---|
| iter_total | -1.667e-05 |
| alloc | 1.292e-06 |
| touch | -2.910e-07 |
| result_drop | 4.200e-08 |
| wrap（owned） - copy + src_drop（copy） | -1.763e-05 |

## gb10 N=256（記録のみ）

- 健全性 A（resident_public iter_total の木の間 ratio）: k=7 中央値=0.9966 成立=False

### before の木の同じ表（記録のみ）

| 項目 | 値 (s) | D_iter 比 |
|---|---|---|
| D_iter | 1.008e-06 | 1.00 |
| S1 | 6.831e-06 | 6.78 |
| S2 | -6.640e-06 | -6.59 |
| S3 | -4.800e-08 | -0.05 |
| S4 | 3.520e-07 | 0.35 |
| residual | -2.550e-07 | -0.25 |
| result_drop | 4.800e-08 | 0.05 |

### resident_public の木の間 ratio（after/before の 15 個の中央値。記録のみ）

| 区間 | 中央値 ratio |
|---|---|
| iter_total | 0.9966 |
| matmul | 0.9952 |
| readout | 0.9167 |
| checksum | 1.0071 |
| result_drop | 1.1600 |

### 写し 2 診断腕の木の間 ratio（記録のみ）

- mirror_owned: k=11 中央値=1.0301 木の差が経路の外にも出ている=False
- mirror_copy: k=9 中央値=1.0534 木の差が経路の外にも出ている=False

### alloc_probe 2 診断腕の区間差（after の木・owned - copy の 15 個の中央値。記録のみ）

| 区間 | 差 (s) |
|---|---|
| iter_total | -2.351e-06 |
| alloc | 6.400e-08 |
| touch | -1.600e-08 |
| result_drop | 8.000e-08 |
| wrap（owned） - copy + src_drop（copy） | -2.368e-06 |

## gb10 N=512（記録のみ）

- 健全性 A（resident_public iter_total の木の間 ratio）: k=6 中央値=0.9887 成立=False

### before の木の同じ表（記録のみ）

| 項目 | 値 (s) | D_iter 比 |
|---|---|---|
| D_iter | -1.059e-04 | 1.00 |
| S1 | -8.878e-05 | 0.84 |
| S2 | -2.507e-05 | 0.24 |
| S3 | -3.200e-08 | 0.00 |
| S4 | 1.248e-06 | -0.01 |
| residual | -8.223e-06 | 0.08 |
| result_drop | -1.600e-08 | 0.00 |

### resident_public の木の間 ratio（after/before の 15 個の中央値。記録のみ）

| 区間 | 中央値 ratio |
|---|---|
| iter_total | 0.9887 |
| matmul | 0.9657 |
| readout | 0.8750 |
| checksum | 1.3440 |
| result_drop | 1.0286 |

### 写し 2 診断腕の木の間 ratio（記録のみ）

- mirror_owned: k=9 中央値=1.0105 木の差が経路の外にも出ている=False
- mirror_copy: k=8 中央値=1.0194 木の差が経路の外にも出ている=False

### alloc_probe 2 診断腕の区間差（after の木・owned - copy の 15 個の中央値。記録のみ）

| 区間 | 差 (s) |
|---|---|
| iter_total | -1.107e-05 |
| alloc | -1.280e-07 |
| touch | 3.200e-08 |
| result_drop | 9.600e-08 |
| wrap（owned） - copy + src_drop（copy） | -1.077e-05 |
