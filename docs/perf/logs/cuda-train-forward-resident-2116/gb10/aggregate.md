checksum: run 間・腕間一致 OK

## forward_resident（上位。5 run 中央値。単位 us）

| batch | arm/phase | median | min-max |
|---|---|---|---|
| 16 | public/forward_resident | 102.1 | 101.3-102.3 |
| 16 | decomposed/register | 0.2 | 0.2-0.2 |
| 16 | decomposed/l1_linear_relu | 57.5 | 56.9-57.5 |
| 16 | decomposed/l2_linear | 30.1 | 29.7-30.1 |
| 16 | decomposed/mse_loss | 14.2 | 14.0-14.3 |
| 16 | paired/residual | 0.4 | 0.1-0.5 |
| 64 | public/forward_resident | 131.8 | 130.8-131.8 |
| 64 | decomposed/register | 0.2 | 0.2-0.2 |
| 64 | decomposed/l1_linear_relu | 78.8 | 78.3-79.7 |
| 64 | decomposed/l2_linear | 37.2 | 36.9-37.4 |
| 64 | decomposed/mse_loss | 14.6 | 14.4-14.6 |
| 64 | paired/residual | 0.4 | 0.1-1.4 |
| 256 | public/forward_resident | 149.4 | 148.3-149.5 |
| 256 | decomposed/register | 0.3 | 0.2-0.3 |
| 256 | decomposed/l1_linear_relu | 89.4 | 88.7-89.9 |
| 256 | decomposed/l2_linear | 43.3 | 43.1-43.7 |
| 256 | decomposed/mse_loss | 15.6 | 15.5-16.0 |
| 256 | paired/residual | 0.5 | 0.1-0.6 |
| 1024 | public/forward_resident | 255.2 | 251.3-257.2 |
| 1024 | decomposed/register | 0.3 | 0.3-0.4 |
| 1024 | decomposed/l1_linear_relu | 173.2 | 168.6-175.1 |
| 1024 | decomposed/l2_linear | 59.2 | 58.9-59.9 |
| 1024 | decomposed/mse_loss | 23.0 | 22.7-23.4 |
| 1024 | paired/residual | -0.1 | -2.8-0.3 |

## batch=16: F = public/forward_resident = 102.1 us

注: 各項は計測スケジュールの異なる独立した参考指標（H1・H4=syncsplit、H3=nosync、H2・H6=prod 単独、H5=上位 API）であり、排他的な内訳ではない。F 比の和は 1.0 を超えうる・合算して F の内訳とは読まない。

| 項 | 値(us) | F 比 | 判定 |
|---|---|---|---|
| H1 層境界の upload/download 呼び出し全体 | 20.4 | 0.20 | 寄与あり |
| H2 未融合 relu の往復 | 14.0 | 0.14 | 非支持 |
| H3 呼び出しごとのデバイス確保 | 5.5 | 0.05 | 非支持 |
| H4 カーネル実行(syncsplit) | 53.2 | 0.52 | 支持（支配項） |
| H5 上位 API 固定費(register+残差) | 0.6 | 0.01 | 非支持 |
| H6 mse の往復 | 13.5 | 0.13 | 非支持 |

支配項: H4 カーネル実行(syncsplit)
fidelity(nosync total / prod total): l1=0.95, l2=0.89（範囲内）
層 l1 の支配区間(nosync 内比 0.40 以上): d2h_download
層 l2 の支配区間(nosync 内比 0.40 以上): d2h_download
補助: decomposed の F 比 l1=0.56 l2=0.29 mse=0.14
補助: what-if(融合 relu) total = 54.1 us / (prod l1 + prod relu = 71.0 us) = 0.76（bits_equal=[True]。record-only）

## batch=64: F = public/forward_resident = 131.8 us

注: 各項は計測スケジュールの異なる独立した参考指標（H1・H4=syncsplit、H3=nosync、H2・H6=prod 単独、H5=上位 API）であり、排他的な内訳ではない。F 比の和は 1.0 を超えうる・合算して F の内訳とは読まない。

| 項 | 値(us) | F 比 | 判定 |
|---|---|---|---|
| H1 層境界の upload/download 呼び出し全体 | 31.4 | 0.24 | 寄与あり |
| H2 未融合 relu の往復 | 17.2 | 0.13 | 非支持 |
| H3 呼び出しごとのデバイス確保 | 5.6 | 0.04 | 非支持 |
| H4 カーネル実行(syncsplit) | 69.7 | 0.53 | 支持（支配項） |
| H5 上位 API 固定費(register+残差) | 0.6 | 0.00 | 非支持 |
| H6 mse の往復 | 13.5 | 0.10 | 非支持 |

支配項: H4 カーネル実行(syncsplit)
fidelity(nosync total / prod total): l1=0.96, l2=0.92（範囲内）
層 l1 の支配区間(nosync 内比 0.40 以上): d2h_download
層 l2 の支配区間(nosync 内比 0.40 以上): d2h_download
補助: decomposed の F 比 l1=0.60 l2=0.28 mse=0.11
補助: what-if(融合 relu) total = 75.9 us / (prod l1 + prod relu = 95.7 us) = 0.79（bits_equal=[True]。record-only）

## batch=256: F = public/forward_resident = 149.4 us

注: 各項は計測スケジュールの異なる独立した参考指標（H1・H4=syncsplit、H3=nosync、H2・H6=prod 単独、H5=上位 API）であり、排他的な内訳ではない。F 比の和は 1.0 を超えうる・合算して F の内訳とは読まない。

| 項 | 値(us) | F 比 | 判定 |
|---|---|---|---|
| H1 層境界の upload/download 呼び出し全体 | 61.0 | 0.41 | 支持（支配項） |
| H2 未融合 relu の往復 | 34.6 | 0.23 | 寄与あり |
| H3 呼び出しごとのデバイス確保 | 5.9 | 0.04 | 非支持 |
| H4 カーネル実行(syncsplit) | 69.5 | 0.47 | 支持（支配項） |
| H5 上位 API 固定費(register+残差) | 0.8 | 0.01 | 非支持 |
| H6 mse の往復 | 14.8 | 0.10 | 非支持 |

支配項: H1 層境界の upload/download 呼び出し全体, H4 カーネル実行(syncsplit)
fidelity(nosync total / prod total): l1=0.97, l2=0.93（範囲内）
層 l1 の支配区間(nosync 内比 0.40 以上): d2h_download
層 l2 の支配区間(nosync 内比 0.40 以上): d2h_download
補助: decomposed の F 比 l1=0.60 l2=0.29 mse=0.10
補助: what-if(融合 relu) total = 84.9 us / (prod l1 + prod relu = 121.7 us) = 0.70（bits_equal=[True]。record-only）

## batch=1024: F = public/forward_resident = 255.2 us

注: 各項は計測スケジュールの異なる独立した参考指標（H1・H4=syncsplit、H3=nosync、H2・H6=prod 単独、H5=上位 API）であり、排他的な内訳ではない。F 比の和は 1.0 を超えうる・合算して F の内訳とは読まない。

| 項 | 値(us) | F 比 | 判定 |
|---|---|---|---|
| H1 層境界の upload/download 呼び出し全体 | 154.6 | 0.61 | 支持（支配項） |
| H2 未融合 relu の往復 | 89.5 | 0.35 | 寄与あり |
| H3 呼び出しごとのデバイス確保 | 5.9 | 0.02 | 非支持 |
| H4 カーネル実行(syncsplit) | 97.4 | 0.38 | 寄与あり |
| H5 上位 API 固定費(register+残差) | 0.2 | 0.00 | 非支持 |
| H6 mse の往復 | 19.5 | 0.08 | 非支持 |

支配項: H1 層境界の upload/download 呼び出し全体
fidelity(nosync total / prod total): l1=0.98, l2=0.94（範囲内）
層 l1 の支配区間(nosync 内比 0.40 以上): d2h_download
層 l2 の支配区間(nosync 内比 0.40 以上): h2d_upload, d2h_download
補助: decomposed の F 比 l1=0.68 l2=0.23 mse=0.09
補助: what-if(融合 relu) total = 163.5 us / (prod l1 + prod relu = 256.2 us) = 0.64（bits_equal=[True]。record-only）

非後退 ratio（record_only）= public/forward_resident(batch 64) 131.8 us / 155.9 us = 0.85（<= 1.00）。バイナリ（HEAD 対 registry 0.9.0）・ハーネス（テスト対 bench-fandhe）が異なる。

## param_readout（batch 64。単位 us）

| arm/phase | median | min-max |
|---|---|---|
| readout_r0_mirror/param_readout | 78.5 | 73.6-82.5 |
| readout_r1_split/param_readout | 75.1 | 73.5-100.3 |
| readout_r1_split/param_contiguous | 0.4 | 0.4-0.4 |
| readout_r1_split/param_to_vec | 34.7 | 33.0-37.2 |
| readout_r1_split/grad_contiguous | 0.3 | 0.3-0.3 |
| readout_r1_split/grad_to_vec | 36.5 | 34.9-39.5 |
| readout_r1_split/noncontig_grad_count | 0.0（個数） | 0.0-0.0 |
| readout_r2_pretouched/param_readout | 76.7 | 71.2-78.4 |
| readout_r3_cpu_control/param_readout | 47.4 | 46.0-61.8 |

P1 host alloc / page fault: (R0-R2)/R0 = 0.02 -> 非支持
P2 非 contiguous の gather: noncontig_grad_count=0.0・grad_contiguous/R0 = 0.00 -> 非支持
P3 CUDA 由来の差: R0/R3 = 1.65 -> CUDA 由来の差あり
P4 sync 待ち: 構造上 0（測定対象外。RULE.txt）
