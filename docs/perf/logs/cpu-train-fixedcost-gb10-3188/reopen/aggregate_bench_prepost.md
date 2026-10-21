round 順序: 奇数 round は pre→head、偶数 round は head→pre。値は各 round の median_s の 5 round 中央値。

| セル | phase | pre median(us) | pre min-max | head median(us) | head min-max | head/pre | checksum pre/head |
|---|---|---|---|---|---|---|---|
| train_reuse_total | total | 967.9 | 956.3-982.0 | 835.5 | 812.8-842.2 | 0.863 | 一致 |
| train_reuse_phases | tape_build | 0.2 | 0.2-0.3 | 0.2 | 0.2-0.2 | 0.846 | 一致 |
| train_reuse_phases | leaf_register | 0.7 | 0.6-0.8 | 0.8 | 0.8-0.9 | 1.212 | 一致 |
| train_reuse_phases | forward_resident | 221.8 | 214.8-227.3 | 218.7 | 207.3-229.4 | 0.986 | 一致 |
| train_reuse_phases | loss_readout | 0.2 | 0.1-0.2 | 0.2 | 0.1-0.2 | 0.909 | 一致 |
| train_reuse_phases | backward | 449.7 | 437.5-461.4 | 453.0 | 434.5-457.5 | 1.007 | 一致 |
| train_reuse_phases | device_update | 293.9 | 284.6-296.5 | 172.7 | 170.1-174.1 | 0.588 | 一致 |
| train_reuse_phases | tape_drop | 2.1 | 2.0-2.2 | 2.2 | 2.1-2.3 | 1.038 | 一致 |
| train_reuse_phases | step_total | 979.1 | 947.1-1000.8 | 860.7 | 829.9-872.8 | 0.879 | 一致 |
| infer_reuse_total | total | 189.4 | 170.3-209.9 | 192.8 | 172.9-204.2 | 1.018 | 一致 |
| infer_reuse_phases | predict_resident | 187.3 | 179.9-194.6 | 187.9 | 178.1-194.8 | 1.003 | 一致 |
| infer_reuse_phases | host_copy | 0.2 | 0.1-0.2 | 0.2 | 0.1-0.2 | 0.956 | 一致 |
| infer_reuse_phases | checksum | 0.3 | 0.3-0.3 | 0.3 | 0.3-0.3 | 1.000 | 一致 |
| infer_reuse_phases | iter_total | 188.5 | 180.6-195.2 | 189.2 | 178.7-195.5 | 1.004 | 一致 |
| infer_fresh_total | total | 176.4 | 170.2-184.9 | 175.8 | 171.7-186.9 | 0.997 | 一致 |
| infer_fresh_phases | predict | 187.3 | 175.0-193.9 | 179.9 | 161.0-189.3 | 0.960 | 一致 |
| infer_fresh_phases | host_copy | 0.1 | 0.1-0.2 | 0.1 | 0.1-0.1 | 0.941 | 一致 |
| infer_fresh_phases | checksum | 0.3 | 0.3-0.4 | 0.3 | 0.3-0.3 | 0.952 | 一致 |
| infer_fresh_phases | iter_total | 187.9 | 175.7-195.0 | 180.5 | 161.6-189.9 | 0.961 | 一致 |

infer の reuse/fresh 比（total セル。1 未満なら reuse の方が速い）:
- pre: reuse 189.4 us / fresh 176.4 us = 1.074
- head: reuse 192.8 us / fresh 175.8 us = 1.097
