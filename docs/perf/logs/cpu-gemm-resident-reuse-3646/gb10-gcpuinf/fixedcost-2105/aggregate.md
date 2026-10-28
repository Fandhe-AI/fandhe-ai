| arm | phase | median(us) | min-max(us) | q1/q3 median(us) | run 内 min-max(us) | minflt/iter(median) |
|---|---|---|---|---|---|---|
| fresh | predict | 184.0 | 176.0-185.6 | 167.6/197.5 | 128.5-358.7 | null |
| fresh | host_copy | 0.7 | 0.7-0.8 | 0.6/0.9 | 0.4-1.5 | null |
| fresh | checksum | 0.3 | 0.3-0.3 | 0.3/0.5 | 0.3-0.5 | null |
| fresh | iter_total | 185.6 | 177.8-187.1 | 169.6/198.7 | 129.7-360.7 | null |
| reuse | predict | 188.4 | 183.8-190.9 | 173.6/200.1 | 126.0-406.2 | null |
| reuse | host_copy | 0.8 | 0.8-0.9 | 0.7/1.0 | 0.4-2.1 | null |
| reuse | checksum | 0.3 | 0.3-0.4 | 0.3/0.5 | 0.3-0.5 | null |
| reuse | iter_total | 190.8 | 185.6-192.4 | 174.7/203.7 | 127.3-407.6 | null |
| reuse_decomposed | tape_new | 0.1 | 0.1-0.1 | 0.0/0.2 | 0.0-0.6 | null |
| reuse_decomposed | snapshot | 0.7 | 0.7-0.8 | 0.6/0.9 | 0.3-1.7 | null |
| reuse_decomposed | tape_build | 1.1 | 1.1-1.1 | 0.8/1.7 | 0.6-3.1 | null |
| reuse_decomposed | upload | 5.4 | 5.4-5.7 | 3.3/7.3 | 2.3-16.8 | null |
| reuse_decomposed | l1_linear_forward_device | 143.2 | 140.3-144.6 | 134.1/150.2 | 104.1-561.7 | null |
| reuse_decomposed | l2_linear_forward_device | 38.9 | 36.3-41.2 | 27.9/47.1 | 13.8-165.1 | null |
| reuse_decomposed | forward_resident | 185.9 | 183.4-188.5 | 170.8/203.9 | 128.4-618.2 | null |
| reuse_decomposed | readout | 1.0 | 0.9-1.0 | 0.8/1.2 | 0.5-3.1 | null |
| reuse_decomposed | host_copy | 1.0 | 0.9-1.0 | 0.8/1.1 | 0.5-1.8 | null |
| reuse_decomposed | checksum | 0.3 | 0.3-0.3 | 0.3/0.5 | 0.3-0.5 | null |
| reuse_decomposed | iter_total | 190.5 | 188.5-192.4 | 174.5/208.5 | 131.4-623.5 | null |
| reuse_decomposed | chain_public | 189.5 | 185.1-193.5 | 176.7/203.8 | 128.5-436.6 | null |
| fresh_layers | l1_linear_relu_fused | 127.9 | 124.7-131.8 | 119.5/139.2 | 95.0-416.7 | null |
| fresh_layers | l2_linear_gemm_add | 49.9 | 48.6-51.9 | 41.5/57.4 | 16.5-214.6 | null |
| ablation | l1_host_unfused_gemm_add_relu | 164.0 | 159.7-174.3 | 150.4/201.3 | 127.1-388.4 | null |
| ablation | input_to_vec_copy | 5.6 | 5.2-5.7 | 5.0/7.5 | 3.0-18.1 | null |

checksum: run 間一致 OK / reuse_decomposed==reuse OK / fresh==reuse: True
ratio = reuse/fresh iter_total = 1.0280（非後退基準 <= 1.00: 逆転が再現）
diff = 5.2 us

| 区間 | 差分(us) | diff 比 | 判定 |
|---|---|---|---|
| H1 カーネル差(L1) | 15.3 | 2.95 | 支持 |
| L2 差 | -11.0 | -2.11 | - |
| H2 コピー/確保(upload+readout) | 6.4 | 1.23 | 支持 |
| H4 tape 固定費(tape_build+残差-H6) | -1.3 | -0.25 | - |
| H6 chain 検証・tracked 差 | 2.7 | 0.51 | 支持 |

帰属: H1 カーネル差(L1), H2 コピー/確保(upload+readout), H6 chain 検証・tracked 差
