| arm | phase | median(us) | min-max(us) | q1/q3 median(us) | run 内 min-max(us) | minflt/iter(median) |
|---|---|---|---|---|---|---|
| fresh | predict | 189.8 | 187.9-199.9 | 176.8/203.2 | 131.4-527.0 | null |
| fresh | host_copy | 0.7 | 0.7-0.8 | 0.6/0.8 | 0.4-1.3 | null |
| fresh | checksum | 0.5 | 0.3-0.5 | 0.3/0.5 | 0.3-0.5 | null |
| fresh | iter_total | 190.9 | 189.1-201.9 | 178.9/205.8 | 132.8-529.6 | null |
| reuse | predict | 191.0 | 185.7-199.7 | 177.1/204.5 | 130.8-441.2 | null |
| reuse | host_copy | 0.8 | 0.7-0.8 | 0.6/0.9 | 0.3-4.6 | null |
| reuse | checksum | 0.4 | 0.3-0.5 | 0.3/0.5 | 0.3-2.2 | null |
| reuse | iter_total | 193.3 | 187.6-201.2 | 178.5/206.7 | 131.9-443.1 | null |
| reuse_decomposed | tape_new | 0.1 | 0.1-0.2 | 0.0/0.2 | 0.0-0.5 | null |
| reuse_decomposed | snapshot | 0.8 | 0.7-0.9 | 0.6/1.0 | 0.4-1.7 | null |
| reuse_decomposed | tape_build | 1.5 | 1.1-1.6 | 0.9/1.8 | 0.6-2.8 | null |
| reuse_decomposed | upload | 6.5 | 5.3-6.6 | 4.8/7.5 | 2.2-16.9 | null |
| reuse_decomposed | l1_linear_forward_device | 145.8 | 143.6-153.8 | 138.9/153.3 | 114.1-290.1 | null |
| reuse_decomposed | l2_linear_forward_device | 39.0 | 34.7-41.0 | 29.1/47.2 | 14.2-191.5 | null |
| reuse_decomposed | forward_resident | 190.2 | 185.8-201.2 | 175.2/207.6 | 142.1-364.9 | null |
| reuse_decomposed | readout | 0.9 | 0.8-1.0 | 0.7/1.1 | 0.4-3.2 | null |
| reuse_decomposed | host_copy | 0.8 | 0.8-0.9 | 0.7/0.9 | 0.5-1.5 | null |
| reuse_decomposed | checksum | 0.4 | 0.3-0.4 | 0.3/0.5 | 0.3-0.5 | null |
| reuse_decomposed | iter_total | 194.9 | 190.9-206.7 | 178.2/213.0 | 144.8-371.0 | null |
| reuse_decomposed | chain_public | 197.5 | 192.3-204.1 | 181.2/212.5 | 128.0-349.9 | null |
| fresh_layers | l1_linear_relu_fused | 135.8 | 131.2-137.9 | 123.9/145.3 | 96.6-239.9 | null |
| fresh_layers | l2_linear_gemm_add | 48.8 | 48.4-50.2 | 39.8/56.8 | 17.9-103.5 | null |
| ablation | l1_host_unfused_gemm_add_relu | 191.9 | 161.0-209.9 | 154.3/222.7 | 125.5-262.0 | null |
| ablation | input_to_vec_copy | 7.0 | 6.3-7.6 | 5.6/8.5 | 2.8-15.5 | null |

checksum: run 間一致 OK / reuse_decomposed==reuse OK / fresh==reuse: True
ratio = reuse/fresh iter_total = 1.0124（非後退基準 <= 1.00: 逆転が再現）
diff = 2.4 us

| 区間 | 差分(us) | diff 比 | 判定 |
|---|---|---|---|
| H1 カーネル差(L1) | 9.9 | 4.19 | 支持 |
| L2 差 | -9.8 | -4.13 | - |
| H2 コピー/確保(upload+readout) | 7.4 | 3.12 | 支持 |
| H4 tape 固定費(tape_build+残差-H6) | -6.5 | -2.74 | - |
| H6 chain 検証・tracked 差 | 6.4 | 2.70 | 支持 |

帰属: H1 カーネル差(L1), H2 コピー/確保(upload+readout), H6 chain 検証・tracked 差
