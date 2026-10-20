| arm | phase | median(us) | min-max(us) | q1/q3 median(us) | run 内 min-max(us) | minflt/iter(median) |
|---|---|---|---|---|---|---|
| fresh | predict | 192.2 | 187.3-195.5 | 178.3/204.9 | 133.2-313.6 | null |
| fresh | host_copy | 0.8 | 0.7-0.8 | 0.6/0.8 | 0.4-1.3 | null |
| fresh | checksum | 0.4 | 0.3-0.4 | 0.3/0.5 | 0.3-5.7 | null |
| fresh | iter_total | 193.7 | 189.1-197.3 | 179.9/206.1 | 134.4-314.9 | null |
| reuse | predict | 195.2 | 190.1-197.9 | 179.5/208.5 | 129.6-375.6 | null |
| reuse | host_copy | 0.7 | 0.7-0.8 | 0.6/0.9 | 0.3-1.5 | null |
| reuse | checksum | 0.4 | 0.4-0.5 | 0.3/0.5 | 0.3-0.5 | null |
| reuse | iter_total | 197.2 | 192.2-199.4 | 181.2/209.7 | 130.7-378.0 | null |
| reuse_decomposed | tape_new | 0.1 | 0.1-0.2 | 0.1/0.2 | 0.0-0.4 | null |
| reuse_decomposed | snapshot | 0.8 | 0.8-0.8 | 0.6/1.0 | 0.3-4.1 | null |
| reuse_decomposed | tape_build | 1.4 | 1.2-1.5 | 0.9/1.8 | 0.5-11.8 | null |
| reuse_decomposed | upload | 6.1 | 5.7-6.8 | 4.8/7.3 | 2.2-15.3 | null |
| reuse_decomposed | l1_linear_forward_device | 146.0 | 144.1-150.0 | 138.0/155.7 | 104.5-480.6 | null |
| reuse_decomposed | l2_linear_forward_device | 41.4 | 38.4-42.6 | 29.3/47.9 | 13.6-82.6 | null |
| reuse_decomposed | forward_resident | 195.4 | 189.8-197.6 | 180.4/208.7 | 137.3-557.2 | null |
| reuse_decomposed | readout | 0.8 | 0.8-0.9 | 0.7/1.1 | 0.3-4.4 | null |
| reuse_decomposed | host_copy | 0.8 | 0.8-0.9 | 0.7/1.0 | 0.5-1.9 | null |
| reuse_decomposed | checksum | 0.4 | 0.3-0.4 | 0.3/0.5 | 0.3-0.5 | null |
| reuse_decomposed | iter_total | 198.7 | 194.2-202.7 | 183.5/212.5 | 140.2-562.2 | null |
| reuse_decomposed | chain_public | 196.0 | 189.7-199.8 | 183.6/211.0 | 131.4-1744.9 | null |
| fresh_layers | l1_linear_relu_fused | 133.9 | 130.2-134.5 | 123.3/141.7 | 96.1-270.4 | null |
| fresh_layers | l2_linear_gemm_add | 50.3 | 48.5-50.8 | 40.4/56.5 | 18.8-257.3 | null |
| ablation | l1_host_unfused_gemm_add_relu | 198.1 | 182.4-204.3 | 162.2/220.5 | 137.7-267.5 | null |
| ablation | input_to_vec_copy | 6.7 | 6.4-7.5 | 5.7/8.8 | 3.0-23.5 | null |

checksum: run 間一致 OK / reuse_decomposed==reuse OK / fresh==reuse: True
ratio = reuse/fresh iter_total = 1.0179（非後退基準 <= 1.00: 逆転が再現）
diff = 3.5 us

| 区間 | 差分(us) | diff 比 | 判定 |
|---|---|---|---|
| H1 カーネル差(L1) | 12.1 | 3.48 | 支持 |
| L2 差 | -8.9 | -2.58 | - |
| H2 コピー/確保(upload+readout) | 7.0 | 2.01 | 支持 |
| H4 tape 固定費(tape_build+残差-H6) | -0.0 | -0.00 | - |
| H6 chain 検証・tracked 差 | -0.2 | -0.05 | - |

帰属: H1 カーネル差(L1), H2 コピー/確保(upload+readout)
