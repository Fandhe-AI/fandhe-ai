| arm | phase | median(us) | min-max(us) |
|---|---|---|---|
| fresh_off | tape_build | 10.5 | 10.2-11.2 |
| fresh_off | leaf_register | 0.1 | 0.1-0.1 |
| fresh_off | forward | 550.0 | 546.2-556.7 |
| fresh_off | to_tensor | 179.0 | 178.6-180.2 |
| fresh_off | host_copy | 0.2 | 0.1-0.2 |
| fresh_off | checksum | 2.3 | 2.3-2.3 |
| fresh_off | iter_total | 741.7 | 738.4-748.2 |
| fresh_on | tape_build | 0.0 | 0.0-0.0 |
| fresh_on | leaf_register | 0.1 | 0.1-0.1 |
| fresh_on | forward | 548.9 | 545.4-555.2 |
| fresh_on | to_tensor | 178.9 | 177.8-180.2 |
| fresh_on | host_copy | 0.2 | 0.1-0.2 |
| fresh_on | checksum | 2.3 | 2.3-2.3 |
| fresh_on | iter_total | 731.5 | 728.2-739.8 |
| reuse_off | predict_resident | 321.3 | 319.8-324.0 |
| reuse_off | host_copy | 0.2 | 0.2-0.2 |
| reuse_off | checksum | 2.3 | 2.3-2.3 |
| reuse_off | iter_total | 324.0 | 322.5-326.8 |
| reuse_on | predict_resident | 309.1 | 307.6-312.9 |
| reuse_on | host_copy | 0.2 | 0.2-0.2 |
| reuse_on | checksum | 2.3 | 2.3-2.3 |
| reuse_on | iter_total | 311.8 | 310.2-315.7 |
| reuse_decomposed_off | provider_select | 10.6 | 10.5-11.0 |
| reuse_decomposed_off | tape_new | 0.0 | 0.0-0.0 |
| reuse_decomposed_off | tape_build | 10.8 | 10.6-11.1 |
| reuse_decomposed_off | snapshot | 0.2 | 0.2-0.2 |
| reuse_decomposed_off | chain | 308.1 | 307.4-312.3 |
| reuse_decomposed_off | host_copy | 0.2 | 0.2-0.2 |
| reuse_decomposed_off | checksum | 2.3 | 2.3-2.3 |
| reuse_decomposed_off | iter_total | 321.8 | 320.8-326.4 |
| reuse_decomposed_on | provider_select | 0.0 | 0.0-0.0 |
| reuse_decomposed_on | tape_new | 0.0 | 0.0-0.0 |
| reuse_decomposed_on | tape_build | 0.1 | 0.1-0.1 |
| reuse_decomposed_on | snapshot | 0.2 | 0.2-0.2 |
| reuse_decomposed_on | chain | 309.8 | 308.2-314.5 |
| reuse_decomposed_on | host_copy | 0.2 | 0.2-0.2 |
| reuse_decomposed_on | checksum | 2.3 | 2.3-2.3 |
| reuse_decomposed_on | iter_total | 312.7 | 311.2-317.5 |
| tape_build_micro | probe_gpu_core_count | 9.2 | 9.2-9.8 |
| tape_build_micro | provider_select | 8.4 | 8.3-8.8 |
| tape_build_micro | tape_new | 0.0 | 0.0-0.0 |
| tape_build_micro | tape_for | 8.2 | 8.1-8.5 |

| arm | counter | run 間 median | run 間 min-max |
|---|---|---|---|
| fresh_off | encode_per_iter | 3.000 | 3.000-3.000 |
| fresh_off | command_buffers_per_iter | 3.000 | 3.000-3.000 |
| fresh_off | wait_per_iter | 3.000 | 3.000-3.000 |
| fresh_off | verify_probe_per_iter | 1.000 | 1.000-1.000 |
| fresh_off | verify_hit_per_iter | 0.000 | 0.000-0.000 |
| fresh_off | uploads_per_iter | 0.000 | 0.000-0.000 |
| fresh_off | upload_bytes_per_iter | 0.000 | 0.000-0.000 |
| fresh_off | downloads_per_iter | 0.000 | 0.000-0.000 |
| fresh_off | download_bytes_per_iter | 0.000 | 0.000-0.000 |
| fresh_on | encode_per_iter | 3.000 | 3.000-3.000 |
| fresh_on | command_buffers_per_iter | 3.000 | 3.000-3.000 |
| fresh_on | wait_per_iter | 3.000 | 3.000-3.000 |
| fresh_on | verify_probe_per_iter | 0.000 | 0.000-0.000 |
| fresh_on | verify_hit_per_iter | 1.000 | 1.000-1.000 |
| fresh_on | uploads_per_iter | 0.000 | 0.000-0.000 |
| fresh_on | upload_bytes_per_iter | 0.000 | 0.000-0.000 |
| fresh_on | downloads_per_iter | 0.000 | 0.000-0.000 |
| fresh_on | download_bytes_per_iter | 0.000 | 0.000-0.000 |
| reuse_off | encode_per_iter | 2.000 | 2.000-2.000 |
| reuse_off | command_buffers_per_iter | 1.000 | 1.000-1.000 |
| reuse_off | wait_per_iter | 1.000 | 1.000-1.000 |
| reuse_off | verify_probe_per_iter | 1.000 | 1.000-1.000 |
| reuse_off | verify_hit_per_iter | 0.000 | 0.000-0.000 |
| reuse_off | uploads_per_iter | 1.000 | 1.000-1.000 |
| reuse_off | upload_bytes_per_iter | 200704.000 | 200704.000-200704.000 |
| reuse_off | downloads_per_iter | 1.000 | 1.000-1.000 |
| reuse_off | download_bytes_per_iter | 2560.000 | 2560.000-2560.000 |
| reuse_on | encode_per_iter | 2.000 | 2.000-2.000 |
| reuse_on | command_buffers_per_iter | 1.000 | 1.000-1.000 |
| reuse_on | wait_per_iter | 1.000 | 1.000-1.000 |
| reuse_on | verify_probe_per_iter | 0.000 | 0.000-0.000 |
| reuse_on | verify_hit_per_iter | 1.000 | 1.000-1.000 |
| reuse_on | uploads_per_iter | 1.000 | 1.000-1.000 |
| reuse_on | upload_bytes_per_iter | 200704.000 | 200704.000-200704.000 |
| reuse_on | downloads_per_iter | 1.000 | 1.000-1.000 |
| reuse_on | download_bytes_per_iter | 2560.000 | 2560.000-2560.000 |
| reuse_decomposed_off | encode_per_iter | 2.000 | 2.000-2.000 |
| reuse_decomposed_off | command_buffers_per_iter | 1.000 | 1.000-1.000 |
| reuse_decomposed_off | wait_per_iter | 1.000 | 1.000-1.000 |
| reuse_decomposed_off | verify_probe_per_iter | 1.000 | 1.000-1.000 |
| reuse_decomposed_off | verify_hit_per_iter | 0.000 | 0.000-0.000 |
| reuse_decomposed_off | uploads_per_iter | 1.000 | 1.000-1.000 |
| reuse_decomposed_off | upload_bytes_per_iter | 200704.000 | 200704.000-200704.000 |
| reuse_decomposed_off | downloads_per_iter | 1.000 | 1.000-1.000 |
| reuse_decomposed_off | download_bytes_per_iter | 2560.000 | 2560.000-2560.000 |
| reuse_decomposed_on | encode_per_iter | 2.000 | 2.000-2.000 |
| reuse_decomposed_on | command_buffers_per_iter | 1.000 | 1.000-1.000 |
| reuse_decomposed_on | wait_per_iter | 1.000 | 1.000-1.000 |
| reuse_decomposed_on | verify_probe_per_iter | 0.000 | 0.000-0.000 |
| reuse_decomposed_on | verify_hit_per_iter | 1.000 | 1.000-1.000 |
| reuse_decomposed_on | uploads_per_iter | 1.000 | 1.000-1.000 |
| reuse_decomposed_on | upload_bytes_per_iter | 200704.000 | 200704.000-200704.000 |
| reuse_decomposed_on | downloads_per_iter | 1.000 | 1.000-1.000 |
| reuse_decomposed_on | download_bytes_per_iter | 2560.000 | 2560.000-2560.000 |

checksum: run 間一致 OK / off==on OK / reuse_decomposed==reuse OK / fresh==reuse: False
ratio_tape_build_fresh (on/off) = 0.0039（非後退基準 <= 1.00: 満たす。記録のみ）
ratio_tape_build_decomposed (on/off) = 0.0078（非後退基準 <= 1.00: 満たす。記録のみ）
ratio_iter_fresh (on/off) = 0.9862（非後退基準 <= 1.00: 満たす。記録のみ）
ratio_iter_reuse (on/off) = 0.9623（非後退基準 <= 1.00: 満たす。記録のみ）

H1: micro.provider_select / fresh_off.tape_build = 0.80 -> 支持（閾値 0.5）
H1a（補助）: probe_gpu_core_count / provider_select = 1.10（残り -0.10 が MTLCopyAllDevices・name 等の推定）
H2: micro.tape_new = 0.0 us -> 支持（閾値 1 us）
H3: reuse_off の encode=2・command_buffers=1・wait=1（全 run） -> 支持
H4: (c) の wall - GPU busy は手動転記で判定（集計対象外）
