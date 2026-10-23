# cuda 集計（自動生成）

| テスト | run1 | run2 | run3 | run4 | run5 |
|---|---|---|---|---|---|
| relu_mask_device_parity | pass: 5 passed / 0 failed (rc=0) | pass: 5 passed / 0 failed (rc=0) | pass: 5 passed / 0 failed (rc=0) | pass: 5 passed / 0 failed (rc=0) | pass: 5 passed / 0 failed (rc=0) |
| mse_device_parity | pass: 5 passed / 0 failed (rc=0) | pass: 5 passed / 0 failed (rc=0) | pass: 5 passed / 0 failed (rc=0) | pass: 5 passed / 0 failed (rc=0) | pass: 5 passed / 0 failed (rc=0) |
| gemm_device_input_parity | pass: 7 passed / 0 failed (rc=0) | pass: 7 passed / 0 failed (rc=0) | pass: 7 passed / 0 failed (rc=0) | pass: 7 passed / 0 failed (rc=0) | pass: 7 passed / 0 failed (rc=0) |
| gemm_device_input_diag_tests_3223 | pass: 1 passed / 0 failed (rc=0) | pass: 1 passed / 0 failed (rc=0) | pass: 1 passed / 0 failed (rc=0) | pass: 1 passed / 0 failed (rc=0) | pass: 1 passed / 0 failed (rc=0) |
| train_step_parts_sync_diag_tests_3224 | fail: 2 passed / 2 failed (rc=101) | fail: 2 passed / 2 failed (rc=101) | fail: 2 passed / 2 failed (rc=101) | fail: 2 passed / 2 failed (rc=101) | fail: 2 passed / 2 failed (rc=101) |

退避済み invalid 試行: 0 件

総合: fail（または invalid を含む）
