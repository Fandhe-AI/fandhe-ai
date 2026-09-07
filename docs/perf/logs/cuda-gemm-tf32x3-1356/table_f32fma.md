### f32fma（REQ-2 判定行） 対 f32_simt

| 形状 | scale | fail/total（5 シード） | max_abs_diff | max_rel_err | max_fail_abs_diff |
|---|---|---|---|---|---|
| 1x1x1 (sub-K-tile, TF32 suite) | 0.1 | 0/5 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 1x1x1 (sub-K-tile, TF32 suite) | 1 | 0/5 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 1x1x1 (sub-K-tile, TF32 suite) | 10 | 0/5 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 1x1x1 (sub-K-tile, TF32 suite) | 100 | 0/5 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 32x32x32 (block tile) | 0.1 | 0/5120 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 32x32x32 (block tile) | 1 | 0/5120 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 32x32x32 (block tile) | 10 | 0/5120 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 32x32x32 (block tile) | 100 | 0/5120 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 64x64x64 (block tile x2) | 0.1 | 0/20480 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 64x64x64 (block tile x2) | 1 | 0/20480 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 64x64x64 (block tile x2) | 10 | 0/20480 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 64x64x64 (block tile x2) | 100 | 0/20480 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 128x128x128 (block tile x4) | 0.1 | 0/81920 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 128x128x128 (block tile x4) | 1 | 0/81920 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 128x128x128 (block tile x4) | 10 | 0/81920 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 128x128x128 (block tile x4) | 100 | 0/81920 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x256 (K sweep base) | 0.1 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x256 (K sweep base) | 1 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x256 (K sweep base) | 10 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x256 (K sweep base) | 100 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 512x512x512 (block tile x16) | 0.1 | 0/1310720 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 512x512x512 (block tile x16) | 1 | 0/1310720 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 512x512x512 (block tile x16) | 10 | 0/1310720 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 512x512x512 (block tile x16) | 100 | 0/1310720 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 17x23x19 (non-multiple edge, TF32 suite) | 0.1 | 0/1955 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 17x23x19 (non-multiple edge, TF32 suite) | 1 | 0/1955 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 17x23x19 (non-multiple edge, TF32 suite) | 10 | 0/1955 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 17x23x19 (non-multiple edge, TF32 suite) | 100 | 0/1955 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 17x19x23 (non-multiple edge, f16 suite) | 0.1 | 0/1615 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 17x19x23 (non-multiple edge, f16 suite) | 1 | 0/1615 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 17x19x23 (non-multiple edge, f16 suite) | 10 | 0/1615 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 17x19x23 (non-multiple edge, f16 suite) | 100 | 0/1615 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 33x31x65 (non-multiple edge) | 0.1 | 0/5115 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 33x31x65 (non-multiple edge) | 1 | 0/5115 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 33x31x65 (non-multiple edge) | 10 | 0/5115 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 33x31x65 (non-multiple edge) | 100 | 0/5115 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 100x100x100 (non-multiple edge) | 0.1 | 0/50000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 100x100x100 (non-multiple edge) | 1 | 0/50000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 100x100x100 (non-multiple edge) | 10 | 0/50000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 100x100x100 (non-multiple edge) | 100 | 0/50000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 130x70x90 (non-multiple edge) | 0.1 | 0/45500 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 130x70x90 (non-multiple edge) | 1 | 0/45500 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 130x70x90 (non-multiple edge) | 10 | 0/45500 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 130x70x90 (non-multiple edge) | 100 | 0/45500 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 64x96x128 (non-square) | 0.1 | 0/30720 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 64x96x128 (non-square) | 1 | 0/30720 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 64x96x128 (non-square) | 10 | 0/30720 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 64x96x128 (non-square) | 100 | 0/30720 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x512 (K sweep) | 0.1 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x512 (K sweep) | 1 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x512 (K sweep) | 10 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x512 (K sweep) | 100 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x1024 (K sweep) | 0.1 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x1024 (K sweep) | 1 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x1024 (K sweep) | 10 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x1024 (K sweep) | 100 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 0.1 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 1 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 10 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 100 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |

### f32fma（REQ-2 判定行） 対 mma_tf32

| 形状 | scale | fail/total（5 シード） | max_abs_diff | max_rel_err | max_fail_abs_diff |
|---|---|---|---|---|---|
| 1x1x1 (sub-K-tile, TF32 suite) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 32x32x32 (block tile) | 0.1 | 176/5120 | 2.211e-05 | 5.585e-01 | 2.211e-05 |
| 32x32x32 (block tile) | 1 | 807/5120 | 1.857e-03 | 2.556e-01 | 1.857e-03 |
| 32x32x32 (block tile) | 10 | 936/5120 | 2.700e-01 | 7.124e-01 | 1.917e-01 |
| 32x32x32 (block tile) | 100 | 945/5120 | 2.031e+01 | 9.061e-01 | 2.031e+01 |
| 64x64x64 (block tile x2) | 0.1 | 1587/20480 | 3.366e-05 | 6.078e-01 | 3.366e-05 |
| 64x64x64 (block tile x2) | 1 | 3374/20480 | 3.377e-03 | 1.046e+00 | 3.377e-03 |
| 64x64x64 (block tile x2) | 10 | 3999/20480 | 3.503e-01 | 1.186e+00 | 3.181e-01 |
| 64x64x64 (block tile x2) | 100 | 3821/20480 | 3.554e+01 | 6.074e-01 | 3.554e+01 |
| 128x128x128 (block tile x4) | 0.1 | 9853/81920 | 5.199e-05 | 1.466e+00 | 5.034e-05 |
| 128x128x128 (block tile x4) | 1 | 13140/81920 | 4.336e-03 | 1.911e+00 | 4.225e-03 |
| 128x128x128 (block tile x4) | 10 | 15521/81920 | 5.544e-01 | 1.949e+00 | 5.061e-01 |
| 128x128x128 (block tile x4) | 100 | 15143/81920 | 4.986e+01 | 1.848e+00 | 4.986e+01 |
| 256x256x256 (K sweep base) | 0.1 | 48557/327680 | 7.097e-05 | 1.958e+00 | 7.097e-05 |
| 256x256x256 (K sweep base) | 1 | 53426/327680 | 6.828e-03 | 1.925e+00 | 6.828e-03 |
| 256x256x256 (K sweep base) | 10 | 62631/327680 | 8.137e-01 | 1.966e+00 | 8.137e-01 |
| 256x256x256 (K sweep base) | 100 | 60426/327680 | 8.010e+01 | 1.910e+00 | 8.010e+01 |
| 512x512x512 (block tile x16) | 0.1 | 214260/1310720 | 1.099e-04 | 1.972e+00 | 1.099e-04 |
| 512x512x512 (block tile x16) | 1 | 212553/1310720 | 9.985e-03 | 1.984e+00 | 9.985e-03 |
| 512x512x512 (block tile x16) | 10 | 249591/1310720 | 1.254e+00 | 1.991e+00 | 1.254e+00 |
| 512x512x512 (block tile x16) | 100 | 239348/1310720 | 1.120e+02 | 1.984e+00 | 1.120e+02 |
| 17x23x19 (non-multiple edge, TF32 suite) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x23x19 (non-multiple edge, TF32 suite) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x23x19 (non-multiple edge, TF32 suite) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x23x19 (non-multiple edge, TF32 suite) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 100x100x100 (non-multiple edge) | 0.1 | 5108/50000 | 4.027e-05 | 1.934e+00 | 4.027e-05 |
| 100x100x100 (non-multiple edge) | 1 | 7956/50000 | 3.625e-03 | 1.884e+00 | 3.625e-03 |
| 100x100x100 (non-multiple edge) | 10 | 9314/50000 | 4.395e-01 | 1.772e+00 | 4.395e-01 |
| 100x100x100 (non-multiple edge) | 100 | 9044/50000 | 4.146e+01 | 1.998e+00 | 4.146e+01 |
| 130x70x90 (non-multiple edge) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 64x96x128 (non-square) | 0.1 | 3586/30720 | 4.596e-05 | 1.290e+00 | 4.596e-05 |
| 64x96x128 (non-square) | 1 | 5067/30720 | 4.179e-03 | 1.743e+00 | 4.178e-03 |
| 64x96x128 (non-square) | 10 | 5966/30720 | 5.252e-01 | 1.489e+00 | 4.692e-01 |
| 64x96x128 (non-square) | 100 | 5676/30720 | 4.828e+01 | 1.848e+00 | 4.828e+01 |
| 256x256x512 (K sweep) | 0.1 | 53660/327680 | 1.083e-04 | 1.911e+00 | 1.011e-04 |
| 256x256x512 (K sweep) | 1 | 52794/327680 | 9.237e-03 | 1.952e+00 | 9.237e-03 |
| 256x256x512 (K sweep) | 10 | 62101/327680 | 1.129e+00 | 1.924e+00 | 1.129e+00 |
| 256x256x512 (K sweep) | 100 | 60069/327680 | 1.085e+02 | 2.000e+00 | 1.085e+02 |
| 256x256x1024 (K sweep) | 0.1 | 56494/327680 | 1.510e-04 | 1.958e+00 | 1.510e-04 |
| 256x256x1024 (K sweep) | 1 | 53474/327680 | 1.296e-02 | 1.891e+00 | 1.296e-02 |
| 256x256x1024 (K sweep) | 10 | 63059/327680 | 1.542e+00 | 1.958e+00 | 1.542e+00 |
| 256x256x1024 (K sweep) | 100 | 60311/327680 | 1.499e+02 | 1.913e+00 | 1.499e+02 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 0.1 | 59400/327680 | 3.074e-04 | 1.961e+00 | 3.074e-04 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 1 | 53251/327680 | 2.532e-02 | 1.969e+00 | 2.532e-02 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 10 | 62807/327680 | 3.103e+00 | 1.973e+00 | 3.103e+00 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 100 | 60229/327680 | 2.910e+02 | 1.930e+00 | 2.910e+02 |

### f32fma（REQ-2 判定行） 対 mma_tf32x3

| 形状 | scale | fail/total（5 シード） | max_abs_diff | max_rel_err | max_fail_abs_diff |
|---|---|---|---|---|---|
| 1x1x1 (sub-K-tile, TF32 suite) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 32x32x32 (block tile) | 0.1 | 0/5120 | 2.608e-08 | 4.990e-04 | 0.000e+00 |
| 32x32x32 (block tile) | 1 | 0/5120 | 2.861e-06 | 5.407e-04 | 0.000e+00 |
| 32x32x32 (block tile) | 10 | 0/5120 | 3.052e-04 | 3.773e-04 | 0.000e+00 |
| 32x32x32 (block tile) | 100 | 0/5120 | 3.125e-02 | 4.888e-04 | 0.000e+00 |
| 64x64x64 (block tile x2) | 0.1 | 0/20480 | 8.196e-08 | 9.666e-03 | 0.000e+00 |
| 64x64x64 (block tile x2) | 1 | 0/20480 | 8.583e-06 | 2.061e-02 | 0.000e+00 |
| 64x64x64 (block tile x2) | 10 | 2/20480 | 7.935e-04 | 1.787e-02 | 1.335e-04 |
| 64x64x64 (block tile x2) | 100 | 2/20480 | 8.594e-02 | 1.613e-02 | 1.055e-02 |
| 128x128x128 (block tile x4) | 0.1 | 0/81920 | 2.086e-07 | 3.396e-02 | 0.000e+00 |
| 128x128x128 (block tile x4) | 1 | 0/81920 | 2.098e-05 | 3.658e-02 | 0.000e+00 |
| 128x128x128 (block tile x4) | 10 | 30/81920 | 2.197e-03 | 2.606e-02 | 5.428e-04 |
| 128x128x128 (block tile x4) | 100 | 25/81920 | 1.875e-01 | 2.110e-02 | 6.610e-02 |
| 256x256x256 (K sweep base) | 0.1 | 0/327680 | 6.109e-07 | 1.112e+00 | 0.000e+00 |
| 256x256x256 (K sweep base) | 1 | 29/327680 | 5.913e-05 | 1.170e+00 | 1.918e-05 |
| 256x256x256 (K sweep base) | 10 | 182/327680 | 6.104e-03 | 9.572e-01 | 1.675e-03 |
| 256x256x256 (K sweep base) | 100 | 163/327680 | 5.781e-01 | 8.429e-01 | 1.738e-01 |
| 512x512x512 (block tile x16) | 0.1 | 0/1310720 | 1.639e-06 | 4.628e-01 | 0.000e+00 |
| 512x512x512 (block tile x16) | 1 | 1123/1310720 | 1.850e-04 | 4.596e-01 | 5.864e-05 |
| 512x512x512 (block tile x16) | 10 | 1488/1310720 | 1.782e-02 | 6.208e-01 | 5.802e-03 |
| 512x512x512 (block tile x16) | 100 | 1503/1310720 | 1.719e+00 | 3.975e-01 | 6.164e-01 |
| 17x23x19 (non-multiple edge, TF32 suite) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x23x19 (non-multiple edge, TF32 suite) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x23x19 (non-multiple edge, TF32 suite) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x23x19 (non-multiple edge, TF32 suite) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 100x100x100 (non-multiple edge) | 0.1 | 0/50000 | 1.714e-07 | 6.221e-03 | 0.000e+00 |
| 100x100x100 (non-multiple edge) | 1 | 0/50000 | 1.717e-05 | 3.473e-03 | 0.000e+00 |
| 100x100x100 (non-multiple edge) | 10 | 11/50000 | 1.587e-03 | 3.961e-03 | 2.884e-04 |
| 100x100x100 (non-multiple edge) | 100 | 13/50000 | 1.562e-01 | 2.801e-03 | 3.311e-02 |
| 130x70x90 (non-multiple edge) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 64x96x128 (non-square) | 0.1 | 0/30720 | 1.788e-07 | 6.972e-03 | 0.000e+00 |
| 64x96x128 (non-square) | 1 | 0/30720 | 2.003e-05 | 9.274e-03 | 0.000e+00 |
| 64x96x128 (non-square) | 10 | 8/30720 | 1.831e-03 | 1.296e-02 | 4.248e-04 |
| 64x96x128 (non-square) | 100 | 9/30720 | 2.031e-01 | 8.149e-03 | 3.948e-02 |
| 256x256x512 (K sweep) | 0.1 | 0/327680 | 1.550e-06 | 7.267e-01 | 0.000e+00 |
| 256x256x512 (K sweep) | 1 | 321/327680 | 1.488e-04 | 6.769e-01 | 6.615e-05 |
| 256x256x512 (K sweep) | 10 | 398/327680 | 1.562e-02 | 9.461e-01 | 5.726e-03 |
| 256x256x512 (K sweep) | 100 | 405/327680 | 1.531e+00 | 7.246e-01 | 5.420e-01 |
| 256x256x1024 (K sweep) | 0.1 | 0/327680 | 4.143e-06 | 2.764e-01 | 0.000e+00 |
| 256x256x1024 (K sweep) | 1 | 757/327680 | 3.815e-04 | 2.175e-01 | 1.619e-04 |
| 256x256x1024 (K sweep) | 10 | 787/327680 | 4.199e-02 | 2.162e-01 | 1.466e-02 |
| 256x256x1024 (K sweep) | 100 | 797/327680 | 4.062e+00 | 2.041e-01 | 1.602e+00 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 0.1 | 27/327680 | 4.214e-05 | 1.710e+00 | 1.299e-05 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 1 | 2881/327680 | 3.807e-03 | 1.487e+00 | 1.213e-03 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 10 | 2902/327680 | 3.799e-01 | 1.549e+00 | 1.178e-01 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 100 | 2876/327680 | 3.812e+01 | 1.423e+00 | 1.309e+01 |

