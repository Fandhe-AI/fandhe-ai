### f64（診断行） 対 f32_simt

| 形状 | scale | fail/total（5 シード） | max_abs_diff | max_rel_err | max_fail_abs_diff |
|---|---|---|---|---|---|
| 1x1x1 (sub-K-tile, TF32 suite) | 0.1 | 0/5 | 1.092e-10 | 4.126e-08 | 0.000e+00 |
| 1x1x1 (sub-K-tile, TF32 suite) | 1 | 0/5 | 1.208e-08 | 2.725e-08 | 0.000e+00 |
| 1x1x1 (sub-K-tile, TF32 suite) | 10 | 0/5 | 1.623e-06 | 2.883e-08 | 0.000e+00 |
| 1x1x1 (sub-K-tile, TF32 suite) | 100 | 0/5 | 7.167e-05 | 4.807e-08 | 0.000e+00 |
| 32x32x32 (block tile) | 0.1 | 0/5120 | 1.261e-08 | 2.205e-04 | 0.000e+00 |
| 32x32x32 (block tile) | 1 | 0/5120 | 1.709e-06 | 1.366e-04 | 0.000e+00 |
| 32x32x32 (block tile) | 10 | 0/5120 | 1.252e-04 | 2.341e-04 | 0.000e+00 |
| 32x32x32 (block tile) | 100 | 0/5120 | 1.267e-02 | 2.023e-04 | 0.000e+00 |
| 64x64x64 (block tile x2) | 0.1 | 0/20480 | 3.560e-08 | 4.792e-03 | 0.000e+00 |
| 64x64x64 (block tile x2) | 1 | 0/20480 | 3.353e-06 | 2.049e-03 | 0.000e+00 |
| 64x64x64 (block tile x2) | 10 | 0/20480 | 2.848e-04 | 7.668e-04 | 0.000e+00 |
| 64x64x64 (block tile x2) | 100 | 0/20480 | 3.126e-02 | 7.294e-04 | 0.000e+00 |
| 128x128x128 (block tile x4) | 0.1 | 0/81920 | 6.522e-08 | 8.904e-03 | 0.000e+00 |
| 128x128x128 (block tile x4) | 1 | 0/81920 | 8.392e-06 | 7.264e-03 | 0.000e+00 |
| 128x128x128 (block tile x4) | 10 | 5/81920 | 7.136e-04 | 3.800e-03 | 1.672e-04 |
| 128x128x128 (block tile x4) | 100 | 4/81920 | 9.034e-02 | 4.274e-03 | 9.585e-03 |
| 256x256x256 (K sweep base) | 0.1 | 0/327680 | 1.565e-07 | 5.514e-01 | 0.000e+00 |
| 256x256x256 (K sweep base) | 1 | 0/327680 | 1.679e-05 | 5.293e-01 | 0.000e+00 |
| 256x256x256 (K sweep base) | 10 | 24/327680 | 1.376e-03 | 1.405e+00 | 3.093e-04 |
| 256x256x256 (K sweep base) | 100 | 26/327680 | 1.443e-01 | 1.892e+00 | 3.969e-02 |
| 512x512x512 (block tile x16) | 0.1 | 0/1310720 | 3.508e-07 | 1.223e-01 | 0.000e+00 |
| 512x512x512 (block tile x16) | 1 | 0/1310720 | 3.869e-05 | 2.040e-01 | 0.000e+00 |
| 512x512x512 (block tile x16) | 10 | 178/1310720 | 3.829e-03 | 2.373e-01 | 8.882e-04 |
| 512x512x512 (block tile x16) | 100 | 176/1310720 | 3.741e-01 | 6.085e-02 | 7.473e-02 |
| 17x23x19 (non-multiple edge, TF32 suite) | 0.1 | 0/1955 | 6.936e-09 | 4.627e-05 | 0.000e+00 |
| 17x23x19 (non-multiple edge, TF32 suite) | 1 | 0/1955 | 6.571e-07 | 8.708e-05 | 0.000e+00 |
| 17x23x19 (non-multiple edge, TF32 suite) | 10 | 0/1955 | 8.803e-05 | 3.553e-05 | 0.000e+00 |
| 17x23x19 (non-multiple edge, TF32 suite) | 100 | 0/1955 | 7.139e-03 | 1.796e-04 | 0.000e+00 |
| 17x19x23 (non-multiple edge, f16 suite) | 0.1 | 0/1615 | 7.809e-09 | 3.148e-04 | 0.000e+00 |
| 17x19x23 (non-multiple edge, f16 suite) | 1 | 0/1615 | 8.413e-07 | 1.323e-03 | 0.000e+00 |
| 17x19x23 (non-multiple edge, f16 suite) | 10 | 0/1615 | 8.600e-05 | 1.530e-04 | 0.000e+00 |
| 17x19x23 (non-multiple edge, f16 suite) | 100 | 0/1615 | 8.840e-03 | 9.275e-04 | 0.000e+00 |
| 33x31x65 (non-multiple edge) | 0.1 | 0/5115 | 3.588e-08 | 3.483e-04 | 0.000e+00 |
| 33x31x65 (non-multiple edge) | 1 | 0/5115 | 2.500e-06 | 3.227e-04 | 0.000e+00 |
| 33x31x65 (non-multiple edge) | 10 | 0/5115 | 2.737e-04 | 6.887e-04 | 0.000e+00 |
| 33x31x65 (non-multiple edge) | 100 | 0/5115 | 2.994e-02 | 2.805e-04 | 0.000e+00 |
| 100x100x100 (non-multiple edge) | 0.1 | 0/50000 | 5.310e-08 | 5.971e-04 | 0.000e+00 |
| 100x100x100 (non-multiple edge) | 1 | 0/50000 | 5.444e-06 | 6.990e-03 | 0.000e+00 |
| 100x100x100 (non-multiple edge) | 10 | 2/50000 | 5.589e-04 | 2.075e-03 | 1.209e-04 |
| 100x100x100 (non-multiple edge) | 100 | 3/50000 | 5.380e-02 | 1.372e-03 | 5.461e-03 |
| 130x70x90 (non-multiple edge) | 0.1 | 0/45500 | 4.696e-08 | 1.931e-02 | 0.000e+00 |
| 130x70x90 (non-multiple edge) | 1 | 0/45500 | 4.464e-06 | 2.252e-02 | 0.000e+00 |
| 130x70x90 (non-multiple edge) | 10 | 3/45500 | 4.901e-04 | 9.234e-02 | 4.256e-05 |
| 130x70x90 (non-multiple edge) | 100 | 5/45500 | 5.382e-02 | 6.841e-02 | 3.645e-03 |
| 64x96x128 (non-square) | 0.1 | 0/30720 | 6.402e-08 | 2.138e-03 | 0.000e+00 |
| 64x96x128 (non-square) | 1 | 0/30720 | 6.107e-06 | 3.104e-03 | 0.000e+00 |
| 64x96x128 (non-square) | 10 | 2/30720 | 5.896e-04 | 2.290e-03 | 9.045e-05 |
| 64x96x128 (non-square) | 100 | 2/30720 | 6.426e-02 | 1.434e-03 | 4.549e-03 |
| 256x256x512 (K sweep) | 0.1 | 0/327680 | 3.391e-07 | 1.064e-01 | 0.000e+00 |
| 256x256x512 (K sweep) | 1 | 0/327680 | 3.164e-05 | 2.769e-01 | 0.000e+00 |
| 256x256x512 (K sweep) | 10 | 37/327680 | 3.654e-03 | 7.928e-01 | 7.714e-04 |
| 256x256x512 (K sweep) | 100 | 38/327680 | 3.548e-01 | 8.898e-02 | 8.464e-02 |
| 256x256x1024 (K sweep) | 0.1 | 0/327680 | 9.328e-07 | 3.924e-02 | 0.000e+00 |
| 256x256x1024 (K sweep) | 1 | 4/327680 | 7.418e-05 | 3.293e-02 | 1.869e-05 |
| 256x256x1024 (K sweep) | 10 | 62/327680 | 6.614e-03 | 3.365e-02 | 1.501e-03 |
| 256x256x1024 (K sweep) | 100 | 52/327680 | 6.134e-01 | 5.058e-02 | 2.503e-01 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 0.1 | 0/327680 | 3.021e-06 | 7.608e-02 | 0.000e+00 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 1 | 71/327680 | 3.425e-04 | 7.938e-02 | 6.691e-05 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 10 | 119/327680 | 2.657e-02 | 5.092e-02 | 7.633e-03 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 100 | 123/327680 | 2.765e+00 | 5.346e-02 | 5.631e-01 |

### f64（診断行） 対 mma_tf32

| 形状 | scale | fail/total（5 シード） | max_abs_diff | max_rel_err | max_fail_abs_diff |
|---|---|---|---|---|---|
| 1x1x1 (sub-K-tile, TF32 suite) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 32x32x32 (block tile) | 0.1 | 176/5120 | 2.211e-05 | 5.585e-01 | 2.211e-05 |
| 32x32x32 (block tile) | 1 | 807/5120 | 1.857e-03 | 2.557e-01 | 1.857e-03 |
| 32x32x32 (block tile) | 10 | 936/5120 | 2.699e-01 | 7.124e-01 | 1.917e-01 |
| 32x32x32 (block tile) | 100 | 945/5120 | 2.032e+01 | 9.062e-01 | 2.032e+01 |
| 64x64x64 (block tile x2) | 0.1 | 1588/20480 | 3.366e-05 | 6.096e-01 | 3.366e-05 |
| 64x64x64 (block tile x2) | 1 | 3373/20480 | 3.377e-03 | 1.046e+00 | 3.377e-03 |
| 64x64x64 (block tile x2) | 10 | 3999/20480 | 3.504e-01 | 1.185e+00 | 3.181e-01 |
| 64x64x64 (block tile x2) | 100 | 3820/20480 | 3.554e+01 | 6.071e-01 | 3.554e+01 |
| 128x128x128 (block tile x4) | 0.1 | 9854/81920 | 5.197e-05 | 1.466e+00 | 5.034e-05 |
| 128x128x128 (block tile x4) | 1 | 13140/81920 | 4.336e-03 | 1.911e+00 | 4.225e-03 |
| 128x128x128 (block tile x4) | 10 | 15518/81920 | 5.542e-01 | 1.949e+00 | 5.060e-01 |
| 128x128x128 (block tile x4) | 100 | 15139/81920 | 4.986e+01 | 1.847e+00 | 4.986e+01 |
| 256x256x256 (K sweep base) | 0.1 | 48555/327680 | 7.097e-05 | 1.957e+00 | 7.097e-05 |
| 256x256x256 (K sweep base) | 1 | 53431/327680 | 6.828e-03 | 1.923e+00 | 6.828e-03 |
| 256x256x256 (K sweep base) | 10 | 62631/327680 | 8.136e-01 | 1.966e+00 | 8.136e-01 |
| 256x256x256 (K sweep base) | 100 | 60429/327680 | 8.010e+01 | 1.908e+00 | 8.010e+01 |
| 512x512x512 (block tile x16) | 0.1 | 214250/1310720 | 1.099e-04 | 1.971e+00 | 1.099e-04 |
| 512x512x512 (block tile x16) | 1 | 212552/1310720 | 9.986e-03 | 1.994e+00 | 9.986e-03 |
| 512x512x512 (block tile x16) | 10 | 249580/1310720 | 1.254e+00 | 1.993e+00 | 1.254e+00 |
| 512x512x512 (block tile x16) | 100 | 239354/1310720 | 1.121e+02 | 1.984e+00 | 1.121e+02 |
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
| 100x100x100 (non-multiple edge) | 0.1 | 5111/50000 | 4.027e-05 | 1.934e+00 | 4.027e-05 |
| 100x100x100 (non-multiple edge) | 1 | 7957/50000 | 3.625e-03 | 1.884e+00 | 3.625e-03 |
| 100x100x100 (non-multiple edge) | 10 | 9315/50000 | 4.394e-01 | 1.772e+00 | 4.394e-01 |
| 100x100x100 (non-multiple edge) | 100 | 9044/50000 | 4.146e+01 | 1.999e+00 | 4.146e+01 |
| 130x70x90 (non-multiple edge) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 64x96x128 (non-square) | 0.1 | 3586/30720 | 4.595e-05 | 1.290e+00 | 4.595e-05 |
| 64x96x128 (non-square) | 1 | 5065/30720 | 4.179e-03 | 1.743e+00 | 4.177e-03 |
| 64x96x128 (non-square) | 10 | 5967/30720 | 5.252e-01 | 1.489e+00 | 4.691e-01 |
| 64x96x128 (non-square) | 100 | 5676/30720 | 4.828e+01 | 1.847e+00 | 4.828e+01 |
| 256x256x512 (K sweep) | 0.1 | 53641/327680 | 1.082e-04 | 1.912e+00 | 1.012e-04 |
| 256x256x512 (K sweep) | 1 | 52800/327680 | 9.237e-03 | 1.951e+00 | 9.237e-03 |
| 256x256x512 (K sweep) | 10 | 62099/327680 | 1.130e+00 | 1.923e+00 | 1.130e+00 |
| 256x256x512 (K sweep) | 100 | 60056/327680 | 1.085e+02 | 2.000e+00 | 1.085e+02 |
| 256x256x1024 (K sweep) | 0.1 | 56497/327680 | 1.510e-04 | 1.958e+00 | 1.510e-04 |
| 256x256x1024 (K sweep) | 1 | 53478/327680 | 1.295e-02 | 1.891e+00 | 1.295e-02 |
| 256x256x1024 (K sweep) | 10 | 63060/327680 | 1.543e+00 | 1.958e+00 | 1.543e+00 |
| 256x256x1024 (K sweep) | 100 | 60312/327680 | 1.499e+02 | 1.902e+00 | 1.499e+02 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 0.1 | 59384/327680 | 3.075e-04 | 1.960e+00 | 3.075e-04 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 1 | 53249/327680 | 2.533e-02 | 1.958e+00 | 2.533e-02 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 10 | 62807/327680 | 3.101e+00 | 1.972e+00 | 3.101e+00 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 100 | 60230/327680 | 2.908e+02 | 1.934e+00 | 2.908e+02 |

### f64（診断行） 対 mma_tf32x3

| 形状 | scale | fail/total（5 シード） | max_abs_diff | max_rel_err | max_fail_abs_diff |
|---|---|---|---|---|---|
| 1x1x1 (sub-K-tile, TF32 suite) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 32x32x32 (block tile) | 0.1 | 0/5120 | 1.998e-08 | 2.786e-04 | 0.000e+00 |
| 32x32x32 (block tile) | 1 | 0/5120 | 2.130e-06 | 4.428e-04 | 0.000e+00 |
| 32x32x32 (block tile) | 10 | 0/5120 | 2.273e-04 | 3.616e-04 | 0.000e+00 |
| 32x32x32 (block tile) | 100 | 0/5120 | 2.777e-02 | 6.910e-04 | 0.000e+00 |
| 64x64x64 (block tile x2) | 0.1 | 0/20480 | 7.454e-08 | 1.441e-02 | 0.000e+00 |
| 64x64x64 (block tile x2) | 1 | 0/20480 | 6.012e-06 | 1.860e-02 | 0.000e+00 |
| 64x64x64 (block tile x2) | 10 | 2/20480 | 6.354e-04 | 1.712e-02 | 1.108e-04 |
| 64x64x64 (block tile x2) | 100 | 2/20480 | 6.429e-02 | 1.685e-02 | 1.091e-02 |
| 128x128x128 (block tile x4) | 0.1 | 0/81920 | 1.858e-07 | 2.529e-02 | 0.000e+00 |
| 128x128x128 (block tile x4) | 1 | 0/81920 | 1.849e-05 | 2.953e-02 | 0.000e+00 |
| 128x128x128 (block tile x4) | 10 | 25/81920 | 1.944e-03 | 2.509e-02 | 3.967e-04 |
| 128x128x128 (block tile x4) | 100 | 26/81920 | 1.606e-01 | 2.528e-02 | 4.984e-02 |
| 256x256x256 (K sweep base) | 0.1 | 0/327680 | 5.433e-07 | 1.164e+00 | 0.000e+00 |
| 256x256x256 (K sweep base) | 1 | 21/327680 | 5.621e-05 | 1.164e+00 | 1.398e-05 |
| 256x256x256 (K sweep base) | 10 | 177/327680 | 5.255e-03 | 1.106e+00 | 1.754e-03 |
| 256x256x256 (K sweep base) | 100 | 174/327680 | 5.471e-01 | 1.140e+00 | 1.751e-01 |
| 512x512x512 (block tile x16) | 0.1 | 0/1310720 | 1.645e-06 | 5.285e-01 | 0.000e+00 |
| 512x512x512 (block tile x16) | 1 | 1120/1310720 | 1.576e-04 | 4.642e-01 | 5.523e-05 |
| 512x512x512 (block tile x16) | 10 | 1490/1310720 | 1.601e-02 | 5.028e-01 | 5.389e-03 |
| 512x512x512 (block tile x16) | 100 | 1499/1310720 | 1.579e+00 | 4.313e-01 | 5.457e-01 |
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
| 100x100x100 (non-multiple edge) | 0.1 | 0/50000 | 1.293e-07 | 6.814e-03 | 0.000e+00 |
| 100x100x100 (non-multiple edge) | 1 | 0/50000 | 1.549e-05 | 5.125e-03 | 0.000e+00 |
| 100x100x100 (non-multiple edge) | 10 | 10/50000 | 1.310e-03 | 3.916e-03 | 2.390e-04 |
| 100x100x100 (non-multiple edge) | 100 | 13/50000 | 1.326e-01 | 3.535e-03 | 2.493e-02 |
| 130x70x90 (non-multiple edge) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 64x96x128 (non-square) | 0.1 | 0/30720 | 1.712e-07 | 7.667e-03 | 0.000e+00 |
| 64x96x128 (non-square) | 1 | 0/30720 | 1.580e-05 | 1.235e-02 | 0.000e+00 |
| 64x96x128 (non-square) | 10 | 8/30720 | 1.688e-03 | 1.248e-02 | 4.340e-04 |
| 64x96x128 (non-square) | 100 | 8/30720 | 1.855e-01 | 7.065e-03 | 3.984e-02 |
| 256x256x512 (K sweep) | 0.1 | 0/327680 | 1.447e-06 | 7.430e-01 | 0.000e+00 |
| 256x256x512 (K sweep) | 1 | 309/327680 | 1.438e-04 | 7.664e-01 | 5.583e-05 |
| 256x256x512 (K sweep) | 10 | 407/327680 | 1.410e-02 | 7.398e-01 | 5.808e-03 |
| 256x256x512 (K sweep) | 100 | 406/327680 | 1.430e+00 | 7.179e-01 | 4.978e-01 |
| 256x256x1024 (K sweep) | 0.1 | 0/327680 | 3.707e-06 | 2.757e-01 | 0.000e+00 |
| 256x256x1024 (K sweep) | 1 | 758/327680 | 3.849e-04 | 2.432e-01 | 1.461e-04 |
| 256x256x1024 (K sweep) | 10 | 787/327680 | 4.282e-02 | 2.412e-01 | 1.382e-02 |
| 256x256x1024 (K sweep) | 100 | 785/327680 | 4.136e+00 | 2.145e-01 | 1.486e+00 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 0.1 | 24/327680 | 4.080e-05 | 1.723e+00 | 1.269e-05 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 1 | 2888/327680 | 3.465e-03 | 1.448e+00 | 1.215e-03 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 10 | 2910/327680 | 3.617e-01 | 1.525e+00 | 1.116e-01 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 100 | 2879/327680 | 3.811e+01 | 1.404e+00 | 1.272e+01 |

