# candle-parity-tolerance-baseline-impact 生出力（イシュー #1238）

`--scale-mode exact`。行数: 45（実測値。`ParityPath` 別内訳は本文 doc 参照）。

## 行別 入力規模（M = S_A・S_B）

| path | context | m | n | k | seed | dtype | S_A | S_B | M=S_A*S_B |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| WmmaTf32 | wmma_tf32 32x32x32 seed=2000 | 32 | 32 | 32 | 0x7D0 | f32 | 0.999650 | 0.998510 | 0.998161 |
| WmmaTf32 | wmma_tf32 256x256x4096 seed=8888 (PoC-v2-5 stress) | 256 | 256 | 4096 | 0x22B8 | f32 | 0.999997 | 0.999999 | 0.999996 |
| WmmaTf32 | wmma_tf32 64x64x64 seed=2001 (#1106 diagnostic) | 64 | 64 | 64 | 0x7D1 | f32 | 0.999927 | 0.999929 | 0.999855 |
| WmmaTf32 | wmma_tf32 128x128x128 seed=2002 (#1106 diagnostic) | 128 | 128 | 128 | 0x7D2 | f32 | 0.999962 | 0.999881 | 0.999843 |
| WmmaTf32 | wmma_tf32 512x512x512 seed=2003 (#1106 diagnostic) | 512 | 512 | 512 | 0x7D3 | f32 | 0.999998 | 1.000000 | 0.999998 |
| WmmaTf32 | wmma_tf32 64x96x128 seed=2004 (#1106 diagnostic) | 64 | 96 | 128 | 0x7D4 | f32 | 0.999948 | 0.999971 | 0.999918 |
| WmmaTf32 | wmma_tf32 17x23x19 seed=2006 (#1106 diagnostic) | 17 | 23 | 19 | 0x7D6 | f32 | 0.995926 | 0.994091 | 0.990041 |
| WmmaTf32 | wmma_tf32 33x31x65 seed=2007 (#1106 diagnostic) | 33 | 31 | 65 | 0x7D7 | f32 | 0.999388 | 0.999993 | 0.999381 |
| WmmaTf32 | wmma_tf32 512x512x4096 seed=0xFACADE (#1106 diagnostic) | 512 | 512 | 4096 | 0xFACADE | f32 | 1.000000 | 1.000000 | 1.000000 |
| WmmaTf32Opt | wmma_tf32_opt 512x512x512 seed=0x7A0 (tensor_core_parity_record) | 512 | 512 | 512 | 0x7A0 | f32 | 0.999999 | 1.000000 | 0.999999 |
| WmmaTf32Opt | wmma_tf32_opt 64x64x64 seed=3000 | 64 | 64 | 64 | 0xBB8 | f32 | 0.999291 | 0.999688 | 0.998979 |
| WmmaTf32Opt | wmma_tf32_opt 512x512x4096 seed=0xC0FFEE | 512 | 512 | 4096 | 0xC0FFEE | f32 | 0.999999 | 1.000000 | 0.999999 |
| WmmaTf32Opt | wmma_tf32_opt 128x128x128 seed=0xBB9 (#1106 diagnostic) | 128 | 128 | 128 | 0xBB9 | f32 | 0.999973 | 0.999807 | 0.999780 |
| WmmaTf32Opt | wmma_tf32_opt 512x512x512 seed=0xBBA (#1106 diagnostic) | 512 | 512 | 512 | 0xBBA | f32 | 0.999989 | 0.999998 | 0.999987 |
| WmmaTf32Opt | wmma_tf32_opt 63x65x33 seed=0xBBB (#1106 diagnostic) | 63 | 65 | 33 | 0xBBB | f32 | 0.999071 | 0.999870 | 0.998941 |
| WmmaTf32Opt | wmma_tf32_opt 65x63x17 seed=0xBBC (#1106 diagnostic) | 65 | 63 | 17 | 0xBBC | f32 | 0.999149 | 0.999391 | 0.998540 |
| WmmaTf32Opt | wmma_tf32_opt 64x96x256 seed=0xBBD (#1106 diagnostic) | 64 | 96 | 256 | 0xBBD | f32 | 0.999890 | 0.999936 | 0.999826 |
| WmmaTf32Opt | wmma_tf32_opt 4096x4096x4096 seed=0xBEEF (#1106 diagnostic) | 4096 | 4096 | 4096 | 0xBEEF | f32 | 1.000000 | 1.000000 | 1.000000 |
| WmmaTf32Staged | wmma_tf32_staged 512x512x4096 seed=0xC0FFEE | 512 | 512 | 4096 | 0xC0FFEE | f32 | 0.999999 | 1.000000 | 0.999999 |
| WmmaTf32Staged | wmma_tf32_staged 64x64x64 seed=0xFA0 (#1106 diagnostic) | 64 | 64 | 64 | 0xFA0 | f32 | 0.999907 | 0.999955 | 0.999862 |
| WmmaTf32Staged | wmma_tf32_staged 128x128x128 seed=0xFA1 (#1106 diagnostic) | 128 | 128 | 128 | 0xFA1 | f32 | 0.999882 | 0.999836 | 0.999719 |
| WmmaTf32Staged | wmma_tf32_staged 512x512x512 seed=0xFA2 (#1106 diagnostic) | 512 | 512 | 512 | 0xFA2 | f32 | 0.999995 | 0.999996 | 0.999992 |
| WmmaTf32Staged | wmma_tf32_staged 60x68x36 seed=0xFA3 (#1106 diagnostic) | 60 | 68 | 36 | 0xFA3 | f32 | 0.999535 | 0.999941 | 0.999476 |
| WmmaTf32Staged | wmma_tf32_staged 68x60x20 seed=0xFA4 (#1106 diagnostic) | 68 | 60 | 20 | 0xFA4 | f32 | 0.999576 | 0.999902 | 0.999478 |
| WmmaTf32Staged | wmma_tf32_staged 64x96x256 seed=0xFA5 (#1106 diagnostic) | 64 | 96 | 256 | 0xFA5 | f32 | 0.999954 | 0.999992 | 0.999946 |
| WmmaTf32Staged | wmma_tf32_staged 4096x4096x4096 seed=0xBEEF (#1106 diagnostic) | 4096 | 4096 | 4096 | 0xBEEF | f32 | 1.000000 | 1.000000 | 1.000000 |
| MmaF16 | mma_f16 256x256x4096 seed=9999 | 256 | 256 | 4096 | 0x270F | f16 | 1.000000 | 1.000000 | 1.000000 |
| WmmaF16 | wmma_f16 256x256x4096 seed=8888 (run_f16 effective route; GB10: opt) | 256 | 256 | 4096 | 0x22B8 | f16 | 1.000000 | 1.000000 | 1.000000 |
| WmmaF16 | wmma_f16 256x256x4096 seed=8889 (run_f16 effective route; GB10: opt) | 256 | 256 | 4096 | 0x22B9 | f16 | 1.000000 | 1.000000 | 1.000000 |
| MmaTf32 | mma_tf32 16x8x8 seed=5000 (#1122 triage) | 16 | 8 | 8 | 0x1388 | f32 | 0.994913 | 0.989194 | 0.984162 |
| MmaTf32 | mma_tf32 64x64x64 seed=5001 (#1122 triage) | 64 | 64 | 64 | 0x1389 | f32 | 0.999850 | 0.999716 | 0.999566 |
| MmaTf32 | mma_tf32 128x128x128 seed=5002 (#1122 triage) | 128 | 128 | 128 | 0x138A | f32 | 0.999839 | 0.999900 | 0.999740 |
| MmaTf32 | mma_tf32 512x512x512 seed=5003 (#1122 triage) | 512 | 512 | 512 | 0x138B | f32 | 0.999994 | 0.999997 | 0.999991 |
| MmaTf32 | mma_tf32 60x68x36 seed=5004 (#1122 triage) | 60 | 68 | 36 | 0x138C | f32 | 0.999471 | 0.999984 | 0.999455 |
| MmaTf32 | mma_tf32 68x60x20 seed=5005 (#1122 triage) | 68 | 60 | 20 | 0x138D | f32 | 0.999539 | 0.999505 | 0.999044 |
| MmaTf32 | mma_tf32 96x68x72 seed=5006 (#1122 triage) | 96 | 68 | 72 | 0x138E | f32 | 0.999961 | 0.999812 | 0.999772 |
| MmaTf32 | mma_tf32 64x96x256 seed=5007 (#1122 triage) | 64 | 96 | 256 | 0x138F | f32 | 0.999880 | 0.999987 | 0.999867 |
| MmaTf32 | mma_tf32 4x4x4 seed=5008 (#1122 triage) | 4 | 4 | 4 | 0x1390 | f32 | 0.985476 | 0.791864 | 0.780363 |
| MmaTf32 | mma_tf32 64x64x64 seed=1 (#1122 smoke_env_adaptive) | 64 | 64 | 64 | 0x1 | f32 | 0.999794 | 0.999856 | 0.999651 |
| MmaTf32 | mma_tf32 4096x4096x4096 seed=9001 (#1122 triage k4096_stress) | 4096 | 4096 | 4096 | 0x2329 | f32 | 1.000000 | 1.000000 | 1.000000 |
| MmaTf32VsWmmaStaged | mma_tf32_vs_wmma_tf32_staged 512x512x512 seed=6002 (#1122 triage) | 512 | 512 | 512 | 0x1772 | f32 | 0.999986 | 0.999997 | 0.999982 |
| MmaTf32VsWmmaStaged | mma_tf32_vs_wmma_tf32_staged 4096x4096x4096 seed=9002 (#1122 triage k4096_stress) | 4096 | 4096 | 4096 | 0x232A | f32 | 1.000000 | 1.000000 | 1.000000 |
| SpecializedMmaF16 | specialized_mma_f16 256x512x1024 seed=4003 compiled=DYNAMIC_ALL (#1159 sweep) | 256 | 512 | 1024 | 0xFA3 | f16 | 1.000000 | 1.000000 | 1.000000 |
| SpecializedMmaF16 | specialized_mma_f16 256x512x1024 seed=4003 compiled=STATIC_NK (#1159 sweep) | 256 | 512 | 1024 | 0xFA3 | f16 | 1.000000 | 1.000000 | 1.000000 |
| SpecializedMmaF16 | specialized_mma_f16 256x512x1024 seed=4003 compiled=STATIC_MNK (#1159 sweep) | 256 | 512 | 1024 | 0xFA3 | f16 | 1.000000 | 1.000000 | 1.000000 |

## 候補 A（スケール付き絶対誤差。M は上表の行別値を使用）

各セルは `bound/分類`（分類: no-op／全救済／部分／未確定／分類不能）。

| 候補 | row0 | row1 | row2 | row3 | row4 | row5 | row6 | row7 | row8 | row9 | row10 | row11 | row12 | row13 | row14 | row15 | row16 | row17 | row18 | row19 | row20 | row21 | row22 | row23 | row24 | row25 | row26 | row27 | row28 | row29 | row30 | row31 | row32 | row33 | row34 | row35 | row36 | row37 | row38 | row39 | row40 | row41 | row42 | row43 | row44 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A c=0.125 eps=2^-23 K*M | 4.760e-07/no-op | 6.103e-05/部分／未確定 | 9.535e-07/no-op | 1.907e-06/no-op | 7.629e-06/no-op | 1.907e-06/no-op | 2.803e-07/no-op | 9.680e-07/no-op | 6.104e-05/部分／未確定 | 7.629e-06/no-op | 9.527e-07/no-op | 6.104e-05/部分／未確定 | 1.907e-06/no-op | 7.629e-06/no-op | 4.912e-07/no-op | 2.529e-07/no-op | 3.814e-06/no-op | 6.104e-05/部分／未確定 | 6.104e-05/部分／未確定 | 9.535e-07/no-op | 1.907e-06/no-op | 7.629e-06/no-op | 5.362e-07/no-op | 2.979e-07/no-op | 3.814e-06/no-op | 6.104e-05/部分／未確定 | 6.104e-05/部分／未確定 | 6.104e-05/部分／未確定 | 6.104e-05/部分／未確定 | 1.173e-07/no-op | 9.533e-07/no-op | 1.907e-06/no-op | 7.629e-06/no-op | 5.361e-07/no-op | 2.977e-07/no-op | 1.073e-06/no-op | 3.814e-06/no-op | 4.651e-08/no-op | 9.533e-07/no-op | 6.104e-05/部分／未確定 | 7.629e-06/no-op | 6.104e-05/部分／未確定 | 1.526e-05/部分／未確定 | 1.526e-05/部分／未確定 | 1.526e-05/部分／未確定 |
| A c=0.25 eps=2^-23 K*M | 9.519e-07/no-op | 1.221e-04/部分／未確定 | 1.907e-06/no-op | 3.814e-06/no-op | 1.526e-05/部分／未確定 | 3.814e-06/no-op | 5.606e-07/no-op | 1.936e-06/no-op | 1.221e-04/部分／未確定 | 1.526e-05/分類不能（ceiling 未実測） | 1.905e-06/no-op | 1.221e-04/部分／未確定 | 3.814e-06/no-op | 1.526e-05/部分／未確定 | 9.824e-07/no-op | 5.059e-07/no-op | 7.628e-06/no-op | 1.221e-04/部分／未確定 | 1.221e-04/部分／未確定 | 1.907e-06/no-op | 3.814e-06/no-op | 1.526e-05/部分／未確定 | 1.072e-06/no-op | 5.957e-07/no-op | 7.629e-06/no-op | 1.221e-04/部分／未確定 | 1.221e-04/部分／未確定 | 1.221e-04/部分／未確定 | 1.221e-04/部分／未確定 | 2.346e-07/no-op | 1.907e-06/no-op | 3.814e-06/no-op | 1.526e-05/部分／未確定 | 1.072e-06/no-op | 5.955e-07/no-op | 2.145e-06/no-op | 7.628e-06/no-op | 9.303e-08/no-op | 1.907e-06/no-op | 1.221e-04/部分／未確定 | 1.526e-05/部分／未確定 | 1.221e-04/部分／未確定 | 3.052e-05/部分／未確定 | 3.052e-05/部分／未確定 | 3.052e-05/部分／未確定 |
| A c=0.5 eps=2^-23 K*M | 1.904e-06/no-op | 2.441e-04/部分／未確定 | 3.814e-06/no-op | 7.628e-06/no-op | 3.052e-05/部分／未確定 | 7.629e-06/no-op | 1.121e-06/no-op | 3.872e-06/no-op | 2.441e-04/部分／未確定 | 3.052e-05/分類不能（ceiling 未実測） | 3.811e-06/no-op | 2.441e-04/部分／未確定 | 7.628e-06/no-op | 3.052e-05/部分／未確定 | 1.965e-06/no-op | 1.012e-06/no-op | 1.526e-05/部分／未確定 | 2.441e-04/部分／未確定 | 2.441e-04/部分／未確定 | 3.814e-06/no-op | 7.627e-06/no-op | 3.052e-05/部分／未確定 | 2.145e-06/no-op | 1.191e-06/no-op | 1.526e-05/部分／未確定 | 2.441e-04/部分／未確定 | 2.441e-04/部分／未確定 | 2.441e-04/部分／未確定 | 2.441e-04/部分／未確定 | 4.693e-07/no-op | 3.813e-06/no-op | 7.627e-06/no-op | 3.052e-05/部分／未確定 | 2.145e-06/no-op | 1.191e-06/no-op | 4.291e-06/no-op | 1.526e-05/部分／未確定 | 1.861e-07/no-op | 3.813e-06/no-op | 2.441e-04/部分／未確定 | 3.052e-05/部分／未確定 | 2.441e-04/部分／未確定 | 6.104e-05/部分／未確定 | 6.104e-05/部分／未確定 | 6.104e-05/部分／未確定 |
| A c=1.0 eps=2^-23 K*M | 3.808e-06/no-op | 4.883e-04/部分／未確定 | 7.628e-06/no-op | 1.526e-05/部分／未確定 | 6.104e-05/部分／未確定 | 1.526e-05/部分／未確定 | 2.242e-06/no-op | 7.744e-06/no-op | 4.883e-04/部分／未確定 | 6.104e-05/分類不能（ceiling 未実測） | 7.622e-06/no-op | 4.883e-04/部分／未確定 | 1.526e-05/部分／未確定 | 6.103e-05/部分／未確定 | 3.930e-06/no-op | 2.024e-06/no-op | 3.051e-05/部分／未確定 | 4.883e-04/部分／未確定 | 4.883e-04/部分／未確定 | 7.628e-06/no-op | 1.525e-05/部分／未確定 | 6.103e-05/部分／未確定 | 4.289e-06/no-op | 2.383e-06/no-op | 3.052e-05/部分／未確定 | 4.883e-04/部分／未確定 | 4.883e-04/部分／未確定 | 4.883e-04/部分／未確定 | 4.883e-04/部分／未確定 | 9.386e-07/no-op | 7.626e-06/no-op | 1.525e-05/部分／未確定 | 6.103e-05/部分／未確定 | 4.289e-06/no-op | 2.382e-06/no-op | 8.581e-06/no-op | 3.051e-05/部分／未確定 | 3.721e-07/no-op | 7.627e-06/no-op | 4.883e-04/部分／未確定 | 6.103e-05/部分／未確定 | 4.883e-04/部分／未確定 | 1.221e-04/部分／未確定 | 1.221e-04/部分／未確定 | 1.221e-04/部分／未確定 |
| A c=2.0 eps=2^-23 K*M | 7.615e-06/no-op | 9.766e-04/部分／未確定 | 1.526e-05/部分／未確定 | 3.051e-05/部分／未確定 | 1.221e-04/部分／未確定 | 3.052e-05/部分／未確定 | 4.485e-06/no-op | 1.549e-05/部分／未確定 | 9.766e-04/部分／未確定 | 1.221e-04/分類不能（ceiling 未実測） | 1.524e-05/部分／未確定 | 9.766e-04/部分／未確定 | 3.051e-05/部分／未確定 | 1.221e-04/部分／未確定 | 7.859e-06/no-op | 4.047e-06/no-op | 6.102e-05/部分／未確定 | 9.766e-04/部分／未確定 | 9.766e-04/部分／未確定 | 1.526e-05/部分／未確定 | 3.051e-05/部分／未確定 | 1.221e-04/部分／未確定 | 8.579e-06/no-op | 4.766e-06/no-op | 6.103e-05/部分／未確定 | 9.766e-04/部分／未確定 | 9.766e-04/部分／未確定 | 9.766e-04/部分／未確定 | 9.766e-04/部分／未確定 | 1.877e-06/no-op | 1.525e-05/部分／未確定 | 3.051e-05/部分／未確定 | 1.221e-04/部分／未確定 | 8.578e-06/no-op | 4.764e-06/no-op | 1.716e-05/部分／未確定 | 6.103e-05/部分／未確定 | 7.442e-07/no-op | 1.525e-05/部分／未確定 | 9.766e-04/部分／未確定 | 1.221e-04/全救済 | 9.766e-04/部分／未確定 | 2.441e-04/部分／未確定 | 2.441e-04/部分／未確定 | 2.441e-04/部分／未確定 |
| A c=0.125 eps=2^-24 K*M | 2.380e-07/no-op | 3.052e-05/部分／未確定 | 4.768e-07/no-op | 9.535e-07/no-op | 3.815e-06/no-op | 9.536e-07/no-op | 1.402e-07/no-op | 4.840e-07/no-op | 3.052e-05/部分／未確定 | 3.815e-06/no-op | 4.764e-07/no-op | 3.052e-05/部分／未確定 | 9.535e-07/no-op | 3.815e-06/no-op | 2.456e-07/no-op | 1.265e-07/no-op | 1.907e-06/no-op | 3.052e-05/部分／未確定 | 3.052e-05/部分／未確定 | 4.768e-07/no-op | 9.534e-07/no-op | 3.815e-06/no-op | 2.681e-07/no-op | 1.489e-07/no-op | 1.907e-06/no-op | 3.052e-05/部分／未確定 | 3.052e-05/部分／未確定 | 3.052e-05/部分／未確定 | 3.052e-05/部分／未確定 | 5.866e-08/no-op | 4.766e-07/no-op | 9.534e-07/no-op | 3.815e-06/no-op | 2.681e-07/no-op | 1.489e-07/no-op | 5.363e-07/no-op | 1.907e-06/no-op | 2.326e-08/no-op | 4.767e-07/no-op | 3.052e-05/部分／未確定 | 3.815e-06/no-op | 3.052e-05/部分／未確定 | 7.629e-06/no-op | 7.629e-06/no-op | 7.629e-06/no-op |
| A c=0.25 eps=2^-24 K*M | 4.760e-07/no-op | 6.103e-05/部分／未確定 | 9.535e-07/no-op | 1.907e-06/no-op | 7.629e-06/no-op | 1.907e-06/no-op | 2.803e-07/no-op | 9.680e-07/no-op | 6.104e-05/部分／未確定 | 7.629e-06/no-op | 9.527e-07/no-op | 6.104e-05/部分／未確定 | 1.907e-06/no-op | 7.629e-06/no-op | 4.912e-07/no-op | 2.529e-07/no-op | 3.814e-06/no-op | 6.104e-05/部分／未確定 | 6.104e-05/部分／未確定 | 9.535e-07/no-op | 1.907e-06/no-op | 7.629e-06/no-op | 5.362e-07/no-op | 2.979e-07/no-op | 3.814e-06/no-op | 6.104e-05/部分／未確定 | 6.104e-05/部分／未確定 | 6.104e-05/部分／未確定 | 6.104e-05/部分／未確定 | 1.173e-07/no-op | 9.533e-07/no-op | 1.907e-06/no-op | 7.629e-06/no-op | 5.361e-07/no-op | 2.977e-07/no-op | 1.073e-06/no-op | 3.814e-06/no-op | 4.651e-08/no-op | 9.533e-07/no-op | 6.104e-05/部分／未確定 | 7.629e-06/no-op | 6.104e-05/部分／未確定 | 1.526e-05/部分／未確定 | 1.526e-05/部分／未確定 | 1.526e-05/部分／未確定 |
| A c=0.5 eps=2^-24 K*M | 9.519e-07/no-op | 1.221e-04/部分／未確定 | 1.907e-06/no-op | 3.814e-06/no-op | 1.526e-05/部分／未確定 | 3.814e-06/no-op | 5.606e-07/no-op | 1.936e-06/no-op | 1.221e-04/部分／未確定 | 1.526e-05/分類不能（ceiling 未実測） | 1.905e-06/no-op | 1.221e-04/部分／未確定 | 3.814e-06/no-op | 1.526e-05/部分／未確定 | 9.824e-07/no-op | 5.059e-07/no-op | 7.628e-06/no-op | 1.221e-04/部分／未確定 | 1.221e-04/部分／未確定 | 1.907e-06/no-op | 3.814e-06/no-op | 1.526e-05/部分／未確定 | 1.072e-06/no-op | 5.957e-07/no-op | 7.629e-06/no-op | 1.221e-04/部分／未確定 | 1.221e-04/部分／未確定 | 1.221e-04/部分／未確定 | 1.221e-04/部分／未確定 | 2.346e-07/no-op | 1.907e-06/no-op | 3.814e-06/no-op | 1.526e-05/部分／未確定 | 1.072e-06/no-op | 5.955e-07/no-op | 2.145e-06/no-op | 7.628e-06/no-op | 9.303e-08/no-op | 1.907e-06/no-op | 1.221e-04/部分／未確定 | 1.526e-05/部分／未確定 | 1.221e-04/部分／未確定 | 3.052e-05/部分／未確定 | 3.052e-05/部分／未確定 | 3.052e-05/部分／未確定 |
| A c=1.0 eps=2^-24 K*M | 1.904e-06/no-op | 2.441e-04/部分／未確定 | 3.814e-06/no-op | 7.628e-06/no-op | 3.052e-05/部分／未確定 | 7.629e-06/no-op | 1.121e-06/no-op | 3.872e-06/no-op | 2.441e-04/部分／未確定 | 3.052e-05/分類不能（ceiling 未実測） | 3.811e-06/no-op | 2.441e-04/部分／未確定 | 7.628e-06/no-op | 3.052e-05/部分／未確定 | 1.965e-06/no-op | 1.012e-06/no-op | 1.526e-05/部分／未確定 | 2.441e-04/部分／未確定 | 2.441e-04/部分／未確定 | 3.814e-06/no-op | 7.627e-06/no-op | 3.052e-05/部分／未確定 | 2.145e-06/no-op | 1.191e-06/no-op | 1.526e-05/部分／未確定 | 2.441e-04/部分／未確定 | 2.441e-04/部分／未確定 | 2.441e-04/部分／未確定 | 2.441e-04/部分／未確定 | 4.693e-07/no-op | 3.813e-06/no-op | 7.627e-06/no-op | 3.052e-05/部分／未確定 | 2.145e-06/no-op | 1.191e-06/no-op | 4.291e-06/no-op | 1.526e-05/部分／未確定 | 1.861e-07/no-op | 3.813e-06/no-op | 2.441e-04/部分／未確定 | 3.052e-05/部分／未確定 | 2.441e-04/部分／未確定 | 6.104e-05/部分／未確定 | 6.104e-05/部分／未確定 | 6.104e-05/部分／未確定 |
| A c=2.0 eps=2^-24 K*M | 3.808e-06/no-op | 4.883e-04/部分／未確定 | 7.628e-06/no-op | 1.526e-05/部分／未確定 | 6.104e-05/部分／未確定 | 1.526e-05/部分／未確定 | 2.242e-06/no-op | 7.744e-06/no-op | 4.883e-04/部分／未確定 | 6.104e-05/分類不能（ceiling 未実測） | 7.622e-06/no-op | 4.883e-04/部分／未確定 | 1.526e-05/部分／未確定 | 6.103e-05/部分／未確定 | 3.930e-06/no-op | 2.024e-06/no-op | 3.051e-05/部分／未確定 | 4.883e-04/部分／未確定 | 4.883e-04/部分／未確定 | 7.628e-06/no-op | 1.525e-05/部分／未確定 | 6.103e-05/部分／未確定 | 4.289e-06/no-op | 2.383e-06/no-op | 3.052e-05/部分／未確定 | 4.883e-04/部分／未確定 | 4.883e-04/部分／未確定 | 4.883e-04/部分／未確定 | 4.883e-04/部分／未確定 | 9.386e-07/no-op | 7.626e-06/no-op | 1.525e-05/部分／未確定 | 6.103e-05/部分／未確定 | 4.289e-06/no-op | 2.382e-06/no-op | 8.581e-06/no-op | 3.051e-05/部分／未確定 | 3.721e-07/no-op | 7.627e-06/no-op | 4.883e-04/部分／未確定 | 6.103e-05/部分／未確定 | 4.883e-04/部分／未確定 | 1.221e-04/部分／未確定 | 1.221e-04/部分／未確定 | 1.221e-04/部分／未確定 |
| A c=1.0 eps=2^-23 sqrtK*M | 6.731e-07/no-op | 7.629e-06/no-op | 9.535e-07/no-op | 1.348e-06/no-op | 2.697e-06/no-op | 1.349e-06/no-op | 5.144e-07/no-op | 9.605e-07/no-op | 7.629e-06/no-op | 2.697e-06/no-op | 9.527e-07/no-op | 7.629e-06/no-op | 1.348e-06/no-op | 2.697e-06/no-op | 6.841e-07/no-op | 4.908e-07/no-op | 1.907e-06/no-op | 7.629e-06/no-op | 7.629e-06/no-op | 9.535e-07/no-op | 1.348e-06/no-op | 2.697e-06/no-op | 7.149e-07/no-op | 5.328e-07/no-op | 1.907e-06/no-op | 7.629e-06/no-op | 7.629e-06/no-op | 7.629e-06/no-op | 7.629e-06/no-op | 3.318e-07/no-op | 9.533e-07/no-op | 1.348e-06/no-op | 2.697e-06/no-op | 7.149e-07/no-op | 5.326e-07/no-op | 1.011e-06/no-op | 1.907e-06/no-op | 1.861e-07/no-op | 9.533e-07/no-op | 7.629e-06/no-op | 2.697e-06/no-op | 7.629e-06/no-op | 3.815e-06/no-op | 3.815e-06/no-op | 3.815e-06/no-op |

## 候補 A-4（緩すぎる参考値。K*u*(K*S_A*S_B) 上界）

| row | bound | class |
|---|---:|---|
| 0 (wmma_tf32 32x32x32 seed=2000) | 6.092e-05 | 部分／未確定 |
| 1 (wmma_tf32 256x256x4096 seed=8888 (PoC-v2-5 stress)) | 1.000e+00 | 部分／未確定 |
| 2 (wmma_tf32 64x64x64 seed=2001 (#1106 diagnostic)) | 2.441e-04 | 部分／未確定 |
| 3 (wmma_tf32 128x128x128 seed=2002 (#1106 diagnostic)) | 9.764e-04 | 部分／未確定 |
| 4 (wmma_tf32 512x512x512 seed=2003 (#1106 diagnostic)) | 1.562e-02 | 部分／未確定 |
| 5 (wmma_tf32 64x96x128 seed=2004 (#1106 diagnostic)) | 9.765e-04 | 部分／未確定 |
| 6 (wmma_tf32 17x23x19 seed=2006 (#1106 diagnostic)) | 2.130e-05 | 部分／未確定 |
| 7 (wmma_tf32 33x31x65 seed=2007 (#1106 diagnostic)) | 2.517e-04 | 部分／未確定 |
| 8 (wmma_tf32 512x512x4096 seed=0xFACADE (#1106 diagnostic)) | 1.000e+00 | 部分／未確定 |
| 9 (wmma_tf32_opt 512x512x512 seed=0x7A0 (tensor_core_parity_record)) | 1.562e-02 | 分類不能（ceiling 未実測） |
| 10 (wmma_tf32_opt 64x64x64 seed=3000) | 2.439e-04 | 部分／未確定 |
| 11 (wmma_tf32_opt 512x512x4096 seed=0xC0FFEE) | 1.000e+00 | 部分／未確定 |
| 12 (wmma_tf32_opt 128x128x128 seed=0xBB9 (#1106 diagnostic)) | 9.763e-04 | 部分／未確定 |
| 13 (wmma_tf32_opt 512x512x512 seed=0xBBA (#1106 diagnostic)) | 1.562e-02 | 部分／未確定 |
| 14 (wmma_tf32_opt 63x65x33 seed=0xBBB (#1106 diagnostic)) | 6.484e-05 | 部分／未確定 |
| 15 (wmma_tf32_opt 65x63x17 seed=0xBBC (#1106 diagnostic)) | 1.720e-05 | 部分／未確定 |
| 16 (wmma_tf32_opt 64x96x256 seed=0xBBD (#1106 diagnostic)) | 3.906e-03 | 部分／未確定 |
| 17 (wmma_tf32_opt 4096x4096x4096 seed=0xBEEF (#1106 diagnostic)) | 1.000e+00 | 部分／未確定 |
| 18 (wmma_tf32_staged 512x512x4096 seed=0xC0FFEE) | 1.000e+00 | 部分／未確定 |
| 19 (wmma_tf32_staged 64x64x64 seed=0xFA0 (#1106 diagnostic)) | 2.441e-04 | 部分／未確定 |
| 20 (wmma_tf32_staged 128x128x128 seed=0xFA1 (#1106 diagnostic)) | 9.763e-04 | 部分／未確定 |
| 21 (wmma_tf32_staged 512x512x512 seed=0xFA2 (#1106 diagnostic)) | 1.562e-02 | 部分／未確定 |
| 22 (wmma_tf32_staged 60x68x36 seed=0xFA3 (#1106 diagnostic)) | 7.721e-05 | 部分／未確定 |
| 23 (wmma_tf32_staged 68x60x20 seed=0xFA4 (#1106 diagnostic)) | 2.383e-05 | 部分／未確定 |
| 24 (wmma_tf32_staged 64x96x256 seed=0xFA5 (#1106 diagnostic)) | 3.906e-03 | 部分／未確定 |
| 25 (wmma_tf32_staged 4096x4096x4096 seed=0xBEEF (#1106 diagnostic)) | 1.000e+00 | 部分／未確定 |
| 26 (mma_f16 256x256x4096 seed=9999) | 1.000e+00 | 部分／未確定 |
| 27 (wmma_f16 256x256x4096 seed=8888 (run_f16 effective route; GB10: opt)) | 1.000e+00 | 部分／未確定 |
| 28 (wmma_f16 256x256x4096 seed=8889 (run_f16 effective route; GB10: opt)) | 1.000e+00 | 部分／未確定 |
| 29 (mma_tf32 16x8x8 seed=5000 (#1122 triage)) | 3.754e-06 | no-op |
| 30 (mma_tf32 64x64x64 seed=5001 (#1122 triage)) | 2.440e-04 | 部分／未確定 |
| 31 (mma_tf32 128x128x128 seed=5002 (#1122 triage)) | 9.763e-04 | 部分／未確定 |
| 32 (mma_tf32 512x512x512 seed=5003 (#1122 triage)) | 1.562e-02 | 部分／未確定 |
| 33 (mma_tf32 60x68x36 seed=5004 (#1122 triage)) | 7.721e-05 | 部分／未確定 |
| 34 (mma_tf32 68x60x20 seed=5005 (#1122 triage)) | 2.382e-05 | 部分／未確定 |
| 35 (mma_tf32 96x68x72 seed=5006 (#1122 triage)) | 3.089e-04 | 部分／未確定 |
| 36 (mma_tf32 64x96x256 seed=5007 (#1122 triage)) | 3.906e-03 | 部分／未確定 |
| 37 (mma_tf32 4x4x4 seed=5008 (#1122 triage)) | 7.442e-07 | no-op |
| 38 (mma_tf32 64x64x64 seed=1 (#1122 smoke_env_adaptive)) | 2.441e-04 | 部分／未確定 |
| 39 (mma_tf32 4096x4096x4096 seed=9001 (#1122 triage k4096_stress)) | 1.000e+00 | 部分／未確定 |
| 40 (mma_tf32_vs_wmma_tf32_staged 512x512x512 seed=6002 (#1122 triage)) | 1.562e-02 | 部分／未確定 |
| 41 (mma_tf32_vs_wmma_tf32_staged 4096x4096x4096 seed=9002 (#1122 triage k4096_stress)) | 1.000e+00 | 部分／未確定 |
| 42 (specialized_mma_f16 256x512x1024 seed=4003 compiled=DYNAMIC_ALL (#1159 sweep)) | 6.250e-02 | 部分／未確定 |
| 43 (specialized_mma_f16 256x512x1024 seed=4003 compiled=STATIC_NK (#1159 sweep)) | 6.250e-02 | 部分／未確定 |
| 44 (specialized_mma_f16 256x512x1024 seed=4003 compiled=STATIC_MNK (#1159 sweep)) | 6.250e-02 | 部分／未確定 |

## 候補 B-2（ULP ベース。t*ulp(K*S_A*S_B) 上界）

| t | row | bound | class |
|---:|---|---:|---|
| 1 | 0 (wmma_tf32 32x32x32 seed=2000) | 1.907e-06 | no-op |
| 1 | 1 (wmma_tf32 256x256x4096 seed=8888 (PoC-v2-5 stress)) | 2.441e-04 | 部分／未確定 |
| 1 | 2 (wmma_tf32 64x64x64 seed=2001 (#1106 diagnostic)) | 3.815e-06 | no-op |
| 1 | 3 (wmma_tf32 128x128x128 seed=2002 (#1106 diagnostic)) | 7.629e-06 | no-op |
| 1 | 4 (wmma_tf32 512x512x512 seed=2003 (#1106 diagnostic)) | 3.052e-05 | 部分／未確定 |
| 1 | 5 (wmma_tf32 64x96x128 seed=2004 (#1106 diagnostic)) | 7.629e-06 | no-op |
| 1 | 6 (wmma_tf32 17x23x19 seed=2006 (#1106 diagnostic)) | 1.907e-06 | no-op |
| 1 | 7 (wmma_tf32 33x31x65 seed=2007 (#1106 diagnostic)) | 7.629e-06 | no-op |
| 1 | 8 (wmma_tf32 512x512x4096 seed=0xFACADE (#1106 diagnostic)) | 2.441e-04 | 部分／未確定 |
| 1 | 9 (wmma_tf32_opt 512x512x512 seed=0x7A0 (tensor_core_parity_record)) | 3.052e-05 | 分類不能（ceiling 未実測） |
| 1 | 10 (wmma_tf32_opt 64x64x64 seed=3000) | 3.815e-06 | no-op |
| 1 | 11 (wmma_tf32_opt 512x512x4096 seed=0xC0FFEE) | 2.441e-04 | 部分／未確定 |
| 1 | 12 (wmma_tf32_opt 128x128x128 seed=0xBB9 (#1106 diagnostic)) | 7.629e-06 | no-op |
| 1 | 13 (wmma_tf32_opt 512x512x512 seed=0xBBA (#1106 diagnostic)) | 3.052e-05 | 部分／未確定 |
| 1 | 14 (wmma_tf32_opt 63x65x33 seed=0xBBB (#1106 diagnostic)) | 3.815e-06 | no-op |
| 1 | 15 (wmma_tf32_opt 65x63x17 seed=0xBBC (#1106 diagnostic)) | 1.907e-06 | no-op |
| 1 | 16 (wmma_tf32_opt 64x96x256 seed=0xBBD (#1106 diagnostic)) | 1.526e-05 | 部分／未確定 |
| 1 | 17 (wmma_tf32_opt 4096x4096x4096 seed=0xBEEF (#1106 diagnostic)) | 4.883e-04 | 部分／未確定 |
| 1 | 18 (wmma_tf32_staged 512x512x4096 seed=0xC0FFEE) | 2.441e-04 | 部分／未確定 |
| 1 | 19 (wmma_tf32_staged 64x64x64 seed=0xFA0 (#1106 diagnostic)) | 3.815e-06 | no-op |
| 1 | 20 (wmma_tf32_staged 128x128x128 seed=0xFA1 (#1106 diagnostic)) | 7.629e-06 | no-op |
| 1 | 21 (wmma_tf32_staged 512x512x512 seed=0xFA2 (#1106 diagnostic)) | 3.052e-05 | 部分／未確定 |
| 1 | 22 (wmma_tf32_staged 60x68x36 seed=0xFA3 (#1106 diagnostic)) | 3.815e-06 | no-op |
| 1 | 23 (wmma_tf32_staged 68x60x20 seed=0xFA4 (#1106 diagnostic)) | 1.907e-06 | no-op |
| 1 | 24 (wmma_tf32_staged 64x96x256 seed=0xFA5 (#1106 diagnostic)) | 1.526e-05 | 部分／未確定 |
| 1 | 25 (wmma_tf32_staged 4096x4096x4096 seed=0xBEEF (#1106 diagnostic)) | 4.883e-04 | 部分／未確定 |
| 1 | 26 (mma_f16 256x256x4096 seed=9999) | 4.883e-04 | 部分／未確定 |
| 1 | 27 (wmma_f16 256x256x4096 seed=8888 (run_f16 effective route; GB10: opt)) | 4.883e-04 | 部分／未確定 |
| 1 | 28 (wmma_f16 256x256x4096 seed=8889 (run_f16 effective route; GB10: opt)) | 4.883e-04 | 部分／未確定 |
| 1 | 29 (mma_tf32 16x8x8 seed=5000 (#1122 triage)) | 4.768e-07 | no-op |
| 1 | 30 (mma_tf32 64x64x64 seed=5001 (#1122 triage)) | 3.815e-06 | no-op |
| 1 | 31 (mma_tf32 128x128x128 seed=5002 (#1122 triage)) | 7.629e-06 | no-op |
| 1 | 32 (mma_tf32 512x512x512 seed=5003 (#1122 triage)) | 3.052e-05 | 部分／未確定 |
| 1 | 33 (mma_tf32 60x68x36 seed=5004 (#1122 triage)) | 3.815e-06 | no-op |
| 1 | 34 (mma_tf32 68x60x20 seed=5005 (#1122 triage)) | 1.907e-06 | no-op |
| 1 | 35 (mma_tf32 96x68x72 seed=5006 (#1122 triage)) | 7.629e-06 | no-op |
| 1 | 36 (mma_tf32 64x96x256 seed=5007 (#1122 triage)) | 1.526e-05 | 部分／未確定 |
| 1 | 37 (mma_tf32 4x4x4 seed=5008 (#1122 triage)) | 2.384e-07 | no-op |
| 1 | 38 (mma_tf32 64x64x64 seed=1 (#1122 smoke_env_adaptive)) | 3.815e-06 | no-op |
| 1 | 39 (mma_tf32 4096x4096x4096 seed=9001 (#1122 triage k4096_stress)) | 2.441e-04 | 部分／未確定 |
| 1 | 40 (mma_tf32_vs_wmma_tf32_staged 512x512x512 seed=6002 (#1122 triage)) | 3.052e-05 | 部分／未確定 |
| 1 | 41 (mma_tf32_vs_wmma_tf32_staged 4096x4096x4096 seed=9002 (#1122 triage k4096_stress)) | 2.441e-04 | 部分／未確定 |
| 1 | 42 (specialized_mma_f16 256x512x1024 seed=4003 compiled=DYNAMIC_ALL (#1159 sweep)) | 1.221e-04 | 部分／未確定 |
| 1 | 43 (specialized_mma_f16 256x512x1024 seed=4003 compiled=STATIC_NK (#1159 sweep)) | 1.221e-04 | 部分／未確定 |
| 1 | 44 (specialized_mma_f16 256x512x1024 seed=4003 compiled=STATIC_MNK (#1159 sweep)) | 1.221e-04 | 部分／未確定 |
| 2 | 0 (wmma_tf32 32x32x32 seed=2000) | 3.815e-06 | no-op |
| 2 | 1 (wmma_tf32 256x256x4096 seed=8888 (PoC-v2-5 stress)) | 4.883e-04 | 部分／未確定 |
| 2 | 2 (wmma_tf32 64x64x64 seed=2001 (#1106 diagnostic)) | 7.629e-06 | no-op |
| 2 | 3 (wmma_tf32 128x128x128 seed=2002 (#1106 diagnostic)) | 1.526e-05 | 部分／未確定 |
| 2 | 4 (wmma_tf32 512x512x512 seed=2003 (#1106 diagnostic)) | 6.104e-05 | 部分／未確定 |
| 2 | 5 (wmma_tf32 64x96x128 seed=2004 (#1106 diagnostic)) | 1.526e-05 | 部分／未確定 |
| 2 | 6 (wmma_tf32 17x23x19 seed=2006 (#1106 diagnostic)) | 3.815e-06 | no-op |
| 2 | 7 (wmma_tf32 33x31x65 seed=2007 (#1106 diagnostic)) | 1.526e-05 | 部分／未確定 |
| 2 | 8 (wmma_tf32 512x512x4096 seed=0xFACADE (#1106 diagnostic)) | 4.883e-04 | 部分／未確定 |
| 2 | 9 (wmma_tf32_opt 512x512x512 seed=0x7A0 (tensor_core_parity_record)) | 6.104e-05 | 分類不能（ceiling 未実測） |
| 2 | 10 (wmma_tf32_opt 64x64x64 seed=3000) | 7.629e-06 | no-op |
| 2 | 11 (wmma_tf32_opt 512x512x4096 seed=0xC0FFEE) | 4.883e-04 | 部分／未確定 |
| 2 | 12 (wmma_tf32_opt 128x128x128 seed=0xBB9 (#1106 diagnostic)) | 1.526e-05 | 部分／未確定 |
| 2 | 13 (wmma_tf32_opt 512x512x512 seed=0xBBA (#1106 diagnostic)) | 6.104e-05 | 部分／未確定 |
| 2 | 14 (wmma_tf32_opt 63x65x33 seed=0xBBB (#1106 diagnostic)) | 7.629e-06 | no-op |
| 2 | 15 (wmma_tf32_opt 65x63x17 seed=0xBBC (#1106 diagnostic)) | 3.815e-06 | no-op |
| 2 | 16 (wmma_tf32_opt 64x96x256 seed=0xBBD (#1106 diagnostic)) | 3.052e-05 | 部分／未確定 |
| 2 | 17 (wmma_tf32_opt 4096x4096x4096 seed=0xBEEF (#1106 diagnostic)) | 9.766e-04 | 部分／未確定 |
| 2 | 18 (wmma_tf32_staged 512x512x4096 seed=0xC0FFEE) | 4.883e-04 | 部分／未確定 |
| 2 | 19 (wmma_tf32_staged 64x64x64 seed=0xFA0 (#1106 diagnostic)) | 7.629e-06 | no-op |
| 2 | 20 (wmma_tf32_staged 128x128x128 seed=0xFA1 (#1106 diagnostic)) | 1.526e-05 | 部分／未確定 |
| 2 | 21 (wmma_tf32_staged 512x512x512 seed=0xFA2 (#1106 diagnostic)) | 6.104e-05 | 部分／未確定 |
| 2 | 22 (wmma_tf32_staged 60x68x36 seed=0xFA3 (#1106 diagnostic)) | 7.629e-06 | no-op |
| 2 | 23 (wmma_tf32_staged 68x60x20 seed=0xFA4 (#1106 diagnostic)) | 3.815e-06 | no-op |
| 2 | 24 (wmma_tf32_staged 64x96x256 seed=0xFA5 (#1106 diagnostic)) | 3.052e-05 | 部分／未確定 |
| 2 | 25 (wmma_tf32_staged 4096x4096x4096 seed=0xBEEF (#1106 diagnostic)) | 9.766e-04 | 部分／未確定 |
| 2 | 26 (mma_f16 256x256x4096 seed=9999) | 9.766e-04 | 部分／未確定 |
| 2 | 27 (wmma_f16 256x256x4096 seed=8888 (run_f16 effective route; GB10: opt)) | 9.766e-04 | 部分／未確定 |
| 2 | 28 (wmma_f16 256x256x4096 seed=8889 (run_f16 effective route; GB10: opt)) | 9.766e-04 | 部分／未確定 |
| 2 | 29 (mma_tf32 16x8x8 seed=5000 (#1122 triage)) | 9.537e-07 | no-op |
| 2 | 30 (mma_tf32 64x64x64 seed=5001 (#1122 triage)) | 7.629e-06 | no-op |
| 2 | 31 (mma_tf32 128x128x128 seed=5002 (#1122 triage)) | 1.526e-05 | 部分／未確定 |
| 2 | 32 (mma_tf32 512x512x512 seed=5003 (#1122 triage)) | 6.104e-05 | 部分／未確定 |
| 2 | 33 (mma_tf32 60x68x36 seed=5004 (#1122 triage)) | 7.629e-06 | no-op |
| 2 | 34 (mma_tf32 68x60x20 seed=5005 (#1122 triage)) | 3.815e-06 | no-op |
| 2 | 35 (mma_tf32 96x68x72 seed=5006 (#1122 triage)) | 1.526e-05 | 部分／未確定 |
| 2 | 36 (mma_tf32 64x96x256 seed=5007 (#1122 triage)) | 3.052e-05 | 部分／未確定 |
| 2 | 37 (mma_tf32 4x4x4 seed=5008 (#1122 triage)) | 4.768e-07 | no-op |
| 2 | 38 (mma_tf32 64x64x64 seed=1 (#1122 smoke_env_adaptive)) | 7.629e-06 | no-op |
| 2 | 39 (mma_tf32 4096x4096x4096 seed=9001 (#1122 triage k4096_stress)) | 4.883e-04 | 部分／未確定 |
| 2 | 40 (mma_tf32_vs_wmma_tf32_staged 512x512x512 seed=6002 (#1122 triage)) | 6.104e-05 | 部分／未確定 |
| 2 | 41 (mma_tf32_vs_wmma_tf32_staged 4096x4096x4096 seed=9002 (#1122 triage k4096_stress)) | 4.883e-04 | 部分／未確定 |
| 2 | 42 (specialized_mma_f16 256x512x1024 seed=4003 compiled=DYNAMIC_ALL (#1159 sweep)) | 2.441e-04 | 部分／未確定 |
| 2 | 43 (specialized_mma_f16 256x512x1024 seed=4003 compiled=STATIC_NK (#1159 sweep)) | 2.441e-04 | 部分／未確定 |
| 2 | 44 (specialized_mma_f16 256x512x1024 seed=4003 compiled=STATIC_MNK (#1159 sweep)) | 2.441e-04 | 部分／未確定 |
| 4 | 0 (wmma_tf32 32x32x32 seed=2000) | 7.629e-06 | no-op |
| 4 | 1 (wmma_tf32 256x256x4096 seed=8888 (PoC-v2-5 stress)) | 9.766e-04 | 部分／未確定 |
| 4 | 2 (wmma_tf32 64x64x64 seed=2001 (#1106 diagnostic)) | 1.526e-05 | 部分／未確定 |
| 4 | 3 (wmma_tf32 128x128x128 seed=2002 (#1106 diagnostic)) | 3.052e-05 | 部分／未確定 |
| 4 | 4 (wmma_tf32 512x512x512 seed=2003 (#1106 diagnostic)) | 1.221e-04 | 部分／未確定 |
| 4 | 5 (wmma_tf32 64x96x128 seed=2004 (#1106 diagnostic)) | 3.052e-05 | 部分／未確定 |
| 4 | 6 (wmma_tf32 17x23x19 seed=2006 (#1106 diagnostic)) | 7.629e-06 | no-op |
| 4 | 7 (wmma_tf32 33x31x65 seed=2007 (#1106 diagnostic)) | 3.052e-05 | 部分／未確定 |
| 4 | 8 (wmma_tf32 512x512x4096 seed=0xFACADE (#1106 diagnostic)) | 9.766e-04 | 部分／未確定 |
| 4 | 9 (wmma_tf32_opt 512x512x512 seed=0x7A0 (tensor_core_parity_record)) | 1.221e-04 | 分類不能（ceiling 未実測） |
| 4 | 10 (wmma_tf32_opt 64x64x64 seed=3000) | 1.526e-05 | 部分／未確定 |
| 4 | 11 (wmma_tf32_opt 512x512x4096 seed=0xC0FFEE) | 9.766e-04 | 部分／未確定 |
| 4 | 12 (wmma_tf32_opt 128x128x128 seed=0xBB9 (#1106 diagnostic)) | 3.052e-05 | 部分／未確定 |
| 4 | 13 (wmma_tf32_opt 512x512x512 seed=0xBBA (#1106 diagnostic)) | 1.221e-04 | 部分／未確定 |
| 4 | 14 (wmma_tf32_opt 63x65x33 seed=0xBBB (#1106 diagnostic)) | 1.526e-05 | 部分／未確定 |
| 4 | 15 (wmma_tf32_opt 65x63x17 seed=0xBBC (#1106 diagnostic)) | 7.629e-06 | no-op |
| 4 | 16 (wmma_tf32_opt 64x96x256 seed=0xBBD (#1106 diagnostic)) | 6.104e-05 | 部分／未確定 |
| 4 | 17 (wmma_tf32_opt 4096x4096x4096 seed=0xBEEF (#1106 diagnostic)) | 1.953e-03 | 部分／未確定 |
| 4 | 18 (wmma_tf32_staged 512x512x4096 seed=0xC0FFEE) | 9.766e-04 | 部分／未確定 |
| 4 | 19 (wmma_tf32_staged 64x64x64 seed=0xFA0 (#1106 diagnostic)) | 1.526e-05 | 部分／未確定 |
| 4 | 20 (wmma_tf32_staged 128x128x128 seed=0xFA1 (#1106 diagnostic)) | 3.052e-05 | 部分／未確定 |
| 4 | 21 (wmma_tf32_staged 512x512x512 seed=0xFA2 (#1106 diagnostic)) | 1.221e-04 | 部分／未確定 |
| 4 | 22 (wmma_tf32_staged 60x68x36 seed=0xFA3 (#1106 diagnostic)) | 1.526e-05 | 部分／未確定 |
| 4 | 23 (wmma_tf32_staged 68x60x20 seed=0xFA4 (#1106 diagnostic)) | 7.629e-06 | no-op |
| 4 | 24 (wmma_tf32_staged 64x96x256 seed=0xFA5 (#1106 diagnostic)) | 6.104e-05 | 部分／未確定 |
| 4 | 25 (wmma_tf32_staged 4096x4096x4096 seed=0xBEEF (#1106 diagnostic)) | 1.953e-03 | 部分／未確定 |
| 4 | 26 (mma_f16 256x256x4096 seed=9999) | 1.953e-03 | 部分／未確定 |
| 4 | 27 (wmma_f16 256x256x4096 seed=8888 (run_f16 effective route; GB10: opt)) | 1.953e-03 | 部分／未確定 |
| 4 | 28 (wmma_f16 256x256x4096 seed=8889 (run_f16 effective route; GB10: opt)) | 1.953e-03 | 部分／未確定 |
| 4 | 29 (mma_tf32 16x8x8 seed=5000 (#1122 triage)) | 1.907e-06 | no-op |
| 4 | 30 (mma_tf32 64x64x64 seed=5001 (#1122 triage)) | 1.526e-05 | 部分／未確定 |
| 4 | 31 (mma_tf32 128x128x128 seed=5002 (#1122 triage)) | 3.052e-05 | 部分／未確定 |
| 4 | 32 (mma_tf32 512x512x512 seed=5003 (#1122 triage)) | 1.221e-04 | 部分／未確定 |
| 4 | 33 (mma_tf32 60x68x36 seed=5004 (#1122 triage)) | 1.526e-05 | 部分／未確定 |
| 4 | 34 (mma_tf32 68x60x20 seed=5005 (#1122 triage)) | 7.629e-06 | no-op |
| 4 | 35 (mma_tf32 96x68x72 seed=5006 (#1122 triage)) | 3.052e-05 | 部分／未確定 |
| 4 | 36 (mma_tf32 64x96x256 seed=5007 (#1122 triage)) | 6.104e-05 | 部分／未確定 |
| 4 | 37 (mma_tf32 4x4x4 seed=5008 (#1122 triage)) | 9.537e-07 | no-op |
| 4 | 38 (mma_tf32 64x64x64 seed=1 (#1122 smoke_env_adaptive)) | 1.526e-05 | 部分／未確定 |
| 4 | 39 (mma_tf32 4096x4096x4096 seed=9001 (#1122 triage k4096_stress)) | 9.766e-04 | 部分／未確定 |
| 4 | 40 (mma_tf32_vs_wmma_tf32_staged 512x512x512 seed=6002 (#1122 triage)) | 1.221e-04 | 部分／未確定 |
| 4 | 41 (mma_tf32_vs_wmma_tf32_staged 4096x4096x4096 seed=9002 (#1122 triage k4096_stress)) | 9.766e-04 | 部分／未確定 |
| 4 | 42 (specialized_mma_f16 256x512x1024 seed=4003 compiled=DYNAMIC_ALL (#1159 sweep)) | 4.883e-04 | 部分／未確定 |
| 4 | 43 (specialized_mma_f16 256x512x1024 seed=4003 compiled=STATIC_NK (#1159 sweep)) | 4.883e-04 | 部分／未確定 |
| 4 | 44 (specialized_mma_f16 256x512x1024 seed=4003 compiled=STATIC_MNK (#1159 sweep)) | 4.883e-04 | 部分／未確定 |

## 候補 B-1・B-3（机上分類不能）

B-1（`max_partial` 基準）・B-3（`exact` 基準）は全 45 行とも 「机上分類不能（実行時適用不可／要トレース）」（部分和トレースまたは厳密真値が本データセットには存在しない）。

## 集計（候補ごとの分類件数）

| 候補 | no-op | 全救済 | 部分／未確定 | 分類不能（ceiling未実測） |
|---|---:|---:|---:|---:|
| A c=0.125 eps=2^-23 K*M | 31 | 0 | 14 | 0 |
| A c=0.25 eps=2^-23 K*M | 25 | 0 | 19 | 1 |
| A c=0.5 eps=2^-23 K*M | 22 | 0 | 22 | 1 |
| A c=1.0 eps=2^-23 K*M | 17 | 0 | 27 | 1 |
| A c=2.0 eps=2^-23 K*M | 10 | 1 | 33 | 1 |
| A c=0.125 eps=2^-24 K*M | 34 | 0 | 11 | 0 |
| A c=0.25 eps=2^-24 K*M | 31 | 0 | 14 | 0 |
| A c=0.5 eps=2^-24 K*M | 25 | 0 | 19 | 1 |
| A c=1.0 eps=2^-24 K*M | 22 | 0 | 22 | 1 |
| A c=2.0 eps=2^-24 K*M | 17 | 0 | 27 | 1 |
| A c=1.0 eps=2^-23 sqrtK*M | 45 | 0 | 0 | 0 |

## 契約 5 項目への影響（構造的事実。OR 追加の単調性による）

| 契約検査項目（`assert_no_parity_regression`） | 影響 |
|---|---|
| provenance 確定（`baseline_provenance_unconfirmed=true` の行数: 0/45） | 無影響（候補追加とは独立） |
| `total` 完全一致 | 無影響（要素数は不変） |
| `fail_count <= baseline_fail_count` | 単調非増加のため恒常成立 （OR 追加は fail→pass のみ生じ、pass→fail は生じない） |
| `mean_abs_diff <= ceiling` | 無影響（全セル集計は bit 同一。候補は判定式のみを変え計算対象の値は変えない） |
| `max_abs_diff`/`max_rel_err <= ceiling`（`Some` のみ） | 無影響（同上） |

