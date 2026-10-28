# G-CPU-INF の算出（RULE.txt 末尾。#3646）

単位 us。5 run の run 間中央値。D = fandhe 窓 - PyTorch 窓。
純カーネル差 = fandhe 層カーネル合計 - PyTorch mm 合計（l1_fwd 64x256x784 + l2_fwd 64x10x256）。
窓の構成差 = D - 純カーネル差（checksum・readout・fandhe 固有の固定費・bias／活性化など、窓に入るが mm 以外の部分の残差）。

- PyTorch 窓（bench_py infer）: 122.9（run 別 138.9, 120.4, 132.3, 121.3, 122.9）
- PyTorch mm_pure l1_fwd 98.0 + l2_fwd 10.9 = 108.9

| fandhe の窓 | 窓 | run 別 | D | fandhe 層カーネル合計 | 純カーネル差 | 窓の構成差 | 純カーネル差/D | 構成差/D |
|---|---|---|---|---|---|---|---|---|
| fresh 通常行 | 173.6 | 173.6, 165.1, 165.3, 178.9, 174.4 | +50.7 | 177.8 | +68.9 | -18.2 | 1.36 | -0.36 |
| reuse 通常行 | 159.3 | 173.5, 172.6, 159.3, 154.2, 159.1 | +36.4 | 182.1 | +73.2 | -36.8 | 2.01 | -1.01 |
| reuse --phases iter_total | 164.5 | 157.9, 173.5, 165.6, 164.5, 149.7 | +41.7 | 182.1 | +73.2 | -31.5 | 1.76 | -0.76 |

注: fandhe 層カーネル合計は #2105 診断（fixedcost-2105）の層別区間で、fresh 行には fresh_layers（L1 は linear+relu 融合・L2 は gemm+add）、
reuse 行には reuse_decomposed の l1／l2 forward_device を当てた。層別区間は診断の別プロセスの値で窓の値とは同一プロセスではない。
