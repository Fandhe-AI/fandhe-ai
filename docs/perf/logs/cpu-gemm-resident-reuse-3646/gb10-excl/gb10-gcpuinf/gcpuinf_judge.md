# G-CPU-INF の算出（RULE.txt 末尾。#3646）

単位 us。5 run の run 間中央値。D = fandhe 窓 - PyTorch 窓。
純カーネル差 = fandhe 層カーネル合計 - PyTorch mm 合計（l1_fwd 64x256x784 + l2_fwd 64x10x256）。
窓の構成差 = D - 純カーネル差（checksum・readout・fandhe 固有の固定費・bias／活性化など、窓に入るが mm 以外の部分の残差）。

- PyTorch 窓（bench_py infer）: 126.7（run 別 137.6, 126.7, 141.0, 114.4, 118.9）
- PyTorch mm_pure l1_fwd 106.2 + l2_fwd 11.0 = 117.2

| fandhe の窓 | 窓 | run 別 | D | fandhe 層カーネル合計 | 純カーネル差 | 窓の構成差 | 純カーネル差/D | 構成差/D |
|---|---|---|---|---|---|---|---|---|
| fresh 通常行 | 171.6 | 180.1, 162.5, 171.6, 171.8, 170.7 | +44.9 | 177.1 | +59.9 | -15.1 | 1.34 | -0.34 |
| reuse 通常行 | 161.7 | 160.7, 164.4, 161.7, 152.7, 168.4 | +34.9 | 183.4 | +66.2 | -31.3 | 1.90 | -0.90 |
| reuse --phases iter_total | 161.2 | 148.6, 173.1, 168.5, 156.1, 161.2 | +34.5 | 183.4 | +66.2 | -31.7 | 1.92 | -0.92 |

注: fandhe 層カーネル合計は #2105 診断（fixedcost-2105）の層別区間で、fresh 行には fresh_layers（L1 は linear+relu 融合・L2 は gemm+add）、
reuse 行には reuse_decomposed の l1／l2 forward_device を当てた。層別区間は診断の別プロセスの値で窓の値とは同一プロセスではない。
