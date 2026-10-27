
#### Mac mini（M4）

| セル | 新条件（0.12.0） | 旧条件（0.12.0・旧腕） | 0.11.0 | 新条件の run 別比 |
|---|---|---|---|---|
| gemm Metal N=256 | 1 位 1.16×（vs candle） | 僅差 0.93×（vs candle） | 2 位 0.84×（vs candle） | 1.16 1.17 1.13 1.14 1.16 |
| gemm Metal N=512 | 1 位 1.06×（vs candle） | 2 位 0.84×（vs candle） | 2 位 0.80×（vs candle） | 0.98 1.08 1.06 1.05 1.09 |
| gemm Metal N=1024 ★ | 僅差 0.99×（vs candle） | 2 位 0.86×（vs candle） | 2 位 0.69×（vs candle） | 0.95 1.01 0.99 1.00 0.97 |
| gemm Metal N=2048 ★ | 僅差 0.94×（vs candle） | 2 位 0.81×（vs candle） | 2 位 0.73×（vs candle） | 0.94 0.94 0.95 0.94 0.94 |
| gemm Metal N=4096 ★ | 2 位 0.85×（vs candle） | 2 位 0.75×（vs candle） | 2 位 0.58×（vs candle） | 0.88 0.85 0.91 0.85 0.85 |
| gemm CPU N=256 ★ | 3 位 0.70×（vs pytorch） | 3 位 0.68×（vs pytorch） | 1 位 1.04×（vs pytorch） | 0.68 0.70 0.70 0.74 0.70 |
| gemm CPU N=512 ★ | 2 位 0.77×（vs pytorch） | 2 位 0.78×（vs pytorch） | 1 位 1.04×（vs pytorch） | 1.05 0.77 0.77 0.77 0.77 |
| gemm CPU N=1024 ★ | 2 位 0.67×（vs pytorch） | 2 位 0.68×（vs pytorch） | 2 位 0.89×（vs pytorch） | 0.71 0.66 0.66 0.67 0.69 |
| gemm CPU N=2048 ★ | 2 位 0.53×（vs pytorch） | 2 位 0.53×（vs pytorch） | 2 位 0.69×（vs pytorch） | 0.54 0.54 0.53 0.52 0.52 |
| train Metal | 1 位 1.01×（vs pytorch） | 4 位 0.32×（vs pytorch） | 3 位 0.32×（vs pytorch） | 1.01 1.01 1.03 1.02 0.99 |
| infer Metal | 1 位 1.02×（vs candle） | 僅差 0.94×（vs candle） | 1 位 1.13×（vs candle） | 1.03 1.01 1.02 1.01 1.03 |
| train CPU ★ | 3 位 0.25×（vs pytorch） | 3 位 0.26×（vs pytorch） | 4 位 0.20×（vs pytorch） | 0.25 0.26 0.25 0.25 0.26 |
| infer CPU ★ | 4 位 0.15×（vs pytorch） | 4 位 0.15×（vs pytorch） | 2 位 0.25×（vs pytorch） | 0.16 0.15 0.15 0.15 0.15 |

#### GB10

| セル | 新条件（0.12.0） | 旧条件（0.12.0・旧腕） | 0.11.0 | 新条件の run 別比 |
|---|---|---|---|---|
| gemm CUDA N=256 | 1 位 1.17×（vs candle） | 2 位 0.89×（vs candle） | 2 位 0.84×（vs candle） | 1.19 1.15 1.17 1.17 1.16 |
| gemm CUDA N=512 | 1 位 1.12×（vs candle） | 僅差 0.95×（vs candle） | 3 位 0.42×（vs candle） | 1.06 1.12 1.15 1.15 1.12 |
| gemm CUDA N=1024 | 1 位 1.15×（vs candle） | 僅差 0.96×（vs candle） | 3 位 0.40×（vs candle） | 1.14 1.15 1.16 1.21 1.15 |
| gemm CUDA N=2048 | 1 位 1.13×（vs burn） | 僅差 0.98×（vs burn） | 3 位 0.48×（vs burn） | 1.14 1.11 1.15 1.13 1.12 |
| gemm CUDA N=4096 | 1 位 1.92×（vs burn） | 1 位 1.72×（vs burn） | 1 位 1.01×（vs burn） | 1.93 1.85 1.93 1.91 1.92 |
| gemm CPU N=256 ★ | 3 位 0.71×（vs pytorch） | 1 位 1.42×（vs pytorch） | 3 位 0.74×（vs pytorch） | 0.68 0.69 0.71 0.82 0.75 |
| gemm CPU N=512 ★ | 2 位 0.68×（vs candle） | 僅差 0.92×（vs candle） | 2 位 0.70×（vs candle） | 0.77 0.68 0.61 0.95 0.63 |
| gemm CPU N=1024 ★ | 2 位 0.87×（vs candle） | 1 位 1.20×（vs candle） | 僅差 0.98×（vs candle） | 0.82 0.87 0.84 0.88 0.89 |
| gemm CPU N=2048 ★ | 僅差 0.92×（vs tensorflow） | 1 位 1.65×（vs tensorflow） | 1 位 1.06×（vs tensorflow） | 0.99 0.92 0.89 1.09 0.91 |
| gemm CPU N=4096 ★ | 僅差 0.96×（vs tensorflow） | 1 位 1.19×（vs tensorflow） | 1 位 1.21×（vs tensorflow） | 0.96 0.97 0.94 0.97 0.96 |
| train CUDA | 1 位 2.01×（vs candle） | 僅差 0.98×（vs candle） | 2 位 0.83×（vs candle） | 2.00 2.01 2.04 2.00 2.05 |
| infer CUDA ★ | 3 位 0.51×（vs pytorch） | 3 位 0.44×（vs pytorch） | 3 位 0.40×（vs pytorch） | 0.51 0.51 0.52 0.52 0.51 |
| train CPU ★ | 2 位 0.55×（vs pytorch） | 2 位 0.48×（vs pytorch） | 3 位 0.35×（vs pytorch） | 0.55 0.56 0.49 0.60 0.46 |
| infer CPU ★ | 2 位 0.78×（vs pytorch） | 2 位 0.68×（vs pytorch） | 2 位 0.60×（vs pytorch） | 0.70 0.71 0.78 1.00 0.84 |

- 新条件: `tally 10 4 13 invalid 1 noise 0 total 27`
- 旧条件: `tally 5 7 15 invalid 1 noise 0 total 27`
- 0.11.0: `tally 6 1 20 invalid 1 noise 1 total 27`
- ★ = 新条件の比が 1.0 以下
