# #3613 3 腕 A/B 集計（GB10・機械出力。手で直さない）

ラベル: `3613-r1` / 規則: `RULE.txt` §4（閾値 1.00・5 round・5/5 一貫）

## 前提ゲート

- P-TREE: 成立
- P-R0: 成立
- P-BIT: 成立
- P-RT: 成立
- P-LOCK: 成立
- P-GATE: 成立
- P-RUNAB: 成立
- P-COLLECT: 成立

## 判定セルの帰属分類

- **gemm 256 fresh**: 非再現（ノイズ帯・判定不能）
- **gemm 2048 reuse**: 非再現（ノイズ帯・判定不能）

## セル別の組比（後者/前者の round 別比と中央値。> 1.00 = 後退）

| セル | 判定 | 組 | round 別比 | 中央値比 | 5/5 後退 |
|---|---|---|---|---|---|
| gemm 256 fresh | 判定 | pair0 (after/before) | 0.964 / 0.924 / 1.020 / 0.804 / 0.978 | 0.964 | いいえ |
| gemm 256 fresh | 判定 | pair1 (prime/before) | 1.134 / 0.862 / 0.948 / 0.855 / 0.876 | 0.876 | いいえ |
| gemm 256 fresh | 判定 | pair2 (after/prime) | 0.850 / 1.072 / 1.076 / 0.941 / 1.117 | 1.072 | いいえ |
| gemm 256 reuse | 併記 | pair0 (after/before) | 0.831 / 0.956 / 1.018 / 1.072 / 1.023 | 1.018 | いいえ |
| gemm 256 reuse | 併記 | pair1 (prime/before) | 0.969 / 0.976 / 1.027 / 0.934 / 0.947 | 0.969 | いいえ |
| gemm 256 reuse | 併記 | pair2 (after/prime) | 0.857 / 0.980 / 0.991 / 1.148 / 1.079 | 0.991 | いいえ |
| gemm 512 fresh | 併記 | pair0 (after/before) | 0.854 / 1.050 / 1.039 / 0.952 / 1.023 | 1.023 | いいえ |
| gemm 512 fresh | 併記 | pair1 (prime/before) | 0.820 / 1.086 / 0.933 / 1.020 / 1.030 | 1.020 | いいえ |
| gemm 512 fresh | 併記 | pair2 (after/prime) | 1.042 / 0.966 / 1.114 / 0.933 / 0.993 | 0.993 | いいえ |
| gemm 512 reuse | 併記 | pair0 (after/before) | 1.160 / 1.078 / 1.081 / 1.027 / 0.827 | 1.078 | いいえ |
| gemm 512 reuse | 併記 | pair1 (prime/before) | 1.254 / 1.187 / 0.993 / 1.104 / 0.973 | 1.104 | いいえ |
| gemm 512 reuse | 併記 | pair2 (after/prime) | 0.924 / 0.908 / 1.090 / 0.931 / 0.850 | 0.924 | いいえ |
| gemm 1024 fresh | 併記 | pair0 (after/before) | 0.775 / 0.761 / 1.019 / 1.111 / 1.000 | 1.000 | いいえ |
| gemm 1024 fresh | 併記 | pair1 (prime/before) | 0.863 / 0.914 / 1.168 / 0.889 / 0.905 | 0.905 | いいえ |
| gemm 1024 fresh | 併記 | pair2 (after/prime) | 0.898 / 0.833 / 0.872 / 1.250 / 1.105 | 0.898 | いいえ |
| gemm 1024 reuse | 併記 | pair0 (after/before) | 0.927 / 0.997 / 0.930 / 0.981 / 0.986 | 0.981 | いいえ |
| gemm 1024 reuse | 併記 | pair1 (prime/before) | 1.006 / 1.029 / 0.992 / 1.073 / 1.021 | 1.021 | いいえ |
| gemm 1024 reuse | 併記 | pair2 (after/prime) | 0.921 / 0.969 / 0.938 / 0.915 / 0.966 | 0.938 | いいえ |
| gemm 2048 fresh | 併記 | pair0 (after/before) | 0.981 / 0.985 / 0.960 / 1.006 / 1.011 | 0.985 | いいえ |
| gemm 2048 fresh | 併記 | pair1 (prime/before) | 0.987 / 0.920 / 0.915 / 0.967 / 0.970 | 0.967 | いいえ |
| gemm 2048 fresh | 併記 | pair2 (after/prime) | 0.994 / 1.070 / 1.049 / 1.041 / 1.042 | 1.042 | いいえ |
| gemm 2048 reuse | 判定 | pair0 (after/before) | 0.918 / 0.751 / 0.869 / 0.810 / 0.952 | 0.869 | いいえ |
| gemm 2048 reuse | 判定 | pair1 (prime/before) | 0.944 / 0.826 / 0.890 / 1.013 / 0.985 | 0.944 | いいえ |
| gemm 2048 reuse | 判定 | pair2 (after/prime) | 0.973 / 0.910 / 0.976 / 0.800 / 0.967 | 0.967 | いいえ |
| train 64 fresh | 併記 | pair0 (after/before) | 1.058 / 1.062 / 1.096 / 1.090 / 1.057 | 1.062 | はい |
| train 64 fresh | 併記 | pair1 (prime/before) | 0.949 / 0.981 / 1.028 / 1.049 / 1.003 | 1.003 | いいえ |
| train 64 fresh | 併記 | pair2 (after/prime) | 1.115 / 1.083 / 1.067 / 1.039 / 1.054 | 1.067 | はい |
| train 64 reuse | 併記 | pair0 (after/before) | 0.971 / 1.027 / 1.081 / 1.019 / 0.978 | 1.019 | いいえ |
| train 64 reuse | 併記 | pair1 (prime/before) | 0.973 / 0.969 / 1.001 / 1.013 / 0.972 | 0.973 | いいえ |
| train 64 reuse | 併記 | pair2 (after/prime) | 0.998 / 1.060 / 1.080 / 1.005 / 1.006 | 1.006 | いいえ |
| infer 64 fresh | 併記 | pair0 (after/before) | 0.952 / 0.978 / 0.921 / 1.058 / 0.951 | 0.952 | いいえ |
| infer 64 fresh | 併記 | pair1 (prime/before) | 1.053 / 0.947 / 0.989 / 1.028 / 1.016 | 1.016 | いいえ |
| infer 64 fresh | 併記 | pair2 (after/prime) | 0.905 / 1.033 / 0.931 / 1.028 / 0.936 | 0.936 | いいえ |
| infer 64 reuse | 併記 | pair0 (after/before) | 0.936 / 0.911 / 0.956 / 1.058 / 1.057 | 0.956 | いいえ |
| infer 64 reuse | 併記 | pair1 (prime/before) | 1.011 / 0.817 / 1.013 / 1.006 / 0.992 | 1.006 | いいえ |
| infer 64 reuse | 併記 | pair2 (after/prime) | 0.925 / 1.115 / 0.943 / 1.052 / 1.066 | 1.052 | いいえ |

checksum 3 腕完全一致: 全セル一致

注: 分類は帰属の候補であり判定（ADOPT／REJECT）ではない。既定 ON へ戻すかは所有者の判断。

