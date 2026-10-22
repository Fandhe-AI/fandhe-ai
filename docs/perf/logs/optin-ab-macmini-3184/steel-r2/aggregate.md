gate=OK
runs=5 reference_only=False
arm=LU verdict=UNDETERMINED :: 性能規則の判定 UNDETERMINED: 上記いずれにも該当せず | N=512: median=0.9989 ratios=[0.9972, 1.0175, 1.0011, 0.9975, 0.9989]; N=1024: median=0.9980 ratios=[0.998, 1.0001, 1.001, 0.9975, 0.9964]; N=2048: median=1.0008 ratios=[0.9987, 1.0008, 0.9973, 1.0021, 1.0008]; N=4096: median=0.9996 ratios=[1.0018, 1.0001, 0.9996, 0.9979, 0.9994]
arm=LU underlying_verdict=UNDETERMINED
arm=LU underlying_reason: 性能規則の判定 UNDETERMINED: 上記いずれにも該当せず
arm=T0U verdict=REJECT :: 性能規則の判定 REJECT: 中央値 >1.00 かつ 5/5 run 符号一貫の N あり | N=512: median=1.0378 ratios=[1.0439, 1.0416, 1.0175, 1.0378, 1.0189]; N=1024: median=0.8987 ratios=[0.8996, 0.8987, 0.901, 0.8957, 0.8955]; N=2048: median=0.7657 ratios=[0.7641, 0.763, 0.7657, 0.7665, 0.7659]; N=4096: median=0.7748 ratios=[0.7739, 0.7748, 0.7751, 0.7742, 0.7762]
arm=T0U underlying_verdict=REJECT
arm=T0U underlying_reason: 性能規則の判定 REJECT: 中央値 >1.00 かつ 5/5 run 符号一貫の N あり
arm=T0U-LU verdict=REJECT :: 性能規則の判定 REJECT: 中央値 >1.00 かつ 5/5 run 符号一貫の N あり | N=512: median=1.0547 ratios=[1.0575, 1.0547, 1.0365, 1.0589, 1.0372]; N=1024: median=0.8903 ratios=[0.8903, 0.8913, 0.8923, 0.8867, 0.887]; N=2048: median=0.7587 ratios=[0.7582, 0.7561, 0.7588, 0.7605, 0.7587]; N=4096: median=0.7575 ratios=[0.7571, 0.7575, 0.7586, 0.7575, 0.7574]
arm=T0U-LU underlying_verdict=REJECT
arm=T0U-LU underlying_reason: 性能規則の判定 REJECT: 中央値 >1.00 かつ 5/5 run 符号一貫の N あり
arm=T0U-LU-FB verdict=UNDETERMINED :: 性能規則の判定 UNDETERMINED: 上記いずれにも該当せず | N=512: median=1.0093 ratios=[1.0093, 1.0153, 0.9982, 1.0142, 0.9954]; N=1024: median=0.8526 ratios=[0.8526, 0.8545, 0.8542, 0.8499, 0.8483]; N=2048: median=0.7262 ratios=[0.726, 0.7236, 0.7262, 0.7274, 0.7266]; N=4096: median=0.7235 ratios=[0.7226, 0.7232, 0.7243, 0.7235, 0.7238]
arm=T0U-LU-FB underlying_verdict=UNDETERMINED
arm=T0U-LU-FB underlying_reason: 性能規則の判定 UNDETERMINED: 上記いずれにも該当せず
