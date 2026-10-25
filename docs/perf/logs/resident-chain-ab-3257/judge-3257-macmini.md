# #3257 判定（rule_sha256=23670ac1d98792fb…）

| 機体 | task | device | before 中央値 ms | after 中央値 ms | ratio_ab | ratio_r（5 round） | 判定 | 理由 | 相手最速÷before | 相手最速÷after |
|---|---|---|---|---|---|---|---|---|---|---|
| macmini | train | cpu | 0.4991 | 0.5009 | 1.004 | 0.948, 0.997, 1.007, 0.994, 1.018 | REGRESS | 後退（ratio>1.00 または 5/5 round で ratio_r>1.00） | 0.27x | 0.27x |
| macmini | infer | cpu | 0.2050 | 0.2010 | 0.980 | 0.995, 0.984, 0.966, 0.975, 0.977 | NEUTRAL | 改善・後退どちらの条件も満たさない（RULE の undetermined） | 0.15x | 0.16x |
| macmini | train | metal | 1.5679 | 0.5047 | 0.322 | 0.224, 0.323, 0.322, 0.324, 0.318 | IMPROVE | 改善（ratio<=0.95 かつ 5/5 round で ratio_r<1.00） | 0.31x | 0.97x |
| macmini | infer | metal | 0.3331 | 0.3279 | 0.984 | 0.953, 0.939, 1.075, 1.005, 0.776 | NEUTRAL | 改善・後退どちらの条件も満たさない（RULE の undetermined） | 0.96x | 0.98x |
