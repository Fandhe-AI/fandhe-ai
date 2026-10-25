# #3257 判定（rule_sha256=23670ac1d98792fb…）

| 機体 | task | device | before 中央値 ms | after 中央値 ms | ratio_ab | ratio_r（5 round） | 判定 | 理由 | 相手最速÷before | 相手最速÷after |
|---|---|---|---|---|---|---|---|---|---|---|
| gb10 | train | cpu | 0.7231 | 0.7659 | 1.059 | 1.092, 0.893, 0.954, 1.102, 1.077 | REGRESS | 後退（ratio>1.00 または 5/5 round で ratio_r>1.00） | 0.52x | 0.49x |
| gb10 | infer | cpu | 0.1911 | 0.1620 | 0.848 | 0.904, 0.857, 0.906, 0.798, 0.718 | IMPROVE | 改善（ratio<=0.95 かつ 5/5 round で ratio_r<1.00） | 0.72x | 0.85x |
| gb10 | train | cuda | 0.2765 | 0.1343 | 0.486 | 0.485, 0.487, 0.487, 0.485, 0.490 | IMPROVE | 改善（ratio<=0.95 かつ 5/5 round で ratio_r<1.00） | 0.98x | 2.02x |
| gb10 | infer | cuda | 0.0899 | 0.0776 | 0.863 | 0.854, 0.854, 0.862, 0.863, 0.873 | IMPROVE | 改善（ratio<=0.95 かつ 5/5 round で ratio_r<1.00） | 0.44x | 0.51x |
