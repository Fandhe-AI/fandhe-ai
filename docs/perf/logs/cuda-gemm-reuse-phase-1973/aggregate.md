# イシュー #1973 Layer A 集計（自動生成）

### N=1024

| phase | 5 run 中央値 (ms) | iter_total 比 |
| --- | --- | --- |
| matmul | 1.6601 | 75.9% |
| to_tensor | 0.0001 | 0.0% |
| host_copy | 0.0001 | 0.0% |
| checksum | 0.5280 | 24.1% |
| iter_total | 2.1882 | - |

### N=2048

| phase | 5 run 中央値 (ms) | iter_total 比 |
| --- | --- | --- |
| matmul | 6.3041 | 74.6% |
| to_tensor | 0.0003 | 0.0% |
| host_copy | 0.0002 | 0.0% |
| checksum | 2.1393 | 25.3% |
| iter_total | 8.4470 | - |

### N=4096

| phase | 5 run 中央値 (ms) | iter_total 比 |
| --- | --- | --- |
| matmul | 30.1819 | 77.7% |
| to_tensor | 0.0005 | 0.0% |
| host_copy | 0.0002 | 0.0% |
| checksum | 8.6272 | 22.2% |
| iter_total | 38.8332 | - |

