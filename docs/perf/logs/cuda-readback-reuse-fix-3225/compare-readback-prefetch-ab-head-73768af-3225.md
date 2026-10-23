## group: gate (modes=reuse sizes=512,1024,2048,4096 threshold=0.95)
| size/mode | before median | after median | after/before | checksum | 判定 |
|---|---|---|---|---|---|
| 512/reuse | 564.9 us (min 557.7 us / max 580.7 us) | 265.6 us (min 260.8 us / max 663.0 us) | 0.4703 | 完全一致 | 非後退 |
| 1024/reuse | 2.397 ms (min 2.188 ms / max 2.460 ms) | 976.7 us (min 963.8 us / max 990.9 us) | 0.4075 | 完全一致 | 非後退 |
| 2048/reuse | 8.631 ms (min 8.600 ms / max 8.737 ms) | 4.230 ms (min 4.207 ms / max 4.247 ms) | 0.4901 | 完全一致 | 非後退 |
| 4096/reuse | 38.759 ms (min 38.719 ms / max 38.976 ms) | 22.797 ms (min 22.705 ms / max 22.910 ms) | 0.5882 | 完全一致 | 非後退 |

## group: fresh4096 (modes=fresh sizes=4096 threshold=0.95)
| size/mode | before median | after median | after/before | checksum | 判定 |
|---|---|---|---|---|---|
| 4096/fresh | 39.754 ms (min 39.259 ms / max 39.910 ms) | 22.432 ms (min 22.276 ms / max 22.514 ms) | 0.5643 | 完全一致 | 非後退 |

## group: guard (modes=fresh sizes=512,1024,2048 threshold=1.03)
| size/mode | before median | after median | after/before | checksum | 判定 |
|---|---|---|---|---|---|
| 512/fresh | 265.0 us (min 258.7 us / max 267.1 us) | 258.7 us (min 258.2 us / max 260.2 us) | 0.9761 | 完全一致 | 非後退 |
| 1024/fresh | 956.7 us (min 949.9 us / max 963.0 us) | 966.8 us (min 960.8 us / max 971.3 us) | 1.0106 | 完全一致 | 非後退 |
| 2048/fresh | 4.196 ms (min 4.189 ms / max 4.203 ms) | 4.201 ms (min 4.188 ms / max 4.218 ms) | 1.0012 | 完全一致 | 非後退 |

## group: n256 (modes=reuse,fresh sizes=256 threshold=1.03)
| size/mode | before median | after median | after/before | checksum | 判定 |
|---|---|---|---|---|---|
| 256/fresh | 88.0 us (min 86.9 us / max 88.3 us) | 88.4 us (min 86.1 us / max 89.5 us) | 1.0054 | 完全一致 | 非後退 |
| 256/reuse | 92.1 us (min 89.9 us / max 92.5 us) | 91.9 us (min 90.5 us / max 92.7 us) | 0.9978 | 完全一致 | 非後退 |

## checksum_gate (全 run・全腕の完全一致 + 既知値 1e-5 以内。N=256 を含む)
checksum_gate: OK
