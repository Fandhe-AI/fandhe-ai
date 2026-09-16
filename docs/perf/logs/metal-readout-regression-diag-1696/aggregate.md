# Metal readout legacy 後退の 4 腕診断 集計（イシュー #1696）

主系列（単一腕・単一サイズ・プロセス分離・各 5 run）の readback／host_read 中央値・起動間 spread（ノイズ床）・checksum 一致を集計する。判定規則は `docs/perf/metal-readout-legacy-regression-four-arm-diag.md` §5 を正とする。

| N | 腕 | readback median (5 run) | readback ノイズ床 (max-min) | host_read median (5 run) | run 数 | checksum 一致 |
|---|----|--------------------------|------------------------------|---------------------------|--------|----------------|
| 1024 | LegacyToVec | 0.0552 ms | 0.0294 ms | 0.8450 ms | 5 | OK |
| 1024 | BorrowedKeepAlive | 0.2566 ms | 0.0155 ms | 0.5887 ms | 5 | OK |
| 1024 | BorrowedWithDummyAllocFree | 0.0573 ms | 0.0328 ms | 1.4460 ms | 5 | OK |
| 1024 | PretouchedReusedDest | 0.0536 ms | 0.0026 ms | 0.8148 ms | 5 | OK |
| 2048 | LegacyToVec | 0.3030 ms | 0.1225 ms | 3.4332 ms | 5 | OK |
| 2048 | BorrowedKeepAlive | 0.9989 ms | 0.0458 ms | 2.4290 ms | 5 | OK |
| 2048 | BorrowedWithDummyAllocFree | 0.5000 ms | 0.1216 ms | 5.7405 ms | 5 | OK |
| 2048 | PretouchedReusedDest | 0.2400 ms | 0.0159 ms | 3.3443 ms | 5 | OK |
| 4096 | LegacyToVec | 1.0155 ms | 0.1179 ms | 14.3533 ms | 5 | OK |
| 4096 | BorrowedKeepAlive | 4.7529 ms | 3.4180 ms | 13.2079 ms | 5 | OK |
| 4096 | BorrowedWithDummyAllocFree | 2.1807 ms | 1.3505 ms | 23.3745 ms | 5 | OK |
| 4096 | PretouchedReusedDest | 1.7036 ms | 2.1348 ms | 24.8725 ms | 5 | OK |

## 腕差（規則 4: 規模照合）

N=1024 の `BorrowedKeepAlive - LegacyToVec` の **`readback` 差**（`matmul` 区間の GPU 待ち＋ホストへの読み出しに対応する内訳。`docs/perf/metal-readout-legacy-regression-four-arm-diag.md` §2）を`lowlayer-diagnosis-2026-09-12.md` §5 の Δ≈0.47 ms（matmul 区間差）と比較する。`host_read`（legacy の 2 本目確保・コピー等、`matmul` 区間の外側で発生する追加コスト）は規模照合の対象に含めず、参考情報として別掲する。checksum が参照値と一致しない起動は当該起動を無効として除外する（規則 5）。

- N=1024: readback 差（規模照合対象） LegacyToVec=0.0552 ms, BorrowedKeepAlive=0.2566 ms, 差=0.2014 ms / host_read 差（参考。規模照合対象外） LegacyToVec=0.8450 ms, BorrowedKeepAlive=0.5887 ms, 差=-0.2563 ms
- N=2048: readback 差（規模照合対象） LegacyToVec=0.3030 ms, BorrowedKeepAlive=0.9989 ms, 差=0.6959 ms / host_read 差（参考。規模照合対象外） LegacyToVec=3.4332 ms, BorrowedKeepAlive=2.4290 ms, 差=-1.0042 ms
- N=4096: readback 差（規模照合対象） LegacyToVec=1.0155 ms, BorrowedKeepAlive=4.7529 ms, 差=3.7374 ms / host_read 差（参考。規模照合対象外） LegacyToVec=14.3533 ms, BorrowedKeepAlive=13.2079 ms, 差=-1.1454 ms

