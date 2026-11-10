# 診断 (3) 集計 #4154

- status: formal（ゲート: PASS PASS PASS PASS PASS）
- RULE: fixed-at-utc=2026-11-10T03:13:51Z tree_sha=3d6ba9e98de38e0280fcb9c276a38fcb9ed96561 tree_fingerprint=838481a85e5002f779213d67de032a391f48746291da21ab5e1c350f3335b46f
- aggregate_script_sha256=2877b967e6a4be8157c5ae4f657dee82fd7443ca98f619e24294fa2152c4789d reach_threshold=1.00 noise_band=NB-RANGE5

## M-MTL-TRN（macmini・相手 pytorch・kernel_owner=hardware）

| run | F1 | F2 | 観測比 |
|---|---|---|---|
| 1 | - | - | 0.5946 |
| 2 | - | - | 1.0398 |
| 3 | - | - | 1.0273 |
| 4 | - | - | 1.1897 |
| 5 | - | - | 1.0589 |
| 1 時計不一致 | fandhe 純カーネル 0.4601 ms（gpu_timestamp） | 相手 純カーネル 0.4239 ms（mps_event） | |
| 2 時計不一致 | fandhe 純カーネル 0.2339 ms（gpu_timestamp） | 相手 純カーネル 0.4339 ms（mps_event） | |
| 3 時計不一致 | fandhe 純カーネル 0.2340 ms（gpu_timestamp） | 相手 純カーネル 0.4308 ms（mps_event） | |
| 4 時計不一致 | fandhe 純カーネル 0.2340 ms（gpu_timestamp） | 相手 純カーネル 0.4214 ms（mps_event） | |
| 5 時計不一致 | fandhe 純カーネル 0.2343 ms（gpu_timestamp） | 相手 純カーネル 0.4234 ms（mps_event） | |
| 中央値 | - | - | 1.0398 |

- 分類: 分類保留／確度: 低／前分類との異同: 同
- 理由: 確度低: F1 の時計不一致（比較不能。境界を併記）

