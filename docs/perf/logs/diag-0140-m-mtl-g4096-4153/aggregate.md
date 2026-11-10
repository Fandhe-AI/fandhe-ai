# 診断 (3) 集計 #4153

- status: formal（ゲート: PASS PASS PASS PASS PASS）
- RULE: fixed-at-utc=2026-11-10T04:18:57Z tree_sha=33b9851a382bc588b07fc561b7b4765de4cd9ed5 tree_fingerprint=838481a85e5002f779213d67de032a391f48746291da21ab5e1c350f3335b46f
- aggregate_script_sha256=2877b967e6a4be8157c5ae4f657dee82fd7443ca98f619e24294fa2152c4789d reach_threshold=1.00 noise_band=NB-RANGE5

## M-MTL-G4096（macmini・相手 candle・kernel_owner=constraint）

| run | F1 | F2 | 観測比 |
|---|---|---|---|
| 1 | - | 0.8160 | 0.9466 |
| 2 | - | 0.7669 | 0.8818 |
| 3 | - | 0.8198 | 0.9435 |
| 4 | - | 0.8309 | 0.9436 |
| 5 | - | 0.8299 | 0.9457 |
| 中央値 | - | 0.8198 | 0.9436 |
| NB-RANGE5 F2 | 0.7669〜0.8309（閾値をまたがない） | | |

- 分類: 分類保留／確度: 高／前分類との異同: 変更（自分たちの制約 → 分類保留）
- 理由: どれも届かない（残差は固定費など別区間。別診断へ）

