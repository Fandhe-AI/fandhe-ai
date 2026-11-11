# 診断 (3) 集計 #4153（参考）

- status: partial（ゲート: TIMEOUT PASS PASS PASS PASS）
- RULE: fixed-at-utc=2026-11-10T10:20:40Z tree_sha=33b9851a382bc588b07fc561b7b4765de4cd9ed5 tree_fingerprint=838481a85e5002f779213d67de032a391f48746291da21ab5e1c350f3335b46f
- aggregate_script_sha256=2877b967e6a4be8157c5ae4f657dee82fd7443ca98f619e24294fa2152c4789d reach_threshold=1.00 noise_band=NB-RANGE5

## M-MTL-G4096（macmini・相手 candle・kernel_owner=constraint）

| run | F1 | F2 | 観測比 |
|---|---|---|---|
| 2 | - | 0.7593 | 0.8873 |
| 3 | - | 0.7859 | 0.9096 |
| 4 | - | 0.7621 | 0.8668 |
| 5 | - | 0.7579 | 0.8704 |
| 中央値 | - | 0.7607 | 0.8789 |
| NB-RANGE5 F2 | 0.7579〜0.7859（閾値をまたがない） | | |

- 分類: 分類保留／確度: 低／前分類との異同: 変更（自分たちの制約 → 分類保留）
- 理由: 確度低: どれも届かない（残差は固定費など別区間。別診断へ）（部分通過）

