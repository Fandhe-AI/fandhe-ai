# #3679 再現判定（repro_judge.py の出力。手で書き換えない）

RULE.txt sha256: `75c4f3c5306b9d16bc8e50c60e024bb292015eb8ee9cc8e5f561aa04ec35feb3`

## 結論

| セル | 機体 | 腕 | N | 結論 | k/15 | 15 個の中央値 ratio |
|---|---|---|---|---|---|---|
| T1 | gb10 | resident_split | 512 | **未実測** | - | - |
| T2 | macmini | resident_public | 256 | **判定不能（系列不足）** | - | - |
| T3 | macmini | gemm_device_download | 4096 | **判定不能（系列不足）** | - | - |

- T1: gb10 のデータ無し
- T2: ゲートを pass した系列が 1（起動 1）。3 未満
- T3: ゲートを pass した系列が 1（起動 1）。3 未満

## 系列の扱い

- macmini macmini（先頭腕 after）: 判定に使用

## セルごとの ratio_r（run1..5。系列順）


## ノイズの参考（T1〜T3 以外の診断セル。系列ごと。記録のみ）

- macmini 系列1: 5/5 > 1.00 が 4 件・5/5 < 1.00 が 8 件（38 セル中）
