# 0.12.0 の 6 フレームワーク同席計測 — Mac mini（Apple M4）（イシュー #3273）

**状態: 未実測（計測部品と事前登録の草案のみ）。** 目的・判定規則・受け入れ条件は `../framework-compare-0.12.0-remeasure/README.md` を正とする（二重管理しない）。本ディレクトリに実測値はない。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.txt` | Mac mini の事前登録規則（未固定の草案） |
| `scripts/macmini-run-0120.sh` | 1 round。事前検査（`Apple M4`・SHA・共有ファイル一致・腕判別 grep・`bench_py.py` sha256）、新旧 2 ツリー、旧条件腕 26 行、期待行数 48/32/26 と skipped 4 件のみ許容 |
| `scripts/macmini-0120a-loop.sh` | 系列 A（ゲートなし・対照） |
| `scripts/macmini-0120b-loop.sh` | 系列 B（load1 < 3.0 ゲート・`GATE-INVALID` 検証つき。5/5 完走のときだけ正式） |
| `aggregate/median_rounds.py` | 0.11.0 版のコピー |

## 結果

未実測。0.11.0 の Mac mini 記録は `../framework-compare-0.11.0-macmini-remeasure/`（M4 Max の値は Mac mini の値として使わない）。

### #3256 の `#[ignore]` 実機テスト（M1〜M6・Metal）

未実測（記入欄）。
