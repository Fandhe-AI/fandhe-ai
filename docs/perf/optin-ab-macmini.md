# Mac mini の未計測 opt-in 候補 A/B（イシュー #3184）

状態: **基盤（事前登録規則・after 腕 patch・手順）まで完了。実測は未実施**。全ゲートの Mac mini 判定は undetermined（未実測）。
main の定数・環境変数既定は変更していない。規則は `logs/optin-ab-macmini-3184/RULE.txt`（計測前に単独コミットで固定）。

## 条件差

| 項目 | 継承 RULE の想定 | 本計測 |
|---|---|---|
| 機体 | M4 Max + GB10 | Mac mini M4（P4+E6、GPU 10 コア。Apple Silicon 枠として扱う） |
| 負荷 | 継承登録値 | load1 < 3.0（外側ゲート＋round 別事後検査） |

## ゲート別判定

| ゲート | 継承 RULE | Mac mini | GB10（#3190） | 総合 |
|---|---|---|---|---|
| G1 `HOST_ARENA_DEFAULT_ENABLED` | 2104 | 未実測 | 未実測 | undetermined |
| G2 `REDUCTION_SEQUENTIAL_FALLBACK_ENABLED` | 2101／2102 | 未実測 | 未実測 | undetermined |
| G3 `TRAIN_FORWARD_ENCODE_ONLY_DEFAULT_ENABLED` | 2113 | 未実測 | 未実測 | undetermined |
| G4 `READBACK_DEST_DEFAULT` | 2112 | 未実測 | 未実測 | undetermined |
| G5 readout legacy／borrowed | §15.1（#3265 へ） | 未実測 | 未実測 | undetermined |
| G6 steel 候補 | 2111 | 未実測 | 未実測 | undetermined |
| G7 `METAL_DEVICE_VERIFY_CACHE_DEFAULT_ENABLED` | 2114（ADOPT 判定なし） | 未実測 | 未実測 | record_only |

## 申し送り

- 実測は `logs/optin-ab-macmini-3184/README.md` の手順で CPU 系と Metal 系に分けて実施する（継承スクリプトの負荷閾値は編集せず外側で課す）。
- #3227 は本表で ADOPT 条件が成立したゲートのみを対象にする（現状なし）。G5 の結果は #3265 が使う。
