# .rev-stamp・規則・バイナリの突き合わせ（イシュー #3247）

計測で使った 3 ツリーの `.rev-stamp`（実 HEAD の 40 桁を 1 行）と、RULE・計測パッチ・bench バイナリの sha256 を、機体ごとの `env_info-*.txt` と突き合わせた記録。判定側（`judge_dinput_skip_ab.py`）も同じ照合を fail-closed で行い、不一致なら判定不能にする。

| 項目 | 期待値（RULE／リポジトリ） | Mac mini の env_info | GB10 の env_info | 一致 |
|---|---|---|---|---|
| before の HEAD（`rev_stamp_head_before`） | `b6579c3f52689a232b81d9184f20c30778c6d26c`（RULE の before） | 同左 | 同左 | はい |
| after の HEAD（`rev_stamp_head_after`） | `4fb2a39c859ac4c40705aa37461f4d34b0fc5dd7`（RULE の after） | 同左 | 同左 | はい |
| runner の HEAD（`rev_stamp_head_runner`） | `a5fa15fd32fafeb6ab4b0565453fbd13ffe857b6`（RULE を含む main の先頭。#3561） | 同左 | 同左 | はい |
| `rule_sha256` | `b4d2915487e8e020bee2956487c5678cec749e74f57b68fd2ba6f77210bc0e9b`（`sha256` で `RULE.txt` を再計算して一致） | 同左 | 同左 | はい |
| `patch_sha256` | `9a2c05004c08adb2289f475e932f93f29b598b6d9d3ab9efe4a710644f40fd13`（`bench-var-no-grad.patch` を再計算して一致） | 同左 | 同左 | はい |

## ツリーの作り方

- ノード（Mac mini・GB10）は github に届かないため、ローカルで作った bundle（runner・before・after のブランチ ref を含む）を rsync で送り、ノード上で `git init` → `git fetch <bundle> <ref>` → `git checkout --detach FETCH_HEAD` で 3 ツリーを作った。
- 各ツリーの直下に実 HEAD の 40 桁を 1 行だけ書いた `.rev-stamp` を置き、`.git/info/exclude` へ追記した。
- 計測パッチは runner ツリーにだけ `git apply` した。before／after は未変更。runner の変更追跡ファイルの差分は `bench-fandhe/src/main.rs` の 1 件だけ（`git status --porcelain` で確認）。

## bench バイナリの sha256（腕ごとに別値。バイナリ本体は収録しない）

| 機体 | before | after |
|---|---|---|
| Mac mini | `985ccb5afa9b3e78b0b13d56bb3ddb84ffccdfe9a202bfb15dae397b26ca3681` | `7903bb4db93fac393dbf2ce96a108abe1a95e6c1d468b25698f16409ae49822d` |
| GB10 | `985af0c2bf7bc9a24414975822c45c7c35c61ffcc444c316e3f7ad8ebab0b78a` | `419698317c0bf3694cc16b151bd95f134605a746868dfdcc919cc9db00982b01` |

機体の同定: Mac mini は `cpu_brand=Apple M4`（M4 Max の開発機ではない）、GB10 は `uname -m = aarch64`・`gpu=NVIDIA GB10`（いずれも env_info に記録）。
