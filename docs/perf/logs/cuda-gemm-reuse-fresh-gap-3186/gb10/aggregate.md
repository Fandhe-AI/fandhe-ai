# #3186 集計（RULE.txt に従う）

専有ゲート: 全 run pass {'run1': 'pass', 'run2': 'pass', 'run3': 'pass', 'run4': 'pass', 'run5': 'pass'}

## 判定 W（窓の構成＝tape 蓄積の寄与）

| N | Layer A fresh (us) | Layer A reuse (us) | gap_A | gap_retention | retention_share | 判定 |
|---|---|---|---|---|---|---|
| 512 | 259.0 | 537.0 | 278.0 | 292.7 | 1.053 | 支持［record_only］ |
| 1024 | 958.0 | 2171.8 | 1213.8 | 1295.8 | 1.067 | 支持 |
| 2048 | 4195.2 | 8168.4 | 3973.1 | 3909.8 | 0.984 | 支持 |
| 4096 | 38706.5 | 38521.2 | -185.4 | 1213.4 | - | 非再現（reuse−fresh が 0 以下。判定しない） |

## 判定 F（N=4096 fresh の未説明分）

- Layer A fresh: 38706.5 us / Layer B Σ(drop_reused): 13462.0 us / host checksum: 8757.7 us
- residual_fresh_4096 = 16486.8 us / page_cost_4096 = 15739.4 us
- share_F = 0.955 -> 帰属できた
- 記録のみ: drop_reused の D2H 区間 11175.5 us（帯域下限の推定約 1100 us との差 10075.5 us）

## 二峰性（N=4096）

- 高モード閾値: 55877.4 us（5 x drop_reused の D2H 中央値）
- 判定: 未タッチ宛先が条件

| 腕 | run1..5 の高モード試行数 |
|---|---|
| drop_untouched | [2, 2, 4, 5, 0] |
| keep_untouched | [0, 0, 0, 0, 0] |
| drop_pretouched_split | [0, 0, 0, 0, 0] |
| keep_pretouched_split | [0, 0, 0, 0, 0] |
| drop_reused | [0, 0, 0, 0, 0] |

## minflt 合計（計測 20 試行・run 間中央値。付記のみ）

- N=512: drop_pretouched=1, keep_pretouched=5121, drop_untouched=1, keep_untouched=3153, drop_reused=1, drop_pinned=0, keep_pinned=5120, drop_pretouched_split=1, keep_pretouched_split=5121
- N=1024: drop_pretouched=1, keep_pretouched=20501, drop_untouched=1, keep_untouched=4177, drop_reused=1, drop_pinned=0, keep_pinned=20500, drop_pretouched_split=1, keep_pretouched_split=20501
- N=2048: drop_pretouched=1, keep_pretouched=81941, drop_untouched=1, keep_untouched=41041, drop_reused=1, drop_pinned=0, keep_pinned=81940, drop_pretouched_split=1, keep_pretouched_split=81941
- N=4096: drop_pretouched=327701, keep_pretouched=327701, drop_untouched=589905, keep_untouched=655441, drop_reused=98305, drop_pinned=327700, keep_pinned=327700, drop_pretouched_split=327701, keep_pretouched_split=327701

## 区間表（run 間中央値 [min-max]・us）

| N | 腕 | readback 区間 | min | max |
|---|---|---|---|---|
| 512 | drop_pretouched | 63.4 | 62.8 | 67.8 |
| 512 | keep_pretouched | 356.1 | 336.7 | 368.2 |
| 512 | drop_untouched | 52.9 | 52.3 | 53.4 |
| 512 | keep_untouched | 8184.1 | 5013.1 | 12229.5 |
| 512 | drop_reused | 52.8 | 52.4 | 53.2 |
| 512 | drop_pinned | 82.2 | 81.7 | 83.4 |
| 512 | keep_pinned | 401.7 | 376.1 | 414.3 |
| 512 | drop_pretouched_split | 66.6 | 63.0 | 67.0 |
| 512 | keep_pretouched_split | 345.3 | 343.8 | 485.8 |
| 1024 | drop_pretouched | 248.1 | 247.8 | 253.2 |
| 1024 | keep_pretouched | 1543.9 | 1505.7 | 1661.1 |
| 1024 | drop_untouched | 247.8 | 247.3 | 248.3 |
| 1024 | keep_untouched | 36452.6 | 32421.6 | 41488.6 |
| 1024 | drop_reused | 248.7 | 248.3 | 250.1 |
| 1024 | drop_pinned | 384.2 | 377.6 | 389.5 |
| 1024 | keep_pinned | 1568.1 | 1533.2 | 1696.3 |
| 1024 | drop_pretouched_split | 248.1 | 247.4 | 253.0 |
| 1024 | keep_pretouched_split | 1480.5 | 1383.1 | 1688.8 |
| 2048 | drop_pretouched | 1436.4 | 1434.3 | 1441.6 |
| 2048 | keep_pretouched | 5346.2 | 4884.4 | 5650.9 |
| 2048 | drop_untouched | 1438.0 | 1436.8 | 1438.9 |
| 2048 | keep_untouched | 144844.6 | 125021.5 | 151094.3 |
| 2048 | drop_reused | 1438.8 | 1438.1 | 1443.4 |
| 2048 | drop_pinned | 2023.2 | 1967.1 | 2033.6 |
| 2048 | keep_pinned | 6792.9 | 6551.5 | 6904.3 |
| 2048 | drop_pretouched_split | 1434.8 | 1432.7 | 1441.2 |
| 2048 | keep_pretouched_split | 5484.9 | 5118.2 | 5674.1 |
| 4096 | drop_pretouched | 26914.8 | 21334.4 | 27989.4 |
| 4096 | keep_pretouched | 28128.2 | 27326.3 | 28865.0 |
| 4096 | drop_untouched | 29913.7 | 25917.9 | 32315.9 |
| 4096 | keep_untouched | 29620.4 | 22930.7 | 31398.5 |
| 4096 | drop_reused | 11175.5 | 11024.9 | 11492.9 |
| 4096 | drop_pinned | 30379.6 | 30234.2 | 30914.8 |
| 4096 | keep_pinned | 31096.5 | 30846.3 | 32459.7 |
| 4096 | drop_pretouched_split | 27472.0 | 24594.3 | 28317.0 |
| 4096 | keep_pretouched_split | 27382.4 | 27176.2 | 27832.4 |

## run 別（run 番号付き。readback 区間 us）

- N=512 drop_pretouched: r1=63.4, r2=63.6, r3=67.8, r4=62.8, r5=63.4
- N=512 keep_pretouched: r1=356.1, r2=336.7, r3=342.4, r4=368.2, r5=362.3
- N=512 drop_untouched: r1=53.4, r2=52.3, r3=52.9, r4=52.4, r5=53.4
- N=512 keep_untouched: r1=12229.5, r2=10705.7, r3=5013.1, r4=8184.1, r5=7118.7
- N=512 drop_reused: r1=53.1, r2=52.8, r3=53.2, r4=52.8, r5=52.4
- N=512 drop_pinned: r1=82.6, r2=82.1, r3=81.7, r4=83.4, r5=82.2
- N=512 keep_pinned: r1=386.3, r2=376.1, r3=404.2, r4=401.7, r5=414.3
- N=512 drop_pretouched_split: r1=67.0, r2=63.0, r3=66.8, r4=66.6, r5=63.7
- N=512 keep_pretouched_split: r1=485.8, r2=343.8, r3=345.3, r4=345.0, r5=363.4
- N=1024 drop_pretouched: r1=247.8, r2=247.9, r3=251.5, r4=248.1, r5=253.2
- N=1024 keep_pretouched: r1=1590.5, r2=1661.1, r3=1533.9, r4=1505.7, r5=1543.9
- N=1024 drop_untouched: r1=247.3, r2=248.3, r3=247.8, r4=247.7, r5=248.0
- N=1024 keep_untouched: r1=36452.6, r2=36600.4, r3=34966.1, r4=41488.6, r5=32421.6
- N=1024 drop_reused: r1=250.1, r2=248.7, r3=249.0, r4=248.6, r5=248.3
- N=1024 drop_pinned: r1=377.6, r2=378.2, r3=389.5, r4=384.2, r5=389.0
- N=1024 keep_pinned: r1=1565.5, r2=1696.3, r3=1613.9, r4=1533.2, r5=1568.1
- N=1024 drop_pretouched_split: r1=247.4, r2=247.8, r3=251.8, r4=248.1, r5=253.0
- N=1024 keep_pretouched_split: r1=1385.7, r2=1383.1, r3=1480.5, r4=1688.8, r5=1513.9
- N=2048 drop_pretouched: r1=1435.0, r2=1434.3, r3=1441.6, r4=1438.6, r5=1436.4
- N=2048 keep_pretouched: r1=4884.4, r2=5457.1, r3=5346.2, r4=5340.6, r5=5650.9
- N=2048 drop_untouched: r1=1438.2, r2=1438.0, r3=1436.8, r4=1438.0, r5=1438.9
- N=2048 keep_untouched: r1=144844.6, r2=137395.7, r3=125021.5, r4=145952.6, r5=151094.3
- N=2048 drop_reused: r1=1438.8, r2=1438.8, r3=1443.4, r4=1440.2, r5=1438.1
- N=2048 drop_pinned: r1=2033.6, r2=2014.4, r3=1967.1, r4=2023.2, r5=2024.6
- N=2048 keep_pinned: r1=6597.5, r2=6792.9, r3=6904.3, r4=6873.1, r5=6551.5
- N=2048 drop_pretouched_split: r1=1432.9, r2=1441.2, r3=1434.8, r4=1435.1, r5=1432.7
- N=2048 keep_pretouched_split: r1=5674.1, r2=5118.2, r3=5349.5, r4=5553.1, r5=5484.9
- N=4096 drop_pretouched: r1=27966.7, r2=26914.8, r3=25788.4, r4=21334.4, r5=27989.4
- N=4096 keep_pretouched: r1=27934.5, r2=28865.0, r3=28128.2, r4=27326.3, r5=28509.1
- N=4096 drop_untouched: r1=25917.9, r2=32315.9, r3=29913.7, r4=31593.0, r5=29189.1
- N=4096 keep_untouched: r1=29620.4, r2=22930.7, r3=31398.5, r4=29843.7, r5=24376.3
- N=4096 drop_reused: r1=11024.9, r2=11443.2, r3=11037.2, r4=11175.5, r5=11492.9
- N=4096 drop_pinned: r1=30317.2, r2=30234.2, r3=30914.8, r4=30400.1, r5=30379.6
- N=4096 keep_pinned: r1=32459.7, r2=31013.5, r3=30846.3, r4=31111.1, r5=31096.5
- N=4096 drop_pretouched_split: r1=28317.0, r2=27691.9, r3=27287.2, r4=24594.3, r5=27472.0
- N=4096 keep_pretouched_split: r1=27245.8, r2=27382.4, r3=27176.2, r4=27408.7, r5=27832.4
