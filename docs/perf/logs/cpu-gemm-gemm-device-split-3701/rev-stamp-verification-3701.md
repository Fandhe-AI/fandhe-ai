# #3701 計測ツリーの同定（GB10 専有 3 系列・Mac mini 3 系列・2026-10-30）

計測機は `<cuda-node>`（GB10・aarch64・compute プロセス 0 件の専有機体）と `<metal-node>`（Mac mini・Apple M4・aarch64）。
`/home/<user>`・`/Users/<user>` 配下のパスは `<home>` へマスクした（数値は変えていない）。ホスト名・ユーザー名は回収後の grep で 0 件を確認した。
マスク前後で `split_inner_judge.py` の出力（ratio・結論・全表）は完全に一致した（`diff` が空。系列の指定パスを同じにして比較した）。

## ツリー

| 項目 | 値 |
|---|---|
| runner（部品のコミット） | `976f691910d64807c98658f4c88d30e532ab9a7b`（#3702 の squash。診断・RULE.txt・部品を含む。計時点なし） |
| after | `976f691910d64807c98658f4c88d30e532ab9a7b+probe-3701.patch(20bafe4d9e6e)`（runner に `probe-3701.patch`〈sha256 `20bafe4d9e6e422f0cebd8732f57437c58dfa3e084a8b4c2febd330691c90600`〉を当てた木） |
| before | `976f691910d64807c98658f4c88d30e532ab9a7b+probe-3701.patch(20bafe4d9e6e)+before-arm.patch(c4dbdedb4906)`（after に `before-arm.patch`〈sha256 `c4dbdedb49067c404870c3831c49fd864c05ab40d5c9aec4618dbd3e3ef69ecc`〉を当てた木） |
| `.rev-stamp` 2 行目（3 本とも） | `dirty=0` |
| 転送方式 | `git archive --format=tar 976f69191` を展開した木から `make_after_tree.sh`・`make_before_tree.sh` で 3 本を作り、ssh の tar へ流して展開（rsync は不使用） |
| ノード上の `*.local.md`・`.env*`・`settings.local.json` | `find` で両機体とも 0 件 |
| RULE.txt の固定時刻 | PR #3702 の最初のコミット `79936242bb`（2026-10-30T01:15:46Z「事前登録規則を固定する」。squash のため main の祖先には無い）。RULE.txt の sha256 は main の版と一致（下記） |
| `976f69191` と作業ブランチの差 | `git diff --stat 976f69191 HEAD -- crates/ scripts/bench/framework-compare` は空（本 PR は docs のみ） |

## ツリー指紋（`crates` と `scripts/bench/framework-compare` の tracked 相当ファイル。`Cargo.lock`・`target*` を除く。相対パスを `LC_ALL=C sort` し `<sha256>  <path>` の出力全体を sha256）

ローカルの展開木と両ノードの展開木で一致した。

| 木 | 指紋 |
|---|---|
| runner | `1c441a6eac4977bc402cdb017900bd2ece9c7723390a756ad43008185f2ae375` |
| after | `b8dcc9f6bee54f2c3962e6f9ee8aac4e023c5d7a495a677342b2e4840f9658c5` |
| before | `59fb9459d2dd716fcba7d8974a956ea61e60753de8cb0ec889b61330be4ae11b` |

## ビルドと腕のバイナリ

- 事前ビルドは各機体で `build_split_arms.sh` を 1 回だけ実行（hw-lock を取る前。計測中はビルドしない）。出力は `arms/gb10-arms.txt`・`arms/macmini-arms.txt`。両機体とも 1 回目で通過した（スクリプトが `LC_ALL=C` を自前で export するため、#3693 のロケール回避は不要だった）
- Mac mini の `diff` は `.claude/skills/*` のディレクトリループ警告を stderr に出したが、検査は通過した（#3693 と同じ）

| 機体 | diag-before sha256 | diag-after sha256 | rustc |
|---|---|---|---|
| gb10 | `227d38ed5ad2f94c536350980fba1e038152375d79e9dcb7f26fb4509dcb3a7b` | `90c2cd7a509662eec7740dd673ac5082564a403ae9ac7560aa735db0e67b6800` | 1.97.1 |
| macmini | `4e8931bd89e15c13e539fcd505d4e33a34543f347c1a5ff30c5ac57a5a7b294b` | `dc9a046ac6e98e1f22757f491994125c0e6ce3a7ad29b0ce1242900ad5a96810` | 1.99.0 |

## 規則・スクリプトの同定

- 6 系列すべての `env_info.txt` の `rule_sha256` が `08b0782b77009959b90a15cdecdfc89449623d2d9e77643adbaeaf7fc6b4bee9`（マージ済み RULE.txt と一致。`split_inner_judge.py` も検査する）
- スクリプトの sha256（runner の版。編集していない）:
  `orchestrate.sh` `2d2916920a17fa075ac43ba6f3d366dca025e38044467e50568c2a948d221777`・
  `session-scripts/session-3701.sh` `8826b7451739f871f47db7820cb5d8136d56ff980428a0c5da261790b0155d41`・
  `split_inner_judge.py` `82e2f7132044d3f914fa83ce83df3c8a7e9b5a72c2b701f8c4d1159193e6df4b`・
  `build_split_arms.sh` `f5e14ae4d38e4d100fde4e032d5a7efa085ed1dec01197f2b9c1d8a0e422e584`・
  `make_after_tree.sh` `c6cb9c7084ff97879566f1ad273a5e68a6d936e0dc35e70d85f23bcb0c68421e`・
  `make_before_tree.sh` `41d1278cc30aee97f001cf87073421a5e4af1500234b4bba5d6f7e0b6b7a528e`・
  `excl_check.py`（#3646 の版）`ab9dcae3bc53fdf29ab31b7be84cd56764dff70791df0285e8d5e26f488d6ab9`
- 判定器・RULE・スクリプトは変更していない（実データでも入力不備にならず、修正は不要だった）

## 系列の時刻と順（judge が検査しない項目を手で確認）

| 機体 | 系列 | 先頭腕 | 開始（UTC） | 完了（UTC） | 前の完了からの間隔 | ゲート |
|---|---|---|---|---|---|---|
| gb10 | s1 | before | 02:24:41 | 02:27:20 | - | 5/5 run pass・`excl=pass` |
| gb10 | s2 | after | 02:58:20 | 02:59:00 | 31 分 00 秒 | 5/5 run pass・`excl=pass` |
| gb10 | s3 | before | 03:30:00 | 03:31:09 | 31 分 00 秒 | 5/5 run pass・`excl=pass` |
| macmini | s1 | before | 02:24:44 | 02:28:52 | - | 5/5 run pass |
| macmini | s2 | after | 02:59:52 | 02:59:59 | 31 分 00 秒 | 5/5 run pass |
| macmini | s3 | before | 03:30:59 | 03:31:06 | 31 分 00 秒 | 5/5 run pass |

- 系列間は 30 分以上。先頭腕は `first_arm.txt` が before／after／before の順で交互。各 run の腕順は `load_gate.log` の `order=`（ABBA）
- 全系列がゲートを通ったため s4・s5 は起動していない。judge には起動順の s1・s2・s3 を渡した。値を見て系列を選んでいない
- GB10 の `gpu_poll.log`（30 秒間隔）は全行 `apps=0`（s1 6 行・s2 2 行・s3 3 行）。pre／post スナップショットも含め `excl_check.py` が 3 系列とも pass。起動前の `nvidia-smi --query-compute-apps` も 0 件
- 系列の所要時間は s1 が GB10 約 160 秒・Mac mini 約 250 秒、s2・s3 は 7〜70 秒だった。s1 の長さの大半は負荷ゲートの待機で説明できる（`load_gate.log` の `waited=`。GB10 s1 は run1 で 90 秒＋run5 で 60 秒の計 150 秒、Mac mini s1 は run1 で 240 秒。s1 の run2〜4 は待機 0 秒）。待機を除いた実行時間は GB10 s1 が約 10 秒、Mac mini s1 が約 10 秒で、診断処理自体が s1 だけ遅いとは読めない。待機の原因（run 開始時の load1 が GB10 で 2.20・1.65、Mac mini で 17.90 と高かったこと）は load_gate.log の記録までで、それ以上は調べていない。run の出力量（jsonl 約 28 KB）と標本数は系列間で同等で、判定器は全系列を受理した。s1 は before 先頭の系列のため、判定は ABBA で腕の順を打ち消す設計に依る
- 2 機体は別ハードのため並行して起動した（hw-lock は機体ごと。計測中に他のビルドは走らせていない）
- 系列の起動は `session-3701.sh` をそのまま使い、系列の間隔だけ、切り離した直列ラッパー（リポジトリ外・値に触れない `sleep 1860` のみ。sha256 `78a76efcda6f271db1f3345aa1ab7df04afd335bf5e628d046b7276eb3cf0f19`）であけた
