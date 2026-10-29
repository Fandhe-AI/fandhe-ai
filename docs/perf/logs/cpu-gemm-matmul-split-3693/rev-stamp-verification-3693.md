# #3693 計測ツリーの同定（GB10 専有 3 系列・Mac mini 3 系列・2026-10-29）

計測機は `<cuda-node>`（GB10・aarch64・compute プロセス 0 件の専有機体）と `<metal-node>`（Mac mini・Apple M4・aarch64）。
`/home/<user>`・`/Users/<user>` 配下のパスは `<home>` へマスクした（数値は変えていない）。ホスト名・ユーザー名は回収後の grep で 0 件を確認した。
マスク前後で `split_judge.py` の ratio・結論・全表は一致した（差は「系列の扱い」節の系列名だけで、マスク前はディレクトリ名 `gb10`／`macmini` が出る）。

## ツリー

| 項目 | 値 |
|---|---|
| runner・after（部品のコミット） | `91c57a281e7a57a3f9ad383f7746222a83575fd1`（#3697 の squash。診断・RULE.txt・部品を含む） |
| before | `91c57a281e7a57a3f9ad383f7746222a83575fd1+before-arm.patch(c4dbdedb4906)`（after に `before-arm.patch`〈sha256 `c4dbdedb49067c404870c3831c49fd864c05ab40d5c9aec4618dbd3e3ef69ecc`〉を当てた木） |
| `.rev-stamp` 2 行目（3 本とも） | `dirty=0` |
| 転送方式 | `git archive --format=tar <sha>` を展開した木を ssh の tar へ流して展開（rsync は不使用） |
| ノード上の `*.local.md`・`.env*`・`settings.local.json` | `find` で両機体とも 0 件 |
| RULE.txt の固定時刻 | PR #3697 の最初のコミット `9015d073e1`（2026-10-29T06:14:19Z「事前登録規則を固定する」）。RULE.txt の sha256 は main の版と一致（下記） |
| `91c57a281` と `origin/main` の差 | `git diff --stat 91c57a281 origin/main -- crates/ scripts/bench/framework-compare` は空（その後の変更は docs のみ） |

## ツリー指紋（`crates` と `scripts/bench/framework-compare` の tracked ファイル。`Cargo.lock`・`target*` を除く。相対パスを `LC_ALL=C sort` し `<sha256>  <path>` の出力全体を sha256）

ローカルの展開木と両ノードの展開木で一致した。

| 木 | 指紋 |
|---|---|
| runner | `99dc66d91bf38c9285ef3e795f3ac20b9bb88c4d908c662ae6012d5d3d990edb` |
| before | `de59442bf983eb16a8a3d77fc8c034f5bdb944519f998f5128a2724ffb1474ba` |
| after | `99dc66d91bf38c9285ef3e795f3ac20b9bb88c4d908c662ae6012d5d3d990edb`（runner と同一） |

## ビルドと腕のバイナリ

- 事前ビルドは各機体で `build_split_arms.sh` を 1 回だけ実行（hw-lock を取る前。計測中はビルドしない）。出力は `arms/gb10-arms.txt`・`arms/macmini-arms.txt`
- GB10 のロケールが日本語のため、`build_split_arms.sh` のハード条件検査（`diff -rq` の出力文字列 `Files … differ` との完全一致）が 1 回目は文言違いで中止した（`ファイル … は異なります`。差は `resident_input.rs` の 1 ファイルだけで、条件自体は満たしていた）。スクリプトは編集せず、`LC_ALL=C` を環境に付けて再実行し通過した（計測前。計測値には無関係）。Mac mini は英語ロケールで 1 回目から通過
- Mac mini の `diff` は `.claude/skills/update-issue-tree` のディレクトリループ警告を stderr に出したが、検査は通過した

| 機体 | diag-before sha256 | diag-after sha256 | rustc |
|---|---|---|---|
| gb10 | `3e92ae0f61114e0fe55b1bbacf8e00bb1c83c4958439b7fec81555f49a8e339f` | `cc958418a376da57b21d0af67772e9c6b8ce9a012eead8ea31228343c930b2c9` | 1.97.1 |
| macmini | `b8a8782cc0c332790885505771df00425abeac3dc919d9248fd84c5f839f9752` | `d7ed7c8165e473267c5bafb97d966373e648155e9e9021f9d1e3722a4eccf6f0` | 1.99.0 |

## 規則・スクリプトの同定

- 6 系列すべての `env_info.txt` の `rule_sha256` が `2fe453eb45382a29bab55eac4b4195712fb0b289f9a8920e4e8c749b63ba56d8`（マージ済み RULE.txt と一致。`split_judge.py` も検査する）
- スクリプトの sha256（runner の版。編集していない）:
  `orchestrate.sh` `c217a70f27d4f1bcfabb4cab0f9047c33daf116a2f73950e34fb79b6ad27e83d`・
  `session-scripts/session-3693.sh` `29bdf05e8d882e8cb4bbd76f46aef2d5a2378933a68f077ca095d6a822c9f099`・
  `split_judge.py` `93889d0686c8a6e7b3c16cd2ccb74d44d0019d4b37a007a75a79281ff31ed9b0`・
  `build_split_arms.sh` `73c72a647388a1e78ac8ae72c4448c57eca4bab0eabc46605a4ac4277953f79f`・
  `make_before_tree.sh` `d527fccddeaa3e590909c11b2b6f68fb6b15d260e2cd099450a955f27c184c41`・
  `excl_check.py` `ab9dcae3bc53fdf29ab31b7be84cd56764dff70791df0285e8d5e26f488d6ab9`
- `session-3693.sh`・`orchestrate.sh` のヘッダコメントは旧名 `build_ab_arms.sh` を参照しているが、実体は `build_split_arms.sh`（挙動は同一。runner の sha を保つため直していない）
- 判定器・RULE・スクリプトは変更していない（実データでも入力不備にならず、判定器の修正は不要だった）

## 系列の時刻と順（judge が検査しない項目を手で確認）

| 機体 | 系列 | 先頭腕 | 開始（UTC） | 完了（UTC） | 前の完了からの間隔 | ゲート |
|---|---|---|---|---|---|---|
| gb10 | s1 | before | 09:58:12 | 10:00:52 | - | 5/5 run pass・`excl=pass` |
| gb10 | s2 | after | 10:31:52 | 10:32:01 | 31 分 00 秒 | 5/5 run pass・`excl=pass` |
| gb10 | s3 | before | 11:03:02 | 11:03:11 | 31 分 01 秒 | 5/5 run pass・`excl=pass` |
| macmini | s1 | before | 09:59:13 | 10:01:50 | - | 5/5 run pass |
| macmini | s2 | after | 10:32:51 | 10:32:58 | 31 分 01 秒 | 5/5 run pass |
| macmini | s3 | before | 11:03:58 | 11:04:35 | 31 分 00 秒 | 5/5 run pass |

- 系列間は 30 分以上。先頭腕は `first_arm.txt` が before／after／before の順で交互。各 run の腕順は `load_gate.log` の `order=`（ABBA）
- 全系列がゲートを通ったため s4・s5 は起動していない。judge には起動順の s1・s2・s3 を渡した
- GB10 の `gpu_poll.log`（30 秒間隔）は全行 `apps=0`（s1 6 行・s2 1 行・s3 1 行）。pre／post スナップショットも含め `excl_check.py` が 3 系列とも pass
- 系列の所要時間は s1 が約 160 秒（両機体）、s2・s3 は 9〜37 秒だった。run の出力量（jsonl 約 15 KB）と標本数は同等で判定器は全系列を受理した。s1 だけ長い理由は調べていない（初回起動に関わる何か、という推測にとどまる）。s1 は before 先頭の系列のため、判定は ABBA で腕の順を打ち消す設計に依る
- 2 機体は別ハードのため並行して起動した（hw-lock は機体ごと。計測中に他のビルドは走らせていない）
- 系列の起動は `session-3693.sh` をそのまま使い、s2・s3 の起動時刻だけ、切り離した直列ラッパー（リポジトリ外・値に触れない `sleep 1860` のみ。sha256 `86a730d27ff94938ffd33999948a0637ea6c775af675e16a1ced558d66d84e9a`）であけた
