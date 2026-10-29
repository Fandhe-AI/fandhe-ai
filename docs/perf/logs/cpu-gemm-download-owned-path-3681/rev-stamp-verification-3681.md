# #3681 計測ツリーの同定（GB10 専有 1 系列・Mac mini 1 系列・2026-10-29）

計測機は `<cuda-node>`（GB10・aarch64・compute プロセス 0 件の専有機体）と `<metal-node>`（Mac mini・Apple M4・aarch64）。
`/home/<user>`・`/Users/<user>` 配下のパスは `<home>` へマスクした（ホスト名・ユーザー名は出力に現れなかった。数値は変えていない）。
マスク前のデータとマスク後のコミット対象データのそれぞれに `path_judge.py` を実行し、出力 `judge-3681.md` はバイト一致した（差なし）。

## ツリー

| 項目 | 値 |
|---|---|
| runner（部品のコミット） | `82bfc2e28e9b08186c31d898d6aa1a302e434957`（`origin/main`。RULE.txt〈#3686〉を含む） |
| before | `a7aac2f0227add243845418e0897ac0675c5b622` |
| after | `f7ce20fc91730216e8f3f60514b181f32ebfee48` |
| `.rev-stamp` 2 行目（3 本とも） | `dirty=0` |
| 転送方式 | `git archive --format=tar <sha>` を ssh の tar へ流して展開（tracked ファイルのみ。rsync は不使用） |
| ノード上の `*.local.md`・`.env*`・`settings.local.json` | `find` で両機体とも 0 件 |

## ツリー指紋（`crates` と `scripts/bench/framework-compare` の tracked ファイル。`Cargo.lock`・`target*` を除く。相対パスを `LC_ALL=C sort` し `<sha256>  <path>` の出力全体を sha256）

ローカルの展開木とノード側で一致した（両機体とも同値。#3679 と同じ値で、crates 配下は #3679 の runner から不変）。

| 木 | 指紋 |
|---|---|
| runner | `d614ef292216df82e7276ffdc3feee236a5fcfc6b8fc8624935d6268626f4746` |
| before | `1d0ec276c8fa8495cddfaa2e59d1d4ec77d341b5c3c26d631a01d0bf7ce206cf` |
| after | `cae11995aea01ee7d6f72d0485aff168e83f554556e81ba3e246ea82ac1f4d8c` |

## ビルドと腕のバイナリ

- 事前ビルドは各機体で `build_ab_arms.sh` を 1 回だけ実行（hw-lock を取る前。計測中はビルドしない）。出力は `arms/gb10-arms.txt`・`arms/macmini-arms.txt`
- `framework-compare/Cargo.lock` の sha256 は両機体とも `77fc6c23eba323a1f8b84f29aac2ebd163d5939917e6049f194556885cf0305c`（ビルド後に復元済みで #3679 と同値）

| 機体 | diag-before | diag-after | bench-before | bench-after | rustc |
|---|---|---|---|---|---|
| gb10 | `103ad827ac2af8feff47d710845d87d95adf15b6950922d7a8d414bef9b53733` | `5796acc84e60b5a50d0d79dccf37faaa353b76fb56ccdf3b605e58150fd2e9ed` | `159ad8d146921443bae7e83a164431892f1b2457d33714b82c8818941887db26` | `70d620c8a0cdf084539ce4fd5c72992e0815712c8f35e9998a00836ab0b3022a` | 1.97.1 |
| macmini | `da0c8b82c95cfe360ec6270a8ebaff8cfc0cf86e95730e526975fe0f8e2e880c` | `6152f48de487065d00da69fc91549a0287393f57c3f3ffd24187df0c892afcd7` | `43ced4c3cf581f2609478b272a71f3fb2499a3c17bf96a36f15e08a21ad3dbec` | `8e31f6afb7fa4ce1c8da3bcf59419f3f3682c6163a48434912ef3da56ee4965b` | 1.99.0 |

diag の sha256 は #3679 の同機体の値と一致する（同じ木・同じ rustc からのビルド）。

## 規則・スクリプトの同定

- 2 機体の `env_info.txt` の `rule_sha256` が `48b3da16e68c53cf111a14e0fb431d3cb149f5816705e43c04339002dcf7614e`（マージ済み RULE.txt と一致。`path_judge.py` も検査する）
- スクリプトの sha256（runner の版。編集していない）:
  `orchestrate.sh` `bb9339c32ffa5a5dc619177a53a3df8ee371370837029cdb6472a8b805d90215`・
  `session-scripts/session-3681.sh` `b2160c283e45ea0418e35cfef49d4c35e9d13115ddf130fdd9904d6f468c04e8`・
  `path_judge.py` `1afa378d9dfd84ec278b0ffb2c684ea518f604c4fd7f8dfeedea9337ba4ad535`・
  `build_ab_arms.sh` `07fb2d64864b5c3519de4d70283ac3ed49a4885758bd94ba328ecd60964d0fc3`・
  `excl_check.py` `ab9dcae3bc53fdf29ab31b7be84cd56764dff70791df0285e8d5e26f488d6ab9`・
  `ab_judge.py` `748deb3bf7bf3cb55244aaf4961031db9b18fdbea9917ff7b867133491252af4`（`path_judge.py` の `AB_JUDGE_SHA256` と一致）
- 判定器・RULE・スクリプトは変更していない（実データでも入力不備にならず、判定器の修正は不要だった）。`python3 -I path_judge.py --self-test` は PASS

## RULE の固定時刻と計測開始の比較（RULE が先であることの根拠）

| 項目 | 時刻 |
|---|---|
| RULE.txt のコミット（`8931dcf43`。`git log -1 --format=%cI -- RULE.txt`） | `2026-10-29T05:51:45+09:00`（= `2026-10-28T20:51:45Z`） |
| gb10 計測開始（`env_info.txt` の `date_utc`・`steps.log` 先頭） | `2026-10-29T04:20:06Z` |
| macmini 計測開始（同） | `2026-10-29T04:20:06Z` |

RULE の固定は 2 機体の計測開始より約 7 時間 28 分前（`04:20:06Z` − `20:51:45Z`）。

## 系列の時刻とゲート

| 機体 | 系列 | 開始（UTC） | 完了（UTC） | ゲート |
|---|---|---|---|---|
| gb10 | s1 | 04:20:06 | 04:55:11 | 5/5 run pass・`excl=pass`（`excl_check` rc=0） |
| macmini | s1 | 04:20:06 | 05:04:43 | 5/5 run pass |

- 終端行は 2 機体とも `done. rc=0`（orchestrate・excl_check〈gb10 のみ〉・path_judge の rc はすべて 0）
- GB10 の `gpu_poll.log`（30 秒間隔）は 71 行すべて `apps=0`（最初 `04:20:06Z`・最後 `04:55:08Z`）
- 2 機体は別ハードのため並行して起動した（hw-lock は機体ごと。計測中に他のビルドは走らせていない）。系列は 1 本ずつで、追加の系列は起動していない（RULE が認めるやり直しは計測前に止まった場合の 1 回だけで、該当しなかった）
