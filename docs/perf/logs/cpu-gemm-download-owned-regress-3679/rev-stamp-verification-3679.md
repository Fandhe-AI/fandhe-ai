# #3679 計測ツリーの同定（GB10 専有 3 系列・Mac mini 3 系列・2026-10-29）

計測機は `<cuda-node>`（GB10・aarch64・compute プロセス 0 件の専有機体）と `<metal-node>`（Mac mini・Apple M4・aarch64）。
`/home/<user>`・`/Users/<user>` 配下のパスは `<home>` へマスクした（ホスト名・ユーザー名は出力に現れなかった。数値は変えていない）。
マスク前後で `repro_judge.py` の ratio・結論・区間行は全行一致した（差は「系列の扱い」節の系列名だけで、マスク前はディレクトリ名 `gb10`／`macmini` が出る）。

## ツリー

| 項目 | 値 |
|---|---|
| runner（部品のコミット） | `7f803368346fc5e9c393b7610e417f48881b834e`（`origin/main`。RULE.txt〈#3684〉を含む） |
| before | `a7aac2f0227add243845418e0897ac0675c5b622` |
| after | `f7ce20fc91730216e8f3f60514b181f32ebfee48` |
| `.rev-stamp` 2 行目（3 本とも） | `dirty=0` |
| 転送方式 | `git archive --format=tar <sha>` を ssh の tar へ流して展開（tracked ファイルのみ。rsync は不使用） |
| ノード上の `*.local.md`・`.env*`・`settings.local.json` | `find` で両機体とも 0 件 |
| RULE.txt のコミット時刻（固定時刻） | `2026-10-29T02:43:19+09:00`（`76f1dc369`） |

## ツリー指紋（`crates` と `scripts/bench/framework-compare` の tracked ファイル。`Cargo.lock`・`target*` を除く。相対パスを `LC_ALL=C sort` し `<sha256>  <path>` の出力全体を sha256）

Mac 側の展開木とノード側で一致した（両機体とも同値）。

| 木 | 指紋 |
|---|---|
| runner | `d614ef292216df82e7276ffdc3feee236a5fcfc6b8fc8624935d6268626f4746` |
| before | `1d0ec276c8fa8495cddfaa2e59d1d4ec77d341b5c3c26d631a01d0bf7ce206cf` |
| after | `cae11995aea01ee7d6f72d0485aff168e83f554556e81ba3e246ea82ac1f4d8c` |

## ビルドと腕のバイナリ

- 事前ビルドは各機体で `build_ab_arms.sh` を 1 回だけ実行（hw-lock を取る前。計測中はビルドしない）。出力は `arms/gb10-arms.txt`・`arms/macmini-arms.txt`（`bench-*` は本 Issue で使わない）
- `framework-compare/Cargo.lock` の sha256 は `77fc6c23eba323a1f8b84f29aac2ebd163d5939917e6049f194556885cf0305c`（before の `git show a7aac2f0:…/Cargo.lock` と一致。ビルド後に復元済み）

| 機体 | diag-before sha256 | diag-after sha256 | rustc |
|---|---|---|---|
| gb10 | `103ad827ac2af8feff47d710845d87d95adf15b6950922d7a8d414bef9b53733` | `5796acc84e60b5a50d0d79dccf37faaa353b76fb56ccdf3b605e58150fd2e9ed` | 1.97.1 |
| macmini | `da0c8b82c95cfe360ec6270a8ebaff8cfc0cf86e95730e526975fe0f8e2e880c` | `6152f48de487065d00da69fc91549a0287393f57c3f3ffd24187df0c892afcd7` | 1.99.0 |

## 規則・スクリプトの同定

- 6 系列すべての `env_info.txt` の `rule_sha256` が `75c4f3c5306b9d16bc8e50c60e024bb292015eb8ee9cc8e5f561aa04ec35feb3`（マージ済み RULE.txt と一致。`repro_judge.py` も検査する）
- スクリプトの sha256（runner の版。編集していない）:
  `orchestrate.sh` `10bb17719234e780137a3ee5b9198191b20b4e4cda056ea3146ea388e5ed5eba`・
  `session-scripts/session-3679.sh` `a57ad919e5f96b4cb52cc8ece6fd338b064c98a802501294b3ec7b0cf503d4ab`・
  `repro_judge.py` `da209abec8a4964aa0694cc0788541c0896cbf9c7c6a28ce83fe9cc8a6649559`・
  `build_ab_arms.sh` `07fb2d64864b5c3519de4d70283ac3ed49a4885758bd94ba328ecd60964d0fc3`・
  `excl_check.py` `ab9dcae3bc53fdf29ab31b7be84cd56764dff70791df0285e8d5e26f488d6ab9`
- 判定器・RULE・スクリプトは変更していない（実データでも入力不備にならず、判定器の修正は不要だった）

## 系列の時刻と順（judge が検査しない項目を手で確認）

| 機体 | 系列 | 先頭腕 | 開始（UTC） | 完了（UTC） | 前の完了からの間隔 | ゲート |
|---|---|---|---|---|---|---|
| gb10 | s1 | before | 00:49:13 | 01:18:52 | - | 5/5 run pass・`excl=pass` |
| gb10 | s2 | after | 01:49:18 | 02:31:58 | 30 分 26 秒 | 5/5 run pass・`excl=pass` |
| gb10 | s3 | before | 03:02:58 | 03:33:08 | 31 分 00 秒 | 5/5 run pass・`excl=pass` |
| macmini | s1 | before | 00:49:16 | 01:28:57 | - | 5/5 run pass |
| macmini | s2 | after | 01:59:26 | 02:37:56 | 30 分 29 秒 | 5/5 run pass |
| macmini | s3 | before | 03:08:56 | 03:48:56 | 31 分 00 秒 | 5/5 run pass |

- 系列間は 30 分以上（steps.log の開始時刻と前系列の `done.` 行の時刻の差）。先頭腕は `first_arm.txt` が before／after／before の順で交互
- 全系列がゲートを通ったため s4・s5 は起動していない。judge には起動順の s1・s2・s3 を渡した
- GB10 の `gpu_poll.log`（30 秒間隔）は全行 `apps=0`（s1 60 行・s2 86 行・s3 61 行。各系列で最初と最後の時刻は `gb10/sN/session/gpu_poll.log`）
- 2 機体は別ハードのため並行して起動した（hw-lock は機体ごと。計測中に他のビルドは走らせていない）
- 系列の起動は `session-3679.sh` をそのまま使い、s2・s3 の起動時刻だけ、切り離した待機スクリプト（リポジトリ外・値に触れない `sleep` のみ）で 30 分以上あけた
