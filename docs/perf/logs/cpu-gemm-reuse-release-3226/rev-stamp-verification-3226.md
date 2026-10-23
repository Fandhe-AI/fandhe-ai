# .rev-stamp とツリー指紋の突合（イシュー #3226）

計測 sha（A/B の before／after・runner が共通で指す commit）: `3c5549027edd58c15917432add08dfd2d34f9d6d`
（ローカルブランチ `perf/3226-eager-release-ab` の計測ツール commit。`git bundle` でノードへ転送し、ノード上で
`git init` → `git fetch <bundle>` → `git checkout --detach` で 3 クローン〈runner・before・after〉を作った。
after のみ `on-arm.patch`〈`EAGER_MATMUL_RELEASE_DEFAULT_ENABLED` の 1 行〉を未コミットで適用）。

## .rev-stamp（各ツリー直下 1 行目）

スクリプトは起動前に before・after・runner の `.rev-stamp` 1 行目が実 HEAD と一致しなければ停止する
（`env_info-3226-*.txt` の `rev_stamp_head` ／ `rev_stamp_head_after` ／ `rev_stamp_head_runner` に記録）。

| 機体 | before | after | runner |
|---|---|---|---|
| GB10 | 3c5549027edd | 3c5549027edd | 3c5549027edd |
| Mac mini | 3c5549027edd | 3c5549027edd | 3c5549027edd |

## ツリー指紋

対象は `crates` と `scripts/bench/framework-compare` の tracked ファイル 2182 件（`Cargo.lock`・`target*`・`results` を除く。
各ファイルの `sha256` を `path sha256` 行に並べソートして連結し、その sha256 を取る）。
ノード側は作業ツリーを直接ハッシュし、手元側は `git archive <計測 sha>` の展開物（after は `on-arm.patch` を適用したもの）を同じ方法でハッシュした。

| ツリー | GB10 | Mac mini | 手元 `git archive` 展開物 | 一致 |
|---|---|---|---|---|
| before | `4fb260640c067bb8f7cdb5523970624d72bebb61af8c72c3e8812a61de8db846` | 同左 | 同左 | 一致 |
| runner | 同上（before と同値） | 同左 | 同左 | 一致 |
| after（patch 適用） | `2c61e2423ae7f47c0e182342ec82be698789642102906bdd99374b26494feef2` | 同左 | 同左 | 一致 |

## RULE のドリフトガード

`env_info-3226-*.txt` の `rule_sha256` = `2d4b390c7a0c2cfe48d7b66daaa897acb400ee94043df0cd1b3638c32b6d6f47`
（両機体）。手元の `shasum -a 256 RULE.txt` と一致（RULE.txt は計測前から未変更）。

## バイナリ

`sha-3226-{before,after}-*.txt` に各腕の `bench-fandhe` の sha256 を収録（ELF は収録しない）。
