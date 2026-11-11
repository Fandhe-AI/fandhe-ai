# .rev-stamp とツリー指紋の突合（イシュー #4158）

計測コミット（before／after／runner が共通で指す commit）: `d50251b385a32f0383c4f2b60a287b84cf0f95d8`
（`RULE.txt` を単独で追加した #4224 のマージ後の main。`git bundle` でノードへ転送し、ノード上で
`git init` → `git fetch <bundle>` → `git checkout --detach FETCH_HEAD` で 3 クローン〈runner・before・after〉を作った。
ノード上で GitHub への fetch はしていない。after のみ `on-arm.patch` を未コミットで `git apply` した）。

## source_commit との差分（RULE [対象]・[ドリフトガード]）

`git diff --quiet 00ce065462130995cb35d44fd216efa6f279a9ea d50251b385a32f0383c4f2b60a287b84cf0f95d8 -- crates scripts/bench/framework-compare`
は成り立った（差分なし）。手元とノードの runner クローンの両方で計測前に確かめた。

## .rev-stamp（各ツリー直下。1 行目 = HEAD・2 行目 `dirty=0`）

runner は起動前に before・after・runner の `.rev-stamp` 1 行目が実 HEAD と一致しなければ停止する
（`gb10/env_info-4158-4158-gb10.txt` の `rev_stamp_head`／`rev_stamp_head_after`／`rev_stamp_head_runner`）。

| 機体 | before | after | runner |
|---|---|---|---|
| GB10 | d50251b385a3 | d50251b385a3 | d50251b385a3 |

after の作業ツリーの差分は `crates/autodiff/src/eager_release.rs` の 1 ファイル 1 行（`on-arm.patch`。sha256
`efb25b5fb70947dcf268518232debfc83f8e7e951b3d8a5f88bfa0e650f33978` を転送後のノード上でも確認）。

## ツリー指紋

対象は `crates` と `scripts/bench/framework-compare` の tracked ファイル 2412 件（`Cargo.lock`・`target*`・`results` を除く。
各ファイルの `path sha256` 行を `LC_ALL=C` でソートして連結し、その sha256 を取る）。
ノード側は計測後の各クローンの `git ls-files` を直接ハッシュし、手元側は `git archive <計測コミット>` の展開物
（after は `on-arm.patch` を適用したもの）を同じ方法でハッシュした。

| ツリー | GB10 | 手元 `git archive` 展開物 | 一致 |
|---|---|---|---|
| before | `da329d5c9e757ebb558836a3d670108c64a84449a62abbf25fa40ae6049b7c21` | 同左 | 一致 |
| runner | 同上（before と同値） | 同左 | 一致 |
| after（patch 適用） | `eaea2eb59cdd3e16408683e59edfb4f58c9293d9c995be4cd74838bfab68de37` | 同左 | 一致 |

## RULE のドリフトガード

`gb10/env_info-4158-4158-gb10.txt` の `rule_sha256` = `ede20e07961403c18a0d09010f0f039aee91cee92144662d80842bb9ff015f4d`。
手元の `shasum -a 256 RULE.txt` と一致（RULE.txt は計測前から未変更）。

## precheck・バイナリ

- precheck（`AB_PRECHECK=assert`）: 両腕の `size_gate_max_bytes=33554432`、before `enabled=false released=false`・after
  `enabled=true released=true`（`gb10/eager-release-report-4158-*.txt`）。
- `gb10/sha-4158-{before,after}-4158-gb10.txt` に各腕の `bench-fandhe` の sha256 を収録（ELF は収録しない）。
