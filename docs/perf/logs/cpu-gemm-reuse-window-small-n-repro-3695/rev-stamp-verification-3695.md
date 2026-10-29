# #3695 計測ツリーの同定（GB10 専有 3 系列 s1〜s3・2026-10-29）

`gb10-excl/s1`〜`s3` の診断系列と bench 系列に共通する同定記録。計測機は `<cuda-node>`（GB10・aarch64・compute プロセス 0 件の機体）。
`/home/<user>` 配下のパスは `<home>` へマスクした（ホスト名・ユーザー名はそれ以外の出力に現れなかった。数値は変えていない。マスク後の `judge_3695.py` の出力はマスク前と全行一致）。

## ツリー

| 項目 | 値 |
|---|---|
| 固定コミット（`.rev-stamp` 1 行目。3 系列とも同じ） | `38eb8672cc59b0c17495ee73b2007e4114ede886`（`origin/main`。RULE.txt と部品〈#3698 の `a3df41554`〉を祖先に含むことを確認） |
| `.rev-stamp` 2 行目 | `dirty=0` |
| `env_info_3695.txt` の `rev_stamp_head`・`rev_stamp_dirty` | s1〜s3 とも上記と一致 |
| 転送方式 | `git archive --format=tar <FIX>` を ssh の tar へ流して展開（tracked ファイルのみ） |
| ノード上の `*.local.md`・`.env*`・`settings.local.json` | `find` で 0 件を確認 |
| ツリー指紋（`crates` と `scripts/bench/framework-compare` の tracked ファイル。`Cargo.lock`・`target*` を除く。`<sha256>  <path>` の出力を `LC_ALL=C sort` し全体を sha256） | ノード・Mac（同じ `<FIX>` の `git archive` 展開木）とも `07cd244a1dae6833867f3882eaf7c1fa96b221c1d6b653950d1eab54b329970c`（一致） |
| `framework-compare/Cargo.lock` の sha256（ビルド後。Mac の `git show <FIX>:…` と一致＝復元済み） | `77fc6c23eba323a1f8b84f29aac2ebd163d5939917e6049f194556885cf0305c` |

## bench バイナリ

| 項目 | 値 |
|---|---|
| `bench-fandhe-head` の sha256 | `ec79278bc23c961e64b504be47af27bc3f0103ca4d04ab6137755a6b79c7db6b`（s1〜s3 の `env_info_3695.txt` の `bench_binary_sha256` に一致） |
| path patch | 計測木の `crates/facade`（`<home>` へマスク） |
| 4 腕 | 同一バイナリ。差は CLI フラグと環境変数だけ（README の腕の表） |
| toolchain | rustc 1.97.1（8bab26f4f 2026-07-14・aarch64-unknown-linux-gnu）・cargo 1.97.1 |

## 事前登録規則・実行物

| ファイル | sha256 |
|---|---|
| `RULE.txt` | `65681f10381885460ba4756ffe577b5d8e4eb95bd5a1276639589546eb9d5e53`（s1〜s3 の `rule_sha256` と一致） |
| `session-scripts/session-3695-excl.sh` | `daa48e140bd69532ea18ff7d251922e02ca4d058c953dace29292204b6f2cac9` |
| `judge_3695.py` | `e7ce74d5c470a58f7efab9948daa07ca9b870a829d7c255125d9ae1db4af7256` |
| `../cpu-gemm-reuse-window-small-n-3680/judge_3680.py` | `8c33bfa8977e184d4c2f4290127717a6460293493dfa43dca719090d84d059f2` |
| `../cpu-gemm-reuse-window-small-n-3680/build_bench_head.sh` | `142b7bdd977cc4a2fdb29b14dfe0a70c8c5b002c1ea3cac9c3162fc6e84dfd49` |
| `../cpu-gemm-reuse-window-3671/orchestrate.sh` | `6de799d9ace8901209d21a9e4a4e4f7d2a77b273f5fd978a162ed92c07929827` |
| `../cpu-gemm-reuse-window-3671/aggregate.py` | `fe54b8764d73f824a520c75b07c98376c270383a88a48331b94686260fe6019b` |
| `../cpu-gemm-reuse-window-3671/bench_judge.py` | `40d11c289c83a1ded7141342638ac706358afc78e8e55b21bdaa26129d9433d3` |
| `../cpu-gemm-resident-reuse-3646/excl_check.py` | `ab9dcae3bc53fdf29ab31b7be84cd56764dff70791df0285e8d5e26f488d6ab9` |

規則・閾値（0.5）・tolerance・parity baseline・判定器の判定式は本 PR で変更していない（計測部品はいずれも無改変）。実データで `judge_3695.py` は失敗せず、是正は要らなかった。

## 専有・ゲート

| 系列 | `excl_status.txt` | `gpu_poll.log` | bench の load ゲート（`load_gate_bench.log`） | 診断の load ゲート |
|---|---|---|---|---|
| s1 | `diag excl=pass`・`bench excl=pass` | 239 件すべて `apps=0` | `gate=pass` 301・`gate=unpassed` 0 | run1〜5 とも pass |
| s2 | 同上 | 211 件すべて `apps=0` | pass 301・unpassed 0 | 同上 |
| s3 | 同上 | 235 件すべて `apps=0` | pass 301・unpassed 0 | 同上 |

3 系列とも `steps.log` は `orchestrate-diag rc=0`・`excl_check rc=0`・`done. rc=0`。ゲート不通過の系列はなく、s4・s5 は起動していない。

## 時刻表（`session-scripts/steps.log`、UTC）

| 系列 | hwlock acquired | 診断 rc=0 | bench done | hwlock released | 前の系列の released からの間隔 |
|---|---|---|---|---|---|
| s1 | 2026-10-29T11:43:46Z | 12:05:01Z | 13:43:17Z | 13:43:17Z | - |
| s2 | 2026-10-29T14:16:15Z | 14:34:02Z | 16:01:44Z | 16:01:44Z | 1978 秒（>= 1800） |
| s3 | 2026-10-29T16:36:05Z | 16:54:18Z | 18:33:29Z | 18:33:29Z | 2061 秒（>= 1800） |

## 計測後の再確認

- `judge_3695.py --self-test`・`excl_check.py --self-test`: ともに `self-test: ok`（`python3 -I`、計測前）
- 実ホスト名・`/home/`・ユーザー名の grep（`gb10-excl/` 配下）: 0 件（`fandhe-ai`〈クレート名〉は残してある）
- マスク前後の `judge_3695.py` 出力: 全行一致
