# #3680 計測ツリーの同定（GB10 専有系列・2026-10-28〜29）

`gb10-excl/` の診断系列と bench 系列に共通する同定記録。計測機は `<cuda-node>`（GB10・aarch64・compute プロセス 0 件の機体）。
`/home/<user>` 配下のパスは `<home>` へマスクした（ホスト名・ユーザー名は出力に現れなかった。数値は変えていない。マスク後の `judge_3680.py` の出力はマスク前と全行一致）。

## ツリー

| 項目 | 値 |
|---|---|
| 固定コミット（`.rev-stamp` 1 行目） | `8931dcf43a8fb0fdfb68264a61227b310833a1bd`（#3686 マージ直後の `origin/main`。RULE.txt〈#3685〉と `download_owned`〈#3678〉を含む祖先であることを確認） |
| `.rev-stamp` 2 行目 | `dirty=0` |
| `env_info_3680.txt` の `rev_stamp_head`・`rev_stamp_dirty` | 上記と一致 |
| 転送方式 | `git archive --format=tar <FIX>` を ssh の tar へ流して展開（tracked ファイルのみ。rsync は不使用） |
| ノード上の `*.local.md`・`.env*`・`settings.local.json` | `find` で 0 件を確認 |
| ツリー指紋（`crates` と `scripts/bench/framework-compare` の tracked ファイル。`Cargo.lock`・`target*` を除く。相対パスを `LC_ALL=C sort`・`<sha256>  <path>` の出力全体を再度 sha256） | ノード・Mac（同じ `<FIX>` の `git archive` 展開木）とも `d614ef292216df82e7276ffdc3feee236a5fcfc6b8fc8624935d6268626f4746`（2512 ファイル。一致） |
| `framework-compare/Cargo.lock` の sha256（ビルド後。Mac の `git show <FIX>:…` と一致＝復元済み） | `77fc6c23eba323a1f8b84f29aac2ebd163d5939917e6049f194556885cf0305c` |
| `make deps-forbidden`（Mac・計測後） | pass（承認ピンのまま・禁止リスト混入なし） |

## bench バイナリ

| 項目 | 値 |
|---|---|
| `bench-fandhe-head` の sha256 | `4274b6f5abecd9ed350627c94df7763bd9515c1bb7612a23f2e73c958dbf2894`（`env_info_3680.txt` の `bench_binary_sha256` と `session-scripts/bench_arms.txt` に一致） |
| path patch（`bench_arms.txt`） | 計測木の `crates/facade`（`<home>` へマスク） |
| 5 腕 | 同一バイナリ。差は 1 要因（README の腕の表） |
| toolchain | rustc 1.97.1（8bab26f4f 2026-07-14・aarch64-unknown-linux-gnu）・cargo 1.97.1 |

## 事前登録規則・実行物

| ファイル | sha256 |
|---|---|
| `RULE.txt` | `e643bc9919eade71cf8ae1125ff58412ea4e06c03b0c706ccec6d45b89f7b9d1`（`env_info_3680.txt` の `rule_sha256` と一致） |
| `session-scripts/session-3680-excl.sh` | `4785c2378a2f98b5ddf2e23bb4dbb8de1c3deb2f70c549a4772ca5388e7105eb` |
| `../cpu-gemm-reuse-window-3671/orchestrate.sh` | `6de799d9ace8901209d21a9e4a4e4f7d2a77b273f5fd978a162ed92c07929827` |
| `../cpu-gemm-reuse-window-3671/aggregate.py` | `fe54b8764d73f824a520c75b07c98376c270383a88a48331b94686260fe6019b` |
| `../cpu-gemm-reuse-window-3671/bench_judge.py` | `40d11c289c83a1ded7141342638ac706358afc78e8e55b21bdaa26129d9433d3` |
| `judge_3680.py` | `8c33bfa8977e184d4c2f4290127717a6460293493dfa43dca719090d84d059f2` |
| `build_bench_head.sh` | `142b7bdd977cc4a2fdb29b14dfe0a70c8c5b002c1ea3cac9c3162fc6e84dfd49` |
| `../cpu-gemm-resident-reuse-3646/excl_check.py` | `ab9dcae3bc53fdf29ab31b7be84cd56764dff70791df0285e8d5e26f488d6ab9` |

規則・閾値（0.5）・tolerance・parity baseline・判定器の判定式は本 PR で変更していない（計測部品はいずれも無改変）。実データで `judge_3680.py` は失敗せず、是正は要らなかった。

## 専有・ゲート

| 項目 | 結果 |
|---|---|
| `excl_status.txt` | `diag excl=pass`・`bench excl=pass`（`excl_check.py --series diag,bench`） |
| `gpu_poll.log` | 275 件すべて `apps=0`（最初 2026-10-28T21:43:13Z・最後 2026-10-29T00:00:22Z） |
| bench の起動前 load ゲート | `load_gate_bench.log` の `gate=pass` 376 件・`gate=unpassed` 0 件 |
| 診断の load ゲート | `gb10/load_gate_status.txt` が run1〜5 とも `gate=pass` |

## 時刻表（`session-scripts/steps.log`）

| 時刻 | 事象 |
|---|---|
| 2026-10-28T21:43:13Z | hwlock acquired・start |
| 2026-10-28T22:03:32Z | 診断 `orchestrate-diag rc=0` |
| 2026-10-29T00:00:34Z | bench done・session body done・`done.`・hwlock released |

## 計測後の再確認

- `judge_3680.py --self-test`・`excl_check.py --self-test`: ともに `self-test: ok`（`python3 -I`）
- 実ホスト名・`/home/`・ユーザー名の grep（`gb10-excl/` 配下）: 0 件（`fandhe-ai`〈クレート名〉は残してある）
