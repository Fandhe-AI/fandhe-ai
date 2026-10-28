# #3671 計測ツリーの同定（GB10 専有系列・2026-10-28）

`gb10-excl/` の全系列に共通する同定記録。計測機は `<cuda-node>`（GB10・aarch64・compute プロセス 0 件の機体）。
ホスト名・`/home` 配下のパス・ユーザー名は回収時にマスクした（数値は変えていない）。

## ツリー

| 項目 | 値 |
|---|---|
| 固定コミット（`.rev-stamp` 1 行目） | `302f6ae9dce4fe6b9f5e387c669b3338f80ab637`（PR #3675 マージ直後の `origin/main`） |
| `.rev-stamp` 2 行目 | `dirty=0` |
| `env_info.txt` の `rev_stamp_head`・`rev_stamp_dirty` | 上記と一致（`gb10`・`gb10-solo`・`gb10-minflt` の各 `env_info.txt`。各 `aggregate.md` 先頭にも同値） |
| 転送方式 | `git archive --format=tar <FIX>` を ssh の tar へ流して展開（tracked ファイルのみ。rsync は不使用） |
| ノード上の `*.local.md`・`.env*`・`settings.local.json` | `find` で 0 件を確認 |
| ツリー指紋（`crates` と `scripts/bench/framework-compare` の tracked ファイル。`Cargo.lock`・`target*` を除く。相対パスを `LC_ALL=C sort`・`<sha256>  <path>` の出力全体を再度 sha256） | ノード・Mac（同じ `<FIX>` の worktree）とも `1d0ec276c8fa8495cddfaa2e59d1d4ec77d341b5c3c26d631a01d0bf7ce206cf`（2512 ファイル。一致） |
| `framework-compare/Cargo.lock` の sha256（ビルド後。Mac の同ファイルと一致＝復元済み） | `77fc6c23eba323a1f8b84f29aac2ebd163d5939917e6049f194556885cf0305c` |
| `make deps-forbidden`（Mac・計測後） | pass（承認ピンのまま・禁止リスト混入なし） |

## 事前登録規則（不変）

| ファイル | sha256 | `env_info.txt` |
|---|---|---|
| `RULE.txt` | `3f90896aab94a2052a8757edad0e11e5e12027fec3648a5dd6e890a42d706059` | `rule_sha256` と一致 |
| `RULE.v2.txt` | `7349ec07ee4cd99ee1bff2250a9d6fbb6cbc1eb5bc1fe564f0ce528ab1ea7553` | `rule_v2_sha256` と一致 |
| `RULE.v3.txt` | `2bcc33665e367cc0d9ed3eabe441c4cda08e6993ac5535b22d3e0b71276ac765` | `rule_v3_sha256` と一致 |

規則・閾値（0.5〜2.0）・tolerance・parity baseline・判定器の判定式は本 PR で変更していない。

## v0.12.0 との差分（`bench_headpatch` の解釈用）

ノードには `.git` が無く `env_info.txt` の `diff_stat_v0.12.0_head` は「未実測」と出る。同定は Mac で取った。

`git diff --stat v0.12.0 <FIX> -- crates/tensor-core crates/autodiff crates/backend-cpu crates/facade/src`:
`crates/backend-cpu/examples/cpu_pack_kernel_breakdown.rs`（+123）・`cpu_sme_thin_m64_reopen.rs`（+108）の 2 ファイルのみ。
**`src` の差分はない。** よって `bench_headpatch`（path patch）と `bench_base`（registry `=0.12.0`）の差はソースの差ではなく、
ビルド構成（path 依存か registry 依存か・ビルドディレクトリ）の差である。

## 実行物

| 実行物 | sha256 |
|---|---|
| `session-scripts/session-3671-excl.sh` | `d1aa7512711de32809534f13ac1122fef8e4c594b83024f78ede3afa58589bc9` |
| `orchestrate.sh` | `31b032266617220b080c1d0c0bb563491982b2f08c3685ca9d74fd686095a9d9` |
| `aggregate.py` | `fe54b8764d73f824a520c75b07c98376c270383a88a48331b94686260fe6019b` |
| `bench_judge.py` | `40d11c289c83a1ded7141342638ac706358afc78e8e55b21bdaa26129d9433d3` |
| `build_bench_arms.sh` | `29bfce3dbca7855c0812510e3ec5e16b85f796de1b4d412c175e87cdfca5a7b6` |
| `../cpu-gemm-resident-reuse-3646/excl_check.py` | `ab9dcae3bc53fdf29ab31b7be84cd56764dff70791df0285e8d5e26f488d6ab9` |
| `bench-fandhe-base`（registry `fandhe-ai =0.12.0`） | `2db037259df250eb73c5b652b074026c7e192035340f62fd5a3fa3bfc56ee636` |
| `bench-fandhe-head`（`crates/facade` の path patch） | `51f8b0db4002a1aaeb18d7fd7cd209cbb91dd0b75238ceffaa2676f9f1956570` |

toolchain: rustc 1.97.1（8bab26f4f 2026-07-14）・cargo 1.97.1・host aarch64-unknown-linux-gnu（`session-scripts/bench_arms.txt`）。

判定器の `--self-test`（`aggregate.py`・`bench_judge.py`・`excl_check.py`）は計測後に Mac で実行し、いずれも `self-test: ok`。

## セッションの経過（`session-scripts/steps.log`・UTC）

| 時刻 | イベント |
|---|---|
| 11:14:36 | `hwlock acquired`・`start tree=302f6ae9…` |
| 11:36:08 | `orchestrate-diag rc=0`（診断の正式系列 5 run） |
| 11:38:57 | `orchestrate-solo rc=0` |
| 11:52:28 / 12:05:23 / 12:18:54 | `bench base`／`tunables`／`headpatch done` |
| 12:22:01 | `orchestrate-minflt rc=0`・`session body done`・`done.`（`rc=` なし。完走）・`hwlock released` |

ロックは 11:14:36 に待ちなしで取得（タイムアウトなし）。負荷ゲートは diag・solo の全 run と bench の 75 ケースすべてが `gate=pass`（`unpassed` 0 件）。
bench の `rc.log` は 3 腕とも 75 行・全行 `rc=0`。

## スモーク（計数しない・ロック取得前）

別の一時ディレクトリで実行し、出力は削除した（系列に数えていない）。

- 診断テスト `cpu_gemm_reuse_window_phases` は N=256 から `DIAG_JSON` 行を出す（全体は約 97 秒。出力は N ごとにまとまって出るため 60 秒の打ち切りでは 0 行になった。短い timeout では判定できない）
- `cpu_gemm_reuse_window_solo`（`GEMM_WINDOW_ONLY_ARM=resident_public`・`GEMM_WINDOW_ONLY_N=256`）は `DIAG_JSON` を 4 行出力
- `bench-fandhe-base`・`bench-fandhe-head` の `--task gemm --device cpu --size 256` の `--mode fresh`／`--mode reuse`／`--mode reuse --phases` はすべて rc=0・`parity_fail_count=0`
- スモーク中（120 秒・1 秒間隔）の `nvidia-smi --query-compute-apps` は 0 行

## 専有

`session-scripts/excl_status.txt`（`excl_check.py --series diag,solo,base,tunables,headpatch,minflt`・exit 0）は 6 系列すべて `excl=pass`。
`gpu_poll.log` は 135 件すべて `apps=0`。`session.log` は空。

## 未検証・限界

- ノードに `.git` が無く、`diff_stat_v0.12.0_head` は `env_info.txt` 上「未実測」（上記は Mac での取得値）
- 本記録の系列は `<cuda-node>` 1 台・1 回の取得。別機体や M4 Max の値で代替しない
