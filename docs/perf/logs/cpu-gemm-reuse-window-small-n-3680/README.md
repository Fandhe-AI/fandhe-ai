# #3680: N=256・512 の bench reuse 窓の差を診断腕で切り分ける

状態: **未実測**（計測部品と事前登録規則のみ。GB10 の値は無い。Linux・M4 Max の値で代替しない）。
判定規則は `RULE.txt` が正（実測と部品より前に単独コミットで固定。固定後は変えない）。
経緯と読解は `docs/perf/cpu-gemm-reuse-fresh-inversion.md` §13。

## 固定の順序

1. `RULE.txt`（単独コミット）
2. 部品: bench-fandhe の `--diag-verify-after-loop`、`build_bench_head.sh`、`session-scripts/session-3680-excl.sh`、`judge_3680.py`
3. 実測（別 PR）→ 回収・マスク → `judge-3680.md` → doc §13 へ機械的に転記

## 腕（bench 側。同一バイナリ。差は 1 要因）

| 腕 | 1 要因 |
|---|---|
| base | 変更なし |
| vafter | `--diag-verify-after-loop`（reuse／phases のみ。fresh は同じ系列の中でフラグなし） |
| tunboth | `GLIBC_TUNABLES=glibc.malloc.mmap_threshold=33554432:glibc.malloc.trim_threshold=1073741824` |
| tunmmap | `GLIBC_TUNABLES=glibc.malloc.mmap_threshold=33554432` |
| tuntrim | `GLIBC_TUNABLES=glibc.malloc.trim_threshold=1073741824` |

診断側は #3671 の `orchestrate.sh gb10`（`cpu_gemm_reuse_window_phases`。変更しない）を同じ木で使う。

## 実行手順（実機に接続できるセッションで）

1. 計測木を `git archive` で展開し、ルートに `.rev-stamp`（1 行目 HEAD の sha・2 行目 `dirty=0`）を置く。HEAD は RULE.txt を含むこと
2. hw-lock の**前に**ビルドを完了する（計測中のビルド並走は禁止）:
   - `build_bench_head.sh <木の絶対パス> <新規 bindir>`（path patch。`Cargo.lock` は trap で復元される）
   - 診断テストバイナリ: 木で `cargo test --release --locked -p fandhe-ai --test cpu_gemm_reuse_window_diag --no-run`
3. `session-scripts/session-3680-excl.sh` を `setsid nohup` で切り離して起動（`$HOME/work/r3680-excl/{runner,bench-bin}` に木と bindir を置く前提。出力先が既にあれば中止する）
4. 回収時にホスト名・`/home`・ユーザー名をマスクする
5. `python3 -I docs/perf/logs/cpu-gemm-resident-reuse-3646/excl_check.py <out> --series diag,bench` で `excl_status.txt` を作る
6. `python3 -I judge_3680.py --diag <out>/gb10 --bench-root <out>/bench --excl <out>/excl_status.txt --gate-log <out>/load_gate_bench.log --env <out>/env_info_3680.txt`

所要時間の目安: #3671 の bench 3 腕 75 起動の実績からの推定であり実測ではない。本系列は bench が 5 腕 × 5 N × 3 モード × 5 run の 375 起動に診断 5 run が加わるため、半日以上を見込む。

## 検証済み／未検証

| 項目 | 状態 |
|---|---|
| bench-fandhe の fmt／clippy／unit test（OFF と ON の bit 一致・受理条件・checksum 不一致の拒否） | 検証済み（M4 Max。値は記録しない） |
| `judge_3680.py --self-test`（(a)(b)(c)・G_diag 負・複数支持・支持なし・データ不備・参考記録化） | 検証済み |
| `shellcheck`／`bash -n`（2 本） | 検証済み |
| `build_bench_head.sh` の実ビルド・path patch の解決確認 | 未検証（`.rev-stamp` 付きの木で実行が要る） |
| `session-3680-excl.sh` の実機実行 | 未検証 |
| GB10 の実測・判定 | 未実測 |
