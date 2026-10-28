# #3671 N=256〜2048 の bench reuse 窓と診断の差の切り分け

## 状態

**計測部品のみ（実測前）。** 事前登録規則（`RULE.txt`・`RULE.v2.txt`）・診断テスト・オーケストレータ・集計器・判定器・
セッションスクリプトを置いた。**GB10 の専有系列は未実測**で、原因は未特定（「特定できた」とも「特定できず」とも書かない）。
このため PR は `Refs #3671`。`Closes` は専有を満たした正式系列で完了条件を満たしたときに限る。

0.12.0 の正式値と #3274 の tally は変えていない。tolerance・parity baseline・bench の窓・本番コードも変えていない。

## 固定の順序

1. `RULE.txt`（単独コミット）→ 2. `RULE.v2.txt`（基準の再確認の式だけ置換。別コミット）→ 3. 部品一式（診断テスト・スクリプト・本ファイル）
RULE の sha256 と `RULE.v2.txt`・`RULE.v3.txt`（H-A2 の「単一スレッド」誤記の訂正。記述のみ）の sha256 は計測時に `env_info.txt` と `rev-stamp-verification-3671.md` へ記録する（未作成。計測時に作る）。

## 腕

| 系列 | 腕 | 1 要因 | 仮説 |
|---|---|---|---|
| 診断（`diag`） | `fresh`・`resident_public` | 基準 | — |
| | `rp_verify_touch` | 窓の外で N² の参照と出力を全要素比較 | H-V（キャッシュ・ページ） |
| | `rp_verify_gap` | 同じ比較回数を L1 常駐の 4 KiB で行う | H-V の分解（空き時間のみ） |
| | `rp_hold_footprint` | N² Vec を 5 本保持するだけ | H-A1 |
| | `rp_ref_compute` | 並列（scoped worker）の O(N³) 参照 GEMM を前処理で 1 回 | H-A2 |
| 診断（`solo`） | `rp_solo` | 1 プロセス 1 腕 1 N・1 ラウンド（N ごとに別プロセス） | H-P |
| bench | `bench_base` | 基準（registry `fandhe-ai =0.12.0`） | — |
| | `bench_tunables` | `GLIBC_TUNABLES`（#3646 と同じリテラル） | H-A3 |
| | `bench_headpatch` | `crates/facade` の path patch | H-B |

`Tape::reset` の腕は作らない（bench の窓も診断も呼ばない。読解の根拠は `RULE.txt`。CI テスト
`matmul_resident_does_not_grow_tape_leaves` が葉数不変だけを補強する。ノード数は公開 API で数えられない）。

## 実行手順（GB10。`docs/real-hardware-verification-env.md` に従う）

1. 固定コミットの tracked ファイルだけを tar で転送する（openrsync の `--delete-excluded` は使わない）。計測ツリーのルートに `.rev-stamp`
   （1 行目 HEAD・2 行目 `dirty=<件数>`）を置き、ツリー指紋を `git archive` と突き合わせる。
2. hw-lock の前にビルドを済ませる:
   `cargo test --release --locked -p fandhe-ai --test cpu_gemm_reuse_window_diag --no-run` と
   `bash docs/perf/logs/cpu-gemm-reuse-window-3671/build_bench_arms.sh <bench-bin>`（`framework-compare/Cargo.lock` は終了時に復元される。
   `make deps-forbidden` で承認ピンのままであることを確認する）。
3. `setsid nohup bash session-scripts/session-3671-excl.sh` で切り離して起動する。順に `diag`（5 run）→ `solo`（5 run）→
   `base`・`tunables`・`headpatch`（N×3 モード×5 run）→ `minflt`（1 run・判定外）。作業ディレクトリは `$HOME/work/r3671-excl`（既存なら中止）。
4. 専有の判定: 回収した `session-scripts/` で
   `python3 -I docs/perf/logs/cpu-gemm-resident-reuse-3646/excl_check.py <dir> --series diag,solo,base,tunables,headpatch,minflt`。
5. 回収時にホスト名は `<cuda-node>`・`/home` は `<home>` にマスクする（数値は変えない）。
6. 集計と判定:
   `python3 -I aggregate.py <gb10>`・`python3 -I aggregate.py <gb10-solo>`・
   `python3 -I bench_judge.py --diag <gb10> --solo <gb10-solo> --bench-root <bench> --excl <excl_status.txt> --gate-log <load_gate_bench.log>`

M4 Max は疎通確認のみ（`orchestrate.sh m4max`。`record_only`）。Linux x86・M4 Max の値で GB10 の判定を代替しない。

## 検証した項目と未検証の項目

| 項目 | 状態 |
|---|---|
| 診断テストの bit 一致（N=64・128・全腕）・葉数不変 | CI で実行可能。ローカル（M4 Max）で pass |
| `aggregate.py --self-test`・`bench_judge.py --self-test`・`excl_check.py --self-test` | pass |
| `excl_check.py` の既定 series の挙動 | 不変。`gb10-excl/session-scripts/` に再実行して `excl_status.txt` がバイト単位で同じ |
| `shellcheck -x`（3 本の sh） | 警告なし |
| `orchestrate.sh`・`session-3671-excl.sh`・`build_bench_arms.sh` の実走 | **未検証**（GB10 実機が必要。実装環境は macOS） |
| GB10 の実測（全系列） | **未実測** |
