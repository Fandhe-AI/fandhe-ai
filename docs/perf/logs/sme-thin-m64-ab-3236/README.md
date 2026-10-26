# m=64 専用 SME 薄型カーネルの既定を Mac mini の A/B で決める（イシュー #3236）

## 状態

**実測済み（判定: undetermined。2026-10-24・Mac mini）**。機械出力は `aggregate.md`、解釈と処置は
`docs/perf/cpu-gemm-sme-thin-m64-decision.md` §11.1。計測は #3580 後の main（コミット `3f122311f`。`macmini/rev-stamp.txt`）で、
RULE-amend-1（P-TREE の `SME_PRODUCTION_ENABLED` を両腕 false に訂正。計測前に追記）に従った。

## 目的

#3235 で入れた m=64 専用 4 タイル SME カーネルは既定 OFF（`SME_THIN_M64_ENABLED = false`）で、性能は未測定。
Mac mini（Apple M4）で OFF（main）と ON（定数 1 行を反転した計測専用ツリー）を 5 round 測り、事前登録規則で既定と k 下限を決める。
反転 PR は ADOPT／ADOPT-k のときだけ、独立監査と所有者確認の後に別 PR で作る。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（計測前にコミットして固定。訂正は末尾への追記のみ） |
| `on-arm.patch` | after 腕。`mod.rs` の `SME_THIN_M64_ENABLED` を true にする 1 行（sha256 は `RULE.txt` に固定） |
| `orchestrate_macmini.sh` | ロック → P-TREE → P-R0 → P-BIT → ビルド → GEMM／K 格子 5 round → framework-compare A/B → 記録 run → マスク収録（`--dry-run` あり） |
| `aggregate.py` | 厳密集計（標準ライブラリのみ。`python3 -I aggregate.py --self-test`／`python3 -I aggregate.py macmini`） |
| `aggregate.md`・`macmini/` | 実測の機械出力と収録ログ（`aggregate.md` は手で直さない） |
| `../../../../scripts/bench/framework-compare/run_ab_sme_thin_m64_macmini_cpu.sh` | framework-compare の A/B 本体（#3232 の `run_ab_sme_default_on_cpu.sh` の派生。元は変更しない） |
| `../../../../crates/backend-cpu/examples/cpu_sme_thin_m64_kgrid.rs` | K 格子の計時 binary |

## 手順（概要。詳細は各スクリプト冒頭）

1. 本ディレクトリを含むコミットの `git bundle` を作り、Mac mini へ rsync して clone する（main ツリー。クリーンに保つ）。接続情報はリポジトリに書かない。
2. `RAYON_NUM_THREADS` を unset し、他の cargo・bench が動いていないことを確かめる。
3. macOS に `setsid` は無い。`nohup` のサブシェルから `zsh -lc` で
   `ON_MAIN=<ツリー> TORCH_PY=<venv python> WORK=<作業> bash docs/perf/logs/sme-thin-m64-ab-3236/orchestrate_macmini.sh` を起動し、`pgrep` で生存確認する。
   完了は `done.` と `load_gate.log` の `run_ab exit=`／`collect exit=` で確認する（`rc=` を grep しない）。
4. 回収後 `python3 -I aggregate.py macmini > aggregate.md` で集計し、決定記録 §11 を更新する。

## マスク規則・実機ロック

- ホスト名は `masked`、`$HOME` は `<home>`、ユーザー名と絶対パスは除去する。
- 実機ロック `~/work/.fandhe-hw-lock` を取得してから測り、取得・解放時刻を `hwlock.log` に残す。ロックを持たずに測った値は記録に使わない。
- Mac mini の値と M4 Max（開発機）の値を混ぜない。

## スコープ外

- 既定を反転する PR（ADOPT のとき。独立した security-auditor の監査と所有者の確認が前提）。
- GB10 の計測（SME 非搭載で経路が変わらない）。
- epilogue の asm 融合・A pack の並列化・`ldc=n` への直接ストア・n%16≠0 のマスク付きストア・スレッド方針の本番化（決定記録 §8・§10）。
- #3232（SME 既定 ON の A/B）の実測。

## 実績（2026-10-24）

- 転送段のロック: 01:08:28Z 取得・01:08:35Z 解放（bundle を clone し `rev-parse HEAD` とクリーン状態を確認）。計測段のロック: スクリプトが 01:08:41Z に取得し、終了後に不在を確認。
- 転送は bundle → clone 方式（P-TREE が `.git` を使うため rsync のみは不可）。収録ログのマスク漏れ（ホスト名・ユーザー名・絶対パス）は grep で 0 件を確認した。
- 計測後、Mac mini 側の作業ツリー・target・bundle は削除済み。

## l2_dx 再計測系列（イシュー #3615・RULE-reopen-1）

#3236 で undetermined だった `l2_dx` を、判定の閾値を変えずに反復数（warmup 200／iters 2000）と round 数（11）を増やして測り直す系列。
規則は `RULE.txt` 末尾の RULE-reopen-1（計測前にコミット済み）。既存の `macmini/`・`aggregate.md`・`aggregate.py`・`orchestrate_macmini.sh` は差し替えない。

| パス | 内容 |
|---|---|
| `orchestrate_macmini_reopen.sh` | `orchestrate_macmini.sh` の派生（ラベル `3236-reopen-r1`・11 round・framework-compare と記録 run は無し。`--dry-run` あり） |
| `aggregate_reopen.py` | 厳密集計（`python3 -I aggregate_reopen.py --self-test`／`python3 -I aggregate_reopen.py macmini-reopen`） |
| `../../../../crates/backend-cpu/examples/cpu_sme_thin_m64_reopen.rs` | 計時 binary（`<before\|after> <gemm\|kgrid>`。出力に `warmup=`／`iters=` を含む） |
| `macmini-reopen/`・`aggregate-reopen.md` | 実測後に追加する収録ログと機械出力 |

状態: **未実測**（計測部品と規則のみ。Mac mini での実測はこのコミット時点で未実施）。既定 `SME_THIN_M64_ENABLED` はこの issue では反転しない。
