# #3681: download_owned の A/B を、非処置セルを matmul_resident の経路に絞った規則で再事前登録して再計測する

**状態: 未実測**（計測部品と事前登録規則のみ。実測値は無く、他機体・他機体種の値で代替しない）。
背景・事実・限界は `docs/perf/cpu-gemm-reuse-fresh-inversion.md` §14。

#3672 の REJECT（`../cpu-gemm-download-owned-3672/judge-3672.md`）を見たあとの再登録である。#3672 の
`RULE.txt`・`RULE.v2.txt`・判定は変えず、#3672 の生データの再判定にも使わない（新しいセッションで別に測る）。

## 構成

| ファイル | 役割 |
|---|---|
| `RULE.txt` | 事前登録規則（単独コミットで最初に固定。以後編集しない。固定時刻はそのコミットのコミット時刻） |
| `orchestrate.sh` | 1 機体・1 系列。`../cpu-gemm-download-owned-3672/orchestrate.sh` の写しで、`rule_sha256` を本ディレクトリの `RULE.txt` から記録する（`RULE.v2.txt` は無いので `rule_v2_sha256` 行を除いた）。before／after の ABBA・5 run・負荷ゲート・マスクは同じ |
| `session-scripts/session-3681.sh` | hw-lock・GB10 の poller と前後スナップショット・orchestrate・`excl_check.py`・参考判定。`#3679`・`#3680` と同じ `$HOME/work/.fandhe-hw-lock` で排他 |
| `path_judge.py` | RULE を機械的に適用する判定器（標準ライブラリのみ・fail-closed・`--self-test` 付き） |

## #3672 の判定との差（1 点だけ）

REJECT の「5/5 run 後退」と ADOPT の「非処置セル中央値 ratio ≤ 1.00」を、`matmul_resident` を計測窓の中で呼ぶ
**経路上のセル（処置 2 + 非処置 28）**にだけ適用する。経路外の診断腕（`fresh`・`resident_split`・`backend_drop`・
`gemm_device_*`）は記録のみ。checksum は全腕・全 N のまま。閾値（処置 ratio ≤ 0.95・5/5 run < 1.00・rho_REM ≥ 0.5）は
#3672 と同じ。経路の決め方（file:line）は `RULE.txt` を正とする。

`path_judge.py` は `../cpu-gemm-download-owned-3672/ab_judge.py` のローダー・セル統計・rho 計算を流用する
（編集しない。sha256 を固定して照合し、不一致なら終了コード 2）。`RULE.txt` の `treat-cell`／`path-cell` 行と
判定器の定数の集合等価、`env_info.txt` の `rule_sha256` と `RULE.txt` の一致も検査する。

## 腕

- before = `a7aac2f0227add243845418e0897ac0675c5b622`・after = `f7ce20fc91730216e8f3f60514b181f32ebfee48`
- 木と事前ビルドは `../cpu-gemm-download-owned-3672/build_ab_arms.sh <before木> <after木> <bin>` を流用する（編集しない）

## 実行手順（実ホスト名は git 管理外の `.local.md` から引き、公開物には書かない）

1. アーカイブ（`git archive`）を 3 本作る（runner＝部品のコミット・before・after）。各木に `.rev-stamp`（1 行目 sha・2 行目 `dirty=0`）を置き、ツリー指紋を記録する
2. 各機体の `$HOME/work/r3681/{runner,before,after}` へ rsync・展開。DGX では openrsync の `--delete-excluded` で framework-compare の target が消えるため、転送後に先にビルドが通ることを確認する
3. hw-lock を取る前に `build_ab_arms.sh` を 1 回実行して `$HOME/work/r3681/bin` を作る（計測中はビルドしない）
4. `session-3681.sh <gb10|macmini> s1` を切り離し起動する（GB10 は `setsid nohup`、Mac mini は `nohup` のサブシェル。Mac は bash 3.2）。完了待ちの grep は `^done\. rc=` に行頭アンカーを付ける
5. マスクして回収（実ホスト名・HOME・ユーザー名の grep が 0 件）、`gb10/`・`macmini/` に置く。大きな生出力はコミットしない
6. `python3 -I path_judge.py --host gb10=<dir> --host macmini=<dir> --out judge-3681.md`（手で書き換えない）
7. `rev-stamp-verification-3681.md` に腕の同定、RULE のコミット時刻（`git log -1 --format=%cI -- RULE.txt`）と各機体の計測開始時刻（`env_info.txt` の `date_utc`・`steps.log` 先頭）、専有の合否を並べる

## 検証済み（実機なし）

- `python3 -I path_judge.py --self-test` が PASS（経路外の 5/5 後退でも ADOPT／経路上の 5/5 後退は REJECT／経路上の中央値 > 1.00 は判定不能／処置セルの REJECT・0.95 超／経路外腕の checksum 不一致は REJECT／rho_REM < 0.5／片機体欠落・record_only／RULE sha・セル列挙の不一致・欠落の fail-closed）
- `bash -n`・`shellcheck`・引数の拒否ケース（相対パス・不正な機体名・不正な label・未知オプション）
- `path_judge.py` は #3672 の回収データに対して実行していない（新しい規則で旧データを判定した結果を作らないため）
