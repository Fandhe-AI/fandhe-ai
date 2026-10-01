# sm_121 ISA プローブの実測ログ（イシュー #2122）

親 #2121 Phase 4。sm_121（DGX Spark GB10）で使える命令とアーキ固有機能（tcgen05・mma 形状・
wgmma・setmaxnreg・cluster・DSMEM・デバイス属性・Hopper との差分）のプローブ。設計・結果表は
[`docs/cuda-sm121-isa-probe.md`](../../../cuda-sm121-isa-probe.md)、判定規則は本ディレクトリの
[`RULE.txt`](./RULE.txt)（実測前に固定。機械可読行が唯一の契約）を正とする。

## 位置づけ

- 本ディレクトリには**実測結果を含まない**（GB10 実測は後続 PR で行う）。開発機（RTX 3060・sm_86）の
  スモーク結果・ログはコミットしない（G0 が不成立になり全セルが判定不能になる設計）。
- TMA（`cp.async.bulk.tensor`）の意味論プローブ（`tma.*`。R-TMA-BASE／R-TMA-SEM／R-TMA-XFER）と既存 `tma_probe_real_device` の再実行（legacy 3 件）は PR-B で追加した。結果は未実測。

## 構成

| ファイル | 役割 |
|---|---|
| `RULE.txt` | 事前登録判定規則。`CLAUSE:`／`STAGE:`／`STATUS:`／`VERDICT:`／`INDETERMINATE:`／`PROBE:`／`TARGET:`／`PROCESS:` 行が機械可読の契約（レジストリ・aggregate.py・orchestrate.sh が参照） |
| `orchestrate.sh` | GB10 での一括実行。起動列は RULE.txt から導出。`--dry-run`（起動一覧のみ）・`--dev-smoke`（開発機。出力先はリポジトリ外の `LOG_DIR` 必須） |
| `aggregate.py` | 集計（python3 標準ライブラリのみ）。`--self-test`（全条項の陽性・判定不能 fixture・RULE.txt との照合・引用照合・起動一覧照合）。`--out` は検証後に atomic に書く |
| `guide_claims.tsv` | R-GUIDE の引用元（`path:line`）・原文引用・測定項目・比較方法 |
| `env_info.txt` `compile.log` `exec/<probe>@<target>.log` `legacy-<name>.log` `device_attributes_dump.log` `aggregate.md` | 実測時に生成される生ログ・集計（未生成） |

## 実行手順（GB10 実機。別セッション）

```
./docs/perf/logs/sm121-isa-probe-2122/orchestrate.sh
python3 docs/perf/logs/sm121-isa-probe-2122/aggregate.py --out docs/perf/logs/sm121-isa-probe-2122/aggregate.md
```

- GB10 ノードには `.git` が無いため、転送元で rsync の直前に `.rev-stamp`（1 行目 HEAD・2 行目 `dirty=<件数>`）
  を作る（手順は [`docs/cuda-sm121-isa-probe.md`](../../../cuda-sm121-isa-probe.md) §6.1）。`orchestrate.sh` は
  git 作業ツリーでなければ `.rev-stamp` から provenance を読み、どちらも無ければ開始前に停止する。正式実行で clean でない
  （`dirty=0` の行が無い §3 の 1 行形式を含む。submodule ポインタの変更も dirty）場合はプロセスを 1 つも
  起動せず `exit 1` で停止する。timeout 秒数は `RULE.txt` の `PROCESS:` 行（`timeout=`）が正。
- 共有ノードのため、ノードが空いているときに実行する（意図的な不正命令を含む。プロセスは分離し
  外部 `timeout` 付き）。
- `orchestrate.sh` は出力先（既定は本ディレクトリ）に既存ログがあると開始前に `exit 1` で停止する
  （上書き・追加起動をしない）。再計測は空の別ディレクトリを `LOG_DIR=<dir>` で指定し、集計は
  `aggregate.py --log-dir <dir>` で行う。
- `aggregate.py` の終了コード: 0＝正常（または全ログ無しで未実測）／2＝完全性違反／3＝G0 不成立。
  非 0 の系列は結果として反映せず、原因を記録して系列全体をやり直す。
- 保存するファイル: 上表の生ログ・`aggregate.md`。保存前にホスト名・`/home/<name>` が masked／`<home>`
  に置換されていることを確認する（`aggregate.py` は `/home/<name>` 形のみ検出する）。

## 結果欄

未記入（GB10 実測待ち）。
