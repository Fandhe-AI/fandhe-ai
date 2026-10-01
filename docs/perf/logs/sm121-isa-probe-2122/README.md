# sm_121 ISA プローブの実測ログ（イシュー #2122）

親 #2121 Phase 4。sm_121（DGX Spark GB10）で使える命令とアーキ固有機能（tcgen05・mma 形状・
wgmma・setmaxnreg・cluster・DSMEM・デバイス属性・Hopper との差分）のプローブ。設計・結果表は
[`docs/cuda-sm121-isa-probe.md`](../../../cuda-sm121-isa-probe.md)、判定規則は本ディレクトリの
[`RULE.txt`](./RULE.txt)（実測前に固定。機械可読行が唯一の契約）を正とする。

## 位置づけ

- 本ディレクトリは **GB10 実測結果（2026-10-01・正式実行 1 系列）を含む**（PR-C）。PR-A・PR-B の時点では実測結果を含まなかった。
  開発機（RTX 3060・sm_86）のスモーク結果・ログはコミットしない（G0 が不成立になり全セルが判定不能になる設計）。
- TMA（`cp.async.bulk.tensor`）の意味論プローブ（`tma.*`。R-TMA-BASE／R-TMA-SEM／R-TMA-XFER）と既存 `tma_probe_real_device` の再実行（legacy 3 件）は PR-B で追加した。GB10 で実測済み（下の「結果欄」）。

## 構成

| ファイル | 役割 |
|---|---|
| `RULE.txt` | 事前登録判定規則。`CLAUSE:`／`STAGE:`／`STATUS:`／`VERDICT:`／`INDETERMINATE:`／`PROBE:`／`TARGET:`／`PROCESS:` 行が機械可読の契約（レジストリ・aggregate.py・orchestrate.sh が参照） |
| `orchestrate.sh` | GB10 での一括実行。起動列は RULE.txt から導出。`--dry-run`（起動一覧のみ）・`--dev-smoke`（開発機。出力先はリポジトリ外の `LOG_DIR` 必須） |
| `aggregate.py` | 集計（python3 標準ライブラリのみ）。`--self-test`（全条項の陽性・判定不能 fixture・RULE.txt との照合・引用照合・起動一覧照合）。`--out` は検証後に atomic に書く |
| `guide_claims.tsv` | R-GUIDE の引用元（`path:line`）・原文引用・測定項目・比較方法 |
| `env_info.txt` `compile.log` `exec/<probe>@<target>.log` `legacy-<name>.log` `device_attributes_dump.log` `aggregate.md` | 実測時に生成される生ログ・集計（2026-10-01 の GB10 正式実行 1 系列で生成・収録済み） |

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

GB10 実測済み（2026-10-01）。判定・観測の転記と注記は [`docs/cuda-sm121-isa-probe.md`](../../../cuda-sm121-isa-probe.md) §5・§5.1、集計の正は本ディレクトリの [`aggregate.md`](./aggregate.md)。

- 実測環境: `<cuda-node>`（NVIDIA GB10・cc 12.1）・ドライバ 580.173.02・NVRTC 13.0・Linux 6.17.0-1031-nvidia aarch64。`env_info.txt`: `mode=gb10`・`git_head=7b6019aeb7856eb9cb2e28929a735b9938bc1503`・`git_clean=1`・`git_source=rev-stamp`・`start_utc=2026-10-01T06:33:39Z`・`end_utc=2026-10-01T06:35:34Z`。常駐サービス 2 プロセスが CUDA コンテキストを保持したまま（`utilization.gpu` 0%）、ユーザー承認のうえ実行した。
- 系列: 起動プロセス 219 件（device_attributes_dump 1・compile 1・exec 210・legacy 7）はすべて exit 0（timeout・process_failed 0 件）。`aggregate.py` は exit 0（G0 成立）で、転送元での再集計結果はノード側の `aggregate.md` とバイト一致。捨てた系列はない。
- 判定の内訳: 判定不能 0 件・想定外の受理（UNEXPECTED_ACCEPT）0 件・LEGACY_CONTRADICTION／LEGACY_INCONCLUSIVE 0 件。home の S2 は 67 件すべて ok、対照 `ctl.copy`・`ctl.raw`・`ctl.rawmap` は 3 target すべて成立。
- 要点（判定語のまま）: tcgen05／TMEM（`tc5.alloc`・`tc5.ld`）と `wgmma.m64n8k16` は 3 target とも「ptxas 拒否（オフライン）」、`tc5.cross` は「ロード失敗」。cluster 2／4／8・DSMEM・`clu.rt2`／`rt4` は「成立」、`clu.dims16` は「実行時エラー」（`CUDA_ERROR_INVALID_CLUSTER_SIZE`）。`snr.*` と `mma.f8f6f4`／`mma.block_scale` は `compute_121` だけが「ptxas 拒否（オフライン）」。TMA の R-TMA-BASE・R-TMA-XFER は「成立」、R-TMA-SEM は「受理のみ（実行意味論は未検証）」（観測は要素座標＝内側次元が先・OOB はゼロ／NaN 埋め・swizzle は標準の XOR モデルと一致）。R-GUIDE の不一致は C02・C03・C06・C09。
