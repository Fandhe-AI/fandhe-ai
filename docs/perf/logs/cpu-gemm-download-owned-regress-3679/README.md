# #3679: download_owned 後に 5/5 run で後退した 3 セルの再現確認

**状態: 未実測**（計測部品と事前登録規則のみ。実測値は無く、他機体・他機体種の値で代替しない）。
背景・事実・限界は `docs/perf/cpu-gemm-reuse-fresh-inversion.md` §12。

## 構成

| ファイル | 役割 |
|---|---|
| `RULE.txt` | 事前登録規則（単独コミットで最初に固定。以後編集しない。固定時刻はそのコミットのコミット時刻） |
| `orchestrate.sh` | 1 機体・1 系列。診断だけを 5 run・ABBA で回す。`--first` で系列の先頭腕を指定 |
| `session-scripts/session-3679.sh` | hw-lock・GB10 の poller と前後スナップショット・orchestrate・`excl_check.py`・参考判定 |
| `repro_judge.py` | RULE を機械的に適用する判定器（標準ライブラリのみ・fail-closed・`--self-test` 付き） |

## 腕

- before = `a7aac2f0227add243845418e0897ac0675c5b622`・after = `f7ce20fc91730216e8f3f60514b181f32ebfee48`
- 木と事前ビルドは `../cpu-gemm-download-owned-3672/build_ab_arms.sh <before木> <after木> <bin>` を流用する（編集しない。bench のバイナリも作られるが本 Issue では使わない）

## 実行手順（実ホスト名は git 管理外の `.local.md` から引き、公開物には書かない）

1. `git archive` を 3 本作る（runner＝部品のコミット・before・after）。各木に `.rev-stamp`（1 行目 sha・2 行目 `dirty=0`）を置き、ツリー指紋を記録する
2. 各機体の `$HOME/work/r3679/{runner,before,after}` へ rsync・展開。DGX では openrsync の `--delete-excluded` で framework-compare の target が消えるため、転送後に先にビルドが通ることを確認する
3. hw-lock を取る前に `build_ab_arms.sh` を 1 回実行して `$HOME/work/r3679/bin` を作る（計測中はビルドしない）
4. 系列を 30 分以上あけて切り離し起動する（GB10 は `setsid nohup`、Mac mini は `nohup` のサブシェル）。完了待ちの grep は `^done\. rc=` に行頭アンカーを付ける
   - `session-3679.sh <gb10|macmini> s1 before`・`s2 after`・`s3 before`・（ゲート不通過があれば）`s4 after`・`s5 before`
5. マスクして回収（実ホスト名・HOME・ユーザー名の grep が 0 件）、`gb10/sN`・`macmini/sN` に置く。nm の生出力のような大きなものはコミットしない
6. `python3 -I repro_judge.py --host gb10=<s1>,<s2>,<s3> --host macmini=<s1>,<s2>,<s3> --out judge-3679.md`（系列は起動順。手で書き換えない）

## 検証済み（実機なし）

- `python3 -I repro_judge.py --self-test` が PASS（再現した／再現しない／判定不能／系列不足／checksum 不一致／k=12 でも系列中央値 ≤ 1.00／欠損・重複・RULE の sha 不一致の fail-closed）
- `bash -n`・引数の拒否ケース（相対パス・許可外の文字・`--first` 不正値）

## 実行の試み（2026-10-29・自動運転セッション）

- **状態: 未実測のまま**。本セッションでは `session-3679.sh` を 1 本も起動していない。RULE.txt・判定器・腕は変更していない。
- GB10（`<cuda-node>`）: 候補 3 ノードすべてで `nvidia-smi --query-compute-apps` が 1 件以上の常駐 compute プロセスを返した。常駐のあるノードでは `excl_check` が構造的に fail し record_only になるため使わず、他者の常駐サービスは止めない。専有できるノードが空くまで T1 は未実測。
- Mac mini（`<metal-node>`）: 到達でき hw-lock も空いていたが、1 機体 3 系列で最短 3〜4 時間かかり、前景ポーリング制約のあるこのセッションでは完走できないため起動しなかった。T2・T3 も未実測。
- 次の実行者は RULE.txt の sha256（`75c4f3c5…`）を確認し、`session-3679.sh <host> s1 before` から起動する。
