# 0.13.0 の 6 フレームワーク同席計測 — Mac mini（Apple M4）

**状態: 実測済み（2026-10-31。系列 B 5/5 完走）。** 目的・判定規則・受け入れ条件・結果の表・1.0 以下のセルの理由・事前確認の結果は `../framework-compare-0.13.0-remeasure/README.md` を正とする（二重管理しない）。本ディレクトリは Mac mini の事前登録規則・スクリプト・集計器と、計測の生ログ・中央値を収める。GB10 の値とは混ぜない。事前登録規則 `RULE.txt` は 2026-10-31T04:56:14Z に固定し、以後編集していない。

## 計測の要点

- 機体: Mac mini（`<metal-node>`・Apple M4・10 コア・16 GB）。`macmini-run-0130.sh` が run ごとに `machdep.cpu.brand_string == "Apple M4"` を検査し、不一致なら run を止める（5/5 完走。M4 Max の値は使っていない）。
- 腕: 単腕 PIN_SHA `5d05a50235138645b0b069060ed54dfd2c46db7f`（main の #3709 マージ）。全 5 run の `tree.txt` が `fandhe-ai v0.13.0`、`rev-stamp.txt` が RULE の SHA と一致する（`../framework-compare-0.13.0-remeasure/aggregate/check_identity.py` が rc=0。行数 Rust 48・Python 32）。
- Python 3 FW: JSONL の `version` 欄は 0.12.0 と一致（pytorch 2.14.0・tensorflow 2.16.2・scipy 1.18.1）。計測後（2026-10-31T06 時台 UTC）に同じ venv の `pip freeze`（GB10 は `pip freeze`＋`python --version`、Mac mini は `uv pip freeze`）を取り `env/` に収録した。0.12.0 の `env/` と byte 一致（numpy・keras を含む全パッケージが同じ版）。
- 系列: **系列 B のみ**（各 run 前に load1 < 3.0 を最大 30 分待つゲート付き・5 run）。**系列 A（対照）は行わない**ため、ノイズ帯は「対照系列なし・未算出」（0 件ではない）。
- 日時・負荷: 2026-10-31 04:58:54Z〜05:08:30Z（run1 開始〜run5 終了。`loop-macmini.log`）。開始時 load1 = 1.39／2.57／2.35／2.00／1.77（`gate.log` 8 行・各 run の `uptime_before.txt`）。run2・run4・run5 の前は直前 run の終了直後の load1 が 4.21・3.15・3.10 で 1 分ずつ待機し、run1・run3 は待機なし。`GATE-INVALID`・`GATE-TIMEOUT` なし。
- 完走・skipped: `loop-macmini.log` が `ALL-DONE completed=5/5`、全 5 run の最終行が `done. rows=48 pyrows=32 skipped=4`。skipped は burn Metal GEMM N=512〜4096 fresh の既知の記録拒否 4 件（`MEASURE_ERROR: gemm checksum is degenerate (0)`）だけで、5 run の `skipped-macmini-0.13.0.log` は同一内容・0.12.0 の同ログとも同一。
- 実機ロック: `~/work/.fandhe-hw-lock`（owner `rel-0.13.0 2026-10-31T04:49:10Z`）。05:35:42Z に解除（GB10 の計測と同じロック）。
- 実機テスト（#3256 の M1〜M6・#3272 の X1）: 未実施（範囲外。RULE.txt）。

## 結果

表は `../framework-compare-0.13.0-remeasure/README.md` に載せる。Mac mini の 13 セルの tally は 1 位 3（gemm Metal N=256・512、infer Metal）・僅差 3（gemm Metal N=1024・2048、train Metal）・負け 7（gemm Metal N=4096、gemm CPU N=256〜2048、train CPU、infer CPU）。GB10 を合わせた 27 セルの tally は `13 3 11 invalid 1 noise na total 27`。

0.12.0 からの区分変化は Mac mini では train Metal の 1 位（1.01）→ 僅差（0.95）の 1 セルだけ（fandhe-ai は 0.12.0 比 1.00 で、PyTorch MPS の窓が 0.5180→0.4902 ms に縮んだ）。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.txt` | Mac mini の事前登録規則（固定 2026-10-31T04:56:14Z） |
| `scripts/macmini-run-0130.sh`・`scripts/macmini-0130b-loop.sh` | 1 round・系列 B（`macmini-run-0120.sh`／`macmini-0120b-loop.sh` の単腕版。本計測で使った版） |
| `macmini-series-b/run{1..5}/` | 各 run の `results-macmini-0.13.0.jsonl`（48 行）・`results-macmini-py-0.13.0.jsonl`（32 行）・`tree.txt`・`rev-stamp.txt`・`run.log`・`uptime_*.txt`・`skipped-macmini-0.13.0.log` |
| `macmini-series-b/results-macmini-0.13.0-median5.jsonl`・`results-macmini-py-0.13.0-median5.jsonl` | 5 run 中央値 |
| `macmini-series-b/gate.log`・`macmini-series-b/build/` | ゲートのログ・事前ビルド（`BUILD_ONLY=1`）の検査・ビルドのログ（結果 JSONL は作らない） |
| `loop-macmini.log` | 系列 B のループのログ |
| `smoke/` | RULE 固定前の行数スモーク（判定に使わない） |
| `aggregate/median_rounds.py` | 0.12.0 版のコピー（変更なし）。同一性検査は GB10 側の `check_identity.py` を使う |

作業ディレクトリ名 `macmini-0130b/` は収納時に `macmini-series-b/` へ改名した（0.12.0 と同じ。`run_0130.sh` の既定がこの名前を読む）。`env/` は計測後に収録した（0.12.0 と byte 一致）。

## 再現手順

```bash
# 0. 展開ツリー（単腕）。.rev-stamp は 40 桁 1 行
PIN=5d05a50235138645b0b069060ed54dfd2c46db7f
git archive $PIN | tar -x -C "${HOME}/work/r0130/tree" && echo $PIN > "${HOME}/work/r0130/tree/.rev-stamp"
# 1. 作業ディレクトリ D に macmini-run-0130.sh・macmini-0130b-loop.sh・py/bench_py.py・venv-torch・venv-tf を置き、実機ロックを取ってから
#    （macOS は bash 3.2・setsid なし。ROOT・D・EXPECT_SHA は export が必要）
export ROOT="${HOME}/work/r0130/tree" D="${HOME}/work/r0130" EXPECT_SHA=$PIN
(nohup bash "${D}/macmini-0130b-loop.sh" > "${D}/loop-macmini.log" 2>&1 &)
#    先にビルド専用の呼び出し（BUILD_ONLY=1）が走り、その後 run1〜run5 を SKIP_BUILD=1 で回す。生存確認は loop-macmini.log
# 2. 集計（GB10 側のディレクトリで）
M=../framework-compare-0.13.0-macmini-remeasure/macmini-series-b
python3 aggregate/median_rounds.py $M/results-macmini-0.13.0-median5.jsonl $M/run{1..5}/results-macmini-0.13.0.jsonl   # py も同様
python3 aggregate/check_identity.py $PIN results-macmini-0.13.0.jsonl=48,results-macmini-py-0.13.0.jsonl=32 $M/run{1..5}
```
