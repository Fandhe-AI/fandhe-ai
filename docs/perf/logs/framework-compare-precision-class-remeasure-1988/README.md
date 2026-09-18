# 精度クラス反映後の GB10 再計測（イシュー #1988）

## 目的

#1989 で承認・#1987（PR #2046）で実装した精度クラス（burn cuda の `tf32: true` 行のみ単位丸め
`u = 2^-11` で第 3 救済項を評価。fandhe-ai 行・candle 行・tolerance 4 定数・判定式・本体
`compare`／`assert_parity`／`ParityBaseline` は不変）を反映した bench-burn で、0.9.0 スコアボード
（`docs/perf/logs/framework-compare-0.9.0-remeasure/`）の「判定不能」6 セルを DGX Spark GB10 で
再計測し、解消／残存を確定してスコアボードを再生成する。判定規則は `RULE.txt`（計測前に固定）。

## 結果（要約）

| セル | 判定 | 根拠 |
|---|---|---|
| burn cuda gemm N=256 | **解消** | 5/5 run で `parity_fail_count = 0`・rescued 10,538（5 run 同値） |
| burn cuda gemm N=512 | **解消** | 同 0・rescued 42,361 |
| burn cuda gemm N=1024 | **解消** | 同 0・rescued 169,929 |
| burn cuda gemm N=2048 | **解消** | 同 0・rescued 681,454 |
| burn cuda gemm N=4096 | **解消** | 同 0・rescued 2,729,050 |
| PyTorch cpu gemm N=4096 | **残存** | 5/5 run で `parity_fail_count = 1`・rescued 643（#1985 と同一）。#1989 承認 T1: spec REQ-2 (b-1) 上正当な判定不能・係数 c=0.5／u=2^-24 は不変 |

- rescued の実測値は `docs/candle-parity-precision-class-decision.md` §3 の机上計算と全 5 形状で一致し、
  `median_s`・`parity_max_abs_err` は 0.9.0 記録と同水準（bound だけが変わり fail が 0 になった）。
- bound は 0.9.0 記録の **8192 倍**（`2^-11 / 2^-24 = 2^13`）。`RULE.txt` の検算欄に書いた「2048 倍」は
  誤記（`u=2^-11` の値 1/2048 と比を取り違えた）で、合否条件ではないため事後に規則は変更せず本 README で訂正する。
  実測 bound（1.562e-2／3.125e-2／6.250e-2／1.250e-1／2.500e-1）は decision doc §3 の値と一致する。
- 専有ゲート（load1 < 1.0 かつ gpu_util 0%）は 5/5 run 通過（`gb10/load_gate.log`。torch 20 スレッド直後は
  load1 が 4〜5 まで上がるため各 run 前に 2〜3 分待機）。計測失敗なし（`gb10/run*/err.log` は 0 バイト）。
- スコアボード集計: 判定対象 27 行のうち 勝ち 5・僅差 2・負け 20 は不変で、**判定不能セル 6 → 1**。
  burn が有効化した影響で GB10 CUDA GEMM N=1024／2048 は 2 位 → 3 位（burn が candle より速い）、
  N=4096 は 1 位のまま「最速他 FW ÷ fandhe-ai」が candle 比 1.36× → burn 比 1.03×（`scoreboard/gen_1988.out`）。
  この比は 2026-09-16 セッションの fandhe-ai 行と 2026-09-18 セッションの burn 行の比（セッション混在）である。
  RULE.txt の要求どおり本セッション内の同一 run 内比（`aggregate.md`）と突き合わせると、各 N で最速相手
  （candle／burn の小さい側）の中央値は 256 candle 0.8374／512 candle 0.4330／1024 candle 0.4066／2048 burn 0.4811／
  4096 burn 1.0145 で、5 形状とも「<1／>1」の側がスコアボード比（0.83／0.43／0.41／0.48／1.03）と一致し、
  符号が異なるセルは無かった。
- 差し替え対象の限定: fandhe-ai／candle／burn の非対象行は 0.9.0 本体 JSONL のまま。Python FW 行はリポジトリ収録の
  `results-dgx-py-0.8.0.jsonl`（2026-09-12 別セッション）を基底とし PyTorch cpu N=4096 のみ差し替えた
  （0.9.0 版 Artifact が用いた Python FW の実ファイルは未収録・元 Artifact も読めないため、他の Python FW セルが
  0.9.0 版と同値かは未検証。parity 値は決定的に一致する）。

## 公開 URL

- 再生成スコアボード（本 issue 版）: https://claude.ai/artifact/AJrFcmiMxNVrXLDLaPG2vu
- 0.9.0 版（https://claude.ai/artifact/Xr4j45QVkgqyext4THAsMW）は本セッションのアカウント一覧に無く
  in-place 更新できないため、新規 Artifact として公開した（0.9.0 版は不変のまま履歴として残る）。

## ディレクトリ構成

| パス | 内容 |
|---|---|
| `RULE.txt` | 事前登録判定規則（計測前にコミット） |
| （注） | RULE.txt の保存物一覧のうちノード側の計測ログ（build.log・load_gate.log・run{1..5}・env_info.txt・uptime）は `gb10/` 配下へ収納した（規則の変更ではない） |
| `orchestrate_gb10.sh` | ノード側オーケストレーション（3 バイナリ再ビルド → 専有ゲート付き 5 run） |
| `gb10/build.log` | 再ビルド記録（bench-burn の mtime 更新・`precision_class` 参照数・`fandhe-ai v0.9.0` ピン・torch 版） |
| `gb10/load_gate.log`・`uptime_before.txt`／`uptime_after.txt` | 専有ゲート履歴・負荷 |
| `gb10/run{1..5}/results.jsonl` | 各 run の bench-fandhe（reuse／fresh）・bench-candle・bench-burn CUDA GEMM 5 形状（20 行） |
| `gb10/run{1..5}/py.jsonl` | 各 run の `bench_py.py --framework pytorch --task gemm --device cpu --size 4096`（1 行） |
| `gb10/env_info.txt` | 実行環境（rev・driver・torch。内部ホスト名・絶対パスは `<home>` へマスク） |
| `aggregate.py` | 集計（python3 標準ライブラリのみ・`--self-test`）。採用 run（median_s 中央値）の行を派生 JSONL へ追記 |
| `aggregate.md` | run 別 fail／rescued／bound／median_s・セル判定・同一 run 内比・ゲート結果 |
| `results-dgx-0.9.0-precision-class.jsonl` | 0.9.0 本体 `results-dgx-0.9.0.jsonl` ＋ burn cuda 5 行（採用 run）を末尾追記（`gen_1988.py --gb`。index は後勝ち） |
| `results-dgx-py-precision-class.jsonl` | 0.9.0 流用 `results-dgx-py-0.8.0.jsonl` ＋ PyTorch cpu 4096 行（採用 run）を末尾追記（`--gb-py`） |
| `summary-run1-a-tf32.md` | `summarize.py`（#2046 後）を run1 に当てた出力。(a-tf32) 節の「要素単位検証」列が `ok（精度クラス TF32〈u=2^-11〉救済 N 要素）` になることが実装反映の一次証拠 |
| `scoreboard/gen_1988.py`・`body_1988.html`・`gen_1988.out` | スコアボード生成器（0.9.0 版 `gen_090.py` の派生。`--body`／`--style` 追加・burn `tf32:true` 行の △ title 出し分け）・本文・生成時の検証出力 |

## 再現手順

```bash
# ノード（GB10。Mac から --delete --exclude target で転送後、.rev-stamp を書いてから起動）
bash docs/perf/logs/framework-compare-precision-class-remeasure-1988/orchestrate_gb10.sh
# Mac（集計・派生 JSONL・スコアボード）
cd docs/perf/logs/framework-compare-precision-class-remeasure-1988
python3 aggregate.py --self-test
python3 aggregate.py --logs gb10 --md aggregate.md \
  --base-gb ../framework-compare-0.9.0-remeasure/gb10/results-dgx-0.9.0.jsonl --out-gb results-dgx-0.9.0-precision-class.jsonl \
  --base-py ../lowlayer-diagnosis-2026-09-12/dgx/results-dgx-py-0.8.0.jsonl --out-py results-dgx-py-precision-class.jsonl
R=../framework-compare-0.9.0-remeasure; L=../lowlayer-diagnosis-2026-09-12/dgx
python3 scoreboard/gen_1988.py --m4 $R/m4max-series-b/results-m4max-0.9.0-median5.jsonl \
  --m4-prev ../../../../scripts/bench/framework-compare/results/raw/results-m4max-0.8.0.jsonl \
  --gb results-dgx-0.9.0-precision-class.jsonl --gb-extra $R/gb10/results-dgx-0.9.0-extra.jsonl \
  --gb-py results-dgx-py-precision-class.jsonl \
  --gb-prev $L/results-dgx-0.8.0.jsonl $L/results-dgx-0.8.0-extra.jsonl $L/results-dgx-py-0.8.0.jsonl \
  --out fandhe-ai-0.9.0-scoreboard-1988.html
```

## スコアボード生成器の非後退確認

`gen_1988.py --body $R/scoreboard/body_090.html` を 0.9.0 の元入力（`--gb $R/gb10/results-dgx-0.9.0.jsonl`・
`--gb-py $L/results-dgx-py-0.8.0.jsonl`）で実行した HTML は、`gen_090.py` の出力と **byte 同一**（`cmp` で確認。
△ title の出し分けは burn `tf32:true` かつ fail=0 の行にしか効かず、0.9.0 データでは判定不能経路のため出力は不変）。
本文の差分は `diff $R/scoreboard/body_090.html scoreboard/body_1988.html` のとおり固定文言（更新日・GB10 節の
注記・計測条件・再現手順）のみ。

## レビュー是正（PR #2047・codex P2 2 件）

- `aggregate.py`: `parity_fail_count` の欠損・null・負数・非整数を `or 0` で成功扱いにせず、入力不正として集計を停止する
  （`--self-test` に拒否ケース 4 件を追加。収録ログでの `aggregate.md` 出力は byte 同一）。
- `aggregate.py`: 派生 JSONL へ採用する対象セルは 5 run 完備を必須とし、1〜4 run のセルがあれば出力せず停止する
  （`--self-test` に 4 run の拒否ケースを追加。収録ログでの派生 JSONL 2 本は byte 同一）。
- `orchestrate_gb10.sh`: 3 バイナリの再ビルド失敗時は計測開始前に停止する（本実測では 3 本とも rc=0。`gb10/build.log`）。
  いずれも本実測の結果・判定には影響しない（計測後の是正であり、規則の変更ではない）。

## 変更していないもの

tolerance 4 定数・判定式・`BASELINES`・本体 `compare`／`assert_parity`／`ParityBaseline`・`bench_py.py`
（係数 0.5・u=2^-24 リテラル）・0.9.0 ディレクトリの全ファイル・fandhe-ai／candle 行と対象外の全セル。
採否判定（ADOPT／REJECT）を伴わない記録であり、新規 issue の起票はしていない。
