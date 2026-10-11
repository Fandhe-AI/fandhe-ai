# fandhe-ai 0.11.0 framework-compare の Apple 側を Mac mini で計測し直す

## 目的

0.11.0 正式系列（`../framework-compare-0.11.0-remeasure/`・2026-10-10）の Apple 側は M4 Max 開発機で計測した。
M4 Max 開発機は他アプリの並走で負荷が安定しないため、Metal の計測・実機テストの実行先を計測専用の
Mac mini（Apple M4・10 コア・16 GB）へ移した（2026-10-11 所有者指示・PR #3001・`docs/real-hardware-verification-env.md` §7）。
本ディレクトリは、同じ 6 フレームワーク同席計測（fandhe-ai／candle／burn＋PyTorch／TensorFlow／SciPy）を Mac mini で
回した生ログ・集計・スコアボード生成器を収める。GB10 側は計測し直しておらず、0.11.0 正式系列の値をそのまま使う。

## ディレクトリ構成

| パス | 内容 |
|---|---|
| `macmini-series-a/` | Mac mini・系列 A（ゲートなし・対照）5 run。`run{1..5}/` と中央値 JSONL 2 本 |
| `macmini-series-b/` | Mac mini・系列 B（load1 < 3.0 ゲート付き）5 run。**正式系列**。`run{1..5}/`・中央値 JSONL 2 本・`gate.log`・`RULE.txt`（事前登録規則） |
| `loop-macmini.log` | 系列 A → 系列 B の連続実行ログ |
| `scripts/` | `macmini-run-0110.sh`（1 ラウンド）・`macmini-0110a-loop.sh`（系列 A）・`macmini-0110b-loop.sh`（系列 B・ゲート実装） |
| `aggregate/median_rounds.py` | 5 run → セルごとの中央値 run の行。0.11.0 正式系列からのコピー（変更なし） |
| `scoreboard/` | `gen_0110_macmini.py`・`body_0110_macmini.html`・`style.css`・`gen_args_macmini.txt`（負荷注記などの引数）・`gen_0110_macmini.out`（判定一覧） |

各 run ディレクトリは `run.log`・`results-macmini-0.11.0.jsonl`（fandhe-ai／candle／burn）・`results-macmini-py-0.11.0.jsonl`（Python 3 FW）・
`skipped-macmini-0.11.0.log`・`tree.txt`・`rev-stamp.txt`・`uptime_*.txt` を持つ。

スクリプトの M4 Max 版（`../framework-compare-0.11.0-remeasure/scripts/m4max-*.sh`）からの差分は、出力ファイル名とパス、
ゲート閾値（8.0 → 3.0）、ツリー同定の取得元（git 履歴のない展開ツリーのため `.rev-stamp` から読む）、venv の位置だけである。
失敗検知・registry ピン検査・期待行数検査・`gate()` の数値検証（`GATE-INVALID`）は維持した。

## 計測条件

- **機体**: Mac mini（Apple M4・10 コア・16 GB）・macOS 27.0.1・計測専用機（平常 load1 は約 1.3）。
- **ツリー**: main `624d0ee4`（GB10 の 0.11.0 計測と同一）。`git archive 624d0ee4 scripts/bench/framework-compare` を展開して
  `.rev-stamp`（フル SHA）を置き、rsync で転送した。全 10 run の `rev-stamp.txt` が同じ SHA。
- **registry ピン**: `fandhe-ai =0.11.0`・`candle-core =0.11.0`・`burn =0.21.0`。全 10 run の `tree.txt` が `fandhe-ai v0.11.0` の 1 行で一致した
  （一致しなければ `macmini-run-0110.sh` が計測を中止する）。
- **Python 3 FW**: Python 3.12.15（uv 管理・ユーザー領域）。`venv-torch` は torch 2.14.0・scipy 1.18.1・numpy 2.5.4、
  `venv-tf` は tensorflow 2.16.2・tensorflow-metal 1.2.0・numpy 1.26.4・keras 3.15.1。FW の版は M4 Max 計測と同じ。
- **bench_py.py**: `docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py`（sha256 `9fd2c08f…2409b`。全文は `RULE.txt`）。
  M4 Max 計測が使った版の記録はないが、Python 3 FW の checksum は M4 Max 系列 B の中央値行と全 32 セルで一致した
  （TensorFlow CPU の学習 loss のみ 2e-8 の差）。checksum の一致は同じ入力・同じ計算であることの確認で、スクリプトの bit 同一性の証明ではない。
- **日時**: 2026-10-11 11:45:34Z〜12:16:26Z（系列 A 11:45:34Z〜12:02:30Z・系列 B 12:02:30Z〜12:16:26Z）。`RULE.txt` の固定は 11:45:34Z（結果が出る前）。
- **負荷**: 各 run 開始時の load1（`uptime_before.txt`）は系列 A が 1.34／1.65／1.96／3.70／2.85、系列 B が 2.67／2.49／2.67／2.63／1.85。
  計測ハーネス自身が run 後の load1 を 3 前後まで押し上げるため、系列 B のゲートは直前 run の残余の減衰待ちとして働いた
  （`gate.log` 7 エントリ。run2・run5 の前に 3.68・3.08 で 1 分ずつ待機。`GATE-INVALID`・`GATE-TIMEOUT` なし）。
- **計測失敗**: 全 10 run で `bench-burn gemm metal 512/1024/2048/4096 fresh` が MEASURE_ERROR（結果テンソル全ゼロ。M4 Max 計測と同じ upstream 既知バグ）。それ以外の失敗はない。
- **完走**: `loop-macmini.log` は系列 A・B とも `ALL-DONE completed=5/5`。全 10 run の `run.log` 最終行は `done. rows=48 pyrows=32 skipped=4`。

## 事前登録規則と採否

`macmini-series-b/RULE.txt` のとおり、系列 B が 5/5 run ゲート通過で完走したため系列 B を正式値とする。
系列 A は対照で、A/B 間で勝敗区分が反転した行をノイズ帯として示す。

## 結果（Mac mini 系列 B・GB10 は 0.11.0 正式系列のまま）

勝敗区分は 0.11.0 正式系列と同じ規則（同一 run 内の「有効な相手最速 ÷ fandhe-ai reuse」の 5 run 中央値。1 超で 1 位・0.90 以上で僅差）。
`scoreboard/gen_0110_macmini.out` より、判定対象 27 行の内訳:

| 区分 | 行数 | 内訳 |
|---|---|---|
| 1 位 | 6 | Mac mini 3（gemm CPU N=256・N=512・infer Metal）・GB10 3 |
| 僅差 | 1 | GB10 1（gemm CPU N=1024） |
| 負け | 20 | Mac mini 10・GB10 10 |
| 判定不能セル | 1 | GB10 PyTorch CPU N=4096 の 1 要素 |
| ノイズ帯 | 1 | Mac mini infer Metal（系列 B は 1 位 1.13×、系列 A は 3 位 0.72×） |

Mac mini の 13 行（同一 run 内比の 5 run 中央値）と、M4 Max 系列 B（0.11.0 正式系列）の同じ行:

| 行 | Mac mini | M4 Max |
|---|---|---|
| gemm Metal N=256 | 負け 0.84×（vs candle） | 負け 0.88× |
| gemm Metal N=512 | 負け 0.80× | 負け 0.78× |
| gemm Metal N=1024 | 負け 0.69× | 負け 0.72× |
| gemm Metal N=2048 | 負け 0.73× | 負け 0.68× |
| gemm Metal N=4096 | 負け 0.58× | 負け 0.65× |
| gemm CPU N=256 | **1 位 1.04×**（vs PyTorch） | 負け 0.77× |
| gemm CPU N=512 | **1 位 1.04×**（vs PyTorch） | 僅差 0.95× |
| gemm CPU N=1024 | 負け 0.89× | 負け 0.87× |
| gemm CPU N=2048 | 負け 0.69× | 負け 0.80× |
| train Metal | 負け 0.32× | 負け 0.39× |
| infer Metal | 1 位 1.13×（ノイズ帯） | 1 位 1.41× |
| train CPU | 負け 0.20× | 負け 0.22× |
| infer CPU | 負け 0.25× | 負け 0.20× |

機体が違う（コア数・メモリが異なる）ため、絶対 ms は比べない。対戦相手比の区分が変わったのは
gemm CPU N=256（負け → 1 位）と N=512（僅差 → 1 位）の 2 行。原因の切り分けはしていない。
infer Metal は系列 B の run 別比が 1.31／0.71／0.90／1.40／1.13 とばらつき、系列 A では負けになる。1 位の区分は安定していない。

## 0.10.0 比を出さない

0.10.0 と 0.11.0 正式系列は M4 Max で計測しているため、Mac mini の行には同一セルの 0.10.0 比を出さない
（`gen_0110_macmini.py` の `--m4-prev` を空にすると「0.10.0 比なし（機体変更）」になる）。GB10 の 0.10.0 比は 0.11.0 正式系列と同じ。

## 役割・機能の対応表

fandhe-ai 列の判定（タグ `v0.11.0` の facade 公開面の監査・`docs/perf/framework-compare-feature-matrix-0.11.0.md`）は変えていない。
タグ以降に main へ入った実機結果を「main」注記として各セルに足した。注記の内容は crates.io の 0.11.0 には含まれない。

| 行 | main 注記の出典 |
|---|---|
| dtype | f16／bf16 opt-in の小形状 6 Op が GB10・M4 Max で CPU と bit 一致（#2988）。Tensor Core 経路の大形状は未実測（#3009）。int8 の演算経路はない |
| 自動微分・演算の範囲 | Phase 4 の 14 機能（19 テスト）が GB10（`784b4e79`）・Mac mini（`c8e1ceb1`）で全件 pass（#3006・#3008） |
| 演算の範囲 | CUDA・Metal の線形代数カーネルは既定 OFF の内部ゲートと準備段階まで（#2982・#2984・#2986）。ゲート ON でも `Unsupported` を返し、facade には公開していない（`docs/backend-gpu-linalg-decision.md`） |
| バックエンド・ハード | int8 は GB10 の INT8 MMA プローブ（#2978）と Transformer 分母の計測（#3012・#3014）のみ。量子化は spec の除外事項で REQ 未承認 |

## 再現

```bash
# Mac mini（D=作業ディレクトリ。venv-torch／venv-tf と bench_py.py を D に置く）
bash scripts/macmini-0110a-loop.sh; bash scripts/macmini-0110b-loop.sh
# 集計（系列・ファイル種別ごと）
python3 aggregate/median_rounds.py macmini-series-b/results-macmini-0.11.0-median5.jsonl macmini-series-b/run{1..5}/results-macmini-0.11.0.jsonl
python3 aggregate/median_rounds.py macmini-series-b/results-macmini-py-0.11.0-median5.jsonl macmini-series-b/run{1..5}/results-macmini-py-0.11.0.jsonl
# スコアボード（GB10 は 0.11.0 正式系列を参照。注記の引数は scoreboard/gen_args_macmini.txt の各行）
G=../framework-compare-0.11.0-remeasure/gb10
python3 scoreboard/gen_0110_macmini.py \
  --m4 macmini-series-b/results-macmini-0.11.0-median5.jsonl --m4-py macmini-series-b/results-macmini-py-0.11.0-median5.jsonl \
  --m4-alt macmini-series-a/results-macmini-0.11.0-median5.jsonl --m4-py-alt macmini-series-a/results-macmini-py-0.11.0-median5.jsonl \
  --m4-runs macmini-series-b --m4-alt-runs macmini-series-a \
  --gb $G/results-dgx-0.11.0-median5.jsonl --gb-extra $G/results-dgx-0.11.0-extra-median5.jsonl \
  --gb-py $G/results-dgx-py-0.11.0-median5.jsonl --gb-runs $G \
  "--m4-series=…" "--m4-load=…" "--m4-date=…" "--m4-skipped-note=…" "--gb-load=…"
```

## 公開先

スコアボードは claude.ai Artifact として公開した（非公開の個人 Artifact。<https://claude.ai/artifact/RKWKJec9BeyvoH5QEDJtxQ>。2026-10-11・`gen_0110_macmini.py` 出力）。
M4 Max 版の 0.11.0 ページは上書きしていない。
