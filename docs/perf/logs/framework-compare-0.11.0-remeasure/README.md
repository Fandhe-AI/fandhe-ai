# fandhe-ai 0.11.0 正式系列 framework-compare 再計測

## 目的

crates.io `fandhe-ai =0.11.0` が 2026-10-09 に公開された（`release-all.yml` run 37933003465・タグ `v0.11.0`）。
これを受けて framework-compare の承認ピンを `=0.11.0` へ更新した（PR #2962）。
そのうえで GEMM・train・infer の対戦成績を両機体で計測し直し、生ログ・集計結果・スコアボード生成器をここに収めた。

手順・計測条件・ディレクトリ構成は 0.10.0 版（`../framework-compare-0.10.0-remeasure/`）と同じ。
生データは PR #2964（GB10）・PR #2963（M4 Max）で収めた。0.10.0 版からの違いは次の 2 点。

- **GB10 は最初から 5 ラウンドの予定で計測した。** 0.10.0 では run1 を見てから run2〜5 を足したが、今回は run1 → run2〜5 を続けて実行した。
- **比較元は 0.10.0 の 5 ラウンド中央値。** 「0.10.0 比」は、M4 Max は 0.10.0 系列 B、GB10 は 0.10.0 の `*-median5.jsonl` との比。

## ディレクトリ構成

| パス | 内容 |
|---|---|
| `m4max-series-a/` | Apple M4 Max・系列 A（ゲートなし）5 run。<br>`run{1..5}/`、`results-m4max-0.11.0-median5.jsonl`（fandhe-ai／candle／burn）、`results-m4max-py-0.11.0-median5.jsonl`（Python 3 FW） |
| `m4max-series-b/` | Apple M4 Max・系列 B（load1 < 8.0 ゲート付き）5 run。**正式系列**。<br>`run{1..5}/`、中央値 JSONL 2 本（系列 A と同じ構成）、`gate.log`、`RULE.txt`（事前登録規則） |
| `loop-m4max.log` | 系列 A → 系列 B の連続実行ログ |
| `gb10/` | DGX Spark GB10 の 5 run。<br>`run{1..5}/`（0.10.0 版と同じファイル群）、3 系統の `*-median5.jsonl`、`loop.log`（run2〜5）、`tree.txt`（registry ピン解決と Fresh ビルドの確認。計測前と計測後） |
| `scripts/` | 0.10.0 版から版数（`0.10.0`→`0.11.0`、`0100`→`0110`、出力ファイル名、ログディレクトリ名）だけを置き換えた計測スクリプト |
| `aggregate/median_rounds.py` | 5 run → セルごとの中央値 run の行。0.10.0 版と同一 |
| `scoreboard/` | `gen_0110.py`、`body_0110.html`、`style.css`、`gen_args.txt`（負荷注記の引数）、`gen_0110.out`（判定一覧の標準出力） |

`gen_0110.py` は `gen_0100.py` の改変版。判定ロジックは変えておらず、変更点は次の 4 つ。

- 版数
- 入力ファイル名
- 比較元（`--m4-prev`・`--gb-prev` の既定を 0.10.0 の 5 ラウンド中央値にした）
- 本文テンプレート名

`body_0110.html` で 0.10.0 版から書き換えたのは次の 4 か所。

- 日付と版数
- GB10 の 5 ラウンド化の説明
- 「0.10.0→0.11.0 の変化」節
- 役割・機能の対応表の fandhe-ai 列

ログ中の個人パスの置き換え方は次のとおり。

- リポジトリの絶対パスは `<repo>` にした。
- DGX のホームは `~` にした。
- 作業ディレクトリの絶対パスは、収めたファイルには出てこない。

`RULE.txt` 内の `m4max-0110a/`・`m4max-0110b/` は、計測時の作業ディレクトリ名である（収納時に `m4max-series-a/`・`m4max-series-b/` へ改名した）。

## 計測条件

- **registry ピン**: `fandhe-ai =0.11.0`・`candle-core =0.11.0`・`burn =0.21.0`（`deps-policy.md` 第 9 区分）。
  - M4 Max: 全 10 run の `run*/tree.txt` が `fandhe-ai v0.11.0`。外れたら計測しない設計。
  - GB10: 計測前と計測後の 2 回、次を `gb10/tree.txt` に記録した。
    - `GEMM_GATE_PATCH_FACADE_PATH` が未設定であること
    - `cargo tree -p bench-fandhe --locked` の `fandhe-ai v0.11.0`（path なし）
    - `Cargo.lock` の `source = "registry+…crates.io-index"`
    - `cargo build --release -p bench-fandhe --locked -v` が `Fresh`
  - 各 `run.log` の `WARN: registry 0.11.0 line not found in log` は 0.10.0 と同じ理由で出たもの（`run_all_cuda.sh` が `cargo tree` を出力しない）。
- **ツリー**: main `624d0ee4`（PR #2962 マージ後）。DGX へは同じツリーを rsync した。
- **macOS 専用 feature**: 計測前に `cargo check -p bench-fandhe --locked --features <feature>` を実行した。
  - 対象は 5 種（`metal-split-k-toggle`・`pinned-h2d-toggle`・`managed-placement`・`device-checksum`・`graph-step`）。
  - 結果はすべて rc=0・警告なし。
  - PR #2962 では Linux 上で確かめられなかった項目。
- **Python 3 FW**: 版は 0.10.0 と同じ。計測スクリプトは `docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py` と同一内容のコピー。
  - M4 Max: 隔離 venv（Python 3.12.12）。torch 2.14.0／scipy 1.18.1、TF は 2.16.2＋tensorflow-metal 1.2.0 の別 venv。
  - GB10: Python 3.12.3、torch 2.14.0+cu130・TF 2.21.0（CPU のみ計測）・scipy 1.18.1。
- **Apple M4 Max**（macOS 27.0.1〈26A434〉・rustc 1.98.1）:
  - 日時: 2026-10-10 08:42:29〜08:55:34 UTC。
  - 系列 A（ゲートなし）: 各 run 開始時の load1 は 5.03／5.82／4.77／5.40／4.39。run1 のみビルドを含む。
  - 系列 B: 5/5 run がゲートを 1 回目の判定で通過して完走した。
    - 通過時の load1 は 4.81／4.92／4.75／5.14／4.55（`gate.log` 全 5 エントリ）。
    - ゲートは開始時点の判定だけで、run 中の負荷は統制していない。
  - 計測失敗: 全 10 run で `bench-burn gemm metal 512/1024/2048/4096 fresh` が MEASURE_ERROR になった（結果テンソル全ゼロ。0.8.0 以来の upstream 既知バグ）。それ以外の失敗はない。
- **DGX Spark GB10**（NVIDIA GB10・driver 580.173.02・CUDA 13.0・rustc 1.97.0）:
  - 日時: 2026-10-10 08:41:17〜09:01:43 UTC に 5 ラウンドを連続実行した。1 ラウンドは約 4 分。
  - 負荷: 各ラウンド開始時の load1 は 1.01／6.34／7.30／6.21／5.82（直前ラウンドの残余を含む）。開始時の GPU 使用率は 0〜1%。
  - 計測失敗: なし。
    - 5 ラウンドとも `run_all rc=0` で、FAIL 行はない。
    - 行数は stage1 112・skipped 0、stage2 8、stage3 28 で、期待値と一致した。
    - `extra.err`・`py.err` は空。

## 事前登録規則と実績

`m4max-series-b/RULE.txt`（2026-10-10T08:42:17Z。系列 A の計測開始より前に固定）は 0.10.0 版の転記。変えたのは次の 4 か所。

- タイムスタンプ
- 作業ディレクトリ名
- 絶対 ms 差分の比較元（0.10.0 → 0.11.0）
- ツリー同定（`624d0ee4`）

**実績**: 系列 B が 5/5 run 完走したので、**系列 B を正式値**にした。

**ノイズ帯**（勝敗区分が系列 A と B で異なる行）: **0 行**。系列 A を正式値にしても、内訳は 1 位 4・僅差 2・負け 21 で変わらない。
0.10.0 版はノイズ帯が 4 行あった。そのときの系列 A の開始時 load1 は 5.30〜28.51 だった。今回は系列 A も 4.39〜5.82 で、両系列の負荷差が小さかった。

## 結果（正式: M4 系列 B・GB10 5 ラウンド中央値）

判定方法は 0.10.0 版と同じ。

- 区分と比は、同一 run 内の対戦相手比の 5 run 中央値で決める。1 を超えれば 1 位、0.90 以上なら僅差、それ未満なら負け。
- 順位と相手名は、5 ラウンド中央値で数える。

内訳は `scoreboard/gen_0110.out` より、判定対象 27 行について次のとおり。

| 区分 | 行数 | 該当 |
|---|---|---|
| 1 位 | 4 | M4 Max infer Metal（1.41×）、GB10 gemm CUDA N=4096（1.01×）、GB10 gemm CPU N=2048（1.06×）・N=4096（1.21×） |
| 僅差 | 2 | M4 Max gemm CPU N=512（0.95×）、GB10 gemm CPU N=1024（0.98×） |
| 負け | 21 | 上記以外 |
| 判定不能セル | 1 | GB10 PyTorch CPU N=4096 の 1 要素。全 5 run 共通で、0.10.0 と同じ |
| ノイズ帯 | 0 | |

中央値同士の比で区分が変わるのは M4 Max gemm Metal N=256 の 1 行だけで、同一 run 内比では 0.88×（負け）、中央値同士の比では 0.94×（僅差）になる。

### 0.10.0 版からの区分の変化

0.10.0 版の内訳は 1 位 5・僅差 3・負け 19 だった。区分が変わったのは次の 3 行で、いずれも 0.10.0 版でノイズ帯だった行。

| 行 | 0.10.0（系列 B） | 0.11.0（系列 B） | 0.11.0 の run 別 |
|---|---|---|---|
| M4 Max gemm Metal N=256 | 僅差 0.95× | 2 位 0.88× | 0.81／0.95／0.82／0.88／0.97 |
| M4 Max gemm CPU N=512 | 1 位 1.05× | 僅差 0.95× | 0.95／1.16／0.94／1.08／0.95 |
| M4 Max gemm CPU N=1024 | 僅差 0.96× | 2 位 0.87× | 0.87／0.85／0.91／0.89／0.87 |

GB10 の 14 行は区分が変わっていない。

v0.11.0 の変更（214 コミット）の中心は facade 公開面の拡大で、registry 既定経路の GEMM・学習・推論の選択ロジックとカーネルは変わっていない。backend 3 クレートの差分は、新規 op とスカラー演算カーネルの追加である。
fandhe-ai 自身の中央値の 0.10.0 比は、GB10 の reuse 行で 0.95〜1.04×、M4 Max で 0.91〜1.15×だった。M4 の絶対 ms は負荷差の交絡を含む参考値。

### round_spread が 1.5 を超えたセル

| セル | run1〜run5（ms） |
|---|---|
| GB10 burn gemm CUDA N=256（TF32 既定） | 0.175／0.170／0.172／0.171／0.625 |
| GB10 burn gemm CUDA N=512（TF32 既定） | 0.494／0.344／0.690／0.452／0.391 |
| GB10 burn infer CPU | 0.750／0.240／0.236／0.242／0.743 |
| GB10 burn infer CUDA | 0.435／0.380／0.104／0.442／0.750 |
| GB10 PyTorch gemm CPU N=256 | 0.262／0.271／0.418／0.263／0.484 |
| GB10 PyTorch train CUDA | 1.463／0.419／1.534／1.303／1.230 |
| GB10 TensorFlow infer CPU | 0.503／0.556／1.134／1.101／1.252 |
| GB10 TensorFlow train CPU | 2.713／2.650／4.077／4.399／2.828 |
| M4 Max（系列 B）fandhe-ai infer Metal fresh | 0.631／0.585／0.643／0.465／0.713（判定は reuse 行） |

0.10.0 版で揺れが run1 に集中していた candle／PyTorch の infer CUDA は、今回は 1.5 を超えていない。
全セルの round_spread は各 `*-median5.jsonl` の `round_spread` フィールドにある。

## 役割・機能の対応表（計測外）

スコアボード下部の「役割・機能の対応表」の fandhe-ai 列は、タグ `v0.11.0` の facade 公開面の再監査（`docs/perf/framework-compare-feature-matrix-0.11.0.md`・#2961）に合わせて書き換えた。

- 0.10.0 で「部分的」だった 9 行のうち 7 行（dtype・自動微分・演算の範囲・NN 層・最適化・相互運用・推論）は本文を更新した。判定はいずれも「部分的」のまま。
- バックエンド・事前学習済みモデルの 2 行は、再監査文書が「変化なし」としているので変えていない。
- 演算数（0.10.0 版の「約 110」）は再監査文書に裏づけがないので、数を書かない形にした。
- PyTorch・TensorFlow・SciPy・Hugging Face・LangChain の列は変更していない。

## 再現

```bash
# M4 Max（ROOT=リポジトリルート、D=作業ディレクトリ。venv-torch／venv-tf を D の隣に、bench_py.py のコピーを D/py に置く）
bash scripts/m4max-0110a-loop.sh; bash scripts/m4max-0110b-loop.sh
# GB10（~/work にツリーを rsync 済み。run1 は dgx-run-0110.sh を LOGD=~/work/dgx-0110-logs/run1 で、run2〜5 は dgx-0110-loop.sh）
LOGD=~/work/dgx-0110-logs/run1 bash ~/work/dgx-run-0110.sh; bash ~/work/dgx-0110-loop.sh
# 集計（系列・ファイル種別ごと）
python3 aggregate/median_rounds.py m4max-series-b/results-m4max-0.11.0-median5.jsonl m4max-series-b/run{1..5}/results-m4max-0.11.0.jsonl
python3 aggregate/median_rounds.py gb10/results-dgx-0.11.0-median5.jsonl gb10/run{1..5}/results-dgx-0.11.0.jsonl
# スコアボード（比較元 0.10.0 は ../framework-compare-0.10.0-remeasure/ を既定参照。負荷注記は gen_args.txt）
python3 scoreboard/gen_0110.py \
  --m4 m4max-series-b/results-m4max-0.11.0-median5.jsonl --m4-py m4max-series-b/results-m4max-py-0.11.0-median5.jsonl \
  --m4-alt m4max-series-a/results-m4max-0.11.0-median5.jsonl --m4-py-alt m4max-series-a/results-m4max-py-0.11.0-median5.jsonl \
  --gb gb10/results-dgx-0.11.0-median5.jsonl --gb-extra gb10/results-dgx-0.11.0-extra-median5.jsonl \
  --gb-py gb10/results-dgx-py-0.11.0-median5.jsonl \
  --m4-runs m4max-series-b --m4-alt-runs m4max-series-a --gb-runs gb10 \
  "--m4-series=…" "--m4-load=…" "--gb-load=…"   # 3 引数の実値は scoreboard/gen_args.txt の各行
```

## 公開先

スコアボードは claude.ai Artifact として公開した（非公開の個人 Artifact。<https://claude.ai/artifact/6q53rzd19gD8rJwfVHdh6L>）。0.10.0 版のページは上書きしていない。
