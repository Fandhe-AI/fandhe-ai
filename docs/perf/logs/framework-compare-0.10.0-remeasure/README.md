# fandhe-ai 0.10.0 正式系列 framework-compare 再計測

## 目的

crates.io `fandhe-ai =0.10.0` の公開（2026-10-03。`release-all.yml` run 37112020053・タグ `v0.10.0`）を受け、
framework-compare の承認ピンを `=0.10.0` へ更新した（PR #2497）。そのうえで GEMM・train・infer の
対戦成績を両機体で計測し直し、生ログ・集計スクリプト・スコアボード生成器をここへ収める。
形式は 0.9.0 版（`../framework-compare-0.9.0-remeasure/`）にそろえている。

0.9.0 版からの主な違い:

- **Python 3 参照 FW（PyTorch・TensorFlow・SciPy）も両機体で実測した。** 各ラウンドの中で Rust セルの直後に計測している。
  0.9.0 版では M4 Max が 2026-09-12 版ページからの転記値、GB10 が別セッション値の流用だったが、これを解消した。
- **GB10 も 5 ラウンドの中央値にした**（0.9.0 版は単一セッション）。後述の「事前登録外の拡張」を参照。
- **GB10 の結果から TF32 opt-in 行を判定対象外にした。** `run_all_cuda.sh` は 0.10.0 で TF32 opt-in スイープ（#1983）を出力する。
  fandhe-ai／candle の `"tf32": true` 行はキーで分離し、f32 同士の判定から外した。

## ディレクトリ構成

| パス | 内容 |
|---|---|
| `m4max-series-a/` | Apple M4 Max・系列 A（共有負荷下・ゲートなし）5 run。<br>`run{1..5}/`、`results-m4max-0.10.0-median5.jsonl`（fandhe-ai／candle／burn）、`results-m4max-py-0.10.0-median5.jsonl`（Python 3 FW） |
| `m4max-series-b/` | Apple M4 Max・系列 B（load1 < 8.0 ゲート付き）5 run。**正式系列**。<br>`run{1..5}/`、中央値 JSONL 2 本（系列 A と同じ構成）、`gate.log`（ゲート判定の履歴）、`RULE.txt`（事前登録規則） |
| `loop-m4max.log` | 系列 A → 系列 B の連続実行ログ |
| `gb10/` | DGX Spark GB10 の 5 run。<br>`run{1..5}/`（`results-dgx-0.10.0.jsonl`＝`run_all_cuda.sh` 全セル、`-extra`＝CPU gemm reuse と N=4096 の追加計測、`results-dgx-py-0.10.0.jsonl`＝Python 3 FW、`run_all_cuda.log`、`uptime_*.txt`、`gpu_util_before.txt`、`skipped-dgx-0.10.0.log`、`run.log`）<br>3 系統それぞれの `*-median5.jsonl`、`loop.log`（run2〜5 の連続実行）、`tree.txt`（registry ピン解決の確認） |
| `scripts/` | 計測オーケストレーション。<br>`m4max-run-0100.sh`（1 ラウンド）、`m4max-0100a-loop.sh`（系列 A）、`m4max-0100b-loop.sh`（系列 B・ゲート実装）、`dgx-run-0100.sh`（GB10 1 ラウンド）、`dgx-0100-loop.sh`（GB10 run2〜5） |
| `aggregate/median_rounds.py` | 5 run → セルごとの中央値 run の行。0.9.0 版の改変。差分は下表のとおり |
| `scoreboard/` | スコアボード生成器。<br>`gen_0100.py`、`body_0100.html`、`style.css`、`gen_args.txt`（負荷注記の引数）、`gen_0100.out`（判定一覧の標準出力） |

`median_rounds.py` の 0.9.0 版からの差分:

| 項目 | 内容 |
|---|---|
| キー | セルのキーに `tf32`（真偽）を加えた。こうしないと GB10 の f32 行と TF32 行が同じセルに混ざる |
| 欠損・重複 | run の欠損セル・run 内の重複セルはエラー終了する |

`RULE.txt` 内のディレクトリ名 `m4max-0100a/`・`m4max-0100b/` は、計測時の作業ディレクトリ名である（収納時に `m4max-series-a/`・`m4max-series-b/` へ改名）。
ログ中のリポジトリの絶対パスは `<repo>`、作業ディレクトリは `<scratch>`、DGX のホームは `~` に置き換えてある。

## 計測条件

- **registry ピン**: `fandhe-ai =0.10.0`・`candle-core =0.11.0`・`burn =0.21.0`（`deps-policy.md` 第 9 区分）。
  - M4 Max: 各 `run*/tree.txt`（`cargo tree` の `fandhe-ai v0.10.0`）で確認し、外れたら計測しない設計にした。
  - GB10: `dgx-run-0100.sh` が `run_all_cuda.log` で確認しようとした行は、`run_all_cuda.sh` が `cargo tree` を出力しないため見つからず、各 `run.log` に WARN が出た。そこで計測後に DGX 上で次を取得し、`gb10/tree.txt` に収めた。
    - `GEMM_GATE_PATCH_FACADE_PATH` が未設定であること（path patch なし）
    - `cargo tree -p bench-fandhe --locked` の `fandhe-ai v0.10.0`（path なし）
    - `Cargo.lock` の `source = "registry+…crates.io-index"`
    - `cargo build --release -p bench-fandhe --locked -v` が `Fresh`（計測バイナリがこの解決で作られていること）
- **ツリー**: 計測時は PR #2497 のブランチ（`c8bb4132`）。squash マージ後の main `9f5b6cda` とツリーは同一。DGX は同じツリーを rsync した。
- **Python 3 FW**:
  - M4 Max は隔離 venv で計測した（torch 2.14.0／scipy 1.18.1、TF は 2.16.2＋tensorflow-metal 1.2.0 の別 venv）。
  - GB10 は torch 2.14.0+cu130・TF 2.21.0（CPU のみ）・scipy 1.18.1。
  - 計測スクリプトは `scripts/bench/framework-compare/` 配下の `bench_py.py` と同一内容のコピー。
- **Apple M4 Max**:
  - 系列 A（ゲートなし）: 各 run 開始時の load1 は 5.30／28.51／14.65／10.62／9.20（他セッション並走の共有負荷下）。run1 のみビルドを含む。
  - 系列 B: 各 run 開始前に load1 < 8.0 を最大 30 分待つ。5/5 run がゲートを通過して完走した（通過時 load1 は 7.36／5.50／6.44／6.41／7.80。`gate.log` 全 9 エントリ）。ゲートは開始時点の判定だけなので、run 中の負荷は統制していない（系列 B run3 では candle CPU N=512 が 20.6 ms と単発で揺れた）。
  - 計測失敗: 全 10 run で `bench-burn gemm metal 512/1024/2048/4096 fresh` が MEASURE_ERROR（結果テンソル全ゼロ。0.8.0／0.9.0 と同じ upstream 既知バグ）。それ以外の失敗はない。
- **DGX Spark GB10**:
  - 環境: NVIDIA GB10・driver 580.173.02・CUDA 13.0。
  - 日時: 2026-10-03 09:15:33〜09:49:46 UTC に 5 ラウンドを連続実行（1 ラウンドは約 4〜6.5 分）。
  - 負荷: 各ラウンド開始時の load1 は 1.39／1.65／6.50／3.04／6.54（直前ラウンドの残余を含む）。開始時の GPU 使用率は 0〜3%（常駐の推論サービスはアイドル）。
  - 計測失敗: なし（全 run の `skipped-dgx-0.10.0.log` が空）。

## 事前登録規則と実績

`m4max-series-b/RULE.txt`（2026-10-03T09:26:07Z。系列 A・B いずれの結果も出る前に固定）:

- 系列 B は、5 run すべてがゲートを通過して完走した場合だけ正式値にする。部分完走や GATE-TIMEOUT なら系列 A を正式値にする。
- 両系列を併記し、A/B 間で verdict が反転したセルはノイズ帯として明示する。
- 0.9.0 → 0.10.0 の絶対 ms 差分は、M4 では負荷差の交絡があるので参考値とする。

**実績**: 系列 B が 5/5 run 完走したので、**系列 B を正式値**にした。

**ノイズ帯の定義**: 勝敗区分（1 位／僅差／負け）が系列 A と B で異なる行。2 位⇄3 位のように「負け」の中で順位が動くだけの場合は反転として数えない。

該当は 4 行:

| 行 | 正式（系列 B） | 対照（系列 A） |
|---|---|---|
| gemm Metal N=256 | 僅差 0.97× | 2 位 0.85× |
| gemm CPU N=512 | 1 位 1.05× | 2 位 0.88× |
| gemm CPU N=1024 | 僅差 0.99× | 4 位 0.62× |
| infer Metal | 1 位 1.45× | 2 位 0.86× |

系列 A を正式値にした場合、M4 Max 13 行の内訳は 1 位 0・僅差 0・負け 13 になる（系列 B では 1 位 2・僅差 2・負け 9）。
M4 の CPU／推論セルの勝敗は、共有負荷の有無で動く範囲にある。

### 事前登録外の拡張（GB10 の 5 ラウンド化）

RULE.txt が事前に固定したのは M4 の系列 A/B だけで、GB10 は当初 0.9.0 版と同じ単一セッションの予定だった。

初回セッション（`gb10/run1/`）では、candle／burn／PyTorch の CUDA 小形状と推論が 0.9.0 や他ラウンドに比べて大きく揺れた。
その **run1 の結果を見たうえで** run2〜5 を追加した。

run1 単独で判定した場合の GB10 14 行の内訳は、1 位 6・僅差 1・負け 7。5 ラウンド中央値（正式）では 1 位 4・僅差 0・負け 10 になる。
この 2 つの比較で区分が動いたのは次の 3 行。

| 行 | run1 単独 | 5 ラウンド中央値 | 原因 |
|---|---|---|---|
| gemm CUDA N=256 | 1 位 1.71× | 2 位 0.84× | candle が run1 だけ 0.154 ms（run2〜5 は 0.075〜0.077 ms） |
| infer CUDA | 1 位 1.84× | 3 位 0.41× | candle が run1 だけ 0.182 ms（run2〜5 は 0.040〜0.042 ms）。PyTorch も run1 だけ 2.43 ms（run2〜5 は 0.040〜0.041 ms） |
| gemm CPU N=256 | 僅差 0.95× | 3 位 0.68× | PyTorch・TensorFlow が run1 だけ遅い（PyTorch 0.711 ms に対し run2〜5 は 0.248〜0.273 ms） |

揺れはほぼ run1 に集中しており、中央値は run2〜5 の水準に収まる。fandhe-ai の reuse 行は run 間でほぼ不変だった（例: infer CUDA reuse の round_spread は 1.00）。

GB10 で round_spread が 1.5 を超えたセルとその値:

| セル | run1〜run5（ms） |
|---|---|
| candle infer CUDA | 0.182／0.041／0.042／0.042／0.040 |
| PyTorch infer CUDA | 2.431／0.040／0.041／0.040／0.041 |
| PyTorch train CUDA | 2.529／0.795／1.124／1.023／1.366 |
| burn gemm CUDA N=256 | 0.266／0.169／0.177／0.318／0.172 |
| SciPy train CPU | 6.123／1.552／0.974／1.456／1.095 |
| fandhe-ai gemm CPU N=2048 fresh | 26.6／25.1／18.0／17.5／17.1（判定は reuse 行） |

全セルの round_spread は各 `*-median5.jsonl` の `round_spread` フィールドにある。

## 結果（正式: M4 系列 B・GB10 5 ラウンド中央値）

`scoreboard/gen_0100.out` より、判定対象 27 行の内訳:

| 区分 | 行数 |
|---|---|
| 1 位 | 6 |
| 僅差 | 2 |
| 負け | 19 |
| 判定不能セル | 1（GB10 PyTorch CPU N=4096 の 1 要素。全 5 run 共通） |
| ノイズ帯 | 4 |

参考として、0.9.0 版（#1988 精度クラス反映版）の内訳は 1 位 5・僅差 2・負け 20 だった。
ただし 0.10.0 版では次の 2 点で比較相手そのものが変わっているので、内訳の差を fandhe-ai 自身の変化とは読まない。

- Python 3 FW が転記・流用値から実測へ変わった。
- burn CUDA GEMM は #1988 版から有効セル（精度クラス TF32・#1989 承認）として扱っている。

fandhe-ai 自身の中央値は、GB10 の reuse 行で 0.9.0 比 0.96〜1.05× の範囲だった。v0.10.0 の性能改善候補はすべて既定 OFF の opt-in で、既定経路の GEMM・学習・推論の選択ロジックは変わっていない。

## 役割・機能の対応表（計測外）

ページ下部の「役割・機能の対応表」の fandhe-ai 列は、2026-10-03 にタグ `v0.10.0` の facade 公開面で再監査した（0.9.0 版ページまでは 0.8.0 時点の記述のままだった）。

| 項目 | 内容 |
|---|---|
| 判定基準 | facade の `pub use`／`pub mod` から到達できるものだけを「ある」とする。内部クレートにだけ実装があるもの（高階微分・`generate()`・KV キャッシュ・npy／npz・Adadelta 等）は「未公開」と書く |
| 主な変化 | 相互運用（ONNX import／export・safetensors・`save_model`／`load_model`）が facade から到達可能になった。NN 層（Conv・Pool・Norm・Embedding・MHA・TransformerEncoder・RNN）、optimizer 7 種と LR scheduler 7 種、DataLoader・`fit`・AMP、`ModelRegistry` を反映した |
| 他列 | PyTorch・TensorFlow・SciPy・Hugging Face・LangChain の列は変更していない |

## DGX スクリプトの失敗検知（事後確認）

`scripts/dgx-run-0100.sh` は計測時に実行したものをそのまま収納している。各段階の失敗は終了コードの表示（`run_all rc=`）と `FAIL` 行の出力だけで、ラウンドを中止しない。このため、次の 3 点を事後に確認した。

| 確認項目 | 結果 |
|---|---|
| `run_all rc=` | 5 ラウンドとも 0 |
| 各段階の行数 | 全ラウンドで stage1 112 行・skipped 0、stage2 8 行、stage3 28 行 |
| 失敗の痕跡 | `FAIL` 行・`extra.err`・`py.err` は全ラウンド空 |

5 ラウンドの結果 JSONL はすべて内容が異なり、前ラウンドの成果物を流用したものはない。集計の `median_rounds.py` も、セルの欠損・重複があればエラー終了する。次回以降のスクリプトは、失敗時に非零で終了させる。

## 再現

```bash
# M4 Max（ROOT=リポジトリルート、D=作業ディレクトリ。venv-torch／venv-tf と bench_py.py のコピーを D の隣と D/py に置く）
bash scripts/m4max-0100a-loop.sh; bash scripts/m4max-0100b-loop.sh
# GB10（~/work にツリーを rsync 済み。run1 は dgx-run-0100.sh を単独で、run2〜5 は dgx-0100-loop.sh）
bash ~/work/dgx-run-0100.sh; bash ~/work/dgx-0100-loop.sh
# 集計（系列・ファイル種別ごと）
python3 aggregate/median_rounds.py m4max-series-b/results-m4max-0.10.0-median5.jsonl m4max-series-b/run{1..5}/results-m4max-0.10.0.jsonl
python3 aggregate/median_rounds.py gb10/results-dgx-0.10.0-median5.jsonl gb10/run{1..5}/results-dgx-0.10.0.jsonl
# スコアボード（比較元 0.9.0 は ../framework-compare-0.9.0-remeasure/ を既定参照。負荷注記は gen_args.txt）
python3 scoreboard/gen_0100.py \
  --m4 m4max-series-b/results-m4max-0.10.0-median5.jsonl --m4-py m4max-series-b/results-m4max-py-0.10.0-median5.jsonl \
  --m4-alt m4max-series-a/results-m4max-0.10.0-median5.jsonl --m4-py-alt m4max-series-a/results-m4max-py-0.10.0-median5.jsonl \
  --gb gb10/results-dgx-0.10.0-median5.jsonl --gb-extra gb10/results-dgx-0.10.0-extra-median5.jsonl \
  --gb-py gb10/results-dgx-py-0.10.0-median5.jsonl \
  "--m4-series=…" "--m4-load=…" "--gb-load=…"   # 3 引数の実値は scoreboard/gen_args.txt の各行
```

## 公開先

スコアボードは claude.ai Artifact として公開した（非公開の個人 Artifact。<https://claude.ai/artifact/WHm2rindpSRHoGBNF1B6dX>）。0.9.0 版のページは上書きしていない。
