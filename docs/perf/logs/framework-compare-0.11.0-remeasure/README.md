# fandhe-ai 0.11.0 正式系列 framework-compare 再計測

## 目的

crates.io `fandhe-ai =0.11.0` の公開（2026-10-09。`release-all.yml` publish run 37933003465・タグ `v0.11.0`）を受け、
framework-compare の承認ピンを `=0.11.0` へ更新した（PR #2962）。そのうえで GEMM・train・infer の
対戦成績を両機体で計測し直した（生データ: M4 Max は PR #2963、GB10 は PR #2964）。
本ディレクトリには、生ログ・集計スクリプト・スコアボード生成器を収めている。
形式は 0.10.0 版（`../framework-compare-0.10.0-remeasure/`）にそろえている。

0.10.0 版からの主な違い:

- **失敗検知込みのスクリプトで計測した。** 0.10.0 版は計測後に失敗検知を追加し事後確認で補った。0.11.0 版は最初から検知込みの版で回している（後述の「スクリプトの失敗検知」節）。
- **M4 系列 B のゲート判定（`gate.log`）の数値検証は計測後に fail-closed 化した**（PR #2963 の codex 指摘）。計測時の 5 判定はすべて数値で取得できていた（後述）。
- **スコアボードの比較元を 0.10.0 にした。** 0.10.0 版は 0.9.0 を比較元にしていた。
- Python 3 参照 FW（PyTorch・TensorFlow・SciPy）を両機体で同一ラウンド内に実測する方式、GB10 の 5 ラウンド中央値、GB10 の TF32 opt-in 行を判定対象外にする扱いは 0.10.0 版を継承した。

## ディレクトリ構成

| パス | 内容 |
|---|---|
| `m4max-series-a/` | Apple M4 Max・系列 A（ゲートなし）5 run。<br>`run{1..5}/`、`results-m4max-0.11.0-median5.jsonl`（fandhe-ai／candle／burn）、`results-m4max-py-0.11.0-median5.jsonl`（Python 3 FW） |
| `m4max-series-b/` | Apple M4 Max・系列 B（load1 < 8.0 ゲート付き）5 run。**正式系列**。<br>`run{1..5}/`、中央値 JSONL 2 本（系列 A と同じ構成）、`gate.log`（ゲート判定の履歴）、`RULE.txt`（事前登録規則） |
| `loop-m4max.log` | 系列 A → 系列 B の連続実行ログ |
| `gb10/` | DGX Spark GB10 の 5 run。<br>`run{1..5}/`（`results-dgx-0.11.0.jsonl`＝`run_all_cuda.sh` 全セル、`-extra`＝CPU gemm reuse と N=4096 の追加計測、`results-dgx-py-0.11.0.jsonl`＝Python 3 FW、`run_all_cuda.log`、`uptime_*.txt`、`gpu_util_before.txt`、`skipped-dgx-0.11.0.log`、`run.log`）<br>3 系統それぞれの `*-median5.jsonl`、`loop.log`（run2〜5 の連続実行）、`tree.txt`（registry ピン解決の確認） |
| `scripts/` | 計測オーケストレーション。<br>`m4max-run-0110.sh`（1 ラウンド）、`m4max-0110a-loop.sh`（系列 A）、`m4max-0110b-loop.sh`（系列 B・ゲート実装）、`dgx-run-0110.sh`（GB10 1 ラウンド）、`dgx-0110-loop.sh`（GB10 run2〜5） |
| `aggregate/median_rounds.py` | 5 run → セルごとの中央値 run の行。0.10.0 版から変更なし（コピー） |
| `scoreboard/` | スコアボード生成器。<br>`gen_0110.py`、`body_0110.html`、`style.css`、`gen_args.txt`（負荷注記の引数）、`gen_0110.out`（判定一覧の標準出力） |

`gen_0110.py` の 0.10.0 版（`gen_0100.py`）からの差分は、版表記（0.10.0 → 0.11.0）、比較元（0.9.0 → 0.10.0）、入出力のファイル名、
`--gb-prev` の既定（0.10.0 版ディレクトリには 5 ラウンド中央値の JSONL だけがあるため `*-median5.jsonl` を指す）のみ。判定ロジック（同一 run 内比・1 位／僅差／負け・ノイズ帯の定義・判定不能の扱い）は変えていない。

`RULE.txt` 内のディレクトリ名 `m4max-0110a/`・`m4max-0110b/` は、計測時の作業ディレクトリ名である（収納時に `m4max-series-a/`・`m4max-series-b/` へ改名）。
生ログには `/Users`・`/home` で始まる絶対パスが含まれていないことを検索で確認した（0.10.0 版のような置換は不要だった）。

## 計測条件

- **registry ピン**: `fandhe-ai =0.11.0`・`candle-core =0.11.0`・`burn =0.21.0`（`deps-policy.md` 第 9 区分）。
  - M4 Max: 各 `run*/tree.txt`（`cargo tree` の `fandhe-ai v0.11.0`）で確認した。10 run すべて同一の 1 行で、`m4max-run-0110.sh` は一致しなければ計測を中止する。
  - GB10: `run_all_cuda.sh` が `cargo tree` を出力しないため、各 `run.log` に `WARN: registry 0.11.0 line not found in log` が出る（0.10.0 版と同じ現象）。そこで `gb10/tree.txt` に、計測前（run1 開始前・事前ビルド直後）と計測後（run5 完了後）の確認を収めた。
    - `GEMM_GATE_PATCH_FACADE_PATH` が未設定（path patch なし）
    - `cargo tree -p bench-fandhe --locked` の `fandhe-ai v0.11.0`（path なし）
    - `Cargo.lock` の `source = "registry+…crates.io-index"`
    - `cargo build --release -p bench-fandhe --locked -v` が `Fresh`（計測バイナリがこの解決で作られていること）
- **ツリー**: main `624d0ee4`（PR #2962 マージ後・registry `fandhe-ai 0.11.0`）。M4 Max の全 10 run の `run.log` 先頭行と GB10 の全 5 run の `run.log` 先頭行が `sha=624d0ee4`。DGX は同じツリーを rsync した（`RULE.txt`）。
- **Python 3 FW**（版数は各 JSONL の `version` フィールドと `RULE.txt` による）:
  - M4 Max: PyTorch 2.14.0（MPS）・SciPy 1.18.1・TensorFlow 2.16.2＋tensorflow-metal 1.2.0（`RULE.txt`）。`m4max-run-0110.sh` は PyTorch／SciPy に `venv-torch`、TensorFlow に別の `venv-tf` を使う。
  - GB10: PyTorch 2.14.0+cu130・TensorFlow 2.21.0（CPU のみ）・SciPy 1.18.1。
  - 計測スクリプトは `bench_py.py`（`scripts/bench/framework-compare/` 配下のもの）。M4 は `m4max-run-0110.sh` が作業ディレクトリの `py/bench_py.py`、GB10 は `dgx-run-0110.sh` が `~/work/bench_py.py` を呼ぶ。これらがリポジトリ内の `bench_py.py` と同一内容かの記録はない。
- **Apple M4 Max**:
  - 系列 A（ゲートなし）: 各 run 開始時の load1 は 5.03／5.82／4.77／5.40／4.39（各 `run*/uptime_before.txt`）。run1 のみビルドを含む。
  - 系列 B: 各 run 開始前に load1 < 8.0 を最大 30 分待つ。5/5 run がゲートを通過して完走した（`gate.log` は 5 エントリ。いずれも初回判定で通過し、通過時 load1 は 4.81／4.92／4.75／5.14／4.55。`loop-m4max.log` は系列 A・B とも `ALL-DONE completed=5/5`）。ゲートは開始時点の判定だけなので、run 中の負荷は統制していない。
  - 日時: `loop-m4max.log` の完了時刻で、系列 A が 2026-10-10 08:45:01Z〜08:49:42Z、系列 B が 08:50:52Z〜08:55:34Z（1 run あたり約 70 秒）。
  - 計測失敗: 全 10 run で `bench-burn gemm metal 512/1024/2048/4096 fresh` が MEASURE_ERROR（結果テンソル全ゼロ。0.8.0／0.9.0／0.10.0 と同じ upstream 既知バグ）。それ以外の失敗はない。
- **DGX Spark GB10**:
  - 環境: NVIDIA GB10。driver・CUDA ランタイムの版は 0.11.0 の計測ログに記録がない（0.10.0 版 README には driver 580.173.02・CUDA 13.0 とあるが、0.11.0 の計測で再確認していないため転記しない）。PyTorch は `+cu130` wheel。
  - 日時: 2026-10-10 08:41:17Z〜09:01:43Z に 5 ラウンドを連続実行（`run.log` の start／done。1 ラウンドは約 4 分）。
  - 負荷: 各ラウンド開始時の load1 は 1.01／6.34／7.30／6.21／5.82（各 `uptime_before_all.txt`。直前ラウンドの残余を含む）。開始時の GPU 使用率は 0／1／0／0／0%（`gpu_util_before.txt`）。
  - 計測失敗: なし（全 run の `skipped-dgx-0.11.0.log`・`extra.err`・`py.err` が 0 バイト）。

## 事前登録規則と実績

`m4max-series-b/RULE.txt`（2026-10-10T08:42:17Z。系列 A・B いずれの結果も出る前に固定。系列 A の run1 開始は 08:42:29Z）:

- 系列 B は、5 run すべてがゲートを通過して完走した場合だけ正式値にする。部分完走や GATE-TIMEOUT なら系列 A を正式値にする。
- 両系列を併記し、A/B 間で verdict が反転したセルをノイズ帯として明示する。
- 0.10.0 → 0.11.0 の絶対 ms 差分は、M4 では負荷差の交絡があるので参考値とする。

**実績**: 系列 B が 5/5 run 完走したので、**系列 B を正式値**にした。

**ノイズ帯の定義**: 勝敗区分（1 位／僅差／負け）が系列 A と B で異なる行。2 位⇄3 位のように「負け」の中で順位が動くだけの場合は反転として数えない。

**該当は 0 行**（`gen_0110.out` に `[ノイズ帯: …]` の印が付く行がない）。M4 Max 13 行の内訳は系列 A・B とも 1 位 1・僅差 1・負け 11 で同じ（系列 A のみで生成して確認）。

RULE.txt が事前に固定したのは M4 の系列 A/B だけで、GB10 の 5 ラウンド化が事前に決まっていたかの記録は本ディレクトリにない。

GB10 で `round_spread` が 1.5 を超えたセル（5 ラウンドの `median_s`。run1〜run5、ms）:

| セル | run1〜run5 |
|---|---|
| burn gemm CUDA N=256（TF32 行） | 0.175／0.170／0.172／0.171／0.625 |
| burn gemm CUDA N=512（TF32 行） | 0.494／0.344／0.690／0.452／0.391 |
| burn infer CUDA（TF32 行） | 0.435／0.380／0.104／0.442／0.750 |
| burn infer CPU | 0.750／0.240／0.236／0.242／0.743 |
| PyTorch gemm CPU N=256 | 0.262／0.271／0.418／0.263／0.484 |
| PyTorch train CUDA | 1.463／0.419／1.534／1.303／1.230 |
| TensorFlow infer CPU | 0.503／0.556／1.134／1.101／1.252 |
| TensorFlow train CPU | 2.713／2.650／4.077／4.399／2.828 |

fandhe-ai の GB10 行で `round_spread` が 1.5 を超えるセルはない。M4 Max（系列 B）で超えるのは fandhe-ai infer Metal の fresh 行（1.53）だけで、判定列は reuse 行。全セルの `round_spread` は各 `*-median5.jsonl` にある。

## 結果（正式: M4 系列 B・GB10 5 ラウンド中央値）

勝敗区分と「最速他 FW ÷ fandhe-ai」の比は、`RULE.txt` の主判定どおり**同一 run 内の対戦相手比**で出す。run ごとに「その run の有効な相手最速 ÷ fandhe-ai reuse」を求め、5 run の中央値が 1 を超えれば 1 位、0.90 以上なら僅差、それ未満なら負けとする。表のセル値・順位（N 位）・「vs」の相手名は、framework ごとに選んだ 5 ラウンド中央値から出す。

`scoreboard/gen_0110.out` より、判定対象 27 行の内訳:

| 区分 | 行数 | 内訳 |
|---|---|---|
| 1 位 | 4 | M4 Max 1（infer Metal）・GB10 3（gemm CUDA N=4096・gemm CPU N=2048・N=4096） |
| 僅差 | 2 | M4 Max 1（gemm CPU N=512）・GB10 1（gemm CPU N=1024） |
| 負け | 21 | M4 Max 11・GB10 10 |
| 判定不能セル | 1 | GB10 PyTorch CPU N=4096 の 1 要素（複合判定を外れる。順位に含めない） |
| ノイズ帯 | 0 | |

`gen_0110.out` の注記が 1 件ある。M4 Max gemm Metal N=256 は、同一 run 内比の 5 run 中央値が 0.88×（負け）、framework ごとの中央値同士の比が 0.94×（僅差）で、区分が変わる。正式の区分は同一 run 内比に従い「負け」。

### 0.10.0 との比較

0.10.0 版（同じ判定ロジック）の内訳は 1 位 5・僅差 3・負け 19・判定不能 1・ノイズ帯 4 だった。区分が動いた行は M4 Max の 3 行で、GB10 の区分は 14 行とも同じ。

| 行 | 0.10.0 | 0.11.0 |
|---|---|---|
| M4 Max gemm Metal N=256 | 僅差 0.95× | 負け（2 位）0.88× |
| M4 Max gemm CPU N=512 | 1 位 1.05× | 僅差 0.95× |
| M4 Max gemm CPU N=1024 | 僅差 0.96× | 負け（2 位）0.87× |

M4 Max の 3 行は、0.10.0 版では系列 A/B で区分が反転したノイズ帯の行（Metal N=256・CPU N=512・CPU N=1024）と同じ行である。ただし 0.10.0 版の系列 A は load1 5.30〜28.51 の高負荷下で、0.11.0 版の系列 A は 4.39〜5.82 と負荷条件が異なる。この差が区分の動きの原因かは、本データからは切り分けられない。

fandhe-ai 自身の reuse 行の中央値（時間比 0.11.0 ÷ 0.10.0。gemm・train・infer の計 13 行〈M4 Max〉・14 行〈GB10〉）:

| 機体 | 範囲 |
|---|---|
| GB10 | 0.95〜1.04×（最小 gemm CPU N=256、最大 gemm CUDA N=512） |
| M4 Max（系列 B 同士・負荷差の交絡を含む参考値） | 0.91〜1.15×（最小 gemm CPU N=256、最大 gemm Metal N=4096） |

0.10.0 版の GB10 の 0.9.0 比（0.96〜1.05×）と同程度の幅である。ただしこの幅が計測ゆらぎか fandhe-ai の変化かは、本データからは切り分けられない。v0.11.0 の変更内容と性能の関係は本計測の範囲外。

## 役割・機能の対応表（計測外）

ページ下部の「役割・機能の対応表」の fandhe-ai 列は、タグ `v0.11.0`（`6b14fdb4`）の facade 公開面で再監査した結果（監査記録の日付 2026-10-09・`docs/perf/framework-compare-feature-matrix-0.11.0.md`）に更新した（#2682）。

| 項目 | 内容 |
|---|---|
| 判定基準 | facade の `pub use`／`pub mod` から到達できるものだけを公開扱いとする。未公開・保留が 1 項目でも残る行は「部分的」。承認済みの非目標（対象外）は行を下げない |
| 主な変化 | 9 行とも判定は「部分的」のまま、内訳が公開側へ動いた。`Sequential::add_*` 31→65、optimizer 拡充・param groups・EMA・SWA、高階微分・関数型 AD、`generate`・KV キャッシュ・speculative decoding、npy／npz、f64 autograd。監査対象外の 4 行は 0.10.0 版を継承し、層・役割の公開モジュール列挙（`inference`・`text` 追加）のみ更新。実機 parity は一部のみ実測済みで、残りは未実測 |
| 他列 | PyTorch・TensorFlow・SciPy・Hugging Face・LangChain の列は変更していない |

## スクリプトの失敗検知

0.11.0 の計測（2026-10-10・main `624d0ee4`）は、0.10.0 の計測後に追加した失敗検知（PR #2498 の codex 指摘）を含む版のスクリプトで回した。`dgx-run-0110.sh`・`dgx-0110-loop.sh`・`m4max-0110a-loop.sh`・`m4max-0110b-loop.sh` の冒頭コメントに、その旨が記されている。

| 対象 | 確認結果 |
|---|---|
| GB10 5 ラウンド（`run.log`） | `run_all rc=0`、`stage1-done rows=112 skipped=0`、`stage2-done rows=8`、`stage3-done pyrows=28`、`done.`。`FAIL` 行・`extra.err`・`py.err` は全ラウンド空 |
| M4 Max 10 run（`run.log` 最終行） | 10 run とも `done. rows=48 pyrows=32 skipped=4`。`skipped-m4max-0.11.0.log` は 10 run とも 4 件で、いずれも burn Metal GEMM N=512〜4096 の既知の記録拒否 |
| ループ | `loop-m4max.log` は系列 A・B とも `ALL-DONE completed=5/5`。`gb10/loop.log` は `loop done.` |

注意点が 2 つある。

- **M4 系列 B のゲートの数値検証は計測後に追加した。** `m4max-0110b-loop.sh` の `gate()` は、`sysctl` の失敗や非数値の load1 を `GATE-INVALID` として打ち切る（PR #2963 の codex 指摘）。計測時の `gate.log` は 5 エントリすべて数値（load1 = 4.55〜5.14）で、`GATE-INVALID`・`GATE-TIMEOUT` は出ていない。検証の有無で判定は変わらない。
- **`m4max-run-0110.sh` の期待行数検査について。** 検査は 0.10.0 の計測後に追加したもので、本ディレクトリの `m4max-run-0110.sh` は 0.11.0 の計測に使った版として PR #2963 で収めた（同ファイルのコメントもこの来歴に合わせた）。事後確認として、上表のとおり 10 run の最終行（行数）と skipped の内容は期待どおりだった。

## 再現

```bash
# M4 Max（ROOT=リポジトリルート、D=作業ディレクトリ。venv-torch／venv-tf と bench_py.py のコピーを D の隣と D/py に置く）
bash scripts/m4max-0110a-loop.sh; bash scripts/m4max-0110b-loop.sh
# GB10（~/work にツリーを rsync 済み。run1 は dgx-run-0110.sh を単独で、run2〜5 は dgx-0110-loop.sh）
bash ~/work/dgx-run-0110.sh; bash ~/work/dgx-0110-loop.sh
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

未公開。対応表は #2682 で更新済み。Artifact としての公開は所有者の操作で行う。
