# burn cuda（TF32 経路）GEMM parity fail 要素の真値突合（イシュー #1984）

## 位置づけ

0.9.0 正式再計測（#1967）で burn cuda gemm fresh は N=256〜4096 の全セルが
`parity_fail_count > 0`（10538〜2728488・`max_abs 1.6e-3〜7.1e-3`・現行第 3 項 bound
`1.9e-6〜3.1e-5`）となりスコアボード規則上「判定不能」であった。本ディレクトリは
`FRAMEWORK_COMPARE_PARITY_DUMP` で取得した fail 要素ダンプを #1184 と同じ手順
（`parity_dump_truth.py`。`Fraction` 厳密真値・f32 FMA 逐次参照の bit 再現）で突合し、
TF32 の単位丸め `u=2^-11`・`c=0.5` を用いた第 3 項（線形 K／√K 形）で机上救済される
件数を `parity_tolerance_candidates.py`（§9 拡張の `--extra-eps`／`--extra-c`）で算出した
**事実の記録**である。判定・tolerance・判定式（`bench-common::parity`・
`compare_gemm_gate.py`・`BASELINES`）は変更していない。**採否判定は書かない**
（#1986 以降のユーザー承認事項）。

## 事前登録規則

`RULE.txt`（2026-09-18T01:48:35Z に固定）のとおり。要点:

- 対象は burn cuda gemm fresh N=256／512／1024／2048／4096（registry ピン `fandhe-ai =0.9.0`
  の framework-compare・bench-burn は `--no-default-features --features cuda`）
- ダンプ上限は N≤1024 が 200000（全 fail 要素・`truncated=false` を確認）、N=2048／4096 が
  4096（**行優先先頭 4096 件のサンプル**。母集団推定とは書かない）。verify は 40 call
  呼ばれるため突合には call=39 の行のみ使う。call 間で `fail_count` が変動した場合は
  `summary-N.txt` から所見として記録する
- 記録事項 (a)〜(c)・採否判定なし

## 実行コマンド

ノード側は実行の要点を再構成した形（ダンプの取得・call=39 行の抽出・`summary-N.txt` の
分離を行った。実際の起動は `run_all_cuda.sh` 系と同じ bench-burn 引数）。Mac 側は実際に
実行したコマンドそのもの。

```bash
# ノード（DGX Spark GB10）。framework-compare を registry ピン 0.9.0 のまま再ビルド
cd scripts/bench/framework-compare
cargo build --release -p bench-burn --no-default-features --features cuda
uptime > uptime_before.txt
for n in 256 512 1024; do
  FRAMEWORK_COMPARE_PARITY_DUMP=200000 target/release/bench-burn --task gemm --device cuda \
    --mode fresh --size $n 2> parity-dump-burn-cuda-raw-$n.txt >> results-burn-cuda-dump.jsonl
done
for n in 2048 4096; do
  FRAMEWORK_COMPARE_PARITY_DUMP=4096 target/release/bench-burn --task gemm --device cuda \
    --mode fresh --size $n 2> parity-dump-burn-cuda-raw-$n.txt >> results-burn-cuda-dump.jsonl
done
uptime > uptime_after.txt
# call=39 の行のみ抽出・40 call の summary を分離（ダンプなし実行の stderr は stderr-nodump-N.txt）
for n in 256 512 1024 2048 4096; do
  grep 'PARITY_DUMP_SUMMARY' parity-dump-burn-cuda-raw-$n.txt > summary-$n.txt
  grep -E 'PARITY_DUMP(_SUMMARY)? call=39 ' parity-dump-burn-cuda-raw-$n.txt > parity-dump-burn-cuda-$n.txt
done

# Mac（机上突合。GPU 不使用）
cd scripts/bench/framework-compare
L=../../../docs/perf/logs/parity-burn-tf32-truth-1984
for n in 256 512 1024 2048 4096; do
  python3 parity_dump_truth.py --n $n < $L/parity-dump-burn-cuda-$n.txt > $L/truth-$n.txt 2> $L/truth-$n.err
  python3 parity_tolerance_candidates.py --n $n --dump burn_cuda_$n=$L/parity-dump-burn-cuda-$n.txt \
    --extra-eps 'tf32u2^-11=0.00048828125' --extra-c 0.5 > $L/candidates-$n.md 2> $L/candidates-$n.err
done
cd $L && python3 summarize_truth.py > truth-summary.md
```

`parity_dump_truth.py`・`parity_tolerance_candidates.py` は本イシューで変更していない。
`.gz` 圧縮済みの入力は `gunzip -k` で展開してから上記を再実行する（`summarize_truth.py`
は `.gz` を直接読める）。

## 生成物

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則 |
| `parity-dump-burn-cuda-N.txt`（1024 は `.gz`） | call=39 の `PARITY_DUMP` 行（末尾に同 call の `PARITY_DUMP_SUMMARY` 1 行）。N≤1024 は全 fail 要素・N=2048／4096 は先頭 4096 件 |
| `summary-N.txt` | 40 call 分の `PARITY_DUMP_SUMMARY`（call 間の決定性の証拠） |
| `results-burn-cuda-dump.jsonl` | ダンプ有効時の JSONL（`parity_*` 列は 0.9.0 正式再計測と同値） |
| `stderr-nodump-N.txt` | ダンプなし実行の stderr（全 5 本 0 バイト＝通常実行では stderr 出力なし） |
| `uptime_before.txt`／`uptime_after.txt` | 負荷記録 |
| `truth-N.txt`（1024 は `.gz`）／`truth-N.err`（全 0 バイト） | `parity_dump_truth.py --n N` の per-element 表 |
| `candidates-N.md`（1024 は `.gz`）／`candidates-N.err`（全 0 バイト） | `parity_tolerance_candidates.py` の候補 A/B 表（`EXTRA c=0.5 eps=tf32u2^-11` 行を含む） |
| `summarize_truth.py`／`truth-summary.md` | `truth-N.txt`・`summary-N.txt`・ダンプの `abs=` を機械集計した (a)〜(e) の表 |
| `env_info.txt` | 実行環境（hostname masked） |

10 MB 超のファイルは `gzip -9` で `.gz` として収録している（`parity-dump-burn-cuda-1024.txt`
31 MB・`truth-1024.txt` 28 MB・`candidates-1024.md` 約 19 MB）。`parity-dump-burn-cuda-512.txt`
（7.7 MB）・`truth-512.txt`（6.7 MB）・`candidates-512.md`（4.5 MB）もリポジトリ
容量の観点から同様に `gzip -9` した（`.gz`。内容は無改変。`zcat` で読む）。

## 結果

### (a) 件数・row 範囲・40 call の決定性

| N | 解析行数 | row 範囲 | fail_count（40 call 全てで同一） | dumped | truncated |
|---:|---:|---|---:|---:|---|
| 256 | 10538 | 0〜255 | 10538 | 10538 | false |
| 512 | 42361 | 0〜511 | 42361 | 42361 | false |
| 1024 | 169929 | 0〜1023 | 169929 | 169929 | false |
| 2048 | 4096 | 0〜12 | 681407 | 4096 | true |
| 4096 | 4096 | 0〜6 | 2728488 | 4096 | true |

- 40 call すべてで `fail_count` が同一（call 間変動なし）。`parity_dump_truth.py`／
  `parity_tolerance_candidates.py` の重複 idx 検出も非 0 終了なし（`*.err` 全 0 バイト）
- N=2048／4096 の解析行は row 0〜12／0〜6 に集中しており、母集団（681407／2728488 件）の
  row 分布は本ダンプからは分からない

### (b) 真値からの距離（`truth-summary.md`）

| N | \|actual−exact\|（burn TF32）min／中央値／max | \|ref−exact\|（f32 FMA 参照）min／中央値／max | actual が近い | ref が近い |
|---:|---|---|---:|---:|
| 256 | 1.028e-05／4.033e-04／1.561e-03 | 5.613e-12／1.302e-07／1.458e-06 | 0 | 10538 |
| 512 | 1.124e-05／5.722e-04／2.342e-03 | 5.322e-12／2.546e-07／3.528e-06 | 0 | 42361 |
| 1024 | 1.047e-05／8.082e-04／3.423e-03 | 2.554e-12／5.096e-07／8.643e-06 | 0 | 169929 |
| 2048 | 2.053e-05／1.134e-03／4.127e-03 | 5.094e-11／9.998e-07／8.940e-06 | 0 | 4096 |
| 4096 | 3.273e-05／1.628e-03／5.428e-03 | 2.725e-10／2.046e-06／2.522e-05 | 0 | 4096 |

`fma_bit_match` は全 N・全解析行で True（100%）: RNG 再現と f32 FMA 逐次参照の契約が
ダンプの `ref_bits` と bit 一致している。

### (c) u=2^-11・c=0.5・M=fixed0.25 の第 3 項での救済数

bound は線形 K 形 `0.5·2^-11·K·0.25 = K/16384`・√K 形 `√K/16384`。救済数は
`candidates-N.md` の `EXTRA c=0.5 eps=tf32u2^-11 {K|sqrtK}*0.25` 行（表の値は
「fail 数 = 総数 − 救済数」なので換算して転記）。`summarize_truth.py`（ダンプの `abs=`
全精度値で `<=` 判定）と一致することを確認済み。

| N | 母集団 fail_count | 線形 K bound | JSONL `parity_max_abs_err` | max_abs < bound | 線形 K 救済（解析行） | √K bound | √K 救済（解析行） | √K 救済率 |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 256 | 10538 | 1.562e-02 | 1.581e-03 | yes | 10538/10538（全件） | 9.766e-04 | 10365/10538（全件） | 98.36% |
| 512 | 42361 | 3.125e-02 | 2.343e-03 | yes | 42361/42361（全件） | 1.381e-03 | 41605/42361（全件） | 98.22% |
| 1024 | 169929 | 6.250e-02 | 3.643e-03 | yes | 169929/169929（全件） | 1.953e-03 | 166955/169929（全件） | 98.25% |
| 2048 | 681407 | 1.250e-01 | 4.941e-03 | yes | 4096/4096（行優先先頭 4096 件） | 2.762e-03 | 4023/4096（行優先先頭 4096 件） | 98.22%（サンプル） |
| 4096 | 2728488 | 2.500e-01 | 7.117e-03 | yes | 4096/4096（行優先先頭 4096 件） | 3.906e-03 | 4015/4096（行優先先頭 4096 件） | 98.02%（サンプル） |

- **線形 K 形は母集団でも全数救済と機械的に言える**: JSONL の `parity_max_abs_err`
  （fail 要素に限らず全要素の `|ref−actual|` の最大値）が 1.58e-3〜7.12e-3 で、線形 K
  bound（1.56e-2〜0.25）を全 N で下回る。これはサンプル偏りに依存しない
- **√K 形は N≤1024 が母集団値（97〜98% 台）、N=2048／4096 はサンプル内の値**であり、母集団
  の救済数は本ダンプからは分からない
- 参考: JSONL の `parity_scaled_abs_rescued`（0／0／0／47／562）は現行契約（`u=2^-24`・
  c=0.5・線形 K）での救済数であり、本表の `u=2^-11` とは別物

## 所見（数値事実から言えることのみ。断定しない）

- 全 N・全解析行で参照実装（f32 FMA 逐次）が burn の TF32 実測値より真値に近い
  （`|ref−exact|` は 1e-12〜2.5e-5・中央値 1.3e-7〜2.0e-6、`|actual−exact|` は 1e-5〜5.4e-3・
  中央値 4.0e-4〜1.6e-3）。fail 要素における両者の差 `|ref−actual|` は `|actual−exact|` と
  ほぼ等しく、`|ref−exact|` はその 1/100〜1/1000 の大きさである（比較の数値事実のみ）
- `|actual−exact|` の中央値 4.0e-4〜1.6e-3 は、TF32 の単位丸め `u=2^-11`（4.9e-4）に
  入力スケール（`|a|,|b| < 0.5`）と K 個の積和の誤差蓄積を掛けた大きさと同じ桁である。
  線形 K 形 bound（`0.5·u·K·0.25`）を全要素が下回ること・√K 形では 1.6〜2.0% が超えることは、
  この誤差が「各積を TF32 へ丸めた誤差の K 個の和」として振る舞うという見方と矛盾しない。
  ただし本記録は burn（cubecl）側のカーネルがどの段階で TF32 丸めを行うかを確認しておらず、
  「TF32 の単位丸めで説明できる」とまでは断定しない
- `max_rel_err` が 1.55〜2.0（JSONL）であるのは、真値が 0 近傍の要素（例 N=256 idx=65512:
  exact −1.50e-3・actual −1.78e-3）で相対誤差が発散するためであり、絶対誤差の分布とは
  切り分けて読む必要がある

## 限界

- N=2048／4096 の解析行は行優先先頭 4096 件（row 0〜12／0〜6）のサンプルであり、母集団の
  `|actual−exact|` 分布・√K 救済率は不明。線形 K の全数救済のみ JSONL の最大値から母集団に
  ついて言える
- 40 call の `fail_count` 同一は決定性の証拠だが、call=39 以外の call のダンプ内容
  （`ref_bits`／`actual_bits`）は保存しておらず、要素ごとの bit 同一性までは確認していない
- 入力は U[-0.5,0.5)・固定シード・正方の 1 条件のみ。スケール依存性（`S_A·S_B`）は対象外
- 本記録は現行判定への OR 追加としての救済数のみを扱う。現行 pass 要素が候補判定で
  新たに fail に転じうるかは評価不能（`parity_tolerance_candidates.py` の評価モデルの限界）

## 秘密情報・内部実値の非混入確認

内部ホスト名・ユーザー名付きメールアドレス・ホームディレクトリの絶対パスを対象とした
`grep -rnE` が本ディレクトリ（`.gz` は `zgrep`）で 0 件であることを確認済み。
