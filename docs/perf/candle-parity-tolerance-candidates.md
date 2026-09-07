# tolerance 判定候補（スケール付き絶対誤差／ULP ベース）の机上評価

## 1. 位置づけ

イシューツリー #1234（ルート）→ #1236（親）配下、本 issue #1237 の成果物。
`docs/perf/cuda-gemm-candle-gate-remeasurement.md` §5 が記録した「N=2048 で
candle 側 CUDA/CPU GEMM 出力が現行複合判定を各 2 要素ずつ外れ判定不能になる」
問題について、`docs/perf/logs/cuda-gemm-candle-parity-1184/`（イシュー #1184。
fail 4 要素の実値取得・厳密真値突合）のダンプ実値のみを入力に、候補判定
（スケール付き絶対誤差・ULP ベース）を現行複合判定へ OR 追加した場合の
fail 数を機械的に算出する。

**本 issue の範囲は事実（fail 数・緩和上限）の算出までであり、推奨案・
採否は #1239（決定記録 draft）が扱う**。tolerance 契約（`RELATIVE_TOLERANCE`／
`ABSOLUTE_RESCUE_THRESHOLD`・`PARITY_REL_TOL`／`PARITY_ABS_TOL`）の変更自体は
本 issue では行わず、変更にはユーザー承認が必須（イシュー #1241）。

## 2. 入力データと評価モデル

### 2.1 入力

- `docs/perf/logs/cuda-gemm-candle-parity-1184/parity-dump-cuda-2048.txt`（candle/cuda
  〈cuBLAS〉N=2048 fail 2 要素）
- `docs/perf/logs/cuda-gemm-candle-parity-1184/parity-dump-cpu-2048.txt`（candle/cpu
  〈gemm crate〉N=2048 fail 2 要素）
- いずれも転送元コミット `4e1ad9cdb809969be3b98602a9d8f9cd23006c1f`（イシュー
  #1184 env_info.txt。GB10 実機実測・40 call 全件 bit 完全一致の決定的ダンプ）

### 2.2 評価モデルの限界（必読）

- **ダンプには現行複合判定で fail した要素しか含まれない**。よって各候補は
  「現行判定への OR 追加（単調緩和）」としてのみ評価でき、
  `fail 数（候補）= 入力要素数（2） − 救済件数` である。**「候補で置き換える」
  判定**（現行 pass 要素が新たに fail に転じうるか）は本ダンプからは評価
  不能であり、スコープ外とする
- `parity_max_abs_err=3.62e-5`／`max_rel=0.281`（`docs/perf/cuda-gemm-candle-gate-remeasurement.md`
  §5.3）は fail 要素以外の passing 要素由来であり、OR 追加ではそれらは
  pass のまま変わらない
- 対象は **K=2048・正方・入力 U[-0.5,0.5)・固定シードの 1 条件のみ**。他形状
  （N=512/1024/4096）・他シードへの外挿は行わない（§5 の K スイープ表は
  A-1 の閾値式自体が K のみの関数であるための参考値であり、実測 fail 数の
  外挿ではない）
- `exact`（有理数演算による厳密真値）を使う指標（`d_ex=|actual-exact|`・
  `d_ref_ex=|ref-exact|`・候補 B-3）は**診断専用**であり、実行時に本体
  `assert_parity`／`ParityBaseline` が使えるのは `d=|ref-actual|` と
  入力・部分和から導ける量（`max_ab`・`sum_abs_ab`・`max_partial`）だけで
  ある。各表の「実行時適用可否」列でこれを区別する

## 3. 候補判定の定義

記号: `d = |ref − actual|`（実行時適用可能）、`d_ex = |actual − exact|`
（診断専用）、`eps_f32 ∈ {2^-23（machine epsilon）, 2^-24（unit roundoff）}`、
`K = 2048`。

### 候補 A（スケール付き絶対誤差。pass 条件 `d <= bound`）

| 系列 | 式 | 備考 |
|---|---|---|
| A-1 | `bound = c × eps_f32 × K × M`、`M = 0.25`（入力範囲 U[-0.5,0.5) からの事前上界 `max|a|·max|b|`） | `c ∈ {0.125, 0.25, 0.5, 1, 2}` × `eps_f32` 2 種のスイープ |
| A-2 | A-1 の `M` を各要素の実測 `max_k|a_k| × max_k|b_k|` に置換 | 代表として `c=1, eps=2^-23` のみ算出（実装は #1238 引き継ぎ） |
| A-3 | `bound = c × eps_f32 × √K × M`（√K スケール） | 救済しないことの記録用 |
| A-4 | `bound = K × u × Σ_k|a_k·b_k|`（`u=2^-24`。古典的前進誤差上界） | 緩すぎることの記録用（参考値） |

### 候補 B（ULP ベース。pass 条件 `err <= t × ulp(base)`）

| 系列 | `base` | `t` |
|---|---|---|
| B-1 | `max\|partial\|`（f32 FMA 逐次累積の部分和絶対値の最大） | `{8, 16, 32, 48, 64}` と `t = c×√K`（`c ∈ {1, 2}`。§5.3 の `√K·ulp(max\|partial\|)` フロアの係数化） |
| B-2 | `Σ_k\|a_k·b_k\|`（実行時 1 パスで計算可能な代替基準） | `{1, 2, 4}` |
| B-3 | `exact`（出力値自体の ULP。無意味であることの記録用） | `{1e3, 1e4, 1e5}` |

各候補は `d` 基準（実行時適用可能）と `d_ex` 基準（診断専用）の両方で救済件数
を算出する（§4 表参照）。

## 4. 結果

再現手順は §8。生出力は
`docs/perf/logs/candle-parity-tolerance-candidates-1237/candidates-2048.md`。

### 4.1 要素別メトリクス

| device | idx | row | col | d=\|ref-actual\| | d_ex=\|actual-exact\|（診断） | d_ref_ex=\|ref-exact\|（診断） | max\|partial\| | max_ab（実測） | Σ\|ab\| | fma_bit_match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cuda | 13850 | 6 | 1562 | 1.125e-05 | 9.165e-06 | 2.084e-06 | 3.969e+00 | 0.2495 | 128.423 | True |
| cuda | 4130484 | 2016 | 1716 | 1.104e-05 | 1.717e-06 | 9.325e-06 | 6.099e+00 | 0.2498 | 129.068 | True |
| cpu | 1372466 | 670 | 306 | 1.265e-05 | 1.452e-05 | 1.869e-06 | 5.613e+00 | 0.2497 | 127.978 | True |
| cpu | 1633751 | 797 | 1495 | 1.163e-05 | 3.255e-06 | 8.370e-06 | 4.748e+00 | 0.2500 | 130.064 | True |

`fma_bit_match=True` は 4 要素すべてで成立しており、`parity_dump_truth.py`
（イシュー #1184）の突合結果（RNG 再現・参照実装の f32 FMA 逐次契約が
正しく機能している）と矛盾しない。

### 4.2 候補 A（スケール付き絶対誤差）

| 候補 | bound(K=2048) | cuda fail | cpu fail | 実行時適用可否 |
|---|---:|---:|---:|---|
| A-1 c=0.125 eps=2^-23 K*0.25 | 7.629e-06 | 2/2 | 2/2 | 適用可能 |
| A-1 c=0.25 eps=2^-23 K*0.25 | 1.526e-05 | 0/2 | 0/2 | 適用可能 |
| A-1 c=0.5 eps=2^-23 K*0.25 | 3.052e-05 | 0/2 | 0/2 | 適用可能 |
| A-1 c=1.0 eps=2^-23 K*0.25 | 6.104e-05 | 0/2 | 0/2 | 適用可能 |
| A-1 c=2.0 eps=2^-23 K*0.25 | 1.221e-04 | 0/2 | 0/2 | 適用可能 |
| A-1 c=0.125 eps=2^-24 K*0.25 | 3.815e-06 | 2/2 | 2/2 | 適用可能 |
| A-1 c=0.25 eps=2^-24 K*0.25 | 7.629e-06 | 2/2 | 2/2 | 適用可能 |
| A-1 c=0.5 eps=2^-24 K*0.25 | 1.526e-05 | 0/2 | 0/2 | 適用可能 |
| A-1 c=1.0 eps=2^-24 K*0.25 | 3.052e-05 | 0/2 | 0/2 | 適用可能 |
| A-1 c=2.0 eps=2^-24 K*0.25 | 6.104e-05 | 0/2 | 0/2 | 適用可能 |
| A-2 c=1 eps=2^-23 K*max_ab（実測） | 6.092e-05（cuda idx=13850 の値。実測 `max_ab` は要素依存） | 0/2 | 0/2 | 適用可能 |
| A-3 c=1.0 eps=2^-23 sqrtK*0.25（√K スケール） | 1.349e-06 | 2/2 | 2/2 | 適用可能（**1 件も救済しない**） |
| A-4（K*u*Σ\|ab\|。古典的前進誤差上界） | 1.568e-02 | 0/2 | 0/2 | 適用可能（**緩すぎる参考値**。現行絶対閾値比 1568 倍） |

**境界の要点**: `eps_f32=2^-23`（machine epsilon）系では `c=0.25` 以上で 4 要素
すべて救済、`c=0.125` は全 fail のまま。`eps_f32=2^-24`（unit roundoff）系では
`c=0.5` 以上で全救済、`c=0.25` が境界（全 fail）。**√K スケール（A-3）はいずれの
候補係数でも 1 件も救済しない**（§5.3 の「√K·ulp フロア」は `max|partial|` 基準
であり、入力規模基準の √K とは別軸であることの確認）。

### 4.3 K スイープ（A-1 の緩和上限。現行絶対閾値 1e-5 との比）

| N=K | c=1 eps=2^-23 bound | 1e-5 比 | c=1 eps=2^-24 bound | 1e-5 比 |
|---:|---:|---:|---:|---:|
| 512 | 1.526e-05 | 1.53x | 7.629e-06 | 0.76x |
| 1024 | 3.052e-05 | 3.05x | 1.526e-05 | 1.53x |
| 2048 | 6.104e-05 | 6.10x | 3.052e-05 | 3.05x |
| 4096 | 1.221e-04 | 12.21x | 6.104e-05 | 6.10x |

A-1 の閾値式は `K` のみの線形関数のため、N=512 では `eps=2^-23` 系でも現行
絶対閾値の 1.5 倍程度に収まる一方、N=4096 では 12 倍まで拡大する。**本表は
式の外挿であり、他形状の実測 fail 数を示すものではない**（§2.2）。

### 4.4 候補 B（ULP ベース）

| 候補 | cuda fail(d基準) | cpu fail(d基準) | cuda fail(d_ex基準,診断) | cpu fail(d_ex基準,診断) | 実行時適用可否 | Phase 2 実装可否 |
|---|---:|---:|---:|---:|---|---|
| B-1 t=8 max_partial | 2/2 | 2/2 | 1/2 | 1/2 | 部分和トレースが要る | 部分和トレース追加実装が要る |
| B-1 t=16 max_partial | 2/2 | 2/2 | 1/2 | 1/2 | 同上 | 同上 |
| B-1 t=32 max_partial | 1/2 | 0/2 | 1/2 | 0/2 | 同上 | 同上 |
| B-1 t=48 max_partial | 0/2 | 0/2 | 0/2 | 0/2 | 同上 | 同上 |
| B-1 t=64 max_partial | 0/2 | 0/2 | 0/2 | 0/2 | 同上 | 同上 |
| B-1 t=1×√K≈45.3 max_partial | 1/2 | 0/2 | 0/2 | 0/2 | 同上 | 同上 |
| B-1 t=2×√K≈90.5 max_partial | 0/2 | 0/2 | 0/2 | 0/2 | 同上 | 同上 |
| B-2 t=1 sum_abs_ab | 0/2 | 1/2 | 0/2 | 1/2 | 実行時 1 パス追加で計算可能 | 1 パス追加で実装可能 |
| B-2 t=2 sum_abs_ab | 0/2 | 0/2 | 0/2 | 0/2 | 同上 | 同上 |
| B-2 t=4 sum_abs_ab | 0/2 | 0/2 | 0/2 | 0/2 | 同上 | 同上 |
| B-3 t=1e3 exact | 2/2 | 2/2 | 2/2 | 2/2 | **実行時には使えない（診断専用）** | 実装不可（真値は実行時に得られない） |
| B-3 t=1e4 exact | 2/2 | 2/2 | 1/2 | 1/2 | 同上 | 同上 |
| B-3 t=1e5 exact | 0/2 | 0/2 | 0/2 | 0/2 | 同上 | 同上 |

**境界の要点**: `max|partial|` 基準（B-1）は `t=32` 付近から部分的救済が始まり
`t=48` 以上で全救済。`t=1×√K≈45.3` は cuda 側が 1 件のみ救済（2 件中 1 件が
`fail(d基準)` として残る）で cpu 側は d基準・d_ex基準とも全件救済
（`√K·ulp(max|partial|)` は §5.3 の対象要素の丸め誤差フロアと近い値だが、
4 要素全てを均一には説明しない）。`sum_abs_ab` 基準（B-2）は `t=1` で
cuda 側が全件救済・cpu 側は 1 件のみ救済（1 件が fail のまま残る）、
`t=2` 以上で 4 要素すべて救済（fail 0/2 が全列で成立）。`exact` 基準（B-3）は出力値そのものの ULP を使うため、救済に
必要な `t` が 1e3〜1e4 桁と非現実的に大きく、実行時に使えないことと合わせて
候補として不適格である。

## 5. 未変更事項の確認

以下は本 issue で変更していない（`git diff --stat` で確認済み）:

```
$ git diff --stat main -- crates/ scripts/bench/framework-compare/bench-common/ \
  scripts/bench/framework-compare/compare_gemm_gate.py \
  scripts/bench/framework-compare/summarize.py \
  scripts/bench/framework-compare/parity_dump_truth.py docs/spec/
(出力なし)
```

- `crates/backend-cpu/src/parity.rs::RELATIVE_TOLERANCE`（1e-3）・
  `ABSOLUTE_RESCUE_THRESHOLD`（1e-5）
- `scripts/bench/framework-compare/bench-common/src/parity.rs::PARITY_REL_TOL`（1e-3）・
  `PARITY_ABS_TOL`（1e-5）
- `scripts/bench/framework-compare/compare_gemm_gate.py`・`summarize.py`（判定
  ロジック・判定不能条件）
- `scripts/bench/framework-compare/parity_dump_truth.py`（同梱 `truth-2048.txt`
  との bit 一致を §8 で再確認済み）
- `docs/spec/`（正本 submodule。編集しない）

## 6. 所見（事実のみ）

- 候補 A（スケール付き絶対誤差）は `eps_f32 × K × 0.25` 系で `c ∈ [0.25, 0.5]`
  程度から fail 4 要素すべてを救済する。これは現行絶対閾値 `1e-5` の
  1.5〜3 倍程度の緩和幅に相当する（§4.3）
- √K スケール（A-3）は本ダンプの fail 要素を 1 件も救済しない。これは
  `docs/perf/cuda-gemm-candle-gate-remeasurement.md` §5.3 が確認した
  「丸め誤差フロアは `max|partial|` 基準の √K·ulp である」ことと整合し、
  「入力規模基準の √K」とは別軸であることを示す
- 候補 B（ULP ベース）は基準の取り方で結果が大きく異なる。`max|partial|`
  基準（部分和トレースが必要）は `t≈32〜48` 程度で救済に転じる一方、
  `sum_abs_ab` 基準（実行時 1 パスで計算可能）は `t=1` の時点で 4 要素中
  3 件（cuda 側 2 件・cpu 側 1 件）を救済し `t=2` 以上で 4 要素すべてを
  救済する。`exact`（出力値）基準は非現実的に大きい `t`
  が必要で、かつ実行時に使えない値のため候補として不適格
- 推奨案・採否・定数値の提案は本 issue の範囲外（#1239 で扱う）

## 7. #1238 への引き継ぎ

- **A-2（実 `max_ab` 基準）の入力規模導出方法**: 本机上計算では代表 1 要素の
  `max_ab`（`max_k|a_k| × max_k|b_k|`）のみ算出した。本体 `assert_parity`／
  `ParityBaseline` 側で採用する場合、GEMM の A・B 全体（対象要素の行・列に
  限らない）からどう `max_ab` を求めるか（全体の `max|A|·max|B|` を 1 回だけ
  事前計算して全要素で共有する案が有力。行・列ごとの計算は候補要素が
  多い場合にコスト増）は #1238 で検討する
- **B-1（`max|partial|` 基準）の部分和トレース要否**: 実行時に `max|partial|`
  を得るには GEMM 累積ループ内で部分和の絶対値最大を追跡する追加コストが
  要る（現状の `assert_parity` は最終値のみ受け取る）。カーネル側の変更を
  伴うため、CPU 参照実装（`f32::mul_add` 逐次）限定でも可否・コストを
  #1238 で検討する
- **定数ピン止めテストとの関係**: `scripts/bench/framework-compare/bench-common/src/parity.rs`
  には `RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD` からの乖離を検出する
  ピン止めテスト（`extract_f64_const` 経由）がある。候補 A/B いずれを採用
  する場合も、既存の複合判定（相対 or 絶対）への **OR 追加**として実装する
  想定であり、既存 2 定数自体は変更しない前提（#1238 で契約整理の一部として
  再確認する）

## 8. 再現手順

```bash
cd scripts/bench/framework-compare
python3 parity_tolerance_candidates.py --n 2048 \
  --dump cuda=../../../docs/perf/logs/cuda-gemm-candle-parity-1184/parity-dump-cuda-2048.txt \
  --dump cpu=../../../docs/perf/logs/cuda-gemm-candle-parity-1184/parity-dump-cpu-2048.txt \
  > ../../../docs/perf/logs/candle-parity-tolerance-candidates-1237/candidates-2048.md

# 未変更確認（parity_dump_truth.py 自体の再現性）
python3 parity_dump_truth.py --n 2048 < ../../../docs/perf/logs/cuda-gemm-candle-parity-1184/parity-dump-cuda-2048.txt
python3 parity_dump_truth.py --n 2048 < ../../../docs/perf/logs/cuda-gemm-candle-parity-1184/parity-dump-cpu-2048.txt
# → ../../../docs/perf/logs/cuda-gemm-candle-parity-1184/truth-2048.txt と完全一致すること
```

入力: `docs/perf/logs/cuda-gemm-candle-parity-1184/`（転送元コミット
`4e1ad9cdb809969be3b98602a9d8f9cd23006c1f`）。
実行環境・commit の記録: `docs/perf/logs/candle-parity-tolerance-candidates-1237/env_info.txt`。
